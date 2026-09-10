# Reproduction guide

## Scope and safety

This is the **captured, tested lab recipe**, not a portable one-command installer. The published scripts retain their measured configuration; publication did not rerun the GPU experiment or silently refactor the serving path. Offline verification is portable. Deployment requires adapting the explicit inventory below.

Use two **dedicated, idle 128 GB DGX Sparks**, working GPU containers, local NVMe, and an already configured RoCE fabric. Never stop unrelated containers to make a preflight pass. Do not run these scripts unchanged against the original lab IPs. The API binds to `0.0.0.0` without authentication: keep it on a trusted, firewalled network, not the public Internet.

The initial experiment hit host-memory/OOM failures and once required power-cycling. Eager mode and the local guard reduce risk but do not make arbitrary settings safe. Do not skip the cold-cache preparation or replace the explicit text-only settings.

**Drafter license:** Inco's checkpoint is CC BY-NC-ND 4.0. Obtain appropriate permission before commercial use. No weights are included.

## 1. Verify the published evidence first (CPU only)

Python 3.11+; standard library only:

```bash
python3 verify_saved_results.py evidence/text128-dflash-resume
python3 -m unittest discover -s tests -v
```

This checks the original captured runs; it does not query the original endpoint. `test_port_guard.py` adds Linux-specific socket lifecycle checks. The synthetic `kernel_probe.py` is a GPU test, not an offline test or an inference benchmark.

## 2. Adapt the inventory in a working copy

Save a diff before making changes. `SOURCE_MANIFEST.json` pins the original tested files, so its hash test is intentionally expected to fail after local customization. Do not rewrite the original published evidence.

| Setting | Original value | Locations to review |
|---|---|---|
| SSH user | `raulwesche` | `prepare.py`, `workflow.py`, `retry_text128.py`, `cache_ready_retry.py`, `download/controller.py` |
| Management hosts | `10.0.0.109`, `10.0.0.183` | same controllers |
| Rank hostnames | `spark-78f1`, `spark-366f` | `launch_node.HOSTS` |
| Fabric addresses | `10.10.10.1`, `10.10.10.5` | `launch_node.HOSTS`, `--master-addr`, NCCL address range |
| Fabric device / HCA | `enp1s0f0np0`, `rocep1s0f0` | `launch_node.command()` environment |
| GID / fabric range | GID 3, `10.10.10.0/24` | `launch_node.command()` environment |
| Remote recipe root | `/home/raulwesche/nvidia-glm53-tp2` | `prepare.REMOTE`, `launch_node.BASE`, `memory_guard.ROOT` |
| Target root | `/home/raulwesche/models/nvidia-GLM-5.3-Flash-NVFP4` | `launch_node.MODEL`, `download/download_node.py`, `workflow.py` tokenizer mount |
| Target download receipts | `/home/raulwesche/nvidia-glm53-download-*.json` | downloader, launcher and `workflow.wait_downloads()` |
| Draft cache root | `/home/raulwesche/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2` | `launch_node.DRAFT_REPO` |
| Inference URL | `http://10.0.0.109:8910` | `probes.URL` |

`workflow.py` also contains read-only checks against an **unrelated Qwen service** at `http://10.0.0.229:8900/v1/models`. Remove those two lab-only checks when adapting the controller; they are not a dependency of GLM. No script should contact that service in your reproduction.

Before rerunning the campaign, change `retry_text128.py`'s `w.OUT` to a fresh directory such as `evidence/local-run-001`. The shipped `evidence/text128-dflash-resume` directory is published evidence and must never be overwritten. Do not use `--resume-baseline` on another machine/configuration; that recovery option was only for matching the same measured lab run after a launcher-only failure.

## 3. Download pinned target and drafter on each node

Provision Python 3.11+ with `huggingface_hub` in a dedicated download environment. After adapting `download/download_node.py`, copy it to each node and run it there:

```bash
python3 -m venv ~/glm-download-venv
~/glm-download-venv/bin/python -m pip install huggingface_hub
# Run from a supervised terminal; hard deadline includes a large model download.
timeout --kill-after=20s 10800s ~/glm-download-venv/bin/python -u download_node.py
```

The target downloader pins revision `423acf37583782c51c142d145aef733d72943d93`, verifies every file against the Hub manifest (SHA-256 for LFS; Git blob SHA-1 for small files), and writes the status/manifest/verification receipts consumed by the launcher. Each node needs 204,476,277,515 bytes for the target, plus draft, image, dependencies, cache and staging space. The downloader enforces an additional staging reserve.

The optional `download/controller.py` does this over SSH for the two hosts; if using a venv, adapt its remote Python executable too. Do not assume a locally activated venv applies to an SSH command.

After accepting the drafter's license, use its exact revision and the cache root configured in `launch_node.py`:

```bash
timeout --kill-after=20s 1800s ~/glm-download-venv/bin/hf download \
  incoai/GLM-5.3-Flash-DFlash2 \
  --revision bf582e4eacc1810f76656d1811693ff6c6737d2a \
  --cache-dir /home/raulwesche/.cache/huggingface/hub
```

Adapt that home path. The snapshot must land under `models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a`; `--local-dir` would require changing the launcher mount/layout.

Verify `model.safetensors` against SHA-256 `b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac` (2,342,169,800 bytes). `config.json` is 1,273 bytes, with Git-blob SHA-1 `083085aaf95d59bad59c58f1160865683460254e`—that is **not** ordinary file SHA-1. The original two-node verification receipts are included under `evidence/text128-dflash-resume/`.

## 4. Prepare the pinned runtime and cache

On the controller workstation, after adapting the inventory:

```bash
python3 prepare.py
```

This pulls/checks the pinned image, installs a separate Transformers/tokenizers overlay, copies the SM121 sparse-index patch, and writes `prepared.json`. It does not install a new host inference stack. Review its commands first: image pulling and dependency downloads require Internet access; eventual model serving is offline.

On **each node**, copy `kernel_probe.py` into the remote recipe root, keep all target/model containers stopped, and run the following **after replacing paths for your own inventory**:

```bash
BASE=/home/raulwesche/nvidia-glm53-tp2
IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6

timeout --kill-after=10s 950s docker run --rm \
  --name nvidia-glm53-kernel-probe --label wesche.task=nvidia-glm53-kernel-probe \
  --gpus all --network none --memory 32g --memory-swap 32g --cpus 4 --pids-limit 256 \
  -e MAX_JOBS=1 -e FLASHINFER_NVCC_THREADS=1 \
  -e FLASHINFER_WORKSPACE_BASE=/cache -e TRITON_CACHE_DIR=/cache/triton \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e PYTHONPATH=/transformers-overlay \
  -v "$BASE/cache:/cache" -v "$BASE/transformers-overlay:/transformers-overlay:ro" \
  -v "$BASE/kernel_probe.py:/probe.py:ro" \
  --entrypoint python3 "$IMAGE" /probe.py
```

Require `cache/kernel-probe-result.json` to end at stage `complete`. These are zero-input kernel fixtures, not GLM answers or throughput evidence. On a first cold compile, monitor host compiler RSS as well as GPU memory.

The measured campaign compiled on one node, copied/hash-verified the cache, and then ran these fixtures on the second. `cache_ready_retry.py` records that tested transfer path, but **its final action starts the full experiment**; do not use it as a passive copy utility. Its cache-relative directory is version/SM-specific and must match the pinned image. Running the fixture independently on each node is an alternative cache-preparation procedure, not a newly benchmarked serving profile.

## 5. Launch and compare safely

The fully guarded controller copies `launch_node.py`, `chat_template.jinja` and the instrumented runner to both nodes; arms the local memory guards; checks download receipts, runtime and template; runs baseline/7/3; preserves grades; and restores its selection:

```bash
# Only after adapting all inventory, removing lab-only health hooks,
# preparing target/draft/cache, and setting a fresh w.OUT:
nohup python3 -u retry_text128.py > local-campaign.log 2>&1 &
```

This intentionally performs a long experiment. It sets `vm.swappiness=0` and drops host page caches before launches via passwordless sudo on **your dedicated test nodes**. It will not stop unrelated containers or GPU workloads to proceed. It uses a label-checked named container and refuses occupied listener ports. Inspect all these side effects before granting privileges.

The controller has bounded startup, transport and campaign lifetimes. Individual long-input requests have a separate prefill allowance. The stored requests set `max_tokens` to the remaining serving window; this is an explicit request limit, **not** a claim that token-limit fields were omitted. A context-limit finish was not accepted as a natural benchmark stop.

To inspect a launch without starting it, on each adapted hostname:

```bash
python3 launch_node.py --dry-run --speculative 7
```

For a manually orchestrated depth-7 launch, first place all overlay/patch/template mounts exactly as the dry run shows and arm a fresh `memory_guard.py` on both nodes; launch **rank 1 before rank 0**. The guard has a five-hour deadline and can kill only the label-matched experiment container. Do not leave a stale `memory-guard.done` file, or reuse a stale/dead guard. Prefer the guarded controller rather than treating `launch_node.py` alone as safe cold-start orchestration.

The normal campaign disarms temporary guards after completion and leaves the selected server online with no restart policy. It does not install production monitoring, authentication or a system service. Scoped cleanup is `python3 launch_node.py --stop` on each node; it checks the ownership label before stopping/removing the named container.

## 6. Verify actual generation, not just configuration

Check `/v1/models`, `/health`, a real streamed completion and positive `vllm:spec_decode_num_drafts_total` / `num_accepted_tokens_total` on `/metrics`. Use your own controller's raw receipts to recompute rates with `verify_saved_results.py` (it expects the same three-arm/four-prompt matrix).

Keep short ACK latency separate from sustained throughput. A selected-server restart health response is not a new full quality score. Preserve strict failures, code that was not executed, skipped checks, natural stopping behavior, long-context failures and memory-guard events. Never turn the fastest measured arm into a clean quality pass by extracting or repairing its best code block.
