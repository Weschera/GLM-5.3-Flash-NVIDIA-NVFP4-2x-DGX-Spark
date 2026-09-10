# Official NVIDIA GLM-5.3-Flash NVFP4 on 2× DGX Spark

**TP2 works. DFlash2 raised the selected configuration from 15.18 to 20.50 tok/s median (+35%).** A depth-3 diagnostic reached 24.78 tok/s (+63%), but failed the coding response contract and was not selected.

This is a measured, text-only community recipe—not an NVIDIA-published GB10 recipe, not a full quality leaderboard, and not a claim of production readiness.

> **License warning:** the NVIDIA target is MIT-licensed, but the [Inco DFlash2 drafter](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a) is **CC BY-NC-ND 4.0**. Treat the accelerated recipe as noncommercial research/evaluation unless you obtain suitable permission. No model weights are redistributed here.

## Results

| Configuration | Median decode tok/s | Relative to baseline | Outcome |
|---|---:|---:|---|
| Official NVIDIA target, no speculation | 15.18 | 1.00× | Real TP2 baseline; code-output contract failed |
| DFlash2, 7 speculative tokens | **20.50** | **1.35×** | Selected; code execution and tool checks passed; two strict format failures remain |
| DFlash2, 3 speculative tokens | 24.78 | 1.63× | Faster diagnostic; three code blocks plus self-correction; rejected by the response-contract gate |

- Four sustained requests per arm: two English technical tutorials and two Korean essays, concurrency one. All 12 stopped naturally.
- All three configurations correctly retrieved beginning/middle/end markers from **120,124 input tokens**. The configured ceiling is **131,072 tokens**.
- Raw SSE reconstruction independently verified all 12 speed samples and all three long-input traces.
- Same request settings, but different output texts and lengths. Ratios are **decode throughput**, not exact-output speedups or matched-task wall-time acceleration. One benchmark boot per arm; no confidence intervals or claim of losslessness.
- **Custom forced-OFF template:** an empty `<think></think>` prefix is used, while the template retains maximum-effort system text. Planning sometimes appeared in final content. These are diagnostic measurements, not native reasoning-OFF qualification.

### What “selected” means

Depth 7 completed JSON arithmetic, isolated code execution (fixed cases + 200 generated cases + input nonmutation), tool calling, tool-result continuation, explicit reasoning-channel checks, and long-input retrieval. It still fenced its code and added explanation when a numeric-only answer was requested. Those are recorded failures, not repaired passes.

Depth 3 emitted three code blocks with self-correction text. Its response failed the frozen contract **before code execution**, and later tool/reasoning checks were skipped. This does not establish that its algorithm was wrong or that DFlash caused the behavior. The baseline also had an output-contract failure.

The selected depth-7 server was restarted and returned `READY`; its model, context and speculative configuration were read back on both nodes. That final restart had a health probe, not a repeat of the complete quality suite. The full benchmark belongs to the preceding trial boot.

## Pinned recipe

- Target: [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4), revision `423acf37583782c51c142d145aef733d72943d93`.
- Target verified on each node: **44 files, 204,476,277,515 bytes**, with Hub hashes checked.
- Draft: [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`.
- Runtime: `ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`.
- TP2 over RoCE; eager execution; **Marlin W4A16 MoE**, not a claim of native W4A4 MoE throughput.
- FP8 E4M3 KV; 6 GiB explicit KV allocation; 131072 context; max sequences 1; prefill batch 128; prefix caching disabled.
- Text-only: `--language-model-only --limit-mm-per-prompt '{"image":0,"video":0}'`.
- Separate Transformers 5.16.1 / tokenizers 0.23.1 overlay. Target weights are unmodified and read-only.
- Persistent FlashInfer/Triton caches and single-job compilation. Cold FP4 compilation must be completed **without the target loaded** first.

## Reproduce / inspect

**Read [REPRODUCE.md](REPRODUCE.md) before running any deployment script.** These are the tested lab scripts, not a portable installer. Hostnames, local paths, network devices, management/fabric IPs and an unrelated read-only health check are deliberately visible and must be adapted to your own machines. Do not point them at another person's hardware.

Offline evidence verification (no SSH, Docker, GPU, model download or inference):

```bash
python3 verify_saved_results.py evidence/text128-dflash-resume
python3 -m unittest discover -s tests -v
```

The code/source and measured receipts are included; weights, credentials, machine process inventories and Docker environment dumps are not. Publication only packages the existing experiment and does not redeploy it.

- [Final comparison and failure records](evidence/text128-dflash-resume/comparison.json)
- [Final summary](evidence/text128-dflash-resume/summary.json) — status `complete_with_quality_caveats`
- [Raw-stream verification](evidence/text128-dflash-resume/raw-evidence-final-verification.json)
- [Experiment history and failed starts](docs/EXPERIMENT.md)
- [Detailed results](docs/RESULTS.md)
- [Source hashes](SOURCE_MANIFEST.json)

## What made it fit

1. Weights loaded on TP2, but initial startup failed during multimodal profiling and then cold kernel compilation; this was not proof that the target weights alone could not fit.
2. Disable image/video profiling explicitly; keep the useful 128K ceiling rather than silently reducing context to 32K.
3. Reduce prefill/concurrency. Precompile representative FP4 kernels without target weights, persist the cache, and serialize compilation.
4. Arm node-local, label-scoped memory guards during the experiment. They are best-effort protection, not a guarantee against sudden driver allocation peaks.
5. Fix the restart preflight to tolerate reusable TIME_WAIT sockets with `SO_REUSEADDR`, while still rejecting an actual listener. Do not mistake that launcher error for draft incompatibility.

The temporary guards disarm at normal campaign completion; this repository does not install a permanent production watchdog. Original OOM/format failures remain documented. No MTP claim: NVIDIA's export lacks an exported MTP head. No DSpark result is claimed.

## Credits and licenses

- **Z.ai:** base architecture and model.
- **NVIDIA:** official NVFP4 checkpoint and Model Optimizer work.
- **Inco:** DFlash2 drafter (noncommercial/no-derivatives license).
- **Tony / [tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark):** pinned GB10 runtime and SM121 patches; source revision `050081dc41ce6edd4d3f15fa19dc3410ba4210e3`.
- **vLLM contributors:** serving engine and Apache-2.0 source files.
- **Mia:** earlier EXL3 work informed methodology; this is not her checkpoint or a matched quality comparison against EXL3.

See [NOTICE.md](NOTICE.md) for component licenses and provenance. Original recipe tooling is MIT; that does not override the licenses of the model, drafter, or vendored files.
