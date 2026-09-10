# NVIDIA GLM-5.3-Flash NVFP4: TP2 experiment

This is a custom DGX Spark compatibility/tuning experiment, **not an NVIDIA-published GB10 recipe**. The deployment/speed experiment is complete with quality caveats: `evidence/text128-dflash-resume/summary.json` reports `complete_with_quality_caveats`, and `final-live-verification.json` confirms the restored depth-7 server is healthy. This is not a clean quality qualification or a production-readiness claim. Earlier failures remain preserved.

## Completed trial matrix and verified selected server

The official NVIDIA target has completed real TP2 generation. Baseline, depth 7 and depth 3 each completed four sustained speed requests and a 120124-token input retrieval test. All 12 speed streams and all three retrieval traces passed independent raw-evidence verification. The trial matrix is complete. Depth 7 was restored on both Sparks, returned READY naturally, and passed fresh HTTP/model/container checks. The controller exited and no benchmark request remains active. Memory guards were disarmed on normal campaign completion; they are not persistent production watchdogs. Sustained speed and full quality checks belong to the recorded benchmark boots, not the final restart health probe.

- Baseline median: **15.1812 tok/s**.
- DFlash2 depth 3 median: **24.7775 tok/s**, **1.63211x** baseline (63.21% higher decode throughput). It passed long-input retrieval but emitted three fenced copies of its coding answer with self-correction commentary. The strict response contract failed before code execution, so later tool/reasoning checks were skipped. This is not proof that its algorithm was wrong and is not evidence that DFlash caused the behavior. Depth 3 is excluded from selection by the frozen qualification gate; no block is repaired or retroactively counted as a pass.
- DFlash2 depth 7 median: **20.5001 tok/s**, a **1.35035x** ratio of medians (35.04% higher decode throughput).
- Depth-7 sample range: 19.1503–26.8939 tok/s. Every sample stopped naturally; outputs and lengths differ, so the ratio is not matched-task wall-time acceleration or proof of exact losslessness.
- End-of-arm counters, including warmup and quality traffic: 5439 drafts, 38073 proposed tokens, 9444 accepted tokens; 24.805% acceptance and 2.73635 mean advancement length.
- Depth 7 passed JSON arithmetic, isolated code execution (fixed cases, 200 generated cases, nonmutation), tool calling, tool-result continuation, and three-marker long-input retrieval.
- Strict failures remain: code had a Markdown fence, and the thinking-ON final response added explanation rather than only the requested number. No clean quality pass is claimed. The forced-OFF template caveat below applies to throughput.

Evidence: `evidence/text128-dflash-resume/comparison.json` and `dflash2-7/`. The offline `verify_saved_results.py` independently reconstructs content, reasoning, usage and first-token timing from raw SSE JSONL, recomputes rates, checks exact request parity, and validates retrieval answers. It does not mutate run evidence or send inference traffic. Literal SSE `[DONE]` was not retained in the JSONL; the verifier explicitly trusts that field from the parser receipt.

```bash
# During the sweep: explicitly permit an incomplete matrix.
python3 verify_saved_results.py evidence/text128-dflash-resume --allow-partial
# Final verification: exits 2 if any of the 12 speed samples is missing.
python3 verify_saved_results.py evidence/text128-dflash-resume
```

The final trial-matrix check verified all 12 completed speed samples and all three completed retrieval traces (`raw-evidence-all-arms-verified.json`). Negative tests reject altered content, token usage, speed, and request settings; an incomplete matrix cannot pass the default final check. Those tests are harness validation, not additional model-quality scores.

## Scope

- Target: `nvidia/GLM-5.3-Flash-NVFP4`, revision `423acf37583782c51c142d145aef733d72943d93`.
- Local NVMe on both hosts: `/home/raulwesche/models/nvidia-GLM-5.3-Flash-NVFP4`.
- TP rank 0: spark-78f1, management `10.0.0.109`, fabric `10.10.10.1`.
- TP rank 1: spark-366f, management `10.0.0.183`, fabric `10.10.10.5`.
- Intended endpoint: `http://10.0.0.109:8910/v1`.
- No changes to Qwen/Hermes on spark-9f73/spark-b610. Its endpoint is read only for before/after health evidence.
- Launch and cleanup affect only the label-checked `nvidia-glm53-tp2` container. Refuse startup if any other container or GPU compute process is active on either selected node.

## Pinned runtime and credits

- Tony's GB10 runtime: `ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`.
- Image ID, confirmed on both hosts: `sha256:35c6f70ffcba62fd67d7b9d4b4e8300ad177201792ce9cdb1ea18fd449bc23b6`.
- Tony source revision: `050081dc41ce6edd4d3f15fa19dc3410ba4210e3` of `tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`.
- vLLM: `0.1.dev20051+g487ecf187`.
- Isolated Transformers overlay: `transformers==5.16.1`, `tokenizers==0.23.1`. The image's tokenizers 0.22.2 fails the Transformers 5.16.1 dependency check. No host Python environment is upgraded.
- SM121 sparse-index top-k patch comes from Tony's pinned repository. It avoids the known long-context shared-memory kernel failure.
- DFlash2: Inco's `incoai/GLM-5.3-Flash-DFlash2`, revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`; config and model weights hash-verified on both hosts. Target architecture by Z.ai; target quantization by NVIDIA. Mia's EXL3 work informs comparison methodology but is not the target checkpoint here.

## Runtime changes and limits

- NVIDIA files remain unchanged and are mounted read-only. The separate `chat_template.jinja` changes only the final assistant prefix so `enable_thinking=false` appends `<think></think>` instead of the original unconditional `<think>`. ON/OFF rendering must pass an actual target-tokenizer check before launch.
- TP2 over RoCE, eager execution, Marlin MoE, FP8 E4M3 KV, 6 GiB explicit KV allocation, context configured to 131072, max sequences 1, batched tokens 128, prefix caching disabled for the controlled comparison.
- Marlin is a W4A16 execution path for the quantized MoE weights; this is not a claim of NVIDIA's native W4A4 throughput. Dense NVFP4 kernel selection remains automatic and must be read from actual startup logs.
- Use explicit `--language-model-only --limit-mm-per-prompt '{"image":0,"video":0}'`. The original preflight text-only conclusion was incorrect: after weights loaded at a reported 88.89 GiB/rank, startup profiled a maximum-size video and recovered kernel logs showed OOM/memory pressure. Vision/video is excluded from this retry. The first explicit text-only retry (batch 1024) still crossed the local memory floor during the dummy forward pass, before the configured KV allocation was returned; weights loaded at 88.23 GiB/rank. The guard stopped rank 0 at 4524138496 available host bytes and cleanup removed both test containers. This disproves multimedia profiling as the sole problem, but is not a completed OOM-to-capacity test. Current batch-128 retry adds startup-only per-layer memory traces; hooks are removed before real inference.
- Runtime context configuration is not proof of full-window quality or 1M-context support. The retry calibrates an actual 118K–122K input using the target tokenizer, then checks three markers at the beginning, middle, and end. This is a capacity/retrieval smoke test, not a comprehensive long-context quality evaluation.
- Swappiness is set to zero on the test nodes. Before each launch, idle-workload guards run, then sync and page-cache reclamation clear the download/verification cache. There is no model deletion or swapoff operation.
- MTP is not used: NVIDIA's export omits the MTP head. Do not use `num_nextn_predict_layers` alone as evidence of exported weights.
- A DSpark-labelled third-party checkpoint is not interchangeable with NVIDIA's official checkpoint. DFlash2 is the first directly compatible acceleration candidate tested here; no DSpark performance claim is made.

## First-layer memory investigation

The batch-128 retry also crossed the host-memory guard inside decoder layer 0, before returning from its first forward and before KV allocation. Both nodes recovered cleanly. A separate synthetic dense-FP4 kernel probe then exposed first-use CUTLASS compilation: individual `cicc` compiler processes exceeded 6 GiB RSS, while probe tensor allocations were about 130 MB. The original launcher left Ninja build parallelism unrestricted and its FlashInfer cache inside the disposable container.

The remedy under test is serial compilation (`MAX_JOBS=1`, `FLASHINFER_NVCC_THREADS=1`) plus persistent `FLASHINFER_WORKSPACE_BASE=/cache` and `TRITON_CACHE_DIR=/cache/triton`. Representative zero-input FP4 GEMMs passed on the first Spark after compilation, then passed on the second Spark using a copied, hash-verified cache. The transfer receipt covers 39 files / 11411778 bytes. These are kernel-fixture results, **not GLM inference or token-speed results**. The official target must still pass the guarded TP2 run.

`kernel_probe.py` is the isolated diagnostic. `cache_ready_retry.py` waits for its successful source receipt and process exit, hashes/transfers the pinned cache, executes the fixtures on the second idle node, then replaces itself with the guarded TP2 workflow. It expects the source kernel probe to have been launched separately, with a bounded lifetime and a memory-limited container. Cache preparation progress is in `cache-retry-status.json`; the subsequent model workflow uses `text128-status.json`, with combined stdout in `cache-ready-workflow.log`.

## Restart-port fix and baseline reuse

The completed diagnostic baseline measured a 15.181243819412654 tok/s median across four naturally completed sustained samples. It passed the 120124-input-token marker retrieval probe. Its short code response violated the single-code-block contract, so quality remains failed and subsequent short checks were not completed.

Both initial speculative attempts failed the head launcher's plain port bind with EADDRINUSE before target loading or inference. A Linux socket regression reproduced the same failure using reusable TIME_WAIT sockets. The corrected preflight sets SO_REUSEADDR, deliberately not SO_REUSEPORT; tests on both Sparks confirm TIME_WAIT passes and a real listener remains rejected. This was a launcher failure, not a speculative speed/correctness result.

Resume only the speculative arms without repeating the completed baseline:

```bash
python3 -u retry_text128.py --resume-baseline evidence/text128-diagnostic-sweep/comparison.json
```

Reuse requires matching exact inference commands, target revision, template hash, probe-code hash and completed baseline measurements. Original failures remain unchanged. Current results are `evidence/text128-dflash-resume/`, stdout is `dflash-resume.log`, and the controller remains `text128-status.json`.

## Automation

```bash
python3 -u prepare.py
python3 -u retry_text128.py
```

Run the bounded workflow in a supervised background process. The retry first arms a node-local, label-scoped memory guard on both test nodes; a 6 GiB MemAvailable floor stops only the experiment. This guard is best-effort protection, not a guarantee against sudden driver-level allocation failures. It keeps 128K configured and the 6 GiB KV allocation, reducing prefill/concurrency and excluding multimedia instead of shrinking the serving window. It has a single-instance lock, SSH/HTTP inactivity deadlines (long-input prefill gets a separate allowance), hard per-request wall deadlines, startup deadlines, a global deadline, and signal-triggered scoped cleanup. Download verification is performed by the separate downloader under `~/.hermes/cache/nvidia-glm53-download/`; the workflow reads fresh remote verification receipts before starting.

Sequence:

1. Wait for both complete, hash-verified target copies.
2. Verify runtime, copied recipe hashes, target tokenizer ON/OFF rendering, and read-only Hermes health.
3. Launch worker then head with no speculation; persist sustained throughput first, then test near-128K input independently of short-answer grading.
4. Repeat the same settings with DFlash2 speculative token counts 7 and 3.
5. Require actual draft counters in `/metrics`, not just accepted configuration flags.
6. Select among semantic-passing variants, preferring fewer strict format failures before speed. If no variant qualifies, retain the baseline explicitly as unqualified. Save the fastest measured diagnostic separately; read back model list, health, and both containers.

If the non-speculative baseline fails a runtime/safety gate, stop the test containers and preserve diagnostics rather than pretending speculative acceleration fixes correctness. Strict code-fence and final-answer-format failures are retained in `quality-report.json` and do not abort the remaining runtime/context/speed probes. `quality_passed` remains false for those variants; a completed comparison with a non-perfect selected variant has `status: complete_with_quality_caveats`, not a clean quality pass. If a speculative variant fails, record the failure and retain the baseline as an eligible fallback.

## First real generation and harness correction

With the compiled kernel cache, both ranks completed all decoder layers, allocated KV cache, and initialized successfully with 131072 configured context. The server returned READY and the exact JSON arithmetic result, generated a function that passed fixed cases plus 200 generated interval tests and nonmutation, completed a tool round trip, and exposed nonempty separate reasoning. The raw code used a Markdown fence and the thinking-on final answer added explanation instead of only the requested number. Those are preserved strict failures; the first harness aborted and cleaned up before long-context or sustained throughput tests.

The follow-up sweep keeps identical prompts and records those strict failures separately from runtime gates. It does not retroactively rescore earlier results or claim a 128K input has passed yet. Full target evidence from the first working startup is under `evidence/text128-precompiled/`; follow-up runtime progress is `text128-status.json`, stdout is `qualified-sweep.log`, and results go to `evidence/text128-qualified-sweep/`.

## Diagnostic sweep after a semantic failure

The `text128-qualified-sweep` baseline generated a function that genuinely failed its first nonempty execution case: the implementation kept a stale `current_end` after starting a new interval. Its raw output, requests, and failure remain unchanged. A prior boot passed the same frozen test; temperature zero and seed 37 did not yield identical code across these boots. No cause for the numerical/output variation has been established.

The current diagnostic workflow records throughput before grading, runs the unchanged near-128K retrieval probe independently of short coding checks, and distinguishes `QualityFailure` from transport/server failures. Generated-answer failures remain failed and can leave later short checks uncompleted; the report states this explicitly. Runtime faults still propagate to cleanup. A semantic-failing draft is not eligible for selection merely because it is fast. `diagnostic-fastest.json` is a measured-speed observation, not a quality qualification.

Current results: `evidence/text128-diagnostic-sweep/`; stdout: `diagnostic-sweep.log`; controller: `text128-status.json`. Source snapshots and a frozen-prompt/unchanged-archive-generator check are retained. `test_diagnostic_grading.py` passes four harness regression tests, including an actual isolated replay of the recorded bad model code that remains failed. Unit-test results are not model inference scores.

## Measurement rules

- Temperature 0, top_p 1, seed 37, repetition penalty 1; same prompts and context/KV configuration.
- Throughput requests use a custom **forced-OFF assistant prefix**, not an established native non-reasoning mode. The template still renders `Reasoning Effort: Max`, and the base model card documents low/high/max budgets, not OFF. One English sample emitted extensive self-correction in final content despite an empty separate reasoning channel. Preserve these measurements as diagnostic; use a separate native-thinking control before attributing quality failures to the quant or claiming normal deployment quality. The explicit thinking-ON probe remains separate.
- Output token allowance is the entire remaining serving window from `/tokenize`. A `length` finish is a failed probe, not a completed response.
- Save raw SSE events and exact request payloads, response content/reasoning, usage, finish reason, TTFT, wall time, client-estimated decode throughput, and end-to-end throughput.
- Correctness: strict JSON, a generated merge-interval function tested in a network-disabled/read-only CPU container against fixed and generated cases, tool call and tool round trip, ON/OFF reasoning checks, and long-context retrieval.
- Extended Korean answers must have no U+FFFD replacement characters. This checks the failure reported for older third-party ModelOpt GLM quantizations in vLLM issue 54150; it does not assume NVIDIA's checkpoint has that defect.
- Throughput: two repeats each of English technical prose and Korean prose, at concurrency one. Generated texts may differ across numerical/speculative paths. These are microbenchmarks, not Spark Bench scores, exact losslessness proofs, or quality parity claims.
- Server speculative acceptance counters are saved separately. First-stream-burst effects make client decode tok/s an estimate; end-to-end rates and complete raw timestamps are retained.

`evidence/harness-selftest/` contains a **reference implementation used to validate the test harness**, not model-generated output. Do not report it as model performance.
