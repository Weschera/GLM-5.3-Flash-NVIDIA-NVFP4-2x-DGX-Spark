# NVIDIA GLM-5.3-Flash NVFP4 on two DGX Sparks

## Outcome

Deployment and acceleration experiment complete, **with quality caveats**. The official NVIDIA checkpoint is running at `http://10.0.0.109:8910/v1` on spark-78f1 and spark-366f. DFlash2 depth 7 is retained. The Qwen/Hermes pair was not modified and its API remained healthy.

## Measured decode throughput

Each arm ran the same four requests: two English technical tutorials and two Korean essays, concurrency one. All ended naturally. Client rates were reconstructed from raw streamed responses, first-token timestamps, wall time and server token usage.

- Baseline: **15.18 tokens/s median**.
- DFlash2 depth 7: **20.50 tokens/s**, **35.0% higher** than baseline; selected.
- DFlash2 depth 3: **24.78 tokens/s**, **63.2% higher**; not selected because its coding response failed the frozen output contract before the remaining short checks could finish.

These are ratios of per-arm medians. Outputs and lengths differ, so they are not exact-output speedups, matched-task wall-time improvements, or proof of speculative losslessness. One benchmark boot per arm; no repeated-boot confidence intervals.

## Context and quality

All three arms correctly retrieved the first, middle and final markers from **120124 input tokens** with natural completion. The configured serving ceiling is **131072 tokens**. This is a text-only retrieval/capacity smoke test, not comprehensive long-context quality or multimodal validation.

Depth 7 passed JSON arithmetic, isolated code execution (fixed cases plus 200 generated cases and input nonmutation), tool calling, tool-result continuation, and exposed reasoning in the explicit thinking-ON probe. It still violated two strict format requirements: Markdown fences around code and explanatory text when only a numeric answer was requested.

Depth 3 repeated its function in three code blocks with self-correction commentary. The complete response failed the contract. Its blocks were not executed or repaired into a pass, and later tool/reasoning checks were skipped. This is not proof that the algorithm was wrong or that speculation caused the behavior.

Throughput uses a **custom forced-OFF assistant prefix** while the template retains maximum-effort system text. Planning can appear in final content. These are diagnostic serving measurements, not native non-reasoning quality qualification. The prior baseline's output failures remain recorded.

## Running configuration

- Official target: `nvidia/GLM-5.3-Flash-NVFP4` at `423acf37583782c51c142d145aef733d72943d93`.
- Drafter: `incoai/GLM-5.3-Flash-DFlash2` at `bf582e4eacc1810f76656d1811693ff6c6737d2a`.
- Runtime: Tony's pinned vLLM image, digest `sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`.
- TP2 over RoCE; eager execution; Marlin MoE; FP8 E4M3 KV with 6 GiB explicit allocation; 128-token prefill batching; max sequences one; prefix cache disabled; text-only.
- Persistent precompiled kernel cache and serialized compilation avoid the earlier cold-start host-memory peak. Launch-port checking permits reusable TIME_WAIT sockets while still rejecting a live listener.
- Checkpoint files remain unchanged. Transformers compatibility overlay and the custom template are external to the target.
- MTP was not used: the NVIDIA export has no exported MTP head. No DSpark performance claim is made.

## Final live verification

The selected depth-7 containers were read back on both nodes. `/health` and `/v1/models` returned HTTP 200; the fresh restarted server generated **READY** and recorded accepted speculative tokens. No requests remained active. Qwen/Hermes returned HTTP 200 with its original model.

The final restart received a health-generation check, **not a repeat of the full benchmark**. Performance and full quality evidence refer to the recorded trial boot. The controller exited normally; its temporary memory guards were disarmed. No persistent watchdog or agent-fleet migration was added.

## Evidence and reproduction

Working recipe: `/Users/wesche/projects/nvidia-glm53-tp2/`.

- `evidence/text128-dflash-resume/summary.json`: final selection and scoped completion status.
- `comparison.json`: all three immutable trial records, including failures.
- `raw-evidence-final-verification.json`: all 12 speed streams and three long-context streams independently verified.
- `final-live-verification.json`: fresh endpoint/container/configuration readback.
- `selected-relaunch/final-health-result.json`: actual restarted-server generation.
- `README.md`, `launch_node.py`, `prepare.py`, `retry_text128.py`: pinned setup, launch, guards and campaign commands.

Credits: Z.ai for the base architecture/weights; NVIDIA for the official quantization; Inco for the drafter; Tony for the GB10 runtime/patches. Mia's EXL3 results informed comparison methodology but are not this deployment or a matched quality comparison.
