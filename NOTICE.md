# Third-party notices

This repository redistributes no target/draft weights, kernel-cache binaries or container images.

## Original orchestration and measurement tooling

Original experiment scripts and documentation are published by Weschera under the MIT license in LICENSE. Model-generated probe outputs remain immutable experiment evidence, not recommended application code.

## vLLM-derived files

`docker/model_runner_profile_trace.py` and `docker/sparse_attn_indexer_kpool_sm121.py` carry the upstream Apache-2.0 SPDX copyright/license headers. Their license is Apache-2.0, not the root MIT license; see licenses/Apache-2.0.txt. Copyright contributors to the vLLM project.

The sparse indexer file was sourced through Tony's GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark repository at 050081dc41ce6edd4d3f15fa19dc3410ba4210e3. The model runner is derived from the pinned runtime's vLLM runner and modified to add startup-only per-layer memory tracing, removed before inference. These modifications are marked by PROFILE_MEMORY instrumentation. No license is inferred for unrelated upstream repository files.

## Chat template and model

`chat_template.jinja` derives from the Z.ai/NVIDIA GLM-5.3-Flash template and changes the final assistant prefix to expose the experiment's forced-OFF branch. NVIDIA's pinned target model card declares MIT. Preserve applicable upstream copyright and license terms when downloading the full model; the repository's MIT license is not a replacement for upstream notices.

## Drafter — important restriction

Inco's pinned model card declares **CC BY-NC-ND 4.0**, not MIT/Apache. The accelerated configuration must not be advertised as licensed for commercial use. Obtain appropriate permission for uses outside the drafter's terms. This repository links to the checkpoint and does not redistribute or relicense it.

Sources:
- https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/423acf37583782c51c142d145aef733d72943d93
- https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a
- https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/tree/050081dc41ce6edd4d3f15fa19dc3410ba4210e3
