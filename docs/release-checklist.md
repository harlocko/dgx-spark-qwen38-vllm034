# Release checks

## Validated on the existing DGX Spark

The following checks were completed on 20 November 2026 with the repository checked out from GitHub:

1. Both Docker images were rebuilt with `--no-cache`. The FP8 and MTP/YaRN hook installers completed successfully.
2. Six installed Python modules and patched/original vLLM files matched the historical working image by SHA256.
3. Seven offline tests passed, including a check that the launcher supplies `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and `HF_HUB_OFFLINE=1`.
4. After correcting the missing long-context environment variable, the rebuilt image started with the existing FP8-hybrid checkpoint and passed live API checks: text generation and a tool call with the expected function arguments.
5. vLLM reported successful CUDA graph capture (0.16 GiB). Startup to the health check took roughly 5 minutes 40 seconds.
6. The test container was removed and the original container was restarted. Its health endpoint responded successfully.

The first live launch attempt failed before loading the model because the long-context override was missing. That failure is resolved in the launcher tested above.

## Not yet validated

- A full installation on a clean DGX Spark, including fresh model download and FP8 conversion.
- End-to-end retrieval or quality at the configured 500,000-token context limit.
- Multimodal operation or simultaneous-request performance. This profile runs text only with one concurrent sequence.
- General compatibility with other GPU architectures or unpinned vLLM versions.
- The recorded Gate7 agent run still has one serializer hidden-test failure. Do not describe Gate7 as fully passing.

## Before changing repository visibility

Inspect the current tree for credentials, private paths, model weights, generated checkpoints and transient files. Confirm the Apache-2.0 license notices and third-party attribution remain intact. Verify the release documentation against the latest commit before making the repository public.

Do not start two copies of this checkpoint at once on a single 128 GB Spark. For another live test, keep the working container available for rollback and schedule a controlled service interruption.
