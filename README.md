# Qwen3.8 Flash Next on DGX Spark: vLLM 0.34

This repository contains the vLLM 0.34 configuration tested on one NVIDIA DGX Spark (GB10) in November 2026. It uses NVIDIA's NVFP4 checkpoint with FP8 side-layer conversion, disk offload for the Engram table, a 500,000-token configured context limit, two-token MTP speculative decoding and decode-only CUDA Graphs.

The code is an adaptation of [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX), with an FP8 converter from [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound). The original projects established the single-Spark deployment, hybrid checkpoint conversion and earlier vLLM compatibility work. This repository focuses on the local migration to vLLM 0.34, its two compatibility hooks, and the measured effect of CUDA Graphs. See [THIRD_PARTY.md](THIRD_PARTY.md).

The published upstream project currently documents a vLLM 0.33 configuration with higher concurrency and additional context profiles. This repository is not intended as a replacement for those profiles; it records a tested single-request v0.34 variant.

## Requirements

- DGX Spark or compatible NVIDIA GB10 system, Linux aarch64, 128 GB unified memory.
- Docker with NVIDIA GPU access and a working NVIDIA driver.
- Fast local NVMe storage with space for the NVIDIA checkpoint and an additional FP8 hybrid layout. Check the current size of the upstream files before downloading.
- A working Hugging Face download path for `nvidia/Qwen3.8-Flash-Next-NVFP4`.

The image is based on an official vLLM v0.34.0 ARM64 digest. It has not been qualified on other GPU architectures.

## Build and run

Run these commands from the repository root. They build new local images; they do not replace or stop an existing container.

```bash
bash scripts/build.sh
bash scripts/download-weights.sh
bash scripts/prepare-hybrid.sh
bash scripts/serve.sh
bash scripts/smoke-test.sh
```

The first model download is large. Both preparation scripts use `HF_CACHE` if you keep your Hugging Face cache outside `~/.cache/huggingface`. The hybrid preparation process converts selected BF16 tensors to blockwise FP8 and creates a sibling snapshot named `<revision>-fp8hybrid`; it leaves the downloaded original checkpoint in place.

To reuse an existing prepared hybrid snapshot, skip downloading and conversion. Ensure that `HF_CACHE` points to its parent cache and that `refs/main` points to the matching revision, or set `MODEL_REVISION` when launching.

`serve.sh` defaults to a loopback-only API at `http://127.0.0.1:18300/v1`. For access from another machine, explicitly set `BIND_IP` to the desired interface address and restrict network access independently:

```bash
BIND_IP=192.0.2.10 RESTART_POLICY=unless-stopped bash scripts/serve.sh
```

The example address is documentation-only. Use an address assigned to the server. The script refuses to replace a container of the same name and does not stop other containers using port 18300. To use a different port or container name, set `PORT` and `NAME`. **Do not run this model alongside another copy on the same 128 GB Spark**: two loaded weight sets may exhaust unified memory even if the containers use separate ports.

Startup can take several minutes because weights must be loaded and CUDA graphs captured. Check progress with `docker logs -f qwen38-flash-v034-mtp2-cgdecode`.

### Effective vLLM settings

The arguments in `scripts/serve.sh` match the inspected command line of the tested container, except for the mount path and IP binding, which have been made host-independent.

| Setting | Value |
| --- | --- |
| Base image | vLLM 0.34.0, pinned official digest |
| Checkpoint | NVIDIA NVFP4, FP8-hybrid snapshot |
| Engram | `{"disk_offload":true}` |
| Context limit | `500000` configured |
| Long-context guard override | `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |
| Hugging Face access during serving | `HF_HUB_OFFLINE=1` |
| Max concurrent sequences | `1` |
| Max batched tokens | `2048` |
| KV cache allocation | `25769803776` bytes (24 GiB) |
| Speculative decoding | MTP, 2 tokens |
| Compilation | `mode=0`, `FULL_DECODE_ONLY`, capture sizes `[1,3]` |
| Vision | Disabled with `--language-model-only` |
| Prefix caching | Enabled |
| Tool parser | `qwen3_coder` |
| Reasoning parser | `qwen3` |

The 500,000-token value is a serving configuration limit, not proof of an end-to-end 500,000-token retrieval test under this build. The checkpoint declares a 262,144-token maximum, so the launcher sets `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` to allow the YaRN-based 500,000-token profile. This bypasses vLLM's length check and should not be used without the matching RoPE/YaRN configuration. The memory settings and batch limits were selected for one request at a time.

## Measured results

The tested v0.34 configuration reached 46.55 tokens/s in a controlled single-stream generation test, compared with 41.69 tokens/s for the same v0.34 configuration with eager execution, a gain of 11.65%. CUDA graph capture was reported as complete; the engine reported about 0.16 GiB of additional memory for graph capture.

On the agentic benchmarks, the decode-only graph variant completed V6 in 196 seconds with 24/24 tests passing and Gate7 in 489 seconds with one hidden contractual failure. Gate7 reproduced a known failure from earlier runs: preservation of a `None` field in an existing serializer. These are individual agent runs, so wall-clock differences are not a controlled measure of the runtime change alone.

The complete comparison and test limitations are in [docs/benchmarks.md](docs/benchmarks.md). The differences from the earlier MAPFIX build are described in [docs/technical-notes.md](docs/technical-notes.md).

## What has and has not been verified

The source files were recovered from the actual development checkout. The Docker image identifier and vLLM argument list were taken from the running, benchmarked container. The 0.34 compatibility hooks were validated on that system through V6, Gate7 and a functional runtime check.

This cleaned distribution changes image tags, paths and port-binding defaults. On 20 November 2026, both images were rebuilt without Docker layer cache on the existing DGX Spark, and the six inspected compatibility modules matched the historical image by SHA256. The first launcher startup test failed because `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` was missing; the launcher was corrected and seven offline tests passed. A subsequent live test loaded the existing FP8-hybrid checkpoint, reached a healthy API after roughly 5 minutes 40 seconds, generated a text response, made a function call with the expected arguments, and captured CUDA graphs (0.16 GiB reported). The original serving container was restored and its health endpoint responded. A fresh checkpoint download and conversion, and a full installation on a clean DGX Spark, have not yet been reproduced. The configured 500,000-token limit has not been tested end to end.

Known constraints:

- The hooks patch internal vLLM Python classes and may break with a different vLLM build. The base image digest is intentionally pinned.
- Tool calling and agentic behavior were exercised, but Gate7 still has one hidden-test failure.
- Vision was not tested in this configuration.
- Only one simultaneous request was configured. No concurrency comparison was performed against the upstream v0.33 profiles or the previous v0.32 MAPFIX setup.
- Model weights, checkpoints and derived FP8 shards are not included. Their licenses and access conditions are separate.
- The server is not an authentication gateway. Keep remote access behind a firewall or trusted network boundary.

## Source history and attribution

See [docs/provenance.md](docs/provenance.md) for the inspected image identity and exact source archive checksums. The package includes the full Apache-2.0 license and retains credit for the upstream work. These modifications are not an official NVIDIA or vLLM release.
