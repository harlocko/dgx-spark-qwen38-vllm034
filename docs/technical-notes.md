# Technical notes: migration to vLLM 0.34

The previous local v0.32 MAPFIX build used patches for Mamba scheduling, sparse-attention Q/K/V handling and PLE offload. Those patches were not copied into the v0.34 image. Instead, the base image comes from vLLM 0.34 and uses its `--engram-config '{"disk_offload":true}'` path to read selected n-gram rows from safetensors storage.

Two adaptations remained necessary for the recovered NVFP4-hybrid checkpoint and long-context speculative decoding.

## Hybrid FP8 adapter

`src/install_v034_fp8_hook.py` checks the `ModelOptMixedPrecisionConfig` interface in the pinned vLLM installation and adds an import hook to its `modelopt.py` module. `src/vllm_fp8_hybrid_v034.py` then recognizes selected FP8 side layers in the safetensors metadata and routes them through vLLM's blockwise FP8 linear method. The NVFP4 expert path remains handled by native ModelOpt code.

This adapter is enabled by `VLLM_FP8_HYBRID=1` in `Dockerfile.v034-core` and expects the `-fp8hybrid` checkpoint produced by the conversion tool. It is a version-specific monkey patch, not a general-purpose vLLM extension.

## MTP and YaRN override

`src/install_v034_mtp_yarn.py` applies a second import hook to `vllm.config.speculative`. The final `vllm_mtp_yarn_v034_v3.py` keeps only RoPE-related Hugging Face override values for the MTP draft config. It handles the temporary `dummy_` config used by vLLM when discovering a draft architecture and deep-copies the input config to avoid changing the caller's object.

The `--hf-overrides` JSON sets YaRN-related parameters under `text_config.rope_parameters`. The serving command keeps MTP at two speculative tokens and sets the configured limit to 500,000 tokens for both the serving model and speculative config.

## Decode CUDA Graphs

The original eager configuration used `--enforce-eager`. The selected configuration instead uses:

```json
{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,3]}
```

`mode=0` disables `torch.compile` compilation while allowing full CUDA graph capture for uniform decode. The GPU runtime reported a successful decode graph capture and 0.16 GiB for the captured graph state. The two capture sizes correspond to the decode paths exercised in this configuration, including speculative decoding.

vLLM's documentation: https://docs.vllm.ai/en/v0.34.0/design/cuda_graphs/

Engram disk-offload documentation: https://docs.vllm.ai/en/v0.34.0/features/engram/

## Memory and performance interpretation

The launcher reserves exactly 24 GiB for KV cache. The `docker stats` memory value is not a complete representation of all weight, KV and device allocations on a unified-memory system. `nvidia-smi` may also report `N/A` for physical GPU memory on GB10.

The 500k context and single-request limit were chosen for this deployment, not because the hardware or upstream recipes categorically lack higher-context or multi-request modes. Different KV sizing, batch budgets and concurrency targets need their own functional and memory tests.
