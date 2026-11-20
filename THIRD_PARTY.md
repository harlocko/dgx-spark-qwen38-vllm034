# Third-party code and attribution

This recipe builds on work published by other developers. These sources are credited here because several files are copied or adapted rather than newly authored for this repository.

## blazux/qwen3.8-Flash-DGX

Source: https://github.com/blazux/qwen3.8-Flash-DGX

License: Apache-2.0. The archived source included a license notice naming copyright 2026 blazux.

Copied and locally adjusted:

- `scripts/download-weights.sh`
- `scripts/prepare-hybrid.sh`

These two scripts retain their original structure. Their default Docker image name was updated to the vLLM 0.34 core image in this repository. Other references to the upstream tooling may remain in the comments.

The upstream repository supplied the original deployment, disk-offload and hybrid-checkpoint preparation work, including the development history leading to the local v0.32 baseline.

## Saren-Arterius/qwen3.8-Flash-DGX-AutoRound

Source: https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound

License: Apache-2.0 according to the archived source attribution.

- `tools/fp8_convert.py` is copied from the `blazux` archive, which explicitly attributes the converter to Saren-Arterius and identifies it as a copy of that fork's converter.
- `src/vllm_fp8_hybrid_v034.py` identifies itself as a port of Saren-Arterius's FP8 compatibility approach from AutoGPTQ to vLLM's ModelOpt NVFP4 configuration class.

## NVIDIA, vLLM and model weights

The Docker build uses the official vLLM project image pinned by digest. The NVIDIA checkpoint is downloaded separately from Hugging Face and is not included here. This repository does not change or redistribute the pretrained model weights.

Model files, the Docker base image, and dependencies retain their own licensing terms. The Apache-2.0 license in this repository does not replace those terms.
