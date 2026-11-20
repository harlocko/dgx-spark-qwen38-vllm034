# Provenance of the November 2026 v0.34 configuration

The sources were exported from a local development checkout on 19 November 2026. The export is preserved separately; this repository includes only the files needed for the final v0.34 variant and its preparation tools.

## Source snapshot

- Original archive: `dgx-spark-publication-sources-20261119.tar.gz`
- Archive SHA256: `0037e998e8575871b1ffc5012dd5d21a1222b007aca2d949ff73c3adecb9f1c5`
- Local baseline commit in that checkout: `727cf4a24e503ec17e18e828005a3a562b7a1c73`
- Important: this commit identifies the local baseline. It is not presented as an upstream GitHub commit.

The original export command explicitly added untracked v0.34 files to a `git archive HEAD` of the baseline. Those files were not yet committed in the source checkout.

## Runtime snapshot

- Running image tag: `qwen38-flash-dgx:test-v034-yarn-mtp-v3-20261118`
- Running image ID: `sha256:e04bdb7d1c115cae7e0b8a76c4568dc5c445fc574dd6f721f724a1d1dd3ceb0a`
- vLLM base image reference in the recovered Dockerfile: `vllm/vllm-openai@sha256:e40eb98263fa13418dda684147d73e8ec4a2b0e056a37e955dc189cd1b11dd16`
- Container used the `Dockerfile.v034-core` FP8 hook and `Dockerfile.v034-yarn-mtp-v3` MTP/YaRN adapter.
- CUDA graphs were enabled at launch, not compiled into a separate image.

The tested container mounted a local Hugging Face cache read-only and served a `-fp8hybrid` snapshot. Its original host-specific directory is intentionally omitted. The public launcher mounts the user's cache at `/hf` inside the container and constructs the equivalent model path from its revision.

## Differences from recovered files

- `Dockerfile.v034-yarn-mtp-v3` has a new parent image tag (`qwen38-flash-dgx:pub-v034-core`) to match the public build script. The patch source copied into the image is unchanged.
- Both upstream checkpoint preparation scripts now default to the public v0.34 core image tag, rather than the archived v0.31 image tag. The conversion and download logic is otherwise unchanged.
- The public `scripts/serve.sh` was reconstructed from the inspected container command and uses a generic `/hf` mount and loopback binding. It was not part of the original running image build.
- Older experimental v0.34 adapters (v1/v2) and v0.32 patches are not included in this release candidate. They are retained in the archived development export.

A clean-system rebuild of the public package has not yet been performed. The original image ID therefore cannot be assumed to match the output of the public Docker build.

## SHA256 of selected original source files

These hashes refer to the files inside the source export, before the public naming changes.

| File | Original SHA256 |
| --- | --- |
| `Dockerfile.v034-core` | `517d35ca8f2f3faab65367588014f49a9fe9704e4b49b2dc445179332a05da81` |
| `Dockerfile.v034-yarn-mtp-v3` | `02da45e28c91f7d3430ff4c7f548bc6589a0de0bb57a059fcfe766c8c7db61e8` |
| `src/install_v034_fp8_hook.py` | `9e04b02c12074c39c92238670c9a1120d24c64102a1abb1392237ddb873863ec` |
| `src/install_v034_mtp_yarn.py` | `0ed693951656d10bddafa437bff696992ef02031feb0692e9863fa0b926129ed` |
| `src/vllm_fp8_hybrid_v034.py` | `7126a86a05c560b66f6138bfec1955f34f536290c1e8d73ddbe88f04a1079434` |
| `src/vllm_mtp_yarn_v034_v3.py` | `819ac4faf3a70521be6ca153998eb2ce321b5a4ad4c6072ee017b9a917392674` |
| `scripts/download-weights.sh` | `9ab620f2463da899d6ec45f61a086edf4d71464d0c00d79a1fc4311213065e9a` |
| `scripts/prepare-hybrid.sh` | `e11673adcdb5fb422f31be81b81c527a4cf4f6702baf027b654d5e6b1cfd92a2` |
| `tools/fp8_convert.py` | `52c80c0d6303384dd644fcb60663c28750d9c700343cf2b8258fcf8d1f890629` |
