# Release checks

The source audit is complete, but two steps still require a DGX Spark test run before describing this repository as independently reproducible.

1. Build both images using `scripts/build.sh` and verify the hook installer output. The rebuilt image is not expected to have exactly the same image ID as the historical container.
2. Using a prepared NVIDIA FP8-hybrid checkpoint, start the public launcher and run `scripts/smoke-test.sh`. Check the capture log, basic tool calls and a short deterministic generation. For a completely clean installation, test download and FP8 conversion too.

Do not start the public image at the same time as the existing model on a single 128 GB Spark: the second set of weights and cache may exhaust unified memory. End-to-end testing requires another suitable machine or a controlled downtime window with the original container preserved for rollback. A different port does not isolate GPU memory. Keep automatic restart disabled until the new version is accepted.

Inspect `git diff --check`, the credential scan and license notices before uploading the repository. No model weights, generated checkpoints, user cache, private keys or benchmark working directories belong in Git.

The `Gate7` serializer hidden-test failure is a known limitation of the recorded run. Do not label Gate7 as fully passing.
