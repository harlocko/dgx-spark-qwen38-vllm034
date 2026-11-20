#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The pinned base digest in Dockerfile.v034-core was used on ARM64/GB10.
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "This build has only been evaluated on Linux ARM64 (DGX Spark/GB10)." >&2
    exit 1
fi

CORE_IMAGE="qwen38-flash-dgx:pub-v034-core"
FINAL_IMAGE="qwen38-flash-dgx:pub-v034-mtp2-cgdecode"

echo "Building $CORE_IMAGE"
docker build --platform linux/arm64 \
    -f Dockerfile.v034-core -t "$CORE_IMAGE" .

echo "Building $FINAL_IMAGE"
docker build --platform linux/arm64 \
    -f Dockerfile.v034-yarn-mtp-v3 -t "$FINAL_IMAGE" .

echo "Images ready:"
docker image inspect --format '{{.RepoTags}} {{.Id}}' "$CORE_IMAGE" "$FINAL_IMAGE"
