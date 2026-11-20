#!/usr/bin/env bash
set -Eeuo pipefail

# Launches a new container only. Never stops or replaces a running server.
IMAGE="${IMAGE:-qwen38-flash-dgx:pub-v034-mtp2-cgdecode}"
NAME="${NAME:-qwen38-flash-v034-mtp2-cgdecode}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-}"
BIND_IP="${BIND_IP:-127.0.0.1}"
PORT="${PORT:-18300}"
RESTART_POLICY="${RESTART_POLICY:-no}"

if [[ "$MODEL" != "nvidia/Qwen3.8-Flash-Next-NVFP4" ]]; then
    echo "Only the NVIDIA NVFP4 checkpoint was tested with this profile." >&2
    exit 1
fi

if [[ ! -d "$HF_CACHE/hub" ]]; then
    echo "HF cache not found: $HF_CACHE" >&2
    exit 1
fi

REPO_PATH="hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4"
MODEL_REPO="$HF_CACHE/$REPO_PATH"
if [[ -z "$MODEL_REVISION" ]]; then
    [[ -r "$MODEL_REPO/refs/main" ]] || {
        echo "Cannot find model revision. Set MODEL_REVISION or download the checkpoint." >&2
        exit 1
    }
    MODEL_REVISION="$(cat "$MODEL_REPO/refs/main")"
fi

if [[ ! "$MODEL_REVISION" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid model revision" >&2
    exit 1
fi

MODEL_DIR="$MODEL_REPO/snapshots/${MODEL_REVISION}-fp8hybrid"
if [[ ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "Hybrid checkpoint not found or incomplete: $MODEL_DIR" >&2
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image missing: $IMAGE (run scripts/build.sh)" >&2
    exit 1
fi

if docker container inspect "$NAME" >/dev/null 2>&1; then
    echo "Container already exists: $NAME; no changes made." >&2
    exit 1
fi

MODEL_IN_CONTAINER="/hf/$REPO_PATH/snapshots/${MODEL_REVISION}-fp8hybrid"
ROPE_OVERRIDES='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
SPECULATIVE='{"method":"mtp","num_speculative_tokens":2,"max_model_len":500000}'
COMPILATION='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,3]}'

# A private loopback binding is the safer default. Set BIND_IP explicitly for a LAN endpoint.
docker run -d \
    --name "$NAME" \
    --gpus all \
    --network bridge \
    --shm-size 8g \
    --restart "$RESTART_POLICY" \
    --mount "type=bind,src=$HF_CACHE,dst=/hf,readonly" \
    -p "$BIND_IP:$PORT:18300" \
    --entrypoint vllm \
    "$IMAGE" serve "$MODEL_IN_CONTAINER" \
    --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 \
    --port 18300 \
    --model-impl vllm \
    --language-model-only \
    --engram-config '{"disk_offload":true}' \
    --load-format safetensors \
    --safetensors-load-strategy lazy \
    --dtype bfloat16 \
    --hf-overrides "$ROPE_OVERRIDES" \
    --max-model-len 500000 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 2048 \
    --kv-cache-memory 25769803776 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config "$SPECULATIVE" \
    --compilation-config "$COMPILATION"

echo "Container created: $NAME"
echo "Health endpoint: http://$BIND_IP:$PORT/health"
