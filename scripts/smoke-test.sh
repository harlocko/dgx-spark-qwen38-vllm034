#!/usr/bin/env bash
set -Eeuo pipefail

API_URL="${API_URL:-http://127.0.0.1:18300}"

echo "Checking $API_URL/health"
curl --fail --silent --show-error --max-time 10 "$API_URL/health" >/dev/null

echo "Checking model listing"
curl --fail --silent --show-error --max-time 15 "$API_URL/v1/models"
echo

echo "Testing a short completion"
curl --fail --silent --show-error --max-time 120 \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Reply with the word ready."}],"temperature":0,"max_tokens":40}' \
    "$API_URL/v1/chat/completions"
echo
