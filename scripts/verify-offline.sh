#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m unittest discover -s tests -v
if [[ -d .git ]]; then
    git diff --check
fi
