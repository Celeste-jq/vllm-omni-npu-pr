#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TASKQUEUEENABLE=1

IMAGE_PATH="${IMAGE_PATH:?IMAGE_PATH is required}"
PROMPT="${PROMPT:?PROMPT is required}"
CONCURRENCY="${CONCURRENCY:?CONCURRENCY is required}"

python3 tools/hunyuan_image3_it2i_npu_experiment.py run \
  --image-path "$IMAGE_PATH" \
  --prompt "$PROMPT" \
  --concurrency-values "$CONCURRENCY" \
  "$@"
