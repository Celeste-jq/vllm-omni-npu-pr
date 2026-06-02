#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TASKQUEUEENABLE=1

python3 tools/hunyuan_image3_it2i_npu_experiment.py prepare-configs "$@"
