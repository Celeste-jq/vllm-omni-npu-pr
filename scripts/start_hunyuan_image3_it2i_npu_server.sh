#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8092}"
BASE_CONFIG="${BASE_CONFIG:-vllm_omni/deploy/hunyuan_image3_it2i_npu_aclgraph_rope_vitdp.yaml}"
BATCH_SIZE="${BATCH_SIZE:-1}"
AR_GPU_MEMORY_UTILIZATION="${AR_GPU_MEMORY_UTILIZATION:-0.78}"
DIT_GPU_MEMORY_UTILIZATION="${DIT_GPU_MEMORY_UTILIZATION:-0.66}"
CONFIG_OUT_DIR="${CONFIG_OUT_DIR:-/tmp/hunyuan_image3_it2i_npu_configs}"
LOG_DIR="${LOG_DIR:-/tmp/hunyuan_image3_it2i_npu_logs}"

mkdir -p "$CONFIG_OUT_DIR" "$LOG_DIR"

RENDERED_CONFIG="$CONFIG_OUT_DIR/hunyuan_image3_it2i_npu_b${BATCH_SIZE}_ar${AR_GPU_MEMORY_UTILIZATION}_dit${DIT_GPU_MEMORY_UTILIZATION}.yaml"
LOG_FILE="$LOG_DIR/server_${PORT}_b${BATCH_SIZE}.log"

export TASKQUEUEENABLE=1

python3 tools/hunyuan_image3_it2i_npu_experiment.py render-config \
  --base-config "$BASE_CONFIG" \
  --output-path "$RENDERED_CONFIG" \
  --batch-size "$BATCH_SIZE" \
  --ar-gpu-memory-utilization "$AR_GPU_MEMORY_UTILIZATION" \
  --dit-gpu-memory-utilization "$DIT_GPU_MEMORY_UTILIZATION"

echo "[info] rendered_config=$RENDERED_CONFIG"
echo "[info] log_file=$LOG_FILE"
echo "[info] batch_size=$BATCH_SIZE max_num_seqs=$BATCH_SIZE"
echo "[info] ar_gpu_memory_utilization=$AR_GPU_MEMORY_UTILIZATION"
echo "[info] dit_gpu_memory_utilization=$DIT_GPU_MEMORY_UTILIZATION"
echo "[info] port=$PORT"
echo "[case] model=$MODEL host=$HOST port=$PORT base_config=$BASE_CONFIG"
echo "[case] fixed_features=aclgraph,RoPE,AR_ViT_DP"
echo "[case] cudagraph_capture_sizes are derived around batch_size=$BATCH_SIZE"

exec vllm serve "$MODEL" \
  --omni \
  --host "$HOST" \
  --port "$PORT" \
  --deploy-config "$RENDERED_CONFIG" \
  2>&1 | tee "$LOG_FILE"
