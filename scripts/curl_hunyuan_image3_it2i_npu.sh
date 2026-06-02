#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${MODE:-single}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8092}"
MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
IMAGE_PATH="${IMAGE_PATH:?IMAGE_PATH is required}"
PROMPT="${PROMPT:-Turn the product into a clean studio advertisement.}"
SIZE="${SIZE:-1024x1024}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-png}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
SEED="${SEED:-42}"
RESULT_DIR="${RESULT_DIR:-/tmp/hunyuan_image3_it2i_manual}"
CONCURRENCY="${CONCURRENCY:-1}"
REQUESTS="${REQUESTS:-$CONCURRENCY}"

mkdir -p "$RESULT_DIR"

endpoint="http://${HOST}:${PORT}/v1/images/edits"

print_case_header() {
  echo "[case] mode=$MODE host=$HOST port=$PORT"
  echo "[case] model=$MODEL image_path=$IMAGE_PATH"
  echo "[case] prompt=$PROMPT"
  echo "[case] size=$SIZE output_format=$OUTPUT_FORMAT"
  echo "[case] num_inference_steps=$NUM_INFERENCE_STEPS guidance_scale=$GUIDANCE_SCALE seed=$SEED"
  if [[ "$MODE" == "parallel" ]]; then
    echo "[case] concurrency=$CONCURRENCY requests=$REQUESTS"
  fi
}

print_single_summary() {
  local output_json="$1"
  python3 - "$output_json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
stage = data.get("stage_durations", {}) or {}
peak = data.get("peak_memory_mb", 0.0)
cot = data.get("cot_output") or ""
images = len(data.get("data", []) or [])
print(f"[result] status=success images={images} peak_memory_mb={peak}")
print(f"[result] stage_durations={json.dumps(stage, ensure_ascii=True)}")
print(f"[result] cot_output_chars={len(cot)}")
PY
}

print_stream_summary() {
  local output_txt="$1"
  python3 - "$output_txt" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
ttft_chunks = 0
final_stage = {}
peak = 0.0
for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    if not line.startswith("data: "):
        continue
    payload = line[6:]
    if payload == "[DONE]":
        continue
    obj = json.loads(payload)
    if obj.get("type") == "ar_delta":
        ttft_chunks += 1
    elif obj.get("type") == "image":
        final_stage = obj.get("stage_durations", {}) or {}
        peak = obj.get("peak_memory_mb", 0.0) or 0.0
print(f"[result] status=success ar_delta_chunks={ttft_chunks} peak_memory_mb={peak}")
print(f"[result] final_stage_durations={json.dumps(final_stage, ensure_ascii=True)}")
PY
}

print_parallel_summary() {
  local result_dir="$1"
  python3 - "$result_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.glob("result_parallel_*.json"))
success = 0
fail = 0
peaks = []
for path in files:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("data"):
            success += 1
            peak = data.get("peak_memory_mb")
            if peak is not None:
                peaks.append(float(peak))
        else:
            fail += 1
    except Exception:
        fail += 1
peak_max = max(peaks) if peaks else 0.0
print(f"[result] total_requests={len(files)} success={success} fail={fail} peak_memory_mb_max={peak_max}")
PY
}

run_single() {
  local output_json="$RESULT_DIR/result_single.json"
  print_case_header
  curl -sS -X POST "$endpoint" \
    -F "model=$MODEL" \
    -F "image=@$IMAGE_PATH" \
    -F "prompt=$PROMPT" \
    -F "size=$SIZE" \
    -F "output_format=$OUTPUT_FORMAT" \
    -F "num_inference_steps=$NUM_INFERENCE_STEPS" \
    -F "guidance_scale=$GUIDANCE_SCALE" \
    -F "seed=$SEED" \
    > "$output_json"
  echo "[info] response_json=$output_json"
  print_single_summary "$output_json"
  echo "[info] dump image:"
  echo "jq -r '.data[0].b64_json' $output_json | base64 -d > $RESULT_DIR/output_single.${OUTPUT_FORMAT}"
}

run_stream() {
  local output_txt="$RESULT_DIR/result_stream.txt"
  print_case_header
  curl -N -X POST "$endpoint" \
    -F "model=$MODEL" \
    -F "image=@$IMAGE_PATH" \
    -F "prompt=$PROMPT" \
    -F "size=$SIZE" \
    -F "output_format=$OUTPUT_FORMAT" \
    -F "num_inference_steps=$NUM_INFERENCE_STEPS" \
    -F "guidance_scale=$GUIDANCE_SCALE" \
    -F "seed=$SEED" \
    -F "stream=true" \
    | tee "$output_txt"
  echo
  echo "[info] stream_log=$output_txt"
  print_stream_summary "$output_txt"
}

run_parallel() {
  local total="${REQUESTS}"
  local max="${CONCURRENCY}"
  local running=0
  local i
  print_case_header
  for i in $(seq 1 "$total"); do
    (
      local seed_i=$((SEED + i))
      local output_json="$RESULT_DIR/result_parallel_${i}.json"
      curl -sS -X POST "$endpoint" \
        -F "model=$MODEL" \
        -F "image=@$IMAGE_PATH" \
        -F "prompt=$PROMPT" \
        -F "size=$SIZE" \
        -F "output_format=$OUTPUT_FORMAT" \
        -F "num_inference_steps=$NUM_INFERENCE_STEPS" \
        -F "guidance_scale=$GUIDANCE_SCALE" \
        -F "seed=$seed_i" \
        > "$output_json"
      echo "[done] request=$i output=$output_json"
    ) &
    running=$((running + 1))
    if [[ "$running" -ge "$max" ]]; then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
  print_parallel_summary "$RESULT_DIR"
}

case "$MODE" in
  single)
    run_single
    ;;
  stream)
    run_stream
    ;;
  parallel)
    run_parallel
    ;;
  *)
    echo "[error] MODE must be one of: single, stream, parallel" >&2
    exit 2
    ;;
esac
