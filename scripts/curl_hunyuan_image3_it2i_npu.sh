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
  local e2e_s="$2"
  python3 - "$output_json" "$e2e_s" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
e2e_s = float(sys.argv[2])
data = json.loads(path.read_text(encoding="utf-8"))
stage = data.get("stage_durations", {}) or {}
peak = data.get("peak_memory_mb", 0.0)
cot = data.get("cot_output") or ""
images = len(data.get("data", []) or [])
print(f"[result] status=success images={images} e2e_s={e2e_s:.3f} peak_memory_mb={peak}")
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
ttft_s = 0.0
e2e_s = 0.0
final_stage = {}
peak = 0.0
ar_text_chars = 0
for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    if "\t" not in line:
        continue
    ts_str, raw = line.split("\t", 1)
    try:
        elapsed_s = float(ts_str)
    except ValueError:
        continue
    if not raw.startswith("data: "):
        continue
    payload = raw[6:]
    if payload == "[DONE]":
        if e2e_s == 0.0:
            e2e_s = elapsed_s
        continue
    obj = json.loads(payload)
    if obj.get("type") == "ar_delta":
        ttft_chunks += 1
        ar_text_chars += len(obj.get("delta", ""))
        if ttft_s == 0.0:
            ttft_s = elapsed_s
    elif obj.get("type") == "image":
        e2e_s = elapsed_s
        final_stage = obj.get("stage_durations", {}) or {}
        peak = obj.get("peak_memory_mb", 0.0) or 0.0
print(
    f"[result] status=success ttft_s={ttft_s:.3f} e2e_s={e2e_s:.3f} "
    f"ar_delta_chunks={ttft_chunks} ar_text_chars={ar_text_chars} peak_memory_mb={peak}"
)
print(f"[result] final_stage_durations={json.dumps(final_stage, ensure_ascii=True)}")
PY
}

print_parallel_summary() {
  local result_dir="$1"
  python3 - "$result_dir" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.glob("result_parallel_*.txt"))

def percentile(values, q):
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values = sorted(float(v) for v in values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)

def mean(values):
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else 0.0

def find_stage_value(stage, names):
    for name in names:
        value = stage.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0

success = 0
fail = 0
peaks = []
ttfts = []
e2es = []
ar_stages = []
dit_stages = []
for path in files:
    try:
        ttft_s = 0.0
        e2e_s = 0.0
        final_stage = {}
        peak = 0.0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "\t" not in line:
                continue
            ts_str, raw = line.split("\t", 1)
            try:
                elapsed_s = float(ts_str)
            except ValueError:
                continue
            if not raw.startswith("data: "):
                continue
            payload = raw[6:]
            if payload == "[DONE]":
                if e2e_s == 0.0:
                    e2e_s = elapsed_s
                continue
            obj = json.loads(payload)
            if obj.get("type") == "ar_delta" and ttft_s == 0.0:
                ttft_s = elapsed_s
            elif obj.get("type") == "image":
                e2e_s = elapsed_s
                final_stage = obj.get("stage_durations", {}) or {}
                peak = float(obj.get("peak_memory_mb", 0.0) or 0.0)
        if e2e_s > 0.0:
            success += 1
            if ttft_s > 0.0:
                ttfts.append(ttft_s)
            e2es.append(e2e_s)
            if peak > 0.0:
                peaks.append(peak)
            ar_stages.append(find_stage_value(final_stage, ("stage_0", "ar", "prefill", "text", "llm")))
            dit_stages.append(find_stage_value(final_stage, ("stage_1", "dit", "diffusion", "image")))
        else:
            fail += 1
    except Exception:
        fail += 1

peak_max = max(peaks) if peaks else 0.0
print(f"[result] total_requests={len(files)} success={success} fail={fail} success_rate={success / len(files):.3f}" if files else "[result] total_requests=0 success=0 fail=0 success_rate=0.000")
print(
    f"[result] ttft_mean_s={mean(ttfts):.3f} ttft_p50_s={percentile(ttfts, 0.50):.3f} "
    f"ttft_p95_s={percentile(ttfts, 0.95):.3f}"
)
print(
    f"[result] e2e_mean_s={mean(e2es):.3f} e2e_p50_s={percentile(e2es, 0.50):.3f} "
    f"e2e_p95_s={percentile(e2es, 0.95):.3f}"
)
print(
    f"[result] ar_stage_mean_s={mean(ar_stages):.3f} dit_stage_mean_s={mean(dit_stages):.3f} "
    f"peak_memory_mb_max={peak_max:.3f}"
)
PY
}

timestamp_stream_to_file() {
  local output_txt="$1"
  python3 - "$output_txt" <<'PY'
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
started = time.perf_counter()
with path.open("w", encoding="utf-8") as f:
    for raw in sys.stdin:
        elapsed = time.perf_counter() - started
        f.write(f"{elapsed:.6f}\t{raw}")
        f.flush()
        sys.stdout.write(raw)
        sys.stdout.flush()
PY
}

run_single() {
  local output_json="$RESULT_DIR/result_single.json"
  local started_s ended_s e2e_s
  print_case_header
  started_s="$(python3 -c 'import time; print(time.perf_counter())')"
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
  ended_s="$(python3 -c 'import time; print(time.perf_counter())')"
  e2e_s="$(python3 - "$started_s" "$ended_s" <<'PY'
import sys
print(f"{float(sys.argv[2]) - float(sys.argv[1]):.6f}")
PY
)"
  echo "[info] response_json=$output_json"
  print_single_summary "$output_json" "$e2e_s"
  echo "[info] dump image:"
  echo "jq -r '.data[0].b64_json' $output_json | base64 -d > $RESULT_DIR/output_single.${OUTPUT_FORMAT}"
}

run_stream() {
  local output_txt="$RESULT_DIR/result_stream.txt"
  print_case_header
  curl -sN -X POST "$endpoint" \
    -F "model=$MODEL" \
    -F "image=@$IMAGE_PATH" \
    -F "prompt=$PROMPT" \
    -F "size=$SIZE" \
    -F "output_format=$OUTPUT_FORMAT" \
    -F "num_inference_steps=$NUM_INFERENCE_STEPS" \
    -F "guidance_scale=$GUIDANCE_SCALE" \
    -F "seed=$SEED" \
    -F "stream=true" \
    | timestamp_stream_to_file "$output_txt"
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
      local output_txt="$RESULT_DIR/result_parallel_${i}.txt"
      curl -sN -X POST "$endpoint" \
        -F "model=$MODEL" \
        -F "image=@$IMAGE_PATH" \
        -F "prompt=$PROMPT" \
        -F "size=$SIZE" \
        -F "output_format=$OUTPUT_FORMAT" \
        -F "num_inference_steps=$NUM_INFERENCE_STEPS" \
        -F "guidance_scale=$GUIDANCE_SCALE" \
        -F "seed=$seed_i" \
        -F "stream=true" \
        | timestamp_stream_to_file "$output_txt" >/dev/null
      echo "[done] request=$i output=$output_txt"
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
