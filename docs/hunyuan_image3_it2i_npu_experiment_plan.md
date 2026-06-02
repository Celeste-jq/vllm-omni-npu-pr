# HunyuanImage3 IT2I NPU Experiment Plan

## Scope

This document defines the NPU-side experiment plan for HunyuanImage-3.0 `it2i` benchmarking in the `hunyuanimag3_ar_test` branch.

The experiment scope is intentionally fixed:

- Single service deployment only
- Task is `it2i` only
- Features always enabled:
  - `aclgraph`
  - `RoPE`
  - `AR ViT DP`

The variables under test are:

- `batch_size`
- `max_num_seqs`
- request concurrency
- `gpu_memory_utilization`

In this experiment suite:

- `batch_size == max_num_seqs == concurrency`

## Files

Main experiment assets:

- Experiment runner:
  - `tools/hunyuan_image3_it2i_npu_experiment.py`
- Base deploy config:
  - `vllm_omni/deploy/hunyuan_image3_it2i_npu_aclgraph_rope_vitdp.yaml`
- Memory presets:
  - `vllm_omni/deploy/hunyuan_image3_it2i_npu_aclgraph_rope_vitdp_mem075.yaml`
  - `vllm_omni/deploy/hunyuan_image3_it2i_npu_aclgraph_rope_vitdp_mem080.yaml`
- Convenience scripts:
  - `scripts/prepare_hunyuan_image3_it2i_npu_configs.sh`
  - `scripts/run_hunyuan_image3_it2i_npu_probe.sh`
  - `scripts/run_hunyuan_image3_it2i_npu_single.sh`
  - `scripts/run_hunyuan_image3_it2i_npu_sweep.sh`

## Runtime Requirements

Run this on the NPU server with an environment that already supports:

- `vllm serve --omni`
- `torch`
- `aiohttp`
- `PyYAML`
- the HunyuanImage3 model and NPU runtime stack

The scripts also expect:

- `TASKQUEUEENABLE=1`
- NPU-visible devices mapped consistently with the deploy YAMLs

## Deployment Principles

The deploy config is generated per experiment point.

Fixed behavior:

- stage 0 is AR
- stage 1 is DiT
- `mm_encoder_tp_mode: data` enables AR ViT DP
- `hf_overrides.rope_parameters` keeps HunyuanImage3 RoPE settings
- `compilation_config.cudagraph_mode: FULL_DECODE_ONLY`
- stage 0 uses `enforce_eager: false`

Per experiment point:

- `batch_size` is chosen first by the experiment runner
- `max_num_seqs` for both stage 0 and stage 1 is then rewritten to that value
- `edges[].max_inflight` is aligned to the same value
- `cudagraph_capture_sizes` is generated around the same target concurrency

So the control direction is:

1. choose experiment `batch_size`
2. rewrite YAML `max_num_seqs`
3. launch service

## KV Cache Based Concurrency Estimation

The runner logs KV cache information from rank 0 using:

- `[kv-cache-profile] ... num_blocks=...`

The experiment suite uses:

- `blocks_per_request = ceil((input_tokens + output_tokens) / 128)`
- `max_concurrency = num_blocks // blocks_per_request`

This is used only to estimate a target concurrency region.

The actual sweep keeps:

- small values
- powers of two
- several values around the target region

## Metrics

The suite records the following metrics.

Client-side:

- `TTFT`
  - defined as time from request start to first `ar_delta` chunk from `/v1/images/edits?stream=true`
- `E2E`
  - defined as time from request start to the final `image` chunk
- request success/failure
- request throughput
- AR delta count
- AR text character count

Server-side:

- `num_blocks`
- `stage_durations`
- `peak_memory_mb`

Derived summary metrics:

- `ttft_mean_s`
- `ttft_p50_s`
- `ttft_p95_s`
- `e2e_mean_s`
- `e2e_p50_s`
- `e2e_p95_s`
- `ar_stage_mean_s`
- `dit_stage_mean_s`
- `peak_memory_mb_max`
- `request_throughput_qps`

## Result Artifacts

Each run writes results under the chosen results directory.

Important outputs:

- `summary.csv`
  - one row per experiment point
- `requests.jsonl`
  - one record per request
- `generated_configs/`
  - the exact YAML used for each point
- `server_logs/`
  - raw server logs including `kv-cache-profile`

## Recommended Execution Flow

### 1. Prepare configs

```bash
bash scripts/prepare_hunyuan_image3_it2i_npu_configs.sh
```

This materializes the preset YAMLs and confirms the config generation path is usable.

### 2. Probe run

```bash
IMAGE_PATH=/path/to/input.png \
PROMPT='Turn the product into a clean studio advertisement.' \
bash scripts/run_hunyuan_image3_it2i_npu_probe.sh
```

Purpose:

- verify the NPU service can boot
- verify the `it2i` request path works
- capture initial `num_blocks`
- validate metrics and logging shape

This run uses:

- `batch_size = 1`
- `max_num_seqs = 1`

### 3. Single-point validation

```bash
IMAGE_PATH=/path/to/input.png \
PROMPT='Turn the product into a clean studio advertisement.' \
CONCURRENCY=8 \
bash scripts/run_hunyuan_image3_it2i_npu_single.sh
```

Purpose:

- validate one chosen experiment point
- inspect stage timings and memory before full sweep

This run uses:

- `batch_size = 8`
- `max_num_seqs = 8`

### 4. Full sweep

```bash
IMAGE_PATH=/path/to/input.png \
PROMPT='Turn the product into a clean studio advertisement.' \
bash scripts/run_hunyuan_image3_it2i_npu_sweep.sh
```

Purpose:

- run the generated concurrency matrix
- compare memory presets
- locate the best throughput/latency operating region

## How To Specify Batch Size

Batch size is not independently configured from concurrency in this suite.

Use either:

- single-point mode:

```bash
CONCURRENCY=16 bash scripts/run_hunyuan_image3_it2i_npu_single.sh
```

- explicit sweep list:

```bash
IMAGE_PATH=/path/to/input.png \
PROMPT='...' \
bash scripts/run_hunyuan_image3_it2i_npu_sweep.sh \
  --concurrency-values 1,2,4,8,12,16
```

In both cases:

- `batch_size = concurrency`
- `max_num_seqs = concurrency`

## Suggested First Sweep

If you want a controlled first pass before using automatic estimation, start with:

```bash
IMAGE_PATH=/path/to/input.png \
PROMPT='Turn the product into a clean studio advertisement.' \
bash scripts/run_hunyuan_image3_it2i_npu_sweep.sh \
  --concurrency-values 1,2,4,8,12,16,20,24,32
```

Then inspect:

- `summary.csv`
- `server_logs/*.log`

Focus on:

- where `TTFT` starts to jump
- whether `DiT` stage time dominates
- whether throughput still scales
- whether `peak_memory_mb` approaches the limit

## What To Compare

For each point, compare:

- `TTFT` vs concurrency
- `E2E` vs concurrency
- `AR stage` vs `DiT stage`
- throughput vs concurrency
- memory preset vs latency stability

A good operating point is typically where:

- throughput still improves
- `TTFT` and `E2E` have not sharply regressed
- `DiT` is not the obvious bottleneck
- memory headroom remains acceptable

## Validation Status In This Branch

What was verified locally in the coding environment:

- Python syntax for modified and new Python files
- shell syntax for the new command scripts
- smoke validation for:
  - concurrency point generation
  - `num_blocks` log parsing
  - max concurrency estimation formula

What was not fully verified locally:

- full `pytest` execution
- real NPU service boot
- actual `it2i` request execution against NPU hardware

Reason:

- the local coding environment did not have the full NPU runtime dependencies available

So the final verification must be performed on the NPU server.
