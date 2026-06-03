# SPDX-License-Identifier: Apache-2.0
"""Run one HunyuanImage3 NPU IT2I batch test and print the metrics table.

The script intentionally exposes only batch_size as the experiment variable.
It sends batch_size streaming /v1/images/edits requests at the same time so the
server can admit one full batch, then prints the current YAML config and result
metrics needed for AR/DiT balance analysis.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY_CONFIG = REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3_it2i_npu_aclgraph_rope_vitdp.yaml"
KV_CACHE_PROFILE_PATTERNS = (
    re.compile(
        r"\[kv-cache-profile\].*?\brank=(?P<rank>\d+).*?"
        r"\bstage_id=(?P<stage_id>[^ ]+).*?"
        r"\bnum_blocks=(?P<num_blocks>\d+).*?"
        r"\bblock_size=(?P<block_size>\d+)"
    ),
    re.compile(
        r"\[kv-cache-profile\].*?\brank=(?P<rank>\d+).*?"
        r"\bnum_blocks=(?P<num_blocks>\d+).*?"
        r"\bblock_size=(?P<block_size>\d+)"
    ),
)
KV_CACHE_NUM_BLOCK_PATTERNS = (
    re.compile(r"\[kv-cache-profile\].*?\bnum_blocks=(\d+)\b"),
    re.compile(r"\bnum_blocks=(\d+)\b"),
    re.compile(r"\bnum_blocks:\s*(\d+)\b"),
    re.compile(r"\bnum_gpu_blocks=(\d+)\b"),
)
VIT_DP_SHARD_PATTERN = re.compile(
    r"HunyuanImage3 AR ViT DP shard:.*?\btp_rank=(\d+).*?\btp_size=(\d+).*?"
    r"\bglobal_batch=(\d+).*?\blocal_count=(\d+).*?\blocal_batch_size=(\d+)"
)
VIT_DP_TIMING_PATTERN = re.compile(
    r"HunyuanImage3 AR ViT timing:.*?\buse_data_parallel=(True|False).*?"
    r"\btp_rank=(\d+).*?\btp_size=(\d+).*?\bglobal_batch=(\d+).*?"
    r"\blocal_count=(\d+).*?\bforward_ms=([0-9.]+).*?"
    r"\ball_gather_ms=([0-9.]+).*?\btotal_ms=([0-9.]+)"
)
VIT_DP_ENCODE_STATE_PATTERN = re.compile(
    r"HunyuanImage3 AR ViT encode state:.*?\buse_data_parallel=(True|False).*?"
    r"\btp_rank=(\d+).*?\btp_size=(\d+).*?\bbatch_size=(\d+)"
)
AR_GENERATED_TOKENS_PATTERN = re.compile(r"\[ar2diffusion\] Request \d+: AR generated (\d+) tokens")
HUNYUAN_IMAGE3_INPUT_TOKENS_PATTERN = re.compile(
    r"\[HunyuanImage3 token-profile\].*?\bprompt_tokens=(\d+)\b"
)


@dataclass
class RequestMetric:
    request_index: int
    success: bool
    http_status: int
    first_event_s: float
    ttft_s: float
    e2e_s: float
    ar_delta_count: int
    ar_text_chars: int
    stage_durations: dict[str, float]
    event_types: list[str]
    peak_memory_mb: float
    image_paths: list[str]
    error: str


class SSEDecoder:
    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, chunk: bytes) -> list[str]:
        self.buffer += chunk.decode("utf-8", errors="replace")
        events: list[str] = []
        while "\n\n" in self.buffer:
            raw_event, self.buffer = self.buffer.split("\n\n", 1)
            data_lines = [line[6:] for line in raw_event.splitlines() if line.startswith("data: ")]
            if data_lines:
                events.append("\n".join(data_lines))
        return events


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required: pip install pyyaml") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (values[hi] - values[lo]) * (idx - lo))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def collect_kv_cache_log_candidates(log_text: str, *, limit: int = 20) -> list[str]:
    candidates: list[str] = []
    for line in log_text.splitlines():
        lower_line = line.lower()
        if "kv-cache-profile" in lower_line or "num_blocks" in lower_line or "num_gpu_blocks" in lower_line:
            candidates.append(line[-500:])
            if len(candidates) >= limit:
                break
    return candidates


def parse_kv_cache_profile(log_text: str) -> dict[str, Any]:
    fallback: dict[str, Any] = {}
    for pattern in KV_CACHE_PROFILE_PATTERNS:
        for match in pattern.finditer(log_text):
            profile = {
                "num_blocks": int(match.group("num_blocks")),
                "block_size": int(match.group("block_size")),
                "kv_cache_profile_source": "kv-cache-profile",
                "kv_cache_profile_found": True,
                "kv_cache_log_candidates": collect_kv_cache_log_candidates(log_text),
            }
            stage_id = match.groupdict().get("stage_id")
            if stage_id in ("0", "None") or stage_id is None:
                return profile
            fallback = profile
    if fallback:
        return fallback
    for pattern in KV_CACHE_NUM_BLOCK_PATTERNS:
        match = pattern.search(log_text)
        if match:
            return {
                "num_blocks": int(match.group(1)),
                "block_size": 128,
                "kv_cache_profile_source": "fallback-pattern",
                "kv_cache_profile_found": True,
                "kv_cache_log_candidates": collect_kv_cache_log_candidates(log_text),
            }
    return {
        "num_blocks": 0,
        "block_size": 128,
        "kv_cache_profile_source": "missing",
        "kv_cache_profile_found": False,
        "kv_cache_log_candidates": collect_kv_cache_log_candidates(log_text),
    }


def merge_kv_cache_profile(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    if current.get("num_blocks", 0) > 0:
        return current
    return previous


def print_kv_cache_profile(profile: dict[str, Any], *, phase: str) -> None:
    print(
        f"[kv-cache] phase={phase} num_blocks={profile.get('num_blocks', 0)} "
        f"block_size={profile.get('block_size', 128)} "
        f"found={str(bool(profile.get('kv_cache_profile_found'))).lower()} "
        f"source={profile.get('kv_cache_profile_source', 'missing')}"
    )
    if not profile.get("num_blocks", 0):
        print("[warning] num_blocks missing; max_concurrency will be 0. Check server_log for [kv-cache-profile].")


def parse_vit_dp_batch_logs(log_text: str) -> dict[str, Any]:
    shard_records = [
        {
            "tp_rank": int(match.group(1)),
            "tp_size": int(match.group(2)),
            "global_batch": int(match.group(3)),
            "local_count": int(match.group(4)),
            "local_batch_size": int(match.group(5)),
        }
        for match in VIT_DP_SHARD_PATTERN.finditer(log_text)
    ]
    timing_records = [
        {
            "use_data_parallel": match.group(1) == "True",
            "tp_rank": int(match.group(2)),
            "tp_size": int(match.group(3)),
            "global_batch": int(match.group(4)),
            "local_count": int(match.group(5)),
            "forward_ms": float(match.group(6)),
            "all_gather_ms": float(match.group(7)),
            "total_ms": float(match.group(8)),
        }
        for match in VIT_DP_TIMING_PATTERN.finditer(log_text)
    ]
    encode_state_records = [
        {
            "use_data_parallel": match.group(1) == "True",
            "tp_rank": int(match.group(2)),
            "tp_size": int(match.group(3)),
            "global_batch": int(match.group(4)),
        }
        for match in VIT_DP_ENCODE_STATE_PATTERN.finditer(log_text)
    ]
    global_batches = [record["global_batch"] for record in shard_records]
    timing_global_batches = [record["global_batch"] for record in timing_records]
    encode_state_global_batches = [record["global_batch"] for record in encode_state_records]
    local_counts = [record["local_count"] for record in shard_records]
    timing_totals = [record["total_ms"] for record in timing_records]
    fallback_global_batches = global_batches or timing_global_batches or encode_state_global_batches
    return {
        "vit_dp_request_shard_log_count": len(shard_records),
        "vit_dp_request_timing_log_count": len(timing_records),
        "vit_dp_request_encode_state_log_count": len(encode_state_records),
        "vit_dp_request_global_batch_max": max(fallback_global_batches, default=0),
        "vit_dp_request_global_batches": sorted(set(fallback_global_batches)),
        "vit_dp_request_local_counts": local_counts,
        "vit_dp_request_nonzero_local_count_logs": sum(1 for value in local_counts if value > 0),
        "vit_dp_request_timing_total_ms_mean": mean(timing_totals),
        "vit_dp_request_timing_total_ms_max": max(timing_totals) if timing_totals else 0.0,
    }


def parse_ar_generated_tokens(log_text: str) -> list[int]:
    return [int(match.group(1)) for match in AR_GENERATED_TOKENS_PATTERN.finditer(log_text)]


def parse_input_tokens(log_text: str) -> list[int]:
    return [int(match.group(1)) for match in HUNYUAN_IMAGE3_INPUT_TOKENS_PATTERN.finditer(log_text)]


def duration_value_seconds(key: str, value: float) -> float:
    value = float(value)
    return value / 1000.0 if key.endswith("_ms") else value


def find_stage_duration_seconds(stage_durations: dict[str, float], candidates: tuple[str, ...]) -> float:
    for key in candidates:
        value = stage_durations.get(key)
        if value is not None:
            return duration_value_seconds(key, float(value))
    return 0.0


def union_stage_duration_keys(metrics: list[RequestMetric]) -> list[str]:
    keys: set[str] = set()
    for metric in metrics:
        keys.update(metric.stage_durations.keys())
    return sorted(keys)


def union_event_types(metrics: list[RequestMetric]) -> list[str]:
    event_types: set[str] = set()
    for metric in metrics:
        event_types.update(metric.event_types)
    return sorted(event_types)


def config_summary(config: dict[str, Any]) -> dict[str, Any]:
    stages = config.get("stages") or []
    ar_stage = stages[0] if len(stages) > 0 and isinstance(stages[0], dict) else {}
    dit_stage = stages[1] if len(stages) > 1 and isinstance(stages[1], dict) else {}
    edges = config.get("edges") or []
    edge = edges[0] if edges and isinstance(edges[0], dict) else {}
    rope_parameters = (ar_stage.get("hf_overrides") or {}).get("rope_parameters") or {}
    compilation = ar_stage.get("compilation_config") or {}
    ar_sampling = ar_stage.get("default_sampling_params") or {}
    return {
        "pipeline": config.get("pipeline", ""),
        "ar_max_num_seqs": ar_stage.get("max_num_seqs"),
        "dit_max_num_seqs": dit_stage.get("max_num_seqs"),
        "edge_max_inflight": edge.get("max_inflight"),
        "ar_gpu_memory_utilization": ar_stage.get("gpu_memory_utilization"),
        "dit_gpu_memory_utilization": dit_stage.get("gpu_memory_utilization"),
        "ar_max_num_batched_tokens": ar_stage.get("max_num_batched_tokens"),
        "dit_max_num_batched_tokens": dit_stage.get("max_num_batched_tokens"),
        "cudagraph_mode": compilation.get("cudagraph_mode"),
        "cudagraph_capture_sizes": compilation.get("cudagraph_capture_sizes"),
        "enforce_eager": ar_stage.get("enforce_eager"),
        "is_comprehension": ar_stage.get("is_comprehension"),
        "rope_enabled": bool(rope_parameters),
        "rope_parameters": rope_parameters,
        "ar_vit_dp_enabled": ar_stage.get("mm_encoder_tp_mode") == "data",
        "mm_encoder_tp_mode": ar_stage.get("mm_encoder_tp_mode"),
        "ar_profiler_config": ar_stage.get("profiler_config"),
        "ar_max_tokens": ar_sampling.get("max_tokens"),
    }


def build_ar_profiler_config(args: argparse.Namespace, profiler_dir: Path) -> dict[str, Any]:
    return {
        "profiler": "torch",
        "torch_profiler_dir": str(profiler_dir),
        "torch_profiler_record_shapes": bool(args.profiler_record_shapes),
        "torch_profiler_with_stack": bool(args.profiler_with_stack),
        "torch_profiler_with_memory": bool(args.profiler_with_memory),
        "torch_profiler_with_flops": bool(args.profiler_with_flops),
        "torch_profiler_use_gzip": bool(args.profiler_use_gzip),
        "torch_profiler_dump_cuda_time_total": bool(args.profiler_dump_cuda_time_total),
    }


def prepare_deploy_config_for_run(
    *,
    config: dict[str, Any],
    source_path: Path,
    output_dir: Path,
    profile_ar: bool,
    profiler_config: dict[str, Any] | None,
) -> Path:
    if not profile_ar:
        return source_path
    run_config = json.loads(json.dumps(config))
    stages = run_config.get("stages") or []
    if not stages or not isinstance(stages[0], dict):
        raise ValueError("deploy config must contain stage 0 for AR profiling")
    stages[0]["profiler_config"] = profiler_config
    rendered_path = output_dir / "deploy_with_ar_profiler.yaml"
    dump_yaml(rendered_path, run_config)
    return rendered_path


def infer_batch_size(config: dict[str, Any]) -> int:
    summary = config_summary(config)
    for key in ("ar_max_num_seqs", "dit_max_num_seqs", "edge_max_inflight"):
        value = summary.get(key)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("unable to infer batch_size from YAML max_num_seqs/max_inflight")


def build_server_env() -> dict[str, str]:
    env = os.environ.copy()
    # torch_npu graph capture checks TASK_QUEUE_ENABLE, while some scripts and
    # docs use TASKQUEUEENABLE. Set both so parent-shell values cannot leak in.
    env["TASK_QUEUE_ENABLE"] = "1"
    env["TASKQUEUEENABLE"] = "1"
    return env


class ManagedServer:
    def __init__(self, *, model: str, deploy_config: Path, host: str, port: int, log_file: Path) -> None:
        self.model = model
        self.deploy_config = deploy_config
        self.host = host
        self.port = port
        self.log_file = log_file
        self.process: subprocess.Popen[str] | None = None
        self._log_thread: threading.Thread | None = None

    def start(self) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        env = build_server_env()
        cmd = [
            "vllm",
            "serve",
            self.model,
            "--omni",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--deploy-config",
            str(self.deploy_config),
        ]
        print("[server] " + " ".join(cmd), flush=True)
        print(
            "[server-env] "
            f"TASK_QUEUE_ENABLE={env.get('TASK_QUEUE_ENABLE')} "
            f"TASKQUEUEENABLE={env.get('TASKQUEUEENABLE')}",
            flush=True,
        )
        self.process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(REPO_ROOT),
        )
        self._log_thread = threading.Thread(target=self._stream_logs, name="server-log-stream", daemon=True)
        self._log_thread.start()

    def _stream_logs(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        with self.log_file.open("w", encoding="utf-8") as log_handle:
            for line in self.process.stdout:
                log_handle.write(line)
                log_handle.flush()
                print(line, end="", flush=True)

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self._log_thread is not None:
            self._log_thread.join(timeout=5)

    def read_log(self) -> str:
        if not self.log_file.exists():
            return ""
        return self.log_file.read_text(encoding="utf-8", errors="replace")

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        health_url = f"http://{self.host}:{self.port}/health"
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"server exited early, see log: {self.log_file}")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:  # noqa: S310
                    if response.status == 200:
                        return
            except Exception:
                time.sleep(2)
        raise TimeoutError(f"server did not become ready within {timeout_s}s, see log: {self.log_file}")


async def send_one_request(
    session: Any,
    *,
    api_url: str,
    model: str | None,
    image_path: Path,
    prompt: str,
    size: str,
    output_format: str,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    seed: int,
    request_index: int,
    timeout_s: float,
    image_dir: Path | None,
) -> RequestMetric:
    import aiohttp

    form = aiohttp.FormData()
    if model is not None:
        form.add_field("model", model)
    form.add_field("image", image_path.read_bytes(), filename=image_path.name, content_type="image/png")
    form.add_field("prompt", prompt)
    form.add_field("size", size)
    form.add_field("output_format", output_format)
    if num_inference_steps is not None:
        form.add_field("num_inference_steps", str(num_inference_steps))
    if guidance_scale is not None:
        form.add_field("guidance_scale", str(guidance_scale))
    form.add_field("seed", str(seed + request_index))
    form.add_field("stream", "true")

    decoder = SSEDecoder()
    started = time.perf_counter()
    first_event_s = 0.0
    ttft_s = 0.0
    e2e_s = 0.0
    ar_delta_count = 0
    ar_text_chars = 0
    stage_durations: dict[str, float] = {}
    event_types: list[str] = []
    peak_memory_mb = 0.0
    image_paths: list[str] = []
    error = ""

    try:
        async with session.post(api_url, data=form, timeout=aiohttp.ClientTimeout(total=timeout_s)) as response:
            async for chunk in response.content.iter_any():
                for event in decoder.feed(chunk):
                    now = time.perf_counter()
                    if event == "[DONE]":
                        if e2e_s == 0.0:
                            e2e_s = now - started
                        continue
                    payload = json.loads(event)
                    if first_event_s == 0.0:
                        first_event_s = now - started
                    event_type = str(payload.get("type") or payload.get("object") or "unknown")
                    event_types.append(event_type)
                    if payload.get("type") == "ar_delta":
                        ar_delta_count += 1
                        ar_text_chars += len(payload.get("delta") or "")
                        if ttft_s == 0.0:
                            ttft_s = now - started
                    elif payload.get("type") == "image":
                        e2e_s = now - started
                        stage_durations = {
                            key: float(value) for key, value in (payload.get("stage_durations") or {}).items()
                        }
                        peak_memory_mb = float(payload.get("peak_memory_mb") or 0.0)
                        if image_dir is not None:
                            data_items = payload.get("data") or []
                            for image_index, image_item in enumerate(data_items):
                                b64_json = image_item.get("b64_json")
                                if not b64_json:
                                    continue
                                output_format = payload.get("output_format") or "png"
                                image_bytes = base64.b64decode(b64_json)
                                image_path = image_dir / f"request_{request_index:03d}_image_{image_index:02d}.{output_format}"
                                image_path.write_bytes(image_bytes)
                                image_paths.append(str(image_path))
                    elif payload.get("object") == "error":
                        error = json.dumps(payload.get("error", {}), ensure_ascii=True)
            if response.status != 200 and not error:
                error = response.reason or f"http_status={response.status}"
            return RequestMetric(
                request_index=request_index,
                success=response.status == 200 and not error and e2e_s > 0.0,
                http_status=response.status,
                first_event_s=first_event_s,
                ttft_s=ttft_s or first_event_s,
                e2e_s=e2e_s,
                ar_delta_count=ar_delta_count,
                ar_text_chars=ar_text_chars,
                stage_durations=stage_durations,
                event_types=event_types,
                peak_memory_mb=peak_memory_mb,
                image_paths=image_paths,
                error=error,
            )
    except Exception as exc:  # noqa: BLE001
        return RequestMetric(
            request_index=request_index,
            success=False,
            http_status=0,
            first_event_s=first_event_s,
            ttft_s=ttft_s,
            e2e_s=e2e_s,
            ar_delta_count=ar_delta_count,
            ar_text_chars=ar_text_chars,
            stage_durations=stage_durations,
            event_types=event_types,
            peak_memory_mb=peak_memory_mb,
            image_paths=image_paths,
            error=str(exc),
        )


async def run_batch_requests(
    args: argparse.Namespace,
    batch_size: int,
    *,
    image_dir: Path | None,
) -> tuple[float, list[RequestMetric]]:
    import aiohttp

    api_url = f"http://{args.host}:{args.port}/v1/images/edits"
    connector = aiohttp.TCPConnector(limit=batch_size, limit_per_host=batch_size)
    image_path = Path(args.image_path).expanduser().resolve()
    async with aiohttp.ClientSession(connector=connector) as session:
        started = time.perf_counter()
        metrics = await asyncio.gather(
            *[
                send_one_request(
                    session,
                    api_url=api_url,
                    model=args.model,
                    image_path=image_path,
                    prompt=args.prompt,
                    size=args.size,
                    output_format=args.output_format,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                    request_index=index,
                    timeout_s=args.request_timeout_s,
                    image_dir=image_dir,
                )
                for index in range(batch_size)
            ]
        )
        return time.perf_counter() - started, list(metrics)


def summarize_image_paths(metrics: list[RequestMetric]) -> list[str]:
    paths: list[str] = []
    for metric in metrics:
        paths.extend(metric.image_paths)
    return paths


def summarize_results(
    *,
    batch_size: int,
    warmup_runs: int,
    wall_time_s: float,
    metrics: list[RequestMetric],
    kv_cache_profile: dict[str, Any],
    input_tokens: int,
    output_tokens: int,
    observed_input_tokens: list[int],
    observed_ar_output_tokens: list[int],
    vit_dp_summary: dict[str, Any],
    saved_image_paths: list[str],
) -> dict[str, Any]:
    successes = [metric for metric in metrics if metric.success]
    first_events = [metric.first_event_s for metric in successes if metric.first_event_s > 0.0]
    ttfts = [metric.ttft_s for metric in successes if metric.ttft_s > 0.0]
    e2es = [metric.e2e_s for metric in successes if metric.e2e_s > 0.0]
    peaks = [metric.peak_memory_mb for metric in successes if metric.peak_memory_mb > 0.0]
    tpot_values_s: list[float] = []
    ttft_source = "ar_delta" if any(metric.ar_delta_count > 0 for metric in successes) else "first_event_fallback"
    ar_stages = [
        find_stage_duration_seconds(
            metric.stage_durations,
            (
                "ar_stage_0",
                "stage_0_gen_s",
                "stage_0_gen_sec",
                "stage_0_gen_ms",
                "stage_0",
                "ar",
                "prefill",
                "text",
                "llm",
            ),
        )
        for metric in successes
    ]
    dit_stages = [
        find_stage_duration_seconds(
            metric.stage_durations,
            (
                "stage_1_gen_s",
                "stage_1_gen_sec",
                "stage_1_gen_ms",
                "stage_1",
                "dit",
                "diffusion",
                "image",
            ),
        )
        for metric in successes
    ]
    block_size = kv_cache_profile.get("block_size") or 128
    num_blocks = kv_cache_profile.get("num_blocks") or 0
    observed_input_tokens_for_capacity = max(observed_input_tokens, default=0)
    input_tokens_for_capacity = observed_input_tokens_for_capacity or input_tokens
    input_tokens_for_capacity_source = (
        "server_log_token_profile"
        if observed_input_tokens_for_capacity
        else "cli_estimate_for_max_concurrency"
    )
    observed_output_tokens_for_capacity = mean([float(value) for value in observed_ar_output_tokens])
    output_tokens_for_capacity = observed_output_tokens_for_capacity or float(output_tokens)
    output_tokens_for_capacity_source = (
        "server_log_ar2diffusion_mean"
        if observed_output_tokens_for_capacity
        else "cli_estimate_for_max_concurrency"
    )
    tokens_per_request = input_tokens_for_capacity + output_tokens_for_capacity
    blocks_per_request = max(1, math.ceil(tokens_per_request / block_size))
    estimated_max_batch = num_blocks // blocks_per_request if num_blocks else 0
    result = {
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "num_requests": batch_size,
        "num_blocks": num_blocks or 0,
        "block_size": block_size,
        "kv_cache_profile_found": bool(kv_cache_profile.get("kv_cache_profile_found")),
        "kv_cache_profile_source": kv_cache_profile.get("kv_cache_profile_source", "missing"),
        "kv_cache_log_candidates": kv_cache_profile.get("kv_cache_log_candidates", []),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_tokens_source": "cli_default",
        "output_tokens_source": "cli_default",
        "input_tokens_for_capacity": input_tokens_for_capacity,
        "input_tokens_for_capacity_source": input_tokens_for_capacity_source,
        "observed_input_tokens": observed_input_tokens,
        "observed_input_tokens_mean": mean([float(value) for value in observed_input_tokens]),
        "observed_input_tokens_max": max(observed_input_tokens, default=0),
        "observed_input_tokens_source": "server_log_token_profile" if observed_input_tokens else "missing",
        "output_tokens_for_capacity_mean": observed_output_tokens_for_capacity,
        "output_tokens_for_capacity": output_tokens_for_capacity,
        "output_tokens_for_capacity_source": output_tokens_for_capacity_source,
        "observed_ar_output_tokens": observed_ar_output_tokens,
        "observed_ar_output_tokens_mean": mean([float(value) for value in observed_ar_output_tokens]),
        "observed_ar_output_tokens_max": max(observed_ar_output_tokens, default=0),
        "observed_ar_output_tokens_source": "server_log_ar2diffusion" if observed_ar_output_tokens else "missing",
        "tokens_per_request": tokens_per_request,
        "blocks_per_request": blocks_per_request,
        "max_concurrency_formula": "num_blocks // ceil((input_tokens_for_capacity + output_tokens_for_capacity_mean) / block_size)",
        "estimated_max_batch": estimated_max_batch,
        "max_concurrency": estimated_max_batch,
        "success": len(successes),
        "fail": len(metrics) - len(successes),
        "success_rate": len(successes) / len(metrics) if metrics else 0.0,
        "wall_time_s": wall_time_s,
        "throughput_qps": len(successes) / wall_time_s if wall_time_s > 0.0 else 0.0,
        "first_event_mean_s": mean(first_events),
        "first_event_p50_s": percentile(first_events, 0.50),
        "first_event_p95_s": percentile(first_events, 0.95),
        "ttft_mean_s": mean(ttfts),
        "ttft_p50_s": percentile(ttfts, 0.50),
        "ttft_p90_s": percentile(ttfts, 0.90),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "ttft_source": ttft_source,
        "tpot_p50_s": percentile(tpot_values_s, 0.50),
        "tpot_available": bool(tpot_values_s),
        "tpot_note": "N/A for current IT2I image-only responses because no per-token stream timing is returned.",
        "e2e_mean_s": mean(e2es),
        "e2e_p50_s": percentile(e2es, 0.50),
        "e2e_p95_s": percentile(e2es, 0.95),
        "ar_stage_mean_s": mean(ar_stages),
        "dit_stage_mean_s": mean(dit_stages),
        "peak_memory_mb_max": max(peaks) if peaks else 0.0,
        "ar_delta_mean": mean([float(metric.ar_delta_count) for metric in successes]),
        "ar_text_chars_mean": mean([float(metric.ar_text_chars) for metric in successes]),
        "request_throughput_qps": len(successes) / wall_time_s if wall_time_s > 0.0 else 0.0,
        "total_token_throughput_tok_s": (len(successes) / wall_time_s * tokens_per_request) if wall_time_s > 0.0 else 0.0,
        "total_token_throughput_note": "Uses real prompt_tokens and real AR generated tokens when present; otherwise falls back to CLI estimates.",
        "response_event_types": union_event_types(metrics),
        "stage_duration_keys": union_stage_duration_keys(metrics),
        "saved_image_paths": saved_image_paths,
        "first_error": next((metric.error for metric in metrics if metric.error), ""),
    }
    result.update(vit_dp_summary)
    return result


def print_case_config(
    *,
    args: argparse.Namespace,
    batch_size: int,
    deploy_config: Path,
    summary: dict[str, Any],
) -> None:
    print("\n========== case config ==========")
    print(f"deploy_config={deploy_config}")
    print(f"model={args.model}")
    print(f"image_path={args.image_path}")
    print(f"prompt_chars={len(args.prompt)}")
    print(f"batch_size={batch_size}")
    print(f"max_num_seqs(ar/dit)={summary['ar_max_num_seqs']}/{summary['dit_max_num_seqs']}")
    print(f"edge_max_inflight={summary['edge_max_inflight']}")
    print(
        "gpu_memory_utilization(ar/dit)="
        f"{summary['ar_gpu_memory_utilization']}/{summary['dit_gpu_memory_utilization']}"
    )
    print(f"max_num_batched_tokens(ar/dit)={summary['ar_max_num_batched_tokens']}/{summary['dit_max_num_batched_tokens']}")
    print(f"cudagraph_mode={summary['cudagraph_mode']}")
    print(f"cudagraph_capture_sizes={summary['cudagraph_capture_sizes']}")
    print("TASK_QUEUE_ENABLE=1")
    print("TASKQUEUEENABLE=1")
    print(f"rope_enabled={str(summary['rope_enabled']).lower()}")
    print(f"ar_vit_dp_enabled={str(summary['ar_vit_dp_enabled']).lower()}")
    print(f"mm_encoder_tp_mode={summary['mm_encoder_tp_mode']}")
    print(f"profile_ar={str(args.profile_ar).lower()}")
    if args.profile_ar:
        print(f"ar_profiler_config={summary.get('ar_profiler_config')}")
    if summary["ar_max_num_seqs"] != batch_size or summary["dit_max_num_seqs"] != batch_size:
        print(
            "[warning] YAML max_num_seqs and batch_size do not match. "
            "For this experiment, set both stage max_num_seqs values to batch_size."
        )
    if summary["edge_max_inflight"] not in (None, batch_size):
        print("[warning] YAML edge max_inflight does not match batch_size.")
    if batch_size >= 16 and summary["dit_max_num_seqs"] and summary["dit_max_num_seqs"] >= batch_size:
        print(
            "[warning] batch_size >= 16 with DiT max_num_seqs >= batch_size is likely to OOM. "
            "If OOM happens during diffusion steps, lower stage 1 max_num_seqs and edge max_inflight first."
        )
    if summary["cudagraph_capture_sizes"]:
        large_capture_sizes = [size for size in summary["cudagraph_capture_sizes"] if isinstance(size, int) and size >= batch_size]
        if batch_size >= 16 and large_capture_sizes:
            print(
                "[warning] cudagraph_capture_sizes includes batch-size graph capture. "
                "If OOM happens during graph capture, remove large capture sizes and keep [1, 2, 4, 8]."
            )
    if summary.get("ar_max_tokens", 0) and summary["ar_max_tokens"] > 2048:
        print(
            "[warning] AR max_tokens is larger than 2048. "
            "For IT2I tests observed AR output is about 500-600 tokens; max_tokens=8192 inflates KV memory."
        )


def build_benchmark_configuration(config: dict[str, Any], deploy_config: Path, batch_size: int) -> str:
    return (
        f"{deploy_config.name} | batch={batch_size} | "
        f"ar_seqs={config['ar_max_num_seqs']} dit_seqs={config['dit_max_num_seqs']} | "
        f"inflight={config['edge_max_inflight']} | "
        f"ar_mem={config['ar_gpu_memory_utilization']} dit_mem={config['dit_gpu_memory_utilization']} | "
        f"ar_tokens={config['ar_max_tokens']} | "
        f"cudagraph={config['cudagraph_mode']}"
    )


def build_benchmark_observation(summary: dict[str, Any]) -> str:
    notes: list[str] = []
    if summary["fail"] > 0:
        notes.append(f"{summary['fail']} failed")
    else:
        notes.append("all requests succeeded")
    if summary["vit_dp_request_global_batch_max"] and summary["vit_dp_request_global_batch_max"] < summary["batch_size"]:
        notes.append("AR ViT DP global_batch < batch_size")
    if summary["response_event_types"] == ["image"]:
        notes.append("image-only response; TTFT uses first event fallback; TPOT unavailable")
    if summary["saved_image_paths"]:
        notes.append(f"saved {len(summary['saved_image_paths'])} images")
    if summary["peak_memory_mb_max"] > 0.0:
        notes.append(f"peak_mem={summary['peak_memory_mb_max']:.0f}MB")
    return "; ".join(notes)


def print_benchmark_results(
    summary: dict[str, Any],
    config: dict[str, Any],
    deploy_config: Path,
    batch_size: int,
) -> None:
    configuration = build_benchmark_configuration(config, deploy_config, batch_size)
    observation = build_benchmark_observation(summary)
    rows = [
        ("Configuration", configuration),
        ("Request Throughput", f"{summary['request_throughput_qps']:.4f} req/s"),
        ("Mean TTFT", f"{summary['ttft_mean_s'] * 1000.0:.2f} ms"),
        ("P50 TTFT", f"{summary['ttft_p50_s'] * 1000.0:.2f} ms"),
        ("P90 TTFT", f"{summary['ttft_p90_s'] * 1000.0:.2f} ms"),
        ("P50 TPOT", f"{summary['tpot_p50_s'] * 1000.0:.2f} ms" if summary["tpot_available"] else "N/A"),
        ("Total Token Throughput", f"{summary['total_token_throughput_tok_s']:.2f} tok/s"),
        ("Observation", observation),
    ]
    print("\n========== Benchmark Results ==========")
    for title, value in rows:
        print(f"{title:<24} {value}")


def print_result_summary(summary: dict[str, Any]) -> None:
    print("\n========== result summary ==========")
    print(
        f"batch_size={summary['batch_size']} warmup_runs={summary['warmup_runs']} num_requests={summary['num_requests']} "
        f"success={summary['success']} fail={summary['fail']} success_rate={summary['success_rate']:.3f}"
    )
    print("\n========== max concurrency calculation ==========")
    print(f"formula={summary['max_concurrency_formula']}")
    print(
        f"num_blocks={summary['num_blocks']} block_size={summary['block_size']} "
        f"kv_source={summary['kv_cache_profile_source']} kv_found={str(summary['kv_cache_profile_found']).lower()}"
    )
    print(
        f"input_tokens_for_capacity={summary['input_tokens_for_capacity']} "
        f"source={summary['input_tokens_for_capacity_source']} "
        f"observed_input_tokens={summary['observed_input_tokens']}"
    )
    print(
        f"output_tokens_for_capacity_mean={summary['output_tokens_for_capacity_mean']:.3f} "
        f"source={summary['output_tokens_for_capacity_source']} "
        f"observed_ar_output_tokens={summary['observed_ar_output_tokens']}"
    )
    print(f"tokens_per_request={summary['tokens_per_request']}")
    print(f"blocks_per_request=ceil(tokens_per_request/block_size)={summary['blocks_per_request']}")
    print(f"max_concurrency=num_blocks//blocks_per_request={summary['max_concurrency']}")
    if summary["num_blocks"] == 0:
        print("[warning] num_blocks was not found in server_log; max_concurrency cannot be computed.")
    if summary["observed_input_tokens_source"] == "missing":
        print("[warning] prompt_tokens was not found in server_log; input token count fell back to --input-tokens.")
    if summary["observed_ar_output_tokens_source"] == "missing":
        print("[warning] AR generated tokens were not found in server_log; output token count fell back to --output-tokens.")
    print(f"wall_time_s={summary['wall_time_s']:.3f} throughput_qps={summary['throughput_qps']:.3f}")
    print(
        f"ttft_mean_s={summary['ttft_mean_s']:.3f} "
        f"ttft_p50_s={summary['ttft_p50_s']:.3f} ttft_p95_s={summary['ttft_p95_s']:.3f} "
        f"ttft_source={summary['ttft_source']}"
    )
    tpot_text = f"{summary['tpot_p50_s']:.6f}" if summary["tpot_available"] else "N/A"
    print(
        f"tpot_p50_s={tpot_text} "
        f"tpot_available={str(summary['tpot_available']).lower()} note={summary['tpot_note']}"
    )
    print(f"total_token_throughput_note={summary['total_token_throughput_note']}")
    print(
        f"e2e_mean_s={summary['e2e_mean_s']:.3f} "
        f"e2e_p50_s={summary['e2e_p50_s']:.3f} e2e_p95_s={summary['e2e_p95_s']:.3f}"
    )
    print(
        f"ar_stage_mean_s={summary['ar_stage_mean_s']:.3f} "
        f"dit_stage_mean_s={summary['dit_stage_mean_s']:.3f} "
        f"peak_memory_mb_max={summary['peak_memory_mb_max']:.3f}"
    )
    print(
        f"ar_delta_mean={summary['ar_delta_mean']:.3f} "
        f"ar_text_chars_mean={summary['ar_text_chars_mean']:.3f}"
    )
    print(f"response_event_types={summary['response_event_types']}")
    print(f"stage_duration_keys={summary['stage_duration_keys']}")
    print(f"saved_images={len(summary['saved_image_paths'])}")
    if summary["saved_image_paths"]:
        print(f"first_saved_image={summary['saved_image_paths'][0]}")
    if not summary["stage_duration_keys"]:
        print("[warning] Response image chunks did not include stage_durations.")
    elif summary["ar_stage_mean_s"] == 0.0 or summary["dit_stage_mean_s"] == 0.0:
        print("[warning] stage_durations were returned, but AR or DiT duration keys were not recognized.")
    print(
        "vit_dp_request="
        f"global_batch_max={summary['vit_dp_request_global_batch_max']} "
        f"global_batches={summary['vit_dp_request_global_batches']} "
        f"shard_logs={summary['vit_dp_request_shard_log_count']} "
        f"encode_state_logs={summary['vit_dp_request_encode_state_log_count']} "
        f"timing_logs={summary['vit_dp_request_timing_log_count']} "
        f"nonzero_local_count_logs={summary['vit_dp_request_nonzero_local_count_logs']}"
    )
    print(
        "vit_dp_timing="
        f"total_ms_mean={summary['vit_dp_request_timing_total_ms_mean']:.3f} "
        f"total_ms_max={summary['vit_dp_request_timing_total_ms_max']:.3f}"
    )
    if summary["vit_dp_request_shard_log_count"] == 0:
        print("[warning] No request-stage AR ViT DP shard logs found. Check server_log directly.")
    if summary["vit_dp_request_encode_state_log_count"] == 0:
        print("[warning] No request-stage AR ViT DP encode-state logs found. Check server_log directly.")
    elif summary["vit_dp_request_global_batch_max"] < summary["batch_size"]:
        print(
            "[warning] Request-stage AR ViT global_batch is smaller than batch_size. "
            "For online tests, ensure stage max_num_seqs and edge max_inflight both equal batch_size."
        )
    if summary["first_error"]:
        print(f"first_error={summary['first_error']}")


def write_outputs(
    output_dir: Path,
    config: dict[str, Any],
    result: dict[str, Any],
    metrics: list[RequestMetric],
    *,
    profiler_dir: Path | None,
    profiler_config: dict[str, Any] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": config,
                "result": result,
                "profiler_dir": str(profiler_dir) if profiler_dir else "",
                "profiler_config": profiler_config,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as f:
        for metric in metrics:
            f.write(json.dumps(asdict(metric), ensure_ascii=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one HunyuanImage3 IT2I NPU batch-size test.")
    parser.add_argument("--deploy-config", default=str(DEFAULT_DEPLOY_CONFIG))
    parser.add_argument("--model", default=None)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--prompt", default="Make the scene snowy while preserving the main subject.")
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to stage 0 max_num_seqs in YAML.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--output-format", default="png")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--server-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup batch runs before the measured test.")
    parser.add_argument("--profile-ar", action="store_true", help="Enable torch profiler only on AR stage 0.")
    parser.add_argument("--profiler-record-shapes", action="store_true")
    parser.add_argument("--profiler-with-stack", action="store_true")
    parser.add_argument("--profiler-with-memory", action="store_true")
    parser.add_argument("--profiler-with-flops", action="store_true")
    parser.add_argument("--profiler-use-gzip", action="store_true")
    parser.add_argument("--profiler-dump-cuda-time-total", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    deploy_config = Path(args.deploy_config).expanduser().resolve()
    image_path = Path(args.image_path).expanduser().resolve()
    if not deploy_config.exists():
        raise FileNotFoundError(f"deploy config does not exist: {deploy_config}")
    if not image_path.exists():
        raise FileNotFoundError(f"image path does not exist: {image_path}")

    config = load_yaml(deploy_config)
    batch_size = args.batch_size or infer_batch_size(config)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT / "test_outputs" / "hunyuan_image3_it2i_batch_test" / f"batch_{batch_size}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    image_dir = output_dir / "images"
    profiler_dir = output_dir / "ar_profiler"
    profiler_config = build_ar_profiler_config(args, profiler_dir) if args.profile_ar else None
    run_deploy_config = prepare_deploy_config_for_run(
        config=config,
        source_path=deploy_config,
        output_dir=output_dir,
        profile_ar=args.profile_ar,
        profiler_config=profiler_config,
    )
    run_config = load_yaml(run_deploy_config)
    config_info = config_summary(run_config)
    log_file = output_dir / "server.log"

    print_case_config(args=args, batch_size=batch_size, deploy_config=run_deploy_config, summary=config_info)
    print(f"\n[info] output_dir={output_dir}")
    print(f"[info] image_dir={image_dir}")
    print(f"[info] server_log={log_file}")
    if args.profile_ar:
        print(f"[info] ar_profiler_dir={profiler_dir}")
        print(f"[info] rendered_deploy_config={run_deploy_config}")

    server = ManagedServer(model=args.model, deploy_config=run_deploy_config, host=args.host, port=args.port, log_file=log_file)
    metrics: list[RequestMetric] = []
    wall_time_s = 0.0
    kv_cache_profile: dict[str, Any] = {
        "num_blocks": 0,
        "block_size": 128,
        "kv_cache_profile_source": "missing",
        "kv_cache_profile_found": False,
        "kv_cache_log_candidates": [],
    }
    request_log_start = 0
    vit_dp_summary: dict[str, Any] = parse_vit_dp_batch_logs("")
    try:
        server.start()
        server.wait_ready(args.server_timeout_s)
        log_text = server.read_log()
        kv_cache_profile = parse_kv_cache_profile(log_text)
        print_kv_cache_profile(kv_cache_profile, phase="after server ready")
        if args.warmup_runs > 0:
            print(f"[warmup] runs={args.warmup_runs} batch_size={batch_size}")
            for warmup_idx in range(args.warmup_runs):
                print(f"[warmup] {warmup_idx + 1}/{args.warmup_runs}")
                _, warmup_metrics = asyncio.run(
                    run_batch_requests(args, batch_size, image_dir=None)
                )
                warmup_success = sum(1 for metric in warmup_metrics if metric.success)
                print(f"[warmup] completed success={warmup_success}/{len(warmup_metrics)}")
        log_text = server.read_log()
        request_log_start = len(log_text)
        image_dir.mkdir(parents=True, exist_ok=True)
        wall_time_s, metrics = asyncio.run(run_batch_requests(args, batch_size, image_dir=image_dir))
    finally:
        server.stop()
    log_text = server.read_log()
    kv_cache_profile = merge_kv_cache_profile(kv_cache_profile, parse_kv_cache_profile(log_text))
    print_kv_cache_profile(kv_cache_profile, phase="final")
    request_log_text = log_text[request_log_start:]
    vit_dp_summary = parse_vit_dp_batch_logs(request_log_text)
    if vit_dp_summary["vit_dp_request_shard_log_count"] == 0 and vit_dp_summary["vit_dp_request_encode_state_log_count"] == 0:
        vit_dp_summary = parse_vit_dp_batch_logs(log_text)
    observed_input_tokens = parse_input_tokens(request_log_text)
    observed_ar_output_tokens = parse_ar_generated_tokens(request_log_text)

    result = summarize_results(
        batch_size=batch_size,
        warmup_runs=args.warmup_runs,
        wall_time_s=wall_time_s,
        metrics=metrics,
        kv_cache_profile=kv_cache_profile,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        observed_input_tokens=observed_input_tokens,
        observed_ar_output_tokens=observed_ar_output_tokens,
        vit_dp_summary=vit_dp_summary,
        saved_image_paths=summarize_image_paths(metrics),
    )
    print_benchmark_results(result, config_info, run_deploy_config, batch_size)
    write_outputs(
        output_dir,
        config_info,
        result,
        metrics,
        profiler_dir=profiler_dir if args.profile_ar else None,
        profiler_config=profiler_config,
    )
    print_result_summary(result)
    print(f"\n[done] summary_json={output_dir / 'summary.json'}")
    print(f"[done] requests_jsonl={output_dir / 'requests.jsonl'}")
    print(f"[done] server_log={log_file}")
    return 0 if result["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
