# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


KV_CACHE_PATTERNS = (
    re.compile(r"\[kv-cache-profile\].*?\bnum_blocks=(\d+)\b"),
    re.compile(r"\bnum_blocks=(\d+)\b"),
    re.compile(r"\bnum_blocks:\s*(\d+)\b"),
)


def estimate_max_concurrency(
    num_blocks: int,
    input_tokens: int,
    output_tokens: int,
    *,
    block_size: int = 128,
) -> int:
    blocks_per_request = max(1, math.ceil((input_tokens + output_tokens) / block_size))
    return max(1, num_blocks // blocks_per_request)


def build_concurrency_values(target: int, maximum: int) -> list[int]:
    if maximum < 1:
        raise ValueError("maximum must be >= 1")
    target = max(1, min(target, maximum))
    values = set()
    values.update(v for v in (1, 2, 3, 4) if v <= maximum)

    power = 1
    while power <= maximum:
        values.add(power)
        power *= 2

    for value in range(target - 2, target + 4):
        if 1 <= value <= maximum:
            values.add(value)

    return sorted(values)


def parse_num_blocks_from_log(text: str) -> int | None:
    for pattern in KV_CACHE_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def apply_deploy_overrides(
    config: dict[str, Any],
    *,
    batch_size: int,
    ar_gpu_memory_utilization: float,
    dit_gpu_memory_utilization: float,
    cudagraph_capture_sizes: list[int],
) -> dict[str, Any]:
    stages = config.get("stages", [])
    if len(stages) < 2:
        raise ValueError("expected at least two stages in deploy config")

    ar_stage = stages[0]
    dit_stage = stages[1]

    ar_stage["is_comprehension"] = True
    ar_stage["max_num_seqs"] = batch_size
    ar_stage["gpu_memory_utilization"] = ar_gpu_memory_utilization
    ar_stage["enforce_eager"] = False
    ar_stage["mm_encoder_tp_mode"] = "data"
    ar_stage.setdefault("hf_overrides", {}).setdefault(
        "rope_parameters",
        {"mrope_section": [0, 32, 32], "rope_type": "default"},
    )
    ar_stage["compilation_config"] = {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": cudagraph_capture_sizes,
    }

    dit_stage["max_num_seqs"] = batch_size
    dit_stage["gpu_memory_utilization"] = dit_gpu_memory_utilization

    edges = config.get("edges")
    if isinstance(edges, list):
        for edge in edges:
            if isinstance(edge, dict):
                edge.setdefault("window_size", -1)
                edge.setdefault("max_inflight", batch_size)

    return config


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _read_image_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    frac = rank - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _find_stage_value(stage_durations: dict[str, float], names: tuple[str, ...]) -> float:
    for name in names:
        value = stage_durations.get(name)
        if value is not None:
            return float(value)
    return 0.0


def _build_capture_sizes(concurrency: int) -> list[int]:
    values = {1, concurrency}
    for delta in (-2, -1, 1, 2):
        candidate = concurrency + delta
        if candidate > 0:
            values.add(candidate)
    return sorted(values)


def render_single_run_config(
    base_config: dict[str, Any],
    *,
    batch_size: int,
    ar_gpu_memory_utilization: float,
    dit_gpu_memory_utilization: float,
) -> dict[str, Any]:
    return apply_deploy_overrides(
        json.loads(json.dumps(base_config)),
        batch_size=batch_size,
        ar_gpu_memory_utilization=ar_gpu_memory_utilization,
        dit_gpu_memory_utilization=dit_gpu_memory_utilization,
        cudagraph_capture_sizes=_build_capture_sizes(batch_size),
    )


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


@dataclass
class RequestMetrics:
    request_index: int
    concurrency: int
    success: bool
    http_status: int
    ttft_s: float
    e2e_s: float
    ar_delta_count: int
    ar_text_chars: int
    stage_durations: dict[str, float]
    peak_memory_mb: float
    error: str


class ImageEditSSEDecoder:
    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: bytes) -> list[str]:
        self._buffer += chunk.decode("utf-8")
        events: list[str] = []
        while "\n\n" in self._buffer:
            raw_event, self._buffer = self._buffer.split("\n\n", 1)
            if not raw_event.strip():
                continue
            data_lines = [line[6:] for line in raw_event.splitlines() if line.startswith("data: ")]
            if data_lines:
                events.append("\n".join(data_lines))
        return events


async def _send_single_request(
    session,
    *,
    api_url: str,
    model: str,
    image_name: str,
    image_bytes: bytes,
    prompt: str,
    size: str,
    output_format: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    request_index: int,
    concurrency: int,
    timeout_s: float,
) -> RequestMetrics:
    import aiohttp

    form = aiohttp.FormData()
    form.add_field("model", model)
    form.add_field("prompt", prompt)
    form.add_field("size", size)
    form.add_field("output_format", output_format)
    form.add_field("num_inference_steps", str(num_inference_steps))
    form.add_field("guidance_scale", str(guidance_scale))
    form.add_field("seed", str(seed + request_index))
    form.add_field("stream", "true")
    form.add_field("image", image_bytes, filename=image_name, content_type="image/png")

    decoder = ImageEditSSEDecoder()
    started = time.perf_counter()
    ttft_s = 0.0
    e2e_s = 0.0
    ar_delta_count = 0
    ar_text_chars = 0
    stage_durations: dict[str, float] = {}
    peak_memory_mb = 0.0
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
                    if payload.get("type") == "ar_delta":
                        ar_delta_count += 1
                        ar_text_chars += len(payload.get("delta", ""))
                        if ttft_s == 0.0:
                            ttft_s = now - started
                    elif payload.get("type") == "image":
                        e2e_s = now - started
                        stage_durations = {
                            key: float(value) for key, value in (payload.get("stage_durations") or {}).items()
                        }
                        peak_memory_mb = float(payload.get("peak_memory_mb") or 0.0)
                    elif payload.get("object") == "error":
                        error = json.dumps(payload.get("error", {}), ensure_ascii=True)
            if response.status != 200 and not error:
                error = response.reason or f"http_status={response.status}"
            return RequestMetrics(
                request_index=request_index,
                concurrency=concurrency,
                success=response.status == 200 and not error and e2e_s > 0.0,
                http_status=response.status,
                ttft_s=ttft_s,
                e2e_s=e2e_s,
                ar_delta_count=ar_delta_count,
                ar_text_chars=ar_text_chars,
                stage_durations=stage_durations,
                peak_memory_mb=peak_memory_mb,
                error=error,
            )
    except Exception as exc:  # noqa: BLE001
        return RequestMetrics(
            request_index=request_index,
            concurrency=concurrency,
            success=False,
            http_status=0,
            ttft_s=ttft_s,
            e2e_s=e2e_s,
            ar_delta_count=ar_delta_count,
            ar_text_chars=ar_text_chars,
            stage_durations=stage_durations,
            peak_memory_mb=peak_memory_mb,
            error=str(exc),
        )


async def _run_benchmark(
    *,
    base_url: str,
    model: str,
    image_path: Path,
    prompt: str,
    size: str,
    output_format: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    concurrency: int,
    num_requests: int,
    timeout_s: float,
) -> tuple[float, list[RequestMetrics]]:
    import aiohttp

    connector = aiohttp.TCPConnector(limit=max(concurrency, 1), limit_per_host=max(concurrency, 1))
    api_url = base_url.rstrip("/") + "/v1/images/edits"
    image_bytes = _read_image_bytes(image_path)
    async with aiohttp.ClientSession(connector=connector) as session:
        started = time.perf_counter()
        tasks = [
            _send_single_request(
                session,
                api_url=api_url,
                model=model,
                image_name=image_path.name,
                image_bytes=image_bytes,
                prompt=prompt,
                size=size,
                output_format=output_format,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                request_index=index,
                concurrency=concurrency,
                timeout_s=timeout_s,
            )
            for index in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)
        wall_time_s = time.perf_counter() - started
        return wall_time_s, results


class ManagedServer:
    def __init__(
        self,
        *,
        model: str,
        deploy_config: Path,
        host: str,
        port: int,
        log_file: Path,
        env: dict[str, str],
    ) -> None:
        self.model = model
        self.deploy_config = deploy_config
        self.host = host
        self.port = port
        self.log_file = log_file
        self.env = env
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = self.log_file.open("w", encoding="utf-8")
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
        self.process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=self.env,
            cwd=str(Path(__file__).resolve().parents[1]),
        )

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

    def wait_until_ready(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        health_url = f"http://{self.host}:{self.port}/health"
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"server exited early, see log: {self.log_file}")
            try:
                import urllib.request

                with urllib.request.urlopen(health_url, timeout=2) as response:  # noqa: S310
                    if response.status == 200:
                        return
            except Exception:  # noqa: BLE001
                time.sleep(2)
        raise TimeoutError(f"server did not become ready within {timeout_s}s: {self.log_file}")

    def read_log(self) -> str:
        if not self.log_file.exists():
            return ""
        return self.log_file.read_text(encoding="utf-8", errors="replace")


def _create_run_summary(
    *,
    gpu_memory_preset: str,
    concurrency: int,
    num_requests: int,
    wall_time_s: float,
    num_blocks: int | None,
    metrics: list[RequestMetrics],
) -> dict[str, Any]:
    successes = [metric for metric in metrics if metric.success]
    ttfts = [metric.ttft_s for metric in successes if metric.ttft_s > 0.0]
    e2es = [metric.e2e_s for metric in successes if metric.e2e_s > 0.0]
    peak_memories = [metric.peak_memory_mb for metric in successes if metric.peak_memory_mb > 0.0]
    ar_stage_s = [
        _find_stage_value(metric.stage_durations, ("stage_0", "ar", "prefill", "text", "llm")) for metric in successes
    ]
    dit_stage_s = [
        _find_stage_value(metric.stage_durations, ("stage_1", "dit", "diffusion", "image")) for metric in successes
    ]
    return {
        "gpu_memory_preset": gpu_memory_preset,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "num_blocks": num_blocks or 0,
        "successes": len(successes),
        "failures": len(metrics) - len(successes),
        "request_throughput_qps": len(successes) / wall_time_s if wall_time_s > 0 else 0.0,
        "wall_time_s": wall_time_s,
        "ttft_mean_s": _mean(ttfts),
        "ttft_p50_s": _percentile(ttfts, 0.50),
        "ttft_p95_s": _percentile(ttfts, 0.95),
        "e2e_mean_s": _mean(e2es),
        "e2e_p50_s": _percentile(e2es, 0.50),
        "e2e_p95_s": _percentile(e2es, 0.95),
        "ar_stage_mean_s": _mean(ar_stage_s),
        "dit_stage_mean_s": _mean(dit_stage_s),
        "peak_memory_mb_max": max(peak_memories) if peak_memories else 0.0,
        "ar_delta_mean": _mean([float(metric.ar_delta_count) for metric in successes]),
        "ar_text_chars_mean": _mean([float(metric.ar_text_chars) for metric in successes]),
    }


def cmd_prepare_configs(args: argparse.Namespace) -> int:
    base_config = _load_yaml(Path(args.base_config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    presets = [
        ("mem075", 0.75, 0.63),
        ("mem078", 0.78, 0.66),
        ("mem080", 0.80, 0.68),
    ]
    for name, ar_gmu, dit_gmu in presets:
        config = apply_deploy_overrides(
            json.loads(json.dumps(base_config)),
            batch_size=1,
            ar_gpu_memory_utilization=ar_gmu,
            dit_gpu_memory_utilization=dit_gmu,
            cudagraph_capture_sizes=[1, 2],
        )
        _dump_yaml(output_dir / f"hunyuan_image3_it2i_npu_{name}.yaml", config)
    return 0


def cmd_render_config(args: argparse.Namespace) -> int:
    base_config = _load_yaml(Path(args.base_config))
    rendered = render_single_run_config(
        base_config,
        batch_size=args.batch_size,
        ar_gpu_memory_utilization=args.ar_gpu_memory_utilization,
        dit_gpu_memory_utilization=args.dit_gpu_memory_utilization,
    )
    _dump_yaml(Path(args.output_path), rendered)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    base_config = _load_yaml(Path(args.base_config))
    results_dir = Path(args.results_dir)
    generated_config_dir = results_dir / "generated_configs"
    logs_dir = results_dir / "server_logs"
    rows: list[dict[str, Any]] = []
    raw_jsonl = results_dir / "requests.jsonl"

    mem_presets = [
        ("mem075", 0.75, 0.63),
        ("mem078", 0.78, 0.66),
        ("mem080", 0.80, 0.68),
    ]
    env = os.environ.copy()
    env["TASKQUEUEENABLE"] = "1"

    probe_config = apply_deploy_overrides(
        json.loads(json.dumps(base_config)),
        batch_size=1,
        ar_gpu_memory_utilization=mem_presets[0][1],
        dit_gpu_memory_utilization=mem_presets[0][2],
        cudagraph_capture_sizes=[1, 2],
    )
    probe_path = generated_config_dir / "probe.yaml"
    _dump_yaml(probe_path, probe_config)
    probe_server = ManagedServer(
        model=args.model,
        deploy_config=probe_path,
        host=args.host,
        port=args.port,
        log_file=logs_dir / "probe.log",
        env=env,
    )
    num_blocks = None
    try:
        probe_server.start()
        probe_server.wait_until_ready(args.server_timeout_s)
        num_blocks = parse_num_blocks_from_log(probe_server.read_log())
    finally:
        probe_server.stop()

    estimated_max = estimate_max_concurrency(
        num_blocks=num_blocks or args.fallback_num_blocks,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
    )
    concurrency_values = (
        [int(value) for value in args.concurrency_values.split(",") if value.strip()]
        if args.concurrency_values
        else build_concurrency_values(estimated_max, estimated_max)
    )

    for preset_name, ar_gmu, dit_gmu in mem_presets:
        for concurrency in concurrency_values:
            config = apply_deploy_overrides(
                json.loads(json.dumps(base_config)),
                batch_size=concurrency,
                ar_gpu_memory_utilization=ar_gmu,
                dit_gpu_memory_utilization=dit_gmu,
                cudagraph_capture_sizes=_build_capture_sizes(concurrency),
            )
            config_path = generated_config_dir / f"{preset_name}_c{concurrency}.yaml"
            _dump_yaml(config_path, config)

            server = ManagedServer(
                model=args.model,
                deploy_config=config_path,
                host=args.host,
                port=args.port,
                log_file=logs_dir / f"{preset_name}_c{concurrency}.log",
                env=env,
            )
            try:
                server.start()
                server.wait_until_ready(args.server_timeout_s)
                wall_time_s, metrics = asyncio.run(
                    _run_benchmark(
                        base_url=f"http://{args.host}:{args.port}",
                        model=args.model,
                        image_path=Path(args.image_path),
                        prompt=args.prompt,
                        size=args.size,
                        output_format=args.output_format,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        seed=args.seed,
                        concurrency=concurrency,
                        num_requests=args.num_requests,
                        timeout_s=args.request_timeout_s,
                    )
                )
            finally:
                server.stop()

            rows.append(
                _create_run_summary(
                    gpu_memory_preset=preset_name,
                    concurrency=concurrency,
                    num_requests=args.num_requests,
                    wall_time_s=wall_time_s,
                    num_blocks=num_blocks,
                    metrics=metrics,
                )
            )
            _append_jsonl(
                raw_jsonl,
                [
                    {
                        "gpu_memory_preset": preset_name,
                        "num_blocks": num_blocks,
                        **asdict(metric),
                    }
                    for metric in metrics
                ],
            )

    _write_summary_csv(results_dir / "summary.csv", rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NPU HunyuanImage3 IT2I experiment runner.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    prepare = subparsers.add_parser("prepare-configs")
    prepare.add_argument(
        "--base-config",
        default="vllm_omni/deploy/hunyuan_image3_it2i_npu_aclgraph_rope_vitdp.yaml",
    )
    prepare.add_argument(
        "--output-dir",
        default="vllm_omni/deploy/experiments/hunyuan_image3_it2i_npu",
    )
    prepare.set_defaults(func=cmd_prepare_configs)

    render = subparsers.add_parser("render-config")
    render.add_argument("--base-config", default="vllm_omni/deploy/hunyuan_image3_it2i_npu_aclgraph_rope_vitdp.yaml")
    render.add_argument("--output-path", required=True)
    render.add_argument("--batch-size", type=int, required=True)
    render.add_argument("--ar-gpu-memory-utilization", type=float, default=0.78)
    render.add_argument("--dit-gpu-memory-utilization", type=float, default=0.66)
    render.set_defaults(func=cmd_render_config)

    run = subparsers.add_parser("run")
    run.add_argument("--base-config", default="vllm_omni/deploy/hunyuan_image3_it2i_npu_aclgraph_rope_vitdp.yaml")
    run.add_argument("--model", default="tencent/HunyuanImage-3.0-Instruct")
    run.add_argument("--image-path", required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--results-dir", default="test_outputs/hunyuan_image3_it2i_npu_experiment")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8092)
    run.add_argument("--size", default="1024x1024")
    run.add_argument("--output-format", default="png")
    run.add_argument("--num-inference-steps", type=int, default=50)
    run.add_argument("--guidance-scale", type=float, default=5.0)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--num-requests", type=int, default=8)
    run.add_argument("--request-timeout-s", type=float, default=900.0)
    run.add_argument("--server-timeout-s", type=float, default=900.0)
    run.add_argument("--input-tokens", type=int, default=2048)
    run.add_argument("--output-tokens", type=int, default=1024)
    run.add_argument("--fallback-num-blocks", type=int, default=1024)
    run.add_argument("--concurrency-values", default=None)
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
