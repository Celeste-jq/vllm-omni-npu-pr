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
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY_CONFIG = REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3_it2i_npu_aclgraph_rope_vitdp.yaml"
KV_CACHE_PATTERN = re.compile(r"\[kv-cache-profile\].*?\bnum_blocks=(\d+)\b")


@dataclass
class RequestMetric:
    request_index: int
    success: bool
    http_status: int
    ttft_s: float
    e2e_s: float
    ar_delta_count: int
    ar_text_chars: int
    stage_durations: dict[str, float]
    peak_memory_mb: float
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


def parse_num_blocks(log_text: str) -> int | None:
    match = KV_CACHE_PATTERN.search(log_text)
    return int(match.group(1)) if match else None


def find_stage_value(stage_durations: dict[str, float], candidates: tuple[str, ...]) -> float:
    for key in candidates:
        value = stage_durations.get(key)
        if value is not None:
            return float(value)
    return 0.0


def config_summary(config: dict[str, Any]) -> dict[str, Any]:
    stages = config.get("stages") or []
    ar_stage = stages[0] if len(stages) > 0 and isinstance(stages[0], dict) else {}
    dit_stage = stages[1] if len(stages) > 1 and isinstance(stages[1], dict) else {}
    edges = config.get("edges") or []
    edge = edges[0] if edges and isinstance(edges[0], dict) else {}
    rope_parameters = (ar_stage.get("hf_overrides") or {}).get("rope_parameters") or {}
    compilation = ar_stage.get("compilation_config") or {}
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


class ManagedServer:
    def __init__(self, *, model: str, deploy_config: Path, host: str, port: int, log_file: Path) -> None:
        self.model = model
        self.deploy_config = deploy_config
        self.host = host
        self.port = port
        self.log_file = log_file
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["TASKQUEUEENABLE"] = "1"
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
        print("[server] " + " ".join(cmd), flush=True)
        self.process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
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
    model: str,
    image_path: Path,
    prompt: str,
    size: str,
    output_format: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    request_index: int,
    timeout_s: float,
) -> RequestMetric:
    import aiohttp

    form = aiohttp.FormData()
    form.add_field("model", model)
    form.add_field("image", image_path.read_bytes(), filename=image_path.name, content_type="image/png")
    form.add_field("prompt", prompt)
    form.add_field("size", size)
    form.add_field("output_format", output_format)
    form.add_field("num_inference_steps", str(num_inference_steps))
    form.add_field("guidance_scale", str(guidance_scale))
    form.add_field("seed", str(seed + request_index))
    form.add_field("stream", "true")

    decoder = SSEDecoder()
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
                        ar_text_chars += len(payload.get("delta") or "")
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
            return RequestMetric(
                request_index=request_index,
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
        return RequestMetric(
            request_index=request_index,
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


async def run_batch_requests(args: argparse.Namespace, batch_size: int) -> tuple[float, list[RequestMetric]]:
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
                )
                for index in range(batch_size)
            ]
        )
        return time.perf_counter() - started, list(metrics)


def summarize_results(
    *,
    batch_size: int,
    wall_time_s: float,
    metrics: list[RequestMetric],
    num_blocks: int | None,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    successes = [metric for metric in metrics if metric.success]
    ttfts = [metric.ttft_s for metric in successes if metric.ttft_s > 0.0]
    e2es = [metric.e2e_s for metric in successes if metric.e2e_s > 0.0]
    peaks = [metric.peak_memory_mb for metric in successes if metric.peak_memory_mb > 0.0]
    ar_stages = [
        find_stage_value(metric.stage_durations, ("stage_0", "ar", "prefill", "text", "llm")) for metric in successes
    ]
    dit_stages = [
        find_stage_value(metric.stage_durations, ("stage_1", "dit", "diffusion", "image")) for metric in successes
    ]
    blocks_per_request = max(1, math.ceil((input_tokens + output_tokens) / 128))
    estimated_max_batch = (num_blocks or 0) // blocks_per_request if num_blocks else 0
    return {
        "batch_size": batch_size,
        "num_requests": batch_size,
        "num_blocks": num_blocks or 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "blocks_per_request": blocks_per_request,
        "estimated_max_batch": estimated_max_batch,
        "success": len(successes),
        "fail": len(metrics) - len(successes),
        "success_rate": len(successes) / len(metrics) if metrics else 0.0,
        "wall_time_s": wall_time_s,
        "throughput_qps": len(successes) / wall_time_s if wall_time_s > 0.0 else 0.0,
        "ttft_mean_s": mean(ttfts),
        "ttft_p50_s": percentile(ttfts, 0.50),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "e2e_mean_s": mean(e2es),
        "e2e_p50_s": percentile(e2es, 0.50),
        "e2e_p95_s": percentile(e2es, 0.95),
        "ar_stage_mean_s": mean(ar_stages),
        "dit_stage_mean_s": mean(dit_stages),
        "peak_memory_mb_max": max(peaks) if peaks else 0.0,
        "ar_delta_mean": mean([float(metric.ar_delta_count) for metric in successes]),
        "ar_text_chars_mean": mean([float(metric.ar_text_chars) for metric in successes]),
        "first_error": next((metric.error for metric in metrics if metric.error), ""),
    }


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
    print(f"prompt={args.prompt}")
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
    print(f"TASKQUEUEENABLE=1")
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


def print_result_summary(summary: dict[str, Any]) -> None:
    print("\n========== result summary ==========")
    print(
        f"batch_size={summary['batch_size']} num_requests={summary['num_requests']} "
        f"success={summary['success']} fail={summary['fail']} success_rate={summary['success_rate']:.3f}"
    )
    print(
        f"num_blocks={summary['num_blocks']} input_tokens={summary['input_tokens']} "
        f"output_tokens={summary['output_tokens']} blocks_per_request={summary['blocks_per_request']} "
        f"estimated_max_batch={summary['estimated_max_batch']}"
    )
    print(f"wall_time_s={summary['wall_time_s']:.3f} throughput_qps={summary['throughput_qps']:.3f}")
    print(
        f"ttft_mean_s={summary['ttft_mean_s']:.3f} "
        f"ttft_p50_s={summary['ttft_p50_s']:.3f} ttft_p95_s={summary['ttft_p95_s']:.3f}"
    )
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
    parser.add_argument("--model", default="tencent/HunyuanImage-3.0-Instruct")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--prompt", default="Make the scene snowy while preserving the main subject.")
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to stage 0 max_num_seqs in YAML.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--output-format", default="png")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--server-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--output-dir", default=None)
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

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT / "test_outputs" / "hunyuan_image3_it2i_batch_test" / f"batch_{batch_size}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
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
    print(f"[info] server_log={log_file}")
    if args.profile_ar:
        print(f"[info] ar_profiler_dir={profiler_dir}")
        print(f"[info] rendered_deploy_config={run_deploy_config}")

    server = ManagedServer(model=args.model, deploy_config=run_deploy_config, host=args.host, port=args.port, log_file=log_file)
    metrics: list[RequestMetric] = []
    wall_time_s = 0.0
    try:
        server.start()
        server.wait_ready(args.server_timeout_s)
        log_text = server.read_log()
        num_blocks = parse_num_blocks(log_text)
        wall_time_s, metrics = asyncio.run(run_batch_requests(args, batch_size))
        log_text = server.read_log()
        num_blocks = parse_num_blocks(log_text) or num_blocks
    finally:
        server.stop()

    result = summarize_results(
        batch_size=batch_size,
        wall_time_s=wall_time_s,
        metrics=metrics,
        num_blocks=num_blocks,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
    )
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
