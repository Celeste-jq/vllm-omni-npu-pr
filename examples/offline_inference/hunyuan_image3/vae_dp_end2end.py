"""
HunyuanImage-3.0-Instruct AR VAE DP batch inference script.

This script mirrors the batch-admission harness used by the ViT DP benchmark,
but focuses on it2i/img2img requests so we can stress AR-stage VAE sample-DP
with multiple requests in the same scheduler tick.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoTokenizer

from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
    MAX_IMAGES_PER_REQUEST,
    build_prompt_tokens,
    resolve_stop_token_ids,
    resolve_sys_type,
)
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DEPLOY_CONFIG = str(_REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image_3_moe.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0 AR VAE DP batch inference.")
    parser.add_argument("--model", default="tencent/HunyuanImage-3.0-Instruct", help="Model name or local path.")
    parser.add_argument("--deploy-config", type=str, default=_DEFAULT_DEPLOY_CONFIG, help="Deploy YAML path.")
    parser.add_argument("--output", type=str, default="./vae_dp_end2end_outputs", help="Output directory.")
    parser.add_argument("--prompts", nargs="+", default=None, help="One or more prompts.")
    parser.add_argument("--prompt-file", type=str, default=None, help="Text file with one prompt per line.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of requests to submit. If larger than prompt count, prompts/images are cycled.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Comma-separated image paths shared by all requests.",
    )
    parser.add_argument(
        "--image-paths",
        nargs="+",
        default=None,
        help=(
            "Per-request image groups. Each group is comma-separated, for example "
            "'a.png,b.png' 'c.png,d.png'. Cycled when --batch-size is larger."
        ),
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=None, help="Generated image height. Defaults to input image height.")
    parser.add_argument("--width", type=int, default=None, help="Generated image width. Defaults to input image width.")
    parser.add_argument("--bot-task", default="think_recaption", choices=["think", "recaption", "think_recaption"])
    parser.add_argument("--sys-type", type=str, default="en_unified")
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument("--init-timeout", type=int, default=300)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument(
        "--additional-config",
        type=str,
        default=None,
        help="JSON object forwarded to Omni/additional_config.",
    )
    parser.add_argument("--warmup-runs", type=int, default=0, help="Number of warmup batch-admission runs.")
    parser.add_argument("--profile-runs", type=int, default=1, help="Number of measured batch-admission runs.")

    return parser.parse_args()


def parse_additional_config(raw_value: str | None) -> dict | None:
    if raw_value is None:
        return None
    value = json.loads(raw_value)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"--additional-config must be a JSON object, got {type(value).__name__}")
    return value


def load_prompts(args) -> list[str]:
    prompts: list[str] = []
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompts.extend(line.strip() for line in f if line.strip())
    if args.prompts:
        prompts.extend(args.prompts)
    if not prompts:
        raise ValueError("Provide --prompts or --prompt-file.")
    return prompts


def split_image_group(raw_group: str) -> list[str]:
    paths = [p.strip() for p in raw_group.split(",") if p.strip()]
    if not paths:
        raise ValueError(f"Image group produced no paths: {raw_group!r}")
    if len(paths) > MAX_IMAGES_PER_REQUEST:
        raise ValueError(
            f"Each request supports at most {MAX_IMAGES_PER_REQUEST} images, got {len(paths)}: {raw_group}"
        )
    for path in paths:
        if not os.path.exists(path):
            raise ValueError(f"Image path does not exist: {path}")
    return paths


def load_image_payload(paths: list[str]):
    images = [Image.open(path).convert("RGB") for path in paths]
    return images[0] if len(images) == 1 else images


def build_request_plan(args) -> list[tuple[str, list[str]]]:
    prompts = load_prompts(args)
    if args.image_paths:
        image_groups = [split_image_group(group) for group in args.image_paths]
    elif args.image_path:
        image_groups = [split_image_group(args.image_path)]
    else:
        raise ValueError("Provide --image-path or --image-paths.")

    request_count = args.batch_size or max(len(prompts), len(image_groups))
    if request_count <= 0:
        raise ValueError("--batch-size must be positive.")

    prompt_iter = iter(prompts)
    image_iter = iter(image_groups)
    plan: list[tuple[str, list[str]]] = []
    for _ in range(request_count):
        try:
            prompt = next(prompt_iter)
        except StopIteration:
            prompt_iter = iter(prompts)
            prompt = next(prompt_iter)
        try:
            image_paths = next(image_iter)
        except StopIteration:
            image_iter = iter(image_groups)
            image_paths = next(image_iter)
        plan.append((prompt, image_paths))
    return plan


def count_prompt_images(prompt: dict[str, Any]) -> int:
    image_payload = prompt.get("multi_modal_data", {}).get("image")
    if image_payload is None:
        return 0
    if isinstance(image_payload, (list, tuple)):
        return len(image_payload)
    return 1


def build_request_mm_uuids(req_idx: int, num_images: int, batch_id: str | None = None) -> dict[str, list[str]]:
    prefix = f"{batch_id}-" if batch_id else ""
    return {"image": [f"{prefix}req-{req_idx}-image-{image_idx}" for image_idx in range(num_images)]}


def _create_patched_deploy_config(deploy_config_path: str, stage0_max_num_seqs: int) -> str:
    """Create a temp deploy config with AR VAE DP enabled on every stage 0."""

    with open(deploy_config_path, encoding="utf-8") as f:
        raw_text = f.read()

    lines = raw_text.splitlines()
    patched_lines: list[str] = []
    in_stage0 = False
    stage0_indent: int | None = None
    injected_vae_dp = "ar_vae_tp_mode: data" in raw_text

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if stripped.startswith("- stage_id: 0"):
            in_stage0 = True
            stage0_indent = indent
            patched_lines.append(line)
            continue

        if in_stage0 and stage0_indent is not None and indent <= stage0_indent and stripped.startswith("- stage_id:"):
            in_stage0 = False
            stage0_indent = None

        if in_stage0 and stripped.startswith("max_num_seqs:"):
            current = int(stripped.split(":", 1)[1].strip())
            patched_lines.append(" " * indent + f"max_num_seqs: {max(current, stage0_max_num_seqs)}")
            continue

        if in_stage0 and not injected_vae_dp and stripped == "hf_overrides:":
            patched_lines.append(line)
            patched_lines.append(" " * (indent + 2) + "ar_vae_tp_mode: data")
            injected_vae_dp = True
            continue

        patched_lines.append(line)

    if not injected_vae_dp:
        raise ValueError(f"Failed to inject ar_vae_tp_mode into {deploy_config_path}")

    temp_dir = tempfile.mkdtemp(prefix="hunyuan_image3_vae_dp_")
    patched_path = os.path.join(temp_dir, Path(deploy_config_path).name)
    with open(patched_path, "w", encoding="utf-8") as f:
        f.write("\n".join(patched_lines) + "\n")
    return patched_path


def run_batch_admission(
    omni: Omni,
    *,
    prompts: list[dict[str, Any]],
    sampling_params_list: list[Any],
    log_prefix: str = "[batch-admission]",
) -> list[Any]:
    """Submit a batch of requests before polling outputs."""

    from vllm_omni.engine.messages import OutputMessage
    from vllm_omni.entrypoints.client_request_state import ClientRequestState
    from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics

    sampling_params_list = list(omni.resolve_sampling_params_list(sampling_params_list))
    sampling_params_list = omni._set_final_only_for_llm_stages(sampling_params_list)

    request_ids = [f"{i}_{uuid.uuid4()}" for i in range(len(prompts))]
    wall_start_ts = time.time()
    req_start_ts: dict[str, float] = {}
    req_final_stage_ids: dict[str, int] = {}
    pending_msgs: list[tuple[str, Any]] = []

    try:
        batch_id = uuid.uuid4().hex
        for req_idx, (req_id, prompt) in enumerate(zip(request_ids, prompts)):
            prompt["multi_modal_uuids"] = build_request_mm_uuids(req_idx, count_prompt_images(prompt), batch_id)
            prompt_modalities = prompt.get("modalities", None)
            final_stage_id = omni._compute_final_stage_id(prompt_modalities)
            final_output_stage_ids = omni._compute_final_output_stage_ids(prompt_modalities) or [final_stage_id]
            req_final_stage_ids[req_id] = final_stage_id

            metrics = OrchestratorMetrics(
                omni.num_stages,
                omni.log_stats,
                wall_start_ts,
                final_stage_id,
            )
            req_state = ClientRequestState(req_id)
            req_state.metrics = metrics
            omni.request_states[req_id] = req_state

            req_sp_list = list(sampling_params_list)
            pd_pair = omni._get_pd_separation_pair()
            if pd_pair is not None:
                p_id = pd_pair[0]
                req_sp_list[p_id] = omni._prepare_prefill_sampling_params(req_id, req_sp_list[p_id])

            msg = omni.engine._build_add_request_message(
                request_id=req_id,
                prompt=prompt,
                sampling_params_list=req_sp_list,
                final_stage_id=final_stage_id,
                final_output_stage_ids=final_output_stage_ids,
            )
            pending_msgs.append((req_id, msg))

        enqueue_start = time.time()
        for req_id, msg in pending_msgs:
            omni.engine.request_queue.sync_q.put_nowait(msg)
            req_state = omni.request_states[req_id]
            if req_state.metrics is not None:
                req_state.metrics.stage_first_ts[0] = enqueue_start
            req_start_ts[req_id] = enqueue_start
            print(f"{log_prefix} enqueued {req_id}")

        active_reqs = set(request_ids)
        outputs: list[Any] = []
        while active_reqs:
            msg = omni.engine.try_get_output()
            should_continue, req_id, stage_id, req_state = omni._handle_output_message(msg)
            if should_continue:
                continue

            if req_id not in active_reqs:
                continue

            omni._check_engine_output_error(msg, req_id, stage_id)
            if req_state.metrics is None:
                continue

            output = omni._process_single_result(
                result=msg,
                stage_id=stage_id,
                metrics=req_state.metrics,
                req_start_ts=req_start_ts,
                wall_start_ts=wall_start_ts,
                final_stage_id_for_e2e=req_final_stage_ids[req_id],
            )
            if output is not None:
                outputs.append(output)

            if isinstance(msg, OutputMessage) and msg.finished:
                active_reqs.discard(req_id)
                omni._log_summary_and_cleanup(req_id)

        return outputs
    except Exception:
        if request_ids:
            omni.abort(request_ids)
        raise


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if p <= 0:
        return float(ordered[0])
    if p >= 100:
        return float(ordered[-1])
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _extract_text(req_output: Any) -> str:
    ro = getattr(req_output, "request_output", None)
    text = ""
    if ro and getattr(ro, "outputs", None):
        text = "".join(getattr(item, "text", "") or "" for item in ro.outputs)
    if text:
        return text

    custom_output = getattr(req_output, "custom_output", {}) or {}
    ar_text = custom_output.get("ar_generated_text") if isinstance(custom_output, dict) else None
    if isinstance(ar_text, list):
        return "\n".join(part for part in ar_text if part)
    if isinstance(ar_text, str):
        return ar_text
    return ""


def _first_stage_metrics(req_output: Any) -> dict[str, Any]:
    metrics = getattr(req_output, "metrics", None) or {}
    stage_metrics = metrics.get("stage_metrics") if isinstance(metrics, dict) else None
    if not isinstance(stage_metrics, dict) or not stage_metrics:
        return {}
    stage_zero = stage_metrics.get("0")
    if isinstance(stage_zero, dict):
        return stage_zero
    first_key = sorted(stage_metrics.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)[0]
    maybe_metrics = stage_metrics.get(first_key)
    return maybe_metrics if isinstance(maybe_metrics, dict) else {}


def _benchmark_metrics(outputs: list[Any], *, elapsed_s: float, configuration: str) -> dict[str, Any]:
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    total_input_tokens = 0
    total_output_tokens = 0
    non_empty_text_outputs = 0

    for req_output in outputs:
        if _extract_text(req_output):
            non_empty_text_outputs += 1

        stage_metrics = _first_stage_metrics(req_output)
        num_tokens_in = _safe_int(stage_metrics.get("num_tokens_in"))
        num_tokens_out = _safe_int(stage_metrics.get("num_tokens_out"))
        total_input_tokens += num_tokens_in
        total_output_tokens += num_tokens_out

        ttft = stage_metrics.get("vllm_ttft_ms")
        if ttft is None:
            ttft = stage_metrics.get("serving_time_to_first_output_ms")
        ttft_value = _safe_float(ttft)
        if ttft_value is not None:
            ttft_ms.append(ttft_value)

        tpot = stage_metrics.get("vllm_tpot_ms")
        if tpot is None:
            tpot = stage_metrics.get("time_per_output_unit_ms")
        tpot_value = _safe_float(tpot)
        if (tpot_value is None or tpot_value <= 0.0) and num_tokens_out > 1:
            gen_ms = _safe_float(stage_metrics.get("stage_gen_time_ms"))
            if gen_ms is not None and ttft_value is not None:
                fallback_tpot = (gen_ms - ttft_value) / (num_tokens_out - 1)
                if fallback_tpot > 0.0:
                    tpot_value = fallback_tpot
        if tpot_value is not None and tpot_value > 0.0:
            tpot_ms.append(tpot_value)

    total_tokens = total_input_tokens + total_output_tokens
    return {
        "configuration": configuration,
        "request_throughput": (len(outputs) / elapsed_s) if elapsed_s > 0 else 0.0,
        "mean_ttft_ms": (sum(ttft_ms) / len(ttft_ms)) if ttft_ms else 0.0,
        "p50_ttft_ms": _percentile(ttft_ms, 50.0),
        "p90_ttft_ms": _percentile(ttft_ms, 90.0),
        "p50_tpot_ms": _percentile(tpot_ms, 50.0),
        "total_token_throughput": (total_tokens / elapsed_s) if elapsed_s > 0 else 0.0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "num_requests": len(outputs),
        "non_empty_text_outputs": non_empty_text_outputs,
        "empty_text_outputs": len(outputs) - non_empty_text_outputs,
    }


def _print_benchmark_metrics(metrics: dict[str, Any]) -> None:
    print("=" * 60)
    print("Benchmark Metrics:")
    print(f"  Configuration           : {metrics.get('configuration')}")
    print(f"  Request Throughput      : {metrics.get('request_throughput', 0.0):.3f} req/s")
    print(f"  Mean TTFT               : {metrics.get('mean_ttft_ms', 0.0):.3f} ms")
    print(f"  P50 TTFT                : {metrics.get('p50_ttft_ms', 0.0):.3f} ms")
    print(f"  P90 TTFT                : {metrics.get('p90_ttft_ms', 0.0):.3f} ms")
    print(f"  P50 TPOT                : {metrics.get('p50_tpot_ms', 0.0):.3f} ms")
    print(f"  Total Token Throughput   : {metrics.get('total_token_throughput', 0.0):.3f} tok/s")
    print(
        "  Text Outputs            : "
        f"{metrics.get('non_empty_text_outputs', 0)}/{metrics.get('num_requests', 0)} non-empty"
    )
    print("=" * 60)


def _emit_outputs(outputs: list[Any], output_dir: str, *, print_prefix: str = "") -> None:
    img_idx = 0
    for req_output in outputs:
        ro = getattr(req_output, "request_output", None)
        txt = _extract_text(req_output)
        if txt:
            print(f"{print_prefix}[Output] Text:\n{txt}")

        images = getattr(req_output, "images", None)
        if not images and ro and hasattr(ro, "images"):
            images = ro.images
        if images:
            for j, img in enumerate(images):
                save_path = os.path.join(output_dir, f"output_{img_idx}_{j}.png")
                img.save(save_path)
                print(f"{print_prefix}[Output] Saved image to {save_path}")
            img_idx += 1


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    additional_config = parse_additional_config(args.additional_config)

    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must both be specified or both omitted.")

    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.warmup_runs < 0:
        raise ValueError(f"--warmup-runs must be non-negative, got {args.warmup_runs}")
    if args.profile_runs <= 0:
        raise ValueError(f"--profile-runs must be positive, got {args.profile_runs}")

    if args.image_paths and args.image_path:
        raise ValueError("--image-path and --image-paths are mutually exclusive.")

    prompts = load_prompts(args)
    plan = build_request_plan(args)
    deploy_config = _create_patched_deploy_config(args.deploy_config, len(plan))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    formatted_prompts: list[OmniPromptType] = []
    for req_idx, (prompt, image_paths) in enumerate(plan):
        image_payload = load_image_payload(image_paths)
        first_image = image_payload[0] if isinstance(image_payload, list) else image_payload
        height = args.height if args.height is not None else first_image.height
        width = args.width if args.width is not None else first_image.width
        result = build_prompt_tokens(
            prompt,
            tokenizer,
            task="it2i",
            bot_task=args.bot_task,
            sys_type=args.sys_type,
            num_images=len(image_paths),
        )
        formatted_prompts.append(
            {
                "prompt_token_ids": result.token_ids,
                "prompt": prompt,
                "use_system_prompt": args.sys_type or resolve_sys_type(args.bot_task),
                "modalities": ["image"],
                "multi_modal_data": {"image": image_payload},
                "height": height,
                "width": width,
                "request_index": req_idx,
            }
        )

    omni_kwargs = {
        "model": args.model,
        "vae_use_tiling": args.vae_use_tiling,
        "log_stats": args.log_stats,
        "init_timeout": args.init_timeout,
        "enforce_eager": args.enforce_eager,
        "mode": "image-editing",
        "deploy_config": deploy_config,
    }
    if additional_config is not None:
        omni_kwargs["additional_config"] = additional_config
    omni = Omni(**omni_kwargs)

    params_list = list(omni.default_sampling_params_list)
    ar_stop_token_ids = resolve_stop_token_ids(task="it2i", bot_task=args.bot_task, tokenizer=tokenizer)
    for sp in params_list:
        if isinstance(sp, OmniDiffusionSamplingParams):
            sp.num_inference_steps = args.steps
            sp.guidance_scale = args.guidance_scale
            sp.guidance_scale_provided = True
            sp.seed = args.seed
        elif hasattr(sp, "stop_token_ids"):
            sp.stop_token_ids = ar_stop_token_ids

    print(f"\n{'=' * 72}")
    print("HunyuanImage-3.0 AR VAE DP Batch Test")
    print(f"  Model: {args.model}")
    print(f"  Deploy config: {deploy_config}")
    print(f"  Requests: {len(formatted_prompts)}")
    print(f"  Images per request: {[len(paths) for _, paths in plan]}")
    print(f"  Height/Width: {formatted_prompts[0]['height']}x{formatted_prompts[0]['width']}")
    print(f"  Steps: {args.steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Seed: {args.seed}")
    print(f"  Warmup runs: {args.warmup_runs}")
    print(f"  Profile runs: {args.profile_runs}")
    print(f"{'=' * 72}\n")

    for warmup_idx in range(args.warmup_runs):
        print(f"[warmup] {warmup_idx + 1}/{args.warmup_runs}")
        run_batch_admission(omni, prompts=formatted_prompts, sampling_params_list=params_list, log_prefix="[warmup]")

    run_summaries: list[dict[str, Any]] = []
    outputs_path = os.path.join(args.output, "outputs.json")
    summary_path = os.path.join(args.output, "summary.json")
    for run_idx in range(args.profile_runs):
        print(f"[run] {run_idx + 1}/{args.profile_runs}")
        start = time.perf_counter()
        omni_outputs = run_batch_admission(omni, prompts=formatted_prompts, sampling_params_list=params_list)
        elapsed_s = time.perf_counter() - start
        serialized_outputs = [
            {
                "request_index": req_idx,
                "text": _extract_text(req_output),
                "stage_durations": getattr(req_output, "stage_durations", {}) or {},
                "metrics": getattr(req_output, "metrics", {}) or {},
            }
            for req_idx, req_output in enumerate(omni_outputs)
        ]
        benchmark_metrics = _benchmark_metrics(
            omni_outputs,
            elapsed_s=elapsed_s,
            configuration=deploy_config,
        )
        run_summary = {
            "run_index": run_idx,
            "elapsed_s": elapsed_s,
            "num_requests": len(omni_outputs),
            "num_images_per_request": [len(paths) for _, paths in plan],
            "benchmark_metrics": benchmark_metrics,
            "outputs": serialized_outputs,
        }
        run_summaries.append(run_summary)
        print(f"[run] elapsed_s={elapsed_s:.4f} num_requests={len(omni_outputs)}")
        _print_benchmark_metrics(benchmark_metrics)
        _emit_outputs(omni_outputs, args.output, print_prefix=f"[run {run_idx + 1}] ")

    with open(outputs_path, "w", encoding="utf-8") as f:
        json.dump(run_summaries, f, ensure_ascii=False, indent=2, default=str)
    summary = {
        "mode": "image-editing",
        "model": args.model,
        "deploy_config": deploy_config,
        "image_path": args.image_path,
        "image_paths": args.image_paths,
        "prompts": prompts,
        "batch_size": len(formatted_prompts),
        "warmup_runs": args.warmup_runs,
        "profile_runs": args.profile_runs,
        "batch_admission": True,
        "benchmark_metrics": run_summaries[-1]["benchmark_metrics"] if run_summaries else {},
        "outputs_json": outputs_path,
        "summary_json": summary_path,
        "runs": run_summaries,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[done] summary={summary_path}")
    print(f"[done] outputs={outputs_path}")


if __name__ == "__main__":
    main()
