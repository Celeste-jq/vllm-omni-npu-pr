# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from tools.hunyuan_image3_it2i_npu_experiment import (
    apply_deploy_overrides,
    build_concurrency_values,
    estimate_max_concurrency,
    parse_num_blocks_from_log,
    render_single_run_config,
)
from tools.hunyuan_image3_it2i_batch_test import (
    RequestMetric,
    build_server_env,
    config_summary as batch_config_summary,
    summarize_results as summarize_batch_results,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_estimate_max_concurrency_uses_block_formula() -> None:
    assert estimate_max_concurrency(num_blocks=1000, input_tokens=640, output_tokens=384) == 100


def test_build_concurrency_values_keeps_small_powers_of_two_and_target_neighbors() -> None:
    values = build_concurrency_values(target=21, maximum=32)

    assert values == [1, 2, 3, 4, 8, 16, 19, 20, 21, 22, 23, 24, 32]


def test_parse_num_blocks_from_log_reads_rank_zero_kv_cache_line() -> None:
    text = """
    random line
    [kv-cache-profile] rank=0 runner=NPUARModelRunner stage_id=0 num_blocks=2048 block_size=128
    """

    assert parse_num_blocks_from_log(text) == 2048


def test_apply_deploy_overrides_enables_graph_rope_vit_dp_and_batch_alignment() -> None:
    base = {
        "pipeline": "hunyuan_image_3_moe",
        "stages": [
            {
                "stage_id": 0,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.8,
                "enforce_eager": True,
                "tensor_parallel_size": 4,
                "hf_overrides": {"rope_parameters": {"rope_type": "default", "mrope_section": [0, 32, 32]}},
            },
            {
                "stage_id": 1,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.65,
                "enforce_eager": True,
                "parallel_config": {"tensor_parallel_size": 4},
            },
        ],
    }

    updated = apply_deploy_overrides(
        copy.deepcopy(base),
        batch_size=16,
        ar_gpu_memory_utilization=0.78,
        dit_gpu_memory_utilization=0.66,
        cudagraph_capture_sizes=[14, 15, 16, 17, 18],
    )

    ar_stage, dit_stage = updated["stages"]
    assert ar_stage["max_num_seqs"] == 16
    assert dit_stage["max_num_seqs"] == 16
    assert ar_stage["gpu_memory_utilization"] == pytest.approx(0.78)
    assert dit_stage["gpu_memory_utilization"] == pytest.approx(0.66)
    assert ar_stage["enforce_eager"] is False
    assert ar_stage["is_comprehension"] is True
    assert ar_stage["mm_encoder_tp_mode"] == "data"
    assert ar_stage["compilation_config"] == {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [14, 15, 16, 17, 18],
    }
    assert ar_stage["hf_overrides"]["rope_parameters"]["rope_type"] == "default"


def test_render_single_run_config_uses_requested_batch_and_memory_values() -> None:
    base = {
        "pipeline": "hunyuan_image_3_moe",
        "stages": [
            {"stage_id": 0, "max_num_seqs": 1, "gpu_memory_utilization": 0.8},
            {"stage_id": 1, "max_num_seqs": 1, "gpu_memory_utilization": 0.65},
        ],
        "edges": [{"from": 0, "to": 1}],
    }

    rendered = render_single_run_config(
        copy.deepcopy(base),
        batch_size=12,
        ar_gpu_memory_utilization=0.79,
        dit_gpu_memory_utilization=0.67,
    )

    assert rendered["stages"][0]["max_num_seqs"] == 12
    assert rendered["stages"][1]["max_num_seqs"] == 12
    assert rendered["stages"][0]["gpu_memory_utilization"] == pytest.approx(0.79)
    assert rendered["stages"][1]["gpu_memory_utilization"] == pytest.approx(0.67)
    assert rendered["stages"][0]["compilation_config"]["cudagraph_capture_sizes"] == [10, 11, 12, 13, 14]


def test_batch_config_summary_extracts_key_run_parameters() -> None:
    config = {
        "pipeline": "hunyuan_image3_it2i",
        "stages": [
            {
                "stage_id": 0,
                "max_num_seqs": 16,
                "gpu_memory_utilization": 0.91,
                "mm_encoder_tp_mode": "data",
                "compilation_config": {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [14, 15, 16, 17, 18],
                },
                "hf_overrides": {"rope_parameters": {"rope_type": "default", "mrope_section": [0, 32, 32]}},
            },
            {
                "stage_id": 1,
                "max_num_seqs": 16,
                "gpu_memory_utilization": 0.89,
            },
        ],
        "edges": [{"from": 0, "to": 1, "max_inflight": 16}],
    }

    summary = batch_config_summary(config)

    assert summary["pipeline"] == "hunyuan_image3_it2i"
    assert summary["ar_max_num_seqs"] == 16
    assert summary["dit_max_num_seqs"] == 16
    assert summary["edge_max_inflight"] == 16
    assert summary["ar_gpu_memory_utilization"] == pytest.approx(0.91)
    assert summary["dit_gpu_memory_utilization"] == pytest.approx(0.89)
    assert summary["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert summary["cudagraph_capture_sizes"] == [14, 15, 16, 17, 18]
    assert summary["mm_encoder_tp_mode"] == "data"
    assert summary["rope_enabled"] is True


def test_summarize_batch_results_reports_batch_metrics_without_concurrency_label() -> None:
    metrics = [
        RequestMetric(
            request_index=0,
            success=True,
            http_status=200,
            ttft_s=1.0,
            e2e_s=8.0,
            ar_delta_count=10,
            ar_text_chars=100,
            stage_durations={"stage_0": 1.5, "stage_1": 6.0},
            peak_memory_mb=1000.0,
            error="",
        ),
        RequestMetric(
            request_index=1,
            success=True,
            http_status=200,
            ttft_s=2.0,
            e2e_s=10.0,
            ar_delta_count=20,
            ar_text_chars=200,
            stage_durations={"stage_0": 2.5, "stage_1": 7.0},
            peak_memory_mb=1200.0,
            error="",
        ),
    ]

    summary = summarize_batch_results(
        batch_size=2,
        wall_time_s=10.0,
        metrics=metrics,
        num_blocks=1000,
        input_tokens=640,
        output_tokens=384,
    )

    assert summary["batch_size"] == 2
    assert summary["num_requests"] == 2
    assert summary["success"] == 2
    assert summary["fail"] == 0
    assert summary["success_rate"] == pytest.approx(1.0)
    assert summary["throughput_qps"] == pytest.approx(0.2)
    assert summary["ttft_mean_s"] == pytest.approx(1.5)
    assert summary["e2e_p95_s"] == pytest.approx(9.9)
    assert summary["ar_stage_mean_s"] == pytest.approx(2.0)
    assert summary["dit_stage_mean_s"] == pytest.approx(6.5)
    assert summary["peak_memory_mb_max"] == pytest.approx(1200.0)
    assert summary["blocks_per_request"] == 8
    assert summary["estimated_max_batch"] == 125


def test_batch_server_env_forces_npu_graph_task_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "2")
    monkeypatch.setenv("TASKQUEUEENABLE", "2")

    env = build_server_env()

    assert env["TASK_QUEUE_ENABLE"] == "1"
    assert env["TASKQUEUEENABLE"] == "1"
