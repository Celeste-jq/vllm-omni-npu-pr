# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from tools.hunyuan_image3_it2i_npu_experiment import (
    apply_deploy_overrides,
    build_concurrency_values,
    estimate_max_concurrency,
    parse_num_blocks_from_log,
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
