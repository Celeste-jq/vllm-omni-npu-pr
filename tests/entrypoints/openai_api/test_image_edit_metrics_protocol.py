# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.entrypoints.openai.protocol.images import ImageData, ImageEditImageChunk, ImageGenerationResponse

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_image_generation_response_exposes_stage_metrics() -> None:
    response = ImageGenerationResponse(
        created=123,
        data=[ImageData(b64_json="abc")],
        output_format="png",
        size="1024x1024",
        stage_durations={"ar": 0.3, "dit": 1.2},
        peak_memory_mb=4096.0,
    )

    assert response.stage_durations == {"ar": 0.3, "dit": 1.2}
    assert response.peak_memory_mb == 4096.0


def test_image_edit_stream_image_chunk_exposes_stage_metrics() -> None:
    chunk = ImageEditImageChunk(
        data=[ImageData(b64_json="abc")],
        output_format="png",
        size="1024x1024",
        created=123,
        model="tencent/HunyuanImage-3.0-Instruct",
        stage_durations={"ar": 0.4, "dit": 1.5},
        peak_memory_mb=5120.0,
    )

    assert chunk.stage_durations == {"ar": 0.4, "dit": 1.5}
    assert chunk.peak_memory_mb == 5120.0
