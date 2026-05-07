# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for VoxCPM2 native AR TTS.

These tests verify that /v1/audio/speech covers zero-shot synthesis,
streaming, and reference-audio cloning through the OpenAI-compatible
serving path.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

VOXCPM2_MODEL = "openbmb/VoxCPM2"
REF_AUDIO_URL = load_test_audio_data_url("qwen3_tts/clone_2.wav")
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

voxcpm2_server_params = [
    pytest.param(
        OmniServerParams(
            model=VOXCPM2_MODEL,
            stage_config_path=get_deploy_config_path("voxcpm2.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="voxcpm2",
    )
]


def _speech_payload(*, stream: bool = False, ref_audio: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": VOXCPM2_MODEL,
        "input": "Hello, this is a VoxCPM2 speech test.",
        "stream": stream,
        "response_format": "wav",
    }
    if ref_audio is not None:
        payload["ref_audio"] = ref_audio
        payload["ref_text"] = REF_TEXT
    return payload


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", voxcpm2_server_params, indirect=True)
def test_voxcpm2_speech_non_streaming_001(omni_server, openai_client) -> None:
    """
    Test zero-shot TTS via OpenAI Speech API.
    Deploy Setting: voxcpm2.yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = _speech_payload()
    request_config["model"] = omni_server.model
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", voxcpm2_server_params, indirect=True)
def test_voxcpm2_speech_streaming_002(omni_server, openai_client) -> None:
    """
    Test streamed TTS via OpenAI Speech API.
    Deploy Setting: voxcpm2.yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=True
    Datasets: single request
    """
    request_config = _speech_payload(stream=True)
    request_config["model"] = omni_server.model
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", voxcpm2_server_params, indirect=True)
def test_voxcpm2_speech_reference_audio_003(omni_server, openai_client) -> None:
    """
    Test reference-audio cloning via OpenAI Speech API.
    Deploy Setting: voxcpm2.yaml
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = _speech_payload(ref_audio=REF_AUDIO_URL)
    request_config["model"] = omni_server.model
    openai_client.send_audio_speech_request(request_config)
