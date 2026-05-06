# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VoxCPM2 platform-aware runtime behavior."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from vllm_omni.model_executor.models.voxcpm2 import voxcpm2_talker as tk  # noqa: E402


def test_resolve_runtime_device_prefers_current_omni_platform(monkeypatch) -> None:
    monkeypatch.setattr(
        tk,
        "_resolve_current_omni_platform_device",
        lambda: torch.device("npu:0"),
        raising=False,
    )
    cfg = SimpleNamespace(device_config=SimpleNamespace(device="cuda:0"))

    assert tk._resolve_runtime_device(cfg) == torch.device("npu:0")


def test_resolve_runtime_device_falls_back_to_vllm_config(monkeypatch) -> None:
    monkeypatch.setattr(
        tk,
        "_resolve_current_omni_platform_device",
        lambda: None,
        raising=False,
    )
    cfg = SimpleNamespace(device_config=SimpleNamespace(device="npu:1"))

    assert tk._resolve_runtime_device(cfg) == torch.device("npu:1")


def test_resolve_runtime_device_falls_back_to_cpu_without_config(monkeypatch) -> None:
    monkeypatch.setattr(
        tk,
        "_resolve_current_omni_platform_device",
        lambda: None,
        raising=False,
    )

    assert tk._resolve_runtime_device(SimpleNamespace()) == torch.device("cpu")


def test_resolve_current_omni_platform_device_propagates_platform_errors(monkeypatch) -> None:
    class BrokenPlatform:

        @staticmethod
        def get_torch_device() -> str:
            raise RuntimeError("platform failed")

    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.platforms",
        SimpleNamespace(current_omni_platform=BrokenPlatform()),
    )

    with pytest.raises(RuntimeError, match="platform failed"):
        tk._resolve_current_omni_platform_device()


def test_clear_device_cache_uses_npu_without_touching_cuda(monkeypatch) -> None:
    npu_empty_cache = Mock()
    cuda_empty_cache = Mock(side_effect=AssertionError("cuda cache must not be touched"))
    monkeypatch.setattr(tk.torch, "npu", SimpleNamespace(empty_cache=npu_empty_cache), raising=False)
    monkeypatch.setattr(tk.torch, "cuda", SimpleNamespace(empty_cache=cuda_empty_cache), raising=True)

    tk._clear_device_cache(torch.device("npu"))

    npu_empty_cache.assert_called_once_with()
    cuda_empty_cache.assert_not_called()


def test_runtime_flags_disable_cuda_features_on_npu() -> None:
    assert tk._runtime_flags_for_device(torch.device("npu")) == {
        "enable_cuda_graph": False,
        "enable_torch_compile": False,
        "compile_vae": False,
    }


def test_runtime_flags_keep_cuda_features_on_cuda() -> None:
    assert tk._runtime_flags_for_device(torch.device("cuda")) == {
        "enable_cuda_graph": True,
        "enable_torch_compile": True,
        "compile_vae": True,
    }


def test_talker_constructor_moves_native_tts_to_resolved_npu(monkeypatch) -> None:
    moved_to: list[torch.device] = []
    cleared: list[torch.device] = []

    class DummyScaffold(torch.nn.Module):
        make_empty_intermediate_tensors = object()

        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()

    class DummyResidual(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.loaded_from_native = False

        def load_weights_from_native(self, native_residual_lm):
            self.loaded_from_native = native_residual_lm == "native-residual"
            return 1

    class DummyTTS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion_concat_proj = torch.nn.Linear(1, 1, bias=False)
            self.patch_size = 4
            self.feat_dim = 64
            self.base_lm = "native-base"
            self.residual_lm = "native-residual"

        def to(self, device):
            moved_to.append(torch.device(device))
            return self

    class DummyVoxCPM:
        @staticmethod
        def from_pretrained(model_path, load_denoiser=False, optimize=False):
            assert model_path == "dummy-voxcpm2"
            assert load_denoiser is False
            assert optimize is False
            return SimpleNamespace(tts_model=DummyTTS())

    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            model="dummy-voxcpm2",
            hf_config=SimpleNamespace(sample_rate=48000),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=7),
    )

    monkeypatch.setattr(tk, "MiniCPM4PagedForVoxCPM2", DummyScaffold)
    monkeypatch.setattr(tk, "MiniCPM4PagedResidualLM", DummyResidual)
    monkeypatch.setattr(tk, "import_voxcpm2_core", lambda: DummyVoxCPM)
    monkeypatch.setattr(tk, "_resolve_runtime_device", lambda vllm_config: torch.device("npu"))
    monkeypatch.setattr(tk, "_clear_device_cache", lambda device: cleared.append(torch.device(device)))
    monkeypatch.setattr(tk, "get_speaker_cache", lambda: object())

    talker = tk.VoxCPM2TalkerForConditionalGeneration(vllm_config=cfg)

    assert moved_to == [torch.device("npu")]
    assert cleared == [torch.device("npu")]
    assert talker._device == torch.device("npu")
    assert talker._enable_cuda_graph is False
    assert talker._enable_torch_compile is False
    assert talker._compile_vae is False
    assert talker._max_batch_size == 7
    assert talker._tts.base_lm is None
    assert talker._tts.residual_lm is None
    assert talker.residual_model.loaded_from_native is True
