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
