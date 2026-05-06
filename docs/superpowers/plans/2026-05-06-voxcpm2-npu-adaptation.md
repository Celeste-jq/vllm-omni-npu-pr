# VoxCPM2 NPU Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VoxCPM2 run on Ascend NPU for offline, online, streaming, voice clone, and mixed prefill/decode paths while preserving CUDA behavior.

**Architecture:** Keep the adaptation in VoxCPM2-local code. Add small runtime helpers for device resolution, cache cleanup, optimization gates, and profiling; wire them into `VoxCPM2TalkerForConditionalGeneration`; keep CUDA Graph and TorchInductor compile on CUDA only; use eager execution on NPU.

**Tech Stack:** Python, PyTorch, torch_npu, vLLM-Ascend, vLLM-Omni AR worker, pytest.

---

## File Structure

- Modify `vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py`
  - Owns VoxCPM2 runtime device resolution, side-model placement, profiling, CUDA Graph gating, and audio output state.
- Create `tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py`
  - Unit tests for device helpers, NPU-safe cache cleanup, runtime flag selection, constructor integration, profiling, and CUDA Graph guards.
- Modify `tests/e2e/offline_inference/test_voxcpm2.py`
  - Add NPU hardware marks to the existing offline zero-shot, voice clone, and mixed prefill/decode tests.
- Create `tests/e2e/online_serving/test_voxcpm2.py`
  - Add online OpenAI Speech API E2E coverage for non-streaming, streaming, and reference-audio requests.
- Modify `vllm_omni/deploy/voxcpm2.yaml`
  - Update comments so the deploy config no longer reads CUDA-only.
- Modify `examples/offline_inference/text_to_speech/README.md`
  - Add NPU usage notes under the VoxCPM2 section.
- Modify `examples/online_serving/text_to_speech/README.md`
  - Add NPU serving notes under the VoxCPM2 section.

---

### Task 1: Add VoxCPM2 Runtime Helper Tests

**Files:**
- Create: `tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py`
- Modify: `vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py`

- [ ] **Step 1: Write failing tests for runtime helpers**

Create `tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py` with this content:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VoxCPM2 platform-aware runtime behavior."""

from __future__ import annotations

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
```

- [ ] **Step 2: Run tests and verify they fail because helpers do not exist**

Run:

```bash
pytest tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py -q
```

Expected: FAIL with errors mentioning missing attributes such as `_resolve_runtime_device`, `_clear_device_cache`, or `_runtime_flags_for_device`.

- [ ] **Step 3: Add minimal runtime helpers**

In `vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py`, add these helpers after `_ACTIVE_STATE_LEAK_WARN_MIN`:

```python
def _resolve_current_omni_platform_device() -> torch.device | None:
    try:
        from vllm_omni.platforms import current_omni_platform

        return torch.device(current_omni_platform.get_torch_device())
    except Exception:
        return None


def _resolve_runtime_device(vllm_config: VllmConfig | None) -> torch.device:
    platform_device = _resolve_current_omni_platform_device()
    if platform_device is not None:
        return platform_device

    device = getattr(getattr(vllm_config, "device_config", None), "device", None)
    if isinstance(device, torch.device):
        return device
    if device:
        return torch.device(device)
    return torch.device("cpu")


def _clear_device_cache(device: torch.device) -> None:
    backend = getattr(torch, device.type, None)
    empty_cache = getattr(backend, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _runtime_flags_for_device(device: torch.device) -> dict[str, bool]:
    is_cuda = device.type == "cuda"
    return {
        "enable_cuda_graph": is_cuda,
        "enable_torch_compile": is_cuda,
        "compile_vae": is_cuda,
    }
```

- [ ] **Step 4: Run helper tests and verify they pass**

Run:

```bash
pytest tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py -q
```

Expected: PASS for the helper tests.

- [ ] **Step 5: Commit**

Run:

```bash
git add tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py
git commit -m "test: cover voxcpm2 runtime device helpers"
```

---

### Task 2: Wire Runtime Device Into VoxCPM2 Talker Initialization

**Files:**
- Modify: `tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py`
- Modify: `vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py`

- [ ] **Step 1: Add failing constructor integration test**

Append this test code to `tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py`:

```python
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
```

- [ ] **Step 2: Run constructor test and verify it fails on CUDA hardcoding**

Run:

```bash
pytest tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py::test_talker_constructor_moves_native_tts_to_resolved_npu -q
```

Expected: FAIL because the constructor still calls `native.tts_model.to("cuda")`, stores `"cuda"`, or calls `torch.cuda.empty_cache()`.

- [ ] **Step 3: Change constructor to use resolved device and runtime flags**

In `VoxCPM2TalkerForConditionalGeneration.__init__`, change the native TTS initialization block from:

```python
native = VoxCPM.from_pretrained(model_path, load_denoiser=False, optimize=False)
self._tts: nn.Module = native.tts_model.to("cuda")
self._side_dtype = self._tts.fusion_concat_proj.weight.dtype
self._device = "cuda"
```

to:

```python
native = VoxCPM.from_pretrained(model_path, load_denoiser=False, optimize=False)
self._device = _resolve_runtime_device(vllm_config)
self._tts: nn.Module = native.tts_model.to(self._device)
self._side_dtype = self._tts.fusion_concat_proj.weight.dtype
```

Change:

```python
torch.cuda.empty_cache()
```

to:

```python
_clear_device_cache(self._device)
```

Change:

```python
self._enable_torch_compile = True
self._compile_vae = True
```

to:

```python
runtime_flags = _runtime_flags_for_device(self._device)
self._enable_torch_compile = runtime_flags["enable_torch_compile"]
self._compile_vae = runtime_flags["compile_vae"]
```

Change:

```python
self._perf = _PerfTimer(enabled=_ENABLE_PROFILING)
self._cfm_buffers: _CFMBufferManager | None = None
self._enable_cuda_graph = True
```

to:

```python
self._perf = _PerfTimer(enabled=_ENABLE_PROFILING, device=self._device)
self._cfm_buffers: _CFMBufferManager | None = None
self._enable_cuda_graph = runtime_flags["enable_cuda_graph"]
```

- [ ] **Step 4: Run constructor and helper tests**

Run:

```bash
pytest tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Run existing VoxCPM2 state tests**

Run:

```bash
pytest tests/model_executor/models/voxcpm2/test_talker_state_eviction.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py
git commit -m "fix: initialize voxcpm2 runtime on active device"
```

---

### Task 3: Make VoxCPM2 Profiling Backend-Aware

**Files:**
- Modify: `tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py`
- Modify: `vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py`

- [ ] **Step 1: Add failing NPU profiling test**

Append this test to `tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py`:

```python
def test_perf_timer_on_npu_uses_device_sync_without_cuda_events(monkeypatch) -> None:
    sync_calls: list[str] = []

    class ForbiddenCuda:
        @staticmethod
        def Event(*args, **kwargs):
            raise AssertionError("cuda Event must not be constructed for npu profiling")

        @staticmethod
        def synchronize():
            raise AssertionError("cuda synchronize must not be called for npu profiling")

    monkeypatch.setattr(tk.torch, "cuda", ForbiddenCuda, raising=True)
    monkeypatch.setattr(
        tk.torch,
        "npu",
        SimpleNamespace(synchronize=lambda: sync_calls.append("npu")),
        raising=False,
    )

    timer = tk._PerfTimer(enabled=True, device=torch.device("npu"))
    timer.start("decode_step")
    timer.stop("decode_step")
    summary = timer.breakdown()

    assert "decode_step" in summary
    assert sync_calls
```

- [ ] **Step 2: Run profiling test and verify it fails on `_PerfTimer` signature or CUDA usage**

Run:

```bash
pytest tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py::test_perf_timer_on_npu_uses_device_sync_without_cuda_events -q
```

Expected: FAIL because `_PerfTimer` does not accept `device` or uses `torch.cuda.Event`.

- [ ] **Step 3: Add device synchronization helper**

In `vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py`, add this helper near `_clear_device_cache`:

```python
def _synchronize_device(device: torch.device) -> None:
    backend = getattr(torch, device.type, None)
    synchronize = getattr(backend, "synchronize", None)
    if callable(synchronize):
        synchronize()
```

- [ ] **Step 4: Replace `_PerfTimer` implementation**

Replace the `_PerfTimer` class with this backend-aware version:

```python
class _PerfTimer:
    __slots__ = (
        "_device",
        "_enabled",
        "_timers",
        "_counts",
        "_event_starts",
        "_event_pairs",
        "_wall_starts",
    )

    def __init__(self, enabled: bool = False, device: torch.device | None = None):
        self._device = device or torch.device("cuda")
        self._enabled = enabled
        self._timers: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._event_starts: dict[str, Any] = {}
        self._event_pairs: list[tuple[str, Any, Any]] = []
        self._wall_starts: dict[str, float] = {}

    def start(self, name: str) -> None:
        if not self._enabled:
            return
        if self._device.type == "cuda":
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            self._event_starts[name] = evt
            return
        _synchronize_device(self._device)
        self._wall_starts[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if not self._enabled:
            return
        if self._device.type == "cuda":
            if name not in self._event_starts:
                return
            start_evt = self._event_starts.pop(name)
            end_evt = torch.cuda.Event(enable_timing=True)
            end_evt.record()
            self._event_pairs.append((name, start_evt, end_evt))
            return
        start = self._wall_starts.pop(name, None)
        if start is None:
            return
        _synchronize_device(self._device)
        self._timers[name] = self._timers.get(name, 0.0) + (time.perf_counter() - start) * 1000.0
        self._counts[name] = self._counts.get(name, 0) + 1

    def _resolve(self) -> None:
        if not self._event_pairs:
            return
        torch.cuda.synchronize()
        for name, s, e in self._event_pairs:
            self._timers[name] = self._timers.get(name, 0.0) + s.elapsed_time(e)
            self._counts[name] = self._counts.get(name, 0) + 1
        self._event_pairs.clear()

    def breakdown(self) -> str:
        if not self._enabled:
            return ""
        self._resolve()
        if not self._timers:
            return ""
        total = self._timers.get("decode_step", sum(self._timers.values()))
        lines = [
            "=== VoxCPM2 Decode Step Breakdown ===",
            f"{'Component':<30} | {'ms':>10} | {'%':>6} | {'N':>5} | {'avg':>8}",
            "-" * 70,
        ]
        for name in sorted(self._timers):
            t, c = self._timers[name], self._counts[name]
            pct = t / total * 100 if total else 0.0
            lines.append(f"{name:<30} | {t:>10.2f} | {pct:>5.1f}% | {c:>5} | {t / c:>8.3f}")
        lines.append(f"{'TOTAL':<30} | {total:>10.2f} |")
        return "\n".join(lines)

    def reset(self) -> None:
        self._timers.clear()
        self._counts.clear()
        self._event_starts.clear()
        self._event_pairs.clear()
        self._wall_starts.clear()
```

- [ ] **Step 5: Run profiling tests**

Run:

```bash
pytest tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py::test_perf_timer_on_npu_uses_device_sync_without_cuda_events -q
```

Expected: PASS.

- [ ] **Step 6: Run all VoxCPM2 unit tests**

Run:

```bash
pytest tests/model_executor/models/voxcpm2 -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py
git commit -m "fix: make voxcpm2 profiling device aware"
```

---

### Task 4: Guard CUDA Graph Access On Non-CUDA Devices

**Files:**
- Modify: `tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py`
- Modify: `vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py`

- [ ] **Step 1: Add failing tests for CUDA Graph guard behavior**

Append these tests to `tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py`:

```python
def _make_bare_talker_for_runtime() -> tk.VoxCPM2TalkerForConditionalGeneration:
    talker = tk.VoxCPM2TalkerForConditionalGeneration.__new__(tk.VoxCPM2TalkerForConditionalGeneration)
    talker._device = torch.device("npu")
    talker._enable_cuda_graph = True
    talker._cuda_graph_pool = None
    return talker


def test_can_use_cuda_graph_is_false_on_npu_even_if_flag_is_true() -> None:
    talker = _make_bare_talker_for_runtime()

    assert talker._can_use_cuda_graph(
        graph_ready=True,
        intermediate_tensors=None,
        inputs_embeds=torch.zeros(1, 1),
    ) is False


def test_get_cuda_graph_pool_rejects_npu_before_torch_cuda(monkeypatch) -> None:
    class ForbiddenCuda:
        @staticmethod
        def graph_pool_handle():
            raise AssertionError("cuda graph pool must not be touched on npu")

    monkeypatch.setattr(tk.torch, "cuda", ForbiddenCuda, raising=True)
    talker = _make_bare_talker_for_runtime()

    with pytest.raises(RuntimeError, match="CUDA Graph is only available on CUDA"):
        talker._get_cuda_graph_pool()
```

- [ ] **Step 2: Run guard tests and verify they fail**

Run:

```bash
pytest tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py::test_can_use_cuda_graph_is_false_on_npu_even_if_flag_is_true tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py::test_get_cuda_graph_pool_rejects_npu_before_torch_cuda -q
```

Expected: FAIL because `_can_use_cuda_graph` does not exist and `_get_cuda_graph_pool` has no non-CUDA guard.

- [ ] **Step 3: Add `_can_use_cuda_graph` method**

Add this method inside `VoxCPM2TalkerForConditionalGeneration`, immediately before `_get_cuda_graph_pool`:

```python
    def _can_use_cuda_graph(
        self,
        *,
        graph_ready: bool,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
    ) -> bool:
        return (
            self._device.type == "cuda"
            and self._enable_cuda_graph
            and graph_ready
            and intermediate_tensors is None
            and inputs_embeds is not None
        )
```

- [ ] **Step 4: Use `_can_use_cuda_graph` in `forward`**

Replace this block in `forward`:

```python
can_use_graph = (
    self._enable_cuda_graph and graph_ready and intermediate_tensors is None and inputs_embeds is not None
)
```

with:

```python
can_use_graph = self._can_use_cuda_graph(
    graph_ready=graph_ready,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
)
```

- [ ] **Step 5: Guard `_get_cuda_graph_pool`**

Change `_get_cuda_graph_pool` to:

```python
    def _get_cuda_graph_pool(self) -> tuple:
        if self._device.type != "cuda":
            raise RuntimeError("CUDA Graph is only available on CUDA devices")
        if self._cuda_graph_pool is None:
            self._cuda_graph_pool = torch.cuda.graph_pool_handle()
        return self._cuda_graph_pool
```

- [ ] **Step 6: Run guard tests and all VoxCPM2 unit tests**

Run:

```bash
pytest tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py::test_can_use_cuda_graph_is_false_on_npu_even_if_flag_is_true tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py::test_get_cuda_graph_pool_rejects_npu_before_torch_cuda -q
pytest tests/model_executor/models/voxcpm2 -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py
git commit -m "fix: guard voxcpm2 cuda graph path by device"
```

---

### Task 5: Add NPU Coverage To Offline VoxCPM2 E2E Tests

**Files:**
- Modify: `tests/e2e/offline_inference/test_voxcpm2.py`

- [ ] **Step 1: Update hardware marks**

In `tests/e2e/offline_inference/test_voxcpm2.py`, change each VoxCPM2 hardware decorator from:

```python
@hardware_test(res={"cuda": "L4"}, num_cards=1)
```

to:

```python
@hardware_test(res={"cuda": "L4", "npu": "A3"}, num_cards=1)
```

Apply the change to:

- `test_voxcpm2_zero_shot_001`.
- `test_voxcpm2_voice_clone_002`.
- `test_voxcpm2_prefill_decode_mixed_batch_003`.

- [ ] **Step 2: Run collection for the offline test file**

Run:

```bash
pytest tests/e2e/offline_inference/test_voxcpm2.py --collect-only -q
```

Expected: PASS collection with three tests collected.

- [ ] **Step 3: Commit**

Run:

```bash
git add tests/e2e/offline_inference/test_voxcpm2.py
git commit -m "test: mark voxcpm2 offline e2e for npu"
```

---

### Task 6: Add Online Speech API E2E Coverage For VoxCPM2

**Files:**
- Create: `tests/e2e/online_serving/test_voxcpm2.py`

- [ ] **Step 1: Add online E2E test file**

Create `tests/e2e/online_serving/test_voxcpm2.py` with this content:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E online tests for VoxCPM2 through the OpenAI Speech API."""

from __future__ import annotations

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


def _speech_payload(*, stream: bool = False, ref_audio: str | None = None) -> dict:
    payload = {
        "model": VOXCPM2_MODEL,
        "input": "Hello, this is VoxCPM2 running through the OpenAI Speech API.",
        "stream": stream,
        "response_format": "wav" if stream else "wav",
    }
    if ref_audio is not None:
        payload["ref_audio"] = ref_audio
    return payload


voxcpm2_server_params = [
    pytest.param(
        OmniServerParams(
            model=VOXCPM2_MODEL,
            stage_config_path=get_deploy_config_path("voxcpm2.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="default",
    )
]


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", voxcpm2_server_params, indirect=True)
def test_voxcpm2_speech_non_streaming_001(omni_server, openai_client) -> None:
    openai_client.send_audio_speech_request(_speech_payload(stream=False))


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", voxcpm2_server_params, indirect=True)
def test_voxcpm2_speech_streaming_002(omni_server, openai_client) -> None:
    openai_client.send_audio_speech_request(_speech_payload(stream=True))


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", voxcpm2_server_params, indirect=True)
def test_voxcpm2_speech_reference_audio_003(omni_server, openai_client) -> None:
    openai_client.send_audio_speech_request(_speech_payload(stream=False, ref_audio=REF_AUDIO_URL))
```

- [ ] **Step 2: Run collection for online tests**

Run:

```bash
pytest tests/e2e/online_serving/test_voxcpm2.py --collect-only -q
```

Expected: PASS collection with three tests collected.

- [ ] **Step 3: Commit**

Run:

```bash
git add tests/e2e/online_serving/test_voxcpm2.py
git commit -m "test: add voxcpm2 online speech e2e"
```

---

### Task 7: Update VoxCPM2 NPU Docs And Deploy Comments

**Files:**
- Modify: `vllm_omni/deploy/voxcpm2.yaml`
- Modify: `examples/offline_inference/text_to_speech/README.md`
- Modify: `examples/online_serving/text_to_speech/README.md`

- [ ] **Step 1: Update deploy comment**

In `vllm_omni/deploy/voxcpm2.yaml`, change the header comment from:

```yaml
# Verified on 1x H20 141GB.
```

to:

```yaml
# Verified on CUDA H20 and designed to run on NPU through the eager path.
```

Change the `enforce_eager` comment from:

```yaml
#     not cudagraph-compatible (captured separately via voxcpm2_talker's own
#     _CapturedGraph path).
```

to:

```yaml
#     not framework-graph-compatible. CUDA may use the talker's own
#     _CapturedGraph path; NPU uses eager execution.
```

- [ ] **Step 2: Add offline README NPU notes**

In `examples/offline_inference/text_to_speech/README.md`, under the VoxCPM2 `### Quick start` block, add this section:

````markdown
### NPU
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=0 \
python examples/offline_inference/text_to_speech/voxcpm2/end2end.py \
    --model openbmb/VoxCPM2 \
    --text "Hello, this is a VoxCPM2 demo running on NPU."
```

VoxCPM2 uses the same deploy config on CUDA and NPU. CUDA keeps the talker-local CUDA Graph optimization; NPU runs the talker side path eagerly.
````

- [ ] **Step 3: Add online README NPU notes**

In `examples/online_serving/text_to_speech/README.md`, under the VoxCPM2 `### Launch` block, add this NPU launch example:

````markdown
For NPU:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=0 \
vllm serve openbmb/VoxCPM2 --omni --host 0.0.0.0 --port 8000
```
````

- [ ] **Step 4: Run markdown-adjacent checks**

Run:

```bash
python3 -m py_compile vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py
```

Expected: PASS with no output.

- [ ] **Step 5: Commit**

Run:

```bash
git add vllm_omni/deploy/voxcpm2.yaml examples/offline_inference/text_to_speech/README.md examples/online_serving/text_to_speech/README.md
git commit -m "docs: document voxcpm2 npu execution"
```

---

### Task 8: Final Verification

**Files:**
- Read: `vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py`
- Read: `tests/model_executor/models/voxcpm2/test_talker_npu_runtime.py`
- Read: `tests/e2e/offline_inference/test_voxcpm2.py`
- Read: `tests/e2e/online_serving/test_voxcpm2.py`

- [ ] **Step 1: Run unit verification**

Run:

```bash
python3 -m py_compile vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py
pytest tests/model_executor/models/voxcpm2 -q
```

Expected: PASS.

- [ ] **Step 2: Run E2E collection verification**

Run:

```bash
pytest tests/e2e/offline_inference/test_voxcpm2.py tests/e2e/online_serving/test_voxcpm2.py --collect-only -q
```

Expected: PASS collection.

- [ ] **Step 3: Run CUDA smoke if CUDA is available**

Run:

```bash
pytest tests/e2e/offline_inference/test_voxcpm2.py::test_voxcpm2_zero_shot_001 -q
```

Expected on CUDA machine with model access: PASS and generated audio duration between 0.5s and 30s.

- [ ] **Step 4: Run NPU environment check on Ascend machine**

Run:

```bash
python3 -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
```

Expected on Ascend machine: output starts with `True` and reports at least one NPU.

- [ ] **Step 5: Run NPU offline E2E on Ascend machine**

Run:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 pytest tests/e2e/offline_inference/test_voxcpm2.py -q -m npu
```

Expected on Ascend machine with model access: PASS for zero-shot, voice clone when reference audio is available, and mixed prefill/decode.

- [ ] **Step 6: Run NPU online E2E on Ascend machine**

Run:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 pytest tests/e2e/online_serving/test_voxcpm2.py -q -m npu
```

Expected on Ascend machine with model access: PASS for non-streaming, streaming, and reference-audio speech requests.

- [ ] **Step 7: Inspect git diff**

Run:

```bash
git status --short
git diff --stat
```

Expected: `git status --short` is empty after all task commits, or shows only intentionally uncommitted validation artifacts. `git diff --stat` is empty after all task commits.
