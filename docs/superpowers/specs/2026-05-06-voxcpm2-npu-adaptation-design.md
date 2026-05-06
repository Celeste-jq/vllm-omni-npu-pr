# VoxCPM2 NPU Adaptation Design

## Goal

Make VoxCPM2 run on Huawei Ascend NPU in vLLM-Omni while preserving the existing CUDA behavior.

The first supported NPU surface must include:

- Offline zero-shot TTS.
- Offline voice cloning / continuation inputs.
- Online OpenAI Speech API requests.
- Streaming audio output.
- Batched and mixed prefill/decode scheduling.

## Current State

VoxCPM2 is implemented as a single-stage autoregressive TTS pipeline:

- Pipeline key: `voxcpm2`.
- Model arch: `VoxCPM2TalkerForConditionalGeneration`.
- Deploy config: `vllm_omni/deploy/voxcpm2.yaml`.
- Model-local implementation: `vllm_omni/model_executor/models/voxcpm2/`.

The NPU platform and NPU AR worker already exist. VoxCPM2 currently fails on NPU because the talker implementation contains CUDA-only runtime assumptions:

- Native TTS side model is moved with `to("cuda")`.
- Runtime device is stored as the literal string `"cuda"`.
- Cache cleanup calls `torch.cuda.empty_cache()`.
- Profiling uses `torch.cuda.Event` and `torch.cuda.synchronize()`.
- Decode optimization uses `torch.cuda.CUDAGraph`, `torch.cuda.graph`, and CUDA graph pools.
- `torch.compile` is enabled unconditionally even though the NPU platform reports no TorchInductor support.

The CUDA Graph path is an optimization, not a functional requirement. NPU should use the eager path first.

## Constraints

Keep the diff local to VoxCPM2 wherever possible. Shared framework files should only change when the behavior is a real cross-model contract.

Do not redesign VoxCPM2 as a two-stage model. Its current single-stage native AR shape is the intended architecture.

Do not remove CUDA-specific optimizations. CUDA should continue to use CUDA Graph and existing compile behavior unless a test proves a regression.

Do not introduce new deploy YAML conventions just for VoxCPM2 NPU.

## Approach

### Device Resolution

Add a small VoxCPM2-local runtime helper that resolves the active torch device from `current_omni_platform.get_torch_device()` with a fallback to `vllm_config.device_config.device`.

The helper returns a `torch.device`, not a string. VoxCPM2 should store the resolved value as `self._device`.

Expected behavior:

- CUDA worker resolves to `torch.device("cuda", index)` or `torch.device("cuda")`.
- NPU worker resolves to `torch.device("npu", index)` or `torch.device("npu")`.
- CPU fallback is only for import-time/unit-test construction paths, not a supported inference target.

### Native TTS Side Model

Move the native TTS side modules to the resolved device:

```python
self._tts = native.tts_model.to(self._device)
```

All later `.to(self._device)` calls should continue to work after `self._device` becomes a `torch.device`.

After deleting duplicated native `base_lm` and `residual_lm`, clear memory through a device-aware helper:

- CUDA: `torch.cuda.empty_cache()`.
- NPU: `torch.npu.empty_cache()` when available.
- Other devices: no-op.

### Optimization Gates

Derive feature flags from the resolved device:

- `_enable_cuda_graph = self._device.type == "cuda"`.
- `_enable_torch_compile = self._device.type == "cuda"` for the initial NPU support.
- `_compile_vae = self._device.type == "cuda"`.

NPU support should prioritize correctness and compatibility. NPU graph or compile acceleration can be added later behind explicit NPU-specific evidence.

### Profiling

Replace the CUDA-only profiling timer with a backend-aware timer.

For CUDA:

- Keep `torch.cuda.Event` timing.

For NPU:

- Either use `torch.npu.Event` if available and compatible, or make profiling a wall-clock timer with `torch.npu.synchronize()`.
- If event support is uncertain, prefer a conservative no-op timer unless `VOXCPM2_PROFILE=1` is set.

The timer must never import or call `torch.cuda` on NPU execution paths.

### CUDA Graph Path

Keep `_CapturedGraph`, `_get_cuda_graph_pool`, `_capture_graph`, and `_replay_graph` CUDA-only.

Runtime guards must prevent these methods from being called unless `self._device.type == "cuda"`.

On NPU, decode uses:

- `self.model(...)` for scaffold.
- `self.residual_model(...)` for residual LM.
- Existing CFM and VAE eager execution.

### Inputs, Voice Cloning, And Streaming

The existing prompt construction, speaker cache, request state, VAE sliding-window decode, and `make_omni_output()` contract remain unchanged.

This preserves:

- `build_voxcpm2_prompt()`.
- `_build_prompt_cache()`.
- Raw audio input from the Speech API.
- Reference audio and continuation modes.
- Per-step streaming chunks under `model_outputs`.
- Per-request state cleanup used by mixed prefill/decode scheduling.

### Deploy Config

Keep `vllm_omni/deploy/voxcpm2.yaml` as the default config.

Only comments may need updates to avoid implying CUDA-only execution. Existing fields such as `gpu_memory_utilization` are framework-level names and should not be renamed for NPU.

## Tests

Use test-first changes for behavior that can be validated without NPU hardware.

Add focused unit tests around VoxCPM2-local helpers:

- Device resolution returns the current omni platform device when available.
- Native side runtime setup uses `npu` without touching `torch.cuda`.
- CUDA Graph is disabled for NPU.
- Torch compile flags are disabled for NPU.
- Device-aware cache clearing calls NPU cleanup when available.

Extend hardware/E2E coverage:

- Existing VoxCPM2 offline tests should include NPU markers where the test infrastructure supports it.
- Zero-shot TTS on one NPU.
- Voice clone on one NPU when reference audio is available.
- Mixed prefill/decode batch on one NPU.

Online validation should use the existing OpenAI Speech API path:

- Non-streaming speech request.
- Streaming speech request.
- Uploaded or raw reference audio path if the environment has the needed fixtures.

## Validation Commands

Fast local checks:

```bash
python3 -m py_compile vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py
pytest tests/model_executor/models/voxcpm2 -q
```

NPU environment checks:

```bash
python3 -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
```

NPU E2E checks:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 pytest tests/e2e/offline_inference/test_voxcpm2.py -q -m npu
```

Online smoke check:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 vllm serve openbmb/VoxCPM2 --omni --host 0.0.0.0 --port 8000
python examples/online_serving/text_to_speech/voxcpm2/openai_speech_client.py --text "Hello, this is VoxCPM2 on NPU."
```

## Risks

The largest unknown is operator support in the native VoxCPM2 side modules on NPU, especially LocDiT and AudioVAE. If an operator is unsupported, the fix should stay model-local when possible, or use a known NPU-compatible PyTorch replacement.

PagedAttention backend compatibility is delegated to the existing NPU AR worker and vLLM-Ascend integration. If VoxCPM2 hits a PagedAttention metadata issue, that may require a narrow NPU runner fix, but it should be proven by an NPU failure log before changing shared code.

Disabling compile and CUDA Graph on NPU may make first support slower than CUDA. That is acceptable for the first functional adaptation.

## Acceptance Criteria

- VoxCPM2 no longer hardcodes CUDA for its native TTS side modules.
- CUDA behavior remains unchanged for existing tests.
- NPU execution does not call CUDA-only APIs.
- Offline zero-shot, voice clone, and mixed prefill/decode work on NPU.
- Online Speech API works on NPU in both non-streaming and streaming modes.
- Any unavailable validation is documented with the exact missing dependency or hardware reason.
