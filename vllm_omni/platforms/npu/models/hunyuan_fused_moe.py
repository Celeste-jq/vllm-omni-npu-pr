# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.nn as nn
import vllm.forward_context as _vllm_fc
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    init_model_parallel_group as vllm_init_model_parallel_group,
)
from vllm.distributed import get_ep_group, tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

from vllm_omni.diffusion.distributed.parallel_state import (
    get_data_parallel_world_size,
    get_sequence_parallel_world_size,
    get_world_group,
)
from vllm_omni.diffusion.forward_context import get_forward_context as omni_get_ctx


def _ensure_forward_context_attr(name: str, annotation: Any, default: Any) -> None:
    if name not in _vllm_fc.ForwardContext.__annotations__:
        _vllm_fc.ForwardContext.__annotations__[name] = annotation
    if not hasattr(_vllm_fc.ForwardContext, name):
        setattr(_vllm_fc.ForwardContext, name, default)


def _set_hunyuan_fused_moe_forward_context(num_tokens: int) -> None:
    if not _vllm_fc.is_forward_context_available():
        return

    forward_context = _vllm_fc.get_forward_context()
    forward_context.num_tokens = num_tokens
    forward_context.moe_comm_type = _select_moe_comm_method(vllm_config=omni_get_ctx().vllm_config)
    forward_context.moe_comm_method = _MoECommMethods.get(forward_context.moe_comm_type)
    forward_context.flash_comm_v1_enabled = False


def _init_mc2_group_for_diffusion(
    world_size: int,
    data_parallel_size: int,
    tensor_parallel_size: int,
    backend: str,
    local_rank: int,
) -> None:
    import vllm_ascend.distributed.parallel_state as vllm_ascend_parallel_state

    if getattr(vllm_ascend_parallel_state, "_MC2", None) is not None:
        return
    all_ranks = torch.arange(world_size).reshape(-1, data_parallel_size * tensor_parallel_size)
    group_ranks = all_ranks.unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]

    vllm_ascend_parallel_state._MC2 = vllm_init_model_parallel_group(
        group_ranks,
        local_rank,
        backend,
        group_name="mc2",
    )


def _select_moe_comm_method(vllm_config: VllmConfig) -> MoECommType | None:
    soc_version = get_ascend_device_type()
    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A2}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A3}:
        moe_comm_type = MoECommType.ALLTOALL
    elif soc_version in {AscendDeviceType._310P}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A5}:
        moe_comm_type = MoECommType.ALLTOALL
    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
    return moe_comm_type


def prepare_hunyuan_fused_moe_runtime() -> None:
    world_size = torch.distributed.get_world_size()
    data_parallel_size = get_data_parallel_world_size()
    tensor_parallel_size = get_tensor_model_parallel_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)
    local_rank = get_world_group().local_rank
    _init_mc2_group_for_diffusion(
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        backend=backend,
        local_rank=local_rank,
    )

    moe_comm_type = _select_moe_comm_method(vllm_config=omni_get_ctx().vllm_config)
    _ensure_forward_context_attr("num_tokens", int | None, None)
    _ensure_forward_context_attr("in_profile_run", bool, False)
    _ensure_forward_context_attr("moe_comm_type", MoECommType | None, moe_comm_type)
    _ensure_forward_context_attr("moe_comm_method", Any, _MoECommMethods.get(moe_comm_type))
    _ensure_forward_context_attr("flash_comm_v1_enabled", bool, False)


# NOTE: vLLM v0.20.0 folded SharedFusedMoE into FusedMoE, and vllm-ascend in turn
# removed AscendSharedFusedMoE — the shared-experts / gate / multistream-overlap
# paths now live directly on AscendFusedMoE and are activated by passing
# shared_experts= as a kwarg.
class AscendHunyuanFusedMoE(AscendFusedMoE):
    def __init__(self, *, prefix: str = "", **kwargs: Any) -> None:
        super().__init__(prefix=prefix, **kwargs)
        self._prefix = prefix

    def forward(self, hidden_states: Any, router_logits: Any) -> Any:
        _set_hunyuan_fused_moe_forward_context(hidden_states.shape[0])
        return super().forward(hidden_states, router_logits)

    def __del__(self):
        import vllm_ascend.distributed.parallel_state as vllm_ascend_parallel_state

        if vllm_ascend_parallel_state._MC2:
            vllm_ascend_parallel_state._MC2.destroy()
        vllm_ascend_parallel_state._MC2 = None


class MindIESDHunyuanFusedMoE(nn.Module):
    def __init__(
        self,
        *,
        prefix: str = "",
        shared_experts: nn.Module | None = None,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = False,
        quant_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise NotImplementedError("MindIE-SD Hunyuan MoE adapter currently supports non-quantized weights only.")
        self._prefix = prefix
        self.shared_experts = shared_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize
        ep_group = get_ep_group()
        self.ep_size = getattr(ep_group, "world_size", 1)
        self.ep_rank = getattr(ep_group, "rank_in_group", 0)
        if num_experts % self.ep_size != 0:
            raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={self.ep_size}.")
        self.local_num_experts = num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.local_num_experts
        self.w13_weight = nn.Parameter(torch.empty(self.local_num_experts, hidden_size, 2 * intermediate_size))
        self.w2_weight = nn.Parameter(torch.empty(self.local_num_experts, intermediate_size, hidden_size))
        self.w13_weight.weight_loader = self._load_w13_weight
        self.w2_weight.weight_loader = self._load_w2_weight
        self._shared_expert_stream: Any | None = None

    @staticmethod
    def make_expert_params_mapping(
        model: Any,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            model,
            ckpt_gate_proj_name=ckpt_gate_proj_name,
            ckpt_down_proj_name=ckpt_down_proj_name,
            ckpt_up_proj_name=ckpt_up_proj_name,
            num_experts=num_experts,
            num_redundant_experts=num_redundant_experts,
        )

    def _local_expert_id(self, expert_id: int | None) -> int | None:
        if expert_id is None:
            return None
        local_id = expert_id - self.local_expert_start
        if local_id < 0 or local_id >= self.local_num_experts:
            return None
        return local_id

    @staticmethod
    def _copy_weight(target: Any, loaded_weight: Any) -> None:
        if loaded_weight.shape != target.shape and len(loaded_weight.shape) == 2:
            loaded_weight = loaded_weight.t()
        target.copy_(loaded_weight)

    @staticmethod
    def _w13_shard_start(shard_id: Any, half_intermediate: int) -> int:
        if shard_id in (0, "w1", "gate", "gate_proj"):
            return 0
        if shard_id in (1, "w3", "up", "up_proj"):
            return half_intermediate
        raise ValueError(f"Unsupported Hunyuan MoE w13 shard_id: {shard_id!r}.")

    def _load_w13_weight(
        self,
        param: Any,
        loaded_weight: Any,
        weight_name: str | None = None,
        *,
        shard_id: Any = None,
        expert_id: int | None = None,
        return_success: bool = False,
    ) -> bool | None:
        local_id = self._local_expert_id(expert_id)
        if local_id is None:
            if return_success:
                return False
            return None

        if loaded_weight.shape[-1] == param.shape[2]:
            self._copy_weight(param.data[local_id], loaded_weight)
        else:
            half_intermediate = param.shape[2] // 2
            start = self._w13_shard_start(shard_id, half_intermediate)
            self._copy_weight(param.data[local_id, :, start : start + half_intermediate], loaded_weight)
        if return_success:
            return True
        return None

    def _load_w2_weight(
        self,
        param: Any,
        loaded_weight: Any,
        weight_name: str | None = None,
        *,
        shard_id: Any = None,
        expert_id: int | None = None,
        return_success: bool = False,
    ) -> bool | None:
        local_id = self._local_expert_id(expert_id)
        if local_id is None:
            if return_success:
                return False
            return None
        self._copy_weight(param.data[local_id], loaded_weight)
        if return_success:
            return True
        return None

    @staticmethod
    def _device_group(group: Any) -> Any:
        return getattr(group, "device_group", group)

    @staticmethod
    def _get_sp_size() -> int:
        try:
            return get_sequence_parallel_world_size()
        except AssertionError:
            return 1

    def _load_mindiesd_fused_moe(self) -> Any:
        try:
            from mindiesd.layers.fused_moe import fused_moe
        except ImportError as exc:
            raise ImportError(
                "MindIE-SD Hunyuan MoE backend requires the 'mindiesd' package. "
                "Install MindIE-SD or unset VLLM_OMNI_HUNYUAN_MOE_BACKEND."
            ) from exc
        return fused_moe

    def _get_shared_expert_stream(self) -> Any:
        if self._shared_expert_stream is None:
            self._shared_expert_stream = torch.npu.Stream()
        return self._shared_expert_stream

    def _forward_shared_experts_async(self, hidden_states: Any) -> Any:
        if self.shared_experts is None:
            return None

        current_stream = torch.npu.current_stream()
        shared_stream = self._get_shared_expert_stream()
        shared_stream.wait_stream(current_stream)
        with torch.npu.stream(shared_stream):
            shared_output = self.shared_experts(hidden_states)
            return tensor_model_parallel_all_reduce(shared_output)

    def forward(self, hidden_states: Any, router_logits: Any) -> Any:
        _set_hunyuan_fused_moe_forward_context(hidden_states.shape[0])
        fused_moe = self._load_mindiesd_fused_moe()
        tp_group = self._device_group(get_tp_group())
        ep_group_obj = get_ep_group()
        ep_group = self._device_group(ep_group_obj) if getattr(ep_group_obj, "world_size", 1) > 1 else None
        sp_enabled = self._get_sp_size() > 1
        shared_output = self._forward_shared_experts_async(hidden_states)
        output = fused_moe(
            hidden_states=hidden_states,
            router_logits=router_logits,
            num_experts=self.num_experts,
            top_k=self.top_k,
            w13_weight=self.w13_weight,
            w2_weight=self.w2_weight,
            tp_group=tp_group,
            ep_group=ep_group,
            tokens_full=not sp_enabled,
            renormalize=self.renormalize,
            reduce_results=True,
            dispatcher_type=None,
        )
        if shared_output is not None:
            torch.npu.current_stream().wait_stream(self._get_shared_expert_stream())
            output = output + shared_output
        return output
