"""Physical expert state, migration buffers, and asynchronous lifecycle records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from placemoe.model_adapter import MoEModelAdapter

from ....utils.accelerator_timing import AcceleratorEvent
from .core_planner import (
    QuotaPolicyEntry,
)


def _initial_slot_to_logical(
    num_experts: int,
    base_num_local_experts: int,
    slot_capacity_per_rank: int,
    ep_size: int,
) -> torch.Tensor:
    layout = torch.full((int(ep_size) * int(slot_capacity_per_rank),), -1, dtype=torch.long)
    for logical_expert in range(int(num_experts)):
        rank, local_slot = divmod(logical_expert, int(base_num_local_experts))
        layout[int(rank) * int(slot_capacity_per_rank) + int(local_slot)] = int(logical_expert)
    return layout


def _canonical_physical_slots(
    num_experts: int,
    base_num_local_experts: int,
    slot_capacity_per_rank: int,
) -> torch.Tensor:
    canonical = torch.empty((int(num_experts),), dtype=torch.long)
    for logical_expert in range(int(num_experts)):
        rank, local_slot = divmod(logical_expert, int(base_num_local_experts))
        canonical[logical_expert] = int(rank) * int(slot_capacity_per_rank) + int(local_slot)
    return canonical


@dataclass
class _CostModelTiming:
    step: int
    physical_routes: torch.Tensor
    local_expert_token_counts: torch.Tensor
    local_assignment_count: torch.Tensor
    communication_events: dict[str, tuple[AcceleratorEvent, AcceleratorEvent]] | None
    dispatch_start: AcceleratorEvent
    dispatch_end: AcceleratorEvent
    compute_start: AcceleratorEvent
    compute_end: AcceleratorEvent
    combine_start: AcceleratorEvent
    combine_end: AcceleratorEvent


@dataclass
class ExpertLayerState:
    key: str
    module_id: int
    num_experts: int
    base_num_local_experts: int
    num_local_experts: int
    expert_parameter_names: tuple[str, ...]
    expert_parameters: tuple[torch.nn.Parameter, ...]
    model_adapter: MoEModelAdapter
    logical_to_physical: torch.Tensor
    slot_to_logical: torch.Tensor | None = None
    canonical_physical_slots: torch.Tensor | None = None
    latest_selected_experts: torch.Tensor | None = None
    latest_physical_routes: torch.Tensor | None = None
    latest_route_step: int = -1
    accumulated_tokens_per_local_expert: torch.Tensor | None = None
    latest_hidden_size: int = 0
    latest_bytes_per_element: int = 0
    is_identity: bool = True
    _device_mapping_cache: dict[torch.device, torch.Tensor] = field(default_factory=dict)
    source_logical_to_physical: torch.Tensor | None = None
    _device_source_mapping_cache: dict[tuple[torch.device, int], torch.Tensor] = field(default_factory=dict)
    _device_slot_layout_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    _device_redundant_groups_cache: dict[torch.device, tuple[tuple[int, torch.Tensor], ...]] = field(
        default_factory=dict
    )
    _redundant_copy_groups_cache: tuple[tuple[int, tuple[int, ...]], ...] | None = None
    _replica_grad_schedule_cache: _ReplicaGradSchedule | None = None
    placement_version: int = 0
    cost_model_timings: list[_CostModelTiming] = field(default_factory=list)
    pending_physical_routes: torch.Tensor | None = None
    pending_route_data_ptr: int = 0
    active_quota_policy: tuple[QuotaPolicyEntry, ...] = ()
    fixed_r2_layout: bool = False

    @property
    def primary_parameter(self) -> torch.nn.Parameter:
        return self.expert_parameters[0]

    def named_expert_parameters(self) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        return tuple(zip(self.expert_parameter_names, self.expert_parameters, strict=True))

    def invalidate_cache(self) -> None:
        self._device_mapping_cache.clear()
        self._device_source_mapping_cache.clear()
        self._device_slot_layout_cache.clear()
        self._device_redundant_groups_cache.clear()
        self._redundant_copy_groups_cache = None
        self._replica_grad_schedule_cache = None

    def refresh_identity(self) -> None:
        if self.slot_to_logical is not None:
            expected = _initial_slot_to_logical(
                self.num_experts,
                self.base_num_local_experts,
                self.num_local_experts,
                self.num_experts // self.base_num_local_experts,
            )
            self.is_identity = torch.equal(self.slot_to_logical.cpu(), expected)
        else:
            identity = torch.arange(self.num_experts, dtype=torch.long)
            self.is_identity = torch.equal(self.logical_to_physical.cpu(), identity)

    def mapping_for_device(self, device: torch.device) -> torch.Tensor:
        cached = self._device_mapping_cache.get(device)
        if cached is None:
            cached = self.logical_to_physical.to(device=device, non_blocking=True)
            self._device_mapping_cache[device] = cached
        return cached

    def source_mapping_for_device(self, device: torch.device, source_rank: int) -> torch.Tensor:
        if self.source_logical_to_physical is None:
            raise RuntimeError(f"HierMoE layer {self.key} has no source-rank route LUT.")
        key = (device, int(source_rank))
        cached = self._device_source_mapping_cache.get(key)
        if cached is None:
            cached = self.source_logical_to_physical[int(source_rank)].to(device=device, non_blocking=True)
            self._device_source_mapping_cache[key] = cached
        return cached

    @property
    def slot_layout_enabled(self) -> bool:
        return self.slot_to_logical is not None

    @property
    def num_physical_slots(self) -> int:
        return self.num_local_experts * (self.num_experts // self.base_num_local_experts)

    def copy_slots_for_device(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._device_slot_layout_cache.get(device)
        if cached is not None:
            return cached
        if self.slot_to_logical is None:
            mapping = self.mapping_for_device(device)
            copy_slots = mapping.view(self.num_experts, 1)
            copy_mask = torch.ones_like(copy_slots, dtype=torch.bool)
            cached = (copy_slots, copy_mask)
            self._device_slot_layout_cache[device] = cached
            return cached

        slot_to_logical_cpu = self.slot_to_logical.detach().cpu()
        counts = torch.bincount(slot_to_logical_cpu[slot_to_logical_cpu >= 0], minlength=self.num_experts)
        max_copies = max(1, int(counts.max().item()))
        copy_slots_cpu = torch.full((self.num_experts, max_copies), -1, dtype=torch.long)
        copy_mask_cpu = torch.zeros((self.num_experts, max_copies), dtype=torch.bool)
        offsets = torch.zeros((self.num_experts,), dtype=torch.long)
        for physical_slot, logical_expert in enumerate(slot_to_logical_cpu.tolist()):
            if logical_expert < 0:
                continue
            offset = int(offsets[logical_expert].item())
            copy_slots_cpu[logical_expert, offset] = int(physical_slot)
            copy_mask_cpu[logical_expert, offset] = True
            offsets[logical_expert] += 1
        cached = (
            copy_slots_cpu.to(device=device, non_blocking=True),
            copy_mask_cpu.to(device=device, non_blocking=True),
        )
        self._device_slot_layout_cache[device] = cached
        return cached

    def redundant_copy_groups(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        cached = self._redundant_copy_groups_cache
        if cached is not None:
            return cached
        if self.slot_to_logical is None:
            self._redundant_copy_groups_cache = ()
            return ()

        layout = self.slot_to_logical.detach().cpu()
        active = layout[layout >= 0]
        if active.numel() == 0:
            self._redundant_copy_groups_cache = ()
            return ()

        counts = torch.bincount(active, minlength=self.num_experts)
        groups: list[tuple[int, tuple[int, ...]]] = []
        for logical_expert in torch.nonzero(counts > 1, as_tuple=False).flatten().tolist():
            slots = torch.nonzero(layout == int(logical_expert), as_tuple=False).flatten().tolist()
            groups.append((int(logical_expert), tuple(int(slot) for slot in slots)))
        self._redundant_copy_groups_cache = tuple(groups)
        return self._redundant_copy_groups_cache

    def redundant_copy_groups_for_device(self, device: torch.device) -> tuple[tuple[int, torch.Tensor], ...]:
        cached = self._device_redundant_groups_cache.get(device)
        if cached is not None:
            return cached
        cached = tuple(
            (
                int(logical_expert),
                torch.tensor(slots, dtype=torch.long, device=device),
            )
            for logical_expert, slots in self.redundant_copy_groups()
        )
        self._device_redundant_groups_cache[device] = cached
        return cached


@dataclass(frozen=True)
class _SwapTensorEntry:
    tensor: torch.Tensor
    lhs_slot: int
    rhs_slot: int


@dataclass(frozen=True)
class _CoverTensorEntry:
    tensor: torch.Tensor
    src_slot: int
    dst_slot: int


@dataclass
class _SwapStagingBuffer:
    send: torch.Tensor
    recv: torch.Tensor


@dataclass(frozen=True)
class _PipelineGradResult:
    layer_key: str
    raw_ms: float
    start_event: AcceleratorEvent | None = None
    completion_event: AcceleratorEvent | None = None


@dataclass(frozen=True)
class _OptimizerParamBinding:
    optimizer: Any
    group: dict[str, Any]


@dataclass(frozen=True)
class _ReplicaGradGroup:
    logical_expert: int
    owner_rank: int
    copy_ranks: tuple[int, ...]
    local_slots: tuple[int, ...]


@dataclass
class _ReplicaGradSchedule:
    groups: tuple[_ReplicaGradGroup, ...]
    pairwise: bool
    globally_ordered_pairs: bool


@dataclass(frozen=True)
class _ReplicaGradContribution:
    logical_expert: int
    param_index: int
    local_grad: torch.Tensor
    local_slots: tuple[int, ...]
    local_sum: torch.Tensor

    @property
    def numel(self) -> int:
        return int(self.local_sum.numel())


_SwapBucketItem = tuple[torch.Tensor, int, torch.Tensor, int, int]


_SlotStateItem = tuple[str, torch.Tensor]
