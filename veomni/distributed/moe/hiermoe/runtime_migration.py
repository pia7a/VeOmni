"""Atomic expert and optimizer-state transfers between physical slots."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Iterable

import torch
import torch.distributed as dist

from ....utils.device import get_torch_device, synchronize
from . import runtime_settings as settings
from .core_planner import CORE_MOE_ALGORITHM_VERSION, QuotaPolicyEntry
from .planner import PlacementPlan
from .runtime_settings import _placement_timing_range
from .runtime_tensors import (
    _chunk_swap_bucket,
    _cover_grouped_slot_entries_atomic,
    _ep_global_rank,
    _exchange_or_swap_grouped_slot_entries_collective,
    _local_tensor_view,
    _swap_chunk_nbytes,
    _unpack_swap_chunk,
)
from .runtime_types import (
    ExpertLayerState,
    _CoverTensorEntry,
    _LayerSwapPlan,
    _PendingLayerSwap,
    _PendingPipelinePlan,
    _PipelineMigrationResult,
    _SwapBucketItem,
    _SwapStagingBuffer,
    _SwapTensorEntry,
)


class MigrationMixin:
    """Atomic expert and optimizer-state transfers between physical slots."""

    def _launch_next_pipeline_migration(self) -> None:
        executor = self._pipeline_migration_executor
        if executor is None or self._pipeline_shutdown:
            return
        with self._pipeline_lock:
            if self._pipeline_migration_futures:
                return
            while self._pipeline_next_migration_index < len(self._pipeline_layer_order):
                layer_key = self._pipeline_layer_order[self._pipeline_next_migration_index]
                self._pipeline_next_migration_index += 1
                pending = self._pipeline_pending_plans.get(layer_key)
                if pending is None:
                    continue
                layer = self.layers[layer_key]
                if int(layer.placement_version) != pending.placement_version:
                    self._pipeline_pending_plans.pop(layer_key, None)
                    self._accumulate_metric("hiermoe/pipeline_migration_stale", 1)
                    continue
                device = self._pipeline_device(layer)
                ready_event = self._pipeline_ready_event(device)
                future = executor.submit(
                    self._pipeline_migration_worker,
                    layer_key,
                    pending,
                    ready_event,
                )
                self._pipeline_migration_futures[layer_key] = future
                return

    @torch.no_grad()
    def _pipeline_migration_worker(
        self,
        layer_key: str,
        pending: _PendingPipelinePlan,
        ready_event: Any | None,
    ) -> _PipelineMigrationResult:
        layer = self.layers[layer_key]
        device = self._pipeline_device(layer)
        started = time.perf_counter()

        def run() -> tuple[str, ...]:
            committed = self._execute_placement_plan(
                layer,
                pending.plan,
                timing_prefix="hiermoe_pipeline_migration",
                transfer_group=self._pipeline_migration_group,
                force_staged_transfer=False,
                fast_sparse_transfer=True,
            )
            return tuple(committed)

        committed = self._run_pipeline_stream_task("migration", device, ready_event, run)
        return _PipelineMigrationResult(
            layer_key=layer_key,
            source_step=pending.source_step,
            committed=committed,
            raw_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _finish_pipeline_migration(self, layer_key: str, *, wait: bool) -> bool:
        with self._pipeline_lock:
            future = self._pipeline_migration_futures.get(layer_key)
        if future is None or (not wait and not future.done()):
            return False
        wait_started = time.perf_counter()
        result = future.result()
        exposed_ms = (time.perf_counter() - wait_started) * 1000.0 if wait else 0.0
        with self._pipeline_lock:
            self._pipeline_migration_futures.pop(layer_key, None)
            self._pipeline_pending_plans.pop(layer_key, None)
        self._accumulate_metric("hiermoe/pipeline_migration_jobs", 1)
        self._accumulate_metric("hiermoe/pipeline_migration_raw_ms", result.raw_ms)
        self._accumulate_metric("hiermoe/pipeline_migration_exposed_ms", exposed_ms)
        raw_total = float(self._placement_metrics.get("hiermoe/pipeline_migration_raw_ms", 0.0))
        exposed_total = float(self._placement_metrics.get("hiermoe/pipeline_migration_exposed_ms", 0.0))
        if raw_total > 0.0:
            self._placement_metrics["hiermoe/pipeline_migration_hidden_ratio"] = max(
                0.0,
                min(1.0, 1.0 - exposed_total / raw_total),
            )
        if exposed_ms > 0.01:
            self._accumulate_metric("hiermoe/pipeline_migration_deadline_miss", 1)
        return True

    def advance_pipeline_after_combine(self, _layer_key: str) -> None:
        if not self.fixed_pipeline_overlap or self._ablation_migration_mode != "hidden":
            return
        # Candidate scoring is a compute-only stage whose deadline is the next
        # step, not the end of the current layer. Let the single planner queue
        # carry it diagonally across later layers and collect completed plans at
        # the optimizer boundary.
        with self._pipeline_lock:
            active = tuple(self._pipeline_migration_futures)
        if active and self._finish_pipeline_migration(active[0], wait=False):
            self._launch_next_pipeline_migration()
        elif not active:
            self._launch_next_pipeline_migration()

    def wait_pipeline_migration_before_layer(self, layer_key: str) -> None:
        if not self.fixed_pipeline_overlap or layer_key not in self._pipeline_pending_plans:
            return
        if self._ablation_migration_mode == "blocking":
            pending = self._pipeline_pending_plans[layer_key]
            result = self._pipeline_migration_worker(layer_key, pending, None)
            with self._pipeline_lock:
                self._pipeline_pending_plans.pop(layer_key, None)
            self._accumulate_metric("hiermoe/pipeline_migration_jobs", 1)
            self._accumulate_metric("hiermoe/pipeline_migration_raw_ms", result.raw_ms)
            self._accumulate_metric("hiermoe/pipeline_migration_exposed_ms", result.raw_ms)
            self._placement_metrics["hiermoe/pipeline_migration_hidden_ratio"] = 0.0
            if result.raw_ms > 0.01:
                self._accumulate_metric("hiermoe/pipeline_migration_deadline_miss", 1)
            return
        while layer_key in self._pipeline_pending_plans:
            with self._pipeline_lock:
                active = tuple(self._pipeline_migration_futures)
            if not active:
                self._launch_next_pipeline_migration()
                continue
            self._finish_pipeline_migration(active[0], wait=True)

    @torch.no_grad()
    def _execute_placement_plan(
        self,
        layer: ExpertLayerState,
        plan: PlacementPlan,
        *,
        timing_prefix: str | None = None,
        transfer_group: dist.ProcessGroup | None = None,
        force_staged_transfer: bool = False,
        fast_sparse_transfer: bool = False,
    ) -> list[str]:
        quota_policy = tuple(QuotaPolicyEntry.from_tuple(row) for row in plan.quota_policy)
        final_owner_slots = tuple(int(value) for value in plan.final_owner_slots)
        algorithm_version = getattr(plan, "algorithm_version", None)
        if algorithm_version == CORE_MOE_ALGORITHM_VERSION and len(final_owner_slots) != layer.num_experts:
            raise RuntimeError(
                f"CoRe-MoE placement plan for {layer.key} must provide exactly {layer.num_experts} owner slots."
            )
        if final_owner_slots and len(final_owner_slots) != layer.num_experts:
            raise RuntimeError(
                f"HierMoE placement plan for {layer.key} has {len(final_owner_slots)} owners, "
                f"expected {layer.num_experts}."
            )
        current_layout = self._layer_layout(layer)
        if len(plan.final_layout) != int(current_layout.numel()):
            raise RuntimeError(
                f"HierMoE placement plan for {layer.key} has {len(plan.final_layout)} physical slots, "
                f"expected {current_layout.numel()}."
            )
        if not plan.actions and tuple(int(value) for value in current_layout.tolist()) == plan.final_layout:
            with _placement_timing_range(timing_prefix, "apply"):
                if layer.slot_layout_enabled:
                    validated_layout, validated_owners = self._validate_placement_layout(
                        layer,
                        current_layout,
                        final_owner_slots or None,
                    )
                    quota_policy = self._validate_quota_policy(layer, validated_layout, quota_policy)
                    if validated_owners is not None:
                        self._refresh_layer_mapping_from_slots(layer, validated_owners)
                elif quota_policy:
                    raise RuntimeError(f"HierMoE compact placement for {layer.key} cannot install replica quota rows.")
                layer.active_quota_policy = tuple(quota_policy)
            return []
        working = current_layout.clone()
        origin_by_slot = torch.arange(int(working.numel()), dtype=torch.long)
        committed: list[str] = []
        for action in plan.actions:
            src_slot = int(action.src_slot)
            dst_slot = int(action.dst_slot)
            if not 0 <= src_slot < int(working.numel()) or not 0 <= dst_slot < int(working.numel()):
                raise RuntimeError(
                    f"HierMoE placement action contains an out-of-range physical slot: {action.format()}."
                )
            if action.kind == "swap":
                if (
                    int(working[src_slot].item()) != action.src_logical
                    or int(working[dst_slot].item()) != action.dst_logical
                ):
                    raise RuntimeError(
                        f"HierMoE placement swap does not match the ordered working layout: {action.format()}."
                    )
                working[src_slot], working[dst_slot] = (
                    working[dst_slot].clone(),
                    working[src_slot].clone(),
                )
                origin_by_slot[src_slot], origin_by_slot[dst_slot] = (
                    origin_by_slot[dst_slot].clone(),
                    origin_by_slot[src_slot].clone(),
                )
            elif action.kind == "replica":
                if (
                    action.src_logical < 0
                    or int(working[src_slot].item()) != action.src_logical
                    or int(working[dst_slot].item()) != action.dst_logical
                ):
                    raise RuntimeError(
                        f"HierMoE placement cover does not match the ordered working layout: {action.format()}."
                    )
                working[dst_slot] = action.src_logical
                origin_by_slot[dst_slot] = origin_by_slot[src_slot]
            elif action.kind == "empty":
                if (
                    src_slot != dst_slot
                    or int(working[dst_slot].item()) != action.src_logical
                    or action.dst_logical != -1
                ):
                    raise RuntimeError(
                        f"HierMoE placement empty action does not match the ordered working layout: {action.format()}."
                    )
                working[dst_slot] = -1
                origin_by_slot[dst_slot] = -1
            else:
                raise RuntimeError(f"HierMoE placement plan contains an unknown action kind: {action.kind!r}.")
            committed.append(f"{layer.key}:{action.format()}")

        final_layout = torch.tensor(plan.final_layout, dtype=torch.long)
        if tuple(int(value) for value in working.tolist()) != plan.final_layout:
            raise RuntimeError(f"HierMoE executor diverged from the planner for layer {layer.key}.")

        directed_transfers: list[tuple[int, int]] = []
        zero_slots: list[int] = []
        for dst_slot in range(int(final_layout.numel())):
            desired = int(final_layout[dst_slot].item())
            initial = int(current_layout[dst_slot].item())
            if desired == initial:
                continue
            if desired < 0:
                zero_slots.append(dst_slot)
                continue
            source_origin = int(origin_by_slot[dst_slot].item())
            if (
                source_origin < 0
                or source_origin >= int(current_layout.numel())
                or int(current_layout[source_origin].item()) != desired
            ):
                raise RuntimeError(
                    f"HierMoE placement plan has no original state source for logical expert {desired} "
                    f"at physical slot {dst_slot}."
                )
            directed_transfers.append((source_origin, dst_slot))
        if final_owner_slots:
            for logical_expert, physical_slot in enumerate(final_owner_slots):
                if not 0 <= physical_slot < int(working.numel()):
                    raise RuntimeError(
                        f"HierMoE placement owner slot {physical_slot} for logical expert {logical_expert} "
                        "is out of range."
                    )
                if int(working[physical_slot].item()) != logical_expert:
                    raise RuntimeError(
                        f"HierMoE placement owner slot {physical_slot} does not contain logical expert "
                        f"{logical_expert}."
                    )
        if layer.slot_layout_enabled:
            working, _ = self._validate_placement_layout(layer, working, final_owner_slots or None)
        elif quota_policy:
            raise RuntimeError(f"HierMoE compact placement for {layer.key} cannot install replica quota rows.")
        quota_policy = self._validate_quota_policy(layer, working, quota_policy)

        grouped_entries: dict[tuple[int, int], list[_CoverTensorEntry]] = defaultdict(list)
        written_slots: set[int] = set()
        for _src_slot, dst_slot in directed_transfers:
            if dst_slot in written_slots:
                raise RuntimeError(f"HierMoE placement plan writes physical slot {dst_slot} more than once.")
            written_slots.add(dst_slot)
        overlapping_zero_slots = written_slots.intersection(zero_slots)
        if overlapping_zero_slots:
            slot = min(overlapping_zero_slots)
            raise RuntimeError(f"HierMoE placement plan both copies into and clears physical slot {slot}.")

        originally_missing_grads = tuple(
            param for param in layer.expert_parameters if getattr(param, "grad", None) is None
        )
        swap_actions = tuple(action for action in plan.actions if action.kind == "swap")
        for action in swap_actions:
            lhs_rank = int(action.src_slot) // layer.num_local_experts
            rhs_rank = int(action.dst_slot) // layer.num_local_experts
            if lhs_rank == rhs_rank:
                raise RuntimeError(
                    f"HierMoE planner produced a same-rank swap for layer {layer.key}: "
                    f"rank={lhs_rank}, experts=({action.src_logical}, {action.dst_logical})."
                )
        swap_slots = tuple(slot for action in swap_actions for slot in (int(action.src_slot), int(action.dst_slot)))
        use_pure_swap_transport = (
            bool(swap_actions)
            and not force_staged_transfer
            and len(swap_actions) == len(plan.actions)
            and len(set(swap_slots)) == len(swap_slots)
            and not zero_slots
            and not quota_policy
            and self.ep_group is not None
        )

        if use_pure_swap_transport:
            try:
                state_rows = self._slot_op_state_rows(layer)
                state_tensors = [tensor for _param, items in state_rows for _descriptor, tensor in items]
                if self.debug_validate and self.ep_size > 1:
                    self._validate_optimizer_state_slot_tensors_across_ep(state_rows)
                swap_plans: list[_LayerSwapPlan] = []
                for action in swap_actions:
                    lhs_rank, lhs_slot = divmod(int(action.src_slot), layer.num_local_experts)
                    rhs_rank, rhs_slot = divmod(int(action.dst_slot), layer.num_local_experts)
                    swap_plans.append(
                        _LayerSwapPlan(
                            layer_key=layer.key,
                            logical_lhs=int(action.src_logical),
                            logical_rhs=int(action.dst_logical),
                            lhs_rank=int(lhs_rank),
                            rhs_rank=int(rhs_rank),
                            entries=tuple(
                                _SwapTensorEntry(tensor, lhs_slot=lhs_slot, rhs_slot=rhs_slot)
                                for tensor in state_tensors
                            ),
                        )
                    )
                if fast_sparse_transfer:
                    with _placement_timing_range(timing_prefix, "transfer"):
                        self._execute_swap_plan_batch(swap_plans)
                elif self.expert_swap_mode == "layer":
                    with _placement_timing_range(timing_prefix, "transfer"):
                        self._execute_sparse_group_swap_plans(swap_plans)
                else:
                    self.launch_pending_layer_swap(layer.key, swap_plans, timing_prefix=timing_prefix)
            except Exception:
                for param in originally_missing_grads:
                    param.grad = None
                raise
        else:
            with _placement_timing_range(timing_prefix, "transfer"):
                try:
                    state_tensors = self._slot_op_state_tensors(layer) if directed_transfers or zero_slots else []
                    for src_slot, dst_slot in directed_transfers:
                        src_rank = src_slot // layer.num_local_experts
                        dst_rank = dst_slot // layer.num_local_experts
                        grouped_entries[(src_rank, dst_rank)].extend(
                            self._slot_op_cover_entries_from_tensors(
                                state_tensors,
                                num_local_experts=layer.num_local_experts,
                                src_slot=src_slot,
                                dst_slot=dst_slot,
                            )
                        )
                    zero_entries = {
                        dst_slot: self._slot_op_cover_entries_from_tensors(
                            state_tensors,
                            num_local_experts=layer.num_local_experts,
                            src_slot=dst_slot,
                            dst_slot=dst_slot,
                        )
                        for dst_slot in zero_slots
                    }
                    zero_entry_groups = tuple(
                        (dst_slot // layer.num_local_experts, zero_entries[dst_slot]) for dst_slot in zero_slots
                    )
                    if fast_sparse_transfer:
                        self._execute_sparse_group_slot_transfers(
                            grouped_entries,
                            zero_entry_groups=zero_entry_groups,
                            process_group=self.ep_group if transfer_group is None else transfer_group,
                        )
                    else:
                        _cover_grouped_slot_entries_atomic(
                            grouped_entries,
                            self.ep_rank,
                            self.ep_size,
                            self.ep_group if transfer_group is None else transfer_group,
                            zero_entry_groups=zero_entry_groups,
                            debug_validate=self.debug_validate,
                        )
                except Exception:
                    for param in originally_missing_grads:
                        param.grad = None
                    raise

        with _placement_timing_range(timing_prefix, "apply"):
            if layer.slot_layout_enabled:
                layer.slot_to_logical = working
                layer.fixed_r2_layout = False
                self._refresh_layer_mapping_from_slots(layer, final_owner_slots or None)
            else:
                mapping = torch.empty((layer.num_experts,), dtype=torch.long)
                for physical_slot, logical in enumerate(working.tolist()):
                    mapping[int(logical)] = int(physical_slot)
                layer.logical_to_physical = mapping
                layer.refresh_identity()
                layer.invalidate_cache()
            layer.active_quota_policy = quota_policy
            layer.placement_version += int(bool(plan.actions))
        return committed

    @staticmethod
    def _layer_layout(layer: ExpertLayerState) -> torch.Tensor:
        if layer.slot_to_logical is not None:
            return layer.slot_to_logical.detach().cpu().clone()
        layout = torch.full((layer.num_experts,), -1, dtype=torch.long)
        logical = torch.arange(layer.num_experts, dtype=torch.long)
        layout.scatter_(0, layer.logical_to_physical.to(torch.long), logical)
        return layout

    def _ensure_swap_staging_buffer(
        self,
        device: torch.device,
        dtype: torch.dtype,
        required_numel: int,
    ) -> _SwapStagingBuffer:
        key = (device, dtype)
        cached = self._swap_staging_buffers.get(key)
        if cached is None or int(cached.send.numel()) < int(required_numel):
            cached = _SwapStagingBuffer(
                send=torch.empty((required_numel,), dtype=dtype, device=device),
                recv=torch.empty((required_numel,), dtype=dtype, device=device),
            )
            self._swap_staging_buffers[key] = cached
        return cached

    @torch.no_grad()
    def _execute_sparse_group_slot_transfers(
        self,
        grouped_entries: dict[tuple[int, int], list[_CoverTensorEntry]],
        *,
        zero_entry_groups: Iterable[tuple[int, Iterable[_CoverTensorEntry]]] = (),
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        """Execute a deterministic directed placement with batched sparse P2P.

        Every EP rank has the same exact placement plan, so peers and payload
        sizes are known without a split-size collective. Only source and
        destination ranks participate; all destination slots are published
        after the complete P2P batch succeeds.
        """

        zero_groups = tuple((int(rank), tuple(entries)) for rank, entries in zero_entry_groups)
        all_entries = tuple(entry for entries in grouped_entries.values() for entry in entries) + tuple(
            entry for _rank, entries in zero_groups for entry in entries
        )
        if not all_entries:
            return
        group = self.ep_group if process_group is None else process_group
        if self.ep_size > 1 and group is None and any(src_rank != dst_rank for src_rank, dst_rank in grouped_entries):
            raise RuntimeError("HierMoE sparse placement migration requires an EP process group.")

        bucket_keys: set[tuple[torch.device, torch.dtype]] = set()
        send_buckets: dict[tuple[torch.device, torch.dtype], dict[int, list[tuple[torch.Tensor, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        recv_buckets: dict[tuple[torch.device, torch.dtype], dict[int, list[tuple[torch.Tensor, int, int]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        local_commits: list[tuple[torch.Tensor, int, torch.Tensor | None]] = []

        for (src_rank, dst_rank), entries in sorted(grouped_entries.items()):
            for entry in entries:
                local_tensor = _local_tensor_view(entry.tensor)
                src_view = local_tensor.detach()[entry.src_slot]
                dst_view = local_tensor.detach()[entry.dst_slot]
                if tuple(src_view.shape) != tuple(dst_view.shape):
                    raise RuntimeError("HierMoE sparse placement copied incompatible expert slot shapes.")
                key = (src_view.device, src_view.dtype)
                bucket_keys.add(key)
                if src_rank == dst_rank:
                    if self.ep_rank == src_rank:
                        local_commits.append((local_tensor, int(entry.dst_slot), src_view.clone()))
                elif self.ep_rank == src_rank:
                    flat = src_view.contiguous().view(-1)
                    send_buckets[key][int(dst_rank)].append((flat, int(flat.numel())))
                elif self.ep_rank == dst_rank:
                    recv_buckets[key][int(src_rank)].append((local_tensor, int(entry.dst_slot), int(dst_view.numel())))

        for dst_rank, entries in zero_groups:
            for entry in entries:
                local_tensor = _local_tensor_view(entry.tensor)
                dst_view = local_tensor.detach()[entry.dst_slot]
                bucket_keys.add((dst_view.device, dst_view.dtype))
                if self.ep_rank == dst_rank:
                    local_commits.append((local_tensor, int(entry.dst_slot), None))

        pending_remote: list[tuple[torch.Tensor, dict[int, list[tuple[torch.Tensor, int, int]]], list[int]]] = []
        for device, dtype in sorted(
            bucket_keys,
            key=lambda item: (
                item[0].type,
                -1 if item[0].index is None else int(item[0].index),
                str(item[1]),
            ),
        ):
            key = (device, dtype)
            peer_sends = send_buckets.get(key, {})
            peer_recvs = recv_buckets.get(key, {})
            input_splits = [
                sum(numel for _view, numel in peer_sends.get(peer_rank, ())) for peer_rank in range(self.ep_size)
            ]
            output_splits = [
                sum(numel for _tensor, _slot, numel in peer_recvs.get(peer_rank, ()))
                for peer_rank in range(self.ep_size)
            ]
            send_numel = sum(input_splits)
            recv_numel = sum(output_splits)
            staging = self._ensure_swap_staging_buffer(device, dtype, max(send_numel, recv_numel))
            send_buffer = staging.send[:send_numel]
            recv_buffer = staging.recv[:recv_numel]

            offset = 0
            send_offsets = [0] * self.ep_size
            for peer_rank in range(self.ep_size):
                send_offsets[peer_rank] = offset
                for view, numel in peer_sends.get(peer_rank, ()):
                    send_buffer[offset : offset + numel].view_as(view).copy_(view)
                    offset += numel
            recv_offsets = [0] * self.ep_size
            offset = 0
            for peer_rank, split_size in enumerate(output_splits):
                recv_offsets[peer_rank] = offset
                offset += int(split_size)

            ops: list[dist.P2POp] = []
            if self.ep_size > 1:
                assert group is not None
                for peer_rank in range(self.ep_size):
                    peer_global_rank = _ep_global_rank(group, peer_rank)
                    input_size = int(input_splits[peer_rank])
                    if input_size:
                        start = send_offsets[peer_rank]
                        ops.append(
                            dist.P2POp(
                                dist.isend,
                                send_buffer[start : start + input_size],
                                peer_global_rank,
                                group,
                            )
                        )
                    output_size = int(output_splits[peer_rank])
                    if output_size:
                        start = recv_offsets[peer_rank]
                        ops.append(
                            dist.P2POp(
                                dist.irecv,
                                recv_buffer[start : start + output_size],
                                peer_global_rank,
                                group,
                            )
                        )
                works = dist.batch_isend_irecv(ops) if ops else ()
                for work in works:
                    work.wait()
                if works:
                    # HCCL ``Work.wait`` only guarantees host-side
                    # submission on Ascend.  The receive buffers are consumed
                    # below on the default stream, so establish the missing
                    # communication-stream -> default-stream dependency before
                    # reading them.  Without this fence a following slot copy
                    # can be queued against an in-flight recv and the first
                    # later collective stalls behind ``aclnnInplaceCopy``.
                    synchronize()
            elif send_numel:
                recv_buffer.copy_(send_buffer)
            pending_remote.append((recv_buffer, peer_recvs, output_splits))

        destinations: set[tuple[int, int]] = set()
        for local_tensor, dst_slot, staged in local_commits:
            destination = (id(local_tensor), int(dst_slot))
            if destination in destinations:
                raise RuntimeError("HierMoE sparse placement writes one tensor slot more than once.")
            destinations.add(destination)
            if staged is None:
                local_tensor.detach()[dst_slot].zero_()
            else:
                local_tensor.detach()[dst_slot].copy_(staged)
        for recv_buffer, peer_recvs, output_splits in pending_remote:
            offset = 0
            for peer_rank, split_size in enumerate(output_splits):
                inner_offset = offset
                for local_tensor, dst_slot, numel in peer_recvs.get(peer_rank, ()):
                    destination = (id(local_tensor), int(dst_slot))
                    if destination in destinations:
                        raise RuntimeError("HierMoE sparse placement writes one tensor slot more than once.")
                    destinations.add(destination)
                    staged = recv_buffer[inner_offset : inner_offset + numel].view_as(local_tensor.detach()[dst_slot])
                    local_tensor.detach()[dst_slot].copy_(staged)
                    inner_offset += numel
                if inner_offset - offset != int(split_size):
                    raise RuntimeError("HierMoE sparse placement payload size does not match the exact plan.")
                offset += int(split_size)

    @torch.no_grad()
    def _execute_sparse_group_swap_plans(self, plans: Iterable[_LayerSwapPlan]) -> None:
        """Synchronously exchange pure swaps with one full-group All-to-All per dtype.

        The training lifecycle creates gradients and optimizer state consistently on every
        EP rank. Debug validation checks that descriptor invariant without adding a production
        split-size collective to this path.
        """

        plan_list = tuple(plans)
        if not plan_list:
            return
        if self._pending_layer_swaps:
            pending = next(iter(self._pending_layer_swaps))
            raise RuntimeError(f"HierMoE tried to execute a collective swap while layer {pending} is still pending.")
        if self.ep_group is None or self.ep_size <= 1:
            raise RuntimeError("HierMoE cross-rank expert swap requires an EP process group.")

        bucket_keys: set[tuple[torch.device, torch.dtype]] = set()
        peer_buckets: dict[tuple[torch.device, torch.dtype], dict[int, list[_SwapBucketItem]]] = defaultdict(
            lambda: defaultdict(list)
        )
        occupied_slots: set[tuple[str, int, int]] = set()
        for plan in plan_list:
            if not 0 <= int(plan.lhs_rank) < self.ep_size or not 0 <= int(plan.rhs_rank) < self.ep_size:
                raise RuntimeError(
                    f"HierMoE planner produced an out-of-range swap rank for layer {plan.layer_key}: "
                    f"({plan.lhs_rank}, {plan.rhs_rank})."
                )
            if plan.lhs_rank == plan.rhs_rank:
                raise RuntimeError(
                    f"HierMoE planner produced a same-rank swap for layer {plan.layer_key}: "
                    f"rank={plan.lhs_rank}, experts=({plan.logical_lhs}, {plan.logical_rhs})."
                )
            if not plan.entries:
                raise RuntimeError(f"HierMoE pure swap plan for layer {plan.layer_key} has no tensor entries.")
            for rank, slot in ((plan.lhs_rank, plan.entries[0].lhs_slot), (plan.rhs_rank, plan.entries[0].rhs_slot)):
                occupied = (plan.layer_key, int(rank), int(slot))
                if occupied in occupied_slots:
                    raise RuntimeError(
                        f"HierMoE pure swap plan writes layer {plan.layer_key} rank {rank} slot {slot} more than once."
                    )
                occupied_slots.add(occupied)

            for entry in plan.entries:
                local_tensor = _local_tensor_view(entry.tensor)
                lhs_view = local_tensor.detach()[entry.lhs_slot]
                rhs_view = local_tensor.detach()[entry.rhs_slot]
                if tuple(lhs_view.shape) != tuple(rhs_view.shape):
                    raise RuntimeError("HierMoE pure swap tried to exchange incompatible expert slot shapes.")
                key = (lhs_view.device, lhs_view.dtype)
                bucket_keys.add(key)
                if self.ep_rank == plan.lhs_rank:
                    local_slot = int(entry.lhs_slot)
                    peer_rank = int(plan.rhs_rank)
                elif self.ep_rank == plan.rhs_rank:
                    local_slot = int(entry.rhs_slot)
                    peer_rank = int(plan.lhs_rank)
                else:
                    continue
                send_view = local_tensor.detach()[local_slot]
                numel = int(send_view.numel())
                peer_buckets[key][peer_rank].append(
                    (local_tensor, local_slot, send_view, numel, numel * int(send_view.element_size()))
                )

        def bucket_sort_key(key: tuple[torch.device, torch.dtype]) -> tuple[str, int, str]:
            device, dtype = key
            return (device.type, -1 if device.index is None else int(device.index), str(dtype))

        pending_publish: list[tuple[torch.Tensor, dict[int, list[_SwapBucketItem]]]] = []
        for device, dtype in sorted(bucket_keys, key=bucket_sort_key):
            buckets = peer_buckets.get((device, dtype), {})
            input_splits = [0] * self.ep_size
            output_splits = [0] * self.ep_size
            for peer_rank, items in buckets.items():
                payload_numel = sum(item[3] for item in items)
                input_splits[int(peer_rank)] = payload_numel
                output_splits[int(peer_rank)] = payload_numel

            send_numel = sum(input_splits)
            recv_numel = sum(output_splits)
            staging = self._ensure_swap_staging_buffer(device, dtype, max(send_numel, recv_numel))
            send_buffer = staging.send[:send_numel]
            recv_buffer = staging.recv[:recv_numel]
            offset = 0
            for peer_rank in range(self.ep_size):
                for _local_tensor, _local_slot, send_view, numel, _nbytes in buckets.get(peer_rank, ()):
                    send_buffer[offset : offset + numel].view_as(send_view).copy_(send_view)
                    offset += numel

            dist.all_to_all_single(
                recv_buffer,
                send_buffer,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=self.ep_group,
            )

            pending_publish.append((recv_buffer, buckets))

        for recv_buffer, buckets in pending_publish:
            offset = 0
            for peer_rank in range(self.ep_size):
                for local_tensor, local_slot, _send_view, numel, _nbytes in buckets.get(peer_rank, ()):
                    staged = recv_buffer[offset : offset + numel].view_as(local_tensor.detach()[local_slot])
                    local_tensor.detach()[local_slot].copy_(staged)
                    offset += numel

    def _compile_swap_waves(
        self,
        plans: Iterable[_LayerSwapPlan],
    ) -> list[list[tuple[tuple[torch.device, torch.dtype], int, list[_SwapBucketItem]]]]:
        remote_buckets: dict[tuple[torch.device, torch.dtype], dict[int, list[_SwapBucketItem]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for plan in plans:
            if plan.lhs_rank == plan.rhs_rank:
                raise RuntimeError(
                    f"HierMoE planner produced a same-rank swap for layer {plan.layer_key}: "
                    f"rank={plan.lhs_rank}, experts=({plan.logical_lhs}, {plan.logical_rhs})."
                )
            if self.ep_group is None or self.ep_size <= 1:
                raise RuntimeError("HierMoE cross-rank expert swap requires an EP process group.")
            if self.ep_rank not in (plan.lhs_rank, plan.rhs_rank):
                continue

            peer_rank = plan.rhs_rank if self.ep_rank == plan.lhs_rank else plan.lhs_rank
            peer_global_rank = _ep_global_rank(self.ep_group, peer_rank)
            for entry in plan.entries:
                local_slot = entry.lhs_slot if self.ep_rank == plan.lhs_rank else entry.rhs_slot
                local_tensor = _local_tensor_view(entry.tensor)
                slot_view = local_tensor.detach()[local_slot]
                send_view = slot_view.contiguous().view(-1)
                numel = int(send_view.numel())
                nbytes = numel * int(send_view.element_size())
                remote_buckets[(send_view.device, send_view.dtype)][peer_global_rank].append(
                    (local_tensor, int(local_slot), send_view, numel, nbytes)
                )

        items: list[tuple[tuple[torch.device, torch.dtype], int, list[_SwapBucketItem]]] = []
        for key in sorted(
            remote_buckets,
            key=lambda item: (
                item[0].type,
                -1 if item[0].index is None else int(item[0].index),
                str(item[1]),
            ),
        ):
            for peer_global_rank, bucket in sorted(remote_buckets[key].items()):
                items.extend((key, peer_global_rank, chunk) for chunk in _chunk_swap_bucket(bucket))

        waves: list[list[tuple[tuple[torch.device, torch.dtype], int, list[_SwapBucketItem]]]] = []
        current: list[tuple[tuple[torch.device, torch.dtype], int, list[_SwapBucketItem]]] = []
        current_nbytes = 0
        for item in items:
            item_nbytes = 2 * _swap_chunk_nbytes(item[2])
            if current and current_nbytes + item_nbytes > settings._MAX_SWAP_WAVE_BYTES:
                waves.append(current)
                current = []
                current_nbytes = 0
            current.append(item)
            current_nbytes += item_nbytes
        if current:
            waves.append(current)
        return waves

    def _stage_swap_wave(
        self,
        wave: list[tuple[tuple[torch.device, torch.dtype], int, list[_SwapBucketItem]]],
    ) -> tuple[tuple[Any, ...], list[tuple[torch.Tensor, list[_SwapBucketItem]]]]:
        required: dict[tuple[torch.device, torch.dtype], int] = defaultdict(int)
        for key, _peer_global_rank, chunk in wave:
            required[key] += sum(item[3] for item in chunk)
        buffers = {key: self._ensure_swap_staging_buffer(key[0], key[1], numel) for key, numel in required.items()}
        offsets: dict[tuple[torch.device, torch.dtype], int] = defaultdict(int)
        ops: list[dist.P2POp] = []
        unpack: list[tuple[torch.Tensor, list[_SwapBucketItem]]] = []
        for key, peer_global_rank, chunk in wave:
            numel = sum(item[3] for item in chunk)
            start = offsets[key]
            offsets[key] += numel
            send_segment = buffers[key].send[start : start + numel]
            recv_segment = buffers[key].recv[start : start + numel]
            inner_offset = 0
            for _local_tensor, _local_slot, send_view, item_numel, _nbytes in chunk:
                send_segment[inner_offset : inner_offset + item_numel].copy_(send_view)
                inner_offset += item_numel
            transfer_group = self._swap_group if self._swap_group is not None else self.ep_group
            ops.extend(
                (
                    dist.P2POp(dist.isend, send_segment, peer_global_rank, transfer_group),
                    dist.P2POp(dist.irecv, recv_segment, peer_global_rank, transfer_group),
                )
            )
            unpack.append((recv_segment, chunk))
        works = tuple(dist.batch_isend_irecv(ops)) if ops else ()
        return works, unpack

    @staticmethod
    def _publish_swap_wave(unpack: Iterable[tuple[torch.Tensor, list[_SwapBucketItem]]]) -> None:
        for recv_segment, chunk in unpack:
            _unpack_swap_chunk(recv_segment, chunk)

    def _swap_comm_stream(self, device: torch.device) -> Any:
        cached = self._swap_comm_streams.get(device)
        if cached is not None:
            return cached
        device_api = get_torch_device()
        try:
            cached = device_api.Stream(device=device)
        except TypeError:
            cached = device_api.Stream()
        self._swap_comm_streams[device] = cached
        return cached

    def _execute_swap_plan_batch(
        self,
        plans: Iterable[_LayerSwapPlan],
        *,
        pending_layer_key: str | None = None,
        timing_prefix: str | None = None,
    ) -> None:
        plan_list = tuple(plans)
        if not plan_list:
            return
        if self._pending_layer_swaps:
            pending = next(iter(self._pending_layer_swaps))
            raise RuntimeError(f"HierMoE tried to launch a swap while layer {pending} is still pending.")

        timing_context = _placement_timing_range(timing_prefix, "transfer")
        timing_context.__enter__()
        try:
            waves = self._compile_swap_waves(plan_list)
            if not waves:
                timing_context.__exit__(None, None, None)
                return

            devices = {key[0] for wave in waves for key, _peer, _chunk in wave}
            asynchronous = pending_layer_key is not None and len(waves) == 1 and len(devices) == 1
            device = next(iter(devices))
            if self._swap_group is None and self.ep_group is dist.group.WORLD:
                asynchronous = False
            asynchronous = asynchronous and device.type != "cpu"
            if not asynchronous:
                for wave in waves:
                    works, unpack = self._stage_swap_wave(wave)
                    for work in works:
                        work.wait()
                    self._publish_swap_wave(unpack)
                timing_context.__exit__(None, None, None)
                return

            device_api = get_torch_device()
            comm_stream = self._swap_comm_stream(device)
            try:
                current_stream = device_api.current_stream(device)
            except TypeError:
                current_stream = device_api.current_stream()
            comm_stream.wait_stream(current_stream)
            with device_api.stream(comm_stream):
                works, unpack = self._stage_swap_wave(waves[0])
            self._pending_layer_swaps[pending_layer_key] = _PendingLayerSwap(
                layer_key=pending_layer_key,
                works=works,
                unpack=tuple(unpack),
                device=device,
                timing_context=timing_context,
            )
        except Exception as error:
            timing_context.__exit__(type(error), error, error.__traceback__)
            raise

    def launch_pending_layer_swap(
        self,
        layer_key: str,
        plans: Iterable[_LayerSwapPlan],
        *,
        timing_prefix: str | None = None,
    ) -> None:
        self._execute_swap_plan_batch(
            plans,
            pending_layer_key=layer_key if self.expert_swap_mode == "layer" else None,
            timing_prefix=timing_prefix,
        )

    def wait_pending_layer_swap(self, layer_key: str) -> None:
        pending = self._pending_layer_swaps.pop(layer_key, None)
        if pending is None:
            return
        try:
            device_api = get_torch_device()
            comm_stream = self._swap_comm_stream(pending.device)
            with device_api.stream(comm_stream):
                for work in pending.works:
                    work.wait()
                self._publish_swap_wave(pending.unpack)
                done_event = device_api.Event()
                done_event.record(comm_stream)
            try:
                current_stream = device_api.current_stream(pending.device)
            except TypeError:
                current_stream = device_api.current_stream()
            current_stream.wait_event(done_event)
        except Exception as error:
            pending.timing_context.__exit__(type(error), error, error.__traceback__)
            raise
        pending.timing_context.__exit__(None, None, None)

    def _build_layer_swap_plan(
        self,
        layer_key: str,
        pair: tuple[int, int],
        *,
        validate_optimizer_state: bool = True,
    ) -> _LayerSwapPlan | None:
        layer = self.layers[layer_key]
        lhs, rhs = pair
        physical_lhs = int(layer.logical_to_physical[lhs].item())
        physical_rhs = int(layer.logical_to_physical[rhs].item())
        if physical_lhs == physical_rhs:
            return None

        lhs_rank, lhs_slot = divmod(physical_lhs, layer.num_local_experts)
        rhs_rank, rhs_slot = divmod(physical_rhs, layer.num_local_experts)
        if lhs_rank == rhs_rank:
            raise RuntimeError(
                f"HierMoE planner produced a same-rank swap for layer {layer_key}: "
                f"rank={lhs_rank}, experts=({lhs}, {rhs})."
            )
        state_rows = self._slot_op_state_rows(layer)
        if validate_optimizer_state and self.debug_validate and self.ep_size > 1:
            self._validate_optimizer_state_slot_tensors_across_ep(state_rows)
        entries = [
            _SwapTensorEntry(tensor, lhs_slot=lhs_slot, rhs_slot=rhs_slot)
            for _param, items in state_rows
            for _descriptor, tensor in items
        ]
        return _LayerSwapPlan(
            layer_key=layer_key,
            logical_lhs=int(lhs),
            logical_rhs=int(rhs),
            lhs_rank=int(lhs_rank),
            rhs_rank=int(rhs_rank),
            entries=tuple(entries),
        )

    @torch.no_grad()
    def _execute_swap_plans(self, plans: Iterable[_LayerSwapPlan], *, force_collective: bool = False) -> None:
        plan_list = tuple(plans)
        grouped: dict[tuple[int, int], list[_SwapTensorEntry]] = defaultdict(list)
        for plan in plan_list:
            if plan.lhs_rank == plan.rhs_rank:
                raise RuntimeError(
                    f"HierMoE planner produced a same-rank swap for layer {plan.layer_key}: "
                    f"rank={plan.lhs_rank}, experts=({plan.logical_lhs}, {plan.logical_rhs})."
                )
            lhs_rank = min(plan.lhs_rank, plan.rhs_rank)
            rhs_rank = max(plan.lhs_rank, plan.rhs_rank)
            if plan.lhs_rank == lhs_rank:
                grouped[(lhs_rank, rhs_rank)].extend(plan.entries)
            else:
                grouped[(lhs_rank, rhs_rank)].extend(
                    _SwapTensorEntry(entry.tensor, lhs_slot=entry.rhs_slot, rhs_slot=entry.lhs_slot)
                    for entry in plan.entries
                )

        if force_collective:
            _exchange_or_swap_grouped_slot_entries_collective(grouped, self.ep_rank, self.ep_size, self.ep_group)
        else:
            self._execute_swap_plan_batch(plan_list)

    @torch.no_grad()
    def swap_layer_pair(self, layer_key: str, pair: tuple[int, int]) -> None:
        plan = self._build_layer_swap_plan(layer_key, pair)
        if plan is None:
            return
        self._execute_swap_plans((plan,))
        layer = self.layers[layer_key]
        lhs, rhs = plan.logical_lhs, plan.logical_rhs
        layer.logical_to_physical[lhs], layer.logical_to_physical[rhs] = (
            layer.logical_to_physical[rhs].clone(),
            layer.logical_to_physical[lhs].clone(),
        )
        layer.refresh_identity()
        layer.invalidate_cache()
