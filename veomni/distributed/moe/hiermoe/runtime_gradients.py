"""Replica-gradient aggregation, overlap, and logical gradient-norm masking."""

from __future__ import annotations

import time
import zlib
from collections import defaultdict
from concurrent.futures import Future
from typing import Any, Iterable

import torch
import torch.distributed as dist

from ....utils.accelerator_timing import AcceleratorEvent, record_accelerator_event
from ....utils.device import get_torch_device
from .core_planner import QuotaPolicyEntry
from .runtime_settings import _full_timing_range
from .runtime_tensors import (
    _ep_global_rank,
    _existing_optimizer_state,
    _iter_leaf_optimizers,
    _local_tensor_view,
)
from .runtime_types import (
    ExpertLayerState,
    _CoverTensorEntry,
    _PipelineGradResult,
    _ReplicaGradContribution,
    _ReplicaGradGroup,
    _ReplicaGradSchedule,
    _SlotStateItem,
)


class GradientsMixin:
    """Replica-gradient aggregation, overlap, and logical gradient-norm masking."""

    def _register_pipeline_gradient_hooks(self, layer: ExpertLayerState) -> None:
        if not self.gradient_overlap_enabled:
            return
        unsupported = [
            index
            for index, param in enumerate(layer.expert_parameters)
            if not callable(getattr(param, "register_post_accumulate_grad_hook", None))
        ]
        if unsupported:
            raise RuntimeError(
                "PlaceMoE requires replica-gradient overlap, but layer "
                f"{layer.key!r} cannot register post-accumulate hooks for expert parameter indices "
                f"{unsupported}. Disable PlaceMoE explicitly or use a supported PyTorch parameter implementation; "
                "blocking synchronization is not selected automatically."
            )

        registered_handles: list[Any] = []
        registered_param_ids: list[int] = []
        for param_index, param in enumerate(layer.expert_parameters):
            if id(param) in self._pipeline_grad_hook_params:
                continue
            register = getattr(param, "register_post_accumulate_grad_hook", None)

            def hook(_param: torch.Tensor, *, key: str = layer.key, index: int = param_index) -> None:
                device = _local_tensor_view(_param).device
                self._pipeline_on_gradient_ready(key, index, self._pipeline_ready_event(device))

            try:
                handle = register(hook)
            except (RuntimeError, TypeError) as error:
                for registered_handle in registered_handles:
                    registered_handle.remove()
                    self._pipeline_grad_hook_handles.remove(registered_handle)
                for param_id in registered_param_ids:
                    self._pipeline_grad_hook_params.discard(param_id)
                raise RuntimeError(
                    "PlaceMoE requires replica-gradient overlap, but failed to register a gradient hook for "
                    f"layer {layer.key!r}, parameter index {param_index}: {error}. Blocking synchronization is not "
                    "selected automatically."
                ) from error
            self._pipeline_grad_hook_handles.append(handle)
            self._pipeline_grad_hook_params.add(id(param))
            registered_handles.append(handle)
            registered_param_ids.append(id(param))

    def _pipeline_on_gradient_ready(
        self,
        layer_key: str,
        param_index: int,
        ready_event: Any | None = None,
    ) -> None:
        if not self.gradient_overlap_enabled or self._pipeline_shutdown or not self._pipeline_is_final_microstep():
            return
        with self._pipeline_lock:
            index = int(param_index)
            self._pipeline_grad_ready[layer_key].add(index)
            if ready_event is not None:
                self._pipeline_grad_ready_events[layer_key][index] = ready_event
        self._advance_pipeline_gradient_queue()

    def _advance_pipeline_gradient_queue(self) -> None:
        """Submit ready layers after dispatch backward in deterministic order."""

        order = tuple(reversed(self._pipeline_layer_order or tuple(self.layers)))
        with self._pipeline_grad_submit_lock:
            while True:
                with self._pipeline_lock:
                    if self._pipeline_grad_comm_blocked:
                        return
                    index = self._pipeline_next_grad_index
                    if index >= len(order):
                        return
                    layer_key = order[index]
                    expected_parameters = len(self.layers[layer_key].expert_parameters)
                    if (
                        len(self._pipeline_grad_ready[layer_key]) < expected_parameters
                        or layer_key not in self._pipeline_grad_dispatch_complete
                    ):
                        return
                    self._pipeline_next_grad_index += 1
                self._submit_pipeline_gradient_sync(layer_key)

    def close_pipeline_gradient_window_before_dispatch(self, _layer_key: str) -> None:
        """Drain background gradient communication before training dispatch backward."""

        if not self.gradient_overlap_enabled or not self._pipeline_is_final_microstep():
            return
        if self._owns_pipeline_grad_group:
            # A dedicated gradient process group is allowed to overlap the
            # training group's dispatch collectives. Waiting here can form a
            # cross-group cycle when ranks reach backward layers at different
            # times: one rank waits for gradient P2P while its peer is already
            # waiting for that rank in the next dispatch collective.
            return
        with self._pipeline_grad_submit_lock:
            with self._pipeline_lock:
                self._pipeline_grad_comm_blocked = True
                futures = [
                    (layer_key, future)
                    for layer_key, future in self._pipeline_grad_futures.items()
                    if layer_key not in self._pipeline_grad_window_waited
                ]
        exposed_ms = 0.0
        for layer_key, future in futures:
            wait_started = time.perf_counter()
            self._wait_pipeline_gradient_result(future)
            exposed_ms += (time.perf_counter() - wait_started) * 1000.0
            with self._pipeline_lock:
                self._pipeline_grad_window_waited.add(layer_key)
        self._pipeline_grad_window_exposed_ms += exposed_ms
        self._accumulate_metric("hiermoe/pipeline_grad_sync_window_exposed_ms", exposed_ms)
        if exposed_ms > 0.01:
            self._accumulate_metric("hiermoe/pipeline_grad_sync_window_miss", 1)

    def open_pipeline_gradient_window_after_dispatch(self, layer_key: str) -> None:
        """Admit this layer's redundant-gradient sync after dispatch backward."""

        if not self.gradient_overlap_enabled or not self._pipeline_is_final_microstep():
            return
        with self._pipeline_grad_submit_lock:
            with self._pipeline_lock:
                self._pipeline_grad_dispatch_complete.add(layer_key)
                self._pipeline_grad_comm_blocked = False
        self._advance_pipeline_gradient_queue()

    @torch.no_grad()
    def _submit_pipeline_gradient_sync(self, layer_key: str) -> None:
        if not self.gradient_overlap_enabled or self._pipeline_shutdown:
            return
        layer = self.layers[layer_key]
        schedule = self._replica_grad_schedule_for_layer(layer)
        if not schedule.groups:
            return
        with self._pipeline_lock:
            if layer_key in self._pipeline_grad_futures:
                return
            previous_layer_key = next(reversed(self._pipeline_grad_futures), None)
            previous_future = None if previous_layer_key is None else self._pipeline_grad_futures[previous_layer_key]
        if previous_future is not None:
            wait_started = time.perf_counter()
            self._wait_pipeline_gradient_result(previous_future)
            exposed_ms = (time.perf_counter() - wait_started) * 1000.0
            self._pipeline_grad_window_exposed_ms += exposed_ms
            self._accumulate_metric("hiermoe/pipeline_grad_sync_backpressure_exposed_ms", exposed_ms)
        device = self._pipeline_device(layer)
        dispatch_event = self._pipeline_ready_event(device)
        with self._pipeline_lock:
            parameter_events = tuple(
                self._pipeline_grad_ready_events[layer_key][index]
                for index in sorted(self._pipeline_grad_ready_events[layer_key])
            )
        ready_events = parameter_events + ((dispatch_event,) if dispatch_event is not None else ())
        future: Future[_PipelineGradResult] = Future()
        result = self._pipeline_gradient_worker(layer_key, schedule, ready_events or None)
        future.set_result(result)
        with self._pipeline_lock:
            self._pipeline_grad_futures[layer_key] = future

    @torch.no_grad()
    def _pipeline_gradient_worker(
        self,
        layer_key: str,
        schedule: _ReplicaGradSchedule,
        ready_event: Any | None,
    ) -> _PipelineGradResult:
        layer = self.layers[layer_key]
        device = self._pipeline_device(layer)
        started = time.perf_counter()
        start_event: AcceleratorEvent | None = None
        completion_event: AcceleratorEvent | None = None

        def run() -> None:
            self._zero_inactive_slot_grads_for_layer(layer)
            contributions = self._replica_grad_contributions(layer, schedule)
            if schedule.pairwise:
                self._sync_pairwise_replica_gradients(
                    schedule,
                    contributions,
                    process_group=self._pipeline_grad_group,
                )
            else:
                self._sync_owner_replica_gradients(
                    schedule,
                    contributions,
                    process_group=self._pipeline_grad_group,
                )

        if device.type == "cpu":
            run()
        else:
            device_api = get_torch_device()
            device_api.set_device(device)
            stream = self._pipeline_stream("gradient", device)
            assert stream is not None
            with device_api.stream(stream):
                ready_events = ready_event if isinstance(ready_event, tuple) else (ready_event,)
                for event in ready_events:
                    if event is not None:
                        stream.wait_event(event)
                start_event = record_accelerator_event()
                run()
                completion_event = record_accelerator_event()
        return _PipelineGradResult(
            layer_key=layer_key,
            raw_ms=(time.perf_counter() - started) * 1000.0,
            start_event=start_event,
            completion_event=completion_event,
        )

    @staticmethod
    def _wait_pipeline_gradient_result(
        future: Future[_PipelineGradResult],
    ) -> tuple[_PipelineGradResult, float]:
        result = future.result()
        completion_event = result.completion_event
        if completion_event is not None and completion_event.event is not None:
            completion_event.event.synchronize()
        raw_ms = result.raw_ms
        if result.start_event is not None and completion_event is not None:
            raw_ms = result.start_event.elapsed_time(completion_event)
        return result, raw_ms

    @torch.no_grad()
    def _finish_pipeline_gradient_sync(self) -> None:
        order = self._pipeline_layer_order or tuple(self.layers)
        missing_hooks = {}
        for layer_key in order:
            layer = self.layers[layer_key]
            if not self._replica_grad_schedule_for_layer(layer).groups:
                continue
            expected = set(range(len(layer.expert_parameters)))
            missing = sorted(expected - self._pipeline_grad_ready[layer_key])
            if missing:
                missing_hooks[layer_key] = missing
        if missing_hooks:
            raise RuntimeError(
                "PlaceMoE replica-gradient overlap did not observe all registered expert gradients: "
                f"{missing_hooks}. Verify that every MoE layer uses the PlaceMoE dispatch path; blocking "
                "synchronization is not selected automatically."
            )
        with self._pipeline_grad_submit_lock:
            with self._pipeline_lock:
                self._pipeline_grad_comm_blocked = False
                self._pipeline_grad_dispatch_complete.update(order)
            for layer_key in reversed(order):
                self._submit_pipeline_gradient_sync(layer_key)
        with self._pipeline_lock:
            futures = [
                (layer_key, self._pipeline_grad_futures[layer_key])
                for layer_key in reversed(order)
                if layer_key in self._pipeline_grad_futures
            ]
        raw_ms = 0.0
        deadline_exposed_ms = 0.0
        for layer_key, future in futures:
            if layer_key in self._pipeline_grad_window_waited:
                _result, layer_raw_ms = self._wait_pipeline_gradient_result(future)
            else:
                wait_started = time.perf_counter()
                _result, layer_raw_ms = self._wait_pipeline_gradient_result(future)
                deadline_exposed_ms += (time.perf_counter() - wait_started) * 1000.0
            raw_ms += layer_raw_ms
            with self._pipeline_lock:
                self._pipeline_grad_futures.pop(layer_key, None)
        exposed_ms = self._pipeline_grad_window_exposed_ms + deadline_exposed_ms
        self._accumulate_metric("hiermoe/pipeline_grad_sync_jobs", len(futures))
        self._accumulate_metric("hiermoe/pipeline_grad_sync_raw_ms", raw_ms)
        self._accumulate_metric("hiermoe/pipeline_grad_sync_deadline_exposed_ms", deadline_exposed_ms)
        self._accumulate_metric("hiermoe/pipeline_grad_sync_exposed_ms", exposed_ms)
        if raw_ms > 0.0:
            self._placement_metrics["hiermoe/pipeline_grad_sync_hidden_ratio"] = max(
                0.0,
                min(1.0, 1.0 - exposed_ms / raw_ms),
            )
        if deadline_exposed_ms > 0.01:
            self._accumulate_metric("hiermoe/pipeline_grad_sync_deadline_miss", 1)
        with self._pipeline_lock:
            self._pipeline_grad_window_waited.clear()
            self._pipeline_grad_window_exposed_ms = 0.0
        self._clear_accumulated_token_counts()

    @staticmethod
    def _zero_grad_slots(param: torch.nn.Parameter, zero_slots: torch.Tensor) -> None:
        grad = getattr(param, "grad", None)
        if not torch.is_tensor(grad):
            return
        local_grad = _local_tensor_view(grad)
        if tuple(local_grad.shape) != tuple(_local_tensor_view(param).shape):
            return
        local_zero_slots = zero_slots.to(device=local_grad.device, dtype=torch.bool)
        slot_mask_shape = (int(local_zero_slots.numel()),) + (1,) * (local_grad.ndim - 1)
        local_grad.detach().masked_fill_(local_zero_slots.view(slot_mask_shape), 0)

    @staticmethod
    def _local_grad_for_redundant_sync(param: torch.nn.Parameter) -> torch.Tensor | None:
        grad = getattr(param, "grad", None)
        if not torch.is_tensor(grad):
            # Copy ranks must all participate in redundant-gradient sync. A missing
            # grad is an explicit zero contribution, not a reason to skip P2P.
            param.grad = torch.zeros_like(param)
            grad = param.grad
        local_grad = _local_tensor_view(grad)
        if tuple(local_grad.shape) != tuple(_local_tensor_view(param).shape):
            return None
        return local_grad

    @torch.no_grad()
    def _zero_inactive_slot_grads(self) -> None:
        for layer in self.layers.values():
            self._zero_inactive_slot_grads_for_layer(layer)

    @torch.no_grad()
    def _zero_inactive_slot_grads_for_layer(self, layer: ExpertLayerState) -> None:
        counts = layer.accumulated_tokens_per_local_expert
        if counts is None or not layer.slot_layout_enabled:
            return
        if counts.ndim != 1 or int(counts.numel()) != int(layer.num_local_experts):
            layer.accumulated_tokens_per_local_expert = None
            return
        zero_slots = counts <= 0
        # NPU grouped-matmul backward may leave undefined weight gradients
        # for zero-token groups. Those slots are mathematically inactive.
        for parameter in layer.expert_parameters:
            self._zero_grad_slots(parameter, zero_slots)

    def _clear_accumulated_token_counts(self) -> None:
        for layer in self.layers.values():
            layer.accumulated_tokens_per_local_expert = None

    def _slot_layout_is_device_unique(self, layer: ExpertLayerState, slot_to_logical: torch.Tensor) -> bool:
        for rank in range(self.ep_size):
            seen: set[int] = set()
            start = rank * layer.num_local_experts
            for slot in range(start, start + layer.num_local_experts):
                logical = int(slot_to_logical[slot].item())
                if logical < 0:
                    continue
                if logical in seen:
                    return False
                seen.add(logical)
        return True

    def _validate_placement_layout(
        self,
        layer: ExpertLayerState,
        slot_to_logical: torch.Tensor | Iterable[int],
        owner_slots: torch.Tensor | Iterable[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        layout = torch.as_tensor(slot_to_logical, dtype=torch.long).detach().cpu().reshape(-1).clone()
        if layout.numel() != layer.num_physical_slots:
            raise ValueError(
                f"HierMoE slot layout for {layer.key} has {layout.numel()} slots, expected {layer.num_physical_slots}."
            )
        if bool(((layout < -1) | (layout >= layer.num_experts)).any().item()):
            raise ValueError(f"HierMoE slot layout for {layer.key} contains an invalid logical expert.")
        active = layout[layout >= 0]
        counts = (
            torch.bincount(active, minlength=layer.num_experts)
            if active.numel()
            else torch.zeros((layer.num_experts,), dtype=torch.long)
        )
        if bool((counts <= 0).any().item()):
            raise ValueError(f"HierMoE slot layout for {layer.key} drops at least one logical expert.")
        if not self._slot_layout_is_device_unique(layer, layout):
            raise ValueError(f"HierMoE slot layout for {layer.key} duplicates an expert on one device.")

        if owner_slots is None:
            return layout, None
        owners = torch.as_tensor(owner_slots, dtype=torch.long).detach().cpu().reshape(-1).clone()
        if owners.numel() != layer.num_experts:
            raise ValueError(
                f"HierMoE owner mapping for {layer.key} has {owners.numel()} entries, expected {layer.num_experts}."
            )
        if bool(((owners < 0) | (owners >= layout.numel())).any().item()):
            raise ValueError(f"HierMoE owner mapping for {layer.key} contains an invalid slot.")
        if len({int(value) for value in owners.tolist()}) != layer.num_experts:
            raise ValueError(f"HierMoE owner mapping for {layer.key} contains duplicate owner slots.")
        for logical_expert, physical_slot in enumerate(owners.tolist()):
            if int(layout[int(physical_slot)].item()) != logical_expert:
                raise ValueError(
                    f"HierMoE owner slot {physical_slot} for logical expert {logical_expert} "
                    "does not contain that expert."
                )
        return layout, owners

    def _validate_quota_policy(
        self,
        layer: ExpertLayerState,
        slot_to_logical: torch.Tensor | Iterable[int],
        quota_policy: Iterable[QuotaPolicyEntry],
    ) -> tuple[QuotaPolicyEntry, ...]:
        layout = torch.as_tensor(slot_to_logical, dtype=torch.long).detach().cpu().reshape(-1)
        entries = tuple(quota_policy)
        policy_keys: set[tuple[int, int, tuple[int, ...]]] = set()
        for entry in entries:
            if not 0 <= entry.source_rank < self.ep_size:
                raise ValueError(f"HierMoE quota policy for {layer.key} has an invalid source rank.")
            if not 0 <= entry.logical_expert < layer.num_experts:
                raise ValueError(f"HierMoE quota policy for {layer.key} has an invalid expert.")
            if not entry.destination_ranks or len(entry.destination_ranks) != len(entry.quotas):
                raise ValueError(f"HierMoE quota policy for {layer.key} has inconsistent quota widths.")
            if tuple(sorted(set(entry.destination_ranks))) != entry.destination_ranks or any(
                not 0 <= rank < self.ep_size for rank in entry.destination_ranks
            ):
                raise ValueError(f"HierMoE quota policy for {layer.key} has invalid destination ranks.")
            if any(quota < 0 for quota in entry.quotas):
                raise ValueError(f"HierMoE quota policy for {layer.key} has a negative quota.")
            policy_key = (
                entry.source_rank,
                entry.logical_expert,
                entry.destination_ranks,
            )
            if policy_key in policy_keys:
                raise ValueError(f"HierMoE quota policy for {layer.key} contains duplicate rows.")
            policy_keys.add(policy_key)
            copy_ranks = {
                slot // layer.num_local_experts
                for slot in torch.nonzero(layout == entry.logical_expert, as_tuple=False).flatten().tolist()
            }
            if any(rank not in copy_ranks for rank in entry.destination_ranks):
                raise ValueError(f"HierMoE quota policy for {layer.key} references a rank without a copy.")
        return entries

    def _validate_optimizer_state_slot_tensors_across_ep(
        self,
        rows: Iterable[tuple[torch.nn.Parameter, list[_SlotStateItem]]],
    ) -> None:
        state_rows = list(rows)
        if not state_rows or self.ep_group is None or self.ep_size <= 1:
            return

        bounds_rows = []
        for _param, items in state_rows:
            count = len(items)
            signature_rows = tuple(
                (
                    descriptor,
                    str(_local_tensor_view(tensor).dtype),
                    tuple(int(value) for value in _local_tensor_view(tensor).shape),
                )
                for descriptor, tensor in items
            )
            signature = zlib.crc32(repr(signature_rows).encode("utf-8"))
            bounds_rows.append((count, -count, signature, -signature))

        local_bounds = torch.tensor(
            bounds_rows,
            dtype=torch.long,
            device=_local_tensor_view(state_rows[0][0]).device,
        )
        global_bounds = local_bounds.clone()
        dist.all_reduce(global_bounds, op=dist.ReduceOp.MIN, group=self.ep_group)
        if not torch.equal(global_bounds, local_bounds):
            raise RuntimeError(
                "HierMoE cannot migrate an asymmetric swap payload across the EP group; "
                "ordered parameter, gradient, and optimizer-state descriptors must match on every rank."
            )

    def _optimizer_state_slot_items_for_slot_op(
        self,
        param: torch.nn.Parameter,
    ) -> list[_SlotStateItem]:
        known_state_names = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "compensation", "momentum_buffer")
        items: list[_SlotStateItem] = []
        bindings = self._optimizer_param_bindings.get(id(param), ())
        optimizers = tuple(binding.optimizer for binding in bindings) or tuple(_iter_leaf_optimizers(self.optimizer))
        for optimizer_index, optimizer in enumerate(optimizers):
            state = _existing_optimizer_state(optimizer, param)
            if not state:
                continue
            matching = [
                (state_name, value)
                for state_name, value in state.items()
                if torch.is_tensor(value)
                and tuple(_local_tensor_view(value).shape) == tuple(_local_tensor_view(param).shape)
            ]
            ordered = [item for known_name in known_state_names for item in matching if item[0] == known_name]
            ordered.extend(
                sorted(
                    (item for item in matching if item[0] not in known_state_names),
                    key=lambda item: str(item[0]),
                )
            )
            items.extend((f"optimizer[{optimizer_index}].{state_name}", tensor) for state_name, tensor in ordered)
        return items

    def _slot_op_state_rows(
        self,
        layer: ExpertLayerState,
    ) -> list[tuple[torch.nn.Parameter, list[_SlotStateItem]]]:
        rows: list[tuple[torch.nn.Parameter, list[_SlotStateItem]]] = []
        for param in layer.expert_parameters:
            items: list[_SlotStateItem] = [("parameter", param)]
            grad = getattr(param, "grad", None)
            if torch.is_tensor(grad):
                if tuple(_local_tensor_view(grad).shape) != tuple(_local_tensor_view(param).shape):
                    raise RuntimeError(
                        f"HierMoE cannot migrate a gradient whose shape differs from parameter {tuple(param.shape)}."
                    )
                items.append(("gradient", grad))
            items.extend(self._optimizer_state_slot_items_for_slot_op(param))
            rows.append((param, items))
        return rows

    def _slot_op_state_tensors(self, layer: ExpertLayerState) -> list[torch.Tensor]:
        return [tensor for _param, items in self._slot_op_state_rows(layer) for _descriptor, tensor in items]

    @staticmethod
    def _slot_op_cover_entries_from_tensors(
        tensors: Iterable[torch.Tensor],
        *,
        num_local_experts: int,
        src_slot: int,
        dst_slot: int,
    ) -> list[_CoverTensorEntry]:
        src_local = int(src_slot) % int(num_local_experts)
        dst_local = int(dst_slot) % int(num_local_experts)
        return [_CoverTensorEntry(tensor, src_slot=src_local, dst_slot=dst_local) for tensor in tensors]

    def _refresh_layer_mapping_from_slots(
        self,
        layer: ExpertLayerState,
        owner_slots: torch.Tensor | Iterable[int] | None = None,
    ) -> None:
        if layer.slot_to_logical is None:
            return
        if owner_slots is not None:
            mapping = torch.as_tensor(owner_slots, dtype=torch.long).detach().cpu().reshape(-1).clone()
            if mapping.numel() != layer.num_experts:
                raise RuntimeError(
                    f"HierMoE owner mapping for {layer.key} has {mapping.numel()} entries, "
                    f"expected {layer.num_experts}."
                )
            for logical_expert, physical_slot in enumerate(mapping.tolist()):
                if not 0 <= int(physical_slot) < layer.num_physical_slots:
                    raise RuntimeError(
                        f"HierMoE owner slot {physical_slot} for logical expert {logical_expert} is out of range."
                    )
                if int(layer.slot_to_logical[int(physical_slot)].item()) != logical_expert:
                    raise RuntimeError(
                        f"HierMoE owner slot {physical_slot} does not contain logical expert {logical_expert}."
                    )
            layer.logical_to_physical = mapping
            layer.refresh_identity()
            layer.invalidate_cache()
            return
        mapping = torch.empty((layer.num_experts,), dtype=torch.long)
        for logical_expert in range(layer.num_experts):
            slots = torch.nonzero(layer.slot_to_logical == logical_expert, as_tuple=False).flatten()
            if slots.numel() == 0:
                raise RuntimeError(f"HierMoE slot layout lost all copies of logical expert {logical_expert}.")
            canonical = layer.canonical_physical_slots
            if canonical is not None and int(canonical[logical_expert].item()) in set(slots.tolist()):
                mapping[logical_expert] = int(canonical[logical_expert].item())
            else:
                mapping[logical_expert] = int(slots[0].item())
        layer.logical_to_physical = mapping
        layer.refresh_identity()
        layer.invalidate_cache()

    @staticmethod
    def _owner_rank_for_copy_group(
        layer: ExpertLayerState,
        logical_expert: int,
        slots: Iterable[int],
    ) -> int:
        logical = int(logical_expert)
        if not 0 <= logical < layer.num_experts:
            raise RuntimeError(f"HierMoE redundant gradient sync received invalid logical expert {logical}.")
        slot_values = tuple(int(slot) for slot in slots)
        owner_slot = int(layer.logical_to_physical[logical].item())
        if owner_slot not in slot_values:
            raise RuntimeError(
                f"HierMoE owner slot {owner_slot} for logical expert {logical} is not in its copy group."
            )
        return owner_slot // layer.num_local_experts

    def _replica_grad_schedule_for_layer(self, layer: ExpertLayerState) -> _ReplicaGradSchedule:
        cached = layer._replica_grad_schedule_cache
        if cached is not None:
            return cached

        groups = []
        for logical_expert, slots in layer.redundant_copy_groups():
            copy_ranks = tuple(sorted({int(slot) // layer.num_local_experts for slot in slots}))
            local_slots = tuple(
                int(slot) % layer.num_local_experts
                for slot in slots
                if int(slot) // layer.num_local_experts == self.ep_rank
            )
            groups.append(
                _ReplicaGradGroup(
                    logical_expert=int(logical_expert),
                    owner_rank=self._owner_rank_for_copy_group(layer, int(logical_expert), list(slots)),
                    copy_ranks=copy_ranks,
                    local_slots=local_slots,
                )
            )
        pairwise = all(len(group.copy_ranks) <= 2 for group in groups)
        peer_neighbors = [set() for _ in range(int(self.ep_size))]
        if pairwise:
            for group in groups:
                if len(group.copy_ranks) != 2:
                    continue
                left_rank, right_rank = group.copy_ranks
                peer_neighbors[int(left_rank)].add(int(right_rank))
                peer_neighbors[int(right_rank)].add(int(left_rank))
        cached = _ReplicaGradSchedule(
            groups=tuple(groups),
            pairwise=pairwise,
            # Fixed mirrored R2 has one peer per rank and safely uses one
            # batched wave. Arbitrary partial-capacity layouts form a general
            # rank graph; all ranks must traverse its edges in the same order
            # or HCCL can leave the P2P work pending indefinitely.
            globally_ordered_pairs=pairwise and any(len(neighbors) > 1 for neighbors in peer_neighbors),
        )
        layer._replica_grad_schedule_cache = cached
        return cached

    def _replica_grad_contributions(
        self,
        layer: ExpertLayerState,
        schedule: _ReplicaGradSchedule,
    ) -> dict[tuple[torch.device, torch.dtype], dict[tuple[int, int], _ReplicaGradContribution]]:
        params = layer.expert_parameters
        local_grads = tuple(self._local_grad_for_redundant_sync(param) for param in params)
        contributions: dict[
            tuple[torch.device, torch.dtype],
            dict[tuple[int, int], _ReplicaGradContribution],
        ] = defaultdict(dict)
        # Pre-seed the local parameter buckets so every rank enters the same
        # packed collective even when a partial-capacity layout gives this
        # rank no redundant expert in a particular layer.
        for local_grad in local_grads:
            if local_grad is not None:
                contributions[(local_grad.device, local_grad.dtype)]
        for group in schedule.groups:
            if not group.local_slots:
                continue
            for param_index, local_grad in enumerate(local_grads):
                if local_grad is None:
                    raise RuntimeError(
                        f"HierMoE redundant gradient for layer {layer.key} does not match its local parameter shape."
                    )
                local_sum = local_grad.detach()[group.local_slots[0]].clone()
                for local_slot in group.local_slots[1:]:
                    local_sum.add_(local_grad.detach()[local_slot])
                contributions[(local_sum.device, local_sum.dtype)][(group.logical_expert, param_index)] = (
                    _ReplicaGradContribution(
                        logical_expert=group.logical_expert,
                        param_index=param_index,
                        local_grad=local_grad,
                        local_slots=group.local_slots,
                        local_sum=local_sum,
                    )
                )
        return contributions

    @staticmethod
    def _replica_grad_bucket_sort_key(
        key: tuple[torch.device, torch.dtype],
    ) -> tuple[str, int, str]:
        device, dtype = key
        return (device.type, -1 if device.index is None else int(device.index), str(dtype))

    @staticmethod
    def _unpack_replica_grad(total: torch.Tensor, contribution: _ReplicaGradContribution) -> None:
        synced = total.view_as(contribution.local_sum)
        for local_slot in contribution.local_slots:
            contribution.local_grad.detach()[local_slot].copy_(synced)

    def _replica_grad_buffer(
        self,
        *,
        kind: str,
        peer_rank: int,
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        key = (kind, int(peer_rank), str(device), str(dtype))
        cached = self._replica_grad_buffers.get(key)
        if cached is None or cached.numel() < numel:
            cached = torch.empty((numel,), dtype=dtype, device=device)
            self._replica_grad_buffers[key] = cached
        return cached[:numel]

    def _pack_replica_grad_items(
        self,
        *,
        kind: str,
        peer_rank: int,
        items: Iterable[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        parts = tuple(item.reshape(-1) for item in items)
        numel = sum(int(part.numel()) for part in parts)
        buffer = self._replica_grad_buffer(
            kind=kind,
            peer_rank=peer_rank,
            numel=numel,
            dtype=dtype,
            device=device,
        )
        offset = 0
        for part in parts:
            buffer[offset : offset + part.numel()].copy_(part)
            offset += int(part.numel())
        return buffer

    def _run_replica_grad_p2p_wave(
        self,
        *,
        phase: str,
        send_buffers: dict[int, torch.Tensor],
        recv_specs: dict[int, tuple[int, torch.dtype, torch.device]],
        process_group: dist.ProcessGroup | None = None,
        globally_ordered_pairs: bool = False,
    ) -> dict[int, torch.Tensor]:
        process_group = self.ep_group if process_group is None else process_group
        if process_group is None or self.ep_size <= 1:
            if send_buffers or recv_specs:
                raise RuntimeError("HierMoE redundant gradient synchronization requires an EP process group.")
            return {}

        recv_buffers = {
            peer_rank: self._replica_grad_buffer(
                kind=f"{phase}_recv",
                peer_rank=peer_rank,
                numel=numel,
                dtype=dtype,
                device=device,
            )
            for peer_rank, (numel, dtype, device) in recv_specs.items()
        }

        def _run_peer(peer_rank: int) -> None:
            peer_global_rank = _ep_global_rank(process_group, peer_rank)
            ops: list[dist.P2POp] = []
            send_buffer = send_buffers.get(peer_rank)
            if send_buffer is not None:
                ops.append(dist.P2POp(dist.isend, send_buffer, peer_global_rank, process_group))
            recv_buffer = recv_buffers.get(peer_rank)
            if recv_buffer is not None:
                ops.append(dist.P2POp(dist.irecv, recv_buffer, peer_global_rank, process_group))
            if not ops:
                return
            works = dist.batch_isend_irecv(ops)
            for work in works:
                work.wait()

        if globally_ordered_pairs:
            # An arbitrary EPLB layout can give one expert copies on many
            # ranks.  Issuing every rank's owner-star peer list in its local
            # order can create a different HCCL P2P order on each rank and
            # deadlock.  Traverse the same undirected rank-pair schedule
            # everywhere; only the two participating ranks issue work.
            for left_rank in range(int(self.ep_size)):
                for right_rank in range(left_rank + 1, int(self.ep_size)):
                    if self.ep_rank == left_rank:
                        _run_peer(right_rank)
                    elif self.ep_rank == right_rank:
                        _run_peer(left_rank)
        else:
            ops: list[dist.P2POp] = []
            for peer_rank in sorted(set(send_buffers) | set(recv_buffers)):
                peer_global_rank = _ep_global_rank(process_group, peer_rank)
                send_buffer = send_buffers.get(peer_rank)
                if send_buffer is not None:
                    ops.append(dist.P2POp(dist.isend, send_buffer, peer_global_rank, process_group))
                recv_buffer = recv_buffers.get(peer_rank)
                if recv_buffer is not None:
                    ops.append(dist.P2POp(dist.irecv, recv_buffer, peer_global_rank, process_group))
            if ops:
                works = dist.batch_isend_irecv(ops)
                for work in works:
                    work.wait()
        return recv_buffers

    def _run_replica_grad_all_to_all_wave(
        self,
        *,
        send_buffers: dict[int, torch.Tensor],
        process_group: dist.ProcessGroup | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[int, torch.Tensor]:
        """Exchange an arbitrary pair graph with one globally ordered collective."""

        process_group = self.ep_group if process_group is None else process_group
        if process_group is None or self.ep_size <= 1:
            if send_buffers:
                raise RuntimeError("HierMoE redundant gradient synchronization requires an EP process group.")
            return {}

        split_sizes = [
            0 if rank not in send_buffers else int(send_buffers[rank].numel()) for rank in range(int(self.ep_size))
        ]
        parts = [send_buffers[rank] for rank in range(int(self.ep_size)) if split_sizes[rank] > 0]
        send_buffer = torch.cat(parts, dim=0) if parts else torch.empty((0,), dtype=dtype, device=device)
        recv_buffer = torch.empty_like(send_buffer)
        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=split_sizes,
            input_split_sizes=split_sizes,
            group=process_group,
        )

        received: dict[int, torch.Tensor] = {}
        offset = 0
        for peer_rank, size in enumerate(split_sizes):
            if size > 0:
                received[peer_rank] = recv_buffer[offset : offset + size]
                offset += size
        return received

    def _sync_pairwise_replica_gradients(
        self,
        schedule: _ReplicaGradSchedule,
        contributions_by_bucket: dict[
            tuple[torch.device, torch.dtype],
            dict[tuple[int, int], _ReplicaGradContribution],
        ],
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        for bucket_key in sorted(contributions_by_bucket, key=self._replica_grad_bucket_sort_key):
            device, dtype = bucket_key
            contributions = contributions_by_bucket[bucket_key]
            peer_items: dict[int, list[_ReplicaGradContribution]] = defaultdict(list)
            for group in schedule.groups:
                if not group.local_slots:
                    continue
                for param_index in (0, 1):
                    contribution = contributions.get((group.logical_expert, param_index))
                    if contribution is None:
                        continue
                    remote_ranks = tuple(rank for rank in group.copy_ranks if rank != self.ep_rank)
                    if not remote_ranks:
                        self._unpack_replica_grad(contribution.local_sum, contribution)
                        continue
                    if len(remote_ranks) != 1:
                        raise RuntimeError("Pairwise redundant gradient schedule contains more than one remote copy.")
                    peer_items[remote_ranks[0]].append(contribution)

            send_buffers = {
                peer_rank: self._pack_replica_grad_items(
                    kind="pairwise_send",
                    peer_rank=peer_rank,
                    items=(item.local_sum for item in items),
                    dtype=dtype,
                    device=device,
                )
                for peer_rank, items in peer_items.items()
            }
            recv_specs = {
                peer_rank: (int(send_buffer.numel()), dtype, device) for peer_rank, send_buffer in send_buffers.items()
            }
            if schedule.globally_ordered_pairs:
                # A lexicographic sequence of blocking rank-pair P2P calls is
                # not a true global order: ranks skip inactive edges and can
                # form a wait cycle on a general multi-neighbor replica graph.
                # Pack all incident edges into one collective. Each two-copy
                # edge is symmetric, so the input/output split vector is the
                # same and no split-size exchange is required.
                recv_buffers = self._run_replica_grad_all_to_all_wave(
                    send_buffers=send_buffers,
                    process_group=process_group,
                    dtype=dtype,
                    device=device,
                )
            else:
                recv_buffers = self._run_replica_grad_p2p_wave(
                    phase="pairwise",
                    send_buffers=send_buffers,
                    process_group=process_group,
                    recv_specs=recv_specs,
                    globally_ordered_pairs=False,
                )
            for peer_rank, items in peer_items.items():
                recv_buffer = recv_buffers[peer_rank]
                offset = 0
                for item in items:
                    remote = recv_buffer[offset : offset + item.numel].view_as(item.local_sum)
                    total = item.local_sum + remote
                    self._unpack_replica_grad(total, item)
                    offset += item.numel

    def _sync_owner_replica_gradients(
        self,
        schedule: _ReplicaGradSchedule,
        contributions_by_bucket: dict[
            tuple[torch.device, torch.dtype],
            dict[tuple[int, int], _ReplicaGradContribution],
        ],
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        for bucket_key in sorted(contributions_by_bucket, key=self._replica_grad_bucket_sort_key):
            device, dtype = bucket_key
            contributions = contributions_by_bucket[bucket_key]
            reduce_send_items: dict[int, list[_ReplicaGradContribution]] = defaultdict(list)
            reduce_recv_items: dict[int, list[_ReplicaGradContribution]] = defaultdict(list)
            owner_totals: dict[tuple[int, int], torch.Tensor] = {}

            for group in schedule.groups:
                if self.ep_rank not in group.copy_ranks:
                    continue
                for param_index in (0, 1):
                    contribution = contributions.get((group.logical_expert, param_index))
                    if contribution is None:
                        continue
                    item_key = (group.logical_expert, param_index)
                    if self.ep_rank == group.owner_rank:
                        owner_totals[item_key] = contribution.local_sum.clone()
                        for source_rank in group.copy_ranks:
                            if source_rank != group.owner_rank:
                                reduce_recv_items[source_rank].append(contribution)
                    else:
                        reduce_send_items[group.owner_rank].append(contribution)

            reduce_send_buffers = {
                peer_rank: self._pack_replica_grad_items(
                    kind="reduce_send",
                    peer_rank=peer_rank,
                    items=(item.local_sum for item in items),
                    dtype=dtype,
                    device=device,
                )
                for peer_rank, items in reduce_send_items.items()
            }
            reduce_recv_specs = {
                peer_rank: (sum(item.numel for item in items), dtype, device)
                for peer_rank, items in reduce_recv_items.items()
            }
            reduce_recv_buffers = self._run_replica_grad_p2p_wave(
                phase="reduce",
                send_buffers=reduce_send_buffers,
                process_group=process_group,
                recv_specs=reduce_recv_specs,
                globally_ordered_pairs=True,
            )
            for peer_rank, items in reduce_recv_items.items():
                recv_buffer = reduce_recv_buffers[peer_rank]
                offset = 0
                for item in items:
                    remote = recv_buffer[offset : offset + item.numel].view_as(item.local_sum)
                    owner_totals[(item.logical_expert, item.param_index)].add_(remote)
                    offset += item.numel

            broadcast_send_keys: dict[int, list[tuple[int, int]]] = defaultdict(list)
            broadcast_recv_items: dict[int, list[_ReplicaGradContribution]] = defaultdict(list)
            for group in schedule.groups:
                if self.ep_rank not in group.copy_ranks:
                    continue
                for param_index in (0, 1):
                    contribution = contributions.get((group.logical_expert, param_index))
                    if contribution is None:
                        continue
                    item_key = (group.logical_expert, param_index)
                    if self.ep_rank == group.owner_rank:
                        for destination_rank in group.copy_ranks:
                            if destination_rank != group.owner_rank:
                                broadcast_send_keys[destination_rank].append(item_key)
                    else:
                        broadcast_recv_items[group.owner_rank].append(contribution)

            broadcast_send_buffers = {
                peer_rank: self._pack_replica_grad_items(
                    kind="broadcast_send",
                    peer_rank=peer_rank,
                    items=(owner_totals[item_key] for item_key in item_keys),
                    dtype=dtype,
                    device=device,
                )
                for peer_rank, item_keys in broadcast_send_keys.items()
            }
            broadcast_recv_specs = {
                peer_rank: (sum(item.numel for item in items), dtype, device)
                for peer_rank, items in broadcast_recv_items.items()
            }
            broadcast_recv_buffers = self._run_replica_grad_p2p_wave(
                phase="broadcast",
                send_buffers=broadcast_send_buffers,
                process_group=process_group,
                recv_specs=broadcast_recv_specs,
                globally_ordered_pairs=True,
            )

            for item_key, total in owner_totals.items():
                self._unpack_replica_grad(total, contributions[item_key])
            for peer_rank, items in broadcast_recv_items.items():
                recv_buffer = broadcast_recv_buffers[peer_rank]
                offset = 0
                for item in items:
                    total = recv_buffer[offset : offset + item.numel]
                    self._unpack_replica_grad(total, item)
                    offset += item.numel

    @torch.no_grad()
    def sync_redundant_gradients(self) -> None:
        if self.gradient_overlap_enabled:
            with _full_timing_range("hiermoe_redundant_grad_sync_deadline"):
                self._finish_pipeline_gradient_sync()
            return
        started = time.perf_counter()
        jobs = 0
        with _full_timing_range("hiermoe_redundant_grad_sync"):
            self._zero_inactive_slot_grads()
            for layer in self.layers.values():
                if layer.slot_to_logical is None:
                    continue
                schedule = self._replica_grad_schedule_for_layer(layer)
                if not schedule.groups:
                    continue
                jobs += 1
                contributions = self._replica_grad_contributions(layer, schedule)
                if schedule.pairwise:
                    self._sync_pairwise_replica_gradients(schedule, contributions)
                else:
                    self._sync_owner_replica_gradients(schedule, contributions)
            self._clear_accumulated_token_counts()
        if self.fixed_pipeline_overlap:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._accumulate_metric("hiermoe/pipeline_grad_sync_jobs", jobs)
            self._accumulate_metric("hiermoe/pipeline_grad_sync_raw_ms", elapsed_ms)
            self._accumulate_metric("hiermoe/pipeline_grad_sync_deadline_exposed_ms", elapsed_ms)
            self._accumulate_metric("hiermoe/pipeline_grad_sync_exposed_ms", elapsed_ms)
            self._placement_metrics["hiermoe/pipeline_grad_sync_hidden_ratio"] = 0.0
            if elapsed_ms > 0.01:
                self._accumulate_metric("hiermoe/pipeline_grad_sync_deadline_miss", 1)

    def redundant_grad_norm_masks(self) -> dict[int, torch.Tensor]:
        masks: dict[int, torch.Tensor] = {}
        for layer in self.layers.values():
            if not layer.slot_layout_enabled or not layer.redundant_copy_groups():
                continue
            mask = torch.zeros((layer.num_local_experts,), dtype=torch.bool)
            for physical_slot in layer.logical_to_physical.tolist():
                rank, local_slot = divmod(int(physical_slot), layer.num_local_experts)
                if rank == self.ep_rank:
                    mask[local_slot] = True
            for param in layer.expert_parameters:
                masks[id(param)] = mask
        return masks
