"""Expert registration and logical-to-physical dispatch mapping."""

from __future__ import annotations

import zlib
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from placemoe.model_adapter import resolve_moe_model_adapter

from . import runtime_settings as settings
from .core_planner import assign_tokens_to_copies_with_quota
from .greedy_planner import assign_tokens_to_copies_greedy
from .planner import assign_tokens_to_copies, assign_tokens_to_mirrored_r2
from .runtime_settings import logger
from .runtime_tensors import _cover_grouped_slot_entries_atomic, _local_tensor_view
from .runtime_types import ExpertLayerState, _canonical_physical_slots, _CoverTensorEntry, _initial_slot_to_logical


class RoutingMixin:
    """Expert registration and logical-to-physical dispatch mapping."""

    def _debug_should_log_copy_stats(self, layer_key: str) -> bool:
        if not settings._DEBUG_REDUNDANT_COPY_STATS:
            return False
        layer_keys = sorted(key for key, layer in self.layers.items() if layer.slot_layout_enabled)
        return layer_key in set(layer_keys[: settings._DEBUG_REDUNDANT_COPY_STATS_MAX_LAYERS])

    @staticmethod
    def _debug_slot_stats(tensor: torch.Tensor, local_slot: int) -> torch.Tensor:
        local = _local_tensor_view(tensor).detach()
        values = local[int(local_slot)].to(dtype=torch.float32)
        if values.numel() == 0:
            zero = torch.zeros((), dtype=torch.float32, device=values.device)
            return torch.stack((zero, zero, zero, zero))
        return torch.stack((values.sum(), values.square().sum(), values.abs().max(), values.mean()))

    def _debug_global_slot_stats(self, tensor: torch.Tensor, layer: ExpertLayerState, slot: int) -> torch.Tensor:
        local = _local_tensor_view(tensor)
        stats = torch.zeros((4,), dtype=torch.float32, device=local.device)
        slot_rank, local_slot = divmod(int(slot), layer.num_local_experts)
        if self.ep_rank == slot_rank:
            stats = self._debug_slot_stats(tensor, local_slot)
        if self.ep_group is not None and self.ep_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=self.ep_group)
        return stats

    def _debug_copy_group_stat_delta(
        self,
        tensor: torch.Tensor,
        layer: ExpertLayerState,
        slots: tuple[int, ...],
    ) -> float:
        ref_stats: torch.Tensor | None = None
        max_delta = 0.0
        for slot in slots:
            stats = self._debug_global_slot_stats(tensor, layer, int(slot))
            if ref_stats is None:
                ref_stats = stats
                continue
            delta = float((stats - ref_stats).abs().max().detach().cpu().item())
            max_delta = max(max_delta, delta)
        return max_delta

    def _debug_global_accumulated_counts(self, layer: ExpertLayerState) -> torch.Tensor:
        local_device = _local_tensor_view(layer.primary_parameter).device
        counts = layer.accumulated_tokens_per_local_expert
        if counts is None or counts.ndim != 1 or int(counts.numel()) != int(layer.num_local_experts):
            local_counts = torch.zeros((layer.num_local_experts,), dtype=torch.float32, device=local_device)
        else:
            local_counts = counts.detach().to(device=local_device, dtype=torch.float32)
        if self.ep_group is None or self.ep_size <= 1:
            return local_counts.detach().cpu()
        gathered = torch.empty(
            (self.ep_size * layer.num_local_experts,),
            dtype=torch.float32,
            device=local_device,
        )
        dist.all_gather_into_tensor(gathered, local_counts.contiguous(), group=self.ep_group)
        return gathered.detach().cpu()

    def _debug_log_redundant_copy_stats(
        self,
        phase: str,
        *,
        layer_key: str | None = None,
        layer: ExpertLayerState | None = None,
        include_grads: bool = False,
    ) -> None:
        if not settings._DEBUG_REDUNDANT_COPY_STATS:
            return
        if layer_key is not None and layer is not None:
            items = [(layer_key, layer)] if self._debug_should_log_copy_stats(layer_key) else []
        else:
            items = [
                (key, candidate)
                for key, candidate in sorted(self.layers.items())
                if self._debug_should_log_copy_stats(key)
            ]

        for key, candidate in items:
            groups = candidate.redundant_copy_groups()[: settings._DEBUG_REDUNDANT_COPY_STATS_MAX_GROUPS]
            if not groups:
                continue
            param_delta = 0.0
            grad_delta = 0.0
            grad_groups = 0
            global_counts = self._debug_global_accumulated_counts(candidate) if include_grads else None
            worst_grad_logical = -1
            worst_grad_slots: tuple[int, ...] = ()
            worst_grad_counts: tuple[float, ...] = ()
            for _logical_expert, slots in groups:
                for _param_name, param in candidate.named_expert_parameters():
                    param_delta = max(param_delta, self._debug_copy_group_stat_delta(param, candidate, slots))
                    grad = getattr(param, "grad", None)
                    if not include_grads or not torch.is_tensor(grad):
                        continue
                    if tuple(_local_tensor_view(grad).shape) != tuple(_local_tensor_view(param).shape):
                        continue
                    group_grad_delta = self._debug_copy_group_stat_delta(grad, candidate, slots)
                    grad_delta = max(grad_delta, group_grad_delta)
                    grad_groups += 1
                    if group_grad_delta >= grad_delta and global_counts is not None:
                        worst_grad_logical = int(_logical_expert)
                        worst_grad_slots = tuple(int(slot) for slot in slots)
                        worst_grad_counts = tuple(float(global_counts[int(slot)].item()) for slot in slots)
            if self.ep_rank == 0:
                logger.warning(
                    "HierMoE redundant copy stats phase=%s layer=%s groups=%s "
                    "param_stat_delta=%.6g grad_stat_delta=%.6g grad_groups=%s "
                    "worst_grad_logical=%s worst_grad_slots=%s worst_grad_counts=%s",
                    phase,
                    key,
                    len(groups),
                    param_delta,
                    grad_delta,
                    grad_groups,
                    worst_grad_logical,
                    worst_grad_slots,
                    worst_grad_counts,
                )

    def register_model(self, model: nn.Module) -> None:
        matched_layers = 0
        for key, module in model.named_modules():
            if self._is_expert_module(module):
                self.register_layer(key, module)
                matched_layers += 1
        if matched_layers == 0:
            raise RuntimeError(
                "PlaceMoE did not find a supported expert module in the model. Standard stacked gate_up_proj/down_proj "
                "and gate_proj/up_proj/down_proj experts are detected automatically; register a MoEModelAdapter for "
                "another representation."
            )
        self._normalize_ablation_layer_keys()
        if settings._FIXED_R2_LAYOUT:
            self.install_fixed_r2_layout()
        if self._initial_layout_path or self._ablation_replay_mode == "static":
            self._install_static_ablation_layout()
        if self._pending_state is not None:
            self.load_state_dict(self._pending_state)
            self._pending_state = None
        self._configure_hot_update_training_affinity()

    @torch.no_grad()
    def install_fixed_r2_layout(self) -> None:
        """Install the static two-copy experiment layout before the first forward."""

        if self.ep_size <= 1 or self.ep_size % 2 != 0:
            raise ValueError(f"Fixed R2 requires a positive even EP size, got {self.ep_size}.")
        half_ep = self.ep_size // 2
        incompatible_groups = tuple(
            int(group_size)
            for group_size in self.hierarchy.group_sizes
            if 1 < int(group_size) < self.ep_size and half_ep % int(group_size) != 0
        )
        if incompatible_groups:
            raise ValueError(
                f"Fixed R2 requires every proper hierarchy group size to divide half EP={half_ep}, "
                f"got {incompatible_groups}."
            )
        for key, layer in self.layers.items():
            if not layer.slot_layout_enabled or layer.slot_to_logical is None:
                raise ValueError(f"Fixed R2 requires reserved redundant slots for layer {key}.")
            if layer.num_experts % half_ep != 0:
                raise ValueError(f"Fixed R2 requires num_experts={layer.num_experts} divisible by half EP={half_ep}.")
            expected_slots_per_rank = layer.num_experts // half_ep
            if layer.num_local_experts != expected_slots_per_rank:
                raise ValueError(
                    f"Fixed R2 layer {key} requires {expected_slots_per_rank} slots per rank, "
                    f"got {layer.num_local_experts}."
                )

            logical = torch.arange(layer.num_experts, dtype=torch.long)
            rank_in_half = torch.div(logical, layer.num_local_experts, rounding_mode="floor")
            local_slot = torch.remainder(logical, layer.num_local_experts)
            first_slots = rank_in_half * layer.num_local_experts + local_slot
            second_slots = (half_ep + rank_in_half) * layer.num_local_experts + local_slot
            target_layout = torch.full((layer.num_physical_slots,), -1, dtype=torch.long)
            target_layout[first_slots] = logical
            target_layout[second_slots] = logical

            current_layout = layer.slot_to_logical.detach().cpu()
            if torch.equal(current_layout, target_layout):
                self._refresh_layer_mapping_from_slots(layer, tuple(int(slot) for slot in first_slots.tolist()))
                layer.fixed_r2_layout = True
                continue

            state_tensors = (
                list(layer.expert_parameters) if self.optimizer is None else self._slot_op_state_tensors(layer)
            )
            grouped_entries: dict[tuple[int, int], list[_CoverTensorEntry]] = defaultdict(list)
            for dst_slot, logical_expert in enumerate(target_layout.tolist()):
                if int(current_layout[dst_slot].item()) == logical_expert:
                    continue
                source_slots = torch.nonzero(current_layout == logical_expert, as_tuple=False).flatten()
                if source_slots.numel() == 0:
                    raise RuntimeError(
                        f"Fixed R2 cannot find source state for logical expert {logical_expert} in layer {key}."
                    )
                src_slot = int(source_slots[0].item())
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

            _cover_grouped_slot_entries_atomic(
                grouped_entries,
                self.ep_rank,
                self.ep_size,
                self.ep_group,
                debug_validate=self.debug_validate,
            )
            layer.slot_to_logical = target_layout
            self._refresh_layer_mapping_from_slots(layer, tuple(int(slot) for slot in first_slots.tolist()))
            layer.active_quota_policy = ()
            layer.pending_physical_routes = None
            layer.pending_route_data_ptr = 0
            layer.fixed_r2_layout = True

        logger.info_rank0("HierMoE installed the fixed R2 layout for %s layer(s).", len(self.layers))

    def _is_expert_module(self, module: nn.Module) -> bool:
        return resolve_moe_model_adapter(module) is not None

    def register_layer(self, key: str, module: nn.Module) -> None:
        adapter = resolve_moe_model_adapter(module)
        if adapter is None:
            raise TypeError(f"No PlaceMoE model adapter supports expert layer {key!r}.")
        named_parameters = adapter.expert_parameters(module)
        if not named_parameters:
            raise ValueError(f"PlaceMoE model adapter {adapter.name!r} exposes no parameters for {key!r}.")
        local_parameters = tuple(_local_tensor_view(item.parameter) for item in named_parameters)
        num_local_experts = int(local_parameters[0].shape[0])
        if any(int(parameter.shape[0]) != num_local_experts for parameter in local_parameters):
            shapes = {
                item.name: tuple(parameter.shape)
                for item, parameter in zip(named_parameters, local_parameters, strict=True)
            }
            raise ValueError(f"PlaceMoE expert parameters for {key!r} have inconsistent slot dimensions: {shapes}.")

        num_experts = adapter.num_experts(module)
        if num_experts % self.ep_size != 0:
            raise ValueError(
                f"HierMoE layer {key} has {num_local_experts=} and {num_experts=} with ep_size={self.ep_size}."
            )
        base_num_local_experts = num_experts // self.ep_size
        if num_local_experts not in {
            base_num_local_experts,
            base_num_local_experts + self.redundant_slot_increment_per_device,
        }:
            raise ValueError(
                f"HierMoE layer {key} has {num_local_experts=} and {num_experts=} with ep_size={self.ep_size}."
            )
        slot_layout_enabled = (
            self.redundant_slot_increment_per_device > 0 and num_local_experts > base_num_local_experts
        )
        canonical_slots = (
            _canonical_physical_slots(num_experts, base_num_local_experts, num_local_experts)
            if slot_layout_enabled
            else None
        )
        layer = self.layers.get(key)
        if slot_layout_enabled:
            mapping = canonical_slots.clone() if layer is None else layer.logical_to_physical.detach().cpu().clone()
            slot_to_logical = (
                _initial_slot_to_logical(num_experts, base_num_local_experts, num_local_experts, self.ep_size)
                if layer is None or layer.slot_to_logical is None
                else layer.slot_to_logical.detach().cpu().clone()
            )
            is_identity = torch.equal(
                slot_to_logical,
                _initial_slot_to_logical(num_experts, base_num_local_experts, num_local_experts, self.ep_size),
            )
        else:
            mapping = (
                torch.arange(num_experts, dtype=torch.long)
                if layer is None
                else layer.logical_to_physical.detach().cpu().clone()
            )
            slot_to_logical = None
            is_identity = torch.equal(mapping, torch.arange(num_experts, dtype=torch.long))
        previous_source_lut = None if layer is None else layer.source_logical_to_physical
        registered_layer = ExpertLayerState(
            key=key,
            module_id=id(module),
            num_experts=num_experts,
            base_num_local_experts=base_num_local_experts,
            num_local_experts=num_local_experts,
            expert_parameter_names=tuple(item.name for item in named_parameters),
            expert_parameters=tuple(item.parameter for item in named_parameters),
            model_adapter=adapter,
            logical_to_physical=mapping,
            slot_to_logical=slot_to_logical,
            canonical_physical_slots=canonical_slots,
            is_identity=bool(is_identity),
        )
        if self._hot_update:
            if previous_source_lut is not None and tuple(previous_source_lut.shape) == (self.ep_size, num_experts):
                registered_layer.source_logical_to_physical = previous_source_lut.detach().cpu().clone()
            else:
                # Before the first planner result, every source rank routes to
                # the canonical owner. This gives hot updates a valid M even
                # when training starts without a precomputed PlaceMoE artifact.
                registered_layer.source_logical_to_physical = mapping.view(1, -1).expand(self.ep_size, -1).clone()
        self.layers[key] = registered_layer
        self.module_id_to_key[id(module)] = key
        for parameter in registered_layer.expert_parameters:
            self.param_id_to_key[id(parameter)] = key
        self._register_pipeline_gradient_hooks(self.layers[key])

    def get_layer_key(self, module: nn.Module) -> str | None:
        return self.module_id_to_key.get(id(module))

    def get_layer_key_from_params(self, *params: torch.Tensor | None) -> str | None:
        for param in params:
            if param is None:
                continue
            key = self.param_id_to_key.get(id(param))
            if key is not None:
                return key
        return None

    def has_layer(self, layer_key: str) -> bool:
        return layer_key in self.layers

    @staticmethod
    def _uses_compact_identity_dispatch(layer: ExpertLayerState) -> bool:
        return layer.slot_layout_enabled and layer.is_identity and not layer.redundant_copy_groups()

    @classmethod
    def _routes_for_cost_model_planner(
        cls,
        layer: ExpertLayerState,
        physical_routes: torch.Tensor,
    ) -> torch.Tensor:
        """Encode compact identity routes with the expanded planner slot stride."""

        if not cls._uses_compact_identity_dispatch(layer):
            return physical_routes
        rank_offsets = torch.div(
            physical_routes,
            layer.base_num_local_experts,
            rounding_mode="floor",
        ) * (layer.num_local_experts - layer.base_num_local_experts)
        return physical_routes + rank_offsets

    @staticmethod
    def _validate_checkpoint_replay(
        layer: ExpertLayerState,
        selected_experts: torch.Tensor,
        planned_routes: torch.Tensor,
    ) -> torch.Tensor:
        if RoutingMixin._uses_compact_identity_dispatch(layer):
            return selected_experts
        selected = selected_experts.to(torch.long)
        owner = layer.mapping_for_device(selected.device).index_select(0, selected.reshape(-1)).view_as(selected)
        if not layer.slot_layout_enabled:
            return owner
        copy_slots, copy_mask = layer.copy_slots_for_device(selected.device)
        selected_copy_slots = copy_slots.index_select(0, selected.reshape(-1))
        selected_copy_mask = copy_mask.index_select(0, selected.reshape(-1))
        planned = planned_routes.to(dtype=torch.long)
        valid = ((selected_copy_slots == planned.reshape(-1, 1)) & selected_copy_mask).any(dim=-1)
        return torch.where(valid.view_as(selected), planned, owner)

    def map_logical_to_physical(
        self,
        layer_key: str,
        selected_experts: torch.Tensor,
        *,
        checkpoint_recompute: bool = False,
        checkpoint_replay: Any | None = None,
    ) -> torch.Tensor:
        layer = self.layers.get(layer_key)
        if layer is None:
            return selected_experts
        if checkpoint_recompute and checkpoint_replay is not None:
            replay = checkpoint_replay.next(layer_key)
            if replay is None:
                raise RuntimeError(f"Checkpoint route replay is missing an occurrence for {layer_key}.")
            if replay.shape != selected_experts.shape:
                raise RuntimeError(
                    f"Checkpoint route replay for {layer_key} does not match recompute input: "
                    f"replay_shape={tuple(replay.shape)}, input_shape={tuple(selected_experts.shape)}."
                )
            replay = replay.to(device=selected_experts.device, dtype=torch.long)
            return self._validate_checkpoint_replay(layer, selected_experts, replay)
        pending = layer.pending_physical_routes
        if (
            pending is not None
            and pending.device == selected_experts.device
            and pending.shape == selected_experts.shape
            and layer.pending_route_data_ptr == selected_experts.data_ptr()
        ):
            layer.pending_physical_routes = None
            layer.pending_route_data_ptr = 0
            dispatched_routes = selected_experts if self._uses_compact_identity_dispatch(layer) else pending
            if checkpoint_replay is not None and not checkpoint_recompute:
                checkpoint_replay.record(layer_key, dispatched_routes)
            return dispatched_routes
        if layer.slot_layout_enabled:
            dispatched_routes = (
                selected_experts
                if self._uses_compact_identity_dispatch(layer)
                else self._map_logical_to_slot(layer, selected_experts)
            )
        elif layer.is_identity:
            dispatched_routes = selected_experts
        else:
            mapping = layer.mapping_for_device(selected_experts.device)
            dispatched_routes = mapping.index_select(0, selected_experts.reshape(-1)).view_as(selected_experts)
        if checkpoint_replay is not None and not checkpoint_recompute:
            checkpoint_replay.record(layer_key, dispatched_routes)
        return dispatched_routes

    def num_physical_slots(self, layer_key: str, fallback_num_experts: int) -> int:
        layer = self.layers.get(layer_key)
        if layer is None or not layer.slot_layout_enabled:
            return int(fallback_num_experts)
        if self._uses_compact_identity_dispatch(layer):
            return int(fallback_num_experts)
        return int(layer.num_physical_slots)

    def _map_logical_to_slot(self, layer: ExpertLayerState, selected_experts: torch.Tensor) -> torch.Tensor:
        original_ndim = selected_experts.ndim
        selected = selected_experts.to(torch.long)
        if selected.ndim == 1:
            selected = selected.unsqueeze(-1)

        mapping = layer.mapping_for_device(selected.device)
        chosen = mapping.index_select(0, selected.reshape(-1)).view_as(selected)
        redundant_groups = layer.redundant_copy_groups_for_device(selected.device)
        if not redundant_groups:
            return chosen.squeeze(-1) if original_ndim == 1 else chosen

        return self._map_logical_to_slot_dedup_aware(layer, selected, chosen, redundant_groups, original_ndim)

    def _map_logical_to_slot_dedup_aware(
        self,
        layer: ExpertLayerState,
        selected: torch.Tensor,
        chosen: torch.Tensor,
        redundant_groups: tuple[tuple[int, torch.Tensor], ...],
        original_ndim: int,
    ) -> torch.Tensor:
        del chosen, redundant_groups
        if layer.slot_to_logical is None:
            raise RuntimeError(f"HierMoE layer {layer.key} has no physical slot layout.")
        # A source-conditioned LUT is the authoritative routing policy emitted
        # by PlaceMoE. Static preload and hot-update paths install it without
        # enabling Forward-reuse Cover, so gating it on that optimization
        # silently replaces the scored mapping with the generic greedy mapper.
        if layer.source_logical_to_physical is not None:
            mapping = layer.source_mapping_for_device(selected.device, self.ep_rank)
            physical = mapping.index_select(0, selected.reshape(-1)).view_as(selected)
            return physical.squeeze(-1) if original_ndim == 1 else physical
        if layer.active_quota_policy:
            mapping = assign_tokens_to_copies_with_quota(
                selected,
                layer.slot_to_logical,
                slots_per_rank=layer.num_local_experts,
                source_ranks=self.ep_rank,
                hierarchy=self.hierarchy,
                owner_slots=layer.logical_to_physical,
                quota_policy=layer.active_quota_policy,
                step=max(0, int(layer.latest_route_step)),
                layer_seed=zlib.crc32(layer.key.encode("utf-8")),
            )
            physical = mapping.physical_slots
            return physical.squeeze(-1) if original_ndim == 1 else physical
        copy_slots, copy_mask = layer.copy_slots_for_device(selected.device)
        if layer.fixed_r2_layout and settings._FORCE_FIXED_R2_MIRRORED_REMAP:
            physical = assign_tokens_to_mirrored_r2(
                selected,
                copy_slots,
                source_ranks=self.ep_rank,
                num_ranks=self.ep_size,
            )
            return physical.squeeze(-1) if original_ndim == 1 else physical
        if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
            physical = assign_tokens_to_copies_greedy(
                selected,
                layer.slot_to_logical,
                slots_per_rank=layer.num_local_experts,
                source_ranks=self.ep_rank,
                hierarchy_group_sizes=self.hierarchy.group_sizes,
                num_experts=layer.num_experts,
                step=max(0, int(layer.latest_route_step)),
                layer_seed=zlib.crc32(layer.key.encode("utf-8")),
                max_copies=self.greedy_max_copies_per_expert,
            )
            return physical.squeeze(-1) if original_ndim == 1 else physical
        if layer.fixed_r2_layout:
            physical = assign_tokens_to_mirrored_r2(
                selected,
                copy_slots,
                source_ranks=self.ep_rank,
                num_ranks=self.ep_size,
            )
            return physical.squeeze(-1) if original_ndim == 1 else physical
        physical = assign_tokens_to_copies(
            selected,
            layer.slot_to_logical,
            slots_per_rank=layer.num_local_experts,
            source_ranks=self.ep_rank,
            hierarchy_group_sizes=self.hierarchy.group_sizes,
            owner_slots=layer.logical_to_physical,
            step=max(0, int(layer.latest_route_step)),
            layer_seed=zlib.crc32(layer.key.encode("utf-8")),
            copy_slots=copy_slots,
            copy_mask=copy_mask,
            validate_copy_table=False,
        )
        return physical.squeeze(-1) if original_ndim == 1 else physical

    def record_routing(
        self,
        *,
        layer_key: str,
        selected_experts: torch.Tensor,
        hidden_size: int,
        bytes_per_element: int,
        step: int | None = None,
    ) -> None:
        layer = self.layers.get(layer_key)
        if layer is None:
            return
        layer.latest_selected_experts = selected_experts.detach()
        if step is not None:
            layer.latest_route_step = int(step)
        layer.latest_hidden_size = int(hidden_size)
        layer.latest_bytes_per_element = int(bytes_per_element)
        if (
            self.fixed_pipeline_overlap
            and self._ablation_replay_mode == "off"
            and self._online_freeze_cost_mode == "off"
            and step is not None
        ):
            self._submit_pipeline_plan(layer, selected_experts, int(step))

    def record_forward_physical_routes(self, layer_key: str, physical_routes: torch.Tensor) -> None:
        """Keep the physical routes already consumed by the trainable Forward."""

        if not self._cost_model_verify and self._online_freeze_cost_mode == "off":
            return
        layer = self.layers.get(layer_key)
        if layer is not None:
            layer.latest_physical_routes = physical_routes.detach()

    def mark_route_step(self, layer_key: str, step: int) -> None:
        layer = self.layers.get(layer_key)
        if layer is not None:
            layer.latest_route_step = int(step)
