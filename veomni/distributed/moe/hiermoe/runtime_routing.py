"""Expert registration and logical-to-physical dispatch mapping."""

from __future__ import annotations

import zlib
from typing import Any

import torch
from torch import nn

from placemoe.model_adapter import resolve_moe_model_adapter

from .core_planner import assign_tokens_to_copies_with_quota
from .greedy_planner import assign_tokens_to_copies_greedy
from .planner import assign_tokens_to_copies
from .runtime_tensors import _local_tensor_view
from .runtime_types import ExpertLayerState, _canonical_physical_slots, _initial_slot_to_logical


class RoutingMixin:
    """Expert registration and logical-to-physical dispatch mapping."""

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
        self._normalize_initial_layer_keys()
        if self._initial_layout_path:
            self._install_initial_layout()
        if self._pending_state is not None:
            self.load_state_dict(self._pending_state)
            self._pending_state = None
        self._configure_hot_update_training_affinity()

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

    def record_forward_physical_routes(self, layer_key: str, physical_routes: torch.Tensor) -> None:
        """Keep the physical routes already consumed by the trainable Forward."""

        if not self._cost_model_verify:
            return
        layer = self.layers.get(layer_key)
        if layer is not None:
            layer.latest_physical_routes = physical_routes.detach()

    def mark_route_step(self, layer_key: str, step: int) -> None:
        layer = self.layers.get(layer_key)
        if layer is not None:
            layer.latest_route_step = int(step)

    @staticmethod
    def _layer_layout(layer: ExpertLayerState) -> torch.Tensor:
        if layer.slot_to_logical is not None:
            return layer.slot_to_logical.detach().cpu().clone()
        layout = torch.full((layer.num_experts,), -1, dtype=torch.long)
        logical = torch.arange(layer.num_experts, dtype=torch.long)
        layout.scatter_(0, layer.logical_to_physical.to(torch.long), logical)
        return layout
