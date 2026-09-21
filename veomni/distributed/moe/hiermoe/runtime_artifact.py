"""Initial artifact loading, layer-key matching, and atomic static layout installation."""

from __future__ import annotations

import json
import zlib
from collections import defaultdict
from typing import Any, Sequence

import torch

from ....utils.device import synchronize
from . import runtime_settings as settings
from .planner import PlacementAction, PlacementCost, PlacementPlan
from .runtime_settings import logger
from .runtime_types import (
    ExpertLayerState,
    _initial_slot_to_logical,
    _PendingPipelinePlan,
)


class ArtifactMixin:
    """Install the serialized layout using the established migration primitives."""

    def _load_ablation_replay(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot load HierMoE ablation replay from {path!r}.") from error

        topology = payload.get("topology", {})
        replay_ep_size = int(topology.get("ep_size", -1))
        if replay_ep_size != self.ep_size:
            raise ValueError(f"HierMoE ablation replay uses ep_size={replay_ep_size}, current ep_size={self.ep_size}.")
        raw_steps = payload.get("replay", {}).get("actions_by_step", {})
        if not isinstance(raw_steps, dict) or not raw_steps:
            raise ValueError("HierMoE ablation replay contains no actions_by_step.")

        actions_by_step: dict[int, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
        for raw_step, rows in raw_steps.items():
            step = int(raw_step)
            if not isinstance(rows, list):
                raise ValueError(f"HierMoE ablation replay step {step} is not a list.")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"HierMoE ablation replay step {step} contains a non-mapping action.")
                layer = str(row.get("layer", ""))
                kind = str(row.get("kind", ""))
                body = str(row.get("body", ""))
                if not layer or kind not in {"swap", "replica", "empty"} or not body:
                    raise ValueError(f"HierMoE ablation replay step {step} contains an invalid action: {row!r}.")
                actions_by_step[step][layer].append((kind, body))
        self._ablation_actions_by_step = {
            step: {layer: tuple(actions) for layer, actions in by_layer.items()}
            for step, by_layer in actions_by_step.items()
        }

        raw_layers = payload.get("layers", {})
        if not isinstance(raw_layers, dict) or not raw_layers:
            raise ValueError("HierMoE ablation replay contains no final layer layouts.")
        expected_layouts: dict[str, tuple[int, ...]] = {}
        expected_owner_slots: dict[str, tuple[int, ...]] = {}
        expected_source_luts: dict[str, tuple[tuple[int, ...], ...]] = {}
        for raw_layer, raw_layer_payload in raw_layers.items():
            layer = str(raw_layer)
            if not isinstance(raw_layer_payload, dict):
                raise ValueError(f"HierMoE ablation layer {layer!r} is not a mapping.")
            expected_layouts[layer] = tuple(int(value) for value in raw_layer_payload["slot_to_logical"])
            raw_owners = raw_layer_payload.get("owner_slots")
            if raw_owners is not None:
                expected_owner_slots[layer] = tuple(int(value) for value in raw_owners)
            raw_source_lut = raw_layer_payload.get("source_logical_to_physical")
            if raw_source_lut is not None:
                expected_source_luts[layer] = tuple(tuple(int(value) for value in row) for row in raw_source_lut)
        self._ablation_expected_layouts = expected_layouts
        self._ablation_expected_owner_slots = expected_owner_slots
        self._ablation_expected_source_luts = expected_source_luts
        self._ablation_initial_layout = str(payload.get("source", {}).get("initial_layout", "fixed_r2"))

    def _normalize_ablation_layer_keys(self) -> None:
        """Match replay keys to the model after checkpoint wrapper conversion."""

        if not self._ablation_expected_layouts:
            return

        replay_keys = set(self._ablation_expected_layouts)
        replay_keys.update(self._ablation_expected_owner_slots)
        replay_keys.update(self._ablation_expected_source_luts)
        for layers_by_step in self._ablation_actions_by_step.values():
            replay_keys.update(layers_by_step)

        key_mapping: dict[str, str] = {}
        for replay_key in replay_keys:
            if replay_key in self.layers:
                key_mapping[replay_key] = replay_key
                continue
            marker = ".layers."
            suffix = replay_key[replay_key.index(marker) :] if marker in replay_key else ""
            matches = [model_key for model_key in self.layers if suffix and model_key.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"HierMoE ablation layer {replay_key!r} has no unambiguous model layer; "
                    f"suffix={suffix!r}, matches={matches}."
                )
            key_mapping[replay_key] = matches[0]

        def remap_values(values: dict[str, Any]) -> dict[str, Any]:
            remapped: dict[str, Any] = {}
            for replay_key, value in values.items():
                model_key = key_mapping[replay_key]
                if model_key in remapped:
                    raise RuntimeError(f"Multiple HierMoE replay layers resolve to model layer {model_key!r}.")
                remapped[model_key] = value
            return remapped

        self._ablation_expected_layouts = remap_values(self._ablation_expected_layouts)
        self._ablation_expected_owner_slots = remap_values(self._ablation_expected_owner_slots)
        self._ablation_expected_source_luts = remap_values(self._ablation_expected_source_luts)
        self._ablation_actions_by_step = {
            step: remap_values(layers_by_step) for step, layers_by_step in self._ablation_actions_by_step.items()
        }

    @staticmethod
    def _zero_placement_cost() -> PlacementCost:
        return PlacementCost(
            communication=0.0,
            compute=0.0,
            communication_model_units=0.0,
            peak_communication_rank=-1,
            peak_compute_rank=-1,
            selected_dim=0,
        )

    def _build_ablation_replay_plan(
        self,
        layer: ExpertLayerState,
        specs: Sequence[tuple[str, str]],
    ) -> PlacementPlan:
        initial = self._layer_layout(layer)
        working = initial.clone()
        owners = layer.logical_to_physical.detach().cpu().clone()
        actions: list[PlacementAction] = []
        for kind, body in specs:
            if kind == "swap":
                lhs_text, rhs_text = body.split("<->", maxsplit=1)
                lhs, rhs = int(lhs_text), int(rhs_text)
                lhs_slot, rhs_slot = int(owners[lhs].item()), int(owners[rhs].item())
                if int(working[lhs_slot].item()) != lhs or int(working[rhs_slot].item()) != rhs:
                    raise RuntimeError(f"HierMoE ablation swap {body} does not match layer {layer.key}.")
                actions.append(PlacementAction("swap", lhs_slot, rhs_slot, lhs, rhs))
                working[lhs_slot], working[rhs_slot] = working[rhs_slot].clone(), working[lhs_slot].clone()
                owners[lhs], owners[rhs] = owners[rhs].clone(), owners[lhs].clone()
                continue
            if kind == "replica":
                logical_text, dst_text = body.split("->", maxsplit=1)
                logical, dst_slot = int(logical_text), int(dst_text)
                src_slot = int(owners[logical].item())
                previous = int(working[dst_slot].item())
                if src_slot == dst_slot or int(working[src_slot].item()) != logical:
                    raise RuntimeError(f"HierMoE ablation replica {body} has no valid owner source in {layer.key}.")
                actions.append(PlacementAction("replica", src_slot, dst_slot, logical, previous))
                working[dst_slot] = logical
                if previous >= 0 and int(owners[previous].item()) == dst_slot:
                    remaining = torch.nonzero(working == previous, as_tuple=False).flatten()
                    if remaining.numel() == 0:
                        raise RuntimeError(
                            f"HierMoE ablation replica {body} removes the final copy of victim "
                            f"expert {previous} in {layer.key}."
                        )
                    owners[previous] = int(remaining.min().item())
                continue
            logical_text, slot_text = body.split("@", maxsplit=1)
            logical, slot = int(logical_text), int(slot_text)
            if int(working[slot].item()) != logical or slot in {int(value) for value in owners.tolist()}:
                raise RuntimeError(f"HierMoE ablation empty action {body} is invalid for layer {layer.key}.")
            actions.append(PlacementAction("empty", slot, slot, logical, -1))
            working[slot] = -1

        zero_cost = self._zero_placement_cost()
        final_layout = tuple(int(value) for value in working.tolist())
        return PlacementPlan(
            actions=tuple(actions),
            initial_layout=tuple(int(value) for value in initial.tolist()),
            final_layout=final_layout,
            baseline_cost=zero_cost,
            final_cost=zero_cost,
            swap_rounds=sum(action.kind == "swap" for action in actions),
            replica_rounds=sum(action.kind == "replica" for action in actions),
            planning_ms=0.0,
            route_stats_ms=0.0,
            swap_ms=0.0,
            replica_ms=0.0,
            swap_score_ms=0.0,
            swap_update_ms=0.0,
            swap_collective_ms=0.0,
            replica_score_ms=0.0,
            replica_update_ms=0.0,
            replica_collective_ms=0.0,
            decision_sync_ms=0.0,
            finalization_ms=0.0,
            algorithm_version="hiermoe-ablation-replay-v1",
            layout_digest=f"{zlib.crc32(repr(final_layout).encode()):08x}",
            final_owner_slots=tuple(int(value) for value in owners.tolist()),
        )

    def _validate_ablation_final_layout(self) -> None:
        if set(self._ablation_expected_layouts) != set(self.layers):
            missing = sorted(set(self.layers) - set(self._ablation_expected_layouts))
            unexpected = sorted(set(self._ablation_expected_layouts) - set(self.layers))
            raise RuntimeError(f"HierMoE ablation layer mismatch: missing={missing}, unexpected={unexpected}.")
        for layer_key, layer in self.layers.items():
            actual = tuple(int(value) for value in self._layer_layout(layer).tolist())
            expected = self._ablation_expected_layouts[layer_key]
            if actual != expected:
                raise RuntimeError(f"HierMoE ablation final layout does not match replay output for {layer_key}.")
            expected_owners = self._ablation_expected_owner_slots.get(layer_key)
            if expected_owners is not None:
                actual_owners = tuple(int(value) for value in layer.logical_to_physical.tolist())
                if actual_owners != expected_owners:
                    raise RuntimeError(f"HierMoE ablation owner mapping does not match replay output for {layer_key}.")
            expected_source_lut = self._ablation_expected_source_luts.get(layer_key)
            if expected_source_lut is not None:
                if layer.source_logical_to_physical is None:
                    raise RuntimeError(f"HierMoE ablation source route LUT is missing for {layer_key}.")
                actual_source_lut = tuple(
                    tuple(int(value) for value in row) for row in layer.source_logical_to_physical.tolist()
                )
                if actual_source_lut != expected_source_lut:
                    raise RuntimeError(
                        f"HierMoE ablation source route LUT does not match replay output for {layer_key}."
                    )

    def _install_static_ablation_route_metadata(self) -> None:
        for layer_key, layer in self.layers.items():
            expected_owners = self._ablation_expected_owner_slots.get(layer_key)
            if expected_owners is not None:
                self._refresh_layer_mapping_from_slots(layer, expected_owners)
            expected_source_lut = self._ablation_expected_source_luts.get(layer_key)
            if expected_source_lut is None:
                continue
            source_lut = torch.tensor(expected_source_lut, dtype=torch.long)
            expected_shape = (self.ep_size, layer.num_experts)
            if tuple(source_lut.shape) != expected_shape:
                raise RuntimeError(
                    f"HierMoE ablation source route LUT for {layer_key} has shape "
                    f"{tuple(source_lut.shape)}, expected {expected_shape}."
                )
            if bool(((source_lut < 0) | (source_lut >= layer.num_physical_slots)).any().item()):
                raise RuntimeError(f"HierMoE ablation source route LUT references an invalid slot for {layer_key}.")
            layout = self._layer_layout(layer)
            logical = torch.arange(layer.num_experts, dtype=torch.long).view(1, -1)
            routed_logical = layout.index_select(0, source_lut.reshape(-1)).view_as(source_lut)
            if not torch.equal(routed_logical, logical.expand_as(routed_logical)):
                raise RuntimeError(f"HierMoE ablation source route LUT references the wrong expert for {layer_key}.")
            layer.source_logical_to_physical = source_lut
            layer._device_source_mapping_cache.clear()

    @torch.no_grad()
    def _install_static_ablation_layout(self) -> None:
        if self._initial_layout_path:
            for layer_key, layer in self.layers.items():
                expected_layout = self._ablation_expected_layouts[layer_key]
                validated_layout, _owners = self._validate_placement_layout(
                    layer,
                    expected_layout,
                    self._ablation_expected_owner_slots.get(layer_key),
                )
                layer.slot_to_logical = validated_layout
                layer.fixed_r2_layout = False
                layer.active_quota_policy = ()
                layer.pending_physical_routes = None
                layer.pending_route_data_ptr = 0
                layer.placement_version += 1
            self._install_static_ablation_route_metadata()
            self._validate_ablation_final_layout()
            logger.info_rank0(
                "HierMoE installed preloaded static placement metadata from %s without expert P2P.",
                self._initial_layout_path,
            )
            return

        canonical_empty = self._ablation_initial_layout == "canonical_empty"
        if canonical_empty:
            if settings._FIXED_R2_LAYOUT:
                raise RuntimeError("Canonical-empty static replay must not install the fixed R2 layout.")
            for layer in self.layers.values():
                expected = _initial_slot_to_logical(
                    layer.num_experts,
                    layer.base_num_local_experts,
                    layer.num_local_experts,
                    self.ep_size,
                )
                if not torch.equal(self._layer_layout(layer), expected):
                    raise RuntimeError(
                        f"Canonical-empty static replay has a non-canonical initial {layer.key} layout."
                    )
        elif not settings._FIXED_R2_LAYOUT:
            raise RuntimeError("Static HierMoE ablation replay requires VEOMNI_HIERMOE_FIXED_R2_LAYOUT=1.")
        # A static replay already knows the final layout before training
        # starts.  Replaying one action wave at a time needlessly serializes
        # hundreds of expert transfers and can leave the first FSDP
        # collective queued behind minutes of device copies.  Concatenate all
        # recorded actions per layer, let ``_build_ablation_replay_plan``
        # resolve every final slot back to an original state source, and
        # materialize that layer with one sparse transfer plan.  The final
        # owner/source-LUT metadata is installed below, so static replay does
        # not need the online Cover path's incremental LUT patches.
        specs_by_layer: dict[str, list[tuple[str, str]]] = {layer_key: [] for layer_key in self.layers}
        for step in sorted(self._ablation_actions_by_step):
            by_layer = self._ablation_actions_by_step[step]
            unexpected = set(by_layer) - set(self.layers)
            if unexpected:
                raise RuntimeError(
                    f"HierMoE ablation replay step {step} contains unknown layers: {sorted(unexpected)}."
                )
            for layer_key in self.layers:
                specs_by_layer[layer_key].extend(by_layer.get(layer_key, ()))

        action_count = 0
        for layer_key, layer in self.layers.items():
            plan = self._build_ablation_replay_plan(layer, specs_by_layer[layer_key])
            self._execute_placement_plan(
                layer,
                plan,
                timing_prefix=None,
                transfer_group=self.ep_group,
                force_staged_transfer=False,
                fast_sparse_transfer=True,
            )
            # ``ProcessGroupHCCL::Work.wait`` guarantees that the P2P work was
            # submitted, but the destination-slot copies consuming the shared
            # receive staging buffer may still be queued on the default
            # stream.  Static installation immediately reuses that staging
            # buffer for the next layer; without a device fence dozens of
            # recv/copy waves can accumulate ahead of the first FSDP
            # collective and eventually hit HCCL's dispatch timeout.  This is
            # startup-only work, so finish each layer before reusing the
            # manager-wide staging pool.
            synchronize()
            action_count += len(plan.actions)
        self._install_static_ablation_route_metadata()
        self._validate_ablation_final_layout()
        logger.info_rank0(
            "HierMoE installed static ablation layout from %s using %s replayed actions.",
            settings._ABLATION_REPLAY_PATH,
            action_count,
        )

    def _queue_ablation_replay_step(self, step: int) -> str:
        if self._ablation_replay_mode == "static":
            self.latest_pair = "none"
            return self.latest_pair
        # ``maybe_swap`` receives the zero-based optimizer-step index while
        # action logs use the one-based training-step number shown in metrics.
        logged_step = int(step) + 1
        by_layer = self._ablation_actions_by_step.get(logged_step)
        if by_layer is None:
            if logged_step > max(self._ablation_actions_by_step):
                self._validate_ablation_final_layout()
            self.latest_pair = "none"
            return self.latest_pair
        if set(by_layer) != set(self.layers):
            raise RuntimeError(f"HierMoE ablation replay step {logged_step} does not contain every registered layer.")

        committed: list[str] = []
        for layer_key in self.layers:
            if layer_key in self._pipeline_pending_plans:
                raise RuntimeError(f"HierMoE ablation replay has an unconsumed plan for {layer_key}.")
            layer = self.layers[layer_key]
            plan = self._build_ablation_replay_plan(layer, by_layer[layer_key])
            self._pipeline_pending_plans[layer_key] = _PendingPipelinePlan(
                plan=plan,
                source_step=int(step),
                placement_version=int(layer.placement_version),
            )
            committed.extend(f"{layer_key}:{action.format()}" for action in plan.actions)
        self._accumulate_metric("hiermoe/ablation_replay_actions", len(committed))
        self._accumulate_metric("hiermoe/ablation_replay_logged_step", logged_step)
        self.latest_pair = ",".join(committed) if committed else "none"
        return self.latest_pair
