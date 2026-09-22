"""Checkpoint serialization and restoration of physical expert placement."""

from __future__ import annotations

import zlib
from typing import Any

import torch

from .core_planner import CORE_MOE_ALGORITHM_VERSION, QuotaPolicyEntry
from .runtime_settings import logger
from .runtime_types import _initial_slot_to_logical


class CheckpointMixin:
    """Checkpoint serialization and restoration of physical expert placement."""

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 3,
            "ep_size": self.ep_size,
            "layers": {
                key: (
                    {
                        "num_experts": layer.num_experts,
                        "base_num_local_experts": layer.base_num_local_experts,
                        "num_local_experts": layer.num_local_experts,
                        "logical_to_physical": layer.logical_to_physical.tolist(),
                        "slot_to_logical": layer.slot_to_logical.tolist(),
                        "quota_algorithm_version": CORE_MOE_ALGORITHM_VERSION,
                        "quota_layout_crc32": zlib.crc32(repr(tuple(layer.slot_to_logical.tolist())).encode()),
                        "quota_policy": [entry.as_tuple() for entry in layer.active_quota_policy],
                    }
                    if layer.slot_to_logical is not None
                    else {
                        "num_experts": layer.num_experts,
                        "logical_to_physical": layer.logical_to_physical.tolist(),
                    }
                )
                for key, layer in sorted(self.layers.items())
            },
        }

    def has_non_identity_placement(self) -> bool:
        if self._pending_state:
            for payload in self._pending_state.get("layers", {}).values():
                mapping = payload.get("logical_to_physical", [])
                if list(mapping) != list(range(len(mapping))):
                    return True
        for layer in self.layers.values():
            if layer.slot_to_logical is not None:
                expected = _initial_slot_to_logical(
                    layer.num_experts,
                    layer.base_num_local_experts,
                    layer.num_local_experts,
                    self.ep_size,
                )
                if not torch.equal(layer.slot_to_logical.cpu(), expected):
                    return True
                continue
            identity = torch.arange(layer.num_experts, dtype=torch.long)
            if not torch.equal(layer.logical_to_physical.cpu(), identity):
                return True
        return False

    def load_state_dict(self, state_dict: dict[str, Any] | None) -> None:
        if not state_dict:
            return
        if int(state_dict.get("ep_size", self.ep_size)) != self.ep_size:
            raise ValueError(
                f"HierMoE checkpoint placement was saved with ep_size={state_dict.get('ep_size')}, "
                f"but the current run uses ep_size={self.ep_size}."
            )
        if not self.layers:
            self._pending_state = state_dict
            return
        staged: dict[str, tuple[torch.Tensor | None, torch.Tensor, tuple[QuotaPolicyEntry, ...]]] = {}
        incompatible_quota_key: str | None = None
        for key, payload in state_dict.get("layers", {}).items():
            layer = self.layers.get(key)
            if layer is None:
                raise RuntimeError(
                    f"Checkpoint contains HierMoE placement for unknown layer {key!r}. "
                    "Loading without that placement would change logical expert semantics."
                )
            if payload.get("slot_to_logical") is not None:
                slot_to_logical = torch.tensor(payload["slot_to_logical"], dtype=torch.long)
                if layer.slot_to_logical is None:
                    raise RuntimeError(
                        f"Checkpoint contains HierMoE slot layout for {key}, but current layer has no redundant slots."
                    )
                slot_to_logical, _ = self._validate_placement_layout(layer, slot_to_logical)
                checkpoint_owners = payload.get("logical_to_physical")
                if checkpoint_owners is None:
                    derived_owners: list[int] = []
                    for logical_expert in range(layer.num_experts):
                        slots = torch.nonzero(slot_to_logical == logical_expert, as_tuple=False).flatten().tolist()
                        canonical = (
                            -1
                            if layer.canonical_physical_slots is None
                            else int(layer.canonical_physical_slots[logical_expert].item())
                        )
                        derived_owners.append(canonical if canonical in slots else int(slots[0]))
                    checkpoint_owners = derived_owners
                slot_to_logical, owner_mapping = self._validate_placement_layout(
                    layer,
                    slot_to_logical,
                    checkpoint_owners,
                )
                assert owner_mapping is not None
                policy_version = payload.get("quota_algorithm_version")
                raw_policy = payload.get("quota_policy", ())
                if raw_policy:
                    raise ValueError("Historical quota checkpoints were removed; use a PlaceMoE checkpoint.")
                if raw_policy and policy_version != CORE_MOE_ALGORITHM_VERSION:
                    incompatible_quota_key = incompatible_quota_key or key
                    quota_policy = ()
                else:
                    quota_policy = tuple(QuotaPolicyEntry.from_tuple(row) for row in raw_policy)
                expected_crc = payload.get("quota_layout_crc32")
                actual_crc = zlib.crc32(repr(tuple(slot_to_logical.tolist())).encode())
                if expected_crc is not None and int(expected_crc) != actual_crc:
                    raise ValueError(f"HierMoE checkpoint quota policy for {key} does not match its slot layout.")
                quota_policy = self._validate_quota_policy(layer, slot_to_logical, quota_policy)
                staged[key] = (slot_to_logical, owner_mapping, quota_policy)
                continue

            if layer.slot_to_logical is not None:
                raise RuntimeError(
                    f"Checkpoint contains a compact HierMoE layout for {key}, "
                    "but the current layer reserves redundant slots."
                )
            mapping = torch.tensor(payload["logical_to_physical"], dtype=torch.long)
            if mapping.numel() != layer.num_experts:
                raise ValueError(
                    f"HierMoE checkpoint placement for {key} has {mapping.numel()} entries, "
                    f"expected {layer.num_experts}."
                )
            if sorted(mapping.tolist()) != list(range(layer.num_experts)):
                raise ValueError(f"HierMoE checkpoint placement for {key} is not a valid permutation.")
            staged[key] = (None, mapping, ())

        snapshots = {
            key: (
                None if self.layers[key].slot_to_logical is None else self.layers[key].slot_to_logical.clone(),
                self.layers[key].logical_to_physical.clone(),
                self.layers[key].active_quota_policy,
                self.layers[key].pending_physical_routes,
                self.layers[key].pending_route_data_ptr,
                self.layers[key].fixed_r2_layout,
            )
            for key in staged
        }
        try:
            for key, (slot_to_logical, owner_mapping, quota_policy) in staged.items():
                layer = self.layers[key]
                layer.slot_to_logical = None if slot_to_logical is None else slot_to_logical.clone()
                layer.fixed_r2_layout = False
                layer.logical_to_physical = owner_mapping.clone()
                layer.active_quota_policy = quota_policy
                layer.pending_physical_routes = None
                layer.pending_route_data_ptr = 0
                layer.refresh_identity()
                layer.invalidate_cache()
        except Exception:
            for key, (
                slot_to_logical,
                owner_mapping,
                quota_policy,
                pending_routes,
                pending_data_ptr,
                fixed_r2_layout,
            ) in snapshots.items():
                layer = self.layers[key]
                layer.slot_to_logical = slot_to_logical
                layer.logical_to_physical = owner_mapping
                layer.active_quota_policy = quota_policy
                layer.pending_physical_routes = pending_routes
                layer.pending_route_data_ptr = pending_data_ptr
                layer.fixed_r2_layout = fixed_r2_layout
                layer.refresh_identity()
                layer.invalidate_cache()
            raise
        if incompatible_quota_key is not None:
            logger.warning(
                "HierMoE checkpoint quota policy for %s uses a version other than %r; "
                "restoring expert layouts and clearing all incompatible quota policies.",
                incompatible_quota_key,
                CORE_MOE_ALGORITHM_VERSION,
            )
