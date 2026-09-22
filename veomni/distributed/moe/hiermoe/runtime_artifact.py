"""Install canonical PlaceMoE metadata after weights have been preloaded."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .placemoe.artifacts import validate_placemoe_artifact
from .placemoe.types import LayerPlan
from .runtime_settings import logger


class ArtifactMixin:
    """Load schema-2 artifacts without historical action replay or migration."""

    def _load_initial_artifact(self, path: str) -> dict[str, LayerPlan]:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot load PlaceMoE initial artifact from {path!r}.") from error
        plans = validate_placemoe_artifact(payload)
        if payload["source"].get("initial_layout") != "preloaded":
            raise ValueError("PlaceMoE initial artifacts must use a preloaded layout.")
        if int(payload["topology"]["ep_size"]) != self.ep_size:
            raise ValueError("PlaceMoE initial artifact EP size does not match the current run.")
        return plans

    def _normalize_initial_layer_keys(self) -> None:
        """Resolve checkpoint-wrapper prefixes without accepting ambiguous layers."""
        normalized: dict[str, LayerPlan] = {}
        for artifact_key, plan in self._initial_plans.items():
            model_key = artifact_key
            if model_key not in self.layers:
                marker = ".layers."
                suffix = artifact_key[artifact_key.index(marker) :] if marker in artifact_key else ""
                matches = [key for key in self.layers if suffix and key.endswith(suffix)]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"PlaceMoE artifact layer {artifact_key!r} has no unambiguous model layer; matches={matches}."
                    )
                model_key = matches[0]
            if model_key in normalized:
                raise RuntimeError(f"Multiple PlaceMoE artifact layers resolve to {model_key!r}.")
            normalized[model_key] = plan
        self._initial_plans = normalized

    @torch.no_grad()
    def _install_initial_layout(self) -> None:
        if set(self._initial_plans) != set(self.layers):
            missing = sorted(set(self.layers) - set(self._initial_plans))
            unexpected = sorted(set(self._initial_plans) - set(self.layers))
            raise RuntimeError(f"PlaceMoE artifact layer mismatch: missing={missing}, unexpected={unexpected}.")
        # Validate all metadata first. ParallelPlan already preloaded weights,
        # so installing an initial artifact must not send expert tensors again.
        prepared = {}
        for key, layer in self.layers.items():
            plan = self._initial_plans[key]
            layout, owners = self._validate_placement_layout(
                layer, plan.slot_to_logical.tolist(), plan.owner_slots.tolist()
            )
            source_lut = torch.tensor(plan.source_logical_to_physical, dtype=torch.long)
            if tuple(source_lut.shape) != (self.ep_size, layer.num_experts):
                raise RuntimeError(f"PlaceMoE source mapping shape does not match layer {key!r}.")
            prepared[key] = (layout, owners, source_lut)
        for key, layer in self.layers.items():
            layout, owners, source_lut = prepared[key]
            layer.slot_to_logical = layout
            layer.active_quota_policy = ()
            layer.pending_physical_routes = None
            layer.pending_route_data_ptr = 0
            layer.placement_version += 1
            self._refresh_layer_mapping_from_slots(layer, owners)
            layer.source_logical_to_physical = source_lut
            layer._device_source_mapping_cache.clear()
        logger.info_rank0("PlaceMoE installed preloaded placement metadata from %s.", self._initial_layout_path)
