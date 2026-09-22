# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture training routes as local or EP-wide snapshots for offline validation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from .topology import Hierarchy


_SNAPSHOT_FORMAT = "veomni.hiermoe.route_snapshot"
_SNAPSHOT_VERSION = 1
_LOCAL_SNAPSHOT_FORMAT = "veomni.hiermoe.local_route"
_LOCAL_SNAPSHOT_VERSION = 1
_CAPTURE_PATH_TEMPLATE = os.environ.get("VEOMNI_HIERMOE_ORACLE_CAPTURE_PATH", "").strip()
_CAPTURE_CALLS: dict[tuple[int, str], int] = {}
_CAPTURED: set[tuple[int, str, int]] = set()
_CAPTURE_LAYER_ORDINALS: dict[int, int] = {}


@dataclass(frozen=True)
class RouteSnapshot:
    """Logical top-k routes for one layer invocation on every EP rank."""

    routes_by_rank: tuple[torch.Tensor, ...]
    num_experts: int
    hidden_size: int
    bytes_per_element: int
    hierarchy: Hierarchy
    logical_to_physical: torch.Tensor
    layer_key: str
    step: int
    call_index: int
    smooth_max_gamma: float = 10.0
    selected_dim: int | None = None

    @property
    def ep_size(self) -> int:
        return len(self.routes_by_rank)

    @property
    def communication_dimension(self) -> int:
        if self.selected_dim is not None:
            return int(self.selected_dim)
        return min(2, self.hierarchy.selected_dim)

    def validate(self) -> None:
        if self.ep_size < 1:
            raise ValueError("A route snapshot must contain at least one EP rank.")
        if self.num_experts < 1 or self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts={self.num_experts} must be positive and divisible by ep_size={self.ep_size}."
            )
        if self.hierarchy.ep_size != self.ep_size:
            raise ValueError(
                f"Hierarchy EP size {self.hierarchy.ep_size} does not match snapshot EP size {self.ep_size}."
            )
        if not 1 <= self.communication_dimension <= self.hierarchy.selected_dim:
            raise ValueError(
                f"selected_dim={self.communication_dimension} must be in [1, {self.hierarchy.selected_dim}]."
            )
        if tuple(self.logical_to_physical.shape) != (self.num_experts,):
            raise ValueError("logical_to_physical must contain one physical slot per logical expert.")
        expected = torch.arange(self.num_experts, dtype=torch.long)
        actual = torch.sort(self.logical_to_physical.to(torch.long).cpu()).values
        if not torch.equal(actual, expected):
            raise ValueError("logical_to_physical must be a permutation of physical expert slots.")

        top_k: int | None = None
        for rank, routes in enumerate(self.routes_by_rank):
            if routes.ndim != 2:
                raise ValueError(f"Rank {rank} routes must be two-dimensional, got shape={tuple(routes.shape)}.")
            if top_k is None:
                top_k = int(routes.shape[1])
            elif int(routes.shape[1]) != top_k:
                raise ValueError("All EP ranks must use the same top-k width.")
            if routes.numel() == 0:
                continue
            minimum = int(routes.min().item())
            maximum = int(routes.max().item())
            if minimum < 0 or maximum >= self.num_experts:
                raise ValueError(
                    f"Rank {rank} route IDs must be in [0, {self.num_experts}), got [{minimum}, {maximum}]."
                )


def _normalize_routes(routes: torch.Tensor) -> torch.Tensor:
    routes = routes.detach().to(device="cpu", dtype=torch.long)
    if routes.ndim == 1:
        routes = routes.unsqueeze(-1)
    elif routes.ndim > 2:
        routes = routes.reshape(-1, routes.shape[-1])
    return routes.contiguous()


def save_route_snapshot(snapshot: RouteSnapshot, path: str | Path) -> Path:
    snapshot.validate()
    routes = tuple(_normalize_routes(item) for item in snapshot.routes_by_rank)
    lengths = torch.tensor([item.shape[0] for item in routes], dtype=torch.long)
    max_tokens = int(lengths.max().item())
    top_k = int(routes[0].shape[1])
    padded = torch.full((snapshot.ep_size, max_tokens, top_k), -1, dtype=torch.int32)
    for rank, item in enumerate(routes):
        padded[rank, : item.shape[0]] = item.to(torch.int32)

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": _SNAPSHOT_FORMAT,
            "version": _SNAPSHOT_VERSION,
            "routes": padded,
            "route_lengths": lengths,
            "num_experts": snapshot.num_experts,
            "hidden_size": snapshot.hidden_size,
            "bytes_per_element": snapshot.bytes_per_element,
            "ep_size": snapshot.ep_size,
            "hierarchy_group_sizes": list(snapshot.hierarchy.group_sizes),
            "hierarchy_source": snapshot.hierarchy.source,
            "logical_to_physical": snapshot.logical_to_physical.to(device="cpu", dtype=torch.long),
            "layer_key": snapshot.layer_key,
            "step": snapshot.step,
            "call_index": snapshot.call_index,
            "smooth_max_gamma": snapshot.smooth_max_gamma,
            "selected_dim": snapshot.communication_dimension,
        },
        output,
    )
    return output


def _layer_matches(layer_key: str, requested: str) -> bool:
    requested = requested.strip()
    if not requested:
        return True
    if requested.isdigit():
        match = re.search(r"(?:layers?|layer)\.(\d+)(?:\.|$)", layer_key)
        return match is not None and int(match.group(1)) == int(requested)
    return requested == layer_key or requested in layer_key


def route_capture_enabled() -> bool:
    """Return whether route capture was enabled before process startup."""

    return bool(_CAPTURE_PATH_TEMPLATE)


def route_capture_mode() -> str:
    """Return the configured route capture mode."""

    mode = os.environ.get("VEOMNI_HIERMOE_ORACLE_CAPTURE_MODE", "global").strip().lower()
    if mode not in {"global", "local"}:
        raise ValueError(f"VEOMNI_HIERMOE_ORACLE_CAPTURE_MODE must be either 'global' or 'local', got {mode!r}.")
    return mode


def _layer_index(layer_key: str) -> int:
    matches = re.findall(r"(?:layers?|layer)\.(\d+)(?:\.|$)", layer_key)
    return int(matches[-1]) if matches else -1


def _capture_layer_key(step: int, layer_key: str | None) -> str:
    if layer_key is not None:
        return layer_key

    layer_ordinal = _CAPTURE_LAYER_ORDINALS.get(int(step), 0)
    _CAPTURE_LAYER_ORDINALS[int(step)] = layer_ordinal + 1
    raw_num_layers = os.environ.get("VEOMNI_HIERMOE_ORACLE_CAPTURE_NUM_LAYERS", "").strip()
    if raw_num_layers:
        try:
            num_layers = int(raw_num_layers)
        except ValueError as exc:
            raise ValueError("VEOMNI_HIERMOE_ORACLE_CAPTURE_NUM_LAYERS must be a positive integer.") from exc
        if num_layers < 1:
            raise ValueError("VEOMNI_HIERMOE_ORACLE_CAPTURE_NUM_LAYERS must be a positive integer.")
        layer_ordinal %= num_layers
    return f"model.layers.{layer_ordinal}.mlp.experts"


def _capture_output_path(
    raw_path: str,
    *,
    step: int,
    layer_key: str,
    call_index: int,
    global_rank: int,
    ep_rank: int,
) -> Path:
    return Path(
        raw_path.format(
            step=int(step),
            layer=re.sub(r"[^A-Za-z0-9_.-]+", "_", layer_key),
            layer_index=_layer_index(layer_key),
            call=int(call_index),
            rank=int(global_rank),
            ep_rank=int(ep_rank),
        )
    )


def _save_local_route_snapshot(
    *,
    routes: torch.Tensor,
    path: Path,
    global_rank: int,
    ep_rank: int,
    ep_size: int,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    hierarchy: Hierarchy,
    logical_to_physical: torch.Tensor,
    slot_to_logical: torch.Tensor | None,
    layer_key: str,
    step: int,
    call_index: int,
    smooth_max_gamma: float,
    selected_dim: int,
) -> Path:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": _LOCAL_SNAPSHOT_FORMAT,
            "version": _LOCAL_SNAPSHOT_VERSION,
            "routes": _normalize_routes(routes).to(torch.int32),
            "global_rank": int(global_rank),
            "ep_rank": int(ep_rank),
            "ep_size": int(ep_size),
            "num_experts": int(num_experts),
            "hidden_size": int(hidden_size),
            "bytes_per_element": int(bytes_per_element),
            "hierarchy_group_sizes": list(hierarchy.group_sizes),
            "hierarchy_source": hierarchy.source,
            "logical_to_physical": logical_to_physical.detach().to(device="cpu", dtype=torch.long),
            "slot_to_logical": (
                None if slot_to_logical is None else slot_to_logical.detach().to(device="cpu", dtype=torch.long)
            ),
            "layer": _layer_index(layer_key),
            "layer_key": layer_key,
            "step": int(step),
            "call_index": int(call_index),
            "smooth_max_gamma": float(smooth_max_gamma),
            "selected_dim": int(selected_dim),
        },
        output,
    )
    return output


def maybe_capture_route_snapshot(
    *,
    selected_experts: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    bytes_per_element: int,
    ep_group: dist.ProcessGroup | None,
    hierarchy: Hierarchy,
    layer_key: str | None,
    step: int,
    logical_to_physical: torch.Tensor | None = None,
    slot_to_logical: torch.Tensor | None = None,
    smooth_max_gamma: float = 10.0,
    selected_dim: int = 1,
) -> Path | None:
    """Capture one route invocation selected by debug-only environment variables."""

    raw_path = _CAPTURE_PATH_TEMPLATE
    if not raw_path:
        return None
    target_step = int(os.environ.get("VEOMNI_HIERMOE_ORACLE_CAPTURE_STEP", "-1"))
    target_layer = os.environ.get("VEOMNI_HIERMOE_ORACLE_CAPTURE_LAYER", "")
    target_call = int(os.environ.get("VEOMNI_HIERMOE_ORACLE_CAPTURE_CALL", "0"))
    layer_key = _capture_layer_key(step, layer_key)
    if (target_step >= 0 and int(step) != target_step) or not _layer_matches(layer_key, target_layer):
        return None

    call_key = (int(step), layer_key)
    call_index = _CAPTURE_CALLS.get(call_key, 0)
    _CAPTURE_CALLS[call_key] = call_index + 1
    capture_key = (int(step), layer_key, call_index)
    if call_index != target_call or capture_key in _CAPTURED:
        return None
    _CAPTURED.add(capture_key)

    local_routes = selected_experts.detach().to(dtype=torch.int32)
    if local_routes.ndim == 1:
        local_routes = local_routes.unsqueeze(-1)
    elif local_routes.ndim > 2:
        local_routes = local_routes.reshape(-1, local_routes.shape[-1])
    local_routes = local_routes.contiguous()

    initialized = dist.is_initialized()
    ep_size = dist.get_world_size(ep_group) if ep_group is not None and initialized else 1
    global_rank = dist.get_rank() if initialized else 0
    ep_rank = dist.get_rank(ep_group) if ep_group is not None and initialized else 0
    mapping = (
        logical_to_physical.detach().to(device="cpu", dtype=torch.long)
        if logical_to_physical is not None
        else torch.arange(num_experts, dtype=torch.long)
    )
    mode = route_capture_mode()
    if mode == "local":
        if "{rank" not in raw_path and "{ep_rank" not in raw_path:
            raise ValueError(
                "Local HierMoE route capture path must contain a {rank} or {ep_rank} field "
                "so ranks on the same host do not overwrite each other."
            )
        output = _capture_output_path(
            raw_path,
            step=step,
            layer_key=layer_key,
            call_index=call_index,
            global_rank=global_rank,
            ep_rank=ep_rank,
        )
        return _save_local_route_snapshot(
            routes=local_routes,
            path=output,
            global_rank=global_rank,
            ep_rank=ep_rank,
            ep_size=ep_size,
            num_experts=num_experts,
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
            hierarchy=hierarchy,
            logical_to_physical=mapping,
            slot_to_logical=slot_to_logical,
            layer_key=layer_key,
            step=step,
            call_index=call_index,
            smooth_max_gamma=smooth_max_gamma,
            selected_dim=selected_dim,
        )

    if slot_to_logical is not None:
        raise RuntimeError(
            "Global HierMoE route capture does not support redundant expert slots; "
            "use VEOMNI_HIERMOE_ORACLE_CAPTURE_MODE=local."
        )
    if ep_size == 1:
        routes_by_rank = (local_routes.cpu(),)
    else:
        shape = torch.tensor(local_routes.shape, dtype=torch.long, device=local_routes.device)
        gathered_shapes = [torch.empty_like(shape) for _ in range(ep_size)]
        dist.all_gather(gathered_shapes, shape, group=ep_group)
        shapes = torch.stack(gathered_shapes).cpu()
        if not bool((shapes[:, 1] == shapes[0, 1]).all().item()):
            raise RuntimeError("All EP ranks must use the same top-k width for route capture.")
        max_tokens = int(shapes[:, 0].max().item())
        padded = torch.full(
            (max_tokens, int(shapes[0, 1].item())),
            -1,
            dtype=torch.int32,
            device=local_routes.device,
        )
        padded[: local_routes.shape[0]] = local_routes
        gathered = torch.empty(
            (ep_size * max_tokens, padded.shape[1]),
            dtype=padded.dtype,
            device=padded.device,
        )
        dist.all_gather_into_tensor(gathered, padded, group=ep_group)
        gathered = gathered.view(ep_size, max_tokens, padded.shape[1]).cpu()
        routes_by_rank = tuple(gathered[rank, : int(shapes[rank, 0].item())].contiguous() for rank in range(ep_size))

    if global_rank != 0:
        return None
    output = _capture_output_path(
        raw_path,
        step=step,
        layer_key=layer_key,
        call_index=call_index,
        global_rank=global_rank,
        ep_rank=ep_rank,
    )
    return save_route_snapshot(
        RouteSnapshot(
            routes_by_rank=tuple(routes.to(torch.long) for routes in routes_by_rank),
            num_experts=int(num_experts),
            hidden_size=int(hidden_size),
            bytes_per_element=int(bytes_per_element),
            hierarchy=hierarchy,
            logical_to_physical=mapping,
            layer_key=layer_key,
            step=int(step),
            call_index=call_index,
            smooth_max_gamma=float(smooth_max_gamma),
            selected_dim=int(selected_dim),
        ),
        output,
    )
