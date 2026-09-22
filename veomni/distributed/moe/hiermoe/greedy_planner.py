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

"""Nearest-copy routing with deterministic tie breaking."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .calibration_cost import GREEDY_COMMUNICATION_PHASE_MULTIPLIER
from .planner import _hierarchy_distance, _route_hash


GREEDY_COVER_ALGORITHM_VERSION = "hiermoe-greedy-cover-p1-exact-stats-v9-early-proxy-topk"
GREEDY_COMPUTE_PHASE_MULTIPLIER = 3.0
_ACTION_SWAP = 0
_ACTION_COVER = 1
_LAYER_COMPUTE_STREAMS: dict[tuple[str, int | None, int], tuple[object, ...]] = {}


def assign_tokens_to_copies_greedy(
    selected_experts: torch.Tensor,
    slot_to_logical: torch.Tensor,
    *,
    slots_per_rank: int,
    source_ranks: int | torch.Tensor,
    hierarchy_group_sizes: Sequence[int],
    num_experts: int,
    token_ordinals: torch.Tensor | None = None,
    step: int = 0,
    layer_seed: int = 0,
    max_copies: int = 8,
    route_hashes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map every logical route to its nearest available physical copy.

    Hierarchy distance is compared lexicographically from coarse groups to the
    source rank. Stable route hashes distribute exact ties. The route choice is
    independent across logical experts, which makes swap/cover deltas sparse:
    only routes of the two experts changed by an action can move.
    """

    original_selected_ndim = selected_experts.ndim
    selected = selected_experts.to(torch.long)
    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    if selected.ndim != 2:
        raise ValueError(f"selected_experts must have rank 1 or 2, got shape={tuple(selected.shape)}.")

    layouts = slot_to_logical.to(device=selected.device, dtype=torch.long, non_blocking=True)
    squeeze_layout = layouts.ndim == 1
    if squeeze_layout:
        layouts = layouts.unsqueeze(0)
    if layouts.ndim != 2:
        raise ValueError(f"slot_to_logical must have rank 1 or 2, got shape={tuple(layouts.shape)}.")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive.")
    if slots_per_rank <= 0 or layouts.shape[1] % int(slots_per_rank) != 0:
        raise ValueError("The physical layout must contain an integral number of ranks.")

    batch, num_slots = layouts.shape
    copy_limit = max(1, min(int(max_copies), num_slots))
    logical_ids = torch.arange(num_experts, dtype=torch.long, device=selected.device)
    slot_ids = torch.arange(num_slots, dtype=torch.long, device=selected.device).view(1, num_slots, 1)
    matches = layouts.unsqueeze(-1) == logical_ids.view(1, 1, num_experts)
    masked_slots = torch.where(matches, slot_ids, torch.full_like(slot_ids, num_slots))
    copy_slots = masked_slots.sort(dim=1).values[:, :copy_limit].transpose(1, 2).contiguous()
    copy_valid = copy_slots < num_slots
    if bool((copy_slots[:, :, 0] >= num_slots).any().item()):
        raise ValueError("Every logical expert must retain at least one physical copy.")

    num_tokens, top_k = selected.shape
    if isinstance(source_ranks, int):
        sources = torch.full((num_tokens,), int(source_ranks), dtype=torch.long, device=selected.device)
    else:
        sources = source_ranks.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
        if sources.numel() != num_tokens:
            raise ValueError(f"source_ranks has {sources.numel()} values for {num_tokens} tokens.")
    ordinals = (
        torch.arange(num_tokens, dtype=torch.long, device=selected.device)
        if token_ordinals is None
        else token_ordinals.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
    )
    if ordinals.numel() != num_tokens:
        raise ValueError(f"token_ordinals has {ordinals.numel()} values for {num_tokens} tokens.")

    selected_slots = copy_slots.index_select(1, selected.reshape(-1)).view(batch, num_tokens, top_k, copy_limit)
    selected_valid = copy_valid.index_select(1, selected.reshape(-1)).view(batch, num_tokens, top_k, copy_limit)
    safe_selected_slots = selected_slots.clamp(max=max(0, num_slots - 1))
    copy_ranks = torch.div(safe_selected_slots, int(slots_per_rank), rounding_mode="floor")
    distance = _hierarchy_distance(sources, copy_ranks, hierarchy_group_sizes)
    invalid_score = torch.iinfo(torch.long).max
    score = torch.where(selected_valid, distance, torch.full_like(distance, invalid_score))
    minimum = score.min(dim=-1, keepdim=True).values
    tied = selected_valid & (score == minimum)
    tie_order = tied.to(torch.long).cumsum(dim=-1) - 1
    if route_hashes is None:
        route_hashes = _route_hash(
            selected,
            token_ordinals=ordinals,
            step=step,
            layer_seed=layer_seed,
        )
    else:
        route_hashes = route_hashes.to(device=selected.device, dtype=torch.long, non_blocking=True)
        if route_hashes.shape != selected.shape:
            raise ValueError("route_hashes must match selected_experts after rank normalization.")
    tie_count = tied.sum(dim=-1, keepdim=True).clamp_min(1)
    tie_target = torch.remainder(route_hashes.view(1, num_tokens, top_k, 1), tie_count)
    chosen_copy = (tied & (tie_order == tie_target)).to(torch.long).argmax(dim=-1)
    physical = safe_selected_slots.gather(3, chosen_copy.unsqueeze(-1)).squeeze(-1)

    if squeeze_layout:
        physical = physical[0]
    if original_selected_ndim == 1:
        physical = physical.squeeze(-1)
    return physical


__all__ = [
    "GREEDY_COMMUNICATION_PHASE_MULTIPLIER",
    "GREEDY_COMPUTE_PHASE_MULTIPLIER",
    "GREEDY_COVER_ALGORITHM_VERSION",
    "assign_tokens_to_copies_greedy",
]
