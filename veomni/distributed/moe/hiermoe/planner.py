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

"""Deterministic physical-copy routing and shared hierarchy/hash helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import ModuleType

import torch


ReduceSum = Callable[[torch.Tensor], torch.Tensor | None]
_FUSED_REPLICA_MAX_TOKENS = 16_384
_FUSED_REPLICA_OPS_CACHE: dict[tuple[str, int, bool], ModuleType | None] = {}
_EXACT_COPY_MAX_COMBINATIONS = 4_096
_COPY_CHOICE_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}


def _hierarchy_distance(source_ranks: torch.Tensor, destination_ranks: torch.Tensor, group_sizes: Sequence[int]):
    source = source_ranks.view(1, -1, 1, 1)
    distance = torch.full_like(destination_ranks, len(group_sizes) + 1, dtype=torch.long)
    distance = torch.where(destination_ranks == source, torch.zeros_like(distance), distance)
    for level, raw_size in reversed(tuple(enumerate(group_sizes, start=1))):
        size = max(1, int(raw_size))
        same_group = torch.div(destination_ranks, size, rounding_mode="floor") == torch.div(
            source, size, rounding_mode="floor"
        )
        distance = torch.where(same_group & (destination_ranks != source), torch.full_like(distance, level), distance)
    return distance


def _route_hash(
    selected: torch.Tensor,
    *,
    token_ordinals: torch.Tensor | None,
    step: int,
    layer_seed: int,
) -> torch.Tensor:
    num_tokens = selected.shape[0]
    if token_ordinals is None:
        token_ids = torch.arange(num_tokens, device=selected.device, dtype=torch.long)
    else:
        token_ids = token_ordinals.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
        if token_ids.numel() != num_tokens:
            raise ValueError(f"token_ordinals has {token_ids.numel()} values for {num_tokens} tokens.")
    token_ids = token_ids.view(num_tokens, 1)
    logical = selected.to(torch.long)
    value = token_ids * 1_000_003 + logical * 65_537 + int(step) * 131 + int(layer_seed) * 17
    value = torch.remainder(value * 48_271 + 1, 2_147_483_647)
    return torch.remainder(value, 1_048_573)


def _copy_scores(
    selected: torch.Tensor,
    copy_ranks: torch.Tensor,
    source_ranks: torch.Tensor,
    owner_slots: torch.Tensor,
    *,
    slots_per_rank: int,
    num_ranks: int,
    hierarchy_group_sizes: Sequence[int],
) -> torch.Tensor:
    """Return lexicographic copy scores without route-order dependencies."""

    num_tokens, top_k = selected.shape
    owner_ranks = torch.div(owner_slots, max(1, int(slots_per_rank)), rounding_mode="floor")
    route_owner_ranks = owner_ranks.index_select(0, selected.reshape(-1)).view(num_tokens, top_k)
    owner_rank_counts = torch.zeros((num_tokens, max(1, int(num_ranks))), dtype=torch.long, device=selected.device)
    owner_rank_counts.scatter_add_(1, route_owner_ranks, torch.ones_like(route_owner_ranks))

    batch, _, _, num_copies = copy_ranks.shape
    flat_copy_ranks = copy_ranks.reshape(batch, num_tokens, top_k * num_copies)
    needed_counts = owner_rank_counts.unsqueeze(0).expand(batch, -1, -1).gather(2, flat_copy_ranks)
    needed_counts = needed_counts.view(batch, num_tokens, top_k, num_copies)
    is_own_owner = copy_ranks == route_owner_ranks.view(1, num_tokens, top_k, 1)
    already_needed = needed_counts > is_own_owner.to(torch.long)

    distance = _hierarchy_distance(source_ranks, copy_ranks, hierarchy_group_sizes)
    distance_scale = len(hierarchy_group_sizes) + 2
    return (~already_needed).to(torch.long) * distance_scale + distance


def _copy_choice_table(top_k: int, num_copies: int, device: torch.device) -> torch.Tensor:
    cache_key = (str(device), int(top_k), int(num_copies))
    cached = _COPY_CHOICE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    combinations = int(num_copies) ** int(top_k)
    combination_ids = torch.arange(combinations, dtype=torch.long, device=device).view(-1, 1)
    powers = torch.tensor(
        [int(num_copies) ** index for index in range(int(top_k))],
        dtype=torch.long,
        device=device,
    ).view(1, -1)
    cached = torch.remainder(torch.div(combination_ids, powers, rounding_mode="floor"), int(num_copies))
    _COPY_CHOICE_CACHE[cache_key] = cached
    return cached


def _remote_group_count_for_choices(
    selected_copy_slots: torch.Tensor,
    choices: torch.Tensor,
    source_groups: torch.Tensor,
    *,
    slots_per_rank: int,
    group_size: int,
) -> torch.Tensor:
    num_tokens, top_k, _num_copies = selected_copy_slots.shape
    num_combinations = int(choices.shape[0])
    counts = torch.zeros((num_tokens, num_combinations), dtype=torch.long, device=selected_copy_slots.device)
    previous_groups = []
    for route_index in range(top_k):
        option_indices = choices[:, route_index]
        chosen_slots = selected_copy_slots[:, route_index].index_select(1, option_indices)
        chosen_ranks = torch.div(chosen_slots, int(slots_per_rank), rounding_mode="floor")
        chosen_groups = torch.div(chosen_ranks, int(group_size), rounding_mode="floor")
        is_first_remote = chosen_groups != source_groups.view(-1, 1)
        for previous in previous_groups:
            is_first_remote.logical_and_(chosen_groups != previous)
        counts.add_(is_first_remote)
        previous_groups.append(chosen_groups)
    return counts


def _assign_tokens_to_copies_exact_vectorized(
    selected: torch.Tensor,
    selected_copy_slots: torch.Tensor,
    selected_copy_valid: torch.Tensor,
    *,
    slots_per_rank: int,
    source_ranks: torch.Tensor,
    hierarchy_group_sizes: Sequence[int],
    num_ranks: int,
    token_ordinals: torch.Tensor | None,
    step: int,
    layer_seed: int,
) -> torch.Tensor | None:
    num_tokens, top_k = selected.shape
    num_copies = int(selected_copy_slots.shape[-1])
    combinations = num_copies**top_k
    if combinations > _EXACT_COPY_MAX_COMBINATIONS:
        return None
    if num_tokens == 0:
        return selected.clone()

    choices = _copy_choice_table(top_k, num_copies, selected.device)
    num_combinations = int(choices.shape[0])
    valid = torch.ones((num_tokens, num_combinations), dtype=torch.bool, device=selected.device)
    for route_index in range(top_k):
        valid.logical_and_(selected_copy_valid[:, route_index].index_select(1, choices[:, route_index]))

    sorted_selected = selected.sort(dim=-1).values
    if top_k > 1 and bool((sorted_selected[:, 1:] == sorted_selected[:, :-1]).any().item()):
        for lhs in range(top_k):
            lhs_slots = selected_copy_slots[:, lhs].index_select(1, choices[:, lhs])
            for rhs in range(lhs + 1, top_k):
                same_logical = selected[:, lhs] == selected[:, rhs]
                if bool(same_logical.any().item()):
                    rhs_slots = selected_copy_slots[:, rhs].index_select(1, choices[:, rhs])
                    valid.logical_and_(
                        torch.logical_or(
                            torch.logical_not(same_logical.view(-1, 1)),
                            lhs_slots == rhs_slots,
                        )
                    )

    hierarchy_levels = sorted(
        {max(1, int(group_size)) for group_size in hierarchy_group_sizes if 1 < int(group_size) < int(num_ranks)},
        reverse=True,
    )
    hierarchy_levels.append(1)
    score = torch.zeros((num_tokens, num_combinations), dtype=torch.long, device=selected.device)
    score_base = top_k + 1
    for group_size in hierarchy_levels:
        source_groups = torch.div(source_ranks, int(group_size), rounding_mode="floor")
        group_count = _remote_group_count_for_choices(
            selected_copy_slots,
            choices,
            source_groups,
            slots_per_rank=slots_per_rank,
            group_size=group_size,
        )
        score.mul_(score_base).add_(group_count)

    invalid_score = torch.iinfo(torch.long).max
    score = torch.where(valid, score, torch.full_like(score, invalid_score))
    minimum = score.min(dim=-1, keepdim=True).values
    tied = valid & (score == minimum)
    route_hash = _route_hash(
        selected,
        token_ordinals=token_ordinals,
        step=step,
        layer_seed=layer_seed,
    )
    tie_modulus = 2_147_483_647
    tie_target = torch.remainder(route_hash.sum(dim=-1, keepdim=True), tie_modulus)
    candidate_hash = torch.zeros_like(score)
    for route_index in range(top_k):
        option_indices = choices[:, route_index]
        chosen_slots = selected_copy_slots[:, route_index].index_select(1, option_indices)
        candidate_hash.mul_(1_000_003).add_(chosen_slots + 1)
        candidate_hash.remainder_(tie_modulus)
    tie_mixed = torch.remainder(
        candidate_hash * (tie_target + 1_000_003) + tie_target * 48_271 + 1,
        tie_modulus,
    )
    tie_score = torch.remainder(tie_mixed * 48_271 + 1, tie_modulus)
    best_combination = torch.where(
        tied,
        tie_score,
        torch.full_like(tie_score, tie_modulus),
    ).argmin(dim=-1)
    best_choices = choices.index_select(0, best_combination)
    return selected_copy_slots.gather(2, best_choices.unsqueeze(-1)).squeeze(-1)


def assign_tokens_to_copies(
    selected_experts: torch.Tensor,
    slot_to_logical: torch.Tensor,
    *,
    slots_per_rank: int,
    source_ranks: int | torch.Tensor,
    hierarchy_group_sizes: Sequence[int],
    owner_slots: torch.Tensor | None = None,
    token_ordinals: torch.Tensor | None = None,
    step: int = 0,
    layer_seed: int = 0,
    max_copies: int = 2,
    copy_slots: torch.Tensor | None = None,
    copy_mask: torch.Tensor | None = None,
    validate_copy_table: bool = True,
) -> torch.Tensor:
    """Map logical routes to physical slots with deterministic locality priorities.

    The function accepts one layout ``[slots]`` or a candidate batch
    ``[candidates, slots]``. Candidate layouts are evaluated independently.
    """

    original_ndim = selected_experts.ndim
    selected = selected_experts.to(torch.long)
    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    layouts = slot_to_logical.to(device=selected.device, dtype=torch.long, non_blocking=True)
    squeeze_layout = layouts.ndim == 1
    if squeeze_layout:
        layouts = layouts.unsqueeze(0)
    if layouts.ndim != 2:
        raise ValueError(f"slot_to_logical must be rank 1 or 2, got shape={tuple(layouts.shape)}.")

    batch, num_slots = layouts.shape
    owners = None
    if owner_slots is not None:
        owners = owner_slots.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
        num_experts = int(owners.numel())
    else:
        selected_max = int(selected.max().item()) if selected.numel() else -1
        layout_max = int(layouts.max().item()) if layouts.numel() else -1
        num_experts = max(selected_max, layout_max) + 1
    if num_experts == 0:
        empty = selected.unsqueeze(0).expand(batch, *selected.shape)
        return empty[0] if squeeze_layout else empty
    if (layouts >= num_experts).any():
        raise ValueError("slot_to_logical contains a logical expert outside selected_experts' expert range.")
    if owners is None:
        first_layout = layouts[0]
        slot_index = torch.arange(num_slots, device=selected.device, dtype=torch.long)
        owners = torch.full((num_experts,), num_slots, device=selected.device, dtype=torch.long)
        owners.scatter_reduce_(
            0,
            first_layout.clamp_min(0),
            torch.where(first_layout >= 0, slot_index, torch.full_like(slot_index, num_slots)),
            reduce="amin",
            include_self=True,
        )

    if copy_slots is None:
        copy_limit = max(1, min(int(max_copies), num_slots))
        logical_ids = torch.arange(num_experts, device=selected.device, dtype=torch.long)
        slot_ids = torch.arange(num_slots, device=selected.device, dtype=torch.long).view(1, num_slots, 1)
        matches = layouts.unsqueeze(-1) == logical_ids.view(1, 1, num_experts)
        masked_slots = torch.where(matches, slot_ids, torch.full_like(slot_ids, num_slots))
        routed_copy_slots = masked_slots.sort(dim=1).values[:, :copy_limit].transpose(1, 2).contiguous()
        copy_valid = routed_copy_slots < num_slots
    else:
        cached_slots = copy_slots.to(device=selected.device, dtype=torch.long, non_blocking=True)
        if cached_slots.ndim != 2 or cached_slots.shape[0] != num_experts:
            raise ValueError(
                f"copy_slots must have shape [{num_experts}, copies], got shape={tuple(cached_slots.shape)}."
            )
        if cached_slots.shape[1] == 0:
            raise ValueError("copy_slots must contain at least one copy column.")
        routed_copy_slots = cached_slots.unsqueeze(0).expand(batch, -1, -1)
        if copy_mask is None:
            requested_valid = routed_copy_slots >= 0
        else:
            cached_mask = copy_mask.to(device=selected.device, dtype=torch.bool, non_blocking=True)
            if cached_mask.shape != cached_slots.shape:
                raise ValueError(
                    f"copy_mask must match copy_slots shape={tuple(cached_slots.shape)}, got {tuple(cached_mask.shape)}."
                )
            requested_valid = cached_mask.unsqueeze(0).expand(batch, -1, -1)
        in_bounds = (routed_copy_slots >= 0) & (routed_copy_slots < num_slots)
        safe_cached_slots = routed_copy_slots.clamp(min=0, max=max(0, num_slots - 1))
        cached_logicals = layouts.gather(1, safe_cached_slots.reshape(batch, -1)).view_as(routed_copy_slots)
        expected_logicals = torch.arange(num_experts, device=selected.device).view(1, num_experts, 1)
        matches_logical = cached_logicals == expected_logicals
        if validate_copy_table and bool((requested_valid & ~(in_bounds & matches_logical)).any().item()):
            raise ValueError("copy_slots contains a masked slot that does not hold the corresponding logical expert.")
        copy_valid = requested_valid & in_bounds & matches_logical
        copy_limit = int(cached_slots.shape[1])
        routed_copy_slots = torch.where(copy_valid, routed_copy_slots, torch.full_like(routed_copy_slots, num_slots))
    if (routed_copy_slots[:, :, 0] >= num_slots).any():
        raise ValueError("Every logical expert must retain at least one physical copy.")

    num_tokens, top_k = selected.shape
    flat_selected = selected.reshape(-1)
    routed_slots = routed_copy_slots.index_select(1, flat_selected).view(batch, num_tokens, top_k, copy_limit)
    valid = copy_valid.index_select(1, flat_selected).view(batch, num_tokens, top_k, copy_limit)
    safe_slots = routed_slots.clamp(max=max(0, num_slots - 1))
    copy_ranks = torch.div(safe_slots, max(1, int(slots_per_rank)), rounding_mode="floor")
    if isinstance(source_ranks, int):
        source = torch.full((num_tokens,), int(source_ranks), dtype=torch.long, device=selected.device)
    else:
        source = source_ranks.to(device=selected.device, dtype=torch.long, non_blocking=True).reshape(-1)
        if int(source.numel()) != num_tokens:
            raise ValueError(f"source_ranks has {source.numel()} values for {num_tokens} tokens.")

    if squeeze_layout:
        exact = _assign_tokens_to_copies_exact_vectorized(
            selected,
            safe_slots[0],
            valid[0],
            slots_per_rank=max(1, int(slots_per_rank)),
            source_ranks=source,
            hierarchy_group_sizes=hierarchy_group_sizes,
            num_ranks=max(1, num_slots // max(1, int(slots_per_rank))),
            token_ordinals=token_ordinals,
            step=step,
            layer_seed=layer_seed,
        )
        if exact is not None:
            return exact.squeeze(-1) if original_ndim == 1 else exact

    score = _copy_scores(
        selected,
        copy_ranks,
        source,
        owners,
        slots_per_rank=slots_per_rank,
        num_ranks=max(1, num_slots // max(1, int(slots_per_rank))),
        hierarchy_group_sizes=hierarchy_group_sizes,
    )
    score = torch.where(valid, score, torch.full_like(score, 1 << 50))
    minimum = score.min(dim=-1, keepdim=True).values
    tied = valid & (score == minimum)
    tie_order = tied.to(torch.long).cumsum(dim=-1) - 1
    route_hash = _route_hash(selected, token_ordinals=token_ordinals, step=step, layer_seed=layer_seed)
    target = torch.remainder(
        route_hash.view(1, num_tokens, top_k, 1),
        tied.sum(dim=-1, keepdim=True),
    )
    chosen = (tied & (tie_order == target)).to(torch.long).argmax(dim=-1, keepdim=True)
    physical = safe_slots.gather(-1, chosen).squeeze(-1)
    if squeeze_layout:
        physical = physical[0]
    if original_ndim == 1:
        physical = physical.squeeze(-1)
    return physical
