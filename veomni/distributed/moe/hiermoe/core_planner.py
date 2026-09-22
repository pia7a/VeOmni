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

"""Quota route representations and deterministic copy mapping."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .planner import _route_hash
from .topology import Hierarchy


CORE_MOE_ALGORITHM_VERSION = "core-moe-v2"
GatherFixed = Callable[[torch.Tensor], torch.Tensor]

_ROUTE_SUMMARY_SCHEMA_VERSION = 4
_FUSED_PLANNER_ABI_VERSION = 2
_FUSED_PROTOCOL_ABI_SHIFT = 16
_FUSED_PROTOCOL_CAPABILITY_MASK = (1 << _FUSED_PROTOCOL_ABI_SHIFT) - 1
_FUSED_CAP_COLLECTIVE = 1 << 0
_FUSED_CAP_SWAP_SELECT = 1 << 1
_FUSED_CAP_REPLICA_PROJECT = 1 << 2
_FUSED_CAP_REPLICA_MATCH = 1 << 3
_FUSED_CAP_QUOTA_MAP = 1 << 4
_FUSED_CAP_QUOTA_POLICY = 1 << 5


@dataclass(frozen=True)
class QuotaPolicyEntry:
    source_rank: int
    logical_expert: int
    destination_ranks: tuple[int, ...]
    quotas: tuple[int, ...]

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.source_rank,
            self.logical_expert,
            len(self.destination_ranks),
            *self.destination_ranks,
            *self.quotas,
        )

    @classmethod
    def from_tuple(cls, values: Sequence[int]) -> "QuotaPolicyEntry":
        row = tuple(int(value) for value in values)
        if len(row) < 3:
            raise ValueError("A quota policy row must contain source, expert, and destination count.")
        count = row[2]
        if count < 1 or len(row) != 3 + 2 * count:
            raise ValueError(f"Invalid quota policy row length: {row!r}.")
        return cls(
            source_rank=row[0],
            logical_expert=row[1],
            destination_ranks=row[3 : 3 + count],
            quotas=row[3 + count :],
        )


@dataclass(frozen=True)
class QuotaMapping:
    physical_slots: torch.Tensor
    policy: tuple[QuotaPolicyEntry, ...]


@dataclass(frozen=True)
class QuotaTensorTables:
    copy_slots: torch.Tensor
    copy_counts: torch.Tensor
    owner_ranks: torch.Tensor
    quota_weights: torch.Tensor
    quota_configured: torch.Tensor


def _copy_slots_by_logical(layout: Sequence[int], num_experts: int) -> tuple[tuple[int, ...], ...]:
    copies: list[list[int]] = [[] for _ in range(num_experts)]
    for slot, logical in enumerate(layout):
        if logical >= 0:
            if logical >= num_experts:
                raise ValueError(f"Layout slot {slot} contains invalid logical expert {logical}.")
            copies[logical].append(slot)
    if any(not slots for slots in copies):
        raise ValueError("Every logical expert must retain at least one physical copy.")
    return tuple(tuple(slots) for slots in copies)


def _communication_class(
    destination_rank: int,
    *,
    source_rank: int,
    other_owner_ranks: Sequence[int],
    hierarchy: Hierarchy,
) -> tuple[int, ...]:
    visited = (int(source_rank), *(int(rank) for rank in other_owner_ranks))
    levels = tuple(int(size) for size in hierarchy.group_sizes[: max(0, hierarchy.selected_dim - 1)])
    values = [int(all(destination_rank // size != rank // size for rank in visited)) for size in reversed(levels)]
    values.append(int(destination_rank not in visited))
    return tuple(values)


def _waterfill_quota(loads: dict[int, float], destinations: Sequence[int], total: int) -> dict[int, int]:
    quota = {int(rank): 0 for rank in destinations}
    remaining = max(0, int(total))
    if remaining == 0 or not destinations:
        return quota
    ordered = sorted((float(loads[int(rank)]), int(rank)) for rank in destinations)
    active = 1
    while active < len(ordered):
        next_level = ordered[active][0]
        current_level = ordered[active - 1][0]
        required = max(0, math.ceil(next_level - current_level)) * active
        if required > remaining:
            break
        if required:
            increment, extra = divmod(required, active)
            for index in range(active):
                quota[ordered[index][1]] += increment + int(index < extra)
            remaining -= required
        active += 1
    increment, extra = divmod(remaining, active)
    for index in range(active):
        quota[ordered[index][1]] += increment + int(index < extra)
    return quota


def assign_tokens_to_copies_with_quota(
    selected_experts: torch.Tensor,
    slot_to_logical: torch.Tensor,
    *,
    slots_per_rank: int,
    source_ranks: int | torch.Tensor,
    hierarchy: Hierarchy,
    owner_slots: torch.Tensor,
    token_ordinals: torch.Tensor | None = None,
    token_weights: torch.Tensor | None = None,
    quota_policy: Sequence[QuotaPolicyEntry] | None = None,
    step: int = 0,
    layer_seed: int = 0,
) -> QuotaMapping:
    """Map routes with communication-first, load-aware integer quotas.

    This eager implementation is deliberately scalar and serves as the oracle
    for the fused planner and dispatch kernels.
    """

    original_device = selected_experts.device
    selected = selected_experts.detach().to(device="cpu", dtype=torch.long)
    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    layout = slot_to_logical.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    owners = owner_slots.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    num_tokens, top_k = selected.shape
    num_experts = int(owners.numel())
    if isinstance(source_ranks, int):
        sources = torch.full((num_tokens,), int(source_ranks), dtype=torch.long)
    else:
        sources = source_ranks.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if sources.numel() != num_tokens:
        raise ValueError(f"source_ranks has {sources.numel()} values for {num_tokens} tokens.")
    ordinals = (
        torch.arange(num_tokens, dtype=torch.long)
        if token_ordinals is None
        else token_ordinals.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    )
    weights = (
        torch.ones((num_tokens,), dtype=torch.float64)
        if token_weights is None
        else token_weights.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    )
    if ordinals.numel() != num_tokens or weights.numel() != num_tokens:
        raise ValueError("token_ordinals and token_weights must match the token count.")

    layout_values = [int(value) for value in layout.tolist()]
    copies = _copy_slots_by_logical(layout_values, num_experts)
    owner_ranks = [int(slot) // int(slots_per_rank) for slot in owners.tolist()]
    rank_slots: list[dict[int, int]] = []
    for logical_slots in copies:
        rank_slots.append({slot // int(slots_per_rank): slot for slot in logical_slots})
    route_hashes = _route_hash(
        selected,
        token_ordinals=ordinals,
        step=step,
        layer_seed=layer_seed,
    ).to(device="cpu")

    records: list[tuple[int, int, int, int, tuple[int, ...], float, int]] = []
    route_positions: dict[tuple[int, int], list[int]] = defaultdict(list)
    for token in range(num_tokens):
        positions: dict[int, list[int]] = defaultdict(list)
        for position, logical in enumerate(selected[token].tolist()):
            positions[int(logical)].append(position)
        for logical, logical_positions in positions.items():
            multiplicity = len(logical_positions)
            other_owners = [owner_ranks[int(other)] for other in positions if int(other) != int(logical)]
            candidate_ranks = tuple(sorted(rank_slots[logical]))
            classes = {
                rank: _communication_class(
                    rank,
                    source_rank=int(sources[token]),
                    other_owner_ranks=other_owners,
                    hierarchy=hierarchy,
                )
                for rank in candidate_ranks
            }
            minimum = min(classes.values())
            eligible = tuple(rank for rank in candidate_ranks if classes[rank] == minimum)
            route_hash = int(route_hashes[token, logical_positions[0]])
            records.append(
                (
                    token,
                    logical,
                    int(sources[token]),
                    multiplicity,
                    eligible,
                    float(weights[token]),
                    route_hash,
                )
            )
            route_positions[(token, logical)] = logical_positions

    rank_loads = dict.fromkeys(range(hierarchy.ep_size), 0.0)
    chosen_rank: dict[tuple[int, int], int] = {}
    buckets: dict[tuple[int, int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        token, logical, source, multiplicity, eligible, weight, _ = record
        if len(eligible) == 1:
            rank = eligible[0]
            chosen_rank[(token, logical)] = rank
            rank_loads[rank] += multiplicity * weight
        else:
            buckets[(source, logical, eligible)].append(index)

    policy_lookup = {
        (entry.source_rank, entry.logical_expert, entry.destination_ranks): entry.quotas
        for entry in (quota_policy or ())
    }
    policy: list[QuotaPolicyEntry] = []
    ordered_buckets = sorted(
        buckets.items(),
        key=lambda item: (
            -sum(records[index][3] * records[index][5] for index in item[1]),
            item[0][0],
            item[0][1],
            item[0][2],
        ),
    )
    for (source, logical, destinations), indices in ordered_buckets:
        record_units = {index: float(records[index][3]) * float(records[index][5]) for index in indices}
        total = int(round(sum(record_units.values())))
        configured = policy_lookup.get((source, logical, destinations))
        if configured is None or sum(configured) <= 0:
            quotas = _waterfill_quota(rank_loads, destinations, total)
        else:
            raw = [total * int(value) / sum(configured) for value in configured]
            rounded = [math.floor(value) for value in raw]
            remainder = total - sum(rounded)
            order = sorted(
                range(len(destinations)), key=lambda index: (-(raw[index] - rounded[index]), destinations[index])
            )
            for index in order[:remainder]:
                rounded[index] += 1
            quotas = {rank: int(rounded[index]) for index, rank in enumerate(destinations)}
        ordered_indices = sorted(
            indices,
            key=lambda index: (
                records[index][6],
                int(ordinals[records[index][0]].item()),
            ),
        )
        assigned = dict.fromkeys(destinations, 0.0)
        if all(math.isclose(record_units[index], 1.0) for index in ordered_indices):
            # Unit-weight routes can realize the integer quota exactly.  Stable
            # hash ordering selects token identity, while cumulative intervals
            # consume each destination's quota instead of treating it as a
            # probabilistic hash weight.
            quota_total = sum(max(0, int(quotas[destination])) for destination in destinations)
            destination_index = 0
            consumed = 0
            for position, index in enumerate(ordered_indices):
                while destination_index + 1 < len(destinations) and position >= consumed + max(
                    0, int(quotas[destinations[destination_index]])
                ):
                    consumed += max(0, int(quotas[destinations[destination_index]]))
                    destination_index += 1
                rank = destinations[destination_index] if quota_total else destinations[position % len(destinations)]
                token = records[index][0]
                chosen_rank[(token, logical)] = rank
                assigned[rank] += 1.0
        else:
            # A repeated logical expert in top-k is one indivisible route with
            # multiplicity m.  Fill the largest quota deficit first, then the
            # lighter projected rank, with rank id as the deterministic tie.
            for index in ordered_indices:
                token = records[index][0]
                units = record_units[index]
                rank = min(
                    destinations,
                    key=lambda destination: (
                        assigned[destination] + units - float(quotas[destination]),
                        rank_loads[destination] + assigned[destination] + units,
                        destination,
                    ),
                )
                chosen_rank[(token, logical)] = rank
                assigned[rank] += units
        for destination in destinations:
            rank_loads[destination] += assigned[destination]
        policy.append(
            QuotaPolicyEntry(
                source_rank=source,
                logical_expert=logical,
                destination_ranks=tuple(destinations),
                quotas=tuple(int(quotas[rank]) for rank in destinations),
            )
        )

    physical = torch.empty_like(selected)
    for (token, logical), positions in route_positions.items():
        rank = chosen_rank[(token, logical)]
        slot = rank_slots[logical][rank]
        for position in positions:
            physical[token, position] = slot
    if selected_experts.ndim == 1:
        physical = physical.squeeze(-1)
    return QuotaMapping(
        physical_slots=physical.to(device=original_device, non_blocking=True),
        policy=tuple(policy),
    )


__all__ = [
    "CORE_MOE_ALGORITHM_VERSION",
    "QuotaMapping",
    "QuotaPolicyEntry",
    "QuotaTensorTables",
    "assign_tokens_to_copies_with_quota",
]
