#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"Planner arguments, topology coefficients, and bounded search budgets."

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from veomni.distributed.moe.hiermoe.topology import expected_hierarchy_group_sizes


_FULL_SEARCH_DEFAULTS = {
    "replica_candidate_limit": 64,
    "partition_restarts": 3,
    "alternations": 3,
    "lut_iterations": 6,
    "partition_iterations": 24,
    "assignment_iterations": 12,
    "community_shortlist": 2,
    "community_sweeps": 4,
}


_FAST_APPROX_DEFAULTS = {
    "replica_candidate_limit": 1,
    "partition_restarts": 2,
    "alternations": 2,
    "lut_iterations": 2,
    "partition_iterations": 8,
    "assignment_iterations": 4,
    "community_shortlist": 2,
    "community_sweeps": 2,
}


@dataclass(frozen=True)
class _CapacityPlan:
    primary_slots_per_rank: int
    reserved_replicas: int
    active_replicas: int
    empty_slots: int


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def _parse_str_list(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one string.")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument(
        "--update-mode",
        choices=("full", "mapping"),
        default="full",
        help="Optimize both L and M, or optimize only M under a fixed input layout.",
    )
    parser.add_argument(
        "--input-layout",
        type=Path,
        default=None,
        help=(
            "Current PlaceMoE artifact. Required by --update-mode mapping; under full search, "
            "the current L/M pair is retained as an exact incumbent candidate."
        ),
    )
    parser.add_argument("--optimize-steps", type=_parse_int_list, default=(1,))
    parser.add_argument("--validation-steps", type=_parse_int_list, default=(2,))
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument(
        "--layer-name-template",
        default="model.language_model.layers.{layer}.mlp.experts",
        help="Python format template for the runtime expert-module key.",
    )
    parser.add_argument(
        "--layer-keys",
        type=_parse_str_list,
        default=(),
        help=(
            "Comma-separated runtime expert-module keys in captured layer order. "
            "This overrides --layer-name-template and is required when hot replanning "
            "a model whose layer names are not represented by the default template."
        ),
    )
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument(
        "--call-indices",
        type=_parse_int_list,
        default=(0,),
        help="Captured Forward call indices to include for every optimizer step.",
    )
    parser.add_argument(
        "--forward-repeats",
        type=int,
        default=1,
        help=(
            "Forward microbatches captured per optimizer step. Repeated "
            "forwards are stored in consecutive layer-index blocks."
        ),
    )
    parser.add_argument(
        "--expected-total-layers",
        type=int,
        default=48,
        help="Total MoE layers in the model. Used only to gate full-model E2E eligibility.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(48, max(1, (os.cpu_count() or 1) // 4)),
        help="Independent layer planner processes. One preserves the legacy serial execution.",
    )
    parser.add_argument(
        "--candidate-workers",
        type=int,
        default=4,
        help="Concurrent exact candidate builders within each layer process.",
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=1,
        help="PyTorch CPU threads available to each layer planner process.",
    )
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument(
        "--hierarchy-group-sizes",
        type=_parse_int_list,
        default=(),
        help="Optional runtime hierarchy chain; legacy callers retain ranks-per-node,EP.",
    )
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument(
        "--slots-per-rank",
        type=int,
        default=None,
        help="Total reserved expert slots per rank. Defaults to primary capacity plus four redundant slots.",
    )
    parser.add_argument(
        "--redundant-slots-per-rank",
        type=int,
        default=None,
        help="Optional per-rank redundant capacity used to derive or validate slots-per-rank.",
    )
    parser.add_argument(
        "--primary-slots-per-rank",
        type=int,
        default=None,
        help="Canonical owner capacity per rank. Defaults to num_experts / ep_size.",
    )
    parser.add_argument(
        "--active-redundant-slots",
        type=int,
        default=None,
        help=(
            "Number of redundant copies to activate globally. Defaults to all reserved slots; "
            "unused uniformly reserved slots remain EMPTY."
        ),
    )
    parser.add_argument(
        "--replica-candidate-limit",
        type=int,
        default=None,
        help="Maximum exact-budget replica allocations evaluated per logical partition.",
    )
    parser.add_argument(
        "--fast-approx",
        action="store_true",
        help=(
            "Use compact search defaults and disable calibrated proposals. Explicit search-budget "
            "arguments override the fast defaults."
        ),
    )
    parser.add_argument("--partition-restarts", type=int, default=None)
    parser.add_argument("--alternations", type=int, default=None)
    parser.add_argument("--lut-iterations", type=int, default=None)
    parser.add_argument("--partition-iterations", type=int, default=None)
    parser.add_argument("--assignment-iterations", type=int, default=None)
    parser.add_argument("--community-shortlist", type=int, default=None)
    parser.add_argument("--community-sweeps", type=int, default=None)
    parser.add_argument(
        "--include-community-block-candidates",
        action="store_true",
        help=("Deprecated compatibility flag; topology-general affinity-community proposals are enabled by default."),
    )
    parser.add_argument(
        "--disable-community-block-candidates",
        action="store_true",
        help="Disable affinity-community placement and source-node block-mapping proposals.",
    )
    parser.add_argument(
        "--communication-blind-proposals",
        action="store_true",
        help=(
            "Generate candidates and source LUTs from assignment demand only. "
            "This is required for a strict compute-only ablation: token "
            "co-occurrence, source locality, and hierarchical communication "
            "must not influence either proposal generation or exact selection."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--inter-ms-per-byte", type=float, default=6.765449326279194e-08)
    parser.add_argument(
        "--mid-ms-per-byte",
        type=float,
        default=None,
        help="Optional middle-link coefficient for a three-stage hierarchy.",
    )
    parser.add_argument("--intra-ms-per-byte", type=float, default=5.02482606728045e-09)
    parser.add_argument("--route-ms-per-assignment", type=float, default=8.746548178958447e-05)
    parser.add_argument("--communication-phase-multiplier", type=float, default=3.1)
    parser.add_argument("--compute-ms-per-assignment", type=float, default=2.82807e-05)
    parser.add_argument("--compute-phase-multiplier", type=float, default=4.19)
    parser.add_argument(
        "--comparison-validation-ms",
        type=float,
        default=6116.241273880005,
        help=(
            "Comparison-only held-out cost. It is never used to generate a candidate. "
            "Ignored when --comparison-layout is not 'none'."
        ),
    )
    parser.add_argument(
        "--comparison-layout",
        choices=("none", "mirrored-r2"),
        default="none",
        help=(
            "Optionally evaluate a matched comparison layout on the same held-out routes. "
            "The main paper matrix uses mirrored-r2 instead of a topology-specific constant."
        ),
    )
    parser.add_argument("--output-layout", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _configure_search(args: argparse.Namespace) -> None:
    """Resolve mode-specific defaults while preserving explicit search budgets."""

    defaults = _FAST_APPROX_DEFAULTS if args.fast_approx else _FULL_SEARCH_DEFAULTS
    search_fields = tuple(defaults)
    requested = {
        name: int(defaults[name] if getattr(args, name, None) is None else getattr(args, name))
        for name in search_fields
    }
    for name, value in requested.items():
        setattr(args, name, value)
    invalid = [name for name, value in requested.items() if value < 0 or (name != "community_sweeps" and value == 0)]
    if invalid:
        raise ValueError(f"Planner search limits must be positive: {', '.join(invalid)}.")
    update_mode = str(getattr(args, "update_mode", "full"))
    mapping_only = update_mode == "mapping"
    reported_fields = ("lut_iterations",) if mapping_only else search_fields
    communication_blind = bool(getattr(args, "communication_blind_proposals", False))
    community_proposals = bool(
        not mapping_only
        and not communication_blind
        and (
            not getattr(args, "disable_community_block_candidates", False)
            or getattr(args, "include_community_block_candidates", False)
        )
    )
    legacy_structured_proposals = False
    legacy_hyperedge_proposals = False
    args.search_budget = {
        "mode": "fast_approx" if args.fast_approx else "full",
        "update_mode": update_mode,
        "requested": {name: requested[name] for name in reported_fields},
        "effective": {name: int(getattr(args, name)) for name in reported_fields},
        "calibrated_proposals": not mapping_only and not args.fast_approx,
        "normalized_proposals": not mapping_only,
        "community_proposals": community_proposals,
        "legacy_structured_proposals": legacy_structured_proposals,
        "legacy_hyperedge_proposals": legacy_hyperedge_proposals,
        "legacy_proposals": legacy_structured_proposals or legacy_hyperedge_proposals,
    }


def _hierarchy_coefficients(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...], tuple[float, ...], float]:
    """Return the runtime hierarchy, per-stage affinity weights, and compute weight."""

    hierarchy_group_sizes = tuple(
        int(size)
        for size in (
            getattr(args, "hierarchy_group_sizes", ())
            or expected_hierarchy_group_sizes(args.ep_size, args.ranks_per_node)
        )
    )
    if hierarchy_group_sizes[-1] != args.ep_size or len(hierarchy_group_sizes) not in {1, 2, 3}:
        raise ValueError("PlaceMoE requires a one-, two-, or three-stage EP hierarchy.")
    if len(hierarchy_group_sizes) == 1 and hierarchy_group_sizes != expected_hierarchy_group_sizes(
        args.ep_size, args.ranks_per_node
    ):
        raise ValueError("A one-stage PlaceMoE hierarchy requires all EP ranks to be on one node.")
    if any(lhs <= 0 or rhs % lhs for lhs, rhs in zip(hierarchy_group_sizes, hierarchy_group_sizes[1:], strict=False)):
        raise ValueError("Hierarchy group sizes must form an increasing divisibility chain.")
    middle = getattr(args, "mid_ms_per_byte", None)
    if len(hierarchy_group_sizes) == 1:
        link_coefficients = (float(args.intra_ms_per_byte),)
    elif len(hierarchy_group_sizes) == 3:
        link_coefficients = (
            float(args.inter_ms_per_byte),
            float(args.inter_ms_per_byte if middle is None else middle),
            float(args.intra_ms_per_byte),
        )
    else:
        link_coefficients = (
            float(args.inter_ms_per_byte),
            float(args.intra_ms_per_byte),
        )
    if any(coefficient < 0.0 for coefficient in link_coefficients):
        raise ValueError("Hierarchy link coefficients must be non-negative.")
    payload_bytes = float(args.hidden_size * args.bytes_per_element)
    multiplier = float(args.communication_phase_multiplier) * payload_bytes
    level_omegas = tuple(multiplier * coefficient for coefficient in link_coefficients)
    gamma = float(args.compute_phase_multiplier) * float(args.compute_ms_per_assignment)
    return hierarchy_group_sizes, level_omegas, gamma


def _partition_coefficients(args: argparse.Namespace) -> tuple[float, float, float]:
    """Return compatibility aliases for coarse, rank, and compute coefficients."""

    _hierarchy, level_omegas, gamma = _hierarchy_coefficients(args)
    return level_omegas[0], level_omegas[-1], gamma


def _validate_configuration(args: argparse.Namespace) -> _CapacityPlan:
    if args.ep_size % args.ranks_per_node:
        raise ValueError("EP size must be divisible by ranks per node.")
    _hierarchy_coefficients(args)
    if args.num_experts <= 0 or args.ep_size <= 0:
        raise ValueError("Expert and EP sizes must be positive.")
    community_shortlist = int(getattr(args, "community_shortlist", 2))
    community_sweeps = int(getattr(args, "community_sweeps", 4))
    if community_shortlist <= 0 or community_sweeps < 0:
        raise ValueError("Community shortlist must be positive and sweeps must be non-negative.")
    if args.num_experts % args.ep_size:
        raise ValueError("This initializer requires logical experts to divide evenly across EP ranks.")

    primary_slots = args.num_experts // args.ep_size
    if args.primary_slots_per_rank is None:
        args.primary_slots_per_rank = primary_slots
    elif int(args.primary_slots_per_rank) != primary_slots:
        raise ValueError(
            "Primary slots per rank must equal num_experts / ep_size; "
            f"expected {primary_slots}, got {args.primary_slots_per_rank}."
        )
    configured_redundant = getattr(args, "redundant_slots_per_rank", None)
    if args.slots_per_rank is None:
        redundant_per_rank = 4 if configured_redundant is None else int(configured_redundant)
        if redundant_per_rank < 0:
            raise ValueError("Redundant slots per rank must be non-negative.")
        args.slots_per_rank = primary_slots + redundant_per_rank
    elif configured_redundant is not None and int(args.slots_per_rank) != primary_slots + int(configured_redundant):
        raise ValueError("slots-per-rank does not match primary plus redundant slots per rank.")
    if args.slots_per_rank < primary_slots:
        raise ValueError("Physical slots per rank cannot be smaller than the canonical owner capacity.")

    reserved = args.ep_size * (args.slots_per_rank - primary_slots)
    active = reserved if args.active_redundant_slots is None else int(args.active_redundant_slots)
    if active < 0 or active > reserved:
        raise ValueError(f"Active redundant slots must be between zero and reserved capacity {reserved}.")
    # A rank cannot hold the same logical expert twice. Consequently an
    # expert has at most one copy on every EP rank.
    if active > args.num_experts * (args.ep_size - 1):
        raise ValueError("Active replica budget exceeds the duplicate-free EP placement capacity.")
    return _CapacityPlan(
        primary_slots_per_rank=primary_slots,
        reserved_replicas=reserved,
        active_replicas=active,
        empty_slots=reserved - active,
    )
