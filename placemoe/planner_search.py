#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"Per-layer layout/mapping search and held-out route evaluation."

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial

import numpy as np
import torch

from placemoe.planner_candidates import (
    _build_candidate,
    _Candidate,
    _group_route_statistics,
    _logical_base_partitions,
    _logical_instances,
    _replica_allocations,
    _source_statistics,
)
from placemoe.planner_config import _CapacityPlan, _hierarchy_coefficients, _partition_coefficients
from veomni.distributed.moe.hiermoe.placemoe import (
    CommunityMappingConfig,
    LayerPlan,
    OptimizerConfig,
    PlacementConfig,
    PlaceMoETopology,
    ProfileStatistics,
    community_intersection_hits,
    community_node_placements,
    mirrored_r2_plan,
    optimize_community_mapping,
    optimize_fixed_layout_mapping,
)
from veomni.distributed.moe.hiermoe.placemoe.route_replay import (
    HybridCost,
    HybridEvaluator,
    load_routes,
)


def _plan_fixed_mapping_layer(
    layer: int,
    *,
    args: argparse.Namespace,
    optimize_samples: list[list[torch.Tensor]],
    validation_samples: list[list[torch.Tensor]],
    source_demand: np.ndarray,
    source_affinity: np.ndarray,
    evaluator: HybridEvaluator,
    layer_started: float,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Optimize the source mapping while preserving the input artifact's layout."""

    layer_offset = layer - args.layer_start
    layer_key = args.layer_keys[layer_offset] if args.layer_keys else args.layer_name_template.format(layer=layer)
    current_plan = args.input_plans[layer_key]
    topology = PlaceMoETopology(
        ep_size=args.ep_size,
        ranks_per_node=args.ranks_per_node,
        num_experts=args.num_experts,
        slots_per_rank=args.slots_per_rank,
    )
    hierarchy_group_sizes, level_omegas, gamma = _hierarchy_coefficients(args)
    node_omega = level_omegas[0]
    rank_omega = level_omegas[-1]
    statistics = ProfileStatistics(demand=source_demand, affinity=source_affinity)
    evaluated: dict[bytes, HybridCost] = {}

    def evaluate(plan: LayerPlan) -> float:
        key = plan.source_logical_to_physical.tobytes()
        cost = evaluated.get(key)
        if cost is None:
            cost = evaluator.evaluate(optimize_samples, plan.source_logical_to_physical)
            evaluated[key] = cost
        return cost.total_ms

    result = optimize_fixed_layout_mapping(
        statistics,
        current_plan,
        OptimizerConfig(
            topology=topology,
            primary_slots_per_rank=args.primary_slots_per_rank,
            node_omega=node_omega,
            rank_omega=rank_omega,
            gamma=gamma,
            mapping_sweep_limit=args.lut_iterations,
            hierarchy_group_sizes=hierarchy_group_sizes,
            level_omegas=level_omegas,
        ),
        evaluate,
    )
    best = result.best
    optimize_cost = evaluated[best.plan.source_logical_to_physical.tobytes()]
    validation_cost = evaluator.evaluate(validation_samples, best.plan.source_logical_to_physical)
    current_validation_cost = evaluator.evaluate(
        validation_samples,
        current_plan.source_logical_to_physical,
    )
    selection_reason = "optimize_routes"
    if validation_cost.total_ms > current_validation_cost.total_ms:
        best = next(
            candidate
            for candidate in result.candidates
            if np.array_equal(
                candidate.plan.source_logical_to_physical,
                current_plan.source_logical_to_physical,
            )
        )
        optimize_cost = evaluated[best.plan.source_logical_to_physical.tobytes()]
        validation_cost = current_validation_cost
        selection_reason = "incumbent_validation_fallback"
    layer_ms = (time.perf_counter() - layer_started) * 1000.0
    row: dict[str, object] = {
        "layer": layer,
        "strategy": "placemoe_fixed_layout_mapping",
        "selection_reason": selection_reason,
        "candidate_count": len(result.candidates),
        "candidates": [
            {
                "strategy": (
                    "current"
                    if np.array_equal(
                        candidate.plan.source_logical_to_physical,
                        current_plan.source_logical_to_physical,
                    )
                    else "mapping_candidate"
                ),
                "mapping_sweeps": candidate.mapping_sweeps,
                "mapping_changes": candidate.mapping_changes,
                "optimize": asdict(evaluated[candidate.plan.source_logical_to_physical.tobytes()]),
            }
            for candidate in result.candidates
        ],
        "planner_ms": layer_ms,
        "winner_planner_ms": layer_ms,
        "exact_route_evaluations": len(evaluated),
        "alternations": 0,
        "mapping_sweeps": best.mapping_sweeps,
        "mapping_changes": best.mapping_changes,
        "copy_counts": [int(value) for value in best.plan.copy_counts[: args.num_experts].tolist()],
        "optimize": asdict(optimize_cost),
        "validation": asdict(validation_cost),
        "comparison_validation": asdict(current_validation_cost),
    }
    print(
        f"layer={layer:02d} strategy=placemoe_fixed_layout_mapping "
        f"candidates={len(result.candidates)} optimize_ms={optimize_cost.total_ms:.3f} "
        f"validation_ms={validation_cost.total_ms:.3f} planner_ms={layer_ms:.1f}",
        flush=True,
    )
    return (
        layer,
        best.plan.slot_to_logical.copy(),
        best.plan.owner_slots.copy(),
        best.plan.source_logical_to_physical.copy(),
        row,
    )


def _plan_layer(
    layer: int,
    *,
    args: argparse.Namespace,
    capacity: _CapacityPlan,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    torch.set_num_threads(int(args.worker_threads))
    evaluator = HybridEvaluator(args)
    layer_started = time.perf_counter()
    optimize_samples = load_routes(
        args.route_root,
        steps=args.optimize_steps,
        layer=layer,
        ep_size=args.ep_size,
        call_indices=args.call_indices,
        forward_repeats=args.forward_repeats,
        layer_stride=args.expected_total_layers,
    )
    validation_samples = load_routes(
        args.route_root,
        steps=args.validation_steps,
        layer=layer,
        ep_size=args.ep_size,
        call_indices=args.call_indices,
        forward_repeats=args.forward_repeats,
        layer_stride=args.expected_total_layers,
    )
    source_demand, source_affinity = _source_statistics(
        optimize_samples,
        num_experts=args.num_experts,
    )
    if args.update_mode == "mapping":
        return _plan_fixed_mapping_layer(
            layer,
            args=args,
            optimize_samples=optimize_samples,
            validation_samples=validation_samples,
            source_demand=source_demand,
            source_affinity=source_affinity,
            evaluator=evaluator,
            layer_started=layer_started,
        )
    if args.communication_blind_proposals:
        source_affinity.fill(0.0)
    # These are exact identities, not approximations: source statistics retain
    # one row per EP rank, and their affinity sum is the same global pair
    # count previously rebuilt by a second complete route scan.
    demand_by_rank = source_demand
    affinity = source_affinity.sum(axis=0)
    node_omega, _, gamma = _partition_coefficients(args)
    if args.communication_blind_proposals:
        node_omega = 0.0
    partitions = _logical_base_partitions(
        affinity,
        demand_by_rank.sum(axis=0),
        num_nodes=args.ep_size // args.ranks_per_node,
        restarts=args.partition_restarts,
        iterations=args.partition_iterations,
        assignment_iterations=args.assignment_iterations,
        seed=args.seed + 100_003 * layer,
        omega=node_omega,
        gamma=gamma,
    )
    replica_allocations = _replica_allocations(
        affinity,
        demand_by_rank.sum(axis=0),
        replicas=capacity.active_replicas,
        restarts=args.partition_restarts,
        iterations=args.partition_iterations,
        assignment_iterations=args.assignment_iterations,
        seed=args.seed + 100_003 * layer + 53_123,
        candidate_limit=args.replica_candidate_limit,
        omega=node_omega,
        gamma=gamma,
    )
    candidates: list[_Candidate] = []
    candidate_jobs: list[dict[str, object]] = []
    seen_replica_sets: set[tuple[bytes, bytes]] = set()
    seen_generic_allocations: set[bytes] = set()
    cost_cache: dict[bytes, HybridCost] = {}
    for partition_index, partition in enumerate(partitions):
        for combination_index, replica_experts in enumerate(replica_allocations):
            key = (partition.tobytes(), replica_experts.tobytes())
            if key in seen_replica_sets:
                continue
            seen_replica_sets.add(key)
            logical_instances = _logical_instances(
                args.num_experts,
                replica_experts,
                total_slots=args.ep_size * args.slots_per_rank,
            )
            allocation_key = logical_instances.tobytes()
            if allocation_key not in seen_generic_allocations:
                seen_generic_allocations.add(allocation_key)
                # Full search keeps the calibrated and normalized heuristics
                # as independent state machines. The bounded search keeps the
                # normalized state machine because held-out EP32/EP64 replays
                # showed it dominates the calibrated proposal when the
                # partition and assignment budgets are aggressively capped.
                for proposal_restart in range(args.partition_restarts):
                    proposal_seed = args.seed + 100_003 * layer + 997 * proposal_restart + 31 * combination_index
                    if not args.fast_approx:
                        candidate_jobs.append(
                            {
                                "logical_instances": logical_instances,
                                "demand_by_source": source_demand,
                                "affinity_by_source": source_affinity,
                                "seed": proposal_seed,
                                "strategy": f"placemoe_p{proposal_restart}_c{combination_index}",
                            }
                        )
                    candidate_jobs.append(
                        {
                            "logical_instances": logical_instances,
                            "demand_by_source": source_demand,
                            "affinity_by_source": source_affinity,
                            "seed": proposal_seed,
                            "strategy": f"placemoe_normalized_p{proposal_restart}_c{combination_index}",
                            "calibrated_partition_refinement": False,
                        }
                    )
            community_enabled = not args.disable_community_block_candidates or args.include_community_block_candidates
            if community_enabled and not args.communication_blind_proposals:
                community_nodes = community_node_placements(
                    logical_instances,
                    partition,
                    source_demand,
                    PlacementConfig(
                        ep_size=args.ep_size,
                        ranks_per_node=args.ranks_per_node,
                        slots_per_rank=args.slots_per_rank,
                        node_omega=node_omega,
                        rank_omega=0.0,
                        gamma=gamma,
                    ),
                    candidate_limit=max(args.community_shortlist, 8 * args.community_shortlist),
                )
                if community_nodes:
                    assignments_by_community, community_mask_histogram = _group_route_statistics(
                        optimize_samples,
                        partition,
                        ranks_per_node=args.ranks_per_node,
                    )
                    intersection_hits = community_intersection_hits(community_mask_histogram)
                    community_rows: list[tuple[float, int, np.ndarray, np.ndarray]] = []
                    for community_index, instance_nodes in enumerate(community_nodes):
                        community_mapping = optimize_community_mapping(
                            logical_instances,
                            instance_nodes,
                            partition,
                            assignments_by_community,
                            community_mask_histogram,
                            CommunityMappingConfig(
                                ranks_per_node=args.ranks_per_node,
                                communication_ms_per_token=(
                                    args.communication_phase_multiplier
                                    * args.hidden_size
                                    * args.bytes_per_element
                                    * args.inter_ms_per_byte
                                ),
                                assignment_ms_per_assignment=(
                                    args.compute_phase_multiplier * args.compute_ms_per_assignment
                                    + args.communication_phase_multiplier * args.route_ms_per_assignment
                                ),
                                sweep_limit=args.community_sweeps,
                            ),
                            intersection_hits=intersection_hits,
                        )
                        community_rows.append(
                            (
                                community_mapping.proxy_cost,
                                community_index,
                                instance_nodes,
                                community_mapping.mapping,
                            )
                        )
                    for _, community_index, instance_nodes, community_mapping in sorted(
                        community_rows,
                        key=lambda row: (row[0], row[1]),
                    )[: args.community_shortlist]:
                        candidate_jobs.append(
                            {
                                "logical_instances": logical_instances,
                                "demand_by_source": source_demand,
                                "affinity_by_source": source_affinity,
                                "seed": (
                                    args.seed
                                    + 100_003 * layer
                                    + 997 * partition_index
                                    + 31 * combination_index
                                    + 17 * community_index
                                ),
                                "strategy": (
                                    f"community_block_{community_index}_p{partition_index}_c{combination_index}"
                                ),
                                "fixed_instance_nodes": instance_nodes,
                                "fixed_initial_lut": community_mapping,
                                "refine_fixed_initial_lut": True,
                            }
                        )
    build_candidate = partial(
        _build_candidate,
        optimize_samples,
        evaluator=evaluator,
        args=args,
        cost_cache=cost_cache,
    )
    if args.candidate_workers == 1:
        built_candidates = [build_candidate(**job) for job in candidate_jobs]
    else:
        with ThreadPoolExecutor(
            max_workers=min(args.candidate_workers, len(candidate_jobs)),
        ) as executor:
            built_candidates = list(executor.map(lambda job: build_candidate(**job), candidate_jobs))
    candidates.extend(candidate for candidate in built_candidates if candidate is not None)
    topology = PlaceMoETopology(
        ep_size=args.ep_size,
        ranks_per_node=args.ranks_per_node,
        num_experts=args.num_experts,
        slots_per_rank=args.slots_per_rank,
    )
    if args.ep_size % 2 == 0 and topology.total_slots == 2 * args.num_experts:
        seed_plan = mirrored_r2_plan(topology)
        seed_key = seed_plan.source_logical_to_physical.tobytes()
        seed_cost = cost_cache.get(seed_key)
        if seed_cost is None:
            seed_cost = evaluator.evaluate(optimize_samples, seed_plan.source_logical_to_physical)
            cost_cache[seed_key] = seed_cost
        candidates.append(
            _Candidate(
                strategy="placemoe_uniform_seed",
                layout=seed_plan.slot_to_logical.copy(),
                owners=seed_plan.owner_slots.copy(),
                lut=seed_plan.source_logical_to_physical.copy(),
                lut_instances=seed_plan.source_logical_to_physical.copy(),
                logical_instances=seed_plan.slot_to_logical.copy(),
                instance_ranks=np.arange(topology.total_slots, dtype=np.int64) // topology.slots_per_rank,
                optimize_cost=seed_cost,
                planner_ms=0.0,
                alternations=0,
            )
        )
    current_plan: LayerPlan | None = None
    if args.input_layout is not None:
        layer_offset = layer - args.layer_start
        layer_key = args.layer_keys[layer_offset] if args.layer_keys else args.layer_name_template.format(layer=layer)
        current_plan = args.input_plans[layer_key]
        current_key = current_plan.source_logical_to_physical.tobytes()
        current_cost = cost_cache.get(current_key)
        if current_cost is None:
            current_cost = evaluator.evaluate(optimize_samples, current_plan.source_logical_to_physical)
            cost_cache[current_key] = current_cost
        active_slots = np.flatnonzero(current_plan.slot_to_logical >= 0)
        candidates.append(
            _Candidate(
                strategy="placemoe_current_incumbent",
                layout=current_plan.slot_to_logical.copy(),
                owners=current_plan.owner_slots.copy(),
                lut=current_plan.source_logical_to_physical.copy(),
                lut_instances=current_plan.source_logical_to_physical.copy(),
                logical_instances=current_plan.slot_to_logical[active_slots].copy(),
                instance_ranks=active_slots // topology.slots_per_rank,
                optimize_cost=current_cost,
                planner_ms=0.0,
                alternations=0,
            )
        )
    if not candidates:
        raise RuntimeError(f"No feasible recursive classifier candidate for layer {layer}.")
    best = min(candidates, key=lambda item: item.optimize_cost.total_ms)
    validation_cost = evaluator.evaluate(validation_samples, best.lut)
    comparison_validation_cost: HybridCost | None = None
    selection_reason = "optimize_routes"
    if current_plan is not None:
        comparison_validation_cost = evaluator.evaluate(
            validation_samples,
            current_plan.source_logical_to_physical,
        )
        if validation_cost.total_ms > comparison_validation_cost.total_ms:
            best = next(candidate for candidate in candidates if candidate.strategy == "placemoe_current_incumbent")
            validation_cost = comparison_validation_cost
            selection_reason = "incumbent_validation_fallback"
    elif args.comparison_layout == "mirrored-r2":
        if args.ep_size % 2 or args.ep_size * args.slots_per_rank != 2 * args.num_experts:
            raise ValueError("mirrored-r2 comparison requires an even EP size and exactly two full expert copies.")
        r2_plan = mirrored_r2_plan(topology)
        comparison_validation_cost = evaluator.evaluate(validation_samples, r2_plan.source_logical_to_physical)
    active_layout = best.layout[best.layout >= 0]
    copy_counts = np.bincount(active_layout, minlength=args.num_experts)
    layer_ms = (time.perf_counter() - layer_started) * 1000.0
    row: dict[str, object] = {
        "layer": layer,
        "strategy": best.strategy,
        "selection_reason": selection_reason,
        "candidate_count": len(candidates),
        "candidates": [
            {
                "strategy": candidate.strategy,
                "planner_ms": candidate.planner_ms,
                "optimize": asdict(candidate.optimize_cost),
            }
            for candidate in sorted(
                candidates,
                key=lambda item: item.optimize_cost.total_ms,
            )
        ],
        "planner_ms": layer_ms,
        "winner_planner_ms": best.planner_ms,
        "exact_route_evaluations": len(cost_cache),
        "alternations": best.alternations,
        "copy_counts": [int(value) for value in copy_counts.tolist()],
        "optimize": asdict(best.optimize_cost),
        "validation": asdict(validation_cost),
        "comparison_validation": (
            asdict(comparison_validation_cost) if comparison_validation_cost is not None else None
        ),
    }
    print(
        f"layer={layer:02d} strategy={best.strategy} candidates={len(candidates)} "
        f"optimize_ms={best.optimize_cost.total_ms:.3f} "
        f"validation_ms={validation_cost.total_ms:.3f} "
        f"planner_ms={layer_ms:.1f}",
        flush=True,
    )
    return layer, best.layout, best.owners, best.lut, row
