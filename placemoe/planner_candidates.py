#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"Deterministic PlaceMoE candidate construction; no training runtime side effects."

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch

from placemoe.planner_config import _hierarchy_coefficients
from veomni.distributed.moe.hiermoe.placemoe import (
    MappingConfig,
    OptimizerConfig,
    PartitionConfig,
    PlacementConfig,
    PlaceMoETopology,
    ProfileStatistics,
    build_replica_allocations,
    materialize_plan,
    optimize_mapping,
    optimize_replica_allocation,
    partition_items,
    profile_route_statistics,
    project_statistics_to_copies,
    repair_rank_placement,
)
from veomni.distributed.moe.hiermoe.placemoe.route_replay import (
    HybridCost,
    HybridEvaluator,
)


@dataclass(frozen=True)
class _Candidate:
    strategy: str
    layout: np.ndarray
    owners: np.ndarray
    lut: np.ndarray
    lut_instances: np.ndarray
    logical_instances: np.ndarray
    instance_ranks: np.ndarray
    optimize_cost: HybridCost
    planner_ms: float
    alternations: int


def _source_statistics(
    samples: list[list[torch.Tensor]],
    *,
    num_experts: int,
) -> tuple[np.ndarray, np.ndarray]:
    statistics = profile_route_statistics(samples, num_experts=num_experts)
    return statistics.demand.copy(), statistics.affinity.copy()


def _logical_base_partitions(
    affinity: np.ndarray,
    demand: np.ndarray,
    *,
    num_nodes: int,
    restarts: int,
    iterations: int,
    assignment_iterations: int,
    seed: int,
    omega: float = 1.0,
    gamma: float = 1.0,
) -> list[np.ndarray]:
    num_experts = int(affinity.shape[0])
    if num_experts % num_nodes:
        raise ValueError("Logical experts must divide evenly across nodes.")
    config = PartitionConfig(
        capacities=(num_experts // num_nodes,) * num_nodes,
        ranks_per_group=(1,) * num_nodes,
        omega=omega,
        gamma=gamma,
        restarts=restarts,
        assignment_iterations=assignment_iterations,
        exchange_limit=iterations,
        seed=seed,
    )
    results: list[np.ndarray] = []
    for result in partition_items(affinity, demand, config):
        groups = sorted(
            (tuple(np.flatnonzero(result.labels == label).tolist()) for label in range(num_nodes)),
            key=lambda row: row,
        )
        canonical = np.full_like(result.labels, -1)
        for label, experts in enumerate(groups):
            canonical[list(experts)] = label
        results.append(canonical)
    return results


def _replica_allocations(
    affinity: np.ndarray,
    demand: np.ndarray,
    *,
    replicas: int,
    restarts: int,
    iterations: int,
    assignment_iterations: int,
    seed: int,
    candidate_limit: int,
    omega: float = 1.0,
    gamma: float = 1.0,
) -> list[np.ndarray]:
    """Return exact-size replica multisets for any feasible global budget.

    Full expert-library copies are peeled off first. The remaining budget is
    represented as a union of equal affinity classes whose size is
    ``gcd(num_experts, residual)``. This is the capacity-general form of the
    previous whole-node-class enumeration: for example, 96 replicas among 128
    experts becomes three classes chosen from four, while 16 replicas becomes
    one class chosen from eight.
    """

    num_experts = int(affinity.shape[0])
    if affinity.shape != (num_experts, num_experts):
        raise ValueError("Replica affinity must be square.")
    if demand.shape != (num_experts,):
        raise ValueError("Replica demand shape does not match the expert count.")
    if replicas < 0:
        raise ValueError("Replica capacity must be non-negative.")
    if candidate_limit <= 0:
        raise ValueError("Replica candidate limit must be positive.")

    _, residual = divmod(int(replicas), num_experts)
    if residual == 0:
        return build_replica_allocations(
            [],
            demand,
            additional_copies=replicas,
            candidate_limit=candidate_limit,
        )

    group_size = int(np.gcd(num_experts, residual))
    num_groups = num_experts // group_size
    partitions = (
        [np.arange(num_experts, dtype=np.int64)]
        if group_size == 1
        else _logical_base_partitions(
            affinity,
            demand,
            num_nodes=num_groups,
            restarts=restarts,
            iterations=iterations,
            assignment_iterations=assignment_iterations,
            seed=seed,
            omega=omega,
            gamma=gamma,
        )
    )
    return build_replica_allocations(
        partitions,
        demand,
        additional_copies=replicas,
        candidate_limit=candidate_limit,
    )


def _logical_instances(
    num_experts: int,
    replica_experts: np.ndarray,
    *,
    total_slots: int | None = None,
) -> np.ndarray:
    active = np.concatenate(
        [
            np.arange(num_experts, dtype=np.int64),
            replica_experts.astype(np.int64, copy=False),
        ]
    )
    if total_slots is None:
        return active
    if total_slots < len(active):
        raise ValueError("Physical slot capacity is smaller than the active expert instance count.")
    return np.pad(active, (0, total_slots - len(active)), constant_values=-1)


def _group_route_statistics(
    samples: list[list[torch.Tensor]],
    logical_groups: np.ndarray,
    *,
    ranks_per_node: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize exact token-group incidence once for all structured seeds."""

    ep_size = len(samples[0])
    if ep_size % ranks_per_node:
        raise ValueError("Source ranks must divide evenly into nodes.")
    num_nodes = ep_size // ranks_per_node
    num_groups = int(logical_groups.max()) + 1
    if num_groups >= 63:
        raise ValueError("Group-mask statistics require fewer than 63 logical groups.")
    assignments_by_group = np.zeros((num_nodes, num_groups), dtype=np.float64)
    group_mask_histogram = np.zeros((num_nodes, 1 << num_groups), dtype=np.int64)
    group_lut = torch.from_numpy(logical_groups).to(torch.long)
    for sample in samples:
        for source_rank, route in enumerate(sample):
            source_node = source_rank // ranks_per_node
            groups = group_lut.index_select(0, route.reshape(-1)).view_as(route)
            assignments_by_group[source_node] += torch.bincount(
                groups.reshape(-1),
                minlength=num_groups,
            ).numpy()
            masks = torch.zeros((route.shape[0],), dtype=torch.long)
            for position in range(route.shape[1]):
                masks.bitwise_or_(
                    torch.bitwise_left_shift(
                        torch.ones_like(groups[:, position]),
                        groups[:, position],
                    )
                )
            group_mask_histogram[source_node] += torch.bincount(
                masks,
                minlength=1 << num_groups,
            ).numpy()
    return assignments_by_group, group_mask_histogram


def _mapped_instance_statistics(
    samples: list[list[torch.Tensor]],
    lut_instances: np.ndarray,
    *,
    logical_instances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    statistics = profile_route_statistics(samples, num_experts=lut_instances.shape[1])
    return project_statistics_to_copies(statistics, logical_instances, lut_instances)


def _greedy_ranks_with_fixed_nodes(
    instance_nodes: np.ndarray,
    instance_demand: np.ndarray,
    instance_affinity: np.ndarray,
    *,
    ranks_per_node: int,
    slots_per_rank: int,
    logical_instances: np.ndarray | None = None,
) -> np.ndarray:
    """Compatibility wrapper for the canonical rank-feasibility repair."""

    if logical_instances is None:
        logical_instances = np.arange(len(instance_nodes), dtype=np.int64)
    else:
        logical_instances = np.asarray(logical_instances, dtype=np.int64)
        if logical_instances.shape != instance_nodes.shape:
            raise ValueError("Logical instance shape does not match node assignment.")
    num_nodes = int(instance_nodes.max()) + 1
    ep_size = num_nodes * ranks_per_node
    return repair_rank_placement(
        instance_nodes,
        instance_demand.sum(axis=0),
        instance_affinity.sum(axis=0),
        logical_instances,
        PlacementConfig(
            ep_size=ep_size,
            ranks_per_node=ranks_per_node,
            slots_per_rank=slots_per_rank,
            node_omega=0.0,
            rank_omega=0.0,
            gamma=0.0,
        ),
    )


def _optimize_lut_instances(
    logical_instances: np.ndarray,
    instance_ranks: np.ndarray,
    initial_lut: np.ndarray,
    demand_by_source: np.ndarray,
    affinity_by_source: np.ndarray,
    *,
    ranks_per_node: int,
    iterations: int,
    node_omega: float,
    rank_omega: float,
    gamma: float,
    hierarchy_group_sizes: tuple[int, ...] = (),
    level_omegas: tuple[float, ...] = (),
) -> np.ndarray:
    statistics = ProfileStatistics(demand=demand_by_source, affinity=affinity_by_source)
    result = optimize_mapping(
        logical_instances,
        instance_ranks,
        initial_lut,
        statistics,
        MappingConfig(
            ranks_per_node=ranks_per_node,
            node_omega=node_omega,
            rank_omega=rank_omega,
            gamma=gamma,
            sweep_limit=iterations,
            hierarchy_group_sizes=hierarchy_group_sizes,
            level_omegas=level_omegas,
        ),
    )
    return result.mapping.copy()


def _materialize_layout(
    logical_instances: np.ndarray,
    instance_ranks: np.ndarray,
    lut_instances: np.ndarray,
    demand_by_source: np.ndarray,
    *,
    ep_size: int,
    slots_per_rank: int,
    primary_slots_per_rank: int,
    num_experts: int,
    ranks_per_node: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plan = materialize_plan(
        logical_instances,
        instance_ranks,
        lut_instances,
        demand_by_source,
        PlaceMoETopology(
            ep_size=ep_size,
            ranks_per_node=ranks_per_node,
            num_experts=num_experts,
            slots_per_rank=slots_per_rank,
        ),
        primary_slots_per_rank=primary_slots_per_rank,
    )
    return (
        plan.slot_to_logical.copy(),
        plan.owner_slots.copy(),
        plan.source_logical_to_physical.copy(),
    )


def _build_placemoe_candidate(
    samples: list[list[torch.Tensor]],
    *,
    logical_instances: np.ndarray,
    demand_by_source: np.ndarray,
    affinity_by_source: np.ndarray,
    evaluator: HybridEvaluator,
    args: argparse.Namespace,
    seed: int,
    strategy: str,
    cost_cache: dict[bytes, HybridCost] | None,
    started: float,
    calibrated_partition_refinement: bool = True,
) -> _Candidate | None:
    """Build one allocation through the canonical PlaceMoE optimizer."""

    communication_blind = bool(args.communication_blind_proposals)
    statistics = ProfileStatistics(
        demand=demand_by_source,
        affinity=np.zeros_like(affinity_by_source) if communication_blind else affinity_by_source,
    )
    hierarchy_group_sizes, level_omegas, gamma = _hierarchy_coefficients(args)
    if communication_blind:
        level_omegas = tuple(0.0 for _ in level_omegas)
    node_omega = level_omegas[0]
    rank_omega = level_omegas[-1]
    topology = PlaceMoETopology(
        ep_size=args.ep_size,
        ranks_per_node=args.ranks_per_node,
        num_experts=args.num_experts,
        slots_per_rank=args.slots_per_rank,
    )
    evaluated: dict[bytes, HybridCost] = {}

    def evaluate(plan) -> float:
        key = plan.source_logical_to_physical.tobytes()
        cost = None if cost_cache is None else cost_cache.get(key)
        if cost is None:
            cost = evaluator.evaluate(samples, plan.source_logical_to_physical)
            if cost_cache is not None:
                cost_cache[key] = cost
        evaluated[key] = cost
        return cost.compute_ms if communication_blind else cost.total_ms

    try:
        result = optimize_replica_allocation(
            statistics,
            logical_instances,
            OptimizerConfig(
                topology=topology,
                primary_slots_per_rank=args.primary_slots_per_rank,
                node_omega=node_omega,
                rank_omega=rank_omega,
                gamma=gamma,
                rounds=args.alternations,
                assignment_iterations=args.assignment_iterations,
                node_exchange_limit=args.partition_iterations,
                rank_exchange_limit=max(1, args.partition_iterations // 2),
                mapping_sweep_limit=args.lut_iterations,
                prefer_node_local=not communication_blind,
                seed=seed,
                normalized_mapping_weights=(() if calibrated_partition_refinement else (8.0, 32.0, 128.0)),
                calibrated_mapping_refinement=calibrated_partition_refinement,
                carry_mapping_across_rounds=calibrated_partition_refinement,
                calibrated_partition_refinement=calibrated_partition_refinement,
                hierarchy_group_sizes=hierarchy_group_sizes,
                level_omegas=level_omegas,
            ),
            evaluate,
        )
    except RuntimeError:
        return None
    best = result.best
    plan = best.plan
    cost = evaluated[plan.source_logical_to_physical.tobytes()]
    return _Candidate(
        strategy=strategy,
        layout=plan.slot_to_logical.copy(),
        owners=plan.owner_slots.copy(),
        lut=plan.source_logical_to_physical.copy(),
        lut_instances=best.instance_mapping.copy(),
        logical_instances=best.logical_instances.copy(),
        instance_ranks=best.instance_ranks.copy(),
        optimize_cost=cost,
        planner_ms=(time.perf_counter() - started) * 1000.0,
        alternations=best.round_index + 1,
    )


def _build_candidate(
    samples: list[list[torch.Tensor]],
    *,
    logical_instances: np.ndarray,
    demand_by_source: np.ndarray,
    affinity_by_source: np.ndarray,
    evaluator: HybridEvaluator,
    args: argparse.Namespace,
    seed: int,
    strategy: str,
    fixed_instance_nodes: np.ndarray | None = None,
    fixed_initial_lut: np.ndarray | None = None,
    cost_cache: dict[bytes, HybridCost] | None = None,
    calibrated_partition_refinement: bool = True,
) -> _Candidate | None:
    started = time.perf_counter()
    if fixed_instance_nodes is None:
        return _build_placemoe_candidate(
            samples,
            logical_instances=logical_instances,
            demand_by_source=demand_by_source,
            affinity_by_source=affinity_by_source,
            evaluator=evaluator,
            args=args,
            seed=seed,
            strategy=strategy,
            cost_cache=cost_cache,
            started=started,
            calibrated_partition_refinement=calibrated_partition_refinement,
        )
    communication_blind = bool(args.communication_blind_proposals)
    if fixed_initial_lut is None:
        raise ValueError("Community candidates require their source mapping.")
    instance_demand, instance_affinity = _mapped_instance_statistics(
        samples,
        fixed_initial_lut,
        logical_instances=logical_instances,
    )
    best: _Candidate | None = None
    hierarchy_group_sizes, level_omegas, gamma = _hierarchy_coefficients(args)
    if communication_blind:
        level_omegas = tuple(0.0 for _ in level_omegas)
    node_omega = level_omegas[0]
    rank_omega = level_omegas[-1]
    for alternation in range(args.alternations):
        instance_ranks = _greedy_ranks_with_fixed_nodes(
            fixed_instance_nodes,
            instance_demand,
            instance_affinity,
            ranks_per_node=args.ranks_per_node,
            slots_per_rank=args.slots_per_rank,
            logical_instances=logical_instances,
        )
        # Community proposals always compare the original mapping and its refinement.
        lut_variants = [
            fixed_initial_lut,
            _optimize_lut_instances(
                logical_instances,
                instance_ranks,
                fixed_initial_lut,
                demand_by_source,
                affinity_by_source,
                ranks_per_node=args.ranks_per_node,
                iterations=args.lut_iterations,
                node_omega=node_omega,
                rank_omega=rank_omega,
                gamma=gamma,
                hierarchy_group_sizes=hierarchy_group_sizes,
                level_omegas=level_omegas,
            ),
        ]
        for lut_instances in lut_variants:
            try:
                layout, owners, lut = _materialize_layout(
                    logical_instances,
                    instance_ranks,
                    lut_instances,
                    demand_by_source,
                    ep_size=args.ep_size,
                    slots_per_rank=args.slots_per_rank,
                    primary_slots_per_rank=args.primary_slots_per_rank,
                    num_experts=args.num_experts,
                    ranks_per_node=args.ranks_per_node,
                )
            except RuntimeError:
                continue
            cache_key = lut.tobytes()
            cost = None if cost_cache is None else cost_cache.get(cache_key)
            if cost is None:
                cost = evaluator.evaluate(samples, lut)
                if cost_cache is not None:
                    cost_cache[cache_key] = cost
            candidate = _Candidate(
                strategy=strategy,
                layout=layout,
                owners=owners,
                lut=lut,
                lut_instances=lut_instances.copy(),
                logical_instances=logical_instances.copy(),
                instance_ranks=instance_ranks.copy(),
                optimize_cost=cost,
                planner_ms=(time.perf_counter() - started) * 1000.0,
                alternations=alternation + 1,
            )
            if best is None or cost.total_ms < best.optimize_cost.total_ms:
                best = candidate
        if best is None:
            continue
        instance_demand, instance_affinity = _mapped_instance_statistics(
            samples,
            best.lut_instances,
            logical_instances=logical_instances,
        )
        if communication_blind:
            instance_affinity.fill(0.0)
    if best is None:
        return None
    return _Candidate(
        **{
            **best.__dict__,
            "planner_ms": (time.perf_counter() - started) * 1000.0,
        }
    )
