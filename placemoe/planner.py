#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"PlaceMoE planner CLI: execute per-layer searches and serialize their results."

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from placemoe.planner_config import _CapacityPlan, _configure_search, _parse_args, _validate_configuration
from placemoe.planner_search import _plan_layer
from veomni.distributed.moe.hiermoe.placemoe import (
    LayerPlan,
    PlaceMoETopology,
    build_placemoe_artifact,
    validate_placemoe_artifact,
)
from veomni.distributed.moe.hiermoe.topology import expected_hierarchy_group_sizes


def _is_e2e_eligible(
    *,
    layer_start: int,
    layers: int,
    expected_total_layers: int,
    validation_total_ms: float,
    comparison_validation_ms: float,
) -> bool:
    return bool(
        layer_start == 0 and layers == expected_total_layers and validation_total_ms <= comparison_validation_ms
    )


def _selected_replica_summary(
    rows: list[dict[str, object]],
    *,
    num_experts: int,
    capacity: _CapacityPlan,
) -> dict[str, object]:
    """Report actual selected replica use without hiding mixed per-layer budgets."""

    active_by_layer = [sum(int(value) for value in row["copy_counts"]) - num_experts for row in rows]
    empty_by_layer = [capacity.reserved_replicas - active for active in active_by_layer]
    uniform = len(set(active_by_layer)) == 1
    return {
        "replica_slots": capacity.active_replicas,
        "reserved_replica_slots": capacity.reserved_replicas,
        "requested_active_replica_slots": capacity.active_replicas,
        "requested_empty_slots": capacity.empty_slots,
        "active_replica_slots": active_by_layer[0] if uniform else None,
        "empty_slots": empty_by_layer[0] if uniform else None,
        "selected_active_replica_slots_by_layer": active_by_layer,
        "selected_empty_slots_by_layer": empty_by_layer,
        "selected_active_replica_slots_min": min(active_by_layer),
        "selected_active_replica_slots_max": max(active_by_layer),
    }


def _preloaded_replay_payload(
    *,
    layouts: list[np.ndarray],
    owners: list[np.ndarray],
    luts: list[np.ndarray],
    args: argparse.Namespace,
    algorithm: str,
) -> dict[str, object]:
    """Serialize validated plans for direct startup preload or a hot update."""

    plans: dict[str, LayerPlan] = {}
    for offset, (layout, owner, lut) in enumerate(zip(layouts, owners, luts, strict=True)):
        layer = args.layer_start + offset
        name = args.layer_keys[offset] if args.layer_keys else args.layer_name_template.format(layer=layer)
        plans[name] = LayerPlan(
            slot_to_logical=layout,
            owner_slots=owner,
            source_logical_to_physical=lut,
        )
    return build_placemoe_artifact(
        plans,
        PlaceMoETopology(
            ep_size=args.ep_size,
            ranks_per_node=args.ranks_per_node,
            num_experts=args.num_experts,
            slots_per_rank=args.slots_per_rank,
        ),
        source={
            "algorithm": algorithm,
            "search_mode": args.search_budget["mode"],
            "search_budget": args.search_budget,
            "route_root": str(args.route_root.resolve()),
            "optimize_steps": list(args.optimize_steps),
            "validation_steps": list(args.validation_steps),
            "layer_name_template": args.layer_name_template,
            "layer_keys": list(args.layer_keys),
            "update_mode": getattr(args, "update_mode", "full"),
            "hierarchy_group_sizes": list(
                getattr(args, "hierarchy_group_sizes", ())
                or expected_hierarchy_group_sizes(args.ep_size, args.ranks_per_node)
            ),
        },
    )


def main() -> None:
    args = _parse_args()
    _configure_search(args)
    if args.layer_keys and len(args.layer_keys) != args.layers:
        raise ValueError(f"--layer-keys contains {len(args.layer_keys)} keys, expected {args.layers}.")
    if len(set(args.layer_keys)) != len(args.layer_keys):
        raise ValueError("--layer-keys must contain unique runtime module keys.")
    if not args.layer_keys and "{layer}" not in args.layer_name_template:
        raise ValueError("--layer-name-template must contain the '{layer}' placeholder.")
    capacity = _validate_configuration(args)
    if args.update_mode == "mapping" and args.input_layout is None:
        raise ValueError("--input-layout is required by --update-mode mapping.")
    if args.input_layout is not None:
        payload = json.loads(args.input_layout.read_text(encoding="utf-8"))
        args.input_plans = validate_placemoe_artifact(payload)
        topology = payload["topology"]
        expected_topology = {
            "ep_size": args.ep_size,
            "ranks_per_node": args.ranks_per_node,
            "num_experts": args.num_experts,
            "slots_per_rank": args.slots_per_rank,
        }
        if any(int(topology[name]) != value for name, value in expected_topology.items()):
            raise ValueError("Input layout topology does not match the requested planner topology.")
        expected_keys = (
            set(args.layer_keys)
            if args.layer_keys
            else {
                args.layer_name_template.format(layer=layer)
                for layer in range(args.layer_start, args.layer_start + args.layers)
            }
        )
        if set(args.input_plans) != expected_keys:
            raise ValueError("Input layout layer keys do not match the requested planner layers.")
        active_copy_counts = {
            name: int(plan.copy_counts[: args.num_experts].sum()) - args.num_experts
            for name, plan in args.input_plans.items()
        }
        if args.update_mode == "full" and any(
            active_replicas > capacity.active_replicas for active_replicas in active_copy_counts.values()
        ):
            raise ValueError("Input layout replica budget exceeds the requested full-search budget.")
    if args.workers <= 0 or args.candidate_workers <= 0 or args.worker_threads <= 0:
        raise ValueError("workers, candidate-workers, and worker-threads must be positive.")
    layer_ids = tuple(range(args.layer_start, args.layer_start + args.layers))
    worker = partial(_plan_layer, args=args, capacity=capacity)
    wall_started = time.perf_counter()
    if args.workers == 1:
        results = [worker(layer) for layer in layer_ids]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(layer_ids))) as executor:
            results = list(executor.map(worker, layer_ids))
    wall_ms = (time.perf_counter() - wall_started) * 1000.0
    results.sort(key=lambda item: item[0])
    layouts = [item[1] for item in results]
    owners = [item[2] for item in results]
    luts = [item[3] for item in results]
    rows = [item[4] for item in results]

    validation_total = sum(float(row["validation"]["total_ms"]) for row in rows)
    replica_summary = _selected_replica_summary(
        rows,
        num_experts=args.num_experts,
        capacity=capacity,
    )
    if args.input_layout is not None:
        comparison_ms = sum(float(row["comparison_validation"]["total_ms"]) for row in rows)
    elif args.comparison_layout == "mirrored-r2":
        comparison_ms = sum(float(row["comparison_validation"]["total_ms"]) for row in rows)
    else:
        comparison_ms = float(args.comparison_validation_ms)
    report = {
        "schema_version": 1,
        "algorithm": "placemoe-v1",
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"input_plans", "output_layout", "output_report"}
        },
        "layers": rows,
        "aggregate": {
            "search_mode": args.search_budget["mode"],
            "search_budget": args.search_budget,
            **replica_summary,
            "optimize_total_ms": sum(float(row["optimize"]["total_ms"]) for row in rows),
            "validation_total_ms": validation_total,
            "planner_total_ms": sum(float(row["planner_ms"]) for row in rows),
            "planner_mean_ms_per_layer": sum(float(row["planner_ms"]) for row in rows) / len(rows),
            "planner_wall_ms": wall_ms,
            "workers": min(args.workers, len(layer_ids)),
            "candidate_workers": args.candidate_workers,
            "exact_route_evaluations": sum(int(row["exact_route_evaluations"]) for row in rows),
            "comparison_validation_ms": comparison_ms,
            "validation_gain_ms": comparison_ms - validation_total,
            "validation_speedup": comparison_ms / validation_total,
            "e2e_eligible": _is_e2e_eligible(
                layer_start=args.layer_start,
                layers=args.layers,
                expected_total_layers=args.expected_total_layers,
                validation_total_ms=validation_total,
                comparison_validation_ms=comparison_ms,
            ),
        },
    }
    payload = _preloaded_replay_payload(
        layouts=layouts,
        owners=owners,
        luts=luts,
        args=args,
        algorithm="placemoe-v1",
    )
    report["aggregate"]["serialization_mode"] = "preloaded"
    for path, value in ((args.output_layout, payload), (args.output_report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"validation_total_ms={validation_total:.6f} "
        f"comparison_ms={comparison_ms:.6f} "
        f"speedup={comparison_ms / validation_total:.6f} "
        f"e2e_eligible={report['aggregate']['e2e_eligible']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
