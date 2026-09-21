"""Replay the archived EP16/E128 profile without GPU/NPU execution.

Run this exact script with each source revision in separate Python processes.
Compare output JSON byte-for-byte: wall time is deliberately excluded. This
checks captured inputs, synthetic feasible mappings, and six candidate rounds,
not distributed synchronization or measured accelerator performance.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from veomni.distributed.moe.hiermoe.placemoe import (
    OptimizerConfig,
    PlaceMoETopology,
    optimize_replica_allocation,
    profile_route_statistics,
)
from veomni.distributed.moe.hiermoe.placemoe.route_replay import HybridEvaluator


parser = argparse.ArgumentParser(description="Emit deterministic CPU replay evidence for refactor comparison.")
parser.add_argument("--snapshot", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reference", type=Path, help="Require exact equality with a previous replay result.")
options = parser.parse_args()
p = torch.load(options.snapshot, map_location="cpu", weights_only=True)
if p.get("format") != "veomni.hiermoe.route_snapshot" or p["ep_size"] != 16 or p["num_experts"] != 128:
    raise ValueError("This regression fixture requires the archived EP16/E128 route snapshot.")
routes = [p["routes"][r, : int(p["route_lengths"][r])].long() for r in range(16)]
statistics = profile_route_statistics([routes], num_experts=128)


def digest(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


result = {
    "snapshot_sha256": hashlib.sha256(options.snapshot.read_bytes()).hexdigest(),
    "snapshot": {"tokens": sum(len(r) for r in routes), "ep_size": 16, "experts": 128, "top_k": 8},
    "demand": digest(statistics.demand),
    "affinity": digest(statistics.affinity),
    "evaluations": {},
}
for hierarchy in [(16,), (8, 16), (2, 8, 16)]:
    args = argparse.Namespace(
        ep_size=16,
        ranks_per_node=16 if len(hierarchy) == 1 else 8,
        num_experts=128,
        slots_per_rank=16,
        hidden_size=2048,
        bytes_per_element=2,
        inter_ms_per_byte=6.765449326279194e-8,
        intra_ms_per_byte=5.02482606728045e-9,
        mid_ms_per_byte=2e-8,
        compute_ms_per_assignment=2.82807e-5,
        route_ms_per_assignment=8.746548178958447e-5,
        communication_phase_multiplier=3.1,
        compute_phase_multiplier=4.19,
        hierarchy_group_sizes=hierarchy,
    )
    evaluator = HybridEvaluator(args)
    base = np.array([(e // 8) * 16 + e % 8 for e in range(128)])
    for kind in ["identity", "replicated"]:
        lut = np.tile(base, (16, 1))
        if kind == "replicated":
            # A second copy occupies the upper eight slots on the opposite half of EP ranks.
            # Prefer the node-local copy; every LUT entry names one of these two valid copies.
            replica = ((np.arange(128) // 8 + 8) % 16) * 16 + 8 + np.arange(128) % 8
            for r in range(16):
                lut[r] = np.where((base // 16) // 8 == r // 8, base, replica)
        counts = []
        for r, route in enumerate(routes):
            physical = torch.tensor(lut[r])[route]
            counts.append(evaluator.planner._local_packed_counts(physical).numpy())
            counts.append(evaluator.planner._local_packed_assignment_counts(physical).numpy())
        result["evaluations"][str(hierarchy) + kind] = {
            "cost": asdict(evaluator.evaluate([routes], lut)),
            "counts": [digest(v) for v in counts],
        }
# Bounded deterministic optimizer on actual source-conditioned profile statistics.
args.hierarchy_group_sizes = (8, 16)
evaluator = HybridEvaluator(args)
topology = PlaceMoETopology(ep_size=16, ranks_per_node=8, num_experts=128, slots_per_rank=16)
config = OptimizerConfig(
    topology=topology,
    primary_slots_per_rank=8,
    node_omega=args.inter_ms_per_byte
    * args.hidden_size
    * args.bytes_per_element
    * args.communication_phase_multiplier,
    rank_omega=args.intra_ms_per_byte
    * args.hidden_size
    * args.bytes_per_element
    * args.communication_phase_multiplier,
    gamma=args.compute_ms_per_assignment * args.compute_phase_multiplier,
    rounds=2,
    assignment_iterations=4,
    node_exchange_limit=2,
    rank_exchange_limit=2,
    mapping_sweep_limit=2,
    seed=20260728,
)
res = optimize_replica_allocation(
    statistics,
    np.tile(np.arange(128), 2),
    config,
    lambda plan: evaluator.evaluate([routes], plan.source_logical_to_physical).total_ms,
)
result["optimizer"] = [
    {
        "layout": v.plan.slot_to_logical.tolist(),
        "owners": v.plan.owner_slots.tolist(),
        "mapping": v.plan.source_logical_to_physical.tolist(),
        "cost": v.cost,
        "round": v.round_index,
        "mapping_changes": v.mapping_changes,
    }
    for v in res.candidates
]
options.output.parent.mkdir(parents=True, exist_ok=True)
options.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
print("Replay complete:", result["snapshot"], len(res.candidates), "candidates")

if options.reference is not None:
    reference = json.loads(options.reference.read_text())
    if result != reference:
        raise SystemExit("Replay differs from the reference; inspect the two JSON outputs.")
    print("Exact reference parity: PASS")
