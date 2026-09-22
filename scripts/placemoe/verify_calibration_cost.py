"""Compare calibration statistics with the pre-extraction planner in another checkout.

Run with --legacy on revision 6ffcd36, then use --reference on the current tree.
Only deterministic counts and costs are serialized; no device timing is involved.
"""

import argparse
import json
from pathlib import Path

import torch

from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.topology import Hierarchy


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--legacy", action="store_true")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
if args.legacy:
    from veomni.distributed.moe.hiermoe.greedy_planner import GreedyCommunicationPlanner as Model
else:
    from veomni.distributed.moe.hiermoe.calibration_cost import CalibrationCostModel as Model
out = []
for sizes in [(8,), (2, 8), (2, 4, 8)]:
    model = Model(
        hierarchy=Hierarchy(ep_size=8, group_sizes=sizes, source="fixture"),
        perf_model=HierMoEPerfModel.default(),
        hidden_size=128,
        bytes_per_element=2,
        slots_per_rank=4,
        smooth_max_gamma=10.0,
    )
    g = torch.Generator().manual_seed(401)
    routes = torch.randint(0, 32, (8, 3, 97, 4), generator=g)
    packed = torch.stack([model._local_packed_counts(r) for r in routes])
    assignments = torch.stack([model._local_packed_assignment_counts(r) for r in routes])
    out.append(
        [
            packed.tolist(),
            assignments.tolist(),
            [t.tolist() for t in model._communication_cost_details(packed.sum(0))],
            [t.tolist() for t in model._source_aware_communication_cost_details(packed)],
        ]
    )
args.output.write_text(json.dumps(out, sort_keys=True))
if args.reference is not None and args.output.read_bytes() != args.reference.read_bytes():
    raise AssertionError("Calibration counts or costs differ from the baseline.")
print("Verified calibration counts and costs for three hierarchy depths.")
