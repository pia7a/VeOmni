"""Compare retained planning paths on deterministic CPU fixtures across revisions.

Use this same script in two source environments. Only measured wall/device time
is excluded; candidate order, predicted costs, actions and all route tables are
compared. These small fixtures complement the full archived-profile replay.
"""

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from placemoe.planner_config import _configure_search, _parse_args, _validate_configuration
from placemoe.planner_search import _plan_layer
from veomni.distributed.moe.hiermoe import runtime_settings
from veomni.distributed.moe.hiermoe.core_planner import CoReMoEPlanner
from veomni.distributed.moe.hiermoe.expert_swap import ExpertSwapManager
from veomni.distributed.moe.hiermoe.greedy_planner import GreedyCommunicationPlanner
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.placemoe import LayerPlan
from veomni.distributed.moe.hiermoe.planner import CurrentRoutePlanner
from veomni.distributed.moe.hiermoe.topology import Hierarchy


def normalize(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items() if key not in {"planner_ms", "winner_planner_ms"}}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def verify(root):
    generator = torch.Generator().manual_seed(20260922)
    samples = []
    for step in (1, 2):
        routes = [torch.stack([torch.randperm(8, generator=generator)[:2] for _ in range(16)]) for _ in range(4)]
        directory = root / f"step{step:04d}"
        directory.mkdir()
        torch.save(
            {"format": "hiermoe-local-route-bundle-v1", "ep_size": 4, "routes_by_rank": routes},
            directory / "layer00_call0_all_ranks.pt",
        )
        samples.append(routes)
    results = {}
    for ranks_per_node in (2, 4):
        for slots in (2, 4):
            for fast in (False, True):
                sys.argv = [
                    "planner",
                    "--route-root",
                    str(root),
                    "--output-layout",
                    str(root / "layout.json"),
                    "--output-report",
                    str(root / "report.json"),
                    "--ep-size",
                    "4",
                    "--num-experts",
                    "8",
                    "--ranks-per-node",
                    str(ranks_per_node),
                    "--slots-per-rank",
                    str(slots),
                    "--layers",
                    "1",
                    "--workers",
                    "1",
                    "--candidate-workers",
                    "1",
                    "--partition-restarts",
                    "1",
                    "--alternations",
                    "2",
                    "--replica-candidate-limit",
                    "1",
                    "--partition-iterations",
                    "2",
                    "--assignment-iterations",
                    "2",
                    "--lut-iterations",
                    "2",
                    "--community-shortlist",
                    "1",
                    "--community-sweeps",
                    "2",
                ] + (["--fast-approx"] if fast else [])
                args = _parse_args()
                _configure_search(args)
                capacity = _validate_configuration(args)
                key = f"node{ranks_per_node}/slots{slots}/fast{fast}"
                result = _plan_layer(0, args=args, capacity=capacity)
                results[f"{key}/full"] = normalize(result)
                args.input_layout = root / "incumbent.json"
                args.input_plans = {
                    args.layer_name_template.format(layer=0): LayerPlan(
                        slot_to_logical=result[1], owner_slots=result[2], source_logical_to_physical=result[3]
                    )
                }
                results[f"{key}/incumbent"] = normalize(_plan_layer(0, args=args, capacity=capacity))
                args.update_mode = "mapping"
                _configure_search(args)
                results[f"{key}/mapping"] = normalize(_plan_layer(0, args=args, capacity=capacity))
    for planner_type in (CurrentRoutePlanner, CoReMoEPlanner, GreedyCommunicationPlanner):
        for groups in ((4,), (2, 4)):
            for replicas in (0, 2):
                # CoRe requires a summary from every rank. Repeat a deterministic
                # local payload to exercise its CPU algorithm with a synthetic
                # gather callback; this does not validate distributed collectives.
                extra = (
                    {"gather_fixed": lambda payload: payload.unsqueeze(0).repeat(4, 1)}
                    if planner_type is CoReMoEPlanner
                    else {}
                )
                planner = planner_type(
                    hierarchy=Hierarchy(ep_size=4, group_sizes=groups, source="refactor-fixture"),
                    perf_model=HierMoEPerfModel.default(),
                    hidden_size=16,
                    bytes_per_element=2,
                    slots_per_rank=4,
                    forward_compute_per_assignment=0.01,
                    **extra,
                )
                layout = torch.tensor([0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1, 6, 7, -1, -1])
                owners = torch.tensor([0, 1, 4, 5, 8, 9, 12, 13])
                plan = planner.plan(
                    samples[0][0],
                    layout,
                    owners,
                    source_ranks=0,
                    max_swaps=1,
                    max_replicas=replicas,
                    step=2,
                    layer_seed=17,
                )
                # PlacementPlan's top-level *_ms fields are observed timings, while
                # its nested baseline/final costs are predictions and remain intact.
                stable = {key: value for key, value in asdict(plan).items() if not key.endswith("_ms")}
                results[f"{planner_type.__name__}/{groups}/replicas{replicas}"] = normalize(stable)
    for scope in ("rank", "node"):
        with patch.multiple(
            runtime_settings,
            _FORWARD_REUSE_COVER_SERVICE_SCOPE=scope,
            _FORWARD_REUSE_COVER_COMPUTE_WEIGHT=3.5,
            _ONLINE_LUT_START_STEP=9,
        ):
            manager = ExpertSwapManager(
                ep_group=None,
                ep_size=4,
                ep_rank=0,
                expert_swap_interval=1,
                expert_swap_max_pairs_per_layer=0,
                redundant_slot_increment_per_device=0,
                max_replica_rounds=0,
                smooth_max_gamma=10.0,
                hierarchy=Hierarchy(ep_size=4, group_sizes=(2, 4), source="refactor-fixture"),
                perf_model=HierMoEPerfModel.default(),
            )
        # Metrics must retain construction-time values even after settings change.
        manager._begin_metrics_step(1)
        results[f"runtime_metrics/{scope}"] = manager.placement_metrics()
        manager.shutdown_pipeline()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="placemoe-paths-") as directory:
        result = verify(Path(directory))
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n")
    if args.reference is not None and args.output.read_bytes() != args.reference.read_bytes():
        raise AssertionError("Planning outputs differ from the reference revision.")
    print(f"Verified {len(result)} planning scenarios.")


if __name__ == "__main__":
    main()
