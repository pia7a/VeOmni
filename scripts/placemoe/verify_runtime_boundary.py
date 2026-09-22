"""Compare preloaded placement, routing and checkpoint behavior across source trees.

Run this same script with each checkout on PYTHONPATH. All inputs are CPU tensors;
this checks metadata and route replay, not distributed transport correctness.
"""

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from veomni.distributed.moe.hiermoe import runtime_settings as settings
from veomni.distributed.moe.hiermoe.expert_swap import ExpertSwapManager
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.placemoe.artifacts import build_placemoe_artifact
from veomni.distributed.moe.hiermoe.placemoe.types import LayerPlan, PlaceMoETopology
from veomni.distributed.moe.hiermoe.state import _HierMoECheckpointReplay
from veomni.distributed.moe.hiermoe.topology import Hierarchy


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.gate_proj = nn.Parameter(torch.arange(18.0).reshape(3, 3, 2))
        self.up_proj = nn.Parameter(torch.arange(18.0).reshape(3, 3, 2) + 20)
        self.down_proj = nn.Parameter(torch.arange(18.0).reshape(3, 2, 3) + 40)


def verify(path):
    model = nn.Module()
    model.experts = Experts()
    weights = [parameter.clone() for parameter in model.parameters()]
    layout = (0, 1, 2, 2, 3, 0)
    owners = (0, 1, 3, 4)
    lut = ((0, 1, 2, 4), (5, 1, 3, 4))
    artifact = build_placemoe_artifact(
        {"experts": LayerPlan(slot_to_logical=layout, owner_slots=owners, source_logical_to_physical=lut)},
        PlaceMoETopology(ep_size=2, ranks_per_node=2, num_experts=4, slots_per_rank=3),
    )
    path.write_text(json.dumps(artifact))
    config = settings._PLACEMOE_RUNTIME_CONFIG
    with patch.multiple(
        settings,
        _INITIAL_LAYOUT_PATH=str(path),
        _HOT_UPDATE=False,
        _PLACEMOE_RUNTIME_CONFIG=replace(config, calibration=replace(config.calibration, auto_generate=False)),
    ):
        manager = ExpertSwapManager(
            ep_group=None,
            ep_size=2,
            ep_rank=0,
            expert_swap_interval=1,
            expert_swap_max_pairs_per_layer=0,
            redundant_slot_increment_per_device=1,
            max_replica_rounds=0,
            smooth_max_gamma=10.0,
            hierarchy=Hierarchy(ep_size=2, group_sizes=(1, 2), source="fixture"),
            perf_model=HierMoEPerfModel.default(),
            expert_swap_selector="hiermoe_greedy_cover_p1",
            fixed_pipeline_overlap=True,
            activation_checkpointing_enabled=True,
        )
        try:
            manager.register_model(model)
            for before, after in zip(weights, model.parameters(), strict=True):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            layer = manager.layers["experts"]
            selected = torch.tensor([[0, 2], [1, 3]])
            manager.configure_pipeline_microstep(1, 0, 1)
            manager.record_routing(
                layer_key="experts", selected_experts=selected, hidden_size=3, bytes_per_element=4, step=1
            )
            replay = _HierMoECheckpointReplay()
            forward = manager.map_logical_to_physical("experts", selected, checkpoint_replay=replay)
            checkpoint = manager.state_dict()
            manager.maybe_swap(1)
            # A new valid source LUT may choose another copy. Recompute must use
            # the physical routes captured by the original forward occurrence.
            layer.source_logical_to_physical = torch.tensor(((5, 1, 3, 4), (0, 1, 2, 4)))
            layer._device_source_mapping_cache.clear()
            replay.reset_reader()
            recompute = manager.map_logical_to_physical(
                "experts", selected, checkpoint_recompute=True, checkpoint_replay=replay
            )
            torch.testing.assert_close(forward, recompute, rtol=0, atol=0)
            manager.load_state_dict(checkpoint)
            assert manager.state_dict() == checkpoint
            return {
                "layout": layer.slot_to_logical.tolist(),
                "owners": layer.logical_to_physical.tolist(),
                "forward": forward.tolist(),
                "recompute": recompute.tolist(),
                "checkpoint": checkpoint,
                "source_lut_after_load": layer.source_logical_to_physical.tolist(),
                "checkpoint_contains_source_lut": "source_logical_to_physical" in checkpoint["layers"]["experts"],
                "weights": [parameter.tolist() for parameter in model.parameters()],
            }
        finally:
            manager.shutdown_pipeline()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
with tempfile.TemporaryDirectory(prefix="placemoe-runtime-") as directory:
    result = verify(Path(directory) / "initial.json")
args.output.write_text(json.dumps(result, sort_keys=True) + "\n")
if args.reference is not None and args.output.read_bytes() != args.reference.read_bytes():
    raise AssertionError("Runtime metadata, routes, checkpoint, or weights differ from the baseline.")
print(
    "Verified preloaded placement, routes, recompute and layout metadata parity; recorded existing Source LUT omission."
)
