"""Single-layer Ascend EP benchmark; run with torchrun on eight idle devices.

Use --mode baseline or hierarchical with identical dimensions and seeds. The
hierarchical mode isolates the production dispatcher with a fixed identity
placement and no redundant slots, planner, calibration, or gradient replicas.
The [4, 8] hierarchy is logical: its timings do not model an inter-node link.
"""

import argparse
import gc
import hashlib
import importlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from veomni.distributed.moe.hiermoe import state as runtime_state
from veomni.distributed.moe.hiermoe.perf_model import HierMoEPerfModel
from veomni.distributed.moe.hiermoe.state import HierMoEState
from veomni.distributed.moe.hiermoe.topology import Hierarchy
from veomni.ops.kernels.moe.npu_group_gemm import npu_ep_fused_moe_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "hierarchical"), required=True)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=1536)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-tensors", type=Path)
    parser.add_argument(
        "--hash-tensors", action="store_true", help="Hash complete outputs and gradients after timing."
    )
    parser.add_argument(
        "--save-output-rank", type=int, help="Save one rank's complete output after the last iteration."
    )
    parser.add_argument("--memory-fraction", type=float, default=1.0)
    parser.add_argument("--diagnose", action="store_true", help="Synchronize stages to attribute memory peaks.")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("npu", local_rank)
    torch.npu.set_device(device)
    torch.npu.set_per_process_memory_fraction(args.memory_fraction, device)
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    size = dist.get_world_size()
    if size != 8 or args.experts % size:
        raise ValueError("This benchmark requires EP=8 and experts divisible by 8.")
    torch.manual_seed(1701 + rank)
    torch.npu.manual_seed(1701 + rank)
    torch.npu.set_compile_mode(jit_compile=False)
    if args.mode == "hierarchical":
        runtime_state._STATE = HierMoEState(
            enable=True,
            token_dedup=True,
            expert_swap=False,
            expert_swap_interval=1,
            expert_swap_max_pairs_per_layer=0,
            redundant_slot_increment_per_device=0,
            max_slot_op_search_rounds=0,
            max_replica_rounds=0,
            expert_swap_mode="step",
            debug_validate=False,
            current_step=0,
            log_interval=100000,
            hierarchy=Hierarchy(size, (4, 8), "benchmark", size),
            perf_model=HierMoEPerfModel.default(),
            active=True,
            placement_mapping_enabled=False,
        )
    else:
        runtime_state._STATE = None
    # Generate routing on CPU so device-specific RNG implementations do not
    # change the routes between baseline and hierarchical execution.
    routes = torch.rand(args.tokens, args.experts).topk(args.top_k, dim=-1).indices.to(device)
    weights = torch.rand(args.tokens, args.top_k)
    weights = (weights / weights.sum(-1, keepdim=True)).to(device, torch.bfloat16).requires_grad_()
    x = torch.randn(args.tokens, args.hidden, dtype=torch.bfloat16, device=device).requires_grad_()
    w1 = (
        torch.randn(args.experts // size, 2 * args.intermediate, args.hidden, dtype=torch.bfloat16, device=device)
        * args.hidden**-0.5
    ).requires_grad_()
    w2 = (
        torch.randn(args.experts // size, args.hidden, args.intermediate, dtype=torch.bfloat16, device=device)
        * args.intermediate**-0.5
    ).requires_grad_()
    upstream = torch.randn_like(x) / args.tokens
    parameters = (x, weights, w1, w2)
    stage_samples = []
    peak_samples = []
    if args.diagnose:
        module = importlib.import_module("veomni.ops.kernels.moe.npu_group_gemm")
        communication = importlib.import_module("veomni.distributed.moe.hiermoe.all_to_all")

        def wrap(function, name):
            def measured(*positional, **keywords):
                torch.npu.synchronize()
                peak_samples.append((torch.npu.max_memory_allocated(), torch.npu.max_memory_reserved()))
                segment_start = len(peak_samples)
                torch.npu.reset_peak_memory_stats()
                before = torch.npu.memory_allocated()
                stage_start = time.perf_counter()
                result = function(*positional, **keywords)
                torch.npu.synchronize()
                stage_ms = (time.perf_counter() - stage_start) * 1000
                peak_samples.append((torch.npu.max_memory_allocated(), torch.npu.max_memory_reserved()))
                stage_samples.append(
                    {
                        "name": name,
                        "elapsed_ms": stage_ms,
                        "before_gib": before / 2**30,
                        "after_gib": torch.npu.memory_allocated() / 2**30,
                        "peak_gib": max(p[0] for p in peak_samples[segment_start:]) / 2**30,
                        "input_shape": list(positional[0].shape)
                        if positional and isinstance(positional[0], torch.Tensor)
                        else None,
                    }
                )
                return result

            return measured

        for name in (
            "rank_dedup_dispatch",
            "rank_dedup_combine",
            "npu_group_gemm",
            "_swiglu",
            "alltoall_dispatch",
            "alltoall_combine",
        ):
            setattr(module, name, wrap(getattr(module, name), name))
        name = "_index_add_dim0_cast_output"
        setattr(communication, name, wrap(getattr(communication, name), name))
    samples = []
    for iteration in range(args.warmup + args.iterations):
        stage_samples.clear()
        peak_samples.clear()
        for parameter in parameters:
            parameter.grad = None
        gc.collect()
        if iteration == 0 or iteration == args.warmup:
            torch.npu.empty_cache()
        dist.barrier()
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        initial = torch.npu.memory_allocated()
        start = time.perf_counter()
        output = npu_ep_fused_moe_forward(
            args.experts,
            weights,
            routes,
            x,
            None,
            None,
            w2,
            w1,
            dist.group.WORLD,
        )
        torch.npu.synchronize()
        forward_end = time.perf_counter()
        forward_peak = torch.npu.max_memory_allocated()
        if args.diagnose:
            forward_peak = max(forward_peak, max(p[0] for p in peak_samples))
        output.backward(upstream)
        torch.npu.synchronize()
        end = time.perf_counter()
        sample = {
            "iteration": iteration,
            "forward_ms": (forward_end - start) * 1000,
            "backward_ms": (end - forward_end) * 1000,
            "total_ms": (end - start) * 1000,
            "initial_gib": initial / 2**30,
            "forward_peak_gib": forward_peak / 2**30,
            "peak_gib": torch.npu.max_memory_allocated() / 2**30,
            "reserved_peak_gib": torch.npu.max_memory_reserved() / 2**30,
        }
        if args.diagnose:
            sample["stages"] = list(stage_samples)
            sample["peak_gib"] = max(sample["peak_gib"], forward_peak / 2**30)
            sample["reserved_peak_gib"] = max(sample["reserved_peak_gib"], max(p[1] for p in peak_samples) / 2**30)
        if rank == 0:
            print(json.dumps(sample), flush=True)
        if iteration >= args.warmup:
            samples.append(sample)
        if args.save_tensors and iteration == args.warmup + args.iterations - 1:
            args.save_tensors.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "output": output.detach().cpu(),
                    "input_grad": x.grad.cpu(),
                    "routing_grad": weights.grad.cpu(),
                    "w1_grad": w1.grad.cpu(),
                    "w2_grad": w2.grad.cpu(),
                },
                args.save_tensors / f"rank{rank}.pt",
            )
        if args.hash_tensors and iteration == args.warmup + args.iterations - 1:
            sample["sha256"] = {}
            for name, tensor in zip(
                ("output", "input_grad", "routing_grad", "w1_grad", "w2_grad"),
                (output, x.grad, weights.grad, w1.grad, w2.grad),
            ):
                digest = hashlib.sha256()
                chunk_rows = 1 if tensor.ndim == 3 else 1024
                for start in range(0, tensor.shape[0], chunk_rows):
                    block = tensor[start : start + chunk_rows].detach().cpu().contiguous()
                    digest.update(block.view(torch.uint8).numpy().tobytes())
                sample["sha256"][name] = digest.hexdigest()
        if args.save_output_rank == rank and iteration == args.warmup + args.iterations - 1:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(output.detach().cpu(), args.output.with_suffix(f".rank{rank}.pt"))
        del output
    all_samples = [None] * size
    dist.all_gather_object(all_samples, {"rank": rank, "samples": samples})
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"config": vars(args), "ranks": all_samples}, default=str, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
