# Existing-operator PlaceMoE optimization, 2026-09-28

Baseline: `b6c389d`. No custom CUDA/Ascend C kernel, new mode, or algorithm
change is introduced. Measurements use the same single-layer 8-device
Ascend 910B1 benchmark as the preceding dispatch report: EP8, logical
[4, 8] hierarchy, 32,768 local tokens, H4096, I1536, E128, top-k8, BF16,
identity placement and zero replicas. CANN9.0 / torch2.9 / torch_npu2.9.

## Trace findings

A warmed-up rank-0 CPU/NPU profiler trace identifies these device kernel
times across the whole forward/backward (sums, not critical-path times):
GroupedMatmul 94.1 ms, InplaceScatterAdd 19.1 ms, GatherV3 16.6 ms,
Cast 14.9 ms, Mul 13.9 ms, InplaceIndexAddWithSorted 12.4 ms,
Slice 11.1 ms, Cumsum 4.87 ms, Bincount 3.87 ms. HCCL task durations
sum to 183.8 ms; this is not a proof that all communication is exposed.
Wait-event durations overlap work and must not be added to kernel time.

The annotated weighted-combine backward launches 18.27 ms of device work.
It multiplies a gathered gradient by routing weights, then copies that
large temporary into an already allocated gradient tensor. Writing the
multiply directly to that destination removes the temporary and copy.
The original differentiable path is retained when autograd is recording
higher-order derivatives, since out= operators cannot construct that graph.

## Verified backward improvement

| Metric | Original | Direct output |
| --- | ---: | ---: |
| Weighted-backward isolated microbenchmark | 18.72 ms | 15.15 ms |
| Full forward | 218.33 ms | 217.76 ms |
| Full backward | 202.38 ms | 198.55 ms |
| Forward + backward | 420.52 ms | 417.53 ms |
| Peak allocated memory | 11.204 GiB | 11.204 GiB |

Ordinary benchmarks use three warmups and ten measured iterations. Each
reported metric is the median of per-iteration maxima over eight ranks.
The total improvement observed here is about 0.7%; forward is unchanged
by this patch, so its timing difference is noise. No full-model or
cross-node training speedup is claimed.

All 40 complete output/gradient hashes match. The fixed upstream gradient
is independent of the output; this validates that VJP, not every loss.
CPU combine tests: 47 passed, including a new second-derivative comparison.
Full PlaceMoE CPU tests: 235 passed, 12 skipped, the same two pre-existing
CLI accelerator-preflight failures (baseline: 234 passed). Independent
review: safe. Modified files pass Ruff; repository-wide formatting still
reports the same 24 unrelated files.

## Contiguous integer prefix scans

The tall, narrow boolean hit matrices were scanned along their first axis.
The NPU path now transposes and materializes that small matrix, scans along
its contiguous last axis, then presents the result in the original logical
shape. All five 2D/3D hierarchy call sites use the same helper. The int32
prefix values and unique-token ordering remain exact; no floating-point
reduction changes. CPU/CUDA execution retains the original layout.

| Final measurement | Original b6c389d | Both changes |
| --- | ---: | ---: |
| Forward, ordinary run | 218.33 ms | 211.72 ms |
| Backward, ordinary run | 202.38 ms | 198.49 ms |
| Forward + backward, ordinary run | 420.52 ms | 409.93 ms |
| Peak allocated / reserved | 11.204 / 15.256 GiB | 11.204 / 15.256 GiB |
| Weighted backward device kernels, trace | 18.27 ms | 15.03 ms |
| AI-vector Cumsum device kernels, trace | 4.869 ms | 0.163 ms |
| TensorMove calls across the traced step | 16 | 0 |

The total observed improvement is about 2.5% against this iteration's
baseline. This is a limited single-node measurement, not a statistical
or full-training guarantee. Combine **forward** remains unchanged.
The extra transpose work is included in the ordinary timing. An isolated
small 127-by-16 matrix was slightly slower (0.123 to 0.185 ms); this layout
optimization targets the measured long-sequence bottleneck, and does not
promise improvement for every shape.

All 40 complete output/gradient hashes still match the original run.
Real CPU/NPU dispatch tests: **44 passed**, including both empty dimensions,
unused destinations, long prefixes and strided inputs. PlaceMoE CPU suite:
**247 passed, 24 skipped, the same two existing failures**. Additional skips
are NPU test parameters covered by the actual device test run. Both code
changes were independently reviewed as safe. Full training, physical
cross-node links and full three-level distributed execution were not tested.

## Evidence and reproduction

[Compact evidence](placemoe_operator_speed_evidence_20260928.json) includes
per-iteration metrics and complete output/gradient hashes. Run the existing
`scripts/placemoe/benchmark_moe_memory.py` under an immutable baseline
checkout via PYTHONPATH, then current code, on otherwise idle devices:

```sh
python -m torch.distributed.run --nproc_per_node=8 \
  scripts/placemoe/benchmark_moe_memory.py --mode hierarchical \
  --tokens 32768 --warmup 3 --iterations 10 --hash-tensors \
  --output /experiment/result.json
```

Raw logs, microbenchmarks, profiler wrapper and rank-0 traces remain in
`/tmp/placemoe-longseq/op-*`. The NPU profiler reported 24 incomplete memory
records; memory claims above use the separate unprofiled allocator
measurements, not those incomplete trace records.
