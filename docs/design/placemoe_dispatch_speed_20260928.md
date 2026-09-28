# PlaceMoE dispatch overhead reduction, 2026-09-28

Baseline: `2eed88f`. This change replaces per-expert NPU `nonzero` scans
with the existing token-permute sorting helper, and derives expert counts
from sorted IDs with `npu_moe_compute_expert_tokens`. The integer counts,
stable assignment order, inverse permutation, routes, deduplication,
collective sequence, accumulation precision and chunk sizes are unchanged.
There are no new configuration fields or runtime modes. CPU/CUDA grouping
keeps its existing implementation.

## Single-node measurements

Eight idle Ascend 910B1 devices, CANN 9.0, torch 2.9.0,
torch_npu 2.9.0.post2. The committed single-layer benchmark uses EP=8,
logical hierarchy [4, 8], 32,768 local tokens, hidden size 4,096,
intermediate size 1,536, 128 experts, top-k=8, BF16, identity placement,
and no redundant experts. This does not model physical inter-node links
or a complete 128K Ulysses training workload.

For each iteration take the maximum across eight ranks, then the median
across measured iterations. Ordinary runs use three warmups and 7–10
measured iterations. Stage profiling runs are separate, with explicit
synchronization and two warmups followed by 3–5 measured iterations.

| Measurement | Original | Final |
| --- | ---: | ---: |
| Forward, ordinary run | 225.7–226.5 ms | 221.9 ms |
| Backward, ordinary run | 202.4–203.2 ms | 202.1 ms |
| Forward + backward, ordinary run | 428.9–429.2 ms | 423.0 ms |
| Maximum allocated memory | 11.205 GiB | 11.204 GiB |
| Maximum reserved memory | 15.256 GiB | 15.256 GiB |
| Dispatch, synchronized profiling | 98.8 ms | 91.6 ms |
| Dispatch final grouping, device events | 11.84 ms | 7.82 ms |
| Combine, synchronized profiling | 107.6 ms | 105.2 ms |

Dispatch improves about 7.2% in these profiling runs; final grouping
improves about 34%. End-to-end observed improvement is only about 1.4%.
These are limited measurements, not a statistical performance guarantee.
Combine was **not modified**: its small timing difference cannot be
attributed to this change. It remains a bottleneck. Column-copy casting,
contiguous scratch, index dtype changes and public index-add alternatives
did not show useful improvement and were not retained. No new reduction
kernel or accumulation-order change is included.

## Correctness and tests

- The final implementation matches the first original run in all 40 full
  SHA256 hashes: output, input gradient, routing-weight gradient and both
  expert-weight gradients on each of eight ranks.
- The original implementation is not itself bitwise repeatable. In a
  second original run, rank 5's output hash changed. Comparing that run
  with final gives one differing element out of 134,217,728 on rank 5,
  maximum absolute difference 0.0009765625 (two BF16 ULP at that value),
  relative L2 error 3.47e-7. All 32 gradient hashes still match. Thus the
  matching first-run hashes are not a guarantee of deterministic output.
- Gradients use identical, output-independent upstream derivatives. This
  validates the tested VJP, not every possible loss-dependent gradient.
- Actual CPU/NPU sorting tests: **20 passed**, covering empty, single,
  skewed, large random and strided IDs, unused experts and both inverse
  modes. Counts and stable indices are compared exactly with CPU results.
- PlaceMoE CPU suite: before **224 passed, 2 skipped, 2 failed**; after
  **234 passed, 12 skipped, the same 2 failed**. The failures are the two
  existing CLI preparation tests requiring accelerator preflight in this
  CPU environment. NPU parameter cases are covered in the device run.
- Whole-suite baseline collection still stops at five existing environment
  errors (CI sample paths, e2e/model dependencies and missing Triton).
- `make quality`: lint passes; the same 24 unrelated formatting failures
  remain. Both modified Python files pass lint and formatting checks.
- Independent review of the final code and tests: **safe**.

## Reproduction and evidence

Use `scripts/placemoe/benchmark_moe_memory.py` in the same NPU environment:

```sh
python -m torch.distributed.run --nproc_per_node=8 \
  scripts/placemoe/benchmark_moe_memory.py \
  --mode hierarchical --tokens 32768 --warmup 3 --iterations 10 \
  --hash-tensors --save-output-rank 5 --output /experiment/result.json
```

Run original code from an immutable `git archive 2eed88f` checkout through
`PYTHONPATH`, then current code, serially on otherwise idle devices. Do not
compare synchronized diagnostic timing with ordinary timing.

[Compact evidence](placemoe_dispatch_speed_evidence_20260928.json) contains
all per-iteration maximum-rank metrics and the full original/repeated/final
hashes. Raw per-rank JSON, profiling instrumentation, microbenchmarks and
the saved rank-5 tensors remain locally under `/tmp/placemoe-longseq/`
(`speed-*` files). Profiling used `speed_profile.py`, a temporary wrapper
around the committed benchmark enabling existing internal events; no
additional production instrumentation was added.
