# PlaceMoE 长序列显存定位与 NPU 优化（2026-09-28）

在单节点 8 卡、逻辑两级拓扑下，确认零副本 PlaceMoE 的 combine 前向和反向会产生额外的大张量。修复后，32k 本地 token 场景的峰值 allocated 显存减少约 24.3%，单层前后向耗时减少约 13.0%。相同显存预算下，原始 baseline 通过、旧 PlaceMoE OOM、优化后通过。

这定位并修复了可复现的额外显存问题，但不等于复现用户完整 128k 训练。缺少用户模型、batch、EP 配置和原始堆栈，不能断言实际训练只有这一个原因。

## 实验范围和统计口径

- 基线源码：`384b240` 的独立 git archive。源码路径和日志分离；未修改历史 baseline。
- 设备：8 × Ascend 910B1，物理每卡 64 GiB；驱动报告 26.0.rc1。镜像 `placemoe:ascend-910b-cann9-torch2.9`，PyTorch 2.9.0+cpu、torch_npu 2.9.0.post2、CANN 9.0.0。
- 单层 MoE：EP=8，128 experts，top-k=8，hidden=4096，expert intermediate=1536，BF16，合并 gate/up 权重；输入、路由权重和两组专家权重均反向传播。
- 原始 baseline 和 PlaceMoE 调用同一个 `npu_ep_fused_moe_forward`，由 HierMoE 状态选择生产通信路径；专家 MLP 实现相同。
- PlaceMoE 固定 identity placement、零副本、逻辑层级 `[4,8]`，不运行规划器、校准或布局迁移。
- 路由由固定 CPU 随机种子生成，每个 token 选择不重复专家，权重归一化。各 rank 种子不同，各版本输入和参数相同。
- 每卡 16,384 或 32,768 tokens。它们对应此前讨论的序列分片量，但实验没有 Attention、实际 Ulysses 交换、FSDP、优化器状态或多层 activation checkpoint。
- 常规计时 warmup=2、测量=5。每次同步后计时，按每次迭代的最慢 rank 取耗时，再取五次中位数。显存取所有 rank、测量迭代的最大值。
- `peak_gib` 是 PyTorch allocator 的 peak allocated，`reserved_peak_gib` 是 peak reserved；均不是整卡进程外/HCCL 全部显存。`--diagnose` 增加阶段同步，仅用于定位，不能与常规耗时混用。
- 单节点 `[4,8]` 只模拟两级调用结构，没有跨节点慢链路。因此本实验不能验证论文跨节点加速比。

## 常规实测

| 每卡 tokens | 版本 | peak allocated / GiB | peak reserved / GiB | 前后向中位耗时 / ms |
| ---: | --- | ---: | ---: | ---: |
| 16,384 | 原始 baseline | 5.95 | 7.53 | 127.2 |
| 16,384 | 修改前 PlaceMoE | 7.92 | 10.11 | 246.1 |
| 16,384 | 优化后 PlaceMoE | 6.00 | 8.02 | 221.8 |
| 32,768 | 原始 baseline | 11.35 | 14.28 | 236.9 |
| 32,768 | 修改前 PlaceMoE | 14.80 | 17.43 | 491.3 |
| 32,768 | 优化后 PlaceMoE | 11.21 | 15.26 | 427.4 |

16k 场景 allocated 峰值减少约 24.2%，耗时减少约 9.9%。32k 场景优化后 allocated 接近 baseline，但 reserved 仍略高；两类指标不能混为一谈。

优化后仍比原始 baseline 慢。阶段同步诊断中，rank 0 的专家计算耗时接近，但原始 dispatch/combine 约 34/37 ms，优化后分层 dispatch/combine 约 105/103 ms。分层路由元数据构造、排序、split-size 的主机同步、额外通信阶段及聚合仍有开销。本次没有为了改善单节点数字而切换回原始通信路径，也没有改变放置算法。

## 根因与消融

### 1. dispatch 的历史缓冲区存活过长

原两级 dispatch 在构造最终专家输入时仍持有已完成阶段的 send/receive payload。现在在 paired collective 的两个 `wait()` 之后，以及下一阶段 `index_select` 完成之后释放不再使用的引用。索引选择反向需要索引和输入形状，无需源张量数值。

rank 0 的局部 dispatch 峰值从约 6.76 GiB 降到 4.42 GiB。但是仅此修改时，32k 整体峰值仍约 14.80 GiB；因此它不是本实验的整体峰值主因。

### 2. combine 完整 FP32 缓冲区与分块反向

旧实现虽然按行分块转换 source 到 FP32，仍整块分配 `[num_rows, hidden]` FP32 聚合输出。旧 autograd 在每块切片反向时还会还原完整 source 梯度，并通过 FP32 `index_select` 产生大临时张量。

新实现保留 FP32 累加和原行分块顺序，按 1024 列划分相互独立的归约；非加权聚合反向直接执行一次源精度 gather，不再构造逐块 slice backward 图。

### 3. 加权聚合的广播反向

旧实现显式生成完整 `expert_outputs * weights`，广播乘法反向还要生成 assignment×hidden 的乘积，再沿 hidden 维归约成每行一个路由梯度。

新实现前向按列生成加权值，随即累加；反向按 16,384 行分块计算路由权重梯度和专家输出梯度。每个权重梯度仍沿完整 hidden 维求和，保留乘法精度和求和边界。

| 32k 消融版本 | peak allocated / GiB | 前后向 / ms |
| --- | ---: | ---: |
| 修改前 | 14.80 | 491.3 |
| 仅提前释放 dispatch | 14.80 | 479.0 |
| 加上聚合直接 gather 反向 | 13.63 | 420.0 |
| 加上 FP32 前向按列分块 | 13.63 | 428.5 |
| 完整加权聚合优化 | 11.21 | 427.4 |

不同消融单独启动进程，耗时有运行间噪声。按列分块增加了 kernel 调用，并非每一步都使耗时下降；完整优化以显存峰值和数学行为为主要约束。

## 人为显存预算下复现 OOM

使用 `torch.npu.set_per_process_memory_fraction` 限制 allocator 预算，模拟完整训练只剩部分显存可用于该层；这不是在物理 64 GiB 全空闲条件下重现完整模型 OOM。

| fraction / 约允许 GiB | 原始 baseline | 旧 PlaceMoE | 优化后 PlaceMoE |
| --- | --- | --- | --- |
| 0.22 / 13.41 | OOM | OOM | 通过 |
| 0.24 / 14.63 | 通过 | OOM | 通过 |

0.22 是首次试验，预算对 baseline 也过紧，因此另用相同的 0.24 预算完成三方对照。0.24 时旧版在 `output.backward()` 中失败；rank 1 的堆栈明确指向旧 `_NpuIndexAddDim0.backward` 的 `grad_output.index_select`，申请 1 GiB，其余部分 rank 申请约 2 GiB 时失败。完整日志保存在证据目录。

## 正确性与回归

- 真实 NPU、8 ranks、每卡 16k tokens、hidden=1152、intermediate=64 的完整张量对照：输出、输入梯度、路由权重梯度、w1/w2 梯度全部逐元素一致。该配置同时跨越行和列分块边界。
- 与性能实验相同的 32k/H4096/I1536 配置：8 ranks 的所有四类梯度 SHA-256 一致，7 个 rank 的输出 SHA-256 一致。rank 5 的 134,217,728 个输出中有 1 个不同：[12084,682] 从 -0.17578125 变为 -0.1748046875，相差 1 个 BF16 ULP（0.0009765625），该 rank 相对 L2 误差为 3.61e-7；通过 `rtol=1/128, atol=1e-7` 的 BF16 容差检查。旧/旧、新/新重复运行各自完全稳定，不能将此归因于原算子随机波动。
- 数值消融定位：仅修改 gather 反向及 dispatch 生命周期时，40 组摘要全部匹配旧版；加入列分块后，40 组摘要全部匹配最终优化版。因此这 1 ULP 差异来自列分块归约，非缓冲区提前释放或反向公式错误。
- 上述完整梯度使用固定、独立于输出的随机 upstream 做 VJP 比较；梯度一致不代表输出相关训练 loss 的梯度也逐字节一致。列分块保持 FP32 精度及行块边界，但 NPU kernel 的内部归约顺序可能随宽度改变；本优化不保证所有输入上的 bitwise 前向等价。
- 小尺寸、多行分块的早期对照也保持逐元素一致；基线本身与原始 baseline 存在 BF16 归约路径差异，不能把这种既有差异当作本次优化引入。
- CPU PlaceMoE 完整集：223 passed、2 skipped、2 failed；两个失败仍是原有 CLI prepare 的 NPU preflight 环境限制。随后新增混合 dtype 用例单独 1 passed。新增测试覆盖 FP32/BF16/FP16、重复索引、空输入、非连续输入、行列分块、只训练一个输入及 FP32 routing weights + BF16 source。
- 24 个生产规划场景与历史 JSON 逐字节一致。
- 全仓 pytest 收集前后同样在 5 处因环境依赖失败。`make quality` lint 通过；format 仍有基线已有的 24 个文件。修改文件单独 lint/format 检查通过。

这里只验证单层固定放置，没有覆盖真实跨节点流依赖、副本同步、热迁移、完整模型收敛和端到端训练吞吐。

独立复核结论为 safe，已独立复算结果表，并检查数值边界、广播梯度及异步释放时机。OOM 日志以 `.txt` 保存，避免被仓库的 `.log` 忽略规则排除。

## 文件与复现

| 文件 | 职责 |
| --- | --- |
| `veomni/distributed/moe/hiermoe/all_to_all.py` | NPU 分列聚合、直接 gather 反向、加权聚合及 dispatch 生命周期优化 |
| `tests/distributed/test_hiermoe_combine.py` | 数值/梯度和分块反向回归 |
| `scripts/placemoe/benchmark_moe_memory.py` | 可复现的单层八卡 benchmark、阶段定位和完整张量摘要 |
| `docs/design/placemoe_long_sequence_evidence/` | 原始 JSON、OOM 日志、测试证据和汇总 |

在配置好 Ascend 依赖的八卡空闲环境运行，例如：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc_per_node=8 scripts/placemoe/benchmark_moe_memory.py --mode hierarchical --tokens 32768 --output /tmp/moe.json
```

将 `--mode` 改为 `baseline` 比较原始路径；加 `--memory-fraction 0.24` 复现预算对照，加 `--diagnose` 定位阶段峰值，加 `--hash-tensors` 在计时之后对完整输出和梯度计算摘要。用 `PYTHONPATH` 指向 `384b240` 的独立 checkout 可测修改前 PlaceMoE，benchmark 脚本使用本次新增版本。`after-*` 证据对应 gather 反向消融，`final-*` 对应列分块消融，最终实现的证据前缀是 `optimized-*`。
