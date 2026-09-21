# PlaceMoE 结构重构与离线验证（2026-09-21）

基线：`c1b9945`，分支 `feature/placemoe-7002cb32`。对照本地论文第 III 节。

## 范围和行为约束

保留 PlaceMoE 的候选顺序、搜索预算、随机种子、平局处理、统计/成本计算顺序、布局与映射格式。采用原函数迁移和显式模块依赖，而非重写算法。所有通信、迁移、梯度和普通规划流水线钩子保留原实现。

`hiermoe/` Python 行数由 42,739 降到 35,149；顶层 `placemoe/` 由 3,154 降到 2,579。合计减少 8,165 行（17.8%，包含注释和空行）。这不是相对原版 VeOmni 的差异统计。

删除独立的 `cpu_planner.py`、`layer_owner_planner.py`、`npu_layer_owner_planner.py`、`legacy_batched_selector.py`、`forward_cover_planner.py`、`online_lut_planner.py`；删除旧 exact-pair 估计实现，以及顶层规划器的旧 structured/hyperedge 候选和不可达辅助函数。旧实验选项明确报错，不静默回退。

仍保留 `core_planner.py`、`planner.py`、`greedy_planner.py`、`statistical_scorer.py` 中当前初始化、路由和校准实际使用的实现。它们不是仅凭文件名就能删除的死代码。比如零副本校准仍需要 `_cpu_exact_planner_for_layer`，现在由校准模块持有。部分旧名称和退役入口错误提示用于清楚地拒绝不再支持的配置。

## 验证结果

| 检查 | 原提交 | 重构后 |
|---|---|---|
| CPU 专项 pytest | 142 passed / 2 skipped / 2 failed | 150 passed / 2 skipped / 2 failed |
| 新增退役入口检查 | 无 | 8 passed |
| 真实路由统计、计数、成本和候选输出 | 参考 JSON | 与参考 JSON 字节一致 |
| 改动文件 Ruff 检查与格式检查 | — | 通过 |
| `git diff --check` | — | 通过 |
| 全仓库 `make quality` | 已有 24 文件格式不符合当前 Ruff | 同样已有格式问题；改动文件单独检查通过 |

两个失败完全相同：`test_prepare_reuses_valid_artifacts_without_running_calibrators` 和 `test_prepare_generates_missing_runtime_and_model_artifacts`。两者进入 NPU 预检，在 CPU 环境因 `rms_norm_implementation='npu'` 不可用而失败，没有新增回归。全量 `pytest tests/` 的原始基线在收集阶段因缺少 Triton 失败，不能宣称全仓库测试通过。

CPU 验证使用独立镜像容器中的 PyTorch 2.9.0+cpu。测试启动器只绕过 `veomni`、`veomni.distributed`、`veomni.distributed.moe` 三个包的无关加速器注册初始化；没有替换被测算法、张量算子或通信计数。因而这些结果也不覆盖正常顶层包的加速器注册流程。

### 真实路由回放

输入为 `profile/processed/hiermoe_oracle_ep16_step0_layer24/step0_layer24_call0.pt`：EP16、128 专家、top-8、260,270 token。使用全部有效路由，不使用 padding。统计比较包括完整需求与共同选择矩阵的哈希。

以该输入构造 rank-only、两层和三层拓扑的 identity 与合法双副本 LUT，共六组**离线场景**，比较 192 个计数数组的哈希及各场景的通信/计算预测成本。这些拓扑变体不是六组独立硬件采样。两轮固定预算优化产生的六个候选，其 layout、owners、source LUT、mapping changes 和成本逐项一致。

真实训练没有执行。未验证 NPU/NCCL/HCCL 时序、死锁、流间依赖、真实梯度重叠或性能是否退化；旧 profile 不能证明这些性质。留出步、多模型和多种路由分布的覆盖仍有限。

详细结果见 [验证摘要](placemoe_refactor_evidence/summary.json)、[参考 JSON](placemoe_refactor_evidence/replay_reference.json)、[原始专项日志](placemoe_refactor_evidence/baseline-focused.txt) 和 [重构后专项日志](placemoe_refactor_evidence/after-focused.txt)。

### 复现

在具有 CPU PyTorch、NumPy、SciPy、scikit-learn 和 pytest 的环境中，从仓库根目录执行：

```bash
export PLACEMOE_SOURCE="$PWD"
export PYTHONPATH="$PWD/tests/tools/placemoe_cpu_bootstrap:$PWD"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python scripts/placemoe/verify_refactor.py   --snapshot profile/processed/hiermoe_oracle_ep16_step0_layer24/step0_layer24_call0.pt   --output /tmp/placemoe-replay.json   --reference docs/design/placemoe_refactor_evidence/replay_reference.json
```

参考结果固定于上述环境。跨 PyTorch/NumPy/BLAS 版本的浮点差异不能直接判定为算法回归，应在同环境比较两个版本。

## 文件职责清单

下表列出两个目录中所有 Python 文件。算法层的叶子模块没有反向依赖运行时管理器；管理器组合各执行职责。`runtime_settings.py` 是可变设置的唯一持有者。

### 公开入口：placemoe/

| 文件 | 行数 | 功能 |
|---|---:|---|
| [__init__.py](../../placemoe/__init__.py) | 51 | 包入口与公开接口导出。 |
| [bridge.py](../../placemoe/bridge.py) | 36 | VeOmni 训练器与 PlaceMoE 生命周期的接口桥接。 |
| [calibrate_network.py](../../placemoe/calibrate_network.py) | 551 | 公开的网络校准命令入口。 |
| [model_adapter.py](../../placemoe/model_adapter.py) | 178 | 标准专家参数表示与模型适配协议（内部文件转发公开接口）。 |
| [planner.py](../../placemoe/planner.py) | 236 | 稳定的规划命令入口、跨层执行编排和结果报告；不包含候选算法。 |
| [planner_candidates.py](../../placemoe/planner_candidates.py) | 647 | 确定性候选构造及布局/映射转换。 |
| [planner_config.py](../../placemoe/planner_config.py) | 390 | 规划 CLI 参数、搜索预算、容量和层级系数。 |
| [planner_search.py](../../placemoe/planner_search.py) | 490 | 逐层候选编排、现有计划比较与留出路由评估。 |

### 运行时：hiermoe/

| 文件 | 行数 | 功能 |
|---|---:|---|
| [__init__.py](../../veomni/distributed/moe/hiermoe/__init__.py) | 80 | 包入口与公开接口导出。 |
| [all_to_all.py](../../veomni/distributed/moe/hiermoe/all_to_all.py) | 2593 | 分层去重 dispatch/combine、反向通信及上下文；通信算法和流水线钩子未改。 |
| [bridge.py](../../veomni/distributed/moe/hiermoe/bridge.py) | 111 | VeOmni 训练器与 PlaceMoE 生命周期的接口桥接。 |
| [core_planner.py](../../veomni/distributed/moe/hiermoe/core_planner.py) | 2561 | 当前运行时仍调用的副本配额、映射和评分支持；保留其行为。 |
| [expert_swap.py](../../veomni/distributed/moe/hiermoe/expert_swap.py) | 931 | 管理器构造与调度入口，组合各功能模块；不再集中所有实现。 |
| [greedy_planner.py](../../veomni/distributed/moe/hiermoe/greedy_planner.py) | 4226 | 初始化/运行时仍调用的布局评分与槽位规划；复用独立 traffic 计数。 |
| [metrics.py](../../veomni/distributed/moe/hiermoe/metrics.py) | 85 | 运行时指标记录、读取与清空。 |
| [oracle.py](../../veomni/distributed/moe/hiermoe/oracle.py) | 1078 | 路由快照采集、读取和离线通信分析。 |
| [perf_model.py](../../veomni/distributed/moe/hiermoe/perf_model.py) | 493 | 硬件通信性能模型和启动校准。 |
| [planner.py](../../veomni/distributed/moe/hiermoe/planner.py) | 2224 | 当前运行时布局动作、计划类型和初始化评分/规划支持。 |
| [routing.py](../../veomni/distributed/moe/hiermoe/routing.py) | 42 | 层级通信的路由辅助接口。 |
| [runtime_artifact.py](../../veomni/distributed/moe/hiermoe/runtime_artifact.py) | 382 | 初始布局文件、层名匹配、静态布局安装及既有序列化格式。 |
| [runtime_calibration.py](../../veomni/distributed/moe/hiermoe/runtime_calibration.py) | 1665 | 运行期计时、代价模型拟合/验证、自动校准及 CPU 评分器构造。 |
| [runtime_checkpoint.py](../../veomni/distributed/moe/hiermoe/runtime_checkpoint.py) | 188 | 物理布局与映射的检查点保存、兼容性校验和恢复。 |
| [runtime_gradients.py](../../veomni/distributed/moe/hiermoe/runtime_gradients.py) | 1373 | 副本梯度聚合、重叠窗口、梯度归一化掩码及相关槽位约束。 |
| [runtime_hot_update.py](../../veomni/distributed/moe/hiermoe/runtime_hot_update.py) | 600 | 异步规划进程启动、路由采集、结果校验和布局/映射原子发布。 |
| [runtime_migration.py](../../veomni/distributed/moe/hiermoe/runtime_migration.py) | 1012 | 参数和优化器状态的槽位迁移、打包、传输与发布。 |
| [runtime_pipeline.py](../../veomni/distributed/moe/hiermoe/runtime_pipeline.py) | 348 | 设备流、事件窗口、微步状态和线程/进程组生命周期。 |
| [runtime_planning.py](../../veomni/distributed/moe/hiermoe/runtime_planning.py) | 853 | 保留的初始化规划与 collective/score 流水线；退役入口显式报错。 |
| [runtime_routing.py](../../veomni/distributed/moe/hiermoe/runtime_routing.py) | 599 | 专家注册、物理槽位路由、LUT 查表和路由记录。 |
| [runtime_settings.py](../../veomni/distributed/moe/hiermoe/runtime_settings.py) | 336 | 运行时设置的唯一持有者、环境参数和配置安装。 |
| [runtime_tensors.py](../../veomni/distributed/moe/hiermoe/runtime_tensors.py) | 880 | 专家参数扩展、优化器状态访问和底层张量传输原语。 |
| [runtime_types.py](../../veomni/distributed/moe/hiermoe/runtime_types.py) | 536 | 专家层状态、迁移缓冲、异步任务和梯度同步记录类型。 |
| [state.py](../../veomni/distributed/moe/hiermoe/state.py) | 826 | 全局训练状态、模型/优化器绑定以及生命周期入口。 |
| [statistical_scorer.py](../../veomni/distributed/moe/hiermoe/statistical_scorer.py) | 2950 | 保留的槽位规划器依赖的统计评分和候选增量计数。 |
| [topology.py](../../veomni/distributed/moe/hiermoe/topology.py) | 94 | 通信层级结构定义与推断。 |
| [traffic.py](../../veomni/distributed/moe/hiermoe/traffic.py) | 676 | 精确去重通信计数、非去重 assignment 计数及校准成本；不包含候选搜索。 |
| [triton_segment_sum.py](../../veomni/distributed/moe/hiermoe/triton_segment_sum.py) | 106 | 统计评分所需的可选 Triton 分段归约。 |

### 论文算法与准备流程：hiermoe/placemoe/

| 文件 | 行数 | 功能 |
|---|---:|---|
| [__init__.py](../../veomni/distributed/moe/hiermoe/placemoe/__init__.py) | 110 | 包入口与公开接口导出。 |
| [allocation.py](../../veomni/distributed/moe/hiermoe/placemoe/allocation.py) | 109 | 论文副本预算分配和有界动态规划候选列表。 |
| [artifacts.py](../../veomni/distributed/moe/hiermoe/placemoe/artifacts.py) | 95 | 布局/映射 artifact 构造、版本和合法性校验。 |
| [calibration.py](../../veomni/distributed/moe/hiermoe/placemoe/calibration.py) | 1197 | 持久化校准数据、拟合、作用域校验与性能预测。 |
| [cli.py](../../veomni/distributed/moe/hiermoe/placemoe/cli.py) | 1216 | prepare/doctor 等用户命令和配置检查。 |
| [mapping.py](../../veomni/distributed/moe/hiermoe/placemoe/mapping.py) | 810 | 来源相关的副本映射初始化、逐项和分组映射优化。 |
| [materialize.py](../../veomni/distributed/moe/hiermoe/placemoe/materialize.py) | 139 | 将实例布局和映射转换为物理槽位计划。 |
| [model_adapter.py](../../veomni/distributed/moe/hiermoe/placemoe/model_adapter.py) | 34 | 标准专家参数表示与模型适配协议（内部文件转发公开接口）。 |
| [optimizer.py](../../veomni/distributed/moe/hiermoe/placemoe/optimizer.py) | 383 | 固定预算下的布局—映射交替优化；算法实现未改。 |
| [partition.py](../../veomni/distributed/moe/hiermoe/placemoe/partition.py) | 422 | 容量约束的谱嵌入、分配和交换细化。 |
| [placement.py](../../veomni/distributed/moe/hiermoe/placemoe/placement.py) | 517 | 分层实例布局、节点分配和 rank 容量/唯一性修复。 |
| [preparation.py](../../veomni/distributed/moe/hiermoe/placemoe/preparation.py) | 373 | 校准准备阶段、artifact 复用和启动前检查。 |
| [route_replay.py](../../veomni/distributed/moe/hiermoe/placemoe/route_replay.py) | 233 | 完整 token 路由加载及留出样本成本评估。 |
| [seeds.py](../../veomni/distributed/moe/hiermoe/placemoe/seeds.py) | 48 | 确定性比较布局种子。 |
| [statistics.py](../../veomni/distributed/moe/hiermoe/placemoe/statistics.py) | 140 | 来源条件需求、共同选择统计及逻辑专家到副本统计投影。 |
| [types.py](../../veomni/distributed/moe/hiermoe/placemoe/types.py) | 182 | 拓扑、统计数据、布局/映射计划及其不变量。 |

### 异步规划支持：hiermoe/placemoe/runtime/

| 文件 | 行数 | 功能 |
|---|---:|---|
| [__init__.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/__init__.py) | 41 | 包入口与公开接口导出。 |
| [config.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/config.py) | 551 | 规范的 typed 配置、路径、校准参数及资源约束。 |
| [controller.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/controller.py) | 50 | 单个异步规划任务的生命周期。 |
| [cpu_affinity.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/cpu_affinity.py) | 289 | 训练与规划进程 CPU 亲和性分配。 |
| [planner_process.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/planner_process.py) | 243 | 规划命令构造、进程启动与终止。 |
| [planner_supervisor.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/planner_supervisor.py) | 70 | 规划子进程的信号转发与父进程退出处理。 |
| [scheduler.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/scheduler.py) | 49 | 布局与映射独立更新周期的状态机。 |

## 维护和边界

运行时功能使用 mixin 分组以保持原有管理器身份、调用顺序和状态布局。它们仍共同操作管理器状态，不应被当成可独立实例化的服务。进一步替换为组合对象需要单独的状态所有权设计和分布式验证，不在这次行为保持重构内。

新的算法改动应与本次结构变化分开提交和验证。不要因某个实现文件仍然较长，就删除当前初始化或校准的调用路径。
