# PlaceMoE 第二轮精简与行为验证（2026-09-22）

基线提交：`880818a`。延续第一轮重构，保持已支持配置下的算法、候选顺序、平局处理、成本公式以及通信窗口。

## 实际减少的复杂度

两个目标目录的 Python 行数（包含注释和空行）由 **37,728 降至 35,439**，本轮减少 **2,289 行（6.1%）**。其中 `hiermoe/` 为 35,149 → 32,960，顶层 `placemoe/` 为 2,579 → 2,479。相对首次重构前的 45,893 行，累计减少 10,454 行（22.8%）。未新增生产模块。

- 删除已经被构造器拒绝的 CPU/process、NPU layer-owner、Forward Cover、Online LUT、exact-pair 和 legacy-batched 的内部调度分支、专用任务状态及错误占位方法；入口仍清楚拒绝退役模式。
- 删除无调用的 Forward LUT 评分器、旧路由摘要工具及旧候选评分辅助实现。保留当前 CurrentRoute/CoRe/Greedy 规划器的可达算法。
- 删除无调用的旧梯度 bucket/wave 同步、槽位操作和优化器状态交换辅助闭包；保留当前梯度计划、迁移、校准和通信窗口。
- 删除 NPU Forward 中仅用于退役模式的统计钩子及其缓存字段。原调用在允许的配置下立即返回，不承担训练通信或事件记录。
- 社区候选现在明确要求其现用初始映射，删除只有历史调用才会使用的可选参数组合和 node-proxy LUT 构造。原始映射与细化映射的评估顺序不变。
- 20 个仅供旧日志指标使用的共享属性收束为一个构造时指标快照。保留 20 个指标键、数值及原配置校验，后续设置变化不会改写快照。

构造器、调度和状态清理是本轮行为等价审查的重点。四个规划/评分模块保留的 **187 个函数体 AST 不变**；另一个构造函数仅删除未被使用的 rank-distance 缓存。AST 相同不是硬件正确性的证明。

## 无多节点 NPU 验证

| 检查 | 基线 `880818a` | 本轮结果 |
|---|---|---|
| 所有 `tests/distributed/test_placemoe*.py` | 168 passed / 2 skipped / 3 failed | 169 passed / 2 skipped / 2 failed |
| 完整历史 EP16 profile 回放 | 原有参考结果 | 逐字节一致 |
| 规划入口和初始化规划器 | 同一脚本在固定 git archive 快照运行 | 36 个场景逐字节一致 |
| 构造时历史指标快照 | rank/node 两种作用域 | 2 个场景逐字节一致 |
| 独立调用链与差异审阅 | — | safe；无删除符号的悬空生产调用 |
| 改动文件 lint / 格式 / diff 检查 | — | 通过 |
| 全仓库 `make quality` | 已有 24 个格式问题文件 | 同样 24 个文件；lint 通过 |

本轮扩大了测试文件范围，因此通过数不能直接与第一轮的 150 项通过数比较。修复的一项基线失败是测试对进程组工厂的补丁打到了配置模块；现在补丁实际定义模块。另两处测试改为直接补丁 `torch.distributed`，不再依赖配置模块的无用导入。生产逻辑没有为通过测试而绕过 NPU 前置检查。

剩余失败仍是 `test_prepare_reuses_valid_artifacts_without_running_calibrators` 和 `test_prepare_generates_missing_runtime_and_model_artifacts`，因为 CPU 环境无法满足 NPU RMSNorm 预检。完整仓库测试在第一轮因缺少 Triton 无法收集，本轮不宣称全量测试通过。

38 个新增场景由确定性 EP4/E8 小样本产生：24 个完整搜索/已有方案/仅映射组合、12 个 CurrentRoute/CoRe/Greedy CPU 规划场景，以及 2 个配置变化后的指标快照。覆盖单节点/两层拓扑、无副本/双副本容量和完整/快速搜索，包含社区候选。CoRe 使用显式的合成 gather 回调；它验证相同输入及回调下的算法输出，不验证真实 collective。参考数据仅排除实际运行耗时，保留候选顺序、预测成本、布局、映射、动作和路由。

完整历史回放沿用 EP16/E128/Top-8 的 260,270 个有效 token、六种离线拓扑/映射组合、192 个计数数组和六个优化候选，SHA256 与第一轮参考完全相同。

证据：[摘要](placemoe_cleanup_evidence/summary.json)、[基线测试](placemoe_cleanup_evidence/baseline-tests.txt)、[最终测试](placemoe_cleanup_evidence/after-tests.txt)、[规划参考](placemoe_cleanup_evidence/planning_reference.json)、[算法 AST 审核](placemoe_cleanup_evidence/algorithm_ast.json)。

### 复现

复用第一轮的 CPU 环境和 `tests/tools/placemoe_cpu_bootstrap`；设置 `PLACEMOE_SOURCE` 为待测版本根目录，`PYTHONPATH` 为该版本 bootstrap 目录及根目录，设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。从仓库根目录执行：

```bash
python scripts/placemoe/verify_planning_paths.py --output /tmp/planning.json --reference docs/design/placemoe_cleanup_evidence/planning_reference.json
python scripts/placemoe/verify_refactor.py --snapshot profile/processed/hiermoe_oracle_ep16_step0_layer24/step0_layer24_call0.pt --output /tmp/profile.json --reference docs/design/placemoe_refactor_evidence/replay_reference.json
pytest -q tests/distributed/test_placemoe*.py
```

比较旧版本时，将同一个新增比较脚本放在旧版本之外执行，通过上述环境变量选择旧源码；本轮基线来自 `git archive 880818a`。固定软件环境为独立 CPU 容器的 PyTorch 2.9.0、pytest 9.1.1、Ruff 0.16.8。bootstrap 只绕过三个上层包的无关加速器注册，不替换算法或张量计算。

## 当前文件与功能清单

下表覆盖两个目标目录中的全部 Python 文件；本轮没有新增或删除文件，只减少内部实现。

| 文件 | 当前行数 | 功能 |
|---|---:|---|
| [placemoe/__init__.py](../../placemoe/__init__.py) | 51 | 包入口与公开接口导出。 |
| [placemoe/bridge.py](../../placemoe/bridge.py) | 36 | VeOmni 训练器与 PlaceMoE 生命周期的接口桥接。 |
| [placemoe/calibrate_network.py](../../placemoe/calibrate_network.py) | 551 | 公开的网络校准命令入口。 |
| [placemoe/model_adapter.py](../../placemoe/model_adapter.py) | 178 | 标准专家参数表示与模型适配协议（内部文件转发公开接口）。 |
| [placemoe/planner.py](../../placemoe/planner.py) | 236 | 稳定的规划命令入口、跨层执行编排和结果报告；不包含候选算法。 |
| [placemoe/planner_candidates.py](../../placemoe/planner_candidates.py) | 548 | 确定性候选构造及布局/映射转换。 |
| [placemoe/planner_config.py](../../placemoe/planner_config.py) | 390 | 规划 CLI 参数、搜索预算、容量和层级系数。 |
| [placemoe/planner_search.py](../../placemoe/planner_search.py) | 489 | 逐层候选编排、现有计划比较与留出路由评估。 |
| [veomni/distributed/moe/hiermoe/__init__.py](../../veomni/distributed/moe/hiermoe/__init__.py) | 80 | 包入口与公开接口导出。 |
| [veomni/distributed/moe/hiermoe/all_to_all.py](../../veomni/distributed/moe/hiermoe/all_to_all.py) | 2593 | 分层去重 dispatch/combine、反向通信及上下文；通信算法和流水线钩子未改。 |
| [veomni/distributed/moe/hiermoe/bridge.py](../../veomni/distributed/moe/hiermoe/bridge.py) | 111 | VeOmni 训练器与 PlaceMoE 生命周期的接口桥接。 |
| [veomni/distributed/moe/hiermoe/core_planner.py](../../veomni/distributed/moe/hiermoe/core_planner.py) | 2339 | 当前运行时仍调用的副本配额、映射和评分支持；保留其行为。 |
| [veomni/distributed/moe/hiermoe/expert_swap.py](../../veomni/distributed/moe/hiermoe/expert_swap.py) | 765 | 管理器构造与调度入口，组合各功能模块；不再集中所有实现。 |
| [veomni/distributed/moe/hiermoe/greedy_planner.py](../../veomni/distributed/moe/hiermoe/greedy_planner.py) | 4151 | 初始化/运行时仍调用的布局评分与槽位规划；复用独立 traffic 计数。 |
| [veomni/distributed/moe/hiermoe/metrics.py](../../veomni/distributed/moe/hiermoe/metrics.py) | 85 | 运行时指标记录、读取与清空。 |
| [veomni/distributed/moe/hiermoe/oracle.py](../../veomni/distributed/moe/hiermoe/oracle.py) | 1078 | 路由快照采集、读取和离线通信分析。 |
| [veomni/distributed/moe/hiermoe/perf_model.py](../../veomni/distributed/moe/hiermoe/perf_model.py) | 493 | 硬件通信性能模型和启动校准。 |
| [veomni/distributed/moe/hiermoe/planner.py](../../veomni/distributed/moe/hiermoe/planner.py) | 1979 | 当前运行时布局动作、计划类型和初始化评分/规划支持。 |
| [veomni/distributed/moe/hiermoe/routing.py](../../veomni/distributed/moe/hiermoe/routing.py) | 42 | 层级通信的路由辅助接口。 |
| [veomni/distributed/moe/hiermoe/runtime_artifact.py](../../veomni/distributed/moe/hiermoe/runtime_artifact.py) | 382 | 初始布局文件、层名匹配、静态布局安装及既有序列化格式。 |
| [veomni/distributed/moe/hiermoe/runtime_calibration.py](../../veomni/distributed/moe/hiermoe/runtime_calibration.py) | 1567 | 运行期计时、代价模型拟合/验证、自动校准及 CPU 评分器构造。 |
| [veomni/distributed/moe/hiermoe/runtime_checkpoint.py](../../veomni/distributed/moe/hiermoe/runtime_checkpoint.py) | 188 | 物理布局与映射的检查点保存、兼容性校验和恢复。 |
| [veomni/distributed/moe/hiermoe/runtime_gradients.py](../../veomni/distributed/moe/hiermoe/runtime_gradients.py) | 1085 | 副本梯度聚合、重叠窗口、梯度归一化掩码及相关槽位约束。 |
| [veomni/distributed/moe/hiermoe/runtime_hot_update.py](../../veomni/distributed/moe/hiermoe/runtime_hot_update.py) | 598 | 异步规划进程启动、路由采集、结果校验和布局/映射原子发布。 |
| [veomni/distributed/moe/hiermoe/runtime_migration.py](../../veomni/distributed/moe/hiermoe/runtime_migration.py) | 1012 | 参数和优化器状态的槽位迁移、打包、传输与发布。 |
| [veomni/distributed/moe/hiermoe/runtime_pipeline.py](../../veomni/distributed/moe/hiermoe/runtime_pipeline.py) | 336 | 设备流、事件窗口、微步状态和线程/进程组生命周期。 |
| [veomni/distributed/moe/hiermoe/runtime_planning.py](../../veomni/distributed/moe/hiermoe/runtime_planning.py) | 717 | 保留的初始化规划与 collective/score 流水线；退役入口显式报错。 |
| [veomni/distributed/moe/hiermoe/runtime_routing.py](../../veomni/distributed/moe/hiermoe/runtime_routing.py) | 584 | 专家注册、物理槽位路由、LUT 查表和路由记录。 |
| [veomni/distributed/moe/hiermoe/runtime_settings.py](../../veomni/distributed/moe/hiermoe/runtime_settings.py) | 312 | 运行时设置的唯一持有者、环境参数和配置安装。 |
| [veomni/distributed/moe/hiermoe/runtime_tensors.py](../../veomni/distributed/moe/hiermoe/runtime_tensors.py) | 645 | 专家参数扩展、优化器状态访问和底层张量传输原语。 |
| [veomni/distributed/moe/hiermoe/runtime_types.py](../../veomni/distributed/moe/hiermoe/runtime_types.py) | 473 | 专家层状态、迁移缓冲、异步任务和梯度同步记录类型。 |
| [veomni/distributed/moe/hiermoe/state.py](../../veomni/distributed/moe/hiermoe/state.py) | 826 | 全局训练状态、模型/优化器绑定以及生命周期入口。 |
| [veomni/distributed/moe/hiermoe/statistical_scorer.py](../../veomni/distributed/moe/hiermoe/statistical_scorer.py) | 2342 | 保留的槽位规划器依赖的统计评分和候选增量计数。 |
| [veomni/distributed/moe/hiermoe/topology.py](../../veomni/distributed/moe/hiermoe/topology.py) | 94 | 通信层级结构定义与推断。 |
| [veomni/distributed/moe/hiermoe/traffic.py](../../veomni/distributed/moe/hiermoe/traffic.py) | 676 | 精确去重通信计数、非去重 assignment 计数及校准成本；不包含候选搜索。 |
| [veomni/distributed/moe/hiermoe/triton_segment_sum.py](../../veomni/distributed/moe/hiermoe/triton_segment_sum.py) | 106 | 统计评分所需的可选 Triton 分段归约。 |
| [veomni/distributed/moe/hiermoe/placemoe/__init__.py](../../veomni/distributed/moe/hiermoe/placemoe/__init__.py) | 110 | 包入口与公开接口导出。 |
| [veomni/distributed/moe/hiermoe/placemoe/allocation.py](../../veomni/distributed/moe/hiermoe/placemoe/allocation.py) | 109 | 论文副本预算分配和有界动态规划候选列表。 |
| [veomni/distributed/moe/hiermoe/placemoe/artifacts.py](../../veomni/distributed/moe/hiermoe/placemoe/artifacts.py) | 95 | 布局/映射 artifact 构造、版本和合法性校验。 |
| [veomni/distributed/moe/hiermoe/placemoe/calibration.py](../../veomni/distributed/moe/hiermoe/placemoe/calibration.py) | 1197 | 持久化校准数据、拟合、作用域校验与性能预测。 |
| [veomni/distributed/moe/hiermoe/placemoe/cli.py](../../veomni/distributed/moe/hiermoe/placemoe/cli.py) | 1216 | prepare/doctor 等用户命令和配置检查。 |
| [veomni/distributed/moe/hiermoe/placemoe/mapping.py](../../veomni/distributed/moe/hiermoe/placemoe/mapping.py) | 810 | 来源相关的副本映射初始化、逐项和分组映射优化。 |
| [veomni/distributed/moe/hiermoe/placemoe/materialize.py](../../veomni/distributed/moe/hiermoe/placemoe/materialize.py) | 139 | 将实例布局和映射转换为物理槽位计划。 |
| [veomni/distributed/moe/hiermoe/placemoe/model_adapter.py](../../veomni/distributed/moe/hiermoe/placemoe/model_adapter.py) | 34 | 标准专家参数表示与模型适配协议（内部文件转发公开接口）。 |
| [veomni/distributed/moe/hiermoe/placemoe/optimizer.py](../../veomni/distributed/moe/hiermoe/placemoe/optimizer.py) | 383 | 固定预算下的布局—映射交替优化；算法实现未改。 |
| [veomni/distributed/moe/hiermoe/placemoe/partition.py](../../veomni/distributed/moe/hiermoe/placemoe/partition.py) | 422 | 容量约束的谱嵌入、分配和交换细化。 |
| [veomni/distributed/moe/hiermoe/placemoe/placement.py](../../veomni/distributed/moe/hiermoe/placemoe/placement.py) | 517 | 分层实例布局、节点分配和 rank 容量/唯一性修复。 |
| [veomni/distributed/moe/hiermoe/placemoe/preparation.py](../../veomni/distributed/moe/hiermoe/placemoe/preparation.py) | 373 | 校准准备阶段、artifact 复用和启动前检查。 |
| [veomni/distributed/moe/hiermoe/placemoe/route_replay.py](../../veomni/distributed/moe/hiermoe/placemoe/route_replay.py) | 233 | 完整 token 路由加载及留出样本成本评估。 |
| [veomni/distributed/moe/hiermoe/placemoe/seeds.py](../../veomni/distributed/moe/hiermoe/placemoe/seeds.py) | 48 | 确定性比较布局种子。 |
| [veomni/distributed/moe/hiermoe/placemoe/statistics.py](../../veomni/distributed/moe/hiermoe/placemoe/statistics.py) | 140 | 来源条件需求、共同选择统计及逻辑专家到副本统计投影。 |
| [veomni/distributed/moe/hiermoe/placemoe/types.py](../../veomni/distributed/moe/hiermoe/placemoe/types.py) | 182 | 拓扑、统计数据、布局/映射计划及其不变量。 |
| [veomni/distributed/moe/hiermoe/placemoe/runtime/__init__.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/__init__.py) | 41 | 包入口与公开接口导出。 |
| [veomni/distributed/moe/hiermoe/placemoe/runtime/config.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/config.py) | 551 | 规范的 typed 配置、路径、校准参数及资源约束。 |
| [veomni/distributed/moe/hiermoe/placemoe/runtime/controller.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/controller.py) | 50 | 单个异步规划任务的生命周期。 |
| [veomni/distributed/moe/hiermoe/placemoe/runtime/cpu_affinity.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/cpu_affinity.py) | 289 | 训练与规划进程 CPU 亲和性分配。 |
| [veomni/distributed/moe/hiermoe/placemoe/runtime/planner_process.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/planner_process.py) | 243 | 规划命令构造、进程启动与终止。 |
| [veomni/distributed/moe/hiermoe/placemoe/runtime/planner_supervisor.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/planner_supervisor.py) | 70 | 规划子进程的信号转发与父进程退出处理。 |
| [veomni/distributed/moe/hiermoe/placemoe/runtime/scheduler.py](../../veomni/distributed/moe/hiermoe/placemoe/runtime/scheduler.py) | 49 | 布局与映射独立更新周期的状态机。 |

## 剩余边界

实际训练仍需多 NPU 验证梯度同步、通信顺序、流间依赖及性能。当前保留的 Greedy、CoRe、CurrentRoute 和校准路径仍有真实调用，不能仅按文件长度删除。管理器的活跃 mixin 仍共享状态，本轮消除了退役模式的任务对象、分支和散落指标属性；并未将全部活跃运行时改写成独立服务。
