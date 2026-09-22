# PlaceMoE 旧规划器和迁移调度精简（2026-09-22）

本轮基线为 `4c1a4ba`，完成用户批准的旧搜索与迁移调度跨模块删除。保留论文采用的生产规划、成本模型、预加载布局、热更新、校准、梯度同步和 activation checkpoint 重计算。

## 规模与边界

物理行数包含空行、注释，仅统计两个目标目录的 Python 文件。

| 目录 | 修改前 | 修改后 |
| --- | ---: | ---: |
| `veomni/distributed/moe/hiermoe/` | 32,171 | 18,765 |
| `placemoe/` | 2,479 | 2,479 |
| 合计 | 34,650 | 21,244 |

减少 13,406 行（38.7%），当前共 57 个文件。本轮未以压缩排版降低行数。

删除 CurrentRoute、CoReMoE、Greedy 历史搜索类，以及 `runtime_planning.py`、`runtime_migration.py`、`statistical_scorer.py`；同步移除训练钩子中的旧规划窗口、异步任务和迁移调用。三个旧 planner 文件仅保留仍需的副本路由函数及数据表示。热更新使用的布局查询移入路由模块；原子搬运、优化器状态和副本梯度操作保留。

生产入口限定 step 模式和零旧搜索预算，旧诊断、冻结、消融及 fixed-R2 环境设置明确报错；非空旧 quota checkpoint 不再接受。初始布局使用预加载的 schema 2 产物。`calibrate-model` 仍保留，通过专用校准环境设置启动原采样、拟合、验证流程。具体兼容边界见使用文档。

生产 held-out 候选评估仍调用 `route_replay.py`，因此保留。少量低层交换辅助函数和兼容常量尚未清除，不能将本轮描述为已删除全部历史代码。

## CPU 验证与限制

隔离 CPU 容器使用 Python 3.11、PyTorch 2.9，计算线程设为 1。

| 验证 | 结果 |
| --- | --- |
| 生产规划 24 场景 | 与独立基线目录输出逐字节一致；旧 14 个场景仍留在历史验证脚本，不作为当前生产测试 |
| EP16/E128/top-8 历史 profile | 260,270 tokens；6 种布局/拓扑、192 个计数摘要、6 个优化候选与原参考一致 |
| 校准成本 | 一至三层拓扑的通信特征及两类成本与原参考一致 |
| 运行时边界 | 预加载权重、布局、路由、activation checkpoint 重计算和布局元数据与基线一致 |
| 算法 AST | 19 个完整模块、12 个保留路由函数与基线一致，见证据 JSON |
| PlaceMoE 和 combine CPU 测试 | 179 passed、2 skipped、2 failed；两个失败与基线相同，均为 CLI prepare 的 NPU preflight 环境限制 |
| 全仓测试收集 | 前后同样在 CI 样本配置、e2e、模型及 Triton 依赖的 5 处失败 |
| `make quality` | lint 通过；format 仍为基线已有的 24 个文件不通过 |

基线 PlaceMoE 测试为 171 passed、2 skipped、2 failed，另有 combine 测试 1 passed。本轮增加 7 个通过项，覆盖旧入口拒绝、quota 拒绝、校准启动契约和 dispatch 梯度事件钩子。

**已有缺陷：** `state_dict()` 未保存 Source LUT，`load_state_dict()` 也未恢复它。热更新后的 checkpoint 恢复可能留下与布局不匹配的路由表。本轮保持基线行为，验证记录了这一遗漏；布局元数据一致不能证明完整路由状态恢复正确。该缺陷需独立修复。

离线验证不能证明多节点 NPU 的梯度同步、通信顺序、流间依赖正确，也不能证明吞吐和显存没有退化。

复现（需已有 CPU bootstrap 或完整训练依赖）：

```bash
python scripts/placemoe/verify_planning_paths.py --production-only --output /tmp/planning.json --reference docs/design/placemoe_runtime_cleanup_evidence/production_planning_reference.json
python scripts/placemoe/verify_runtime_boundary.py --output /tmp/runtime.json --reference docs/design/placemoe_runtime_cleanup_evidence/runtime_boundary_reference.json
python scripts/placemoe/verify_refactor.py --snapshot profile/processed/hiermoe_oracle_ep16_step0_layer24/step0_layer24_call0.pt --output /tmp/profile.json --reference docs/design/placemoe_refactor_evidence/replay_reference.json
python scripts/placemoe/verify_calibration_cost.py --output /tmp/cost.json --reference docs/design/placemoe_cleanup_evidence/calibration_cost_reference.json
```

独立提交复核结论为 safe（限上述生产支持范围），跨 mixin 方法检查未发现悬空调用。修改的 23 个 Python 文件 lint 和格式检查通过。

验证参考、AST 对照和本轮日志位于 `placemoe_runtime_cleanup_evidence/`。旧规划验证脚本不带 `--production-only` 的模式仅适用于仍有旧类的历史 checkout。

## 当前文件与功能清单

### `placemoe/`

| 文件 | 行数 | 功能 |
| --- | ---: | --- |
| `__init__.py` | 51 | 公共接口 |
| `bridge.py` | 36 | 版本化桥接接口 |
| `calibrate_network.py` | 551 | 网络校准微基准 |
| `model_adapter.py` | 178 | 模型专家参数适配 |
| `planner.py` | 236 | 离线规划 CLI 和报告 |
| `planner_candidates.py` | 548 | 候选构造 |
| `planner_config.py` | 390 | 规划参数与预算 |
| `planner_search.py` | 489 | 逐层搜索与 held-out 评估 |

### `veomni/distributed/moe/hiermoe/`

| 文件 | 行数 | 功能 |
| --- | ---: | --- |
| `__init__.py` | 76 | 分布式 MoE 公共入口 |
| `all_to_all.py` | 2507 | 分层 dispatch/combine、自动微分通信钩子 |
| `bridge.py` | 111 | 训练框架与 PlaceMoE 适配 |
| `calibration_cost.py` | 142 | 校准通信特征和成本公式 |
| `core_planner.py` | 354 | quota 数据表示及副本映射函数 |
| `expert_swap.py` | 359 | 运行时 mixin 组合与生命周期入口 |
| `greedy_planner.py` | 138 | 最近副本路由 |
| `metrics.py` | 85 | 统计指标 |
| `oracle.py` | 411 | 训练路由快照采集 |
| `perf_model.py` | 493 | 性能模型拟合与预测 |
| `planner.py` | 381 | 确定性副本分配 |
| `routing.py` | 42 | 路由接口 |
| `runtime_artifact.py` | 77 | 预加载布局与 Source LUT 安装 |
| `runtime_calibration.py` | 1140 | 运行时计时、校准和验证 |
| `runtime_checkpoint.py` | 190 | 布局元数据保存与恢复 |
| `runtime_gradients.py` | 1082 | 副本梯度同步 |
| `runtime_hot_update.py` | 598 | 生产热更新与布局应用 |
| `runtime_pipeline.py` | 116 | 梯度流、事件及关闭清理 |
| `runtime_routing.py` | 360 | 层注册、路由映射和记录 |
| `runtime_settings.py` | 208 | 运行时环境配置和旧入口拒绝 |
| `runtime_tensors.py` | 645 | 参数、优化器状态和 slot 搬运原语 |
| `runtime_types.py` | 281 | 共享状态与记录类型 |
| `state.py` | 799 | 全局状态、训练钩子及 checkpoint 重计算 |
| `topology.py` | 94 | 通信拓扑 |
| `traffic.py` | 676 | 精确通信计数与流量特征 |
| `triton_segment_sum.py` | 106 | 分段归约内核 |

### `veomni/distributed/moe/hiermoe/placemoe/`

| 文件 | 行数 | 功能 |
| --- | ---: | --- |
| `__init__.py` | 110 | 论文算法公共接口 |
| `allocation.py` | 109 | 副本预算分配 |
| `artifacts.py` | 95 | 布局产物序列化与校验 |
| `calibration.py` | 1197 | 校准产物读取、拟合与构建 |
| `cli.py` | 1209 | prepare/calibrate/plan 等命令 |
| `mapping.py` | 810 | Source LUT 构造与映射 |
| `materialize.py` | 139 | 布局物化 |
| `model_adapter.py` | 34 | 模型适配兼容导出 |
| `optimizer.py` | 383 | 布局优化编排 |
| `partition.py` | 422 | 专家分区 |
| `placement.py` | 517 | 专家放置 |
| `preparation.py` | 373 | 训练前产物准备 |
| `route_replay.py` | 233 | 生产候选的路由回放评估 |
| `runtime/__init__.py` | 41 | 运行时接口导出 |
| `runtime/config.py` | 551 | 生产运行时配置 |
| `runtime/controller.py` | 50 | 热更新控制 |
| `runtime/cpu_affinity.py` | 289 | 规划进程 CPU 亲和性 |
| `runtime/planner_process.py` | 243 | 规划子进程入口 |
| `runtime/planner_supervisor.py` | 70 | 规划子进程生命周期 |
| `runtime/scheduler.py` | 49 | 更新时机调度 |
| `seeds.py` | 48 | 确定性随机种子 |
| `statistics.py` | 140 | 路由统计 |
| `types.py` | 182 | 论文算法数据类型 |
