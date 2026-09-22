# PlaceMoE 第三轮精简记录（2026-09-22）

本轮完成独立诊断代码清理和校准成本模型拆分。尚未完成生产模式之外的全部历史策略删除；旧规划器及迁移调度暂时保留。

## 范围与规模

基线为 `6ffcd36`。按两个目标目录下全部 Python 文件的物理行数统计（含注释和空行）：

| 目录 | 重构前 | 重构后 |
| --- | ---: | ---: |
| `veomni/distributed/moe/hiermoe/` | 32,960 | 32,171 |
| `placemoe/` | 2,479 | 2,479 |
| 合计 | 35,439 | 34,650 |

净减少 789 行。本轮没有修改顶层 `placemoe/` 算法，也没有修改生产通信、梯度同步、checkpoint 重计算和热更新的执行顺序。

删除 oracle 中无调用的旧离线成本曲线、交换/副本模拟以及旧快照读取辅助函数；保留训练路由采集和快照保存。删除冗余副本专用调试日志及其三个环境变量读取。原有校准数学计算独立为 `CalibrationCostModel`，旧 Greedy 和自动校准共享这份实现；保留公式、运算顺序、参数规范化和正值检查。

## 文件与功能

路径均相对于 `veomni/distributed/moe/hiermoe/`。

| 文件 | 当前职责／本轮变化 |
| --- | --- |
| `calibration_cost.py` | 独立的通信成本统计；接收端和源/目的端瓶颈评分 |
| `traffic.py` | 共用 token/assignment 计数和分层流量特征；本轮未改 |
| `runtime_calibration.py` | 采集时间、拟合和验证成本；直接构造成本模型，删除无调用工厂 |
| `greedy_planner.py` | 暂保留旧搜索；通过继承共用成本计算，删除重复公式 |
| `oracle.py` | 训练路由采集及快照保存；删除旧离线曲线分析 |
| `runtime_routing.py` | 注册、路由映射与路由记录；删除副本专用诊断方法 |
| `runtime_gradients.py` | 副本梯度同步；仅删除诊断日志调用 |
| `runtime_pipeline.py` | 流、事件和任务生命周期；仅删除诊断日志调用 |
| `runtime_settings.py` | 配置读取；删除无用副本诊断设置 |

验证辅助文件：`scripts/placemoe/verify_calibration_cost.py`，基线数据在 `docs/design/placemoe_cleanup_evidence/calibration_cost_reference.json`。

## 不依赖多节点 NPU 的验证

隔离 CPU 容器中使用 PyTorch 2.9、单计算线程，与独立基线目录比较。

- PlaceMoE 最终完整回归：171 passed、2 skipped、2 failed；基线为 169 passed、2 skipped、2 failed。新增通过项为零/负 smoothing 参数拒绝测试；没有新增失败。两个失败均为 CLI prepare 的 NPU preflight 在 CPU 环境失败。相关 adapter 测试文件单独验证为 17 passed。
- 38 个规划场景：与既有基线 JSON 字节完全一致，包括候选、布局、路由和预测成本。
- 既有 EP16/E128/top-8 profile：260,270 tokens，6 种拓扑/布局组合的 192 个计数摘要和 6 个优化候选，与原基线完全一致。
- 独立成本抽取对照：一层、两层、三层拓扑，每种包含 8 个源 rank、3 个 batch、97 tokens、top-4；计数及两种成本输出与原 Greedy 完全一致。基线与当前输出 SHA-256 均为 `293c78ff3775cde772fc58e5ab7b24c8584642e31291614424905eb7a68a2b0a`。
- 全仓 `pytest tests/ --maxfail=5`：前后均因缺少 CI 样本配置和 Triton 等环境依赖，在相同五处收集失败。
- `make quality`：Ruff lint 通过；format 检查仍报告基线已有的 24 个文件。修改文件单独检查通过。

复现命令（需要配置仓库已有 CPU bootstrap 或正常训练依赖）：

```bash
python scripts/placemoe/verify_planning_paths.py --output /tmp/planning.json --reference docs/design/placemoe_cleanup_evidence/planning_reference.json
python scripts/placemoe/verify_refactor.py --snapshot profile/processed/hiermoe_oracle_ep16_step0_layer24/step0_layer24_call0.pt --output /tmp/profile.json --reference docs/design/placemoe_refactor_evidence/replay_reference.json
python scripts/placemoe/verify_calibration_cost.py --output /tmp/calibration.json --reference docs/design/placemoe_cleanup_evidence/calibration_cost_reference.json
```

验证脚本支持 `--legacy`，可在 `6ffcd36` 源码环境中重新生成成本基线。离线证据不能证明多节点 NPU 通信顺序、流依赖、梯度同步和性能没有退化。

## 未完成范围与审批阻塞

跨模块删除旧 CurrentRoute/CoRe/Greedy 搜索、旧规划/迁移调度的操作被自动审批拒绝。理由是对共享运行时依赖的安全证据不足；第二次还指出当时存在测试失败。本轮恢复了未完成的拆分，仅提交上述独立改动。

后续删除必须保留热更新使用的 `_layer_layout`、原子 slot 搬运、优化器状态及副本梯度函数；从混合通信钩子中仅移除旧 planner 部分。同时需明确拒绝旧 quota checkpoint 或保留其原有处理，不能把零搜索预算等同于所有副作用都消失。当前代码仍维持原 checkpoint 兼容行为。
