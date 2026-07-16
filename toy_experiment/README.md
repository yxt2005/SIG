# toy_experiment 使用说明

本文档说明小拓扑实验的运行方式、常用配置修改方法，以及结果文件位置。


## 目录结构

- `data/params.json`：实验参数配置，包括求解模式、CVaR 约束、双层节点层约束等。
- `data/failure_events_1.json` 到 `data/failure_events_4.json`：四类人为设定的故障场景及其概率。
- `run_toy.py`：运行一次小拓扑实验。
- `run_toy_comparison.py`：批量运行单层模型与双层模型对比实验。
- `run_toy_equivalence_scan.py`：扫描不同参数下是否存在等价最优解。
- `results/runs/xxx/`：每一次实验的完整结果目录。`xxx` 为顺序编号。
- `results/comparisons/xxx/`：每一组对比实验的汇总结果目录。`xxx` 为顺序编号。
- `results/equivalence_scans/xxx/`：每一组等价解扫参实验的汇总结果目录。`xxx` 为顺序编号。

`results/` 目录默认不上传到 GitHub；论文或讨论文档需要引用的图，应复制到 `paper/` 目录后再引用。

## 单次实验

运行命令：

```powershell
python toy_experiment\run_toy.py
```

脚本会读取 `toy_experiment/data/params.json`，求解一次模型，并自动生成新的结果目录：

```text
toy_experiment/results/runs/001/
toy_experiment/results/runs/002/
...
```

每次运行的主要输出包括：

- `toy_topology_solution.png`：完整四宫格拓扑结果图，包括候选拓扑、正常工作态和故障场景。
- `toy_working_state.png`：若由比较实验生成，则为单独的工作态子图。
- `metrics.csv`：本次实验的核心指标。
- `placements.json`：任务放置结果。
- `path_allocations.csv`：路径预留带宽分配。
- `scenario_losses.csv`：各故障场景下的损失和 Goodput 信息。
- `link_loads.csv`：链路预留负载。
- `params.json`：本次运行时使用的参数副本，便于复现。
- `run_metadata.json`：本次运行的轻量索引。

最近一次运行位置记录在：

```text
toy_experiment/results/latest_run.json
```

所有单次实验的总汇总表为：

```text
toy_experiment/results/summary.csv
```

## 修改求解模式

求解模式由 `data/params.json` 中的 `solver_mode` 控制。

### 单层模型

```json
{
  "solver_mode": "single_level",
  "risk_mode": "cvar_constraint",
  "cvar_bound": 0.1
}
```

含义：

- 使用单层联合优化模型；
- 最终真实 CVaR 约束为 `cvar_bound`；
- 优化目标为资源利用率，CVaR 作为约束。

### 双层平均分流模型

```json
{
  "solver_mode": "two_layer_average_split",
  "node_layer_risk_mode": "cvar_constraint",
  "node_layer_cvar_bound": 0.565,
  "risk_mode": "cvar_constraint",
  "cvar_bound": 0.1
}
```

含义：

- 第一层使用平均分流代理模型选择计算节点；
- `node_layer_cvar_bound` 是节点层代理 CVaR 约束；
- 第二层固定节点放置后重新优化真实路径预留带宽；
- `cvar_bound` 是链路层最终真实 CVaR 约束，应与单层模型对比时保持一致。

## 单层模型的带宽分配 baseline

单层模型通过 `data/params.json` 中的 `routing_mode` 选择带宽分配方式：

```json
{
  "solver_mode": "single_level",
  "routing_mode": "mcf"
}
```

支持三个取值：

- `mcf`：现有多路径连续流模型。各候选路径的 `x[p]` 由优化器决定，不要求均分，并允许为故障场景预留超过名义需求的带宽。
- `single_path`：每个任务在所选计算节点的输入段和输出段各选择一条路径。新增二元变量 `z[p]`，约束为 `sum_p z[p] = y[i,m]`、`x[p] = demand * z[p]`。
- `ecmp`：在所选计算节点对应的全部候选路径上严格均分名义带宽需求。若候选路径数为 `K`，则 `x[p] = demand / K * y[i,m]`。

三种 baseline 可批量运行：

```powershell
python toy_experiment\run_toy_bandwidth_comparison.py --link-bound 0.8
```

也可只运行部分模式：

```powershell
python toy_experiment\run_toy_bandwidth_comparison.py --modes mcf ecmp --link-bound 0.8 --no-plot
```

结果写入 `results/bandwidth_comparisons/xxx/summary.csv`，每个 case 的完整结果仍写入 `results/runs/xxx/`。

注意：当前默认 `cvar_bound=0.1` 下，严格单路径和严格 ECMP 无法满足可靠性约束，因此会返回不可行；这反映 baseline 本身的能力边界。要得到三者均可行的数值对比，应放宽 `--link-bound`（当前 toy 数据下可使用 `0.8`）。

### CVaR 约束—资源扫描

推荐扫描 CVaR 约束上限，同时比较可靠性要求与链路资源开销：

```powershell
python toy_experiment\run_toy_bandwidth_pareto.py
```

默认对 MCF、单路径和 ECMP 扫描 `CVaR bound=0.1,0.2,...,0.8`。每个扫描点直接调用一次原单层模型：以资源利用率为目标、以对应 CVaR 上限为约束，不增加次级目标。自动图以约束上限 `Γ` 为横轴、最大链路利用率为纵轴；不可行点不绘制。`eval_cvar` 仍保留在汇总表中。脚本自动输出：

- `results/bandwidth_pareto/xxx/summary.csv`：全部扫描点、不可行状态及 `is_pareto` 标记；
- `results/bandwidth_pareto/xxx/summary.json`：JSON 格式完整结果；
- `results/bandwidth_pareto/xxx/routing_cvar_bound_resource_curve.png`：自动生成的约束—资源曲线；
- `results/bandwidth_pareto/xxx/routing_cvar_bound_resource_curve.svg`：同图的矢量版本。

自定义扫描范围或模式：

```powershell
python toy_experiment\run_toy_bandwidth_pareto.py --link-bounds 0.05,0.1,0.2,0.4,0.6,0.8 --modes mcf single_path ecmp
```
## 候选路径完整性

`data/paths.json` 采用分层转发规则：源节点先到 `a/c`，可选经过一次 `a-c` 横向链路，再进入计算节点；输出侧从计算节点到 `b/d`，可选经过一次 `b-d` 横向链路，再到目标节点。它不是无向图上所有简单路径的全集（例如不会包含绕经另一个源节点的路径）。

按这一定义，每个“任务–计算节点–输入/输出阶段”现在均有 4 条对称候选路径，共 12 组、48 条路径。已补充的方向包括您指出的 `s1-a-c-mA` 与 `mA-b-d-t1`，并同步补齐其他计算节点和任务的对称项。
## 修改 CVaR 约束

单层模型和双层链路层的最终真实 CVaR 约束由 `cvar_bound` 控制，例如：

```json
{
  "risk_mode": "cvar_constraint",
  "cvar_bound": 0.3
}
```

双层模型节点层代理 CVaR 约束由 `node_layer_cvar_bound` 控制，例如：

```json
{
  "node_layer_risk_mode": "cvar_constraint",
  "node_layer_cvar_bound": 0.7
}
```

对比单层和双层时，建议固定相同的最终真实约束 `cvar_bound`，再扫描不同的 `node_layer_cvar_bound`。

## 输出等价最优解

小拓扑中可能存在多个 objective value 相同的等价解。默认情况下，求解器会启用 Gurobi solution pool，输出所有与最优目标值相同的离散等价解。

可在 `data/params.json` 中控制：

```json
{
  "enumerate_equivalent_solutions": true,
  "solution_pool_limit": 50,
  "equivalent_obj_tol": 1e-6
}
```

含义：

- `enumerate_equivalent_solutions`：是否枚举等价解；
- `solution_pool_limit`：最多保留多少个 solution pool 解；
- `equivalent_obj_tol`：判断 objective value 是否相同的容忍误差。

输出文件位于每次运行的 `results/runs/xxx/` 下：

- `equivalent_solutions.csv`：等价解指标对比表；
- `equivalent_solutions.json`：等价解放置与指标的 JSON 索引；
- `equivalent_path_allocations.csv`：等价解的路径预留带宽明细。

注意：双层模型中，节点层含整数放置变量，可以枚举等价节点放置；固定放置后的链路层是连续 LP，理论上可能存在无限多个连续等价流量解。因此当前输出的是每个等价节点放置对应的一个代表性链路层最优解。

## 等价解扫参实验

如果需要确认“单层模型是否在其他参数下也没有等价解”，可以运行专门的扫参脚本：

```powershell
python toy_experiment\run_toy_equivalence_scan.py
```

默认扫描：

- 单层模型：`Γ_L=0.1,0.3,0.5`；
- 双层模型：`Γ_L=0.1,0.3,0.5` 与 `Γ_N=0.565,0.6,0.7,0.8` 的组合。

也可以手动指定扫描范围：

```powershell
python toy_experiment\run_toy_equivalence_scan.py --link-bounds 0.05,0.1,0.2,0.3,0.5 --node-bounds 0.55,0.565,0.6,0.7,0.8
```

如果只扫描单层模型：

```powershell
python toy_experiment\run_toy_equivalence_scan.py --modes single --link-bounds 0.05,0.1,0.2,0.3,0.5
```

扫参结果写入：

```text
toy_experiment/results/equivalence_scans/001/
toy_experiment/results/equivalence_scans/002/
...
```

主要文件：

- `summary.csv`：每个参数组合的等价解数量、放置方案和核心指标；
- `summary.json`：同样结果的 JSON 版本，便于后续脚本读取。

注意：该脚本用于快速判断不同离散放置方案是否存在等价最优解。由于路径带宽分配含连续变量，同一离散放置下仍可能存在连续意义上的等价流量解，不能仅凭 solution pool 完全枚举。
## 修改故障场景

默认综合故障场景写在：

```text
toy_experiment/data/failure_events_4.json
```

当前四类故障文件均只需要写故障场景，不需要手动写正常场景。程序会自动把正常场景概率设为：

```text
1 - 所有故障场景概率之和
```

例如：

```json
[
  {
    "scenario_id": 1,
    "label": "mA compute site unavailable",
    "probability": 0.005,
    "event_ids": ["site_mA"],
    "failed_nodes": ["mA"],
    "failed_edges": []
  },
  {
    "scenario_id": 2,
    "label": "shared input link a-c failure",
    "probability": 0.015,
    "event_ids": ["link_ac"],
    "failed_nodes": [],
    "failed_edges": [["a", "c"]]
  }
]
```

注意：

- 所有故障场景概率之和必须小于或等于 1；
- 若故障概率和为 0.16，则正常场景概率自动为 0.84；
- `failed_nodes` 写失效节点编号；
- `failed_edges` 写失效边的两个端点，例如 `["a", "c"]`。

## 单双层比较实验

比较实验脚本为：

```powershell
python toy_experiment\run_toy_comparison.py --link-bound 0.3 --node-bounds 0.6,0.7,0.8
```

含义：

- `--link-bound 0.3`：固定最终真实 CVaR 约束 $\Gamma_L=0.3$；
- `--node-bounds 0.6,0.7,0.8`：双层模型扫描节点层代理约束 $\Gamma_N=0.6,0.7,0.8$；
- 默认会同时运行一个单层模型作为基准；
- 每个 case 都会生成一个新的 `results/runs/xxx/`；
- 整组实验会生成一个新的 `results/comparisons/xxx/`。

常用示例：

```powershell
python toy_experiment\run_toy_comparison.py --link-bound 0.1 --node-bounds 0.6,0.7,0.8
python toy_experiment\run_toy_comparison.py --link-bound 0.3 --node-bounds 0.6,0.7,0.8
python toy_experiment\run_toy_comparison.py --link-bound 0.5 --node-bounds 0.6,0.7,0.8
```

如果只想运行双层模型，不运行单层基准：

```powershell
python toy_experiment\run_toy_comparison.py --link-bound 0.3 --node-bounds 0.6,0.7,0.8 --skip-single
```

如果只想生成数值结果，不生成拓扑图：

```powershell
python toy_experiment\run_toy_comparison.py --link-bound 0.3 --node-bounds 0.6,0.7,0.8 --no-plot
```

## 比较实验结果位置

每一组比较实验会写入：

```text
toy_experiment/results/comparisons/001/
toy_experiment/results/comparisons/002/
...
```

主要文件：

- `summary.csv`：本组单层和双层模型的指标对比表。
- `toy_working_state_grid.png`：本组工作态对比图，采用两排两列布局。
- `toy_working_state_grid.svg`：同一张图的 SVG 版本。

其中 `toy_working_state_grid.png` 是讨论文档和论文中最常用的对比图。它只展示每个模型的工作状态子图，不包含故障子图，便于横向比较单层模型和多个双层模型结果。

## 故障类型对比实验

为了比较不同故障类型下的模型表现，`data/` 下补充了四个故障场景文件：

- `failure_events_1.json`：无故障，仅自动生成 normal 场景；
- `failure_events_2.json`：只有链路故障；
- `failure_events_3.json`：只有节点故障；
- `failure_events_4.json`：同时考虑链路故障和节点故障，内容与原综合场景一致。

单次实验中可以在 `data/params.json` 中修改：

```json
{
  "failure_scenarios_file": "failure_events_2.json"
}
```

批量运行四类故障场景对比：

```powershell
python toy_experiment\run_toy_failure_type_comparison.py
```

脚本会保持其他参数不变，只切换 `failure_scenarios_file`，并为每个 case 生成一个新的 `results/runs/xxx/`。整组对比结果写入：

```text
toy_experiment/results/failure_type_comparisons/001/
toy_experiment/results/failure_type_comparisons/002/
...
```

主要文件：

- `summary.csv`：四类故障场景的求解状态与核心指标；
- `summary.json`：同样结果的 JSON 版本；
- `toy_failure_type_working_state_grid.png`：四类故障场景的工作态对比图；
- `toy_failure_type_working_state_grid.svg`：同一张图的 SVG 版本。

如果某一类故障场景下模型不可行，则对应子图会保留空白占位，并在 `summary.csv` 中记录失败原因。
## 指标说明

常用比较指标包括：

- `eval_cvar`：根据最终解重新评估得到的真实 CVaR；
- `model_u_node_max`：最大节点利用率；
- `model_u_link_max`：最大链路利用率；
- `total_reserved_bandwidth`：总分配/预留带宽；
- `effective_throughput`：有效吞吐 Goodput；
- `min_loss`：最小场景损失；
- `max_loss`：最大场景损失；
- `expected_loss`：概率加权平均损失。
