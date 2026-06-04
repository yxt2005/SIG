# toy_experiment 使用说明

本文档说明小拓扑实验的运行方式、常用配置修改方法，以及结果文件位置。

## 目录结构

- `data/params.json`：实验参数配置，包括求解模式、CVaR 约束、双层节点层约束等。
- `data/failure_scenarios.json`：人为设定的故障场景及其概率。
- `run_toy.py`：运行一次小拓扑实验。
- `run_toy_comparison.py`：批量运行单层模型与双层模型对比实验。
- `results/runs/xxx/`：每一次实验的完整结果目录。`xxx` 为顺序编号。
- `results/comparisons/xxx/`：每一组对比实验的汇总结果目录。`xxx` 为顺序编号。

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

## 修改故障场景

故障场景写在：

```text
toy_experiment/data/failure_scenarios.json
```

当前文件只需要写故障场景，不需要手动写正常场景。程序会自动把正常场景概率设为：

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


