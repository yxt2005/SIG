"""Toy 实验单层 MILP 求解器。

输入：
    - config 配置信息（来自toy_config.py及data/*）。
输出：
    - solution_bundle：包含任务放置 placement、路径流量分配 allocations，同时统计场景损失和指标。
    - results/*.csv/json：总结实验结果。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toy_config import Task, edge_key, enumerate_scenarios, path_edges


@dataclass(frozen=True)
class PathVar:
    """一条候选路径变量。

    关键变量：
        task_id：任务编号。
        phase：in 表示输入，out 表示输出端。
        compute_node：候选计算节点。
        path_id：优化变量 x 的唯一索引。
        path_nodes/path_edges：路径经过的节点和无向边。
    """

    task_id: str
    phase: str
    compute_node: str
    path_id: str
    path_nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


def _build_path_vars(config: dict) -> list[PathVar]:
    """从配置文件中的候选路径生成优化变量索引列表。"""
    path_vars = []
    for (task_id, compute_node, phase), items in config["paths"].items():
        for item in items:
            nodes = tuple(item["path_nodes"])
            path_vars.append(
                PathVar(
                    task_id=task_id,
                    phase=phase,
                    compute_node=compute_node,
                    path_id=item["path_id"],
                    path_nodes=nodes,
                    edges=path_edges(nodes),
                )
            )
    return path_vars


def _path_available(path_var: PathVar, scenario: dict[str, Any]) -> float:
    """判断路径在某个故障场景下是否可用；可用返回 1，不可用返回 0。"""
    if path_var.compute_node in scenario["failed_nodes"]:
        return 0.0
    if set(path_var.edges) & scenario["failed_edges"]:
        return 0.0
    return 1.0


def _weighted_cvar(losses: list[float], probs: list[float], beta: float) -> tuple[float, float]:
    """用离散场景损失计算加权 CVaR 和对应 VaR 阈值 alpha。"""
    candidate_alphas = sorted(set([0.0, 1.0] + [float(loss) for loss in losses]))
    best_alpha = 0.0
    best_value = float("inf")
    for alpha in candidate_alphas:
        value = alpha + sum(prob * max(loss - alpha, 0.0) for loss, prob in zip(losses, probs)) / (1.0 - beta)
        if value < best_value:
            best_value = value
            best_alpha = alpha
    return float(best_value), float(best_alpha)


def _evaluate_solution(config: dict, placement: dict[str, str], allocations: list[dict], scenarios: list[dict[str, Any]]):
    """在给定任务放置和路径流量分配下，逐场景评估服务损失、链路负载和风险指标。"""
    alloc_by_path = {row["path_id"]: float(row["allocation"]) for row in allocations}
    path_vars_by_id = {pv.path_id: pv for pv in _build_path_vars(config)}

    scenario_rows = []
    scenario_losses = {}
    scenario_goodputs = {}
    for scenario in scenarios:
        sid = int(scenario["scenario_id"])
        task_loss_map = {}
        for task in config["tasks"]:
            phase_losses = {}
            for phase, demand in (("in", task.b_in), ("out", task.b_out)):
                delivered = 0.0
                for row in allocations:
                    if row["task"] != task.task_id or row["phase"] != phase:
                        continue
                    pv = path_vars_by_id[row["path_id"]]
                    delivered += _path_available(pv, scenario) * alloc_by_path[row["path_id"]]
                phase_losses[phase] = min(1.0, max(0.0, 1.0 - delivered / float(demand)))
            completion_ratio = min(
                1.0,
                max(0.0, 1.0 - phase_losses["in"]),
                max(0.0, 1.0 - phase_losses["out"]),
            )
            task_goodput = completion_ratio * (float(task.b_in) + float(task.b_out))
            task_loss_map[task.task_id] = {
                "in": phase_losses["in"],
                "out": phase_losses["out"],
                "task": max(phase_losses["in"], phase_losses["out"]),
                "completion_ratio": completion_ratio,
                "goodput": task_goodput,
            }

        if config["loss_aggregation"] == "average":
            system_loss = sum(v["task"] for v in task_loss_map.values()) / len(task_loss_map)
        else:
            system_loss = max(v["task"] for v in task_loss_map.values())
        scenario_losses[sid] = system_loss
        scenario_goodputs[sid] = sum(v["goodput"] for v in task_loss_map.values())

        for task in config["tasks"]:
            scenario_rows.append(
                {
                    "scenario_id": sid,
                    "probability": float(scenario["probability"]),
                    "event_ids": ";".join(scenario["event_ids"]) if scenario["event_ids"] else "normal",
                    "event_labels": "; ".join(scenario["event_labels"]) if scenario["event_labels"] else "normal",
                    "failed_nodes": ";".join(sorted(scenario["failed_nodes"])),
                    "failed_edges": ";".join(f"{u}-{v}" for u, v in sorted(scenario["failed_edges"])),
                    "task": task.task_id,
                    "compute_node": placement[task.task_id],
                    "loss_in": task_loss_map[task.task_id]["in"],
                    "loss_out": task_loss_map[task.task_id]["out"],
                    "loss_task": task_loss_map[task.task_id]["task"],
                    "completion_ratio": task_loss_map[task.task_id]["completion_ratio"],
                    "goodput_task": task_loss_map[task.task_id]["goodput"],
                    "system_loss": system_loss,
                    "scenario_goodput": scenario_goodputs[sid],
                }
            )

    link_loads = []
    max_util = 0.0
    for edge, capacity in config["capacities"].items():
        load = 0.0
        for row in allocations:
            if f"{edge[0]}-{edge[1]}" in row["path_edges"]:
                load += float(row["allocation"])
        util = load / float(capacity) if capacity > 0 else 0.0
        max_util = max(max_util, util)
        link_loads.append(
            {
                "edge": f"{edge[0]}-{edge[1]}",
                "capacity": float(capacity),
                "reserved_load": load,
                "reserved_utilization": util,
            }
        )

    scenario_loss_list = [scenario_losses[int(s["scenario_id"])] for s in scenarios]
    scenario_prob_list = [float(s["probability"]) for s in scenarios]
    cvar, alpha = _weighted_cvar(scenario_loss_list, scenario_prob_list, float(config["beta"]))
    expected_loss = sum(prob * loss for prob, loss in zip(scenario_prob_list, scenario_loss_list))
    effective_throughput = sum(
        float(s["probability"]) * scenario_goodputs[int(s["scenario_id"])]
        for s in scenarios
    )
    availability = sum(prob for prob, loss in zip(scenario_prob_list, scenario_loss_list) if loss <= 1e-9)

    return {
        "scenario_rows": scenario_rows,
        "link_loads": link_loads,
        "metrics": {
            "cvar": cvar,
            "alpha": alpha,
            "expected_loss": float(expected_loss),
            "min_loss": float(min(scenario_loss_list)),
            "max_loss": float(max(scenario_loss_list)),
            "effective_throughput": float(effective_throughput),
            "availability": float(availability),
            "total_reserved_bandwidth": float(sum(row["allocation"] for row in allocations)),
            "maximum_link_utilization": float(max_util),
            "beta": float(config["beta"]),
            "lambda_weight": float(config["lambda_weight"]),
            "risk_weight": float(config["risk_weight"]),
        },
    }


def solve_toy_single_level(config: dict) -> dict:
    """建立并求解 toy MILP。

    输入：
        config：toy_config.default_config() 返回的完整实验配置。
    输出：
        dict：best 字段保存最优解，scenarios 字段保存用于评估的故障场景。

    关键变量：
        y[task, m]：任务是否部署到计算节点 m。
        x[path_id]：路径预留带宽。
        U_node_max/U_link_max：最大节点/链路资源利用率。
        loss_task/loss_sys：任务级和系统级场景损失。
        alpha/u_aux：CVaR 线性化辅助变量。
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError("gurobipy is required for the toy single-level MILP solver.") from exc

    tasks: list[Task] = config["tasks"]
    task_by_id = {task.task_id: task for task in tasks}
    compute_nodes = config["compute_nodes"]
    beta = float(config["beta"])
    lambda_weight = float(config["lambda_weight"])
    risk_weight = float(config["risk_weight"])
    risk_mode = config.get("risk_mode", "weighted")
    cvar_bound = config.get("cvar_bound")
    scenarios = config.get("failure_scenarios") or enumerate_scenarios(config["failure_events"])
    path_vars = _build_path_vars(config)
    path_by_id = {pv.path_id: pv for pv in path_vars}

    #========1. 建立优化模型=======
    try:
        model = gp.Model("toy_single_level")
    except gp.GurobiError as exc:
        raise RuntimeError(f"Failed to create Gurobi model for toy experiment: {exc}") from exc
    model.Params.OutputFlag = 0

    #========2. 定义决策变量=======
    y_keys = [(task.task_id, m) for task in tasks for m in task.candidates]
    y = model.addVars(y_keys, vtype=GRB.BINARY, name="y")
    x = model.addVars([pv.path_id for pv in path_vars], lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    u_node_max = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="U_node_max")
    u_link_max = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="U_link_max")
    alpha = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="alpha")
    loss_task = model.addVars(
        [(int(s["scenario_id"]), task.task_id) for s in scenarios for task in tasks],
        lb=0.0,
        ub=1.0,
        vtype=GRB.CONTINUOUS,
        name="loss_task",
    )
    loss_sys = model.addVars([int(s["scenario_id"]) for s in scenarios], lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="loss_sys")
    u_aux = model.addVars([int(s["scenario_id"]) for s in scenarios], lb=0.0, vtype=GRB.CONTINUOUS, name="u")

    #========3. 添加任务放置约束=======
    for task in tasks:
        model.addConstr(gp.quicksum(y[(task.task_id, m)] for m in task.candidates) == 1, name=f"place_{task.task_id}")

    #========4. 添加计算节点容量约束=======
    for node_id, info in compute_nodes.items():
        model.addConstr(
            gp.quicksum(
                float(task.compute_demand) * y[(task.task_id, node_id)]
                for task in tasks
                if node_id in task.candidates
            )
            <= float(info.capacity) * u_node_max,
            name=f"node_cap_{node_id}",
        )

    #========5. 添加任务带宽需求与路径激活约束=======
    for task in tasks:
        for m in task.candidates:
            for phase, demand in (("in", task.b_in), ("out", task.b_out)):
                candidate_paths = [pv for pv in path_vars if pv.task_id == task.task_id and pv.compute_node == m and pv.phase == phase]
                if not candidate_paths:
                    model.addConstr(y[(task.task_id, m)] == 0, name=f"forbid_no_path_{task.task_id}_{m}_{phase}")
                    continue
                model.addConstr(
                    gp.quicksum(x[pv.path_id] for pv in candidate_paths) >= float(demand) * y[(task.task_id, m)],
                    name=f"demand_{task.task_id}_{m}_{phase}",
                )
                for pv in candidate_paths:
                    path_capacity = min(float(config["capacities"][edge]) for edge in pv.edges)
                    model.addConstr(x[pv.path_id] <= path_capacity * y[(task.task_id, m)], name=f"active_{pv.path_id}")

    #========6. 添加链路容量约束=======
    for edge, capacity in config["capacities"].items():
        model.addConstr(
            gp.quicksum(x[pv.path_id] for pv in path_vars if edge_key(*edge) in pv.edges)
            <= float(capacity) * u_link_max,
            name=f"link_cap_{edge[0]}_{edge[1]}",
        )

    #========7. 添加故障场景损失和 CVaR 线性化约束=======
    for scenario in scenarios:
        sid = int(scenario["scenario_id"])
        for task in tasks:
            delivered = {}
            for phase, demand in (("in", task.b_in), ("out", task.b_out)):
                expr = gp.LinExpr()
                for pv in path_vars:
                    if pv.task_id == task.task_id and pv.phase == phase:
                        expr += _path_available(pv, scenario) * x[pv.path_id]
                delivered[phase] = expr
                model.addConstr(
                    loss_task[(sid, task.task_id)] >= 1.0 - expr / float(demand),
                    name=f"loss_{phase}_{sid}_{task.task_id}",
                )

            if config["loss_aggregation"] == "average":
                pass
            else:
                model.addConstr(loss_sys[sid] >= loss_task[(sid, task.task_id)], name=f"loss_sys_{sid}_{task.task_id}")

        if config["loss_aggregation"] == "average":
            model.addConstr(
                loss_sys[sid] >= gp.quicksum(loss_task[(sid, task.task_id)] for task in tasks) / float(len(tasks)),
                name=f"loss_sys_avg_{sid}",
            )
        model.addConstr(u_aux[sid] >= loss_sys[sid] - alpha, name=f"u_aux_{sid}")

    #========8. 设置目标函数并求解=======
    cvar_expr = alpha + gp.quicksum(float(s["probability"]) * u_aux[int(s["scenario_id"])] for s in scenarios) / (1.0 - beta)
    resource_expr = lambda_weight * u_link_max + (1.0 - lambda_weight) * u_node_max
    if risk_mode == "cvar_constraint":
        model.addConstr(cvar_expr <= float(cvar_bound) + 1e-9, name="cvar_sla_bound")
        model.setObjective(resource_expr, GRB.MINIMIZE)
    elif risk_mode == "weighted":
        model.setObjective(risk_weight * cvar_expr + resource_expr, GRB.MINIMIZE)
    else:
        raise ValueError(f"Unsupported risk_mode: {risk_mode}. Use weighted or cvar_constraint.")
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        raise RuntimeError(f"Toy MILP found no feasible solution: gurobi_status_{model.Status}")

    #========9. 提取最优任务放置和路径预留带宽=======
    placement = {}
    for task in tasks:
        placement[task.task_id] = max(task.candidates, key=lambda m: y[(task.task_id, m)].X)

    allocations = []
    for pv in path_vars:
        value = float(x[pv.path_id].X)
        if value <= 1e-8:
            continue
        allocations.append(
            {
                "task": pv.task_id,
                "phase": pv.phase,
                "compute_node": pv.compute_node,
                "path_id": pv.path_id,
                "path_nodes": list(pv.path_nodes),
                "path_edges": [f"{u}-{v}" for u, v in pv.edges],
                "allocation": value,
            }
        )

    #========10. 评估并打包输出=======
    evaluated = _evaluate_solution(config, placement, allocations, scenarios)
    metrics = evaluated["metrics"]
    metrics.update(
        {
            "model_objective": float(model.ObjVal),
            "model_cvar": float(cvar_expr.getValue()),
            "model_alpha": float(alpha.X),
            "model_u_node_max": float(u_node_max.X),
            "model_u_link_max": float(u_link_max.X),
            "risk_mode": risk_mode,
            "cvar_bound": None if cvar_bound is None else float(cvar_bound),
            "cvar_bound_slack": None if cvar_bound is None else float(cvar_bound) - float(cvar_expr.getValue()),
            "beta_mode": config.get("beta_mode", "fixed"),
            "beta_margin": float(config.get("beta_margin", 0.0)),
            "normal_probability": float(config.get("normal_probability", 0.0)),
            "solver_status": "optimal" if model.Status == GRB.OPTIMAL else "suboptimal",
        }
    )

    metrics["solver_mode"] = "single_level"
    best = {
        "status": metrics["solver_status"],
        "reason": "ok",
        "placement": placement,
        "allocations": allocations,
        "scenario_rows": evaluated["scenario_rows"],
        "link_loads": evaluated["link_loads"],
        "metrics": metrics,
    }
    return {"best": best, "all_results": [best], "scenarios": scenarios}


def _candidate_paths(path_vars: list[PathVar], task_id: str, compute_node: str, phase: str) -> list[PathVar]:
    """筛选某个任务、候选计算节点和阶段对应的候选路径。"""
    return [
        pv for pv in path_vars
        if pv.task_id == task_id and pv.compute_node == compute_node and pv.phase == phase
    ]


def _path_survival_ratio(paths: list[PathVar], scenario: dict[str, Any]) -> float:
    """平均分流代理中，一组候选路径在场景下的平均存活比例。"""
    if not paths:
        return 0.0
    return sum(_path_available(path, scenario) for path in paths) / float(len(paths))


def _build_average_split_proxy(config: dict, path_vars: list[PathVar], scenarios: list[dict[str, Any]]):
    """为双层第一层构建平均分流代理损失和代理链路负载。

    proxy_loss[(sid, task, m)] 表示任务放置到 m 时，在场景 sid 下的代理任务损失。
    proxy_load[(task, m, edge)] 表示任务放置到 m 时，平均分流对链路 edge 的正常态预留负载。
    """
    proxy_loss: dict[tuple[int, str, str], float] = {}
    proxy_load: dict[tuple[str, str, tuple[str, str]], float] = {}

    for task in config["tasks"]:
        for m in task.candidates:
            for phase, demand in (("in", task.b_in), ("out", task.b_out)):
                paths = _candidate_paths(path_vars, task.task_id, m, phase)
                if not paths:
                    continue
                split = float(demand) / float(len(paths))
                for pv in paths:
                    for edge in pv.edges:
                        key = (task.task_id, m, edge)
                        proxy_load[key] = proxy_load.get(key, 0.0) + split

            for scenario in scenarios:
                sid = int(scenario["scenario_id"])
                in_paths = _candidate_paths(path_vars, task.task_id, m, "in")
                out_paths = _candidate_paths(path_vars, task.task_id, m, "out")
                in_ratio = _path_survival_ratio(in_paths, scenario)
                out_ratio = _path_survival_ratio(out_paths, scenario)
                loss = max(1.0 - in_ratio, 1.0 - out_ratio, 0.0)
                proxy_loss[(sid, task.task_id, m)] = min(1.0, max(0.0, float(loss)))

    return proxy_loss, proxy_load


def _solve_toy_node_layer_average_split(config: dict, path_vars: list[PathVar], scenarios: list[dict[str, Any]]):
    """双层第一层：用平均分流代理模型求解任务放置。"""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError("gurobipy is required for the toy two-layer node solver.") from exc

    tasks: list[Task] = config["tasks"]
    compute_nodes = config["compute_nodes"]
    beta = float(config["beta"])
    lambda_weight = float(config["lambda_weight"])
    risk_weight = float(config["risk_weight"])
    risk_mode = config.get("node_layer_risk_mode", "weighted")
    cvar_bound = config.get("node_layer_cvar_bound")
    proxy_loss, proxy_load = _build_average_split_proxy(config, path_vars, scenarios)

    model = gp.Model("toy_two_layer_node_average_split")
    model.Params.OutputFlag = 0

    y_keys = [(task.task_id, m) for task in tasks for m in task.candidates]
    y = model.addVars(y_keys, vtype=GRB.BINARY, name="y")
    u_node_max = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="U_node_max")
    u_link_proxy = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="U_link_proxy")
    alpha = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="alpha_node")
    loss_task = model.addVars(
        [(int(s["scenario_id"]), task.task_id) for s in scenarios for task in tasks],
        lb=0.0,
        ub=1.0,
        vtype=GRB.CONTINUOUS,
        name="loss_task_node",
    )
    loss_sys = model.addVars([int(s["scenario_id"]) for s in scenarios], lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="loss_sys_node")
    u_aux = model.addVars([int(s["scenario_id"]) for s in scenarios], lb=0.0, vtype=GRB.CONTINUOUS, name="u_node")

    for task in tasks:
        model.addConstr(gp.quicksum(y[(task.task_id, m)] for m in task.candidates) == 1, name=f"place_{task.task_id}")
        for m in task.candidates:
            if not _candidate_paths(path_vars, task.task_id, m, "in") or not _candidate_paths(path_vars, task.task_id, m, "out"):
                model.addConstr(y[(task.task_id, m)] == 0, name=f"forbid_no_path_{task.task_id}_{m}")

    for node_id, info in compute_nodes.items():
        model.addConstr(
            gp.quicksum(
                float(task.compute_demand) * y[(task.task_id, node_id)]
                for task in tasks
                if node_id in task.candidates
            )
            <= float(info.capacity) * u_node_max,
            name=f"node_cap_{node_id}",
        )

    for edge, capacity in config["capacities"].items():
        model.addConstr(
            gp.quicksum(
                float(proxy_load.get((task.task_id, m, edge), 0.0)) * y[(task.task_id, m)]
                for task in tasks
                for m in task.candidates
            )
            <= float(capacity) * u_link_proxy,
            name=f"proxy_link_cap_{edge[0]}_{edge[1]}",
        )

    for scenario in scenarios:
        sid = int(scenario["scenario_id"])
        for task in tasks:
            model.addConstr(
                loss_task[(sid, task.task_id)]
                >= gp.quicksum(float(proxy_loss[(sid, task.task_id, m)]) * y[(task.task_id, m)] for m in task.candidates),
                name=f"proxy_loss_{sid}_{task.task_id}",
            )
            if config["loss_aggregation"] == "max":
                model.addConstr(loss_sys[sid] >= loss_task[(sid, task.task_id)], name=f"node_loss_sys_{sid}_{task.task_id}")

        if config["loss_aggregation"] == "average":
            model.addConstr(
                loss_sys[sid] >= gp.quicksum(loss_task[(sid, task.task_id)] for task in tasks) / float(len(tasks)),
                name=f"node_loss_sys_avg_{sid}",
            )
        model.addConstr(u_aux[sid] >= loss_sys[sid] - alpha, name=f"node_u_aux_{sid}")

    cvar_expr = alpha + gp.quicksum(float(s["probability"]) * u_aux[int(s["scenario_id"])] for s in scenarios) / (1.0 - beta)
    resource_expr = lambda_weight * u_link_proxy + (1.0 - lambda_weight) * u_node_max
    if risk_mode == "cvar_constraint":
        model.addConstr(cvar_expr <= float(cvar_bound) + 1e-9, name="node_cvar_sla_bound")
        model.setObjective(resource_expr, GRB.MINIMIZE)
    elif risk_mode == "weighted":
        model.setObjective(risk_weight * cvar_expr + resource_expr, GRB.MINIMIZE)
    else:
        raise ValueError(f"Unsupported risk_mode: {risk_mode}. Use weighted or cvar_constraint.")
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        raise RuntimeError(f"Toy two-layer node model found no feasible solution: gurobi_status_{model.Status}")

    placement = {}
    for task in tasks:
        placement[task.task_id] = max(task.candidates, key=lambda m: y[(task.task_id, m)].X)

    return {
        "status": "optimal" if model.Status == GRB.OPTIMAL else "suboptimal",
        "placement": placement,
        "node_objective": float(model.ObjVal),
        "node_cvar": float(cvar_expr.getValue()),
        "node_alpha": float(alpha.X),
        "node_u_node_max": float(u_node_max.X),
        "node_u_link_proxy": float(u_link_proxy.X),
        "node_cvar_bound_slack": None if cvar_bound is None else float(cvar_bound) - float(cvar_expr.getValue()),
        "node_risk_mode": risk_mode,
        "node_cvar_bound": None if cvar_bound is None else float(cvar_bound),
    }


def _solve_toy_link_layer_fixed_placement(
    config: dict,
    path_vars: list[PathVar],
    scenarios: list[dict[str, Any]],
    placement: dict[str, str],
):
    """双层第二层：固定任务放置后优化真实路径预留带宽。"""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError("gurobipy is required for the toy two-layer link solver.") from exc

    tasks: list[Task] = config["tasks"]
    beta = float(config["beta"])
    risk_weight = float(config["risk_weight"])
    risk_mode = config.get("risk_mode", "weighted")
    cvar_bound = config.get("cvar_bound")
    selected_paths = [pv for pv in path_vars if placement[pv.task_id] == pv.compute_node]

    model = gp.Model("toy_two_layer_link")
    model.Params.OutputFlag = 0

    x = model.addVars([pv.path_id for pv in selected_paths], lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    u_link_max = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="U_link_max")
    alpha = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="alpha_link")
    loss_task = model.addVars(
        [(int(s["scenario_id"]), task.task_id) for s in scenarios for task in tasks],
        lb=0.0,
        ub=1.0,
        vtype=GRB.CONTINUOUS,
        name="loss_task_link",
    )
    loss_sys = model.addVars([int(s["scenario_id"]) for s in scenarios], lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="loss_sys_link")
    u_aux = model.addVars([int(s["scenario_id"]) for s in scenarios], lb=0.0, vtype=GRB.CONTINUOUS, name="u_link")

    for task in tasks:
        m = placement[task.task_id]
        for phase, demand in (("in", task.b_in), ("out", task.b_out)):
            paths = _candidate_paths(selected_paths, task.task_id, m, phase)
            if not paths:
                raise RuntimeError(f"No {phase}-paths for task {task.task_id} with selected node {m}.")
            model.addConstr(
                gp.quicksum(x[pv.path_id] for pv in paths) >= float(demand),
                name=f"demand_{task.task_id}_{phase}",
            )
            for pv in paths:
                path_capacity = min(float(config["capacities"][edge]) for edge in pv.edges)
                model.addConstr(x[pv.path_id] <= path_capacity, name=f"path_cap_{pv.path_id}")

    for edge, capacity in config["capacities"].items():
        model.addConstr(
            gp.quicksum(x[pv.path_id] for pv in selected_paths if edge_key(*edge) in pv.edges)
            <= float(capacity) * u_link_max,
            name=f"link_cap_{edge[0]}_{edge[1]}",
        )

    for scenario in scenarios:
        sid = int(scenario["scenario_id"])
        for task in tasks:
            for phase, demand in (("in", task.b_in), ("out", task.b_out)):
                expr = gp.LinExpr()
                for pv in selected_paths:
                    if pv.task_id == task.task_id and pv.phase == phase:
                        expr += _path_available(pv, scenario) * x[pv.path_id]
                model.addConstr(
                    loss_task[(sid, task.task_id)] >= 1.0 - expr / float(demand),
                    name=f"loss_{phase}_{sid}_{task.task_id}",
                )
            if config["loss_aggregation"] == "max":
                model.addConstr(loss_sys[sid] >= loss_task[(sid, task.task_id)], name=f"loss_sys_{sid}_{task.task_id}")

        if config["loss_aggregation"] == "average":
            model.addConstr(
                loss_sys[sid] >= gp.quicksum(loss_task[(sid, task.task_id)] for task in tasks) / float(len(tasks)),
                name=f"loss_sys_avg_{sid}",
            )
        model.addConstr(u_aux[sid] >= loss_sys[sid] - alpha, name=f"u_aux_{sid}")

    cvar_expr = alpha + gp.quicksum(float(s["probability"]) * u_aux[int(s["scenario_id"])] for s in scenarios) / (1.0 - beta)
    if risk_mode == "cvar_constraint":
        model.addConstr(cvar_expr <= float(cvar_bound) + 1e-9, name="link_cvar_sla_bound")
        model.setObjective(u_link_max, GRB.MINIMIZE)
    elif risk_mode == "weighted":
        model.setObjective(risk_weight * cvar_expr + u_link_max, GRB.MINIMIZE)
    else:
        raise ValueError(f"Unsupported risk_mode: {risk_mode}. Use weighted or cvar_constraint.")
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        raise RuntimeError(f"Toy two-layer link model found no feasible solution: gurobi_status_{model.Status}")

    allocations = []
    for pv in selected_paths:
        value = float(x[pv.path_id].X)
        if value <= 1e-8:
            continue
        allocations.append(
            {
                "task": pv.task_id,
                "phase": pv.phase,
                "compute_node": pv.compute_node,
                "path_id": pv.path_id,
                "path_nodes": list(pv.path_nodes),
                "path_edges": [f"{u}-{v}" for u, v in pv.edges],
                "allocation": value,
            }
        )

    return {
        "status": "optimal" if model.Status == GRB.OPTIMAL else "suboptimal",
        "allocations": allocations,
        "link_objective": float(model.ObjVal),
        "link_cvar": float(cvar_expr.getValue()),
        "link_alpha": float(alpha.X),
        "link_u_link_max": float(u_link_max.X),
        "link_cvar_bound_slack": None if cvar_bound is None else float(cvar_bound) - float(cvar_expr.getValue()),
    }


def solve_toy_two_layer_average_split(config: dict) -> dict:
    """求解 toy 平均分流双层模型，并复用单层模型的输出结构。"""
    scenarios = config.get("failure_scenarios") or enumerate_scenarios(config["failure_events"])
    path_vars = _build_path_vars(config)

    node_result = _solve_toy_node_layer_average_split(config, path_vars, scenarios)
    link_result = _solve_toy_link_layer_fixed_placement(config, path_vars, scenarios, node_result["placement"])
    evaluated = _evaluate_solution(config, node_result["placement"], link_result["allocations"], scenarios)

    metrics = evaluated["metrics"]
    metrics.update(
        {
            "solver_mode": "two_layer_average_split",
            "model_objective": float(link_result["link_objective"]),
            "model_cvar": float(link_result["link_cvar"]),
            "model_alpha": float(link_result["link_alpha"]),
            "model_u_node_max": float(node_result["node_u_node_max"]),
            "model_u_link_max": float(link_result["link_u_link_max"]),
            "risk_mode": config.get("risk_mode", "weighted"),
            "cvar_bound": config.get("cvar_bound"),
            "cvar_bound_slack": link_result["link_cvar_bound_slack"],
            "beta_mode": config.get("beta_mode", "fixed"),
            "beta_margin": float(config.get("beta_margin", 0.0)),
            "normal_probability": float(config.get("normal_probability", 0.0)),
            "node_layer_objective": float(node_result["node_objective"]),
            "node_layer_cvar": float(node_result["node_cvar"]),
            "node_layer_alpha": float(node_result["node_alpha"]),
            "node_layer_u_node_max": float(node_result["node_u_node_max"]),
            "node_layer_u_link_proxy": float(node_result["node_u_link_proxy"]),
            "node_layer_cvar_bound_slack": node_result["node_cvar_bound_slack"],
            "node_layer_risk_mode": node_result["node_risk_mode"],
            "node_layer_cvar_bound": node_result["node_cvar_bound"],
            "link_layer_objective": float(link_result["link_objective"]),
            "link_layer_cvar": float(link_result["link_cvar"]),
            "link_layer_alpha": float(link_result["link_alpha"]),
            "link_layer_u_link_max": float(link_result["link_u_link_max"]),
            "link_layer_cvar_bound_slack": link_result["link_cvar_bound_slack"],
            "solver_status": "optimal" if node_result["status"] == "optimal" and link_result["status"] == "optimal" else "suboptimal",
        }
    )

    best = {
        "status": metrics["solver_status"],
        "reason": "ok",
        "placement": node_result["placement"],
        "allocations": link_result["allocations"],
        "scenario_rows": evaluated["scenario_rows"],
        "link_loads": evaluated["link_loads"],
        "metrics": metrics,
    }
    return {"best": best, "all_results": [best], "scenarios": scenarios}


def solve_toy(config: dict) -> dict:
    """根据 config['solver_mode'] 调度单层或平均分流双层求解器。"""
    solver_mode = config.get("solver_mode", "single_level")
    if solver_mode == "single_level":
        return solve_toy_single_level(config)
    if solver_mode == "two_layer_average_split":
        return solve_toy_two_layer_average_split(config)
    raise ValueError(f"Unsupported solver_mode: {solver_mode}.")


def write_results(solution_bundle: dict, output_dir: str | Path):
    """将求解结果写入 CSV/JSON，供画图、论文表格和复现实验使用。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = solution_bundle["best"]

    with (output_dir / "placements.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "status": best["status"],
                "reason": best["reason"],
                "placement": best["placement"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if best["allocations"]:
        with (output_dir / "path_allocations.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(best["allocations"][0].keys()))
            writer.writeheader()
            writer.writerows(best["allocations"])

    with (output_dir / "scenario_losses.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(best["scenario_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(best["scenario_rows"])

    with (output_dir / "link_loads.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(best["link_loads"][0].keys()))
        writer.writeheader()
        writer.writerows(best["link_loads"])

    metric_row = dict(best["metrics"])
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_row.keys()))
        writer.writeheader()
        writer.writerow(metric_row)

    return output_dir


def printable_summary(solution_bundle: dict) -> str:
    """终端输出摘要，展示任务放置、风险指标和每条路径预留带宽。"""
    best = solution_bundle["best"]
    metrics = best["metrics"]
    lines = [
        "[toy] selected placement: " + ", ".join(f"{k}->{v}" for k, v in best["placement"].items()),
        (
            "[toy] selected metrics: "
            f"solver_mode={metrics.get('solver_mode', 'single_level')}, "
            f"risk_mode={metrics['risk_mode']}, beta={metrics['beta']:.6f}, "
            f"obj={metrics['model_objective']:.6f}, model_CVaR={metrics['model_cvar']:.6f}, "
            f"eval_CVaR={metrics['cvar']:.6f}, availability={metrics['availability']:.6f}, "
            f"expected_loss={metrics['expected_loss']:.6f}, max_loss={metrics['max_loss']:.6f}, "
            f"U_node={metrics['model_u_node_max']:.6f}, U_link={metrics['model_u_link_max']:.6f}"
        ),
        "[toy] selected path allocations:",
    ]
    for row in best["allocations"]:
        lines.append(
            f"[toy]   {row['task']} {row['phase']} {row['path_id']} "
            f"{'->'.join(row['path_nodes'])}: x={row['allocation']:.6f}"
        )
    return "\n".join(lines)
