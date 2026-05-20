from __future__ import annotations

import time
from typing import Dict, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

from Algorithms.common import (
    build_node_scenario_matrix,
    build_tunnel_edge_matrix,
    build_tunnel_scenario_matrix,
)


def _build_path_capacity(T: Sequence[Sequence[int]], capacity: Sequence[float]):
    """计算每条候选路径的瓶颈容量，路径编号与输入 T 保持从 1 开始的编号方式。"""
    path_capacity = {}
    for tunnel_id, tunnel_edges in enumerate(T, start=1):
        if len(tunnel_edges) == 0:
            path_capacity[tunnel_id] = 0.0
        else:
            path_capacity[tunnel_id] = min(float(capacity[e - 1]) for e in tunnel_edges)
    return path_capacity


def _build_task_path_sets(tasks, assignment: Dict[int, int], pair_to_tunnels):
    in_paths = {}
    out_paths = {}
    local_in = {}
    local_out = {}
    for task in tasks:
        tid = int(task["task_idx"])
        src = int(task["src"])
        dst = int(task["dst"])
        m = int(assignment[tid])
        local_in[tid] = src == m
        local_out[tid] = dst == m
        in_paths[tid] = [] if local_in[tid] else list(pair_to_tunnels.get((src, m), []))
        out_paths[tid] = [] if local_out[tid] else list(pair_to_tunnels.get((m, dst), []))
    return in_paths, out_paths, local_in, local_out


def solve_link_layer(
    tasks,
    assignment: Dict[int, int],
    edges: Sequence[Tuple[int, int]],
    capacity,
    beta: float,
    scenarios,
    scenario_probs,
    node_scenarios,
    pair_to_tunnels,
    cpu_capacity: Dict[int, float],
    T,
    risk_weight: float = 0.5,
    loss_aggregation: str = "average",
    optimization_mode: str = "weighted",
    cvar_tolerance: float = 0.0,
):
    """在给定任务放置 assignment 后，求解链路层路由和故障损失。

    链路层只优化路径流量 x，不再改变任务放置。可选目标有两种：
    weighted：最小化 risk_weight * CVaR + (1 - risk_weight) * U_link_max；
    cvar_then_u：先最小化 CVaR，再在 CVaR 容忍范围内最小化 U_link_max。
    """
    t0 = time.time()
    nscenarios = len(scenarios)
    nedges = len(edges)
    if nscenarios == 0:
        return {"status": "infeasible", "reason": "no scenarios"}

    loss_aggregation = str(loss_aggregation).lower()
    if loss_aggregation not in {"max", "average"}:
        raise ValueError("loss_aggregation must be 'max' or 'average'.")
    optimization_mode = str(optimization_mode).lower()
    if optimization_mode not in {"weighted", "cvar_then_u"}:
        raise ValueError("optimization_mode must be 'weighted' or 'cvar_then_u'.")
    cvar_tolerance = max(0.0, float(cvar_tolerance))

    L = build_tunnel_edge_matrix(T, nedges)
    X_path = build_tunnel_scenario_matrix(T, edges, scenarios)
    path_capacity = _build_path_capacity(T, capacity)
    compute_nodes = sorted(int(m) for m in cpu_capacity.keys())
    if node_scenarios is None:
        node_scenarios = [{m: 1.0 for m in compute_nodes} for _ in range(nscenarios)]
    if len(node_scenarios) != nscenarios:
        raise ValueError("node_scenarios must have the same length as scenarios.")
    eta_node = build_node_scenario_matrix(compute_nodes, node_scenarios)
    node_col = {int(m): idx for idx, m in enumerate(compute_nodes)}

    in_paths, out_paths, local_in, local_out = _build_task_path_sets(tasks, assignment, pair_to_tunnels)
    for task in tasks:
        tid = int(task["task_idx"])
        m = int(assignment[tid])
        if not local_in[tid] and len(in_paths[tid]) == 0:
            return {"status": "infeasible", "reason": f"no in-path for task {tid} with selected node {m}"}
        if not local_out[tid] and len(out_paths[tid]) == 0:
            return {"status": "infeasible", "reason": f"no out-path for task {tid} with selected node {m}"}

    x_in_keys = [(int(task["task_idx"]), int(tid)) for task in tasks if not local_in[int(task["task_idx"])] for tid in in_paths[int(task["task_idx"])]]
    x_out_keys = [(int(task["task_idx"]), int(tid)) for task in tasks if not local_out[int(task["task_idx"])] for tid in out_paths[int(task["task_idx"])]]

    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.Threads = 1

    # ===================== 1. 决策变量：固定放置后的输入段/输出段路径流量 =====================
    # x_in/x_out 只在已选执行节点对应的路径集合上定义；U_link_max 表示正常状态最大链路利用率。
    # alpha、loss_task、loss_sys、u_aux 用于把故障损失写成 CVaR 线性形式。
    x_in = model.addVars(x_in_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="x_in")
    x_out = model.addVars(x_out_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="x_out")
    u_link_max = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="U_link_max")
    alpha = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="alpha_link")
    loss_task = model.addVars(
        [(s + 1, int(task["task_idx"])) for s in range(nscenarios) for task in tasks],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="loss_task_link",
    )
    loss_sys = model.addVars(range(1, nscenarios + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="loss_sys_link")
    u_aux = model.addVars(range(1, nscenarios + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="u_link")

    for task in tasks:
        tid = int(task["task_idx"])
        if not local_in[tid]:
            # ===================== 2. 流量需求约束：非本地输入段需要在候选路径上分配至少 b_in 带宽 =====================
            # 使用 >= 允许冗余带宽；冗余可能降低故障损失，但会提高 U_link_max 并进入目标函数。
            model.addConstr(
                gp.quicksum(x_in[(tid, t)] for t in in_paths[tid]) >= float(task["b_in"]),
                name=f"in_flow_t{tid}",
            )
        if not local_out[tid]:
            model.addConstr(
                gp.quicksum(x_out[(tid, t)] for t in out_paths[tid]) >= float(task["b_out"]),
                name=f"out_flow_t{tid}",
            )

    # ===================== 3. 路径容量约束：每条路径流量不超过该路径的瓶颈容量 =====================
    # 固定放置后不需要 y 激活变量，直接用路径瓶颈容量约束 x，可减少无意义的可行域。
    for tid, tunnel_id in x_in_keys:
        model.addConstr(
            x_in[(tid, tunnel_id)] <= float(path_capacity[tunnel_id]),
            name=f"in_path_cap_t{tid}_p{tunnel_id}",
        )
    for tid, tunnel_id in x_out_keys:
        model.addConstr(
            x_out[(tid, tunnel_id)] <= float(path_capacity[tunnel_id]),
            name=f"out_path_cap_t{tid}_p{tunnel_id}",
        )

    # ===================== 4. 链路容量约束：经过链路 e 的全部路径流量不超过 cap_e * U_link_max =====================
    for e in range(nedges):
        link_load = (
            gp.quicksum(x_in[k] * L[k[1] - 1, e] for k in x_in_keys)
            + gp.quicksum(x_out[k] * L[k[1] - 1, e] for k in x_out_keys)
        )
        model.addConstr(link_load <= float(capacity[e]) * u_link_max, name=f"link_cap_e{e+1}")

    # ===================== 5. 故障场景损失约束：路径故障和执行节点故障共同决定每个任务的交付比例 =====================
    # loss_aggregation=max 时场景损失取最坏任务；average 时取任务平均损失。
    for s_idx in range(1, nscenarios + 1):
        for task in tasks:
            tid = int(task["task_idx"])
            m = int(assignment[tid])
            eta_m = float(eta_node[s_idx - 1, node_col[m]]) if m in node_col else 1.0
            b_in = float(task["b_in"])
            b_out = float(task["b_out"])

            if local_in[tid]:
                delivered_in = eta_m * b_in
            else:
                delivered_in = gp.quicksum(x_in[(tid, t)] * eta_m * X_path[s_idx - 1, t - 1] for t in in_paths[tid])
            if local_out[tid]:
                delivered_out = eta_m * b_out
            else:
                delivered_out = gp.quicksum(x_out[(tid, t)] * eta_m * X_path[s_idx - 1, t - 1] for t in out_paths[tid])

            if b_in > 0:
                model.addConstr(loss_task[(s_idx, tid)] >= 1 - delivered_in / b_in, name=f"loss_in_s{s_idx}_t{tid}")
            if b_out > 0:
                model.addConstr(loss_task[(s_idx, tid)] >= 1 - delivered_out / b_out, name=f"loss_out_s{s_idx}_t{tid}")
            if loss_aggregation == "max":
                model.addConstr(loss_sys[s_idx] >= loss_task[(s_idx, tid)], name=f"loss_sys_s{s_idx}_t{tid}")

        if loss_aggregation == "average":
            model.addConstr(
                loss_sys[s_idx]
                >= gp.quicksum(loss_task[(s_idx, int(task["task_idx"]))] for task in tasks) / float(len(tasks)),
                name=f"loss_sys_avg_s{s_idx}",
            )

        model.addConstr(u_aux[s_idx] >= loss_sys[s_idx] - alpha, name=f"u_aux_s{s_idx}")
        model.addConstr(u_aux[s_idx] >= 0, name=f"u_nonneg_s{s_idx}")

    cvar_expr = alpha + (1.0 / (1.0 - beta)) * gp.quicksum(
        float(scenario_probs[s - 1]) * u_aux[s] for s in range(1, nscenarios + 1)
    )
    weighted_obj = float(risk_weight) * cvar_expr + (1.0 - float(risk_weight)) * u_link_max
    first_stage_cvar = None
    # ===================== 6. 目标函数：支持“加权折中”和“先损失后利用率”的两阶段优化 =====================
    if optimization_mode == "cvar_then_u":
        model.setObjective(cvar_expr, GRB.MINIMIZE)
    else:
        model.setObjective(weighted_obj, GRB.MINIMIZE)
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        return {"status": "infeasible", "reason": f"gurobi_status_{model.Status}"}

    if optimization_mode == "cvar_then_u":
        first_stage_cvar = float(cvar_expr.getValue())
        model.addConstr(
            cvar_expr <= first_stage_cvar + cvar_tolerance + 1e-9,
            name="link_cvar_then_u_quality",
        )
        model.setObjective(u_link_max, GRB.MINIMIZE)
        model.optimize()
        if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
            return {"status": "infeasible", "reason": f"gurobi_status_{model.Status}_stage2"}

    x_in_val = {k: float(x_in[k].X) for k in x_in_keys if x_in[k].X > 1e-12}
    x_out_val = {k: float(x_out[k].X) for k in x_out_keys if x_out[k].X > 1e-12}
    loss_task_val = {(s, int(task["task_idx"])): float(loss_task[(s, int(task["task_idx"]))].X) for s in range(1, nscenarios + 1) for task in tasks}
    loss_sys_val = {s: float(loss_sys[s].X) for s in range(1, nscenarios + 1)}
    u_aux_val = {s: float(u_aux[s].X) for s in range(1, nscenarios + 1)}

    delivered_in = {}
    delivered_out = {}
    for s_idx in range(1, nscenarios + 1):
        for task in tasks:
            tid = int(task["task_idx"])
            m = int(assignment[tid])
            eta_m = float(eta_node[s_idx - 1, node_col[m]]) if m in node_col else 1.0
            if local_in[tid]:
                delivered_in[(s_idx, tid)] = eta_m * float(task["b_in"])
            else:
                delivered_in[(s_idx, tid)] = float(sum(x_in[(tid, t)].X * eta_m * X_path[s_idx - 1, t - 1] for t in in_paths[tid]))
            if local_out[tid]:
                delivered_out[(s_idx, tid)] = eta_m * float(task["b_out"])
            else:
                delivered_out[(s_idx, tid)] = float(sum(x_out[(tid, t)].X * eta_m * X_path[s_idx - 1, t - 1] for t in out_paths[tid]))

    link_loads = {}
    u_link_recomputed = 0.0
    for e in range(nedges):
        load = 0.0
        for (tid, t), value in x_in_val.items():
            load += value * L[t - 1, e]
        for (tid, t), value in x_out_val.items():
            load += value * L[t - 1, e]
        link_loads[e + 1] = float(load)
        if capacity[e] > 0:
            u_link_recomputed = max(u_link_recomputed, float(load / capacity[e]))

    solve_time = time.time() - t0
    cvar_value = float(cvar_expr.getValue())
    u_link_final = max(float(u_link_max.X), u_link_recomputed)
    weighted_obj_value = float(risk_weight) * cvar_value + (1.0 - float(risk_weight)) * u_link_final

    return {
        "status": "optimal" if model.Status == GRB.OPTIMAL else "suboptimal",
        "lower_obj": cvar_value,
        "cvar": cvar_value,
        "alpha": float(alpha.X),
        "solve_time_sec": solve_time,
        "x_in": x_in_val,
        "x_out": x_out_val,
        "loss_task": loss_task_val,
        "loss_sys": loss_sys_val,
        "u_aux": u_aux_val,
        "delivered_in": delivered_in,
        "delivered_out": delivered_out,
        "link_loads": link_loads,
        "u_link_max": u_link_final,
        "link_obj": weighted_obj_value,
        "link_solver_obj": float(model.ObjVal),
        "link_optimization_mode": optimization_mode,
        "link_cvar_tolerance": cvar_tolerance,
        "link_first_stage_cvar": first_stage_cvar,
        "risk_weight": float(risk_weight),
        "loss_aggregation": loss_aggregation,
        "L": L,
        "X_path": X_path,
        "Eta_node": eta_node,
    }
