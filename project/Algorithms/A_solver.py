from __future__ import annotations

import random
import time
from typing import Dict, List, Sequence, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from Algorithms.common import build_node_scenario_matrix, build_tunnel_edge_matrix, build_tunnel_scenario_matrix
from util import compute_node_usage


def _assignment_repr(tasks, assignment: Dict[int, int]) -> str:
    """把任务放置结果整理成写入 candidate_summary.csv 的紧凑字符串。"""
    parts = []
    for task in tasks:
        tid = int(task["task_idx"])
        parts.append(f"i{tid}->m{int(assignment[tid])}")
    return ";".join(parts)


def _task_path_sets_for_all_candidates(tasks, pair_to_tunnels):
    """为每个“任务-候选计算节点”预先整理输入段、输出段可选路径。

    单层模型同时决定放置变量 y 和路由变量 x，因此在建模前必须把每个候选放置对应的
    src->m 与 m->dst 路径集合都准备好。若源点或目的点就是执行节点，则对应段为本地段，
    不需要经过链路路径。
    """
    in_paths: Dict[Tuple[int, int], List[int]] = {}
    out_paths: Dict[Tuple[int, int], List[int]] = {}
    local_in: Dict[Tuple[int, int], bool] = {}
    local_out: Dict[Tuple[int, int], bool] = {}

    for task in tasks:
        tid = int(task["task_idx"])
        s = int(task["src"])
        d = int(task["dst"])
        for raw_m in task["candidate_ms"]:
            m = int(raw_m)
            local_in[(tid, m)] = s == m
            local_out[(tid, m)] = d == m
            in_paths[(tid, m)] = [] if s == m else list(pair_to_tunnels.get((s, m), []))
            out_paths[(tid, m)] = [] if d == m else list(pair_to_tunnels.get((m, d), []))

    return in_paths, out_paths, local_in, local_out


def _build_path_capacity(T: Sequence[Sequence[int]], capacity: Sequence[float]):
    """计算每条候选路径的瓶颈容量，路径编号与输入 T 保持从 1 开始的编号方式。"""
    path_capacity: Dict[int, float] = {}
    for tunnel_id, tunnel_edges in enumerate(T, start=1):
        if len(tunnel_edges) == 0:
            path_capacity[tunnel_id] = 0.0
        else:
            path_capacity[tunnel_id] = min(float(capacity[e - 1]) for e in tunnel_edges)
    return path_capacity


def solve_A(
    tasks,
    edges,
    capacity,
    beta,
    lambda_weight,
    scenarios,
    scenario_probs,
    pair_to_tunnels,
    cpu_capacity,
    T,
    node_scenarios=None,
    risk_weight: float = 1.0,
    risk_mode: str = "weighted",
    cvar_bound: float | None = None,
    loss_aggregation: str = "max",
    enable_mip_gap: bool = False,
    mip_gap: float = 0.01,
    time_limit_sec: float | None = None,
    seed: int = 1,
):
    """求解单层方案 A，把放置、路由、利用率和故障损失放在同一个 MILP 中联合优化。

    默认 weighted 模式的目标函数为：
        risk_weight * CVaR + lambda_weight * U_link_max + (1 - lambda_weight) * U_node_max

    cvar_constraint 模式把 CVaR 解释为 SLA 风险约束：
        CVaR <= cvar_bound
        min lambda_weight * U_link_max + (1 - lambda_weight) * U_node_max

    其中 CVaR 衡量故障场景下的尾部损失，U_link_max 衡量正常状态最大链路利用率，
    U_node_max 衡量正常状态最大节点利用率。该模型相当于单层“上帝视角”基准。
    """
    random.seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    nscenarios = len(scenarios)
    nedges = len(edges)
    if nscenarios == 0:
        raise RuntimeError("No scenarios provided for Scheme A.")
    loss_aggregation = str(loss_aggregation).lower()
    if loss_aggregation not in {"max", "average"}:
        raise ValueError("loss_aggregation must be 'max' or 'average'.")
    risk_mode = str(risk_mode or "weighted").lower()
    if risk_mode not in {"weighted", "cvar_constraint"}:
        raise ValueError("risk_mode must be 'weighted' or 'cvar_constraint'.")
    if risk_mode == "cvar_constraint":
        if cvar_bound is None:
            raise ValueError("cvar_bound is required when risk_mode is 'cvar_constraint'.")
        cvar_bound = max(0.0, float(cvar_bound))
    else:
        cvar_bound = None

    L = build_tunnel_edge_matrix(T, nedges)
    X_path = build_tunnel_scenario_matrix(T, edges, scenarios)
    compute_nodes = sorted(int(m) for m in cpu_capacity.keys())
    if node_scenarios is None:
        node_scenarios = [{m: 1.0 for m in compute_nodes} for _ in range(nscenarios)]
    if len(node_scenarios) != nscenarios:
        raise ValueError("node_scenarios must have the same length as scenarios.")
    Eta_node = build_node_scenario_matrix(compute_nodes, node_scenarios)
    node_col = {int(m): idx for idx, m in enumerate(compute_nodes)}
    path_capacity = _build_path_capacity(T, capacity)
    in_paths, out_paths, local_in, local_out = _task_path_sets_for_all_candidates(tasks, pair_to_tunnels)

    y_keys = [
        (int(task["task_idx"]), int(m))
        for task in tasks
        for m in task["candidate_ms"]
    ]
    x_in_keys = [
        (int(task["task_idx"]), int(m), int(tunnel_id))
        for task in tasks
        for m in task["candidate_ms"]
        if not local_in[(int(task["task_idx"]), int(m))]
        for tunnel_id in in_paths[(int(task["task_idx"]), int(m))]
    ]
    x_out_keys = [
        (int(task["task_idx"]), int(m), int(tunnel_id))
        for task in tasks
        for m in task["candidate_ms"]
        if not local_out[(int(task["task_idx"]), int(m))]
        for tunnel_id in out_paths[(int(task["task_idx"]), int(m))]
    ]

    model = gp.Model()
    model.Params.OutputFlag = 0
    if enable_mip_gap:
        model.Params.MIPGap = float(mip_gap)
    if time_limit_sec is not None and float(time_limit_sec) > 0:
        model.Params.TimeLimit = float(time_limit_sec)

    # ===================== 1. 决策变量：同时决定任务放置和端到端流量分配 =====================
    # y[i,m] 表示任务 i 是否放在计算节点 m；x_in/x_out 表示输入段和输出段在候选路径上的带宽。
    # 因为 y 与 x 同时优化，模型可以直接在全局范围内权衡放置、路由、容量和故障损失。
    y = model.addVars(y_keys, vtype=GRB.BINARY, name="y")
    x_in = model.addVars(x_in_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="x_in")
    x_out = model.addVars(x_out_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="x_out")

    # U_node_max 和 U_link_max 是正常状态下的最大利用率变量，取值上界 1 表示不允许超过物理容量。
    # alpha、loss_task、loss_sys、u_aux 构成 CVaR 线性化：先计算每个场景损失，再取尾部风险。
    u_node_max = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="U_node_max")
    u_link_max = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="U_link_max")

    alpha = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="alpha")
    loss_task = model.addVars(
        [(s + 1, int(task["task_idx"])) for s in range(nscenarios) for task in tasks],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="loss_task",
    )
    loss_sys = model.addVars(range(1, nscenarios + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="loss_sys")
    u_aux = model.addVars(range(1, nscenarios + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="u")

    # ===================== 2. 放置约束：每个任务必须且只能选择一个候选计算节点 =====================
    for task in tasks:
        tid = int(task["task_idx"])
        model.addConstr(
            gp.quicksum(y[(tid, int(m))] for m in task["candidate_ms"]) == 1,
            name=f"place_t{tid}",
        )

    # 若某个候选节点缺少输入段或输出段路径，则禁止选择该候选，避免后续流量约束不可行。
    for task in tasks:
        tid = int(task["task_idx"])
        for raw_m in task["candidate_ms"]:
            m = int(raw_m)
            missing_in = (not local_in[(tid, m)]) and len(in_paths[(tid, m)]) == 0
            missing_out = (not local_out[(tid, m)]) and len(out_paths[(tid, m)]) == 0
            if missing_in or missing_out:
                model.addConstr(y[(tid, m)] == 0, name=f"forbid_no_path_t{tid}_m{m}")

    # ===================== 3. 节点容量约束：所有放在节点 m 的任务 CPU 需求不能超过 cap_m * U_node_max =====================
    for m, cap in cpu_capacity.items():
        model.addConstr(
            gp.quicksum(
                float(task["cpu_demand"]) * y[(int(task["task_idx"]), int(m))]
                for task in tasks
                if int(m) in [int(x) for x in task["candidate_ms"]]
            )
            <= float(cap) * u_node_max,
            name=f"node_cap_m{int(m)}",
        )

    # ===================== 4. 流量需求约束：被选中的候选节点必须获得足够输入流量并送出足够输出流量 =====================
    # 这里使用 >= 而不是 ==，允许模型在正常状态预留冗余带宽以降低故障损失；代价会通过 U_link_max 进入目标函数。
    for task in tasks:
        tid = int(task["task_idx"])
        b_in = float(task["b_in"])
        b_out = float(task["b_out"])
        for raw_m in task["candidate_ms"]:
            m = int(raw_m)
            if not local_in[(tid, m)]:
                in_sum = gp.quicksum(x_in[(tid, m, t)] for t in in_paths[(tid, m)])
                model.addConstr(
                    in_sum >= b_in * y[(tid, m)],
                    name=f"in_flow_t{tid}_m{m}",
                )
            if not local_out[(tid, m)]:
                out_sum = gp.quicksum(x_out[(tid, m, t)] for t in out_paths[(tid, m)])
                model.addConstr(
                    out_sum >= b_out * y[(tid, m)],
                    name=f"out_flow_t{tid}_m{m}",
                )

    # ===================== 5. 路径激活约束：只有对应候选节点被选择时，该候选路径上的流量变量才允许为正 =====================
    # 上界取路径瓶颈容量，既收紧模型，也避免使用过大的常数。
    for tid, m, tunnel_id in x_in_keys:
        model.addConstr(
            x_in[(tid, m, tunnel_id)] <= path_capacity[tunnel_id] * y[(tid, m)],
            name=f"in_path_active_t{tid}_m{m}_p{tunnel_id}",
        )
    for tid, m, tunnel_id in x_out_keys:
        model.addConstr(
            x_out[(tid, m, tunnel_id)] <= path_capacity[tunnel_id] * y[(tid, m)],
            name=f"out_path_active_t{tid}_m{m}_p{tunnel_id}",
        )

    # ===================== 6. 链路容量约束：所有经过链路 e 的路径流量之和不能超过 cap_e * U_link_max =====================
    for e in range(nedges):
        link_load = (
            gp.quicksum(x_in[k] * L[k[2] - 1, e] for k in x_in_keys)
            + gp.quicksum(x_out[k] * L[k[2] - 1, e] for k in x_out_keys)
        )
        model.addConstr(link_load <= float(capacity[e]) * u_link_max, name=f"link_cap_e{e+1}")

    # ===================== 7. 故障场景损失约束：计算每个场景下任务实际交付比例，并线性化 CVaR =====================
    # 路径可用性由 X_path 给出，执行节点可用性由 Eta_node 给出；本地段不占链路，但仍受执行节点故障影响。
    for s_idx in range(1, nscenarios + 1):
        for task in tasks:
            tid = int(task["task_idx"])
            b_in = float(task["b_in"])
            b_out = float(task["b_out"])

            delivered_in = gp.LinExpr()
            delivered_out = gp.LinExpr()
            for raw_m in task["candidate_ms"]:
                m = int(raw_m)
                eta_m = float(Eta_node[s_idx - 1, node_col[m]]) if m in node_col else 1.0
                if local_in[(tid, m)]:
                    delivered_in += eta_m * b_in * y[(tid, m)]
                else:
                    delivered_in += gp.quicksum(
                        x_in[(tid, m, t)] * eta_m * X_path[s_idx - 1, t - 1]
                        for t in in_paths[(tid, m)]
                    )

                if local_out[(tid, m)]:
                    delivered_out += eta_m * b_out * y[(tid, m)]
                else:
                    delivered_out += gp.quicksum(
                        x_out[(tid, m, t)] * eta_m * X_path[s_idx - 1, t - 1]
                        for t in out_paths[(tid, m)]
                    )

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
    resource_obj_expr = float(lambda_weight) * u_link_max + (1.0 - float(lambda_weight)) * u_node_max
    # ===================== 8. 目标函数：支持加权风险目标或 CVaR 约束下的资源最小化 =====================
    # weighted 模式保留原始目标；cvar_constraint 模式把 CVaR 上界 Gamma 当作 SLA 约束，
    # 在满足尾部故障损失要求的前提下最小化正常态资源利用率。
    if risk_mode == "cvar_constraint":
        model.addConstr(cvar_expr <= float(cvar_bound) + 1e-9, name="cvar_sla_bound")
        model.setObjective(resource_obj_expr, GRB.MINIMIZE)
    else:
        model.setObjective(
            float(risk_weight) * cvar_expr + resource_obj_expr,
            GRB.MINIMIZE,
        )
    model.optimize()

    if model.Status == GRB.TIME_LIMIT and getattr(model, "SolCount", 0) > 0:
        pass
    elif model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        raise RuntimeError(f"Scheme A found no feasible solution: gurobi_status_{model.Status}")

    assignment: Dict[int, int] = {}
    for task in tasks:
        tid = int(task["task_idx"])
        best_m = max((int(m) for m in task["candidate_ms"]), key=lambda m: y[(tid, m)].X)
        assignment[tid] = best_m

    # 输出阶段把 x 从“任务-候选节点-路径”聚合成“任务-路径”，使结果文件与链路层输出格式保持一致。
    x_in_val: Dict[Tuple[int, int], float] = {}
    for tid, m, tunnel_id in x_in_keys:
        value = float(x_in[(tid, m, tunnel_id)].X)
        if value > 1e-12:
            x_in_val[(tid, tunnel_id)] = x_in_val.get((tid, tunnel_id), 0.0) + value

    x_out_val: Dict[Tuple[int, int], float] = {}
    for tid, m, tunnel_id in x_out_keys:
        value = float(x_out[(tid, m, tunnel_id)].X)
        if value > 1e-12:
            x_out_val[(tid, tunnel_id)] = x_out_val.get((tid, tunnel_id), 0.0) + value

    loss_task_val = {
        (s, int(task["task_idx"])): float(loss_task[(s, int(task["task_idx"]))].X)
        for s in range(1, nscenarios + 1)
        for task in tasks
    }
    loss_sys_val = {s: float(loss_sys[s].X) for s in range(1, nscenarios + 1)}
    u_aux_val = {s: float(u_aux[s].X) for s in range(1, nscenarios + 1)}

    delivered_in_val = {}
    delivered_out_val = {}
    for s_idx in range(1, nscenarios + 1):
        for task in tasks:
            tid = int(task["task_idx"])
            b_in = float(task["b_in"])
            b_out = float(task["b_out"])
            din = 0.0
            dout = 0.0
            for raw_m in task["candidate_ms"]:
                m = int(raw_m)
                y_val = float(y[(tid, m)].X)
                eta_m = float(Eta_node[s_idx - 1, node_col[m]]) if m in node_col else 1.0
                if local_in[(tid, m)]:
                    din += eta_m * b_in * y_val
                else:
                    din += sum(
                        float(x_in[(tid, m, t)].X) * eta_m * X_path[s_idx - 1, t - 1]
                        for t in in_paths[(tid, m)]
                    )
                if local_out[(tid, m)]:
                    dout += eta_m * b_out * y_val
                else:
                    dout += sum(
                        float(x_out[(tid, m, t)].X) * eta_m * X_path[s_idx - 1, t - 1]
                        for t in out_paths[(tid, m)]
                    )
            delivered_in_val[(s_idx, tid)] = float(din)
            delivered_out_val[(s_idx, tid)] = float(dout)

    link_loads = {}
    u_link_recomputed = 0.0
    for e in range(nedges):
        load = 0.0
        for tid, m, tunnel_id in x_in_keys:
            load += float(x_in[(tid, m, tunnel_id)].X) * L[tunnel_id - 1, e]
        for tid, m, tunnel_id in x_out_keys:
            load += float(x_out[(tid, m, tunnel_id)].X) * L[tunnel_id - 1, e]
        link_loads[e + 1] = float(load)
        if capacity[e] > 0:
            u_link_recomputed = max(u_link_recomputed, float(load / capacity[e]))

    _, _, u_node_recomputed = compute_node_usage(tasks, assignment, cpu_capacity)
    u_link_final = max(float(u_link_max.X), u_link_recomputed)
    u_node_final = max(float(u_node_max.X), u_node_recomputed)
    cvar_value = float(cvar_expr.getValue())
    total_score = float(model.ObjVal)
    resource_obj_value = float(lambda_weight) * u_link_final + (1.0 - float(lambda_weight)) * u_node_final
    solve_time = time.time() - t0

    lower = {
        "status": "optimal" if model.Status == GRB.OPTIMAL else ("time_limit" if model.Status == GRB.TIME_LIMIT else "suboptimal"),
        "lower_obj": cvar_value,
        "cvar": cvar_value,
        "alpha": float(alpha.X),
        "solve_time_sec": solve_time,
        "x_in": x_in_val,
        "x_out": x_out_val,
        "loss_task": loss_task_val,
        "loss_sys": loss_sys_val,
        "u_aux": u_aux_val,
        "delivered_in": delivered_in_val,
        "delivered_out": delivered_out_val,
        "link_loads": link_loads,
        "u_link_max": u_link_final,
        "u_node_max": u_node_final,
        "single_level_obj": total_score,
        "single_level_resource_obj": resource_obj_value,
        "single_level_risk_mode": risk_mode,
        "cvar_bound": cvar_bound,
        "cvar_bound_slack": (float(cvar_bound) - cvar_value) if cvar_bound is not None else None,
        "risk_weight": float(risk_weight),
        "loss_aggregation": loss_aggregation,
        "enable_mip_gap": bool(enable_mip_gap),
        "mip_gap": float(mip_gap) if enable_mip_gap else None,
        "time_limit_sec": float(time_limit_sec) if time_limit_sec is not None and float(time_limit_sec) > 0 else None,
        "mip_gap_reached": float(model.MIPGap) if model.IsMIP else None,
        "obj_bound": float(model.ObjBound) if model.IsMIP else None,
    }

    candidate_row = {
        "candidate_id": 1,
        "assignment_repr": _assignment_repr(tasks, assignment),
        "is_feasible": True,
        "node_feasible": True,
        "lower_status": lower["status"],
        "u_node_max": u_node_final,
        "u_link_max": u_link_final,
        "lower_obj": cvar_value,
        "cvar": cvar_value,
        "alpha": float(alpha.X),
        "beta": beta,
        "lambda_weight": lambda_weight,
        "risk_weight": float(risk_weight),
        "single_level_risk_mode": risk_mode,
        "cvar_bound": cvar_bound if cvar_bound is not None else "",
        "cvar_bound_slack": (float(cvar_bound) - cvar_value) if cvar_bound is not None else "",
        "loss_aggregation": loss_aggregation,
        "enable_mip_gap": bool(enable_mip_gap),
        "mip_gap": float(mip_gap) if enable_mip_gap else "",
        "time_limit_sec": float(time_limit_sec) if time_limit_sec is not None and float(time_limit_sec) > 0 else "",
        "mip_gap_reached": float(model.MIPGap) if model.IsMIP else "",
        "total_score": total_score,
        "resource_obj": resource_obj_value,
        "solve_time_sec": solve_time,
        "note": "single_level_scheme_A",
    }

    return {
        "best_assignment": assignment,
        "best_score": total_score,
        "u_node_max": u_node_final,
        "u_link_max": u_link_final,
        "best_lower": lower,
        "candidate_rows": [candidate_row],
        "total_evaluated": 1,
        "search_mode": "single_level_A",
        "L": L,
        "X_path": X_path,
        "Eta_node": Eta_node,
    }
