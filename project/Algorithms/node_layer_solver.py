from __future__ import annotations

import time
from typing import Dict, Mapping, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

from Algorithms.common import build_node_scenario_matrix, build_tunnel_scenario_matrix
from util import compute_node_usage


# ===================== 1. 通用候选工具：检查路径可行性、格式化放置方案、生成 no-good 约束 =====================
# 这些函数不属于某个具体 layered 版本，主要服务于节点层候选生成和候选去重。


def _candidate_has_required_paths(task, m: int, pair_to_tunnels) -> bool:
    src = int(task["src"])
    dst = int(task["dst"])
    has_in = src == m or len(pair_to_tunnels.get((src, m), [])) > 0
    has_out = dst == m or len(pair_to_tunnels.get((m, dst), [])) > 0
    return has_in and has_out


def _assignment_repr(tasks, assignment: Dict[int, int]) -> str:
    return ";".join(f"i{int(task['task_idx'])}->m{int(assignment[int(task['task_idx'])])}" for task in tasks)


def _extract_assignment(tasks, y, var_value) -> Dict[int, int]:
    """根据 y 变量值提取每个任务选择的计算节点。"""
    assignment: Dict[int, int] = {}
    for task in tasks:
        tid = int(task["task_idx"])
        assignment[tid] = max(
            (int(m) for m in task["candidate_ms"]),
            key=lambda m: var_value(y[(tid, int(m))]),
        )
    return assignment


def _add_no_good_constraint(model, tasks, y, assignment: Mapping[int, int], name: str):
    """排除一个已经生成过的完整放置方案，避免候选集中出现重复 assignment。"""
    selected_terms = []
    for task in tasks:
        tid = int(task["task_idx"])
        selected_m = int(assignment.get(tid, -1))
        if (tid, selected_m) in y:
            selected_terms.append(y[(tid, selected_m)])
    if selected_terms:
        return model.addConstr(gp.quicksum(selected_terms) <= len(selected_terms) - 1, name=name)
    return None


def _assignment_network_affinity(tasks, assignment: Mapping[int, int], network_affinity) -> float | None:
    """计算一个完整放置方案的平均网络亲和度；没有亲和度表时返回 None。"""
    if network_affinity is None:
        return None
    total_affinity = 0.0
    for task in tasks:
        tid = int(task["task_idx"])
        selected_m = int(assignment[tid])
        total_affinity += float(network_affinity.get((tid, selected_m), 1.0))
    return total_affinity / float(len(tasks))


# ===================== 2. layered2 代理模型工具：用平均分流近似节点层对链路故障和链路负载的感知 =====================
# layered2 不在节点层引入真实路由变量 x，而是基于固定 k_paths 平均分流预先计算线性代理系数。


def build_average_split_proxy_loss(
    tasks,
    compute_nodes: Sequence[int],
    scenarios,
    node_scenarios,
    pair_to_tunnels,
    T,
    edges,
    k_paths: int,
) -> Dict[Tuple[int, int, int], float]:
    """为 layered2 预计算节点层代理损失：假设候选路径平均分流，再估计故障场景下的任务损失。"""
    nscenarios = len(scenarios)
    compute_nodes = [int(m) for m in compute_nodes]
    if node_scenarios is None:
        node_scenarios = [{m: 1.0 for m in compute_nodes} for _ in range(nscenarios)]
    if len(node_scenarios) != nscenarios:
        raise ValueError("node_scenarios must have the same length as scenarios.")

    X_path = build_tunnel_scenario_matrix(T, edges, scenarios)
    eta_node = build_node_scenario_matrix(compute_nodes, node_scenarios)
    node_col = {int(m): idx for idx, m in enumerate(compute_nodes)}
    k_paths = max(1, int(k_paths))

    def average_path_survival(s_idx: int, tunnel_ids: Sequence[int]) -> float:
        selected = list(tunnel_ids)[:k_paths]
        if not selected:
            return 0.0
        return float(sum(float(X_path[s_idx - 1, int(tid) - 1]) for tid in selected) / len(selected))

    proxy_loss: Dict[Tuple[int, int, int], float] = {}
    for s_idx in range(1, nscenarios + 1):
        for task in tasks:
            tid = int(task["task_idx"])
            src = int(task["src"])
            dst = int(task["dst"])
            for raw_m in task["candidate_ms"]:
                m = int(raw_m)
                eta_m = float(eta_node[s_idx - 1, node_col[m]]) if m in node_col else 1.0

                in_survival = 1.0 if src == m else average_path_survival(s_idx, pair_to_tunnels.get((src, m), []))
                out_survival = 1.0 if dst == m else average_path_survival(s_idx, pair_to_tunnels.get((m, dst), []))

                delivered_in_ratio = eta_m * in_survival
                delivered_out_ratio = eta_m * out_survival
                loss = max(1.0 - delivered_in_ratio, 1.0 - delivered_out_ratio, 0.0)
                proxy_loss[(s_idx, tid, m)] = min(1.0, max(0.0, float(loss)))
    return proxy_loss


def build_average_split_proxy_link_load(
    tasks,
    pair_to_tunnels,
    T,
    nedges: int,
    k_paths: int,
) -> Dict[Tuple[int, int, int], float]:
    """为 layered2 预计算节点层代理链路负载：假设候选路径平均分流，得到每个放置选择对链路的占用。"""
    k_paths = max(1, int(k_paths))
    proxy_load: Dict[Tuple[int, int, int], float] = {}

    def add_path_load(tid: int, m: int, tunnel_ids: Sequence[int], bandwidth: float):
        selected = list(tunnel_ids)[:k_paths]
        if not selected:
            return
        split = float(bandwidth) / float(len(selected))
        for tunnel_id in selected:
            if int(tunnel_id) <= 0 or int(tunnel_id) > len(T):
                continue
            for edge_id in T[int(tunnel_id) - 1]:
                if 1 <= int(edge_id) <= int(nedges):
                    key = (tid, m, int(edge_id))
                    proxy_load[key] = proxy_load.get(key, 0.0) + split

    for task in tasks:
        tid = int(task["task_idx"])
        src = int(task["src"])
        dst = int(task["dst"])
        b_in = float(task["b_in"])
        b_out = float(task["b_out"])
        for raw_m in task["candidate_ms"]:
            m = int(raw_m)
            if src != m:
                add_path_load(tid, m, pair_to_tunnels.get((src, m), []), b_in)
            if dst != m:
                add_path_load(tid, m, pair_to_tunnels.get((m, dst), []), b_out)
    return proxy_load


# ===================== 3. layered3 网络指标工具：生成网络亲和度，用于节点层候选排序 =====================
# 这些指标只影响候选生成，不替代链路层的真实路由优化。


def build_network_affinity_cost(
    tasks,
    pair_to_tunnels,
    T,
    k_paths: int,
    count_weight: float = 0.4,
    overlap_weight: float = 0.4,
    length_weight: float = 0.2,
) -> Dict[Tuple[int, int], float]:
    """为 layered3 预计算网络亲和度：路径少、重叠高、路径长的候选放置会得到更高惩罚。"""
    k_paths = max(1, int(k_paths))
    max_path_len = max((len(path) for path in T if len(path) > 0), default=1)

    def leg_cost(tunnel_ids: Sequence[int]) -> float:
        selected = [int(tid) for tid in list(tunnel_ids)[:k_paths] if 0 < int(tid) <= len(T)]
        if not selected:
            return 1.0

        edge_counts: Dict[int, int] = {}
        total_len = 0.0
        for tunnel_id in selected:
            path = T[tunnel_id - 1]
            total_len += len(path)
            for edge_id in path:
                edge_counts[int(edge_id)] = edge_counts.get(int(edge_id), 0) + 1

        path_count_penalty = 1.0 / float(len(selected))
        overlap_penalty = max(edge_counts.values(), default=0) / float(len(selected))
        length_penalty = (total_len / float(len(selected))) / float(max_path_len)
        return (
            float(count_weight) * path_count_penalty
            + float(overlap_weight) * overlap_penalty
            + float(length_weight) * length_penalty
        )

    affinity: Dict[Tuple[int, int], float] = {}
    for task in tasks:
        tid = int(task["task_idx"])
        src = int(task["src"])
        dst = int(task["dst"])
        b_in = float(task["b_in"])
        b_out = float(task["b_out"])
        denom = b_in + b_out
        for raw_m in task["candidate_ms"]:
            m = int(raw_m)
            in_cost = 0.0 if src == m else leg_cost(pair_to_tunnels.get((src, m), []))
            out_cost = 0.0 if dst == m else leg_cost(pair_to_tunnels.get((m, dst), []))
            if denom > 0:
                cost = (b_in * in_cost + b_out * out_cost) / denom
            else:
                cost = 0.5 * (in_cost + out_cost)
            affinity[(tid, m)] = float(cost)
    return affinity


# ===================== 4. 节点层主求解器：构建基础放置模型，并按 layered1/2/3 参数生成候选方案 =====================
# 外部仍然只调用 solve_node_layer；后续若拆 layered1/2/3，应优先复用本函数中的公共建模块。


def _add_placement_constraints(model, tasks, y, pair_to_tunnels):
    """添加基础放置约束：每个任务选一个候选节点，并禁止缺少路径的候选。"""
    for task in tasks:
        tid = int(task["task_idx"])
        model.addConstr(
            gp.quicksum(y[(tid, int(m))] for m in task["candidate_ms"]) == 1,
            name=f"place_t{tid}",
        )
        for raw_m in task["candidate_ms"]:
            m = int(raw_m)
            if not _candidate_has_required_paths(task, m, pair_to_tunnels):
                model.addConstr(y[(tid, m)] == 0, name=f"forbid_no_path_t{tid}_m{m}")


def _add_node_capacity_constraints(model, tasks, y, cpu_capacity, u_node_max):
    """添加节点容量约束：节点 m 的 CPU 总需求不能超过 cap_m * U_node_max。"""
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


def _add_proxy_link_constraints(model, tasks, y, proxy_link_load, link_capacity, u_link_proxy):
    """添加代理链路利用率约束，返回每条链路的代理利用率表达式列表。"""
    proxy_link_util_exprs = []
    for edge_idx, cap in enumerate(link_capacity, start=1):
        if float(cap) <= 0:
            continue
        proxy_load_expr = gp.quicksum(
            float(proxy_link_load.get((int(task["task_idx"]), int(m), edge_idx), 0.0))
            * y[(int(task["task_idx"]), int(m))]
            for task in tasks
            for m in task["candidate_ms"]
        )
        proxy_util_expr = proxy_load_expr / float(cap)
        proxy_link_util_exprs.append(proxy_util_expr)
        model.addConstr(proxy_util_expr <= u_link_proxy, name=f"proxy_link_util_e{edge_idx}")
    return proxy_link_util_exprs


def _add_node_loss_constraints(
    model,
    tasks,
    y,
    nscenarios: int,
    eta_node,
    node_col,
    proxy_loss,
    loss_aggregation: str,
    loss_task,
    loss_sys,
    u_aux,
    alpha,
):
    """添加节点层损失和 CVaR 辅助约束。"""
    for s_idx in range(1, nscenarios + 1):
        for task in tasks:
            tid = int(task["task_idx"])
            if proxy_loss is None:
                selected_available = gp.quicksum(
                    float(eta_node[s_idx - 1, node_col[int(m)]]) * y[(tid, int(m))]
                    for m in task["candidate_ms"]
                )
                model.addConstr(loss_task[(s_idx, tid)] >= 1 - selected_available, name=f"node_loss_s{s_idx}_t{tid}")
            else:
                selected_proxy_loss = gp.quicksum(
                    float(proxy_loss.get((s_idx, tid, int(m)), 1.0)) * y[(tid, int(m))]
                    for m in task["candidate_ms"]
                )
                model.addConstr(
                    loss_task[(s_idx, tid)] >= selected_proxy_loss,
                    name=f"node_proxy_loss_s{s_idx}_t{tid}",
                )
            if loss_aggregation == "max":
                model.addConstr(loss_sys[s_idx] >= loss_task[(s_idx, tid)], name=f"node_loss_sys_s{s_idx}_t{tid}")

        if loss_aggregation == "average":
            model.addConstr(
                loss_sys[s_idx]
                >= gp.quicksum(loss_task[(s_idx, int(task["task_idx"]))] for task in tasks) / float(len(tasks)),
                name=f"node_loss_sys_avg_s{s_idx}",
            )

        model.addConstr(u_aux[s_idx] >= loss_sys[s_idx] - alpha, name=f"node_u_aux_s{s_idx}")
        model.addConstr(u_aux[s_idx] >= 0, name=f"node_u_nonneg_s{s_idx}")


def solve_node_layer(
    tasks,
    compute_nodes: Sequence[int],
    cpu_capacity: Dict[int, float],
    node_scenarios,
    node_scenario_probs,
    pair_to_tunnels,
    beta: float,
    risk_weight: float = 0.5,
    loss_aggregation: str = "average",
    seed: int = 1,
    proxy_loss: Mapping[Tuple[int, int, int], float] | None = None,
    proxy_link_load: Mapping[Tuple[int, int, int], float] | None = None,
    link_capacity: Sequence[float] | None = None,
    proxy_link_cap_enforced: bool = True,
    proxy_link_weight: float = 0.0,
    network_affinity: Mapping[Tuple[int, int], float] | None = None,
    node_loss_model: str = "node_failure",
    risk_mode: str = "weighted",
    cvar_bound: float | None = None,
    candidate_budget: int = 1,
    node_u_tol: float = 0.0,
    node_cvar_tol: float = 1e-4,
    affinity_tol: float | None = None,
    forbidden_assignments: Sequence[Mapping[int, int]] | None = None,
):
    """求解分层算法的节点层模型，只决定任务放置，不引入真实路由变量 x。

    基础目标为 risk_weight * 节点层 CVaR + node_load_weight * U_node_max。
    layered2 可额外加入代理链路利用率 U_link_proxy；layered3 可额外按网络亲和度生成候选方案。
    节点层输出一个或多个放置候选，再交给链路层做真实流量分配和故障损失评估。
    """
    t0 = time.time()
    nscenarios = len(node_scenarios)
    if nscenarios == 0:
        raise RuntimeError("No node scenarios provided for layered node model.")

    loss_aggregation = str(loss_aggregation).lower()
    if loss_aggregation not in {"max", "average"}:
        raise ValueError("loss_aggregation must be 'max' or 'average'.")
    risk_mode = str(risk_mode).lower()
    if risk_mode not in {"weighted", "cvar_constraint"}:
        raise ValueError("risk_mode must be 'weighted' or 'cvar_constraint'.")
    if risk_mode == "cvar_constraint":
        if cvar_bound is None:
            raise ValueError("cvar_bound is required when node risk_mode is 'cvar_constraint'.")
        cvar_bound = max(0.0, float(cvar_bound))
    else:
        cvar_bound = None

    compute_nodes = [int(m) for m in compute_nodes]
    eta_node = build_node_scenario_matrix(compute_nodes, node_scenarios)
    node_col = {int(m): idx for idx, m in enumerate(compute_nodes)}
    y_keys = [(int(task["task_idx"]), int(m)) for task in tasks for m in task["candidate_ms"]]
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.Seed = int(seed)
    model.Params.Threads = 1
    candidate_budget = max(1, int(candidate_budget))
    node_u_tol = max(0.0, float(node_u_tol))
    node_cvar_tol = max(0.0, float(node_cvar_tol))
    affinity_tol = None if affinity_tol is None else max(0.0, float(affinity_tol))

    # ===================== 1. 决策变量：节点层只选择任务放置，不创建真实流量变量 x =====================
    # y[i,m] 表示任务 i 是否放在计算节点 m；U_node_max 表示最大节点利用率。
    # layered2 提供代理链路负载时，U_link_proxy 表示平均分流规则下的最大代理链路利用率。
    y = model.addVars(y_keys, vtype=GRB.BINARY, name="y")
    u_node_max = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="U_node_max")
    u_link_proxy = None
    if proxy_link_load is not None:
        proxy_ub = 1.0 if proxy_link_cap_enforced else GRB.INFINITY
        u_link_proxy = model.addVar(lb=0.0, ub=proxy_ub, vtype=GRB.CONTINUOUS, name="U_link_proxy")
    alpha = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="alpha_node")
    loss_task = model.addVars(
        [(s + 1, int(task["task_idx"])) for s in range(nscenarios) for task in tasks],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="loss_task_node",
    )
    loss_sys = model.addVars(range(1, nscenarios + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="loss_sys_node")
    u_aux = model.addVars(range(1, nscenarios + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="u_node")

    # ===================== 2. 放置可行性约束：每个任务选一个候选节点，并排除缺少输入/输出路径的候选 =====================
    _add_placement_constraints(model, tasks, y, pair_to_tunnels)

    for idx, assignment in enumerate(forbidden_assignments or []):
        # 已经评估过的完整放置方案通过 no-good 约束排除，用于生成不重复候选。
        _add_no_good_constraint(model, tasks, y, assignment, name=f"forbidden_assignment_{idx}")

    # ===================== 3. 节点容量约束：节点 m 的 CPU 总需求不能超过 cap_m * U_node_max =====================
    _add_node_capacity_constraints(model, tasks, y, cpu_capacity, u_node_max)

    proxy_link_util_exprs = []
    if proxy_link_load is not None:
        # ===================== 4. 代理链路容量约束：用平均分流的线性负载近似节点层对链路的影响 =====================
        # 该约束不等价于真实链路层路由，只用于在节点层提前过滤明显拥塞的放置方案。
        if link_capacity is None:
            raise ValueError("link_capacity is required when proxy_link_load is provided.")
        if u_link_proxy is None:
            raise RuntimeError("U_link_proxy was not initialized.")
        proxy_link_util_exprs = _add_proxy_link_constraints(
            model,
            tasks,
            y,
            proxy_link_load,
            link_capacity,
            u_link_proxy,
        )

    # ===================== 5. 节点层损失约束：按节点故障或代理链路故障损失计算每个场景的任务损失 =====================
    # proxy_loss=None 时只看执行节点是否故障；否则使用 layered2 预先计算的平均分流代理损失。
    _add_node_loss_constraints(
        model,
        tasks,
        y,
        nscenarios=nscenarios,
        eta_node=eta_node,
        node_col=node_col,
        proxy_loss=proxy_loss,
        loss_aggregation=loss_aggregation,
        loss_task=loss_task,
        loss_sys=loss_sys,
        u_aux=u_aux,
        alpha=alpha,
    )

    cvar_expr = alpha + (1.0 / (1.0 - beta)) * gp.quicksum(
        float(node_scenario_probs[s - 1]) * u_aux[s] for s in range(1, nscenarios + 1)
    )
    proxy_link_weight = float(proxy_link_weight) if proxy_link_load is not None else 0.0
    if proxy_link_weight < 0:
        raise ValueError("proxy_link_weight must be non-negative.")
    if risk_mode == "cvar_constraint":
        if proxy_link_weight > 1.0 + 1e-9:
            raise ValueError("proxy_link_weight must be <= 1 when node risk_mode is 'cvar_constraint'.")
        node_load_weight = 1.0 - proxy_link_weight
    else:
        node_load_weight = 1.0 - float(risk_weight) - proxy_link_weight
        if node_load_weight < -1e-9:
            raise ValueError("risk_weight + proxy_link_weight must be <= 1.")
    node_load_weight = max(0.0, node_load_weight)
    proxy_link_term = proxy_link_weight * u_link_proxy if u_link_proxy is not None else 0.0
    network_affinity_expr = 0.0
    if network_affinity is not None:
        network_affinity_expr = (
            gp.quicksum(
                float(network_affinity.get((int(task["task_idx"]), int(m)), 1.0))
                * y[(int(task["task_idx"]), int(m))]
                for task in tasks
                for m in task["candidate_ms"]
            )
            / float(len(tasks))
        )
    blended_obj = (
        float(risk_weight) * cvar_expr
        + node_load_weight * u_node_max
        + proxy_link_term
    )
    # ===================== 6. 节点层目标函数：节点层 CVaR、节点最大利用率、代理链路项加权求和 =====================
    resource_obj = node_load_weight * u_node_max + proxy_link_term
    if risk_mode == "cvar_constraint":
        model.addConstr(cvar_expr <= float(cvar_bound) + 1e-9, name="node_cvar_sla_bound")
        model.setObjective(resource_obj, GRB.MINIMIZE)
    else:
        model.setObjective(blended_obj, GRB.MINIMIZE)
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        raise RuntimeError(f"Layered node model found no feasible solution: gurobi_status_{model.Status}")

    def var_value(var):
        return float(var.X)

    def build_solution(sol_idx: int, source: str):
        assignment = _extract_assignment(tasks, y, var_value)

        _, _, u_node_recomputed = compute_node_usage(tasks, assignment, cpu_capacity)
        u_node_final = max(var_value(u_node_max), u_node_recomputed)
        alpha_value = var_value(alpha)
        cvar_value = alpha_value + (1.0 / (1.0 - beta)) * sum(
            float(node_scenario_probs[s - 1]) * var_value(u_aux[s])
            for s in range(1, nscenarios + 1)
        )
        u_link_proxy_value = var_value(u_link_proxy) if u_link_proxy is not None else None
        network_affinity_value = _assignment_network_affinity(tasks, assignment, network_affinity)

        node_obj = (
            (0.0 if risk_mode == "cvar_constraint" else float(risk_weight)) * cvar_value
            + float(node_load_weight) * u_node_final
            + float(proxy_link_weight) * (u_link_proxy_value or 0.0)
        )
        return {
            "status": "optimal" if model.Status == GRB.OPTIMAL else "suboptimal",
            "assignment": assignment,
            "assignment_repr": _assignment_repr(tasks, assignment),
            "node_obj": float(node_obj),
            "node_cvar": float(cvar_value),
            "node_alpha": float(alpha_value),
            "u_node_max": float(u_node_final),
            "solve_time_sec": time.time() - t0,
            "loss_aggregation": loss_aggregation,
            "risk_mode": risk_mode,
            "risk_weight": float(risk_weight),
            "cvar_bound": cvar_bound,
            "cvar_bound_slack": (float(cvar_bound) - cvar_value) if cvar_bound is not None else None,
            "node_load_weight": float(node_load_weight),
            "proxy_link_weight": float(proxy_link_weight),
            "u_link_proxy": u_link_proxy_value,
            "network_affinity": network_affinity_value,
            "node_loss_model": node_loss_model,
            "candidate_index": int(sol_idx),
            "candidate_source": source,
        }

    candidate_solutions = []
    seen_assignments = set()

    def remember_solution(solution):
        assignment_repr = solution["assignment_repr"]
        if assignment_repr in seen_assignments:
            return False
        seen_assignments.add(assignment_repr)
        candidate_solutions.append(solution)
        return True

    if network_affinity is not None:
        # ===================== 7. layered3 候选生成：固定节点利用率和节点层 CVaR 质量，再枚举低网络亲和度邻域 =====================
        # 节点层只负责生成少量候选放置；真实流量分配、链路损失和链路利用率统一交给链路层评估。
        model.setObjective(u_node_max, GRB.MINIMIZE)
        model.optimize()
        if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
            raise RuntimeError(f"Layered node min-load model failed: gurobi_status_{model.Status}")
        u_node_best = float(u_node_max.X)
        model.addConstr(u_node_max <= min(1.0, u_node_best + node_u_tol) + 1e-9, name="affinity_u_node_quality")

        model.setObjective(cvar_expr, GRB.MINIMIZE)
        model.optimize()
        if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
            raise RuntimeError(f"Layered node min-risk model failed: gurobi_status_{model.Status}")
        cvar_best = float(cvar_expr.getValue())
        model.addConstr(cvar_expr <= cvar_best + node_cvar_tol + 1e-9, name="affinity_cvar_quality")

        model.setObjective(network_affinity_expr, GRB.MINIMIZE)
        model.optimize()
        if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
            raise RuntimeError(f"Layered node min-affinity model failed: gurobi_status_{model.Status}")
        affinity_best = float(network_affinity_expr.getValue())
        if affinity_tol is not None:
            model.addConstr(
                network_affinity_expr <= affinity_best + affinity_tol + 1e-9,
                name="affinity_neighborhood_quality",
            )

        no_good_count = 0
        target_count = max(1, candidate_budget)
        while len(candidate_solutions) < target_count and no_good_count < target_count * 3:
            model.setObjective(network_affinity_expr, GRB.MINIMIZE)
            model.optimize()
            if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
                break
            solution = build_solution(no_good_count, f"min_affinity_rank_{no_good_count + 1}")
            remember_solution(solution)
            _add_no_good_constraint(
                model,
                tasks,
                y,
                solution["assignment"],
                name=f"affinity_no_good_{no_good_count}",
            )
            no_good_count += 1
        gurobi_solution_count = len(candidate_solutions)
    else:
        # 单候选模式直接返回节点层加权目标的最优放置，供 layered1/2 使用。
        solution = build_solution(0, "weighted_node_objective")
        if risk_mode == "cvar_constraint" and solution.get("candidate_source") == "weighted_node_objective":
            solution["candidate_source"] = "node_cvar_constraint_resource"
        remember_solution(solution)
        gurobi_solution_count = 1

    if not candidate_solutions:
        fallback_source = "node_cvar_constraint_resource" if risk_mode == "cvar_constraint" else "weighted_node_objective"
        candidate_solutions.append(build_solution(0, fallback_source))

    total_node_solve_time = time.time() - t0
    for solution in candidate_solutions:
        solution["solve_time_sec"] = total_node_solve_time

    primary = dict(candidate_solutions[0])
    primary["candidate_solutions"] = candidate_solutions
    primary["candidate_budget"] = candidate_budget
    primary["node_u_tol"] = node_u_tol
    primary["node_cvar_tol"] = node_cvar_tol
    primary["affinity_tol"] = affinity_tol
    primary["proxy_link_cap_enforced"] = bool(proxy_link_cap_enforced)
    primary["candidate_count"] = len(candidate_solutions)
    primary["gurobi_solution_count"] = gurobi_solution_count

    return primary
