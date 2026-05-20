from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from Algorithms.common import build_tunnel_scenario_matrix
from parsers import (
    build_tasks_from_demand_and_taskfile,
    load_paths_for_all_pairs,
    next_run,
    read_demand,
    read_node_resources,
    read_task_file,
    read_topology,
)
from phase1_correctness import run_phase1_correctness
from util import build_joint_failure_scenarios, weibull_probs
from util import compute_node_usage


def read_input(message, default, typ):
    # ===================== 1. 消融实验输入工具：空输入使用默认值，非空输入按指定类型解析 =====================
    raw = input(message)
    if len(raw.strip()) == 0:
        return default
    if typ is str:
        return raw.strip()
    try:
        return typ(raw)
    except Exception as exc:
        raise ValueError(f"Invalid input for '{message.strip()}': {raw}") from exc


def _to_jsonable(value):
    # 将 numpy 类型转换成 json 可写类型，保证 details.json 能保存完整中间结果。
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cvar(losses: Sequence[float], probs: Sequence[float], beta: float):
    # 对固定损失序列直接枚举 alpha，计算经验 CVaR；这里不依赖 Gurobi。
    candidates = sorted({0.0, *[float(loss) for loss in losses]})
    best_alpha = 0.0
    best_value = float("inf")
    tail_scale = 1.0 / max(1e-12, 1.0 - float(beta))
    for alpha in candidates:
        value = alpha + tail_scale * sum(
            float(prob) * max(0.0, float(loss) - alpha)
            for loss, prob in zip(losses, probs)
        )
        if value < best_value:
            best_value = value
            best_alpha = alpha
    return best_value, best_alpha


def _x_value(x_map: Mapping[Tuple[int, ...], float], tid: int, m: int, tunnel_id: int) -> float:
    # 单层模型输出可能带候选节点维度，链路层输出通常不带；该函数统一两种 x 的读取方式。
    return float(x_map.get((tid, m, tunnel_id), x_map.get((tid, tunnel_id), 0.0)))


def evaluate_fixed_routing_failure(
    *,
    tasks,
    assignment: Dict[int, int],
    x_in: Mapping[Tuple[int, ...], float],
    x_out: Mapping[Tuple[int, ...], float],
    edges,
    capacity,
    T,
    scenarios,
    scenario_probs,
    node_scenarios,
    pair_to_tunnels,
    cpu_capacity,
    beta: float,
    loss_aggregation: str,
):
    """在固定放置和固定路径流量下，重新评估考虑故障后的实际 CVaR。"""
    # ===================== 2. 固定解故障评估：不重新优化，只检查给定 y/x 在故障场景下的真实损失 =====================
    # 该函数用于比较“不考虑故障优化得到的解”在加入故障后会产生多大尾部损失。
    x_path = build_tunnel_scenario_matrix(T, edges, scenarios)
    compute_nodes = sorted(int(m) for m in cpu_capacity.keys())
    node_col = {int(m): idx for idx, m in enumerate(compute_nodes)}
    eta_node = np.ones((len(scenarios), len(compute_nodes)), dtype=float)
    for s_idx, state in enumerate(node_scenarios):
        for m in compute_nodes:
            eta_node[s_idx, node_col[m]] = float(state.get(m, 1.0))

    scenario_losses = []
    task_losses = {}
    for s_idx in range(1, len(scenarios) + 1):
        per_task_losses = []
        for task in tasks:
            tid = int(task["task_idx"])
            src = int(task["src"])
            dst = int(task["dst"])
            m = int(assignment[tid])
            eta_m = float(eta_node[s_idx - 1, node_col[m]]) if m in node_col else 1.0
            b_in = float(task["b_in"])
            b_out = float(task["b_out"])

            if src == m:
                delivered_in = eta_m * b_in
            else:
                delivered_in = sum(
                    _x_value(x_in, tid, m, int(tunnel_id)) * eta_m * x_path[s_idx - 1, int(tunnel_id) - 1]
                    for tunnel_id in pair_to_tunnels.get((src, m), [])
                )
            if dst == m:
                delivered_out = eta_m * b_out
            else:
                delivered_out = sum(
                    _x_value(x_out, tid, m, int(tunnel_id)) * eta_m * x_path[s_idx - 1, int(tunnel_id) - 1]
                    for tunnel_id in pair_to_tunnels.get((m, dst), [])
                )

            loss = 0.0
            if b_in > 0:
                loss = max(loss, 1.0 - delivered_in / b_in)
            if b_out > 0:
                loss = max(loss, 1.0 - delivered_out / b_out)
            loss = min(1.0, max(0.0, float(loss)))
            task_losses[(s_idx, tid)] = loss
            per_task_losses.append(loss)

        if str(loss_aggregation).lower() == "average":
            scenario_loss = sum(per_task_losses) / max(1, len(per_task_losses))
        else:
            scenario_loss = max(per_task_losses, default=0.0)
        scenario_losses.append(float(scenario_loss))

    cvar_value, alpha = _cvar(scenario_losses, scenario_probs, beta)

    link_loads = {edge_idx + 1: 0.0 for edge_idx in range(len(edges))}
    for key, value in x_in.items():
        tunnel_id = int(key[-1])
        for edge_id in T[tunnel_id - 1]:
            link_loads[int(edge_id)] += float(value)
    for key, value in x_out.items():
        tunnel_id = int(key[-1])
        for edge_id in T[tunnel_id - 1]:
            link_loads[int(edge_id)] += float(value)
    u_link_max = 0.0
    for edge_id, load in link_loads.items():
        cap = float(capacity[int(edge_id) - 1])
        if cap > 0:
            u_link_max = max(u_link_max, float(load) / cap)
    _, _, u_node_max = compute_node_usage(tasks, assignment, cpu_capacity)

    return {
        "fixed_x_eval_cvar": float(cvar_value),
        "fixed_x_eval_alpha": float(alpha),
        "fixed_x_eval_u_link_max": float(u_link_max),
        "fixed_x_eval_u_node_max": float(u_node_max),
        "fixed_x_eval_loss_aggregation": str(loss_aggregation).lower(),
        "scenario_losses": {idx + 1: loss for idx, loss in enumerate(scenario_losses)},
        "task_losses": {f"{s}_{tid}": loss for (s, tid), loss in task_losses.items()},
    }


def _load_problem(args: Namespace):
    # ===================== 3. 读取公共问题数据：四组消融实验共享同一拓扑、任务、路径和故障场景 =====================
    # 这样不同算法/风险设置之间的结果可直接比较，避免输入数据变化影响结论。
    edges, capacity, _, nodes = read_topology(args.topology)
    node_rows, compute_nodes, cpu_capacity, node_failure_probs = read_node_resources(args.topology)
    placement_policy = args.placement_policy
    allow_same_src_dst = bool(args.allow_same_src_dst) and placement_policy == "exclude_endpoints"
    demand, flows = read_demand(
        f"{args.topology}/demand",
        len(nodes),
        args.num_demand,
        scale=args.demand_scale,
        downscale=args.demand_downscale,
        include_cycles=allow_same_src_dst,
    )
    task_cpu_map = read_task_file(f"data/{args.topology}/task.txt")
    tasks = build_tasks_from_demand_and_taskfile(
        demand=demand,
        flows=flows,
        task_cpu_map=task_cpu_map,
        rho=args.rho,
        candidate_ms=compute_nodes,
        max_tasks=args.max_tasks,
        placement_policy=placement_policy,
    )
    T, _, k_used, _, pair_to_tunnels = load_paths_for_all_pairs(
        topology=args.topology,
        path_mode=args.paths,
        links=edges,
        nodes=nodes,
        k_paths=args.k_paths,
        edge_disjoint=args.edge_disjoint,
    )
    link_probs = weibull_probs(len(edges), shape=0.8, scale=args.weibull_scale)
    scenarios, node_scenarios, scenario_probs = build_joint_failure_scenarios(
        link_probs=link_probs,
        compute_nodes=compute_nodes,
        node_failure_probs=node_failure_probs,
        cutoff=args.cutoff,
        include_node_failures=True,
    )
    return {
        "edges": edges,
        "capacity": capacity,
        "tasks": tasks,
        "T": T,
        "k_used": k_used,
        "pair_to_tunnels": pair_to_tunnels,
        "cpu_capacity": cpu_capacity,
        "scenarios": scenarios,
        "node_scenarios": node_scenarios,
        "scenario_probs": scenario_probs,
    }


def _base_args(**overrides):
    # ===================== 4. 默认参数模板：集中维护消融实验所需的完整 Namespace 字段 =====================
    # 后续每个 case 只覆盖算法和风险相关参数，避免遗漏 phase1_correctness 所需字段。
    data = {
        "topology": "B4",
        "num_demand": 1,
        "demand_scale": 1.0,
        "demand_downscale": 1.0,
        "max_tasks": 144,
        "placement_policy": "exclude_endpoints",
        "allow_same_src_dst": False,
        "rho": 0.5,
        "paths": "KSP",
        "k_paths": 6,
        "edge_disjoint": False,
        "beta": 0.99,
        "lambda_weight": 0.5,
        "cutoff": 0.0001,
        "weibull_scale": 0.001,
        "seed": 1,
        "algorithm": "A",
        "risk_weight": 0.5,
        "loss_aggregation": "max",
        "node_risk_weight": 0.5,
        "link_risk_weight": 0.5,
        "node_loss_aggregation": "average",
        "link_loss_aggregation": "max",
        "link_optimization_mode": "weighted",
        "link_cvar_tolerance": 0.0,
        "node_proxy_link_weight": 0.0,
        "layered3_candidate_budget": 8,
        "layered3_node_u_tol": 0.0,
        "layered3_node_cvar_tol": 1e-4,
        "layered3_affinity_tol": None,
        "enable_mip_gap": False,
        "mip_gap": 0.0,
    }
    data.update(overrides)
    return Namespace(**data)


def _with_common(common: Namespace, **overrides):
    # 从交互输入中继承公共参数，再覆盖某个 case 的算法配置。
    data = vars(_base_args()).copy()
    for key in [
        "topology",
        "num_demand",
        "demand_scale",
        "demand_downscale",
        "max_tasks",
        "placement_policy",
        "allow_same_src_dst",
        "rho",
        "paths",
        "k_paths",
        "edge_disjoint",
        "beta",
        "lambda_weight",
        "cutoff",
        "weibull_scale",
        "seed",
        "enable_mip_gap",
        "mip_gap",
        "layered3_candidate_budget",
    ]:
        data[key] = getattr(common, key)
    data.update(overrides)
    return Namespace(**data)


def run_phase1_fault_ablation(common: Namespace):
    # ===================== 5. 消融实验主流程：分别运行无故障目标和故障感知目标，再统一做固定解故障评估 =====================
    # 输出 summary.csv 便于横向对比节点利用率、链路利用率、优化 CVaR 和固定解实际 CVaR。
    out_dir = Path(next_run("data/raw/phase1_fault_ablation"))
    problem = _load_problem(common)

    cases = [
        # A_no_failure：单层模型把故障损失权重置为 0，只优化正常状态利用率。
        (
            "A_no_failure",
            _with_common(
                common,
                algorithm="A",
                risk_weight=0.0,
                loss_aggregation="max",
            ),
            False,
        ),
        # A_failure_aware：单层模型正常考虑故障 CVaR，用作故障感知基准。
        (
            "A_failure_aware",
            _with_common(
                common,
                algorithm="A",
                risk_weight=0.5,
                loss_aggregation="max",
            ),
            True,
        ),
        # layered3_no_failure：分层模型关闭节点层和链路层风险权重，观察正常状态利用率基线。
        (
            "layered3_no_failure",
            _with_common(
                common,
                algorithm="layered3",
                node_risk_weight=0.0,
                link_risk_weight=0.0,
                node_loss_aggregation="average",
                link_loss_aggregation="max",
                link_optimization_mode="weighted",
                layered3_node_u_tol=0.0,
                layered3_node_cvar_tol=1.0,
            ),
            False,
        ),
        # layered3_failure_aware：分层模型保留故障风险项，用于和无故障版本及单层模型比较。
        (
            "layered3_failure_aware",
            _with_common(
                common,
                algorithm="layered3",
                node_risk_weight=0.5,
                link_risk_weight=0.5,
                node_loss_aggregation="average",
                link_loss_aggregation="max",
                link_optimization_mode="weighted",
                layered3_node_u_tol=0.0,
                layered3_node_cvar_tol=1e-4,
            ),
            True,
        ),
    ]

    summary_rows = []
    details = {}
    for case_name, args, failure_aware in cases:
        # 每个 case 先调用 phase1_correctness 得到最优放置和路由，再用同一批故障场景复算固定解损失。
        print(f"\n========== Running ablation case: {case_name} ==========")
        case_output_dir, result = run_phase1_correctness(args)
        lower = result["best_lower"]
        fixed_eval = evaluate_fixed_routing_failure(
            tasks=problem["tasks"],
            assignment=result["best_assignment"],
            x_in=lower.get("x_in", {}),
            x_out=lower.get("x_out", {}),
            edges=problem["edges"],
            capacity=problem["capacity"],
            T=problem["T"],
            scenarios=problem["scenarios"],
            scenario_probs=problem["scenario_probs"],
            node_scenarios=problem["node_scenarios"],
            pair_to_tunnels=problem["pair_to_tunnels"],
            cpu_capacity=problem["cpu_capacity"],
            beta=args.beta,
            loss_aggregation="max",
        )

        row = {
            "case_name": case_name,
            "algorithm": args.algorithm,
            "failure_aware": bool(failure_aware),
            "risk_weight": getattr(args, "risk_weight", ""),
            "node_risk_weight": getattr(args, "node_risk_weight", ""),
            "link_risk_weight": getattr(args, "link_risk_weight", ""),
            "candidate_budget": getattr(args, "layered3_candidate_budget", ""),
            "u_node_max": result["u_node_max"],
            "u_link_max": result["u_link_max"],
            "optimized_cvar": lower.get("cvar"),
            "fixed_x_eval_cvar": fixed_eval["fixed_x_eval_cvar"],
            "fixed_x_eval_alpha": fixed_eval["fixed_x_eval_alpha"],
            "fixed_x_eval_u_node_max": fixed_eval["fixed_x_eval_u_node_max"],
            "fixed_x_eval_u_link_max": fixed_eval["fixed_x_eval_u_link_max"],
            "solve_time_sec": lower.get("solve_time_sec"),
            "phase1_output_dir": str(Path(case_output_dir).resolve()),
        }
        summary_rows.append(row)
        details[case_name] = {
            "args": vars(args),
            "phase1_output_dir": str(Path(case_output_dir).resolve()),
            "summary": row,
            "fixed_eval": fixed_eval,
        }

    summary_path = out_dir / "summary.csv"
    # ===================== 6. 写出结果：summary.csv 存核心指标，details.json 存每个 case 的参数和固定解损失明细 =====================
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (out_dir / "details.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(details), f, ensure_ascii=False, indent=2)

    print("\n========== Fault Ablation Summary ==========")
    print(f"Output dir : {out_dir.resolve()}")
    for row in summary_rows:
        print(
            f"{row['case_name']}: "
            f"U_node={float(row['u_node_max']):.6f}, "
            f"U_link={float(row['u_link_max']):.6f}, "
            f"opt_CVaR={float(row['optimized_cvar']):.6f}, "
            f"fixed_CVaR={float(row['fixed_x_eval_cvar']):.6f}"
        )
    print("==========================================\n")
    return str(out_dir), summary_rows


def main():
    # ===================== 7. 交互入口：只读取消融实验真正需要调节的公共参数 =====================
    print("Asking for fault ablation inputs. Press enter for default values\n")
    common = _base_args(
        topology=read_input("Topology (B4): ", "B4", str),
        num_demand=read_input("Num demand row (1): ", 1, int),
        demand_scale=read_input("Demand scale (1.0): ", 1.0, float),
        demand_downscale=read_input("Demand downscale (1.0): ", 1.0, float),
        max_tasks=read_input("Max tasks (144): ", 144, int),
        placement_policy=read_input("Placement policy all/exclude_endpoints (exclude_endpoints): ", "exclude_endpoints", str),
        allow_same_src_dst=read_input("Allow same source and destination tasks? (False): ", False, bool),
        rho=read_input("Out traffic ratio rho (0.5): ", 0.5, float),
        paths=read_input("Paths KSP/EDGE_DISJOINT (KSP): ", "KSP", str),
        k_paths=read_input("K paths (6): ", 6, int),
        edge_disjoint=False,
        beta=read_input("CVaR beta (0.99): ", 0.99, float),
        lambda_weight=read_input("Single-level lambda weight (0.5): ", 0.5, float),
        cutoff=read_input("Scenario probability cutoff (0.0001): ", 0.0001, float),
        weibull_scale=read_input("Weibull scale (0.001): ", 0.001, float),
        seed=read_input("Random seed (1): ", 1, int),
        enable_mip_gap=True,
        mip_gap=read_input("A_solver MIP gap (0.07): ", 0.07, float),
        layered3_candidate_budget=read_input("Layered3 candidate budget (8): ", 8, int),
    )
    run_phase1_fault_ablation(common)


if __name__ == "__main__":
    main()
