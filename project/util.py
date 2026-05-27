from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def get_probabilities(scenarios: Sequence[Sequence[float]], probabilities: Sequence[float]) -> List[float]:
    p: List[float] = []
    for s in scenarios:
        prob = 1.0
        for i in range(len(s)):
            prob *= (1 - s[i]) * probabilities[i] + s[i] * (1 - probabilities[i])
        p.append(prob)
    return p


def k_scenarios(nedges: int, k: int, probabilities: Sequence[float], first: bool = True):
    scenarios: List[np.ndarray] = []
    if first:
        scenarios.append(np.ones(nedges, dtype=float))
    for i in range(1, k + 1):
        for bits in itertools.combinations(range(1, nedges + 1), i):
            s = np.ones(nedges, dtype=float)
            for bit in bits:
                s[bit - 1] = 0
            scenarios.append(s)
    probs = get_probabilities(scenarios, probabilities)
    probs_arr = np.array(probs, dtype=float)
    probs_arr = probs_arr / np.sum(probs_arr)
    return scenarios, probs_arr.tolist()


def all_scenarios(nedges: int, probabilities: Sequence[float], first: bool = True):
    scenarios: List[np.ndarray] = []
    probs: List[float] = []
    if first:
        scenario = np.ones(nedges, dtype=float)
        scenarios.append(scenario)
        prob = 1.0
        for i in range(len(scenario)):
            prob *= (1 - scenario[i]) * probabilities[i] + scenario[i] * (1 - probabilities[i])
        probs.append(prob)
    for i in range(1, nedges + 1):
        for bits in itertools.combinations(range(1, nedges + 1), i):
            s = np.ones(nedges, dtype=float)
            for bit in bits:
                s[bit - 1] = 0
            prob = 1.0
            for j in range(len(s)):
                prob *= (1 - s[j]) * probabilities[j] + s[j] * (1 - probabilities[j])
            probs.append(prob)
            scenarios.append(s)
    probs_arr = np.array(probs, dtype=float)
    probs_arr = probs_arr / np.sum(probs_arr)
    return scenarios, probs_arr.tolist()


def sub_scenarios_recursion(
    original: Sequence[float],
    cutoff: float,
    remaining: Sequence[float] | None = None,
    partial: List[int] | None = None,
    scenarios: List[np.ndarray] | None = None,
    probabilities: List[float] | None = None,
):
    if partial is None:
        partial = []
    if scenarios is None:
        scenarios = []
    if probabilities is None:
        probabilities = []
    if remaining is None:
        remaining = list(original)

    original_arr = np.array(original, dtype=float)
    if len(partial) == 0:
        scenarios.append(np.ones(len(original_arr), dtype=float))
        probabilities.append(float(np.prod(1 - original_arr)))
        remaining = original_arr
    else:
        probs = 1 - original_arr
        bitmap = np.ones(len(original_arr), dtype=float)
        for index in partial:
            probs[index - 1] = original_arr[index - 1]
            bitmap[index - 1] = 0
        product = float(np.prod(probs))
        if product >= cutoff:
            scenarios.append(bitmap)
            probabilities.append(product)
        else:
            return scenarios, probabilities

    rem = list(remaining)
    for i in range(len(rem)):
        n = len(original_arr) - len(rem) + i + 1
        sub_scenarios_recursion(
            original_arr,
            cutoff,
            rem[i + 1 :],
            partial + [n],
            scenarios,
            probabilities,
        )
    return scenarios, probabilities


def sub_scenarios(original: Sequence[float], cutoff: float, first: bool = True, last: bool = True):
    scenarios, probabilities = sub_scenarios_recursion(original, cutoff)
    if not first:
        scenarios = scenarios[1:]
        probabilities = probabilities[1:]
    if last and len(scenarios) > 0:
        scenarios.append(np.zeros(len(scenarios[0]), dtype=float))
        probabilities.append(1 - sum(probabilities))
    p = np.array(probabilities, dtype=float)
    if p.sum() <= 0:
        raise ValueError("Scenario probabilities sum to 0 after cutoff")
    p = p / p.sum()
    return scenarios, p.tolist()


def weibull_probs(num: int, shape: float = 0.8, scale: float = 0.0001):
    draws = scale * np.random.weibull(shape, size=num)
    return draws.tolist()


def split_joint_failure_scenarios(
    joint_scenarios: Sequence[Sequence[float]],
    num_links: int,
    compute_nodes: Sequence[int],
):
    link_scenarios: List[np.ndarray] = []
    node_scenarios: List[Dict[int, float]] = []
    for scenario in joint_scenarios:
        link_scenarios.append(np.array(scenario[:num_links], dtype=float))
        node_state: Dict[int, float] = {}
        for offset, node_id in enumerate(compute_nodes):
            idx = num_links + offset
            node_state[int(node_id)] = float(scenario[idx]) if idx < len(scenario) else 1.0
        node_scenarios.append(node_state)
    return link_scenarios, node_scenarios


def build_joint_failure_scenarios(
    link_probs: Sequence[float],
    compute_nodes: Sequence[int],
    node_failure_probs: Dict[int, float],
    cutoff: float,
    include_node_failures: bool = False,
):
    """Build cutoff scenarios over links and optional compute-node failures.

    State value 1 means available; 0 means failed. Probabilities are treated as
    independent failure probabilities, matching the existing link scenario code.
    """
    node_probs = []
    if include_node_failures:
        node_probs = [float(node_failure_probs.get(int(m), 0.0)) for m in compute_nodes]

    event_probs = list(float(p) for p in link_probs) + node_probs
    joint_scenarios, scenario_probs = sub_scenarios(event_probs, cutoff, first=True, last=False)
    link_scenarios, node_scenarios = split_joint_failure_scenarios(
        joint_scenarios,
        len(link_probs),
        compute_nodes if include_node_failures else [],
    )

    if not include_node_failures:
        node_scenarios = [{int(m): 1.0 for m in compute_nodes} for _ in link_scenarios]

    return link_scenarios, node_scenarios, scenario_probs


def compute_node_usage(tasks, assignment: Dict[int, int], cpu_capacity: Dict[int, float]):
    used: Dict[int, float] = {m: 0.0 for m in cpu_capacity}
    for task in tasks:
        tid = int(task["task_idx"])
        m = int(assignment[tid])
        used[m] = used.get(m, 0.0) + float(task["cpu_demand"])
    util: Dict[int, float] = {}
    max_util = 0.0
    for m, cap in cpu_capacity.items():
        if cap <= 0:
            util[m] = 0.0
        else:
            util[m] = used.get(m, 0.0) / cap
        max_util = max(max_util, util[m])
    return used, util, max_util


def _path_to_strings(tunnel_edges: Sequence[int], edges: Sequence[Tuple[int, int]]):
    if len(tunnel_edges) == 0:
        return "", ""
    edge_pairs = [edges[eid - 1] for eid in tunnel_edges]
    edge_str = "|".join(f"({u},{v})" for u, v in edge_pairs)
    nodes = [edge_pairs[0][0]] + [v for _, v in edge_pairs]
    node_str = "->".join(str(n) for n in nodes)
    return node_str, edge_str


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def _assignment_node(assignment: Dict[int, int], task_id: int) -> int:
    if task_id in assignment:
        return int(assignment[task_id])
    return int(assignment[str(task_id)])  # type: ignore[index]


def _task_path_weights(x_map: Dict[Tuple[int, int], float], task_id: int) -> Dict[int, float]:
    weights: Dict[int, float] = {}
    for key, value in x_map.items():
        if len(key) < 2 or int(key[0]) != int(task_id):
            continue
        tunnel_id = int(key[-1])
        amount = float(value)
        if amount > 1e-12:
            weights[tunnel_id] = weights.get(tunnel_id, 0.0) + amount
    return weights


def _is_tunnel_alive(tunnel_edges: Sequence[int], scenario: Sequence[float]) -> bool:
    if len(tunnel_edges) == 0:
        return False
    for edge_id in tunnel_edges:
        if not (1 <= int(edge_id) <= len(scenario)):
            return False
        if float(scenario[int(edge_id) - 1]) <= 0.5:
            return False
    return True


def _add_routed_load(link_loads: List[float], T: Sequence[Sequence[int]], tunnel_id: int, routed: float):
    if routed <= 0:
        return
    if not (0 < int(tunnel_id) <= len(T)):
        return
    for edge_id in T[int(tunnel_id) - 1]:
        if 1 <= int(edge_id) <= len(link_loads):
            link_loads[int(edge_id) - 1] += float(routed)


def _max_link_utilization(link_loads: Sequence[float], capacity: Sequence[float]) -> float:
    if not link_loads:
        return 0.0
    return max(float(load) / max(float(capacity[idx]), 1e-12) for idx, load in enumerate(link_loads))


def compute_teavar_style_evaluation_metrics(
    tasks: Sequence[Dict[str, object]],
    assignment: Dict[int, int],
    lower: Dict[str, object],
    edges: Sequence[Tuple[int, int]],
    capacity: Sequence[float],
    T: Sequence[Sequence[int]],
    scenarios: Sequence[Sequence[float]],
    scenario_probs: Sequence[float],
    node_scenarios: Sequence[Dict[int, float]] | None,
    availability_target: float = 0.0,
):
    """基于求解出的放置和 x，计算正常态真实负载与故障重分配评价指标。"""
    x_in: Dict[Tuple[int, int], float] = lower.get("x_in", {})  # type: ignore[assignment]
    x_out: Dict[Tuple[int, int], float] = lower.get("x_out", {})  # type: ignore[assignment]
    normal_loads = [0.0 for _ in edges]

    for task in tasks:
        tid = int(task["task_idx"])
        m = _assignment_node(assignment, tid)
        for direction, demand, weights in (
            ("in", float(task["b_in"]), _task_path_weights(x_in, tid)),
            ("out", float(task["b_out"]), _task_path_weights(x_out, tid)),
        ):
            if demand <= 0:
                continue
            endpoint_local = (
                direction == "in" and int(task["src"]) == m
            ) or (
                direction == "out" and int(task["dst"]) == m
            )
            if endpoint_local:
                continue
            total_weight = sum(weights.values())
            if total_weight <= 1e-12:
                continue
            for tunnel_id, weight in weights.items():
                _add_routed_load(normal_loads, T, tunnel_id, demand * float(weight) / float(total_weight))

    actual_normal_u_link_max = _max_link_utilization(normal_loads, capacity)
    total_required = max(
        sum(float(task["b_in"]) + float(task["b_out"]) for task in tasks),
        1e-12,
    )
    scenario_losses: Dict[int, float] = {}
    scenario_fail_u_link_max: Dict[int, float] = {}

    for s_idx, scenario in enumerate(scenarios, start=1):
        link_loads = [0.0 for _ in edges]
        delivered_before_overflow = 0.0
        node_state = node_scenarios[s_idx - 1] if node_scenarios is not None and s_idx <= len(node_scenarios) else {}

        for task in tasks:
            tid = int(task["task_idx"])
            m = _assignment_node(assignment, tid)
            if float(node_state.get(m, 1.0)) <= 0.5:
                continue

            for direction, demand, weights in (
                ("in", float(task["b_in"]), _task_path_weights(x_in, tid)),
                ("out", float(task["b_out"]), _task_path_weights(x_out, tid)),
            ):
                if demand <= 0:
                    continue
                endpoint_local = (
                    direction == "in" and int(task["src"]) == m
                ) or (
                    direction == "out" and int(task["dst"]) == m
                )
                if endpoint_local:
                    delivered_before_overflow += demand
                    continue

                alive = {
                    tunnel_id: weight
                    for tunnel_id, weight in weights.items()
                    if 0 < int(tunnel_id) <= len(T) and _is_tunnel_alive(T[int(tunnel_id) - 1], scenario)
                }
                total_alive_weight = sum(alive.values())
                if total_alive_weight <= 1e-12:
                    continue
                delivered_before_overflow += demand
                for tunnel_id, weight in alive.items():
                    _add_routed_load(link_loads, T, tunnel_id, demand * float(weight) / float(total_alive_weight))

        overflow = 0.0
        for edge_idx, load in enumerate(link_loads):
            overflow += max(0.0, float(load) - float(capacity[edge_idx]))

        delivered_after_overflow = max(0.0, delivered_before_overflow - overflow)
        scenario_losses[s_idx] = min(1.0, max(0.0, 1.0 - delivered_after_overflow / total_required))
        scenario_fail_u_link_max[s_idx] = _max_link_utilization(link_loads, capacity)

    availability = sum(
        float(scenario_probs[s_idx - 1])
        for s_idx, loss in scenario_losses.items()
        if float(loss) <= float(availability_target) + 1e-12
    )
    expected_fail_u_link_max = sum(
        float(scenario_probs[s_idx - 1]) * value
        for s_idx, value in scenario_fail_u_link_max.items()
    )
    worst_fail_u_link_max = max(scenario_fail_u_link_max.values(), default=0.0)
    return {
        "actual_normal_u_link_max": float(actual_normal_u_link_max),
        "availability": float(availability),
        "availability_target": float(availability_target),
        "expected_fail_u_link_max": float(expected_fail_u_link_max),
        "worst_fail_u_link_max": float(worst_fail_u_link_max),
    }


def _append_experiment_summary(output_dir: Path, config: Dict[str, object], best_result: Dict[str, object]):
    summary_path = output_dir.parent.parent / "experiment_summary.csv"
    lower = best_result["best_lower"]
    evaluation_metrics = best_result.get("evaluation_metrics", {})
    summary_loss_aggregation = config.get("loss_aggregation")
    if summary_loss_aggregation is None:
        node_agg = config.get("node_loss_aggregation")
        link_agg = config.get("link_loss_aggregation")
        if node_agg is not None and link_agg is not None:
            node_model = config.get("node_loss_model")
            if node_model is not None:
                summary_loss_aggregation = f"node:{node_agg}/{node_model}|link:{link_agg}"
            else:
                summary_loss_aggregation = f"node:{node_agg}|link:{link_agg}"
        else:
            summary_loss_aggregation = lower.get("loss_aggregation")
    header = [
        "run_id",
        "algorithm",
        "num_tasks",
        "demand_scale",
        "demand_downscale",
        "loss_aggregation",
        "single_level_risk_mode",
        "single_level_cvar_bound",
        "cvar_bound_slack",
        "link_optimization_mode",
        "link_cvar_bound",
        "link_cvar_bound_slack",
        "layered2_node_risk_mode",
        "layered2_node_cvar_bound",
        "layered2_node_cvar_bound_slack",
        "solve_time_sec",
        "best_score",
        "u_node_max",
        "reserved_u_link_max",
        "actual_normal_u_link_max",
        "availability",
        "expected_fail_u_link_max",
        "worst_fail_u_link_max",
        "model_cvar",
        "model_alpha",
        "mip_gap",
    ]
    row = [
        output_dir.name,
        config.get("algorithm"),
        config.get("num_tasks"),
        config.get("demand_scale"),
        config.get("demand_downscale"),
        summary_loss_aggregation,
        config.get("single_level_risk_mode"),
        config.get("single_level_cvar_bound"),
        lower.get("cvar_bound_slack"),
        lower.get("link_optimization_mode", config.get("link_optimization_mode")),
        lower.get("link_cvar_bound", config.get("link_cvar_bound")),
        lower.get("link_cvar_bound_slack"),
        lower.get("layered2_node_risk_mode", config.get("layered2_node_risk_mode")),
        lower.get("layered2_node_cvar_bound", config.get("layered2_node_cvar_bound")),
        lower.get("layered2_node_cvar_bound_slack"),
        lower.get("solve_time_sec"),
        best_result.get("best_score"),
        best_result.get("u_node_max"),
        best_result.get("reserved_u_link_max", best_result.get("u_link_max")),
        evaluation_metrics.get("actual_normal_u_link_max"),
        evaluation_metrics.get("availability"),
        evaluation_metrics.get("expected_fail_u_link_max"),
        evaluation_metrics.get("worst_fail_u_link_max"),
        lower.get("cvar"),
        lower.get("alpha"),
        config.get("mip_gap"),
    ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header is not None and existing_header != header:
            summary_path = output_dir.parent.parent / "experiment_summary_latest.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def write_phase1_results(
    output_dir: str | Path,
    config: Dict[str, object],
    tasks: Sequence[Dict[str, object]],
    best_result: Dict[str, object],
    candidate_rows: Sequence[Dict[str, object]],
    edges: Sequence[Tuple[int, int]],
    capacity: Sequence[float],
    T: Sequence[Sequence[int]],
    scenarios: Sequence[Sequence[float]],
    scenario_probs: Sequence[float],
    node_rows: Sequence[Dict[str, object]],
    sanity_checks: Dict[str, object],
    node_scenarios: Sequence[Dict[int, float]] | None = None,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_json(out / "config.json", config)

    assignment: Dict[int, int] = best_result["best_assignment"]
    lower = best_result["best_lower"]
    reserved_u_link_max = float(best_result["u_link_max"])
    evaluation_metrics = compute_teavar_style_evaluation_metrics(
        tasks=tasks,
        assignment=assignment,
        lower=lower,
        edges=edges,
        capacity=capacity,
        T=T,
        scenarios=scenarios,
        scenario_probs=scenario_probs,
        node_scenarios=node_scenarios,
        availability_target=float(config.get("availability_target", 0.0) or 0.0),
    )
    best_result["reserved_u_link_max"] = reserved_u_link_max
    best_result["evaluation_metrics"] = evaluation_metrics
    global_metrics = {
        "best_score": best_result["best_score"],
        "u_node_max": best_result["u_node_max"],
        "reserved_u_link_max": reserved_u_link_max,
        "actual_normal_u_link_max": evaluation_metrics["actual_normal_u_link_max"],
        "availability": evaluation_metrics["availability"],
        "availability_target": evaluation_metrics["availability_target"],
        "expected_fail_u_link_max": evaluation_metrics["expected_fail_u_link_max"],
        "worst_fail_u_link_max": evaluation_metrics["worst_fail_u_link_max"],
        "model_cvar": lower.get("cvar"),
        "model_alpha": lower.get("alpha"),
        "beta": config.get("beta"),
        "lambda_weight": config.get("lambda_weight"),
        "single_level_risk_mode": lower.get("single_level_risk_mode", config.get("single_level_risk_mode")),
        "single_level_cvar_bound": lower.get("cvar_bound", config.get("single_level_cvar_bound")),
        "cvar_bound_slack": lower.get("cvar_bound_slack"),
        "single_level_resource_obj": lower.get("single_level_resource_obj"),
        "loss_aggregation": lower.get("loss_aggregation", config.get("loss_aggregation", "max")),
        "scenario_count": len(scenarios),
        "solver_status": lower.get("status"),
        "solve_time_sec": lower.get("solve_time_sec"),
        "link_optimization_mode": lower.get("link_optimization_mode", config.get("link_optimization_mode")),
        "link_cvar_tolerance": lower.get("link_cvar_tolerance", config.get("link_cvar_tolerance")),
        "link_cvar_bound": lower.get("link_cvar_bound", config.get("link_cvar_bound")),
        "link_cvar_bound_slack": lower.get("link_cvar_bound_slack"),
        "link_first_stage_cvar": lower.get("link_first_stage_cvar"),
        "node_cvar": lower.get("node_cvar"),
        "node_alpha": lower.get("node_alpha"),
        "node_obj": lower.get("node_obj"),
        "node_solve_time_sec": lower.get("node_solve_time_sec"),
        "node_loss_model": lower.get("node_loss_model", config.get("node_loss_model")),
        "node_proxy_cvar": lower.get("node_proxy_cvar"),
        "node_proxy_alpha": lower.get("node_proxy_alpha"),
        "node_proxy_obj": lower.get("node_proxy_obj"),
        "node_proxy_split_rule": lower.get("node_proxy_split_rule", config.get("node_proxy_split_rule")),
        "node_proxy_k_paths": lower.get("node_proxy_k_paths", config.get("node_proxy_k_paths")),
        "node_proxy_k_paths_source": lower.get(
            "node_proxy_k_paths_source",
            config.get("node_proxy_k_paths_source"),
        ),
        "node_proxy_capacity_constraints": lower.get(
            "node_proxy_capacity_constraints",
            config.get("node_proxy_capacity_constraints"),
        ),
        "node_proxy_link_weight": lower.get("node_proxy_link_weight", config.get("node_proxy_link_weight")),
        "node_proxy_u_link_max": lower.get("node_proxy_u_link_max"),
        "layered2_node_risk_mode": lower.get("layered2_node_risk_mode", config.get("layered2_node_risk_mode")),
        "layered2_node_cvar_bound": lower.get("layered2_node_cvar_bound", config.get("layered2_node_cvar_bound")),
        "layered2_node_cvar_bound_slack": lower.get("layered2_node_cvar_bound_slack"),
        "network_affinity": lower.get("network_affinity"),
        "network_affinity_model": lower.get("network_affinity_model", config.get("network_affinity_model")),
        "layered3_candidate_budget": lower.get(
            "layered3_candidate_budget",
            config.get("layered3_candidate_budget"),
        ),
        "layered3_node_u_tol": lower.get("layered3_node_u_tol", config.get("layered3_node_u_tol")),
        "layered3_node_cvar_tol": lower.get(
            "layered3_node_cvar_tol",
            config.get("layered3_node_cvar_tol"),
        ),
        "layered3_affinity_tol": lower.get("layered3_affinity_tol", config.get("layered3_affinity_tol")),
        "selection_score": lower.get("selection_score", best_result.get("best_score")),
        "node_load_weight": lower.get("node_load_weight"),
        "link_cvar": lower.get("link_cvar"),
        "link_alpha": lower.get("link_alpha"),
        "link_obj": lower.get("link_obj"),
        "layered_obj": lower.get("layered_obj"),
        "link_solve_time_sec": lower.get("link_solve_time_sec"),
        "total_link_solve_time_sec": lower.get("total_link_solve_time_sec"),
    }

    task_rows_header = [
        "task_idx",
        "src_dst",
        "selected_exec_node",
        "model_cvar",
        "model_alpha",
        "b_in",
        "b_out",
        "cpu_demand",
    ]
    task_rows = []
    for task in tasks:
        tid = int(task["task_idx"])
        task_rows.append(
            [
                tid,
                f"{int(task['src'])}->{int(task['dst'])}",
                int(assignment[tid]),
                float(lower.get("cvar", 0.0)),
                float(lower.get("alpha", 0.0)),
                float(task["b_in"]),
                float(task["b_out"]),
                float(task["cpu_demand"]),
            ]
        )

    _write_json(
        out / "best_solution.json",
        {
            "global_metrics": global_metrics,
            "task_rows_header": task_rows_header,
            "task_rows": task_rows,
        },
    )

    candidate_header = [
        "candidate_id",
        "is_feasible",
        "node_feasible",
        "lower_status",
        "node_obj",
        "node_cvar",
        "node_candidate_index",
        "node_candidate_source",
        "node_proxy_u_link_max",
        "u_node_max",
        "network_affinity",
        "link_optimization_mode",
        "single_level_risk_mode",
        "cvar_bound",
        "cvar_bound_slack",
        "link_cvar_tolerance",
        "link_cvar_bound",
        "link_cvar_bound_slack",
        "link_first_stage_cvar",
        "layered2_node_risk_mode",
        "layered2_node_cvar_bound",
        "layered2_node_cvar_bound_slack",
        "link_cvar",
        "link_alpha",
        "cvar",
        "alpha",
        "link_obj",
        "layered_obj",
        "selection_score",
        "u_link_max",
        "total_score",
        "resource_obj",
        "solve_time_sec",
        "note",
    ]
    candidate_csv_rows = []
    for row in candidate_rows:
        candidate_csv_rows.append([row.get(h, "") for h in candidate_header])
    _write_csv(out / "candidate_summary.csv", candidate_header, candidate_csv_rows)

    x_in: Dict[Tuple[int, int], float] = lower.get("x_in", {})
    x_out: Dict[Tuple[int, int], float] = lower.get("x_out", {})
    alloc_rows = []
    for task in tasks:
        tid = int(task["task_idx"])
        m = int(assignment[tid])
        src_dst = f"{int(task['src'])}->{int(task['dst'])}"
        for (ti, tunnel_id), bw in x_in.items():
            if int(ti) != tid or bw <= 0:
                continue
            tunnel_edges = T[int(tunnel_id) - 1]
            path_nodes, _ = _path_to_strings(tunnel_edges, edges)
            alloc_rows.append(
                [
                    tid,
                    src_dst,
                    "in",
                    int(task["src"]),
                    m,
                    int(tunnel_id),
                    path_nodes,
                    float(bw),
                ]
            )
        for (ti, tunnel_id), bw in x_out.items():
            if int(ti) != tid or bw <= 0:
                continue
            tunnel_edges = T[int(tunnel_id) - 1]
            path_nodes, _ = _path_to_strings(tunnel_edges, edges)
            alloc_rows.append(
                [
                    tid,
                    src_dst,
                    "out",
                    m,
                    int(task["dst"]),
                    int(tunnel_id),
                    path_nodes,
                    float(bw),
                ]
            )

    _write_csv(
        out / "alloc_paths.csv",
        ["task_idx", "src_dst", "direction", "from_node", "to_node", "path_id", "path_nodes", "alloc_bw"],
        alloc_rows,
    )

    scenario_rows = []
    u_aux = lower.get("u_aux", {})
    delivered_in = lower.get("delivered_in", {})
    delivered_out = lower.get("delivered_out", {})
    alpha = float(lower.get("alpha", 0.0))
    loss_aggregation = str(lower.get("loss_aggregation", config.get("loss_aggregation", "max"))).lower()
    if loss_aggregation not in {"max", "average"}:
        loss_aggregation = "max"

    # 导出场景损失时使用 delivered_* 重算，避免 LP 辅助变量在退化解中的非唯一取值干扰结果分析。
    scenario_loss_sys: Dict[int, float] = {}
    scenario_loss_task: Dict[Tuple[int, int], float] = {}
    for s_idx in range(1, len(scenarios) + 1):
        max_loss = 0.0
        loss_sum = 0.0
        for task in tasks:
            tid = int(task["task_idx"])
            b_in = float(task["b_in"])
            b_out = float(task["b_out"])
            din = float(delivered_in.get((s_idx, tid), 0.0))
            dout = float(delivered_out.get((s_idx, tid), 0.0))
            loss_t = max(
                0.0,
                1 - din / b_in if b_in > 0 else 0.0,
                1 - dout / b_out if b_out > 0 else 0.0,
            )
            scenario_loss_task[(s_idx, tid)] = float(loss_t)
            max_loss = max(max_loss, float(loss_t))
            loss_sum += float(loss_t)
        if loss_aggregation == "average" and len(tasks) > 0:
            scenario_loss_sys[s_idx] = float(loss_sum / len(tasks))
        else:
            scenario_loss_sys[s_idx] = float(max_loss)

    for s_idx in range(1, len(scenarios) + 1):
        for task in tasks:
            tid = int(task["task_idx"])
            scenario_rows.append(
                [
                    s_idx,
                    float(scenario_probs[s_idx - 1]),
                    float(scenario_loss_sys.get(s_idx, 0.0)),
                    float(u_aux.get(s_idx, 0.0)),
                    alpha,
                    tid,
                    float(scenario_loss_task.get((s_idx, tid), 0.0)),
                    float(delivered_in.get((s_idx, tid), 0.0)),
                    float(delivered_out.get((s_idx, tid), 0.0)),
                ]
            )
    _write_csv(
        out / "scenario_losses.csv",
        [
            "scenario_id",
            "scenario_prob",
            "loss_sys",
            "cvar_aux_u",
            "alpha",
            "task_idx",
            "loss_task",
            "delivered_in",
            "delivered_out",
        ],
        scenario_rows,
    )

    scenario_state_rows = []
    for s_idx, scenario in enumerate(scenarios, start=1):
        failed_edges = [str(e_idx) for e_idx, state in enumerate(scenario, start=1) if float(state) <= 0.5]
        node_state = node_scenarios[s_idx - 1] if node_scenarios is not None and s_idx <= len(node_scenarios) else {}
        failed_nodes = [str(int(node_id)) for node_id, state in sorted(node_state.items()) if float(state) <= 0.5]
        scenario_state_rows.append(
            [
                s_idx,
                float(scenario_probs[s_idx - 1]),
                ";".join(failed_edges),
                ";".join(failed_nodes),
            ]
        )
    _write_csv(
        out / "scenario_states.csv",
        ["scenario_id", "scenario_prob", "failed_edges", "failed_compute_nodes"],
        scenario_state_rows,
    )

    link_rows = []
    link_loads = lower.get("link_loads", {})
    for e_idx, (u, v) in enumerate(edges, start=1):
        load = float(link_loads.get(e_idx, 0.0))
        cap = float(capacity[e_idx - 1])
        util = load / cap if cap > 0 else 0.0
        link_rows.append([e_idx, u, v, cap, load, util, reserved_u_link_max])
    _write_csv(
        out / "link_loads.csv",
        ["edge_id", "u", "v", "capacity", "reserved_load", "reserved_utilization", "reserved_u_link_max"],
        link_rows,
    )

    cpu_capacity = {int(row["node_id"]): float(row["cpu"]) for row in node_rows if int(row["is_compute"]) == 1}
    used, util_map, _ = compute_node_usage(tasks, assignment, cpu_capacity)
    node_rows_csv = []
    for row in node_rows:
        node_id = int(row["node_id"])
        is_compute = int(row["is_compute"])
        cap = float(row["cpu"]) if is_compute == 1 else 0.0
        used_v = float(used.get(node_id, 0.0)) if is_compute == 1 else 0.0
        util_v = float(util_map.get(node_id, 0.0)) if is_compute == 1 else 0.0
        node_rows_csv.append([node_id, is_compute, cap, used_v, util_v, float(best_result["u_node_max"])])
    _write_csv(
        out / "node_utils.csv",
        ["node_id", "is_compute", "cpu_capacity", "cpu_used", "utilization", "u_node_max"],
        node_rows_csv,
    )

    _write_json(out / "sanity_checks.json", sanity_checks)
    _append_experiment_summary(out, config, best_result)
