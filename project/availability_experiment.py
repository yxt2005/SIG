from __future__ import annotations

import csv
import json
from argparse import Namespace
from decimal import Decimal
from pathlib import Path
from typing import Iterable, List

from parsers import next_run
from phase1_correctness import run_phase1_correctness
from Algorithms.common import build_tunnel_scenario_matrix


# ===================== 1. 参数解析工具：生成 scale 序列、需求行列表，并把 numpy 类型转换成 JSON 可写格式 =====================
# Availability 实验需要对多个 demand scale 循环运行，同一套工具保证命令行输入和结果文件中的参数一致。


def collect_scales(start: float, step: float, finish: float) -> List[float]:
    if step == 0:
        raise ValueError("scale step must not be 0.")
    start_d = Decimal(str(start))
    step_d = Decimal(str(step))
    finish_d = Decimal(str(finish))
    places = max(
        max(0, -start_d.as_tuple().exponent),
        max(0, -step_d.as_tuple().exponent),
        max(0, -finish_d.as_tuple().exponent),
    )
    quant = Decimal(1).scaleb(-places)
    values: List[float] = []
    value = start_d
    eps = Decimal("1e-18")
    if step_d > 0:
        while value <= finish_d + eps:
            values.append(float(value.quantize(quant)))
            value += step_d
    else:
        while value >= finish_d - eps:
            values.append(float(value.quantize(quant)))
            value += step_d
    return values


def parse_demand_rows(raw: str | Iterable[int]) -> List[int]:
    if isinstance(raw, str):
        rows = [int(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        rows = [int(item) for item in raw]
    if not rows:
        raise ValueError("At least one demand row is required.")
    if any(row <= 0 for row in rows):
        raise ValueError("Demand rows are 1-based and must be positive.")
    return rows


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


# ===================== 2. Availability 计算：按 TEAVAR 口径统计 P(loss <= target) =====================
# 这里的 loss 来自每个算法求解后导出的 scenario_losses.csv；它是事后评价指标，不是 CVaR 本身。


def read_scenario_losses(output_dir: str | Path):
    losses = {}
    probs = {}
    with (Path(output_dir) / "scenario_losses.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario_id = int(row["scenario_id"])
            probs[scenario_id] = float(row["scenario_prob"])
            losses[scenario_id] = float(row["loss_sys"])
    return losses, probs


def availability_from_losses(losses: dict[int, float], probs: dict[int, float], target: float):
    return sum(float(probs[sid]) for sid, loss in losses.items() if float(loss) <= float(target) + 1e-12)


def _task_path_weights(x_map, task_id: int):
    weights = {}
    for key, value in x_map.items():
        if len(key) < 2 or int(key[0]) != int(task_id):
            continue
        tunnel_id = int(key[-1])
        amount = float(value)
        if amount > 1e-12:
            weights[tunnel_id] = weights.get(tunnel_id, 0.0) + amount
    return weights


def _assignment_node(assignment, task_id: int):
    if task_id in assignment:
        return int(assignment[task_id])
    return int(assignment[str(task_id)])


def _add_routed_load(link_loads, T, tunnel_id: int, routed: float):
    if routed <= 0:
        return
    if not (0 < int(tunnel_id) <= len(T)):
        return
    for edge_id in T[int(tunnel_id) - 1]:
        if 1 <= int(edge_id) <= len(link_loads):
            link_loads[int(edge_id) - 1] += float(routed)


def _max_link_utilization(link_loads, capacity):
    if not link_loads:
        return 0.0
    return max(float(load) / max(float(capacity[idx]), 1e-12) for idx, load in enumerate(link_loads))


def calculate_actual_normal_u_link_max(result):
    """按求解出的放置和 x 权重归一化后，计算无故障状态下真实业务流量的最大链路利用率。"""
    context = result["availability_context"]
    tasks = context["tasks"]
    capacity = [float(x) for x in context["capacity"]]
    T = context["T"]
    assignment = result["best_assignment"]
    lower = result["best_lower"]
    x_in = lower.get("x_in", {})
    x_out = lower.get("x_out", {})
    link_loads = [0.0 for _ in context["edges"]]

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
                _add_routed_load(link_loads, T, tunnel_id, demand * float(weight) / float(total_weight))

    return _max_link_utilization(link_loads, capacity)


def evaluate_fixed_allocation_link_utilization(result, scenario_probs):
    """固定分配口径下，按故障后仍存活的 x 直接计算各场景最大链路利用率。"""
    context = result["availability_context"]
    tasks = context["tasks"]
    capacity = [float(x) for x in context["capacity"]]
    T = context["T"]
    edges = context["edges"]
    scenarios = context["scenarios"]
    node_scenarios = context["node_scenarios"]
    assignment = result["best_assignment"]
    lower = result["best_lower"]
    x_in = lower.get("x_in", {})
    x_out = lower.get("x_out", {})
    x_path = build_tunnel_scenario_matrix(T, edges, scenarios)
    scenario_fail_u_link_max = {}

    for s_idx in range(1, len(scenarios) + 1):
        link_loads = [0.0 for _ in edges]
        node_state = node_scenarios[s_idx - 1] if node_scenarios is not None else {}
        for task in tasks:
            tid = int(task["task_idx"])
            m = _assignment_node(assignment, tid)
            if float(node_state.get(m, 1.0)) <= 0.5:
                continue
            for direction, weights in (
                ("in", _task_path_weights(x_in, tid)),
                ("out", _task_path_weights(x_out, tid)),
            ):
                endpoint_local = (
                    direction == "in" and int(task["src"]) == m
                ) or (
                    direction == "out" and int(task["dst"]) == m
                )
                if endpoint_local:
                    continue
                for tunnel_id, amount in weights.items():
                    if 0 < tunnel_id <= len(T) and float(x_path[s_idx - 1, tunnel_id - 1]) > 0.5:
                        _add_routed_load(link_loads, T, tunnel_id, amount)
        scenario_fail_u_link_max[s_idx] = _max_link_utilization(link_loads, capacity)

    expected_fail_u_link_max = sum(
        float(scenario_probs[s_idx - 1]) * value
        for s_idx, value in scenario_fail_u_link_max.items()
    )
    worst_fail_u_link_max = max(scenario_fail_u_link_max.values(), default=0.0)
    return {
        "expected_fail_u_link_max": float(expected_fail_u_link_max),
        "worst_fail_u_link_max": float(worst_fail_u_link_max),
    }


def evaluate_teavar_reallocation(result, scenario_probs, target: float):
    """按 TEAVAR availability 评价口径，对算法输出的正常态分流做故障后重归一化评价。"""
    context = result["availability_context"]
    tasks = context["tasks"]
    edges = context["edges"]
    capacity = [float(x) for x in context["capacity"]]
    T = context["T"]
    scenarios = context["scenarios"]
    node_scenarios = context["node_scenarios"]
    assignment = result["best_assignment"]
    lower = result["best_lower"]
    x_in = lower.get("x_in", {})
    x_out = lower.get("x_out", {})
    x_path = build_tunnel_scenario_matrix(T, edges, scenarios)

    scenario_losses = {}
    scenario_fail_u_link_max = {}
    total_required = sum(float(task["b_in"]) + float(task["b_out"]) for task in tasks)
    total_required = max(total_required, 1e-12)

    for s_idx in range(1, len(scenarios) + 1):
        link_loads = [0.0 for _ in edges]
        delivered_before_overflow = 0.0
        node_state = node_scenarios[s_idx - 1] if node_scenarios is not None else {}

        for task in tasks:
            tid = int(task["task_idx"])
            m = _assignment_node(assignment, tid)
            node_up = float(node_state.get(m, 1.0)) > 0.5
            if not node_up:
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
                    if 0 < tunnel_id <= len(T) and float(x_path[s_idx - 1, tunnel_id - 1]) > 0.5
                }
                total_alive_weight = sum(alive.values())
                if total_alive_weight <= 1e-12:
                    continue
                delivered_before_overflow += demand
                for tunnel_id, weight in alive.items():
                    _add_routed_load(link_loads, T, tunnel_id, demand * float(weight) / float(total_alive_weight))

        overflow = 0.0
        for edge_idx, load in enumerate(link_loads):
            excess = max(0.0, float(load) - capacity[edge_idx])
            overflow += excess

        delivered_after_overflow = max(0.0, delivered_before_overflow - overflow)
        loss = min(1.0, max(0.0, 1.0 - delivered_after_overflow / total_required))
        scenario_losses[s_idx] = float(loss)
        scenario_fail_u_link_max[s_idx] = _max_link_utilization(link_loads, capacity)

    availability = availability_from_losses(scenario_losses, {i + 1: p for i, p in enumerate(scenario_probs)}, target)
    expected_fail_u_link_max = sum(
        float(scenario_probs[s_idx - 1]) * value
        for s_idx, value in scenario_fail_u_link_max.items()
    )
    worst_fail_u_link_max = max(scenario_fail_u_link_max.values(), default=0.0)
    return {
        "availability": float(availability),
        "expected_fail_u_link_max": float(expected_fail_u_link_max),
        "worst_fail_u_link_max": float(worst_fail_u_link_max),
    }


def append_dict_row(path: Path, row: dict):
    """每完成一次算法运行就追加一行，避免长实验中途停止时丢失已完成结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ===================== 3. 算法参数模板：对比单层 A、无故障感知 A_no_failure 和默认 layered3 =====================
# beta 默认采用 TEAVAR 口径：无故障场景概率减 0.01；每次故障采样通过 seed 改变 Weibull 链路概率。


def base_solver_args(common: Namespace, *, algorithm: str, scale: float, demand_row: int, seed: int, output_root: str):
    single_level_algorithm = algorithm in {"A", "A_no_failure", "A_agnostic"}
    risk_weight = 0.0 if algorithm in {"A_no_failure", "A_agnostic"} else common.risk_weight
    single_level_risk_mode = "weighted" if algorithm in {"A_no_failure", "A_agnostic"} else common.single_level_risk_mode
    single_level_cvar_bound = (
        None
        if single_level_risk_mode != "cvar_constraint" or algorithm in {"A_no_failure", "A_agnostic"}
        else common.single_level_cvar_bound
    )
    data = {
        "topology": common.topology,
        "num_demand": demand_row,
        "demand_scale": scale,
        "demand_downscale": common.demand_downscale,
        "max_tasks": common.max_tasks,
        "placement_policy": common.placement_policy,
        "allow_same_src_dst": common.allow_same_src_dst,
        "rho": common.rho,
        "paths": common.paths,
        "k_paths": common.k_paths,
        "edge_disjoint": common.edge_disjoint,
        "beta": common.beta,
        "beta_mode": "teavar",
        "teavar_beta_margin": common.teavar_beta_margin,
        "lambda_weight": common.lambda_weight,
        "cutoff": common.cutoff,
        "weibull_scale": common.weibull_scale,
        "seed": seed,
        "algorithm": algorithm,
        "include_node_failures": common.failure_scope == "joint",
        "output_root": output_root,
        "quiet": True,
        "risk_weight": risk_weight,
        "single_level_risk_mode": single_level_risk_mode,
        "single_level_cvar_bound": single_level_cvar_bound,
        "loss_aggregation": common.loss_aggregation,
        "node_risk_weight": common.node_risk_weight,
        "link_risk_weight": common.link_risk_weight,
        "node_loss_aggregation": common.node_loss_aggregation,
        "link_loss_aggregation": common.link_loss_aggregation,
        "link_optimization_mode": common.link_optimization_mode,
        "link_cvar_tolerance": common.link_cvar_tolerance,
        "node_proxy_link_weight": common.node_proxy_link_weight,
        "layered3_candidate_budget": common.layered3_candidate_budget,
        "layered3_node_u_tol": common.layered3_node_u_tol,
        "layered3_node_cvar_tol": common.layered3_node_cvar_tol,
        "layered3_affinity_tol": common.layered3_affinity_tol,
        "enable_mip_gap": common.enable_mip_gap if single_level_algorithm else False,
        "mip_gap": common.mip_gap if single_level_algorithm else 0.0,
        "time_limit_sec": common.a_time_limit_sec if single_level_algorithm else None,
    }
    return Namespace(**data)


def default_common_args(**overrides):
    data = {
        "topology": "B4",
        "algorithms": ["A", "A_no_failure", "layered3"],
        "demand_rows": [1],
        "failure_iterations": 1,
        "scale_start": 1.0,
        "scale_step": 0.2,
        "scale_finish": 2.0,
        "demand_downscale": 2.0,
        "max_tasks": 144,
        "placement_policy": "exclude_endpoints",
        "allow_same_src_dst": False,
        "rho": 0.5,
        "paths": "KSP",
        "k_paths": 6,
        "edge_disjoint": False,
        "beta": 0.99,
        "teavar_beta_margin": 0.01,
        "lambda_weight": 0.5,
        "cutoff": 1e-4,
        "weibull_scale": 0.001,
        "seed": 1,
        "failure_scope": "joint",
        "availability_target": 0.0,
        "risk_weight": 0.5,
        "single_level_risk_mode": "weighted",
        "single_level_cvar_bound": None,
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
        "enable_mip_gap": True,
        "mip_gap": 0.01,
        "a_time_limit_sec": 600.0,
        "availability_eval_mode": "teavar_reallocation",
    }
    data.update(overrides)
    data["demand_rows"] = parse_demand_rows(data["demand_rows"])
    return Namespace(**data)


# ===================== 4. 实验主循环：scale、demand 行、故障采样、算法四层循环，并实时打印阶段性日志 =====================
# 每次算法运行都会产生一个子目录；汇总文件保存在 data/raw/availability/<run_id>/ 下。


def run_availability_experiment(common: Namespace):
    scales = collect_scales(common.scale_start, common.scale_step, common.scale_finish)
    out_dir = Path(next_run("data/raw/availability"))
    run_root = out_dir / "solver_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    total = len(scales) * len(common.demand_rows) * int(common.failure_iterations) * len(common.algorithms)
    done = 0
    rows = []
    summary_path = out_dir / "summary.csv"
    print(f"[availability] output_dir={out_dir.resolve()}")
    print(
        f"[availability] algorithms={common.algorithms}, scales={scales}, "
        f"demand_rows={common.demand_rows}, failure_iterations={common.failure_iterations}, "
        f"failure_scope={common.failure_scope}"
    )

    for scale_idx, scale in enumerate(scales, start=1):
        print(f"[availability] scale {scale_idx}/{len(scales)} started: scale={scale}")
        for demand_row in common.demand_rows:
            for iteration in range(1, int(common.failure_iterations) + 1):
                # 同一 demand 行和同一故障采样编号在所有 scale 下复用同一随机种子，
                # 使 Availability 曲线比较的是负载规模变化，而不是不同故障样本变化。
                sample_seed = int(common.seed) + int(demand_row) * 100 + iteration
                for algorithm in common.algorithms:
                    done += 1
                    print(
                        f"[availability] [{done}/{total}] run algorithm={algorithm}, "
                        f"scale={scale}, demand_row={demand_row}, sample={iteration}, seed={sample_seed}"
                    )
                    solver_args = base_solver_args(
                        common,
                        algorithm=algorithm,
                        scale=scale,
                        demand_row=demand_row,
                        seed=sample_seed,
                        output_root=str(run_root / algorithm),
                    )
                    solver_output_dir, result = run_phase1_correctness(solver_args)
                    actual_normal_u_link_max = calculate_actual_normal_u_link_max(result)
                    if common.availability_eval_mode == "fixed_x":
                        losses, probs = read_scenario_losses(solver_output_dir)
                        fixed_link_metrics = evaluate_fixed_allocation_link_utilization(result, result["scenario_probs"])
                        availability_metrics = {
                            "availability": availability_from_losses(losses, probs, common.availability_target),
                            **fixed_link_metrics,
                        }
                    elif common.availability_eval_mode == "teavar_reallocation":
                        availability_metrics = evaluate_teavar_reallocation(
                            result,
                            result["scenario_probs"],
                            common.availability_target,
                        )
                    else:
                        raise ValueError("availability_eval_mode must be fixed_x or teavar_reallocation.")
                    row = {
                        "algorithm": algorithm,
                        "scale": scale,
                        "u_node_max": result["u_node_max"],
                        "reserved_u_link_max": result["u_link_max"],
                        "actual_normal_u_link_max": actual_normal_u_link_max,
                        "availability": availability_metrics["availability"],
                        "expected_fail_u_link_max": availability_metrics["expected_fail_u_link_max"],
                        "worst_fail_u_link_max": availability_metrics["worst_fail_u_link_max"],
                        "model_cvar": result["best_lower"].get("cvar"),
                        "single_level_risk_mode": result["best_lower"].get("single_level_risk_mode"),
                        "single_level_cvar_bound": result["best_lower"].get("cvar_bound"),
                        "cvar_bound_slack": result["best_lower"].get("cvar_bound_slack"),
                        "failure_scope": common.failure_scope,
                        "eval_mode": common.availability_eval_mode,
                        "demand_row": demand_row,
                        "failure_iteration": iteration,
                        "solver_status": result["best_lower"].get("status"),
                        "solve_time_sec": result["best_lower"].get("solve_time_sec"),
                        "solver_output_dir": str(Path(solver_output_dir).resolve()),
                    }
                    rows.append(row)
                    append_dict_row(summary_path, row)
                    print(
                        f"[availability] [{done}/{total}] done algorithm={algorithm}, "
                        f"availability={float(row['availability']):.6f}, "
                        f"U_node={float(row['u_node_max']):.6f}, "
                        f"reserved_U_link={float(row['reserved_u_link_max']):.6f}, "
                        f"actual_U_link={float(row['actual_normal_u_link_max']):.6f}, "
                        f"expected_fail_U_link={float(row['expected_fail_u_link_max']):.6f}"
                    )

    grouped = {}
    for row in rows:
        key = (row["algorithm"], row["scale"])
        grouped.setdefault(key, []).append(row)
    aggregate_rows = []
    for (algorithm, scale), vals in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        count = len(vals)
        solve_times = [
            float(v["solve_time_sec"])
            for v in vals
            if v.get("solve_time_sec") not in {None, ""}
        ]
        cvar_values = [
            float(v["model_cvar"])
            for v in vals
            if v.get("model_cvar") not in {None, ""}
        ]
        aggregate_rows.append(
            {
                "algorithm": algorithm,
                "scale": scale,
                "u_node_max_mean": sum(float(v["u_node_max"]) for v in vals) / count,
                "reserved_u_link_max_mean": sum(float(v["reserved_u_link_max"]) for v in vals) / count,
                "actual_normal_u_link_max_mean": sum(float(v["actual_normal_u_link_max"]) for v in vals) / count,
                "availability_mean": sum(float(v["availability"]) for v in vals) / count,
                "expected_fail_u_link_max_mean": sum(float(v["expected_fail_u_link_max"]) for v in vals) / count,
                "worst_fail_u_link_max": max(float(v["worst_fail_u_link_max"]) for v in vals),
                "model_cvar_mean": sum(cvar_values) / len(cvar_values) if cvar_values else "",
                "single_level_risk_mode": next(
                    (v.get("single_level_risk_mode") for v in vals if v.get("single_level_risk_mode")),
                    "",
                ),
                "single_level_cvar_bound": next(
                    (v.get("single_level_cvar_bound") for v in vals if v.get("single_level_cvar_bound") not in {None, ""}),
                    "",
                ),
                "solve_time_sec_mean": sum(solve_times) / len(solve_times) if solve_times else "",
                "sample_count": count,
            }
        )
    aggregate_path = out_dir / "availability_by_scale.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    details = {
        "config": vars(common),
        "scales": scales,
        "summary_csv": str(summary_path.resolve()),
        "availability_by_scale_csv": str(aggregate_path.resolve()),
    }
    with (out_dir / "details.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(details), f, ensure_ascii=False, indent=2)

    print(f"[availability] finished. summary={summary_path.resolve()}")
    print(f"[availability] aggregate={aggregate_path.resolve()}")
    return str(out_dir), rows, aggregate_rows
