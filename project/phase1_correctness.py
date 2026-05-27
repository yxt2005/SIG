from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from Algorithms.A_solver import solve_A
from Algorithms.link_layer_solver import solve_link_layer
from Algorithms.layered1_node_solver import solve_layered1_node_layer
from Algorithms.layered2_node_solver import prepare_layered2_node_data, solve_layered2_node_layer
from Algorithms.layered3_node_solver import prepare_layered3_node_data, solve_layered3_node_layer
from parsers import (
    build_tasks_from_demand_and_taskfile,
    load_paths_for_all_pairs,
    next_run,
    read_demand,
    read_node_resources,
    read_task_file,
    read_topology,
)
from simulation import run_sanity_checks
from util import build_joint_failure_scenarios, weibull_probs, write_phase1_results


def run_phase1_correctness(args: Namespace):
    np.random.seed(args.seed)

    # ===================== 1. 解析算法名称：把用户输入映射到结果目录和内部求解器名称 =====================
    # layered 保留为 layered1 的别名；当前入口只支持单层 A 和三个分层版本。
    algorithm_names = {
        "A": "A_solver",
        "A_no_failure": "A_solver",
        "A_agnostic": "A_solver",
        "layered": "layered1_solver",
        "layered1": "layered1_solver",
        "layered2": "layered2_solver",
        "layered3": "layered3_solver",
    }
    if args.algorithm not in algorithm_names:
        raise ValueError("Unsupported algorithm. Use A, A_no_failure, layered1, layered2, or layered3.")
    requested_algorithm = str(args.algorithm)
    failure_agnostic_single_level = requested_algorithm in {"A_no_failure", "A_agnostic"}
    algorithm = algorithm_names[args.algorithm]

    output_algorithm_name = "A_no_failure_solver" if failure_agnostic_single_level else algorithm
    output_root = getattr(args, "output_root", f"data/raw/phase1_correctness/{output_algorithm_name}")
    output_dir = next_run(output_root)

    # ===================== 2. 读取拓扑、节点资源和任务需求：把原始 demand/task 文件转换成任务列表 =====================
    # tasks 中每个任务包含源点、目的点、输入/输出流量、CPU 需求和候选计算节点，是后续建模的统一输入。
    edges, capacity, _, nodes = read_topology(args.topology)
    node_rows, compute_nodes, cpu_capacity, node_failure_probs = read_node_resources(args.topology)
    task_file = f"data/{args.topology}/task.txt"

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
    task_cpu_map = read_task_file(task_file)
    tasks = build_tasks_from_demand_and_taskfile(
        demand=demand,
        flows=flows,
        task_cpu_map=task_cpu_map,
        rho=args.rho,
        candidate_ms=compute_nodes,
        max_tasks=args.max_tasks,
        placement_policy=placement_policy,
    )
    if len(tasks) == 0:
        raise ValueError("No tasks built from demand/task mapping.")

    # ===================== 3. 读取候选路径：为所有源宿对准备 K 条候选路径，并建立“节点对 -> 路径编号”索引 =====================
    # 单层模型和链路层模型都会复用同一批 k_paths，保证节点层代理评估与链路层真实评估的路径集合一致。
    T, Tf, k_used, flows_all, pair_to_tunnels = load_paths_for_all_pairs(
        topology=args.topology,
        path_mode=args.paths,
        links=edges,
        nodes=nodes,
        k_paths=args.k_paths,
        edge_disjoint=args.edge_disjoint,
    )

    enable_node_failures = bool(getattr(args, "include_node_failures", True)) and algorithm in {
        "A_solver",
        "layered1_solver",
        "layered2_solver",
        "layered3_solver",
    }

    # ===================== 4. 构建故障场景：链路层使用链路+节点联合故障，节点层只使用节点故障场景 =====================
    # layered2 的节点层代理损失需要看到链路故障，因此后面会在 layered2 分支中改用联合场景。
    link_probs = weibull_probs(len(edges), shape=0.8, scale=args.weibull_scale)
    scenarios, node_scenarios, scenario_probs = build_joint_failure_scenarios(
        link_probs=link_probs,
        compute_nodes=compute_nodes,
        node_failure_probs=node_failure_probs,
        cutoff=args.cutoff,
        include_node_failures=enable_node_failures,
    )
    _, node_layer_scenarios, node_layer_probs = build_joint_failure_scenarios(
        link_probs=[],
        compute_nodes=compute_nodes,
        node_failure_probs=node_failure_probs,
        cutoff=args.cutoff,
        include_node_failures=enable_node_failures,
    )
    effective_beta = float(args.beta)
    if str(getattr(args, "beta_mode", "fixed")).lower() == "teavar":
        beta_margin = float(getattr(args, "teavar_beta_margin", 0.01))
        effective_beta = max(0.0, min(0.999999, float(scenario_probs[0]) - beta_margin))

    if algorithm == "A_solver":
        # ===================== 5A. 单层算法 A：在一个 MILP 中联合优化放置、路由、容量利用率和故障 CVaR =====================
        # 该分支作为性能上界/基准；weighted 模式使用 risk_weight 加权 CVaR，
        # cvar_constraint 模式使用 Gamma 上界控制 CVaR，并在风险约束下最小化资源利用率。
        risk_weight = 0.0 if failure_agnostic_single_level else float(args.risk_weight)
        single_level_risk_mode = "weighted" if failure_agnostic_single_level else str(
            getattr(args, "single_level_risk_mode", "weighted") or "weighted"
        ).lower()
        if single_level_risk_mode not in {"weighted", "cvar_constraint"}:
            raise ValueError("single_level_risk_mode must be 'weighted' or 'cvar_constraint'.")
        single_level_cvar_bound = getattr(args, "single_level_cvar_bound", None)
        if single_level_risk_mode == "cvar_constraint":
            if single_level_cvar_bound is None:
                raise ValueError("single_level_cvar_bound is required when single_level_risk_mode is cvar_constraint.")
            single_level_cvar_bound = max(0.0, float(single_level_cvar_bound))
        else:
            single_level_cvar_bound = None
        loss_aggregation = str(getattr(args, "loss_aggregation", "average")).lower()
        node_risk_weight = None
        link_risk_weight = None
        node_loss_aggregation = None
        link_loss_aggregation = None
        link_optimization_mode = None
        link_cvar_tolerance = None
        link_cvar_bound = None
        layered2_node_risk_mode = None
        layered2_node_cvar_bound = None
        node_loss_model = None
        node_proxy_split_rule = None
        node_proxy_k_paths = None
        node_proxy_k_paths_source = None
        node_proxy_capacity_constraints = None
        node_proxy_link_weight = None
        network_affinity_model = None
        layered3_candidate_budget = None
        layered3_node_u_tol = None
        layered3_node_cvar_tol = None
        layered3_affinity_tol = None
        result = solve_A(
            tasks=tasks,
            edges=edges,
            capacity=capacity,
            beta=effective_beta,
            lambda_weight=args.lambda_weight,
            scenarios=scenarios,
            scenario_probs=scenario_probs,
            node_scenarios=node_scenarios,
            pair_to_tunnels=pair_to_tunnels,
            cpu_capacity=cpu_capacity,
            T=T,
            risk_weight=risk_weight,
            risk_mode=single_level_risk_mode,
            cvar_bound=single_level_cvar_bound,
            loss_aggregation=loss_aggregation,
            enable_mip_gap=args.enable_mip_gap,
            mip_gap=args.mip_gap,
            time_limit_sec=getattr(args, "time_limit_sec", None),
            seed=args.seed,
        )
    elif algorithm in {"layered1_solver", "layered2_solver", "layered3_solver"}:
        # ===================== 5B. 分层算法公共参数：先由节点层生成放置候选，再由链路层固定放置求真实路由 =====================
        # layered1 节点层只看节点故障；layered2 加入平均分流代理损失/代理链路负载；
        # layered3 主要生成网络亲和度邻域候选，并由链路层按 CVaR 与 U_link 选择。
        risk_weight = None
        single_level_risk_mode = None
        single_level_cvar_bound = None
        loss_aggregation = None
        node_risk_weight = float(args.node_risk_weight)
        link_risk_weight = float(args.link_risk_weight)
        node_loss_aggregation = str(getattr(args, "node_loss_aggregation", "average")).lower()
        link_loss_aggregation = str(getattr(args, "link_loss_aggregation", "average")).lower()
        is_layered2 = algorithm == "layered2_solver"
        is_layered3 = algorithm == "layered3_solver"
        link_optimization_mode = "weighted" if is_layered3 else str(
            getattr(args, "link_optimization_mode", "weighted") or "weighted"
        ).lower()
        if link_optimization_mode not in {"weighted", "cvar_then_u", "cvar_constraint"}:
            raise ValueError("link_optimization_mode must be 'weighted', 'cvar_then_u', or 'cvar_constraint'.")
        raw_link_cvar_tolerance = getattr(args, "link_cvar_tolerance", 0.0)
        link_cvar_tolerance = 0.0 if is_layered3 else max(
            0.0,
            float(0.0 if raw_link_cvar_tolerance is None else raw_link_cvar_tolerance),
        )
        raw_link_cvar_bound = getattr(args, "link_cvar_bound", None)
        if (not is_layered3) and link_optimization_mode == "cvar_constraint":
            if raw_link_cvar_bound is None:
                raise ValueError("link_cvar_bound is required when link_optimization_mode is cvar_constraint.")
            link_cvar_bound = max(0.0, float(raw_link_cvar_bound))
        else:
            link_cvar_bound = None
        layered2_node_risk_mode = str(getattr(args, "layered2_node_risk_mode", "weighted") or "weighted").lower()
        if not is_layered2:
            layered2_node_risk_mode = None
            layered2_node_cvar_bound = None
        else:
            if layered2_node_risk_mode not in {"weighted", "cvar_constraint"}:
                raise ValueError("layered2_node_risk_mode must be 'weighted' or 'cvar_constraint'.")
            raw_node_cvar_bound = getattr(args, "layered2_node_cvar_bound", None)
            if layered2_node_risk_mode == "cvar_constraint":
                if raw_node_cvar_bound is None:
                    raise ValueError("layered2_node_cvar_bound is required when layered2_node_risk_mode is cvar_constraint.")
                layered2_node_cvar_bound = max(0.0, float(raw_node_cvar_bound))
            else:
                layered2_node_cvar_bound = None
        node_loss_model = "average_split_proxy" if is_layered2 else "node_failure"
        node_proxy_split_rule = "average" if is_layered2 else None
        node_proxy_k_paths = int(args.k_paths) if is_layered2 else None
        node_proxy_k_paths_source = "k_paths" if is_layered2 else None
        node_proxy_capacity_constraints = bool(is_layered2)
        node_proxy_link_weight = float(getattr(args, "node_proxy_link_weight", 0.0) or 0.0) if is_layered2 else 0.0
        network_affinity_model = "path_count_overlap_length" if is_layered3 else None
        raw_layered3_candidate_budget = getattr(args, "layered3_candidate_budget", 8)
        layered3_candidate_budget = max(
            1,
            int(8 if raw_layered3_candidate_budget is None else raw_layered3_candidate_budget),
        ) if is_layered3 else 1
        raw_layered3_node_u_tol = getattr(args, "layered3_node_u_tol", 0.0)
        raw_layered3_node_cvar_tol = getattr(args, "layered3_node_cvar_tol", 1e-4)
        layered3_node_u_tol = (
            max(0.0, float(0.0 if raw_layered3_node_u_tol is None else raw_layered3_node_u_tol))
            if is_layered3
            else None
        )
        layered3_node_cvar_tol = (
            max(0.0, float(1e-4 if raw_layered3_node_cvar_tol is None else raw_layered3_node_cvar_tol))
            if is_layered3
            else None
        )
        raw_layered3_affinity_tol = getattr(args, "layered3_affinity_tol", None)
        layered3_affinity_tol = (
            None if raw_layered3_affinity_tol is None else max(0.0, float(raw_layered3_affinity_tol))
        ) if is_layered3 else None

        layered2_node_data = None
        layered3_node_data = None
        if is_layered2:
            # layered2 的执行逻辑集中在 layered2_node_solver：节点层使用联合故障场景、平均分流代理损失和代理链路容量。
            layered2_node_data = prepare_layered2_node_data(
                tasks=tasks,
                compute_nodes=compute_nodes,
                scenarios=scenarios,
                scenario_probs=scenario_probs,
                node_scenarios=node_scenarios,
                pair_to_tunnels=pair_to_tunnels,
                T=T,
                edges=edges,
                k_paths=args.k_paths,
                nedges=len(edges),
            )
            node_loss_model = layered2_node_data["node_loss_model"]
            node_proxy_split_rule = layered2_node_data["node_proxy_split_rule"]
            node_proxy_k_paths = layered2_node_data["node_proxy_k_paths"]
            node_proxy_k_paths_source = layered2_node_data["node_proxy_k_paths_source"]
            node_proxy_capacity_constraints = layered2_node_data["node_proxy_capacity_constraints"]
        elif is_layered3:
            # layered3 的执行逻辑集中在 layered3_node_solver：节点层生成网络亲和度候选，链路层负责真实路由评估。
            layered3_node_data = prepare_layered3_node_data(
                tasks=tasks,
                pair_to_tunnels=pair_to_tunnels,
                T=T,
                k_paths=args.k_paths,
            )
            node_loss_model = layered3_node_data["node_loss_model"]
            node_proxy_split_rule = layered3_node_data["node_proxy_split_rule"]
            node_proxy_k_paths = layered3_node_data["node_proxy_k_paths"]
            node_proxy_k_paths_source = layered3_node_data["node_proxy_k_paths_source"]
            node_proxy_capacity_constraints = layered3_node_data["node_proxy_capacity_constraints"]
            network_affinity_model = layered3_node_data["network_affinity_model"]

        def run_node_model():
            # 这里按算法版本分发到各自的节点层入口，避免 layered1/2/3 的参数在一个调用中相互缠绕。
            if is_layered2:
                return solve_layered2_node_layer(
                    tasks=tasks,
                    compute_nodes=compute_nodes,
                    cpu_capacity=cpu_capacity,
                    pair_to_tunnels=pair_to_tunnels,
                    beta=effective_beta,
                    node_risk_weight=node_risk_weight,
                    node_loss_aggregation=node_loss_aggregation,
                    seed=args.seed,
                    node_data=layered2_node_data,
                    link_capacity=capacity,
                    node_proxy_link_weight=node_proxy_link_weight,
                    node_risk_mode=layered2_node_risk_mode or "weighted",
                    node_cvar_bound=layered2_node_cvar_bound,
                )
            if is_layered3:
                return solve_layered3_node_layer(
                    tasks=tasks,
                    compute_nodes=compute_nodes,
                    cpu_capacity=cpu_capacity,
                    node_scenarios=node_layer_scenarios,
                    node_scenario_probs=node_layer_probs,
                    pair_to_tunnels=pair_to_tunnels,
                    beta=effective_beta,
                    node_risk_weight=node_risk_weight,
                    node_loss_aggregation=node_loss_aggregation,
                    seed=args.seed,
                    node_data=layered3_node_data,
                    candidate_budget=layered3_candidate_budget,
                    node_u_tol=layered3_node_u_tol or 0.0,
                    node_cvar_tol=layered3_node_cvar_tol or 0.0,
                    affinity_tol=layered3_affinity_tol,
                )
            return solve_layered1_node_layer(
                tasks=tasks,
                compute_nodes=compute_nodes,
                cpu_capacity=cpu_capacity,
                node_scenarios=node_layer_scenarios,
                node_scenario_probs=node_layer_probs,
                pair_to_tunnels=pair_to_tunnels,
                beta=effective_beta,
                node_risk_weight=node_risk_weight,
                node_loss_aggregation=node_loss_aggregation,
                seed=args.seed,
            )

        # ===================== 6B-1. 节点层求解：layered1/2 返回单个放置，layered3 返回网络亲和度邻域候选 =====================
        node_result = run_node_model()

        node_candidates = node_result.get("candidate_solutions", [node_result]) if is_layered3 else [node_result]
        evaluated_candidates = []
        candidate_rows = []
        total_link_solve_time = 0.0

        # ===================== 7. 链路层评估候选：对每个节点层放置方案固定 y，重新优化真实路径流量 x =====================
        # candidate_summary.csv 中每一行对应一个候选放置及其链路层评估结果。
        for candidate_id, node_candidate in enumerate(node_candidates, start=1):
            link_lower = solve_link_layer(
                tasks=tasks,
                assignment=node_candidate["assignment"],
                edges=edges,
                capacity=capacity,
                beta=effective_beta,
                scenarios=scenarios,
                scenario_probs=scenario_probs,
                node_scenarios=node_scenarios,
                pair_to_tunnels=pair_to_tunnels,
                cpu_capacity=cpu_capacity,
                T=T,
                risk_weight=link_risk_weight,
                loss_aggregation=link_loss_aggregation,
                optimization_mode=link_optimization_mode,
                cvar_tolerance=link_cvar_tolerance,
                cvar_bound=link_cvar_bound,
            )
            if link_lower.get("status") not in {"optimal", "suboptimal"}:
                candidate_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "assignment_repr": node_candidate["assignment_repr"],
                        "is_feasible": False,
                        "node_feasible": True,
                        "lower_status": link_lower.get("status"),
                        "node_obj": node_candidate["node_obj"],
                        "node_cvar": node_candidate["node_cvar"],
                        "u_node_max": node_candidate["u_node_max"],
                        "network_affinity": node_candidate.get("network_affinity"),
                        "total_score": "",
                        "selection_score": "",
                        "solve_time_sec": node_candidate["solve_time_sec"],
                        "note": link_lower.get("reason", f"{algorithm}_node_link_infeasible"),
                    }
                )
                continue

            link_solve_time = float(link_lower["solve_time_sec"])
            total_link_solve_time += link_solve_time
            candidate_total_time = float(node_result["solve_time_sec"]) + link_solve_time
            link_lower["link_solve_time_sec"] = link_solve_time
            link_lower["node_solve_time_sec"] = node_result["solve_time_sec"]
            link_lower["node_cvar"] = node_candidate["node_cvar"]
            link_lower["node_alpha"] = node_candidate["node_alpha"]
            link_lower["node_obj"] = node_candidate["node_obj"]
            link_lower["node_loss_model"] = node_candidate["node_loss_model"]
            if is_layered2:
                link_lower["node_proxy_cvar"] = node_candidate["node_cvar"]
                link_lower["node_proxy_alpha"] = node_candidate["node_alpha"]
                link_lower["node_proxy_obj"] = node_candidate["node_obj"]
                link_lower["node_proxy_split_rule"] = node_proxy_split_rule
                link_lower["node_proxy_k_paths"] = node_proxy_k_paths
                link_lower["node_proxy_k_paths_source"] = node_proxy_k_paths_source
                link_lower["node_proxy_capacity_constraints"] = node_proxy_capacity_constraints
                link_lower["node_proxy_link_weight"] = node_candidate["proxy_link_weight"]
                link_lower["node_proxy_u_link_max"] = node_candidate["u_link_proxy"]
                link_lower["node_load_weight"] = node_candidate["node_load_weight"]
                link_lower["layered2_node_risk_mode"] = node_candidate.get("risk_mode")
                link_lower["layered2_node_cvar_bound"] = node_candidate.get("cvar_bound")
                link_lower["layered2_node_cvar_bound_slack"] = node_candidate.get("cvar_bound_slack")
            if is_layered3:
                link_lower["network_affinity"] = node_candidate["network_affinity"]
                link_lower["network_affinity_model"] = network_affinity_model
                link_lower["node_load_weight"] = node_candidate["node_load_weight"]
                link_lower["layered3_candidate_budget"] = layered3_candidate_budget
                link_lower["link_optimization_mode"] = link_optimization_mode
                link_lower["link_cvar_tolerance"] = link_cvar_tolerance
                link_lower["layered3_node_u_tol"] = layered3_node_u_tol
                link_lower["layered3_node_cvar_tol"] = layered3_node_cvar_tol
                link_lower["layered3_affinity_tol"] = layered3_affinity_tol
                link_lower["node_candidate_source"] = node_candidate.get("candidate_source")
                link_lower["node_proxy_u_link_max"] = node_candidate.get("u_link_proxy")
            link_lower["link_cvar"] = link_lower["cvar"]
            link_lower["link_alpha"] = link_lower["alpha"]
            link_lower["layered_obj"] = float(node_candidate["node_obj"]) + float(link_lower["link_obj"])
            link_lower["selection_score"] = float(link_lower["link_obj"])
            link_lower["u_node_max"] = node_candidate["u_node_max"]

            candidate_row = {
                "candidate_id": candidate_id,
                "assignment_repr": node_candidate["assignment_repr"],
                "is_feasible": True,
                "node_feasible": True,
                "lower_status": link_lower["status"],
                "u_node_max": node_candidate["u_node_max"],
                "u_link_max": link_lower["u_link_max"],
                "lower_obj": link_lower["cvar"],
                "cvar": link_lower["cvar"],
                "alpha": link_lower["alpha"],
                "link_cvar": link_lower["link_cvar"],
                "link_alpha": link_lower["link_alpha"],
                "beta": effective_beta,
                "lambda_weight": args.lambda_weight,
                "node_risk_weight": node_risk_weight,
                "link_risk_weight": link_risk_weight,
                "node_loss_aggregation": node_loss_aggregation,
                "link_loss_aggregation": link_loss_aggregation,
                "link_optimization_mode": link_optimization_mode,
                "link_cvar_tolerance": link_cvar_tolerance,
                "link_cvar_bound": link_lower.get("link_cvar_bound"),
                "link_cvar_bound_slack": link_lower.get("link_cvar_bound_slack"),
                "link_first_stage_cvar": link_lower.get("link_first_stage_cvar"),
                "node_loss_model": node_loss_model,
                "node_obj": node_candidate["node_obj"],
                "node_cvar": node_candidate["node_cvar"],
                "node_alpha": node_candidate["node_alpha"],
                "layered2_node_risk_mode": node_candidate.get("risk_mode"),
                "layered2_node_cvar_bound": node_candidate.get("cvar_bound"),
                "layered2_node_cvar_bound_slack": node_candidate.get("cvar_bound_slack"),
                "node_candidate_index": node_candidate.get("candidate_index"),
                "node_candidate_source": node_candidate.get("candidate_source"),
                "node_proxy_split_rule": node_proxy_split_rule,
                "node_proxy_k_paths": node_proxy_k_paths,
                "node_proxy_k_paths_source": node_proxy_k_paths_source,
                "node_proxy_capacity_constraints": node_proxy_capacity_constraints,
                "node_proxy_link_weight": node_proxy_link_weight,
                "node_proxy_u_link_max": node_candidate["u_link_proxy"],
                "network_affinity": node_candidate["network_affinity"],
                "network_affinity_model": network_affinity_model,
                "node_load_weight": node_candidate["node_load_weight"],
                "link_obj": link_lower["link_obj"],
                "layered_obj": link_lower["layered_obj"],
                "selection_score": link_lower["selection_score"],
                "total_score": link_lower["selection_score"],
                "solve_time_sec": candidate_total_time,
                "note": f"{algorithm}_node_link",
            }
            candidate_rows.append(candidate_row)
            evaluated_candidates.append(
                {
                    "node": node_candidate,
                    "link": link_lower,
                    "row": candidate_row,
                    "selection_score": link_lower["selection_score"],
                }
            )

        if not evaluated_candidates:
            raise RuntimeError("Layered link model found no feasible solution for any node-layer candidate.")

        best_eval = min(evaluated_candidates, key=lambda item: (item["selection_score"], item["link"]["u_link_max"]))
        best_node = best_eval["node"]
        best_link = best_eval["link"]
        total_solve_time = float(node_result["solve_time_sec"]) + total_link_solve_time
        best_link["total_link_solve_time_sec"] = total_link_solve_time
        best_link["solve_time_sec"] = total_solve_time
        result = {
            "best_assignment": best_node["assignment"],
            "best_score": best_link["selection_score"],
            "u_node_max": best_node["u_node_max"],
            "u_link_max": best_link["u_link_max"],
            "best_lower": best_link,
            "candidate_rows": candidate_rows,
            "total_evaluated": len(evaluated_candidates),
            "search_mode": algorithm.replace("_solver", ""),
            "node_layer": node_result,
            "selected_node_layer": best_node,
            "link_layer": best_link,
            "L": best_link.get("L"),
            "X_path": best_link.get("X_path"),
            "Eta_node": best_link.get("Eta_node"),
        }
    # ===================== 8. 结果校验与写出：复算关键指标并保存配置、候选汇总、最优解和场景信息 =====================
    sanity_checks = run_sanity_checks(
        tasks=tasks,
        best_assignment=result["best_assignment"],
        lower_result={
            **result["best_lower"],
            "u_node_max": result["u_node_max"],
            "u_link_max": result["u_link_max"],
        },
        capacity=capacity,
        cpu_capacity=cpu_capacity,
        scenarios=scenarios,
        seed=args.seed,
    )

    config = {
        "topology": args.topology,
        "num_demand": args.num_demand,
        "demand_scale": args.demand_scale,
        "demand_downscale": args.demand_downscale,
        "rho": args.rho,
        "placement_policy": placement_policy,
        "allow_same_src_dst": allow_same_src_dst,
        "task_file": task_file,
        "paths": args.paths,
        "k_paths": args.k_paths,
        "edge_disjoint": args.edge_disjoint,
        "beta": effective_beta,
        "beta_mode": getattr(args, "beta_mode", "fixed"),
        "teavar_beta_margin": getattr(args, "teavar_beta_margin", None),
        "lambda_weight": args.lambda_weight,
        "scenario_mode": "cutoff",
        "scenario_prob_source": "weibull",
        "node_failures_enabled": enable_node_failures,
        "node_failure_source": "nodes.txt",
        "node_failure_probs": {str(k): float(v) for k, v in node_failure_probs.items()},
        "cutoff": args.cutoff,
        "weibull_shape": 0.8,
        "weibull_scale": args.weibull_scale,
        "seed": args.seed,
        "algorithm": output_algorithm_name,
        "internal_algorithm": algorithm,
        "requested_algorithm": requested_algorithm,
        "failure_agnostic": failure_agnostic_single_level,
        "risk_weight": risk_weight,
        "single_level_risk_mode": single_level_risk_mode,
        "single_level_cvar_bound": single_level_cvar_bound,
        "loss_aggregation": loss_aggregation,
        "node_risk_weight": node_risk_weight,
        "link_risk_weight": link_risk_weight,
        "node_loss_aggregation": node_loss_aggregation,
        "link_loss_aggregation": link_loss_aggregation,
        "link_optimization_mode": link_optimization_mode,
        "link_cvar_tolerance": link_cvar_tolerance,
        "link_cvar_bound": link_cvar_bound,
        "layered2_node_risk_mode": layered2_node_risk_mode,
        "layered2_node_cvar_bound": layered2_node_cvar_bound,
        "node_loss_model": node_loss_model,
        "node_proxy_split_rule": node_proxy_split_rule,
        "node_proxy_k_paths": node_proxy_k_paths,
        "node_proxy_k_paths_source": node_proxy_k_paths_source,
        "node_proxy_capacity_constraints": node_proxy_capacity_constraints,
        "node_proxy_link_weight": node_proxy_link_weight,
        "network_affinity_model": network_affinity_model,
        "layered3_candidate_budget": layered3_candidate_budget,
        "layered3_node_u_tol": layered3_node_u_tol,
        "layered3_node_cvar_tol": layered3_node_cvar_tol,
        "layered3_affinity_tol": layered3_affinity_tol,
        "enable_mip_gap": args.enable_mip_gap,
        "mip_gap": args.mip_gap if args.enable_mip_gap else None,
        "time_limit_sec": getattr(args, "time_limit_sec", None),
        "max_tasks": args.max_tasks,
        "computed_paths_k": k_used,
        "num_tasks": len(tasks),
        "num_edges": len(edges),
        "num_scenarios": len(scenarios),
        "output_dir": output_dir,
    }

    write_phase1_results(
        output_dir=output_dir,
        config=config,
        tasks=tasks,
        best_result=result,
        candidate_rows=result["candidate_rows"],
        edges=edges,
        capacity=capacity,
        T=T,
        scenarios=scenarios,
        scenario_probs=scenario_probs,
        node_rows=node_rows,
        sanity_checks=sanity_checks,
        node_scenarios=node_scenarios,
    )
    result["scenario_probs"] = scenario_probs
    result["scenario_count"] = len(scenarios)
    result["effective_beta"] = effective_beta
    result["availability_context"] = {
        "tasks": tasks,
        "edges": edges,
        "capacity": capacity,
        "T": T,
        "scenarios": scenarios,
        "node_scenarios": node_scenarios,
    }

    if not bool(getattr(args, "quiet", False)):
        print("")
        print("========== Phase-1 Summary ==========")
        print(f"Output dir      : {Path(output_dir).resolve()}")
        print(f"Tasks           : {len(tasks)}")
        print(f"Scenarios       : {len(scenarios)}")
        print(f"Candidates eval : {result['total_evaluated']}")
        print(f"Algorithm       : {algorithm}")
        print(f"Best score      : {result['best_score']:.6f}")
        print(f"u_node_max      : {result['u_node_max']:.6f}")
        print(f"reserved_u_link : {result['reserved_u_link_max']:.6f}")
        print(f"actual_u_link   : {result['evaluation_metrics']['actual_normal_u_link_max']:.6f}")
        print(f"availability    : {result['evaluation_metrics']['availability']:.6f}")
        print(f"expected_fail_u : {result['evaluation_metrics']['expected_fail_u_link_max']:.6f}")
        print(f"worst_fail_u    : {result['evaluation_metrics']['worst_fail_u_link_max']:.6f}")
        print(f"model_CVaR      : {result['best_lower']['cvar']:.6f}")
        print(f"model_alpha     : {result['best_lower']['alpha']:.6f}")
        print("====================================")
        print("")

    return output_dir, result
