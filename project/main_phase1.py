from __future__ import annotations

import argparse
from argparse import Namespace

from phase1_correctness import run_phase1_correctness


def read_input(message, default, typ):
    # ===================== 交互输入工具：空输入沿用默认值，非空输入按指定类型转换 =====================
    raw = input(message)
    if len(raw.strip()) == 0:
        return default
    if typ is str:
        return raw.strip()
    if typ is bool:
        val = raw.strip().lower()
        if val in {"true", "1", "yes", "y"}:
            return True
        if val in {"false", "0", "no", "n"}:
            return False
        raise ValueError(f"Invalid boolean input for '{message.strip()}': {raw}")
    try:
        return typ(raw)
    except Exception as exc:
        raise ValueError(f"Invalid input for '{message.strip()}': {raw}") from exc


def parse_commandline():
    # ===================== 命令行参数：允许直接指定实验类型，也允许进入交互式输入 =====================
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Type of experiment to run. Only supports: phase1_correctness",
    )
    return parser.parse_args()


def main():
    parsed = parse_commandline()
    print("Asking for inputs. Press enter for default values\n")

    # ===================== 1. 选择实验：主入口默认运行 phase1_correctness，可转到故障消融实验 =====================
    experiment = (
        parsed.experiment
        if parsed.experiment is not None
        else read_input("Experiment (phase1_correctness): ", "phase1_correctness", str)
    )
    if experiment != "phase1_correctness":
        if experiment == "phase1_fault_ablation":
            from phase1_fault_ablation import main as fault_ablation_main

            fault_ablation_main()
            return
        raise ValueError(
            f"Unsupported experiment: {experiment}. This entrypoint supports phase1_correctness or phase1_fault_ablation."
        )

    print(f"\nRunning experiment [{experiment}]...")

    # ===================== 2. 设置通用实验参数：拓扑、需求规模、任务数量和候选路径设置 =====================
    # 这些参数会共同决定任务集合、候选计算节点和链路层可用路径，是所有算法共用的输入。
    topology = read_input("Topology (B4): ", "B4", str)
    num_demand = read_input("Num demand row (1): ", 1, int)
    demand_scale = read_input("Demand scale (1.0): ", 1.0, float)
    demand_downscale = read_input("Demand downscale (1.0): ", 1.0, float)
    max_tasks = read_input("Max tasks (144): ", 144, int)

    placement_policy = read_input("Placement policy all/exclude_endpoints (exclude_endpoints): ", "exclude_endpoints", str)
    if placement_policy not in {"all", "exclude_endpoints"}:
        raise ValueError(f"Unsupported placement_policy: {placement_policy}. Use all or exclude_endpoints.")
    
    if placement_policy == "exclude_endpoints":
        allow_same_src_dst = read_input("Allow same source and destination tasks? (False): ", False, bool)
    else:
        allow_same_src_dst = False

    rho = read_input("Rho for b_out=rho*b_in (0.5): ", 0.5, float)
    paths = read_input("Paths mode (KSP): ", "KSP", str)
    k_paths = read_input("K paths for KSP mode (6): ", 6, int)
    # 候选路径是否要求边不相交。默认允许共享链路。
    edge_disjoint = read_input("Edge disjoint for KSP? (false): ", False, bool)
    beta = read_input("Beta (0.99): ", 0.99, float)
    lambda_weight = read_input("Lambda weight (0.5): ", 0.5, float)

    # 故障概率模型参数：Weibull 分布的 scale 参数和场景采样的概率截断阈值。Weibull 分布的 shape 参数固定为 0.8。
    cutoff = read_input("Scenario cutoff (1e-4): ", 1e-4, float)
    weibull_scale = read_input("Weibull scale (0.001): ", 0.001, float)
    seed = read_input("Random seed (1): ", 1, int)

    # ===================== 3. 选择算法：单层算法 A 或分层算法 layered1/layered2/layered3 =====================
    algorithm = read_input("Algorithm A/layered1/layered2/layered3 (layered3): ", "layered3", str)
    if algorithm not in {"A", "layered", "layered1", "layered2", "layered3"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use A, layered1, layered2, or layered3.")
    
    if algorithm == "A":
        # ===================== 6A. 单层算法 A 参数：选择风险目标形式、损失聚合方式和可选 MIPGap =====================
        # weighted 保持原模型：CVaR 与资源利用率加权求和；cvar_constraint 使用 Gamma 作为 CVaR SLA 上界。
        single_level_risk_mode = read_input(
            "Single-level risk mode weighted/cvar_constraint (weighted): ",
            "weighted",
            str,
        )
        if single_level_risk_mode not in {"weighted", "cvar_constraint"}:
            raise ValueError("Single-level risk mode must be weighted or cvar_constraint.")
        if single_level_risk_mode == "weighted":
            risk_weight = read_input("Risk weight for Algorithm A CVaR term (0.5): ", 0.5, float)
            single_level_cvar_bound = None
        else:
            risk_weight = 0.0
            single_level_cvar_bound = read_input("Single-level CVaR bound Gamma (0.05): ", 0.05, float)
            if single_level_cvar_bound < 0:
                raise ValueError("Single-level CVaR bound Gamma must be >= 0.")
        loss_aggregation = read_input("Algorithm A loss aggregation max/average (average): ", "average", str)
        if loss_aggregation not in {"max", "average"}:
            raise ValueError(f"Unsupported loss_aggregation: {loss_aggregation}. Use max or average.")
        enable_mip_gap = read_input("Enable MIPGap for Algorithm A? (True): ", True, bool)
        if enable_mip_gap:
            mip_gap = read_input("MIPGap for Algorithm A (0.01): ", 0.01, float)
        else:
            mip_gap = 0.0
        node_risk_weight = None
        link_risk_weight = None
        node_loss_aggregation = None
        link_loss_aggregation = None
        link_optimization_mode = None
        link_cvar_tolerance = None
        link_cvar_bound = None
        layered2_node_risk_mode = None
        layered2_node_cvar_bound = None
        node_proxy_link_weight = None
        layered3_candidate_budget = None
        layered3_node_u_tol = None
        layered3_node_cvar_tol = None
        layered3_affinity_tol = None
    elif algorithm in {"layered", "layered1", "layered2", "layered3"}:
        # ===================== 6B. 分层算法公共参数：分别设置节点层和链路层的风险权重与损失聚合方式 =====================
        risk_weight = None
        single_level_risk_mode = None
        single_level_cvar_bound = None
        loss_aggregation = None
        node_risk_weight = read_input("Layered node-layer risk weight (0.5): ", 0.5, float)
        link_risk_weight = read_input("Layered link-layer risk weight (0.5): ", 0.5, float)
        node_loss_aggregation = read_input("Layered node-layer loss aggregation max/average (average): ", "average", str)
        link_loss_aggregation = read_input("Layered link-layer loss aggregation max/average (max): ", "max", str)
        if algorithm == "layered3":
            link_optimization_mode = "weighted"
        else:
            link_optimization_mode = read_input(
                "Layered link optimization weighted/cvar_then_u/cvar_constraint (weighted): ",
                "weighted",
                str,
            )
        if link_optimization_mode not in {"weighted", "cvar_then_u", "cvar_constraint"}:
            raise ValueError("Link optimization mode must be weighted, cvar_then_u, or cvar_constraint.")
        link_cvar_tolerance = 0.0
        link_cvar_bound = None
        if link_optimization_mode == "cvar_then_u":
            link_cvar_tolerance = read_input(
                "Link CVaR tolerance inside link solver (0.0): ",
                0.0,
                float,
            )
        if link_optimization_mode == "cvar_constraint":
            link_cvar_bound = read_input("Layered link CVaR bound Gamma (0.05): ", 0.05, float)
            if link_cvar_bound < 0:
                raise ValueError("Layered link CVaR bound Gamma must be >= 0.")
        if algorithm == "layered2":
            layered2_node_risk_mode = read_input(
                "Layered2 node risk mode weighted/cvar_constraint (weighted): ",
                "weighted",
                str,
            )
            if layered2_node_risk_mode not in {"weighted", "cvar_constraint"}:
                raise ValueError("Layered2 node risk mode must be weighted or cvar_constraint.")
            if layered2_node_risk_mode == "cvar_constraint":
                layered2_node_cvar_bound = read_input("Layered2 node proxy CVaR bound Gamma (0.05): ", 0.05, float)
                if layered2_node_cvar_bound < 0:
                    raise ValueError("Layered2 node proxy CVaR bound Gamma must be >= 0.")
                node_proxy_link_weight = read_input("Layered2 node resource link weight (0.5): ", 0.5, float)
                if node_proxy_link_weight > 1.0 + 1e-9:
                    raise ValueError("For layered2 cvar_constraint, node resource link weight must be <= 1.")
            else:
                layered2_node_cvar_bound = None
                node_proxy_link_weight = read_input("Layered2 proxy link weight (0.1): ", 0.1, float)
                if node_risk_weight + node_proxy_link_weight > 1.0 + 1e-9:
                    raise ValueError("For layered2, node_risk_weight + node_proxy_link_weight must be <= 1.")
        if algorithm != "layered2":
            node_proxy_link_weight = None
            layered2_node_risk_mode = None
            layered2_node_cvar_bound = None
        if algorithm == "layered3":
            # layered3 固定使用网络亲和度邻域候选，链路层按 link_cvar 与 U_link 的加权和选择。
            layered3_candidate_budget = read_input("Layered3 candidate budget (8): ", 8, int)
            layered3_node_u_tol = read_input("Layered3 node U tolerance (0.0): ", 0.0, float)
            layered3_node_cvar_tol = read_input("Layered3 node CVaR tolerance (1e-4): ", 1e-4, float)
            layered3_affinity_tol = read_input("Layered3 affinity tolerance (-1 disables): ", -1.0, float)
            if layered3_affinity_tol < 0:
                layered3_affinity_tol = None
        else:
            layered3_candidate_budget = None
            layered3_node_u_tol = None
            layered3_node_cvar_tol = None
            layered3_affinity_tol = None
        if node_loss_aggregation not in {"max", "average"}:
            raise ValueError(
                f"Unsupported node_loss_aggregation: {node_loss_aggregation}. Use max or average."
            )
        if link_loss_aggregation not in {"max", "average"}:
            raise ValueError(
                f"Unsupported link_loss_aggregation: {link_loss_aggregation}. Use max or average."
            )
        if link_cvar_tolerance < 0:
            raise ValueError("Link CVaR tolerance inside link solver must be >= 0.")
        enable_mip_gap = False
        mip_gap = 0.0
    args = Namespace(
        # ===================== 7. 组装实验参数：把交互输入统一打包给 phase1_correctness 入口 =====================
        topology=topology,
        num_demand=num_demand,
        demand_scale=demand_scale,
        demand_downscale=demand_downscale,
        max_tasks=max_tasks,
        placement_policy=placement_policy,
        allow_same_src_dst=allow_same_src_dst,
        rho=rho,
        paths=paths,
        k_paths=k_paths,
        edge_disjoint=edge_disjoint,
        beta=beta,
        lambda_weight=lambda_weight,
        cutoff=cutoff,
        weibull_scale=weibull_scale,
        seed=seed,
        algorithm=algorithm,
        risk_weight=risk_weight,
        single_level_risk_mode=single_level_risk_mode,
        single_level_cvar_bound=single_level_cvar_bound,
        loss_aggregation=loss_aggregation,
        node_risk_weight=node_risk_weight,
        link_risk_weight=link_risk_weight,
        node_loss_aggregation=node_loss_aggregation,
        link_loss_aggregation=link_loss_aggregation,
        link_optimization_mode=link_optimization_mode,
        link_cvar_tolerance=link_cvar_tolerance,
        link_cvar_bound=link_cvar_bound,
        layered2_node_risk_mode=layered2_node_risk_mode,
        layered2_node_cvar_bound=layered2_node_cvar_bound,
        node_proxy_link_weight=node_proxy_link_weight,
        layered3_candidate_budget=layered3_candidate_budget,
        layered3_node_u_tol=layered3_node_u_tol,
        layered3_node_cvar_tol=layered3_node_cvar_tol,
        layered3_affinity_tol=layered3_affinity_tol,
        enable_mip_gap=enable_mip_gap,
        mip_gap=mip_gap,
    )
    run_phase1_correctness(args)


if __name__ == "__main__":
    main()
