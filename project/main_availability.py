from __future__ import annotations

from availability_experiment import default_common_args, parse_demand_rows, run_availability_experiment


def read_input(message, default, typ):
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


def parse_algorithms(raw: str):
    algorithms = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"A", "A_no_failure", "A_agnostic", "layered3"}
    if not algorithms:
        raise ValueError("At least one algorithm is required.")
    unknown = [item for item in algorithms if item not in allowed]
    if unknown:
        raise ValueError(f"Unsupported algorithms: {unknown}. Use A, A_no_failure, and/or layered3.")
    return algorithms


def main():
    print("Asking for availability experiment inputs. Press enter for default values\n")
    algorithms = parse_algorithms(
        read_input(
            "Algorithms comma-separated A,A_no_failure,layered3 (A,A_no_failure,layered3): ",
            "A,A_no_failure,layered3",
            str,
        )
    )
    demand_rows = parse_demand_rows(read_input("Demand rows comma-separated (1): ", "1", str))
    failure_scope = read_input("Failure scope joint/link_only (joint): ", "joint", str)
    if failure_scope not in {"joint", "link_only"}:
        raise ValueError("Failure scope must be joint or link_only.")
    availability_eval_mode = read_input(
        "Availability eval mode teavar_reallocation/fixed_x (teavar_reallocation): ",
        "teavar_reallocation",
        str,
    )
    if availability_eval_mode not in {"teavar_reallocation", "fixed_x"}:
        raise ValueError("Availability eval mode must be teavar_reallocation or fixed_x.")

    single_level_risk_mode = "weighted"
    single_level_cvar_bound = None
    risk_weight = 0.5
    if "A" in algorithms:
        single_level_risk_mode = read_input(
            "Single-level risk mode weighted/cvar_constraint (weighted): ",
            "weighted",
            str,
        )
        if single_level_risk_mode not in {"weighted", "cvar_constraint"}:
            raise ValueError("Single-level risk mode must be weighted or cvar_constraint.")
        if single_level_risk_mode == "cvar_constraint":
            single_level_cvar_bound = read_input("Single-level CVaR bound Gamma (0.05): ", 0.05, float)
            if single_level_cvar_bound < 0:
                raise ValueError("Single-level CVaR bound Gamma must be >= 0.")
            risk_weight = 0.0
        else:
            risk_weight = read_input("Risk weight for Algorithm A CVaR term (0.5): ", 0.5, float)

    layered3_candidate_budget = 8
    layered3_node_u_tol = 0.0
    layered3_node_cvar_tol = 1e-4
    layered3_affinity_tol = None
    if "layered3" in algorithms:
        layered3_candidate_budget = read_input("Layered3 candidate budget (8): ", 8, int)
        layered3_node_u_tol = read_input("Layered3 node U tolerance (0.0): ", 0.0, float)
        layered3_node_cvar_tol = read_input("Layered3 node CVaR tolerance (1e-4): ", 1e-4, float)
        layered3_affinity_tol = read_input("Layered3 affinity tolerance (-1 disables): ", -1.0, float)
        if layered3_affinity_tol < 0:
            layered3_affinity_tol = None
        if layered3_candidate_budget < 1:
            raise ValueError("Layered3 candidate budget must be >= 1.")
        if layered3_node_u_tol < 0:
            raise ValueError("Layered3 node U tolerance must be >= 0.")
        if layered3_node_cvar_tol < 0:
            raise ValueError("Layered3 node CVaR tolerance must be >= 0.")

    common = default_common_args(
        topology=read_input("Topology (B4): ", "B4", str),
        algorithms=algorithms,
        demand_rows=demand_rows,
        failure_iterations=read_input("Failure scenario samples (1): ", 1, int),
        scale_start=read_input("Scale start (1.0): ", 1.0, float),
        scale_step=read_input("Scale step (0.2): ", 0.2, float),
        scale_finish=read_input("Scale finish (2.0): ", 2.0, float),
        demand_downscale=read_input("Demand downscale base (2.0): ", 2.0, float),
        max_tasks=read_input("Max tasks (144): ", 144, int),
        placement_policy=read_input("Placement policy all/exclude_endpoints (exclude_endpoints): ", "exclude_endpoints", str),
        allow_same_src_dst=read_input("Allow same source and destination tasks? (False): ", False, bool),
        rho=read_input("Rho for b_out=rho*b_in (0.5): ", 0.5, float),
        paths=read_input("Paths mode (KSP): ", "KSP", str),
        k_paths=read_input("K paths for KSP mode (6): ", 6, int),
        edge_disjoint=read_input("Edge disjoint for KSP? (false): ", False, bool),
        teavar_beta_margin=read_input("TEAVAR beta margin (0.01): ", 0.01, float),
        lambda_weight=read_input("Single-level lambda weight (0.5): ", 0.5, float),
        cutoff=read_input("Scenario cutoff (1e-4): ", 1e-4, float),
        weibull_scale=read_input("Weibull scale (0.001): ", 0.001, float),
        seed=read_input("Base random seed (1): ", 1, int),
        failure_scope=failure_scope,
        availability_eval_mode=availability_eval_mode,
        availability_target=read_input("Availability target loss (0.0): ", 0.0, float),
        risk_weight=risk_weight,
        single_level_risk_mode=single_level_risk_mode,
        single_level_cvar_bound=single_level_cvar_bound,
        loss_aggregation=read_input("Algorithm A loss aggregation max/average (max): ", "max", str),
        node_risk_weight=read_input("Layered node-layer risk weight (0.5): ", 0.5, float),
        link_risk_weight=read_input("Layered link-layer risk weight (0.5): ", 0.5, float),
        node_loss_aggregation=read_input("Layered node-layer loss aggregation max/average (average): ", "average", str),
        link_loss_aggregation=read_input("Layered link-layer loss aggregation max/average (max): ", "max", str),
        link_optimization_mode="weighted",
        link_cvar_tolerance=0.0,
        layered3_candidate_budget=layered3_candidate_budget,
        layered3_node_u_tol=layered3_node_u_tol,
        layered3_node_cvar_tol=layered3_node_cvar_tol,
        layered3_affinity_tol=layered3_affinity_tol,
        enable_mip_gap=read_input("Enable MIPGap for Algorithm A? (True): ", True, bool),
        mip_gap=read_input("MIPGap for Algorithm A (0.01): ", 0.01, float),
        a_time_limit_sec=read_input("Time limit seconds for Algorithm A, 0 disables (600): ", 600.0, float),
    )
    run_availability_experiment(common)


if __name__ == "__main__":
    main()
