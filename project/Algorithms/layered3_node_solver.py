from __future__ import annotations

from Algorithms.node_layer_solver import build_network_affinity_cost, solve_node_layer


def prepare_layered3_node_data(
    *,
    tasks,
    pair_to_tunnels,
    T,
    k_paths: int,
):
    """layered3 节点层预处理：只构造网络亲和度，用于生成邻域候选。"""
    network_affinity = build_network_affinity_cost(
        tasks=tasks,
        pair_to_tunnels=pair_to_tunnels,
        T=T,
        k_paths=k_paths,
    )
    return {
        "network_affinity": network_affinity,
        "network_affinity_model": "path_count_overlap_length",
        "node_loss_model": "node_failure",
        "node_proxy_split_rule": None,
        "node_proxy_k_paths": None,
        "node_proxy_k_paths_source": None,
        "node_proxy_capacity_constraints": False,
    }


def solve_layered3_node_layer(
    *,
    tasks,
    compute_nodes,
    cpu_capacity,
    node_scenarios,
    node_scenario_probs,
    pair_to_tunnels,
    beta: float,
    node_risk_weight: float,
    node_loss_aggregation: str,
    seed: int,
    node_data,
    candidate_budget: int,
    node_u_tol: float,
    node_cvar_tol: float,
    affinity_tol,
):
    """layered3 节点层：生成 node_failure + U_node 约束下的 network_affinity 邻域候选。"""
    return solve_node_layer(
        tasks=tasks,
        compute_nodes=compute_nodes,
        cpu_capacity=cpu_capacity,
        node_scenarios=node_scenarios,
        node_scenario_probs=node_scenario_probs,
        pair_to_tunnels=pair_to_tunnels,
        beta=beta,
        risk_weight=node_risk_weight,
        loss_aggregation=node_loss_aggregation,
        seed=seed,
        proxy_loss=None,
        proxy_link_load=None,
        link_capacity=None,
        proxy_link_cap_enforced=False,
        proxy_link_weight=0.0,
        network_affinity=node_data["network_affinity"],
        node_loss_model=node_data["node_loss_model"],
        candidate_budget=candidate_budget,
        node_u_tol=node_u_tol,
        node_cvar_tol=node_cvar_tol,
        affinity_tol=affinity_tol,
    )
