from __future__ import annotations

from Algorithms.node_layer_solver import (
    build_average_split_proxy_link_load,
    build_average_split_proxy_loss,
    solve_node_layer,
)


def prepare_layered2_node_data(
    *,
    tasks,
    compute_nodes,
    scenarios,
    scenario_probs,
    node_scenarios,
    pair_to_tunnels,
    T,
    edges,
    k_paths: int,
    nedges: int,
):
    """layered2 节点层预处理：构造平均分流代理损失和代理链路负载。

    layered2 的核心区别是让节点层粗略感知链路故障与链路容量，但仍不引入真实路由变量 x。
    这里固定使用与链路层一致的 k_paths，在这些候选路径上平均分流，得到线性的代理系数。
    """
    proxy_loss = build_average_split_proxy_loss(
        tasks=tasks,
        compute_nodes=compute_nodes,
        scenarios=scenarios,
        node_scenarios=node_scenarios,
        pair_to_tunnels=pair_to_tunnels,
        T=T,
        edges=edges,
        k_paths=k_paths,
    )
    proxy_link_load = build_average_split_proxy_link_load(
        tasks=tasks,
        pair_to_tunnels=pair_to_tunnels,
        T=T,
        nedges=nedges,
        k_paths=k_paths,
    )
    return {
        "node_scenarios": node_scenarios,
        "node_scenario_probs": scenario_probs,
        "proxy_loss": proxy_loss,
        "proxy_link_load": proxy_link_load,
        "node_loss_model": "average_split_proxy",
        "node_proxy_split_rule": "average",
        "node_proxy_k_paths": int(k_paths),
        "node_proxy_k_paths_source": "k_paths",
        "node_proxy_capacity_constraints": True,
    }


def solve_layered2_node_layer(
    *,
    tasks,
    compute_nodes,
    cpu_capacity,
    pair_to_tunnels,
    beta: float,
    node_risk_weight: float,
    node_loss_aggregation: str,
    seed: int,
    node_data,
    link_capacity,
    node_proxy_link_weight: float,
    node_risk_mode: str = "weighted",
    node_cvar_bound: float | None = None,
):
    """layered2 节点层：在节点故障模型上加入平均分流代理链路项。

    节点层目标包含节点层 CVaR、最大节点利用率和 U_link_proxy。U_link_proxy 不是真实路由结果，
    只是平均分流规则下的最大代理链路利用率；真实链路损失仍由后续链路层重新优化得到。
    """
    return solve_node_layer(
        tasks=tasks,
        compute_nodes=compute_nodes,
        cpu_capacity=cpu_capacity,
        node_scenarios=node_data["node_scenarios"],
        node_scenario_probs=node_data["node_scenario_probs"],
        pair_to_tunnels=pair_to_tunnels,
        beta=beta,
        risk_weight=node_risk_weight,
        loss_aggregation=node_loss_aggregation,
        seed=seed,
        proxy_loss=node_data["proxy_loss"],
        proxy_link_load=node_data["proxy_link_load"],
        link_capacity=link_capacity,
        proxy_link_cap_enforced=True,
        proxy_link_weight=node_proxy_link_weight,
        network_affinity=None,
        node_loss_model=node_data["node_loss_model"],
        risk_mode=node_risk_mode,
        cvar_bound=node_cvar_bound,
        candidate_budget=1,
    )
