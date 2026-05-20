from __future__ import annotations

from Algorithms.node_layer_solver import solve_node_layer


def solve_layered1_node_layer(
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
):
    """layered1 节点层：只根据计算节点故障和节点容量生成一个放置方案。

    layered1 是最基础的双层模型。节点层不看链路故障、不构造代理链路负载，也不生成候选池；
    它只优化节点层 CVaR 与最大节点利用率。得到固定放置后，再交给链路层求真实路由。
    """
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
        network_affinity=None,
        node_loss_model="node_failure",
        candidate_budget=1,
    )
