from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


# ===================== 1. 路径-链路矩阵工具：把路径集合转成容量约束可直接使用的系数 =====================


def _find_reverse_edge_index(edges: Sequence[Tuple[int, int]], edge_idx_1_based: int):
    """查找某条有向边的反向边编号，返回从 1 开始的边编号。

    项目中的路径以有向边编号表示，但故障场景中一条物理链路故障通常会同时影响两个方向。
    因此在判断路径是否可用时，需要同时检查边 e 与它的反向边是否出现在路径中。
    """
    u, v = edges[edge_idx_1_based - 1]
    reverse_edge = (v, u)
    for idx, edge in enumerate(edges, start=1):
        if edge == reverse_edge:
            return idx
    return None


def build_tunnel_edge_matrix(T: Sequence[Sequence[int]], nedges: int):
    """构建路径-链路关联矩阵 L，L[t, e] 表示路径 t 是否经过链路 e。

    该矩阵用于把路径流量变量转换为链路负载：
        load_e = sum_t x_t * L[t, e]
    单层模型和链路层模型都会用它生成链路容量约束与链路利用率统计。
    """
    ntunnels = len(T)
    L = np.zeros((ntunnels, nedges), dtype=float)
    for tunnel_idx, tunnel_edges in enumerate(T):
        for edge_id in tunnel_edges:
            if 1 <= edge_id <= nedges:
                L[tunnel_idx, edge_id - 1] = 1.0
    return L


# ===================== 2. 故障场景矩阵工具：预计算路径/节点在每个场景下是否可用 =====================


def build_tunnel_scenario_matrix(
    T: Sequence[Sequence[int]],
    edges: Sequence[Tuple[int, int]],
    scenarios: Sequence[Sequence[float]],
):
    """构建场景-路径可用矩阵 X，链路或反向链路故障时路径不可用。

    X[s, t] = 1 表示场景 s 下路径 t 可用，否则为 0。链路层损失约束会用它计算
    故障场景下每条路径实际能交付的流量。这里提前生成矩阵，可以避免在优化模型中
    反复写复杂的路径可用性判断。
    """
    nedges = len(edges)
    ntunnels = len(T)
    nscenarios = len(scenarios)
    X = np.ones((nscenarios, ntunnels), dtype=float)
    for scenario_idx in range(nscenarios):
        for tunnel_idx in range(ntunnels):
            if len(T[tunnel_idx]) == 0:
                X[scenario_idx, tunnel_idx] = 0.0
                continue
            for edge_idx_0_based in range(nedges):
                if scenarios[scenario_idx][edge_idx_0_based] != 0:
                    continue
                edge_id = edge_idx_0_based + 1
                reverse_edge_id = _find_reverse_edge_index(edges, edge_id)
                if edge_id in T[tunnel_idx] or (
                    reverse_edge_id is not None and reverse_edge_id in T[tunnel_idx]
                ):
                    X[scenario_idx, tunnel_idx] = 0.0
                    break
    return X


def build_node_scenario_matrix(
    compute_nodes: Sequence[int],
    node_scenarios: Sequence[Dict[int, float]] | None,
):
    """构建场景-计算节点可用矩阵 Eta。

    Eta[s, m] = 1 表示场景 s 下计算节点 m 可用，否则为 0。节点层损失模型和
    链路层端到端交付量都会使用该矩阵，把执行节点故障纳入任务损失计算。
    """
    if node_scenarios is None:
        return np.ones((0, len(compute_nodes)), dtype=float)

    Eta = np.ones((len(node_scenarios), len(compute_nodes)), dtype=float)
    for scenario_idx, scenario in enumerate(node_scenarios):
        for node_idx, node_id in enumerate(compute_nodes):
            Eta[scenario_idx, node_idx] = float(scenario.get(int(node_id), 1.0))
    return Eta
