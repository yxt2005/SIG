from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _find_reverse_edge_index(edges: Sequence[Tuple[int, int]], edge_idx_1_based: int):
    u, v = edges[edge_idx_1_based - 1]
    rev = (v, u)
    for i, e in enumerate(edges, start=1):
        if e == rev:
            return i
    return None


def calculate_loss_reallocation(
    edges: Sequence[Tuple[int, int]],
    capacity: Sequence[float],
    demand: Sequence[float],
    flows: Sequence[Tuple[int, int]],
    T: Sequence[Sequence[int]],
    Tf: Sequence[Sequence[int]],
    k: int,
    splittingratios,
    scenarios: Sequence[Sequence[float]],
    probabilities: Sequence[float],
    progress: bool = False,
):
    nedges = len(edges)
    nflows = len(flows)
    ntunnels = len(T)
    nscenarios = len(scenarios)

    X = np.ones((nscenarios, ntunnels), dtype=float)
    for s in range(nscenarios):
        for t in range(ntunnels):
            if len(T[t]) == 0:
                X[s, t] = 0
            else:
                for e in range(nedges):
                    if scenarios[s][e] == 0:
                        e_idx = e + 1
                        back_edge = _find_reverse_edge_index(edges, e_idx)
                        if e_idx in T[t] or (back_edge is not None and back_edge in T[t]):
                            X[s, t] = 0

    L = np.zeros((ntunnels, nedges), dtype=float)
    for t in range(ntunnels):
        for e in range(nedges):
            if (e + 1) in T[t]:
                L[t, e] = 1

    splittingratios = np.array(splittingratios, dtype=float, copy=True)
    routed = np.zeros((nscenarios, nflows, k), dtype=float)
    t_total = np.zeros(nscenarios, dtype=float)

    for s in range(nscenarios):
        for f in range(nflows):
            totalup = 0.0
            for t in range(len(Tf[f])):
                totalup += splittingratios[f, t] * X[s, Tf[f][t] - 1]
            if totalup == 0:
                splittingratios[f, :] = splittingratios[f, :] + 0.2
                for t in range(len(Tf[f])):
                    totalup += splittingratios[f, t] * X[s, Tf[f][t] - 1]

            for t in range(len(Tf[f])):
                if totalup != 0:
                    routed[s, f, t] = splittingratios[f, t] / totalup * demand[f] * X[s, Tf[f][t] - 1]

        t_total[s] = np.sum(routed[s, :, :])

        congestion_loss = 0.0
        for e in range(nedges):
            edge_utilization = 0.0
            for f in range(nflows):
                edge_utilization += sum(
                    routed[s, f, t] * L[Tf[f][t] - 1, e] * X[s, Tf[f][t] - 1]
                    for t in range(len(Tf[f]))
                )
            congestion_loss += max(0.0, round((edge_utilization - capacity[e]) * 1000) / 1000)
        t_total[s] -= congestion_loss

    demand_sum = float(np.sum(demand))
    umax = [round((1 - x / demand_sum) * 100000) / 100000 for x in t_total]
    return umax


def pdf(losses, probabilities=None, sla=0):
    if probabilities is None:
        count = 0
        for loss in losses:
            if (1 - loss) >= sla:
                count += 1
        return count / len(losses)

    pairs = sorted(zip(losses, probabilities), key=lambda x: x[0])
    total = 0.0
    for loss, prob in pairs:
        if loss > sla:
            break
        total += prob
    return total


def var_uniform(losses, beta):
    usorted = sorted(losses)
    probabilities = [1 / len(losses)] * len(losses)
    total = 0.0
    loss = 0.0
    for u, p in zip(usorted, probabilities):
        total += p
        loss = u
        if total >= beta:
            break
    return loss


def var(losses, probabilities, beta):
    pairs = sorted(zip(losses, probabilities), key=lambda x: x[0])
    total = 0.0
    loss = 0.0
    for u, p in pairs:
        total += p
        loss = u
        if total >= beta:
            break
    return loss


def cvar(losses, probabilities, beta):
    pairs = sorted(zip(losses, probabilities), key=lambda x: x[0])
    total = 0.0
    loss = 0.0
    prob_total = 0.0
    for u, p in pairs:
        total += p
        if total >= beta:
            prob_total += p
            loss += u * p
    return loss / prob_total if prob_total > 0 else 0.0


def simulate_utilization_no_failures(edges, capacity, demand, flows, T, Tf, k, a, bandwidth_allowed):
    nedges = len(edges)
    nflows = len(flows)
    ntunnels = len(T)

    L = np.zeros((ntunnels, nedges), dtype=float)
    for t in range(ntunnels):
        for e in range(nedges):
            if (e + 1) in T[t]:
                L[t, e] = 1

    routed = np.zeros((nflows, k), dtype=float)
    a = np.array(a, dtype=float, copy=True)
    for f in range(nflows):
        totalup = 0.0
        for t in range(len(Tf[f])):
            totalup += a[f, t]
        if totalup == 0:
            a[f, :] = a[f, :] + 0.2
            for t in range(len(Tf[f])):
                totalup += a[f, t]
        for t in range(len(Tf[f])):
            routed[f, t] = a[f, t] / totalup * demand[f]

    edge_utilization = np.zeros(nedges, dtype=float)
    for e in range(nedges):
        use = 0.0
        for f in range(nflows):
            use += sum(routed[f, t] * L[Tf[f][t] - 1, e] for t in range(len(Tf[f])))
        edge_utilization[e] = use / capacity[e]
    return edge_utilization
