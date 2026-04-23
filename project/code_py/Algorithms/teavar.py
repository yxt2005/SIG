from __future__ import annotations

from typing import Sequence, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from util import print_results


def _find_reverse_edge_index(edges: Sequence[Tuple[int, int]], edge_idx_1_based: int):
    u, v = edges[edge_idx_1_based - 1]
    rev = (v, u)
    for i, e in enumerate(edges, start=1):
        if e == rev:
            return i
    return None


def teavar(
    env,
    edges,
    capacity,
    flows,
    demand,
    beta,
    k,
    T,
    Tf,
    scenarios,
    scenario_probs,
    explain=False,
    verbose=False,
    utilization=False,
    average=False,
):
    nedges = len(edges)
    nflows = len(flows)
    ntunnels = len(T)
    nscenarios = len(scenarios)
    p = scenario_probs

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

    model = gp.Model(env=env) if env is not None else gp.Model()
    model.Params.OutputFlag = 0

    a = model.addVars(nflows, k, lb=0.0, vtype=GRB.CONTINUOUS, name="a")
    alpha = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="alpha")
    umax = model.addVars(nscenarios, lb=0.0, vtype=GRB.CONTINUOUS, name="umax")
    u = model.addVars(nscenarios, nflows, lb=0.0, vtype=GRB.CONTINUOUS, name="u")

    for e in range(nedges):
        model.addConstr(
            gp.quicksum(
                a[f, t] * L[Tf[f][t] - 1, e]
                for f in range(nflows)
                for t in range(len(Tf[f]))
            )
            <= capacity[e]
        )

    for s in range(nscenarios):
        for f in range(nflows):
            satisfied = gp.quicksum(
                a[f, t] * X[s, Tf[f][t] - 1] for t in range(len(Tf[f]))
            ) / demand[f]
            model.addConstr(u[s, f] >= 1 - satisfied)

    for s in range(nscenarios):
        if average:
            model.addConstr(umax[s] + alpha >= gp.quicksum(u[s, f] for f in range(nflows)) / nflows)
        else:
            for f in range(nflows):
                model.addConstr(umax[s] + alpha >= u[s, f])

    model.setObjective(
        alpha + (1 / (1 - beta)) * gp.quicksum(p[s] * umax[s] for s in range(nscenarios)),
        GRB.MINIMIZE,
    )
    model.optimize()

    a_val = np.zeros((nflows, k), dtype=float)
    for f in range(nflows):
        for t in range(k):
            a_val[f, t] = a[f, t].X

    u_val = np.zeros((nscenarios, nflows), dtype=float)
    for s in range(nscenarios):
        for f in range(nflows):
            u_val[s, f] = u[s, f].X

    umax_val = np.array([umax[s].X for s in range(nscenarios)], dtype=float)
    alpha_val = float(alpha.X)
    obj_val = float(model.ObjVal)

    if explain:
        print_results(
            obj_val,
            alpha_val,
            a_val,
            u_val,
            umax_val,
            edges,
            scenarios,
            T,
            Tf,
            L,
            capacity,
            verbose=verbose,
            utilization=utilization,
        )

    return alpha_val, obj_val, a_val, umax_val
