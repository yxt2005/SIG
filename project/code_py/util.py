from __future__ import annotations

import itertools
from typing import List, Sequence, Tuple

import numpy as np


def print_results(
    o: float,
    alpha: float,
    a: np.ndarray,
    u: np.ndarray,
    umax: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    scenarios: Sequence[Sequence[float]],
    T: Sequence[Sequence[int]],
    Tf: Sequence[Sequence[int]],
    L: np.ndarray,
    capacity: Sequence[float],
    verbose: bool = False,
    utilization: bool = True,
):
    print("Objective value: ", o)
    print("")
    print("------------------ Allocations ----------------------\n")
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            print(f"Flow {i + 1}, tunnel {j + 1} allocated : {a[i, j]}")
            print("Edges in use: ", end="")
            for e in T[Tf[i][j] - 1]:
                print(edges[e - 1], end="")
            print("\n")
    if verbose:
        print("--------------- Loss Breakdown ---------------------\n")
        for s in range(len(umax)):
            if s == 0:
                print("Scenario 1: ", scenarios[s])
            else:
                print(f"Scenario {s + 1}: ", scenarios[s])
            print("Edges: ", end="")
            for i in range(len(scenarios[s])):
                if scenarios[s][i] == 0.0:
                    print(edges[i], end=" ")
            print("go down\n")
            for f in range(u.shape[1]):
                print(f"Loss on flow {f + 1} = {u[s, f]}")
            print("umax = ", umax[s])
            print("Max loss = ", umax[s] + alpha)
            print("")
    print("------------------------------------------------\n")
    if utilization:
        for e in range(len(edges)):
            print(f"EDGE: {e + 1} : {edges[e]}")
            print("capacity: ", capacity[e])
            used = 0.0
            for f in range(a.shape[0]):
                for t in range(a.shape[1]):
                    used += a[f, t] * L[Tf[f][t] - 1, e]
            print("used: ", used)
            print("")


def get_probabilities(scenarios: Sequence[Sequence[float]], probabilities: Sequence[float]) -> List[float]:
    p: List[float] = []
    for s in scenarios:
        prob = 1.0
        for i in range(len(s)):
            prob *= (1 - s[i]) * probabilities[i] + s[i] * (1 - probabilities[i])
        p.append(prob)
    return p


def k_scenarios(nedges: int, k: int, probabilities: Sequence[float], first: bool = True):
    scenarios: List[np.ndarray] = []
    if first:
        scenarios.append(np.ones(nedges, dtype=float))
    for i in range(1, k + 1):
        for bits in itertools.combinations(range(1, nedges + 1), i):
            s = np.ones(nedges, dtype=float)
            for bit in bits:
                s[bit - 1] = 0
            scenarios.append(s)
    probs = get_probabilities(scenarios, probabilities)
    return scenarios, probs


def all_scenarios(nedges: int, probabilities: Sequence[float], first: bool = True):
    scenarios: List[np.ndarray] = []
    probs: List[float] = []
    if first:
        scenario = np.ones(nedges, dtype=float)
        scenarios.append(scenario)
        prob = 1.0
        for i in range(len(scenario)):
            prob *= (1 - scenario[i]) * probabilities[i] + scenario[i] * (1 - probabilities[i])
        probs.append(prob)
    for i in range(1, nedges + 1):
        for bits in itertools.combinations(range(1, nedges + 1), i):
            s = np.ones(nedges, dtype=float)
            for bit in bits:
                s[bit - 1] = 0
            prob = 1.0
            for j in range(len(s)):
                prob *= (1 - s[j]) * probabilities[j] + s[j] * (1 - probabilities[j])
            probs.append(prob)
            scenarios.append(s)
    probs_arr = np.array(probs, dtype=float)
    probs_arr = probs_arr / np.sum(probs_arr)
    return scenarios, probs_arr.tolist()


def sub_scenarios_recursion(
    original: Sequence[float],
    cutoff: float,
    remaining: Sequence[float] | None = None,
    offset: int = 0,
    partial: List[int] | None = None,
    scenarios: List[np.ndarray] | None = None,
    probabilities: List[float] | None = None,
):
    if partial is None:
        partial = []
    if scenarios is None:
        scenarios = []
    if probabilities is None:
        probabilities = []
    if remaining is None:
        remaining = list(original)

    original_arr = np.array(original, dtype=float)
    if len(partial) == 0:
        scenarios.append(np.ones(len(original_arr), dtype=float))
        probabilities.append(float(np.prod(1 - original_arr)))
        remaining = original_arr
    else:
        probs = 1 - original_arr
        bitmap = np.ones(len(original_arr), dtype=float)
        for index in partial:
            probs[index - 1] = original_arr[index - 1]
            bitmap[index - 1] = 0
        product = float(np.prod(probs))
        if product >= cutoff:
            scenarios.append(bitmap)
            probabilities.append(product)
        else:
            return scenarios, probabilities

    rem = list(remaining)
    for i in range(len(rem)):
        offset = len(original_arr) - len(rem)
        n = offset + i + 1
        sub_scenarios_recursion(
            original_arr,
            cutoff,
            rem[i + 1 :],
            offset,
            partial + [n],
            scenarios,
            probabilities,
        )
    return scenarios, probabilities


def sub_scenarios(original: Sequence[float], cutoff: float, first: bool = True, last: bool = True, progress: bool = True):
    scenarios, probabilities = sub_scenarios_recursion(original, cutoff)
    if not first:
        scenarios = scenarios[1:]
        probabilities = probabilities[1:]
    if last and len(scenarios) > 0:
        scenarios.append(np.zeros(len(scenarios[0]), dtype=float))
        probabilities.append(1 - sum(probabilities))
    if sum(probabilities) < 1 and sum(probabilities) > 0:
        p = np.array(probabilities, dtype=float)
        p = p / np.sum(p)
        probabilities = p.tolist()
    return scenarios, probabilities


def weibull_probs(num: int, shape: float = 0.8, scale: float = 0.0001):
    # Julia Distributions.Weibull(shape, scale) = scale * numpy.weibull(shape)
    draws = scale * np.random.weibull(shape, size=num)
    return draws.tolist()
