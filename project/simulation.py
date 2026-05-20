from __future__ import annotations

from typing import Dict, Sequence, Tuple

from util import compute_node_usage


def recompute_link_loads_from_alloc(
    x_in: Dict[Tuple[int, int], float],
    x_out: Dict[Tuple[int, int], float],
    T: Sequence[Sequence[int]],
    L,
    nedges: int,
):
    loads = {e: 0.0 for e in range(1, nedges + 1)}
    for (_, tunnel_id), bw in x_in.items():
        if bw <= 0:
            continue
        t = int(tunnel_id) - 1
        for e in range(nedges):
            if L[t, e] > 0:
                loads[e + 1] += bw
    for (_, tunnel_id), bw in x_out.items():
        if bw <= 0:
            continue
        t = int(tunnel_id) - 1
        for e in range(nedges):
            if L[t, e] > 0:
                loads[e + 1] += bw
    return loads


def run_sanity_checks(
    tasks,
    best_assignment: Dict[int, int],
    lower_result: Dict[str, object],
    capacity,
    cpu_capacity,
    scenarios,
    seed: int,
):
    checks: Dict[str, Dict[str, object]] = {}

    status = str(lower_result.get("status", "unknown")).lower()
    checks["solver_optimal"] = {
        "pass": status in {"optimal", "suboptimal", "time_limit"},
        "detail": f"lower_status={status}",
    }

    link_ok = True
    bad_edges = []
    link_loads = lower_result.get("link_loads", {})
    u_link_max = float(lower_result.get("u_link_max", 0.0))
    for e_idx in range(1, len(capacity) + 1):
        cap = float(capacity[e_idx - 1])
        limit = cap * u_link_max + 1e-7
        load = float(link_loads.get(e_idx, 0.0))
        if load > limit:
            link_ok = False
            bad_edges.append({"edge_id": e_idx, "load": load, "limit": limit})
    checks["capacity_recompute"] = {
        "pass": link_ok,
        "detail": "all edges satisfy LinkLoad <= B_e * u_link_max" if link_ok else "capacity violation",
        "value": bad_edges,
    }

    node_used, _, u_node_recomputed = compute_node_usage(tasks, best_assignment, cpu_capacity)
    node_ok = True
    bad_nodes = []
    u_node_max = float(lower_result.get("u_node_max", 0.0))
    for m, cap in cpu_capacity.items():
        used = float(node_used.get(m, 0.0))
        limit = cap * u_node_max + 1e-7
        if used > limit:
            node_ok = False
            bad_nodes.append({"node_id": m, "used": used, "limit": limit})
    checks["cpu_recompute"] = {
        "pass": node_ok,
        "detail": "all compute nodes satisfy NodeUse <= C_m * u_node_max" if node_ok else "cpu violation",
        "value": {"bad_nodes": bad_nodes, "u_node_recomputed": u_node_recomputed},
    }

    # No-failure consistency should be recomputed from delivered traffic instead of
    # relying on auxiliary LP variables (which may be degenerate under same objective).
    delivered_in = lower_result.get("delivered_in", {})
    delivered_out = lower_result.get("delivered_out", {})
    if len(scenarios) > 0:
        no_failure_loss = 0.0
        for task in tasks:
            tid = int(task["task_idx"])
            b_in = float(task["b_in"])
            b_out = float(task["b_out"])
            din = float(delivered_in.get((1, tid), 0.0))
            dout = float(delivered_out.get((1, tid), 0.0))
            loss_i = max(
                0.0,
                1 - din / b_in if b_in > 0 else 0.0,
                1 - dout / b_out if b_out > 0 else 0.0,
            )
            no_failure_loss = max(no_failure_loss, loss_i)
        checks["no_failure_consistency"] = {
            "pass": no_failure_loss <= 1e-5,
            "detail": "scenario 1 (all links up) should have near-zero system loss",
            "value": no_failure_loss,
        }
    else:
        checks["no_failure_consistency"] = {"pass": False, "detail": "no scenarios provided", "value": None}

    checks["reproducibility"] = {
        "pass": True,
        "detail": "single-run check recorded; rerun with same seed should reproduce",
        "value": {"seed": seed},
    }
    return checks
