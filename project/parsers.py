from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def _read_rows(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(line.split())
    return rows


def _resolve_path(path_or_rel: str) -> Path:
    p = Path(path_or_rel)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return BASE_DIR / path_or_rel


def _parse_node_token(token: str) -> int:
    t = token.strip().strip("(),")
    t = t.replace("h", "").replace("s", "")
    return int(t)


def _find_edge_index(edges: Sequence[Tuple[int, int]], edge: Tuple[int, int]) -> int | None:
    for i, e in enumerate(edges, start=1):
        if e == edge:
            return i
    return None


def next_run(dir_rel: str) -> str:
    out_dir = _resolve_path(dir_rel)
    out_dir.mkdir(parents=True, exist_ok=True)
    counter_path = out_dir / "counter.txt"
    if not counter_path.exists():
        counter_path.write_text("1\n", encoding="utf-8")

    c = int(counter_path.read_text(encoding="utf-8").split()[0])
    counter_path.write_text(f"{c + 1}\n", encoding="utf-8")
    new_dir = out_dir / str(c)
    new_dir.mkdir(parents=False, exist_ok=False)
    print("Output written to", new_dir)
    return str(new_dir)


def read_topology(
    topology: str,
    zeroindex: bool = False,
    downscale: float = 1.0,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, List[str]]:
    dir_path = BASE_DIR / "data" / topology
    topo_rows = _read_rows(dir_path / "topology.txt")[1:]  # skip header
    node_rows = _read_rows(dir_path / "nodes.txt")[1:]  # skip header

    offset = 1 if zeroindex else 0
    links: List[Tuple[int, int]] = []
    capacities: List[float] = []
    probabilities: List[float] = []
    for row in topo_rows:
        # NOTE: topology.txt header in B4 is to_node/from_node, but path files
        # are aligned with reading row[0] -> row[1] as directed edge.
        u = int(float(row[0]))
        v = int(float(row[1]))
        cap = float(row[2]) / downscale / 1000.0
        prob = float(row[3])
        links.append((u + offset, v + offset))
        capacities.append(cap)
        probabilities.append(prob)

    nodes = [r[0] for r in node_rows]
    return links, np.array(capacities, dtype=float), np.array(probabilities, dtype=float), nodes


def read_node_resources(topology: str):
    rows = _read_rows(BASE_DIR / "data" / topology / "nodes.txt")
    if len(rows) <= 1:
        raise ValueError("nodes.txt has no node rows")
    header = [h.lower() for h in rows[0]]
    data_rows = rows[1:]

    if "is_compute" not in header or "cpu" not in header:
        raise ValueError("nodes.txt must include 'is_compute' and 'cpu' columns in header")
    is_compute_idx = header.index("is_compute")
    cpu_idx = header.index("cpu")
    prob_idx = None
    for name in ("prob_failure", "node_prob_failure", "failure_prob"):
        if name in header:
            prob_idx = header.index(name)
            break
    name_idx = 0

    node_rows = []
    compute_nodes: List[int] = []
    cpu_capacity: Dict[int, float] = {}
    node_failure_probs: Dict[int, float] = {}
    for idx, row in enumerate(data_rows, start=1):
        if len(row) <= max(is_compute_idx, cpu_idx):
            raise ValueError(f"nodes.txt row {idx + 1} missing is_compute/cpu values")
        node_name = row[name_idx]
        is_compute = int(float(row[is_compute_idx]))
        cpu = float(row[cpu_idx])
        failure_prob = float(row[prob_idx]) if prob_idx is not None and len(row) > prob_idx else 0.0

        node_rows.append(
            {
                "node_id": idx,
                "node_name": node_name,
                "is_compute": is_compute,
                "cpu": cpu,
                "prob_failure": failure_prob,
            }
        )
        if is_compute == 1:
            compute_nodes.append(idx)
            cpu_capacity[idx] = cpu
            node_failure_probs[idx] = failure_prob

    if not compute_nodes:
        raise ValueError("No compute nodes found in nodes.txt (is_compute=1)")
    return node_rows, compute_nodes, cpu_capacity, node_failure_probs


def ignore_cycles(demand: np.ndarray, zeroindex: bool = False) -> np.ndarray:
    offset = 1 if zeroindex else 0
    kept = []
    for row in demand:
        if row[0] != row[1]:
            kept.append([row[0] + offset, row[1] + offset, row[2]])
    if not kept:
        return np.zeros((0, 3), dtype=float)
    return np.array(kept, dtype=float)


def parse_matrix(filename: str, num_nodes: int, num_demand: int, include_cycles: bool = False) -> np.ndarray:
    x = np.loadtxt(filename, dtype=float)
    if x.ndim == 1:
        x = np.array([x], dtype=float)
    vec = x[num_demand - 1, :]

    row_count = len(vec) if include_cycles else len(vec) - int(math.sqrt(len(vec)))
    m = np.zeros((row_count, 3), dtype=float)
    from_node = 0
    count = 0
    for i in range(num_nodes**2):
        to_node = i % num_nodes + 1
        if to_node == 1:
            from_node += 1
        if include_cycles or from_node != to_node:
            m[count, 0] = from_node
            m[count, 1] = to_node
            m[count, 2] = vec[i]
            count += 1

    ret = m.copy()
    for row in range(m.shape[0] - 1, -1, -1):
        if m[row, 2] == 0:
            ret = np.delete(ret, row, axis=0)
    return ret


def read_demand(
    filename: str,
    num_nodes: int,
    num_demand: int,
    scale: float = 1.0,
    matrix: bool = True,
    downscale: float = 1.0,
    zeroindex: bool = False,
    include_cycles: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    file_prefix = (BASE_DIR / "data" / filename).resolve()
    if matrix:
        input_demand = parse_matrix(f"{file_prefix}.txt", num_nodes, num_demand, include_cycles=include_cycles)
    else:
        demand_path = file_prefix / f"{num_demand}.txt"
        demand_raw = np.loadtxt(demand_path, skiprows=1)
        if include_cycles:
            offset = 1 if zeroindex else 0
            input_demand = demand_raw.copy()
            input_demand[:, 0] = input_demand[:, 0] + offset
            input_demand[:, 1] = input_demand[:, 1] + offset
        else:
            input_demand = ignore_cycles(demand_raw, zeroindex=zeroindex)

    from_nodes = input_demand[:, 0]
    to_nodes = input_demand[:, 1]
    flows = [(int(from_nodes[i]), int(to_nodes[i])) for i in range(len(from_nodes))]
    demand = input_demand[:, 2] / downscale * scale / 1000.0
    return demand.astype(float), flows


def parse_yates_splitting_ratios(
    filename: str,
    k: int,
    flows: Sequence[Tuple[int, int]],
    zeroindex: bool = False,
) -> np.ndarray:
    fpath = (BASE_DIR / "data" / filename).resolve()
    a = np.zeros((len(flows), k), dtype=float)
    num_flow = 0
    t = 0
    with fpath.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            if "->" in tokens:
                from_node = _parse_node_token(tokens[0]) + (1 if zeroindex else 0)
                to_node = _parse_node_token(tokens[2]) + (1 if zeroindex else 0)
                try:
                    num_flow = flows.index((from_node, to_node))
                except ValueError:
                    num_flow = -1
                t = 0
            else:
                if num_flow < 0:
                    continue
                for i, token in enumerate(tokens):
                    if "@" in token and i + 1 < len(tokens):
                        a[num_flow, t] = math.ceil(float(tokens[i + 1]) * 1000) / 1000
                        t += 1
                        break
    return a


def parse_paths(filename: str, links: Sequence[Tuple[int, int]], flows: Sequence[Tuple[int, int]], zeroindex: bool = False):
    fpath = (BASE_DIR / "data" / filename).resolve()
    nflows = len(flows)
    T: List[List[int]] = []
    Tf: List[List[int]] = [[] for _ in range(nflows)]

    tf: List[int] = []
    from_node = 0
    to_node = 0
    num_flow = 0
    tindex = 1
    max_paths = 0

    with fpath.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "->" in line:
                max_paths = max(max_paths, len(tf))
                if from_node != 0 and num_flow != 0:
                    Tf[num_flow - 1] = tf
                parts = line.replace(":", "").split()
                if re.match(r"^\d+$", parts[0]):
                    from_node = int(parts[0]) + (1 if zeroindex else 0)
                    to_node = int(parts[2]) + (1 if zeroindex else 0)
                else:
                    from_node = _parse_node_token(parts[0]) + (1 if zeroindex else 0)
                    to_node = _parse_node_token(parts[2]) + (1 if zeroindex else 0)
                try:
                    num_flow = flows.index((from_node, to_node)) + 1
                except ValueError:
                    num_flow = 0
                tf = []
            else:
                tunnel_edges: List[int] = []
                if "[" in line and "]" in line:
                    between = line[line.index("[") + 1 : line.index("]")]
                    pairs = re.findall(r"\(([^)]+)\)", between)
                    for pair in pairs:
                        left, right = [x.strip() for x in pair.split(",")]
                        e = (_parse_node_token(left) + (1 if zeroindex else 0), _parse_node_token(right) + (1 if zeroindex else 0))
                        idx = _find_edge_index(links, e)
                        if idx is not None:
                            tunnel_edges.append(idx)
                if num_flow != 0:
                    T.append(tunnel_edges)
                    tf.append(tindex)
                    tindex += 1

    if num_flow != 0:
        Tf[num_flow - 1] = tf
    T.append([])

    for f in range(len(Tf)):
        for _ in range(max_paths - len(Tf[f])):
            Tf[f].append(tindex)
    return T, Tf, max_paths


def get_tunnels(
    nodes: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    capacity: Sequence[float],
    flows: Sequence[Tuple[int, int]],
    k: int,
    edge_disjoint: bool = False,
):
    num_nodes = len(nodes)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(1, num_nodes + 1))
    for u, v in edges:
        graph.add_edge(u, v, weight=1.0)

    T: List[List[int]] = [[]]
    Tf: List[List[int]] = []
    ti = 2
    max_k = 1

    for flow in flows:
        tf: List[int] = []
        curr_k = 0
        edges_used: List[int] = []
        try:
            paths = list(nx.shortest_simple_paths(graph, flow[0], flow[1]))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            paths = []

        for i in range(k):
            path = paths[i] if i < len(paths) else []
            t: List[int] = []
            for n in range(1, len(path)):
                idx = _find_edge_index(edges, (path[n - 1], path[n]))
                if idx is not None:
                    t.append(idx)
            if len(t) == 0:
                break
            add = True
            if edge_disjoint:
                for e in t:
                    if e in edges_used:
                        add = False
                        break
            if add:
                if edge_disjoint:
                    edges_used.extend(t)
                T.append(t)
                tf.append(ti)
                ti += 1
                curr_k += 1
        max_k = max(max_k, curr_k)
        Tf.append(tf)

    for f in range(len(Tf)):
        for _ in range(max_k - len(Tf[f])):
            Tf[f].append(1)
    return T, Tf, max_k, graph


def build_all_pairs_flows(num_nodes: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(1, num_nodes + 1) for j in range(1, num_nodes + 1) if i != j]


def build_pair_to_tunnels(flows: Sequence[Tuple[int, int]], Tf: Sequence[Sequence[int]], T: Sequence[Sequence[int]]):
    pair_to_tunnels: Dict[Tuple[int, int], List[int]] = {}
    for i, pair in enumerate(flows):
        tlist = []
        for tid in Tf[i]:
            if tid <= 0:
                continue
            if tid > len(T):
                continue
            if len(T[tid - 1]) == 0:
                continue
            tlist.append(int(tid))
        # remove duplicates while preserving order
        dedup = list(dict.fromkeys(tlist))
        pair_to_tunnels[pair] = dedup
    return pair_to_tunnels


def load_paths_for_all_pairs(
    topology: str,
    path_mode: str,
    links: Sequence[Tuple[int, int]],
    nodes: Sequence[str],
    k_paths: int,
    edge_disjoint: bool = False,
):
    flows_all = build_all_pairs_flows(len(nodes))
    if path_mode.upper() == "KSP":
        T, Tf, k_used, _ = get_tunnels(nodes, links, [1.0] * len(links), flows_all, k_paths, edge_disjoint=edge_disjoint)
    else:
        T, Tf, k_used = parse_paths(f"{topology}/paths/{path_mode}", links, flows_all)
    pair_to_tunnels = build_pair_to_tunnels(flows_all, Tf, T)
    return T, Tf, k_used, flows_all, pair_to_tunnels


def read_task_file(task_file: str) -> Dict[int, float]:
    path = _resolve_path(task_file)
    rows = _read_rows(path)
    task_cpu: Dict[int, float] = {}
    for row in rows:
        if len(row) < 2:
            continue
        if row[0].startswith("#"):
            continue
        try:
            task_id = int(float(row[0]))
            cpu = float(row[1])
        except ValueError:
            # header row
            continue
        task_cpu[task_id] = cpu
    if not task_cpu:
        raise ValueError(f"No task cpu mapping found in {path}")
    return task_cpu


def build_tasks_from_demand_and_taskfile(
    demand: Sequence[float],
    flows: Sequence[Tuple[int, int]],
    task_cpu_map: Dict[int, float],
    rho: float,
    candidate_ms: Sequence[int],
    max_tasks: int | None = None,
    placement_policy: str = "all",
):
    tasks = []
    n = len(flows)
    cutoff = n if max_tasks is None else min(n, max_tasks)
    for idx in range(cutoff):
        task_idx = idx + 1
        if task_idx not in task_cpu_map:
            raise ValueError(f"task.txt missing cpu for task_id={task_idx}")
        src, dst = flows[idx]
        candidates = [int(m) for m in candidate_ms]
        if placement_policy == "exclude_endpoints":
            candidates = [m for m in candidates if m not in {int(src), int(dst)}]
        if len(candidates) == 0:
            raise ValueError(
                f"No candidate execution nodes left for task_id={task_idx} "
                f"under placement_policy={placement_policy}."
            )
        b_in = float(demand[idx])
        b_out = float(rho) * b_in
        tasks.append(
            {
                "task_idx": task_idx,
                "src": int(src),
                "dst": int(dst),
                "b_in": b_in,
                "b_out": b_out,
                "cpu_demand": float(task_cpu_map[task_idx]),
                "candidate_ms": candidates,
            }
        )
    return tasks
