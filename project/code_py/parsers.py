from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Sequence, Tuple

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
    print("Output written to ", dir_rel)
    out_dir = (BASE_DIR / dir_rel).resolve()
    counter_path = out_dir / "counter.txt"
    if not counter_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        counter_path.write_text("1\n", encoding="utf-8")
    c = int(counter_path.read_text(encoding="utf-8").split()[0])
    counter_path.write_text(f"{c + 1}\n", encoding="utf-8")
    new_dir = out_dir / str(c)
    new_dir.mkdir(parents=False, exist_ok=False)
    return str(new_dir)


def read_topology(topology: str, zeroindex: bool = False, downscale: float = 1) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, List[str]]:
    dir_path = BASE_DIR / "data" / topology
    topo_rows = _read_rows(dir_path / "topology.txt")[1:]  # skip header
    node_rows = _read_rows(dir_path / "nodes.txt")[1:]  # skip header

    ignore: Tuple[int, ...] = ()
    offset = 1 if zeroindex else 0
    links: List[Tuple[int, int]] = []
    capacities: List[float] = []
    probabilities: List[float] = []
    for row in topo_rows:
        from_node = int(float(row[0]))
        to_node = int(float(row[1]))
        cap = float(row[2]) / downscale / 1000.0
        prob = float(row[3])
        if from_node not in ignore and to_node not in ignore:
            links.append((from_node + offset, to_node + offset))
            capacities.append(cap)
            probabilities.append(prob)

    nodes = [r[0] for r in node_rows]
    return links, np.array(capacities, dtype=float), np.array(probabilities, dtype=float), nodes


def ignore_cycles(demand: np.ndarray, zeroindex: bool = False) -> np.ndarray:
    offset = 1 if zeroindex else 0
    kept = []
    for row in demand:
        if row[0] != row[1]:
            kept.append([row[0] + offset, row[1] + offset, row[2]])
    if not kept:
        return np.zeros((0, 3), dtype=float)
    return np.array(kept, dtype=float)


def parse_matrix(filename: str, num_nodes: int, num_demand: int) -> np.ndarray:
    x = np.loadtxt(filename, dtype=float)
    if x.ndim == 1:
        x = np.array([x], dtype=float)
    vec = x[num_demand - 1, :]

    ignore: Tuple[int, ...] = ()
    start_range = 0
    end_range = math.inf
    m = np.zeros((len(vec) - int(math.sqrt(len(vec))), 3), dtype=float)
    from_node = 0
    count = 0
    for i in range(num_nodes**2):
        to_node = i % num_nodes + 1
        if to_node == 1:
            from_node += 1
        if from_node != to_node and i >= start_range and i < end_range and from_node not in ignore and to_node not in ignore:
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
    downscale: float = 1,
    sigfigs: int = 1,
    zeroindex: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    file_prefix = (BASE_DIR / "data" / filename).resolve()
    if matrix:
        input_demand = parse_matrix(f"{file_prefix}.txt", num_nodes, num_demand)
    else:
        demand_path = file_prefix / f"{num_demand}.txt"
        demand_raw = np.loadtxt(demand_path, skiprows=1)
        input_demand = ignore_cycles(demand_raw, zeroindex=zeroindex)

    from_nodes = input_demand[:, 0]
    to_nodes = input_demand[:, 1]
    flows = [(int(from_nodes[i]), int(to_nodes[i])) for i in range(len(from_nodes))]
    demand = input_demand[:, 2] / downscale * scale / 1000.0
    return demand.astype(float), flows


def parse_yates_splitting_ratios(filename: str, k: int, flows: Sequence[Tuple[int, int]], zeroindex: bool = False) -> np.ndarray:
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


def get_tunnels(nodes: Sequence[str], edges: Sequence[Tuple[int, int]], capacity: Sequence[float], flows: Sequence[Tuple[int, int]], k: int, edge_disjoint: bool = False):
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
