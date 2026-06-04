"""toy_config.py - 读取 toy_experiment/data 下的实验配置。

输入：
    - tasks.csv：任务源宿、输入/输出带宽、计算需求和候选计算节点。
    - compute_nodes.csv：计算节点容量、失效概率和备注。
    - edges.csv：物理链路及容量。
    - paths.json：任务到候选计算节点的输入/输出候选路径。
    - failure_scenarios.json：人为定义的故障场景及概率；normal 场景概率自动补齐。
    - failure_events.json：没有显式场景时用于枚举独立故障事件。
    - positions.json：拓扑图节点坐标。
    - params.json：求解器、风险约束和目标函数参数。
输出：
    - default_config() 返回统一 config 字典，供 solve_toy.py 和 plot_toy.py 使用。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    """任务信息。"""

    task_id: str
    source: str
    destination: str
    b_in: float
    b_out: float
    compute_demand: float
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class ComputeNode:
    """计算节点信息。"""

    node_id: str
    capacity: float
    fail_prob: float
    note: str


@dataclass(frozen=True)
class FailureEvent:
    """独立故障事件信息。"""

    event_id: str
    label: str
    probability: float
    failed_nodes: tuple[str, ...] = ()
    failed_edges: tuple[tuple[str, str], ...] = ()


def edge_key(u: str, v: str) -> tuple[str, str]:
    """把无向边端点排序后作为统一键，避免 a-c 与 c-a 被视为不同链路。"""
    return tuple(sorted((u, v)))


def path_edges(path_nodes: list[str] | tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """把节点序列转换成无向边序列。"""
    return tuple(edge_key(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1))


def _split_semicolon(value: str | None) -> tuple[str, ...]:
    """解析 CSV 中用分号分隔的候选节点字段。"""
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(";") if part.strip())


def _load_tasks(path: Path) -> list[Task]:
    """读取 tasks.csv，输出 Task 列表。"""
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        return [
            Task(
                task_id=row["task_id"],
                source=row["source"],
                destination=row["destination"],
                b_in=float(row["b_in"]),
                b_out=float(row["b_out"]),
                compute_demand=float(row["compute_demand"]),
                candidates=_split_semicolon(row["candidates"]),
            )
            for row in rows
        ]


def _load_compute_nodes(path: Path) -> dict[str, ComputeNode]:
    """读取 compute_nodes.csv，输出 node_id 到 ComputeNode 的映射。"""
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        return {
            row["node_id"]: ComputeNode(
                node_id=row["node_id"],
                capacity=float(row["capacity"]),
                fail_prob=float(row["fail_prob"]),
                note=row.get("note", ""),
            )
            for row in rows
        }


def _load_edges(path: Path) -> tuple[tuple[tuple[str, str], ...], dict[tuple[str, str], float]]:
    """读取 edges.csv，输出链路集合和链路容量字典。"""
    edges: list[tuple[str, str]] = []
    capacities: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        for row in rows:
            edge = edge_key(row["u"], row["v"])
            edges.append(edge)
            capacities[edge] = float(row["capacity"])
    return tuple(edges), capacities


def _load_paths(path: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """读取 paths.json，按 (任务, 计算节点, 阶段) 分组候选路径。"""
    with path.open("r", encoding="utf-8") as f:
        raw_paths = json.load(f)
    paths: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in raw_paths:
        key = (item["task"], item["compute_node"], item["phase"])
        paths.setdefault(key, []).append(
            {
                "path_id": item["path_id"],
                "path_nodes": list(item["path_nodes"]),
            }
        )
    return paths


def _load_failure_events(path: Path) -> list[FailureEvent]:
    """读取 failure_events.json，输出可枚举组合的基础故障事件。"""
    with path.open("r", encoding="utf-8") as f:
        raw_events = json.load(f)
    events = []
    for item in raw_events:
        failed_edges = tuple(edge_key(edge[0], edge[1]) for edge in item.get("failed_edges", []))
        events.append(
            FailureEvent(
                event_id=item["event_id"],
                label=item["label"],
                probability=float(item["probability"]),
                failed_nodes=tuple(item.get("failed_nodes", [])),
                failed_edges=failed_edges,
            )
        )
    return events


def _is_normal_scenario(item: dict[str, Any]) -> bool:
    """判断一条显式场景是否是 normal 场景。"""
    event_ids = {str(event_id).lower() for event_id in item.get("event_ids", [])}
    label = str(item.get("label", "")).lower()
    no_failure = not item.get("failed_nodes") and not item.get("failed_edges")
    return no_failure and ("normal" in event_ids or label == "normal")


def _scenario_from_item(item: dict[str, Any], scenario_id: int, probability: float) -> dict:
    """把 JSON 场景条目转换成模型内部使用的场景字典。"""
    failed_edges = {edge_key(edge[0], edge[1]) for edge in item.get("failed_edges", [])}
    event_ids = list(item.get("event_ids", []))
    label = item.get("label", "normal")
    return {
        "scenario_id": int(item.get("scenario_id", scenario_id)),
        "event_ids": event_ids,
        "event_labels": [label] if label else [],
        "probability": float(probability),
        "failed_nodes": set(item.get("failed_nodes", [])),
        "failed_edges": failed_edges,
    }


def _load_failure_scenarios(path: Path) -> list[dict]:
    """读取显式故障场景，并自动补齐 normal 场景概率。

    failure_scenarios.json 允许只填写故障场景及其 probability。程序会自动计算：
        normal_probability = 1 - sum(failure_probability)

    如果文件中已经存在 normal 场景，其 probability 字段会被忽略并自动重写。
    """
    with path.open("r", encoding="utf-8") as f:
        raw_scenarios = json.load(f)

    normal_items = [item for item in raw_scenarios if _is_normal_scenario(item)]
    failure_items = [item for item in raw_scenarios if not _is_normal_scenario(item)]
    failure_prob = sum(float(item["probability"]) for item in failure_items)
    if failure_prob > 1.0 + 1e-8:
        raise ValueError(f"Toy failure scenario probabilities must not exceed 1.0, got {failure_prob:.12f}.")

    normal_prob = max(0.0, 1.0 - failure_prob)
    normal_item = normal_items[0] if normal_items else {
        "scenario_id": 1,
        "label": "normal",
        "event_ids": ["normal"],
        "failed_nodes": [],
        "failed_edges": [],
    }

    scenarios = [_scenario_from_item(normal_item, 1, normal_prob)]
    for idx, item in enumerate(failure_items, start=2):
        normalized = dict(item)
        normalized["scenario_id"] = idx
        scenarios.append(_scenario_from_item(normalized, idx, float(item["probability"])))

    total_prob = sum(float(s["probability"]) for s in scenarios)
    if abs(total_prob - 1.0) > 1e-8:
        raise ValueError(f"Toy failure scenario probabilities must sum to 1.0 after normal completion, got {total_prob:.12f}.")
    return scenarios


def _normal_probability_from_scenarios(scenarios: list[dict]) -> float:
    """从显式场景中统计无故障 normal 场景概率。"""
    normal_prob = 0.0
    for scenario in scenarios:
        if not scenario["failed_nodes"] and not scenario["failed_edges"]:
            normal_prob += float(scenario["probability"])
    return normal_prob


def _normal_probability_from_events(events: list[FailureEvent]) -> float:
    """从独立故障事件概率计算无故障概率。"""
    normal_prob = 1.0
    for event in events:
        normal_prob *= 1.0 - float(event.probability)
    return normal_prob


def _resolve_beta(params: dict, normal_probability: float) -> float:
    """根据 params.json 的 beta_mode 确定 CVaR 置信水平 beta。"""
    beta_mode = params.get("beta_mode", "fixed")
    if beta_mode == "fixed":
        beta = float(params.get("beta", 0.90))
    elif beta_mode == "normal_minus_margin":
        beta = float(normal_probability) - float(params.get("beta_margin", 0.01))
    else:
        raise ValueError(f"Unsupported beta_mode: {beta_mode}. Use fixed or normal_minus_margin.")

    if not 0.0 < beta < 1.0:
        raise ValueError(f"Resolved beta must be in (0, 1), got {beta}.")
    return beta


def default_config(data_dir: str | Path | None = None) -> dict:
    """读取 toy_experiment/data，统一打包求解与绘图所需配置。"""
    #========1. 读取网络拓扑=======
    base = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent / "data"
    with (base / "positions.json").open("r", encoding="utf-8") as f:
        positions = {node: tuple(value) for node, value in json.load(f).items()}
    with (base / "params.json").open("r", encoding="utf-8") as f:
        params = json.load(f)

    edges, capacities = _load_edges(base / "edges.csv")

    #========2. 读取故障事件/显式故障场景=======
    scenario_path = base / "failure_scenarios.json"
    explicit_scenarios = _load_failure_scenarios(scenario_path) if scenario_path.exists() else None
    failure_events = _load_failure_events(base / "failure_events.json")
    normal_probability = (
        _normal_probability_from_scenarios(explicit_scenarios)
        if explicit_scenarios is not None
        else _normal_probability_from_events(failure_events)
    )

    #========3. 解析风险参数=======
    beta = _resolve_beta(params, normal_probability)
    risk_mode = params.get("risk_mode", "weighted")
    if risk_mode not in {"weighted", "cvar_constraint"}:
        raise ValueError(f"Unsupported risk_mode: {risk_mode}. Use weighted or cvar_constraint.")
    cvar_bound = params.get("cvar_bound")
    if risk_mode == "cvar_constraint" and cvar_bound is None:
        raise ValueError("risk_mode=cvar_constraint requires cvar_bound in params.json.")

    node_layer_risk_mode = params.get("node_layer_risk_mode", "weighted")
    if node_layer_risk_mode not in {"weighted", "cvar_constraint"}:
        raise ValueError("node_layer_risk_mode must be weighted or cvar_constraint.")
    node_layer_cvar_bound = params.get("node_layer_cvar_bound")
    if node_layer_risk_mode == "cvar_constraint" and node_layer_cvar_bound is None:
        raise ValueError("node_layer_risk_mode=cvar_constraint requires node_layer_cvar_bound in params.json.")

    solver_mode = params.get("solver_mode", "single_level")
    if solver_mode not in {"single_level", "two_layer_average_split"}:
        raise ValueError("solver_mode must be single_level or two_layer_average_split.")

    #========4. 打包模型输入与输出目录=======
    return {
        "solver_mode": solver_mode,
        "beta": beta,
        "beta_mode": params.get("beta_mode", "fixed"),
        "beta_margin": float(params.get("beta_margin", 0.0)),
        "normal_probability": normal_probability,
        "lambda_weight": float(params.get("lambda_weight", 0.5)),
        "risk_weight": float(params.get("risk_weight", 1.0)),
        "risk_mode": risk_mode,
        "cvar_bound": None if cvar_bound is None else float(cvar_bound),
        "node_layer_risk_mode": node_layer_risk_mode,
        "node_layer_cvar_bound": None if node_layer_cvar_bound is None else float(node_layer_cvar_bound),
        "loss_aggregation": params.get("loss_aggregation", "max"),
        "tasks": _load_tasks(base / "tasks.csv"),
        "compute_nodes": _load_compute_nodes(base / "compute_nodes.csv"),
        "edges": edges,
        "capacities": capacities,
        "paths": _load_paths(base / "paths.json"),
        "failure_events": failure_events,
        "failure_scenarios": explicit_scenarios,
        "positions": positions,
        "params_path": str(base / "params.json"),
        "results_dir": str(Path(__file__).resolve().parent / "results"),
    }


def enumerate_scenarios(failure_events: list[FailureEvent] | tuple[FailureEvent, ...]) -> list[dict]:
    """枚举所有故障事件组合，输出每个场景的概率、故障节点和故障链路。"""
    scenarios = []
    n = len(failure_events)
    for r in range(n + 1):
        for event_subset in combinations(range(n), r):
            event_indices = set(event_subset)
            probability = 1.0
            failed_nodes: set[str] = set()
            failed_edges: set[tuple[str, str]] = set()
            labels = []
            ids = []
            for idx, event in enumerate(failure_events):
                if idx in event_indices:
                    probability *= event.probability
                    failed_nodes.update(event.failed_nodes)
                    failed_edges.update(edge_key(*edge) for edge in event.failed_edges)
                    labels.append(event.label)
                    ids.append(event.event_id)
                else:
                    probability *= 1.0 - event.probability
            scenarios.append(
                {
                    "scenario_id": len(scenarios) + 1,
                    "event_ids": ids,
                    "event_labels": labels,
                    "probability": probability,
                    "failed_nodes": failed_nodes,
                    "failed_edges": failed_edges,
                }
            )
    return scenarios
