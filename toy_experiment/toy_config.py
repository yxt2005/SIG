from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    source: str
    destination: str
    b_in: float
    b_out: float
    compute_demand: float
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class ComputeNode:
    node_id: str
    capacity: float
    fail_prob: float
    note: str


@dataclass(frozen=True)
class FailureEvent:
    event_id: str
    label: str
    probability: float
    failed_nodes: tuple[str, ...] = ()
    failed_edges: tuple[tuple[str, str], ...] = ()


def edge_key(u: str, v: str) -> tuple[str, str]:
    return tuple(sorted((u, v)))


def path_edges(path_nodes: list[str] | tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(edge_key(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1))


def _split_semicolon(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(";") if part.strip())


def _load_tasks(path: Path) -> list[Task]:
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


def _load_failure_scenarios(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        raw_scenarios = json.load(f)

    scenarios = []
    for idx, item in enumerate(raw_scenarios, start=1):
        failed_edges = {edge_key(edge[0], edge[1]) for edge in item.get("failed_edges", [])}
        event_ids = list(item.get("event_ids", []))
        label = item.get("label", "normal")
        scenarios.append(
            {
                "scenario_id": int(item.get("scenario_id", idx)),
                "event_ids": event_ids,
                "event_labels": [label] if label else [],
                "probability": float(item["probability"]),
                "failed_nodes": set(item.get("failed_nodes", [])),
                "failed_edges": failed_edges,
            }
        )

    total_prob = sum(float(s["probability"]) for s in scenarios)
    if abs(total_prob - 1.0) > 1e-8:
        raise ValueError(f"Toy failure scenario probabilities must sum to 1.0, got {total_prob:.12f}.")
    return scenarios


def _normal_probability_from_scenarios(scenarios: list[dict]) -> float:
    normal_prob = 0.0
    for scenario in scenarios:
        if not scenario["failed_nodes"] and not scenario["failed_edges"]:
            normal_prob += float(scenario["probability"])
    return normal_prob


def _normal_probability_from_events(events: list[FailureEvent]) -> float:
    normal_prob = 1.0
    for event in events:
        normal_prob *= 1.0 - float(event.probability)
    return normal_prob


def _resolve_beta(params: dict, normal_probability: float) -> float:
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
    base = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent / "data"
    with (base / "positions.json").open("r", encoding="utf-8") as f:
        positions = {node: tuple(value) for node, value in json.load(f).items()}
    with (base / "params.json").open("r", encoding="utf-8") as f:
        params = json.load(f)

    edges, capacities = _load_edges(base / "edges.csv")
    scenario_path = base / "failure_scenarios.json"
    explicit_scenarios = _load_failure_scenarios(scenario_path) if scenario_path.exists() else None
    failure_events = _load_failure_events(base / "failure_events.json")
    normal_probability = (
        _normal_probability_from_scenarios(explicit_scenarios)
        if explicit_scenarios is not None
        else _normal_probability_from_events(failure_events)
    )
    beta = _resolve_beta(params, normal_probability)
    risk_mode = params.get("risk_mode", "weighted")
    if risk_mode not in {"weighted", "cvar_constraint"}:
        raise ValueError(f"Unsupported risk_mode: {risk_mode}. Use weighted or cvar_constraint.")
    cvar_bound = params.get("cvar_bound")
    if risk_mode == "cvar_constraint" and cvar_bound is None:
        raise ValueError("risk_mode=cvar_constraint requires cvar_bound in params.json.")

    return {
        "beta": beta,
        "beta_mode": params.get("beta_mode", "fixed"),
        "beta_margin": float(params.get("beta_margin", 0.0)),
        "normal_probability": normal_probability,
        "lambda_weight": float(params.get("lambda_weight", 0.5)),
        "risk_weight": float(params.get("risk_weight", 1.0)),
        "risk_mode": risk_mode,
        "cvar_bound": None if cvar_bound is None else float(cvar_bound),
        "loss_aggregation": params.get("loss_aggregation", "max"),
        "tasks": _load_tasks(base / "tasks.csv"),
        "compute_nodes": _load_compute_nodes(base / "compute_nodes.csv"),
        "edges": edges,
        "capacities": capacities,
        "paths": _load_paths(base / "paths.json"),
        "failure_events": failure_events,
        "failure_scenarios": explicit_scenarios,
        "positions": positions,
        "results_dir": str(Path(__file__).resolve().parent / "results"),
    }


def enumerate_scenarios(failure_events: list[FailureEvent] | tuple[FailureEvent, ...]) -> list[dict]:
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
