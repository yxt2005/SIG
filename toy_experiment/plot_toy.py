"""Toy 实验拓扑可视化。

输入：
    - config：由 toy_config.default_config() 读取出的拓扑、任务、路径、故障场景配置。
    - solution_bundle：由 solve_toy.solve_toy() 返回的选址、路径预留和场景损失。
输出：
    - toy_topology_solution.png/svg：四宫格拓扑图，展示候选拓扑、正常态和故障态。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from toy_config import edge_key


TASK_COLORS = {"i1": "#d62728", "i2": "#1f77b4"}


def _scenario_table(best: dict) -> dict[int, dict[str, Any]]:
    """将逐任务的场景损失行合并成逐场景字典，便于绘图时查询故障节点/链路。"""
    grouped: dict[int, dict[str, Any]] = {}
    for row in best["scenario_rows"]:
        sid = int(row["scenario_id"])
        item = grouped.setdefault(
            sid,
            {
                "scenario_id": sid,
                "probability": float(row["probability"]),
                "event_ids": row["event_ids"],
                "event_labels": row["event_labels"],
                "failed_nodes": set(filter(None, row["failed_nodes"].split(";"))),
                "failed_edges": set(),
                "system_loss": float(row["system_loss"]),
            },
        )
        if row["failed_edges"]:
            item["failed_edges"].update(tuple(part.split("-")) for part in row["failed_edges"].split(";") if part)
    for item in grouped.values():
        item["failed_edges"] = {edge_key(u, v) for u, v in item["failed_edges"]}
    return grouped


def _choose_access_failure(best: dict):
    """选择一个会造成损失的接入链路故障场景，用作子图 (c) 的可视化示例。"""
    scenarios = _scenario_table(best)
    access_ids = {
        "upper_input_access",
        "lower_input_access",
        "upper_output_access",
        "lower_output_access",
    }
    candidates = [
        s for s in scenarios.values()
        if any(event_id in s["event_ids"] for event_id in access_ids) and s["system_loss"] > 1e-9
    ]
    if not candidates:
        candidates = [
            s for s in scenarios.values()
            if any(event_id in s["event_ids"] for event_id in access_ids)
        ]
    return max(candidates, key=lambda s: s["system_loss"]) if candidates else None


def _choose_compute_failure(best: dict):
    """选择一个命中已选计算节点的故障场景，用作子图 (d) 的可视化示例。"""
    scenarios = _scenario_table(best)
    selected_nodes = set(best["placement"].values())
    candidates = []
    for scenario in scenarios.values():
        if scenario["failed_nodes"] & selected_nodes:
            candidates.append(scenario)
    return max(candidates, key=lambda s: s["system_loss"]) if candidates else None


def _panel_caption(ax, caption: str):
    """在每个子图下方绘制说明文字。"""
    ax.text(
        0.5,
        -0.08,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )


def _draw_base(ax, config: dict, selected_nodes: set[str] | None = None, failed_nodes=None, failed_edges=None):
    """绘制底图：物理链路、转发节点、计算节点以及故障标记。"""
    selected_nodes = selected_nodes or set()
    failed_nodes = failed_nodes or set()
    failed_edges = failed_edges or set()
    pos = config["positions"]

    ax.set_xlim(-0.35, 5.4)
    ax.set_ylim(-2.05, 2.05)
    ax.axis("off")

    for edge in config["edges"]:
        u, v = edge
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        color = "#d0d0d0"
        lw = 1.0
        ls = "-"
        if edge_key(u, v) in failed_edges:
            color = "#9f9f9f"
            lw = 2.0
            ls = "--"
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=ls, zorder=1)

    for node, (x, y) in pos.items():
        if node in config["compute_nodes"]:
            face = "#ffe8a3" if node in selected_nodes else "#fff2c6"
            rect = FancyBboxPatch(
                (x - 0.27, y - 0.22), 0.54, 0.44,
                boxstyle="round,pad=0.035,rounding_size=0.035",
                facecolor=face,
                edgecolor="#111111",
                linewidth=1.0,
                zorder=4,
            )
            ax.add_patch(rect)
            ax.text(x, y, node, ha="center", va="center", fontsize=12, fontstyle="italic", zorder=11)
        else:
            circle_face = "#ffffff"
            ax.scatter([x], [y], s=390, facecolor=circle_face, edgecolor="#111111", linewidth=1.0, zorder=4)
            ax.text(x, y, node, ha="center", va="center", fontsize=10, zorder=11)
        if node in failed_nodes:
            ax.text(x, y, "X", ha="center", va="center", color="#d62728", fontsize=24, fontweight="bold", zorder=15)


def _task_demand(config: dict) -> dict[tuple[str, str], float]:
    """返回 (任务, 阶段) 到带宽需求的映射，阶段 phase 取 in/out。"""
    demand = {}
    for task in config["tasks"]:
        demand[(task.task_id, "in")] = float(task.b_in)
        demand[(task.task_id, "out")] = float(task.b_out)
    return demand


def _path_failed(allocation: dict, failed_edges: set[tuple[str, str]], failed_nodes: set[str]) -> bool:
    """判断一条候选路径在当前故障场景下是否不可用。"""
    if allocation["compute_node"] in failed_nodes or bool(set(allocation["path_nodes"]) & failed_nodes):
        return True
    for u, v in zip(allocation["path_nodes"][:-1], allocation["path_nodes"][1:]):
        if edge_key(u, v) in failed_edges:
            return True
    return False


def _actual_bandwidth_by_path(
    config: dict,
    allocations: list[dict],
    failed_edges: set[tuple[str, str]],
    failed_nodes: set[str],
) -> dict[str, float]:
    """计算每条路径的实际占用带宽。

    关键变量：
        allocation：模型为路径预留的带宽。
        actual：当前场景下任务流真正使用的带宽。

    计算规则：
        同一任务同一阶段的可用预留总量如果大于需求，则按预留比例分摊需求；
        如果小于需求，则幸存路径全部用满，故障路径实际占用为 0。
    """
    demand = _task_demand(config)
    grouped: dict[tuple[str, str], list[dict]] = {}
    actual_by_path = {row["path_id"]: 0.0 for row in allocations}

    for row in allocations:
        if float(row["allocation"]) <= 1e-7:
            continue
        grouped.setdefault((row["task"], row["phase"]), []).append(row)

    for key, rows in grouped.items():
        usable_rows = [row for row in rows if not _path_failed(row, failed_edges, failed_nodes)]
        usable_reserved = sum(float(row["allocation"]) for row in usable_rows)
        if usable_reserved <= 1e-9:
            continue

        required = demand[key]
        scale = min(1.0, required / usable_reserved)
        for row in usable_rows:
            actual_by_path[row["path_id"]] = float(row["allocation"]) * scale
    return actual_by_path


def _format_bandwidth(value: float) -> str:
    """格式化带宽数值：整数不显示小数，非整数保留两位小数。"""
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def _draw_path(ax, config: dict, allocation: dict, actual_bandwidth: float, failed_edges=None, failed_nodes=None):
    """绘制单条路径，并以“实际 / 预留”的形式标注带宽。"""
    failed_edges = failed_edges or set()
    failed_nodes = failed_nodes or set()
    reserved_bandwidth = float(allocation["allocation"])
    if reserved_bandwidth <= 1e-7:
        return

    path_nodes = allocation["path_nodes"]
    task = allocation["task"]
    color = TASK_COLORS.get(task, "#333333")
    linestyle = "--" if allocation["path_id"].endswith("_2") else "-"
    path_failed = _path_failed(allocation, failed_edges, failed_nodes)

    draw_color = "#9f9f9f" if path_failed else color
    alpha = 0.78 if path_failed else 1.0
    line_width = 2.8 if path_failed else 2.0
    path_zorder = 7 if path_failed else 8
    pos = config["positions"]
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        shrink_a = 23 if u in config["compute_nodes"] else 16
        shrink_b = 23 if v in config["compute_nodes"] else 16
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=draw_color,
                lw=line_width,
                linestyle=linestyle,
                shrinkA=shrink_a,
                shrinkB=shrink_b,
                mutation_scale=10,
                alpha=alpha,
            ),
            zorder=path_zorder,
        )

    label_segment = 0 if len(path_nodes) <= 2 else min(1, len(path_nodes) - 2)
    u_label = path_nodes[label_segment]
    v_label = path_nodes[label_segment + 1]
    x_mid = (pos[u_label][0] + pos[v_label][0]) / 2.0
    y_mid = (pos[u_label][1] + pos[v_label][1]) / 2.0
    task_shift = 0.13 if task == "i1" else -0.13
    phase_shift = 0.08 if allocation["phase"] == "in" else -0.08
    path_shift = 0.08 if allocation["path_id"].endswith("_2") else -0.02
    ax.text(
        x_mid + path_shift,
        y_mid + task_shift + phase_shift,
        f"{_format_bandwidth(actual_bandwidth)} / {_format_bandwidth(reserved_bandwidth)}",
        color=draw_color,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.4),
        zorder=14,
    )


def _draw_allocations(ax, config: dict, best: dict, scenario=None):
    """绘制当前场景下的全部已选路径，并自动计算每条路径的实际占用带宽。"""
    failed_edges = scenario["failed_edges"] if scenario else set()
    failed_nodes = scenario["failed_nodes"] if scenario else set()
    actual_by_path = _actual_bandwidth_by_path(config, best["allocations"], failed_edges, failed_nodes)
    ordered_allocations = []
    for allocation in best["allocations"]:
        path_failed = _path_failed(allocation, failed_edges, failed_nodes)
        ordered_allocations.append((path_failed, allocation))
    for _, allocation in sorted(ordered_allocations, key=lambda item: 0 if item[0] else 1):
        _draw_path(
            ax,
            config,
            allocation,
            actual_by_path.get(allocation["path_id"], 0.0),
            failed_edges=failed_edges,
            failed_nodes=failed_nodes,
        )


def _scenario_caption(scenario: dict | None, fallback: str) -> str:
    """将场景标签转换成更适合论文图的英文描述。"""
    if not scenario:
        return fallback
    label = scenario["event_labels"]
    label_text = label if isinstance(label, str) else "; ".join(label)
    replacements = {
        "upper input access links fail": "forwarding node a unavailable",
        "lower input access links fail": "forwarding node c unavailable",
        "upper output access links fail": "forwarding node b unavailable",
        "lower output access links fail": "forwarding node d unavailable",
        "mA compute site unavailable": "compute site mA unavailable",
        "mB compute site unavailable": "compute site mB unavailable",
        "mC compute site unavailable": "compute site mC unavailable",
    }
    return replacements.get(label_text, label_text)


def _panel_c_visual_scenario(scenario: dict | None):
    """把接入链路故障转换成图 (c) 中更直观的转发节点 a 不可用画法。"""
    if not scenario:
        return None
    visual = dict(scenario)
    visual["failed_nodes"] = {"a"}
    visual["failed_edges"] = {
        edge_key("s1", "a"),
        edge_key("a", "mA"),
    }
    visual["event_labels"] = ["forwarding node a unavailable"]
    return visual


def plot_toy_solution(config: dict, solution_bundle: dict, output_dir: str | Path):
    """生成 toy 拓扑结果图。

    输入：
        config：网络、任务、路径、故障配置。
        solution_bundle：求解器输出的 best/all_results/scenarios。
        output_dir：图片输出目录。
    输出：
        (png_path, svg_path)：生成图片的路径。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = solution_bundle["best"]
    selected_nodes = set(best["placement"].values())
    access_scenario = _choose_access_failure(best)
    compute_scenario = _choose_compute_failure(best)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0][0]
    _draw_base(ax, config)
    pos = config["positions"]
    for u, v in [("a", "c"), ("b", "d")]:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="#777777", linewidth=2.0, linestyle="--", zorder=2)
    _panel_caption(ax, "(a) Candidate topology")

    ax = axes[0][1]
    _draw_base(ax, config, selected_nodes=selected_nodes)
    _draw_allocations(ax, config, best)
    _panel_caption(ax, "(b) Working state from solved model")

    ax = axes[1][0]
    access_visual = _panel_c_visual_scenario(access_scenario)
    _draw_base(
        ax,
        config,
        selected_nodes=selected_nodes,
        failed_nodes=access_visual["failed_nodes"] if access_visual else set(),
        failed_edges=access_visual["failed_edges"] if access_visual else set(),
    )
    _draw_allocations(ax, config, best, scenario=access_visual)
    if access_scenario:
        caption = _scenario_caption(access_visual, "Access-link failure")
        _panel_caption(
            ax,
            f"(c) {caption}, $L_s$={access_scenario['system_loss']:.2f}",
        )
    else:
        _panel_caption(ax, "(c) Access-link failure")

    ax = axes[1][1]
    _draw_base(
        ax,
        config,
        selected_nodes=selected_nodes,
        failed_nodes=compute_scenario["failed_nodes"] if compute_scenario else set(),
    )
    _draw_allocations(ax, config, best, scenario=compute_scenario)
    if compute_scenario:
        caption = _scenario_caption(compute_scenario, "Compute-site unavailable")
        _panel_caption(
            ax,
            f"(d) {caption}, $L_s$={compute_scenario['system_loss']:.2f}",
        )
    else:
        _panel_caption(ax, "(d) Compute-site unavailable")

    fig.subplots_adjust(left=0.035, right=0.99, top=0.98, bottom=0.08, wspace=0.12, hspace=0.22)
    png_path = output_dir / "toy_topology_solution.png"
    svg_path = output_dir / "toy_topology_solution.svg"
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def read_csv_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
