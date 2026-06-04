"""toy_experiment 主入口。

输入：
    - toy_experiment/data 下的拓扑、任务、故障场景与参数配置。
输出：
    - toy_experiment/results/runs/xxx 下的 CSV/JSON 结果与拓扑图。

执行方式：
    python toy_experiment/run_toy.py
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from plot_toy import plot_toy_solution
from solve_toy import printable_summary, solve_toy, write_results
from toy_config import default_config


SUMMARY_FIELDS = [
    "run_id",
    "run_dir",
    "best_placement",
    "solver_mode",
    "risk_mode",
    "cvar_bound",
    "beta",
    "beta_mode",
    "model_cvar",
    "eval_cvar",
    "cvar_bound_slack",
    "model_u_node_max",
    "model_u_link_max",
    "maximum_link_utilization",
    "node_layer_cvar",
    "node_layer_risk_mode",
    "node_layer_cvar_bound",
    "node_layer_u_link_proxy",
    "link_layer_cvar",
    "total_reserved_bandwidth",
    "effective_throughput",
    "min_loss",
    "expected_loss",
    "max_loss",
    "availability",
    "solver_status",
]


def _next_run_dir(results_root: Path) -> Path:
    """生成 runs/xxx 输出目录，避免覆盖历史实验结果。"""
    runs_root = results_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    existing_ids = []
    for child in runs_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            existing_ids.append(int(child.name))
    next_id = max(existing_ids, default=0) + 1
    return runs_root / f"{next_id:03d}"


def _copy_run_params(config: dict, run_dir: Path):
    """复制本次运行使用的 params.json，便于复现实验设置。"""
    shutil.copy2(Path(config["params_path"]), run_dir / "params.json")


def _write_run_metadata(config: dict, solution_bundle: dict, run_dir: Path):
    """写入本次运行的轻量索引，详细参数和指标分别见 params.json 与 metrics.csv。"""
    best = solution_bundle["best"]
    metadata = {
        "run_id": run_dir.name,
        "status": best["status"],
        "reason": best["reason"],
        "solver_mode": best["metrics"].get("solver_mode", config["solver_mode"]),
        "files": {
            "params": "params.json",
            "placements": "placements.json",
            "metrics": "metrics.csv",
            "path_allocations": "path_allocations.csv",
            "scenario_losses": "scenario_losses.csv",
            "link_loads": "link_loads.csv",
            "figure_png": "toy_topology_solution.png",
            "figure_svg": "toy_topology_solution.svg",
        },
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _summary_row_from_parts(run_dir: Path, placement: dict, metrics: dict) -> dict:
    """将 placement 与 metrics 整理为固定 schema 的 results/summary.csv 行。"""
    row = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "best_placement": ";".join(f"{task}->{node}" for task, node in placement.items()),
        "solver_mode": metrics.get("solver_mode"),
        "risk_mode": metrics.get("risk_mode"),
        "cvar_bound": metrics.get("cvar_bound"),
        "beta": metrics.get("beta"),
        "beta_mode": metrics.get("beta_mode"),
        "model_cvar": metrics.get("model_cvar"),
        "eval_cvar": metrics.get("cvar"),
        "cvar_bound_slack": metrics.get("cvar_bound_slack"),
        "model_u_node_max": metrics.get("model_u_node_max"),
        "model_u_link_max": metrics.get("model_u_link_max"),
        "maximum_link_utilization": metrics.get("maximum_link_utilization"),
        "node_layer_cvar": metrics.get("node_layer_cvar"),
        "node_layer_risk_mode": metrics.get("node_layer_risk_mode"),
        "node_layer_cvar_bound": metrics.get("node_layer_cvar_bound"),
        "node_layer_u_link_proxy": metrics.get("node_layer_u_link_proxy"),
        "link_layer_cvar": metrics.get("link_layer_cvar"),
        "total_reserved_bandwidth": metrics.get("total_reserved_bandwidth"),
        "effective_throughput": metrics.get("effective_throughput"),
        "min_loss": metrics.get("min_loss"),
        "expected_loss": metrics.get("expected_loss"),
        "max_loss": metrics.get("max_loss"),
        "availability": metrics.get("availability"),
        "solver_status": metrics.get("solver_status"),
    }
    return {field: row.get(field, "") for field in SUMMARY_FIELDS}


def _summary_header_is_current(summary_path: Path) -> bool:
    """检查现有 summary.csv 表头是否与当前固定 schema 一致。"""
    if not summary_path.exists():
        return True
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f), None)
    return header == SUMMARY_FIELDS


def _enrich_metrics_from_scenario_rows(run_dir: Path, metrics: dict) -> dict:
    """旧结果缺少新指标时，从 scenario_losses.csv 反推最小损失与有效吞吐。"""
    if metrics.get("min_loss") not in (None, "") and metrics.get("effective_throughput") not in (None, ""):
        return metrics

    scenario_path = run_dir / "scenario_losses.csv"
    if not scenario_path.exists():
        return metrics

    demand_by_task = {
        task.task_id: float(task.b_in) + float(task.b_out)
        for task in default_config()["tasks"]
    }
    grouped = {}
    with scenario_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = int(row["scenario_id"])
            item = grouped.setdefault(
                sid,
                {
                    "probability": float(row["probability"]),
                    "system_loss": float(row["system_loss"]),
                    "goodput": 0.0,
                },
            )
            loss_in = float(row["loss_in"])
            loss_out = float(row["loss_out"])
            completion_ratio = min(1.0, max(0.0, 1.0 - loss_in), max(0.0, 1.0 - loss_out))
            item["goodput"] += completion_ratio * demand_by_task.get(row["task"], 0.0)

    if grouped:
        metrics.setdefault("min_loss", min(item["system_loss"] for item in grouped.values()))
        metrics.setdefault(
            "effective_throughput",
            sum(item["probability"] * item["goodput"] for item in grouped.values()),
        )
    return metrics


def _summary_row_from_run(run_dir: Path) -> dict | None:
    """从已完成的 runs/xxx 目录反向恢复一行汇总记录。"""
    placements_path = run_dir / "placements.json"
    metrics_path = run_dir / "metrics.csv"
    if not placements_path.exists() or not metrics_path.exists():
        return None
    with placements_path.open("r", encoding="utf-8") as f:
        placement = json.load(f).get("placement", {})
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        metrics = next(csv.DictReader(f), None)
    if metrics is None:
        return None
    metrics = _enrich_metrics_from_scenario_rows(run_dir, metrics)
    return _summary_row_from_parts(run_dir, placement, metrics)


def rebuild_summary(results_root: Path):
    """从所有已完成的 runs/xxx 目录重建 results/summary.csv。"""
    summary_path = results_root / "summary.csv"
    runs_root = results_root / "runs"
    rows = []
    if runs_root.exists():
        run_dirs = [
            child for child in runs_root.iterdir()
            if child.is_dir() and child.name.isdigit()
        ]
        for run_dir in sorted(run_dirs, key=lambda path: int(path.name)):
            row = _summary_row_from_run(run_dir)
            if row:
                rows.append(row)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _append_summary(results_root: Path, run_dir: Path, solution_bundle: dict):
    """把本次运行的核心指标追加到 results/summary.csv。"""
    summary_path = results_root / "summary.csv"
    if not _summary_header_is_current(summary_path):
        rebuild_summary(results_root)
        return

    best = solution_bundle["best"]
    row = _summary_row_from_parts(run_dir, best["placement"], best["metrics"])
    write_header = not summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_latest_pointer(results_root: Path, run_dir: Path):
    """记录最近一次运行目录，方便快速找到最新结果。"""
    with (results_root / "latest_run.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_dir.name,
                "run_dir": str(run_dir.resolve()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    #========1. 读取网络拓扑=======
    config = default_config()
    results_root = Path(config["results_dir"])

    #========2. 打印实验配置摘要=======
    print("[toy] starting motivating example experiment")
    scenario_count = len(config["failure_scenarios"]) if config.get("failure_scenarios") else 2 ** len(config["failure_events"])
    scenario_source = "explicit_scenarios" if config.get("failure_scenarios") else "event_enumeration"
    print(
        f"[toy] beta={config['beta']}, tasks={len(config['tasks'])}, "
        f"scenario_source={scenario_source}, scenarios={scenario_count}, "
        f"solver_mode={config['solver_mode']}, "
        f"node_layer_risk_mode={config['node_layer_risk_mode']}, "
        f"node_layer_cvar_bound={config['node_layer_cvar_bound']}, "
        f"risk_mode={config['risk_mode']}, cvar_bound={config['cvar_bound']}"
    )

    #========3. 按配置求解模型=======
    solution_bundle = solve_toy(config)
    print(printable_summary(solution_bundle))

    #========4. 写出求解结果=======
    output_dir = _next_run_dir(results_root)
    output_dir = write_results(solution_bundle, output_dir)

    #========5. 绘制拓扑可视化=======
    png_path, svg_path = plot_toy_solution(config, solution_bundle, output_dir)

    #========6. 写出运行元数据与索引=======
    _copy_run_params(config, output_dir)
    _write_run_metadata(config, solution_bundle, output_dir)
    _append_summary(results_root, output_dir, solution_bundle)
    _write_latest_pointer(results_root, output_dir)

    print(f"[toy] results_dir={output_dir.resolve()}")
    print(f"[toy] summary_csv={(results_root / 'summary.csv').resolve()}")
    print(f"[toy] latest_run={(results_root / 'latest_run.json').resolve()}")
    print(f"[toy] figure_png={png_path.resolve()}")
    print(f"[toy] figure_svg={svg_path.resolve()}")


if __name__ == "__main__":
    main()
