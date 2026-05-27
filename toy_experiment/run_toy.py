"""Toy 实验主入口。

输入：
    - toy_experiment/data 下的配置信息。
输出：
    - toy_experiment/results/runs/xxx 下的求解结果 CSV/JSON 与拓扑图。

执行方式：
    python toy_experiment/run_toy.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from plot_toy import plot_toy_solution
from solve_toy import printable_summary, solve_toy, write_results
from toy_config import default_config


def _next_run_dir(results_root: Path) -> Path:
    """生成下一个 runs/xxx 输出目录，避免覆盖历史实验结果。"""
    runs_root = results_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    existing_ids = []
    for child in runs_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            existing_ids.append(int(child.name))
    next_id = max(existing_ids, default=0) + 1
    return runs_root / f"{next_id:03d}"


def _write_run_metadata(config: dict, solution_bundle: dict, run_dir: Path):
    """写入本次运行的关键配置、选址结果和指标，便于之后复现实验。"""
    best = solution_bundle["best"]
    metadata = {
        "run_id": run_dir.name,
        "risk_mode": config["risk_mode"],
        "cvar_bound": config["cvar_bound"],
        "beta": config["beta"],
        "beta_mode": config["beta_mode"],
        "beta_margin": config["beta_margin"],
        "normal_probability": config["normal_probability"],
        "lambda_weight": config["lambda_weight"],
        "risk_weight": config["risk_weight"],
        "loss_aggregation": config["loss_aggregation"],
        "placement": best["placement"],
        "metrics": best["metrics"],
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _append_summary(results_root: Path, run_dir: Path, solution_bundle: dict):
    """把本次运行的核心指标追加到 results/summary.csv。"""
    best = solution_bundle["best"]
    metrics = best["metrics"]
    row = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "best_placement": ";".join(f"{k}->{v}" for k, v in best["placement"].items()),
        "risk_mode": metrics.get("risk_mode"),
        "cvar_bound": metrics.get("cvar_bound"),
        "beta": metrics.get("beta"),
        "beta_mode": metrics.get("beta_mode"),
        "model_cvar": metrics.get("model_cvar"),
        "cvar_bound_slack": metrics.get("cvar_bound_slack"),
        "model_u_node_max": metrics.get("model_u_node_max"),
        "model_u_link_max": metrics.get("model_u_link_max"),
        "total_reserved_bandwidth": metrics.get("total_reserved_bandwidth"),
        "availability": metrics.get("availability"),
        "expected_loss": metrics.get("expected_loss"),
        "max_loss": metrics.get("max_loss"),
        "solver_status": metrics.get("solver_status"),
    }
    summary_path = results_root / "summary.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_latest_pointer(results_root: Path, run_dir: Path):
    """记录最近一次运行目录，方便脚本或人工快速找到最新结果。"""
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
        f"risk_mode={config['risk_mode']}, cvar_bound={config['cvar_bound']}"
    )

    #========3. 求解单层 MILP=======
    solution_bundle = solve_toy(config)
    print(printable_summary(solution_bundle))

    #========4. 写出求解结果=======
    output_dir = _next_run_dir(results_root)
    output_dir = write_results(solution_bundle, output_dir)

    #========5. 绘制拓扑可视化=======
    png_path, svg_path = plot_toy_solution(config, solution_bundle, output_dir)

    #========6. 写出运行元数据与索引=======
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
