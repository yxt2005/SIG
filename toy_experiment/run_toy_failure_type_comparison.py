"""Run toy experiments under four failure-scenario types.

输入：
    - toy_experiment/data/params.json 作为基础参数；
    - 四个故障场景文件：failure_events_1.json 到 failure_events_4.json。
输出：
    - toy_experiment/results/failure_type_comparisons/xxx/summary.csv；
    - toy_experiment/results/failure_type_comparisons/xxx/toy_failure_type_working_state_grid.png/svg。

执行方式：
    python toy_experiment/run_toy_failure_type_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from solve_toy import printable_summary, solve_toy, write_results
from toy_config import default_config


TOY_DIR = Path(__file__).resolve().parent
DATA_DIR = TOY_DIR / "data"


FAILURE_TYPE_CASES = [
    {
        "case_name": "none",
        "title": "No failure",
        "scenario_file": "failure_events_1.json",
    },
    {
        "case_name": "link_only",
        "title": "Link failures only",
        "scenario_file": "failure_events_2.json",
    },
    {
        "case_name": "node_only",
        "title": "Node failures only",
        "scenario_file": "failure_events_3.json",
    },
    {
        "case_name": "link_and_node",
        "title": "Link + node failures",
        "scenario_file": "failure_events_4.json",
    },
]


def _load_base_params() -> dict[str, Any]:
    """读取基础参数，批量实验只覆盖 failure_scenarios_file。"""
    with (DATA_DIR / "params.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _next_batch_dir(root: Path) -> Path:
    """生成 failure_type_comparisons/xxx 输出目录。"""
    root.mkdir(parents=True, exist_ok=True)
    existing_ids = []
    for child in root.iterdir():
        if child.is_dir() and child.name.isdigit():
            existing_ids.append(int(child.name))
    return root / f"{max(existing_ids, default=0) + 1:03d}"


def _next_run_dir(results_root: Path) -> Path:
    """生成 results/runs/xxx 输出目录。"""
    runs_root = results_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    existing_ids = []
    for child in runs_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            existing_ids.append(int(child.name))
    return runs_root / f"{max(existing_ids, default=0) + 1:03d}"


def _copy_run_inputs(config: dict, output_dir: Path):
    """复制本次运行的 params.json 和故障场景文件，便于复现。"""
    shutil.copy2(Path(config["params_path"]), output_dir / "params.json")
    scenario_path = Path(config["failure_scenarios_path"])
    if scenario_path.exists():
        shutil.copy2(scenario_path, output_dir / scenario_path.name)


def _write_run_metadata(config: dict, solution_bundle: dict, output_dir: Path):
    """写入本次运行的轻量索引。"""
    best = solution_bundle["best"]
    metadata = {
        "run_id": output_dir.name,
        "status": best["status"],
        "reason": best["reason"],
        "solver_mode": best["metrics"].get("solver_mode", config["solver_mode"]),
        "failure_scenarios_file": config["failure_scenarios_file"],
        "files": {
            "params": "params.json",
            "failure_scenarios": Path(config["failure_scenarios_path"]).name,
            "placements": "placements.json",
            "metrics": "metrics.csv",
            "path_allocations": "path_allocations.csv",
            "scenario_losses": "scenario_losses.csv",
            "link_loads": "link_loads.csv",
            "equivalent_solutions": "equivalent_solutions.csv",
            "equivalent_path_allocations": "equivalent_path_allocations.csv",
            "figure_png": "toy_topology_solution.png",
            "figure_svg": "toy_topology_solution.svg",
        },
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _case_params(base_params: dict[str, Any], scenario_file: str) -> dict[str, Any]:
    """构造单个故障类型 case 的 params.json。"""
    params = dict(base_params)
    params["failure_scenarios_file"] = scenario_file
    return params


def _summary_row(case: dict[str, str], params: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """初始化一行批量实验汇总记录。"""
    return {
        "case_name": case["case_name"],
        "failure_type": case["title"],
        "failure_scenarios_file": case["scenario_file"],
        "solver_mode": params.get("solver_mode"),
        "cvar_bound": params.get("cvar_bound"),
        "node_layer_cvar_bound": params.get("node_layer_cvar_bound"),
        "run_id": output_dir.name,
        "run_dir": str(output_dir.resolve()),
    }


def _run_case(case: dict[str, str], params: dict[str, Any], batch_dir: Path, make_plot: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """运行一个故障类型 case，并返回汇总行和网格绘图 item。"""
    with tempfile.TemporaryDirectory(prefix="toy_failure_type_") as tmp:
        temp_data_dir = Path(tmp) / "data"
        shutil.copytree(DATA_DIR, temp_data_dir)
        with (temp_data_dir / "params.json").open("w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

        config = default_config(temp_data_dir)
        results_root = Path(config["results_dir"])
        output_dir = _next_run_dir(results_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        row = _summary_row(case, params, output_dir)

        try:
            solution_bundle = solve_toy(config)
            print(printable_summary(solution_bundle))
            write_results(solution_bundle, output_dir)

            figure_png = None
            figure_svg = None
            if make_plot:
                try:
                    from plot_toy import plot_toy_solution

                    figure_png, figure_svg = plot_toy_solution(config, solution_bundle, output_dir)
                except ModuleNotFoundError as exc:
                    print(f"[toy-failure-types] skip per-case figure: {exc}")

            _copy_run_inputs(config, output_dir)
            _write_run_metadata(config, solution_bundle, output_dir)

            best = solution_bundle["best"]
            metrics = best["metrics"]
            row.update(
                {
                    "status": best.get("status"),
                    "reason": best.get("reason"),
                    "placement": ";".join(f"{task}->{node}" for task, node in best["placement"].items()),
                    "eval_cvar": metrics.get("cvar"),
                    "model_u_node_max": metrics.get("model_u_node_max"),
                    "model_u_link_max": metrics.get("model_u_link_max"),
                    "total_reserved_bandwidth": metrics.get("total_reserved_bandwidth"),
                    "effective_throughput": metrics.get("effective_throughput"),
                    "min_loss": metrics.get("min_loss"),
                    "max_loss": metrics.get("max_loss"),
                    "equivalent_solution_count": metrics.get("equivalent_solution_count"),
                    "figure_png": "" if figure_png is None else str(figure_png.resolve()),
                    "figure_svg": "" if figure_svg is None else str(figure_svg.resolve()),
                }
            )
            plot_item = {
                "title": case["title"],
                "config": config,
                "solution_bundle": solution_bundle,
            }
        except Exception as exc:
            reason = repr(exc)
            error = {"case": case, "error": reason, "params": params}
            with (output_dir / "run_error.json").open("w", encoding="utf-8") as f:
                json.dump(error, f, ensure_ascii=False, indent=2)
            with (output_dir / "params.json").open("w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
            row.update({"status": "failed", "reason": reason})
            plot_item = {
                "title": case["title"],
                "config": None,
                "solution_bundle": None,
                "reason": reason,
            }

        with (output_dir / "failure_type_case.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "case_name": case["case_name"],
                    "batch_dir": str(batch_dir.resolve()),
                    "params": params,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return row, plot_item


def _write_batch_summary(batch_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace):
    """写出四类故障场景对比实验汇总。"""
    batch_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        with (batch_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    with (batch_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "rows": rows}, f, ensure_ascii=False, indent=2)


def main():
    #========1. 读取网络拓扑=======
    parser = argparse.ArgumentParser(description="Run toy experiments across four failure-scenario types.")
    parser.add_argument("--no-plot", action="store_true", help="Do not render figures.")
    args = parser.parse_args()

    #========2. 生成四类故障场景 case=======
    base_params = _load_base_params()
    batch_dir = _next_batch_dir(TOY_DIR / "results" / "failure_type_comparisons")

    #========3. 逐类故障场景求解=======
    rows = []
    plot_items = []
    for case in FAILURE_TYPE_CASES:
        params = _case_params(base_params, case["scenario_file"])
        print(f"[toy-failure-types] running {case['case_name']} with {case['scenario_file']}")
        row, plot_item = _run_case(case, params, batch_dir, make_plot=not args.no_plot)
        rows.append(row)
        plot_items.append(plot_item)

    #========4. 写出汇总表与四宫格结果图=======
    _write_batch_summary(batch_dir, rows, args)
    if not args.no_plot:
        try:
            from plot_toy import plot_toy_working_state_grid

            grid_png, grid_svg = plot_toy_working_state_grid(
                plot_items,
                batch_dir,
                stem="toy_failure_type_working_state_grid",
            )
            print(f"[toy-failure-types] grid_png={grid_png.resolve()}")
            print(f"[toy-failure-types] grid_svg={grid_svg.resolve()}")
        except ModuleNotFoundError as exc:
            print(f"[toy-failure-types] skip grid figure: {exc}")
    print(f"[toy-failure-types] batch_summary={(batch_dir / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()
