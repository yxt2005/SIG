"""Run toy single-level vs two-layer comparison experiments.

The comparison keeps the final true CVaR bound identical:
    - single_level: true joint-model CVaR <= Gamma
    - two_layer_average_split: link-layer true CVaR <= Gamma

For two-layer runs, the node layer uses an average-split proxy CVaR bound Gamma_N.
The script can scan several Gamma_N values and writes each case to results/runs/xxx.
It also writes a compact batch summary under results/comparisons/xxx/summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from plot_toy import plot_toy_solution, plot_toy_working_state, plot_toy_working_state_grid
from run_toy import (
    _append_summary,
    _copy_run_params,
    _next_run_dir,
    _write_latest_pointer,
    _write_run_metadata,
)
from solve_toy import printable_summary, solve_toy, write_results
from toy_config import default_config


TOY_DIR = Path(__file__).resolve().parent
DATA_DIR = TOY_DIR / "data"


def _parse_float_list(value: str) -> list[float]:
    """Parse comma-separated float values."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float list: {value}") from exc


def _load_base_params(data_dir: Path) -> dict[str, Any]:
    with (data_dir / "params.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_value_for_name(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _next_batch_dir(comparisons_root: Path) -> Path:
    """Return the next sequential comparison batch directory, e.g. comparisons/001."""
    comparisons_root.mkdir(parents=True, exist_ok=True)
    existing_ids = []
    for child in comparisons_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            existing_ids.append(int(child.name))
    return comparisons_root / f"{max(existing_ids, default=0) + 1:03d}"


def _case_params(base_params: dict[str, Any], solver_mode: str, link_bound: float, node_bound: float | None) -> dict[str, Any]:
    """Build params.json content for one comparison case."""
    params = dict(base_params)
    params["risk_mode"] = "cvar_constraint"
    params["cvar_bound"] = float(link_bound)
    params["solver_mode"] = solver_mode
    params["routing_mode"] = "mcf"

    if solver_mode == "single_level":
        params["node_layer_risk_mode"] = "weighted"
        params["node_layer_cvar_bound"] = None
    elif solver_mode == "two_layer_average_split":
        if node_bound is None:
            raise ValueError("node_bound is required for two_layer_average_split")
        params["node_layer_risk_mode"] = "cvar_constraint"
        params["node_layer_cvar_bound"] = float(node_bound)
    else:
        raise ValueError(f"Unsupported solver_mode: {solver_mode}")
    return params


def _case_plot_title(params: dict[str, Any]) -> str:
    """Build a compact title for working-state comparison panels."""
    link_bound = params.get("cvar_bound")
    if params["solver_mode"] == "single_level":
        routing_mode = params.get("routing_mode", "mcf")
        return f"Single ({routing_mode}), Γ_L={link_bound}"
    return f"Two-layer, Γ_N={params.get('node_layer_cvar_bound')}"


def _run_case(
    case_name: str,
    params: dict[str, Any],
    batch_dir: Path,
    make_plot: bool,
    print_summary: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run one toy experiment case and return one summary row."""
    with tempfile.TemporaryDirectory(prefix="toy_compare_") as tmp:
        temp_data_dir = Path(tmp) / "data"
        shutil.copytree(DATA_DIR, temp_data_dir)
        with (temp_data_dir / "params.json").open("w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

        config = default_config(temp_data_dir)
        results_root = Path(config["results_dir"])
        output_dir = _next_run_dir(results_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        row: dict[str, Any] = {
            "case_name": case_name,
            "solver_mode": params["solver_mode"],
            "routing_mode": params.get("routing_mode", "mcf"),
            "link_cvar_bound": params.get("cvar_bound"),
            "node_layer_cvar_bound": params.get("node_layer_cvar_bound"),
            "run_id": output_dir.name,
            "run_dir": str(output_dir.resolve()),
        }

        try:
            solution_bundle = solve_toy(config)
            if print_summary:
                print(printable_summary(solution_bundle))

            write_results(solution_bundle, output_dir)
            png_path = None
            svg_path = None
            working_png_path = None
            working_svg_path = None
            if make_plot:
                png_path, svg_path = plot_toy_solution(config, solution_bundle, output_dir)
                working_png_path, working_svg_path = plot_toy_working_state(config, solution_bundle, output_dir)

            _copy_run_params(config, output_dir)
            _write_run_metadata(config, solution_bundle, output_dir)
            _append_summary(results_root, output_dir, solution_bundle)
            _write_latest_pointer(results_root, output_dir)

            best = solution_bundle["best"]
            metrics = best["metrics"]
            row.update(
                {
                    "status": best["status"],
                    "reason": best["reason"],
                    "placement": ";".join(f"{task}->{node}" for task, node in best["placement"].items()),
                    "model_cvar": metrics.get("model_cvar"),


                    "eval_cvar": metrics.get("cvar"),
                    "cvar_bound_slack": metrics.get("cvar_bound_slack"),
                    "model_u_node_max": metrics.get("model_u_node_max"),
                    "model_u_link_max": metrics.get("model_u_link_max"),
                    "maximum_link_utilization": metrics.get("maximum_link_utilization"),
                    "node_layer_cvar": metrics.get("node_layer_cvar"),
                    "node_layer_cvar_bound_slack": metrics.get("node_layer_cvar_bound_slack"),
                    "node_layer_u_link_proxy": metrics.get("node_layer_u_link_proxy"),
                    "link_layer_cvar": metrics.get("link_layer_cvar"),
                    "total_reserved_bandwidth": metrics.get("total_reserved_bandwidth"),
                    "effective_throughput": metrics.get("effective_throughput"),
                    "min_loss": metrics.get("min_loss"),
                    "max_loss": metrics.get("max_loss"),
                    "expected_loss": metrics.get("expected_loss"),
                    "figure_png": "" if png_path is None else str(png_path.resolve()),
                    "figure_svg": "" if svg_path is None else str(svg_path.resolve()),
                    "working_state_png": "" if working_png_path is None else str(working_png_path.resolve()),
                    "working_state_svg": "" if working_svg_path is None else str(working_svg_path.resolve()),
                }
            )
            plot_item = {
                "title": _case_plot_title(params),
                "config": config,
                "solution_bundle": solution_bundle,
            }
        except Exception as exc:
            error = {"case_name": case_name, "error": repr(exc), "params": params}
            with (output_dir / "run_error.json").open("w", encoding="utf-8") as f:
                json.dump(error, f, ensure_ascii=False, indent=2)
            with (output_dir / "params.json").open("w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
            row.update({"status": "failed", "reason": repr(exc)})
            plot_item = None

        with (output_dir / "comparison_case.json").open("w", encoding="utf-8") as f:
            json.dump({"case_name": case_name, "batch_dir": str(batch_dir.resolve()), "params": params}, f, ensure_ascii=False, indent=2)
        return row, plot_item


def _write_batch_summary(batch_dir: Path, rows: list[dict[str, Any]]):
    batch_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with (batch_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare toy single-level and average-split two-layer models.")
    parser.add_argument("--link-bound", type=float, default=None, help="Final true CVaR bound Gamma. Defaults to data/params.json cvar_bound.")
    parser.add_argument(
        "--node-bounds",
        type=_parse_float_list,
        default=None,
        help="Comma-separated node-layer proxy CVaR bounds Gamma_N for two-layer runs, e.g. 0.565,0.6.",
    )
    parser.add_argument("--skip-single", action="store_true", help="Skip the single-level benchmark case.")
    parser.add_argument("--no-plot", action="store_true", help="Do not render topology figures for each case.")
    args = parser.parse_args()

    base_params = _load_base_params(DATA_DIR)
    link_bound = float(args.link_bound if args.link_bound is not None else base_params.get("cvar_bound", 0.1))
    node_bounds = args.node_bounds if args.node_bounds is not None else [float(base_params.get("node_layer_cvar_bound", 0.565))]

    batch_dir = _next_batch_dir(TOY_DIR / "results" / "comparisons")

    cases: list[tuple[str, dict[str, Any]]] = []
    if not args.skip_single:
        case_name = f"single_link_{_format_value_for_name(link_bound)}"
        cases.append((case_name, _case_params(base_params, "single_level", link_bound, None)))
    for node_bound in node_bounds:
        case_name = (
            f"two_layer_node_{_format_value_for_name(node_bound)}"
            f"_link_{_format_value_for_name(link_bound)}"
        )
        cases.append((case_name, _case_params(base_params, "two_layer_average_split", link_bound, node_bound)))

    rows = []
    plot_items = []
    for case_name, params in cases:
        print(f"[toy-compare] running {case_name}")
        row, plot_item = _run_case(case_name, params, batch_dir, make_plot=not args.no_plot)
        rows.append(row)
        if plot_item is not None:
            plot_items.append(plot_item)

    _write_batch_summary(batch_dir, rows)
    if plot_items and not args.no_plot:
        grid_png, grid_svg = plot_toy_working_state_grid(plot_items, batch_dir)
        print(f"[toy-compare] working_state_grid_png={grid_png.resolve()}")
        print(f"[toy-compare] working_state_grid_svg={grid_svg.resolve()}")
    print(f"[toy-compare] batch_summary={(batch_dir / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()
