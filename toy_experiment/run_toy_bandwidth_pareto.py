"""Scan CVaR bounds and plot routing risk-resource Pareto frontiers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from run_toy_bandwidth_comparison import ROUTING_MODES, _case_params
from run_toy_comparison import (
    DATA_DIR,
    TOY_DIR,
    _format_value_for_name,
    _load_base_params,
    _next_batch_dir,
    _parse_float_list,
    _run_case,
)


DEFAULT_BOUNDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MODE_STYLE = {
    "mcf": {"label": "MCF", "color": "#d62728", "marker": "o"},
    "single_path": {"label": "Single path", "color": "#1f77b4", "marker": "s"},
    "ecmp": {"label": "ECMP", "color": "#2ca02c", "marker": "^"},
}


def _is_feasible(row: dict[str, Any]) -> bool:
    return row.get("status") in {"optimal", "suboptimal"} and row.get("eval_cvar") not in {None, ""}


def _mark_pareto_points(rows: list[dict[str, Any]], tolerance: float = 1e-9) -> None:
    """Mark nondominated points for lower CVaR and lower maximum link utilization."""
    for row in rows:
        row["is_pareto"] = False

    plotted_modes = [mode for mode in ROUTING_MODES if any(row.get("routing_mode") == mode for row in rows)]
    for mode in plotted_modes:
        feasible = [row for row in rows if row.get("routing_mode") == mode and _is_feasible(row)]
        for candidate in feasible:
            risk = float(candidate["eval_cvar"])
            resource = float(candidate["maximum_link_utilization"])
            dominated = any(
                float(other["eval_cvar"]) <= risk + tolerance
                and float(other["maximum_link_utilization"]) <= resource + tolerance
                and (
                    float(other["eval_cvar"]) < risk - tolerance
                    or float(other["maximum_link_utilization"]) < resource - tolerance
                )
                for other in feasible
                if other is not candidate
            )
            candidate["is_pareto"] = not dominated
        seen_points: set[tuple[float, float]] = set()
        for candidate in sorted(feasible, key=lambda row: float(row["link_cvar_bound"])):
            if not candidate["is_pareto"]:
                continue
            point = (
                round(float(candidate["eval_cvar"]), 9),
                round(float(candidate["maximum_link_utilization"]), 9),
            )
            if point in seen_points:
                candidate["is_pareto"] = False
            else:
                seen_points.add(point)


def _plot_bound_resource_curve(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    plotted_modes = [mode for mode in ROUTING_MODES if any(row.get("routing_mode") == mode for row in rows)]
    for mode in plotted_modes:
        style = MODE_STYLE[mode]
        mode_rows = sorted(
            [row for row in rows if row.get("routing_mode") == mode],
            key=lambda row: float(row["link_cvar_bound"]),
        )
        feasible = [row for row in mode_rows if _is_feasible(row)]

        if feasible:
            ax.plot(
                [float(row["link_cvar_bound"]) for row in feasible],
                [float(row["maximum_link_utilization"]) for row in feasible],
                color=style["color"],
                marker=style["marker"],
                markersize=7,
                linewidth=2.0,
                label=style["label"],
            )
    all_bounds = sorted({float(row["link_cvar_bound"]) for row in rows})
    ax.set_title("CVaR Constraint-Resource Tradeoff")
    ax.set_xlabel("CVaR constraint bound Γ")
    ax.set_ylabel("Maximum link utilization (lower is better)")
    ax.set_xticks(all_bounds)
    ax.set_xlim(min(all_bounds) - 0.03, max(all_bounds) + 0.03)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=False, loc="best")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "routing_cvar_bound_resource_curve.png"
    svg_path = output_dir / "routing_cvar_bound_resource_curve.svg"
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path

def _write_outputs(batch_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    csv_path = batch_dir / "summary.csv"
    json_path = batch_dir / "summary.json"
    batch_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Scan routing CVaR bounds and draw the risk-resource Pareto frontier.")
    parser.add_argument(
        "--link-bounds",
        type=_parse_float_list,
        default=DEFAULT_BOUNDS,
        help="Comma-separated CVaR bounds, e.g. 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=ROUTING_MODES,
        default=list(ROUTING_MODES),
        help="Routing modes to scan.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Write tables without rendering the Pareto figure.")
    args = parser.parse_args()

    bounds = sorted(set(float(value) for value in args.link_bounds))
    if not bounds or any(value <= 0.0 or value > 1.0 for value in bounds):
        raise ValueError("All CVaR bounds must be in (0, 1].")

    base_params = _load_base_params(DATA_DIR)
    batch_dir = _next_batch_dir(TOY_DIR / "results" / "bandwidth_pareto")
    rows: list[dict[str, Any]] = []

    for routing_mode in args.modes:
        for bound in bounds:
            case_name = f"single_{routing_mode}_link_{_format_value_for_name(bound)}"
            params = _case_params(base_params, routing_mode, bound)
            params["enumerate_equivalent_solutions"] = False
            print(f"[toy-pareto] running mode={routing_mode}, cvar_bound={bound:g}")
            row, _ = _run_case(
                case_name,
                params,
                batch_dir,
                make_plot=False,
                print_summary=False,
            )
            rows.append(row)

    _mark_pareto_points(rows)
    csv_path, json_path = _write_outputs(batch_dir, rows)
    print(f"[toy-pareto] summary_csv={csv_path.resolve()}")
    print(f"[toy-pareto] summary_json={json_path.resolve()}")

    if not args.no_plot:
        png_path, svg_path = _plot_bound_resource_curve(rows, batch_dir)
        print(f"[toy-pareto] bound_resource_png={png_path.resolve()}")
        print(f"[toy-pareto] bound_resource_svg={svg_path.resolve()}")


if __name__ == "__main__":
    main()
