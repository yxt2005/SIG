"""Compare the three single-level bandwidth-allocation baselines.

The placement, risk model, failure scenarios, and final CVaR bound are held
constant. Only routing_mode changes: mcf, single_path, or ecmp.
"""

from __future__ import annotations

import argparse
from typing import Any

from plot_toy import plot_toy_working_state_grid
from run_toy_comparison import (
    DATA_DIR,
    TOY_DIR,
    _load_base_params,
    _next_batch_dir,
    _run_case,
    _write_batch_summary,
)


ROUTING_MODES = ("mcf", "single_path", "ecmp")


def _case_params(base_params: dict[str, Any], routing_mode: str, link_bound: float) -> dict[str, Any]:
    params = dict(base_params)
    params.update(
        {
            "solver_mode": "single_level",
            "routing_mode": routing_mode,
            "risk_mode": "cvar_constraint",
            "cvar_bound": float(link_bound),
            "node_layer_risk_mode": "weighted",
            "node_layer_cvar_bound": None,
        }
    )
    return params


def main():
    parser = argparse.ArgumentParser(description="Compare MCF, single-path, and ECMP routing in the toy single-level model.")
    parser.add_argument(
        "--link-bound",
        type=float,
        default=None,
        help="Final true CVaR bound. Defaults to data/params.json cvar_bound.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=ROUTING_MODES,
        default=list(ROUTING_MODES),
        help="Routing baselines to run.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not render topology figures.")
    args = parser.parse_args()

    base_params = _load_base_params(DATA_DIR)
    link_bound = float(args.link_bound if args.link_bound is not None else base_params.get("cvar_bound", 0.1))
    batch_dir = _next_batch_dir(TOY_DIR / "results" / "bandwidth_comparisons")

    rows = []
    plot_items = []
    for routing_mode in args.modes:
        case_name = f"single_{routing_mode}_link_{str(link_bound).replace('.', 'p')}"
        params = _case_params(base_params, routing_mode, link_bound)
        print(f"[toy-bandwidth] running {case_name}")
        row, plot_item = _run_case(case_name, params, batch_dir, make_plot=not args.no_plot)
        rows.append(row)
        if plot_item is not None:
            plot_items.append(plot_item)

    _write_batch_summary(batch_dir, rows)
    if plot_items and not args.no_plot:
        grid_png, grid_svg = plot_toy_working_state_grid(plot_items, batch_dir)
        print(f"[toy-bandwidth] working_state_grid_png={grid_png.resolve()}")
        print(f"[toy-bandwidth] working_state_grid_svg={grid_svg.resolve()}")
    print(f"[toy-bandwidth] batch_summary={(batch_dir / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()
