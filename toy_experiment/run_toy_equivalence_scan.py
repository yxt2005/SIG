"""Scan toy-model parameter settings for equivalent optimal solutions.

输入：
    - toy_experiment/data/params.json 作为基础配置；
    - 命令行给定的链路层 CVaR 约束 Gamma_L 与节点层约束 Gamma_N。
输出：
    - toy_experiment/results/equivalence_scans/xxx/summary.csv；
    - toy_experiment/results/equivalence_scans/xxx/summary.json。

执行方式：
    python toy_experiment/run_toy_equivalence_scan.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from solve_toy import solve_toy
from toy_config import default_config


TOY_DIR = Path(__file__).resolve().parent
DATA_DIR = TOY_DIR / "data"


def _parse_float_list(value: str) -> list[float]:
    """解析逗号分隔的浮点数列表，例如 0.1,0.3,0.5。"""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float list: {value}") from exc


def _parse_mode_list(value: str) -> list[str]:
    """解析需要扫描的模型类型。"""
    mode_alias = {
        "single": "single_level",
        "single_level": "single_level",
        "two": "two_layer_average_split",
        "two_layer": "two_layer_average_split",
        "two_layer_average_split": "two_layer_average_split",
    }
    modes = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        if item not in mode_alias:
            raise argparse.ArgumentTypeError(f"unsupported mode: {item}")
        mapped = mode_alias[item]
        if mapped not in modes:
            modes.append(mapped)
    if not modes:
        raise argparse.ArgumentTypeError("expected at least one mode")
    return modes


def _load_base_params() -> dict[str, Any]:
    """读取基础 params.json，扫参时只覆盖必要字段。"""
    with (DATA_DIR / "params.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _next_scan_dir(scans_root: Path) -> Path:
    """生成 equivalence_scans/xxx 输出目录，避免覆盖历史扫描结果。"""
    scans_root.mkdir(parents=True, exist_ok=True)
    existing_ids = []
    for child in scans_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            existing_ids.append(int(child.name))
    return scans_root / f"{max(existing_ids, default=0) + 1:03d}"


def _case_params(base_params: dict[str, Any], solver_mode: str, link_bound: float, node_bound: float | None) -> dict[str, Any]:
    """构造一个扫参 case 的 params.json 内容。"""
    params = dict(base_params)
    params["solver_mode"] = solver_mode
    params["risk_mode"] = "cvar_constraint"
    params["cvar_bound"] = float(link_bound)
    params["enumerate_equivalent_solutions"] = True

    if solver_mode == "single_level":
        params["node_layer_risk_mode"] = "weighted"
        params["node_layer_cvar_bound"] = None
    elif solver_mode == "two_layer_average_split":
        if node_bound is None:
            raise ValueError("node_bound is required for two-layer scan cases")
        params["node_layer_risk_mode"] = "cvar_constraint"
        params["node_layer_cvar_bound"] = float(node_bound)
    else:
        raise ValueError(f"Unsupported solver_mode: {solver_mode}")
    return params


def _placement_text(placement: dict[str, str]) -> str:
    """把任务放置结果压缩成一行文本，便于 CSV 查看。"""
    return ";".join(f"{task}->{node}" for task, node in sorted(placement.items()))


def _equivalent_placements(solution_bundle: dict) -> list[str]:
    """提取等价解中的任务放置集合。"""
    placements = []
    for result in solution_bundle.get("all_results", []):
        placement = _placement_text(result.get("placement", {}))
        if placement not in placements:
            placements.append(placement)
    return placements


def _run_case(case_id: int, params: dict[str, Any]) -> dict[str, Any]:
    """运行一个参数组合，并返回汇总行；不可行 case 会记录失败原因。"""
    with tempfile.TemporaryDirectory(prefix="toy_equiv_scan_") as tmp:
        temp_data_dir = Path(tmp) / "data"
        shutil.copytree(DATA_DIR, temp_data_dir)
        with (temp_data_dir / "params.json").open("w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

        row: dict[str, Any] = {
            "case_id": case_id,
            "solver_mode": params["solver_mode"],
            "link_cvar_bound": params.get("cvar_bound"),
            "node_layer_cvar_bound": params.get("node_layer_cvar_bound"),
        }
        try:
            config = default_config(temp_data_dir)
            solution_bundle = solve_toy(config)
            best = solution_bundle["best"]
            metrics = best["metrics"]
            placements = _equivalent_placements(solution_bundle)
            row.update(
                {
                    "status": best.get("status", "ok"),
                    "reason": best.get("reason", ""),
                    "equivalent_solution_count": metrics.get("equivalent_solution_count", len(placements)),
                    "equivalent_node_solution_count": metrics.get("equivalent_node_solution_count", ""),
                    "best_placement": _placement_text(best.get("placement", {})),
                    "equivalent_placements": "|".join(placements),
                    "objective_value": metrics.get("objective_value"),
                    "model_cvar": metrics.get("model_cvar"),
                    "eval_cvar": metrics.get("cvar"),
                    "model_u_node_max": metrics.get("model_u_node_max"),
                    "model_u_link_max": metrics.get("model_u_link_max"),
                    "total_reserved_bandwidth": metrics.get("total_reserved_bandwidth"),
                    "effective_throughput": metrics.get("effective_throughput"),
                    "min_loss": metrics.get("min_loss"),
                    "max_loss": metrics.get("max_loss"),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "reason": repr(exc),
                    "equivalent_solution_count": "",
                    "equivalent_node_solution_count": "",
                    "best_placement": "",
                    "equivalent_placements": "",
                }
            )
        return row


def _write_summary(scan_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace):
    """写出 CSV 与 JSON 两种格式的扫参结果。"""
    scan_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with (scan_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (scan_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "rows": rows}, f, ensure_ascii=False, indent=2)


def main():
    #========1. 读取网络拓扑=======
    parser = argparse.ArgumentParser(description="Scan toy equivalent optimal solutions across CVaR bounds.")
    parser.add_argument("--link-bounds", type=_parse_float_list, default=[0.1, 0.3, 0.5], help="Gamma_L list, e.g. 0.1,0.3,0.5.")
    parser.add_argument("--node-bounds", type=_parse_float_list, default=[0.565, 0.6, 0.7, 0.8], help="Gamma_N list for two-layer cases.")
    parser.add_argument("--modes", type=_parse_mode_list, default=["single_level", "two_layer_average_split"], help="Modes to scan: single,two_layer.")
    args = parser.parse_args()

    #========2. 生成扫参 case=======
    base_params = _load_base_params()
    cases: list[dict[str, Any]] = []
    for link_bound in args.link_bounds:
        if "single_level" in args.modes:
            cases.append(_case_params(base_params, "single_level", link_bound, None))
        if "two_layer_average_split" in args.modes:
            for node_bound in args.node_bounds:
                cases.append(_case_params(base_params, "two_layer_average_split", link_bound, node_bound))

    #========3. 逐个求解并记录等价解数量=======
    rows = []
    for case_id, params in enumerate(cases, start=1):
        print(
            "[toy-equiv-scan] running "
            f"case={case_id}, mode={params['solver_mode']}, "
            f"Gamma_L={params.get('cvar_bound')}, Gamma_N={params.get('node_layer_cvar_bound')}"
        )
        rows.append(_run_case(case_id, params))

    #========4. 写出扫参汇总结果=======
    scan_dir = _next_scan_dir(TOY_DIR / "results" / "equivalence_scans")
    _write_summary(scan_dir, rows, args)
    print(f"[toy-equiv-scan] summary_csv={(scan_dir / 'summary.csv').resolve()}")
    print(f"[toy-equiv-scan] summary_json={(scan_dir / 'summary.json').resolve()}")


if __name__ == "__main__":
    main()
