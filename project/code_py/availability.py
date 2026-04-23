from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import List

import numpy as np

from Algorithms.teavar import teavar
from parsers import (
    get_tunnels,
    next_run,
    parse_paths,
    parse_yates_splitting_ratios,
    read_demand,
    read_topology,
)
from simulation import calculate_loss_reallocation, pdf
from util import sub_scenarios, weibull_probs

# 生成实验里的 scales 列表
def _collect_scales(start: float, step: float, finish: float) -> List[float]:
    if step == 0:
        raise ValueError("step must not be 0")

    start_d = Decimal(str(start))
    step_d = Decimal(str(step))
    finish_d = Decimal(str(finish))

    def _decimal_places(d: Decimal) -> int:
        return max(0, -d.as_tuple().exponent)

    places = max(_decimal_places(start_d), _decimal_places(step_d), _decimal_places(finish_d))
    quant = Decimal(1).scaleb(-places)

    vals: List[float] = []
    v = start_d
    eps = Decimal("1e-18")
    if step_d > 0:
        while v <= finish_d + eps:
            vals.append(float(v.quantize(quant)))
            v += step_d
    else:
        while v >= finish_d - eps:
            vals.append(float(v.quantize(quant)))
            v += step_d
    return vals


# 将一行数据以 tab 分隔的形式追加到文件中
def _append_tab_row(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("\t".join(str(v) for v in values))
        f.write("\n")


# 将矩阵数据写入文件，如果是二维矩阵则每行以 tab 分隔，如果是一维矩阵则每行一个值
def _write_matrix(path: Path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        arr = np.array(matrix, dtype=object)
        if arr.ndim == 1:
            for v in arr:
                f.write(f"{v}\n")
        else:
            for row in arr:
                f.write("\t".join(str(v) for v in row))
                f.write("\n")


def availability_plot(
    algorithmns,
    topologies,
    demand_downscales,
    num_demands,
    iterations,
    cutoff,
    start,
    step,
    finish,
    k=12,
    target=0,
    xliml=0.95,
    xlimr=1.0001,
    paths="KSP",
    weibull_scale=0.0001,
    plot=True,
    dirname="./data/raw/availability/",
):
    env = None

    # “按算法分组、每组存一条随 scale 变化的序列”
    availability_vals = [[] for _ in range(len(algorithmns))]

    dir_path = Path(next_run(dirname))
    for algorithmn in algorithmns:
        (dir_path / algorithmn).mkdir(parents=True, exist_ok=False)

    # scenarios_all[0][2] = B4 第3次迭代的场景集合；scenario_probs_all[1][0] = IBM 第1次迭代的场景概率向量
    scenarios_all = []
    scenario_probs_all = []

    for topology in topologies:
        links, capacity, link_probs, nodes = read_topology(topology)
        scenarios_all_top = []
        scenario_probs_top = []
        for _ in range(iterations):
            link_probs = weibull_probs(len(links), shape=0.8, scale=weibull_scale)
            scenarios, probs = sub_scenarios(link_probs, cutoff, first=True, last=False)
            scenarios_all_top.append(scenarios)
            scenario_probs_top.append(probs)
            print(probs)
        scenarios_all.append(scenarios_all_top)
        scenario_probs_all.append(scenario_probs_top)

    scales = _collect_scales(start, step, finish)
    confidence = np.zeros((len(scales), 3 * len(algorithmns) + 1), dtype=float)
    confidence[:, 0] = np.array(scales, dtype=float)

    total = len(scales) * len(topologies) * num_demands * iterations * len(algorithmns)
    done = 0

    for s_idx, scale in enumerate(scales):
        availabilities = [[] for _ in range(len(algorithmns))]
        for t, topology in enumerate(topologies):
            links, capacity, link_probs, nodes = read_topology(topology)
            for d in range(1, num_demands + 1):
                demand, flows = read_demand(f"{topology}/demand", len(nodes), d, scale=scale, downscale=demand_downscales[t])
                if paths != "KSP":
                    T, Tf, k_local = parse_paths(f"{topology}/paths/{paths}", links, flows)
                else:
                    T, Tf, _, _ = get_tunnels(nodes, links, capacity, flows, k)
                    k_local = k
                for i in range(iterations):
                    for alg in range(len(algorithmns)):
                        algorithm = algorithmns[alg]
                        if algorithm == "TEAVAR":
                            _, _, a, _ = teavar(
                                env,
                                links,
                                capacity,
                                flows,
                                demand,
                                scenario_probs_all[t][i][0] - 0.01,
                                k_local,
                                T,
                                Tf,
                                scenarios_all[t][i],
                                scenario_probs_all[t][i],
                                average=True,
                            )
                        elif algorithm == "ECMP":
                            a = np.ones((len(Tf), k_local), dtype=float)
                        elif algorithm in {"SMORE", "FFC-1", "FFC-2", "MaxMin"}:
                            raise NotImplementedError(f"{algorithm} is not ported in this Python rewrite yet.")
                        else:
                            T, Tf, k_local = parse_paths(f"{topology}/paths/{algorithm}", links, flows)
                            a = parse_yates_splitting_ratios(f"{topology}/paths/{algorithm}", k_local, flows)

                        losses = calculate_loss_reallocation(
                            links,
                            capacity,
                            demand,
                            flows,
                            T,
                            Tf,
                            k_local,
                            a,
                            scenarios_all[t][i],
                            scenario_probs_all[t][i],
                        )
                        print(losses)
                        _append_tab_row(dir_path / algorithm / f"{algorithm}_losses.txt", scenario_probs_all[t][i])
                        _append_tab_row(dir_path / algorithm / f"{algorithm}_losses.txt", losses)

                        availability = pdf(losses, scenario_probs_all[t][i], target)
                        print(availability)
                        availabilities[alg].append(availability)

                        done += 1
                        print(
                            f"[{done}/{total}] topology={topology} scale={scale} demand={d}/{num_demands} "
                            f"iteration={i + 1}/{iterations} algorithm={algorithm} availability={availability}"
                        )

        for alg in range(len(algorithmns)):
            mean_val = sum(availabilities[alg]) / (num_demands * iterations * len(topologies))
            confidence[s_idx, (alg) * 3 + 1] = mean_val
            confidence[s_idx, (alg) * 3 + 2] = min(availabilities[alg])
            confidence[s_idx, (alg) * 3 + 3] = max(availabilities[alg])
            _append_tab_row(dir_path / algorithmns[alg] / f"{algorithmns[alg]}_availabilities.txt", availabilities[alg])
            availability_vals[alg].append(mean_val)

    _write_matrix(dir_path / "availabilities", availability_vals)
    _write_matrix(dir_path / "scales", scales)

    params = [
        [
            "algorithmns",
            "topologies",
            "demand_downscales",
            "num_demands",
            "iterations",
            "cutoff",
            "scales",
            "k",
            "target",
            "paths",
            "weibull_scale",
        ],
        [
            json.dumps(algorithmns),
            json.dumps(topologies),
            json.dumps(demand_downscales),
            num_demands,
            iterations,
            cutoff,
            json.dumps(scales),
            k,
            target,
            paths if paths is not None else "",
            weibull_scale,
        ],
    ]
    _write_matrix(dir_path / "params", params)
    _write_matrix(dir_path / "confidence", confidence)

    if plot:
        try:
            import matplotlib.pyplot as plt

            plt.clf()
            for vals in availability_vals:
                plt.plot(vals, scales)
            plt.xlabel("Availability", fontweight="bold")
            plt.ylabel("Demand Scale", fontweight="bold")
            plt.xlim(left=xliml, right=xlimr)
            plt.legend(algorithmns, loc="upper right")
            plt.savefig(dir_path / "plot.png")
            plt.show()
        except Exception as ex:
            print(f"Plot skipped: {ex}")
