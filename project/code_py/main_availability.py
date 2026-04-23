from __future__ import annotations

import argparse

from availability import availability_plot


def read_input(message, default, typ):
    try:
        raw = input(message)
        if typ is str:
            return raw if len(raw) > 0 else default
        if typ is bool:
            if len(raw.strip()) == 0:
                return default
            val = raw.strip().lower()
            if val in {"true", "1", "yes", "y"}:
                return True
            if val in {"false", "0", "no", "n"}:
                return False
            return default
        return typ(raw)
    except Exception:
        return default


def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", help="Type of experiment to run. Only supports: availability")
    parser.add_argument("--plot", "-p", action="store_true", help="Boolean whether or not to plot the results of the given experiment")
    return parser.parse_args()


def main():
    parsed = parse_commandline()
    print("Asking for inputs. Press enter for default values\n")

    experiment = parsed.experiment if parsed.experiment is not None else read_input("Experiment (availability): ", "availability", str)
    plot = parsed.plot or read_input("Plot results? (true): ", True, bool)

    print(f"\nRunning experiment [{experiment}] with plotting [{plot}]...")
    if experiment != "availability":
        raise ValueError(f"Unsupported experiment: {experiment}. This entrypoint only supports availability.")

    topologies = read_input("Topologies (B4): ", "B4", str).split(",")
    algorithms = read_input("Algorithms (TEAVAR): ", "TEAVAR", str).split(",")
    cutoff = read_input("Cutoff (0.0001): ", 0.0001, float)
    weibull_scale = read_input("Weibull scale (0.001): ", 0.001, float)
    num_demands = read_input("Number of demands (1): ", 1, int)
    iterations = read_input("Iterations (1): ", 1, int)
    demand_downscales = [float(i) for i in read_input("Demand downscales (2): ", "2", str).split(",")]
    start = read_input("Start (1): ", 1.0, float)
    step = read_input("Step (0.2): ", 0.2, float)
    finish = read_input("Finish (4.0): ", 4.0, float)

    availability_plot(
        algorithms,
        topologies,
        demand_downscales,
        num_demands,
        iterations,
        cutoff,
        start,
        step,
        finish,
        weibull_scale=weibull_scale,
        paths="KSP",
        k=12,
        plot=plot,
    )
    return "...Complete"


if __name__ == "__main__":
    main()
