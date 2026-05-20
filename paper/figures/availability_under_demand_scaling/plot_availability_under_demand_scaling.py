from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "project").is_dir() and (parent / "paper").is_dir():
            return parent
    raise RuntimeError("Cannot locate project root.")


ROOT = find_project_root()
OUT_DIR = Path(__file__).resolve().parent

SCALES = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
ALGORITHM_LABELS = {
    "A": "Single-level",
    "layered3": "Two-layer",
    "A_no_failure": "Failure-agnostic",
}


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def collect_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    # 2.0 的 Single-level 和 Two-layer 使用最新重跑结果。
    for row in read_summary(ROOT / "project/data/raw/availability/12/summary.csv"):
        scale = round(float(row["scale"]), 1)
        if scale == 2.0 and row["algorithm"] in {"A", "layered3"}:
            rows.append(
                {
                    "algorithm": row["algorithm"],
                    "label": ALGORITHM_LABELS[row["algorithm"]],
                    "scale": scale,
                    "availability_percent": 100.0 * float(row["availability"]),
                    "source": "availability/12",
                }
            )

    # 2.2 和 2.4 的 Single-level 和 Two-layer 由补跑实验提供。
    for row in read_summary(ROOT / "project/data/raw/availability/10/summary.csv"):
        scale = round(float(row["scale"]), 1)
        if scale in {2.2, 2.4} and row["algorithm"] in {"A", "layered3"}:
            rows.append(
                {
                    "algorithm": row["algorithm"],
                    "label": ALGORITHM_LABELS[row["algorithm"]],
                    "scale": scale,
                    "availability_percent": 100.0 * float(row["availability"]),
                    "source": "availability/10",
                }
            )

    # 2.6 到 4.0 的 Single-level 和 Two-layer 来自主扫参实验。
    for row in read_summary(ROOT / "project/data/raw/availability/9/summary.csv"):
        scale = round(float(row["scale"]), 1)
        if scale in set(SCALES[3:]) and row["algorithm"] in {"A", "layered3"}:
            rows.append(
                {
                    "algorithm": row["algorithm"],
                    "label": ALGORITHM_LABELS[row["algorithm"]],
                    "scale": scale,
                    "availability_percent": 100.0 * float(row["availability"]),
                    "source": "availability/9",
                }
            )

    # Failure-agnostic baseline 来自单独实验。
    for row in read_summary(ROOT / "project/data/raw/availability/11/summary.csv"):
        scale = round(float(row["scale"]), 1)
        if scale in set(SCALES) and row["algorithm"] == "A_no_failure":
            rows.append(
                {
                    "algorithm": row["algorithm"],
                    "label": ALGORITHM_LABELS[row["algorithm"]],
                    "scale": scale,
                    "availability_percent": 100.0 * float(row["availability"]),
                    "source": "availability/11",
                }
            )

    return sorted(rows, key=lambda item: (float(item["scale"]), str(item["algorithm"])))


def write_plot_data(rows: list[dict[str, float | str]]) -> Path:
    out_path = OUT_DIR / "availability_under_demand_scaling_data.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scale", "algorithm", "label", "availability_percent", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def plot(rows: list[dict[str, float | str]]) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    styles = {
        "A": {"color": "#1f77b4", "marker": "o", "linewidth": 2.0},
        "layered3": {"color": "#d62728", "marker": "s", "linewidth": 2.0},
        "A_no_failure": {"color": "#2ca02c", "marker": "^", "linewidth": 1.9},
    }

    fig, ax = plt.subplots(figsize=(6.1, 3.35), constrained_layout=True)

    for algorithm in ["A", "layered3", "A_no_failure"]:
        series = [row for row in rows if row["algorithm"] == algorithm]
        xs = [float(row["scale"]) for row in series]
        ys = [float(row["availability_percent"]) for row in series]
        ax.plot(
            xs,
            ys,
            label=ALGORITHM_LABELS[algorithm],
            markersize=5.0,
            **styles[algorithm],
        )

    ax.set_xlabel("Demand Scale")
    ax.set_ylabel("Availability (%)")
    ax.set_xlim(1.95, 4.05)
    ax.set_ylim(96.5, 100.1)
    ax.set_xticks([2.0, 2.4, 2.8, 3.2, 3.6, 4.0])
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
    )

    png_path = OUT_DIR / "availability_under_demand_scaling.png"
    pdf_path = OUT_DIR / "availability_under_demand_scaling.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    rows = collect_rows()
    expected = {(algorithm, scale) for algorithm in ALGORITHM_LABELS for scale in SCALES}
    actual = {(str(row["algorithm"]), float(row["scale"])) for row in rows}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"Missing plotting rows: {missing}")
    data_path = write_plot_data(rows)
    png_path, pdf_path = plot(rows)
    print(f"data: {data_path}")
    print(f"png:  {png_path}")
    print(f"pdf:  {pdf_path}")


if __name__ == "__main__":
    main()
