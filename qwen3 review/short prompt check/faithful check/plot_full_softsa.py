from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from config import OUTPUT_ROOT


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _values(rows: list[dict[str, str]], x: str, y: str, difficulty: str | None = None):
    subset = [r for r in rows if difficulty is None or r["difficulty"] == difficulty]
    return np.asarray([float(r[x]) for r in subset]), np.asarray([float(r[y]) for r in subset])


def _overall_annotation(root: Path, rows: list[dict[str, str]], y_column: str) -> str:
    metrics_path = root / "tables" / "soft_explains_cma.csv"
    if metrics_path.exists():
        metrics = _load(metrics_path)
        overall = next(row for row in metrics if row["group"] == "overall")
        return (
            f"$R^2$ = {float(overall['r2']):.3f}\n"
            f"Pearson = {float(overall['pearson']):.3f}\n"
            f"Spearman = {float(overall['spearman']):.3f}"
        )
    x, y = _values(rows, y_column, "cma_signed")
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(stats.spearmanr(x, y).statistic)
    return f"$R^2$ = {pearson ** 2:.3f}\nPearson = {pearson:.3f}\nSpearman = {spearman:.3f}"


def plot(root: Path, output: Path, y_column: str, y_label: str) -> Path:
    rows = _load(root / "tables" / "case_level.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    palette = {"easy": "#2f78b7", "hard": "#d95f45"}
    fig, axis = plt.subplots(figsize=(6.4, 5.2))

    for difficulty in ("easy", "hard"):
        x, y = _values(rows, "cma_signed", y_column, difficulty)
        axis.scatter(x, y, alpha=.72, color=palette[difficulty], label=difficulty)
        if len(x) > 1 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(x.min(), x.max(), 100)
            axis.plot(grid, slope * grid + intercept, color=palette[difficulty])
    axis.axhline(0, color="0.75", linewidth=.8)
    axis.axvline(0, color="0.75", linewidth=.8)
    axis.set(
        xlabel="CMA signed (logit)",
        ylabel=y_label,
        xlim=(-1.05, 1.05),
        ylim=(-1.05, 1.05),
    )
    axis.text(
        0.03,
        0.97,
        _overall_annotation(root, rows, y_column),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 4},
    )
    axis.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT.parent / "faithful_check_extended" / "balanced_subset" / "full_softsa")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--y-column", default="full_sa_signed")
    parser.add_argument("--y-label", default="Full Soft-SA signed")
    args = parser.parse_args()
    destination = args.output or args.root / "plots" / "cma_vs_full_soft_sa.png"
    print(plot(args.root, destination, args.y_column, args.y_label))
