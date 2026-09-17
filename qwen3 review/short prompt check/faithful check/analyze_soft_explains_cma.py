from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from config import BOOTSTRAP_REPEATS, OUTPUT_ROOT, SEED


VARIANTS = {
    "short": {
        "soft_column": "sa_signed",
        "x_label": "Short Soft-SA signed",
        "figure": "soft_sa_vs_cma.png",
    },
    "reverse_short": {
        "soft_column": "reverse_sa_signed",
        "x_label": "Reverse Short Soft-SA signed",
        "figure": "reverse_soft_sa_vs_cma.png",
    },
    "full": {
        "soft_column": "full_sa_signed",
        "x_label": "Full Soft-SA signed",
        "figure": "full_soft_sa_vs_cma.png",
    },
}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _pairs(rows: list[dict[str, str]], soft_column: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (float(row[soft_column]), float(row["cma_signed"]))
        for row in rows
        if row.get(soft_column) not in (None, "") and row.get("cma_signed") not in (None, "")
    ]
    if not pairs:
        return np.array([]), np.array([])
    return np.asarray([pair[0] for pair in pairs]), np.asarray([pair[1] for pair in pairs])


def _correlation(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float | None, float | None]:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None
    result = stats.pearsonr(x, y) if kind == "pearson" else stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _fit(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    if len(x) < 2 or np.ptp(x) == 0:
        return {key: None for key in ("intercept", "slope", "r2", "mae", "rmse")}
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ np.asarray([intercept, slope])
    residual = y - prediction
    total = np.sum((y - y.mean()) ** 2)
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": float(1 - np.sum(residual ** 2) / total) if total > 0 else None,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
    }


def _bootstrap_ci(
    rows: list[dict[str, str]],
    statistic: Callable[[list[dict[str, str]]], float],
    repeats: int,
) -> tuple[float | None, float | None]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["item_id"])].append(row)
    item_ids = sorted(grouped)
    if len(item_ids) < 2:
        return None, None
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(repeats):
        sampled_ids = rng.choice(item_ids, size=len(item_ids), replace=True)
        sampled_rows = [row for item_id in sampled_ids for row in grouped[str(item_id)]]
        value = statistic(sampled_rows)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None, None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _metrics(
    rows: list[dict[str, str]],
    variant: str,
    group: str,
    soft_column: str,
    repeats: int,
) -> dict[str, Any]:
    x, y = _pairs(rows, soft_column)
    pearson, pearson_p = _correlation(x, y, "pearson")
    spearman, spearman_p = _correlation(x, y, "spearman")
    fit = _fit(x, y)

    def sampled_correlation(sample: list[dict[str, str]], kind: str) -> float:
        sample_x, sample_y = _pairs(sample, soft_column)
        value, _ = _correlation(sample_x, sample_y, kind)
        return float("nan") if value is None else value

    def sampled_slope(sample: list[dict[str, str]]) -> float:
        sample_x, sample_y = _pairs(sample, soft_column)
        value = _fit(sample_x, sample_y)["slope"]
        return float("nan") if value is None else float(value)

    pearson_ci = _bootstrap_ci(rows, lambda sample: sampled_correlation(sample, "pearson"), repeats)
    spearman_ci = _bootstrap_ci(rows, lambda sample: sampled_correlation(sample, "spearman"), repeats)
    slope_ci = _bootstrap_ci(rows, sampled_slope, repeats)
    nonzero = (x != 0) & (y != 0)
    return {
        "variant": variant,
        "group": group,
        "regression": "cma_signed ~ soft_sa_signed",
        "n": len(x),
        "pearson": pearson,
        "pearson_p_value": pearson_p,
        "pearson_ci_low": pearson_ci[0],
        "pearson_ci_high": pearson_ci[1],
        "spearman": spearman,
        "spearman_p_value": spearman_p,
        "spearman_ci_low": spearman_ci[0],
        "spearman_ci_high": spearman_ci[1],
        **fit,
        "slope_ci_low": slope_ci[0],
        "slope_ci_high": slope_ci[1],
        "sign_agreement_rate": (
            float(np.mean(np.sign(x[nonzero]) == np.sign(y[nonzero]))) if nonzero.any() else None
        ),
    }


def _plot(
    rows: list[dict[str, str]],
    soft_column: str,
    x_label: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    colors = {"easy": "#2f78b7", "hard": "#d95f45"}
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    for difficulty in ("easy", "hard"):
        subset = [row for row in rows if row["difficulty"] == difficulty]
        x, y = _pairs(subset, soft_column)
        axis.scatter(x, y, alpha=.72, label=difficulty, color=colors[difficulty])
        if len(x) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(x.min(), x.max(), 100)
            axis.plot(grid, slope * grid + intercept, color=colors[difficulty])
    axis.axhline(0, color="0.75", linewidth=.8)
    axis.axvline(0, color="0.75", linewidth=.8)
    axis.set(
        xlabel=x_label,
        ylabel="CMA signed (logit)",
        xlim=(-1.05, 1.05),
        ylim=(-1.05, 1.05),
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def analyze(root: Path, repeats: int) -> list[Path]:
    outputs: list[Path] = []
    for variant, specification in VARIANTS.items():
        table_directory = root / "tables" / variant
        figure_directory = root / "figures" / variant
        rows = _load_csv(table_directory / "case_level.csv")
        metrics = []
        for group in ("overall", "easy", "hard"):
            subset = rows if group == "overall" else [row for row in rows if row["difficulty"] == group]
            metrics.append(
                _metrics(subset, variant, group, specification["soft_column"], repeats)
            )
        table_path = table_directory / "soft_explains_cma.csv"
        _atomic_csv(table_path, metrics)
        figure_path = figure_directory / specification["figure"]
        _plot(rows, specification["soft_column"], specification["x_label"], figure_path)
        outputs.extend([table_path, figure_path])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regress CMA on each Soft-SA variant")
    parser.add_argument(
        "--root",
        type=Path,
        default=(
            OUTPUT_ROOT.parent
            / "faithful_check_extended"
            / "balanced_subset"
            / "final_results"
        ),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for output in analyze(arguments.root, arguments.bootstrap_repeats):
        print(output)
