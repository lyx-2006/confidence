from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
for candidate in (REPOSITORY_ROOT, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
from scipy import stats

from dp_SA.io_utils import atomic_json, load_jsonl
from config import BOOTSTRAP_REPEATS, OUTPUT_ROOT, SEED


def _json_clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    return value


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
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


def _case_row(trial: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    primary = trial["cma_logit"]
    secondary = trial["cma_log_probability"]
    sa = trial["short_sa"]
    text_gate = manifest["original_text"]
    image_gate = manifest["original_image_gate"]
    return {
        "case_id": trial["case_id"], "item_id": trial["item_id"],
        "difficulty": trial["difficulty"], "text_color": trial["text_color"],
        "image_color": trial["image_color"], "third_color": trial["third_color"],
        "fixed_answer": trial["fixed_answer"],
        "cma_signed": primary["cma_signed"],
        "cma_identifiable": primary["identifiable"],
        "phi_image": primary["phi_image"], "phi_text": primary["phi_text"],
        "interaction": primary["interaction"],
        "cma_log_probability_signed": secondary["cma_signed"],
        "cma_log_probability_identifiable": secondary["identifiable"],
        "sa_raw": sa["soft_sa_image_score"], "sa_signed": sa["soft_sa_signed"],
        "sa_hard_class": sa["argmax_hard_class"],
        "original_text_entropy": text_gate["normalized_entropy"],
        "original_image_entropy": image_gate["normalized_entropy"],
        "text_entropy_match_error": manifest["text_match_deltas"]["entropy_delta"],
        "text_probability_match_error": manifest["text_match_deltas"]["target_probability_delta"],
    }


def _paired(rows: list[dict[str, Any]], x_key: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (float(row[x_key]), float(row["sa_signed"])) for row in rows
        if row.get(x_key) is not None and row.get("sa_signed") is not None
        and math.isfinite(float(row[x_key])) and math.isfinite(float(row["sa_signed"]))
    ]
    if not pairs:
        return np.array([]), np.array([])
    return np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])


def _correlation(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float, float]:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    result = stats.pearsonr(x, y) if kind == "pearson" else stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _simple_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 2 or np.ptp(x) == 0:
        return {key: float("nan") for key in ("intercept", "slope", "r2", "mae", "rmse")}
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ coefficients
    residual = y - predicted
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "intercept": float(coefficients[0]), "slope": float(coefficients[1]),
        "r2": float(1.0 - np.sum(residual ** 2) / total) if total > 0 else float("nan"),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
    }


def _bootstrap_ci(
    rows: list[dict[str, Any]], statistic: Callable[[list[dict[str, Any]]], float], repeats: int,
) -> list[float | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["item_id"])].append(row)
    ids = sorted(grouped)
    if len(ids) < 2:
        return [None, None]
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(repeats):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        replicate = [row for item_id in sampled for row in grouped[str(item_id)]]
        value = float(statistic(replicate))
        if math.isfinite(value):
            values.append(value)
    if not values:
        return [None, None]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _group_metrics(
    rows: list[dict[str, Any]], group: str, x_key: str, score_type: str, repeats: int,
) -> dict[str, Any]:
    x, y = _paired(rows, x_key)
    pearson, pearson_p = _correlation(x, y, "pearson")
    spearman, spearman_p = _correlation(x, y, "spearman")
    fit = _simple_fit(x, y)

    def boot_correlation(kind: str) -> Callable[[list[dict[str, Any]]], float]:
        def value(sample: list[dict[str, Any]]) -> float:
            bx, by = _paired(sample, x_key)
            return _correlation(bx, by, kind)[0]
        return value

    def boot_slope(sample: list[dict[str, Any]]) -> float:
        bx, by = _paired(sample, x_key)
        return _simple_fit(bx, by)["slope"]

    return {
        "group": group, "score_type": score_type, "n": len(x),
        "pearson": pearson, "pearson_p_value": pearson_p,
        "pearson_ci_low": _bootstrap_ci(rows, boot_correlation("pearson"), repeats)[0],
        "pearson_ci_high": _bootstrap_ci(rows, boot_correlation("pearson"), repeats)[1],
        "spearman": spearman, "spearman_p_value": spearman_p,
        "spearman_ci_low": _bootstrap_ci(rows, boot_correlation("spearman"), repeats)[0],
        "spearman_ci_high": _bootstrap_ci(rows, boot_correlation("spearman"), repeats)[1],
        **fit,
        "slope_ci_low": _bootstrap_ci(rows, boot_slope, repeats)[0],
        "slope_ci_high": _bootstrap_ci(rows, boot_slope, repeats)[1],
        "sign_agreement_rate": (
            float(np.mean([
                np.sign(float(row[x_key])) == np.sign(float(row["sa_signed"]))
                for row in rows
                if row.get(x_key) is not None and float(row[x_key]) != 0
                and float(row["sa_signed"]) != 0
            ]))
            if any(
                row.get(x_key) is not None and float(row[x_key]) != 0
                and float(row["sa_signed"]) != 0 for row in rows
            ) else None
        ),
    }


def _multiple_fit(rows: list[dict[str, Any]], adjusted: bool) -> dict[str, Any]:
    usable = [row for row in rows if row.get("cma_signed") is not None]
    if not usable:
        return {"n": 0, "coefficients": {}, "r2": None, "mae": None, "rmse": None}
    names = ["intercept", "cma_signed", "hard", "cma_x_hard"]
    matrix = []
    target = []
    for row in usable:
        cma = float(row["cma_signed"])
        hard = float(row["difficulty"] == "hard")
        values = [1.0, cma, hard, cma * hard]
        if adjusted:
            values.extend([
                float(row["original_text_entropy"]), float(row["original_image_entropy"]),
                float(row["text_entropy_match_error"]),
                float(row["text_probability_match_error"]),
            ])
        if all(math.isfinite(value) for value in values):
            matrix.append(values)
            target.append(float(row["sa_signed"]))
    if adjusted:
        names.extend([
            "original_text_entropy", "original_image_entropy",
            "text_entropy_match_error", "text_probability_match_error",
        ])
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(target, dtype=float)
    if not len(y):
        return {"n": 0, "coefficients": {}, "r2": None, "mae": None, "rmse": None}
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    predicted = x @ coefficients
    residual = y - predicted
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": len(y), "rank": int(np.linalg.matrix_rank(x)),
        "coefficients": dict(zip(names, map(float, coefficients))),
        "r2": float(1 - np.sum(residual ** 2) / total) if total > 0 else None,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
    }


def _robustness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (float(row["cma_signed"]), float(row["cma_log_probability_signed"]))
        for row in rows if row.get("cma_signed") is not None
        and row.get("cma_log_probability_signed") is not None
    ]
    if not pairs:
        return {"n": 0}
    primary = np.asarray([pair[0] for pair in pairs])
    secondary = np.asarray([pair[1] for pair in pairs])
    pearson, pearson_p = _correlation(primary, secondary, "pearson")
    spearman, spearman_p = _correlation(primary, secondary, "spearman")
    nonzero = (primary != 0) & (secondary != 0)
    return {
        "n": len(pairs), "pearson": pearson, "pearson_p_value": pearson_p,
        "spearman": spearman, "spearman_p_value": spearman_p,
        "mae": float(np.mean(np.abs(primary - secondary))),
        "sign_agreement_rate": (
            float(np.mean(np.sign(primary[nonzero]) == np.sign(secondary[nonzero])))
            if nonzero.any() else None
        ),
    }


def _plots(rows: list[dict[str, Any]], root: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    colors = {"easy": "#2f78b7", "hard": "#d95f45"}
    for difficulty in ("easy", "hard"):
        subset = [row for row in rows if row["difficulty"] == difficulty and row["cma_signed"] is not None]
        if subset:
            x = np.asarray([row["cma_signed"] for row in subset], dtype=float)
            y = np.asarray([row["sa_signed"] for row in subset], dtype=float)
            axis.scatter(x, y, alpha=.72, label=difficulty, color=colors[difficulty])
            if len(x) >= 2 and np.ptp(x) > 0:
                fit = np.polyfit(x, y, 1)
                grid = np.linspace(x.min(), x.max(), 100)
                axis.plot(grid, fit[0] * grid + fit[1], color=colors[difficulty])
    axis.axhline(0, color="0.75", linewidth=.8); axis.axvline(0, color="0.75", linewidth=.8)
    axis.set(xlabel="CMA signed (logit)", ylabel="Soft-SA signed", xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
    axis.legend(); fig.tight_layout()
    path = plots / "cma_vs_soft_sa.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    fig, axis = plt.subplots(figsize=(6.4, 4.6))
    axis.hist([row["interaction"] for row in rows], bins=min(24, max(5, len(rows) // 4)))
    axis.set(xlabel="Four-cell interaction", ylabel="Cases")
    fig.tight_layout(); path = plots / "interaction_diagnostic.png"
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    paired = [row for row in rows if row["cma_signed"] is not None and row["cma_log_probability_signed"] is not None]
    fig, axis = plt.subplots(figsize=(5.3, 5.1))
    axis.scatter([row["cma_signed"] for row in paired], [row["cma_log_probability_signed"] for row in paired], alpha=.72)
    axis.plot([-1, 1], [-1, 1], linestyle="--", color="0.5")
    axis.set(xlabel="Logit CMA", ylabel="Log-probability CMA", xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
    fig.tight_layout(); path = plots / "cma_robustness.png"
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))
    return paths


def analyze(root: Path, repeats: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    trials = [row for row in load_jsonl(root / "trials.jsonl") if row.get("status") == "completed"]
    if not trials:
        raise ValueError(f"No completed faithful-check trials found in {root / 'trials.jsonl'}")
    manifests = {row["case_id"]: row for row in load_jsonl(root / "manifest.jsonl")}
    missing = [row["case_id"] for row in trials if row["case_id"] not in manifests]
    if missing:
        raise ValueError(f"Trials missing manifest records: {missing[:5]}")
    rows = [_case_row(trial, manifests[trial["case_id"]]) for trial in trials]
    tables = root / "tables"
    _atomic_csv(tables / "case_level.csv", rows)
    metric_rows: list[dict[str, Any]] = []
    for label, subset in (
        ("overall", rows), ("easy", [row for row in rows if row["difficulty"] == "easy"]),
        ("hard", [row for row in rows if row["difficulty"] == "hard"]),
    ):
        metric_rows.append(_group_metrics(subset, label, "cma_signed", "logit", repeats))
        metric_rows.append(_group_metrics(
            subset, label, "cma_log_probability_signed", "log_probability", repeats,
        ))
    _atomic_csv(tables / "correlation_and_fit.csv", metric_rows)
    identifiable = [row for row in rows if row["cma_signed"] is not None]
    summary = {
        "status": "complete", "trial_count": len(rows),
        "identifiable_count": len(identifiable),
        "unidentifiable_count": len(rows) - len(identifiable),
        "bootstrap_repeats": repeats, "metrics": metric_rows,
        "difficulty_interaction_fit": _multiple_fit(identifiable, adjusted=False),
        "adjusted_fit": _multiple_fit(identifiable, adjusted=True),
        "logit_vs_log_probability_robustness": _robustness(rows),
        "plots": _plots(rows, root) if rows else [],
    }
    cleaned = _json_clean(summary)
    atomic_json(root / "summary.json", cleaned)
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze faithful-check output")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(analyze(args.output_root, args.bootstrap_repeats), ensure_ascii=False, indent=2))
