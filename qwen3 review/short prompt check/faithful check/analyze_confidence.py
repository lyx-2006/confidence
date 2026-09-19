from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
for candidate in (REPOSITORY_ROOT, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dp_SA.io_utils import atomic_json, load_jsonl, sha256_file

from confidence_core import atomic_csv, ols_fit, partial_r2
from config import BOOTSTRAP_REPEATS, SEED


DEFAULT_EXPERIMENT_ROOT = (
    REPOSITORY_ROOT / "qwen3 review" / "short prompt check" / "output"
    / "faithful_check_extended" / "balanced_subset" / "exp"
)
DEFAULT_CONFIDENCE_ROOT = DEFAULT_EXPERIMENT_ROOT / "logit_confidence"
DEFAULT_FINAL_ROOT = DEFAULT_EXPERIMENT_ROOT.parent / "final_results"


MODELS = {
    "M_G": ("G_L",),
    "M_CMA": ("cma_signed",),
    "M_GC": ("G_L", "cma_signed"),
    "M_IT": ("L_i", "L_t"),
    "M_ITC": ("L_i", "L_t", "cma_signed"),
    "M_GC_H": ("G_L", "cma_signed", "Hard"),
    "M_GC_prob": ("G_C", "cma_signed"),
}


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _corr(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float | None, float | None]:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None
    result = stats.pearsonr(x, y) if kind == "pearson" else stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _metric_arrays(rows: Sequence[dict[str, Any]], feature: str, outcome: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(float(row[feature]), float(row[outcome])) for row in rows]
    return np.asarray([pair[0] for pair in pairs]), np.asarray([pair[1] for pair in pairs])


def _fit_metrics(fit: dict[str, Any]) -> dict[str, Any]:
    return {key: fit[key] for key in ("intercept", "r2", "adjusted_r2", "mae", "rmse")}


def _cv_metrics(rows: Sequence[dict[str, Any]], outcome: str, features: Sequence[str]) -> dict[str, float | None]:
    groups = np.asarray([str(row["item_id"]) for row in rows])
    if len(set(groups)) < 5:
        return {"cv_r2": None, "cv_mae": None}
    x = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float64)
    y = np.asarray([float(row[outcome]) for row in rows], dtype=np.float64)
    predictions = np.full(len(rows), np.nan)
    splitter = GroupKFold(n_splits=5)
    for train_index, test_index in splitter.split(x, y, groups):
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(train_index)), x[train_index]]), y[train_index], rcond=None)[0]
        predictions[test_index] = np.column_stack([np.ones(len(test_index)), x[test_index]]) @ beta
    residual = y - predictions
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "cv_r2": float(1.0 - np.sum(residual ** 2) / total) if total > 0 else None,
        "cv_mae": float(np.mean(np.abs(residual))),
    }


def _bootstrap_metrics(rows: Sequence[dict[str, Any]], outcome: str, features: Sequence[str], repeats: int) -> dict[str, float | int | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["item_id"])].append(row)
    item_ids = sorted(grouped)
    rng = np.random.default_rng(SEED)
    values = {key: [] for key in ("r2", "mae", "rmse")}
    for _ in range(repeats):
        sampled = rng.choice(item_ids, size=len(item_ids), replace=True)
        sample = [row for item_id in sampled for row in grouped[str(item_id)]]
        fit = ols_fit(sample, outcome, features)
        for key in values:
            if math.isfinite(float(fit[key])):
                values[key].append(float(fit[key]))
    result: dict[str, float | int | None] = {"bootstrap_valid": len(values["r2"])}
    for key, numbers in values.items():
        if numbers:
            result[f"{key}_ci_low"], result[f"{key}_ci_high"] = map(float, np.percentile(numbers, [2.5, 97.5]))
        else:
            result[f"{key}_ci_low"] = result[f"{key}_ci_high"] = None
    return result


def _load_rows(experiment_root: Path, confidence_root: Path) -> list[dict[str, Any]]:
    confidence = {row["case_id"]: row for row in load_jsonl(confidence_root / "test_fixed_answer_confidence.jsonl")}
    trials = {row["case_id"]: row for row in load_jsonl(experiment_root / "trials.jsonl") if row.get("status") == "completed"}
    full = {row["case_id"]: row for row in load_jsonl(experiment_root / "full_softsa" / "full_softsa.jsonl") if row.get("status") == "completed"}
    short = {row["case_id"]: row for row in trials.values()}
    reverse = {row["case_id"]: row for row in load_jsonl(experiment_root / "reverse_softsa" / "reverse_softsa.jsonl") if row.get("status") == "completed"}
    if len(confidence) != 110 or len(trials) != 110 or len(full) != 110 or len(reverse) != 110:
        raise ValueError(f"expected 110 complete rows: confidence={len(confidence)}, trials={len(trials)}, full={len(full)}, reverse={len(reverse)}")
    rows = []
    for case_id in sorted(confidence):
        c = confidence[case_id]; t = trials[case_id]; f = full[case_id]; r = reverse[case_id]
        cma = float(t["cma_logit"]["cma_signed"])
        if abs(cma - float(f["source_cma_signed"])) > 1e-12:
            raise ValueError(f"CMA mismatch for {case_id}")
        rows.append({
            "case_id": case_id, "item_id": str(c["item_id"]), "difficulty": c["difficulty"], "Hard": int(c["difficulty"] == "hard"),
            "fixed_answer": c["fixed_answer"], "C_i": c["C_i"], "C_t": c["C_t"], "L_i": c["L_i"], "L_t": c["L_t"],
            "G_L": c["G_L"], "G_C": c["G_C"], "cma_signed": cma,
            "full_sa_signed": float(f["soft_sa_signed"]), "short_sa_signed": float(t["short_sa"]["soft_sa_signed"]),
            "reverse_short_sa_signed": float(r["soft_sa_signed"]),
        })
    return rows


def _analyze_variant(rows: list[dict[str, Any]], outcome: str, variant: str, output_root: Path, repeats: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    performance = []; fits: dict[tuple[str, str], dict[str, Any]] = {}
    groups = [("overall", rows), ("easy", [row for row in rows if row["difficulty"] == "easy"]), ("hard", [row for row in rows if row["difficulty"] == "hard"])]
    for group, subset in groups:
        for model, features in MODELS.items():
            fit = ols_fit(subset, outcome, features); fits[group, model] = fit
            row: dict[str, Any] = {"variant": variant, "group": group, "model": model, "features": "+".join(features), "n": len(subset), "item_count": len({r["item_id"] for r in subset}), **_fit_metrics(fit), **_cv_metrics(subset, outcome, features), **_bootstrap_metrics(subset, outcome, features, repeats)}
            for name, value in fit["coefficients"].items(): row[f"beta_{name}"] = value
            for name, value in fit["standardized_coefficients"].items(): row[f"std_beta_{name}"] = value
            performance.append(row)
    contrasts = []
    for group, _subset in groups:
        base = fits[group, "M_G"]; cma = fits[group, "M_CMA"]; joint = fits[group, "M_GC"]
        for name, full, reduced, added in (
            ("CMA_beyond_G", joint, base, "CMA"), ("G_beyond_CMA", joint, cma, "G_L"),
            ("CMA_beyond_IT", fits[group, "M_ITC"], fits[group, "M_IT"], "CMA"),
        ):
            contrasts.append({"variant": variant, "group": group, "contrast": name, "added_predictor": added, "full_model_r2": full["r2"], "reduced_model_r2": reduced["r2"], "partial_r2": partial_r2(full["r2"], reduced["r2"]), "delta_r2": full["r2"] - reduced["r2"]})
    return performance, contrasts


def _correlations(rows: list[dict[str, Any]], outcome: str, variant: str, group: str, subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for feature, label in (("G_L", "confidence_gap_log_odds"), ("G_C", "confidence_gap_probability"), ("cma_signed", "CMA"), ("L_i", "image_log_odds"), ("L_t", "text_log_odds")):
        x, y = _metric_arrays(subset, feature, outcome); pearson, pearson_p = _corr(x, y, "pearson"); spearman, spearman_p = _corr(x, y, "spearman")
        fit = ols_fit(subset, outcome, (feature,))
        result.append({"variant": variant, "group": group, "outcome": outcome, "predictor": label, "predictor_field": feature, "n": len(subset), "pearson": pearson, "pearson_p_value": pearson_p, "spearman": spearman, "spearman_p_value": spearman_p, "slope": fit["coefficients"][feature], "intercept": fit["intercept"], "r2": fit["r2"], "mae": fit["mae"], "rmse": fit["rmse"]})
    return result


def _plot_full(rows: list[dict[str, Any]], performance: list[dict[str, Any]], output_root: Path) -> None:
    figure_root = output_root / "figures" / "full"; figure_root.mkdir(parents=True, exist_ok=True)
    x = np.asarray([row["G_L"] for row in rows]); cma = np.asarray([row["cma_signed"] for row in rows]); y = np.asarray([row["full_sa_signed"] for row in rows])
    fit = ols_fit(rows, "full_sa_signed", ("G_L",)); grid = np.linspace(float(x.min()), float(x.max()), 200)
    fig, ax = plt.subplots(figsize=(7.2, 5.4)); points = ax.scatter(x, y, c=cma, cmap="coolwarm", vmin=-1, vmax=1, s=34, alpha=.85, edgecolors="none"); ax.plot(grid, fit["intercept"] + fit["coefficients"]["G_L"] * grid, color="black", ls="--", lw=1.3); fig.colorbar(points, ax=ax, label="CMA signed"); ax.set(xlabel="Confidence gap $G_L=L_i-L_t$", ylabel="Full Soft-SA signed", title="Confidence gap versus Full Soft-SA"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figure_root / "confidence_gap_vs_full_sa.png", dpi=260); plt.close(fig)
    joint = ols_fit(rows, "full_sa_signed", ("G_L", "cma_signed")); gx = np.linspace(x.min(), x.max(), 100); gy = np.linspace(cma.min(), cma.max(), 100); xx, yy = np.meshgrid(gx, gy); zz = joint["intercept"] + joint["coefficients"]["G_L"] * xx + joint["coefficients"]["cma_signed"] * yy
    fig, ax = plt.subplots(figsize=(7.2, 5.4)); contour = ax.contourf(xx, yy, zz, levels=15, cmap="coolwarm", alpha=.72); ax.scatter(x, cma, c=y, cmap="viridis", vmin=-1, vmax=1, s=34, edgecolors="black", linewidths=.2); fig.colorbar(contour, ax=ax, label="Predicted Full Soft-SA signed"); ax.set(xlabel="Confidence gap $G_L$", ylabel="CMA signed", title="Joint confidence–CMA prediction plane"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figure_root / "confidence_cma_prediction_plane.png", dpi=260); plt.close(fig)
    rows_plot = [row for row in performance if row["group"] == "overall" and row["variant"] == "full"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8)); names = [row["model"] for row in rows_plot]; values = [row["r2"] for row in rows_plot]; ax.bar(names, values, color="#4C78A8"); ax.axhline(0, color="black", lw=.8); ax.set(ylabel="$R^2$", title="Full Soft-SA model fit comparison"); ax.grid(axis="y", alpha=.2); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(figure_root / "full_model_r2.png", dpi=260); plt.close(fig)


def analyze(experiment_root: Path, confidence_root: Path, final_root: Path, repeats: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    rows = _load_rows(experiment_root, confidence_root)
    performance: list[dict[str, Any]] = []; contrasts: list[dict[str, Any]] = []; correlations: list[dict[str, Any]] = []
    variant_outcomes = {"full": "full_sa_signed", "short": "short_sa_signed", "reverse_short": "reverse_short_sa_signed"}
    for variant, outcome in variant_outcomes.items():
        p, c = _analyze_variant(rows, outcome, variant, final_root, repeats); performance.extend(p); contrasts.extend(c)
        for group, subset in (("overall", rows), ("easy", [r for r in rows if r["difficulty"] == "easy"]), ("hard", [r for r in rows if r["difficulty"] == "hard"])):
            correlations.extend(_correlations(rows, outcome, variant, group, subset))
    full_table = final_root / "tables" / "full"; full_table.mkdir(parents=True, exist_ok=True)
    for variant in variant_outcomes:
        (final_root / "tables" / variant).mkdir(parents=True, exist_ok=True)
        atomic_csv(final_root / "tables" / variant / "confidence_model_performance.csv", [row for row in performance if row["variant"] == variant])
        atomic_csv(final_root / "tables" / variant / "confidence_correlations.csv", [row for row in correlations if row["variant"] == variant])
        atomic_csv(final_root / "tables" / variant / "confidence_partial_r2.csv", [row for row in contrasts if row["variant"] == variant])
    atomic_csv(confidence_root / "test_confidence_case_level.csv", rows)
    atomic_csv(full_table / "confidence_model_performance_all_variants.csv", performance)
    atomic_csv(full_table / "confidence_correlations_all_variants.csv", correlations)
    atomic_csv(full_table / "confidence_partial_r2_all_variants.csv", contrasts)
    _plot_full(rows, performance, final_root)
    summary = {"status": "complete", "case_count": len(rows), "item_count": len({row["item_id"] for row in rows}), "bootstrap_repeats": repeats, "primary_variant": "full", "primary_performance": [row for row in performance if row["variant"] == "full" and row["group"] == "overall"], "primary_contrasts": [row for row in contrasts if row["variant"] == "full" and row["group"] == "overall"], "inputs": {"confidence": sha256_file(confidence_root / "test_fixed_answer_confidence.jsonl"), "full_softsa": sha256_file(experiment_root / "full_softsa" / "full_softsa.jsonl"), "trials": sha256_file(experiment_root / "trials.jsonl")}}
    atomic_json(final_root / "confidence_summary.json", _clean(summary)); return _clean(summary)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze calibrated unimodal confidence against CMA and Soft-SA")
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--confidence-root", type=Path, default=DEFAULT_CONFIDENCE_ROOT)
    parser.add_argument("--final-root", type=Path, default=DEFAULT_FINAL_ROOT)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(); print(json.dumps(analyze(args.experiment_root, args.confidence_root, args.final_root, args.bootstrap_repeats), ensure_ascii=False, indent=2))
