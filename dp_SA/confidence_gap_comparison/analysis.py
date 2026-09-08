from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


SEED = 42
BOOTSTRAP_REPEATS = 2000
EPSILON = 1e-6
RIDGE_ALPHA = 1.0
PACKAGE_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PACKAGE_ROOT.parent / "unimodal_logit_confidence" / "output" / "results"
CONFIDENCE_PATH = RESULTS_ROOT / "unimodal_confidence/artifacts/predictions/phase1_confidence_joined.jsonl"
TRAIN_MANIFEST_PATH = RESULTS_ROOT / "shared/manifests/probe_train_manifest.jsonl"
TEST_MANIFEST_PATH = RESULTS_ROOT / "shared/manifests/test_manifest.jsonl"
OUTPUT_ROOT = PACKAGE_ROOT / "output"
MODEL_FEATURES = {"M_G": ("G_L",), "M_IT": ("L_i", "L_t")}


@dataclass(frozen=True)
class FittedModel:
    name: str
    features: tuple[str, ...]
    scaler: StandardScaler
    model: Ridge


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Non-object JSONL row: {path}:{line_number}")
                rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clipped_logit(probability: float, epsilon: float = EPSILON) -> float:
    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Confidence is not a finite probability: {value}")
    clipped = float(np.clip(value, epsilon, 1.0 - epsilon))
    return float(math.log(clipped / (1.0 - clipped)))


def prepare_rows(confidence_rows: Sequence[dict[str, Any]], train_manifest: Sequence[dict[str, Any]],
                 test_manifest: Sequence[dict[str, Any]], *, require_frozen_counts: bool = True
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifests: dict[str, tuple[str, dict[str, Any]]] = {}
    for split, rows in (("train", train_manifest), ("test", test_manifest)):
        for row in rows:
            case_id = str(row["case_id"])
            if case_id in manifests:
                raise ValueError(f"Duplicate/cross-split manifest case: {case_id}")
            manifests[case_id] = (split, row)
    confidence: dict[str, dict[str, Any]] = {}
    for row in confidence_rows:
        case_id = str(row["case_id"])
        if case_id in confidence:
            raise ValueError(f"Duplicate confidence case: {case_id}")
        confidence[case_id] = row
    if set(confidence) != set(manifests):
        raise ValueError("Confidence cases do not exactly match frozen manifests")

    prepared: list[dict[str, Any]] = []
    clip_counts = {"C_i_low": 0, "C_i_high": 0, "C_t_low": 0, "C_t_high": 0}
    for case_id, (split, manifest) in manifests.items():
        source = confidence[case_id]
        for field in ("split", "family_id", "item_id", "condition", "fixed_answer"):
            manifest_value = manifest.get("phase0_normalized_answer") if field == "fixed_answer" else manifest.get(field)
            if str(source.get(field)) != str(manifest_value):
                raise ValueError(f"Frozen join mismatch for {case_id}: {field}")
        c_i = float(source["image_fixed_answer_confidence"])
        c_t = float(source["text_fixed_answer_confidence"])
        clip_counts["C_i_low"] += int(c_i < EPSILON); clip_counts["C_i_high"] += int(c_i > 1.0 - EPSILON)
        clip_counts["C_t_low"] += int(c_t < EPSILON); clip_counts["C_t_high"] += int(c_t > 1.0 - EPSILON)
        l_i, l_t = clipped_logit(c_i), clipped_logit(c_t)
        soft_sa = float(manifest["soft_sa_image_score"])
        if not math.isfinite(soft_sa) or not 0.0 <= soft_sa <= 1.0:
            raise ValueError(f"Invalid soft SA: {case_id}")
        prepared.append({
            "case_id": case_id, "split": split, "family_id": str(manifest["family_id"]),
            "item_id": str(manifest["item_id"]), "condition": str(manifest["condition"]),
            "soft_SA": soft_sa, "V_SA": 2.0 * soft_sa - 1.0,
            "C_i": c_i, "C_t": c_t, "L_i": l_i, "L_t": l_t,
            "G_L": l_i - l_t, "M_L": (l_i + l_t) / 2.0,
        })
    prepared.sort(key=lambda row: (row["split"], row["family_id"], row["case_id"]))
    train = [row for row in prepared if row["split"] == "train"]
    test = [row for row in prepared if row["split"] == "test"]
    if require_frozen_counts and (len(train) != 1112 or len(test) != 100):
        raise ValueError(f"Frozen counts changed: train={len(train)}, test={len(test)}")
    train_families, test_families = ({row["family_id"] for row in group} for group in (train, test))
    train_items, test_items = ({row["item_id"] for row in group} for group in (train, test))
    if train_families & test_families or train_items & test_items:
        raise ValueError("Train/test family or item leakage")
    if require_frozen_counts and (len(test_families) != 50 or {sum(row["family_id"] == family for row in test) for family in test_families} != {2}):
        raise ValueError("Frozen test families are not exactly 50 clusters of two cases")
    audit = {
        "status": "passed", "train_case_count": len(train), "test_case_count": len(test),
        "train_family_count": len(train_families), "test_family_count": len(test_families),
        "train_item_count": len(train_items), "test_item_count": len(test_items),
        "family_overlap_count": 0, "item_overlap_count": 0, "confidence_clip_counts": clip_counts,
    }
    return prepared, audit


def fit_model(name: str, train: Sequence[dict[str, Any]]) -> FittedModel:
    features = MODEL_FEATURES[name]
    x = np.asarray([[float(row[feature]) for feature in features] for row in train], dtype=np.float64)
    y = np.asarray([float(row["V_SA"]) for row in train], dtype=np.float64)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", fit_intercept=True).fit(scaler.transform(x), y)
    return FittedModel(name, features, scaler, model)


def predict(fitted: FittedModel, rows: Sequence[dict[str, Any]]) -> np.ndarray:
    x = np.asarray([[float(row[feature]) for feature in fitted.features] for row in rows], dtype=np.float64)
    return np.asarray(fitted.model.predict(fitted.scaler.transform(x)), dtype=np.float64)


def reparameterize_mit(fitted: FittedModel, train: Sequence[dict[str, Any]],
                       rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, dict[str, float]]:
    """Express the fitted L_i/L_t model exactly in standardized G_L/M_L coordinates."""
    if fitted.features != ("L_i", "L_t"):
        raise ValueError("Only M_IT can be reparameterized as G_L + M_L")
    coefficient_i = float(fitted.model.coef_[0] / fitted.scaler.scale_[0])
    coefficient_t = float(fitted.model.coef_[1] / fitted.scaler.scale_[1])
    raw_intercept = float(fitted.model.intercept_ - coefficient_i * fitted.scaler.mean_[0]
                          - coefficient_t * fitted.scaler.mean_[1])
    raw_beta_g = 0.5 * (coefficient_i - coefficient_t)
    raw_beta_m = coefficient_i + coefficient_t
    train_g = np.asarray([row["G_L"] for row in train], dtype=np.float64)
    train_m = np.asarray([row["M_L"] for row in train], dtype=np.float64)
    mean_g, std_g = float(np.mean(train_g)), float(np.std(train_g))
    mean_m, std_m = float(np.mean(train_m)), float(np.std(train_m))
    if std_g <= 0 or std_m <= 0:
        raise ValueError("G_L and M_L must vary in train")
    standardized_beta_g = raw_beta_g * std_g
    standardized_beta_m = raw_beta_m * std_m
    standardized_intercept = raw_intercept + raw_beta_g * mean_g + raw_beta_m * mean_m
    prediction = np.asarray([
        standardized_intercept
        + standardized_beta_g * ((float(row["G_L"]) - mean_g) / std_g)
        + standardized_beta_m * ((float(row["M_L"]) - mean_m) / std_m)
        for row in rows
    ], dtype=np.float64)
    return prediction, {
        "intercept": standardized_intercept,
        "beta_G": standardized_beta_g, "beta_M": standardized_beta_m,
        "G_L_mean": mean_g, "G_L_std": std_g, "M_L_mean": mean_m, "M_L_std": std_m,
        "raw_intercept": raw_intercept, "raw_beta_G": raw_beta_g, "raw_beta_M": raw_beta_m,
    }


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    values = {
        "r2": float(r2_score(y, prediction)), "pearson": float(pearsonr(y, prediction).statistic),
        "spearman": float(spearmanr(y, prediction).statistic),
        "mae": float(mean_absolute_error(y, prediction)),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("A held-out metric is non-finite")
    return values


def family_bootstrap_delta(test: Sequence[dict[str, Any]], y: np.ndarray,
                           prediction_it: np.ndarray, prediction_gap: np.ndarray,
                           repeats: int = BOOTSTRAP_REPEATS, seed: int = SEED
                           ) -> tuple[float, float, int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(test):
        groups[str(row["family_id"])].append(index)
    families = sorted(groups)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(repeats):
        sampled = rng.choice(families, size=len(families), replace=True)
        indices = np.asarray([index for family in sampled for index in groups[str(family)]], dtype=int)
        delta = float(r2_score(y[indices], prediction_it[indices]) - r2_score(y[indices], prediction_gap[indices]))
        if math.isfinite(delta):
            deltas.append(delta)
    if not deltas:
        raise ValueError("No valid paired family-bootstrap replicate")
    low, high = np.percentile(np.asarray(deltas), [2.5, 97.5])
    return float(low), float(high), len(deltas)


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prediction_figure(path: Path, y: np.ndarray, predictions: dict[str, np.ndarray],
                       metric_rows: dict[str, dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True, constrained_layout=True)
    limits = [float(min(np.min(y), *(np.min(value) for value in predictions.values()))),
              float(max(np.max(y), *(np.max(value) for value in predictions.values())))]
    for axis, name, title in zip(axes, ("M_G", "M_GM"), ("Gap model", "$G_L + M_L$ model"), strict=True):
        prediction = predictions[name]; observed = metric_rows[name]
        axis.scatter(y, prediction, color="#356A9A", alpha=0.72, s=28, edgecolors="none")
        axis.plot(limits, limits, color="0.3", linestyle="--", linewidth=1.2)
        axis.set(title=title, xlabel="Observed $V_{SA}$", ylabel="Predicted $V_{SA}$")
        axis.grid(alpha=0.2)
        axis.text(0.04, 0.96, f"$R^2$ = {observed['r2']:.3f}\nPearson = {observed['pearson']:.3f}\nSpearman = {observed['spearman']:.3f}\nMAE = {observed['mae']:.3f}\nn = {len(y)}",
                  transform=axis.transAxes, va="top", bbox={"boxstyle": "round", "facecolor": "white", "alpha": .88, "edgecolor": ".8"})
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def _confidence_plane(path: Path, test: Sequence[dict[str, Any]], fitted: FittedModel) -> None:
    l_t = np.asarray([row["L_t"] for row in test], dtype=float)
    l_i = np.asarray([row["L_i"] for row in test], dtype=float)
    y = np.asarray([row["V_SA"] for row in test], dtype=float)
    x_grid = np.linspace(float(l_t.min()), float(l_t.max()), 180)
    y_grid = np.linspace(float(l_i.min()), float(l_i.max()), 180)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.column_stack((yy.ravel(), xx.ravel()))  # fitted feature order: L_i, L_t
    zz = fitted.model.predict(fitted.scaler.transform(grid)).reshape(xx.shape)
    fig, axis = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    color_limit = float(np.max(np.abs(y)))
    points = axis.scatter(l_t, l_i, c=y, cmap="coolwarm", vmin=-color_limit, vmax=color_limit, s=34,
                          edgecolors="black", linewidths=.25, alpha=.9)
    contours = axis.contour(xx, yy, zz, levels=9, colors="black", linewidths=.8, alpha=.72)
    axis.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
    fig.colorbar(points, ax=axis, label="Observed $V_{SA}$")
    axis.set(xlabel="$L_t$", ylabel="$L_i$", title="$M_{GM}$ prediction contours and observed signed SA")
    axis.grid(alpha=.15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def run_analysis(output_root: Path = OUTPUT_ROOT, *, bootstrap_repeats: int = BOOTSTRAP_REPEATS,
                 seed: int = SEED) -> dict[str, Any]:
    inputs = (CONFIDENCE_PATH, TRAIN_MANIFEST_PATH, TEST_MANIFEST_PATH)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    rows, audit = prepare_rows(load_jsonl(CONFIDENCE_PATH), load_jsonl(TRAIN_MANIFEST_PATH),
                               load_jsonl(TEST_MANIFEST_PATH))
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    y = np.asarray([row["V_SA"] for row in test], dtype=np.float64)
    fitted = {name: fit_model(name, train) for name in MODEL_FEATURES}
    predictions = {name: predict(model, test) for name, model in fitted.items()}
    predictions["M_GM"], semantic = reparameterize_mit(fitted["M_IT"], train, test)
    equivalence_error = float(np.max(np.abs(predictions["M_GM"] - predictions["M_IT"])))
    if equivalence_error > 1e-12:
        raise ValueError(f"M_GM reparameterization is not prediction-equivalent: {equivalence_error}")
    observed = {name: metrics(y, value) for name, value in predictions.items()}
    delta_r2 = observed["M_GM"]["r2"] - observed["M_G"]["r2"]
    ci_low, ci_high, valid = family_bootstrap_delta(test, y, predictions["M_GM"], predictions["M_G"],
                                                    repeats=bootstrap_repeats, seed=seed)

    performance_rows = [{
        "model": name, "features": "+".join(MODEL_FEATURES[name]) if name in MODEL_FEATURES else "G_L+M_L",
        "outcome": "V_SA", "train_case_count": len(train), "test_case_count": len(test),
        "test_family_count": len({row["family_id"] for row in test}), **observed[name],
    } for name in ("M_G", "M_IT", "M_GM")]
    coefficient_rows: list[dict[str, Any]] = []
    for name, value in fitted.items():
        coefficient_rows.append({"model": name, "term": "intercept", "standardized_coefficient": float(value.model.intercept_),
                                 "predictor_train_mean": 0.0, "predictor_train_std": 1.0})
        for index, feature in enumerate(value.features):
            coefficient_rows.append({"model": name, "term": feature,
                                     "standardized_coefficient": float(value.model.coef_[index]),
                                     "predictor_train_mean": float(value.scaler.mean_[index]),
                                     "predictor_train_std": float(value.scaler.scale_[index])})
    coefficient_rows.extend([
        {"model": "M_GM", "term": "intercept", "standardized_coefficient": semantic["intercept"],
         "predictor_train_mean": 0.0, "predictor_train_std": 1.0},
        {"model": "M_GM", "term": "G_L", "standardized_coefficient": semantic["beta_G"],
         "predictor_train_mean": semantic["G_L_mean"], "predictor_train_std": semantic["G_L_std"]},
        {"model": "M_GM", "term": "M_L", "standardized_coefficient": semantic["beta_M"],
         "predictor_train_mean": semantic["M_L_mean"], "predictor_train_std": semantic["M_L_std"]},
    ])
    contrast_rows = [{
        "contrast": "M_GM_minus_M_G", "left_model": "M_GM", "right_model": "M_G",
        "delta_r2": delta_r2, "ci_low": ci_low, "ci_high": ci_high,
        "bootstrap_repeats": bootstrap_repeats, "valid_bootstrap_repeats": valid,
        "seed": seed, "cluster": "family_id",
    }, {
        "contrast": "M_GM_minus_M_IT", "left_model": "M_GM", "right_model": "M_IT",
        "delta_r2": observed["M_GM"]["r2"] - observed["M_IT"]["r2"], "ci_low": 0.0, "ci_high": 0.0,
        "bootstrap_repeats": bootstrap_repeats, "valid_bootstrap_repeats": valid,
        "seed": seed, "cluster": "family_id",
    }]
    prediction_rows = [{
        **{key: row[key] for key in ("case_id", "family_id", "item_id", "condition", "soft_SA", "V_SA", "C_i", "C_t", "L_i", "L_t", "G_L", "M_L")},
        "predicted_M_G": float(predictions["M_G"][index]), "residual_M_G": float(y[index] - predictions["M_G"][index]),
        "predicted_M_IT": float(predictions["M_IT"][index]), "residual_M_IT": float(y[index] - predictions["M_IT"][index]),
        "predicted_M_GM": float(predictions["M_GM"][index]), "residual_M_GM": float(y[index] - predictions["M_GM"][index]),
    } for index, row in enumerate(test)]

    output_root = Path(output_root)
    _atomic_csv(output_root / "tables/model_performance.csv", performance_rows, list(performance_rows[0]))
    _atomic_csv(output_root / "tables/standardized_coefficients.csv", coefficient_rows, list(coefficient_rows[0]))
    _atomic_csv(output_root / "tables/paired_delta_r2.csv", contrast_rows, list(contrast_rows[0]))
    _atomic_csv(output_root / "artifacts/test_predictions.csv", prediction_rows, list(prediction_rows[0]))
    audit.update({
        "model": {"type": "Ridge", "alpha": RIDGE_ALPHA, "solver": "lsqr", "fit_intercept": True},
        "outcome": "V_SA = 2 * soft_sa_image_score - 1", "confidence_logit_clip": [EPSILON, 1.0 - EPSILON],
        "scaler_fit_split": "train", "model_fit_split": "train", "evaluation_split": "test_once",
        "bootstrap": {"repeats": bootstrap_repeats, "valid_repeats": valid, "seed": seed, "cluster": "family_id", "paired": True},
        "semantic_reparameterization": {"source_model": "M_IT", "target_model": "M_GM",
                                         "max_absolute_prediction_difference": equivalence_error,
                                         "raw_coefficients": {key: semantic[key] for key in ("raw_intercept", "raw_beta_G", "raw_beta_M")}},
        "input_sha256": {str(path.resolve()): sha256_file(path) for path in inputs},
    })
    _atomic_json(output_root / "artifacts/audit.json", audit)
    _prediction_figure(output_root / "figures/fig1_test_predictions.png", y, predictions, observed)
    _confidence_plane(output_root / "figures/fig2_confidence_plane.png", test, fitted["M_IT"])
    return {
        "status": "complete", "output_root": str(output_root.resolve()),
        "train_case_count": len(train), "test_case_count": len(test),
        "test_family_count": len({row["family_id"] for row in test}),
        "bootstrap_repeats": bootstrap_repeats, "valid_bootstrap_repeats": valid,
        "M_G": observed["M_G"], "M_IT": observed["M_IT"], "M_GM": observed["M_GM"],
        "beta_G": semantic["beta_G"], "beta_M": semantic["beta_M"], "delta_r2": delta_r2,
        "delta_r2_ci": [ci_low, ci_high],
    }


__all__ = [
    "BOOTSTRAP_REPEATS", "CONFIDENCE_PATH", "EPSILON", "MODEL_FEATURES", "OUTPUT_ROOT",
    "SEED", "TEST_MANIFEST_PATH", "TRAIN_MANIFEST_PATH", "clipped_logit", "family_bootstrap_delta",
    "fit_model", "metrics", "predict", "prepare_rows", "reparameterize_mit", "run_analysis",
]
