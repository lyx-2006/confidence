"""CPU-only screen for a protocol-shared Source Attribution component.

This module deliberately distinguishes two claims:

``rank transport``
    A shared linear readout preserves item ordering across report protocols,
    possibly despite protocol-specific offsets or scales.

``coordinate invariance``
    The *same* unit direction, origin, and scale give equivalent coordinates
    for the same item without protocol-specific calibration.

The existing protocol panel is observational and contains no common-template
random mappings.  Consequently even a passing coordinate gate is labelled an
existing-panel candidate that still requires new mapping confirmation.

All fitting is item-OOF.  The seven common-template protocols are used for
training; the three legacy protocols are untouched grammar holdouts.  Inputs
are read-only and every output is written below a caller-supplied directory.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from layer_metacognition.hidden_state_store import (
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
)

from .core import atomic_save_npz, ridge_raw_space, sha256_file, write_jsonl_atomic


COMMON_PROTOCOLS = (
    "common_9_ordered",
    "common_3_ordered",
    "common_2_ordered",
    "common_3_reversed",
    "common_2_reversed",
    "common_3_semantic",
    "common_2_semantic",
)
LEGACY_PROTOCOLS = (
    "legacy_normal_numeric",
    "legacy_text_both_image",
    "legacy_binary_text_image",
)
ALL_PROTOCOLS = COMMON_PROTOCOLS + LEGACY_PROTOCOLS


@dataclass(frozen=True)
class ComponentScreenConfig:
    """Pre-registered thresholds and deterministic fitting controls."""

    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    bootstrap_iterations: int = 1000
    control_iterations: int = 200
    random_direction_iterations: int = 200
    seed: int = 42
    expected_items: int | None = 80
    equivalence_fraction: float = 0.25
    icc_minimum: float = 0.75
    icc_ci_lower_minimum: float = 0.60
    within_between_sd_ratio_maximum: float = 0.50
    slope_interval: tuple[float, float] = (0.80, 1.25)
    null_percentile: float = 0.95
    fold_direction_median_cosine_minimum: float = 0.50

    def validate(self) -> None:
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("Ridge alphas must all be positive")
        if self.bootstrap_iterations < 20:
            raise ValueError("bootstrap_iterations must be at least 20")
        if self.control_iterations < 20:
            raise ValueError("control_iterations must be at least 20")
        if self.random_direction_iterations < 20:
            raise ValueError("random_direction_iterations must be at least 20")
        if not 0 < self.equivalence_fraction < 1:
            raise ValueError("equivalence_fraction must lie in (0, 1)")
        if not 0.5 < self.null_percentile < 1:
            raise ValueError("null_percentile must lie in (0.5, 1)")


@dataclass(frozen=True)
class ProtocolPanel:
    bridge_dir: Path
    input_root: Path
    rows: tuple[dict[str, Any], ...]
    case_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    folds: np.ndarray
    protocols: tuple[str, ...]
    hidden: np.ndarray
    semantic_scores: np.ndarray
    old_sa_prediction: np.ndarray
    old_sa_coordinate: np.ndarray
    behavior_prediction: np.ndarray
    behavior_coordinate: np.ndarray
    behavior_target: np.ndarray
    covariates: np.ndarray
    hidden_paths: tuple[Path, ...]


@dataclass(frozen=True)
class SharedTargetModel:
    mean: np.ndarray
    scale: np.ndarray
    loading: np.ndarray
    explained_variance: float

    def transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        return ((values - self.mean) / self.scale) @ self.loading


@dataclass(frozen=True)
class FoldFit:
    fold: int
    alpha: float
    d_raw: np.ndarray
    d_unit: np.ndarray
    raw_intercept: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    train_z_mean: float
    train_z_sd: float
    target_model: SharedTargetModel
    audit: dict[str, Any]


def _latest_completed_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        key = str(row.get("intervention_key") or row.get("case_id"))
        latest[key] = row
    completed = [row for row in latest.values() if row.get("status") == "completed"]
    return sorted(completed, key=lambda row: str(row["case_id"]))


def _resolve_hidden_path(bridge_dir: Path, reference: str) -> Path:
    candidate = Path(reference)
    candidates = (
        candidate,
        bridge_dir / candidate,
        bridge_dir.parent / candidate,
    )
    for value in candidates:
        if value.is_file():
            return value.resolve()
    raise FileNotFoundError(f"Protocol hidden file is missing: {reference}")


def load_protocol_panel(
    bridge_dir: str | Path,
    *,
    expected_items: int | None = 80,
) -> ProtocolPanel:
    """Load and validate the existing 80 x 10 protocol panel without mutation."""

    directory = Path(bridge_dir).resolve()
    results_path = directory / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(f"Protocol bridge results are missing: {results_path}")
    rows = _latest_completed_rows(results_path)
    if expected_items is not None and len(rows) != expected_items:
        raise ValueError(f"Expected {expected_items} completed items, found {len(rows)}")
    if len(rows) < 10:
        raise ValueError("Protocol component screen requires at least 10 items")
    item_ids = tuple(str(row["item_id"]) for row in rows)
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Protocol panel must contain exactly one case per item")

    hidden_values: list[np.ndarray] = []
    hidden_paths: list[Path] = []
    semantic: list[list[float]] = []
    old_prediction: list[list[float]] = []
    old_coordinate: list[list[float]] = []
    behavior_prediction: list[list[float]] = []
    behavior_coordinate: list[list[float]] = []
    for row in rows:
        protocol_payload = row.get("protocols")
        if not isinstance(protocol_payload, dict):
            raise ValueError(f"Case {row['case_id']} has no protocol payload")
        missing = [name for name in ALL_PROTOCOLS if name not in protocol_payload]
        if missing:
            raise ValueError(f"Case {row['case_id']} lacks protocols: {missing}")
        hidden_path = _resolve_hidden_path(directory, str(row["hidden_file"]))
        with np.load(hidden_path, allow_pickle=False) as payload:
            protocols = tuple(str(value) for value in payload["protocols"].tolist())
            hidden = np.asarray(payload["hidden"], dtype=np.float64)
        if protocols != ALL_PROTOCOLS:
            raise ValueError(
                f"Hidden protocol order mismatch for {row['case_id']}: {protocols}"
            )
        if hidden.ndim != 2 or hidden.shape[0] != len(ALL_PROTOCOLS):
            raise ValueError(f"Invalid hidden shape for {row['case_id']}: {hidden.shape}")
        if not np.isfinite(hidden).all():
            raise ValueError(f"Non-finite hidden values for {row['case_id']}")
        hidden_values.append(hidden)
        hidden_paths.append(hidden_path)
        semantic.append(
            [float(protocol_payload[name]["semantic_imageward_score"]) for name in ALL_PROTOCOLS]
        )
        old_prediction.append(
            [float(protocol_payload[name]["ridge_sa_prediction"]) for name in ALL_PROTOCOLS]
        )
        old_coordinate.append(
            [float(protocol_payload[name]["ridge_sa_coordinate"]) for name in ALL_PROTOCOLS]
        )
        behavior_prediction.append(
            [float(protocol_payload[name]["ridge_behavior_prediction"]) for name in ALL_PROTOCOLS]
        )
        behavior_coordinate.append(
            [float(protocol_payload[name]["ridge_behavior_coordinate"]) for name in ALL_PROTOCOLS]
        )

    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    unique_folds = sorted(set(folds.tolist()))
    if len(unique_folds) < 3 or any(int(np.sum(folds == fold)) < 2 for fold in unique_folds):
        raise ValueError("At least three item folds with two test items each are required")
    hidden_array = np.stack(hidden_values, axis=0)
    arrays = [
        np.asarray(semantic, dtype=np.float64),
        np.asarray(old_prediction, dtype=np.float64),
        np.asarray(old_coordinate, dtype=np.float64),
        np.asarray(behavior_prediction, dtype=np.float64),
        np.asarray(behavior_coordinate, dtype=np.float64),
    ]
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("Protocol panel contains non-finite measurements")
    covariates = np.column_stack(
        [
            np.asarray([float(bool(row["final_image"])) for row in rows]),
            np.asarray([float(str(row["difficulty"]) == "hard") for row in rows]),
            np.asarray([float(row["prior_index"]) for row in rows]),
        ]
    )
    return ProtocolPanel(
        bridge_dir=directory,
        input_root=directory.parent,
        rows=tuple(rows),
        case_ids=tuple(str(row["case_id"]) for row in rows),
        item_ids=item_ids,
        folds=folds,
        protocols=ALL_PROTOCOLS,
        hidden=hidden_array,
        semantic_scores=arrays[0],
        old_sa_prediction=arrays[1],
        old_sa_coordinate=arrays[2],
        behavior_prediction=arrays[3],
        behavior_coordinate=arrays[4],
        behavior_target=np.asarray(
            [float(row["behavior_use_residual"]) for row in rows], dtype=np.float64
        ),
        covariates=covariates,
        hidden_paths=tuple(hidden_paths),
    )


def fit_shared_target(scores: np.ndarray) -> SharedTargetModel:
    """Fit a training-only standardized PC1 semantic target."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("Shared target requires a 2D matrix with at least 2x2 values")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-12):
        raise ValueError("Every training protocol must have nonzero semantic variance")
    standardized = (values - mean) / scale
    _, singular, right = np.linalg.svd(standardized, full_matrices=False)
    loading = np.asarray(right[0], dtype=np.float64)
    # All protocols are semantically oriented Text -> Image.  Fix PC sign by
    # their average loading rather than by held-out outcomes.
    if float(np.sum(loading)) < 0:
        loading = -loading
    variance = np.square(singular)
    explained = float(variance[0] / variance.sum())
    return SharedTargetModel(mean, scale, loading, explained)


def _safe_correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    if rank:
        # Stable average ranks without relying on warning-producing scipy paths.
        from scipy.stats import rankdata

        x = rankdata(x, method="average")
        y = rankdata(y, method="average")
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def _bootstrap_ci(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rank: bool,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    estimate = _safe_correlation(x, y, rank=rank)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(x), size=len(x))
        value = _safe_correlation(x[indices], y[indices], rank=rank)
        if value is not None:
            samples.append(value)
    ci = [None, None]
    if samples:
        ci = [float(value) for value in np.quantile(samples, [0.025, 0.975])]
    return {
        "n": int(len(x)),
        "estimate": estimate,
        "ci95": ci,
        "iterations": iterations,
        "valid": len(samples),
    }


def _fit_scaled_ridge(
    hidden: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> tuple[StandardScaler, Ridge, np.ndarray, float]:
    scaler = StandardScaler().fit(hidden)
    ridge = Ridge(alpha=float(alpha), solver="lsqr").fit(scaler.transform(hidden), target)
    raw, intercept = ridge_raw_space(scaler, ridge)  # Ridge exposes the same API as RidgeCV.
    return scaler, ridge, raw, intercept


def _nested_select_alpha(
    panel: ProtocolPanel,
    outer_fold: int,
    alphas: Sequence[float],
) -> tuple[float, dict[str, float]]:
    common_count = len(COMMON_PROTOCOLS)
    outer_train = panel.folds != outer_fold
    scores = panel.semantic_scores[:, :common_count]
    errors = {float(alpha): [] for alpha in alphas}
    for inner_fold in sorted(set(panel.folds[outer_train].tolist())):
        inner_train = outer_train & (panel.folds != inner_fold)
        inner_valid = outer_train & (panel.folds == inner_fold)
        target_model = fit_shared_target(scores[inner_train])
        train_target = target_model.transform(scores[inner_train])
        valid_target = target_model.transform(scores[inner_valid])
        train_hidden = panel.hidden[inner_train, :common_count].reshape(
            -1, panel.hidden.shape[-1]
        )
        valid_hidden = panel.hidden[inner_valid, :common_count].reshape(
            -1, panel.hidden.shape[-1]
        )
        repeated_train = np.repeat(train_target, common_count)
        repeated_valid = np.repeat(valid_target, common_count)
        for alpha in alphas:
            scaler, ridge, _, _ = _fit_scaled_ridge(train_hidden, repeated_train, alpha)
            prediction = ridge.predict(scaler.transform(valid_hidden))
            errors[float(alpha)].append(float(np.mean(np.square(prediction - repeated_valid))))
    mean_errors = {str(alpha): float(np.mean(values)) for alpha, values in errors.items()}
    selected = min(errors, key=lambda alpha: (float(np.mean(errors[alpha])), alpha))
    return float(selected), mean_errors


def _fit_oof_shared_direction(
    panel: ProtocolPanel,
    config: ComponentScreenConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[FoldFit]]:
    n_items, _, hidden_size = panel.hidden.shape
    common_count = len(COMMON_PROTOCOLS)
    predictions = np.full((n_items, len(ALL_PROTOCOLS)), np.nan, dtype=np.float64)
    coordinates = np.full_like(predictions, np.nan)
    target_oof = np.full(n_items, np.nan, dtype=np.float64)
    fits: list[FoldFit] = []
    scores = panel.semantic_scores[:, :common_count]
    for fold in sorted(set(panel.folds.tolist())):
        train = panel.folds != fold
        test = panel.folds == fold
        train_items = {panel.item_ids[index] for index in np.flatnonzero(train)}
        test_items = {panel.item_ids[index] for index in np.flatnonzero(test)}
        overlap = sorted(train_items.intersection(test_items))
        if overlap:
            raise RuntimeError(f"Item leakage in outer fold {fold}: {overlap}")
        target_model = fit_shared_target(scores[train])
        train_target = target_model.transform(scores[train])
        test_target = target_model.transform(scores[test])
        target_oof[test] = test_target
        alpha, inner_errors = _nested_select_alpha(panel, fold, config.alphas)
        train_hidden = panel.hidden[train, :common_count].reshape(-1, hidden_size)
        repeated_target = np.repeat(train_target, common_count)
        scaler, ridge, d_raw, intercept = _fit_scaled_ridge(
            train_hidden, repeated_target, alpha
        )
        norm = float(np.linalg.norm(d_raw))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError(f"Degenerate shared direction in fold {fold}")
        d_unit = d_raw / norm
        if (_safe_correlation(train_hidden @ d_unit, repeated_target, rank=False) or 0) < 0:
            d_unit = -d_unit
        train_z = train_hidden @ d_unit
        train_z_mean = float(np.mean(train_z))
        train_z_sd = float(np.std(train_z, ddof=1))
        if not math.isfinite(train_z_sd) or train_z_sd <= 1e-12:
            raise RuntimeError(f"Invalid training-only coordinate scale in fold {fold}")
        test_hidden = panel.hidden[test].reshape(-1, hidden_size)
        predictions[test] = ridge.predict(scaler.transform(test_hidden)).reshape(
            int(np.sum(test)), len(ALL_PROTOCOLS)
        )
        coordinates[test] = (
            (test_hidden @ d_unit - train_z_mean) / train_z_sd
        ).reshape(int(np.sum(test)), len(ALL_PROTOCOLS))
        audit = {
            "fold": int(fold),
            "train_items": sorted(train_items),
            "test_items": sorted(test_items),
            "train_item_count": len(train_items),
            "test_item_count": len(test_items),
            "item_overlap": overlap,
            "selected_alpha": alpha,
            "inner_grouped_mse": inner_errors,
            "target_explained_variance": target_model.explained_variance,
            "target_mean": target_model.mean.tolist(),
            "target_scale": target_model.scale.tolist(),
            "target_loading": target_model.loading.tolist(),
            "train_z_mean": train_z_mean,
            "train_z_sd": train_z_sd,
            "coordinate_scale_source": "outer-training items and seven common protocols only",
        }
        fits.append(
            FoldFit(
                fold=int(fold),
                alpha=alpha,
                d_raw=d_raw,
                d_unit=d_unit,
                raw_intercept=intercept,
                scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
                scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
                train_z_mean=train_z_mean,
                train_z_sd=train_z_sd,
                target_model=target_model,
                audit=audit,
            )
        )
    if not all(np.isfinite(values).all() for values in (predictions, coordinates, target_oof)):
        raise RuntimeError("OOF shared fit did not cover every item")
    return predictions, coordinates, target_oof, fits


def _fit_covariate_oof(
    panel: ProtocolPanel,
    target_oof: np.ndarray,
    fits: Sequence[FoldFit],
) -> np.ndarray:
    prediction = np.full(len(panel.rows), np.nan, dtype=np.float64)
    common_scores = panel.semantic_scores[:, : len(COMMON_PROTOCOLS)]
    by_fold = {fit.fold: fit for fit in fits}
    for fold in sorted(set(panel.folds.tolist())):
        train = panel.folds != fold
        test = panel.folds == fold
        model = by_fold[int(fold)].target_model
        train_target = model.transform(common_scores[train])
        scaler = StandardScaler().fit(panel.covariates[train])
        ridge = Ridge(alpha=1.0).fit(
            scaler.transform(panel.covariates[train]), train_target
        )
        prediction[test] = ridge.predict(scaler.transform(panel.covariates[test]))
    if not np.isfinite(prediction).all():
        raise RuntimeError("Covariate OOF baseline is incomplete")
    return prediction


def absolute_agreement_icc(values: np.ndarray) -> float | None:
    """Two-way absolute-agreement single-measure ICC(A,1)."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.isfinite(matrix).all():
        return None
    n, k = matrix.shape
    grand = float(matrix.mean())
    row_mean = matrix.mean(axis=1)
    column_mean = matrix.mean(axis=0)
    ms_rows = float(k * np.sum(np.square(row_mean - grand)) / (n - 1))
    ms_columns = float(n * np.sum(np.square(column_mean - grand)) / (k - 1))
    residual = matrix - row_mean[:, None] - column_mean[None, :] + grand
    ms_error = float(np.sum(np.square(residual)) / ((n - 1) * (k - 1)))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    if abs(denominator) <= 1e-12:
        return None
    value = (ms_rows - ms_error) / denominator
    return float(value) if math.isfinite(value) else None


def _mean_ci(values: np.ndarray, iterations: int, seed: int) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = [
        float(np.mean(array[rng.integers(0, len(array), size=len(array))]))
        for _ in range(iterations)
    ]
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def _linear_pair_bootstrap(
    reference: np.ndarray,
    other: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(other, dtype=np.float64)

    def fit(indices: np.ndarray) -> tuple[float, float] | None:
        selected_x, selected_y = x[indices], y[indices]
        if float(np.std(selected_x)) <= 1e-12:
            return None
        slope, intercept = np.polyfit(selected_x, selected_y, 1)
        return float(slope), float(intercept)

    estimate = fit(np.arange(len(x)))
    rng = np.random.default_rng(seed)
    sampled: list[tuple[float, float]] = []
    for _ in range(iterations):
        value = fit(rng.integers(0, len(x), size=len(x)))
        if value is not None and all(math.isfinite(part) for part in value):
            sampled.append(value)
    if estimate is None or not sampled:
        return {"slope": None, "intercept": None, "slope_ci95": [None, None], "intercept_ci95": [None, None]}
    samples = np.asarray(sampled)
    return {
        "slope": estimate[0],
        "intercept": estimate[1],
        "slope_ci95": [float(value) for value in np.quantile(samples[:, 0], [0.025, 0.975])],
        "intercept_ci95": [float(value) for value in np.quantile(samples[:, 1], [0.025, 0.975])],
    }


def coordinate_invariance_metrics(
    coordinates: np.ndarray,
    protocols: Sequence[str] = ALL_PROTOCOLS,
    *,
    config: ComponentScreenConfig | None = None,
) -> dict[str, Any]:
    """Evaluate absolute coordinates; no protocol-specific calibration is applied."""

    cfg = config or ComponentScreenConfig(expected_items=None)
    cfg.validate()
    matrix = np.asarray(coordinates, dtype=np.float64)
    names = tuple(protocols)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("Coordinate matrix and protocol names do not align")
    indices = {name: names.index(name) for name in names}
    if any(name not in indices for name in ALL_PROTOCOLS):
        raise ValueError("Coordinate gate requires all common and legacy protocols")
    common = matrix[:, [indices[name] for name in COMMON_PROTOCOLS]]
    icc = absolute_agreement_icc(common)
    rng = np.random.default_rng(cfg.seed + 701)
    icc_samples: list[float] = []
    for _ in range(cfg.bootstrap_iterations):
        sample = common[rng.integers(0, len(common), size=len(common))]
        value = absolute_agreement_icc(sample)
        if value is not None:
            icc_samples.append(value)
    icc_ci = [None, None]
    if icc_samples:
        icc_ci = [float(value) for value in np.quantile(icc_samples, [0.025, 0.975])]
    between_sd = float(np.std(common.mean(axis=1), ddof=1))
    within_sd = float(np.sqrt(np.mean(np.var(common, axis=1, ddof=1))))
    ratio = float(within_sd / between_sd) if between_sd > 1e-12 else math.inf
    band = float(cfg.equivalence_fraction * between_sd)
    reference = matrix[:, indices[COMMON_PROTOCOLS[0]]]
    comparisons: dict[str, Any] = {}
    for offset, name in enumerate(names[1:], start=1):
        other = matrix[:, indices[name]]
        difference = other - reference
        difference_ci = _mean_ci(
            difference, cfg.bootstrap_iterations, cfg.seed + 1000 + offset
        )
        regression = _linear_pair_bootstrap(
            reference,
            other,
            cfg.bootstrap_iterations,
            cfg.seed + 2000 + offset,
        )
        slope_ci = regression["slope_ci95"]
        intercept_ci = regression["intercept_ci95"]
        mean_equivalent = bool(
            band > 0
            and difference_ci[0] is not None
            and max(abs(float(difference_ci[0])), abs(float(difference_ci[1]))) <= band
        )
        slope_equivalent = bool(
            slope_ci[0] is not None
            and float(slope_ci[0]) >= cfg.slope_interval[0]
            and float(slope_ci[1]) <= cfg.slope_interval[1]
        )
        intercept_equivalent = bool(
            band > 0
            and intercept_ci[0] is not None
            and max(abs(float(intercept_ci[0])), abs(float(intercept_ci[1]))) <= band
        )
        comparisons[name] = {
            "paired_mean_difference": float(np.mean(difference)),
            "paired_mean_ci95": difference_ci,
            "spearman": _safe_correlation(reference, other, rank=True),
            **regression,
            "mean_equivalent": mean_equivalent,
            "slope_equivalent": slope_equivalent,
            "intercept_equivalent": intercept_equivalent,
            "coordinate_equivalent": bool(
                mean_equivalent and slope_equivalent and intercept_equivalent
            ),
        }
    common_pass = all(
        comparisons[name]["coordinate_equivalent"] for name in COMMON_PROTOCOLS[1:]
    )
    legacy_pass = all(
        comparisons[name]["coordinate_equivalent"] for name in LEGACY_PROTOCOLS
    )
    return {
        "reference_protocol": COMMON_PROTOCOLS[0],
        "calibration": "none; same fold direction, training origin, and training scale",
        "common_icc_a1": icc,
        "common_icc_bootstrap_ci95": icc_ci,
        "between_item_sd": between_sd,
        "within_item_protocol_sd": within_sd,
        "within_between_sd_ratio": ratio if math.isfinite(ratio) else None,
        "equivalence_band": band,
        "comparisons": comparisons,
        "common_pairwise_equivalence_passed": common_pass,
        "legacy_holdout_equivalence_passed": legacy_pass,
        "basic_components": {
            "icc_point": bool(icc is not None and icc >= cfg.icc_minimum),
            "icc_lower_ci": bool(
                icc_ci[0] is not None and float(icc_ci[0]) >= cfg.icc_ci_lower_minimum
            ),
            "within_between_ratio": bool(
                math.isfinite(ratio) and ratio <= cfg.within_between_sd_ratio_maximum
            ),
            "common_equivalence": common_pass,
            "legacy_equivalence": legacy_pass,
        },
    }


def _protocol_associations(
    predictions: np.ndarray,
    scores: np.ndarray,
    config: ComponentScreenConfig,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    return {
        name: _bootstrap_ci(
            predictions[:, index],
            scores[:, index],
            rank=True,
            iterations=config.bootstrap_iterations,
            seed=config.seed + seed_offset + index,
        )
        for index, name in enumerate(ALL_PROTOCOLS)
    }


def _fold_direction_stability(fits: Sequence[FoldFit]) -> dict[str, Any]:
    directions = np.stack([fit.d_unit for fit in fits])
    cosine = directions @ directions.T
    values = cosine[np.triu_indices(len(fits), 1)]
    return {
        "folds": [fit.fold for fit in fits],
        "pairwise_cosines": [float(value) for value in values],
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "all_positive": bool(np.all(values > 0)),
    }


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        key: float(value)
        for key, value in zip(
            ("q00", "q05", "q50", "q95", "q99", "q100"),
            np.quantile(array, [0, 0.05, 0.50, 0.95, 0.99, 1]),
        )
    }


def _percentile_pass(observed: float | None, null: Sequence[float], quantile: float) -> bool:
    if observed is None or not math.isfinite(observed):
        return False
    return bool(observed > float(np.quantile(np.asarray(null), quantile)))


def _permutation_controls(
    panel: ProtocolPanel,
    predictions: np.ndarray,
    coordinates: np.ndarray,
    target: np.ndarray,
    config: ComponentScreenConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed + 5000)
    common_count = len(COMMON_PROTOCOLS)
    consensus_prediction = predictions[:, :common_count].mean(axis=1)
    observed_target = _safe_correlation(consensus_prediction, target, rank=True)
    protocol_observed_values = [
        _safe_correlation(predictions[:, index], panel.semantic_scores[:, index], rank=True)
        for index in range(len(ALL_PROTOCOLS))
    ]
    observed_protocol_minimum = float(min(value for value in protocol_observed_values if value is not None))
    observed_icc = absolute_agreement_icc(coordinates[:, :common_count])
    target_null: list[float] = []
    protocol_null: list[float] = []
    pairing_null: list[float] = []
    for _ in range(config.control_iterations):
        target_null.append(
            _safe_correlation(consensus_prediction, rng.permutation(target), rank=True) or 0.0
        )
        shuffled_scores = np.column_stack(
            [rng.permutation(panel.semantic_scores[:, index]) for index in range(len(ALL_PROTOCOLS))]
        )
        values = [
            _safe_correlation(predictions[:, index], shuffled_scores[:, index], rank=True)
            or 0.0
            for index in range(len(ALL_PROTOCOLS))
        ]
        protocol_null.append(float(min(values)))
        shuffled_coordinates = coordinates[:, :common_count].copy()
        for index in range(1, common_count):
            shuffled_coordinates[:, index] = rng.permutation(shuffled_coordinates[:, index])
        pairing_null.append(absolute_agreement_icc(shuffled_coordinates) or 0.0)
    return {
        "item_target_permutation": {
            "observed_spearman": observed_target,
            "null_quantiles": _quantiles(target_null),
            "passed": _percentile_pass(observed_target, target_null, config.null_percentile),
        },
        "within_protocol_score_permutation": {
            "observed_minimum_protocol_spearman": observed_protocol_minimum,
            "null_quantiles": _quantiles(protocol_null),
            "passed": _percentile_pass(
                observed_protocol_minimum, protocol_null, config.null_percentile
            ),
        },
        "same_item_pairing_permutation": {
            "observed_common_icc_a1": observed_icc,
            "null_quantiles": _quantiles(pairing_null),
            "passed": _percentile_pass(observed_icc, pairing_null, config.null_percentile),
        },
    }


def _random_direction_controls(
    panel: ProtocolPanel,
    fits: Sequence[FoldFit],
    candidate_coordinates: np.ndarray,
    target: np.ndarray,
    config: ComponentScreenConfig,
) -> dict[str, Any]:
    iterations = config.random_direction_iterations
    random_coordinates = np.full(
        (len(panel.rows), len(ALL_PROTOCOLS), iterations), np.nan, dtype=np.float64
    )
    rng = np.random.default_rng(config.seed + 6000)
    by_fold = {fit.fold: fit for fit in fits}
    common_count = len(COMMON_PROTOCOLS)
    for fold in sorted(set(panel.folds.tolist())):
        train = panel.folds != fold
        test = panel.folds == fold
        candidate = by_fold[int(fold)].d_unit
        directions = rng.normal(size=(panel.hidden.shape[-1], iterations))
        directions -= candidate[:, None] * (candidate @ directions)[None, :]
        norms = np.linalg.norm(directions, axis=0)
        if np.any(norms <= 1e-12):
            raise RuntimeError("Degenerate random orthogonal direction")
        directions /= norms
        train_hidden = panel.hidden[train, :common_count].reshape(-1, panel.hidden.shape[-1])
        train_projection = train_hidden @ directions
        mean = train_projection.mean(axis=0)
        scale = train_projection.std(axis=0, ddof=1)
        if np.any(scale <= 1e-12):
            raise RuntimeError("Degenerate random direction training scale")
        test_hidden = panel.hidden[test].reshape(-1, panel.hidden.shape[-1])
        projected = ((test_hidden @ directions - mean) / scale).reshape(
            int(np.sum(test)), len(ALL_PROTOCOLS), iterations
        )
        random_coordinates[test] = projected
    if not np.isfinite(random_coordinates).all():
        raise RuntimeError("Random direction controls did not cover all items")
    random_rho: list[float] = []
    random_icc: list[float] = []
    random_joint: list[float] = []
    for iteration in range(iterations):
        values = random_coordinates[:, :, iteration]
        rho = _safe_correlation(values[:, :common_count].mean(axis=1), target, rank=True) or 0.0
        icc = absolute_agreement_icc(values[:, :common_count]) or 0.0
        random_rho.append(rho)
        random_icc.append(icc)
        random_joint.append(max(rho, 0.0) * max(icc, 0.0))
    candidate_rho = _safe_correlation(
        candidate_coordinates[:, :common_count].mean(axis=1), target, rank=True
    )
    candidate_icc = absolute_agreement_icc(candidate_coordinates[:, :common_count])
    candidate_joint = max(candidate_rho or 0.0, 0.0) * max(candidate_icc or 0.0, 0.0)
    return {
        "definition": "per-fold random unit directions orthogonal to the candidate; training-only origin and scale",
        "iterations": iterations,
        "semantic_spearman": {
            "observed": candidate_rho,
            "null_quantiles": _quantiles(random_rho),
            "passed": _percentile_pass(candidate_rho, random_rho, config.null_percentile),
        },
        "common_icc_a1": {
            "observed": candidate_icc,
            "null_quantiles": _quantiles(random_icc),
            "passed": _percentile_pass(candidate_icc, random_icc, config.null_percentile),
        },
        "joint_semantic_x_icc": {
            "observed": candidate_joint,
            "null_quantiles": _quantiles(random_joint),
            "passed": _percentile_pass(candidate_joint, random_joint, config.null_percentile),
        },
    }


def _direction_comparator(
    name: str,
    predictions: np.ndarray,
    coordinates: np.ndarray,
    panel: ProtocolPanel,
    target: np.ndarray,
    config: ComponentScreenConfig,
) -> dict[str, Any]:
    common_count = len(COMMON_PROTOCOLS)
    return {
        "name": name,
        "shared_target_rank": _bootstrap_ci(
            predictions[:, :common_count].mean(axis=1),
            target,
            rank=True,
            iterations=config.bootstrap_iterations,
            seed=config.seed + (8000 if name == "old_sa_direction" else 9000),
        ),
        "protocol_rank_transfer": _protocol_associations(
            predictions, panel.semantic_scores, config, seed_offset=8100
        ),
        "coordinate": coordinate_invariance_metrics(
            coordinates, panel.protocols, config=config
        ),
        "behavior_target_rank": _bootstrap_ci(
            coordinates[:, :common_count].mean(axis=1),
            panel.behavior_target,
            rank=True,
            iterations=config.bootstrap_iterations,
            seed=config.seed + (8200 if name == "old_sa_direction" else 9200),
        ),
        "claim_limit": (
            "comparison only; this pre-existing direction is not reclassified by the candidate gate"
        ),
    }


def classify_component(rank_passed: bool, coordinate_passed: bool) -> str:
    if coordinate_passed:
        return "existing_panel_coordinate_invariant_candidate_requires_random_mapping_confirmation"
    if rank_passed:
        return "rank_transport_candidate_only_no_coordinate_invariance"
    return "no_validated_shared_attribution_component_on_existing_panel"


def _markdown_summary(summary: dict[str, Any]) -> str:
    rank = summary["rank_gate"]
    coordinate = summary["coordinate_gate"]
    target = summary["oof_shared_target"]
    metrics = summary["coordinate_metrics"]
    controls = summary["controls"]

    def number(value: Any, digits: int = 3) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "# Protocol-Shared Internal Attribution Candidate — CPU Screen",
        "",
        f"- Classification: `{summary['classification']}`",
        f"- Items: {summary['n_items']} unique items; hidden site: L18 PANL.",
        (
            f"- Shared target OOF R²: {target['r2']:.3f}; Spearman: "
            f"{number(target['spearman']['estimate'])} "
            f"[{number(target['spearman']['ci95'][0])}, {number(target['spearman']['ci95'][1])}]."
        ),
        (
            f"- Covariate-only R²: {target['covariate_only_r2']:.3f}; incremental R²: "
            f"{target['incremental_r2_over_covariates']:.3f}."
        ),
        f"- Rank gate: **{'PASS' if rank['passed'] else 'FAIL'}**.",
        f"- Coordinate gate: **{'PASS' if coordinate['passed'] else 'FAIL'}**.",
        "",
        "## Rank transport",
        "",
        "| Protocol | OOF Spearman | 95% item-bootstrap CI | Role |",
        "|---|---:|---:|---|",
    ]
    for name, value in summary["protocol_rank_transfer"].items():
        role = "common training family" if name in COMMON_PROTOCOLS else "legacy grammar holdout"
        lines.append(
            f"| {name} | {number(value['estimate'])} | "
            f"[{number(value['ci95'][0])}, {number(value['ci95'][1])}] | {role} |"
        )
    lines.extend(
        [
            "",
            "All rank statistics are item-OOF. Positive rank transfer does not imply a common absolute coordinate.",
            "",
            "## Absolute coordinate audit",
            "",
            (
                f"Common-protocol ICC(A,1) = {number(metrics['common_icc_a1'])} "
                f"[{number(metrics['common_icc_bootstrap_ci95'][0])}, "
                f"{number(metrics['common_icc_bootstrap_ci95'][1])}]; "
                f"within/between SD ratio = {number(metrics['within_between_sd_ratio'])}; "
                f"equivalence band = ±{number(metrics['equivalence_band'])}."
            ),
            "",
            "| Protocol vs common_9_ordered | Mean difference (95% CI) | Slope 95% CI | Equivalent |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, value in metrics["comparisons"].items():
        lines.append(
            f"| {name} | {number(value['paired_mean_difference'])} "
            f"[{number(value['paired_mean_ci95'][0])}, {number(value['paired_mean_ci95'][1])}] | "
            f"[{number(value['slope_ci95'][0])}, {number(value['slope_ci95'][1])}] | "
            f"{'yes' if value['coordinate_equivalent'] else 'no'} |"
        )
    random_control = controls["random_orthogonal_directions"]
    old = controls["old_sa_direction"]
    behavior = controls["candidate_behavior_direction"]
    lines.extend(
        [
            "",
            "## Controls",
            "",
            (
                f"- Candidate joint semantic×ICC statistic: "
                f"{number(random_control['joint_semantic_x_icc']['observed'])}; random-direction "
                f"95th percentile: {number(random_control['joint_semantic_x_icc']['null_quantiles']['q95'])}."
            ),
            (
                f"- Old SA direction ↔ shared target: Spearman "
                f"{number(old['shared_target_rank']['estimate'])} "
                f"[{number(old['shared_target_rank']['ci95'][0])}, "
                f"{number(old['shared_target_rank']['ci95'][1])}]."
            ),
            (
                f"- Candidate behavior direction ↔ shared attribution target: Spearman "
                f"{number(behavior['shared_target_rank']['estimate'])} "
                f"[{number(behavior['shared_target_rank']['ci95'][0])}, "
                f"{number(behavior['shared_target_rank']['ci95'][1])}]."
            ),
            "- Item-target, within-protocol score, and same-item pairing permutation controls are all reported in `controls.json`.",
            "",
            "## Gate audit",
            "",
            "Rank components: "
            + ", ".join(
                f"{name}={'pass' if passed else 'fail'}"
                for name, passed in rank["components"].items()
            )
            + ".",
            "",
            "Coordinate components: "
            + ", ".join(
                f"{name}={'pass' if passed else 'fail'}"
                for name, passed in coordinate["components"].items()
            )
            + ".",
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
            "Protocol-specific affine or monotone calibration is never counted as coordinate invariance.",
            "A positive existing-panel screen still requires common-template random-mapping and row-order forwards.",
            "",
        ]
    )
    return "\n".join(lines)


def run_attribution_component_screen(
    bridge_dir: str | Path,
    output_dir: str | Path,
    *,
    config: ComponentScreenConfig | None = None,
) -> dict[str, Any]:
    """Run the auditable zero-GPU component screen and atomically write outputs."""

    cfg = config or ComponentScreenConfig()
    cfg.validate()
    bridge = Path(bridge_dir).resolve()
    output = Path(output_dir).resolve()
    if output == bridge or output in bridge.parents or bridge in output.parents:
        raise ValueError("Output must be a separate sibling tree, not an input ancestor/descendant")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Candidate-screen output already exists: {output}")
    panel = load_protocol_panel(bridge, expected_items=cfg.expected_items)
    predictions, coordinates, target, fits = _fit_oof_shared_direction(panel, cfg)
    covariate_prediction = _fit_covariate_oof(panel, target, fits)
    common_prediction = predictions[:, : len(COMMON_PROTOCOLS)].mean(axis=1)
    target_association = _bootstrap_ci(
        common_prediction,
        target,
        rank=True,
        iterations=cfg.bootstrap_iterations,
        seed=cfg.seed + 100,
    )
    residual_association = _bootstrap_ci(
        common_prediction - covariate_prediction,
        target - covariate_prediction,
        rank=True,
        iterations=cfg.bootstrap_iterations,
        seed=cfg.seed + 200,
    )
    target_r2 = float(r2_score(target, common_prediction))
    covariate_r2 = float(r2_score(target, covariate_prediction))
    protocol_rank = _protocol_associations(
        predictions, panel.semantic_scores, cfg, seed_offset=300
    )
    coordinate_metrics = coordinate_invariance_metrics(
        coordinates, panel.protocols, config=cfg
    )
    stability = _fold_direction_stability(fits)
    permutations = _permutation_controls(panel, predictions, coordinates, target, cfg)
    random_controls = _random_direction_controls(
        panel, fits, coordinates, target, cfg
    )
    old_comparator = _direction_comparator(
        "old_sa_direction",
        panel.old_sa_prediction,
        panel.old_sa_coordinate,
        panel,
        target,
        cfg,
    )
    behavior_comparator = _direction_comparator(
        "candidate_behavior_direction",
        panel.behavior_prediction,
        panel.behavior_coordinate,
        panel,
        target,
        cfg,
    )

    def lower_positive(metric: dict[str, Any]) -> bool:
        return bool(metric["ci95"][0] is not None and float(metric["ci95"][0]) > 0)

    common_rank_pass = all(lower_positive(protocol_rank[name]) for name in COMMON_PROTOCOLS)
    legacy_rank_pass = all(lower_positive(protocol_rank[name]) for name in LEGACY_PROTOCOLS)
    rank_components = {
        "shared_target_r2_positive": target_r2 > 0,
        "shared_target_spearman_lower_positive": lower_positive(target_association),
        "covariate_incremental_r2_positive": target_r2 - covariate_r2 > 0,
        "covariate_residual_spearman_lower_positive": lower_positive(residual_association),
        "all_common_protocol_rank_lower_positive": common_rank_pass,
        "all_legacy_holdout_rank_lower_positive": legacy_rank_pass,
        "item_target_permutation": permutations["item_target_permutation"]["passed"],
        "protocol_score_permutation": permutations["within_protocol_score_permutation"]["passed"],
        "random_direction_semantic": random_controls["semantic_spearman"]["passed"],
    }
    rank_passed = all(rank_components.values())
    basic = coordinate_metrics["basic_components"]
    coordinate_components = {
        "rank_gate": rank_passed,
        **basic,
        "same_item_pairing_control": permutations["same_item_pairing_permutation"]["passed"],
        "random_direction_icc": random_controls["common_icc_a1"]["passed"],
        "random_direction_joint": random_controls["joint_semantic_x_icc"]["passed"],
        "fold_direction_stability": bool(
            stability["all_positive"]
            and stability["median"] >= cfg.fold_direction_median_cosine_minimum
        ),
    }
    coordinate_passed = all(coordinate_components.values())
    classification = classify_component(rank_passed, coordinate_passed)
    if coordinate_passed:
        interpretation = (
            "The same OOF coordinate passes the existing common and legacy panel, but it remains a "
            "candidate—not a confirmed protocol-invariant Internal Source Attribution—until unseen "
            "random mappings and row orders are measured."
        )
    elif rank_passed:
        interpretation = (
            "A protocol-transportable semantic ordering is recoverable, but absolute coordinates, "
            "fold stability, or matched null controls fail.  This is rank transport only."
        )
    else:
        interpretation = (
            "The existing panel does not validate a shared attribution component under the strict "
            "item-OOF, holdout, covariate, and null-control gate."
        )
    summary = {
        "title": "Protocol-Shared Internal Attribution Candidate — Zero-GPU Screen",
        "status": "completed",
        "n_items": len(panel.rows),
        "protocols": {
            "training": list(COMMON_PROTOCOLS),
            "legacy_grammar_holdout": list(LEGACY_PROTOCOLS),
        },
        "target_definition": (
            "outer-training-only standardized PC1 of seven common protocol semantic scores; "
            "the same item target is repeated across protocol views"
        ),
        "fit_definition": (
            "nested item-fold StandardScaler + stacked shared Ridge; no protocol indicator or "
            "protocol-specific calibration"
        ),
        "oof_shared_target": {
            "r2": target_r2,
            "mae": float(mean_absolute_error(target, common_prediction)),
            "spearman": target_association,
            "covariate_only_r2": covariate_r2,
            "incremental_r2_over_covariates": target_r2 - covariate_r2,
            "residual_spearman": residual_association,
        },
        "protocol_rank_transfer": protocol_rank,
        "coordinate_metrics": coordinate_metrics,
        "fold_direction_stability": stability,
        "controls": {
            "permutations": permutations,
            "random_orthogonal_directions": random_controls,
            "covariates": ["final_answer_side", "difficulty", "prior_index"],
            "old_sa_direction": old_comparator,
            "candidate_behavior_direction": behavior_comparator,
            "behavior_direction_claim_limit": (
                "The pre-existing behavior direction failed its own behavioral target gate and is "
                "not called a validated reliance representation here."
            ),
        },
        "rank_gate": {
            "passed": rank_passed,
            "components": rank_components,
            "claim_if_passed": "protocol-transportable semantic rank only",
        },
        "coordinate_gate": {
            "passed": coordinate_passed,
            "components": coordinate_components,
            "claim_if_passed": (
                "existing-panel coordinate candidate requiring random-mapping confirmation"
            ),
        },
        "classification": classification,
        "interpretation": interpretation,
        "new_forward_requirement": {
            "required_for_strong_invariance_claim": True,
            "missing": [
                "at least three common-template random label bijections",
                "at least two class-row-order permutations",
                "preferably an identical answer-only prefix followed by a post-answer SA query branch",
            ],
            "old_30_item_random_mapping": (
                "preliminary external sensitivity only; one fixed mapping and legacy grammar"
            ),
        },
        "config": asdict(cfg),
    }

    output.mkdir(parents=True, exist_ok=True)
    direction_dir = output / "directions"
    direction_entries: list[dict[str, Any]] = []
    for fit in fits:
        filename = f"fold_{fit.fold}_layer_18_panl.npz"
        atomic_save_npz(
            direction_dir / filename,
            alpha=np.asarray(fit.alpha),
            d_raw=fit.d_raw,
            d_unit=fit.d_unit,
            raw_intercept=np.asarray(fit.raw_intercept),
            scaler_mean=fit.scaler_mean,
            scaler_scale=fit.scaler_scale,
            train_z_mean=np.asarray(fit.train_z_mean),
            train_z_sd=np.asarray(fit.train_z_sd),
            target_mean=fit.target_model.mean,
            target_scale=fit.target_model.scale,
            target_loading=fit.target_model.loading,
            target_explained_variance=np.asarray(fit.target_model.explained_variance),
        )
        direction_entries.append({"fold": fit.fold, "file": filename, **fit.audit})
    atomic_write_json(
        direction_dir / "index.json",
        {
            "format_version": 1,
            "definition": summary["fit_definition"],
            "target": summary["target_definition"],
            "protocol_specific_calibration": False,
            "folds": direction_entries,
        },
    )
    result_rows = []
    for index, row in enumerate(panel.rows):
        result_rows.append(
            {
                "case_id": panel.case_ids[index],
                "item_id": panel.item_ids[index],
                "fold": int(panel.folds[index]),
                "shared_target_oof": float(target[index]),
                "shared_prediction_oof": float(common_prediction[index]),
                "covariate_prediction_oof": float(covariate_prediction[index]),
                "protocols": {
                    name: {
                        "semantic_score": float(panel.semantic_scores[index, protocol_index]),
                        "shared_prediction_oof": float(predictions[index, protocol_index]),
                        "shared_coordinate_oof": float(coordinates[index, protocol_index]),
                        "old_sa_coordinate": float(panel.old_sa_coordinate[index, protocol_index]),
                        "behavior_coordinate": float(panel.behavior_coordinate[index, protocol_index]),
                    }
                    for protocol_index, name in enumerate(ALL_PROTOCOLS)
                },
            }
        )
    write_jsonl_atomic(output / "results.jsonl", result_rows)
    atomic_write_json(
        output / "cohort_manifest.json",
        {
            "n": len(panel.rows),
            "case_ids": list(panel.case_ids),
            "item_ids": list(panel.item_ids),
            "fold_counts": {
                str(fold): int(np.sum(panel.folds == fold))
                for fold in sorted(set(panel.folds.tolist()))
            },
            "protocols": list(panel.protocols),
            "source_results": str(panel.bridge_dir / "results.jsonl"),
            "source_results_sha256": sha256_file(panel.bridge_dir / "results.jsonl"),
            "hidden_files": [
                {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in panel.hidden_paths
            ],
        },
    )
    atomic_write_json(output / "controls.json", summary["controls"])
    atomic_write_json(output / "summary.json", summary)
    atomic_write_text(output / "summary.md", _markdown_summary(summary))
    return summary


__all__ = [
    "ALL_PROTOCOLS",
    "COMMON_PROTOCOLS",
    "LEGACY_PROTOCOLS",
    "ComponentScreenConfig",
    "ProtocolPanel",
    "SharedTargetModel",
    "absolute_agreement_icc",
    "classify_component",
    "coordinate_invariance_metrics",
    "fit_shared_target",
    "load_protocol_panel",
    "run_attribution_component_screen",
]
