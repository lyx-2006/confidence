"""Read-only protocol sensitivity checks for Actual Source Reliance.

The frozen research plan specified that the graded reliance estimand controls
the Full-context answer margin.  The production v2 calibration did not include
that column.  This module reconstructs the specified linear nuisance model as
a *separate sensitivity analysis*: development rows are cross-fitted by item
fold and confirmatory rows receive the matching frozen development-fold model.

Nothing here mutates input rows or replaces the original measurement gate.
The only write is an atomic JSON document at a caller-supplied path.  Donor
reuse, leave-one-item-out (LOO) influence, and single- versus two-donor-average
consistency are reported to make a near-threshold result interpretable without
changing any threshold after seeing the data.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr, t

from layer_metacognition.hidden_state_store import atomic_write_json

from .core import stable_hash


METHOD_COLUMNS = {
    "deletion": "behavior_delete_imageward",
    "replacement": "behavior_replace_imageward",
}
SENSITIVITY_KEYS = {
    "deletion": "full_margin_residual_delete",
    "replacement": "full_margin_residual_replace",
}
REQUIRED_FIELDS = {
    "case_id",
    "item_id",
    "fold",
    "answer_star",
    "answer_star_side",
    "difficulty",
    "prior_strength",
    "full_margin",
    "behavior_delete_imageward",
    "behavior_replace_imageward",
    "behavior_replace_imageward_d1",
    "behavior_replace_imageward_d2",
    "donor1_item_id",
    "donor2_item_id",
}


@dataclass(frozen=True)
class ProtocolSensitivityConfig:
    bootstrap_iterations: int = 1000
    seed: int = 42
    loo_top_n: int = 10
    reliability_reference: float = 0.60

    def validate(self) -> None:
        if self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations must be non-negative")
        if self.loo_top_n < 1:
            raise ValueError("loo_top_n must be positive")
        if not 0.0 < self.reliability_reference < 1.0:
            raise ValueError("reliability_reference must lie in (0, 1)")


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite")
    return number


def _validate_rows(rows: Sequence[dict[str, Any]], label: str) -> None:
    if len(rows) < 3:
        raise ValueError(f"{label} needs at least three rows")
    case_ids: set[str] = set()
    item_ids: set[str] = set()
    for row in rows:
        missing = REQUIRED_FIELDS.difference(row)
        if missing:
            raise ValueError(f"{label} row omits {sorted(missing)}")
        case_id = str(row["case_id"])
        item_id = str(row["item_id"])
        if case_id in case_ids or item_id in item_ids:
            raise ValueError(f"{label} must contain unique cases and items")
        case_ids.add(case_id)
        item_ids.add(item_id)
        if str(row["answer_star_side"]) not in {"text", "image", "other"}:
            raise ValueError(f"Unknown answer_star_side in {case_id}")
        for field in (
            "prior_strength",
            "full_margin",
            *METHOD_COLUMNS.values(),
            "behavior_replace_imageward_d1",
            "behavior_replace_imageward_d2",
        ):
            _finite(row[field], field)


def _feature_spec(answer_vocabulary: Sequence[str]) -> dict[str, Any]:
    vocabulary = sorted({str(value) for value in answer_vocabulary})
    if len(vocabulary) < 2:
        raise ValueError("At least two answer labels are required")
    reference = vocabulary[0]
    return {
        "answer_vocabulary": vocabulary,
        "answer_reference": reference,
        "feature_names": [
            "intercept",
            "choice_image",
            "choice_other",
            "difficulty_hard",
            "prior_strength",
            "full_margin",
            *[f"answer={label}" for label in vocabulary if label != reference],
        ],
    }


def _nuisance_vector(row: dict[str, Any], spec: dict[str, Any]) -> np.ndarray:
    answer = str(row["answer_star"])
    vocabulary = list(spec["answer_vocabulary"])
    if answer not in vocabulary:
        raise ValueError(f"Answer {answer!r} is outside the frozen vocabulary")
    reference = str(spec["answer_reference"])
    side = str(row["answer_star_side"])
    return np.asarray(
        [
            1.0,
            float(side == "image"),
            float(side == "other"),
            float(str(row["difficulty"]) == "hard"),
            _finite(row["prior_strength"], "prior_strength"),
            _finite(row["full_margin"], "full_margin"),
            *[float(answer == label) for label in vocabulary if label != reference],
        ],
        dtype=np.float64,
    )


def fit_full_margin_calibration(
    development_rows: Sequence[dict[str, Any]],
    *,
    answer_vocabulary: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit fold-specific development-only nuisance models.

    For fold ``f``, only development items outside ``f`` are used.  The same
    fitted coefficients can therefore be applied to both development fold
    ``f`` (cross-fit) and an independent confirmatory fold ``f``.
    """

    _validate_rows(development_rows, "development")
    vocabulary = list(answer_vocabulary or sorted({str(row["answer_star"]) for row in development_rows}))
    spec = _feature_spec(vocabulary)
    design = np.stack([_nuisance_vector(row, spec) for row in development_rows])
    folds = np.asarray([int(row["fold"]) for row in development_rows], dtype=np.int64)
    unique_folds = sorted(set(folds.tolist()))
    if len(unique_folds) < 2:
        raise ValueError("At least two item folds are required")
    calibration: dict[str, Any] = {
        "format_version": 1,
        "analysis_only": True,
        "definition": "development-fold nuisance residualization including linear Full answer margin",
        "nuisance": spec,
        "method_columns": dict(METHOD_COLUMNS),
        "folds": {},
    }
    for fold in unique_folds:
        train = folds != fold
        test = folds == fold
        if int(train.sum()) < design.shape[1]:
            raise ValueError(f"Fold {fold} has fewer training rows than nuisance columns")
        train_items = {str(development_rows[index]["item_id"]) for index in np.flatnonzero(train)}
        test_items = {str(development_rows[index]["item_id"]) for index in np.flatnonzero(test)}
        if train_items.intersection(test_items):
            raise RuntimeError(f"Item leakage in fold {fold}")
        singular = np.linalg.svd(design[train], compute_uv=False)
        design_rank = int(np.linalg.matrix_rank(design[train]))
        entry: dict[str, Any] = {
            "train_n": int(train.sum()),
            "development_test_n": int(test.sum()),
            "train_items_sha256": stable_hash(sorted(train_items)),
            "development_test_items_sha256": stable_hash(sorted(test_items)),
            "design_columns": int(design.shape[1]),
            "design_rank": design_rank,
            "condition_number": (
                None
                if design_rank < design.shape[1]
                or singular[-1] <= np.finfo(np.float64).eps
                else float(singular[0] / singular[-1])
            ),
            "methods": {},
        }
        for method, column in METHOD_COLUMNS.items():
            outcome = np.asarray([float(row[column]) for row in development_rows], dtype=np.float64)
            beta = np.linalg.lstsq(design[train], outcome[train], rcond=None)[0]
            entry["methods"][method] = {"nuisance_beta": beta.tolist()}
        calibration["folds"][str(fold)] = entry
    calibration["calibration_fingerprint"] = stable_hash(calibration)
    return calibration


def apply_full_margin_calibration(
    rows: Sequence[dict[str, Any]],
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return copied rows with sensitivity residuals; never mutate inputs."""

    _validate_rows(rows, "application")
    expected = dict(calibration)
    fingerprint = str(expected.pop("calibration_fingerprint"))
    if stable_hash(expected) != fingerprint:
        raise ValueError("Full-margin calibration fingerprint mismatch")
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        fold = str(int(row["fold"]))
        if fold not in calibration["folds"]:
            raise ValueError(f"Calibration omits fold {fold}")
        vector = _nuisance_vector(row, calibration["nuisance"])
        predictions: dict[str, float] = {}
        for method, column in METHOD_COLUMNS.items():
            beta = np.asarray(
                calibration["folds"][fold]["methods"][method]["nuisance_beta"],
                dtype=np.float64,
            )
            prediction = float(vector @ beta)
            predictions[method] = prediction
            row[SENSITIVITY_KEYS[method]] = float(row[column]) - prediction
        # A donor replicate is an alternate observation of the replacement
        # family, so the frozen replacement prediction is subtracted from each.
        row["full_margin_residual_donor1"] = (
            float(row["behavior_replace_imageward_d1"]) - predictions["replacement"]
        )
        row["full_margin_residual_donor2"] = (
            float(row["behavior_replace_imageward_d2"]) - predictions["replacement"]
        )
        row["full_margin_calibration_fingerprint"] = fingerprint
        output.append(row)
    return output


def _values(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _association(rows: Sequence[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    x = _values(rows, left)
    y = _values(rows, right)
    if len(rows) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return {"n": len(rows), "pearson": None, "spearman": None, "kendall": None, "alpha_two_indicator": None, "sign_agreement": None}
    pearson = float(pearsonr(x, y).statistic)
    return {
        "n": len(rows),
        "pearson": pearson,
        "spearman": float(spearmanr(x, y).statistic),
        "kendall": float(kendalltau(x, y).statistic),
        "alpha_two_indicator": float(2.0 * pearson / (1.0 + pearson)),
        "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))),
    }


def _icc_consistency(matrix: np.ndarray, *, average: bool) -> float | None:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        return None
    n, k = values.shape
    subject_means = values.mean(axis=1)
    grand = float(values.mean())
    ms_between = float(k * np.sum((subject_means - grand) ** 2) / (n - 1))
    residual = values - subject_means[:, None] - values.mean(axis=0)[None, :] + grand
    ms_error = float(np.sum(residual**2) / ((n - 1) * (k - 1)))
    if average:
        return None if ms_between <= 0 else float((ms_between - ms_error) / ms_between)
    denominator = ms_between + (k - 1) * ms_error
    return None if denominator <= 0 else float((ms_between - ms_error) / denominator)


def _bootstrap_ci(
    rows: Sequence[dict[str, Any]],
    statistic: Callable[[Sequence[dict[str, Any]]], float | None],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any] | None:
    if iterations == 0:
        return None
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sample = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        value = statistic(sample)
        if value is not None and math.isfinite(float(value)):
            estimates.append(float(value))
    if not estimates:
        return None
    return {
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "valid": len(estimates),
        "iterations": iterations,
    }


def _loo_correlation(
    rows: Sequence[dict[str, Any]],
    left: str,
    right: str,
    *,
    top_n: int,
    reference: float,
) -> dict[str, Any]:
    base = _association(rows, left, right)["pearson"]
    if base is None:
        return {"n": len(rows), "base_pearson": None}
    estimates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample = [value for offset, value in enumerate(rows) if offset != index]
        value = _association(sample, left, right)["pearson"]
        if value is not None:
            estimates.append(
                {
                    "case_id": str(row["case_id"]),
                    "item_id": str(row["item_id"]),
                    "pearson_without_item": value,
                    "delta_from_full": value - base,
                }
            )
    alpha_reference_r = reference / (2.0 - reference)
    ordered = sorted(estimates, key=lambda value: abs(value["delta_from_full"]), reverse=True)
    values = [entry["pearson_without_item"] for entry in estimates]
    return {
        "n": len(rows),
        "base_pearson": base,
        "pearson_range": [min(values), max(values)],
        "max_absolute_delta": max(abs(value - base) for value in values),
        "reference_alpha": reference,
        "equivalent_pearson_reference": alpha_reference_r,
        "deletions_at_or_above_reference": sum(value >= alpha_reference_r for value in values),
        "top_influential": ordered[:top_n],
    }


def _loo_icc(rows: Sequence[dict[str, Any]], left: str, right: str, *, top_n: int) -> dict[str, Any]:
    def value(sample: Sequence[dict[str, Any]], average: bool = False) -> float | None:
        return _icc_consistency(np.column_stack((_values(sample, left), _values(sample, right))), average=average)

    base = value(rows)
    base_average = value(rows, True)
    estimates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample = [entry for offset, entry in enumerate(rows) if offset != index]
        single = value(sample)
        average = value(sample, True)
        if single is not None and average is not None and base is not None:
            estimates.append(
                {
                    "case_id": str(row["case_id"]),
                    "item_id": str(row["item_id"]),
                    "icc_single_without_item": single,
                    "icc_average_without_item": average,
                    "delta_single_from_full": single - base,
                }
            )
    ordered = sorted(estimates, key=lambda entry: abs(entry["delta_single_from_full"]), reverse=True)
    return {
        "n": len(rows),
        "icc_single": base,
        "icc_two_donor_average": base_average,
        "icc_single_range": [
            min(entry["icc_single_without_item"] for entry in estimates),
            max(entry["icc_single_without_item"] for entry in estimates),
        ],
        "top_influential": ordered[:top_n],
    }


def _donor_summary(rows: Sequence[dict[str, Any]], *, top_n: int) -> dict[str, Any]:
    donor_ids = [str(row[f"donor{position}_item_id"]) for row in rows for position in (1, 2)]
    counts = Counter(donor_ids)
    d1 = _values(rows, "behavior_replace_imageward_d1")
    d2 = _values(rows, "behavior_replace_imageward_d2")
    difference = d1 - d2
    standard_error = float(np.std(difference, ddof=1) / math.sqrt(len(difference)))
    critical = float(t.ppf(0.975, len(difference) - 1))
    raw = _loo_icc(rows, "behavior_replace_imageward_d1", "behavior_replace_imageward_d2", top_n=top_n)
    adjusted = _loo_icc(rows, "full_margin_residual_donor1", "full_margin_residual_donor2", top_n=top_n)
    single = adjusted["icc_single"]
    extrapolated: dict[str, float | None] = {}
    for donors in range(1, 7):
        denominator = 1.0 + (donors - 1) * single if single is not None else None
        extrapolated[str(donors)] = (
            None if single is None or denominator is None or denominator <= 0 else float(donors * single / denominator)
        )
    return {
        "slots": len(donor_ids),
        "unique_donor_items": len(counts),
        "maximum_reuse": max(counts.values()),
        "top_reused": [
            {"donor_item_id": donor, "uses": uses}
            for donor, uses in counts.most_common(top_n)
        ],
        "position_check": {
            "donor1_mean": float(np.mean(d1)),
            "donor2_mean": float(np.mean(d2)),
            "paired_mean_difference": float(np.mean(difference)),
            "paired_mean_difference_ci95": [
                float(np.mean(difference) - critical * standard_error),
                float(np.mean(difference) + critical * standard_error),
            ],
        },
        "raw_reliability": raw,
        "full_margin_adjusted_reliability": adjusted,
        "iid_spearman_brown_extrapolation_from_adjusted_single_donor": {
            "warning": "descriptive only; assumes exchangeable independent donor errors and must not be used to rescue the frozen gate",
            "reliability_by_total_donors": extrapolated,
        },
    }


def _split_summary(
    rows: Sequence[dict[str, Any]],
    *,
    config: ProtocolSensitivityConfig,
    seed_offset: int,
) -> dict[str, Any]:
    adjusted = _association(rows, SENSITIVITY_KEYS["deletion"], SENSITIVITY_KEYS["replacement"])
    adjusted["pearson_bootstrap"] = _bootstrap_ci(
        rows,
        lambda sample: _association(sample, SENSITIVITY_KEYS["deletion"], SENSITIVITY_KEYS["replacement"])["pearson"],
        iterations=config.bootstrap_iterations,
        seed=config.seed + seed_offset,
    )
    adjusted["spearman_bootstrap"] = _bootstrap_ci(
        rows,
        lambda sample: _association(sample, SENSITIVITY_KEYS["deletion"], SENSITIVITY_KEYS["replacement"])["spearman"],
        iterations=config.bootstrap_iterations,
        seed=config.seed + seed_offset + 1,
    )
    folds = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        selected = [row for row in rows if int(row["fold"]) == fold]
        folds.append(
            {
                "fold": fold,
                **_association(selected, SENSITIVITY_KEYS["deletion"], SENSITIVITY_KEYS["replacement"]),
            }
        )
    original = None
    if all("graded_residual_delete" in row and "graded_residual_replace" in row for row in rows):
        original = _association(rows, "graded_residual_delete", "graded_residual_replace")
    return {
        "n": len(rows),
        "unique_items": len({str(row["item_id"]) for row in rows}),
        "full_margin_adjusted": adjusted,
        "full_margin_adjusted_fold_metrics": folds,
        "full_margin_adjusted_loo": _loo_correlation(
            rows,
            SENSITIVITY_KEYS["deletion"],
            SENSITIVITY_KEYS["replacement"],
            top_n=config.loo_top_n,
            reference=config.reliability_reference,
        ),
        "stored_original_graded_for_comparison": original,
        "stored_original_graded_loo": (
            None
            if original is None
            else _loo_correlation(
                rows,
                "graded_residual_delete",
                "graded_residual_replace",
                top_n=config.loo_top_n,
                reference=config.reliability_reference,
            )
        ),
        "donors": _donor_summary(rows, top_n=config.loo_top_n),
    }


def run_reliance_protocol_sensitivity(
    development_rows: Sequence[dict[str, Any]],
    confirmatory_rows: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    config: ProtocolSensitivityConfig | None = None,
    answer_vocabulary: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the sensitivity analysis and atomically write its JSON summary."""

    settings = config or ProtocolSensitivityConfig()
    settings.validate()
    _validate_rows(development_rows, "development")
    _validate_rows(confirmatory_rows, "confirmatory")
    development_items = {str(row["item_id"]) for row in development_rows}
    confirmatory_items = {str(row["item_id"]) for row in confirmatory_rows}
    overlap = development_items.intersection(confirmatory_items)
    if overlap:
        raise ValueError(f"Development/confirmatory item overlap: {sorted(overlap)}")

    # Label identities are metadata, not outcomes.  Taking their union merely
    # makes an unseen confirmatory label auditable; its all-zero development
    # column receives a zero least-squares coefficient.
    vocabulary = list(
        answer_vocabulary
        or sorted(
            {
                str(row["answer_star"])
                for row in [*development_rows, *confirmatory_rows]
            }
        )
    )
    calibration = fit_full_margin_calibration(
        development_rows,
        answer_vocabulary=vocabulary,
    )
    development = apply_full_margin_calibration(development_rows, calibration)
    confirmatory = apply_full_margin_calibration(confirmatory_rows, calibration)
    development_answers = {str(row["answer_star"]) for row in development_rows}
    confirmatory_answers = {str(row["answer_star"]) for row in confirmatory_rows}
    result = {
        "title": "Actual Source Reliance protocol sensitivity — Full answer margin",
        "status": "completed",
        "analysis_class": "pre-registered-covariate reconstruction reported as non-gating sensitivity",
        "config": asdict(settings),
        "original_gate": {
            "overwritten": False,
            "replacement_authorized": False,
            "statement": "The production development and confirmatory gate decisions remain authoritative and unchanged.",
        },
        "input_audit": {
            "development_n": len(development_rows),
            "confirmatory_n": len(confirmatory_rows),
            "item_overlap_n": 0,
            "answer_vocabulary": vocabulary,
            "confirmatory_answers_unseen_in_development": sorted(confirmatory_answers - development_answers),
            "confirmatory_n_by_fold": {
                str(fold): sum(int(row["fold"]) == fold for row in confirmatory_rows)
                for fold in sorted({int(row["fold"]) for row in confirmatory_rows})
            },
        },
        "calibration": calibration,
        "splits": {
            "development": _split_summary(development, config=settings, seed_offset=0),
            "confirmatory": _split_summary(confirmatory, config=settings, seed_offset=10),
            "combined_descriptive_only": _split_summary(
                [*development, *confirmatory], config=settings, seed_offset=20
            ),
        },
        "claim_limit": (
            "This analysis diagnoses specification and donor-sampling sensitivity. "
            "It cannot turn a failed frozen confirmatory gate into a pass, and combined metrics are not independent confirmation."
        ),
    }
    atomic_write_json(output_path, result)
    return result


__all__ = [
    "ProtocolSensitivityConfig",
    "apply_full_margin_calibration",
    "fit_full_margin_calibration",
    "run_reliance_protocol_sensitivity",
]
