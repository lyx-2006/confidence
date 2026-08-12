"""Development-fit, externally frozen source-reliance representation analysis.

This module is intentionally separate from :mod:`reliance_representation`.
It consumes method-v2 answer-only behavioral measurements, selects every
layer/position/Ridge hyperparameter using development items only, and applies
the resulting fold-specific model unchanged to the confirmatory items in the
same pre-existing item fold.

Two estimands are kept distinct throughout:

``raw_choice_coupled``
    Fold-standardized deletion/replacement effects for the naturally selected
    answer.  This estimand deliberately retains answer-choice coupling and can
    never be labelled a mediator by this module.

``graded_preregistered``
    Deletion/replacement effects residualized, using training rows only, on the
    explicitly preregistered nuisance design: answer side, answer identity,
    difficulty, prior strength, and Full answer margin.  There is no automatic
    nuisance discovery.

Natural prediction is not causal mediation.  Consequently this module always
emits ``causal_mediator_authorized=false``.  A graded candidate representation
additionally requires an untampered external measurement authorization whose
development and confirmatory donor/measurement gates passed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from layer_metacognition.hidden_state_store import atomic_write_json

from .core import (
    RIDGE_ALPHAS,
    atomic_save_npz,
    item_cluster_bootstrap,
    sha256_file,
    stable_hash,
    write_experiment_summary,
    write_jsonl_atomic,
)
from .reliance_measurement import MEASUREMENT_METHOD_VERSION
from .reliance_representation import (
    PanelCell,
    RELIANCE_LAYERS,
    RELIANCE_POSITIONS,
    fit_ridge,
    load_hidden_matrices,
)


BRIDGE_DIR = "stage3_sa_computational_bridge"
MEASUREMENT_DIR = "01_actual_source_reliance"
EXTERNAL_REPRESENTATION_DIR = "03_reliance_representation_devfit_confirm"
ESTIMANDS = ("raw_choice_coupled", "graded_preregistered")
OBJECTIVES = ("shared", "deletion", "replacement")
FORMAT_VERSION = 1


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _latest_rows(path: str | Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("measurement_method_version", -1)) != MEASUREMENT_METHOD_VERSION:
            raise ValueError(
                f"Mixed/legacy measurement method in {path}: "
                f"{row.get('measurement_method_version')!r}"
            )
        key = str(row.get("intervention_key", ""))
        if not key:
            raise ValueError(f"Measurement row in {path} lacks intervention_key")
        latest[key] = row
    return list(latest.values())


def load_measurement_rows(
    measurement_root: str | Path,
    split: str,
) -> list[dict[str, Any]]:
    """Load only completed method-v2 rows and reject split/schema mixing."""

    if split not in {"development", "confirmatory"}:
        raise ValueError(f"Unknown measurement split: {split}")
    path = Path(measurement_root) / f"{split}_results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    latest = _latest_rows(path)
    for row in latest:
        if row.get("experiment") != "clean_actual_source_reliance":
            raise ValueError(f"Unexpected experiment in {path}: {row.get('experiment')!r}")
        if row.get("split") != split:
            raise ValueError(
                f"Measurement split mixing in {path}: {row.get('split')!r}"
            )
        if row.get("status") not in {"completed", "excluded"}:
            raise ValueError(
                f"Non-terminal measurement row in {path}: {row.get('status')!r}"
            )
    return [dict(row) for row in latest if row.get("status") == "completed"]


def _summary_gate(summary: Mapping[str, Any], name: str) -> bool:
    value = summary.get(name)
    if not isinstance(value, Mapping) or value.get("gate_passed") is None:
        raise ValueError(f"Measurement summary lacks {name}.gate_passed")
    return bool(value["gate_passed"])


def build_measurement_authorization(
    measurement_root: str | Path,
) -> dict[str, Any]:
    """Build a fingerprinted authorization from the two frozen summaries."""

    root = Path(measurement_root)
    summaries: dict[str, Any] = {}
    files: dict[str, Any] = {}
    for split in ("development", "confirmatory"):
        path = root / f"{split}_summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("split") != split or summary.get("status") != "completed":
            raise ValueError(f"Invalid completed {split} measurement summary")
        summaries[split] = {
            "technical": _summary_gate(summary, "technical"),
            "raw": _summary_gate(summary, "raw_reliability"),
            "graded": _summary_gate(summary, "graded_reliability"),
            "donor": _summary_gate(summary, "donor_replicate_reliability"),
            "overall": bool(summary.get("measurement_gate_passed")),
            "completed": int(summary.get("completed", -1)),
            "unique_items": int(summary.get("unique_items", -1)),
        }
        files[split] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    raw_allowed = all(
        summaries[split][gate]
        for split in summaries
        for gate in ("technical", "raw")
    )
    graded_allowed = all(
        summaries[split][gate]
        for split in summaries
        for gate in ("technical", "raw", "graded", "donor", "overall")
    )
    frozen = root / "frozen_measurement_rule.json"
    if not frozen.is_file():
        raise FileNotFoundError(frozen)
    frozen_payload = json.loads(frozen.read_text(encoding="utf-8"))
    frozen_fingerprint = str(frozen_payload.get("rule_fingerprint", ""))
    frozen_without_fingerprint = dict(frozen_payload)
    frozen_without_fingerprint.pop("rule_fingerprint", None)
    if (
        not frozen_fingerprint
        or stable_hash(frozen_without_fingerprint) != frozen_fingerprint
    ):
        raise ValueError("Frozen measurement rule fingerprint mismatch")
    answer_vocabulary = (
        frozen_payload.get("calibration", {})
        .get("nuisance", {})
        .get("answer_vocabulary")
    )
    if not isinstance(answer_vocabulary, list) or len(answer_vocabulary) < 2:
        raise ValueError("Frozen measurement rule lacks its answer vocabulary")
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "measurement_method_version": MEASUREMENT_METHOD_VERSION,
        "source": "frozen method-v2 measurement summaries",
        "summaries": summaries,
        "source_files": files,
        "frozen_rule": {
            "path": str(frozen.resolve()),
            "sha256": sha256_file(frozen),
        },
        "answer_vocabulary": sorted({str(value) for value in answer_vocabulary}),
        "raw_readout_allowed": raw_allowed,
        "graded_candidate_allowed": graded_allowed,
        "causal_mediator_authorized": False,
        "reason": (
            "both measurement splits passed every target/donor gate"
            if graded_allowed
            else "at least one frozen measurement or donor gate failed"
        ),
    }
    payload["authorization_fingerprint"] = stable_hash(payload)
    return payload


def validate_authorization(value: Mapping[str, Any]) -> dict[str, Any]:
    authorization = dict(value)
    fingerprint = str(authorization.pop("authorization_fingerprint", ""))
    if not fingerprint or stable_hash(authorization) != fingerprint:
        raise ValueError("Measurement authorization fingerprint mismatch")
    if int(authorization.get("measurement_method_version", -1)) != MEASUREMENT_METHOD_VERSION:
        raise ValueError("Authorization is not for measurement method v2")
    if authorization.get("causal_mediator_authorized") is not False:
        raise ValueError("A predictive measurement authorization cannot authorize mediation")
    vocabulary = authorization.get("answer_vocabulary")
    if not isinstance(vocabulary, list) or len(set(map(str, vocabulary))) < 2:
        raise ValueError("Measurement authorization lacks a frozen answer vocabulary")
    authorization["authorization_fingerprint"] = fingerprint
    return authorization


def _validate_rows(
    development: Sequence[Mapping[str, Any]],
    confirmatory: Sequence[Mapping[str, Any]],
    *,
    answer_vocabulary: Sequence[str],
) -> list[int]:
    if not development or not confirmatory:
        raise ValueError("Both development and confirmatory rows are required")
    all_case_ids: set[str] = set()
    all_item_ids: set[str] = set()
    fold_sets: dict[str, set[int]] = {}
    vocabulary = {str(value) for value in answer_vocabulary}
    for split, rows in (("development", development), ("confirmatory", confirmatory)):
        folds: set[int] = set()
        for row in rows:
            case_id = str(row.get("case_id", ""))
            item_id = str(row.get("item_id", ""))
            if not case_id or case_id in all_case_ids:
                raise ValueError(f"case_id is missing/duplicated across splits: {case_id!r}")
            if not item_id or item_id in all_item_ids:
                raise ValueError(f"item_id is missing/duplicated across splits: {item_id!r}")
            all_case_ids.add(case_id)
            all_item_ids.add(item_id)
            if row.get("split") != split:
                raise ValueError(f"Row {case_id} is mixed into the wrong split")
            if row.get("status") != "completed":
                raise ValueError(f"Row {case_id} is not completed")
            if int(row.get("measurement_method_version", -1)) != MEASUREMENT_METHOD_VERSION:
                raise ValueError(f"Row {case_id} is not measurement method v2")
            if row.get("experiment") != "clean_actual_source_reliance":
                raise ValueError(f"Row {case_id} is from the wrong experiment")
            if bool(row.get("verbal_sa_leakage")):
                raise ValueError(f"Row {case_id} contains verbal-SA leakage")
            if row.get("teacher_forced_causal_prefix_equal") is not True:
                raise ValueError(f"Row {case_id} failed causal-prefix reconstruction")
            if row.get("selection_measurement_same_forward") is not True:
                raise ValueError(f"Row {case_id} did not reuse the Full selection forward")
            for key in (
                "answer_star",
                "answer_star_side",
                "difficulty",
                "prior_strength",
                "full_margin",
                "behavior_delete_imageward",
                "behavior_replace_imageward",
                "hidden_file",
                "fold",
            ):
                if row.get(key) is None:
                    raise ValueError(f"Row {case_id} lacks {key}")
            for key in (
                "prior_strength",
                "full_margin",
                "behavior_delete_imageward",
                "behavior_replace_imageward",
            ):
                _finite(row[key], f"{case_id}.{key}")
            fold = int(row["fold"])
            folds.add(fold)
            if str(row["answer_star"]) not in vocabulary:
                raise ValueError(
                    f"Row {case_id} answer is outside the frozen answer vocabulary"
                )
        fold_sets[split] = folds
    if fold_sets["development"] != fold_sets["confirmatory"]:
        raise ValueError("Development and confirmatory fold sets differ")
    if fold_sets["development"] != {0, 1, 2, 3, 4}:
        raise ValueError("The frozen five item folds 0..4 are required")
    return sorted(fold_sets["development"])


@dataclass(frozen=True)
class ExplicitNuisanceEncoder:
    answer_vocabulary: tuple[str, ...]
    answer_reference: str
    prior_mean: float
    prior_scale: float
    margin_mean: float
    margin_scale: float
    columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_vocabulary": list(self.answer_vocabulary),
            "answer_reference": self.answer_reference,
            "prior_mean": self.prior_mean,
            "prior_scale": self.prior_scale,
            "margin_mean": self.margin_mean,
            "margin_scale": self.margin_scale,
            "columns": list(self.columns),
        }


def _scale(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    scale = float(np.std(values, ddof=1)) if len(values) > 1 else 1.0
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return mean, scale


def fit_explicit_nuisance_encoder(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    answer_vocabulary: Sequence[str],
) -> ExplicitNuisanceEncoder:
    selected = [rows[int(index)] for index in indices]
    if len(selected) < 2:
        raise ValueError("Nuisance encoder needs at least two training rows")
    vocabulary = tuple(sorted({str(value) for value in answer_vocabulary}))
    if len(vocabulary) < 2:
        raise ValueError("Answer vocabulary must contain at least two classes")
    observed = {str(row["answer_star"]) for row in selected}
    if not observed.issubset(vocabulary):
        raise ValueError("Training answer is outside the frozen answer vocabulary")
    prior_mean, prior_scale = _scale(
        np.asarray([_finite(row["prior_strength"], "prior_strength") for row in selected])
    )
    margin_mean, margin_scale = _scale(
        np.asarray([_finite(row["full_margin"], "full_margin") for row in selected])
    )
    reference = vocabulary[0]
    columns = (
        "intercept",
        "choice_image",
        "choice_other",
        "difficulty_hard",
        "prior_strength",
        "full_margin",
        *(f"answer={answer}" for answer in vocabulary if answer != reference),
    )
    return ExplicitNuisanceEncoder(
        answer_vocabulary=vocabulary,
        answer_reference=reference,
        prior_mean=prior_mean,
        prior_scale=prior_scale,
        margin_mean=margin_mean,
        margin_scale=margin_scale,
        columns=tuple(columns),
    )


def transform_explicit_nuisance(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    encoder: ExplicitNuisanceEncoder,
) -> np.ndarray:
    output: list[list[float]] = []
    for index in indices:
        row = rows[int(index)]
        answer = str(row["answer_star"])
        if answer not in encoder.answer_vocabulary:
            raise ValueError(f"Answer {answer!r} is outside the frozen vocabulary")
        side = str(row["answer_star_side"])
        output.append(
            [
                1.0,
                float(side == "image"),
                float(side == "other"),
                float(str(row["difficulty"]) == "hard"),
                (_finite(row["prior_strength"], "prior_strength") - encoder.prior_mean)
                / encoder.prior_scale,
                (_finite(row["full_margin"], "full_margin") - encoder.margin_mean)
                / encoder.margin_scale,
                *[
                    float(answer == value)
                    for value in encoder.answer_vocabulary
                    if value != encoder.answer_reference
                ],
            ]
        )
    matrix = np.asarray(output, dtype=np.float64)
    if matrix.shape != (len(indices), len(encoder.columns)):
        raise RuntimeError("Explicit nuisance design has an unexpected shape")
    return matrix


@dataclass
class TargetTransform:
    estimand: str
    encoder: ExplicitNuisanceEncoder
    nuisance_beta: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray

    def apply(
        self,
        rows: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        x = transform_explicit_nuisance(rows, indices, self.encoder)
        y = np.asarray(
            [
                [
                    _finite(rows[int(index)]["behavior_delete_imageward"], "deletion"),
                    _finite(rows[int(index)]["behavior_replace_imageward"], "replacement"),
                ]
                for index in indices
            ],
            dtype=np.float64,
        )
        values = y if self.estimand == "raw_choice_coupled" else y - x @ self.nuisance_beta
        standardized = (values - self.target_mean) / self.target_scale
        return {
            "deletion": standardized[:, 0],
            "replacement": standardized[:, 1],
            "shared": standardized.mean(axis=1),
        }, x


def fit_target_transform(
    rows: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    *,
    estimand: str,
    answer_vocabulary: Sequence[str],
) -> TargetTransform:
    if estimand not in ESTIMANDS:
        raise ValueError(f"Unknown estimand: {estimand}")
    encoder = fit_explicit_nuisance_encoder(rows, train_indices, answer_vocabulary)
    x = transform_explicit_nuisance(rows, train_indices, encoder)
    y = np.asarray(
        [
            [
                _finite(rows[int(index)]["behavior_delete_imageward"], "deletion"),
                _finite(rows[int(index)]["behavior_replace_imageward"], "replacement"),
            ]
            for index in train_indices
        ],
        dtype=np.float64,
    )
    nuisance_beta = (
        np.zeros((x.shape[1], 2), dtype=np.float64)
        if estimand == "raw_choice_coupled"
        else np.linalg.lstsq(x, y, rcond=None)[0]
    )
    values = y if estimand == "raw_choice_coupled" else y - x @ nuisance_beta
    target_mean = values.mean(axis=0)
    target_scale = values.std(axis=0, ddof=1)
    if not np.isfinite(target_scale).all() or np.any(target_scale <= 1e-12):
        raise ValueError(f"Degenerate {estimand} target training scale")
    return TargetTransform(
        estimand=estimand,
        encoder=encoder,
        nuisance_beta=nuisance_beta,
        target_mean=target_mean,
        target_scale=target_scale,
    )


@dataclass
class HiddenTransform:
    estimand: str
    nuisance_beta: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    def apply(self, matrix: np.ndarray, indices: Sequence[int], x: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)[np.asarray(indices, dtype=np.int64)]
        residual = values if self.estimand == "raw_choice_coupled" else values - x @ self.nuisance_beta
        return (residual - self.feature_mean) / self.feature_scale


def fit_hidden_transform(
    matrix: np.ndarray,
    train_indices: Sequence[int],
    x_train: np.ndarray,
    *,
    estimand: str,
) -> HiddenTransform:
    values = np.asarray(matrix, dtype=np.float64)[np.asarray(train_indices, dtype=np.int64)]
    nuisance_beta = (
        np.zeros((x_train.shape[1], values.shape[1]), dtype=np.float64)
        if estimand == "raw_choice_coupled"
        else np.linalg.lstsq(x_train, values, rcond=None)[0]
    )
    residual = values if estimand == "raw_choice_coupled" else values - x_train @ nuisance_beta
    mean = residual.mean(axis=0)
    scale = residual.std(axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    return HiddenTransform(estimand, nuisance_beta, mean, scale)


def _inner_splits(folds: np.ndarray, outer_train: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in sorted(set(folds[outer_train].tolist())):
        validation = outer_train[folds[outer_train] == fold]
        training = outer_train[folds[outer_train] != fold]
        if len(training) >= 2 and len(validation):
            output.append((training, validation))
    if len(output) < 2:
        raise ValueError("Development outer train has fewer than two inner folds")
    return output


def _select_inner(
    rows: Sequence[Mapping[str, Any]],
    matrices: Mapping[PanelCell, np.ndarray],
    folds: np.ndarray,
    outer_train: np.ndarray,
    *,
    estimand: str,
    answer_vocabulary: Sequence[str],
    alphas: Sequence[float],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    losses = {
        objective: {
            (cell, float(alpha)): [0.0, 0]
            for cell in matrices
            for alpha in alphas
        }
        for objective in OBJECTIVES
    }
    audits: list[dict[str, Any]] = []
    for training, validation in _inner_splits(folds, outer_train):
        target_fit = fit_target_transform(
            rows, training, estimand=estimand, answer_vocabulary=answer_vocabulary
        )
        train_targets, x_train = target_fit.apply(rows, training)
        validation_targets, x_validation = target_fit.apply(rows, validation)
        for cell, matrix in matrices.items():
            hidden_fit = fit_hidden_transform(
                matrix, training, x_train, estimand=estimand
            )
            h_train = hidden_fit.apply(matrix, training, x_train)
            h_validation = hidden_fit.apply(matrix, validation, x_validation)
            for objective in OBJECTIVES:
                for alpha in alphas:
                    model = fit_ridge(h_train, train_targets[objective], float(alpha))
                    error = model.predict(h_validation) - validation_targets[objective]
                    record = losses[objective][(cell, float(alpha))]
                    record[0] += float(error @ error)
                    record[1] += len(error)
        audits.append(
            {
                "training_n": len(training),
                "validation_n": len(validation),
                "training_folds": sorted(set(folds[training].tolist())),
                "validation_folds": sorted(set(folds[validation].tolist())),
                "item_overlap": sorted(
                    {str(rows[index]["item_id"]) for index in training}.intersection(
                        {str(rows[index]["item_id"]) for index in validation}
                    )
                ),
            }
        )
    if any(value["item_overlap"] for value in audits):
        raise RuntimeError("Inner development selection leaked an item")
    selected: dict[str, dict[str, Any]] = {}
    loss_rows: dict[str, list[dict[str, Any]]] = {}
    for objective in OBJECTIVES:
        candidates = sorted(
            (
                total / count,
                cell.position,
                cell.layer,
                alpha,
                cell,
                count,
            )
            for (cell, alpha), (total, count) in losses[objective].items()
            if count
        )
        if not candidates:
            raise RuntimeError(f"No inner losses for {objective}")
        mse, _position, _layer, alpha, cell, count = candidates[0]
        selected[objective] = {
            "cell": cell,
            "alpha": float(alpha),
            "inner_mse": float(mse),
            "inner_validation_n": int(count),
        }
        loss_rows[objective] = [
            {
                **candidate.to_dict(),
                "alpha": float(candidate_alpha),
                "mse": float(total / count),
                "n": int(count),
            }
            for (candidate, candidate_alpha), (total, count) in sorted(
                losses[objective].items(),
                key=lambda item: (item[0][0].position, item[0][0].layer, item[0][1]),
            )
            if count
        ]
    return selected, {"splits": audits, "losses": loss_rows}


def _fit_baseline(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, target, rcond=None)[0]


def _association(
    rows: Sequence[dict[str, Any]],
    left: str,
    right: str,
    *,
    iterations: int,
) -> dict[str, Any]:
    x = np.asarray([float(row[left]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[right]) for row in rows], dtype=np.float64)
    if len(rows) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return {
            "n": len(rows),
            "pearson": None,
            "spearman": None,
            "spearman_item_bootstrap": {
                "estimate": None,
                "ci95": [None, None],
                "iterations": iterations,
                "valid": 0,
            },
        }
    bootstrap = item_cluster_bootstrap(
        rows,
        lambda sample: spearmanr(
            [float(row[left]) for row in sample],
            [float(row[right]) for row in sample],
        ).statistic,
        iterations=iterations,
    )
    return {
        "n": len(rows),
        "unique_items": len({str(row["item_id"]) for row in rows}),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
        "spearman_item_bootstrap": bootstrap,
    }


def _prediction_metrics(
    rows: Sequence[dict[str, Any]],
    target: str,
    prediction: str,
    *,
    iterations: int,
) -> dict[str, Any]:
    y = np.asarray([float(row[target]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row[prediction]) for row in rows], dtype=np.float64)
    fold_metrics = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        selected = [row for row in rows if int(row["fold"]) == fold]
        association = _association(selected, target, prediction, iterations=iterations)
        fold_metrics.append(
            {
                "fold": fold,
                "n": len(selected),
                "spearman": association["spearman"],
                "r2": float(
                    r2_score(
                        [float(row[target]) for row in selected],
                        [float(row[prediction]) for row in selected],
                    )
                ),
            }
        )
    return {
        "n": len(rows),
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "association": _association(rows, target, prediction, iterations=iterations),
        "fold_metrics": fold_metrics,
        "positive_fold_count": sum(
            value["spearman"] is not None and value["spearman"] > 0
            for value in fold_metrics
        ),
    }


def _split_summary(rows: Sequence[dict[str, Any]], *, iterations: int) -> dict[str, Any]:
    target_reliability = _association(
        rows, "target_deletion", "target_replacement", iterations=iterations
    )
    target_sign_agreement = float(
        np.mean(
            [
                float(row["target_deletion"]) * float(row["target_replacement"]) > 0
                for row in rows
            ]
        )
    )
    shared = _prediction_metrics(
        rows, "target_shared", "prediction_shared", iterations=iterations
    )
    baseline = _prediction_metrics(
        rows, "target_shared", "prediction_nuisance", iterations=iterations
    )
    cross = {
        "deletion_to_replacement": _association(
            rows, "prediction_deletion", "target_replacement", iterations=iterations
        ),
        "replacement_to_deletion": _association(
            rows, "prediction_replacement", "target_deletion", iterations=iterations
        ),
    }
    shared_lower = shared["association"]["spearman_item_bootstrap"]["ci95"][0]
    cross_lowers = [
        value["spearman_item_bootstrap"]["ci95"][0] for value in cross.values()
    ]
    # These are two separately fitted, non-nested predictors.  Their R2
    # difference is useful as a conservative gate (the hidden-only readout
    # must outperform the observed-covariate-only predictor), but it is not a
    # conditional or incremental R2.  A genuinely nested development-fit /
    # confirm-frozen sensitivity is implemented separately.
    hidden_minus_nuisance = shared["r2"] - baseline["r2"]
    statistical_gate = bool(
        shared["r2"] > 0
        and shared_lower is not None
        and shared_lower > 0
        and shared["positive_fold_count"] >= 4
        and hidden_minus_nuisance > 0
        and all(value is not None and value > 0 for value in cross_lowers)
    )
    return {
        "target_reliability": {
            "deletion_vs_replacement": target_reliability,
            "sign_agreement": target_sign_agreement,
        },
        "shared": shared,
        "nuisance_baseline": baseline,
        "hidden_minus_nuisance_r2": float(hidden_minus_nuisance),
        "r2_comparison_is_nested": False,
        "cross_method": cross,
        "statistical_gate_passed": statistical_gate,
        "gate_rule": (
            "R2>0; Spearman CI lower>0; >=4/5 positive folds; "
            "hidden-only R2 exceeds separately fitted nuisance-only R2; "
            "both frozen cross-method CI lowers>0"
        ),
    }


def _save_fold_direction(
    directory: Path,
    *,
    fold: int,
    objective: str,
    cell: PanelCell,
    alpha: float,
    target_fit: TargetTransform,
    hidden_fit: HiddenTransform,
    model: Any,
) -> dict[str, Any]:
    raw_direction = np.asarray(model.coefficient, dtype=np.float64) / hidden_fit.feature_scale
    norm = float(np.linalg.norm(raw_direction))
    unit = raw_direction / norm if norm > 1e-12 else np.zeros_like(raw_direction)
    filename = f"fold_{fold}_{objective}_layer_{cell.layer}_{cell.position}.npz"
    atomic_save_npz(
        directory / filename,
        ridge_coefficient=np.asarray(model.coefficient, dtype=np.float64),
        ridge_intercept=np.asarray(float(model.intercept)),
        raw_direction=raw_direction,
        unit_direction=unit,
        target_nuisance_beta=target_fit.nuisance_beta,
        target_mean=target_fit.target_mean,
        target_scale=target_fit.target_scale,
        hidden_nuisance_beta=hidden_fit.nuisance_beta,
        feature_mean=hidden_fit.feature_mean,
        feature_scale=hidden_fit.feature_scale,
    )
    return {
        "fold": int(fold),
        "objective": objective,
        **cell.to_dict(),
        "alpha": float(alpha),
        "file": filename,
        "direction_norm": norm,
        "estimand": target_fit.estimand,
        "explicit_nuisance": target_fit.encoder.to_dict(),
    }


def fit_external_estimand(
    development_rows: Sequence[Mapping[str, Any]],
    confirmatory_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    estimand: str,
    hidden_root: str | Path | None,
    layers: Sequence[int],
    positions: Sequence[str],
    alphas: Sequence[float],
    answer_vocabulary: Sequence[str],
    bootstrap_iterations: int,
) -> dict[str, Any]:
    """Select on development only and evaluate confirmatory without refitting."""

    if estimand not in ESTIMANDS:
        raise ValueError(f"Unknown estimand: {estimand}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dev = [dict(row) for row in development_rows]
    confirm = [dict(row) for row in confirmatory_rows]
    dev_matrices = load_hidden_matrices(
        dev, hidden_root=hidden_root, layers=layers, positions=positions
    )
    confirm_matrices = load_hidden_matrices(
        confirm, hidden_root=hidden_root, layers=layers, positions=positions
    )
    if set(dev_matrices) != set(confirm_matrices):
        raise RuntimeError("Development and confirmatory hidden panels differ")
    dev_folds = np.asarray([int(row["fold"]) for row in dev], dtype=np.int64)
    confirm_folds = np.asarray([int(row["fold"]) for row in confirm], dtype=np.int64)
    dev_predictions = {
        objective: np.full(len(dev), np.nan, dtype=np.float64) for objective in OBJECTIVES
    }
    confirm_predictions = {
        objective: np.full(len(confirm), np.nan, dtype=np.float64)
        for objective in OBJECTIVES
    }
    dev_targets = {
        objective: np.full(len(dev), np.nan, dtype=np.float64) for objective in OBJECTIVES
    }
    confirm_targets = {
        objective: np.full(len(confirm), np.nan, dtype=np.float64)
        for objective in OBJECTIVES
    }
    dev_baseline = np.full(len(dev), np.nan, dtype=np.float64)
    confirm_baseline = np.full(len(confirm), np.nan, dtype=np.float64)
    fold_audit: list[dict[str, Any]] = []
    direction_entries: list[dict[str, Any]] = []
    direction_dir = destination / "directions"
    for fold in sorted(set(dev_folds.tolist())):
        outer_train = np.flatnonzero(dev_folds != fold)
        dev_test = np.flatnonzero(dev_folds == fold)
        confirm_test = np.flatnonzero(confirm_folds == fold)
        if not len(confirm_test):
            raise ValueError(f"Confirmatory split has no rows in fold {fold}")
        train_items = {str(dev[index]["item_id"]) for index in outer_train}
        held_items = {
            *(str(dev[index]["item_id"]) for index in dev_test),
            *(str(confirm[index]["item_id"]) for index in confirm_test),
        }
        overlap = sorted(train_items.intersection(held_items))
        if overlap:
            raise RuntimeError(f"Fold {fold} training leaked held-out items: {overlap[:5]}")
        selected, inner_audit = _select_inner(
            dev,
            dev_matrices,
            dev_folds,
            outer_train,
            estimand=estimand,
            answer_vocabulary=answer_vocabulary,
            alphas=alphas,
        )
        target_fit = fit_target_transform(
            dev,
            outer_train,
            estimand=estimand,
            answer_vocabulary=answer_vocabulary,
        )
        train_target, x_train = target_fit.apply(dev, outer_train)
        dev_target, x_dev_test = target_fit.apply(dev, dev_test)
        confirm_target, x_confirm_test = target_fit.apply(confirm, confirm_test)
        for objective in OBJECTIVES:
            dev_targets[objective][dev_test] = dev_target[objective]
            confirm_targets[objective][confirm_test] = confirm_target[objective]
        baseline_beta = _fit_baseline(x_train, train_target["shared"])
        dev_baseline[dev_test] = x_dev_test @ baseline_beta
        confirm_baseline[confirm_test] = x_confirm_test @ baseline_beta
        fold_record: dict[str, Any] = {
            "fold": int(fold),
            "development_train_n": len(outer_train),
            "development_heldout_n": len(dev_test),
            "confirmatory_frozen_n": len(confirm_test),
            "development_train_items_sha256": stable_hash(sorted(train_items)),
            "heldout_item_overlap": overlap,
            "confirmatory_used_for_selection_or_fit": False,
            "selected": {},
            "inner_cv": inner_audit,
            "explicit_nuisance_columns": list(target_fit.encoder.columns),
        }
        for objective in OBJECTIVES:
            specification = selected[objective]
            cell = specification["cell"]
            matrix = dev_matrices[cell]
            hidden_fit = fit_hidden_transform(
                matrix, outer_train, x_train, estimand=estimand
            )
            h_train = hidden_fit.apply(matrix, outer_train, x_train)
            h_dev = hidden_fit.apply(matrix, dev_test, x_dev_test)
            h_confirm = hidden_fit.apply(
                confirm_matrices[cell], confirm_test, x_confirm_test
            )
            model = fit_ridge(
                h_train, train_target[objective], float(specification["alpha"])
            )
            dev_predictions[objective][dev_test] = model.predict(h_dev)
            confirm_predictions[objective][confirm_test] = model.predict(h_confirm)
            entry = _save_fold_direction(
                direction_dir,
                fold=fold,
                objective=objective,
                cell=cell,
                alpha=float(specification["alpha"]),
                target_fit=target_fit,
                hidden_fit=hidden_fit,
                model=model,
            )
            direction_entries.append(entry)
            fold_record["selected"][objective] = {
                **cell.to_dict(),
                "alpha": float(specification["alpha"]),
                "inner_mse": float(specification["inner_mse"]),
                "direction_file": entry["file"],
            }
        fold_audit.append(fold_record)
    arrays = [
        *dev_predictions.values(),
        *confirm_predictions.values(),
        *dev_targets.values(),
        *confirm_targets.values(),
        dev_baseline,
        confirm_baseline,
    ]
    if not all(np.isfinite(value).all() for value in arrays):
        raise RuntimeError("Dev-fit/frozen-confirm did not cover every row")

    def records(
        rows: Sequence[Mapping[str, Any]],
        targets: Mapping[str, np.ndarray],
        predictions: Mapping[str, np.ndarray],
        baseline: np.ndarray,
        split: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "split": split,
                "case_id": str(row["case_id"]),
                "item_id": str(row["item_id"]),
                "fold": int(row["fold"]),
                "estimand": estimand,
                "target_deletion": float(targets["deletion"][index]),
                "target_replacement": float(targets["replacement"][index]),
                "target_shared": float(targets["shared"][index]),
                "target_method_disagreement": float(
                    0.5
                    * (
                        targets["deletion"][index]
                        - targets["replacement"][index]
                    )
                ),
                "prediction_deletion": float(predictions["deletion"][index]),
                "prediction_replacement": float(predictions["replacement"][index]),
                "prediction_shared": float(predictions["shared"][index]),
                "prediction_method_disagreement": float(
                    0.5
                    * (
                        predictions["deletion"][index]
                        - predictions["replacement"][index]
                    )
                ),
                "prediction_nuisance": float(baseline[index]),
                "answer_star": str(row["answer_star"]),
                "answer_star_side": str(row["answer_star_side"]),
            }
            for index, row in enumerate(rows)
        ]

    dev_records = records(dev, dev_targets, dev_predictions, dev_baseline, "development")
    confirm_records = records(
        confirm,
        confirm_targets,
        confirm_predictions,
        confirm_baseline,
        "confirmatory",
    )
    write_jsonl_atomic(destination / "development_oof_predictions.jsonl", dev_records)
    write_jsonl_atomic(
        destination / "confirmatory_frozen_predictions.jsonl", confirm_records
    )
    atomic_write_json(destination / "fold_audit.json", {"folds": fold_audit})
    atomic_write_json(
        direction_dir / "index.json",
        {
            "format_version": FORMAT_VERSION,
            "estimand": estimand,
            "definition": (
                "development-only inner selection and fit; unchanged same-fold "
                "confirmatory evaluation"
            ),
            "confirmatory_used_for_selection_or_fit": False,
            "entries": direction_entries,
        },
    )
    summary = {
        "estimand": estimand,
        "development": _split_summary(dev_records, iterations=bootstrap_iterations),
        "confirmatory": _split_summary(
            confirm_records, iterations=bootstrap_iterations
        ),
        "confirmatory_used_for_selection_or_fit": False,
    }
    return summary


def fit_external_reliance_representation(
    development_rows: Sequence[Mapping[str, Any]],
    confirmatory_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    measurement_authorization: Mapping[str, Any],
    hidden_root: str | Path | None,
    layers: Sequence[int] = RELIANCE_LAYERS,
    positions: Sequence[str] = RELIANCE_POSITIONS,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    """Run both estimands while enforcing external claim authorization."""

    authorization = validate_authorization(measurement_authorization)
    answer_vocabulary = [str(value) for value in authorization["answer_vocabulary"]]
    folds = _validate_rows(
        development_rows,
        confirmatory_rows,
        answer_vocabulary=answer_vocabulary,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for estimand in ESTIMANDS:
        estimand_dir = output / estimand
        result = fit_external_estimand(
            development_rows,
            confirmatory_rows,
            estimand_dir,
            estimand=estimand,
            hidden_root=hidden_root,
            layers=tuple(int(value) for value in layers),
            positions=tuple(str(value) for value in positions),
            alphas=tuple(float(value) for value in alphas),
            answer_vocabulary=answer_vocabulary,
            bootstrap_iterations=bootstrap_iterations,
        )
        dev_pass = bool(result["development"]["statistical_gate_passed"])
        confirm_pass = bool(result["confirmatory"]["statistical_gate_passed"])
        statistical_pass = dev_pass and confirm_pass
        result["statistical_pattern_passed"] = statistical_pass
        if estimand == "raw_choice_coupled":
            readout_pass = statistical_pass and bool(
                authorization["raw_readout_allowed"]
            )
            result.update(
                {
                    "measurement_authorized": bool(
                        authorization["raw_readout_allowed"]
                    ),
                    "readout_gate_passed": readout_pass,
                    "candidate_source_use_representation": False,
                    "classification": (
                        "endpoint-coupled behavioral sensitivity readout"
                        if readout_pass
                        else "endpoint-coupled behavioral sensitivity was not externally validated"
                    ),
                }
            )
        else:
            candidate = statistical_pass and bool(
                authorization["graded_candidate_allowed"]
            )
            result.update(
                {
                    "measurement_authorized": bool(
                        authorization["graded_candidate_allowed"]
                    ),
                    # Do not expose an apparently positive readout gate when
                    # the behavioral target itself lacks the preregistered
                    # measurement/donor authorization.
                    "readout_gate_passed": candidate,
                    "candidate_source_use_representation": candidate,
                    "classification": (
                        "candidate noncausal source-use representation"
                        if candidate
                        else "exploratory graded representation; measurement/donor authorization is absent"
                    ),
                }
            )
        result["causal_mediator_authorized"] = False
        result["mediator_claim"] = (
            "prohibited: predictive decoding cannot establish causal mediation"
        )
        write_experiment_summary(estimand_dir, result)
        results[estimand] = result
    summary = {
        "title": "External Actual Source Reliance Representation",
        "status": "completed",
        "development_n": len(development_rows),
        "confirmatory_n": len(confirmatory_rows),
        "folds": folds,
        "answer_vocabulary": answer_vocabulary,
        "selection": (
            "development only; confirmatory never used for fit, scaling, nuisance "
            "estimation, or hyperparameter selection"
        ),
        "measurement_authorization": authorization,
        "estimands": results,
        "causal_mediator_authorized": False,
        "mediator_claim": "prohibited pending an independent causal intervention gate",
    }
    write_experiment_summary(output, summary)
    return summary
