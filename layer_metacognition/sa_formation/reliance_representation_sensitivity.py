"""Post-hoc, non-gating sensitivities for the frozen reliance readouts.

This module is deliberately downstream of, and separate from,
``03_reliance_representation_devfit_confirm``.  It never fits or changes a
hidden-state readout.  It performs two narrowly scoped analyses:

* transport the frozen fold-specific target transform from donor pair d1/d2
  to the prospectively collected donor pair d3/d4, then score the unchanged
  development-OOF / confirmatory-frozen predictions;
* fit two nested *linear calibration* models on development OOF predictions
  only and apply both unchanged to the confirmatory predictions.

Neither analysis is gate-bearing and neither can authorize causal mediation.
All joins are lineage checked against the method-v2 row fingerprint, message
hashes, cohort fingerprints, and the frozen representation identifiers.
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

from .core import (
    SEED,
    item_cluster_bootstrap,
    sha256_file,
    stable_hash,
    write_experiment_summary,
    write_jsonl_atomic,
)
from .donor_replication_extension import EXTENSION_DIR, EXTENSION_METHOD_VERSION
from .reliance_external_representation import (
    BRIDGE_DIR,
    ESTIMANDS,
    EXTERNAL_REPRESENTATION_DIR,
    MEASUREMENT_DIR,
    ExplicitNuisanceEncoder,
    transform_explicit_nuisance,
)
from .reliance_measurement import MEASUREMENT_METHOD_VERSION


SENSITIVITY_DIR = "04_reliance_representation_sensitivities"
FORMAT_VERSION = 1
OBJECTIVES = ("shared", "deletion", "replacement")
SPLITS = ("development", "confirmatory")


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_unique_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Non-object row at {path}:{line_number}")
        case_id = str(row.get("case_id", ""))
        if not case_id:
            raise ValueError(f"Missing case_id at {path}:{line_number}")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id {case_id!r} in {path}")
        seen.add(case_id)
        rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def validate_stable_fingerprint(
    payload: Mapping[str, Any], key: str, *, name: str
) -> str:
    """Validate a ``stable_hash`` field without mutating its source object."""

    value = dict(payload)
    fingerprint = str(value.pop(key, ""))
    if not fingerprint or stable_hash(value) != fingerprint:
        raise ValueError(f"{name} {key} mismatch")
    return fingerprint


def _index_unique(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in output:
            raise ValueError(f"{name} has a missing/duplicated case_id: {case_id!r}")
        output[case_id] = row
    return output


def _require_equal(case_id: str, field: str, values: Sequence[Any]) -> None:
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError(f"Lineage mismatch for {case_id}.{field}: {values!r}")


@dataclass(frozen=True)
class SplitLineage:
    measurement_analysis_sha256: str
    measurement_manifest_fingerprint: str
    extension_manifest_fingerprint: str
    measurement_calibration_fingerprint: str
    margin_repair_calibration_fingerprint: str


def strict_join_split(
    split: str,
    measurement_results: Sequence[Mapping[str, Any]],
    measurement_analysis: Sequence[Mapping[str, Any]],
    donor_analysis: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    lineage: SplitLineage,
) -> list[dict[str, Any]]:
    """Strictly join 01/02/03 rows and verify their row-level lineage.

    ``measurement_results`` may contain structural exclusions; only completed
    method-v2 rows participate.  The completed result must be an exact subset
    of its 01 analysis row, whose full stable hash must equal the fingerprint
    recorded prospectively by 02.
    """

    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if set(predictions) != set(ESTIMANDS):
        raise ValueError("Both raw and graded 03 prediction panels are required")
    completed_results = [
        dict(row)
        for row in measurement_results
        if row.get("status") == "completed"
        and int(row.get("measurement_method_version", -1))
        == MEASUREMENT_METHOD_VERSION
    ]
    sources = {
        "measurement_results": _index_unique(completed_results, "measurement results"),
        "measurement_analysis": _index_unique(measurement_analysis, "measurement analysis"),
        "donor_analysis": _index_unique(donor_analysis, "donor analysis"),
        **{
            f"prediction_{estimand}": _index_unique(rows, f"{estimand} predictions")
            for estimand, rows in predictions.items()
        },
    }
    case_sets = {name: set(values) for name, values in sources.items()}
    reference = case_sets["measurement_analysis"]
    if any(values != reference for values in case_sets.values()):
        counts = {name: len(values) for name, values in case_sets.items()}
        raise ValueError(f"01/02/03 case sets differ in {split}: {counts}")

    joined: list[dict[str, Any]] = []
    for case_id in sorted(reference):
        result = sources["measurement_results"][case_id]
        measurement = sources["measurement_analysis"][case_id]
        donor = sources["donor_analysis"][case_id]
        prediction_rows = {
            estimand: sources[f"prediction_{estimand}"][case_id]
            for estimand in ESTIMANDS
        }
        for key, value in result.items():
            if key not in measurement or measurement[key] != value:
                raise ValueError(
                    f"01 result/analysis drift for {case_id}.{key}"
                )
        if measurement.get("status") != "completed" or measurement.get("split") != split:
            raise ValueError(f"Invalid 01 analysis row for {case_id}")
        if int(measurement.get("measurement_method_version", -1)) != MEASUREMENT_METHOD_VERSION:
            raise ValueError(f"Legacy measurement row for {case_id}")
        if donor.get("status") != "completed" or donor.get("split") != split:
            raise ValueError(f"Invalid 02 analysis row for {case_id}")
        if int(donor.get("extension_method_version", -1)) != EXTENSION_METHOD_VERSION:
            raise ValueError(f"Legacy donor extension row for {case_id}")
        if donor.get("answer_star_reused") is not True:
            raise ValueError(f"02 did not reuse answer_star for {case_id}")
        if donor.get("full_messages_hash_equal") is not True:
            raise ValueError(f"02 Full message reconstruction failed for {case_id}")
        if donor.get("selection_reused_without_forward") is not True:
            raise ValueError(f"02 reselected the answer for {case_id}")
        if bool(donor.get("verbal_sa_leakage")) or bool(donor.get("hidden_captured")):
            raise ValueError(f"02 protocol contamination for {case_id}")

        for estimand, prediction in prediction_rows.items():
            if prediction.get("split") != split or prediction.get("estimand") != estimand:
                raise ValueError(f"Wrong 03 prediction panel for {case_id}/{estimand}")
        _require_equal(
            case_id,
            "case_id",
            [measurement["case_id"], donor["case_id"]]
            + [prediction_rows[value]["case_id"] for value in ESTIMANDS],
        )
        for field in ("item_id", "fold", "answer_star"):
            _require_equal(
                case_id,
                field,
                [measurement[field], donor[field]]
                + [prediction_rows[value][field] for value in ESTIMANDS],
            )
        _require_equal(
            case_id,
            "answer_star_side",
            [measurement["answer_star_side"], donor["answer_star_side"]]
            + [prediction_rows[value]["answer_star_side"] for value in ESTIMANDS],
        )
        if str(measurement.get("manifest_fingerprint", "")) != lineage.measurement_manifest_fingerprint:
            raise ValueError(f"01 manifest fingerprint drift for {case_id}")
        if str(measurement.get("calibration_fingerprint", "")) != lineage.measurement_calibration_fingerprint:
            raise ValueError(f"01 calibration fingerprint drift for {case_id}")
        if str(donor.get("manifest_fingerprint", "")) != lineage.extension_manifest_fingerprint:
            raise ValueError(f"02 manifest fingerprint drift for {case_id}")
        if str(donor.get("margin_repair_calibration_fingerprint", "")) != lineage.margin_repair_calibration_fingerprint:
            raise ValueError(f"02 margin-repair fingerprint drift for {case_id}")
        if str(donor.get("method_v2_analysis_sha256", "")) != lineage.measurement_analysis_sha256:
            raise ValueError(f"02 method-v2 file hash drift for {case_id}")
        if str(donor.get("method_v2_row_fingerprint", "")) != stable_hash(measurement):
            raise ValueError(f"02 method-v2 row fingerprint drift for {case_id}")

        selection_hash = str(measurement.get("selection_rendered_hash", ""))
        full_hash = str(measurement.get("selection", {}).get("messages_hash", ""))
        if not selection_hash or not full_hash:
            raise ValueError(f"01 selection hashes missing for {case_id}")
        _require_equal(
            case_id,
            "selection_rendered_hash",
            [selection_hash, donor.get("method_v2_selection_rendered_hash")],
        )
        _require_equal(
            case_id,
            "full_messages_hash",
            [
                full_hash,
                donor.get("method_v2_full_messages_hash"),
                donor.get("reconstructed_full_messages_hash"),
            ],
        )
        joined.append(
            {
                "case_id": case_id,
                "item_id": str(measurement["item_id"]),
                "fold": int(measurement["fold"]),
                "split": split,
                "measurement": measurement,
                "donor": donor,
                "predictions": prediction_rows,
            }
        )
    return joined


def _encoder_from_dict(value: Mapping[str, Any]) -> ExplicitNuisanceEncoder:
    required = {
        "answer_vocabulary",
        "answer_reference",
        "prior_mean",
        "prior_scale",
        "margin_mean",
        "margin_scale",
        "columns",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(f"Frozen nuisance encoder omits {sorted(missing)}")
    encoder = ExplicitNuisanceEncoder(
        answer_vocabulary=tuple(str(item) for item in value["answer_vocabulary"]),
        answer_reference=str(value["answer_reference"]),
        prior_mean=_finite(value["prior_mean"], "prior_mean"),
        prior_scale=_finite(value["prior_scale"], "prior_scale"),
        margin_mean=_finite(value["margin_mean"], "margin_mean"),
        margin_scale=_finite(value["margin_scale"], "margin_scale"),
        columns=tuple(str(item) for item in value["columns"]),
    )
    if encoder.prior_scale <= 0 or encoder.margin_scale <= 0:
        raise ValueError("Frozen nuisance scales must be positive")
    return encoder


@dataclass(frozen=True)
class FrozenTargetTransform:
    estimand: str
    fold: int
    encoder: ExplicitNuisanceEncoder
    nuisance_beta: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    source_files: tuple[str, ...]

    def apply(
        self,
        rows: Sequence[Mapping[str, Any]],
        deletion: Sequence[float],
        replacement: Sequence[float],
    ) -> dict[str, np.ndarray]:
        if len(rows) != len(deletion) or len(rows) != len(replacement):
            raise ValueError("Frozen target transform inputs have unequal lengths")
        if any(int(row["fold"]) != self.fold for row in rows):
            raise ValueError(f"Rows from another fold were sent to fold {self.fold}")
        indices = np.arange(len(rows), dtype=np.int64)
        x = transform_explicit_nuisance(rows, indices, self.encoder)
        y = np.column_stack(
            [
                np.asarray(deletion, dtype=np.float64),
                np.asarray(replacement, dtype=np.float64),
            ]
        )
        if not np.isfinite(y).all():
            raise ValueError("Fresh-donor endpoint contains non-finite values")
        values = (
            y
            if self.estimand == "raw_choice_coupled"
            else y - x @ self.nuisance_beta
        )
        standardized = (values - self.target_mean) / self.target_scale
        if not np.isfinite(standardized).all():
            raise ValueError("Frozen target transformation produced non-finite values")
        return {
            "deletion": standardized[:, 0],
            "replacement": standardized[:, 1],
            "shared": standardized.mean(axis=1),
        }


def load_frozen_target_transforms(
    representation_root: str | Path,
    estimand: str,
) -> dict[int, FrozenTargetTransform]:
    """Load target transforms from 03 and prove objective copies agree."""

    if estimand not in ESTIMANDS:
        raise ValueError(f"Unknown estimand: {estimand}")
    directory = Path(representation_root) / estimand / "directions"
    index_path = directory / "index.json"
    index = _read_json(index_path)
    if index.get("estimand") != estimand:
        raise ValueError(f"03 direction index estimand mismatch: {estimand}")
    if index.get("confirmatory_used_for_selection_or_fit") is not False:
        raise ValueError("03 direction index claims confirmatory fitting")
    entries = index.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No 03 direction entries for {estimand}")
    by_fold: dict[int, list[dict[str, Any]]] = {}
    for source in entries:
        entry = dict(source)
        fold = int(entry.get("fold", -1))
        by_fold.setdefault(fold, []).append(entry)
    transforms: dict[int, FrozenTargetTransform] = {}
    for fold, fold_entries in sorted(by_fold.items()):
        if {str(value.get("objective")) for value in fold_entries} != set(OBJECTIVES):
            raise ValueError(f"Fold {fold}/{estimand} lacks exactly three objectives")
        if len(fold_entries) != len(OBJECTIVES):
            raise ValueError(f"Fold {fold}/{estimand} has duplicated objectives")
        reference_encoder: ExplicitNuisanceEncoder | None = None
        reference_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        files: list[str] = []
        for entry in sorted(fold_entries, key=lambda value: str(value["objective"])):
            if entry.get("estimand") != estimand:
                raise ValueError(f"Direction entry estimand mismatch in fold {fold}")
            encoder = _encoder_from_dict(entry.get("explicit_nuisance", {}))
            filename = str(entry.get("file", ""))
            path = directory / filename
            if not filename or not path.is_file() or path.parent.resolve() != directory.resolve():
                raise ValueError(f"Invalid direction file for fold {fold}: {filename!r}")
            with np.load(path, allow_pickle=False) as archive:
                required = {"target_nuisance_beta", "target_mean", "target_scale"}
                if not required.issubset(archive.files):
                    raise ValueError(f"Direction file {path} omits target transform")
                arrays = tuple(
                    np.asarray(archive[key], dtype=np.float64).copy()
                    for key in ("target_nuisance_beta", "target_mean", "target_scale")
                )
            nuisance_beta, target_mean, target_scale = arrays
            if nuisance_beta.shape != (len(encoder.columns), 2):
                raise ValueError(f"Bad nuisance beta shape in {path}")
            if target_mean.shape != (2,) or target_scale.shape != (2,):
                raise ValueError(f"Bad target scale shape in {path}")
            if not all(np.isfinite(value).all() for value in arrays) or np.any(target_scale <= 0):
                raise ValueError(f"Non-finite/degenerate target transform in {path}")
            if reference_encoder is None:
                reference_encoder = encoder
                reference_arrays = arrays
            elif encoder != reference_encoder or not all(
                np.array_equal(left, right)
                for left, right in zip(reference_arrays or (), arrays, strict=True)
            ):
                raise ValueError(
                    f"Objective-specific target transforms differ in fold {fold}/{estimand}"
                )
            files.append(str(path.resolve()))
        assert reference_encoder is not None and reference_arrays is not None
        transforms[fold] = FrozenTargetTransform(
            estimand=estimand,
            fold=fold,
            encoder=reference_encoder,
            nuisance_beta=reference_arrays[0],
            target_mean=reference_arrays[1],
            target_scale=reference_arrays[2],
            source_files=tuple(files),
        )
    if set(transforms) != {0, 1, 2, 3, 4}:
        raise ValueError("Frozen target transforms must cover folds 0..4")
    return transforms


def build_fresh_donor_records(
    joined: Sequence[Mapping[str, Any]],
    transforms: Mapping[int, FrozenTargetTransform],
    *,
    estimand: str,
    replay_tolerance: float = 1e-10,
) -> list[dict[str, Any]]:
    """Transform D + fresh M34 and pair with unchanged 03 predictions."""

    output: list[dict[str, Any]] = []
    for fold in sorted(transforms):
        selected = [row for row in joined if int(row["fold"]) == fold]
        if not selected:
            raise ValueError(f"No joined rows for fold {fold}")
        measurements = [row["measurement"] for row in selected]
        deletion = [
            _finite(row["behavior_delete_imageward"], "behavior_delete_imageward")
            for row in measurements
        ]
        old_replacement = [
            _finite(row["behavior_replace_imageward"], "behavior_replace_imageward")
            for row in measurements
        ]
        fresh_replacement = [
            _finite(row["donor"]["behavior_replace_imageward_d34_mean"], "fresh_m34")
            for row in selected
        ]
        transform = transforms[fold]
        original = transform.apply(measurements, deletion, old_replacement)
        fresh = transform.apply(measurements, deletion, fresh_replacement)
        for local_index, row in enumerate(selected):
            prediction = row["predictions"][estimand]
            expected = {
                "deletion": _finite(prediction["target_deletion"], "target_deletion"),
                "replacement": _finite(
                    prediction["target_replacement"], "target_replacement"
                ),
                "shared": _finite(prediction["target_shared"], "target_shared"),
            }
            errors = {
                name: abs(float(original[name][local_index]) - expected[name])
                for name in expected
            }
            if max(errors.values()) > replay_tolerance:
                raise ValueError(
                    f"03 target transform replay failed for {row['case_id']}: {errors}"
                )
            output.append(
                {
                    "split": str(row["split"]),
                    "case_id": str(row["case_id"]),
                    "item_id": str(row["item_id"]),
                    "fold": fold,
                    "estimand": estimand,
                    "answer_star": str(row["measurement"]["answer_star"]),
                    "answer_star_side": str(row["measurement"]["answer_star_side"]),
                    "fresh_target_deletion": float(fresh["deletion"][local_index]),
                    "fresh_target_replacement_m34": float(
                        fresh["replacement"][local_index]
                    ),
                    "fresh_target_shared_d_m34": float(fresh["shared"][local_index]),
                    "frozen_prediction_replacement": _finite(
                        prediction["prediction_replacement"], "prediction_replacement"
                    ),
                    "frozen_prediction_shared": _finite(
                        prediction["prediction_shared"], "prediction_shared"
                    ),
                    "original_target_replay_max_abs_error": max(errors.values()),
                    "hidden_or_readout_refit": False,
                    "gate_bearing": False,
                }
            )
    return sorted(output, key=lambda row: (int(row["fold"]), str(row["case_id"])))


def _safe_association(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None, None
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def _bootstrap_metric(
    rows: Sequence[dict[str, Any]],
    target: str,
    prediction: str,
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    def statistic(sample: Sequence[dict[str, Any]]) -> float:
        y = np.asarray([float(row[target]) for row in sample], dtype=np.float64)
        pred = np.asarray([float(row[prediction]) for row in sample], dtype=np.float64)
        if metric == "r2":
            return float(r2_score(y, pred)) if len(y) >= 2 and np.std(y) > 1e-12 else math.nan
        if metric == "mae":
            return float(mean_absolute_error(y, pred))
        if metric == "spearman":
            return (
                float(spearmanr(y, pred).statistic)
                if len(y) >= 3 and np.std(y) > 1e-12 and np.std(pred) > 1e-12
                else math.nan
            )
        raise ValueError(f"Unknown bootstrap metric: {metric}")

    return item_cluster_bootstrap(
        rows, statistic, iterations=iterations, seed=seed
    )


def prediction_score(
    rows: Sequence[dict[str, Any]],
    target: str,
    prediction: str,
    *,
    bootstrap_iterations: int,
    seed: int = SEED,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot score an empty prediction panel")
    y = np.asarray([_finite(row[target], target) for row in rows], dtype=np.float64)
    pred = np.asarray(
        [_finite(row[prediction], prediction) for row in rows], dtype=np.float64
    )
    pearson, spearman = _safe_association(y, pred)
    fold_metrics = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if int(row["fold"]) == fold]
        fold_y = y[indices]
        fold_pred = pred[indices]
        _, fold_spearman = _safe_association(fold_y, fold_pred)
        fold_metrics.append(
            {
                "fold": fold,
                "n": len(indices),
                "r2": float(r2_score(fold_y, fold_pred)) if len(indices) >= 2 else None,
                "mae": float(mean_absolute_error(fold_y, fold_pred)),
                "spearman": fold_spearman,
            }
        )
    return {
        "n": len(rows),
        "unique_items": len({str(row["item_id"]) for row in rows}),
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "pearson": pearson,
        "spearman": spearman,
        "item_cluster_bootstrap": {
            metric: _bootstrap_metric(
                rows,
                target,
                prediction,
                metric,
                iterations=bootstrap_iterations,
                seed=seed + offset,
            )
            for offset, metric in enumerate(("r2", "mae", "spearman"))
        },
        "fold_metrics": fold_metrics,
        "positive_spearman_fold_count": sum(
            row["spearman"] is not None and row["spearman"] > 0
            for row in fold_metrics
        ),
    }


@dataclass(frozen=True)
class NestedCalibrators:
    estimand: str
    nuisance_coefficient: np.ndarray
    augmented_coefficient: np.ndarray
    development_case_ids_sha256: str
    development_n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimand": self.estimand,
            "nuisance_formula": "target_shared ~ 1 + prediction_nuisance",
            "augmented_formula": (
                "target_shared ~ 1 + prediction_nuisance + prediction_shared"
            ),
            "nuisance_coefficient": self.nuisance_coefficient.tolist(),
            "augmented_coefficient": self.augmented_coefficient.tolist(),
            "fit_split": "development",
            "development_n": self.development_n,
            "development_case_ids_sha256": self.development_case_ids_sha256,
            "confirmatory_used_for_fit": False,
            "post_hoc": True,
            "gate_bearing": False,
        }


def fit_nested_calibrators(
    development_oof: Sequence[Mapping[str, Any]], *, estimand: str
) -> NestedCalibrators:
    """Fit both nested calibrators using development OOF rows only."""

    if len(development_oof) < 3:
        raise ValueError("Nested calibration requires at least three development rows")
    for row in development_oof:
        if row.get("split") != "development" or row.get("estimand") != estimand:
            raise ValueError("Nested calibrator received a non-development/wrong-estimand row")
    target = np.asarray(
        [_finite(row["target_shared"], "target_shared") for row in development_oof],
        dtype=np.float64,
    )
    nuisance = np.asarray(
        [
            _finite(row["prediction_nuisance"], "prediction_nuisance")
            for row in development_oof
        ],
        dtype=np.float64,
    )
    hidden = np.asarray(
        [_finite(row["prediction_shared"], "prediction_shared") for row in development_oof],
        dtype=np.float64,
    )
    x_nuisance = np.column_stack([np.ones(len(target)), nuisance])
    x_augmented = np.column_stack([np.ones(len(target)), nuisance, hidden])
    nuisance_beta = np.linalg.lstsq(x_nuisance, target, rcond=None)[0]
    augmented_beta = np.linalg.lstsq(x_augmented, target, rcond=None)[0]
    if not np.isfinite(nuisance_beta).all() or not np.isfinite(augmented_beta).all():
        raise ValueError("Nested calibration produced non-finite coefficients")
    return NestedCalibrators(
        estimand=estimand,
        nuisance_coefficient=nuisance_beta,
        augmented_coefficient=augmented_beta,
        development_case_ids_sha256=stable_hash(
            sorted(str(row["case_id"]) for row in development_oof)
        ),
        development_n=len(development_oof),
    )


def apply_nested_calibrators(
    rows: Sequence[Mapping[str, Any]], calibrators: NestedCalibrators
) -> list[dict[str, Any]]:
    """Apply already-fitted development calibrators; no fitting occurs here."""

    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if row.get("estimand") != calibrators.estimand:
            raise ValueError("Nested calibrator estimand mismatch")
        nuisance = _finite(row["prediction_nuisance"], "prediction_nuisance")
        hidden = _finite(row["prediction_shared"], "prediction_shared")
        target = _finite(row["target_shared"], "target_shared")
        nuisance_prediction = float(
            np.asarray([1.0, nuisance]) @ calibrators.nuisance_coefficient
        )
        augmented_prediction = float(
            np.asarray([1.0, nuisance, hidden]) @ calibrators.augmented_coefficient
        )
        output.append(
            {
                "split": str(row["split"]),
                "case_id": str(row["case_id"]),
                "item_id": str(row["item_id"]),
                "fold": int(row["fold"]),
                "estimand": calibrators.estimand,
                "target_shared": target,
                "nuisance_calibrated_prediction": nuisance_prediction,
                "nuisance_plus_hidden_calibrated_prediction": augmented_prediction,
                "paired_squared_error_improvement": (
                    (target - nuisance_prediction) ** 2
                    - (target - augmented_prediction) ** 2
                ),
                "calibrator_fit_split": "development",
                "confirmatory_used_for_fit": False,
                "post_hoc": True,
                "gate_bearing": False,
            }
        )
    return output


def paired_squared_error_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    iterations: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """Bootstrap paired MSE improvement, positive when hidden prediction helps."""

    return item_cluster_bootstrap(
        rows,
        lambda sample: float(
            np.mean([float(row["paired_squared_error_improvement"]) for row in sample])
        ),
        iterations=iterations,
        seed=seed,
    )


def summarize_nested_predictions(
    rows: Sequence[dict[str, Any]], *, bootstrap_iterations: int
) -> dict[str, Any]:
    nuisance = prediction_score(
        rows,
        "target_shared",
        "nuisance_calibrated_prediction",
        bootstrap_iterations=bootstrap_iterations,
        seed=SEED + 101,
    )
    augmented = prediction_score(
        rows,
        "target_shared",
        "nuisance_plus_hidden_calibrated_prediction",
        bootstrap_iterations=bootstrap_iterations,
        seed=SEED + 211,
    )
    return {
        "nuisance_only": nuisance,
        "nuisance_plus_hidden": augmented,
        "delta_r2": float(augmented["r2"] - nuisance["r2"]),
        "delta_mae": float(augmented["mae"] - nuisance["mae"]),
        "paired_squared_error_improvement_item_bootstrap": (
            paired_squared_error_bootstrap(
                rows, iterations=bootstrap_iterations, seed=SEED + 307
            )
        ),
        "interpretation_of_positive_paired_improvement": (
            "nuisance+hidden has lower squared error than nuisance-only"
        ),
    }


def _prediction_filename(split: str) -> str:
    return (
        "development_oof_predictions.jsonl"
        if split == "development"
        else "confirmatory_frozen_predictions.jsonl"
    )


def _validate_manifest_rows(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    extension: bool,
) -> None:
    if manifest.get("split") != split:
        raise ValueError(f"Manifest split mismatch: {split}")
    manifest_rows = _index_unique(manifest.get("rows", []), f"{split} manifest")
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in manifest_rows:
            raise ValueError(f"{case_id} absent from {split} manifest")
        expected = manifest_rows[case_id]
        for field in ("item_id", "fold"):
            if str(expected[field]) != str(row[field]):
                raise ValueError(f"Manifest {field} mismatch for {case_id}")
        if extension:
            for field in (
                "answer_star",
                "method_v2_selection_rendered_hash",
                "method_v2_full_messages_hash",
            ):
                if expected[field] != row[field]:
                    raise ValueError(f"Extension manifest {field} mismatch for {case_id}")


@dataclass
class SensitivityInputs:
    bridge_root: Path
    measurement_root: Path
    extension_root: Path
    representation_root: Path
    joined: dict[str, list[dict[str, Any]]]
    prediction_rows: dict[str, dict[str, list[dict[str, Any]]]]
    gate_snapshot: dict[str, Any]
    lineage_audit: dict[str, Any]


def load_sensitivity_inputs(experiment_dir: str | Path) -> SensitivityInputs:
    """Load and fully validate the immutable 01/02/03 source artifacts."""

    bridge = Path(experiment_dir).resolve() / BRIDGE_DIR
    measurement_root = bridge / MEASUREMENT_DIR
    extension_root = bridge / EXTENSION_DIR
    representation_root = bridge / EXTERNAL_REPRESENTATION_DIR
    for path in (measurement_root, extension_root, representation_root):
        if not path.is_dir():
            raise FileNotFoundError(path)

    measurement_rule = _read_json(measurement_root / "frozen_measurement_rule.json")
    validate_stable_fingerprint(
        measurement_rule, "rule_fingerprint", name="measurement rule"
    )
    calibration = measurement_rule.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Measurement rule lacks calibration")
    measurement_calibration_fingerprint = validate_stable_fingerprint(
        calibration, "calibration_fingerprint", name="measurement calibration"
    )
    margin_repair = _read_json(extension_root / "full_margin_protocol_repair.json")
    margin_repair_fingerprint = validate_stable_fingerprint(
        margin_repair,
        "calibration_fingerprint",
        name="full-margin protocol repair",
    )
    extension_rule = _read_json(extension_root / "frozen_extension_rule.json")
    validate_stable_fingerprint(
        extension_rule, "rule_fingerprint", name="extension rule"
    )
    representation_config = _read_json(representation_root / "run_config.json")
    validate_stable_fingerprint(
        representation_config, "config_fingerprint", name="03 run configuration"
    )
    authorization = _read_json(representation_root / "measurement_authorization.json")
    validate_stable_fingerprint(
        authorization,
        "authorization_fingerprint",
        name="03 measurement authorization",
    )
    representation_summary = _read_json(representation_root / "summary.json")
    if representation_summary.get("status") != "completed":
        raise ValueError("03 representation summary is not completed")
    if representation_summary.get("causal_mediator_authorized") is not False:
        raise ValueError("03 unexpectedly authorizes causal mediation")
    if representation_config.get("causal_mediator_authorized") is not False:
        raise ValueError("03 configuration unexpectedly authorizes causal mediation")

    joined: dict[str, list[dict[str, Any]]] = {}
    prediction_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        estimand: {} for estimand in ESTIMANDS
    }
    lineage_audit: dict[str, Any] = {}
    for split in SPLITS:
        result_path = measurement_root / f"{split}_results.jsonl"
        analysis_path = measurement_root / f"{split}_analysis.jsonl"
        donor_path = extension_root / f"{split}_analysis.jsonl"
        measurement_results = _read_unique_jsonl(result_path)
        measurement_analysis = _read_unique_jsonl(analysis_path)
        donor_analysis = _read_unique_jsonl(donor_path)
        measurement_manifest = _read_json(
            measurement_root / f"{split}_cohort_manifest.json"
        )
        measurement_manifest_fingerprint = validate_stable_fingerprint(
            measurement_manifest,
            "manifest_fingerprint",
            name=f"01 {split} manifest",
        )
        extension_manifest = _read_json(
            extension_root / f"{split}_cohort_manifest.json"
        )
        extension_manifest_fingerprint = validate_stable_fingerprint(
            extension_manifest,
            "manifest_fingerprint",
            name=f"02 {split} manifest",
        )
        analysis_sha = sha256_file(analysis_path)
        if str(extension_manifest.get("method_v2_analysis_sha256", "")) != analysis_sha:
            raise ValueError(f"02 {split} manifest points to another 01 analysis file")
        if split == "confirmatory":
            if extension_rule.get("confirmatory_manifest_fingerprint") != extension_manifest_fingerprint:
                raise ValueError("Extension rule confirmatory manifest drift")
            if extension_rule.get("method_v2_analysis_sha256") != analysis_sha:
                raise ValueError("Extension rule method-v2 hash drift")
        completed_results = [
            row for row in measurement_results if row.get("status") == "completed"
        ]
        _validate_manifest_rows(
            measurement_manifest,
            completed_results,
            split=split,
            extension=False,
        )
        _validate_manifest_rows(
            extension_manifest,
            donor_analysis,
            split=split,
            extension=True,
        )
        split_predictions: dict[str, list[dict[str, Any]]] = {}
        for estimand in ESTIMANDS:
            rows = _read_unique_jsonl(
                representation_root / estimand / _prediction_filename(split)
            )
            split_predictions[estimand] = rows
            prediction_rows[estimand][split] = rows
        lineage = SplitLineage(
            measurement_analysis_sha256=analysis_sha,
            measurement_manifest_fingerprint=measurement_manifest_fingerprint,
            extension_manifest_fingerprint=extension_manifest_fingerprint,
            measurement_calibration_fingerprint=measurement_calibration_fingerprint,
            margin_repair_calibration_fingerprint=margin_repair_fingerprint,
        )
        joined[split] = strict_join_split(
            split,
            measurement_results,
            measurement_analysis,
            donor_analysis,
            split_predictions,
            lineage=lineage,
        )
        lineage_audit[split] = {
            "n": len(joined[split]),
            "case_ids_sha256": stable_hash(
                sorted(str(row["case_id"]) for row in joined[split])
            ),
            "measurement_analysis_sha256": analysis_sha,
            "measurement_manifest_fingerprint": measurement_manifest_fingerprint,
            "extension_manifest_fingerprint": extension_manifest_fingerprint,
            "all_case_item_fold_answer_hash_fingerprint_checks_passed": True,
        }

    gate_snapshot = {
        "source_summary_sha256": sha256_file(representation_root / "summary.json"),
        "source_config_fingerprint": representation_config["config_fingerprint"],
        "source_authorization_fingerprint": authorization[
            "authorization_fingerprint"
        ],
        "estimands": {
            estimand: {
                key: representation_summary["estimands"][estimand].get(key)
                for key in (
                    "measurement_authorized",
                    "readout_gate_passed",
                    "candidate_source_use_representation",
                    "classification",
                    "causal_mediator_authorized",
                )
            }
            for estimand in ESTIMANDS
        },
        "modified_by_sensitivity": False,
    }
    return SensitivityInputs(
        bridge_root=bridge,
        measurement_root=measurement_root,
        extension_root=extension_root,
        representation_root=representation_root,
        joined=joined,
        prediction_rows=prediction_rows,
        gate_snapshot=gate_snapshot,
        lineage_audit=lineage_audit,
    )


def run_representation_sensitivities(
    inputs: SensitivityInputs,
    output_dir: str | Path,
    *,
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    """Run the two post-hoc sensitivities and write only the new directory."""

    if bootstrap_iterations < 10:
        raise ValueError("bootstrap_iterations must be at least 10")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimand_summaries: dict[str, Any] = {}
    for estimand in ESTIMANDS:
        destination = output / estimand
        destination.mkdir(parents=True, exist_ok=True)
        transforms = load_frozen_target_transforms(
            inputs.representation_root, estimand
        )
        fresh_summary: dict[str, Any] = {}
        nested_summary: dict[str, Any] = {}
        calibrators = fit_nested_calibrators(
            inputs.prediction_rows[estimand]["development"], estimand=estimand
        )
        for split in SPLITS:
            fresh = build_fresh_donor_records(
                inputs.joined[split], transforms, estimand=estimand
            )
            write_jsonl_atomic(
                destination / f"{split}_fresh_donor_predictions.jsonl", fresh
            )
            fresh_summary[split] = {
                "replacement_only_m34": prediction_score(
                    fresh,
                    "fresh_target_replacement_m34",
                    "frozen_prediction_replacement",
                    bootstrap_iterations=bootstrap_iterations,
                    seed=SEED + (0 if split == "development" else 1000),
                ),
                "shared_deletion_plus_m34": prediction_score(
                    fresh,
                    "fresh_target_shared_d_m34",
                    "frozen_prediction_shared",
                    bootstrap_iterations=bootstrap_iterations,
                    seed=SEED + (100 if split == "development" else 1100),
                ),
                "max_original_target_replay_error": max(
                    float(row["original_target_replay_max_abs_error"])
                    for row in fresh
                ),
                "hidden_or_readout_refit": False,
            }
            nested = apply_nested_calibrators(
                inputs.prediction_rows[estimand][split], calibrators
            )
            write_jsonl_atomic(
                destination / f"{split}_nested_predictions.jsonl", nested
            )
            nested_summary[split] = summarize_nested_predictions(
                nested, bootstrap_iterations=bootstrap_iterations
            )
        summary = {
            "title": f"{estimand} representation sensitivities",
            "status": "completed",
            "estimand": estimand,
            "gate_bearing": False,
            "post_hoc": True,
            "fresh_donor_endpoint": {
                "definition": (
                    "unchanged 03 frozen predictions scored against D and donor3/4 M34 "
                    "after the exact fold-specific 03 target transform"
                ),
                "chronology": (
                    "donor3/4 was prospectively collected relative to the original 01 "
                    "measurement, but its outcomes existed before the 03 readout analysis; "
                    "this cross-target comparison is therefore post-hoc, not a prospective "
                    "external validation of the readout"
                ),
                "target_transform_refit": False,
                "hidden_or_readout_refit": False,
                "splits": fresh_summary,
            },
            "nested_calibration": {
                "definition": (
                    "development OOF linear calibration only; unchanged application "
                    "to confirmatory frozen predictions"
                ),
                "calibrators": calibrators.to_dict(),
                "splits": nested_summary,
                "confirmatory_used_for_fit": False,
                "development_calibration_evaluation_is_in_sample": True,
                "primary_interpretation_split": "confirmatory",
            },
            "original_03_gate_modified": False,
            "causal_mediator_authorized": False,
            "claim_limit": (
                "post-hoc sensitivity only; cannot reverse the original 03 gate, "
                "validate a source-use representation, or establish mediation"
            ),
        }
        write_experiment_summary(destination, summary)
        estimand_summaries[estimand] = summary
    root_summary = {
        "title": "Reliance Representation Sensitivities",
        "status": "completed",
        "format_version": FORMAT_VERSION,
        "development_n": len(inputs.joined["development"]),
        "confirmatory_n": len(inputs.joined["confirmatory"]),
        "lineage_audit": inputs.lineage_audit,
        "source_03_gate_snapshot": inputs.gate_snapshot,
        "estimands": estimand_summaries,
        "gate_bearing": False,
        "post_hoc": True,
        "original_03_gate_modified": False,
        "causal_mediator_authorized": False,
    }
    write_experiment_summary(output, root_summary)
    return root_summary


def required_input_files(experiment_dir: str | Path) -> list[Path]:
    """Return every immutable source file covered by aggregate provenance."""

    bridge = Path(experiment_dir).resolve() / BRIDGE_DIR
    measurement = bridge / MEASUREMENT_DIR
    extension = bridge / EXTENSION_DIR
    representation = bridge / EXTERNAL_REPRESENTATION_DIR
    paths = [
        *(measurement / f"{split}_{kind}.jsonl" for split in SPLITS for kind in ("results", "analysis")),
        *(measurement / f"{split}_cohort_manifest.json" for split in SPLITS),
        measurement / "frozen_measurement_rule.json",
        *(extension / f"{split}_analysis.jsonl" for split in SPLITS),
        *(extension / f"{split}_cohort_manifest.json" for split in SPLITS),
        extension / "full_margin_protocol_repair.json",
        extension / "frozen_extension_rule.json",
        *sorted(path for path in representation.rglob("*") if path.is_file()),
    ]
    unique = sorted({path.resolve() for path in paths}, key=str)
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError("Sensitivity inputs missing: " + ", ".join(missing))
    return unique


def input_provenance(experiment_dir: str | Path) -> dict[str, Any]:
    bridge = Path(experiment_dir).resolve() / BRIDGE_DIR
    files: dict[str, Any] = {}
    for path in required_input_files(experiment_dir):
        relative = str(path.relative_to(bridge))
        files[relative] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    aggregate_payload = {
        key: {"sha256": value["sha256"], "bytes": value["bytes"]}
        for key, value in files.items()
    }
    return {
        "files": files,
        "aggregate_sha256": stable_hash(aggregate_payload),
        "coverage": {
            "01_measurement_rows": True,
            "02_donor_analysis_rows": True,
            "03_all_directions_and_predictions": True,
        },
    }
