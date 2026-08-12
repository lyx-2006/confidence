"""Answer-only internal representation of Actual Source Reliance.

This module deliberately treats behavioral source reliance and verbal Source
Attribution as different outcomes.  It consumes answer-only measurement rows
and hidden-state panels, performs strictly nested item-fold out-of-fold (OOF)
model selection, and optionally applies the resulting fold-specific readout to
Text-first/Image-first History pairs without refitting.

The expected measurement-row schema is intentionally small::

    {
      "case_id": "...", "item_id": "...", "fold": 0,
      "behavior_delete_imageward": 1.2,
      "behavior_replace_imageward": 0.7,
      "nuisance": {
        "answer_identity": "cyan", "final_side": "image", ...
      },
      "hidden_file": "hidden/case.npz"
    }

Each NPZ must contain ``hidden`` (positions x layers x hidden, or layers x
positions x hidden), ``positions``/``position_names``, and
``layers``/``layer_indices``.  No prompt containing a Source Attribution request
is permitted by the GPU capture helper below.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from layer_metacognition.hidden_state_store import atomic_write_json

from .core import (
    RIDGE_ALPHAS,
    SEED,
    atomic_save_npz,
    canonical_message_hash,
    item_cluster_bootstrap,
    paired_effect_summary,
    write_experiment_summary,
    write_jsonl_atomic,
)


RELIANCE_LAYERS = (8, 12, 16, 20, 24, 27)
RELIANCE_POSITIONS = ("pre_answer", "post_answer")
DELETE_KEY = "behavior_delete_imageward"
REPLACE_KEY = "behavior_replace_imageward"
OBJECTIVES = ("shared", "deletion", "replacement")
FORMAT_VERSION = 1


@dataclass(frozen=True)
class PanelCell:
    layer: int
    position: str

    @property
    def key(self) -> str:
        return f"{self.position}|L{self.layer}"

    def to_dict(self) -> dict[str, Any]:
        return {"layer": self.layer, "position": self.position, "key": self.key}


@dataclass
class PreparedTargets:
    train: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    encoder: dict[str, Any]
    x_train: np.ndarray
    x_test: np.ndarray
    nuisance_beta: np.ndarray
    residual_mean: np.ndarray
    residual_scale: np.ndarray


@dataclass
class PreparedHidden:
    train: np.ndarray
    test: np.ndarray
    nuisance_beta: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray


@dataclass
class RidgeFit:
    coefficient: np.ndarray
    intercept: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) @ self.coefficient + self.intercept


def _finite_float(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _first_finite(mapping: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if mapping.get(key) is None:
            continue
        try:
            return _finite_float(mapping[key], name=key)
        except (TypeError, ValueError):
            continue
    return None


def _replicate_average(
    mapping: Mapping[str, Any], stems: Sequence[str]
) -> float | None:
    for stem in stems:
        values = [
            _first_finite(mapping, [f"{stem}_d1"]),
            _first_finite(mapping, [f"{stem}_d2"]),
        ]
        observed = [value for value in values if value is not None]
        if len(observed) == 2:
            return float(np.mean(observed))
    return None


def derive_reliance_indicators(
    row: Mapping[str, Any],
    *,
    delete_key: str = DELETE_KEY,
    replace_key: str = REPLACE_KEY,
) -> tuple[float, float]:
    """Resolve analysis-row aliases or recompute indicators from raw log-probs.

    Measurement runners may emit one replacement estimate directly, or two
    donor replicates named ``replacement_d1``/``replacement_d2``.  The latter
    are averaged here before any fold fitting.  A similarly named deletion pair
    is accepted for symmetry, though deletion normally has one estimate.
    """

    sources: list[Mapping[str, Any]] = [row]
    for key in ("analysis_row", "analysis", "indicators"):
        if isinstance(row.get(key), Mapping):
            sources.insert(0, row[key])
    deletion: float | None = None
    replacement: float | None = None
    for source in sources:
        if deletion is None:
            deletion = _first_finite(
                source,
                (
                    delete_key,
                    "deletion",
                    "delete_imageward",
                    "behavior_deletion_imageward",
                ),
            )
        if deletion is None:
            deletion = _replicate_average(
                source,
                (delete_key, "deletion", "delete_imageward"),
            )
        if replacement is None:
            replacement = _first_finite(
                source,
                (
                    replace_key,
                    "replacement",
                    "replace_imageward",
                    "behavior_replacement_imageward",
                ),
            )
        if replacement is None:
            replacement = _replicate_average(
                source,
                (replace_key, "replacement", "replace_imageward"),
            )

    measurements = row.get("measurements")
    if isinstance(measurements, Mapping):
        logp: dict[str, float] = {}
        for name, value in measurements.items():
            if not isinstance(value, Mapping):
                continue
            raw = value.get(
                "fixed_answer_log_probability", value.get("fixed_answer_logp")
            )
            if raw is None and value.get("fixed_answer_probability") is not None:
                probability = _finite_float(
                    value["fixed_answer_probability"],
                    name=f"measurements.{name}.fixed_answer_probability",
                )
                if probability <= 0:
                    raise ValueError("Fixed-answer probability must be positive")
                raw = math.log(probability)
            if raw is not None:
                logp[str(name)] = _finite_float(
                    raw, name=f"measurements.{name}.fixed_answer_log_probability"
                )
        if deletion is None and {"no_text", "no_image"}.issubset(logp):
            deletion = logp["no_text"] - logp["no_image"]
        if replacement is None:
            replicate_values = []
            for suffix in ("", "_d1", "_d2"):
                text_key = f"replace_text{suffix}"
                image_key = f"replace_image{suffix}"
                if {text_key, image_key}.issubset(logp):
                    replicate_values.append(logp[text_key] - logp[image_key])
            if replicate_values:
                replacement = float(np.mean(replicate_values))
    if deletion is None or replacement is None:
        raise KeyError(
            f"Row {row.get('case_id')} cannot derive deletion/replacement indicators"
        )
    return float(deletion), float(replacement)


def normalize_measurement_row(
    row: Mapping[str, Any],
    *,
    delete_key: str = DELETE_KEY,
    replace_key: str = REPLACE_KEY,
) -> dict[str, Any]:
    """Flatten runner metadata and materialize canonical indicator columns."""

    metadata = row.get("metadata")
    normalized = {**(dict(metadata) if isinstance(metadata, Mapping) else {}), **dict(row)}
    deletion, replacement = derive_reliance_indicators(
        normalized, delete_key=delete_key, replace_key=replace_key
    )
    normalized[delete_key] = deletion
    normalized[replace_key] = replacement
    return normalized


def _canonical_nuisance(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable nuisance mapping while preserving explicit caller data."""

    explicit = row.get("nuisance")
    if explicit is not None and not isinstance(explicit, Mapping):
        raise TypeError("row['nuisance'] must be a mapping")
    values = dict(explicit or {})

    if "answer_identity" not in values:
        for key in ("fixed_answer", "final_answer", "answer_identity"):
            if row.get(key) is not None:
                values["answer_identity"] = str(row[key])
                break
    if "final_side" not in values:
        if row.get("final_side") is not None:
            values["final_side"] = str(row["final_side"])
        elif row.get("decision_side") is not None:
            values["final_side"] = str(row["decision_side"])
        elif row.get("final_image") is not None:
            values["final_side"] = "image" if bool(row["final_image"]) else "text"

    aliases = {
        "difficulty": ("difficulty",),
        "condition": ("condition",),
        "prior_strength": ("prior_strength",),
        "answer_margin": (
            "answer_margin",
            "full_top1_top2_logit_margin",
            "top1_top2_logit_margin",
        ),
    }
    for canonical, candidates in aliases.items():
        if canonical in values:
            continue
        for key in candidates:
            if row.get(key) is not None:
                values[canonical] = row[key]
                break

    missing = [key for key in ("answer_identity", "final_side") if key not in values]
    if missing:
        raise ValueError(
            "Nuisance data must identify the fixed answer and final side; missing "
            + ", ".join(missing)
        )
    return values


def _is_numeric(values: Sequence[Any]) -> bool:
    return bool(values) and all(
        isinstance(value, (bool, int, float, np.integer, np.floating))
        and math.isfinite(float(value))
        for value in values
    )


def fit_nuisance_encoder(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit numeric scaling and categorical levels using training rows only."""

    if not rows:
        raise ValueError("Cannot fit a nuisance encoder without training rows")
    nuisances = [_canonical_nuisance(row) for row in rows]
    keys = sorted({key for value in nuisances for key in value})
    numeric: dict[str, dict[str, float]] = {}
    categorical: dict[str, dict[str, Any]] = {}
    columns = ["intercept"]
    for key in keys:
        observed = [value[key] for value in nuisances if value.get(key) is not None]
        if _is_numeric(observed):
            numbers = np.asarray([float(value) for value in observed], dtype=np.float64)
            mean = float(numbers.mean())
            scale = float(numbers.std(ddof=1)) if len(numbers) > 1 else 1.0
            if not math.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            numeric[key] = {"mean": mean, "scale": scale}
            columns.append(f"numeric:{key}")
        else:
            levels = sorted(
                {str(value.get(key, "<MISSING>")) for value in nuisances}
            )
            reference = levels[0]
            encoded = levels[1:]
            categorical[key] = {
                "reference": reference,
                "levels": levels,
                "encoded_levels": encoded,
            }
            columns.extend(f"categorical:{key}={level}" for level in encoded)
    return {
        "format_version": 1,
        "numeric": numeric,
        "categorical": categorical,
        "columns": columns,
        "fit_n": len(rows),
    }


def transform_nuisance(
    rows: Sequence[Mapping[str, Any]], encoder: Mapping[str, Any]
) -> np.ndarray:
    """Apply a training-fitted encoder; unseen categories map to the reference."""

    output: list[list[float]] = []
    for row in rows:
        nuisance = _canonical_nuisance(row)
        values = [1.0]
        for key, specification in encoder["numeric"].items():
            raw = nuisance.get(key, specification["mean"])
            try:
                number = float(raw)
            except (TypeError, ValueError):
                number = float(specification["mean"])
            if not math.isfinite(number):
                number = float(specification["mean"])
            values.append(
                (number - float(specification["mean"]))
                / float(specification["scale"])
            )
        for key, specification in encoder["categorical"].items():
            raw = str(nuisance.get(key, "<MISSING>"))
            values.extend(
                float(raw == level) for level in specification["encoded_levels"]
            )
        output.append(values)
    matrix = np.asarray(output, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(encoder["columns"]):
        raise RuntimeError("Nuisance design has an unexpected shape")
    return matrix


def _fit_lstsq(design: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(
        np.asarray(design, dtype=np.float64),
        np.asarray(outcomes, dtype=np.float64),
        rcond=None,
    )[0]


def prepare_targets(
    rows: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    *,
    delete_key: str = DELETE_KEY,
    replace_key: str = REPLACE_KEY,
) -> PreparedTargets:
    """Training-fold nuisance residualization and shared-target construction."""

    train_rows = [rows[index] for index in train_indices]
    test_rows = [rows[index] for index in test_indices]
    encoder = fit_nuisance_encoder(train_rows)
    x_train = transform_nuisance(train_rows, encoder)
    x_test = transform_nuisance(test_rows, encoder)
    y_train = np.asarray(
        [
            [
                _finite_float(row[delete_key], name=delete_key),
                _finite_float(row[replace_key], name=replace_key),
            ]
            for row in train_rows
        ],
        dtype=np.float64,
    )
    y_test = np.asarray(
        [
            [
                _finite_float(row[delete_key], name=delete_key),
                _finite_float(row[replace_key], name=replace_key),
            ]
            for row in test_rows
        ],
        dtype=np.float64,
    )
    nuisance_beta = _fit_lstsq(x_train, y_train)
    residual_train = y_train - x_train @ nuisance_beta
    residual_test = y_test - x_test @ nuisance_beta
    residual_mean = residual_train.mean(axis=0)
    residual_scale = residual_train.std(axis=0, ddof=1)
    if (
        not np.isfinite(residual_scale).all()
        or np.any(residual_scale <= 1e-12)
    ):
        raise ValueError("Deletion/replacement residual target has zero training scale")
    standardized_train = (residual_train - residual_mean) / residual_scale
    standardized_test = (residual_test - residual_mean) / residual_scale
    train = {
        "deletion": standardized_train[:, 0],
        "replacement": standardized_train[:, 1],
        "shared": standardized_train.mean(axis=1),
    }
    test = {
        "deletion": standardized_test[:, 0],
        "replacement": standardized_test[:, 1],
        "shared": standardized_test.mean(axis=1),
    }
    return PreparedTargets(
        train=train,
        test=test,
        encoder=encoder,
        x_train=x_train,
        x_test=x_test,
        nuisance_beta=nuisance_beta,
        residual_mean=residual_mean,
        residual_scale=residual_scale,
    )


def prepare_hidden(
    hidden: np.ndarray,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    targets: PreparedTargets,
) -> PreparedHidden:
    """Project answer/final-side nuisance out of hidden using training items only."""

    values = np.asarray(hidden, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("One hidden panel cell must be a finite [cases, hidden] matrix")
    h_train = values[np.asarray(train_indices, dtype=np.int64)]
    h_test = values[np.asarray(test_indices, dtype=np.int64)]
    nuisance_beta = _fit_lstsq(targets.x_train, h_train)
    residual_train = h_train - targets.x_train @ nuisance_beta
    residual_test = h_test - targets.x_test @ nuisance_beta
    feature_mean = residual_train.mean(axis=0)
    feature_scale = residual_train.std(axis=0, ddof=1)
    feature_scale[~np.isfinite(feature_scale) | (feature_scale <= 1e-12)] = 1.0
    return PreparedHidden(
        train=(residual_train - feature_mean) / feature_scale,
        test=(residual_test - feature_mean) / feature_scale,
        nuisance_beta=nuisance_beta,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )


def fit_ridge(values: np.ndarray, target: np.ndarray, alpha: float) -> RidgeFit:
    """Fit an intercept Ridge efficiently in the sample-dual when appropriate."""

    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(y) < 2:
        raise ValueError("Ridge requires matching [n, hidden] values and n>=2 target")
    if alpha <= 0 or not math.isfinite(float(alpha)):
        raise ValueError("Ridge alpha must be finite and positive")
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered_x = x - x_mean
    centered_y = y - y_mean
    if x.shape[1] > x.shape[0]:
        gram = centered_x @ centered_x.T
        dual = np.linalg.solve(
            gram + float(alpha) * np.eye(len(x), dtype=np.float64), centered_y
        )
        coefficient = centered_x.T @ dual
    else:
        gram = centered_x.T @ centered_x
        coefficient = np.linalg.solve(
            gram + float(alpha) * np.eye(x.shape[1], dtype=np.float64),
            centered_x.T @ centered_y,
        )
    intercept = y_mean - float(x_mean @ coefficient)
    return RidgeFit(coefficient=coefficient, intercept=intercept)


def _hidden_path(row: Mapping[str, Any], hidden_root: str | Path | None) -> Path:
    raw = row.get("hidden_file", row.get("hidden_npz"))
    if raw is None:
        raise KeyError(f"Row {row.get('case_id')} has no hidden_file/hidden_npz")
    path = Path(str(raw))
    if not path.is_absolute() and hidden_root is not None:
        path = Path(hidden_root) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_hidden_panel(
    row: Mapping[str, Any],
    *,
    hidden_root: str | Path | None = None,
    layers: Sequence[int] = RELIANCE_LAYERS,
    positions: Sequence[str] = RELIANCE_POSITIONS,
) -> dict[PanelCell, np.ndarray]:
    """Load and normalize one NPZ hidden panel into named vectors."""

    path = _hidden_path(row, hidden_root)
    with np.load(path, allow_pickle=False) as payload:
        hidden_key = "hidden" if "hidden" in payload else "hidden_states"
        layer_key = "layers" if "layers" in payload else "layer_indices"
        position_key = "positions" if "positions" in payload else "position_names"
        missing = [
            key
            for key, present in (
                ("hidden/hidden_states", hidden_key in payload),
                ("layers/layer_indices", layer_key in payload),
                ("positions/position_names", position_key in payload),
            )
            if not present
        ]
        if missing:
            raise ValueError(f"Hidden panel {path} is missing {missing}")
        values = np.asarray(payload[hidden_key], dtype=np.float64)
        saved_layers = [int(value) for value in payload[layer_key].tolist()]
        saved_positions = [str(value) for value in payload[position_key].tolist()]
    if values.ndim != 3:
        raise ValueError(f"Hidden panel must be rank 3, got {values.shape} at {path}")
    if values.shape[:2] == (len(saved_positions), len(saved_layers)):
        normalized = values
    elif values.shape[:2] == (len(saved_layers), len(saved_positions)):
        normalized = values.transpose(1, 0, 2)
    else:
        raise ValueError(
            f"Hidden shape {values.shape} does not match positions={len(saved_positions)} "
            f"and layers={len(saved_layers)} at {path}"
        )
    if not np.isfinite(normalized).all():
        raise ValueError(f"Hidden panel contains non-finite values: {path}")
    missing_layers = sorted(set(map(int, layers)).difference(saved_layers))
    missing_positions = sorted(set(map(str, positions)).difference(saved_positions))
    if missing_layers or missing_positions:
        raise ValueError(
            f"Hidden panel {path} misses layers={missing_layers}, positions={missing_positions}"
        )
    return {
        PanelCell(int(layer), str(position)): normalized[
            saved_positions.index(str(position)), saved_layers.index(int(layer))
        ].copy()
        for position in positions
        for layer in layers
    }


def load_hidden_matrices(
    rows: Sequence[Mapping[str, Any]],
    *,
    hidden_root: str | Path | None = None,
    layers: Sequence[int] = RELIANCE_LAYERS,
    positions: Sequence[str] = RELIANCE_POSITIONS,
) -> dict[PanelCell, np.ndarray]:
    panels = [
        load_hidden_panel(
            row, hidden_root=hidden_root, layers=layers, positions=positions
        )
        for row in rows
    ]
    cells = tuple(panels[0]) if panels else ()
    return {
        cell: np.stack([panel[cell] for panel in panels], axis=0) for cell in cells
    }


def _validate_rows(
    rows: Sequence[Mapping[str, Any]], delete_key: str, replace_key: str
) -> tuple[np.ndarray, list[int]]:
    if not rows:
        raise ValueError("No reliance measurement rows were supplied")
    case_ids: set[str] = set()
    item_to_fold: dict[str, int] = {}
    folds: list[int] = []
    for row in rows:
        case_id = str(row.get("case_id", ""))
        item_id = str(row.get("item_id", ""))
        if not case_id or case_id in case_ids:
            raise ValueError(f"case_id must be non-empty and unique: {case_id!r}")
        if not item_id or row.get("fold") is None:
            raise ValueError(f"Row {case_id} lacks item_id/fold")
        case_ids.add(case_id)
        fold = int(row["fold"])
        previous = item_to_fold.setdefault(item_id, fold)
        if previous != fold:
            raise RuntimeError(
                f"Outer item-fold leakage: item {item_id} appears in folds {previous} and {fold}"
            )
        _finite_float(row[delete_key], name=delete_key)
        _finite_float(row[replace_key], name=replace_key)
        _canonical_nuisance(row)
        folds.append(fold)
    unique_folds = sorted(set(folds))
    if len(unique_folds) < 3:
        raise ValueError("Strict nested OOF requires at least three existing item folds")
    return np.asarray(folds, dtype=np.int64), unique_folds


def _inner_splits(
    folds: np.ndarray, outer_train: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in sorted(set(folds[outer_train].tolist())):
        validation = outer_train[folds[outer_train] == fold]
        training = outer_train[folds[outer_train] != fold]
        if len(training) < 2 or len(validation) < 1:
            continue
        splits.append((training, validation))
    if len(splits) < 2:
        raise ValueError("Outer training data do not contain enough existing folds for inner CV")
    return splits


def _select_models_inner(
    rows: Sequence[Mapping[str, Any]],
    matrices: Mapping[PanelCell, np.ndarray],
    folds: np.ndarray,
    outer_train: np.ndarray,
    *,
    alphas: Sequence[float],
    delete_key: str,
    replace_key: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    cells = tuple(sorted(matrices, key=lambda cell: (cell.position, cell.layer)))
    losses = {
        objective: {
            (cell, float(alpha)): [0.0, 0]
            for cell in cells
            for alpha in alphas
        }
        for objective in OBJECTIVES
    }
    split_audit: list[dict[str, Any]] = []
    for inner_train, inner_validation in _inner_splits(folds, outer_train):
        targets = prepare_targets(
            rows,
            inner_train,
            inner_validation,
            delete_key=delete_key,
            replace_key=replace_key,
        )
        for cell in cells:
            prepared = prepare_hidden(
                matrices[cell], inner_train, inner_validation, targets
            )
            for objective in OBJECTIVES:
                for alpha in alphas:
                    fitted = fit_ridge(
                        prepared.train, targets.train[objective], float(alpha)
                    )
                    prediction = fitted.predict(prepared.test)
                    error = prediction - targets.test[objective]
                    record = losses[objective][(cell, float(alpha))]
                    record[0] += float(error @ error)
                    record[1] += len(error)
        split_audit.append(
            {
                "train_n": len(inner_train),
                "validation_n": len(inner_validation),
                "train_folds": sorted(set(folds[inner_train].tolist())),
                "validation_folds": sorted(
                    set(folds[inner_validation].tolist())
                ),
                "item_overlap": sorted(
                    {str(rows[i]["item_id"]) for i in inner_train}.intersection(
                        {str(rows[i]["item_id"]) for i in inner_validation}
                    )
                ),
            }
        )
    if any(value["item_overlap"] for value in split_audit):
        raise RuntimeError("Inner existing-fold CV leaked an item")
    selected: dict[str, dict[str, Any]] = {}
    loss_audit: dict[str, Any] = {}
    for objective in OBJECTIVES:
        ranked = sorted(
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
        if not ranked:
            raise RuntimeError(f"Inner CV produced no losses for {objective}")
        mse, _position, _layer, alpha, cell, count = ranked[0]
        selected[objective] = {
            "cell": cell,
            "alpha": float(alpha),
            "inner_mse": float(mse),
            "inner_validation_n": int(count),
        }
        loss_audit[objective] = [
            {
                **candidate.to_dict(),
                "alpha": float(candidate_alpha),
                "mse": float(total / count),
                "n": int(count),
            }
            for (candidate, candidate_alpha), (total, count) in sorted(
                losses[objective].items(),
                key=lambda value: (
                    value[0][0].position,
                    value[0][0].layer,
                    value[0][1],
                ),
            )
            if count
        ]
    return selected, {"splits": split_audit, "losses": loss_audit}


def _fit_outer_model(
    rows: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    objective: str,
    alpha: float,
    delete_key: str,
    replace_key: str,
) -> tuple[np.ndarray, PreparedTargets, PreparedHidden, RidgeFit]:
    targets = prepare_targets(
        rows,
        train_indices,
        test_indices,
        delete_key=delete_key,
        replace_key=replace_key,
    )
    hidden = prepare_hidden(matrix, train_indices, test_indices, targets)
    fitted = fit_ridge(hidden.train, targets.train[objective], alpha)
    return fitted.predict(hidden.test), targets, hidden, fitted


def _save_direction(
    directory: Path,
    *,
    fold: int,
    objective: str,
    cell: PanelCell,
    alpha: float,
    targets: PreparedTargets,
    hidden: PreparedHidden,
    fitted: RidgeFit,
) -> dict[str, Any]:
    raw_direction = fitted.coefficient / hidden.feature_scale
    norm = float(np.linalg.norm(raw_direction))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError(
            f"Degenerate reliance direction for fold={fold}, objective={objective}"
        )
    unit = raw_direction / norm
    filename = (
        f"fold_{fold}_{objective}_layer_{cell.layer}_{cell.position}.npz"
    )
    atomic_save_npz(
        directory / filename,
        alpha=np.asarray(float(alpha)),
        ridge_coefficient=fitted.coefficient,
        ridge_intercept=np.asarray(float(fitted.intercept)),
        raw_direction=raw_direction,
        unit_direction=unit,
        hidden_nuisance_beta=hidden.nuisance_beta,
        feature_mean=hidden.feature_mean,
        feature_scale=hidden.feature_scale,
        target_nuisance_beta=targets.nuisance_beta,
        target_residual_mean=targets.residual_mean,
        target_residual_scale=targets.residual_scale,
    )
    return {
        "fold": int(fold),
        "objective": objective,
        **cell.to_dict(),
        "alpha": float(alpha),
        "file": filename,
        "nuisance_encoder": targets.encoder,
        "hidden_size": int(len(raw_direction)),
        "direction_norm": norm,
    }


def _association(
    rows: Sequence[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    iterations: int,
) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get(x_key) is not None
        and row.get(y_key) is not None
        and math.isfinite(float(row[x_key]))
        and math.isfinite(float(row[y_key]))
    ]
    if len(valid) < 3:
        return {
            "n": len(valid),
            "pearson": None,
            "spearman": None,
            "spearman_item_bootstrap": None,
        }
    x = np.asarray([float(row[x_key]) for row in valid], dtype=np.float64)
    y = np.asarray([float(row[y_key]) for row in valid], dtype=np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return {
            "n": len(valid),
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
        valid,
        lambda sample: spearmanr(
            [float(row[x_key]) for row in sample],
            [float(row[y_key]) for row in sample],
        ).statistic,
        iterations=iterations,
    )
    return {
        "n": len(valid),
        "unique_items": len({str(row["item_id"]) for row in valid}),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
        "spearman_item_bootstrap": bootstrap,
    }


def _prediction_metrics(
    rows: Sequence[dict[str, Any]],
    target_key: str,
    prediction_key: str,
    *,
    iterations: int,
) -> dict[str, Any]:
    target = np.asarray([float(row[target_key]) for row in rows], dtype=np.float64)
    prediction = np.asarray(
        [float(row[prediction_key]) for row in rows], dtype=np.float64
    )
    return {
        "n": len(rows),
        "r2": float(r2_score(target, prediction)),
        "mae": float(mean_absolute_error(target, prediction)),
        "association": _association(
            rows, prediction_key, target_key, iterations=iterations
        ),
    }


def fit_reliance_representation(
    rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    hidden_root: str | Path | None = None,
    layers: Sequence[int] = RELIANCE_LAYERS,
    positions: Sequence[str] = RELIANCE_POSITIONS,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    delete_key: str = DELETE_KEY,
    replace_key: str = REPLACE_KEY,
    min_reliable_n: int = 80,
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    """Fit nested-OOF Actual Source Reliance readouts and write directions.

    Layer/position and alpha selection are repeated inside every held-out item
    fold using only the remaining existing folds.  The returned panel-selected
    OOF score therefore includes model-selection uncertainty rather than
    reporting the best test-set cell.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    clean_rows = [
        normalize_measurement_row(
            row, delete_key=delete_key, replace_key=replace_key
        )
        for row in rows
    ]
    folds, unique_folds = _validate_rows(clean_rows, delete_key, replace_key)
    layer_values = tuple(int(value) for value in layers)
    position_values = tuple(str(value) for value in positions)
    alpha_values = tuple(float(value) for value in alphas)
    if (
        not layer_values
        or len(set(layer_values)) != len(layer_values)
        or not position_values
        or len(set(position_values)) != len(position_values)
        or not alpha_values
        or any(value <= 0 or not math.isfinite(value) for value in alpha_values)
    ):
        raise ValueError("Layers, positions, and positive finite alphas must be distinct/non-empty")
    matrices = load_hidden_matrices(
        clean_rows,
        hidden_root=hidden_root,
        layers=layer_values,
        positions=position_values,
    )
    n = len(clean_rows)
    selected_predictions = {
        objective: np.full(n, np.nan, dtype=np.float64) for objective in OBJECTIVES
    }
    targets_oof = {
        objective: np.full(n, np.nan, dtype=np.float64) for objective in OBJECTIVES
    }
    baseline_oof = np.full(n, np.nan, dtype=np.float64)
    cell_predictions = {
        cell: np.full(n, np.nan, dtype=np.float64) for cell in matrices
    }
    fold_selection: list[dict[str, Any]] = []
    direction_entries: list[dict[str, Any]] = []
    direction_dir = output / "directions"
    for fold in unique_folds:
        test_indices = np.flatnonzero(folds == fold)
        train_indices = np.flatnonzero(folds != fold)
        train_items = {str(clean_rows[index]["item_id"]) for index in train_indices}
        test_items = {str(clean_rows[index]["item_id"]) for index in test_indices}
        overlap = sorted(train_items.intersection(test_items))
        if overlap:
            raise RuntimeError(f"Outer item leakage in fold {fold}: {overlap[:5]}")
        selected, inner_audit = _select_models_inner(
            clean_rows,
            matrices,
            folds,
            train_indices,
            alphas=alpha_values,
            delete_key=delete_key,
            replace_key=replace_key,
        )

        # The shared outer target is the common reference for all-cell timing.
        common_targets = prepare_targets(
            clean_rows,
            train_indices,
            test_indices,
            delete_key=delete_key,
            replace_key=replace_key,
        )
        for objective in OBJECTIVES:
            targets_oof[objective][test_indices] = common_targets.test[objective]
        baseline_oof[test_indices] = float(common_targets.train["shared"].mean())

        # Fit every cell with its training-only selected shared alpha for an
        # unbiased descriptive layer/position trajectory.
        shared_losses = inner_audit["losses"]["shared"]
        for cell in matrices:
            candidates = [
                value
                for value in shared_losses
                if value["key"] == cell.key
            ]
            best = min(candidates, key=lambda value: (value["mse"], value["alpha"]))
            prediction, _targets, _hidden, _fitted = _fit_outer_model(
                clean_rows,
                matrices[cell],
                train_indices,
                test_indices,
                objective="shared",
                alpha=float(best["alpha"]),
                delete_key=delete_key,
                replace_key=replace_key,
            )
            cell_predictions[cell][test_indices] = prediction

        fold_record = {
            "fold": int(fold),
            "train_n": len(train_indices),
            "test_n": len(test_indices),
            "train_item_count": len(train_items),
            "test_item_count": len(test_items),
            "item_overlap": overlap,
            "selected": {},
            "inner_cv": inner_audit,
        }
        for objective in OBJECTIVES:
            specification = selected[objective]
            cell = specification["cell"]
            prediction, prepared_targets, prepared_hidden, fitted = _fit_outer_model(
                clean_rows,
                matrices[cell],
                train_indices,
                test_indices,
                objective=objective,
                alpha=float(specification["alpha"]),
                delete_key=delete_key,
                replace_key=replace_key,
            )
            selected_predictions[objective][test_indices] = prediction
            entry = _save_direction(
                direction_dir,
                fold=fold,
                objective=objective,
                cell=cell,
                alpha=float(specification["alpha"]),
                targets=prepared_targets,
                hidden=prepared_hidden,
                fitted=fitted,
            )
            direction_entries.append(entry)
            fold_record["selected"][objective] = {
                **cell.to_dict(),
                "alpha": float(specification["alpha"]),
                "inner_mse": float(specification["inner_mse"]),
                "direction_file": entry["file"],
            }
        fold_selection.append(fold_record)

    arrays = [*selected_predictions.values(), *targets_oof.values(), baseline_oof]
    arrays.extend(cell_predictions.values())
    if not all(np.isfinite(values).all() for values in arrays):
        raise RuntimeError("Nested OOF fitting did not cover every row")

    oof_rows: list[dict[str, Any]] = []
    for index, row in enumerate(clean_rows):
        record = {
            "case_id": str(row["case_id"]),
            "item_id": str(row["item_id"]),
            "fold": int(row["fold"]),
            "target_deletion": float(targets_oof["deletion"][index]),
            "target_replacement": float(targets_oof["replacement"][index]),
            "target_shared": float(targets_oof["shared"][index]),
            "prediction_shared": float(selected_predictions["shared"][index]),
            "prediction_deletion": float(selected_predictions["deletion"][index]),
            "prediction_replacement": float(
                selected_predictions["replacement"][index]
            ),
            "prediction_nuisance_only": float(baseline_oof[index]),
            "answer_identity": str(_canonical_nuisance(row)["answer_identity"]),
            "final_side": str(_canonical_nuisance(row)["final_side"]),
            "cell_predictions": {
                cell.key: float(values[index])
                for cell, values in cell_predictions.items()
            },
        }
        oof_rows.append(record)
    write_jsonl_atomic(output / "oof_predictions.jsonl", oof_rows)
    atomic_write_json(output / "fold_audit.json", {"folds": fold_selection})
    atomic_write_json(
        direction_dir / "index.json",
        {
            "format_version": FORMAT_VERSION,
            "definition": "answer-only Actual Source Reliance; strict outer item-fold and inner existing-fold Ridge selection",
            "layers": list(layer_values),
            "positions": list(position_values),
            "alphas": list(alpha_values),
            "entries": direction_entries,
        },
    )

    reliability = _association(
        oof_rows,
        "target_deletion",
        "target_replacement",
        iterations=bootstrap_iterations,
    )
    sign_agreement = float(
        np.mean(targets_oof["deletion"] * targets_oof["replacement"] > 0)
    )
    reliability_ci = reliability["spearman_item_bootstrap"]["ci95"]
    target_gate = bool(
        n >= int(min_reliable_n)
        and reliability_ci[0] is not None
        and reliability_ci[0] > 0
        and sign_agreement >= 0.60
    )
    shared_metrics = _prediction_metrics(
        oof_rows, "target_shared", "prediction_shared", iterations=bootstrap_iterations
    )
    nuisance_metrics = _prediction_metrics(
        oof_rows,
        "target_shared",
        "prediction_nuisance_only",
        iterations=bootstrap_iterations,
    )
    cross = {
        "deletion_model_to_replacement": _association(
            oof_rows,
            "prediction_deletion",
            "target_replacement",
            iterations=bootstrap_iterations,
        ),
        "replacement_model_to_deletion": _association(
            oof_rows,
            "prediction_replacement",
            "target_deletion",
            iterations=bootstrap_iterations,
        ),
    }
    shared_ci = shared_metrics["association"]["spearman_item_bootstrap"]["ci95"]
    cross_lowers = [
        value["spearman_item_bootstrap"]["ci95"][0] for value in cross.values()
    ]
    incremental_r2 = shared_metrics["r2"] - nuisance_metrics["r2"]
    representation_gate = bool(
        target_gate
        and shared_metrics["r2"] > 0
        and shared_ci[0] is not None
        and shared_ci[0] > 0
        and incremental_r2 > 0
        and all(value is not None and value > 0 for value in cross_lowers)
    )
    panel_metrics = {
        cell.key: _prediction_metrics(
            [
                {**row, "cell_prediction": row["cell_predictions"][cell.key]}
                for row in oof_rows
            ],
            "target_shared",
            "cell_prediction",
            iterations=bootstrap_iterations,
        )
        for cell in matrices
    }
    summary = {
        "title": "Answer-only Actual Source Reliance Representation",
        "status": "completed",
        "n": n,
        "unique_items": len({str(row["item_id"]) for row in clean_rows}),
        "folds": unique_folds,
        "target_definition": "training-fold nuisance residualized and standardized mean of deletion and replacement imageward indicators",
        "nuisance_required": ["answer_identity", "final_side"],
        "target_reliability": {
            "deletion_vs_replacement": reliability,
            "sign_agreement": sign_agreement,
            "gate_passed": target_gate,
            "rule": f"n>={min_reliable_n}, item-bootstrap Spearman CI lower>0, sign agreement>=.60",
        },
        "nested_panel_oof": shared_metrics,
        "nuisance_only_oof": nuisance_metrics,
        "incremental_oof_r2": float(incremental_r2),
        "cross_method": cross,
        "panel_cells_oof": panel_metrics,
        "fold_selected_cells": [
            {"fold": value["fold"], "selected": value["selected"]}
            for value in fold_selection
        ],
        "representation_gate_passed": representation_gate,
        "representation_gate_rule": "target gate; nested-panel OOF R2>0 and Spearman CI lower>0; incremental R2>0; both cross-method Spearman CI lowers>0",
        "classification": (
            "a shared answer-only Actual Source Reliance representation is OOF-validated"
            if representation_gate
            else "the shared answer-only Actual Source Reliance representation is not yet jointly validated"
        ),
    }
    write_experiment_summary(output, summary)
    return summary


def _direction_entry(
    index: Mapping[str, Any], fold: int, objective: str = "shared"
) -> Mapping[str, Any]:
    matches = [
        entry
        for entry in index["entries"]
        if int(entry["fold"]) == int(fold) and entry["objective"] == objective
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one {objective} reliance direction for fold {fold}, got {len(matches)}"
        )
    return matches[0]


def _branch_record(row: Mapping[str, Any], history: str) -> Mapping[str, Any]:
    for container_key in ("history_hidden", "histories"):
        container = row.get(container_key)
        if isinstance(container, Mapping) and history in container:
            value = container[history]
            if isinstance(value, Mapping):
                return value
            return {"hidden_file": value}
    flat = row.get(f"{history}_hidden_file")
    if flat is not None:
        return {"hidden_file": flat}
    raise KeyError(f"History row {row.get('case_id')} has no hidden for {history}")


def _history_behavior_values(row: Mapping[str, Any]) -> tuple[float, float, float | None]:
    source: Mapping[str, Any] = row
    protocols = row.get("protocols")
    if isinstance(protocols, Mapping) and isinstance(protocols.get("answer_only"), Mapping):
        source = protocols["answer_only"]
    deletion = source.get("delta_history_delete", row.get("delta_history_delete"))
    replacement = source.get(
        "delta_history_replace", row.get("delta_history_replace")
    )
    if deletion is None or replacement is None:
        raise KeyError(
            f"History row {row.get('case_id')} lacks deletion/replacement History effects"
        )
    delta_sa = row.get("delta_sa", row.get("old_delta_sa_if_minus_tf"))
    return (
        _finite_float(deletion, name="delta_history_delete"),
        _finite_float(replacement, name="delta_history_replace"),
        None if delta_sa is None else _finite_float(delta_sa, name="delta_sa"),
    )


def _endpoint_matched(row: Mapping[str, Any]) -> bool:
    if row.get("primary_endpoint_matched") is not None:
        return bool(row["primary_endpoint_matched"])
    endpoints = row.get("answer_only_natural_endpoints")
    fixed = row.get("fixed_answer")
    if isinstance(endpoints, Mapping) and fixed is not None:
        return all(str(value) == str(fixed) for value in endpoints.values())
    return True


def _predict_direction(
    hidden: np.ndarray,
    nuisance_row: Mapping[str, Any],
    entry: Mapping[str, Any],
    payload: Mapping[str, np.ndarray],
) -> tuple[float, float]:
    x = transform_nuisance([nuisance_row], entry["nuisance_encoder"])
    vector = np.asarray(hidden, dtype=np.float64).reshape(-1)
    beta = np.asarray(payload["hidden_nuisance_beta"], dtype=np.float64)
    if beta.shape[1] != len(vector):
        raise ValueError("History hidden size differs from the fitted direction")
    residual = vector - (x @ beta)[0]
    standardized = (
        residual - np.asarray(payload["feature_mean"], dtype=np.float64)
    ) / np.asarray(payload["feature_scale"], dtype=np.float64)
    prediction = float(
        standardized @ np.asarray(payload["ridge_coefficient"], dtype=np.float64)
        + float(payload["ridge_intercept"])
    )
    coordinate = float(
        residual @ np.asarray(payload["unit_direction"], dtype=np.float64)
    )
    return prediction, coordinate


def analyze_history_zero_shot(
    history_rows: Sequence[Mapping[str, Any]],
    direction_dir: str | Path,
    *,
    hidden_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    min_primary_n: int = 15,
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    """Apply no-History fold directions to TF/IF History pairs without refit."""

    root = Path(direction_dir)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for raw in history_rows:
        row = dict(raw)
        fold = int(row["fold"])
        entry = _direction_entry(index, fold, "shared")
        path = root / str(entry["file"])
        with np.load(path, allow_pickle=False) as loaded:
            payload = {key: np.asarray(loaded[key]) for key in loaded.files}
        branch_predictions: dict[str, dict[str, float]] = {}
        branch_nuisance: dict[str, dict[str, Any]] = {}
        for history in ("text_first", "image_first"):
            branch = dict(_branch_record(row, history))
            merged = dict(row)
            if isinstance(branch.get("nuisance"), Mapping):
                merged["nuisance"] = {
                    **dict(row.get("nuisance", {})),
                    **dict(branch["nuisance"]),
                }
            panel = load_hidden_panel(
                {**merged, **branch},
                hidden_root=hidden_root,
                layers=[int(entry["layer"])],
                positions=[str(entry["position"])],
            )
            hidden = panel[PanelCell(int(entry["layer"]), str(entry["position"]))]
            prediction, coordinate = _predict_direction(hidden, merged, entry, payload)
            branch_predictions[history] = {
                "prediction": prediction,
                "coordinate": coordinate,
            }
            branch_nuisance[history] = _canonical_nuisance(merged)
        for key in ("answer_identity", "final_side"):
            if branch_nuisance["text_first"][key] != branch_nuisance["image_first"][key]:
                raise ValueError(
                    f"History pair changes {key}; fixed-answer comparison is invalid for {row.get('case_id')}"
                )
        delta_delete, delta_replace, delta_sa = _history_behavior_values(row)
        target_scale = np.asarray(payload["target_residual_scale"], dtype=np.float64)
        delta_behavior = float(
            0.5
            * (delta_delete / target_scale[0] + delta_replace / target_scale[1])
        )
        results.append(
            {
                "case_id": str(row["case_id"]),
                "item_id": str(row["item_id"]),
                "fold": fold,
                "selected_layer": int(entry["layer"]),
                "selected_position": str(entry["position"]),
                "text_first_prediction": branch_predictions["text_first"][
                    "prediction"
                ],
                "image_first_prediction": branch_predictions["image_first"][
                    "prediction"
                ],
                "delta_z_reliance": branch_predictions["image_first"][
                    "prediction"
                ]
                - branch_predictions["text_first"]["prediction"],
                "delta_coordinate": branch_predictions["image_first"][
                    "coordinate"
                ]
                - branch_predictions["text_first"]["coordinate"],
                "delta_behavior_delete": delta_delete,
                "delta_behavior_replace": delta_replace,
                "delta_behavior_shared": delta_behavior,
                "delta_sa": delta_sa,
                "primary_endpoint_matched": _endpoint_matched(row),
            }
        )
    primary = [row for row in results if row["primary_endpoint_matched"]]

    def summarize(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"n": 0}
        return {
            "n": len(selected),
            "delta_z_reliance": paired_effect_summary(
                selected, "delta_z_reliance", iterations=bootstrap_iterations
            ),
            "delta_behavior_shared": paired_effect_summary(
                selected, "delta_behavior_shared", iterations=bootstrap_iterations
            ),
            "delta_z_vs_delta_behavior": _association(
                selected,
                "delta_z_reliance",
                "delta_behavior_shared",
                iterations=bootstrap_iterations,
            ),
            "delta_z_vs_delta_sa": _association(
                selected,
                "delta_z_reliance",
                "delta_sa",
                iterations=bootstrap_iterations,
            ),
        }

    primary_summary = summarize(primary)
    all_summary = summarize(results)
    if primary:
        z_ci = primary_summary["delta_z_reliance"]["ci95"]
        behavior_ci = primary_summary["delta_behavior_shared"]["ci95"]
        alignment_ci = primary_summary["delta_z_vs_delta_behavior"][
            "spearman_item_bootstrap"
        ]["ci95"]
    else:
        z_ci = behavior_ci = alignment_ci = [None, None]
    population_gate = bool(
        len(primary) >= min_primary_n
        and z_ci[0] is not None
        and z_ci[0] > 0
        and behavior_ci[0] is not None
        and behavior_ci[0] > 0
    )
    instance_gate = bool(
        population_gate and alignment_ci[0] is not None and alignment_ci[0] > 0
    )
    summary = {
        "title": "Zero-shot History Treatment Effect on Actual Reliance Representation",
        "status": "completed",
        "n": len(results),
        "primary_endpoint_matched_n": len(primary),
        "primary": primary_summary,
        "all_rows_sensitivity": all_summary,
        "population_shift_gate_passed": population_gate,
        "instance_faithfulness_gate_passed": instance_gate,
        "gate_rule": f"primary n>={min_primary_n}; mean delta-z and behavioral delta CIs lower>0; instance gate additionally requires their Spearman CI lower>0",
        "classification": (
            "History shifts the validated reliance coordinate with instance-wise behavioral alignment"
            if instance_gate
            else (
                "History shifts reliance at the population level without validated instance-wise alignment"
                if population_gate
                else "A zero-shot internal History shift is not established"
            )
        ),
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(destination / "results.jsonl", results)
        write_experiment_summary(destination, summary)
    return summary


def _position_for_char(alignment: Any, char_index: int, name: str) -> int:
    positions = alignment.processed_tokens_for_char_span(char_index, char_index + 1)
    if len(set(positions)) != 1:
        raise ValueError(f"{name} character maps to multiple processed tokens: {positions}")
    return int(positions[0])


def capture_answer_only_pre_post(
    runtime: Any,
    messages: Sequence[dict[str, Any]],
    fixed_answer: str,
    *,
    layers: Sequence[int] = RELIANCE_LAYERS,
) -> dict[str, Any]:
    """GPU helper: capture pre/post-answer states in one SA-free forward.

    ``messages`` may end in the usual ``**Answer**:`` continuation.  This
    function replaces only that final continuation with the exact fixed answer
    and a newline, locates the answer colon and post-answer newline by rendered
    character alignment, and performs one full prefill.  Causal masking makes
    the colon state equivalent to the short pre-answer continuation; runners
    should nevertheless retain a BF16 reconstruction smoke check.
    """

    from layer_metacognition.model_adapter import run_hooked_forward
    from layer_metacognition.token_spans import build_rendered_alignment

    selected_layers = tuple(int(value) for value in layers)
    if not selected_layers or len(set(selected_layers)) != len(selected_layers):
        raise ValueError("Capture layers must be non-empty and distinct")
    if any(
        "Source Attribution" in str(part.get("text", ""))
        for message in messages
        for part in message.get("content", [])
        if isinstance(part, Mapping)
    ):
        raise ValueError("Answer-only reliance capture contains an SA request")
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Answer-only messages must end in an assistant continuation")
    extended = copy.deepcopy(list(messages))
    assistant_text = f"**Answer**: {fixed_answer}\n"
    extended[-1] = {
        "role": "assistant",
        "content": [{"type": "text", "text": assistant_text}],
    }
    rendered, inputs = runtime.generator.prepare_messages(
        extended, assistant_text=assistant_text
    )
    alignment = build_rendered_alignment(
        runtime.generator.tokenizer,
        rendered,
        inputs.input_ids,
        inputs.attention_mask,
    )
    assistant_start = len(rendered) - len(assistant_text)
    pre_position = _position_for_char(
        alignment,
        assistant_start + len("**Answer**:") - 1,
        "pre_answer colon",
    )
    post_position = _position_for_char(
        alignment, len(rendered) - 1, "post_answer newline"
    )
    forward = run_hooked_forward(
        runtime.model,
        inputs,
        runtime.modules,
        {"pre_answer": pre_position, "post_answer": post_position},
        logits_positions=[pre_position],
    )
    hidden = np.stack(
        [
            np.stack(
                [
                    forward.hidden_by_name[position][layer]
                    .detach()
                    .to(device="cpu", dtype=__import__("torch").float16)
                    .numpy()
                    for layer in selected_layers
                ],
                axis=0,
            )
            for position in RELIANCE_POSITIONS
        ],
        axis=0,
    )
    answer_logits = forward.logits_by_position[pre_position].numpy()
    result = {
        "hidden": hidden,
        "layers": np.asarray(selected_layers, dtype=np.int64),
        "positions": np.asarray(RELIANCE_POSITIONS),
        "token_positions": {
            "pre_answer": pre_position,
            "post_answer": post_position,
        },
        "pre_answer_vocab_logits": answer_logits,
        "messages_hash": canonical_message_hash(extended),
        "input_token_count": int(inputs.input_ids.shape[1]),
        "fixed_answer": str(fixed_answer),
        "sa_request_present": False,
    }
    del inputs, forward
    return result


def save_captured_panel(path: str | Path, capture: Mapping[str, Any]) -> None:
    """Atomically save the NPZ portion of ``capture_answer_only_pre_post``."""

    atomic_save_npz(
        path,
        hidden=np.asarray(capture["hidden"], dtype=np.float16),
        layers=np.asarray(capture["layers"], dtype=np.int64),
        positions=np.asarray(capture["positions"]),
    )
