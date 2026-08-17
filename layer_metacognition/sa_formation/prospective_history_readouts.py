"""Frozen readouts and GPU measurements for the prospective History panel.

Stage 09 measures four deliberately distinct quantities:

``B``
    The answer-only behavioral reliance panel (computed elsewhere).
``U``
    The development-fit, fold-frozen Stage-03 raw-choice-coupled shared
    source-use readout.  Its selected layer is fold dependent (L16 or L20).
``U_L18``
    The fixed-L18 Stage-08 source-use candidate.  It is a secondary readout,
    not a replacement for the Stage-03 primary readout.
``A`` / ``V``
    The frozen Stage-10/Stage-06 attribution direction and the verbal
    common-nine Source Attribution report.

Nothing in this module fits a hidden-state direction.  Filesystem loaders
validate the frozen artifacts before exposing small projection objects.  The
GPU helpers accept arbitrary answer-only multi-turn messages, append the
fixed one-token answer to the *same prepared causal prefix*, and capture all
required post-answer layers in one read-only forward.

The low-level projection and prefix-audit functions are intentionally pure so
they can be exercised with synthetic arrays without loading a model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from confidence_test.answer_metrics import normalize_answer
from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from layer_metacognition.hidden_state_store import load_jsonl
from layer_metacognition.model_adapter import run_logits_forward

from .confirmatory_attribution_panel import (
    FrozenDirectionRepository,
    FrozenFoldDirection,
    panel_protocols,
)
from .core import canonical_message_hash, sha256_file, stable_hash
from .fixed_l18_representation_divergence import SourceUseFoldModel
from .reliance_external_representation import (
    ExplicitNuisanceEncoder,
    _fit_baseline,
    fit_target_transform,
    transform_explicit_nuisance,
)
from .reliance_measurement import (
    canonical_answer_token_ids,
    contains_verbal_sa_request,
    restricted_distribution,
)
from .reliance_representation import normalize_measurement_row
from .runtime import (
    Stage3Runtime,
    append_exact_token_ids,
    prepare_measurement,
)
from .second_order import ProtocolAnalyzer


BRIDGE_DIR = "stage3_sa_computational_bridge"
STAGE01_DIR = "01_actual_source_reliance"
STAGE03_DIR = "03_reliance_representation_devfit_confirm"
STAGE06_DIR = "06_confirmatory_attribution_panel"
STAGE08_DIR = "08_fixed_l18_representation_divergence"
RAW_ESTIMAND = "raw_choice_coupled"
PRIMARY_OBJECTIVE = "shared"
POST_ANSWER_POSITION = "post_answer"
SECONDARY_LAYER = 18
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
EXPECTED_DEVELOPMENT_N = 97

_SOURCE_USE_ARRAYS = frozenset(
    {
        "ridge_coefficient",
        "ridge_intercept",
        "raw_direction",
        "unit_direction",
        "hidden_nuisance_beta",
        "feature_mean",
        "feature_scale",
        "target_nuisance_beta",
        "target_mean",
        "target_scale",
    }
)
_STAGE08_ARRAYS = frozenset(
    {
        "alpha",
        "ridge_coefficient",
        "ridge_intercept",
        "raw_direction",
        "unit_direction",
        "hidden_nuisance_beta",
        "feature_mean",
        "feature_scale",
        "target_nuisance_beta",
        "target_residual_mean",
        "target_residual_scale",
        "train_z_mean",
        "train_z_sd",
    }
)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _direction_member(directory: Path, raw_name: Any) -> Path:
    """Resolve one direction file without permitting path traversal."""

    name = str(raw_name or "")
    if not name:
        raise ValueError(f"Direction index under {directory} has an empty filename")
    candidate = Path(name)
    if candidate.is_absolute():
        raise ValueError(f"Direction filename must be relative: {name!r}")
    resolved = (directory / candidate).resolve()
    if resolved.parent != directory.resolve():
        raise ValueError(f"Direction filename escapes its frozen directory: {name!r}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _finite_scalar(value: Any, name: str) -> float:
    number = float(np.asarray(value).reshape(()))
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite")
    return number


def _finite_array(
    value: Any,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def hidden_checksum_payload(
    hidden_by_layer: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    """Return deterministic per-layer and combined checksums for hidden arrays."""

    if not hidden_by_layer:
        raise ValueError("No hidden states were supplied for checksumming")
    entries: list[dict[str, Any]] = []
    for raw_layer in sorted(hidden_by_layer):
        layer = int(raw_layer)
        hidden = np.asarray(hidden_by_layer[raw_layer])
        if hidden.ndim != 1 or not np.isfinite(hidden).all():
            raise ValueError(f"L{layer} hidden must be one finite vector")
        entries.append(
            {
                "layer": layer,
                "shape": list(hidden.shape),
                "dtype": str(hidden.dtype),
                "array_sha256": _array_sha256(hidden),
            }
        )
    return {
        "layers": entries,
        "combined_fingerprint": stable_hash(entries),
    }


def explicit_nuisance_encoder_from_dict(
    value: Mapping[str, Any],
) -> ExplicitNuisanceEncoder:
    """Materialize and validate the exact explicit nuisance encoding."""

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
    vocabulary = tuple(str(item) for item in value["answer_vocabulary"])
    reference = str(value["answer_reference"])
    columns = tuple(str(item) for item in value["columns"])
    if len(vocabulary) < 2 or len(vocabulary) != len(set(vocabulary)):
        raise ValueError("Frozen answer vocabulary must contain distinct classes")
    if reference not in vocabulary:
        raise ValueError("Frozen nuisance answer reference is outside its vocabulary")
    expected_columns = (
        "intercept",
        "choice_image",
        "choice_other",
        "difficulty_hard",
        "prior_strength",
        "full_margin",
        *(f"answer={answer}" for answer in vocabulary if answer != reference),
    )
    if columns != expected_columns:
        raise ValueError(
            "Frozen explicit nuisance columns differ from the Stage-03 contract"
        )
    encoder = ExplicitNuisanceEncoder(
        answer_vocabulary=vocabulary,
        answer_reference=reference,
        prior_mean=_finite_scalar(value["prior_mean"], "prior_mean"),
        prior_scale=_finite_scalar(value["prior_scale"], "prior_scale"),
        margin_mean=_finite_scalar(value["margin_mean"], "margin_mean"),
        margin_scale=_finite_scalar(value["margin_scale"], "margin_scale"),
        columns=columns,
    )
    if encoder.prior_scale <= 0.0 or encoder.margin_scale <= 0.0:
        raise ValueError("Frozen nuisance scales must be positive")
    return encoder


def _validate_common_direction_arrays(
    payload: Mapping[str, Any],
    encoder: ExplicitNuisanceEncoder,
    *,
    hidden_size: int,
    source: Path,
) -> dict[str, np.ndarray | float]:
    coefficient = _finite_array(
        payload["ridge_coefficient"],
        f"{source}.ridge_coefficient",
        shape=(hidden_size,),
    )
    intercept = _finite_scalar(payload["ridge_intercept"], f"{source}.intercept")
    raw = _finite_array(
        payload["raw_direction"], f"{source}.raw_direction", shape=(hidden_size,)
    )
    unit = _finite_array(
        payload["unit_direction"], f"{source}.unit_direction", shape=(hidden_size,)
    )
    nuisance_width = len(encoder.columns)
    hidden_beta = _finite_array(
        payload["hidden_nuisance_beta"],
        f"{source}.hidden_nuisance_beta",
        shape=(nuisance_width, hidden_size),
    )
    feature_mean = _finite_array(
        payload["feature_mean"], f"{source}.feature_mean", shape=(hidden_size,)
    )
    feature_scale = _finite_array(
        payload["feature_scale"], f"{source}.feature_scale", shape=(hidden_size,)
    )
    target_beta = _finite_array(
        payload["target_nuisance_beta"],
        f"{source}.target_nuisance_beta",
        shape=(nuisance_width, 2),
    )
    if np.any(feature_scale <= 0.0):
        raise ValueError(f"{source} contains a non-positive feature scale")
    norm = float(np.linalg.norm(raw))
    if norm <= 1e-12 or not math.isclose(
        float(np.linalg.norm(unit)), 1.0, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError(f"{source} contains a degenerate direction")
    expected_raw = coefficient / feature_scale
    if not np.allclose(raw, expected_raw, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{source} raw direction does not replay coefficient/scale")
    alignment = float(unit @ (raw / norm))
    if not math.isclose(abs(alignment), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{source} unit direction is not aligned with raw direction")
    return {
        "ridge_coefficient": coefficient,
        "ridge_intercept": intercept,
        "raw_direction": raw,
        "unit_direction": unit,
        "hidden_nuisance_beta": hidden_beta,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_nuisance_beta": target_beta,
    }


@dataclass(frozen=True)
class FrozenPrimarySourceUseModel:
    """One Stage-03 raw-choice-coupled shared fold readout."""

    fold: int
    layer: int
    position: str
    alpha: float
    ridge_coefficient: np.ndarray
    ridge_intercept: float
    raw_direction: np.ndarray
    unit_direction: np.ndarray
    hidden_nuisance_beta: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_nuisance_beta: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    nuisance_encoder: ExplicitNuisanceEncoder
    source_file: Path
    source_sha256: str
    audit: dict[str, Any]

    def residual_hidden(
        self, hidden: np.ndarray, nuisance_row: Mapping[str, Any]
    ) -> np.ndarray:
        value = _finite_array(
            hidden,
            f"Stage03 fold {self.fold} hidden",
            shape=self.raw_direction.shape,
        )
        if nuisance_row.get("fold") is not None and int(nuisance_row["fold"]) != self.fold:
            raise ValueError(
                f"Stage03 fold {self.fold} received a fold-{nuisance_row['fold']} row"
            )
        design = transform_explicit_nuisance(
            [nuisance_row], [0], self.nuisance_encoder
        )[0]
        residual = value - design @ self.hidden_nuisance_beta
        if not np.isfinite(residual).all():
            raise ValueError(f"Stage03 fold {self.fold} produced non-finite residuals")
        return residual

    def project(
        self, hidden: np.ndarray, nuisance_row: Mapping[str, Any]
    ) -> tuple[float, float]:
        """Return ``(coordinate, frozen_prediction)`` without refitting."""

        residual = self.residual_hidden(hidden, nuisance_row)
        standardized = (residual - self.feature_mean) / self.feature_scale
        prediction = float(
            standardized @ self.ridge_coefficient + self.ridge_intercept
        )
        coordinate = float(residual @ self.unit_direction)
        if not math.isfinite(prediction) or not math.isfinite(coordinate):
            raise ValueError(f"Stage03 fold {self.fold} produced a non-finite readout")
        return coordinate, prediction

    def transform_behavior(self, deletion: float, replacement: float) -> float:
        """Apply the frozen raw-target scaling and return its shared target.

        Stage-03's raw estimand does not regress behavioral outcomes on
        nuisance variables; it only standardizes deletion and replacement by
        the development-training fold and averages the two indicators.
        """

        observed = np.asarray([deletion, replacement], dtype=np.float64)
        if not np.isfinite(observed).all():
            raise ValueError(f"Stage03 fold {self.fold} received non-finite behavior")
        standardized = (observed - self.target_mean) / self.target_scale
        value = float(np.mean(standardized))
        if not math.isfinite(value):
            raise ValueError(f"Stage03 fold {self.fold} produced a non-finite target")
        return value


def load_stage03_primary_source_use_models(
    experiment_dir: str | Path,
) -> dict[int, FrozenPrimarySourceUseModel]:
    """Load exactly the five frozen Stage-03 raw/shared fold predictors."""

    experiment = Path(experiment_dir).resolve()
    stage = experiment / BRIDGE_DIR / STAGE03_DIR
    summary = _read_json_object(stage / "summary.json")
    if summary.get("status") != "completed":
        raise ValueError("Stage03 source-use representation is not completed")
    authorization = summary.get("measurement_authorization", {})
    if not isinstance(authorization, Mapping) or authorization.get(
        "raw_readout_allowed"
    ) is not True:
        raise ValueError("Stage03 did not authorize its raw descriptive readout")
    directions = stage / RAW_ESTIMAND / "directions"
    index_path = directions / "index.json"
    index = _read_json_object(index_path)
    if index.get("format_version") != 1 or index.get("estimand") != RAW_ESTIMAND:
        raise ValueError("Stage03 raw direction index has an incompatible format")
    if index.get("confirmatory_used_for_selection_or_fit") is not False:
        raise ValueError("Stage03 index claims confirmatory fitting or selection")
    raw_entries = index.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("Stage03 direction index lacks entries")
    entries = [
        dict(entry)
        for entry in raw_entries
        if str(entry.get("objective")) == PRIMARY_OBJECTIVE
    ]
    by_fold = {int(entry.get("fold", -1)): entry for entry in entries}
    if len(entries) != len(by_fold) or tuple(sorted(by_fold)) != EXPECTED_FOLDS:
        raise ValueError("Stage03 must contain one raw/shared model for folds 0..4")

    models: dict[int, FrozenPrimarySourceUseModel] = {}
    for fold in EXPECTED_FOLDS:
        entry = by_fold[fold]
        if entry.get("estimand") != RAW_ESTIMAND:
            raise ValueError(f"Stage03 fold {fold} estimand drifted")
        if str(entry.get("position")) != POST_ANSWER_POSITION:
            raise ValueError(f"Stage03 fold {fold} primary U is not post_answer")
        layer = int(entry.get("layer", -1))
        if layer < 0:
            raise ValueError(f"Stage03 fold {fold} has an invalid selected layer")
        encoder = explicit_nuisance_encoder_from_dict(
            entry.get("explicit_nuisance", {})
        )
        source = _direction_member(directions, entry.get("file"))
        source_sha = sha256_file(source)
        with np.load(source, allow_pickle=False) as archive:
            missing = _SOURCE_USE_ARRAYS.difference(archive.files)
            if missing:
                raise ValueError(f"{source} omits arrays {sorted(missing)}")
            coefficient_shape = np.asarray(archive["ridge_coefficient"]).shape
            if len(coefficient_shape) != 1 or coefficient_shape[0] <= 0:
                raise ValueError(f"{source} has an invalid ridge coefficient shape")
            hidden_size = int(coefficient_shape[0])
            indexed_hidden_size = entry.get("hidden_size")
            if indexed_hidden_size is not None and int(indexed_hidden_size) != hidden_size:
                raise ValueError(f"Stage03 fold {fold} hidden-size/index mismatch")
            arrays = _validate_common_direction_arrays(
                archive, encoder, hidden_size=hidden_size, source=source
            )
            target_mean = _finite_array(
                archive["target_mean"], f"{source}.target_mean", shape=(2,)
            )
            target_scale = _finite_array(
                archive["target_scale"], f"{source}.target_scale", shape=(2,)
            )
        if np.any(target_scale <= 0.0):
            raise ValueError(f"{source} has a non-positive target scale")
        norm = float(np.linalg.norm(arrays["raw_direction"]))
        if not math.isclose(
            norm,
            _finite_scalar(entry.get("direction_norm"), "direction_norm"),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Stage03 fold {fold} direction norm/index mismatch")
        alpha = _finite_scalar(entry.get("alpha"), "alpha")
        if alpha <= 0.0:
            raise ValueError(f"Stage03 fold {fold} alpha must be positive")
        audit = {
            **entry,
            "index_file": str(index_path),
            "index_sha256": sha256_file(index_path),
            "source_file": str(source),
            "source_sha256": source_sha,
            "arrays_validated": sorted(_SOURCE_USE_ARRAYS),
            "no_refit": True,
        }
        models[fold] = FrozenPrimarySourceUseModel(
            fold=fold,
            layer=layer,
            position=POST_ANSWER_POSITION,
            alpha=alpha,
            ridge_coefficient=np.asarray(arrays["ridge_coefficient"]),
            ridge_intercept=float(arrays["ridge_intercept"]),
            raw_direction=np.asarray(arrays["raw_direction"]),
            unit_direction=np.asarray(arrays["unit_direction"]),
            hidden_nuisance_beta=np.asarray(arrays["hidden_nuisance_beta"]),
            feature_mean=np.asarray(arrays["feature_mean"]),
            feature_scale=np.asarray(arrays["feature_scale"]),
            target_nuisance_beta=np.asarray(arrays["target_nuisance_beta"]),
            target_mean=target_mean,
            target_scale=target_scale,
            nuisance_encoder=encoder,
            source_file=source,
            source_sha256=source_sha,
            audit=audit,
        )
    return models


def load_bridge08_source_use_models(
    experiment_dir: str | Path,
) -> dict[int, SourceUseFoldModel]:
    """Load the five checksum-pinned fixed-L18 ``U_L18`` candidates."""

    experiment = Path(experiment_dir).resolve()
    directions = experiment / BRIDGE_DIR / STAGE08_DIR / "directions"
    index_path = directions / "index.json"
    index = _read_json_object(index_path)
    if index.get("format_version") != 1:
        raise ValueError("Bridge08 direction index has an incompatible format")
    if index.get("development_only") is not True:
        raise ValueError("Bridge08 direction index is not development-only")
    if index.get("confirmatory_fit_or_selection") is not False:
        raise ValueError("Bridge08 index claims confirmatory fitting or selection")
    raw_entries = index.get("folds")
    if not isinstance(raw_entries, list):
        raise ValueError("Bridge08 direction index lacks fold entries")
    entries = [dict(entry) for entry in raw_entries]
    by_fold = {int(entry.get("fold", -1)): entry for entry in entries}
    if len(entries) != len(by_fold) or tuple(sorted(by_fold)) != EXPECTED_FOLDS:
        raise ValueError("Bridge08 must contain exactly one model for folds 0..4")

    output: dict[int, SourceUseFoldModel] = {}
    for fold in EXPECTED_FOLDS:
        entry = by_fold[fold]
        hidden_size = int(entry.get("hidden_size", -1))
        if hidden_size <= 0:
            raise ValueError(f"Bridge08 fold {fold} has an invalid hidden size")
        encoder = explicit_nuisance_encoder_from_dict(
            entry.get("explicit_nuisance", {})
        )
        source = _direction_member(directions, entry.get("file"))
        checksum = sha256_file(source)
        if checksum != str(entry.get("sha256", "")):
            raise ValueError(f"Bridge08 fold {fold} direction checksum changed")
        with np.load(source, allow_pickle=False) as archive:
            missing = _STAGE08_ARRAYS.difference(archive.files)
            if missing:
                raise ValueError(f"{source} omits arrays {sorted(missing)}")
            arrays = _validate_common_direction_arrays(
                archive, encoder, hidden_size=hidden_size, source=source
            )
            target_mean = _finite_array(
                archive["target_residual_mean"],
                f"{source}.target_residual_mean",
                shape=(2,),
            )
            target_scale = _finite_array(
                archive["target_residual_scale"],
                f"{source}.target_residual_scale",
                shape=(2,),
            )
            alpha = _finite_scalar(archive["alpha"], f"{source}.alpha")
            train_z_mean = _finite_scalar(
                archive["train_z_mean"], f"{source}.train_z_mean"
            )
            train_z_sd = _finite_scalar(
                archive["train_z_sd"], f"{source}.train_z_sd"
            )
        if alpha <= 0.0 or train_z_sd <= 0.0 or np.any(target_scale <= 0.0):
            raise ValueError(f"Bridge08 fold {fold} contains an invalid scale/alpha")
        if not math.isclose(
            alpha,
            _finite_scalar(entry.get("selected_alpha"), "selected_alpha"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Bridge08 fold {fold} alpha/index mismatch")
        for name, observed in (
            ("train_z_mean", train_z_mean),
            ("train_z_sd", train_z_sd),
        ):
            if not math.isclose(
                observed,
                _finite_scalar(entry.get(name), name),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(f"Bridge08 fold {fold} {name}/index mismatch")
        norm = float(np.linalg.norm(arrays["raw_direction"]))
        if not math.isclose(
            norm,
            _finite_scalar(entry.get("raw_direction_norm"), "raw_direction_norm"),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Bridge08 fold {fold} direction norm/index mismatch")
        output[fold] = SourceUseFoldModel(
            fold=fold,
            alpha=alpha,
            ridge_coefficient=np.asarray(arrays["ridge_coefficient"]),
            ridge_intercept=float(arrays["ridge_intercept"]),
            raw_direction=np.asarray(arrays["raw_direction"]),
            unit_direction=np.asarray(arrays["unit_direction"]),
            hidden_nuisance_beta=np.asarray(arrays["hidden_nuisance_beta"]),
            feature_mean=np.asarray(arrays["feature_mean"]),
            feature_scale=np.asarray(arrays["feature_scale"]),
            target_nuisance_beta=np.asarray(arrays["target_nuisance_beta"]),
            target_residual_mean=target_mean,
            target_residual_scale=target_scale,
            nuisance_encoder=encoder,
            train_z_mean=train_z_mean,
            train_z_sd=train_z_sd,
            audit={
                **entry,
                "index_file": str(index_path),
                "index_sha256": sha256_file(index_path),
                "source_file": str(source),
                "source_sha256": checksum,
                "arrays_validated": sorted(_STAGE08_ARRAYS),
                "no_refit": True,
            },
        )
    return output


@dataclass(frozen=True)
class FrozenNuisanceBaseline:
    """Replayed Stage-03 observed-covariate-only shared predictor."""

    fold: int
    encoder: ExplicitNuisanceEncoder
    coefficient: np.ndarray
    source_file: Path
    source_sha256: str
    training_item_fingerprint: str
    coefficient_sha256: str
    replay_max_abs_error: float | None

    def predict(self, nuisance_row: Mapping[str, Any]) -> float:
        if nuisance_row.get("fold") is not None and int(nuisance_row["fold"]) != self.fold:
            raise ValueError(
                f"Nuisance baseline fold {self.fold} received fold {nuisance_row['fold']}"
            )
        design = transform_explicit_nuisance(
            [nuisance_row], [0], self.encoder
        )[0]
        value = float(design @ self.coefficient)
        if not math.isfinite(value):
            raise ValueError(f"Nuisance baseline fold {self.fold} is non-finite")
        return value

    def to_manifest(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "source_file": str(self.source_file),
            "source_sha256": self.source_sha256,
            "training_item_fingerprint": self.training_item_fingerprint,
            "explicit_nuisance": self.encoder.to_dict(),
            "coefficient": self.coefficient.tolist(),
            "coefficient_sha256": self.coefficient_sha256,
            "replay_max_abs_error": self.replay_max_abs_error,
            "fit_scope": "Stage01 development items outside this fold only",
            "stage09_outcomes_used": False,
        }


def reconstruct_stage03_nuisance_baselines(
    development_rows: Sequence[Mapping[str, Any]],
    primary_models: Mapping[int, FrozenPrimarySourceUseModel],
    *,
    source_file: str | Path,
    source_sha256: str,
    expected_oof_rows: Sequence[Mapping[str, Any]] | None = None,
    replay_tolerance: float = 1e-10,
) -> dict[int, FrozenNuisanceBaseline]:
    """Rebuild the original nuisance baseline from old development data only."""

    rows = [normalize_measurement_row(row) for row in development_rows]
    if len(rows) != EXPECTED_DEVELOPMENT_N:
        raise ValueError(
            f"Nuisance baseline requires {EXPECTED_DEVELOPMENT_N} development rows; "
            f"found {len(rows)}"
        )
    if len({str(row.get("case_id", "")) for row in rows}) != len(rows):
        raise ValueError("Stage01 development rows do not have unique case IDs")
    if len({str(row.get("item_id", "")) for row in rows}) != len(rows):
        raise ValueError("Stage01 development rows do not have unique item IDs")
    if tuple(sorted({int(row["fold"]) for row in rows})) != EXPECTED_FOLDS:
        raise ValueError("Stage01 development rows do not cover folds 0..4")
    if tuple(sorted(primary_models)) != EXPECTED_FOLDS:
        raise ValueError("Primary Stage03 models do not cover folds 0..4")

    expected = (
        {str(row["case_id"]): dict(row) for row in expected_oof_rows}
        if expected_oof_rows is not None
        else {}
    )
    if expected and set(expected) != {str(row["case_id"]) for row in rows}:
        raise ValueError("Stage03 OOF nuisance replay rows differ from Stage01 development")
    output: dict[int, FrozenNuisanceBaseline] = {}
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    for fold in EXPECTED_FOLDS:
        model = primary_models[fold]
        training = np.flatnonzero(folds != fold)
        heldout = np.flatnonzero(folds == fold)
        training_items = sorted(str(rows[index]["item_id"]) for index in training)
        heldout_items = {str(rows[index]["item_id"]) for index in heldout}
        if set(training_items).intersection(heldout_items):
            raise RuntimeError(f"Nuisance baseline fold {fold} leaks held-out items")
        target_fit = fit_target_transform(
            rows,
            training,
            estimand=RAW_ESTIMAND,
            answer_vocabulary=model.nuisance_encoder.answer_vocabulary,
        )
        if target_fit.encoder != model.nuisance_encoder:
            raise ValueError(f"Nuisance encoder replay differs in Stage03 fold {fold}")
        for name, replayed, frozen in (
            ("target_nuisance_beta", target_fit.nuisance_beta, model.target_nuisance_beta),
            ("target_mean", target_fit.target_mean, model.target_mean),
            ("target_scale", target_fit.target_scale, model.target_scale),
        ):
            if not np.allclose(replayed, frozen, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Stage03 fold {fold} {name} does not replay")
        train_target, x_train = target_fit.apply(rows, training)
        coefficient = np.asarray(
            _fit_baseline(x_train, train_target["shared"]), dtype=np.float64
        )
        if coefficient.shape != (len(model.nuisance_encoder.columns),) or not np.isfinite(
            coefficient
        ).all():
            raise ValueError(f"Stage03 fold {fold} nuisance coefficient is invalid")
        heldout_predictions: dict[str, float] = {}
        if len(heldout):
            _targets, x_heldout = target_fit.apply(rows, heldout)
            values = x_heldout @ coefficient
            heldout_predictions = {
                str(rows[index]["case_id"]): float(values[local])
                for local, index in enumerate(heldout)
            }
        replay_errors = [
            abs(
                prediction
                - float(expected[case_id]["prediction_nuisance"])
            )
            for case_id, prediction in heldout_predictions.items()
            if expected
        ]
        replay_error = max(replay_errors) if replay_errors else None
        if replay_error is not None and replay_error > replay_tolerance:
            raise ValueError(
                f"Stage03 fold {fold} nuisance OOF replay error {replay_error} "
                f"exceeds {replay_tolerance}"
            )
        output[fold] = FrozenNuisanceBaseline(
            fold=fold,
            encoder=model.nuisance_encoder,
            coefficient=coefficient,
            source_file=Path(source_file).resolve(),
            source_sha256=str(source_sha256),
            training_item_fingerprint=stable_hash(training_items),
            coefficient_sha256=_array_sha256(coefficient),
            replay_max_abs_error=replay_error,
        )
    return output


def load_stage03_nuisance_baselines(
    experiment_dir: str | Path,
    primary_models: Mapping[int, FrozenPrimarySourceUseModel] | None = None,
) -> dict[int, FrozenNuisanceBaseline]:
    """Load old rows and exactly replay Stage-03's nuisance-only baseline."""

    experiment = Path(experiment_dir).resolve()
    bridge = experiment / BRIDGE_DIR
    source = bridge / STAGE01_DIR / "development_analysis.jsonl"
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha = sha256_file(source)
    provenance = _read_json_object(bridge / STAGE08_DIR / "provenance.json")
    pinned = (
        provenance.get("files", {})
        .get("method01_development_analysis", {})
        .get("sha256")
    )
    if not pinned or source_sha != str(pinned):
        raise ValueError("Stage01 development analysis differs from Bridge08 provenance")
    rows = [
        row
        for row in load_jsonl(source, repair_trailing=False)
        if row.get("status") == "completed"
    ]
    oof_path = (
        bridge
        / STAGE03_DIR
        / RAW_ESTIMAND
        / "development_oof_predictions.jsonl"
    )
    oof = load_jsonl(oof_path, repair_trailing=False)
    return reconstruct_stage03_nuisance_baselines(
        rows,
        primary_models or load_stage03_primary_source_use_models(experiment),
        source_file=source,
        source_sha256=source_sha,
        expected_oof_rows=oof,
    )


def load_attribution_repository(
    experiment_dir: str | Path,
) -> FrozenDirectionRepository:
    """Load Stage06's byte-frozen copy of the Stage10 attribution rule."""

    root = (
        Path(experiment_dir).resolve() / BRIDGE_DIR / STAGE06_DIR
    )
    repository = FrozenDirectionRepository(root)
    # Force all checksum and array checks now rather than on the first GPU item.
    directions = [repository.get(fold) for fold in EXPECTED_FOLDS]
    hidden_sizes = {int(direction.d_unit.size) for direction in directions}
    if len(hidden_sizes) != 1:
        raise ValueError("Frozen attribution directions disagree on hidden size")
    return repository


@dataclass(frozen=True)
class ProspectiveReadoutRepository:
    primary_u: dict[int, FrozenPrimarySourceUseModel]
    secondary_u_l18: dict[int, SourceUseFoldModel]
    attribution: FrozenDirectionRepository
    nuisance_baseline: dict[int, FrozenNuisanceBaseline]

    def required_answer_layers(self, fold: int) -> tuple[int, ...]:
        key = int(fold)
        if key not in self.primary_u or key not in self.secondary_u_l18:
            raise KeyError(f"Frozen readouts omit fold {key}")
        return tuple(sorted({int(self.primary_u[key].layer), SECONDARY_LAYER}))

    def audit_manifest(self) -> dict[str, Any]:
        attribution_rule_path = self.attribution.output_dir / "frozen_rule.json"
        attribution_folds: list[dict[str, Any]] = []
        for entry in sorted(
            self.attribution.rule.get("folds", []),
            key=lambda value: int(value["fold"]),
        ):
            source = self.attribution.output_dir / str(entry["frozen_file"])
            actual_sha256 = sha256_file(source)
            expected_sha256 = str(entry["sha256"])
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Frozen attribution direction checksum changed for fold {entry['fold']}"
                )
            attribution_folds.append(
                {
                    "fold": int(entry["fold"]),
                    "source_file": str(source),
                    "source_sha256": actual_sha256,
                    "expected_sha256": expected_sha256,
                    "bytes": source.stat().st_size,
                }
            )
        payload: dict[str, Any] = {
            "primary_u": {
                str(fold): model.audit for fold, model in sorted(self.primary_u.items())
            },
            "secondary_u_l18": {
                str(fold): model.audit
                for fold, model in sorted(self.secondary_u_l18.items())
            },
            "nuisance_baseline": {
                str(fold): model.to_manifest()
                for fold, model in sorted(self.nuisance_baseline.items())
            },
            "attribution": {
                "rule_file": str(attribution_rule_path),
                "rule_sha256": sha256_file(attribution_rule_path),
                "rule_fingerprint": self.attribution.rule.get("rule_fingerprint"),
                "folds": attribution_folds,
            },
            "fit_performed_by_stage09": False,
        }
        payload["readout_manifest_fingerprint"] = stable_hash(payload)
        return payload


def load_prospective_readout_repository(
    experiment_dir: str | Path,
    *,
    include_nuisance_baseline: bool = True,
) -> ProspectiveReadoutRepository:
    primary = load_stage03_primary_source_use_models(experiment_dir)
    secondary = load_bridge08_source_use_models(experiment_dir)
    primary_sizes = {model.raw_direction.size for model in primary.values()}
    secondary_sizes = {model.raw_direction.size for model in secondary.values()}
    if len(primary_sizes | secondary_sizes) != 1:
        raise ValueError("Primary/secondary U directions disagree on hidden size")
    baseline = (
        load_stage03_nuisance_baselines(experiment_dir, primary)
        if include_nuisance_baseline
        else {}
    )
    return ProspectiveReadoutRepository(
        primary_u=primary,
        secondary_u_l18=secondary,
        attribution=load_attribution_repository(experiment_dir),
        nuisance_baseline=baseline,
    )


def _prospective_target_rows(plan_or_target_rows: Any) -> list[dict[str, Any]]:
    """Normalize either a Stage-09 plan or an explicit candidate-row sequence."""

    value = plan_or_target_rows
    inventory = getattr(value, "inventory", None)
    if inventory is not None and hasattr(inventory, "rows"):
        value = inventory.rows
    elif isinstance(value, Mapping) and "inventory" in value:
        nested = value["inventory"]
        value = nested.get("rows") if isinstance(nested, Mapping) else getattr(
            nested, "rows", nested
        )
    elif isinstance(value, Mapping) and "rows" in value:
        value = value["rows"]
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(
            "target readout exclusion requires a Stage-09 plan or a row sequence"
        )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"Target row {index} is not a mapping")
        rows.append(dict(raw))
    if not rows:
        raise ValueError("Target readout exclusion received no candidate rows")
    return rows


def _item_list(values: Sequence[Any] | set[str]) -> list[str]:
    """Return a unique deterministic item-id list without numeric coercion."""

    return sorted({str(value) for value in values if str(value) != ""})


def _source_overlap_manifest(
    target_items: set[str], source_items: set[str]
) -> dict[str, Any]:
    overlap = _item_list(target_items.intersection(source_items))
    return {
        "item_n": len(source_items),
        "item_fingerprint": stable_hash(_item_list(source_items)),
        "target_overlap_n": len(overlap),
        "target_overlap_items": overlap,
    }


def audit_target_readout_exclusion(
    plan_or_target_rows: Any,
    repository: ProspectiveReadoutRepository,
    experiment_dir: str | Path,
) -> dict[str, Any]:
    """Prove that all 67 prospective target items are readout-external.

    This is a pure CPU/read-only audit.  It binds the prospective candidate
    universe to the exact frozen Stage-03, Bridge-08, and Stage-06 artifacts,
    and checks every candidate row against the immutable Stage-1 item split.
    A formal runner must persist the returned manifest and hard-fail before
    endpoint freezing unless ``passed`` is exactly ``True``.
    """

    experiment = Path(experiment_dir).resolve()
    target_rows = _prospective_target_rows(plan_or_target_rows)
    target_item_values = [str(row.get("item_id", "")) for row in target_rows]
    invalid_target_rows = [
        {
            "row_index": index,
            "case_id": str(row.get("case_id", "")),
            "item_id": str(row.get("item_id", "")),
            "reason": "missing_item_id",
        }
        for index, row in enumerate(target_rows)
        if str(row.get("item_id", "")) == ""
    ]
    target_items = {value for value in target_item_values if value}

    split_path = experiment / "stage1_metacognition" / "item_split" / "split_assignments.json"
    split = _read_json_object(split_path)
    raw_item_to_fold = split.get("item_to_fold")
    if not isinstance(raw_item_to_fold, Mapping):
        raise ValueError(f"Base item split lacks item_to_fold: {split_path}")
    item_to_fold: dict[str, int] = {}
    for item_id, raw_fold in raw_item_to_fold.items():
        try:
            fold = int(raw_fold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid base fold for item {item_id!r}") from exc
        if fold not in EXPECTED_FOLDS:
            raise ValueError(f"Base item {item_id!r} has out-of-range fold {fold}")
        item_to_fold[str(item_id)] = fold

    fold_mismatches: list[dict[str, Any]] = []
    for index, row in enumerate(target_rows):
        item_id = str(row.get("item_id", ""))
        if not item_id:
            continue
        expected_fold = item_to_fold.get(item_id)
        try:
            observed_fold = int(row.get("fold"))
        except (TypeError, ValueError, OverflowError):
            observed_fold = None
        reason = None
        if expected_fold is None:
            reason = "item_missing_from_base_split"
        elif observed_fold != expected_fold:
            reason = "fold_mismatch"
        if reason is not None:
            fold_mismatches.append(
                {
                    "row_index": index,
                    "case_id": str(row.get("case_id", "")),
                    "item_id": item_id,
                    "observed_fold": observed_fold,
                    "expected_fold": expected_fold,
                    "reason": reason,
                }
            )

    bridge = experiment / BRIDGE_DIR

    # Stage-03 was fit from the completed development measurement rows.  Bind
    # that table to Stage-03's own provenance, rather than inferring its items
    # from a later summary.
    stage03_source = bridge / STAGE01_DIR / "development_results.jsonl"
    stage03_source_sha = sha256_file(stage03_source)
    stage03_provenance_path = bridge / STAGE03_DIR / "provenance.json"
    stage03_provenance = _read_json_object(stage03_provenance_path)
    pinned_stage03_source = (
        stage03_provenance.get("inputs", {})
        .get("development_results.jsonl", {})
    )
    stage03_rows = [
        dict(row)
        for row in load_jsonl(stage03_source, repair_trailing=False)
        if row.get("status") == "completed"
    ]
    stage03_items = {
        str(row.get("item_id", ""))
        for row in stage03_rows
        if str(row.get("item_id", ""))
    }
    stage03_primary_folds = sorted(int(fold) for fold in repository.primary_u)
    stage03_index_files = {
        str(model.audit.get("index_file", ""))
        for model in repository.primary_u.values()
    }
    stage03_source_manifest = {
        "source_file": str(stage03_source),
        "source_sha256": stage03_source_sha,
        "provenance_file": str(stage03_provenance_path),
        "provenance_sha256": sha256_file(stage03_provenance_path),
        "provenance_pinned_path": str(pinned_stage03_source.get("path", "")),
        "provenance_pinned_sha256": str(pinned_stage03_source.get("sha256", "")),
        "completed_row_n": len(stage03_rows),
        "unique_case_n": len(
            {str(row.get("case_id", "")) for row in stage03_rows}
        ),
        **_source_overlap_manifest(target_items, stage03_items),
        "repository_primary_folds": stage03_primary_folds,
        "repository_direction_index_files": sorted(stage03_index_files),
    }

    # Bridge-08 entries carry the exact per-fold training item lists used for
    # the fixed-L18 fit.  Take their union, but retain fold-level counts and
    # fingerprints so a future artifact mutation cannot hide behind the union.
    bridge08_folds: list[dict[str, Any]] = []
    bridge08_train_items: set[str] = set()
    bridge08_fold_ids = sorted(int(fold) for fold in repository.secondary_u_l18)
    for fold in bridge08_fold_ids:
        audit = repository.secondary_u_l18[fold].audit
        raw_train = audit.get("train_items")
        if not isinstance(raw_train, list):
            raw_train = []
        train_items = set(_item_list(raw_train))
        bridge08_train_items.update(train_items)
        bridge08_folds.append(
            {
                "fold": fold,
                "train_item_n": len(train_items),
                "declared_train_n": audit.get("train_n"),
                "train_item_fingerprint": stable_hash(_item_list(train_items)),
                "target_overlap_items": _item_list(target_items.intersection(train_items)),
            }
        )
    bridge08_index_path = bridge / STAGE08_DIR / "directions" / "index.json"
    bridge08_source_manifest = {
        "index_file": str(bridge08_index_path),
        "index_sha256": sha256_file(bridge08_index_path),
        "fold_n": len(bridge08_folds),
        "folds": bridge08_folds,
        **_source_overlap_manifest(target_items, bridge08_train_items),
    }

    # Stage-06 is the byte-frozen Stage-10 attribution rule.  Both its training
    # and test items count as prior readout exposure for this prospective test.
    attribution_rule_path = repository.attribution.output_dir / "frozen_rule.json"
    attribution_entries = repository.attribution.rule.get("folds")
    if not isinstance(attribution_entries, list):
        attribution_entries = []
    attribution_folds: list[dict[str, Any]] = []
    attribution_train_items: set[str] = set()
    attribution_test_items: set[str] = set()
    for raw_entry in sorted(
        (dict(entry) for entry in attribution_entries if isinstance(entry, Mapping)),
        key=lambda entry: int(entry.get("fold", -1)),
    ):
        fold = int(raw_entry.get("fold", -1))
        raw_train = raw_entry.get("train_items")
        raw_test = raw_entry.get("test_items")
        train_items = set(_item_list(raw_train if isinstance(raw_train, list) else []))
        test_items = set(_item_list(raw_test if isinstance(raw_test, list) else []))
        attribution_train_items.update(train_items)
        attribution_test_items.update(test_items)
        attribution_folds.append(
            {
                "fold": fold,
                "train_item_n": len(train_items),
                "test_item_n": len(test_items),
                "within_fold_train_test_overlap_items": _item_list(
                    train_items.intersection(test_items)
                ),
                "train_item_fingerprint": stable_hash(_item_list(train_items)),
                "test_item_fingerprint": stable_hash(_item_list(test_items)),
            }
        )
    attribution_exposed_items = attribution_train_items.union(attribution_test_items)
    attribution_source_manifest = {
        "frozen_rule_file": str(attribution_rule_path),
        "frozen_rule_sha256": sha256_file(attribution_rule_path),
        "rule_fingerprint": repository.attribution.rule.get("rule_fingerprint"),
        "fold_n": len(attribution_folds),
        "folds": attribution_folds,
        "train_union_item_n": len(attribution_train_items),
        "train_union_item_fingerprint": stable_hash(_item_list(attribution_train_items)),
        "test_union_item_n": len(attribution_test_items),
        "test_union_item_fingerprint": stable_hash(_item_list(attribution_test_items)),
        **_source_overlap_manifest(target_items, attribution_exposed_items),
    }

    readout_manifest = repository.audit_manifest()
    stage03_overlap = stage03_source_manifest["target_overlap_items"]
    bridge08_overlap = bridge08_source_manifest["target_overlap_items"]
    attribution_overlap = attribution_source_manifest["target_overlap_items"]
    checks = {
        "target_unique_item_n_is_67": len(target_items) == 67,
        "target_rows_have_item_ids": not invalid_target_rows,
        "every_target_row_matches_base_item_split": not fold_mismatches,
        "stage03_development_completed_row_n_is_97": len(stage03_rows) == 97,
        "stage03_development_unique_case_n_is_97": (
            stage03_source_manifest["unique_case_n"] == 97
        ),
        "stage03_development_unique_item_n_is_97": len(stage03_items) == 97,
        "stage03_development_checksum_pinned": (
            str(pinned_stage03_source.get("sha256", "")) == stage03_source_sha
        ),
        "stage03_repository_has_folds_0_to_4": stage03_primary_folds
        == list(EXPECTED_FOLDS),
        "stage03_repository_uses_one_direction_index": len(stage03_index_files) == 1,
        "target_excluded_from_stage03_development": not stage03_overlap,
        "bridge08_repository_has_folds_0_to_4": bridge08_fold_ids
        == list(EXPECTED_FOLDS),
        "bridge08_each_fold_has_declared_training_items": bool(
            len(bridge08_folds) == len(EXPECTED_FOLDS)
            and all(
                entry["train_item_n"] > 0
                and entry["declared_train_n"] == entry["train_item_n"]
                for entry in bridge08_folds
            )
        ),
        "bridge08_train_union_equals_stage03_development": (
            bridge08_train_items == stage03_items
        ),
        "target_excluded_from_bridge08_train_union": not bridge08_overlap,
        "stage06_attribution_has_folds_0_to_4": [
            entry["fold"] for entry in attribution_folds
        ]
        == list(EXPECTED_FOLDS),
        "stage06_each_fold_has_train_and_test_items": bool(
            len(attribution_folds) == len(EXPECTED_FOLDS)
            and all(
                entry["train_item_n"] > 0 and entry["test_item_n"] > 0
                for entry in attribution_folds
            )
        ),
        "stage06_within_fold_train_test_disjoint": all(
            not entry["within_fold_train_test_overlap_items"]
            for entry in attribution_folds
        ),
        "target_excluded_from_stage06_train_test_union": not attribution_overlap,
    }
    payload: dict[str, Any] = {
        "format_version": 1,
        "audit": "stage09_target_readout_exclusion",
        "cpu_only": True,
        "read_only": True,
        "required_before_formal_endpoint_freeze": True,
        "required_action_if_failed": "hard_fail_before_formal_freeze",
        "target": {
            "row_n": len(target_rows),
            "unique_item_n": len(target_items),
            "expected_unique_item_n": 67,
            "item_ids": _item_list(target_items),
            "item_fingerprint": stable_hash(_item_list(target_items)),
            "invalid_rows": invalid_target_rows,
        },
        "base_item_split": {
            "source_file": str(split_path),
            "source_sha256": sha256_file(split_path),
            "mapped_item_n": len(item_to_fold),
            "checked_target_row_n": len(target_rows) - len(invalid_target_rows),
            "mismatch_n": len(fold_mismatches),
            "mismatches": fold_mismatches,
        },
        "sources": {
            "stage03_development_fit": stage03_source_manifest,
            "bridge08_train_union": bridge08_source_manifest,
            "stage06_attribution_train_test_union": attribution_source_manifest,
        },
        "readout_manifest_fingerprint": readout_manifest[
            "readout_manifest_fingerprint"
        ],
        "checks": checks,
        "passed": all(checks.values()),
    }
    payload["fingerprint"] = stable_hash(payload)
    return payload


def _input_tensor(inputs: Any, key: str) -> torch.Tensor:
    try:
        value = inputs[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Prepared inputs omit {key}") from exc
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Prepared input {key} is not a tensor")
    return value


def _nonlanguage_tensor_identity(inputs: Any) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    keys = (
        list(inputs.keys())
        if hasattr(inputs, "keys")
        else []
    )
    for key in keys:
        if key in {"input_ids", "attention_mask"}:
            continue
        value = inputs[key]
        if isinstance(value, torch.Tensor):
            output[str(key)] = {
                "object_id": id(value),
                "data_ptr": int(value.data_ptr()),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "version": int(value._version),
            }
    return output


def audit_single_token_causal_prefix(
    *,
    pre_input_ids: torch.Tensor,
    pre_attention_mask: torch.Tensor,
    post_inputs: Any,
    expected_token_id: int,
    nonlanguage_before: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prove an answer token was appended without changing its causal prefix."""

    post_ids = _input_tensor(post_inputs, "input_ids")
    post_mask = _input_tensor(post_inputs, "attention_mask")
    pre_length = int(pre_input_ids.shape[1])
    checks = {
        "batch_shape_equal": int(post_ids.shape[0]) == int(pre_input_ids.shape[0]),
        "exactly_one_token_appended": int(post_ids.shape[1]) == pre_length + 1,
        "input_ids_prefix_equal": bool(
            torch.equal(post_ids[:, :pre_length], pre_input_ids)
        ),
        "attention_mask_prefix_equal": bool(
            torch.equal(post_mask[:, :pre_length], pre_attention_mask)
        ),
        "appended_attention_enabled": bool(
            post_mask.shape[1] == pre_length + 1
            and torch.all(post_mask[:, pre_length] == 1).item()
        ),
        "appended_token_equal": bool(
            post_ids.shape[1] == pre_length + 1
            and torch.all(post_ids[:, pre_length] == int(expected_token_id)).item()
        ),
    }
    nonlanguage_after = _nonlanguage_tensor_identity(post_inputs)
    if nonlanguage_before is not None:
        checks["nonlanguage_tensor_identity_equal"] = (
            dict(nonlanguage_before) == nonlanguage_after
        )
    passed = bool(checks and all(checks.values()))
    audit = {
        "passed": passed,
        "pre_token_count": pre_length,
        "post_token_count": int(post_ids.shape[1]),
        "expected_appended_token_id": int(expected_token_id),
        "checks": checks,
        "construction": "append exact canonical token to the same prepared inputs",
        "nonlanguage_before": dict(nonlanguage_before or {}),
        "nonlanguage_after": nonlanguage_after,
    }
    if not passed:
        raise RuntimeError(f"Teacher-forced causal prefix changed: {audit}")
    return audit


def _capture_selected_post_answer_layers(
    runtime: Stage3Runtime,
    inputs: Any,
    *,
    post_answer_position: int,
    layers: Sequence[int],
    replay_logits_position: int,
) -> tuple[dict[int, np.ndarray], torch.Tensor, dict[str, Any]]:
    """Run one read-only forward and capture only the requested decoder layers."""

    selected = tuple(sorted(set(int(layer) for layer in layers)))
    if not selected:
        raise ValueError("At least one post-answer layer must be captured")
    if any(layer < 0 or layer >= runtime.modules.num_hidden_layers for layer in selected):
        raise ValueError(f"Requested layer is outside the model: {selected}")
    input_length = int(_input_tensor(inputs, "input_ids").shape[1])
    if post_answer_position < 0 or post_answer_position >= input_length:
        raise ValueError("Post-answer capture position is outside the input")

    calls = {layer: 0 for layer in selected}
    captured: dict[int, np.ndarray] = {}
    handles: list[Any] = []

    def capture(layer: int, output: Any) -> None:
        calls[layer] += 1
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise TypeError(
                f"L{layer} decoder output is not rank-three hidden state"
            )
        if int(tensor.shape[1]) != input_length:
            raise ValueError(f"L{layer} hook observed an unexpected sequence length")
        if int(tensor.shape[2]) != runtime.modules.hidden_size:
            raise ValueError(f"L{layer} hook observed an unexpected hidden size")
        if layer in captured:
            raise RuntimeError(f"L{layer} hook captured more than once")
        captured[layer] = (
            tensor[0, post_answer_position, :]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )

    try:
        for layer in selected:
            handles.append(
                runtime.modules.language_layers[layer].register_forward_hook(
                    lambda _module, _args, output, layer=layer: capture(layer, output)
                )
            )
        replay_logits = run_logits_forward(
            runtime.model,
            inputs,
            [int(replay_logits_position)],
            runtime.modules,
        )[int(replay_logits_position)]
    finally:
        for handle in handles:
            handle.remove()
    if calls != {layer: 1 for layer in selected} or set(captured) != set(selected):
        raise RuntimeError(
            f"Post-answer read-only hooks did not fire exactly once: {calls}"
        )
    if any(not np.isfinite(hidden).all() for hidden in captured.values()):
        raise ValueError("Post-answer readout hook captured non-finite hidden state")
    return (
        captured,
        replay_logits,
        {
            "layers": list(selected),
            "hook_call_count": {str(layer): calls[layer] for layer in selected},
            "hook_exactly_once": True,
            "intervention_registered": False,
            "steering_applied": False,
            "injection_l2": 0.0,
            "forward_count": 1,
            "position": POST_ANSWER_POSITION,
        },
    )


def _distribution_from_vocab_logits(
    vocab_logits: torch.Tensor,
    token_ids: Mapping[str, int],
) -> dict[str, Any]:
    return restricted_distribution(vocab_logits, dict(token_ids))


def restricted_next_answer_distribution(
    runtime: Stage3Runtime,
    messages: Sequence[dict[str, Any]],
    *,
    answer_classes: Sequence[str],
    assistant_text: str = ASSISTANT_ANSWER_PREFILL,
) -> dict[str, Any]:
    """Measure a restricted next-answer distribution on any answer-only History."""

    copied = copy.deepcopy(list(messages))
    if contains_verbal_sa_request(copied):
        raise ValueError("Answer-only readout messages contain a verbal-SA request")
    rendered, inputs = runtime.generator.prepare_messages(
        copied, assistant_text=assistant_text
    )
    position = int(_input_tensor(inputs, "input_ids").shape[1]) - 1
    logits = run_logits_forward(
        runtime.model, inputs, [position], runtime.modules
    )[position]
    token_ids = canonical_answer_token_ids(
        runtime.generator.tokenizer, answer_classes
    )
    result = _distribution_from_vocab_logits(logits, token_ids)
    result.update(
        {
            "messages_hash": canonical_message_hash(copied),
            "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "target_position": position,
            "input_token_count": int(_input_tensor(inputs, "input_ids").shape[1]),
            "verbal_sa_leakage": False,
            "measurement": "restricted next-answer distribution",
        }
    )
    del inputs, logits
    return result


def nuisance_row_for_fixed_answer(
    base_row: Mapping[str, Any],
    *,
    fixed_answer: str,
    answer_margin: float,
) -> dict[str, Any]:
    """Build the explicit Stage-03 nuisance row for one History branch."""

    row = dict(base_row)
    answer = str(normalize_answer(fixed_answer))
    if not answer:
        raise ValueError("The fixed answer is empty")
    text_answer = (
        str(normalize_answer(row["text_answer"]))
        if row.get("text_answer") is not None
        else None
    )
    image_answer = (
        str(normalize_answer(row["image_answer"]))
        if row.get("image_answer") is not None
        else None
    )
    derived_side = (
        "image"
        if image_answer is not None and answer == image_answer
        else "text"
        if text_answer is not None and answer == text_answer
        else "other"
    )
    supplied_side = row.get("answer_star_side")
    if supplied_side is not None and str(supplied_side) != derived_side:
        raise ValueError(
            f"Fixed-answer side disagrees with text/image endpoints: "
            f"{supplied_side!r} != {derived_side!r}"
        )
    row.update(
        {
            "answer_star": answer,
            "answer_star_side": derived_side,
            "full_margin": _finite_scalar(answer_margin, "answer_margin"),
            "fold": int(row["fold"]),
        }
    )
    for key in ("difficulty", "prior_strength"):
        if row.get(key) is None:
            raise ValueError(f"Nuisance row lacks {key}")
    return row


def project_answer_hidden(
    hidden_by_layer: Mapping[int, np.ndarray],
    nuisance_row: Mapping[str, Any],
    readouts: ProspectiveReadoutRepository,
) -> dict[str, Any]:
    """Pure projection of one captured answer state onto both frozen U readouts."""

    fold = int(nuisance_row["fold"])
    primary = readouts.primary_u[fold]
    if primary.layer not in hidden_by_layer or SECONDARY_LAYER not in hidden_by_layer:
        raise ValueError(
            f"Captured hidden omits required layers {primary.layer} and {SECONDARY_LAYER}"
        )
    primary_coordinate, primary_prediction = primary.project(
        hidden_by_layer[primary.layer], nuisance_row
    )
    secondary_coordinate, secondary_prediction = readouts.secondary_u_l18[
        fold
    ].project(hidden_by_layer[SECONDARY_LAYER], nuisance_row)
    baseline_prediction = (
        readouts.nuisance_baseline[fold].predict(nuisance_row)
        if fold in readouts.nuisance_baseline
        else None
    )
    return {
        "primary_u": {
            "source_stage": 3,
            "estimand": RAW_ESTIMAND,
            "objective": PRIMARY_OBJECTIVE,
            "layer": primary.layer,
            "position": primary.position,
            "coordinate": primary_coordinate,
            "frozen_prediction": primary_prediction,
            "source_file": str(primary.source_file),
            "source_sha256": primary.source_sha256,
            "no_refit": True,
        },
        "secondary_u_l18": {
            "source_stage": 8,
            "estimand": RAW_ESTIMAND,
            "objective": PRIMARY_OBJECTIVE,
            "layer": SECONDARY_LAYER,
            "position": POST_ANSWER_POSITION,
            "coordinate": secondary_coordinate,
            "frozen_prediction": secondary_prediction,
            "source_file": str(readouts.secondary_u_l18[fold].audit["source_file"]),
            "source_sha256": str(
                readouts.secondary_u_l18[fold].audit["source_sha256"]
            ),
            "no_refit": True,
        },
        "nuisance_only": {
            "frozen_prediction": baseline_prediction,
            "available": baseline_prediction is not None,
            "stage09_outcomes_used_for_fit": False,
        },
    }


@dataclass
class AnswerOnlyReadoutMeasurement:
    answer_distribution: dict[str, Any]
    answer_star: str
    nuisance_row: dict[str, Any]
    readouts: dict[str, Any]
    hidden_by_layer: dict[int, np.ndarray]
    hidden_checksums: dict[str, Any]
    causal_prefix_audit: dict[str, Any]
    hook_audit: dict[str, Any]
    teacher_forced_messages_hash: str
    teacher_forced_rendered_sha256: str
    length_path_replay: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-safe metadata; hidden vectors remain separately available."""

        return {
            "answer_distribution": self.answer_distribution,
            "answer_star": self.answer_star,
            "nuisance_row": self.nuisance_row,
            "readouts": self.readouts,
            "hidden_checksums": self.hidden_checksums,
            "hidden_layers": sorted(self.hidden_by_layer),
            "causal_prefix_audit": self.causal_prefix_audit,
            "hook_audit": self.hook_audit,
            "teacher_forced_messages_hash": self.teacher_forced_messages_hash,
            "teacher_forced_rendered_sha256": self.teacher_forced_rendered_sha256,
            "length_path_replay": self.length_path_replay,
            "new_forward_count": 2,
        }


def measure_answer_only_readouts(
    runtime: Stage3Runtime,
    messages: Sequence[dict[str, Any]],
    *,
    answer_classes: Sequence[str],
    fixed_answer: str | None,
    base_row: Mapping[str, Any],
    readouts: ProspectiveReadoutRepository,
    assistant_text: str = ASSISTANT_ANSWER_PREFILL,
) -> AnswerOnlyReadoutMeasurement:
    """Measure answer logits, then fixed-answer U/U_L18 in two clean forwards.

    The second forward does not reconstruct the multimodal prefix.  It appends
    the canonical one-token ``A*`` directly to the already prepared first-pass
    tensors, which makes causal-prefix equality structural rather than an
    approximate BF16 hidden-state comparison.
    """

    copied = copy.deepcopy(list(messages))
    if contains_verbal_sa_request(copied):
        raise ValueError("Answer-only readout messages contain a verbal-SA request")
    rendered, inputs = runtime.generator.prepare_messages(
        copied, assistant_text=assistant_text
    )
    pre_ids = _input_tensor(inputs, "input_ids").clone()
    pre_mask = _input_tensor(inputs, "attention_mask").clone()
    nonlanguage = _nonlanguage_tensor_identity(inputs)
    pre_position = int(pre_ids.shape[1]) - 1
    token_ids = canonical_answer_token_ids(
        runtime.generator.tokenizer, answer_classes
    )
    pre_logits = run_logits_forward(
        runtime.model, inputs, [pre_position], runtime.modules
    )[pre_position]
    distribution = _distribution_from_vocab_logits(pre_logits, token_ids)
    distribution.update(
        {
            "messages_hash": canonical_message_hash(copied),
            "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "target_position": pre_position,
            "input_token_count": int(pre_ids.shape[1]),
            "verbal_sa_leakage": False,
        }
    )
    if fixed_answer is None:
        if distribution.get("unique_top1") is not True:
            raise ValueError(
                "No-History endpoint selection requires a unique restricted top-1"
            )
        answer = str(distribution["predicted_answer"])
    else:
        answer = str(normalize_answer(fixed_answer))
    if answer not in token_ids:
        raise ValueError(f"Fixed answer {answer!r} is outside the answer classes")
    answer_token = int(token_ids[answer])
    append_exact_token_ids(inputs, [answer_token])
    prefix_audit = audit_single_token_causal_prefix(
        pre_input_ids=pre_ids,
        pre_attention_mask=pre_mask,
        post_inputs=inputs,
        expected_token_id=answer_token,
        nonlanguage_before=nonlanguage,
    )
    fold = int(base_row["fold"])
    layers = readouts.required_answer_layers(fold)
    post_position = int(_input_tensor(inputs, "input_ids").shape[1]) - 1
    hidden, replay_logits, hook_audit = _capture_selected_post_answer_layers(
        runtime,
        inputs,
        post_answer_position=post_position,
        layers=layers,
        replay_logits_position=pre_position,
    )
    replay = _distribution_from_vocab_logits(replay_logits, token_ids)
    max_logit_error = max(
        abs(
            float(distribution["answer_class_logits"][label])
            - float(replay["answer_class_logits"][label])
        )
        for label in token_ids
    )
    probability_tv = 0.5 * sum(
        abs(
            float(distribution["answer_class_probabilities"][label])
            - float(replay["answer_class_probabilities"][label])
        )
        for label in token_ids
    )
    nuisance = nuisance_row_for_fixed_answer(
        base_row,
        fixed_answer=answer,
        answer_margin=float(distribution["top1_top2_logit_margin"]),
    )
    projections = project_answer_hidden(hidden, nuisance, readouts)
    teacher_messages = copy.deepcopy(copied)
    if not teacher_messages or teacher_messages[-1].get("role") != "assistant":
        raise ValueError("Answer-only messages must end in assistant continuation")
    teacher_messages[-1]["content"] = [
        {"type": "text", "text": f"{assistant_text} {answer}"}
    ]
    teacher_rendered = rendered + " " + answer
    result = AnswerOnlyReadoutMeasurement(
        answer_distribution=distribution,
        answer_star=answer,
        nuisance_row=nuisance,
        readouts=projections,
        hidden_by_layer=hidden,
        hidden_checksums=hidden_checksum_payload(hidden),
        causal_prefix_audit=prefix_audit,
        hook_audit=hook_audit,
        teacher_forced_messages_hash=canonical_message_hash(teacher_messages),
        teacher_forced_rendered_sha256=hashlib.sha256(
            teacher_rendered.encode("utf-8")
        ).hexdigest(),
        length_path_replay={
            "predicted_answer": replay["predicted_answer"],
            "max_restricted_logit_error": float(max_logit_error),
            "restricted_probability_tv": float(probability_tv),
            "equality_required": False,
            "note": (
                "The exact causal prefix is audited structurally; BF16 kernels may "
                "differ when total sequence length changes."
            ),
        },
    )
    del inputs, pre_logits, replay_logits
    return result


@dataclass
class JointCommon9Measurement:
    payload: dict[str, Any]
    hidden: np.ndarray


def measure_joint_common9(
    runtime: Stage3Runtime,
    messages: Sequence[dict[str, Any]],
    *,
    answer_star: str,
    fold: int,
    readouts: ProspectiveReadoutRepository,
    assistant_text: str | None = None,
) -> JointCommon9Measurement:
    """Measure frozen common-nine verbal ``V`` and internal attribution ``A``."""

    answer = str(normalize_answer(answer_star))
    continuation = assistant_text or (
        f"**Answer**: {answer}\n**Source Attribution**:"
    )
    copied = copy.deepcopy(list(messages))
    prepared = prepare_measurement(
        runtime.generator,
        copied,
        assistant_text=continuation,
        answer=answer,
    )
    protocol = panel_protocols()[0]
    if protocol.name != "common_9_ordered":
        raise RuntimeError("Frozen common-nine protocol order drifted")
    analyzer = ProtocolAnalyzer(runtime.generator.tokenizer, protocol.spec)
    direction: FrozenFoldDirection = readouts.attribution.get(int(fold))
    measured = runtime.measure(
        prepared,
        direction,  # runtime accepts the shared frozen-direction interface
        analyzer=analyzer,
    )
    hidden = np.asarray(measured.hidden, dtype=np.float32)
    if hidden.ndim != 1 or not np.isfinite(hidden).all():
        raise ValueError(f"Joint common-nine L18 hidden is invalid: {hidden.shape}")
    if measured.applied_count != 1 or measured.hook_call_count != 1:
        raise RuntimeError("Joint common-nine capture hook did not apply exactly once")
    if measured.injection_l2 != 0.0 or measured.expected_delta_z != 0.0:
        raise RuntimeError("Joint common-nine readout unexpectedly changed activation")
    source = measured.source
    checksum = hidden_checksum_payload({SECONDARY_LAYER: hidden})
    payload = {
        "protocol": protocol.name,
        "answer_star": answer,
        "semantic_imageward_score": float(source["soft_image_score"]),
        "hard_label": str(source["hard_label"]),
        "class_logits": source["class_logits"],
        "class_probabilities": source["class_probabilities"],
        "attribution_coordinate": float(direction.coordinate(hidden)),
        "attribution_prediction": float(direction.predict(hidden)),
        "prefix_hash": prepared.prefix_hash,
        "panl_position": int(prepared.panl_position),
        "target_position": int(prepared.target_position),
        "input_token_count": int(_input_tensor(prepared.inputs, "input_ids").shape[1]),
        "hidden_checksums": checksum,
        "hook_audit": {
            "hook_call_count": int(measured.hook_call_count),
            "hook_applied_count": int(measured.applied_count),
            "hook_exactly_once": (
                measured.applied_count == 1 and measured.hook_call_count == 1
            ),
            "steering_applied": False,
            "injection_l2": float(measured.injection_l2),
        },
        "source_direction": {
            "stage": "Stage10 byte-frozen under Stage06",
            "fold": int(fold),
            "rule_fingerprint": readouts.attribution.rule.get("rule_fingerprint"),
            "no_refit": True,
        },
        "new_forward_count": 1,
    }
    runtime.release_inputs(prepared)
    return JointCommon9Measurement(payload=payload, hidden=hidden)
