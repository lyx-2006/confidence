#!/usr/bin/env python3
"""Train pooled per-layer, per-position probes on an existing manifest."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import shutil
import tempfile
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

from layer_metacognition.hidden_state_store import atomic_write_json

from . import (
    C_GRID,
    DECISION_SIDE_LABELS,
    DECISION_SIDE_LOCATIONS,
    DEFAULT_PROBE_CONDITIONS,
    DEFAULT_PROBE_LOCATIONS,
    HIDDEN_STATE_DEFINITION,
    POSITION_NAMES,
    PROBE_CONDITIONS,
    VERSION_SETTINGS,
    build_probe_tasks,
    normalize_ordered_choices,
)
from .common import iter_jsonl, probe_output_dir
from .hidden_state_loader import HiddenStateLoader
from .probe_metrics import evaluate_required_subsets, majority_label, subset_membership
from .probe_models import (
    build_current_answer_baseline,
    build_hidden_state_probe,
    choose_regularization_C,
)
from .provenance import canonical_fingerprint, validate_manifest_provenance
from .split_utils import (
    load_or_create_answer_pair_split_assignments,
    load_or_create_split_assignments,
    permute_labels_by_unique_key,
    rows_for_answer_pair_outer_fold,
    rows_for_outer_fold,
)


def _trim_process_memory() -> None:
    """Return freed native arrays to the allocator in memory-constrained runs."""

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):  # pragma: no cover - non-glibc platforms
        pass


def filter_task_records(
    manifest: Sequence[dict[str, Any]],
    target_field: str,
    *,
    probe_conditions: Sequence[str] = DEFAULT_PROBE_CONDITIONS,
) -> list[dict[str, Any]]:
    selected = set(str(value) for value in probe_conditions)
    eligibility_fields = {
        "text_only_answer": "eligible_text_probe",
        "image_only_answer": "eligible_image_probe",
        "conflict_label": "eligible_conflict_probe",
        "decision_side": "eligible_decision_side_probe",
    }
    eligibility = eligibility_fields.get(target_field)
    if eligibility is None:
        raise ValueError(f"Unsupported target field: {target_field}")
    return [
        record
        for record in manifest
        if record.get("condition") in selected
        and record.get(eligibility)
        and isinstance(record.get(target_field), str)
    ]


def validate_outer_labels(
    train_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
    target_field: str,
) -> tuple[LabelEncoder | None, dict[str, Any] | None]:
    if not train_records:
        return None, {"type": "EmptyOuterTrain", "message": "No outer-train records"}
    if not test_records:
        return None, {"type": "EmptyOuterTest", "message": "No outer-test records"}
    train_labels = [str(record[target_field]) for record in train_records]
    test_labels = [str(record[target_field]) for record in test_records]
    if target_field == "decision_side":
        invalid = sorted(
            (set(train_labels) | set(test_labels)) - set(DECISION_SIDE_LABELS)
        )
        if invalid:
            return None, {
                "type": "InvalidDecisionSideClass",
                "message": "Decision-Side labels are outside the fixed class space",
                "invalid_classes": invalid,
            }
        encoder = LabelEncoder()
        encoder.classes_ = np.asarray(DECISION_SIDE_LABELS, dtype=object)
    else:
        encoder = LabelEncoder().fit(train_labels)
    if len(encoder.classes_) < 2:
        return None, {
            "type": "SingleClassOuterTrain",
            "message": f"Outer train has one class: {encoder.classes_[0]!r}",
        }
    unseen = sorted(set(test_labels) - set(str(value) for value in encoder.classes_))
    if unseen:
        return None, {
            "type": "UnseenOuterTestClasses",
            "message": "Test labels are absent from the outer-train label space",
            "unseen_classes": unseen,
        }
    return encoder, None


def _features(
    loader: HiddenStateLoader,
    records: Sequence[dict[str, Any]],
    *,
    layer: int,
    position: str,
) -> np.ndarray:
    vectors = [
        loader.load_vector(record, layer=layer, position_name=position)
        for record in records
    ]
    if not vectors:
        raise ValueError("Cannot construct features for an empty record set")
    matrix = np.stack(vectors, axis=0).astype(np.float32, copy=False)
    if matrix.ndim != 2:
        raise ValueError(f"Probe feature matrix must be rank 2, got {matrix.shape}")
    return matrix


class FeatureMatrixCache:
    """Cache one original-order matrix per position/layer for the current run."""

    def __init__(
        self,
        loader: HiddenStateLoader,
        manifest: Sequence[dict[str, Any]],
        *,
        experiment_fingerprint: str,
        manifest_fingerprint: str,
        max_matrices: int = 1,
    ):
        self.loader = loader
        self.manifest = list(manifest)
        self.experiment_fingerprint = experiment_fingerprint
        self.manifest_fingerprint = manifest_fingerprint
        if max_matrices < 1:
            raise ValueError("Feature matrix cache size must be positive")
        self.max_matrices = int(max_matrices)
        self._matrices: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._ordinal = {
            str(record["case_id"]): index for index, record in enumerate(self.manifest)
        }
        if len(self._ordinal) != len(self.manifest):
            raise ValueError("Probe manifest contains duplicate case_id values")
        self.feature_loading_seconds = 0.0
        self.matrix_load_count = 0

    def matrix(self, *, layer: int, position: str) -> np.ndarray:
        key = (
            self.experiment_fingerprint,
            self.manifest_fingerprint,
            str(position),
            int(layer),
        )
        if key not in self._matrices:
            while len(self._matrices) >= self.max_matrices:
                _key, evicted = self._matrices.popitem(last=False)
                del evicted
                _trim_process_memory()
            started = time.perf_counter()
            self._matrices[key] = _features(
                self.loader,
                self.manifest,
                layer=layer,
                position=position,
            )
            self.feature_loading_seconds += time.perf_counter() - started
            self.matrix_load_count += 1
        else:
            self._matrices.move_to_end(key)
        return self._matrices[key]

    def rows(
        self,
        records: Sequence[dict[str, Any]],
        *,
        layer: int,
        position: str,
    ) -> np.ndarray:
        indices = [self._ordinal[str(record["case_id"])] for record in records]
        return self.matrix(layer=layer, position=position)[indices]


def _align_probabilities(
    probabilities: np.ndarray,
    classifier_classes: Sequence[int],
    class_count: int,
) -> np.ndarray:
    output = np.zeros((probabilities.shape[0], class_count), dtype=np.float64)
    for local_index, encoded_class in enumerate(classifier_classes):
        output[:, int(encoded_class)] = probabilities[:, local_index]
    row_sums = output.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-7):
        raise ValueError("Classifier probabilities do not sum to one after alignment")
    return output


def _prediction_record(
    record: dict[str, Any],
    *,
    fold: int,
    task: str,
    position: str,
    layer: int | None,
    setting: str,
    train_version: str,
    test_version: str,
    model_type: str,
    backend: str,
    target_field: str,
    predicted_label: str,
    probabilities: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    raw_field = f"{target_field}_raw"
    memberships = subset_membership(record)
    output = {
        "case_id": record["case_id"],
        "item_id": record["item_id"],
        "prior_index": record["prior_index"],
        "condition": record["condition"],
        "version": record["version"],
        "fold": fold,
        "task": task,
        "position": position,
        "layer": layer,
        "version_setting": setting,
        "train_version": train_version,
        "test_version": test_version,
        "model_type": model_type,
        "backend": backend,
        "true_label": record[target_field],
        "true_label_raw": record.get(raw_field),
        "predicted_label": predicted_label,
        "class_probabilities": {
            str(label): float(probabilities[index])
            for index, label in enumerate(classes)
        },
        "subsets": [name for name, included in memberships.items() if included],
    }
    if target_field == "decision_side":
        label_to_id = {label: index for index, label in enumerate(classes)}
        output.update(
            {
                "true_label_id": int(label_to_id[str(record[target_field])]),
                "predicted_label_id": int(label_to_id[str(predicted_label)]),
                "P(follows_text)": float(probabilities[0]),
                "P(follows_image)": float(probabilities[1]),
            }
        )
    return output


def _decision_direction_arrays(model: Any, backend: str) -> dict[str, np.ndarray]:
    if backend == "torch":
        scaler = model.scaler
        weight = np.asarray(model.weight, dtype=np.float64).reshape(-1)
        intercept = float(np.asarray(model.intercept, dtype=np.float64).reshape(-1)[0])
    else:
        scaler = model.named_steps["scaler"]
        classifier = model.named_steps["classifier"]
        weight = np.asarray(classifier.coef_, dtype=np.float64).reshape(-1)
        intercept = float(np.asarray(classifier.intercept_, dtype=np.float64)[0])
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    d_raw = weight / scale
    norm = float(np.linalg.norm(d_raw))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Decision-Side direction has zero or non-finite norm")
    d_K = d_raw / norm
    raw_intercept = intercept - float(np.dot(d_raw, mean))
    if float(np.dot(d_raw, d_K)) <= 0:
        raise AssertionError("+d_K does not point toward follows_image")
    return {
        "scaler_mean": mean,
        "scaler_scale": scale,
        "weight": weight,
        "intercept": np.asarray([intercept], dtype=np.float64),
        "d_raw": d_raw,
        "d_K": d_K,
        "raw_intercept": np.asarray([raw_intercept], dtype=np.float64),
    }


def _fit_diagnostics(model: Any, backend: str) -> dict[str, Any] | None:
    if backend == "torch":
        return dict(model.diagnostics)
    classifier = model.named_steps["classifier"]
    iterations = int(np.max(np.asarray(classifier.n_iter_, dtype=np.int64)))
    max_iterations = int(classifier.max_iter)
    return {
        "backend": "sklearn",
        "iterations": iterations,
        "max_iterations": max_iterations,
        "converged": iterations < max_iterations,
    }


def _invalid_result(
    *,
    task: str,
    position: str,
    target_field: str,
    layer: int | None,
    fold: int,
    setting: str,
    train_version: str,
    test_version: str,
    model_type: str,
    backend: str,
    train_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
    reason: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "invalid",
        "invalid_reason": reason,
        "task": task,
        "position": position,
        "target_field": target_field,
        "layer": layer,
        "fold": fold,
        "version_setting": setting,
        "train_version": train_version,
        "test_version": test_version,
        "model_type": model_type,
        "backend": backend,
        "train_sample_count": len(train_records),
        "test_sample_count": len(test_records),
        "train_item_count": len({str(record["item_id"]) for record in train_records}),
        "test_item_count": len({str(record["item_id"]) for record in test_records}),
        "classes": [],
        "selected_C": None,
        "c_selection": None,
        "fit_diagnostics": diagnostics,
        "subset_metrics": {},
    }


def _model_classifier_classes(model: Any, backend: str) -> Sequence[int]:
    if backend == "torch":
        return model.classes_
    return model.named_steps["classifier"].classes_


def _predict_model(
    model: Any,
    X: np.ndarray,
    *,
    backend: str,
    encoder: LabelEncoder,
) -> tuple[list[str], np.ndarray]:
    predicted_encoded = np.asarray(model.predict(X), dtype=np.int64)
    probabilities = _align_probabilities(
        np.asarray(model.predict_proba(X), dtype=np.float64),
        _model_classifier_classes(model, backend),
        len(encoder.classes_),
    )
    predicted = [
        str(value) for value in encoder.inverse_transform(predicted_encoded)
    ]
    return predicted, probabilities


def _fit_baseline_model(
    train_records: Sequence[dict[str, Any]],
    target_field: str,
    encoder: LabelEncoder,
) -> Any:
    train_X = np.asarray(
        [[str(record["current_answer"])] for record in train_records], dtype=object
    )
    encoded_train = encoder.transform(
        [str(record[target_field]) for record in train_records]
    )
    return build_current_answer_baseline().fit(train_X, encoded_train)


def _predict_baseline_model(
    model: Any,
    test_records: Sequence[dict[str, Any]],
    encoder: LabelEncoder,
) -> tuple[list[str], np.ndarray]:
    test_X = np.asarray(
        [[str(record["current_answer"])] for record in test_records], dtype=object
    )
    predicted_encoded = np.asarray(model.predict(test_X), dtype=np.int64)
    classifier = model.named_steps["classifier"]
    probabilities = _align_probabilities(
        model.predict_proba(test_X), classifier.classes_, len(encoder.classes_)
    )
    return [
        str(value) for value in encoder.inverse_transform(predicted_encoded)
    ], probabilities


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--manifest-path")
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--probe-conditions",
        nargs="+",
        choices=list(PROBE_CONDITIONS),
        default=list(DEFAULT_PROBE_CONDITIONS),
    )
    parser.add_argument(
        "--answer-probe-location",
        nargs="+",
        choices=list(POSITION_NAMES),
        default=list(DEFAULT_PROBE_LOCATIONS),
    )
    parser.add_argument(
        "--conflict-probe-location",
        nargs="+",
        choices=list(POSITION_NAMES),
        default=list(DEFAULT_PROBE_LOCATIONS),
    )
    parser.add_argument(
        "--decision-side-probe-location",
        nargs="+",
        choices=list(DECISION_SIDE_LOCATIONS),
    )
    parser.add_argument(
        "--split-mode", choices=("item", "answer_pair"), default="item"
    )
    parser.add_argument(
        "--decision-side-only",
        action="store_true",
        help=(
            "Train only requested Decision-Side tasks. This is optional for item "
            "splits and implicit for answer_pair splits."
        ),
    )
    parser.add_argument(
        "--version-settings",
        nargs="+",
        choices=list(VERSION_SETTINGS),
        default=list(VERSION_SETTINGS),
    )
    parser.add_argument("--backend", choices=("sklearn", "torch"), default="sklearn")
    parser.add_argument("--fixed-c", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--shard-cache-size", type=int, default=2)
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[list[int], tuple[str, ...]]:
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")
    if args.permutations < 0:
        raise ValueError("--permutations must be non-negative")
    if args.shard_cache_size < 1:
        raise ValueError("--shard-cache-size must be positive")
    if args.split_mode == "answer_pair" and not args.decision_side_probe_location:
        raise ValueError(
            "--split-mode answer_pair requires --decision-side-probe-location"
        )
    if args.decision_side_only and not args.decision_side_probe_location:
        raise ValueError(
            "--decision-side-only requires --decision-side-probe-location"
        )
    if args.fixed_c is not None and (
        not np.isfinite(args.fixed_c) or args.fixed_c <= 0
    ):
        raise ValueError("--fixed-c must be a finite positive value")
    if args.backend == "torch":
        if args.fixed_c is None:
            raise ValueError("--backend torch requires --fixed-c")
        if args.permutations != 0:
            raise ValueError("--backend torch requires --permutations 0")
        from .torch_logistic_probe import resolve_torch_device

        args.resolved_device = resolve_torch_device(args.device)
        import torch

        args.torch_version = torch.__version__
        args.torch_cuda_version = torch.version.cuda
    elif args.device != "auto":
        raise ValueError("--device is only valid with --backend torch")
    else:
        args.resolved_device = "cpu"
        args.torch_version = None
        args.torch_cuda_version = None
    layers = [int(value) for value in args.layers]
    if not layers or len(layers) != len(set(layers)) or any(value < 0 for value in layers):
        raise ValueError("--layers must contain distinct non-negative layer indices")
    selected_settings = tuple(
        name for name in VERSION_SETTINGS if name in set(args.version_settings)
    )
    if not selected_settings:
        raise ValueError("--version-settings must not be empty")
    return sorted(layers), selected_settings


def _protected_output_paths(output_dir: Path) -> tuple[Path, ...]:
    return (
        output_dir / "run_config.json",
        output_dir / "split_assignments.json",
        output_dir / "decision_side_pair_split_assignments.json",
        output_dir / "layer_probe_metrics.json",
        output_dir / "layer_probe_predictions.jsonl",
        output_dir / "decision_directions",
    )


def _prepare_output(
    output_dir: Path,
    run_config: dict[str, Any],
    *,
    resume: bool,
) -> bool:
    """Protect prior outputs. Return True for an already-complete no-op."""

    config_path = output_dir / "run_config.json"
    metrics_path = output_dir / "layer_probe_metrics.json"
    predictions_path = output_dir / "layer_probe_predictions.jsonl"
    existing_protected = [path for path in _protected_output_paths(output_dir) if path.exists()]
    if not config_path.exists():
        if existing_protected:
            raise FileExistsError(
                "Probe output directory contains protected artifacts without a run "
                f"configuration: {[str(path) for path in existing_protected]}"
            )
        return False
    if not resume:
        raise FileExistsError(
            f"Probe run already exists in {output_dir}; pass --resume only for an "
            "identical configuration"
        )
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    if existing.get("config_fingerprint") != run_config["config_fingerprint"]:
        raise ValueError(
            "Cannot resume Probe run because the immutable configuration differs: "
            f"existing={existing.get('config_fingerprint')} "
            f"requested={run_config['config_fingerprint']}"
        )
    status = existing.get("status")
    if status == "complete":
        if not metrics_path.is_file() or not predictions_path.is_file():
            raise ValueError("Completed Probe run is missing metrics or predictions")
        return True
    if metrics_path.exists() or predictions_path.exists():
        raise FileExistsError(
            "Interrupted Probe run has partial official metrics/predictions; refusing "
            "to overwrite them"
        )
    return False


def _fit_count_from_c_selection(detail: dict[str, Any]) -> int:
    if detail.get("status") != "selected":
        return 0
    scores = detail.get("scores") or {}
    return sum(len(values) for values in scores.values())


def main(argv: list[str] | None = None) -> int:
    total_started = time.perf_counter()
    args = _parser().parse_args(argv)
    layers, selected_setting_names = _validate_args(args)
    probe_conditions = normalize_ordered_choices(
        args.probe_conditions, PROBE_CONDITIONS, "--probe-conditions"
    )
    answer_locations = normalize_ordered_choices(
        args.answer_probe_location, POSITION_NAMES, "--answer-probe-location"
    )
    conflict_locations = normalize_ordered_choices(
        args.conflict_probe_location, POSITION_NAMES, "--conflict-probe-location"
    )
    decision_locations = normalize_ordered_choices(
        args.decision_side_probe_location or (),
        DECISION_SIDE_LOCATIONS,
        "--decision-side-probe-location",
    )
    if args.split_mode == "answer_pair" or args.decision_side_only:
        probe_tasks = build_probe_tasks((), (), decision_locations)
    else:
        probe_tasks = build_probe_tasks(
            answer_locations, conflict_locations, decision_locations
        )
    selected_settings = {
        name: VERSION_SETTINGS[name] for name in selected_setting_names
    }

    experiment_dir = Path(args.experiment_dir).resolve()
    output_dir = probe_output_dir(experiment_dir, args.output_dir)
    manifest_path = (
        Path(args.manifest_path).resolve()
        if args.manifest_path
        else output_dir / "probe_manifest.jsonl"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Probe manifest does not exist: {manifest_path}; run build_probe_manifest first"
        )
    manifest = list(iter_jsonl(manifest_path))
    if not manifest:
        raise ValueError("Probe manifest is empty")
    requested_positions = tuple(
        position for position in POSITION_NAMES
        if position
        in set(answer_locations).union(conflict_locations).union(decision_locations)
    )
    requested_versions = tuple(
        version
        for version in ("v3", "v4")
        if any(version in pair for pair in selected_settings.values())
    )
    provenance = validate_manifest_provenance(
        experiment_dir,
        manifest_path,
        manifest,
        selected_conditions=probe_conditions,
        requested_layers=layers,
        requested_positions=requested_positions,
        requested_versions=requested_versions,
    )
    selected_manifest = [
        record for record in manifest if record.get("condition") in probe_conditions
    ]
    permutation_seeds = [args.seed + index for index in range(args.permutations)]
    immutable_config = {
        "format_version": 2,
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "probe_tasks": {
            key: {"position": value[0], "target_field": value[1]}
            for key, value in probe_tasks.items()
        },
        "version_settings": {
            key: {"train_version": value[0], "test_version": value[1]}
            for key, value in selected_settings.items()
        },
        "layers": layers,
        "n_splits": args.n_splits,
        "inner_n_splits": 3,
        "seed": args.seed,
        "probe_conditions": list(probe_conditions),
        "answer_probe_locations": list(answer_locations),
        "conflict_probe_locations": list(conflict_locations),
        "decision_side_probe_locations": list(decision_locations),
        "decision_side_label_mapping": {
            label: index for index, label in enumerate(DECISION_SIDE_LABELS)
        },
        "split_mode": args.split_mode,
        "decision_side_only": bool(
            args.decision_side_only or args.split_mode == "answer_pair"
        ),
        "backend": args.backend,
        "fixed_C": args.fixed_c,
        "device": args.resolved_device,
        "requested_device": args.device,
        "C_grid": list(C_GRID),
        "C_selection_metric": "balanced_accuracy",
        "C_tie_break": "smaller_C",
        "permutation_count": args.permutations,
        "permutation_seeds": permutation_seeds,
        "permutation_C_policy": "reuse_hidden_state_probe_selected_C",
        "shard_cache_size": args.shard_cache_size,
        "feature_matrix_cache_size": 1,
        "hidden_state_definition": HIDDEN_STATE_DEFINITION,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "torch_version": args.torch_version,
        "torch_cuda_version": args.torch_cuda_version,
        **provenance,
    }
    config_fingerprint = canonical_fingerprint(immutable_config)
    run_config = {
        **immutable_config,
        "config_fingerprint": config_fingerprint,
        "status": "running",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if _prepare_output(output_dir, run_config, resume=args.resume):
        print(
            json.dumps(
                {"status": "complete", "resumed": True, "output_dir": str(output_dir)},
                ensure_ascii=False,
            )
        )
        return 0
    atomic_write_json(output_dir / "run_config.json", run_config)
    item_to_fold: dict[str, int] | None = None
    pair_to_fold: dict[str, int] | None = None
    if args.split_mode == "item":
        assignment = load_or_create_split_assignments(
            output_dir / "split_assignments.json",
            selected_manifest,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        item_to_fold = {
            str(key): int(value) for key, value in assignment["item_to_fold"].items()
        }
    else:
        decision_records = filter_task_records(
            manifest, "decision_side", probe_conditions=probe_conditions
        )
        assignment = load_or_create_answer_pair_split_assignments(
            output_dir / "decision_side_pair_split_assignments.json",
            decision_records,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        pair_to_fold = {
            str(key): int(value) for key, value in assignment["pair_to_fold"].items()
        }

    loader = HiddenStateLoader(experiment_dir, cache_size=args.shard_cache_size)
    feature_case_ids = {
        str(record["case_id"])
        for _task, (_position, target) in probe_tasks.items()
        for record in filter_task_records(
            manifest, target, probe_conditions=probe_conditions
        )
    }
    feature_manifest = [
        record for record in manifest if str(record["case_id"]) in feature_case_ids
    ]
    feature_cache = FeatureMatrixCache(
        loader,
        feature_manifest,
        experiment_fingerprint=provenance["hidden_state_index_fingerprint"],
        manifest_fingerprint=provenance["manifest_fingerprint"],
        max_matrices=1,
    )
    model_cache: dict[tuple[Any, ...], Any] = {}
    model_error_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    c_cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
    encoder_cache: dict[tuple[Any, ...], LabelEncoder] = {}
    fold_results: list[dict[str, Any]] = []
    fit_counts: Counter[str] = Counter()
    fit_seconds = 0.0
    evaluation_seconds = 0.0
    gpu_transfer_seconds = 0.0
    torch_iterations: list[int] = []
    non_converged_count = 0
    direction_index: list[dict[str, Any]] = []
    direction_temp_dir = Path(
        tempfile.mkdtemp(prefix=f".decision_directions.{os.getpid()}.", dir=output_dir)
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".layer_probe_predictions.{os.getpid()}.",
        suffix=".jsonl.tmp",
        dir=output_dir,
    )

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as prediction_handle:
            for task, (position, target_field) in probe_tasks.items():
                task_records = filter_task_records(
                    manifest, target_field, probe_conditions=probe_conditions
                )
                for setting, (train_version, test_version) in selected_settings.items():
                    for fold in range(args.n_splits):
                        split_audit = None
                        if args.split_mode == "answer_pair":
                            assert pair_to_fold is not None
                            train_records, test_records, split_audit = (
                                rows_for_answer_pair_outer_fold(
                                    task_records,
                                    pair_to_fold,
                                    fold=fold,
                                    train_version=train_version,
                                    test_version=test_version,
                                )
                            )
                        else:
                            assert item_to_fold is not None
                            train_records, test_records = rows_for_outer_fold(
                                task_records,
                                item_to_fold,
                                fold=fold,
                                train_version=train_version,
                                test_version=test_version,
                            )
                        encoder, invalid_reason = validate_outer_labels(
                            train_records, test_records, target_field
                        )
                        if invalid_reason is not None:
                            if position == "panl" and target_field != "conflict_label":
                                fold_results.append(
                                    _invalid_result(
                                        task=task,
                                        position=position,
                                        target_field=target_field,
                                        layer=None,
                                        fold=fold,
                                        setting=setting,
                                        train_version=train_version,
                                        test_version=test_version,
                                        model_type="current_answer_only_baseline",
                                        backend="sklearn",
                                        train_records=train_records,
                                        test_records=test_records,
                                        reason=invalid_reason,
                                    )
                                )
                            for layer in layers:
                                fold_results.append(
                                    _invalid_result(
                                        task=task,
                                        position=position,
                                        target_field=target_field,
                                        layer=layer,
                                        fold=fold,
                                        setting=setting,
                                        train_version=train_version,
                                        test_version=test_version,
                                        model_type="hidden_state_probe",
                                        backend=args.backend,
                                        train_records=train_records,
                                        test_records=test_records,
                                        reason=invalid_reason,
                                    )
                                )
                            continue

                        assert encoder is not None
                        train_context = (
                            task,
                            target_field,
                            position,
                            fold,
                            train_version,
                            args.seed,
                            tuple(probe_conditions),
                            provenance["hidden_state_index_fingerprint"],
                            provenance["manifest_fingerprint"],
                        )
                        cached_encoder = encoder_cache.setdefault(train_context, encoder)
                        if not np.array_equal(cached_encoder.classes_, encoder.classes_):
                            raise AssertionError("Cached LabelEncoder class order changed")
                        encoder = cached_encoder
                        classes = [str(value) for value in encoder.classes_]
                        train_labels = [str(record[target_field]) for record in train_records]
                        test_labels = [str(record[target_field]) for record in test_records]
                        majority = majority_label(train_labels)

                        if position == "panl" and target_field in {
                            "text_only_answer",
                            "image_only_answer",
                        }:
                            baseline_key = (
                                "current_answer_only_baseline",
                                *train_context,
                                None,
                                "sklearn",
                                1.0,
                                None,
                            )
                            if baseline_key not in model_cache:
                                started = time.perf_counter()
                                model_cache[baseline_key] = _fit_baseline_model(
                                    train_records, target_field, encoder
                                )
                                fit_seconds += time.perf_counter() - started
                                fit_counts["current_answer_only_baseline"] += 1
                            evaluation_started = time.perf_counter()
                            baseline_predicted, baseline_probabilities = (
                                _predict_baseline_model(
                                    model_cache[baseline_key], test_records, encoder
                                )
                            )
                            baseline_metrics = evaluate_required_subsets(
                                test_records,
                                test_labels,
                                baseline_predicted,
                                baseline_probabilities,
                                classes=classes,
                                majority_class=majority,
                                selected_C=1.0,
                            )
                            evaluation_seconds += time.perf_counter() - evaluation_started
                            fold_results.append(
                                {
                                    "status": "valid",
                                    "invalid_reason": None,
                                    "task": task,
                                    "position": position,
                                    "target_field": target_field,
                                    "layer": None,
                                    "fold": fold,
                                    "version_setting": setting,
                                    "train_version": train_version,
                                    "test_version": test_version,
                                    "model_type": "current_answer_only_baseline",
                                    "backend": "sklearn",
                                    "train_sample_count": len(train_records),
                                    "test_sample_count": len(test_records),
                                    "train_item_count": len(
                                        {str(record["item_id"]) for record in train_records}
                                    ),
                                    "test_item_count": len(
                                        {str(record["item_id"]) for record in test_records}
                                    ),
                                    "classes": classes,
                                    "selected_C": 1.0,
                                    "c_selection": {
                                        "status": "fixed",
                                        "reason": "specified_current_answer_baseline",
                                    },
                                    "fit_diagnostics": None,
                                    "subset_metrics": baseline_metrics,
                                }
                            )
                            for index, record in enumerate(test_records):
                                prediction_handle.write(
                                    json.dumps(
                                        _prediction_record(
                                            record,
                                            fold=fold,
                                            task=task,
                                            position=position,
                                            layer=None,
                                            setting=setting,
                                            train_version=train_version,
                                            test_version=test_version,
                                            model_type="current_answer_only_baseline",
                                            backend="sklearn",
                                            target_field=target_field,
                                            predicted_label=baseline_predicted[index],
                                            probabilities=baseline_probabilities[index],
                                            classes=classes,
                                        ),
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    )
                                    + "\n"
                                )
                        for layer in layers:
                            train_X = feature_cache.rows(
                                train_records, layer=layer, position=position
                            )
                            test_X = feature_cache.rows(
                                test_records, layer=layer, position=position
                            )
                            c_key = (*train_context, int(layer), args.backend)
                            if c_key not in c_cache:
                                if args.fixed_c is not None:
                                    c_cache[c_key] = (
                                        float(args.fixed_c),
                                        {
                                            "status": "fixed",
                                            "reason": "specified_by_fixed_c",
                                        },
                                    )
                                else:
                                    started = time.perf_counter()
                                    c_cache[c_key] = choose_regularization_C(
                                        train_X,
                                        train_labels,
                                        [str(record["item_id"]) for record in train_records],
                                        seed=args.seed,
                                    )
                                    fit_seconds += time.perf_counter() - started
                                    inner_fit_count = _fit_count_from_c_selection(
                                        c_cache[c_key][1]
                                    )
                                    if inner_fit_count:
                                        fit_counts["inner_cv"] += inner_fit_count
                            selected_C, c_selection = c_cache[c_key]
                            model_key = (
                                "hidden_state_probe",
                                task,
                                target_field,
                                position,
                                int(layer),
                                fold,
                                train_version,
                                args.backend,
                                args.resolved_device,
                                float(selected_C),
                                args.seed,
                                tuple(probe_conditions),
                                None,
                                provenance["hidden_state_index_fingerprint"],
                                provenance["manifest_fingerprint"],
                            )
                            if model_key not in model_cache and model_key not in model_error_cache:
                                encoded_train = encoder.transform(train_labels)
                                print(
                                    json.dumps(
                                        {
                                            "event": "probe_fit_started",
                                            "task": task,
                                            "position": position,
                                            "layer": int(layer),
                                            "fold": int(fold),
                                            "version_setting": setting,
                                            "backend": args.backend,
                                            "split_mode": args.split_mode,
                                        },
                                        ensure_ascii=False,
                                    ),
                                    flush=True,
                                )
                                started = time.perf_counter()
                                try:
                                    if args.backend == "torch":
                                        from .torch_logistic_probe import (
                                            TorchProbeNumericalError,
                                            fit_torch_logistic_probe,
                                        )

                                        model = fit_torch_logistic_probe(
                                            train_X,
                                            encoded_train,
                                            C=selected_C,
                                            device=args.resolved_device,
                                            seed=args.seed,
                                            binary_single_logit=(
                                            target_field == "conflict_label"
                                                or target_field == "decision_side"
                                            ),
                                        )
                                    else:
                                        model = build_hidden_state_probe(selected_C)
                                        model.fit(train_X, encoded_train)
                                    model_cache[model_key] = model
                                except Exception as exc:
                                    numerical = (
                                        args.backend == "torch"
                                        and exc.__class__.__name__ == "TorchProbeNumericalError"
                                    )
                                    if not numerical:
                                        raise
                                    model_error_cache[model_key] = {
                                        "type": exc.__class__.__name__,
                                        "message": str(exc),
                                    }
                                finally:
                                    elapsed = time.perf_counter() - started
                                    fit_counts["hidden_state_probe"] += 1
                                    if args.backend != "torch" or model_key in model_error_cache:
                                        fit_seconds += elapsed
                                if model_key in model_cache and args.backend == "torch":
                                    diagnostics = model_cache[model_key].diagnostics
                                    fit_seconds += float(
                                        diagnostics.get("preprocessing_seconds", 0.0)
                                    ) + float(diagnostics.get("fit_seconds", 0.0))
                                    gpu_transfer_seconds += float(
                                        diagnostics.get("gpu_transfer_seconds", 0.0)
                                    )
                                    torch_iterations.append(
                                        int(diagnostics.get("iterations", 0))
                                    )
                                    if not diagnostics.get("converged", False):
                                        non_converged_count += 1
                                print(
                                    json.dumps(
                                        {
                                            "event": "probe_fit_finished",
                                            "task": task,
                                            "position": position,
                                            "layer": int(layer),
                                            "fold": int(fold),
                                            "version_setting": setting,
                                            "status": (
                                                "invalid"
                                                if model_key in model_error_cache
                                                else "valid"
                                            ),
                                        },
                                        ensure_ascii=False,
                                    ),
                                    flush=True,
                                )

                            if model_key in model_error_cache:
                                fold_results.append(
                                    _invalid_result(
                                        task=task,
                                        position=position,
                                        target_field=target_field,
                                        layer=layer,
                                        fold=fold,
                                        setting=setting,
                                        train_version=train_version,
                                        test_version=test_version,
                                        model_type="hidden_state_probe",
                                        backend=args.backend,
                                        train_records=train_records,
                                        test_records=test_records,
                                        reason=model_error_cache[model_key],
                                    )
                                )
                                continue
                            model = model_cache[model_key]
                            if target_field == "decision_side":
                                arrays = _decision_direction_arrays(model, args.backend)
                                direction_name = (
                                    f"{setting}__fold_{fold}__{position}__layer_{layer}.npz"
                                )
                                np.savez_compressed(
                                    direction_temp_dir / direction_name, **arrays
                                )
                                direction_index.append(
                                    {
                                        "file": direction_name,
                                        "task": task,
                                        "position": position,
                                        "layer": int(layer),
                                        "fold": int(fold),
                                        "version_setting": setting,
                                        "backend": args.backend,
                                        "class0": DECISION_SIDE_LABELS[0],
                                        "class1": DECISION_SIDE_LABELS[1],
                                        "positive_direction": "+d_K -> follows_image",
                                    }
                                )
                            fit_diagnostics = _fit_diagnostics(model, args.backend)
                            try:
                                evaluation_started = time.perf_counter()
                                predicted, probabilities = _predict_model(
                                    model, test_X, backend=args.backend, encoder=encoder
                                )
                                evaluation_seconds += time.perf_counter() - evaluation_started
                            except Exception as exc:
                                if args.backend != "torch" or exc.__class__.__name__ != "TorchProbeNumericalError":
                                    raise
                                fold_results.append(
                                    _invalid_result(
                                        task=task,
                                        position=position,
                                        target_field=target_field,
                                        layer=layer,
                                        fold=fold,
                                        setting=setting,
                                        train_version=train_version,
                                        test_version=test_version,
                                        model_type="hidden_state_probe",
                                        backend=args.backend,
                                        train_records=train_records,
                                        test_records=test_records,
                                        reason={"type": exc.__class__.__name__, "message": str(exc)},
                                        diagnostics=fit_diagnostics,
                                    )
                                )
                                evaluation_seconds += time.perf_counter() - evaluation_started
                                continue
                            permuted_predictions: list[list[str]] = []
                            for permutation_seed in permutation_seeds:
                                permutation_key = (
                                    "permuted_hidden_state_probe",
                                    task,
                                    target_field,
                                    position,
                                    int(layer),
                                    fold,
                                    train_version,
                                    "sklearn",
                                    float(selected_C),
                                    args.seed,
                                    tuple(probe_conditions),
                                    int(permutation_seed),
                                    provenance["hidden_state_index_fingerprint"],
                                    provenance["manifest_fingerprint"],
                                )
                                if permutation_key not in model_cache:
                                    permuted_labels = permute_labels_by_unique_key(
                                        train_records,
                                        target_field,
                                        seed=permutation_seed,
                                    )
                                    encoded_permuted = encoder.transform(permuted_labels)
                                    fit_started = time.perf_counter()
                                    permuted_model = build_hidden_state_probe(selected_C)
                                    permuted_model.fit(train_X, encoded_permuted)
                                    fit_seconds += time.perf_counter() - fit_started
                                    fit_counts["permuted_hidden_state_probe"] += 1
                                    model_cache[permutation_key] = permuted_model
                                evaluation_started = time.perf_counter()
                                permuted_predicted, _ = _predict_model(
                                    model_cache[permutation_key],
                                    test_X,
                                    backend="sklearn",
                                    encoder=encoder,
                                )
                                evaluation_seconds += time.perf_counter() - evaluation_started
                                permuted_predictions.append(permuted_predicted)
                            evaluation_started = time.perf_counter()
                            subset_metrics = evaluate_required_subsets(
                                test_records,
                                test_labels,
                                predicted,
                                probabilities,
                                classes=classes,
                                majority_class=majority,
                                selected_C=selected_C,
                                permuted_predictions=permuted_predictions,
                            )
                            evaluation_seconds += time.perf_counter() - evaluation_started
                            fold_results.append(
                                {
                                    "status": "valid",
                                    "invalid_reason": None,
                                    "task": task,
                                    "position": position,
                                    "target_field": target_field,
                                    "layer": layer,
                                    "fold": fold,
                                    "version_setting": setting,
                                    "train_version": train_version,
                                    "test_version": test_version,
                                    "model_type": "hidden_state_probe",
                                    "backend": args.backend,
                                    "train_sample_count": len(train_records),
                                    "test_sample_count": len(test_records),
                                    "train_item_count": len(
                                        {str(record["item_id"]) for record in train_records}
                                    ),
                                    "test_item_count": len(
                                        {str(record["item_id"]) for record in test_records}
                                    ),
                                    "classes": classes,
                                    "selected_C": selected_C,
                                    "c_selection": c_selection,
                                    "fit_diagnostics": fit_diagnostics,
                                    "split_audit": split_audit,
                                    "train_class_counts": dict(
                                        sorted(Counter(train_labels).items())
                                    ),
                                    "test_class_counts": dict(
                                        sorted(Counter(test_labels).items())
                                    ),
                                    "subset_metrics": subset_metrics,
                                }
                            )
                            for index, record in enumerate(test_records):
                                prediction_handle.write(
                                    json.dumps(
                                        _prediction_record(
                                            record,
                                            fold=fold,
                                            task=task,
                                            position=position,
                                            layer=layer,
                                            setting=setting,
                                            train_version=train_version,
                                            test_version=test_version,
                                            model_type="hidden_state_probe",
                                            backend=args.backend,
                                            target_field=target_field,
                                            predicted_label=predicted[index],
                                            probabilities=probabilities[index],
                                            classes=classes,
                                        ),
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    )
                                    + "\n"
                                )
                            del train_X, test_X
                            _trim_process_memory()
            prediction_handle.flush()
            os.fsync(prediction_handle.fileno())
        predictions_path = output_dir / "layer_probe_predictions.jsonl"
        if predictions_path.exists():
            raise FileExistsError(f"Refusing to overwrite predictions: {predictions_path}")
        os.replace(temporary_name, predictions_path)
        if direction_index:
            atomic_write_json(
                direction_temp_dir / "index.json",
                {
                    "format_version": 1,
                    "split_mode": args.split_mode,
                    "class_mapping": {
                        label: index
                        for index, label in enumerate(DECISION_SIDE_LABELS)
                    },
                    "direction_count": len(direction_index),
                    "directions": direction_index,
                },
            )
            direction_dir = output_dir / "decision_directions"
            if direction_dir.exists():
                raise FileExistsError(
                    f"Refusing to overwrite directions: {direction_dir}"
                )
            os.replace(direction_temp_dir, direction_dir)
        else:
            shutil.rmtree(direction_temp_dir)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        if direction_temp_dir.exists():
            shutil.rmtree(direction_temp_dir)
        run_config["status"] = "failed"
        atomic_write_json(output_dir / "run_config.json", run_config)
        raise

    timing = {
        "feature_loading_seconds": float(feature_cache.feature_loading_seconds),
        "gpu_transfer_seconds": float(gpu_transfer_seconds),
        "fit_seconds": float(fit_seconds),
        "evaluation_seconds": float(evaluation_seconds),
        "total_seconds": float(time.perf_counter() - total_started),
        "fit_count": int(sum(fit_counts.values())),
        "fit_count_by_model_type": dict(sorted(fit_counts.items())),
        "mean_iterations": (
            float(np.mean(torch_iterations)) if torch_iterations else None
        ),
        "p95_iterations": (
            float(np.percentile(torch_iterations, 95)) if torch_iterations else None
        ),
        "non_converged_count": int(non_converged_count),
        "feature_matrix_load_count": feature_cache.matrix_load_count,
        "model_cache_entry_count": len(model_cache),
    }
    metric_payload = {
        "format_version": 2,
        "backend": args.backend,
        "probe_conditions": list(probe_conditions),
        "answer_probe_locations": list(answer_locations),
        "conflict_probe_locations": list(conflict_locations),
        "decision_side_probe_locations": list(decision_locations),
        "split_mode": args.split_mode,
        "fold_result_count": len(fold_results),
        "valid_result_count": sum(result["status"] == "valid" for result in fold_results),
        "invalid_result_count": sum(result["status"] == "invalid" for result in fold_results),
        "performance": timing,
        "fold_results": fold_results,
    }
    metrics_path = output_dir / "layer_probe_metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {metrics_path}")
    atomic_write_json(metrics_path, metric_payload)
    run_config.update(
        {
            "status": "complete",
            "fold_result_count": len(fold_results),
            "prediction_file": str(output_dir / "layer_probe_predictions.jsonl"),
            "shard_load_count": loader.shard_load_count,
            "performance": timing,
        }
    )
    atomic_write_json(output_dir / "run_config.json", run_config)
    print(
        json.dumps(
            {
                "status": "complete",
                "fold_results": len(fold_results),
                "valid": metric_payload["valid_result_count"],
                "invalid": metric_payload["invalid_result_count"],
                "performance": timing,
                "task_record_counts": {
                    task: len(
                        filter_task_records(
                            manifest, target, probe_conditions=probe_conditions
                        )
                    )
                    for task, (_position, target) in probe_tasks.items()
                },
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
