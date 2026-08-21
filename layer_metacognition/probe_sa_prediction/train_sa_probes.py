#!/usr/bin/env python3
"""Train item-grouped OOF hard-label and soft-score SA probes."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from layer_metacognition.hidden_state_store import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
)
from layer_metacognition.probe.common import iter_jsonl, sortable_item_id
from layer_metacognition.probe.hidden_state_loader import HiddenStateLoader
from layer_metacognition.probe.provenance import (
    canonical_fingerprint,
    sha256_file,
)
from layer_metacognition.probe.split_utils import (
    load_or_create_split_assignments,
    rows_for_outer_fold,
)
from layer_metacognition.probe.torch_logistic_probe import (
    fit_torch_logistic_probe,
    resolve_torch_device,
)

from . import (
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_LAYERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSITIONS,
    HIDDEN_STATE_DEFINITION,
    SA_CLASSES,
    TASKS,
    job_id,
    prediction_key,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument(
        "--positions",
        nargs="+",
        choices=list(DEFAULT_POSITIONS),
        default=list(DEFAULT_POSITIONS),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of item_id groups; all eligible cases are retained.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[list[int], list[str]]:
    layers = [int(value) for value in args.layers]
    positions = [str(value) for value in args.positions]
    if not layers or len(layers) != len(set(layers)) or any(value < 0 for value in layers):
        raise ValueError("--layers must contain distinct non-negative indices")
    if not positions or len(positions) != len(set(positions)):
        raise ValueError("--positions must contain distinct values")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")
    if args.max_samples is not None and args.max_samples < args.n_splits:
        raise ValueError("--max-samples must be at least --n-splits item groups")
    return sorted(layers), [value for value in DEFAULT_POSITIONS if value in positions]


def _same_sequence(left: Any, right: Sequence[Any]) -> bool:
    return [str(value) for value in left or []] == [str(value) for value in right]


def _validated_records(
    experiment_dir: Path,
    *,
    layers: Sequence[int],
    positions: Sequence[str],
    max_items: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config_path = experiment_dir / "config.json"
    results_path = experiment_dir / "results.jsonl"
    index_path = experiment_dir / "hidden_states" / "index.json"
    for path in (config_path, results_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required source artifact does not exist: {path}")
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    classes = tuple(str(value) for value in source_config.get("source_attribution_classes", []))
    if classes != SA_CLASSES:
        raise ValueError(f"SA probe requires source classes {list(SA_CLASSES)}, found {list(classes)}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    cases = index.get("cases")
    if not isinstance(cases, dict):
        raise ValueError(f"Hidden-state index has no cases object: {index_path}")
    available_layers = [int(value) for value in index.get("layer_indices", [])]
    available_positions = [str(value) for value in index.get("position_names", [])]
    missing_layers = sorted(set(int(value) for value in layers) - set(available_layers))
    missing_positions = sorted(set(positions) - set(available_positions))
    if missing_layers or missing_positions:
        raise ValueError(
            "Source hidden-state schema is incomplete: "
            f"missing_layers={missing_layers}, missing_positions={missing_positions}, "
            f"available_layers={available_layers}, available_positions={available_positions}"
        )

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source in iter_jsonl(results_path):
        raw_case_id = source.get("case_id")
        case_id = "" if raw_case_id is None else str(raw_case_id)
        item_id = source.get("item_id")
        condition = source.get("condition")
        try:
            prior_index = int(source["prior_index"])
        except (KeyError, TypeError, ValueError, OverflowError):
            prior_index = None
        reason: str | None = None
        generated = source.get("generated")
        attribution = generated.get("source_attribution") if isinstance(generated, dict) else None
        reference = source.get("hidden_state_reference")
        parsed_label = attribution.get("parsed_label") if isinstance(attribution, dict) else None
        soft_score = attribution.get("soft_image_score") if isinstance(attribution, dict) else None
        if source.get("status") != "completed":
            continue
        if source.get("attribution_mode") != "joint" or source.get("version") != "v4":
            continue
        if not case_id:
            reason = "missing_case_id"
        elif item_id is None or not str(item_id):
            reason = "missing_item_id"
        elif prior_index is None:
            reason = "missing_or_invalid_prior_index"
        elif condition is None or not str(condition):
            reason = "missing_condition"
        elif str(parsed_label) not in SA_CLASSES:
            reason = "missing_or_invalid_parsed_label"
        elif not isinstance(soft_score, (int, float)) or not math.isfinite(float(soft_score)):
            reason = "missing_or_non_finite_soft_image_score"
        elif not 0.0 <= float(soft_score) <= 1.0:
            reason = "soft_image_score_outside_unit_interval"
        elif not isinstance(reference, dict):
            reason = "missing_hidden_state_reference"
        elif not isinstance(cases.get(case_id), dict):
            reason = "case_missing_from_hidden_index"
        elif not _same_sequence(reference.get("layer_indices"), available_layers):
            reason = "reference_layer_schema_mismatch"
        elif not _same_sequence(reference.get("position_names"), available_positions):
            reason = "reference_position_schema_mismatch"
        elif reference.get("hidden_state_definition") != HIDDEN_STATE_DEFINITION:
            reason = "hidden_state_definition_mismatch"
        if reason is not None:
            failures.append(
                {
                    "case_id": case_id,
                    "item_id": source.get("item_id"),
                    "stage": "input_validation",
                    "reason": reason,
                }
            )
            continue
        records.append(
            {
                "case_id": case_id,
                "item_id": str(item_id),
                "prior_index": prior_index,
                "condition": str(condition),
                "version": "v4",
                "hard_label": str(parsed_label),
                "soft_score": float(soft_score),
                "hidden_state_reference": dict(reference),
            }
        )
    item_ids = sorted({record["item_id"] for record in records}, key=sortable_item_id)
    if max_items is not None:
        selected = set(item_ids[: int(max_items)])
        records = [record for record in records if record["item_id"] in selected]
        item_ids = item_ids[: int(max_items)]
    if len(item_ids) < 2:
        raise ValueError("SA probe requires at least two eligible item_id groups")
    return records, failures, {
        "source_config_path": str(config_path),
        "source_results_path": str(results_path),
        "source_index_path": str(index_path),
        "source_config_fingerprint": sha256_file(config_path),
        "source_results_fingerprint": sha256_file(results_path),
        "hidden_state_index_fingerprint": sha256_file(index_path),
        "selected_case_count": len(records),
        "selected_item_count": len(item_ids),
        "selected_item_ids": item_ids,
        "input_failure_count": len(failures),
    }


def _load_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path)) if path.is_file() else []


def _append_unique_failures(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    existing = {
        (str(row.get("case_id")), str(row.get("stage")), str(row.get("reason")), str(row.get("position")), str(row.get("layer")))
        for row in _load_jsonl_if_present(path)
    }
    if not path.exists():
        atomic_write_text(path, "")
    for row in rows:
        key = (
            str(row.get("case_id")),
            str(row.get("stage")),
            str(row.get("reason")),
            str(row.get("position")),
            str(row.get("layer")),
        )
        if key not in existing:
            append_jsonl(path, dict(row), fsync=False)
            existing.add(key)


def _prepare_run(
    output_dir: Path,
    run_config: dict[str, Any],
    *,
    resume: bool,
) -> tuple[dict[str, Any], bool]:
    config_path = output_dir / "run_config.json"
    protected = (
        config_path,
        output_dir / "progress.json",
        output_dir / "split_assignments.json",
        output_dir / "predictions" / "oof_predictions.jsonl",
        output_dir / "results" / "hard_label_results.json",
        output_dir / "results" / "soft_score_results.json",
    )
    if not config_path.exists():
        existing = [str(path) for path in protected[1:] if path.exists()]
        if existing:
            raise FileExistsError(f"Protected SA probe artifacts exist without run_config: {existing}")
        return run_config, False
    if not resume:
        raise FileExistsError(f"SA probe output already exists; pass --resume: {output_dir}")
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    if saved.get("config_fingerprint") != run_config["config_fingerprint"]:
        raise ValueError("Cannot resume SA probe because immutable configuration differs")
    return saved, saved.get("status") == "complete"


def _feature_matrix(
    loader: HiddenStateLoader,
    records: Sequence[dict[str, Any]],
    *,
    layer: int,
    position: str,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    usable: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        try:
            vector = loader.load_vector(record, layer=layer, position_name=position)
            if vector.ndim != 1 or not bool(np.isfinite(vector).all()):
                raise ValueError(f"invalid vector shape or values: {vector.shape}")
            usable.append(record)
            vectors.append(vector)
        except Exception as exc:
            failures.append(
                {
                    "case_id": record["case_id"],
                    "item_id": record["item_id"],
                    "stage": "feature_loading",
                    "position": position,
                    "layer": int(layer),
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
    if not vectors:
        return usable, np.empty((0, 0), dtype=np.float32), failures
    return usable, np.stack(vectors).astype(np.float32, copy=False), failures


def _append_predictions(
    path: Path,
    rows: Sequence[dict[str, Any]],
    existing_keys: set[tuple[str, str, int, int, str]],
) -> int:
    written = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        key = prediction_key(row)
        if key in existing_keys:
            continue
        append_jsonl(path, row, fsync=False)
        existing_keys.add(key)
        written += 1
    if written:
        with path.open("a", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    return written


def _base_prediction(record: dict[str, Any], *, task: str, position: str, layer: int, fold: int) -> dict[str, Any]:
    return {
        "case_id": record["case_id"],
        "item_id": record["item_id"],
        "prior_index": record["prior_index"],
        "condition": record["condition"],
        "version": record["version"],
        "task": task,
        "position": position,
        "layer": int(layer),
        "fold": int(fold),
    }


def _fit_hard(
    train_X: np.ndarray,
    test_X: np.ndarray,
    train_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
    *,
    device: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed = [label for label in SA_CLASSES if any(row["hard_label"] == label for row in train_records)]
    if len(observed) < 2:
        raise ValueError(f"Hard-label outer train has fewer than two classes: {observed}")
    label_to_local = {label: index for index, label in enumerate(observed)}
    encoded = np.asarray([label_to_local[row["hard_label"]] for row in train_records], dtype=np.int64)
    model = fit_torch_logistic_probe(
        train_X,
        encoded,
        C=1.0,
        device=device,
        seed=seed,
        binary_single_logit=False,
    )
    local_probabilities = np.asarray(model.predict_proba(test_X), dtype=np.float64)
    probabilities = np.zeros((len(test_records), len(SA_CLASSES)), dtype=np.float64)
    for local_index, label in enumerate(observed):
        probabilities[:, SA_CLASSES.index(label)] = local_probabilities[:, local_index]
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(test_records):
        predicted = SA_CLASSES[int(np.argmax(probabilities[index]))]
        rows.append(
            {
                **record,
                "true_label": record["hard_label"],
                "predicted_label": predicted,
                "class_probabilities": {
                    label: float(probabilities[index, class_index])
                    for class_index, label in enumerate(SA_CLASSES)
                },
            }
        )
    return rows, {
        "observed_train_classes": observed,
        "fit_diagnostics": dict(model.diagnostics),
    }


def _fit_soft(
    train_X: np.ndarray,
    test_X: np.ndarray,
    train_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, solver="lsqr")),
        ]
    )
    train_y = np.asarray([row["soft_score"] for row in train_records], dtype=np.float64)
    model.fit(train_X, train_y)
    predicted = np.asarray(model.predict(test_X), dtype=np.float64)
    rows = [
        {
            **record,
            "true_score": float(record["soft_score"]),
            "predicted_score": float(predicted[index]),
        }
        for index, record in enumerate(test_records)
    ]
    return rows, {"alpha": 1.0, "solver": "lsqr"}


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    layers, positions = _validate_args(args)
    experiment_dir = Path(args.experiment_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    records, initial_failures, provenance = _validated_records(
        experiment_dir,
        layers=layers,
        positions=positions,
        max_items=args.max_samples,
    )
    if provenance["selected_item_count"] < args.n_splits:
        raise ValueError(
            f"Need at least {args.n_splits} selected items, found {provenance['selected_item_count']}"
        )
    resolved_device = resolve_torch_device(args.device)
    immutable = {
        "format_version": 1,
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "layers": layers,
        "positions": positions,
        "tasks": list(TASKS),
        "hard_target_field": "generated.source_attribution.parsed_label",
        "soft_target_field": "generated.source_attribution.soft_image_score",
        "source_classes": list(SA_CLASSES),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "max_samples_item_groups": args.max_samples,
        "requested_device": args.device,
        "resolved_hard_probe_device": resolved_device,
        "hard_model": {
            "type": "balanced_l2_multinomial_logistic_regression",
            "backend": "torch",
            "C": 1.0,
        },
        "soft_model": {
            "type": "standard_scaler_plus_ridge",
            "backend": "sklearn",
            "alpha": 1.0,
            "prediction_clipping": False,
        },
        "hidden_state_definition": HIDDEN_STATE_DEFINITION,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        **provenance,
    }
    run_config = {
        **immutable,
        "config_fingerprint": canonical_fingerprint(immutable),
        "status": "running",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config, already_complete = _prepare_run(output_dir, run_config, resume=args.resume)
    if already_complete:
        return {"status": "complete", "resumed": True, "output_dir": str(output_dir)}
    atomic_write_json(output_dir / "run_config.json", run_config)
    failures_path = output_dir / "input_failures.jsonl"
    _append_unique_failures(failures_path, initial_failures)
    assignment = load_or_create_split_assignments(
        output_dir / "split_assignments.json",
        records,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    item_to_fold = {str(key): int(value) for key, value in assignment["item_to_fold"].items()}

    predictions_path = output_dir / "predictions" / "oof_predictions.jsonl"
    existing_predictions = _load_jsonl_if_present(predictions_path)
    existing_keys: set[tuple[str, str, int, int, str]] = set()
    for prediction in existing_predictions:
        key = prediction_key(prediction)
        if key in existing_keys:
            raise ValueError(f"Duplicate existing OOF prediction key: {key}")
        existing_keys.add(key)
    progress_path = output_dir / "progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {}
    )
    completed_jobs = set(str(value) for value in progress.get("completed_jobs", []))
    invalid_jobs: dict[str, Any] = dict(progress.get("invalid_jobs", {}))
    job_audits: dict[str, Any] = dict(progress.get("job_audits", {}))
    total_jobs = len(TASKS) * len(positions) * len(layers) * args.n_splits
    loader = HiddenStateLoader(experiment_dir, cache_size=2)

    def checkpoint(status: str) -> None:
        atomic_write_json(
            progress_path,
            {
                "status": status,
                "total_job_count": total_jobs,
                "completed_job_count": len(completed_jobs),
                "invalid_job_count": len(invalid_jobs),
                "prediction_count": len(existing_keys),
                "completed_jobs": sorted(completed_jobs),
                "invalid_jobs": invalid_jobs,
                "job_audits": job_audits,
                "elapsed_seconds": float(time.perf_counter() - started),
            },
        )

    checkpoint("running")
    for position in positions:
        for layer in layers:
            usable, matrix, feature_failures = _feature_matrix(
                loader, records, layer=layer, position=position
            )
            _append_unique_failures(failures_path, feature_failures)
            ordinal = {row["case_id"]: index for index, row in enumerate(usable)}
            for task in TASKS:
                for fold in range(args.n_splits):
                    identifier = job_id(task, position, layer, fold)
                    if identifier in invalid_jobs:
                        continue
                    train_records, test_records = rows_for_outer_fold(
                        usable,
                        item_to_fold,
                        fold=fold,
                        train_version="v4",
                        test_version="v4",
                    )
                    expected_case_ids = {str(row["case_id"]) for row in test_records}
                    present_case_ids = {
                        key[4]
                        for key in existing_keys
                        if key[:4] == (task, position, int(layer), int(fold))
                    }
                    extra = present_case_ids - expected_case_ids
                    if extra:
                        raise ValueError(f"OOF predictions contain unexpected cases for {identifier}: {sorted(extra)}")
                    if present_case_ids == expected_case_ids and expected_case_ids:
                        completed_jobs.add(identifier)
                        checkpoint("running")
                        continue
                    train_items = {row["item_id"] for row in train_records}
                    test_items = {row["item_id"] for row in test_records}
                    overlap = train_items.intersection(test_items)
                    if overlap:
                        raise AssertionError(f"Item leakage in {identifier}: {sorted(overlap)}")
                    job_audits[identifier] = {
                        "train_sample_count": len(train_records),
                        "test_sample_count": len(test_records),
                        "train_item_count": len(train_items),
                        "test_item_count": len(test_items),
                        "item_overlap_count": 0,
                    }
                    try:
                        if not train_records or not test_records:
                            raise ValueError("Outer fold has an empty train or validation partition")
                        train_indices = [ordinal[row["case_id"]] for row in train_records]
                        test_indices = [ordinal[row["case_id"]] for row in test_records]
                        train_X = matrix[train_indices]
                        test_X = matrix[test_indices]
                        if task == "hard_label":
                            task_rows, diagnostics = _fit_hard(
                                train_X,
                                test_X,
                                train_records,
                                test_records,
                                device=resolved_device,
                                seed=args.seed,
                            )
                        else:
                            task_rows, diagnostics = _fit_soft(
                                train_X, test_X, train_records, test_records
                            )
                        prediction_rows = []
                        for row in task_rows:
                            payload = _base_prediction(
                                row,
                                task=task,
                                position=position,
                                layer=layer,
                                fold=fold,
                            )
                            if task == "hard_label":
                                payload.update(
                                    {
                                        "true_label": row["true_label"],
                                        "predicted_label": row["predicted_label"],
                                        "class_probabilities": row["class_probabilities"],
                                    }
                                )
                            else:
                                payload.update(
                                    {
                                        "true_score": row["true_score"],
                                        "predicted_score": row["predicted_score"],
                                    }
                                )
                            prediction_rows.append(payload)
                        _append_predictions(predictions_path, prediction_rows, existing_keys)
                        completed_jobs.add(identifier)
                        job_audits[identifier]["model_diagnostics"] = diagnostics
                    except Exception as exc:
                        invalid_jobs[identifier] = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    checkpoint("running")
            del matrix
            gc.collect()
    checkpoint("training_complete")
    run_config.update(
        {
            "status": "training_complete",
            "completed_job_count": len(completed_jobs),
            "invalid_job_count": len(invalid_jobs),
            "prediction_count": len(existing_keys),
            "shard_load_count": loader.shard_load_count,
            "training_seconds": float(time.perf_counter() - started),
        }
    )
    atomic_write_json(output_dir / "run_config.json", run_config)
    return {
        "status": "training_complete",
        "completed_jobs": len(completed_jobs),
        "invalid_jobs": len(invalid_jobs),
        "predictions": len(existing_keys),
        "output_dir": str(output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_training(args)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
