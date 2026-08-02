#!/usr/bin/env python3
"""Train per-layer AC/PANL linear probes on an existing Probe manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

from layer_metacognition.hidden_state_store import atomic_write_json

from . import (
    C_GRID,
    EASY_CONDITIONS,
    HIDDEN_STATE_DEFINITION,
    PROBE_TASKS,
    VERSION_SETTINGS,
)
from .common import iter_jsonl, probe_output_dir
from .hidden_state_loader import HiddenStateLoader
from .probe_metrics import (
    evaluate_required_subsets,
    majority_label,
    subset_membership,
)
from .probe_models import (
    build_current_answer_baseline,
    build_hidden_state_probe,
    choose_regularization_C,
)
from .split_utils import (
    load_or_create_split_assignments,
    permute_labels_by_unique_key,
    rows_for_outer_fold,
)


def filter_task_records(
    manifest: Sequence[dict[str, Any]],
    target_field: str,
    *,
    text_scope: str,
) -> list[dict[str, Any]]:
    if target_field == "image_only_answer":
        records = [
            record
            for record in manifest
            if record.get("eligible_image_probe")
            and record.get("condition") in EASY_CONDITIONS
            and isinstance(record.get(target_field), str)
        ]
        forbidden = [
            record
            for record in records
            if record.get("condition") not in EASY_CONDITIONS
        ]
        if forbidden:
            raise AssertionError("Image Probe filter admitted hard/null/irr records")
        return records
    if target_field != "text_only_answer":
        raise ValueError(f"Unsupported target field: {target_field}")
    records = [
        record
        for record in manifest
        if record.get("eligible_text_probe")
        and isinstance(record.get(target_field), str)
    ]
    if text_scope == "matched_easy":
        records = [
            record for record in records if record.get("condition") in EASY_CONDITIONS
        ]
    elif text_scope != "all":
        raise ValueError(f"Unknown text scope: {text_scope}")
    return records


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
    train_version: str,
    test_version: str,
    model_type: str,
    target_field: str,
    predicted_label: str,
    probabilities: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    raw_field = f"{target_field}_raw"
    memberships = subset_membership(record)
    return {
        "case_id": record["case_id"],
        "item_id": record["item_id"],
        "prior_index": record["prior_index"],
        "condition": record["condition"],
        "version": record["version"],
        "fold": fold,
        "task": task,
        "position": position,
        "layer": layer,
        "train_version": train_version,
        "test_version": test_version,
        "model_type": model_type,
        "true_label": record[target_field],
        "true_label_raw": record.get(raw_field),
        "predicted_label": predicted_label,
        "class_probabilities": {
            str(label): float(probabilities[index])
            for index, label in enumerate(classes)
        },
        "subsets": [
            name for name, included in memberships.items() if included
        ],
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
    train_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
    reason: dict[str, Any],
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
        "train_sample_count": len(train_records),
        "test_sample_count": len(test_records),
        "train_item_count": len({str(record["item_id"]) for record in train_records}),
        "test_item_count": len({str(record["item_id"]) for record in test_records}),
        "classes": [],
        "selected_C": None,
        "c_selection": None,
        "subset_metrics": {},
    }


def _fit_current_answer_baseline(
    *,
    train_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
    target_field: str,
    encoder: LabelEncoder,
) -> tuple[list[str], np.ndarray]:
    train_X = np.asarray(
        [[str(record["current_answer"])] for record in train_records],
        dtype=object,
    )
    test_X = np.asarray(
        [[str(record["current_answer"])] for record in test_records],
        dtype=object,
    )
    encoded_train = encoder.transform(
        [str(record[target_field]) for record in train_records]
    )
    model = build_current_answer_baseline()
    model.fit(train_X, encoded_train)
    predicted_encoded = np.asarray(model.predict(test_X), dtype=np.int64)
    classifier = model.named_steps["classifier"]
    probabilities = _align_probabilities(
        model.predict_proba(test_X),
        classifier.classes_,
        len(encoder.classes_),
    )
    return [str(value) for value in encoder.inverse_transform(predicted_encoded)], probabilities


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text-scope",
        choices=["matched_easy", "all"],
        default="matched_easy",
    )
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--shard-cache-size", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")
    if args.permutations < 0:
        raise ValueError("--permutations must be non-negative")
    if args.shard_cache_size < 1:
        raise ValueError("--shard-cache-size must be positive")
    layers = [int(value) for value in args.layers]
    if not layers or len(layers) != len(set(layers)) or any(value < 0 for value in layers):
        raise ValueError("--layers must contain distinct non-negative layer indices")
    layers = sorted(layers)

    experiment_dir = Path(args.experiment_dir).resolve()
    output_dir = probe_output_dir(experiment_dir)
    manifest_path = output_dir / "probe_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Probe manifest does not exist: {manifest_path}; run build_probe_manifest first"
        )
    manifest = list(iter_jsonl(manifest_path))
    if not manifest:
        raise ValueError("Probe manifest is empty")
    available_layers = {
        int(value)
        for record in manifest
        for value in record["hidden_state_reference"].get("layer_indices", [])
    }
    missing_layers = sorted(set(layers) - available_layers)
    if missing_layers:
        raise ValueError(
            f"Requested layers are absent from the manifest: {missing_layers}; "
            f"available={sorted(available_layers)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment = load_or_create_split_assignments(
        output_dir / "split_assignments.json",
        manifest,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    item_to_fold = {
        str(key): int(value) for key, value in assignment["item_to_fold"].items()
    }
    permutation_seeds = [args.seed + index for index in range(args.permutations)]
    run_config = {
        "format_version": 1,
        "experiment_dir": str(experiment_dir),
        "manifest_path": str(manifest_path),
        "probe_tasks": {
            key: {"position": value[0], "target_field": value[1]}
            for key, value in PROBE_TASKS.items()
        },
        "version_settings": {
            key: {"train_version": value[0], "test_version": value[1]}
            for key, value in VERSION_SETTINGS.items()
        },
        "layers": layers,
        "n_splits": args.n_splits,
        "inner_n_splits": 3,
        "seed": args.seed,
        "text_scope": args.text_scope,
        "C_grid": list(C_GRID),
        "C_selection_metric": "balanced_accuracy",
        "C_tie_break": "smaller_C",
        "permutation_count": args.permutations,
        "permutation_seeds": permutation_seeds,
        "permutation_C_policy": "reuse_hidden_state_probe_selected_C",
        "shard_cache_size": args.shard_cache_size,
        "hidden_state_definition": HIDDEN_STATE_DEFINITION,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "status": "running",
    }
    atomic_write_json(output_dir / "run_config.json", run_config)

    loader = HiddenStateLoader(experiment_dir, cache_size=args.shard_cache_size)
    fold_results: list[dict[str, Any]] = []
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".layer_probe_predictions.",
        suffix=".jsonl.tmp",
        dir=output_dir,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as prediction_handle:
            for task, (position, target_field) in PROBE_TASKS.items():
                task_records = filter_task_records(
                    manifest,
                    target_field,
                    text_scope=args.text_scope,
                )
                for setting, (train_version, test_version) in VERSION_SETTINGS.items():
                    for fold in range(args.n_splits):
                        train_records, test_records = rows_for_outer_fold(
                            task_records,
                            item_to_fold,
                            fold=fold,
                            train_version=train_version,
                            test_version=test_version,
                        )
                        encoder, invalid_reason = validate_outer_labels(
                            train_records,
                            test_records,
                            target_field,
                        )
                        if invalid_reason is not None:
                            if position == "panl":
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
                                        train_records=train_records,
                                        test_records=test_records,
                                        reason=invalid_reason,
                                    )
                                )
                            continue
                        assert encoder is not None
                        classes = [str(value) for value in encoder.classes_]
                        train_labels = [
                            str(record[target_field]) for record in train_records
                        ]
                        test_labels = [
                            str(record[target_field]) for record in test_records
                        ]
                        majority = majority_label(train_labels)

                        if position == "panl":
                            baseline_predicted, baseline_probabilities = (
                                _fit_current_answer_baseline(
                                    train_records=train_records,
                                    test_records=test_records,
                                    target_field=target_field,
                                    encoder=encoder,
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
                                            train_version=train_version,
                                            test_version=test_version,
                                            model_type="current_answer_only_baseline",
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
                            train_X = _features(
                                loader,
                                train_records,
                                layer=layer,
                                position=position,
                            )
                            test_X = _features(
                                loader,
                                test_records,
                                layer=layer,
                                position=position,
                            )
                            selected_C, c_selection = choose_regularization_C(
                                train_X,
                                train_labels,
                                [str(record["item_id"]) for record in train_records],
                                seed=args.seed,
                            )
                            encoded_train = encoder.transform(train_labels)
                            model = build_hidden_state_probe(selected_C)
                            model.fit(train_X, encoded_train)
                            predicted_encoded = np.asarray(
                                model.predict(test_X), dtype=np.int64
                            )
                            predicted = [
                                str(value)
                                for value in encoder.inverse_transform(predicted_encoded)
                            ]
                            classifier = model.named_steps["classifier"]
                            probabilities = _align_probabilities(
                                model.predict_proba(test_X),
                                classifier.classes_,
                                len(classes),
                            )
                            permuted_predictions: list[list[str]] = []
                            for permutation_seed in permutation_seeds:
                                permuted_labels = permute_labels_by_unique_key(
                                    train_records,
                                    target_field,
                                    seed=permutation_seed,
                                )
                                encoded_permuted = encoder.transform(permuted_labels)
                                permuted_model = build_hidden_state_probe(selected_C)
                                permuted_model.fit(train_X, encoded_permuted)
                                permuted_encoded_prediction = np.asarray(
                                    permuted_model.predict(test_X),
                                    dtype=np.int64,
                                )
                                permuted_predictions.append(
                                    [
                                        str(value)
                                        for value in encoder.inverse_transform(
                                            permuted_encoded_prediction
                                        )
                                    ]
                                )
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
                                            train_version=train_version,
                                            test_version=test_version,
                                            model_type="hidden_state_probe",
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
            prediction_handle.flush()
            os.fsync(prediction_handle.fileno())
        os.replace(temporary_name, output_dir / "layer_probe_predictions.jsonl")
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        run_config["status"] = "failed"
        atomic_write_json(output_dir / "run_config.json", run_config)
        raise

    metric_payload = {
        "format_version": 1,
        "fold_result_count": len(fold_results),
        "valid_result_count": sum(
            result["status"] == "valid" for result in fold_results
        ),
        "invalid_result_count": sum(
            result["status"] == "invalid" for result in fold_results
        ),
        "fold_results": fold_results,
    }
    atomic_write_json(output_dir / "layer_probe_metrics.json", metric_payload)
    run_config.update(
        {
            "status": "complete",
            "fold_result_count": len(fold_results),
            "prediction_file": str(
                output_dir / "layer_probe_predictions.jsonl"
            ),
            "shard_load_count": loader.shard_load_count,
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
                "task_record_counts": {
                    task: len(
                        filter_task_records(
                            manifest,
                            target,
                            text_scope=args.text_scope,
                        )
                    )
                    for task, (_position, target) in PROBE_TASKS.items()
                },
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
