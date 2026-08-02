"""Deterministic grouped outer folds and unique-key permutations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

from layer_metacognition.hidden_state_store import atomic_write_json

from .common import sortable_item_id


def create_split_assignments(
    records: Sequence[dict[str, Any]],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    item_ids = sorted({str(record["item_id"]) for record in records}, key=sortable_item_id)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(item_ids) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} distinct item_id groups, found {len(item_ids)}"
        )
    values = np.arange(len(item_ids), dtype=np.int64).reshape(-1, 1)
    groups = np.asarray(item_ids, dtype=object)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    item_to_fold: dict[str, int] = {}
    for fold, (_train, test) in enumerate(splitter.split(values, groups=groups)):
        for index in test:
            item_to_fold[item_ids[int(index)]] = fold
    assignment = {
        "format_version": 1,
        "n_splits": int(n_splits),
        "seed": int(seed),
        "group_key": "item_id",
        "item_count": len(item_ids),
        "item_to_fold": item_to_fold,
    }
    validate_split_assignments(records, assignment)
    return assignment


def validate_split_assignments(
    records: Sequence[dict[str, Any]],
    assignment: dict[str, Any],
) -> None:
    item_to_fold = assignment.get("item_to_fold")
    n_splits = int(assignment.get("n_splits", 0))
    if not isinstance(item_to_fold, dict) or n_splits < 2:
        raise ValueError("Invalid split assignment structure")
    record_items = {str(record["item_id"]) for record in records}
    assignment_items = {str(value) for value in item_to_fold}
    if record_items != assignment_items:
        missing = sorted(record_items - assignment_items, key=sortable_item_id)
        extra = sorted(assignment_items - record_items, key=sortable_item_id)
        raise ValueError(f"Split item coverage mismatch: missing={missing}, extra={extra}")
    invalid = {
        item_id: fold
        for item_id, fold in item_to_fold.items()
        if not isinstance(fold, int) or fold < 0 or fold >= n_splits
    }
    if invalid:
        raise ValueError(f"Invalid item fold values: {invalid}")
    seen_folds = set(item_to_fold.values())
    if seen_folds != set(range(n_splits)):
        raise ValueError(f"Split assignments do not cover every fold: {seen_folds}")
    for fold in range(n_splits):
        train_items = {
            item_id for item_id, item_fold in item_to_fold.items() if item_fold != fold
        }
        test_items = {
            item_id for item_id, item_fold in item_to_fold.items() if item_fold == fold
        }
        if train_items.intersection(test_items):
            raise AssertionError(f"Train/test item leakage in fold {fold}")
    for record in records:
        item_id = str(record["item_id"])
        record_fold = record.get("fold")
        if record_fold is not None and int(record_fold) != int(item_to_fold[item_id]):
            raise ValueError(
                f"Case fold disagrees with item fold for {record.get('case_id')}"
            )


def load_or_create_split_assignments(
    path: str | Path,
    records: Sequence[dict[str, Any]],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    destination = Path(path)
    if destination.is_file():
        assignment = json.loads(destination.read_text(encoding="utf-8"))
        if (
            int(assignment.get("n_splits", -1)) != n_splits
            or int(assignment.get("seed", -1)) != seed
        ):
            raise ValueError(
                f"Existing split assignments use n_splits={assignment.get('n_splits')} "
                f"seed={assignment.get('seed')}, requested n_splits={n_splits} seed={seed}"
            )
        validate_split_assignments(records, assignment)
        return assignment
    assignment = create_split_assignments(records, n_splits=n_splits, seed=seed)
    atomic_write_json(destination, assignment)
    return assignment


def rows_for_outer_fold(
    records: Sequence[dict[str, Any]],
    item_to_fold: dict[str, int],
    *,
    fold: int,
    train_version: str,
    test_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [
        record
        for record in records
        if str(record["version"]) == train_version
        and int(item_to_fold[str(record["item_id"])]) != fold
    ]
    test = [
        record
        for record in records
        if str(record["version"]) == test_version
        and int(item_to_fold[str(record["item_id"])]) == fold
    ]
    train_items = {str(record["item_id"]) for record in train}
    test_items = {str(record["item_id"]) for record in test}
    overlap = train_items.intersection(test_items)
    if overlap:
        raise AssertionError(
            f"Cross-version item leakage in fold {fold}: "
            f"{sorted(overlap, key=sortable_item_id)}"
        )
    return train, test


def label_key(record: dict[str, Any], target_field: str) -> tuple[Any, ...]:
    if target_field == "text_only_answer":
        return str(record["item_id"]), int(record["prior_index"])
    if target_field == "image_only_answer":
        return str(record["item_id"]), str(record["condition"])
    raise ValueError(f"Unsupported behavior-label field: {target_field!r}")


def permute_labels_by_unique_key(
    records: Sequence[dict[str, Any]],
    target_field: str,
    *,
    seed: int,
) -> list[str]:
    key_to_label: dict[tuple[Any, ...], str] = {}
    for record in records:
        key = label_key(record, target_field)
        label = record.get(target_field)
        if not isinstance(label, str) or not label:
            raise ValueError(f"Record has no valid {target_field}: {record.get('case_id')}")
        previous = key_to_label.setdefault(key, label)
        if previous != label:
            raise ValueError(f"Behavior label key {key} maps to conflicting labels")
    keys = sorted(key_to_label, key=lambda value: tuple(str(part) for part in value))
    labels = np.asarray([key_to_label[key] for key in keys], dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = labels[rng.permutation(len(labels))]
    mapping = {key: str(shuffled[index]) for index, key in enumerate(keys)}
    return [mapping[label_key(record, target_field)] for record in records]
