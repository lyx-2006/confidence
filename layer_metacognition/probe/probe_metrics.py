"""Classification metrics and required behavioral analysis subsets."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from . import EASY_CONDITIONS

SUBSETS = (
    "pooled_overall",
    "easy_overall",
    "consistent_easy",
    "consistent_hard",
    "conflict_easy",
    "conflict_hard",
    "discriminative_conflict",
    "joint_follows_text",
    "joint_follows_image",
    "joint_follows_neither",
)


def subset_membership(record: dict[str, Any]) -> dict[str, bool]:
    condition = record.get("condition")
    text_answer = record.get("text_only_answer")
    image_answer = record.get("image_only_answer")
    current_answer = record.get("current_answer")
    discriminative = bool(
        text_answer is not None
        and image_answer is not None
        and text_answer != image_answer
    )
    return {
        "pooled_overall": True,
        "easy_overall": condition in EASY_CONDITIONS,
        "consistent_easy": condition == "consistent_easy",
        "consistent_hard": condition == "consistent_hard",
        "conflict_easy": condition == "conflict_easy",
        "conflict_hard": condition == "conflict_hard",
        "discriminative_conflict": condition
        in {"conflict_easy", "conflict_hard"}
        and discriminative,
        "joint_follows_text": discriminative and current_answer == text_answer,
        "joint_follows_image": discriminative and current_answer == image_answer,
        "joint_follows_neither": discriminative
        and current_answer != text_answer
        and current_answer != image_answer,
    }


def majority_label(labels: Sequence[str]) -> str:
    if not labels:
        raise ValueError("Cannot compute majority class from empty labels")
    counts = Counter(str(value) for value in labels)
    maximum = max(counts.values())
    return sorted(label for label, count in counts.items() if count == maximum)[0]


def _empty_metrics(majority_class: str, selected_C: float | None) -> dict[str, Any]:
    return {
        "status": "empty",
        "accuracy": None,
        "balanced_accuracy": None,
        "macro_f1": None,
        "cross_entropy": None,
        "roc_auc": None,
        "sample_count": 0,
        "class_counts": {},
        "item_count": 0,
        "majority_class": majority_class,
        "majority_baseline_accuracy": None,
        "permuted_label_accuracy_mean": None,
        "permuted_label_accuracy_std": None,
        "selected_C": selected_C,
    }


def compute_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    probabilities: np.ndarray,
    *,
    classes: Sequence[str],
    item_ids: Sequence[str],
    majority_class: str,
    selected_C: float | None,
    permuted_predictions: Sequence[Sequence[str]] | None = None,
) -> dict[str, Any]:
    true = np.asarray(true_labels, dtype=object)
    predicted = np.asarray(predicted_labels, dtype=object)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if len(true) == 0:
        return _empty_metrics(majority_class, selected_C)
    if probability_values.shape != (len(true), len(classes)):
        raise ValueError(
            f"Probability shape {probability_values.shape} does not match "
            f"{len(true)} samples and {len(classes)} classes"
        )
    row_sums = probability_values.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(probability_values)) or np.any(row_sums <= 0):
        raise ValueError("Probabilities must be finite with a positive row sum")
    probability_values = probability_values / row_sums
    permutation_accuracies = [
        float(accuracy_score(true, np.asarray(values, dtype=object)))
        for values in (permuted_predictions or [])
    ]
    balanced_accuracy = (
        float(accuracy_score(true, predicted))
        if len(set(str(value) for value in true)) == 1
        else float(balanced_accuracy_score(true, predicted))
    )
    class_counts = Counter(str(value) for value in true)
    class_to_index = {str(label): index for index, label in enumerate(classes)}
    unknown = sorted(set(class_counts) - set(class_to_index))
    if unknown:
        raise ValueError(f"True labels are absent from probability columns: {unknown}")
    true_indices = np.asarray(
        [class_to_index[str(value)] for value in true], dtype=np.int64
    )
    true_probabilities = probability_values[np.arange(len(true)), true_indices]
    cross_entropy = float(
        -np.mean(np.log(np.clip(true_probabilities, np.finfo(np.float64).tiny, 1.0)))
    )
    roc_auc = None
    if len(classes) == 2 and len(class_counts) == 2:
        binary_true = np.asarray(
            [str(value) == str(classes[1]) for value in true], dtype=np.int64
        )
        roc_auc = float(roc_auc_score(binary_true, probability_values[:, 1]))
    return {
        "status": "valid",
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "cross_entropy": cross_entropy,
        "roc_auc": roc_auc,
        "sample_count": int(len(true)),
        "class_counts": dict(sorted(class_counts.items())),
        "item_count": len({str(value) for value in item_ids}),
        "majority_class": majority_class,
        "majority_baseline_accuracy": float(
            accuracy_score(true, np.full(len(true), majority_class, dtype=object))
        ),
        "permuted_label_accuracy_mean": (
            float(np.mean(permutation_accuracies))
            if permutation_accuracies
            else None
        ),
        "permuted_label_accuracy_std": (
            float(np.std(permutation_accuracies, ddof=0))
            if permutation_accuracies
            else None
        ),
        "selected_C": selected_C,
    }


def evaluate_required_subsets(
    records: Sequence[dict[str, Any]],
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    probabilities: np.ndarray,
    *,
    classes: Sequence[str],
    majority_class: str,
    selected_C: float | None,
    permuted_predictions: Sequence[Sequence[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    memberships = [subset_membership(record) for record in records]
    true = np.asarray(true_labels, dtype=object)
    predicted = np.asarray(predicted_labels, dtype=object)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    permuted = [
        np.asarray(values, dtype=object) for values in (permuted_predictions or [])
    ]
    output: dict[str, dict[str, Any]] = {}
    for subset in SUBSETS:
        indices = np.asarray(
            [index for index, value in enumerate(memberships) if value[subset]],
            dtype=np.int64,
        )
        output[subset] = compute_metrics(
            true[indices],
            predicted[indices],
            probability_values[indices],
            classes=classes,
            item_ids=[str(records[index]["item_id"]) for index in indices],
            majority_class=majority_class,
            selected_C=selected_C,
            permuted_predictions=[values[indices] for values in permuted],
        )
    return output
