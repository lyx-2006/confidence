"""Required linear Probe pipelines and inner grouped C selection."""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from . import C_GRID

# The experiment explicitly requires ``penalty="l2"``. Scikit-learn 1.8+
# deprecates spelling the default penalty explicitly, but the requested model
# and behavior remain supported. Suppress only that compatibility warning;
# convergence and data warnings remain visible.
warnings.filterwarnings(
    "ignore",
    message="'penalty' was deprecated in version 1.8.*",
    category=FutureWarning,
)


def build_hidden_state_probe(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=float(C),
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=5000,
                ),
            ),
        ]
    )


def build_current_answer_baseline() -> Pipeline:
    return Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=5000,
                ),
            ),
        ]
    )


def choose_regularization_C(
    X: np.ndarray,
    labels: Sequence[str],
    groups: Sequence[str],
    *,
    seed: int,
    c_grid: Sequence[float] = C_GRID,
    n_splits: int = 3,
) -> tuple[float, dict[str, Any]]:
    y = np.asarray(labels, dtype=object)
    group_values = np.asarray(groups, dtype=object)
    unique_groups = set(str(value) for value in group_values)
    if len(unique_groups) < n_splits:
        return 1.0, {
            "status": "fallback",
            "reason": "insufficient_inner_groups",
            "group_count": len(unique_groups),
        }
    if len(set(str(value) for value in y)) < 2:
        return 1.0, {"status": "fallback", "reason": "single_class_inner_train"}
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(X, y, groups=group_values))
    scores: dict[str, list[float]] = {str(float(C)): [] for C in c_grid}
    for train_indices, validation_indices in splits:
        inner_train_labels = [str(value) for value in y[train_indices]]
        inner_validation_labels = [str(value) for value in y[validation_indices]]
        encoder = LabelEncoder().fit(inner_train_labels)
        unseen = sorted(set(inner_validation_labels) - set(encoder.classes_))
        if unseen:
            return 1.0, {
                "status": "fallback",
                "reason": "inner_validation_unseen_classes",
                "unseen_classes": unseen,
            }
        if len(encoder.classes_) < 2:
            return 1.0, {
                "status": "fallback",
                "reason": "single_class_inner_fold",
            }
        encoded_train = encoder.transform(inner_train_labels)
        encoded_validation = encoder.transform(inner_validation_labels)
        for C in c_grid:
            model = build_hidden_state_probe(float(C))
            model.fit(X[train_indices], encoded_train)
            predicted = model.predict(X[validation_indices])
            score = (
                accuracy_score(encoded_validation, predicted)
                if len(set(int(value) for value in encoded_validation)) == 1
                else balanced_accuracy_score(encoded_validation, predicted)
            )
            scores[str(float(C))].append(
                float(score)
            )
    mean_scores = {
        key: float(np.mean(values)) for key, values in scores.items()
    }
    selected = min(
        (float(value) for value in c_grid),
        key=lambda value: (-mean_scores[str(float(value))], value),
    )
    return selected, {
        "status": "selected",
        "metric": "balanced_accuracy",
        "n_splits": n_splits,
        "scores": scores,
        "mean_scores": mean_scores,
        "tie_break": "smaller_C",
    }
