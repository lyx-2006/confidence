from __future__ import annotations

import math

import numpy as np

from layer_metacognition.probe.probe_metrics import (
    SUBSETS,
    compute_metrics,
    evaluate_required_subsets,
)
from layer_metacognition.probe.probe_models import (
    build_hidden_state_probe,
    choose_regularization_C,
)


def test_metrics_and_probability_cross_entropy() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.25, 0.75]], dtype=np.float64)
    result = compute_metrics(
        ["blue", "yellow"],
        ["blue", "yellow"],
        probabilities,
        classes=["blue", "yellow"],
        item_ids=["1", "2"],
        majority_class="blue",
        selected_C=0.1,
        permuted_predictions=[["yellow", "blue"], ["blue", "yellow"]],
    )
    expected = -(math.log(0.8) + math.log(0.75)) / 2
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert math.isclose(result["cross_entropy"], expected)
    assert result["permuted_label_accuracy_mean"] == 0.5
    assert result["permuted_label_accuracy_std"] == 0.5


def test_required_subsets_include_explicit_empty_records() -> None:
    record = {
        "item_id": "1",
        "condition": "consistent_easy",
        "text_only_answer": "blue",
        "image_only_answer": "blue",
        "current_answer": "blue",
    }
    result = evaluate_required_subsets(
        [record],
        ["blue"],
        ["blue"],
        np.asarray([[1.0, 0.0]]),
        classes=["blue", "yellow"],
        majority_class="blue",
        selected_C=1.0,
    )
    assert set(result) == set(SUBSETS)
    assert result["pooled_overall"]["sample_count"] == 1
    assert result["easy_overall"]["sample_count"] == 1
    assert result["conflict_easy"]["status"] == "empty"
    assert result["consistent_hard"]["status"] == "empty"
    assert result["discriminative_conflict"]["status"] == "empty"


def test_hidden_probe_and_inner_grouped_C_selection() -> None:
    rng = np.random.default_rng(3)
    X = np.vstack(
        [
            rng.normal(-2, 0.1, size=(6, 3)),
            rng.normal(2, 0.1, size=(6, 3)),
        ]
    ).astype(np.float32)
    labels = ["blue"] * 6 + ["yellow"] * 6
    groups = [f"b{index // 2}" for index in range(6)] + [
        f"y{index // 2}" for index in range(6)
    ]
    selected, detail = choose_regularization_C(
        X,
        labels,
        groups,
        seed=42,
        n_splits=3,
    )
    assert selected in {0.01, 0.1, 1.0, 10.0}
    assert detail["status"] == "selected"
    model = build_hidden_state_probe(selected).fit(
        X,
        np.asarray([0] * 6 + [1] * 6),
    )
    assert model.predict(X).shape == (12,)


def test_inner_C_falls_back_when_groups_are_insufficient() -> None:
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    selected, detail = choose_regularization_C(
        X,
        ["a", "b", "a", "b"],
        ["one", "one", "two", "two"],
        seed=42,
        n_splits=3,
    )
    assert selected == 1.0
    assert detail["reason"] == "insufficient_inner_groups"
