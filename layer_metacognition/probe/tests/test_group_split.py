from __future__ import annotations

import numpy as np

from layer_metacognition.probe import PROBE_TASKS
from layer_metacognition.probe.probe_models import build_current_answer_baseline
from layer_metacognition.probe.split_utils import (
    create_split_assignments,
    permute_labels_by_unique_key,
    rows_for_outer_fold,
)
from layer_metacognition.probe.train_layer_probes import (
    filter_task_records,
    validate_outer_labels,
)


def _records() -> list[dict]:
    records = []
    for item in range(10):
        for version in ("v3", "v4"):
            for prior in (0, 1):
                records.append(
                    {
                        "case_id": f"{item}-{prior}-{version}",
                        "item_id": str(item),
                        "prior_index": prior,
                        "condition": "consistent_easy",
                        "version": version,
                        "text_only_answer": "blue" if prior == 0 else "yellow",
                        "image_only_answer": "blue" if item % 2 == 0 else "yellow",
                        "current_answer": "blue",
                        "eligible_text_probe": True,
                        "eligible_image_probe": True,
                    }
                )
    return records


def test_outer_group_split_and_cross_version_isolation() -> None:
    records = _records()
    assignment = create_split_assignments(records, n_splits=5, seed=42)
    mapping = assignment["item_to_fold"]
    for fold in range(5):
        train, test = rows_for_outer_fold(
            records,
            mapping,
            fold=fold,
            train_version="v3",
            test_version="v4",
        )
        train_items = {record["item_id"] for record in train}
        test_items = {record["item_id"] for record in test}
        assert train_items.isdisjoint(test_items)
        assert all(mapping[record["item_id"]] == fold for record in test)


def test_image_filter_never_admits_hard_null_or_irr() -> None:
    records = _records()[:2]
    for condition in ("consistent_hard", "conflict_hard", "null", "irr"):
        records.append(
            {
                **records[0],
                "case_id": condition,
                "condition": condition,
                "eligible_image_probe": True,
            }
        )
    filtered = filter_task_records(
        records,
        "image_only_answer",
        text_scope="matched_easy",
    )
    assert filtered
    assert {record["condition"] for record in filtered} == {"consistent_easy"}


def test_permutation_occurs_at_unique_label_key_level() -> None:
    records = _records()
    permuted = permute_labels_by_unique_key(
        records,
        "image_only_answer",
        seed=7,
    )
    key_to_values: dict[tuple[str, str], set[str]] = {}
    for record, label in zip(records, permuted):
        key = (record["item_id"], record["condition"])
        key_to_values.setdefault(key, set()).add(label)
    assert all(len(values) == 1 for values in key_to_values.values())
    original_key_labels = {
        (record["item_id"], record["condition"]): record["image_only_answer"]
        for record in records
    }
    assert sorted(original_key_labels.values()) == sorted(
        next(iter(values)) for values in key_to_values.values()
    )


def test_unseen_outer_test_class_invalidates_whole_fold() -> None:
    train = [
        {"text_only_answer": "blue"},
        {"text_only_answer": "yellow"},
    ]
    test = [{"text_only_answer": "green"}]
    encoder, reason = validate_outer_labels(train, test, "text_only_answer")
    assert encoder is None
    assert reason["type"] == "UnseenOuterTestClasses"


def test_current_answer_only_baseline_and_four_task_contract() -> None:
    assert set(PROBE_TASKS) == {
        "ac_text_answer",
        "ac_image_answer",
        "panl_text_answer",
        "panl_image_answer",
    }
    X = np.asarray([["blue"], ["yellow"], ["blue"], ["yellow"]], dtype=object)
    y = np.asarray([0, 1, 0, 1])
    model = build_current_answer_baseline().fit(X, y)
    predicted = model.predict(np.asarray([["blue"], ["unknown"]], dtype=object))
    probabilities = model.predict_proba(
        np.asarray([["blue"], ["unknown"]], dtype=object)
    )
    assert predicted.shape == (2,)
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
