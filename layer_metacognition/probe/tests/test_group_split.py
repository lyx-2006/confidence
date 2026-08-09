from __future__ import annotations

import numpy as np

from layer_metacognition.probe import (
    DECISION_SIDE_LABELS,
    PROBE_CONDITIONS,
    PROBE_TASKS,
    build_probe_tasks,
    normalize_ordered_choices,
)
from layer_metacognition.probe.probe_models import build_current_answer_baseline
from layer_metacognition.probe.split_utils import (
    create_answer_pair_split_assignments,
    create_split_assignments,
    permute_labels_by_unique_key,
    rows_for_answer_pair_outer_fold,
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
                        "eligible_conflict_probe": True,
                        "conflict_label": "consistent",
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


def test_image_filter_uses_explicit_easy_and_hard_conditions() -> None:
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
        probe_conditions=("consistent_easy", "consistent_hard", "conflict_hard"),
    )
    assert filtered
    assert {record["condition"] for record in filtered} == {
        "consistent_easy",
        "consistent_hard",
        "conflict_hard",
    }


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


def test_current_answer_only_baseline_and_default_task_contract() -> None:
    assert normalize_ordered_choices(
        ["conflict_hard", "consistent_easy", "conflict_hard"],
        PROBE_CONDITIONS,
        "conditions",
    ) == ("consistent_easy", "conflict_hard")
    assert set(PROBE_TASKS) == {
        "ac_text_answer",
        "ac_image_answer",
        "panl_text_answer",
        "panl_image_answer",
        "ac_conflict",
        "panl_conflict",
    }
    dynamic = build_probe_tasks(["sac", "ltt"], ["ptnl", "ac"])
    assert list(dynamic) == [
        "ltt_text_answer",
        "ltt_image_answer",
        "sac_text_answer",
        "sac_image_answer",
        "ac_conflict",
        "ptnl_conflict",
    ]
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


def test_decision_tasks_and_fixed_class_direction() -> None:
    tasks = build_probe_tasks(("ac",), ("panl",), ("ptnl", "ac"))
    assert tasks["ptnl_decision_side"] == ("ptnl", "decision_side")
    assert tasks["ac_decision_side"] == ("ac", "decision_side")
    train = [{"decision_side": "follows_image"}, {"decision_side": "follows_text"}]
    test = [{"decision_side": "follows_text"}]
    encoder, reason = validate_outer_labels(train, test, "decision_side")
    assert reason is None
    assert encoder is not None
    assert tuple(encoder.classes_) == DECISION_SIDE_LABELS
    assert encoder.transform(["follows_text", "follows_image"]).tolist() == [0, 1]


def test_answer_pair_split_has_no_pair_or_item_leakage() -> None:
    records = []
    for item in range(8):
        for pair in ("blue||yellow", "green||red"):
            records.append(
                {
                    "case_id": f"{item}-{pair}",
                    "item_id": str(item),
                    "version": "v4",
                    "unordered_answer_pair_key": pair,
                }
            )
    assignment = create_answer_pair_split_assignments(records, n_splits=2, seed=42)
    for fold in range(2):
        train, test, audit = rows_for_answer_pair_outer_fold(
            records,
            assignment["pair_to_fold"],
            fold=fold,
            train_version="v4",
            test_version="v4",
        )
        assert {row["item_id"] for row in train}.isdisjoint(
            {row["item_id"] for row in test}
        )
        assert {row["unordered_answer_pair_key"] for row in train}.isdisjoint(
            {row["unordered_answer_pair_key"] for row in test}
        )
        assert audit["item_overlap_count"] == 0
        assert audit["answer_pair_overlap_count"] == 0
    rows_for_answer_pair_outer_fold,
