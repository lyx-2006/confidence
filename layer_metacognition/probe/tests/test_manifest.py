from __future__ import annotations

import json
from pathlib import Path

import pytest

from confidence_test.dataset_utils import ConditionInput, EvaluationCase
from layer_metacognition.probe import HIDDEN_STATE_DEFINITION
from layer_metacognition.probe.build_probe_manifest import build_manifest
from layer_metacognition.probe.generate_unimodal_labels import (
    extract_existing_text_labels,
)


def _case() -> EvaluationCase:
    condition = ConditionInput("consistent_easy", "image.png", "/tmp/image.png", None)
    return EvaluationCase(
        item_id="1",
        item_order=0,
        ground_truth_answer=None,
        text_answer=None,
        conflict_answer=None,
        question="Choose from: blue, yellow.",
        answer_classes=["blue", "yellow"],
        answer_class_error=None,
        prior_index=0,
        prior_bin=None,
        text_clue="A clue",
        record_key="1::0",
        conditions={"consistent_easy": condition},
    )


def _result(answer: str, condition: str = "consistent_easy") -> dict:
    reference = {
        "format_version": 1,
        "layer_indices": [19],
        "position_names": ["ac", "panl"],
        "shard_path": "hidden_states/shard.pt",
        "offset": 0,
        "hidden_size": 4,
        "hidden_state_definition": HIDDEN_STATE_DEFINITION,
    }
    return {
        "case_id": f"1__prior_0__{condition}__v3__joint",
        "item_id": "1",
        "prior_index": 0,
        "condition": condition,
        "version": "v3",
        "attribution_mode": "joint",
        "status": "completed",
        "generated": {
            "initial_answer": answer,
            "initial_answer_result": {
                "parse_success": True,
                "answer_metric_status": "completed",
            },
            "current_answer": answer,
            "current_answer_result": {
                "answer_class_probabilities": {"blue": 0.8, "yellow": 0.2}
            },
        },
        "hidden_state_reference": reference,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_text_only_duplicate_labels_are_normalized_and_consistent(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [_result("Blue"), _result(" blue ")])
    labels, conflicts = extract_existing_text_labels(results, {("1", 0): _case()})
    assert not conflicts
    assert labels[("1", 0)]["text_only_answer"] == "blue"


def test_text_only_conflict_is_reported(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [_result("blue"), _result("yellow")])
    labels, conflicts = extract_existing_text_labels(results, {("1", 0): _case()})
    assert not labels
    assert conflicts[0]["error"]["type"] == "TextOnlyLabelConflict"


def test_manifest_merges_partial_labels_and_excludes_non_easy_images(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    easy = _result("blue")
    hard = _result("blue", condition="conflict_hard")
    _write_jsonl(experiment / "results.jsonl", [easy, hard])
    cases = {}
    for record in (easy, hard):
        ref = dict(record["hidden_state_reference"])
        ref["case_id"] = record["case_id"]
        cases[record["case_id"]] = ref
    index_path = experiment / "hidden_states" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    _write_jsonl(
        experiment / "probe" / "text_only_labels.jsonl",
        [
            {
                "item_id": "1",
                "prior_index": 0,
                "text_only_answer": "Blue",
                "text_only_answer_raw": "Blue",
                "parse_success": True,
                "answer_classes": ["blue", "yellow"],
            }
        ],
    )
    _write_jsonl(
        experiment / "probe" / "image_only_labels.jsonl",
        [
            {
                "item_id": "1",
                "condition": "consistent_easy",
                "image_only_answer": "yellow",
                "image_only_answer_raw": "yellow",
                "parse_success": True,
                "answer_classes": ["blue", "yellow"],
            }
        ],
    )
    manifest, summary = build_manifest(experiment)
    by_condition = {record["condition"]: record for record in manifest}
    assert by_condition["consistent_easy"]["eligible_image_probe"]
    assert not by_condition["conflict_hard"]["eligible_image_probe"]
    assert by_condition["conflict_hard"]["image_only_answer"] is None
    assert summary["image_hard_excluded_count"] == 1
    assert summary["missing_image_label_count"] == 0


def test_manifest_rejects_hidden_reference_disagreement(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    record = _result("blue")
    _write_jsonl(experiment / "results.jsonl", [record])
    index_ref = dict(record["hidden_state_reference"])
    index_ref.update({"case_id": record["case_id"], "offset": 3})
    path = experiment / "hidden_states" / "index.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"cases": {record["case_id"]: index_ref}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="offset"):
        build_manifest(experiment)
