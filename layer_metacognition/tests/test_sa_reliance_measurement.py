from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from layer_metacognition.model_adapter import HookedForwardResult
from layer_metacognition.run_sa_reliance_measurement import validate_output
from layer_metacognition.sa_formation.core import stable_hash
from layer_metacognition.sa_formation.reliance_measurement import (
    AmbiguousEndpointError,
    CONFIRMATORY_N,
    PANEL_CONDITIONS,
    POSITIONS,
    _stack_hidden,
    _causal_prefix_audit,
    _safe_record,
    apply_frozen_calibration,
    build_answer_only_messages,
    build_split_manifest,
    condition_sources,
    contains_verbal_sa_request,
    fit_development_calibration,
    fixed_answer_effects,
    nuisance_feature_spec,
    select_two_donors,
    summarize_measurement,
    verify_confirmatory_allowed,
)


class _Inputs(dict):
    def __getattr__(self, key: str):
        return self[key]


def _case(
    item: str,
    *,
    clue: str = "target clue",
    prior_bin: str = "0.4-0.6",
) -> SimpleNamespace:
    return SimpleNamespace(
        item_id=item,
        prior_index=0,
        question="Which color?",
        text_clue=clue,
        prior_bin=prior_bin,
        answer_classes=["red", "blue", "green"],
        conditions={
            "conflict_easy": SimpleNamespace(resolved_image_path=Path(f"/{item}_full.png")),
            "null": SimpleNamespace(resolved_image_path=Path(f"/{item}_null.png")),
        },
    )


def test_answer_only_prompt_has_no_verbal_sa_request(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not-decoded-in-this-test")
    case = _case("1")
    messages = build_answer_only_messages(
        case,
        text_clue=case.text_clue,
        image_path=str(image),
    )
    assert not contains_verbal_sa_request(messages)
    text = messages[0]["content"][1]["text"]
    assert "Output exactly" in text
    assert "**Answer**" in text
    assert "Source Attribution" not in text


def test_two_donor_replacements_are_symmetric() -> None:
    target = {
        "case_id": "1_case",
        "item_id": "1",
        "prior_index": 0,
        "condition": "conflict_easy",
        "fold": 0,
        "difficulty": "easy",
        "final_image": 1,
    }
    donor1 = {**target, "case_id": "2_case", "item_id": "2"}
    donor2 = {**target, "case_id": "3_case", "item_id": "3"}
    cases = {
        ("1", 0): _case("1"),
        ("2", 0): _case("2", clue="donor one"),
        ("3", 0): _case("3", clue="donor two"),
    }
    sources = condition_sources(cases[("1", 0)], target, (donor1, donor2), cases)
    assert tuple(sources) == PANEL_CONDITIONS
    for index, donor in ((1, donor1), (2, donor2)):
        assert sources[f"replace_text_d{index}"]["text_source_item"] == donor["item_id"]
        assert sources[f"replace_image_d{index}"]["image_source_item"] == donor["item_id"]
        assert sources[f"replace_text_d{index}"]["image_source_item"] == target["item_id"]
        assert sources[f"replace_image_d{index}"]["text_source_item"] == target["item_id"]


def test_deterministic_donors_are_distinct_and_same_fold() -> None:
    target = {
        "case_id": "1_case",
        "item_id": "1",
        "prior_index": 0,
        "condition": "conflict_easy",
        "fold": 2,
        "difficulty": "easy",
        "final_image": 1,
    }
    candidates = [target]
    cases = {("1", 0): _case("1")}
    for item, length in (("2", 7), ("3", 8), ("4", 20), ("5", 6)):
        row = {
            **target,
            "case_id": f"{item}_case",
            "item_id": item,
        }
        candidates.append(row)
        cases[(item, 0)] = _case(item, clue="x" * length)
    left = select_two_donors(target, candidates, cases)
    right = select_two_donors(target, list(reversed(candidates)), cases)
    assert [row["case_id"] for row in left] == [row["case_id"] for row in right]
    assert left[0]["item_id"] != left[1]["item_id"]
    assert all(row["fold"] == target["fold"] for row in left)


def test_fixed_answer_effects_use_equal_family_donor_average() -> None:
    probabilities = {
        "full": 0.8,
        "no_text": 0.6,
        "no_image": 0.3,
        "replace_text_d1": 0.5,
        "replace_image_d1": 0.25,
        "replace_text_d2": 0.4,
        "replace_image_d2": 0.2,
    }
    measurements = {
        condition: {"answer_class_probabilities": {"red": value}}
        for condition, value in probabilities.items()
    }
    effect = fixed_answer_effects(measurements, "red")
    assert np.isclose(effect["behavior_delete_imageward"], np.log(2.0))
    assert np.isclose(effect["behavior_replace_imageward_d1"], np.log(2.0))
    assert np.isclose(effect["behavior_replace_imageward_d2"], np.log(2.0))
    assert np.isclose(effect["behavior_replace_imageward"], np.log(2.0))
    assert np.isclose(effect["replacement_donor_disagreement"], 0.0)


def _calibration_rows(n: int = 50) -> list[dict[str, object]]:
    rows = []
    for index in range(n):
        latent = (index - n / 2) / 10.0
        rows.append(
            {
                "case_id": f"case_{index}",
                "item_id": str(index),
                "fold": index % 5,
                "difficulty": "hard" if index % 2 else "easy",
                "prior_strength": (index % 7) / 10.0,
                "answer_star": "blue" if index % 3 else "red",
                "answer_star_side": "image" if index % 3 else "text",
                "behavior_delete_imageward": latent + 0.05 * ((index % 5) - 2),
                "behavior_replace_imageward": 0.8 * latent + 0.03 * ((index % 7) - 3),
                "behavior_replace_imageward_d1": 0.8 * latent + 0.02,
                "behavior_replace_imageward_d2": 0.8 * latent - 0.02,
            }
        )
    return rows


def test_crossfit_calibration_uses_training_items_and_shared_disagreement_algebra() -> None:
    rows = _calibration_rows()
    calibration = fit_development_calibration(rows, ["red", "blue", "green"])
    assert nuisance_feature_spec(["red", "blue", "green"])["answer_reference"] == "blue"
    scored = apply_frozen_calibration(rows, calibration)
    assert len(scored) == len(rows)
    for row in scored:
        assert np.isclose(
            row["reliance_raw_shared"],
            0.5 * (row["raw_z_delete"] + row["raw_z_replace"]),
        )
        assert np.isclose(
            row["reliance_graded_method_disagreement"],
            0.5 * (row["graded_z_delete"] - row["graded_z_replace"]),
        )

    # Altering held-out fold 0 outcomes cannot change fold-0 calibration.
    altered = [dict(row) for row in rows]
    for row in altered:
        if row["fold"] == 0:
            row["behavior_delete_imageward"] = float(row["behavior_delete_imageward"]) + 1000.0
            row["behavior_replace_imageward"] = float(row["behavior_replace_imageward"]) - 1000.0
    changed = fit_development_calibration(altered, ["red", "blue", "green"])
    assert calibration["folds"]["0"] == changed["folds"]["0"]


def test_frozen_calibration_rejects_tampering() -> None:
    calibration = fit_development_calibration(
        _calibration_rows(), ["red", "blue", "green"]
    )
    calibration["folds"]["0"]["methods"]["deletion"]["raw_mean"] += 1.0
    with pytest.raises(ValueError, match="fingerprint"):
        apply_frozen_calibration(_calibration_rows(), calibration)


def test_hidden_contract_is_positions_by_all_layers() -> None:
    hidden_size = 11
    hidden = {
        position: {
            layer: torch.full((hidden_size,), float(layer + position_index))
            for layer in range(28)
        }
        for position_index, position in enumerate(POSITIONS)
    }
    result = HookedForwardResult(hidden_by_name=hidden, logits_by_position={})
    stacked = _stack_hidden(result, 28)
    assert stacked.shape == (2, 28, hidden_size)
    assert stacked.dtype == np.float16
    assert np.all(stacked[1, 4] == 5.0)


def test_teacher_forced_causal_prefix_audit_checks_text_and_vision() -> None:
    pre = _Inputs(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        pixel_values=torch.arange(8).reshape(2, 4),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
    )
    post = _Inputs(
        input_ids=torch.tensor([[1, 2, 3, 9]]),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        pixel_values=pre["pixel_values"].clone(),
        image_grid_thw=pre["image_grid_thw"].clone(),
    )
    assert _causal_prefix_audit(pre, post)["passed"]
    post["pixel_values"][0, 0] = -1
    with pytest.raises(RuntimeError, match="causal prefix changed"):
        _causal_prefix_audit(pre, post)


def test_tied_endpoint_is_a_terminal_structural_exclusion() -> None:
    audit = {"tied_labels": ["red", "blue"], "selection_stage": "full"}

    def operation():
        raise AmbiguousEndpointError("tie", endpoint_audit=audit)

    row = _safe_record(
        {"intervention_key": "k", "measurement_method_version": 2},
        operation,
    )
    assert row["status"] == "excluded"
    assert row["exclusion_reason"] == "tied_natural_endpoint"
    assert row["endpoint_audit"] == audit


def test_measurement_technical_gate_allows_predeclared_tie_exclusion() -> None:
    rows = _calibration_rows(99)
    calibration = fit_development_calibration(rows, ["red", "blue", "green"])
    scored = apply_frozen_calibration(rows, calibration)
    for row in scored:
        row.update(
            {
                "verbal_sa_leakage": False,
                "hidden_shape": [2, 28, 11],
                "hidden_dtype": "float16",
                "selection_measurement_same_forward": True,
                "teacher_forced_causal_prefix_equal": True,
                "teacher_forced_length_path_max_logit_error": 1.75,
                "teacher_forced_length_path_probability_tv": 0.01,
            }
        )
    summary = summarize_measurement(
        scored,
        split="development",
        failed=0,
        excluded=1,
    )
    assert summary["attempted"] == 100
    assert summary["completed"] == 99
    assert summary["excluded"] == 1
    assert summary["technical"]["gate_passed"]


def test_manifest_explains_77_item_confirmatory_cohort() -> None:
    cohort = []
    donors = {}
    for index in range(CONFIRMATORY_N):
        row = {
            "case_id": f"case_{index}",
            "item_id": str(index),
            "prior_index": 0,
            "condition": "conflict_easy",
            "fold": index % 5,
        }
        cohort.append(row)
        donors[row["case_id"]] = (
            {"case_id": f"d1_{index}", "item_id": f"d1_{index}"},
            {"case_id": f"d2_{index}", "item_id": f"d2_{index}"},
        )
    manifest = build_split_manifest("confirmatory", cohort, donors)
    assert manifest["n"] == 77
    assert manifest["selection_audit"]["remaining_unique_after_development"] == 78
    assert manifest["selection_audit"]["excluded_item_ids"] == ["34"]


def test_confirmatory_requires_valid_passing_frozen_rule(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="completed development"):
        verify_confirmatory_allowed(tmp_path)
    payload = {
        "development_measurement_gate_passed": True,
        "confirmatory_allowed": True,
    }
    payload["rule_fingerprint"] = stable_hash(payload)
    (tmp_path / "frozen_measurement_rule.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    assert verify_confirmatory_allowed(tmp_path)["confirmatory_allowed"] is True
    payload["confirmatory_allowed"] = False
    unsigned = dict(payload)
    unsigned.pop("rule_fingerprint")
    payload["rule_fingerprint"] = stable_hash(unsigned)
    (tmp_path / "frozen_measurement_rule.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="gate failed"):
        verify_confirmatory_allowed(tmp_path)


def test_output_is_fixed_and_protects_existing_stages(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    expected = experiment / "stage3_sa_computational_bridge" / "01_actual_source_reliance"
    assert validate_output(experiment) == expected.resolve()
    with pytest.raises(ValueError, match="fixed"):
        validate_output(experiment, str(experiment / "stage3_sa_truth_audit"))
