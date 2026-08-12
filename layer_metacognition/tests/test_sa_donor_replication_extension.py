from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from layer_metacognition.run_sa_donor_replication_extension import validate_output
from layer_metacognition.sa_formation.core import stable_hash
from layer_metacognition.sa_formation.donor_replication_extension import (
    NEW_CONDITIONS,
    _donor_reuse_audit,
    _method_v2_full_messages_hash,
    apply_full_margin_protocol_repair,
    extension_condition_sources,
    fit_full_margin_protocol_repair,
    measure_extension_case,
    select_extension_donors,
    summarize_extension,
    verify_development_allowed,
)


def _case(
    item: str,
    image: Path,
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
            "conflict_easy": SimpleNamespace(resolved_image_path=image),
            "null": SimpleNamespace(resolved_image_path=image),
        },
    )


def _target(item: str = "1") -> dict[str, object]:
    return {
        "case_id": f"{item}_case",
        "item_id": item,
        "prior_index": 0,
        "condition": "conflict_easy",
        "fold": 2,
        "difficulty": "easy",
        "final_image": 1,
        "text_answer": "red",
        "image_answer": "blue",
    }


def test_extension_donors_are_next_distinct_v2_rank_and_stable(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"x")
    target = _target()
    candidates = [target]
    cases = {("1", 0): _case("1", image)}
    for item, clue_length in (("2", 10), ("3", 12), ("4", 9), ("5", 13), ("6", 8)):
        row = {**target, "case_id": f"{item}_case", "item_id": item}
        candidates.append(row)
        cases[(item, 0)] = _case(item, image, clue="x" * clue_length)
    old = {
        "donor1_case_id": "2_case",
        "donor1_item_id": "2",
        "donor2_case_id": "3_case",
        "donor2_item_id": "3",
    }
    forward = select_extension_donors(target, candidates, cases, old)
    reverse = select_extension_donors(target, list(reversed(candidates)), cases, old)
    assert [row["item_id"] for row in forward] == ["4", "5"]
    assert [row["case_id"] for row in forward] == [row["case_id"] for row in reverse]
    assert len({"1", "2", "3", *(str(row["item_id"]) for row in forward)}) == 5


def test_extension_replacements_are_symmetric(tmp_path: Path) -> None:
    images = {}
    for item in ("1", "4", "5"):
        images[item] = tmp_path / f"{item}.png"
        images[item].write_bytes(item.encode())
    target = _target()
    d3 = {**target, "case_id": "4_case", "item_id": "4"}
    d4 = {**target, "case_id": "5_case", "item_id": "5"}
    cases = {
        ("1", 0): _case("1", images["1"]),
        ("4", 0): _case("4", images["4"], clue="donor three"),
        ("5", 0): _case("5", images["5"], clue="donor four"),
    }
    sources = extension_condition_sources(cases[("1", 0)], target, (d3, d4), cases)
    assert tuple(sources) == NEW_CONDITIONS
    for index, donor in ((3, d3), (4, d4)):
        assert sources[f"replace_text_d{index}"]["text_source_item"] == donor["item_id"]
        assert sources[f"replace_image_d{index}"]["image_source_item"] == donor["item_id"]
        assert sources[f"replace_text_d{index}"]["image_source_item"] == "1"
        assert sources[f"replace_image_d{index}"]["text_source_item"] == "1"


def test_measurement_reuses_answer_and_hash_and_never_requests_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = {}
    for item in ("1", "4", "5"):
        images[item] = tmp_path / f"{item}.png"
        images[item].write_bytes(item.encode())
    target = _target()
    d3 = {
        **target,
        "case_id": "4_case",
        "item_id": "4",
        "text_answer": "red",
        "image_answer": "green",
    }
    d4 = {
        **target,
        "case_id": "5_case",
        "item_id": "5",
        "text_answer": "green",
        "image_answer": "blue",
    }
    cases = {
        ("1", 0): _case("1", images["1"]),
        ("4", 0): _case("4", images["4"], clue="donor three"),
        ("5", 0): _case("5", images["5"], clue="donor four"),
    }

    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return {" red": [1], " blue": [2], " green": [3]}[text]

    runtime = SimpleNamespace(generator=SimpleNamespace(tokenizer=Tokenizer()))
    calls = []
    probabilities = iter((0.8, 0.4, 0.6, 0.3))

    def fake_pre(_runtime, _case_value, source, answer, _ids, *, capture_hidden):
        calls.append((source, answer, capture_hidden))
        probability = next(probabilities)
        return {
            "answer_class_probabilities": {
                "red": probability,
                "blue": (1.0 - probability) / 2,
                "green": (1.0 - probability) / 2,
            },
            "verbal_sa_leakage": False,
        }, None

    monkeypatch.setattr(
        "layer_metacognition.sa_formation.donor_replication_extension._pre_answer_condition",
        fake_pre,
    )
    old = {
        "case_id": "1_case",
        "answer_star": "red",
        "selection_rendered_hash": "stored-rendered-hash",
        "selection": {
            "messages_hash": _method_v2_full_messages_hash(cases[("1", 0)], target)
        },
    }
    measured = measure_extension_case(runtime, target, old, (d3, d4), cases)
    assert len(calls) == 4
    assert all(answer == "red" and capture is False for _, answer, capture in calls)
    assert measured["selection_reused_without_forward"]
    assert measured["full_messages_hash_equal"]
    assert measured["hidden_captured"] is False
    assert np.isclose(measured["behavior_replace_imageward_d3"], np.log(2.0))
    assert np.isclose(measured["behavior_replace_imageward_d4"], np.log(2.0))
    assert measured["donor3_text_matches_answer_star"]
    assert measured["donor4_image_matches_answer_star"] is False


def _summary_rows(n: int = 75) -> list[dict[str, object]]:
    rows = []
    for index in range(n):
        latent = (index - n / 2) / 7.0
        donors = [f"donor_{(4 * index + offset) % 29}" for offset in range(4)]
        sources = {}
        for donor_index in (3, 4):
            sources[f"replace_text_d{donor_index}"] = {
                "text_source_item": donors[donor_index - 1],
                "image_source_item": str(index),
            }
            sources[f"replace_image_d{donor_index}"] = {
                "text_source_item": str(index),
                "image_source_item": donors[donor_index - 1],
            }
        rows.append(
            {
                "case_id": f"case_{index}",
                "item_id": str(index),
                "fold": index % 5,
                "answer_star_side": "image" if index % 2 else "text",
                "full_margin": 0.25 + (index % 12),
                "behavior_delete_imageward": latent + 0.01 * (index % 3),
                "behavior_replace_imageward_d12_mean": latent - 0.01 * (index % 5),
                "behavior_replace_imageward_d34_mean": latent + 0.01 * (index % 7),
                "behavior_replace_imageward_d3": latent + 0.02,
                "behavior_replace_imageward_d4": latent - 0.02,
                "behavior_replace_imageward_d3_minus_d4": 0.04,
                "donor_match_asymmetry_d3_minus_d4": (index % 3) - 1,
                "graded_residual_delete": -latent,
                "legacy_graded_residual_replace_d34": latent,
                "margin_repair_residual_deletion": latent,
                "margin_repair_residual_replacement": latent + 0.02,
                "verbal_sa_leakage": False,
                "answer_star_reused": True,
                "full_messages_hash_equal": True,
                "selection_reused_without_forward": True,
                "hidden_captured": False,
                **{
                    f"donor{donor_index}_item_id": donors[donor_index - 1]
                    for donor_index in (1, 2, 3, 4)
                },
                "condition_sources": sources,
            }
        )
    return rows


def test_primary_extension_gate_ignores_failed_legacy_graded_diagnostic() -> None:
    rows = _summary_rows()
    summary = summarize_extension(
        rows,
        split="confirmatory",
        expected_n=len(rows),
        failed=0,
    )
    assert summary["technical"]["gate_passed"]
    assert summary["donor_split_half"]["gate_passed"]
    assert summary["raw_cross_method_replication"]["gate_passed"]
    assert summary["extension_gate_passed"]
    assert summary["legacy_graded_secondary"]["gate_bearing"] is False
    assert summary["legacy_graded_secondary"]["deletion_vs_fresh_m34"]["spearman"] < 0
    assert summary["dependence_sensitivity"]["m12_vs_m34"][
        "donor_cluster_bootstrap"
    ]["gate_bearing"] is False
    assert summary["donor_reuse_audit"]["fresh_pair"]["maximum_reuse"] > 1
    assert "choice-coupled" in summary["claim"]


def test_full_margin_protocol_repair_is_fingerprinted_and_non_gate_bearing() -> None:
    development = []
    for index in range(75):
        latent = index / 10.0
        development.append(
            {
                "item_id": str(index),
                "fold": index % 5,
                "answer_star": "blue" if index % 2 else "red",
                "answer_star_side": "image" if index % 2 else "text",
                "difficulty": "hard" if index % 3 else "easy",
                "prior_strength": (index % 7) / 10.0,
                "full_margin": 0.5 + index % 9,
                "behavior_delete_imageward": latent,
                "behavior_replace_imageward": 0.8 * latent,
                "behavior_replace_imageward_d34_mean": 0.9 * latent,
            }
        )
    specification = {
        "answer_vocabulary": ["blue", "red"],
        "answer_reference": "blue",
        "feature_names": [
            "intercept",
            "choice_image",
            "choice_other",
            "difficulty_hard",
            "prior_strength",
            "answer=red",
        ],
    }
    repair = fit_full_margin_protocol_repair(
        development,
        {"nuisance": specification},
    )
    assert repair["gate_bearing"] is False
    assert repair["feature_names"][-1] == "full_margin"
    scored = apply_full_margin_protocol_repair(development[0], repair)
    assert scored["margin_repair_calibration_fingerprint"] == repair[
        "calibration_fingerprint"
    ]
    tampered = json.loads(json.dumps(repair))
    tampered["folds"]["0"]["methods"]["deletion"]["beta"][0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        apply_full_margin_protocol_repair(development[0], tampered)


def test_development_requires_passing_frozen_extension_rule(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="completed confirmatory"):
        verify_development_allowed(tmp_path)
    payload = {"development_allowed": True}
    payload["rule_fingerprint"] = stable_hash(payload)
    (tmp_path / "frozen_extension_rule.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    assert verify_development_allowed(tmp_path)["development_allowed"] is True
    payload["development_allowed"] = False
    unsigned = dict(payload)
    unsigned.pop("rule_fingerprint")
    payload["rule_fingerprint"] = stable_hash(unsigned)
    (tmp_path / "frozen_extension_rule.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="prohibited"):
        verify_development_allowed(tmp_path)


def test_donor_reuse_audit_and_fixed_output(tmp_path: Path) -> None:
    audit = _donor_reuse_audit(_summary_rows())
    assert audit["all_four"]["assignments"] == 4 * 75
    assert audit["fresh_pair"]["reused_donors"] > 0
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    expected = (
        experiment
        / "stage3_sa_computational_bridge"
        / "02_donor_replication_extension"
    )
    assert validate_output(experiment) == expected.resolve()
    with pytest.raises(ValueError, match="fixed"):
        validate_output(experiment, str(experiment / "stage3_sa_truth_audit"))
