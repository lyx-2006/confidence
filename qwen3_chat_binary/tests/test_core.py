from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from qwen3_chat_binary.config import DATASET_PATH, POSITIONS
from qwen3_chat_binary.contracts import ensure_fingerprinted_config, parse_alphas, parse_layers
from qwen3_chat_binary.conversation import history_content, stage2_messages
from qwen3_chat_binary.layout import ensure_output_layout
from qwen3_chat_binary.dataset import PAIR_TYPES, QUESTION_TEMPLATE, load_conflict_cases
from qwen3_chat_binary.prompts import ATTRIBUTION_TEMPLATE, LABELS, LABEL_WEIGHTS, PHASE0_TEMPLATE
from qwen3_chat_binary.scoring import attribution_score
from qwen3_chat_binary import smoke as smoke_module
from qwen3_chat_binary.steering import attribution_side, scaled_direction, shared_manifests
from qwen3_chat_binary.counterfactual_common import cma_scores, load_image_record, render_counterfactual
from dp_SA.prompts import PHASE0_TEMPLATE as ORIGINAL_PHASE0_TEMPLATE


def test_phase0_is_exactly_unchanged_and_stage2_has_one_image() -> None:
    assert PHASE0_TEMPLATE == ORIGINAL_PHASE0_TEMPLATE
    messages = stage2_messages("question", "/tmp/image.png", "**Answer**: blue", "native_boundary")
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[1]["content"][0]["text"] == "**Answer**: blue"
    assert messages[-1]["content"][0]["text"] == "**Source Attribution**:"
    assert sum(
        part.get("type") == "image"
        for message in messages for part in message["content"]
    ) == 1
    assert history_content("**Answer**: blue\n", "explicit_newline") == "**Answer**: blue\n\n"


def test_conflict_dataset_is_complete_case_level_input() -> None:
    cases = load_conflict_cases(DATASET_PATH)
    assert len(cases) == 396
    assert [case.case_id for case in cases] == [f"case_{index:04d}" for index in range(396)]
    assert Counter(case.pair_type for case in cases) == {pair_type: 132 for pair_type in PAIR_TYPES}
    assert len({(case.text_clue, case.image_reference) for case in cases}) == 396
    assert all(case.image_path.is_file() for case in cases)
    assert all(case.text_answer != case.image_answer for case in cases)
    assert cases[0].question == QUESTION_TEMPLATE.format(shape=cases[0].shape)


def test_phase1_is_exactly_v28_five_class_prompt() -> None:
    expected = (
        Path(__file__).resolve().parents[1]
        / "prompt test" / "prompt" / "v28_fiveway_rule_before_labels" / "prompt.txt"
    ).read_text(encoding="utf-8").rstrip("\n")
    assert ATTRIBUTION_TEMPLATE == expected
    assert LABELS == ("0", "1", "2", "3", "4")
    assert LABEL_WEIGHTS == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_five_class_score_is_expected_ordinal_position_and_reports_vocab_mass() -> None:
    logits = torch.zeros(8)
    token_ids = (1, 2, 3, 4, 5)
    logits[list(token_ids)] = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    result = attribution_score(logits, token_ids)
    probabilities = torch.softmax(torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0]), dim=0)
    expected = float((probabilities * torch.tensor(LABEL_WEIGHTS)).sum())
    assert result["image_attribution_score"] == pytest.approx(expected)
    assert result["signed_attribution_score"] == pytest.approx(2 * expected - 1)
    assert result["predicted_label"] == "4"
    assert result["predicted_side"] == "image"
    assert set(result["label_logits"]) == set(LABELS)
    assert 0 < result["label_probability_mass"] < 1


def test_attribution_side_uses_continuous_five_class_score() -> None:
    assert attribution_side({"image_attribution_score": 0.49}) == "text"
    assert attribution_side({"image_attribution_score": 0.5}) == "tie"
    assert attribution_side({"image_attribution_score": 0.51}) == "image"


def test_direction_has_three_percent_mean_residual_norm() -> None:
    high = np.asarray([[3.0, 4.0], [4.0, 3.0]], dtype=np.float32)
    low = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    vector, metadata = scaled_direction(high, low)
    assert np.linalg.norm(vector) == pytest.approx(metadata["mean_residual_norm"] * 0.03)


def test_counterfactual_cma_formula() -> None:
    result = cma_scores(0.9, 0.3, 0.7, 0.1)
    assert result["phi_image"] == pytest.approx(0.6)
    assert result["phi_text"] == pytest.approx(0.2)
    assert result["image_share"] == pytest.approx(0.75)
    assert result["cma_signed"] == pytest.approx(0.5)


def test_counterfactual_rerender_is_pixel_exact_and_target_only(tmp_path) -> None:
    case = load_conflict_cases(DATASET_PATH, max_samples=1)[0]
    record = load_image_record(case)
    excluded = {case.text_answer, case.image_answer}
    distractor_colors = {obj["color"] for obj in record["layout"]["objects"] if obj.get("role") != "target"}
    third = next(color for color in ("red", "orange", "yellow", "green", "blue", "cyan", "purple", "pink", "brown", "white", "black", "gray") if color not in excluded | distractor_colors)
    audit = render_counterfactual(case, third, tmp_path / case.case_id)
    assert audit["original_reproduced_pixel_exact"] is True
    assert audit["sharp_changes_within_visible_target"] is True
    assert audit["masks_equal"] is True
    assert Path(audit["counterfactual_image"]).is_file()


def test_contract_validation_and_resume_fingerprint(tmp_path) -> None:
    assert parse_layers([8, 35]) == (8, 35)
    assert parse_alphas([-2, 0, 2]) == (-2.0, 0.0, 2.0)
    with pytest.raises(ValueError):
        parse_layers([7])
    path = tmp_path / "config.json"
    ensure_fingerprinted_config(path, {"x": 1}, resume=False, label="Test")
    ensure_fingerprinted_config(path, {"x": 1}, resume=True, label="Test")
    with pytest.raises(ValueError):
        ensure_fingerprinted_config(path, {"x": 2}, resume=True, label="Test")


def test_shared_smoke_selection_is_case_disjoint_and_common(tmp_path) -> None:
    for variant in ("native_boundary", "explicit_newline"):
        directory = tmp_path / variant / "tables"
        directory.mkdir(parents=True)
        rows = []
        for index in range(30):
            score = index / 29
            rows.append({
                "status": "completed", "variant": variant, "case_id": f"case-{index}",
                "image_attribution_score": score,
                "predicted_label": "4" if score > 0.5 else "0",
                "predicted_side": "image" if score > 0.5 else "text",
            })
        (directory / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    construction, test, summary = shared_manifests(
        tmp_path, variants=("native_boundary", "explicit_newline"), smoke=True
    )
    assert len(construction) == 10
    assert len(test) == 10
    assert {row["case_id"] for row in construction}.isdisjoint({row["case_id"] for row in test})
    assert summary["split_unit"] == "case_id"
    assert summary["common_completed_case_count"] == 30
    assert set(POSITIONS) == {"LAT", "PANL", "PANL+1", "CLE", "SAC"}


def test_formal_selection_uses_extremes_for_direction_and_midpoint_for_test(tmp_path) -> None:
    for variant in ("native_boundary", "explicit_newline"):
        directory = tmp_path / variant / "tables"
        directory.mkdir(parents=True)
        rows = []
        for index in range(240):
            score = (index + 0.5) / 240
            rows.append({
                "status": "completed", "variant": variant, "case_id": f"case-{index:04d}",
                "image_attribution_score": score,
                "predicted_label": "4" if score > 0.5 else "0",
                "predicted_side": "image" if score > 0.5 else "text",
            })
        (directory / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    construction, test, summary = shared_manifests(
        tmp_path, variants=("native_boundary", "explicit_newline"), smoke=False
    )
    assert Counter(row["construction_side"] for row in construction) == {
        "high_text": 25, "high_image": 25,
    }
    assert Counter(row["test_side"] for row in test) == {"text_side": 50, "image_side": 50}
    construction_ids = {row["case_id"] for row in construction}
    assert construction_ids.isdisjoint({row["case_id"] for row in test})
    for side, predicate in (("text_side", lambda score: score < 0.5), ("image_side", lambda score: score > 0.5)):
        chosen = [row for row in test if row["test_side"] == side]
        assert all(predicate(float(row["image_attribution_score"])) for row in chosen)
        distances = [float(row["distance_from_midpoint"]) for row in chosen]
        assert max(distances) < 0.32
    assert summary["test_rule"].startswith("closest")


def test_output_layout_has_experiment_and_variant_sections(tmp_path) -> None:
    variants = ("native_boundary", "explicit_newline")
    ensure_output_layout(tmp_path, variants)
    for section in ("progress", "figures", "tables"):
        assert (tmp_path / section).is_dir()
        for variant in variants:
            assert (tmp_path / variant / section).is_dir()


def test_successful_smoke_deletes_temporary_output(tmp_path, monkeypatch) -> None:
    def fake_capture(**kwargs):
        (kwargs["output_root"] / "tables").mkdir(parents=True)
        return {"status": "complete"}

    def fake_steering(**kwargs):
        (kwargs["output_root"] / "tables").mkdir(parents=True)
        return {"status": "complete"}

    monkeypatch.setattr(smoke_module, "run_capture", fake_capture)
    monkeypatch.setattr(smoke_module, "run_steering", fake_steering)
    monkeypatch.setattr(smoke_module, "_validate_capture_smoke", lambda *_args: None)
    smoke_module.run_smoke(output_root=tmp_path, model_path=tmp_path / "model")
    assert not (tmp_path / "_smoke").exists()


def test_failed_smoke_retains_temporary_output(tmp_path, monkeypatch) -> None:
    def fake_capture(**kwargs):
        (kwargs["output_root"] / "tables").mkdir(parents=True)
        raise RuntimeError("expected smoke failure")

    monkeypatch.setattr(smoke_module, "run_capture", fake_capture)
    with pytest.raises(RuntimeError, match="expected smoke failure"):
        smoke_module.run_smoke(output_root=tmp_path, model_path=tmp_path / "model")
    retained = list((tmp_path / "_smoke").glob("run-*/Capture/tables"))
    assert len(retained) == 1
