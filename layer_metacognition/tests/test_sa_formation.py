from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from layer_metacognition.conversation_builder import render_continued_assistant
from confidence_test.source_attribution_analyzer import parse_joint_answer_source_output
from layer_metacognition.sa_formation.core import (
    GateDecision,
    SAFormationArtifacts,
    SAOOFDirectionRepository,
    assert_endpoint_evidence_equal,
    assert_policy_no_verbal_sa,
    coordinate_delta,
    decide_gate,
    fit_oof_directions,
    initialize_run,
    orthogonal_equal_norm_control,
    ridge_raw_space,
    transplant_delta,
    validate_output_dir,
)
from layer_metacognition.sa_formation.runtime import (
    PreparedMeasurement,
    append_exact_token_ids,
    assistant_message,
    build_history_messages,
    source_prefix_from_generation,
    text_content,
    right_pad_measurement_inputs,
)
from layer_metacognition.sa_formation.followup import (
    _balanced_unique_cases,
    _item_mean_summary,
)
from layer_metacognition.sa_formation.mechanism import (
    _orthogonal_to_span,
    _summarize_label_remapping,
    build_relevance_history_messages,
    label_mappings,
    semantic_class_text,
)
from layer_metacognition.sa_formation.second_order import (
    ProtocolAnalyzer,
    _case_coordinates,
    _summarize_protocol_semantics,
    answer_only_prompt,
    build_answer_history_messages,
    protocol_specs,
)
from layer_metacognition.sa_formation.truth_audit import (
    _js_divergence,
    canonical_leading_answer_tokens,
    common_protocol_prompt,
    common_protocol_specs,
    counterfactual_source_use,
    factorial_contrasts,
    paired_ci_within_equivalence_band,
    posthoc_collapse_nine,
)


def test_ridge_raw_space_matches_scaled_prediction_and_direction_sign() -> None:
    rng = np.random.default_rng(42)
    hidden = rng.normal(size=(100, 5))
    target = hidden @ np.asarray([0.3, -0.2, 0.0, 0.5, 0.1]) + 0.7
    scaler = StandardScaler().fit(hidden)
    ridge = RidgeCV(alphas=[0.1, 1, 10]).fit(scaler.transform(hidden), target)
    raw, intercept = ridge_raw_space(scaler, ridge)
    assert np.allclose(ridge.predict(scaler.transform(hidden)), hidden @ raw + intercept)
    assert np.corrcoef(hidden @ (raw / np.linalg.norm(raw)), target)[0, 1] > 0


def test_coordinate_transplant_clamp_and_orthogonal_equal_norm() -> None:
    unit = np.asarray([3.0, 4.0, 0.0]) / 5.0
    recipient = np.asarray([1.0, 2.0, 3.0])
    donor = np.asarray([-2.0, 5.0, 1.0])
    delta = transplant_delta(recipient, donor, unit)
    assert np.isclose((recipient + delta) @ unit, donor @ unit)
    clamp = coordinate_delta(recipient, unit, 1.25)
    assert np.isclose((recipient + clamp) @ unit, 1.25)
    control = orthogonal_equal_norm_control(unit, np.linalg.norm(delta), seed_material="case")
    assert abs(control @ unit) < 1e-10
    assert np.isclose(np.linalg.norm(control), np.linalg.norm(delta))
    assert np.allclose(control, orthogonal_equal_norm_control(unit, np.linalg.norm(delta), seed_material="case"))


def test_three_level_gate_distinguishes_projection_from_intervention() -> None:
    level1 = decide_gate(True, {"coordinate_effective": True})
    level2 = decide_gate(True, {"coordinate_effective": False})
    level3 = decide_gate(False, {"coordinate_effective": True})
    assert (level1.level, level1.allow_causal_mediator, level1.allow_policy_steering) == (1, True, True)
    assert (level2.level, level2.run_natural_formation, level2.allow_causal_mediator) == (2, True, False)
    assert (level3.level, level3.run_natural_formation) == (3, False)


def test_protocol_rank_agreement_does_not_imply_coordinate_equivalence() -> None:
    equivalent = {"paired_difference": {"ci95": [-0.01, 0.02]}}
    shifted = {"paired_difference": {"ci95": [0.20, 0.30]}}
    assert paired_ci_within_equivalence_band(equivalent, 0.05)
    assert not paired_ci_within_equivalence_band(shifted, 0.05)


class FallbackProcessor:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, continue_final_message=None):
        del tokenize
        if continue_final_message:
            raise TypeError("unsupported")
        rendered = "".join(f"<{message['role']}>" + message["content"][0]["text"] for message in messages)
        return rendered + ("<assistant>" if add_generation_prompt else "")


def test_renderer_fallback_preserves_all_history_turns() -> None:
    messages = [
        {"role": "user", "content": text_content("first")},
        assistant_message("old answer"),
        {"role": "user", "content": text_content("final")},
        assistant_message("**Answer**:"),
    ]
    rendered = render_continued_assistant(FallbackProcessor(), messages, "**Answer**:")
    assert "first" in rendered and "old answer" in rendered and "final" in rendered
    assert rendered.endswith("**Answer**:")


def test_source_prefix_reconstructs_exact_generated_separator() -> None:
    assert source_prefix_from_generation("**Answer**: blue\n**Source Attribution**: 6", "6").endswith(": ")
    assert source_prefix_from_generation("**Answer**: blue\n**Source Attribution**:6", "6").endswith(":")
    assert source_prefix_from_generation("**Answer**: blue\n**Source Attribution**:<6>", "6").endswith("<")
    assert parse_joint_answer_source_output("**Answer**: blue\n**Source Attribution**:<6>") == ("blue", "6", True)
    assert parse_joint_answer_source_output("**Answer**: blue\n**Source Attribution**:<6") == ("blue", "6", True)
    assert parse_joint_answer_source_output("**Answer**: blue\n**Source Attribution**:**6**") == ("blue", "6", True)
    assert parse_joint_answer_source_output("**Answer**: blue\naddCriterion**:6") == ("blue", "6", True)
    assert parse_joint_answer_source_output("**Answer**: blue\naddCriterionation**:6") == (None, None, False)


def test_final_evidence_and_policy_leakage_invariants() -> None:
    left = [{"role": "user", "content": text_content("old")}, {"role": "user", "content": text_content("same final")}, assistant_message("x")]
    right = [{"role": "user", "content": text_content("different old")}, {"role": "user", "content": text_content("same final")}, assistant_message("x")]
    assert_endpoint_evidence_equal(left, right)
    bad = [assistant_message("**Source Attribution**:6"), {"role": "user", "content": text_content("policy")}, assistant_message("**Source Choice**:")]
    with pytest.raises(ValueError, match="leaks"):
        assert_policy_no_verbal_sa(bad)
    good = [assistant_message("**Answer**: blue"), {"role": "user", "content": text_content("policy")}, assistant_message("**Source Choice**:")]
    assert_policy_no_verbal_sa(good)


def test_shape_matching_padding_preserves_valid_prefix() -> None:
    inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]]), "pixel_values": torch.ones(2, 2)}
    prepared = PreparedMeasurement([], "", inputs, "", "x", 1, 2, "hash")
    right_pad_measurement_inputs(prepared, 5, pad_token_id=99)
    assert prepared.inputs["input_ids"].tolist() == [[1, 2, 3, 99, 99]]
    assert prepared.inputs["attention_mask"].tolist() == [[1, 1, 1, 0, 0]]
    assert torch.equal(prepared.inputs["pixel_values"], torch.ones(2, 2))


def test_exact_generated_token_append_preserves_multimodal_inputs() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "pixel_values": torch.ones(2, 2),
    }
    append_exact_token_ids(inputs, [7, 8, 9])
    assert inputs["input_ids"].tolist() == [[1, 2, 3, 7, 8, 9]]
    assert inputs["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1]]
    assert torch.equal(inputs["pixel_values"], torch.ones(2, 2))


def test_followup_item_weighting_and_unique_balanced_selection() -> None:
    pairs = [
        {"item_id": "1", "delta": 1.0},
        {"item_id": "1", "delta": 3.0},
        {"item_id": "2", "delta": 6.0},
    ]
    summary = _item_mean_summary(pairs, "delta")
    assert summary["n"] == 2
    assert summary["unique_items"] == 2
    assert summary["mean"] == 4.0
    rows = []
    for item in range(10):
        for prior in range(2):
            rows.append(
                {
                    "case_id": f"{item}_{prior}",
                    "item_id": str(item),
                    "prior_index": prior,
                    "condition": "conflict_easy" if item % 2 else "conflict_hard",
                    "difficulty": "easy" if item % 2 else "hard",
                    "fold": item % 5,
                    "decision_side": "follows_text" if item % 2 else "follows_image",
                    "final_answer": "red",
                    "text_answer": "red",
                    "image_answer": "blue",
                }
            )
    selected = _balanced_unique_cases(rows, 8)
    assert len(selected) == 8
    assert len({row["item_id"] for row in selected}) == 8


def test_output_protection_resume_and_fingerprint(tmp_path: Path) -> None:
    experiment = tmp_path / "base"
    experiment.mkdir()
    output = validate_output_dir(experiment, experiment / "stage3_sa_formation")
    initialize_run(output, {"a": 1}, resume=False)
    with pytest.raises(FileExistsError):
        initialize_run(output, {"a": 1}, resume=False)
    initialize_run(output, {"a": 1}, resume=True)
    with pytest.raises(ValueError, match="fingerprint"):
        initialize_run(output, {"a": 2}, resume=True)
    with pytest.raises(ValueError):
        validate_output_dir(experiment, experiment / "stage2_bad")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_oof_item_exclusion_and_training_only_sigma(tmp_path: Path) -> None:
    base = tmp_path / "base"
    hidden_dir = base / "hidden_states"
    hidden_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    hidden = rng.normal(size=(20, 1, 1, 3)).astype(np.float32)
    case_ids = [f"{index}__prior_0__conflict_easy__v4__joint" for index in range(20)]
    shard = hidden_dir / "shard.pt"
    torch.save({"case_ids": case_ids, "layer_indices": [18], "position_names": ["panl"], "hidden_states": torch.from_numpy(hidden)}, shard)
    index_cases = {case_id: {"shard_path": "hidden_states/shard.pt", "offset": index} for index, case_id in enumerate(case_ids)}
    _write_json(hidden_dir / "index.json", {"cases": index_cases})
    manifests = []
    results = []
    item_to_fold = {}
    for index, case_id in enumerate(case_ids):
        fold = index % 5
        item_to_fold[str(index)] = fold
        reference = {"shard_path": "hidden_states/shard.pt", "offset": index}
        manifests.append({"case_id": case_id, "item_id": str(index), "prior_index": 0, "condition": "conflict_easy", "version": "v4", "answer_classes": ["red", "blue"], "text_only_answer": "red", "image_only_answer": "blue", "current_answer": "blue", "decision_side": "follows_image", "hidden_state_reference": reference})
        score = float(0.5 + 0.1 * hidden[index, 0, 0, 0] - 0.05 * hidden[index, 0, 0, 1])
        results.append({"case_id": case_id, "status": "completed", "generated": {"source_attribution": {"soft_image_score": score, "parsed_label": "4", "class_probabilities": [1.0]}}})
    _write_jsonl(base / "manifest.jsonl", manifests)
    _write_jsonl(base / "results.jsonl", results)
    _write_json(base / "split.json", {"item_to_fold": item_to_fold})
    dummy = base / "dummy"
    _write_json(dummy, {})
    artifacts = SAFormationArtifacts(base, base / "results.jsonl", hidden_dir / "index.json", base / "manifest.jsonl", dummy, dummy, base / "split.json", dummy, dummy, base, dummy, base, dummy)
    output = tmp_path / "out"
    oof, audit = fit_oof_directions(artifacts, output)
    assert len(oof) == 20
    assert all(not row["item_overlap"] for row in audit["fold_audits"])
    repository = SAOOFDirectionRepository(output / "directions")
    for fold in range(5):
        direction = repository.get(fold)
        train_indices = [index for index in range(20) if index % 5 != fold]
        train_z = hidden[train_indices, 0, 0, :] @ direction.d_unit
        assert np.isclose(direction.sigma_z, np.std(train_z, ddof=1))


def test_label_remappings_are_semantically_indexed_and_unordered() -> None:
    mappings = label_mappings()
    assert mappings["normal_numeric"] == tuple(str(index) for index in range(9))
    assert mappings["reversed_numeric"] == tuple(str(index) for index in reversed(range(9)))
    arbitrary = mappings["arbitrary_tokens"]
    assert len(arbitrary) == len(set(arbitrary)) == 9
    assert not all(ord(arbitrary[index]) + 1 == ord(arbitrary[index + 1]) for index in range(8))
    reversed_text = semantic_class_text(mappings["reversed_numeric"])
    assert "8: The answer was based almost entirely on the text clue." in reversed_text
    assert "0: The answer was based almost entirely on the image." in reversed_text


def test_semantic_gate_requires_all_three_mappings() -> None:
    rows = []
    for index in range(25):
        mappings = {}
        for name in label_mappings():
            effect = 0.1 if name != "arbitrary_tokens" else -0.1
            mappings[name] = {
                "arms": {
                    "-2": {
                        "semantic_imageward_score": 0.5 - effect / 2,
                        "raw_numeric_score": 0.4 if "numeric" in name else None,
                    },
                    "2": {
                        "semantic_imageward_score": 0.5 + effect / 2,
                        "raw_numeric_score": 0.6 if "numeric" in name else None,
                    },
                }
            }
        rows.append(
            {
                "status": "completed",
                "item_id": str(index),
                "mappings": mappings,
            }
        )
    summary = _summarize_label_remapping(rows)
    assert summary["effects"]["normal_numeric"]["semantic_supported"]
    assert summary["effects"]["reversed_numeric"]["semantic_supported"]
    assert not summary["effects"]["arbitrary_tokens"]["semantic_supported"]
    assert not summary["semantic_gate"]["passed"]


def test_relevance_history_changes_only_historical_item(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not decoded by message construction")
    condition = SimpleNamespace(resolved_image_path=image)
    target = SimpleNamespace(
        question="target question",
        text_clue="target clue",
        conditions={"conflict_easy": condition},
    )
    donor = SimpleNamespace(
        question="donor question",
        text_clue="donor clue",
        conditions={"conflict_easy": condition},
    )
    relevant = build_relevance_history_messages(
        target,
        "conflict_easy",
        target,
        "conflict_easy",
        "text",
        "blue",
    )
    irrelevant = build_relevance_history_messages(
        target,
        "conflict_easy",
        donor,
        "conflict_easy",
        "text",
        "blue",
    )
    assert relevant[1] == irrelevant[1]
    assert relevant[-1] == irrelevant[-1]
    assert relevant[-2] == irrelevant[-2]
    assert relevant[0] != irrelevant[0]
    assert "target question" in relevant[-2]["content"][-1]["text"]
    assert "target question" in irrelevant[-2]["content"][-1]["text"]


def test_span_orthogonal_random_control_is_deterministic() -> None:
    first = np.asarray([1.0, 0.0, 0.0, 0.0])
    second = np.asarray([0.0, 1.0, 0.0, 0.0])
    control = _orthogonal_to_span([first, second], "case")
    assert np.isclose(np.linalg.norm(control), 1.0)
    assert abs(control @ first) < 1e-10
    assert abs(control @ second) < 1e-10
    assert np.allclose(control, _orthogonal_to_span([first, second], "case"))


def test_second_order_answer_history_has_context_but_no_sa_request(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"message construction does not decode images")
    condition = SimpleNamespace(resolved_image_path=image)
    target = SimpleNamespace(
        question="Which color wins?",
        text_clue="The text says green.",
        conditions={"conflict_easy": condition},
    )
    donor = SimpleNamespace(
        question="What shape is shown?",
        text_clue="The text says square.",
        conditions={"conflict_easy": condition},
    )
    messages = build_answer_history_messages(
        target,
        "conflict_easy",
        donor,
        "conflict_easy",
        "image",
        "green",
    )
    rendered_text = "\n".join(
        str(part.get("text", ""))
        for message in messages
        for part in message["content"]
        if isinstance(part, dict)
    )
    assert len(messages) == 4
    assert "What shape is shown?" in rendered_text
    assert "Which color wins?" in rendered_text
    assert "Source Attribution" not in rendered_text
    assert "**Answer**:" in answer_only_prompt(target)
    assert "Source Attribution" not in answer_only_prompt(target)


def test_second_order_case_id_and_protocol_semantics() -> None:
    assert _case_coordinates("24__prior_3__conflict_easy__v4__joint") == ("24", 3)
    with pytest.raises(ValueError, match="Cannot parse"):
        _case_coordinates("bad-id")
    specs = {spec.name: spec for spec in protocol_specs()}
    assert specs["normal_numeric"].labels_by_semantic == tuple(str(i) for i in range(9))
    assert specs["reversed_numeric"].labels_by_semantic == tuple(str(i) for i in reversed(range(9)))
    assert specs["binary_text_image"].midpoints == (0.05, 0.95)
    assert all(
        tuple(sorted(spec.midpoints)) == spec.midpoints
        for spec in specs.values()
    )


class _SingleTokenTokenizer:
    def encode(self, label, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(label)]


def test_second_order_protocol_analyzer_rejects_token_collisions() -> None:
    binary = protocol_specs()[-1]
    analyzer = ProtocolAnalyzer(_SingleTokenTokenizer(), binary)
    assert analyzer.encodings == {"T": [ord("T")], "I": [ord("I")]}

    class CollisionTokenizer(_SingleTokenTokenizer):
        def encode(self, label, *, add_special_tokens):
            return [1]

    with pytest.raises(ValueError, match="colliding"):
        ProtocolAnalyzer(CollisionTokenizer(), binary)


def test_second_order_semantic_gate_fails_cleanly_for_insufficient_or_constant_data() -> None:
    insufficient = _summarize_protocol_semantics([])
    assert not insufficient["semantic_target_gate"]["passed"]
    names = [spec.name for spec in protocol_specs()]
    rows = []
    for index in range(6):
        rows.append(
            {
                "status": "completed",
                "item_id": str(index),
                "protocols": {
                    name: {
                        "semantic_imageward_score": 0.5,
                        "ridge_prediction": float(index),
                    }
                    for name in names
                },
            }
        )
    constant = _summarize_protocol_semantics(rows)
    assert not constant["semantic_target_gate"]["passed"]
    assert not constant["shared_component"]["all_protocols_nonconstant"]


def test_counterfactual_source_use_algebra_and_orientation() -> None:
    use = counterfactual_source_use(0.8, 0.2, 0.6)
    assert np.isclose(
        use["relative_image_use_log"],
        np.log(0.6) - np.log(0.2),
    )
    assert np.isclose(
        use["relative_image_use_log"],
        use["remove_image_drop_logp"] - use["remove_text_drop_logp"],
    )
    assert use["behavior_imageward_score"] > 0.5


def test_history_factorial_effect_coding_and_congruence_identity() -> None:
    cells = {"text_at": 1.0, "text_ai": 2.0, "image_at": 3.0, "image_ai": 7.0}
    effects = factorial_contrasts(cells)
    assert np.isclose(effects["modality_main"], 3.5)
    assert np.isclose(effects["prior_answer_main"], 2.5)
    assert np.isclose(effects["interaction"], 3.0)
    assert np.isclose(effects["congruence_main"], effects["interaction"] / 2.0)
    with pytest.raises(ValueError, match="exactly"):
        factorial_contrasts({"text_at": 1.0})


class _LeadingSpaceTokenizer:
    def encode(self, label, *, add_special_tokens):
        assert not add_special_tokens
        if not label.startswith(" "):
            return [99, 100]
        return [sum(ord(character) for character in label)]


def test_truth_audit_uses_only_canonical_leading_answer_tokens() -> None:
    tokens = canonical_leading_answer_tokens(
        _LeadingSpaceTokenizer(), ["red", "blue"]
    )
    assert set(tokens) == {"red", "blue"}
    assert tokens["red"] != tokens["blue"]


def test_common_protocol_panel_and_posthoc_collapse() -> None:
    specs = {spec.name: spec for spec in common_protocol_specs()}
    assert len(specs) == 7
    assert specs["common_3_ordered"].labels_by_semantic == ("0", "4", "8")
    assert specs["common_3_reversed"].labels_by_semantic == ("8", "4", "0")
    assert specs["common_2_semantic"].labels_by_semantic == ("T", "I")
    probabilities = np.zeros(9)
    probabilities[0] = 0.2
    probabilities[4] = 0.3
    probabilities[8] = 0.5
    collapsed = posthoc_collapse_nine(probabilities)
    assert np.allclose(collapsed["ternary_probabilities"], [0.2, 0.3, 0.5])
    assert np.allclose(
        collapsed["binary_conditional_probabilities"], [2 / 7, 5 / 7]
    )
    assert np.allclose(collapsed["binary_split_probabilities"], [0.35, 0.65])
    assert np.isclose(_js_divergence([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]), 0.0)


def test_common_protocol_prompt_keeps_shared_wrapper() -> None:
    case = SimpleNamespace(question="Q?", text_clue="T")
    specs = common_protocol_specs()
    prompts = [common_protocol_prompt(case, spec) for spec in specs]
    invariant = "Then report the relative contribution of the text clue and the image"
    assert all(invariant in prompt for prompt in prompts)
    assert all("**Source Attribution**:<CLASS>" in prompt for prompt in prompts)
    assert "0:" in prompts[0] and "8:" in prompts[0]
