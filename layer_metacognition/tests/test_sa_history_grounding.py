from __future__ import annotations

from types import SimpleNamespace

from layer_metacognition.sa_formation.history_grounding import (
    answer_only_prompt_with_text,
    build_history_perturbation_messages,
    contains_sa_request,
    equivalence_to_zero,
    history_difference_in_differences,
)


def _case() -> SimpleNamespace:
    condition = SimpleNamespace(resolved_image_path="/tmp/original.png")
    return SimpleNamespace(
        question="Which color?",
        text_clue="original clue",
        conditions={"conflict_easy": condition},
    )


def test_history_difference_in_differences_algebra() -> None:
    text = {
        "no_text": -1.0,
        "no_image": -3.0,
        "replace_text": -2.0,
        "replace_image": -5.0,
    }
    image = {
        "no_text": -0.5,
        "no_image": -4.0,
        "replace_text": -1.0,
        "replace_image": -5.5,
    }
    result = history_difference_in_differences(text, image)
    assert result["delete_text_first"] == 2.0
    assert result["delete_image_first"] == 3.5
    assert result["delta_history_delete"] == 1.5
    assert result["replace_text_first"] == 3.0
    assert result["replace_image_first"] == 4.5
    assert result["delta_history_replace"] == 1.5


def test_answer_only_history_perturbation_preserves_history_and_has_no_sa_request(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "layer_metacognition.sa_formation.history_grounding.image_content",
        lambda path, text: [
            {"type": "image", "image": path},
            {"type": "text", "text": text},
        ],
    )
    monkeypatch.setattr(
        "layer_metacognition.sa_formation.second_order.image_content",
        lambda path, text: [
            {"type": "image", "image": path},
            {"type": "text", "text": text},
        ],
    )
    case = _case()
    original = build_history_perturbation_messages(
        protocol="answer_only",
        target_case=case,
        target_condition="conflict_easy",
        modality="text",
        prior_answer="red",
        text_clue="original clue",
        image_path="/tmp/original.png",
    )
    perturbed = build_history_perturbation_messages(
        protocol="answer_only",
        target_case=case,
        target_condition="conflict_easy",
        modality="text",
        prior_answer="red",
        text_clue="[No text clue available.]",
        image_path="/tmp/original.png",
    )
    assert original[:2] == perturbed[:2]
    assert original[-2] != perturbed[-2]
    assert not contains_sa_request(original)
    assert not contains_sa_request(perturbed)
    assert "[No text clue available.]" in answer_only_prompt_with_text(
        case, "[No text clue available.]"
    )


def test_equivalence_requires_entire_ci_inside_standardized_band() -> None:
    inside = {"ci95": [-0.09, 0.08]}
    outside = {"ci95": [-0.09, 0.12]}
    assert equivalence_to_zero(inside, natural_scale=0.5)["passed"]
    assert not equivalence_to_zero(outside, natural_scale=0.5)["passed"]
