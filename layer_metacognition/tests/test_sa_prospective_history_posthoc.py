from __future__ import annotations

import pytest

from layer_metacognition.sa_formation.prospective_history_posthoc import (
    build_answer_match_rows,
    build_posthoc_factorial_rows,
    contrast_specs,
    run_posthoc_analysis,
    validate_and_normalize_rows,
    within_fold_group_bootstrap,
)
from layer_metacognition.sa_formation.prospective_history_response_stats import BRANCHES


def _synthetic_inputs(n: int = 10) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    endpoints: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for index in range(n):
        answer_star_side = "image" if index % 2 == 0 else "text"
        answer_star = "image-answer" if answer_star_side == "image" else "text-answer"
        endpoints.append(
            {
                "case_id": f"case-{index}",
                "item_id": str(index),
                "fold": index % 5,
                "answer_star": answer_star,
                "answer_star_side": answer_star_side,
            }
        )
        for branch in BRANCHES:
            if branch == "no_history":
                relevance = modality = replay = 0
                replayed_answer = None
                token_count = 100
            else:
                relevance = int(branch.startswith("relevant_"))
                modality = int("_image_" in branch)
                replay = int(branch.endswith("_ai"))
                # Relevant replays the target's AT/AI answers; exactly one
                # replay side equals A*.  Irrelevant has an A* match only for
                # the first two items, reproducing partial paired coverage.
                if relevance:
                    replayed_answer = "image-answer" if replay else "text-answer"
                elif index < 2 and replay:
                    replayed_answer = answer_star
                else:
                    replayed_answer = "unrelated"
                token_count = 110 + 20 * modality + replay
            value = float(index + 2 * relevance + 3 * modality + 4 * replay)
            records.append(
                {
                    "status": "completed",
                    "case_id": f"case-{index}",
                    "item_id": str(index),
                    "fold": index % 5,
                    "branch": branch,
                    "answer_star": answer_star,
                    "answer_star_side": answer_star_side,
                    "A_prediction": value,
                    "V": value / 10.0,
                    "joint_common9": {"input_token_count": token_count},
                    "branch_factors": {"replayed_answer": replayed_answer},
                    "history_match_tier": (
                        "exact_ordered_text_image_answer_pair"
                        if index < 2
                        else "fallback_same_fold_difficulty"
                    ),
                    "history_ordered_pair_exact": index < 2,
                    "history_donor_item_id": f"h-{index}",
                    "donor5_item_id": f"d5-{index}",
                    "donor6_item_id": f"d6-{index}",
                }
            )
    return records, endpoints


def test_contrast_specs_are_exactly_twelve() -> None:
    specs = contrast_specs()
    assert len(specs) == 12
    assert len({row["id"] for row in specs}) == 12
    assert {row["effect"] for row in specs} == {
        "modality",
        "replay",
        "interaction",
        "history_vs_none",
    }


def test_normalization_and_token_factorial_algebra() -> None:
    records, endpoints = _synthetic_inputs()
    normalized = validate_and_normalize_rows(
        records, endpoints, require_frozen_shape=False
    )
    rows = build_posthoc_factorial_rows(normalized)
    assert len(rows) == 10
    row = rows[0]
    assert row["relevant_modality_input_token_count"] == pytest.approx(20.0)
    assert row["relevant_replay_input_token_count"] == pytest.approx(1.0)
    assert row["relevant_interaction_input_token_count"] == pytest.approx(0.0)
    assert row["endpoint_side"] == "image"


def test_answer_match_uses_answer_star_and_reports_partial_coverage() -> None:
    records, endpoints = _synthetic_inputs()
    normalized = validate_and_normalize_rows(
        records, endpoints, require_frozen_shape=False
    )
    relevant, relevant_counts = build_answer_match_rows(
        normalized, relevance="relevant"
    )
    irrelevant, irrelevant_counts = build_answer_match_rows(
        normalized, relevance="irrelevant"
    )
    assert len(relevant) == 10
    assert relevant_counts["items_with_at_least_one_match_n"] == 10
    assert len(irrelevant) == 2
    assert irrelevant_counts["items_with_at_least_one_match_n"] == 2
    assert irrelevant_counts["items_without_any_match_n"] == 8


def test_within_fold_group_bootstrap_is_deterministic() -> None:
    rows = [
        {
            "item_id": str(index),
            "fold": index % 5,
            "endpoint_side": "image" if index % 2 == 0 else "text",
            "x": float(index),
        }
        for index in range(10)
    ]

    def statistic(sample: list[dict[str, object]]) -> float:
        image = [float(row["x"]) for row in sample if row["endpoint_side"] == "image"]
        text = [float(row["x"]) for row in sample if row["endpoint_side"] == "text"]
        return sum(image) / len(image) - sum(text) / len(text)

    first = within_fold_group_bootstrap(
        rows, statistic, group_key="endpoint_side", iterations=50, seed=77
    )
    second = within_fold_group_bootstrap(
        rows, statistic, group_key="endpoint_side", iterations=50, seed=77
    )
    assert first == second
    assert first["valid"] == 50
    assert first["seed"] == 77


def test_end_to_end_pure_analysis_omits_composition_changing_pool() -> None:
    records, endpoints = _synthetic_inputs()
    result = run_posthoc_analysis(
        records,
        endpoints,
        iterations=20,
        require_frozen_shape=False,
    )
    assert result["analysis_counts"]["contrast_n"] == 12
    match = result["answer_match_reinterpretation"]
    assert "all_history" not in match
    assert match["definition"]["all_history_pooled_contrast_reported"] is False
    assert match["relevant_history"]["paired_eligible_item_n"] == 10
    assert match["irrelevant_history"]["paired_eligible_item_n"] == 2


def test_endpoint_manifest_mismatch_is_rejected() -> None:
    records, endpoints = _synthetic_inputs()
    records[0]["answer_star_side"] = "text"
    with pytest.raises(ValueError, match="endpoint/report mismatch"):
        validate_and_normalize_rows(records, endpoints, require_frozen_shape=False)

