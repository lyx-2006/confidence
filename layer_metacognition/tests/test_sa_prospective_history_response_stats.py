from __future__ import annotations

import math

import pytest

from layer_metacognition.sa_formation.prospective_history_response_stats import (
    BRANCHES,
    association_summary,
    build_factorial_rows,
    factorial_match_tier_sensitivity,
    fixed_fold_item_bootstrap,
    holm_adjust,
    icc_consistency,
    leave_one_reused_donor_cluster_out,
    qualification_gate,
)


def _factorial_records(n: int = 10) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(n):
        for branch in BRANCHES:
            if branch == "no_history":
                modality = replay = relevance = 0.0
            else:
                relevance = float(branch.startswith("relevant_"))
                modality = float("_image_" in branch)
                replay = float(branch.endswith("_ai"))
            base = float(index) / 10.0
            # Relevant modality effect is 2, irrelevant is 1; replay effect is
            # 3 and the image×AI interaction is .5 in both relevance strata.
            value = (
                base
                + modality * (1.0 + relevance)
                + replay * 3.0
                + modality * replay * 0.5
            )
            rows.append(
                {
                    "status": "completed",
                    "case_id": f"case-{index}",
                    "item_id": str(index),
                    "fold": index % 5,
                    "branch": branch,
                    "history_match_tier": (
                        "exact_ordered_text_image_answer_pair"
                        if index < 2
                        else "fallback_same_fold_difficulty"
                    ),
                    "history_ordered_pair_exact": index < 2,
                    "history_donor_item_id": f"history-{index // 2}",
                    "donor5_item_id": f"d5-{index // 3}",
                    "donor6_item_id": f"d6-{index}",
                    "B_D": value,
                    "B_M56": value,
                    "M5": value,
                    "M6": value,
                    "U_prediction": value,
                    "A_prediction": value,
                    "V": value,
                }
            )
    return rows


def test_factorial_algebra_and_relevance_did() -> None:
    rows = build_factorial_rows(
        _factorial_records(),
        outcomes=("B_D", "B_M56", "M5", "M6", "U_prediction", "A_prediction", "V"),
    )
    assert len(rows) == 10
    row = rows[0]
    assert row["relevant_modality_B_D"] == pytest.approx(2.25)
    assert row["irrelevant_modality_B_D"] == pytest.approx(1.25)
    assert row["relevant_replay_B_D"] == pytest.approx(3.25)
    assert row["relevant_interaction_B_D"] == pytest.approx(0.5)
    assert row["did_modality_B_D"] == pytest.approx(1.0)
    assert row["did_replay_B_D"] == pytest.approx(0.0)
    assert row["history_match_tier"] == "exact_ordered_text_image_answer_pair"
    assert row["history_donor_item_id"] == "history-0"


def test_incomplete_case_is_not_silently_analyzed() -> None:
    records = _factorial_records(2)
    records.pop()
    rows = build_factorial_rows(records, outcomes=("B_D",))
    assert [row["case_id"] for row in rows] == ["case-0"]


def test_complete_branch_grid_without_requested_outcome_is_not_counted() -> None:
    rows = build_factorial_rows(_factorial_records(1), outcomes=("missing",))
    assert rows == []


def test_factorial_requires_consistent_frozen_donor_ids() -> None:
    records = _factorial_records(1)
    records[0].pop("donor5_item_id")
    with pytest.raises(ValueError, match="lacks frozen donor identity donor5_item_id"):
        build_factorial_rows(records, outcomes=("B_D",))


def test_fixed_fold_bootstrap_rejects_duplicate_recipient_items() -> None:
    rows = [
        {"item_id": "1", "fold": 0, "x": 1.0},
        {"item_id": "1", "fold": 0, "x": 2.0},
    ]
    with pytest.raises(ValueError, match="one row per recipient item"):
        fixed_fold_item_bootstrap(rows, lambda values: sum(row["x"] for row in values))


def test_constant_association_is_safe() -> None:
    rows = [
        {"item_id": str(index), "fold": index % 5, "x": 1.0, "y": index}
        for index in range(10)
    ]
    summary = association_summary(rows, "x", "y", iterations=20)
    assert summary["spearman"] is None
    assert summary["spearman_ci95"] == [None, None]


def test_icc_and_holm() -> None:
    assert icc_consistency([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]) == pytest.approx(1.0)
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "missing": None})
    assert adjusted == {"a": pytest.approx(0.02), "b": pytest.approx(0.04), "missing": None}


def test_qualification_requires_frozen_nuisance_comparison() -> None:
    rows = []
    for index in range(40):
        target = float(index)
        rows.append(
            {
                "status": "completed",
                "case_id": f"case-{index}",
                "item_id": str(index),
                "fold": index % 5,
                "B_D": target,
                "B_M56": target + 0.01,
                "M5": target + 0.02,
                "M6": target + 0.03,
                "B_target_shared": target,
                "U_prediction": target,
                "A_prediction": target,
                "V": target,
            }
        )
    summary = qualification_gate(rows, iterations=20)
    assert summary["components"]["frozen_nuisance_available"] is False
    assert summary["passed"] is False
    assert math.isclose(summary["metrics"]["M5_M6_icc"], 1.0)


def _qualification_rows(
    *,
    break_behavior: bool = False,
    break_report: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(40):
        # Nonlinear, monotone targets keep every eight-item fold non-constant
        # and make the frozen predictor beat the deliberately bad nuisance.
        target = float(index + (index % 5) * 0.01)
        row: dict[str, object] = {
            "status": "completed",
            "case_id": f"case-{index}",
            "item_id": f"item-{index}",
            "fold": index % 5,
            "B_D": target,
            "B_M56": target + 0.01,
            "M5": target + 0.02,
            "M6": target + 0.03,
            "B_target_shared": target,
            "U_prediction": target,
            "U_nuisance_prediction": -target,
            "A_prediction": target,
            "V": target + 0.04,
            "answer_only_no_sa": True,
            "causal_prefix_equal": True,
            "answer_hook_exactly_once": True,
            "joint_hook_exactly_once": True,
            "steering_applied": False,
        }
        if break_behavior and index == 0:
            row["U_prediction"] = math.nan
        if break_report and index == 0:
            row["A_prediction"] = math.nan
        rows.append(row)
    return rows


def test_qualification_authorizes_both_tracks_with_strict_forty_row_audit() -> None:
    summary = qualification_gate(_qualification_rows(), iterations=40)
    assert summary["authorizations"] == {
        "behavior_readout_history": True,
        "report_formation_history": True,
        "full_four_layer": True,
    }
    assert summary["technical_audit"]["fold_counts"] == {
        str(fold): 8 for fold in range(5)
    }
    assert summary["technical_audit"]["bootstrap_minimum_valid"] == 38
    assert summary["components"]["behavior_technical_complete"] is True
    assert summary["components"]["report_technical_complete"] is True


def test_qualification_tracks_are_independent() -> None:
    behavior_broken = qualification_gate(
        _qualification_rows(break_behavior=True), iterations=40
    )
    assert behavior_broken["authorizations"]["behavior_readout_history"] is False
    assert behavior_broken["authorizations"]["report_formation_history"] is True

    report_broken = qualification_gate(
        _qualification_rows(break_report=True), iterations=40
    )
    assert report_broken["authorizations"]["behavior_readout_history"] is True
    assert report_broken["authorizations"]["report_formation_history"] is False


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        (lambda rows: rows.__setitem__(39, {**rows[39], "item_id": "item-0"}), "item_id_unique"),
        (lambda rows: rows[0].__setitem__("fold", 1), "fold_counts_exact"),
        (lambda rows: rows[0].__setitem__("causal_prefix_equal", False), "required_structural_fields"),
    ),
)
def test_qualification_rejects_protocol_shape_or_structure(mutation, failed_check) -> None:
    rows = _qualification_rows()
    mutation(rows)
    summary = qualification_gate(rows, iterations=20)
    assert summary["passed"] is False
    assert summary["technical_audit"]["checks"][failed_check] is False


def test_match_tier_and_reused_donor_sensitivities_are_descriptive() -> None:
    contrasts = build_factorial_rows(_factorial_records(10), outcomes=("B_D",))
    tiers = factorial_match_tier_sensitivity(
        contrasts, outcomes=("B_D",), iterations=20
    )
    assert tiers["exact_ordered_pair"]["n"] == 2
    assert tiers["fallback"]["n"] == 8
    assert tiers["exact_ordered_pair"]["descriptive_only"] is True
    assert tiers["fallback"]["gate_bearing"] is False

    donors = leave_one_reused_donor_cluster_out(contrasts, outcomes=("B_D",))
    assert donors["descriptive_only"] is True
    assert donors["conditional_on_frozen_cohort"] is True
    assert donors["roles"]["history_donor"]["reused_donor_cluster_n"] == 5
    effect = donors["roles"]["history_donor"]["effects"][
        "relevant_modality_B_D"
    ]
    assert effect["leave_cluster_out_n"] == 5
    assert effect["analysis_n_range"] == [8, 8]


def test_hard_answer_other_is_kept_as_secondary_total_effect() -> None:
    records = _factorial_records(1)
    for index, row in enumerate(records):
        row["hard_answer_side"] = "other" if index == 0 else "image"
    contrasts = build_factorial_rows(
        records, outcomes=("hard_answer_image", "hard_answer_other")
    )
    assert contrasts[0]["no_history_hard_answer_image"] == 0.0
    assert contrasts[0]["no_history_hard_answer_other"] == 1.0
    assert "relevant_modality_hard_answer_other" in contrasts[0]
