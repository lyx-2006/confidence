from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from layer_metacognition import run_sa_prospective_history_response as runner

from layer_metacognition.sa_formation.core import SAFormationArtifacts
from layer_metacognition.sa_formation.prospective_history_readouts import (
    AnswerOnlyReadoutMeasurement,
    JointCommon9Measurement,
    audit_single_token_causal_prefix,
    audit_target_readout_exclusion,
    hidden_checksum_payload,
    load_prospective_readout_repository,
    nuisance_row_for_fixed_answer,
    project_answer_hidden,
)
from layer_metacognition.sa_formation.prospective_history_response_cohort import (
    BRANCHES,
    EVIDENCE_CONDITIONS,
    EXPECTED_FOLD_CONTEXT_COUNTS,
    HARD_SIDE_QUOTAS,
    _latest_rows,
    build_joint_history_messages,
    build_messages,
    build_plan,
    choose_structural_replacement,
    cohort_candidate_manifest,
    history_branch_factors,
    joint_message_audit,
    message_audit,
    protocol_manifest,
    replacement_stratum,
)


EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "Final_v4_run"
    / "answer_basis_9"
)


@pytest.fixture(scope="module")
def real_plan():
    if not EXPERIMENT.is_dir():
        pytest.skip("Frozen Stage-09 source artifacts are not present")
    artifacts = SAFormationArtifacts.discover(EXPERIMENT)
    return build_plan(artifacts)


@pytest.fixture(scope="module")
def real_readout_repository():
    if not EXPERIMENT.is_dir():
        pytest.skip("Frozen Stage-09 readout artifacts are not present")
    return load_prospective_readout_repository(EXPERIMENT)


def test_real_candidate_inventory_and_primary_allocation_are_frozen(real_plan) -> None:
    assert len(real_plan.inventory.rows) == 395
    assert len({row["item_id"] for row in real_plan.inventory.rows}) == 67
    assert Counter(row["fold"] for row in real_plan.inventory.rows) == Counter(
        EXPECTED_FOLD_CONTEXT_COUNTS
    )
    assert len(real_plan.primary_rows) == 40
    assert len({row["item_id"] for row in real_plan.primary_rows}) == 40
    assert Counter(row["fold"] for row in real_plan.primary_rows) == Counter(
        {fold: 8 for fold in range(5)}
    )
    assert Counter(
        (row["fold"], row["difficulty"]) for row in real_plan.primary_rows
    ) == Counter(
        {(fold, difficulty): 4 for fold in range(5) for difficulty in ("easy", "hard")}
    )
    for fold, expected in HARD_SIDE_QUOTAS.items():
        observed = Counter(
            "image" if row["legacy_final_image"] else "text"
            for row in real_plan.primary_rows
            if row["fold"] == fold and row["difficulty"] == "hard"
        )
        assert observed == Counter(expected)


def test_donors_are_panel_external_and_row_distinct(real_plan) -> None:
    candidate_items = {row["item_id"] for row in real_plan.inventory.rows}
    for row in real_plan.all_rows:
        role_items = {
            row["history_donor"]["item_id"],
            row["donor5"]["item_id"],
            row["donor6"]["item_id"],
        }
        assert len(role_items) == 3
        assert row["item_id"] not in role_items
        assert not role_items.intersection(candidate_items)
        assert all(donor["fold"] == row["fold"] for donor in (
            row["history_donor"], row["donor5"], row["donor6"]
        ))
        assert all(donor["difficulty"] == row["difficulty"] for donor in (
            row["history_donor"], row["donor5"], row["donor6"]
        ))


def test_factorial_branch_names_and_congruence_are_explicit(real_plan) -> None:
    assert BRANCHES == (
        "no_history",
        "relevant_text_at",
        "relevant_text_ai",
        "relevant_image_at",
        "relevant_image_ai",
        "irrelevant_text_at",
        "irrelevant_text_ai",
        "irrelevant_image_at",
        "irrelevant_image_ai",
    )
    row = real_plan.primary_rows[0]
    assert history_branch_factors(row, "relevant_text_at")[
        "history_source_congruent_with_replay"
    ] is True
    assert history_branch_factors(row, "relevant_image_at")[
        "history_source_congruent_with_replay"
    ] is False
    irrelevant = history_branch_factors(row, "irrelevant_image_ai")
    assert irrelevant["history_source_congruent_with_replay"] is True
    assert irrelevant["history_item_id"] != row["item_id"]


def test_message_audit_and_final_turn_equality(real_plan) -> None:
    audit = message_audit(real_plan, include_reserve=False)
    assert audit["passed"] is True
    assert audit["failure_n"] == 0
    assert audit["message_cell_n"] == 40 * 9 * 7
    row = real_plan.primary_rows[0]
    for condition in EVIDENCE_CONDITIONS:
        final_hashes = {
            str(build_messages(real_plan, row, branch, condition)[-2:])
            for branch in BRANCHES
        }
        assert len(final_hashes) == 1


def test_joint_common9_messages_have_identical_target_turn(real_plan) -> None:
    endpoints = {
        row["case_id"]: row["text_answer"] for row in real_plan.primary_rows
    }
    audit = joint_message_audit(real_plan, endpoints)
    assert audit["passed"] is True
    assert audit["failure_n"] == 0
    row = real_plan.primary_rows[0]
    final_turns = set()
    for branch in BRANCHES:
        messages, assistant_text = build_joint_history_messages(
            real_plan,
            row,
            branch,
            answer_star=endpoints[row["case_id"]],
        )
        assert messages[-1]["content"][0]["text"] == assistant_text
        assert "**Source Attribution**:" in assistant_text
        final_turns.add(str(messages[-2:]))
    assert len(final_turns) == 1


def test_reserve_cells_and_structural_replacement_are_frozen(real_plan) -> None:
    easy = next(row for row in real_plan.primary_rows if row["difficulty"] == "easy")
    hard = next(row for row in real_plan.primary_rows if row["difficulty"] == "hard")
    assert "legacy_side=" not in replacement_stratum(easy)
    assert "legacy_side=" in replacement_stratum(hard)
    replacement = choose_structural_replacement(
        real_plan,
        easy,
        real_plan.primary_rows,
        {easy["case_id"]},
    )
    remaining_items = {
        row["item_id"]
        for row in real_plan.primary_rows
        if row["case_id"] != easy["case_id"]
    }
    assert replacement["selection_stratum"] == easy["selection_stratum"]
    assert replacement["item_id"] not in remaining_items
    assert replacement["replacement_for_case_id"] == easy["case_id"]


def test_manifests_keep_all_reserves_and_correct_forward_arithmetic(real_plan) -> None:
    cohort = cohort_candidate_manifest(real_plan)
    assert cohort["candidate_n"] == 395
    assert cohort["primary_n"] == 40
    assert cohort["reserve_exact_context_n"] == 355
    planned = protocol_manifest()["planned_cells"]
    assert planned["behavior_primary_total"] == 2520
    assert planned["postanswer_hidden_primary_total_if_all_branches_measured"] == 360
    assert planned["joint_common9_primary_total_if_all_branches_measured"] == 360
    assert planned["formal_forward_primary_total_if_both_tracks_authorized"] == 3240


def test_hidden_checksum_and_causal_prefix_audit_are_deterministic() -> None:
    hidden = {18: np.asarray([1.0, 2.0], dtype=np.float32)}
    assert hidden_checksum_payload(hidden) == hidden_checksum_payload(hidden)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 7]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    audit = audit_single_token_causal_prefix(
        pre_input_ids=torch.tensor([[1, 2]]),
        pre_attention_mask=torch.tensor([[1, 1]]),
        post_inputs=inputs,
        expected_token_id=7,
    )
    assert audit["passed"] is True
    with pytest.raises(RuntimeError, match="causal prefix changed"):
        audit_single_token_causal_prefix(
            pre_input_ids=torch.tensor([[1, 9]]),
            pre_attention_mask=torch.tensor([[1, 1]]),
            post_inputs=inputs,
            expected_token_id=7,
        )


def test_read_only_source_jsonl_is_never_repaired(tmp_path) -> None:
    source = tmp_path / "old_results.jsonl"
    original = '{"case_id":"ok"}\n{"case_id":'
    source.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSONL"):
        _latest_rows(source)
    assert source.read_text(encoding="utf-8") == original


def test_real_frozen_readout_loaders_and_pure_projection(
    real_plan, real_readout_repository
) -> None:
    if not EXPERIMENT.is_dir():
        pytest.skip("Frozen Stage-09 readout artifacts are not present")
    repository = real_readout_repository
    assert {fold: repository.primary_u[fold].layer for fold in range(5)} == {
        0: 16,
        1: 16,
        2: 20,
        3: 20,
        4: 16,
    }
    assert set(repository.secondary_u_l18) == set(range(5))
    assert set(repository.nuisance_baseline) == set(range(5))
    assert all(
        model.replay_max_abs_error == pytest.approx(0.0)
        for model in repository.nuisance_baseline.values()
    )
    readout_audit = repository.audit_manifest()
    assert len(readout_audit["attribution"]["folds"]) == 5
    assert all(
        entry["source_sha256"] == entry["expected_sha256"]
        for entry in readout_audit["attribution"]["folds"]
    )
    assert len(readout_audit["readout_manifest_fingerprint"]) == 64
    row = dict(real_plan.primary_rows[0])
    row["answer_star_side"] = "image"
    nuisance = nuisance_row_for_fixed_answer(
        row,
        fixed_answer=row["image_answer"],
        answer_margin=1.0,
    )
    fold = row["fold"]
    hidden_size = repository.primary_u[fold].raw_direction.size
    hidden = {
        repository.primary_u[fold].layer: np.zeros(hidden_size),
        18: np.zeros(hidden_size),
    }
    projected = project_answer_hidden(hidden, nuisance, repository)
    assert np.isfinite(projected["primary_u"]["frozen_prediction"])
    assert np.isfinite(projected["secondary_u_l18"]["frozen_prediction"])
    assert projected["nuisance_only"]["available"] is True


def test_real_target_items_are_excluded_from_every_frozen_readout_source(
    real_plan, real_readout_repository
) -> None:
    audit = audit_target_readout_exclusion(
        real_plan, real_readout_repository, EXPERIMENT
    )
    assert audit["passed"] is True
    assert len(audit["fingerprint"]) == 64
    assert audit == audit_target_readout_exclusion(
        real_plan.inventory.rows, real_readout_repository, EXPERIMENT
    )
    assert audit["target"]["row_n"] == 395
    assert audit["target"]["unique_item_n"] == 67
    assert audit["base_item_split"]["mismatch_n"] == 0

    stage03 = audit["sources"]["stage03_development_fit"]
    assert stage03["completed_row_n"] == 97
    assert stage03["unique_case_n"] == 97
    assert stage03["item_n"] == 97
    assert stage03["target_overlap_items"] == []

    bridge08 = audit["sources"]["bridge08_train_union"]
    assert bridge08["fold_n"] == 5
    assert bridge08["item_n"] == 97
    assert bridge08["target_overlap_items"] == []

    stage06 = audit["sources"]["stage06_attribution_train_test_union"]
    assert stage06["fold_n"] == 5
    assert stage06["train_union_item_n"] == 80
    assert stage06["test_union_item_n"] == 80
    assert stage06["item_n"] == 80
    assert stage06["target_overlap_items"] == []
    assert all(audit["checks"].values())


def test_target_readout_exclusion_reports_overlap_without_hiding_manifest(
    real_plan, real_readout_repository
) -> None:
    rows = [dict(row) for row in real_plan.inventory.rows]
    replaced_item = rows[0]["item_id"]
    source_item = real_readout_repository.secondary_u_l18[0].audit["train_items"][0]
    expected_fold = next(
        int(row["fold"])
        for row in real_plan.inventory.source_rows
        if str(row["item_id"]) == str(source_item)
    )
    for row in rows:
        if row["item_id"] == replaced_item:
            row["item_id"] = str(source_item)
            row["fold"] = expected_fold
    audit = audit_target_readout_exclusion(rows, real_readout_repository, EXPERIMENT)
    assert audit["target"]["unique_item_n"] == 67
    assert audit["base_item_split"]["mismatch_n"] == 0
    assert audit["passed"] is False
    assert audit["sources"]["stage03_development_fit"]["target_overlap_items"] == [
        str(source_item)
    ]
    assert audit["sources"]["bridge08_train_union"]["target_overlap_items"] == [
        str(source_item)
    ]
    assert audit["checks"]["target_excluded_from_stage03_development"] is False
    assert audit["checks"]["target_excluded_from_bridge08_train_union"] is False


def test_target_readout_exclusion_audits_every_candidate_row_fold(
    real_plan, real_readout_repository
) -> None:
    rows = [dict(row) for row in real_plan.inventory.rows]
    rows[-1]["fold"] = (int(rows[-1]["fold"]) + 1) % 5
    audit = audit_target_readout_exclusion(rows, real_readout_repository, EXPERIMENT)
    assert audit["passed"] is False
    assert audit["base_item_split"]["checked_target_row_n"] == 395
    assert audit["base_item_split"]["mismatch_n"] == 1
    mismatch = audit["base_item_split"]["mismatches"][0]
    assert mismatch["row_index"] == 394
    assert mismatch["reason"] == "fold_mismatch"
    assert audit["checks"]["every_target_row_matches_base_item_split"] is False


def test_runner_track_authorizations_use_only_exact_authoritative_keys() -> None:
    assert runner._gate_authorizations(
        {
            "passed": False,
            "authorizations": {
                "behavior_readout_history": True,
                "report_formation_history": False,
                "full_four_layer": False,
            },
        }
    ) == (True, False)
    assert runner._gate_authorizations(
        {
            "authorizations": {
                "behavior_readout_history": False,
                "report_formation_history": True,
            }
        }
    ) == (False, True)
    with pytest.raises(ValueError, match="exact authorization keys"):
        runner._gate_authorizations(
            {"authorizations": {"behavior_readout": True, "report_formation": True}}
        )


def test_runner_orphan_npz_is_strictly_reused_or_rejected(tmp_path) -> None:
    path = tmp_path / "hidden" / "orphan.npz"
    arrays = {
        "layers": np.asarray([18], dtype=np.int64),
        "hidden": np.asarray([[1.0, 2.0]], dtype=np.float16),
    }
    first = runner._save_hidden_once(path, **arrays)
    second = runner._save_hidden_once(path, **arrays)
    assert first == second
    with pytest.raises(FileExistsError, match="strict orphan validation"):
        runner._save_hidden_once(
            path,
            layers=np.asarray([18], dtype=np.int64),
            hidden=np.asarray([[1.0, 3.0]], dtype=np.float16),
        )
    with pytest.raises(FileExistsError, match="strict orphan validation"):
        runner._save_hidden_once(
            path,
            layers=np.asarray([18], dtype=np.int32),
            hidden=arrays["hidden"],
        )


def test_runner_terminal_failed_track_resume_never_requires_hidden(tmp_path) -> None:
    terminal = {
        "status": "failed",
        "terminal_track_status": "failed",
        "identical_attempt_n": 2,
    }
    runner._validate_terminal_track(tmp_path, terminal)
    with pytest.raises(ValueError, match="exactly two"):
        runner._validate_terminal_track(
            tmp_path,
            {**terminal, "identical_attempt_n": 1},
        )


def test_runner_merge_tracks_preserves_success_when_other_fails() -> None:
    shared = {
        "format_version": 1,
        "experiment": "prospective_history_response_panel_v2",
        "intervention_key": "case::no_history",
        "case_id": "case",
        "item_id": "item",
        "fold": 0,
        "branch": "no_history",
        "history_match_tier": "fallback_same_fold_difficulty",
        "history_ordered_pair_exact": False,
        "history_donor_item_id": "h",
        "donor5_item_id": "d5",
        "donor6_item_id": "d6",
    }
    behavior = {
        **shared,
        "status": "completed",
        "terminal_track_status": "completed",
        "B_D": 1.0,
        "B_M56": 2.0,
        "elapsed_seconds": 3.0,
        "reused_phase0_full_and_u": True,
        "new_forward_count": 7,
        "formal_branch_forward_count_including_reuse": 9,
        "hidden": {"answer": {"path": "a", "sha256": "x", "bytes": 1}},
    }
    report = {
        **shared,
        "status": "failed",
        "terminal_track_status": "failed",
        "identical_attempt_n": 2,
        "elapsed_seconds": 4.0,
        "reused_phase0_full_and_u": False,
        "error_type": "RuntimeError",
        "error": "boom",
    }
    merged = runner._merge_track_rows(
        behavior, report, behavior_required=True, report_required=True
    )
    assert merged["status"] == "completed"
    assert merged["all_requested_tracks_completed"] is False
    assert merged["behavior_readout_status"] == "completed"
    assert merged["report_formation_status"] == "failed"
    assert merged["B_D"] == 1.0
    assert merged["track_runtime"]["behavior_readout"]["elapsed_seconds"] == 3.0
    assert merged["track_runtime"]["report_formation"]["elapsed_seconds"] == 4.0
    assert merged["formal_branch_forward_count_including_reuse"] == 9


def test_runner_phase0_replacement_can_reuse_failed_item_context(real_plan) -> None:
    failed, same_item = next(
        (row, alternatives)
        for row in real_plan.primary_rows
        if (
            alternatives := [
                reserve
                for reserve in real_plan.reserve_rows
                if reserve["item_id"] == row["item_id"]
                and reserve["selection_stratum"] == row["selection_stratum"]
            ]
        )
    )
    blocked = {
        row["case_id"]
        for row in real_plan.reserve_rows
        if row["selection_stratum"] == failed["selection_stratum"]
        and row["case_id"] != same_item[0]["case_id"]
    }
    replacement = choose_structural_replacement(
        real_plan, failed, real_plan.primary_rows, blocked | {failed["case_id"]}
    )
    other_items = {
        row["item_id"]
        for row in real_plan.primary_rows
        if row["case_id"] != failed["case_id"]
    }
    assert replacement["item_id"] == failed["item_id"]
    assert replacement["item_id"] not in other_items


def test_runner_phase0_plus_phase1_forward_arithmetic_and_track_merge() -> None:
    behavior = {
        "status": "completed",
        "terminal_track_status": "completed",
        "intervention_key": "c::no_history",
        "case_id": "c",
        "item_id": "i",
        "fold": 0,
        "branch": "no_history",
        "new_forward_count": 6,
        "formal_branch_forward_count_including_reuse": 8,
        "reused_phase0_full_and_u": True,
    }
    report = {
        "status": "completed",
        "terminal_track_status": "completed",
        "intervention_key": "c::no_history",
        "case_id": "c",
        "item_id": "i",
        "fold": 0,
        "branch": "no_history",
        "new_forward_count": 1,
        "formal_branch_forward_count_including_reuse": 1,
        "reused_phase0_full_and_u": False,
    }
    merged = runner._merge_track_rows(
        behavior, report, behavior_required=True, report_required=True
    )
    assert merged["new_forward_count"] == 7
    assert merged["formal_branch_forward_count_including_reuse"] == 9


def test_runner_analysis_keeps_terminal_failures_out_of_factorial_rows(tmp_path) -> None:
    runner._upsert_jsonl(
        tmp_path / runner.FORMAL_RESULTS,
        {
            "status": "completed",
            "intervention_key": "c::no_history",
            "case_id": "c",
            "item_id": "i",
            "fold": 0,
            "branch": "no_history",
            "behavior_readout_status": "failed",
            "report_formation_status": "failed",
        },
    )
    summary = runner.analyze(tmp_path)
    assert summary["formal_terminal_branch_n"] == 1
    assert summary["factorial_structural_item_n"] == 0
    assert summary["status"] == "partial"


def _fake_answer_measurement(row, *, predicted_answer: str):
    probabilities = {
        str(row["answer_star"]): 0.6,
        str(row["text_answer"]): 0.3,
        str(predicted_answer): 0.1,
    }
    total = sum(probabilities.values())
    probabilities = {key: value / total for key, value in probabilities.items()}
    distribution = {
        "predicted_answer": str(predicted_answer),
        "unique_top1": True,
        "top1_top2_logit_margin": 0.75,
        "answer_class_probabilities": probabilities,
        "answer_class_logits": {key: float(index) for index, key in enumerate(probabilities)},
        "verbal_sa_leakage": False,
    }
    readout = {
        "primary_u": {"frozen_prediction": 0.2, "coordinate": 0.3},
        "secondary_u_l18": {"frozen_prediction": 0.4, "coordinate": 0.5},
        "nuisance_only": {"frozen_prediction": 0.1, "available": True},
    }
    return AnswerOnlyReadoutMeasurement(
        answer_distribution=distribution,
        answer_star=str(row["answer_star"]),
        nuisance_row={"full_margin": float(row["phase0_full_margin"])},
        readouts=readout,
        hidden_by_layer={18: np.asarray([1.0, 2.0], dtype=np.float32)},
        hidden_checksums={},
        causal_prefix_audit={"passed": True},
        hook_audit={"hook_exactly_once": True},
        teacher_forced_messages_hash="messages",
        teacher_forced_rendered_sha256="rendered",
        length_path_replay={},
    )


def _fake_joint_measurement():
    return JointCommon9Measurement(
        payload={
            "attribution_prediction": 0.25,
            "attribution_coordinate": 0.5,
            "semantic_imageward_score": 0.75,
            "hard_label": "7",
            "hook_audit": {"hook_exactly_once": True},
        },
        hidden=np.asarray([3.0, 4.0], dtype=np.float32),
    )


def test_runner_flatten_saves_audits_donors_and_finite_other_indicator(real_plan) -> None:
    row = dict(real_plan.primary_rows[0])
    row.update(
        {
            "answer_star": row["image_answer"],
            "answer_star_side": "image",
            "phase0_full_margin": 1.25,
            "selection_slot": row["case_id"],
        }
    )
    case = real_plan.cases[(row["item_id"], row["prior_index"])]
    other = next(
        answer
        for answer in case.answer_classes
        if answer not in {row["text_answer"], row["image_answer"]}
    )
    answer = _fake_answer_measurement(row, predicted_answer=other)
    cells = {
        condition: {
            **answer.answer_distribution,
            "answer_class_probabilities": {
                **answer.answer_distribution["answer_class_probabilities"],
                row["answer_star"]: 0.5,
            },
        }
        for condition in EVIDENCE_CONDITIONS
    }
    record = runner._flatten_branch(
        real_plan,
        row,
        "relevant_image_ai",
        cells,
        answer,
        answer.readouts,
        _fake_joint_measurement(),
        {"answer": {}, "attribution": {}},
        elapsed_seconds=1.0,
        new_forward_count=9,
        reused_phase0=False,
    )
    assert record["hard_answer_side"] == "other"
    assert record["hard_answer_image"] == 0.0
    assert record["hard_answer_other"] == 1.0
    assert record["answer_only_no_sa"] is True
    assert record["causal_prefix_equal"] is True
    assert record["answer_hook_exactly_once"] is True
    assert record["joint_hook_exactly_once"] is True
    assert record["history_prefix_equal_across_answer_and_joint"] is True
    assert record["answer_joint_full_prompt_equal"] is False
    assert record["history_match_tier"] == row["history_match_tier"]
    assert record["history_donor_item_id"] == row["history_donor"]["item_id"]
    assert record["donor5_item_id"] == row["donor5"]["item_id"]
    assert record["donor6_item_id"] == row["donor6"]["item_id"]


def test_runner_phase0_reuse_adds_exactly_seven_phase1_forwards(
    real_plan, monkeypatch, tmp_path
) -> None:
    row = dict(real_plan.primary_rows[0])
    row.update(
        {
            "answer_star": row["image_answer"],
            "answer_star_side": "image",
            "phase0_full_margin": 1.25,
            "selection_slot": row["case_id"],
        }
    )
    answer = _fake_answer_measurement(row, predicted_answer=row["image_answer"])
    phase0 = {
        "hidden": {"path": "hidden/phase0.npz", "sha256": "x", "bytes": 1},
    }
    calls = Counter()

    monkeypatch.setattr(runner, "_validate_hidden_reference", lambda *args: None)
    monkeypatch.setattr(runner, "_measurement_from_phase0", lambda *args: answer)

    def restricted(*args, **kwargs):
        calls["restricted"] += 1
        return dict(answer.answer_distribution)

    monkeypatch.setattr(runner, "restricted_next_answer_distribution", restricted)
    monkeypatch.setattr(
        runner,
        "measure_joint_common9",
        lambda *args, **kwargs: (calls.update(["joint"]) or _fake_joint_measurement()),
    )
    monkeypatch.setattr(
        runner,
        "_save_hidden_once",
        lambda path, **arrays: {"path": str(path), "sha256": "y", "bytes": 2},
    )
    primary = SimpleNamespace(
        transform_behavior=lambda deletion, replacement: (deletion + replacement) / 2
    )
    readouts = SimpleNamespace(primary_u={row["fold"]: primary})
    record = runner._measure_branch(
        SimpleNamespace(),
        real_plan,
        row,
        "no_history",
        readouts,
        tmp_path,
        runner.Deadline(1),
        phase0_row=phase0,
        measure_behavior_readout=True,
        measure_report_formation=True,
    )
    assert calls == Counter({"restricted": 6, "joint": 1})
    assert record["new_forward_count"] == 7
    assert record["formal_branch_forward_count_including_reuse"] == 9
    assert record["reused_phase0_full_and_u"] is True
