from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from layer_metacognition.run_sa_confirmatory_attribution_panel import (
    validate_output,
    verify_analysis_rerun_provenance,
)
from layer_metacognition.sa_formation.confirmatory_attribution_panel import (
    ALL_PROTOCOL_NAMES,
    CORE_PROTOCOL_NAMES,
    EXPECTED_COMPLETED_ITEMS,
    FROZEN_EQUIVALENCE_BAND,
    HOLDOUT_PROTOCOL_NAMES,
    JOINT_PROTOCOL_NAMES,
    POSTQUERY_PROTOCOL_NAME,
    RANDOM_LABEL_MAPPINGS,
    ROW_ORDERS,
    FrozenDirectionRepository,
    FrozenFoldDirection,
    audit_answer_star_token,
    audit_prefix_through_answer,
    analyze_confirmatory_panel,
    build_cohort_manifest,
    build_joint_messages,
    build_postquery_messages,
    freeze_stage10_rule,
    frozen_common_coordinate_metrics,
    frozen_rank_gate_components,
    immutable_json,
    load_confirmatory_cohort,
    panel_protocols,
)
from layer_metacognition.sa_formation.core import (
    atomic_save_npz,
    canonical_message_hash,
    initialize_run,
    sha256_file,
    write_jsonl_atomic,
)


def _case(image: Path) -> SimpleNamespace:
    return SimpleNamespace(
        item_id="200",
        prior_index=0,
        question="Which color?",
        text_clue="The text says red.",
        conditions={
            "conflict_easy": SimpleNamespace(resolved_image_path=image),
        },
    )


def test_protocol_freeze_has_exact_strong_controls_and_order() -> None:
    protocols = panel_protocols()
    assert tuple(value.name for value in protocols) == JOINT_PROTOCOL_NAMES
    assert len(CORE_PROTOCOL_NAMES) == 7
    assert RANDOM_LABEL_MAPPINGS == {
        4242: ("B", "T", "Q", "F", "R", "M", "Z", "V", "K"),
        314159: ("R", "Z", "K", "F", "T", "B", "M", "Q", "V"),
        20260811: ("T", "B", "F", "R", "M", "V", "Z", "K", "Q"),
    }
    assert ROW_ORDERS == {
        4242: (3, 4, 0, 8, 5, 1, 2, 7, 6),
        314159: (5, 2, 6, 8, 4, 3, 1, 0, 7),
    }
    assert ALL_PROTOCOL_NAMES == JOINT_PROTOCOL_NAMES + (POSTQUERY_PROTOCOL_NAME,)
    for value in protocols:
        assert sorted(value.display_order) == list(range(len(value.spec.labels_by_semantic)))


def test_contexts_use_answer_star_and_never_old_joint_final(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"x")
    case = _case(image)
    messages, assistant_text = build_joint_messages(
        case,
        "conflict_easy",
        panel_protocols()[0],
        answer_star="red",
    )
    assert assistant_text.startswith("**Answer**: red\n")
    assert "blue" not in assistant_text  # simulated obsolete joint final answer
    assert messages[-1]["content"][0]["text"] == assistant_text

    base, branch, post_assistant = build_postquery_messages(
        case, "conflict_easy", answer_star="red"
    )
    assert base[-1]["content"][0]["text"] == "**Answer**: red"
    assert branch[: len(base)] == base
    assert post_assistant == "**Source Attribution**:"
    assert "**Source Attribution**:" not in base[-1]["content"][0]["text"]


def test_frozen_transform_prediction_and_coordinate_are_exact() -> None:
    direction = FrozenFoldDirection(
        fold=2,
        d_raw=np.asarray([2.0, -1.0]),
        d_unit=np.asarray([0.6, 0.8]),
        raw_intercept=0.5,
        scaler_mean=np.asarray([3.0, 4.0]),
        scaler_scale=np.asarray([2.0, 5.0]),
        train_z_mean=1.0,
        train_z_sd=2.0,
        target_mean=np.arange(7, dtype=np.float64),
        target_scale=np.full(7, 2.0),
        target_loading=np.arange(1, 8, dtype=np.float64) / 10.0,
    )
    hidden = np.asarray([3.0, 4.0])
    scores = np.arange(7, dtype=np.float64) + 2.0
    assert np.isclose(direction.predict(hidden), 2.5)
    assert np.isclose(direction.coordinate(hidden), 2.0)
    assert np.isclose(direction.transform_target(scores), 2.8)


def test_answer_star_token_must_match_method_v2_single_token() -> None:
    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return {" red": [17], " blue": [8, 9]}[text]

    assert audit_answer_star_token(
        Tokenizer(),
        {"answer_star": "red", "method_v2_answer_star_token_id": 17},
    )["passed"]
    assert not audit_answer_star_token(
        Tokenizer(),
        {"answer_star": "blue", "method_v2_answer_star_token_id": 8},
    )["passed"]


def test_postquery_prefix_audit_checks_messages_rendering_and_tokens() -> None:
    base_messages = [
        {"role": "user", "content": [{"type": "text", "text": "q"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**Answer**: red"}]},
    ]
    branch_messages = [
        *base_messages,
        {"role": "user", "content": [{"type": "text", "text": "source?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**Source Attribution**:"}]},
    ]
    audit = audit_prefix_through_answer(
        base_messages=base_messages,
        branch_messages=branch_messages,
        base_rendered="U:q\nA:**Answer**: red",
        branch_rendered="U:q\nA:**Answer**: red\nU:source?\nA:**Source Attribution**:",
        base_inputs={"input_ids": np.asarray([[1, 2, 3, 4]])},
        branch_inputs={"input_ids": np.asarray([[1, 2, 3, 4, 5, 6]])},
        answer_star="red",
    )
    assert audit["passed"]
    assert audit["hidden_numeric_equality_claimed"] is False
    bad = audit_prefix_through_answer(
        base_messages=base_messages,
        branch_messages=branch_messages,
        base_rendered="U:q\nA:**Answer**: red",
        branch_rendered="U:q\nA:**Answer**: red\nU:source?\nA:**Source Attribution**:",
        base_inputs={"input_ids": np.asarray([[1, 2, 3, 4]])},
        branch_inputs={"input_ids": np.asarray([[1, 9, 3, 4, 5, 6]])},
        answer_star="red",
    )
    assert not bad["passed"]
    assert not bad["checks"]["token_prefix_through_answer_equal"]


def _method_row(index: int) -> dict[str, object]:
    answer = "red" if index % 2 == 0 else "blue"
    return {
        "intervention_key": f"actual_reliance|confirmatory|case_{index}",
        "status": "completed",
        "split": "confirmatory",
        "case_id": f"case_{index}",
        "item_id": str(200 + index),
        "prior_index": 0,
        "condition": "conflict_easy",
        "difficulty": "easy",
        "fold": index % 5,
        "answer_star": answer,
        "answer_only_answer": answer,
        "answer_star_side": "text",
        "text_answer": answer,
        "image_answer": "green",
        "final_answer": "obsolete-value-that-must-not-be-used",
        "manifest_fingerprint": "method-v2-fingerprint",
        "measurement_method_version": 2,
        "selection_measurement_same_forward": True,
        "selection_rendered_hash": f"rendered-{index}",
        "selection": {
            "messages_hash": f"selection-{index}",
            "teacher_forced_messages_hash": f"teacher-{index}",
            "canonical_leading_token_ids": {"red": 17, "blue": 23},
        },
        "verbal_sa_leakage": False,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_confirmatory_cohort_is_method_v2_answer_star_and_item_disjoint(tmp_path: Path) -> None:
    method = (
        tmp_path
        / "stage3_sa_computational_bridge"
        / "01_actual_source_reliance"
        / "confirmatory_results.jsonl"
    )
    rows = [_method_row(index) for index in range(EXPECTED_COMPLETED_ITEMS)]
    rows.append(
        {
            "intervention_key": "actual_reliance|confirmatory|excluded",
            "status": "excluded",
            "case_id": "excluded",
            "item_id": "999",
            "exclusion_reason": "tied_natural_endpoint",
        }
    )
    _write_jsonl(method, rows)
    stage10 = (
        tmp_path
        / "stage3_sa_truth_audit"
        / "10_protocol_shared_attribution_component"
        / "cohort_manifest.json"
    )
    stage10.parent.mkdir(parents=True, exist_ok=True)
    stage10.write_text(json.dumps({"item_ids": [str(index) for index in range(80)]}))
    cohort, audit = load_confirmatory_cohort(tmp_path)
    assert len(cohort) == EXPECTED_COMPLETED_ITEMS
    assert audit["item_isolation_passed"]
    assert all(row["answer_source"].endswith("answer_star") for row in cohort)
    assert all("final_answer" not in row for row in cohort)
    manifest = build_cohort_manifest(cohort, audit)
    assert manifest["expected_formal_forward_count"] == 988

    payload = json.loads(stage10.read_text())
    payload["item_ids"].append(cohort[0]["item_id"])
    stage10.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="item overlap"):
        load_confirmatory_cohort(tmp_path)


def test_frozen_coordinate_gate_uses_fixed_band_and_detects_offset() -> None:
    latent = np.linspace(-2.0, 2.0, 120)
    passing = np.column_stack(
        [latent + 0.005 * index * np.sin(latent) for index in range(7)]
    )
    passed = frozen_common_coordinate_metrics(passing, iterations=200)
    assert passed["equivalence_band"] == FROZEN_EQUIVALENCE_BAND
    assert passed["passed"]
    failing = passing.copy()
    failing[:, 3] += 0.5
    failed = frozen_common_coordinate_metrics(failing, iterations=200)
    assert not failed["passed"]
    assert not failed["comparisons"][CORE_PROTOCOL_NAMES[3]]["mean_equivalent"]


def test_frozen_rank_gate_requires_every_random_and_order_holdout() -> None:
    passing_metric = {"spearman_ci95": [0.1, 0.8]}
    protocol_rank = {name: dict(passing_metric) for name in JOINT_PROTOCOL_NAMES}
    components = frozen_rank_gate_components(
        technical_passed=True,
        source_rank_passed=True,
        target_r2=0.1,
        target_association=passing_metric,
        protocol_rank=protocol_rank,
    )
    assert all(components.values())
    protocol_rank[HOLDOUT_PROTOCOL_NAMES[-1]] = {"spearman_ci95": [-0.01, 0.7]}
    components = frozen_rank_gate_components(
        technical_passed=True,
        source_rank_passed=True,
        target_r2=0.1,
        target_association=passing_metric,
        protocol_rank=protocol_rank,
    )
    assert not components["all_random_order_holdout_rank_lower_positive"]


def _write_fake_stage10(root: Path) -> None:
    stage10 = (
        root
        / "stage3_sa_truth_audit"
        / "10_protocol_shared_attribution_component"
    )
    (stage10 / "directions").mkdir(parents=True)
    (stage10 / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "rank_gate": {"passed": True},
                "coordinate_metrics": {"common_pairwise_equivalence_passed": True},
            }
        )
    )
    entries = []
    for fold in range(5):
        name = f"fold_{fold}_layer_18_panl.npz"
        atomic_save_npz(
            stage10 / "directions" / name,
            d_raw=np.asarray([1.0, 0.0]),
            d_unit=np.asarray([1.0, 0.0]),
            raw_intercept=np.asarray(0.25),
            scaler_mean=np.asarray([0.0, 0.0]),
            scaler_scale=np.asarray([1.0, 1.0]),
            train_z_mean=np.asarray(0.5),
            train_z_sd=np.asarray(2.0),
            target_mean=np.zeros(7),
            target_scale=np.ones(7),
            target_loading=np.ones(7) / np.sqrt(7),
        )
        entries.append(
            {
                "fold": fold,
                "file": name,
                "selected_alpha": 1.0,
                "train_z_mean": 0.5,
                "train_z_sd": 2.0,
                "target_mean": [0.0] * 7,
                "target_scale": [1.0] * 7,
                "target_loading": (np.ones(7) / np.sqrt(7)).tolist(),
                "train_items": ["1", "2"],
                "test_items": ["3"],
            }
        )
    (stage10 / "directions" / "index.json").write_text(
        json.dumps({"folds": entries})
    )


def test_rule_is_byte_frozen_and_resume_protected(tmp_path: Path) -> None:
    _write_fake_stage10(tmp_path)
    output = tmp_path / "stage3_sa_computational_bridge" / "06_confirmatory_attribution_panel"
    output.mkdir(parents=True)
    first = freeze_stage10_rule(tmp_path, output)
    second = freeze_stage10_rule(tmp_path, output)
    assert first == second
    repository = FrozenDirectionRepository(output)
    assert np.isclose(repository.get(3).predict(np.asarray([2.0, 5.0])), 2.25)
    assert all((output / entry["frozen_file"]).is_file() for entry in first["folds"])

    immutable_json(output / "immutable.json", {"value": 1})
    immutable_json(output / "immutable.json", {"value": 1})
    with pytest.raises(ValueError, match="refusing overwrite"):
        immutable_json(output / "immutable.json", {"value": 2})


def test_configuration_fingerprint_resume_and_output_protection(tmp_path: Path) -> None:
    run = tmp_path / "fingerprint"
    initialize_run(run, {"experiment": "x", "value": 1}, resume=False)
    initialize_run(run, {"experiment": "x", "value": 1}, resume=True)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        initialize_run(run, {"experiment": "x", "value": 2}, resume=True)

    experiment = tmp_path / "experiment"
    expected = (
        experiment
        / "stage3_sa_computational_bridge"
        / "06_confirmatory_attribution_panel"
    )
    assert validate_output(experiment) == expected.resolve()
    with pytest.raises(ValueError, match="fixed"):
        validate_output(experiment, str(experiment / "stage3_sa_truth_audit"))


def test_analyze_only_allows_recorded_implementation_drift_not_input_drift(
    tmp_path: Path,
) -> None:
    original = {
        "base_inputs": {"base": {"sha256": "a"}},
        "source_inputs": {"source": {"sha256": "b"}},
        "implementation": {"module": {"sha256": "old"}},
        "environment": {"python": "3"},
    }
    current = {
        **original,
        "implementation": {"module": {"sha256": "new"}},
    }
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(original))
    audit = verify_analysis_rerun_provenance(tmp_path, current)
    assert audit["scientific_inputs_identical"]
    assert audit["implementation_identical"] is False
    assert (tmp_path / "analysis_rerun_provenance.json").is_file()
    changed = {
        **current,
        "source_inputs": {"source": {"sha256": "changed"}},
    }
    with pytest.raises(ValueError, match="scientific input provenance drift"):
        verify_analysis_rerun_provenance(tmp_path, changed)


def test_analyze_only_recomputes_complete_synthetic_panel(tmp_path: Path) -> None:
    _write_fake_stage10(tmp_path)
    output = tmp_path / "stage3_sa_computational_bridge" / "06_confirmatory_attribution_panel"
    output.mkdir(parents=True)
    freeze_stage10_rule(tmp_path, output)
    (output / "run_config.json").write_text(json.dumps({"synthetic": True}))
    (output / "provenance.json").write_text(json.dumps({"synthetic": True}))
    endpoint_preflight = {
        "item_isolation_passed": True,
        "stage10_item_overlap": [],
        "completed_n": EXPECTED_COMPLETED_ITEMS,
    }
    cohort = {
        "endpoint_audit": endpoint_preflight,
        "n": EXPECTED_COMPLETED_ITEMS,
        "rows": [],
    }
    (output / "cohort_manifest.json").write_text(json.dumps(cohort))
    result_rows = []
    for index in range(EXPECTED_COMPLETED_ITEMS):
        score = 0.1 + 0.8 * index / (EXPECTED_COMPLETED_ITEMS - 1)
        target = np.sqrt(7.0) * score
        hidden = np.tile(np.asarray([target - 0.25, 0.0]), (len(ALL_PROTOCOL_NAMES), 1))
        hidden_path = output / "hidden" / f"case_{index}.npz"
        atomic_save_npz(
            hidden_path,
            protocols=np.asarray(ALL_PROTOCOL_NAMES),
            joint_protocols=np.asarray(JOINT_PROTOCOL_NAMES),
            hidden=hidden.astype(np.float32),
            layer=np.asarray(18),
        )
        prefix = {
            "passed": True,
            "checks": {
                "rendered_prefix_through_answer_equal": True,
                "token_prefix_through_answer_equal": True,
            },
            "matches_method_v2_teacher_forced_messages_hash": True,
        }
        protocols = {
            name: {
                "semantic_imageward_score": score,
                "hook_exactly_once": True,
                "injection_l2": 0.0,
                **(
                    {"prefix_through_answer_audit": prefix}
                    if name == POSTQUERY_PROTOCOL_NAME
                    else {}
                ),
            }
            for name in ALL_PROTOCOL_NAMES
        }
        result_rows.append(
            {
                "intervention_key": f"confirmatory_attribution|case_{index}",
                "status": "completed",
                "case_id": f"case_{index}",
                "item_id": str(200 + index),
                "fold": index % 5,
                "answer_star": "red",
                "answer_source": "method_v2_confirmatory_results.answer_star",
                "answer_star_token_audit": {"passed": True},
                "protocol_order": list(ALL_PROTOCOL_NAMES),
                "protocols": protocols,
                "hidden_file": hidden_path.name,
                "hidden_sha256": sha256_file(hidden_path),
                "formal_forward_count": len(ALL_PROTOCOL_NAMES),
            }
        )
    write_jsonl_atomic(output / "results.jsonl", result_rows)
    summary = analyze_confirmatory_panel(output)
    assert summary["technical_gate"]["passed"]
    assert summary["frozen_rank_gate"]["passed"]
    assert summary["frozen_common_coordinate_gate"]["passed"]
    assert summary["postquery_report_transfer"]["passed"]
    assert summary["formal_forward_count"] == 988
    assert summary["causal_intervention"] is False
    assert summary["causal_mediator_authorized"] is False
    assert (output / "artifact_manifest.json").is_file()
    assert (output / "summary.md").is_file()
