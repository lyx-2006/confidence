from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer_metacognition.run_sa_three_layer_descriptive_screen import validate_output
from layer_metacognition.sa_formation.three_layer_descriptive_screen import (
    BRIDGE_DIR,
    SCREEN_DIR,
    ThreeLayerPanel,
    ThreeLayerPaths,
    analyze_three_layer_panel,
    build_input_provenance,
    join_three_layer_rows,
)


def _synthetic_inputs(n: int = 25, *, item_only_near_match: bool = False):
    attribution = []
    source = []
    behavior = []
    donor = []
    raw = []
    graded = []
    for index in range(n):
        case_id = f"case_{index}"
        item_id = str(index)
        fold = index % 5
        latent = ((index * 7) % 17 - 8) / 4.0 + 0.03 * index
        answer = "blue" if index % 2 else "green"
        side = "image" if index % 2 else "text"
        difficulty = "hard" if index % 3 == 0 else "easy"
        prior = (index % 7) / 6.0
        margin = 1.0 + 0.11 * (index % 5) + 0.02 * index
        deletion_z = 0.65 * latent + 0.15 * (index % 3)
        replacement_z = 0.45 * latent - 0.08 * (index % 4)
        b_shared = 0.5 * (deletion_z + replacement_z)
        v_target = latent + 0.17 * ((index * 3) % 5)
        a_rank = 0.72 * v_target + 0.13 * ((index * 5) % 7)
        endpoint_matched = index not in {2, 7}
        final_answer = answer if endpoint_matched else "red"
        attribution.append(
            {
                "case_id": case_id,
                "item_id": item_id,
                "fold": fold,
                "shared_prediction_oof": a_rank,
                "shared_target_oof": v_target,
            }
        )
        source.append(
            {
                "case_id": case_id,
                "item_id": item_id,
                "status": "completed",
                "final_answer": final_answer,
                "final_image": int(side == "image"),
            }
        )
        behavior.append(
            {
                "case_id": case_id,
                "item_id": item_id,
                "fold": fold,
                "split": "development",
                "status": "completed",
                "measurement_method_version": 2,
                "answer_star": answer,
                "answer_star_side": side,
                "difficulty": difficulty,
                "prior_strength": prior,
                "full_margin": margin,
                "raw_z_delete": deletion_z,
                "raw_z_replace": replacement_z,
                "reliance_raw_shared": b_shared,
                "behavior_delete_imageward": 2.0 * deletion_z,
            }
        )
        donor.append(
            {
                "case_id": case_id,
                "item_id": item_id,
                "fold": fold,
                "split": "development",
                "status": "completed",
                "extension_method_version": 3,
                "answer_star": answer,
                "answer_star_side": side,
                "difficulty": difficulty,
                "prior_strength": prior,
                "full_margin": margin,
                "behavior_replace_imageward_d34_mean": 1.4 * replacement_z + 0.2,
            }
        )
        raw.append(
            {
                "case_id": case_id,
                "item_id": item_id,
                "fold": fold,
                "split": "development",
                "estimand": "raw_choice_coupled",
                "answer_star": answer,
                "answer_star_side": side,
                "target_shared": b_shared,
                "prediction_shared": 0.7 * b_shared + 0.05 * (index % 4),
            }
        )
        graded.append(
            {
                "case_id": case_id,
                "item_id": item_id,
                "fold": fold,
                "split": "development",
                "estimand": "graded_preregistered",
                "answer_star": answer,
                "answer_star_side": side,
                "target_shared": b_shared - 0.2 * (side == "image"),
                "prediction_shared": 0.4 * b_shared - 0.03 * (index % 3),
            }
        )
    if item_only_near_match:
        attribution.append(
            {
                "case_id": "attribution_case_x",
                "item_id": "999",
                "fold": 0,
                "shared_prediction_oof": 0.2,
                "shared_target_oof": 0.1,
            }
        )
        source.append(
            {
                "case_id": "attribution_case_x",
                "item_id": "999",
                "status": "completed",
                "final_answer": "blue",
                "final_image": 1,
            }
        )
        behavior.append(
            {
                **behavior[0],
                "case_id": "behavior_case_x",
                "item_id": "999",
            }
        )
        donor.append(
            {**donor[0], "case_id": "behavior_case_x", "item_id": "999"}
        )
        raw.append({**raw[0], "case_id": "behavior_case_x", "item_id": "999"})
        graded.append(
            {**graded[0], "case_id": "behavior_case_x", "item_id": "999"}
        )
    calibration = {str(fold): {"mean": 0.2, "sd": 1.4} for fold in range(5)}
    return attribution, source, behavior, donor, raw, graded, calibration


def _join(*, item_only_near_match: bool = False):
    values = _synthetic_inputs(item_only_near_match=item_only_near_match)
    return join_three_layer_rows(
        *values[:6],
        replacement_raw_calibration=values[6],
        confirmatory_case_ids={"behavior": ["confirm_1"]},
        expected_primary_n=23,
        expected_mismatch_n=2,
    )


def test_join_is_case_keyed_and_endpoint_mismatch_is_separate() -> None:
    rows, audit = _join(item_only_near_match=True)
    assert len(rows) == 25
    assert audit["strict_case_overlap_n"] == 25
    assert audit["item_candidate_overlap_n"] == 26
    assert audit["item_only_nonjoin_n"] == 1
    assert audit["item_only_nonjoins"][0]["item_id"] == "999"
    assert audit["item_id_fallback_used"] is False
    primary = [row for row in rows if row["endpoint_matched"]]
    mismatch = [row for row in rows if not row["endpoint_matched"]]
    assert len(primary) == 23
    assert {row["case_id"] for row in mismatch} == {"case_2", "case_7"}
    assert all(row["final_answer"] == row["answer_star"] for row in primary)
    assert all(row["final_answer"] != row["answer_star"] for row in mismatch)


def test_join_rejects_confirmatory_overlap_and_non_development_input() -> None:
    values = _synthetic_inputs()
    with pytest.raises(ValueError, match="overlaps confirmatory"):
        join_three_layer_rows(
            *values[:6],
            replacement_raw_calibration=values[6],
            confirmatory_case_ids={"raw": ["case_3"]},
        )
    values[2][0]["split"] = "confirmatory"
    with pytest.raises(ValueError, match="strictly development"):
        join_three_layer_rows(
            *values[:6],
            replacement_raw_calibration=values[6],
            confirmatory_case_ids={},
        )


def test_same_case_with_different_item_is_rejected() -> None:
    values = _synthetic_inputs()
    values[3][0]["item_id"] = "different_item"
    with pytest.raises(ValueError, match="inconsistent item_ids"):
        join_three_layer_rows(
            *values[:6],
            replacement_raw_calibration=values[6],
            confirmatory_case_ids={},
        )


def test_analysis_marks_constructive_nonindependence_and_writes_atomic_artifacts(
    tmp_path: Path,
) -> None:
    rows, audit = _join()
    panel = ThreeLayerPanel(
        rows=tuple(rows),
        primary_rows=tuple(row for row in rows if row["endpoint_matched"]),
        endpoint_mismatch_rows=tuple(
            row for row in rows if not row["endpoint_matched"]
        ),
        join_audit=audit,
        input_provenance={"input_aggregate_sha256": "synthetic-sha"},
        answer_vocabulary=("blue", "green", "red"),
    )
    output = tmp_path / "screen"
    summary = analyze_three_layer_panel(
        panel, output, bootstrap_iterations=20, config_fingerprint="config-sha"
    )
    assert summary["n"] == 23
    assert summary["endpoint_mismatch_n"] == 2
    assert summary["analysis_scope"]["development_only"] is True
    assert summary["analysis_scope"]["confirmatory_overlap_n"] == 0
    assert summary["analysis_scope"]["causal_mediation_authorized"] is False
    assert summary["construction_dependence"]["A_rank_vs_V_independent"] is False
    assert summary["representation_prediction_diagnostics"]["used_for_gate"] is False
    assert summary["paired_delta_rho"]["primary"]["paired_same_item_resamples"] is True
    for name in ("cohort_manifest.json", "results.jsonl", "summary.json", "summary.md"):
        assert (output / name).is_file()
    manifest = json.loads((output / "cohort_manifest.json").read_text())
    assert manifest["join_key"] == "case_id"
    assert manifest["item_id_fallback_used"] is False
    assert manifest["config_fingerprint"] == "config-sha"
    assert sum(1 for _ in (output / "results.jsonl").open()) == 25
    markdown = (output / "summary.md").read_text()
    assert "constructively non-independent" in markdown
    assert "nonconfirmatory" in markdown


def test_input_aggregate_fingerprint_changes_and_output_is_fixed(tmp_path: Path) -> None:
    file_fields = [
        "attribution_results",
        "attribution_manifest",
        "attribution_source_results",
        "behavior_development",
        "donor_development",
        "raw_representation_development",
        "graded_representation_development",
        "frozen_measurement_rule",
        "behavior_confirmatory",
        "donor_confirmatory",
        "raw_representation_confirmatory",
        "graded_representation_confirmatory",
    ]
    values = {}
    for index, name in enumerate(file_fields):
        path = tmp_path / f"{index}_{name}.json"
        path.write_text(f"{name}\n")
        values[name] = path
    paths = ThreeLayerPaths(experiment_dir=tmp_path, **values)
    first = build_input_provenance(paths)
    values["behavior_development"].write_text("changed\n")
    second = build_input_provenance(paths)
    assert first["input_aggregate_sha256"] != second["input_aggregate_sha256"]
    expected = tmp_path / BRIDGE_DIR / SCREEN_DIR
    assert validate_output(tmp_path) == expected.resolve()
    with pytest.raises(ValueError, match="fixed"):
        validate_output(tmp_path, str(tmp_path / "wrong"))

