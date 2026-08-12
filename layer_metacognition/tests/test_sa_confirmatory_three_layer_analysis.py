from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer_metacognition.run_sa_confirmatory_three_layer_analysis import validate_output
from layer_metacognition.sa_formation.confirmatory_three_layer_analysis import (
    A_FIELD,
    ANALYSIS_DIR,
    BRIDGE_DIR,
    PANEL_DIR,
    ConfirmatoryThreeLayerPanel,
    ConfirmatoryThreeLayerPaths,
    analysis_root,
    analyze_confirmatory_three_layer_panel,
    build_confirmatory_input_provenance,
    join_confirmatory_three_layer_rows,
    load_confirmatory_three_layer_panel,
)
from layer_metacognition.sa_formation.core import sha256_file


def _synthetic_inputs(n: int = 30):
    panel = []
    manifest = []
    raw = []
    graded = []
    raw_fresh = []
    graded_fresh = []
    for index in range(n):
        case_id = f"case_{index:03d}"
        item_id = str(1000 + index)
        fold = index % 5
        side = "other" if index % 10 == 0 else "image" if index % 2 else "text"
        answer = "cyan" if side == "other" else "blue" if side == "image" else "green"
        latent = ((index * 11) % 31 - 15) / 7.0 + 0.025 * index
        a_value = latent + 0.06 * (index % 4)
        v_value = 0.82 * latent + 0.09 * (index % 5)
        raw_b = 0.65 * latent + 0.13 * (index % 3)
        graded_b = 0.28 * latent - 0.10 * (index % 4)
        raw_prediction = 0.72 * raw_b + 0.04 * (index % 5)
        graded_prediction = 0.55 * graded_b - 0.03 * (index % 3)
        common = {
            "case_id": case_id,
            "item_id": item_id,
            "fold": fold,
            "answer_star": answer,
            "answer_star_side": side,
        }
        panel.append(
            {
                "case_id": case_id,
                "item_id": item_id,
                "fold": fold,
                "answer_star": answer,
                "core_consensus_prediction": a_value,
                "frozen_shared_target": v_value,
            }
        )
        manifest.append(dict(common))
        raw.append(
            {
                **common,
                "split": "confirmatory",
                "estimand": "raw_choice_coupled",
                "target_shared": raw_b,
                "prediction_shared": raw_prediction,
            }
        )
        graded.append(
            {
                **common,
                "split": "confirmatory",
                "estimand": "graded_preregistered",
                "target_shared": graded_b,
                "prediction_shared": graded_prediction,
            }
        )
        raw_fresh.append(
            {
                **common,
                "split": "confirmatory",
                "estimand": "raw_choice_coupled",
                "fresh_target_shared_d_m34": raw_b + 0.16 * ((index % 3) - 1),
                "frozen_prediction_shared": raw_prediction,
                "original_target_replay_max_abs_error": 0.0,
                "hidden_or_readout_refit": False,
                "gate_bearing": False,
            }
        )
        graded_fresh.append(
            {
                **common,
                "split": "confirmatory",
                "estimand": "graded_preregistered",
                "fresh_target_shared_d_m34": graded_b - 0.12 * ((index % 4) - 1),
                "frozen_prediction_shared": graded_prediction,
                "original_target_replay_max_abs_error": 0.0,
                "hidden_or_readout_refit": False,
                "gate_bearing": False,
            }
        )
    return panel, manifest, raw, graded, raw_fresh, graded_fresh


def _joined(n: int = 30):
    values = _synthetic_inputs(n)
    rows, audit = join_confirmatory_three_layer_rows(
        *values,
        stage10_item_ids=["1", "2", "3"],
        expected_n=n,
    )
    return rows, audit


def test_exact_case_item_join_and_centered_rows_are_auditable(tmp_path: Path) -> None:
    rows, audit = _joined()
    assert len(rows) == 30
    assert audit["strict_case_set_equality"] is True
    assert audit["strict_item_set_equality"] is True
    assert audit["item_id_fallback_used"] is False
    assert audit["stage10_development_item_overlap"] == []
    assert audit["stage03_stage04_prediction_replay_passed"] is True

    panel = ConfirmatoryThreeLayerPanel(
        rows=tuple(rows),
        join_audit=audit,
        input_provenance={"input_aggregate_sha256": "synthetic-input"},
        source_scope={
            "stage06_technical_gate_passed": True,
            "stage04_post_hoc": True,
        },
    )
    output = tmp_path / "three_layer_analysis"
    summary = analyze_confirmatory_three_layer_panel(
        panel, output, bootstrap_iterations=30
    )
    assert summary["status"] == "complete"
    assert summary["analysis_scope"]["causal_intervention"] is False
    assert summary["construction_and_causal_limits"][
        "A_vs_V_constructively_independent"
    ] is False
    raw = summary["primary_original_B"]["raw_choice_coupled"]
    assert raw["uncentered"]["associations"]["B_vs_A"]["spearman"] is not None
    assert len(raw["uncentered"]["associations"]["B_vs_A"]["fold_metrics"]) == 5
    bootstrap = raw["answer_side_centered"]["associations"]["B_vs_A"][
        "fold_item_cluster_bootstrap"
    ]
    assert bootstrap["cluster_unit"] == "item_id"
    assert bootstrap["stratified_by"] == "fixed_fold"
    assert raw["uncentered"]["paired_rho_difference"][
        "paired_same_item_resamples"
    ] is True
    assert summary["fresh_M34_sensitivity"]["gate_bearing"] is False
    assert summary["fresh_M34_sensitivity"]["post_hoc"] is True

    result_rows = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for side in ("text", "image", "other"):
        selected = [row for row in result_rows if row["answer_star_side"] == side]
        mean = sum(
            row[f"{A_FIELD}_answer_side_centered"] for row in selected
        ) / len(selected)
        assert abs(mean) < 1e-12
    for name in ("results.jsonl", "summary.json", "summary.md"):
        assert (output / name).is_file()
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    assert "Neither A nor V is an intervention" in markdown
    assert "post-hoc, non-gate-bearing" in markdown


def test_join_rejects_nonexact_cases_item_mismatch_and_stage10_overlap() -> None:
    values = _synthetic_inputs()
    shortened = [list(rows) for rows in values]
    shortened[3].pop()
    with pytest.raises(ValueError, match="identical case_id sets"):
        join_confirmatory_three_layer_rows(
            *shortened, stage10_item_ids=[], expected_n=None
        )

    values = _synthetic_inputs()
    values[4][0]["item_id"] = "different-item"
    with pytest.raises(ValueError, match="identical item_id sets"):
        join_confirmatory_three_layer_rows(
            *values, stage10_item_ids=[], expected_n=None
        )

    values = _synthetic_inputs()
    with pytest.raises(ValueError, match="overlap Stage-10"):
        join_confirmatory_three_layer_rows(
            *values, stage10_item_ids=[values[1][0]["item_id"]], expected_n=None
        )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_complete_fixture(root: Path, n: int = 30) -> ConfirmatoryThreeLayerPaths:
    values = _synthetic_inputs(n)
    bridge = root / BRIDGE_DIR
    panel = bridge / PANEL_DIR
    representation = bridge / "03_reliance_representation_devfit_confirm"
    sensitivity = bridge / "04_reliance_representation_sensitivities"
    paths = ConfirmatoryThreeLayerPaths(
        experiment_dir=root,
        panel_analysis=panel / "analysis.jsonl",
        panel_summary=panel / "summary.json",
        panel_manifest=panel / "cohort_manifest.json",
        panel_artifact_manifest=panel / "artifact_manifest.json",
        panel_frozen_rule=panel / "frozen_rule.json",
        stage10_manifest=(
            root
            / "stage3_sa_truth_audit"
            / "10_protocol_shared_attribution_component"
            / "cohort_manifest.json"
        ),
        representation_summary=representation / "summary.json",
        representation_authorization=representation / "measurement_authorization.json",
        raw_representation=(
            representation
            / "raw_choice_coupled"
            / "confirmatory_frozen_predictions.jsonl"
        ),
        graded_representation=(
            representation
            / "graded_preregistered"
            / "confirmatory_frozen_predictions.jsonl"
        ),
        sensitivity_summary=sensitivity / "summary.json",
        raw_fresh_m34=(
            sensitivity
            / "raw_choice_coupled"
            / "confirmatory_fresh_donor_predictions.jsonl"
        ),
        graded_fresh_m34=(
            sensitivity
            / "graded_preregistered"
            / "confirmatory_fresh_donor_predictions.jsonl"
        ),
    )
    _write_jsonl(paths.panel_analysis, values[0])
    _write_json(paths.panel_manifest, {"n": n, "rows": values[1]})
    _write_json(paths.panel_frozen_rule, {"rule": "frozen"})
    _write_json(paths.stage10_manifest, {"item_ids": ["1", "2", "3"]})
    _write_json(paths.representation_summary, {"status": "completed"})
    _write_json(
        paths.representation_authorization,
        {"causal_mediator_authorized": False},
    )
    _write_jsonl(paths.raw_representation, values[2])
    _write_jsonl(paths.graded_representation, values[3])
    _write_json(
        paths.sensitivity_summary,
        {
            "status": "completed",
            "gate_bearing": False,
            "post_hoc": True,
            "original_03_gate_modified": False,
        },
    )
    _write_jsonl(paths.raw_fresh_m34, values[4])
    _write_jsonl(paths.graded_fresh_m34, values[5])
    aggregate = "synthetic-artifact-aggregate"
    _write_json(
        paths.panel_artifact_manifest,
        {
            "aggregate_sha256": aggregate,
            "files": [
                {
                    "path": "analysis.jsonl",
                    "sha256": sha256_file(paths.panel_analysis),
                    "bytes": paths.panel_analysis.stat().st_size,
                }
            ],
        },
    )
    _write_json(
        paths.panel_summary,
        {
            "status": "completed",
            "n": n,
            "technical_gate": {"passed": True},
            "frozen_rank_gate": {"passed": True},
            "frozen_common_coordinate_gate": {"passed": False},
            "artifact_aggregate_sha256": aggregate,
            "causal_intervention": False,
            "causal_mediator_authorized": False,
        },
    )
    return paths


def test_loader_checks_artifacts_and_analysis_does_not_modify_inputs(
    tmp_path: Path,
) -> None:
    paths = _write_complete_fixture(tmp_path)
    provenance = build_confirmatory_input_provenance(paths)
    before = {name: sha256_file(path) for name, path in paths.files().items()}
    panel = load_confirmatory_three_layer_panel(tmp_path, expected_n=30)
    assert panel.input_provenance["input_aggregate_sha256"] == provenance[
        "input_aggregate_sha256"
    ]
    output = analysis_root(tmp_path)
    analyze_confirmatory_three_layer_panel(
        panel, output, bootstrap_iterations=20
    )
    after = {name: sha256_file(path) for name, path in paths.files().items()}
    assert before == after
    assert output.parent == tmp_path / BRIDGE_DIR / PANEL_DIR


def test_loader_rejects_panel_artifact_checksum_and_output_is_fixed(
    tmp_path: Path,
) -> None:
    paths = _write_complete_fixture(tmp_path)
    paths.panel_analysis.write_text(
        paths.panel_analysis.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum"):
        load_confirmatory_three_layer_panel(tmp_path, expected_n=30)

    expected = tmp_path / BRIDGE_DIR / PANEL_DIR / ANALYSIS_DIR
    assert validate_output(tmp_path) == expected.resolve()
    with pytest.raises(ValueError, match="fixed"):
        validate_output(tmp_path, str(tmp_path / "wrong"))
