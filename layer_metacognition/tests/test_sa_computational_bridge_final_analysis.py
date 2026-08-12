from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from layer_metacognition.run_sa_computational_bridge_final_analysis import (
    configuration,
    initialize_outputs,
)
from layer_metacognition.sa_formation.computational_bridge_final_analysis import (
    build_trace_artifacts,
    derive_causal_authorization_gate,
    discover_final_analysis_paths,
    load_final_analysis_inputs,
    validate_output_paths,
    write_final_analysis_outputs,
)
from layer_metacognition.sa_formation.core import sha256_file, stable_hash


def _fingerprinted(payload: dict, key: str) -> dict:
    value = dict(payload)
    value[key] = stable_hash(value)
    return value


def _association(value: float, low: float, high: float, n: int = 8) -> dict:
    return {
        "n": n,
        "unique_items": n,
        "pearson": value,
        "spearman": value,
        "spearman_item_bootstrap": {
            "estimate": value,
            "ci95": [low, high],
            "iterations": 1000,
            "valid": 1000,
        },
    }


def _score(value: float, low: float, high: float, n: int = 8) -> dict:
    return {
        "n": n,
        "r2": value / 2,
        "mae": 0.5,
        "pearson": value,
        "spearman": value,
        "item_cluster_bootstrap": {
            "r2": {"estimate": value / 2, "ci95": [-0.1, 0.4]},
            "mae": {"estimate": 0.5, "ci95": [0.4, 0.6]},
            "spearman": {"estimate": value, "ci95": [low, high]},
        },
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _make_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "experiment"
    paths = discover_final_analysis_paths(root)

    common = _association(0.6, 0.2, 0.8)
    development01 = {
        "status": "completed",
        "split": "development",
        "measurement_gate_passed": True,
    }
    confirm01 = {
        "status": "completed",
        "split": "confirmatory",
        "measurement_gate_passed": False,
        "raw_reliability": {
            "delete_vs_replace": common,
            "gate_passed": True,
        },
        "graded_reliability": {
            "delete_vs_replace": _association(0.3, 0.05, 0.5),
            "gate_passed": False,
        },
        "donor_replicate_reliability": {
            "donor1_vs_donor2": _association(0.4, 0.1, 0.6),
            "gate_passed": False,
        },
    }
    stage01 = {"development": development01, "confirmatory": confirm01}
    rule01 = _fingerprinted({"method": "frozen-01"}, "rule_fingerprint")

    confirm02 = {
        "status": "completed",
        "extension_gate_passed": True,
        "donor_split_half": {"m12_vs_m34": _association(0.7, 0.4, 0.85)},
        "raw_cross_method_replication": {
            "deletion_vs_fresh_m34": _association(0.72, 0.45, 0.86)
        },
    }
    stage02 = {
        "confirmatory": confirm02,
        "development": {"status": "completed", "extension_gate_passed": True},
    }
    rule02 = _fingerprinted({"method": "frozen-02"}, "rule_fingerprint")

    authorization = _fingerprinted(
        {
            "graded_candidate_allowed": False,
            "raw_readout_allowed": True,
            "causal_mediator_authorized": False,
        },
        "authorization_fingerprint",
    )
    raw_shared = {
        "n": 8,
        "r2": 0.4,
        "association": _association(0.65, 0.3, 0.8),
    }
    graded_shared = {
        "n": 8,
        "r2": 0.05,
        "association": _association(0.42, 0.05, 0.67),
    }
    stage03 = {
        "status": "completed",
        "measurement_authorization": authorization,
        "causal_mediator_authorized": False,
        "estimands": {
            "raw_choice_coupled": {
                "readout_gate_passed": False,
                "candidate_source_use_representation": False,
                "confirmatory": {"shared": raw_shared},
            },
            "graded_preregistered": {
                "readout_gate_passed": True,
                "candidate_source_use_representation": False,
                "confirmatory": {"shared": graded_shared},
            },
        },
    }
    amendment = (
        "This is not a conditional or incremental R2.\n"
        "measurement_authorized=false\n"
        "Causal mediation remains unauthorized.\n"
    )

    nested = {
        "nuisance_only": {"n": 8},
        "paired_squared_error_improvement_item_bootstrap": {
            "estimate": 0.1,
            "ci95": [-0.02, 0.2],
        },
    }
    stage04 = {
        "status": "completed",
        "gate_bearing": False,
        "original_03_gate_modified": False,
        "source_03_gate_snapshot": {},
        "estimands": {
            name: {
                "fresh_donor_endpoint": {
                    "splits": {
                        "confirmatory": {
                            "replacement_only_m34": _score(
                                0.65 if name == "raw_choice_coupled" else 0.4,
                                0.2,
                                0.8,
                            )
                        }
                    }
                },
                "nested_calibration": {"splits": {"confirmatory": nested}},
            }
            for name in ("raw_choice_coupled", "graded_preregistered")
        },
    }
    stage05 = {
        "status": "complete",
        "classification": "development-only descriptive",
        "analysis_scope": {
            "causal_mediation_authorized": False,
            "confirmatory_overlap_n": 0,
        },
        "primary_associations": {
            "B_vs_V": {
                "n": 7,
                "spearman": 0.1,
                "spearman_item_bootstrap": {"ci95": [-0.2, 0.4]},
            },
            "B_vs_A": {
                "n": 7,
                "spearman": 0.12,
                "spearman_item_bootstrap": {"ci95": [-0.2, 0.42]},
            },
            "A_vs_V": {
                "n": 7,
                "spearman": 0.62,
                "spearman_item_bootstrap": {"ci95": [0.3, 0.8]},
            },
        },
    }
    rank06 = {
        "n": 8,
        "spearman": 0.55,
        "spearman_ci95": [0.2, 0.75],
    }
    post06 = {
        "n": 8,
        "spearman": 0.2,
        "spearman_ci95": [-0.1, 0.5],
    }
    stage06 = {
        "status": "completed",
        "n": 8,
        "causal_intervention": False,
        "causal_mediator_authorized": False,
        "development_item_overlap": [],
        "frozen_rank_gate": {
            "passed": True,
            "shared_target": {"association": rank06},
        },
        "frozen_common_coordinate_gate": {
            "passed": True,
            "metrics": {"common_icc_a1": 0.9},
        },
        "postquery_report_transfer": {
            "passed": False,
            "frozen_prediction_vs_postquery_report": post06,
        },
    }

    stage10 = {
        "status": "completed",
        "n_items": 8,
        "rank_gate": {"passed": True},
        "coordinate_gate": {"passed": False},
        "oof_shared_target": {
            "spearman": {
                "n": 8,
                "estimate": 0.6,
                "ci95": [0.2, 0.8],
            }
        },
        "coordinate_metrics": {"common_icc_a1": 0.95},
    }

    def three_metric(value: float, low: float, high: float) -> dict:
        return {
            "n": 8,
            "unique_items": 8,
            "spearman": value,
            "fold_item_cluster_bootstrap": {
                "estimate": value,
                "ci95": [low, high],
                "iterations": 1000,
                "valid": 1000,
            },
        }

    a_v = three_metric(0.75, 0.55, 0.86)

    def three_block(ba: float, bv: float) -> dict:
        return {
            "uncentered": {
                "associations": {
                    "B_vs_A": three_metric(ba, -0.2, 0.3),
                    "B_vs_V": three_metric(bv, -0.1, 0.45),
                    "A_vs_V": a_v,
                },
                "paired_rho_difference": {
                    "estimate": ba - bv,
                    "ci95": [-0.4, 0.1],
                },
            },
            "answer_side_centered": {
                "associations": {
                    "B_vs_A": three_metric(-0.1, -0.35, 0.15),
                    "B_vs_V": three_metric(-0.02, -0.25, 0.2),
                    "A_vs_V": a_v,
                },
                "paired_rho_difference": {
                    "estimate": -0.08,
                    "ci95": [-0.35, 0.15],
                },
            },
        }

    values = {
        "stage01_summary": stage01,
        "stage01_confirmatory_summary": confirm01,
        "stage01_frozen_rule": rule01,
        "stage02_summary": stage02,
        "stage02_confirmatory_summary": confirm02,
        "stage02_frozen_rule": rule02,
        "stage03_summary": stage03,
        "stage03_authorization": authorization,
        "stage04_summary": stage04,
        "stage05_summary": stage05,
        "stage06_summary": stage06,
        "stage10_summary": stage10,
        "stage10_cohort_manifest": {"n": 8},
    }
    for name, value in values.items():
        if name == "stage06_summary":
            continue
        _write_json(paths.sources[name], value)
    paths.sources["stage03_amendment"].parent.mkdir(parents=True, exist_ok=True)
    paths.sources["stage03_amendment"].write_text(amendment, encoding="utf-8")
    # Stage 04 must fingerprint the exact Stage-03 summary bytes.
    stage04["source_03_gate_snapshot"]["source_summary_sha256"] = sha256_file(
        paths.sources["stage03_summary"]
    )
    _write_json(paths.sources["stage04_summary"], stage04)

    stage06_root = paths.sources["stage06_summary"].parent
    artifact = stage06_root / "results.jsonl"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"status":"completed"}\n', encoding="utf-8")
    entries = [
        {
            "path": "results.jsonl",
            "sha256": sha256_file(artifact),
            "bytes": artifact.stat().st_size,
        }
    ]
    manifest = {
        "files": entries,
        "aggregate_sha256": stable_hash(entries),
    }
    stage06["artifact_aggregate_sha256"] = manifest["aggregate_sha256"]
    _write_json(paths.sources["stage06_artifact_manifest"], manifest)
    _write_json(paths.sources["stage06_summary"], stage06)

    three_results = paths.sources["stage06_three_layer_results"]
    three_results.parent.mkdir(parents=True, exist_ok=True)
    three_results.write_text(
        "".join(
            json.dumps({"case_id": f"case-{index}", "item_id": f"item-{index}"})
            + "\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    provenance_entry = {
        "path": str(paths.sources["stage03_summary"].resolve()),
        "sha256": sha256_file(paths.sources["stage03_summary"]),
        "bytes": paths.sources["stage03_summary"].stat().st_size,
    }
    provenance_files = {"representation_summary": provenance_entry}
    three_summary = {
        "status": "complete",
        "classification": "confirmatory exact-join descriptive association; noncausal",
        "n": 8,
        "unique_items": 8,
        "analysis_scope": {
            "confirmatory_exact_join": True,
            "causal_intervention": False,
            "causal_mediator_authorized": False,
        },
        "primary_original_B": {
            "raw_choice_coupled": three_block(0.08, 0.27),
            "graded_preregistered": three_block(-0.1, -0.02),
        },
        "join_audit": {
            "strict_case_set_equality": True,
            "strict_item_set_equality": True,
            "per_case_item_id_equality": True,
            "per_case_fold_equality": True,
            "per_case_endpoint_equality": True,
            "stage10_item_isolation_passed": True,
            "stage03_stage04_prediction_replay_passed": True,
            "stage10_development_item_overlap": [],
            "stage04_gate_bearing": False,
        },
        "input_provenance": {
            "files": provenance_files,
            "input_aggregate_sha256": stable_hash(
                {name: value["sha256"] for name, value in provenance_files.items()}
            ),
        },
    }
    three_summary["input_aggregate_sha256"] = three_summary["input_provenance"][
        "input_aggregate_sha256"
    ]
    _write_json(paths.sources["stage06_three_layer_summary"], three_summary)
    paths.sources["stage06_three_layer_markdown"].write_text(
        "# Confirmatory three-layer analysis\n", encoding="utf-8"
    )
    return root


def test_monotonic_gate_ignores_later_scoped_passes() -> None:
    stage01 = {"measurement_gate_passed": False}
    stage03 = {
        "causal_mediator_authorized": False,
        "estimands": {
            "graded_preregistered": {
                "candidate_source_use_representation": False
            }
        },
    }
    authorization = {"graded_candidate_allowed": False}
    stage10 = {"rank_gate": {"passed": True}, "coordinate_gate": {"passed": False}}
    stage06 = {
        "frozen_rank_gate": {"passed": True},
        "frozen_common_coordinate_gate": {"passed": True},
    }
    gate = derive_causal_authorization_gate(
        stage01, stage03, authorization, stage10, stage06
    )
    assert gate["decision"] == "skipped_by_gate"
    assert gate["planned_forwards"] == 0
    assert gate["frozen_upstream"]["stage01_confirmatory_measurement_gate"] is False
    assert gate["frozen_upstream"]["stage03_representation_authorization"] is False
    assert gate["attribution_scope"]["stage06_common_coordinate_gate"] is True
    assert gate["attribution_scope"]["effective_global_coordinate_gate"] is False
    assert gate["monotonicity"]["non_overriding_downstream_stages"] == [
        "02",
        "04",
        "05",
        "06",
    ]


def test_loader_verifies_lineage_fingerprints_and_stage06_manifest(
    tmp_path: Path,
) -> None:
    root = _make_fixture(tmp_path)
    inputs = load_final_analysis_inputs(root)
    assert inputs.validation_audit["stage06"]["all_files_verified"] is True
    assert inputs.documents["stage10_summary"]["rank_gate"]["passed"] is True
    assert inputs.documents["stage10_summary"]["coordinate_gate"]["passed"] is False

    manifest = inputs.paths.sources["stage06_artifact_manifest"]
    value = json.loads(manifest.read_text())
    value["files"][0]["sha256"] = "bad"
    _write_json(manifest, value)
    with pytest.raises(ValueError, match="aggregate fingerprint"):
        load_final_analysis_inputs(root)


def test_final_writer_creates_skip_and_analysis_without_mutating_sources(
    tmp_path: Path,
) -> None:
    root = _make_fixture(tmp_path)
    inputs = load_final_analysis_inputs(root)
    source_hashes = {
        name: sha256_file(path) for name, path in inputs.paths.sources.items()
    }
    config = configuration(
        root,
        inputs.paths.trace_output,
        inputs.paths.analysis_output,
        input_aggregate_sha256=inputs.provenance["aggregate_sha256"],
    )
    fingerprint = initialize_outputs(
        inputs.paths.trace_output,
        inputs.paths.analysis_output,
        config,
        resume=False,
    )
    final = write_final_analysis_outputs(
        inputs,
        config_fingerprint=fingerprint,
        implementation_files=(),
    )
    assert final["causal_divergence_tracing"] == {
        "status": "skipped_by_gate",
        "planned_forwards": 0,
        "actual_forwards": 0,
    }
    trace_summary = json.loads(
        (inputs.paths.trace_output / "summary.json").read_text()
    )
    assert trace_summary["status"] == "skipped_by_gate"
    assert trace_summary["planned_forwards"] == 0
    assert (inputs.paths.trace_output / "results.jsonl").read_text() == ""
    gate = json.loads(
        (inputs.paths.analysis_output / "causal_authorization_gate.json").read_text()
    )
    assert gate["causal_divergence_tracing_authorized"] is False
    assert gate["attribution_scope"]["stage10_rank_gate"] is True
    assert gate["attribution_scope"]["stage10_global_coordinate_gate"] is False
    with (inputs.paths.analysis_output / "core_table.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["row_id"] for row in rows}.issuperset(
        {
            "R01_GRADED",
            "R04_RAW_FRESH",
            "A10_RANK",
            "A06_RANK",
            "T06C_RAW_B_VS_V",
            "T06C_GRADED_B_VS_A",
            "C07_SKIP",
        }
    )
    markdown = (inputs.paths.analysis_output / "FINAL_ANALYSIS.md").read_text()
    assert "skipped by gate" in markdown
    assert "cannot reverse" in markdown
    assert source_hashes == {
        name: sha256_file(path) for name, path in inputs.paths.sources.items()
    }


def test_existing_output_protection_and_resume_fingerprint(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    inputs = load_final_analysis_inputs(root)
    config = configuration(
        root,
        inputs.paths.trace_output,
        inputs.paths.analysis_output,
        input_aggregate_sha256=inputs.provenance["aggregate_sha256"],
    )
    first = initialize_outputs(
        inputs.paths.trace_output,
        inputs.paths.analysis_output,
        config,
        resume=False,
    )
    with pytest.raises(FileExistsError, match="already exist"):
        initialize_outputs(
            inputs.paths.trace_output,
            inputs.paths.analysis_output,
            config,
            resume=False,
        )
    assert (
        initialize_outputs(
            inputs.paths.trace_output,
            inputs.paths.analysis_output,
            config,
            resume=True,
        )
        == first
    )
    changed = dict(config)
    changed["input_aggregate_sha256"] = "changed"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        initialize_outputs(
            inputs.paths.trace_output,
            inputs.paths.analysis_output,
            changed,
            resume=True,
        )


def test_output_paths_are_fixed_and_trace_refuses_authorized_gate(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    expected = discover_final_analysis_paths(root)
    assert validate_output_paths(root) == (
        expected.trace_output.resolve(),
        expected.analysis_output.resolve(),
    )
    with pytest.raises(ValueError, match="fixed"):
        validate_output_paths(root, tmp_path / "elsewhere", None)
    with pytest.raises(ValueError, match="only materializes"):
        build_trace_artifacts(
            {
                "causal_divergence_tracing_authorized": True,
                "gate_fingerprint": "test",
            },
            input_aggregate_sha256="input",
            config_fingerprint="config",
        )
