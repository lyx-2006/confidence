"""Final read-only aggregation and monotonic causal gate for the bridge study.

The causal authorization decision is deliberately computed from frozen,
upstream artifacts only.  Later donor extensions, post-hoc sensitivities,
descriptive screens, and confirmatory report panels are evidence, but they are
not allowed to turn a failed upstream gate into a passing gate.

This module never runs a model forward and never mutates Stages 01--06 or the
Stage-10 attribution screen.  It writes only the fixed Stage-07 skip directory
and the bridge ``analysis`` directory selected by the caller.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from layer_metacognition.hidden_state_store import atomic_write_json, atomic_write_text

from .core import sha256_file, stable_hash, write_csv_atomic, write_jsonl_atomic


BRIDGE_DIR = "stage3_sa_computational_bridge"
TRACE_DIR = "07_causal_divergence_tracing"
ANALYSIS_DIR = "analysis"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class FinalAnalysisPaths:
    experiment_dir: Path
    bridge_root: Path
    trace_output: Path
    analysis_output: Path
    sources: dict[str, Path]


@dataclass(frozen=True)
class FinalAnalysisInputs:
    paths: FinalAnalysisPaths
    documents: dict[str, dict[str, Any]]
    provenance: dict[str, Any]
    validation_audit: dict[str, Any]


def discover_final_analysis_paths(experiment_dir: str | Path) -> FinalAnalysisPaths:
    root = Path(experiment_dir).resolve()
    bridge = root / BRIDGE_DIR
    sources = {
        "stage01_summary": bridge / "01_actual_source_reliance" / "summary.json",
        "stage01_confirmatory_summary": (
            bridge / "01_actual_source_reliance" / "confirmatory_summary.json"
        ),
        "stage01_frozen_rule": (
            bridge / "01_actual_source_reliance" / "frozen_measurement_rule.json"
        ),
        "stage02_summary": bridge / "02_donor_replication_extension" / "summary.json",
        "stage02_confirmatory_summary": (
            bridge / "02_donor_replication_extension" / "confirmatory_summary.json"
        ),
        "stage02_frozen_rule": (
            bridge / "02_donor_replication_extension" / "frozen_extension_rule.json"
        ),
        "stage03_summary": (
            bridge / "03_reliance_representation_devfit_confirm" / "summary.json"
        ),
        "stage03_authorization": (
            bridge
            / "03_reliance_representation_devfit_confirm"
            / "measurement_authorization.json"
        ),
        "stage03_amendment": (
            bridge
            / "03_reliance_representation_devfit_confirm"
            / "PROTOCOL_AMENDMENT.md"
        ),
        "stage04_summary": (
            bridge / "04_reliance_representation_sensitivities" / "summary.json"
        ),
        "stage05_summary": (
            bridge / "05_three_layer_descriptive_screen" / "summary.json"
        ),
        "stage06_summary": (
            bridge / "06_confirmatory_attribution_panel" / "summary.json"
        ),
        "stage06_artifact_manifest": (
            bridge
            / "06_confirmatory_attribution_panel"
            / "artifact_manifest.json"
        ),
        "stage06_three_layer_summary": (
            bridge
            / "06_confirmatory_attribution_panel"
            / "three_layer_analysis"
            / "summary.json"
        ),
        "stage06_three_layer_results": (
            bridge
            / "06_confirmatory_attribution_panel"
            / "three_layer_analysis"
            / "results.jsonl"
        ),
        "stage06_three_layer_markdown": (
            bridge
            / "06_confirmatory_attribution_panel"
            / "three_layer_analysis"
            / "summary.md"
        ),
        "stage10_summary": (
            root
            / "stage3_sa_truth_audit"
            / "10_protocol_shared_attribution_component"
            / "summary.json"
        ),
        "stage10_cohort_manifest": (
            root
            / "stage3_sa_truth_audit"
            / "10_protocol_shared_attribution_component"
            / "cohort_manifest.json"
        ),
    }
    return FinalAnalysisPaths(
        experiment_dir=root,
        bridge_root=bridge,
        trace_output=bridge / TRACE_DIR,
        analysis_output=bridge / ANALYSIS_DIR,
        sources=sources,
    )


def input_readiness(paths: FinalAnalysisPaths) -> dict[str, Any]:
    missing = [name for name, path in paths.sources.items() if not path.is_file()]
    return {
        "ready": not missing,
        "missing": missing,
        "missing_paths": [str(paths.sources[name]) for name in missing],
        "source_count": len(paths.sources),
    }


def validate_output_paths(
    experiment_dir: str | Path,
    trace_output: str | Path | None = None,
    analysis_output: str | Path | None = None,
) -> tuple[Path, Path]:
    paths = discover_final_analysis_paths(experiment_dir)
    trace = Path(trace_output).resolve() if trace_output else paths.trace_output.resolve()
    analysis = (
        Path(analysis_output).resolve()
        if analysis_output
        else paths.analysis_output.resolve()
    )
    if trace != paths.trace_output.resolve():
        raise ValueError(f"Stage-07 output is fixed to {paths.trace_output}; got {trace}")
    if analysis != paths.analysis_output.resolve():
        raise ValueError(
            f"Bridge analysis output is fixed to {paths.analysis_output}; got {analysis}"
        )
    source_roots = {
        path.parent.resolve() if path.is_file() else path.parent.resolve()
        for path in paths.sources.values()
    }
    for output in (trace, analysis):
        if any(
            output == source
            or output.is_relative_to(source)
            or source.is_relative_to(output)
            for source in source_roots
        ):
            raise ValueError("Final aggregate output overlaps a read-only source tree")
    if trace == analysis or trace.is_relative_to(analysis) or analysis.is_relative_to(trace):
        raise ValueError("Stage-07 and analysis outputs must be disjoint siblings")
    return trace, analysis


def _read_object(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not a JSON object: {path}")
    return value


def _validate_stable_field(payload: Mapping[str, Any], field: str, name: str) -> str:
    value = dict(payload)
    fingerprint = str(value.pop(field, ""))
    if not fingerprint or stable_hash(value) != fingerprint:
        raise ValueError(f"{name} {field} mismatch")
    return fingerprint


def _validate_stage06_manifest(
    path: Path, summary: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Stage 06 artifact manifest has no files")
    expected = str(manifest.get("aggregate_sha256", ""))
    if not expected or stable_hash(entries) != expected:
        raise ValueError("Stage 06 artifact aggregate fingerprint mismatch")
    if str(summary.get("artifact_aggregate_sha256", "")) != expected:
        raise ValueError("Stage 06 summary points to another artifact aggregate")
    root = path.parent.resolve()
    failures: list[str] = []
    for entry in entries:
        relative = Path(str(entry.get("path", "")))
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            failures.append(str(relative))
            continue
        if candidate.stat().st_size != int(entry.get("bytes", -1)):
            failures.append(str(relative))
            continue
        if sha256_file(candidate) != str(entry.get("sha256", "")):
            failures.append(str(relative))
    if failures:
        raise ValueError(f"Stage 06 artifact manifest verification failed: {failures[:5]}")
    return {
        "aggregate_sha256": expected,
        "file_count": len(entries),
        "all_files_verified": True,
    }


def _validate_three_layer_confirmatory(
    summary: Mapping[str, Any], results_path: Path
) -> dict[str, Any]:
    if summary.get("status") not in {"complete", "completed"}:
        raise ValueError("Stage 06 confirmatory three-layer analysis is not completed")
    scope = summary.get("analysis_scope", {})
    if scope.get("confirmatory_exact_join") is not True:
        raise ValueError("Stage 06 three-layer analysis is not an exact confirmatory join")
    if scope.get("causal_intervention") is not False:
        raise ValueError("Stage 06 three-layer analysis unexpectedly contains causality")
    if scope.get("causal_mediator_authorized") is not False:
        raise ValueError("Stage 06 three-layer analysis unexpectedly authorizes mediation")
    join = summary.get("join_audit", {})
    required_join = (
        "strict_case_set_equality",
        "strict_item_set_equality",
        "per_case_item_id_equality",
        "per_case_fold_equality",
        "per_case_endpoint_equality",
        "stage10_item_isolation_passed",
        "stage03_stage04_prediction_replay_passed",
    )
    if any(join.get(key) is not True for key in required_join):
        raise ValueError("Stage 06 three-layer exact-join audit failed")
    if join.get("stage10_development_item_overlap") != []:
        raise ValueError("Stage 06 three-layer panel overlaps Stage-10 development")
    if join.get("stage04_gate_bearing") is not False:
        raise ValueError("Stage 06 three-layer analysis promotes Stage 04 to a gate")
    expected_n = int(summary.get("n", -1))
    if expected_n <= 0 or int(summary.get("unique_items", -1)) != expected_n:
        raise ValueError("Stage 06 three-layer summary has an invalid cohort size")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        results_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"Stage 06 three-layer results row {line_number} is not an object"
            )
        rows.append(value)
    case_ids = [str(row.get("case_id", "")) for row in rows]
    item_ids = [str(row.get("item_id", "")) for row in rows]
    if (
        len(rows) != expected_n
        or "" in case_ids
        or "" in item_ids
        or len(set(case_ids)) != expected_n
        or len(set(item_ids)) != expected_n
    ):
        raise ValueError("Stage 06 three-layer results do not match summary n")

    provenance = summary.get("input_provenance", {})
    files = provenance.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Stage 06 three-layer summary lacks input provenance")
    expected_aggregate = str(summary.get("input_aggregate_sha256", ""))
    if provenance.get("input_aggregate_sha256") != expected_aggregate:
        raise ValueError("Stage 06 three-layer input aggregate fields disagree")
    digest_payload: dict[str, str] = {}
    drift: list[str] = []
    for name, entry in files.items():
        path = Path(str(entry.get("path", ""))).resolve()
        expected_hash = str(entry.get("sha256", ""))
        digest_payload[str(name)] = expected_hash
        if (
            not path.is_file()
            or path.stat().st_size != int(entry.get("bytes", -1))
            or sha256_file(path) != expected_hash
        ):
            drift.append(str(name))
    if not expected_aggregate or stable_hash(digest_payload) != expected_aggregate:
        raise ValueError("Stage 06 three-layer input aggregate fingerprint mismatch")
    if drift:
        raise ValueError(
            "Stage 06 three-layer input provenance drift: " + ", ".join(drift[:5])
        )
    return {
        "n": expected_n,
        "unique_items": len(set(item_ids)),
        "results_sha256": sha256_file(results_path),
        "input_aggregate_sha256": expected_aggregate,
        "all_input_files_verified": True,
        "exact_join_verified": True,
    }


def build_input_provenance(paths: FinalAnalysisPaths) -> dict[str, Any]:
    files = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in sorted(paths.sources.items())
    }
    digest_payload = {
        name: {"sha256": value["sha256"], "bytes": value["bytes"]}
        for name, value in files.items()
    }
    return {
        "format_version": FORMAT_VERSION,
        "source_root": str(paths.experiment_dir),
        "files": files,
        "aggregate_sha256": stable_hash(digest_payload),
        "aggregate_definition": (
            "SHA256 of canonical logical-name to source SHA256/bytes mapping"
        ),
        "source_artifacts_read_only": True,
        "transitive_stage06_artifact_coverage": True,
    }


def load_final_analysis_inputs(experiment_dir: str | Path) -> FinalAnalysisInputs:
    paths = discover_final_analysis_paths(experiment_dir)
    readiness = input_readiness(paths)
    if not readiness["ready"]:
        raise FileNotFoundError(
            "Final analysis inputs are incomplete: "
            + ", ".join(readiness["missing_paths"])
        )
    documents: dict[str, dict[str, Any]] = {}
    for name, path in paths.sources.items():
        if path.suffix == ".json":
            documents[name] = _read_object(path, name)

    stage01 = documents["stage01_summary"]
    stage01_confirm = documents["stage01_confirmatory_summary"]
    if stage01.get("confirmatory") != stage01_confirm:
        raise ValueError("Stage 01 aggregate/confirmatory summary drift")
    if stage01_confirm.get("status") != "completed":
        raise ValueError("Stage 01 confirmatory measurement is not completed")
    _validate_stable_field(
        documents["stage01_frozen_rule"], "rule_fingerprint", "Stage 01 rule"
    )

    stage02 = documents["stage02_summary"]
    stage02_confirm = documents["stage02_confirmatory_summary"]
    if stage02.get("confirmatory") != stage02_confirm:
        raise ValueError("Stage 02 aggregate/confirmatory summary drift")
    if stage02_confirm.get("status") != "completed":
        raise ValueError("Stage 02 donor extension is not completed")
    _validate_stable_field(
        documents["stage02_frozen_rule"], "rule_fingerprint", "Stage 02 rule"
    )

    stage03 = documents["stage03_summary"]
    authorization = documents["stage03_authorization"]
    if stage03.get("status") != "completed":
        raise ValueError("Stage 03 representation analysis is not completed")
    authorization_fingerprint = _validate_stable_field(
        authorization, "authorization_fingerprint", "Stage 03 authorization"
    )
    if (
        stage03.get("measurement_authorization", {}).get(
            "authorization_fingerprint"
        )
        != authorization_fingerprint
    ):
        raise ValueError("Stage 03 summary/authorization fingerprint drift")

    stage04 = documents["stage04_summary"]
    if stage04.get("status") != "completed":
        raise ValueError("Stage 04 sensitivity analysis is not completed")
    if stage04.get("gate_bearing") is not False:
        raise ValueError("Stage 04 unexpectedly claims gate-bearing status")
    if stage04.get("original_03_gate_modified") is not False:
        raise ValueError("Stage 04 claims to modify the frozen Stage 03 gate")
    if (
        stage04.get("source_03_gate_snapshot", {}).get("source_summary_sha256")
        != sha256_file(paths.sources["stage03_summary"])
    ):
        raise ValueError("Stage 04 points to another Stage 03 summary")

    stage05 = documents["stage05_summary"]
    if stage05.get("status") not in {"complete", "completed"}:
        raise ValueError("Stage 05 descriptive screen is not completed")
    if stage05.get("analysis_scope", {}).get("causal_mediation_authorized") is not False:
        raise ValueError("Stage 05 unexpectedly authorizes causal mediation")
    if stage05.get("analysis_scope", {}).get("confirmatory_overlap_n") != 0:
        raise ValueError("Stage 05 is not a development-only screen")

    stage06 = documents["stage06_summary"]
    if stage06.get("status") != "completed":
        raise ValueError("Stage 06 confirmatory attribution panel is not completed")
    if stage06.get("causal_intervention") is not False:
        raise ValueError("Stage 06 unexpectedly contains a causal intervention")
    if stage06.get("causal_mediator_authorized") is not False:
        raise ValueError("Stage 06 unexpectedly authorizes a causal mediator")
    if stage06.get("development_item_overlap") != []:
        raise ValueError("Stage 06 overlaps Stage-10 development items")
    stage06_manifest_audit = _validate_stage06_manifest(
        paths.sources["stage06_artifact_manifest"],
        stage06,
        documents["stage06_artifact_manifest"],
    )
    stage06_three_layer_audit = _validate_three_layer_confirmatory(
        documents["stage06_three_layer_summary"],
        paths.sources["stage06_three_layer_results"],
    )

    stage10 = documents["stage10_summary"]
    if stage10.get("status") != "completed":
        raise ValueError("Stage 10 attribution screen is not completed")
    if stage10.get("rank_gate", {}).get("passed") is not True:
        raise ValueError("Frozen Stage-10 rank gate is not the expected PASS")
    if stage10.get("coordinate_gate", {}).get("passed") is not False:
        raise ValueError("Frozen Stage-10 global coordinate gate is not the expected FAIL")

    amendment = paths.sources["stage03_amendment"].read_text(encoding="utf-8")
    required_amendment_terms = (
        "not a conditional or incremental",
        "measurement_authorized=false",
        "Causal mediation remains unauthorized",
    )
    missing_terms = [term for term in required_amendment_terms if term not in amendment]
    if missing_terms:
        raise ValueError(f"Stage 03 protocol amendment is incomplete: {missing_terms}")

    provenance = build_input_provenance(paths)
    validation_audit = {
        "all_required_sources_present": True,
        "stage01_aggregate_matches_confirmatory": True,
        "stage02_aggregate_matches_confirmatory": True,
        "stage03_authorization_fingerprint": authorization_fingerprint,
        "stage04_source_03_hash_verified": True,
        "stage05_development_only": True,
        "stage06": stage06_manifest_audit,
        "stage06_three_layer": stage06_three_layer_audit,
        "stage10_rank_pass_global_coordinate_fail_preserved": True,
    }
    return FinalAnalysisInputs(paths, documents, provenance, validation_audit)


def derive_causal_authorization_gate(
    stage01_confirmatory: Mapping[str, Any],
    stage03_summary: Mapping[str, Any],
    stage03_authorization: Mapping[str, Any],
    stage10_summary: Mapping[str, Any],
    stage06_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive a monotonic gate from frozen upstream sources only.

    ``stage06_summary`` is accepted solely to record its scoped result.  It is
    never an input to the frozen Stage-10 global-coordinate decision and never
    an input to the Stage-01/03 causal authorization conjunction.
    """

    measurement = bool(stage01_confirmatory.get("measurement_gate_passed"))
    graded = stage03_summary.get("estimands", {}).get("graded_preregistered", {})
    representation_components = {
        "graded_measurement_authorized": bool(
            stage03_authorization.get("graded_candidate_allowed")
        ),
        "graded_candidate_source_use_representation": bool(
            graded.get("candidate_source_use_representation")
        ),
        "stage03_causal_mediator_authorized": bool(
            stage03_summary.get("causal_mediator_authorized")
        ),
    }
    representation = all(representation_components.values())
    upstream = measurement and representation
    rank = bool(stage10_summary.get("rank_gate", {}).get("passed"))
    global_coordinate = bool(
        stage10_summary.get("coordinate_gate", {}).get("passed")
    )
    stage06_rank = None
    stage06_common_coordinate = None
    if stage06_summary is not None:
        stage06_rank = bool(
            stage06_summary.get("frozen_rank_gate", {}).get("passed")
        )
        stage06_common_coordinate = bool(
            stage06_summary.get("frozen_common_coordinate_gate", {}).get("passed")
        )
    tracing = upstream and global_coordinate
    blockers: list[str] = []
    if not measurement:
        blockers.append("stage01_confirmatory_measurement_gate_failed")
    if not representation:
        blockers.append("stage03_reliance_representation_not_authorized")
    if not global_coordinate:
        blockers.append("stage10_global_attribution_coordinate_gate_failed")
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "decision": "authorized" if tracing else "skipped_by_gate",
        "causal_divergence_tracing_authorized": tracing,
        "planned_forwards": 0 if not tracing else None,
        "frozen_upstream": {
            "stage01_confirmatory_measurement_gate": measurement,
            "stage03_representation_authorization": representation,
            "stage03_representation_components": representation_components,
            "upstream_causal_authorization": upstream,
        },
        "attribution_scope": {
            "stage10_rank_gate": rank,
            "stage10_global_coordinate_gate": global_coordinate,
            "stage06_confirmatory_rank_gate": stage06_rank,
            "stage06_common_coordinate_gate": stage06_common_coordinate,
            "effective_global_coordinate_gate": global_coordinate,
            "stage06_allowed_to_override_stage10_global_coordinate": False,
        },
        "blockers": blockers,
        "monotonicity": {
            "gate_bearing_sources": [
                "01_actual_source_reliance/confirmatory_summary.json",
                "03_reliance_representation_devfit_confirm/summary.json",
                "03_reliance_representation_devfit_confirm/measurement_authorization.json",
                "stage3_sa_truth_audit/10_protocol_shared_attribution_component/summary.json",
            ],
            "non_overriding_downstream_stages": ["02", "04", "05", "06"],
            "rule": (
                "a failed frozen upstream gate remains failed; downstream extensions, "
                "sensitivities, descriptive associations, and scoped confirmation may "
                "qualify evidence but cannot retroactively authorize causality"
            ),
            "stage02_can_flip_stage01": False,
            "stage04_can_flip_stage03": False,
            "stage05_can_authorize_causality": False,
            "stage06_can_flip_stage10_global_coordinate": False,
        },
    }
    payload["gate_fingerprint"] = stable_hash(payload)
    return payload


def _metric_ci(metric: Mapping[str, Any]) -> tuple[Any, Any]:
    for path in (
        ("spearman_item_bootstrap", "ci95"),
        ("association", "spearman_item_bootstrap", "ci95"),
        ("spearman_bootstrap", "ci95"),
        ("fold_item_cluster_bootstrap", "ci95"),
    ):
        value: Any = metric
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        if isinstance(value, list) and len(value) == 2:
            return value[0], value[1]
    value = metric.get("spearman_ci95")
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    return None, None


def _row(
    row_id: str,
    construct: str,
    experiment: str,
    split: str,
    evidence_class: str,
    endpoint: str,
    *,
    n: Any,
    estimate: Any,
    ci: Sequence[Any] = (None, None),
    gate_bearing: bool,
    gate_passed: Any,
    claim: str,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "construct": construct,
        "experiment": experiment,
        "split": split,
        "evidence_class": evidence_class,
        "endpoint": endpoint,
        "n": n,
        "estimate": estimate,
        "ci_low": ci[0] if len(ci) > 0 else None,
        "ci_high": ci[1] if len(ci) > 1 else None,
        "gate_bearing": gate_bearing,
        "gate_passed": gate_passed,
        "claim": claim,
    }


def build_core_table(inputs: FinalAnalysisInputs, gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    d = inputs.documents
    m01 = d["stage01_confirmatory_summary"]
    raw01 = m01["raw_reliability"]["delete_vs_replace"]
    graded01 = m01["graded_reliability"]["delete_vs_replace"]
    donor01 = m01["donor_replicate_reliability"]["donor1_vs_donor2"]
    m02 = d["stage02_confirmatory_summary"]
    split02 = m02["donor_split_half"]["m12_vs_m34"]
    fresh02 = m02["raw_cross_method_replication"]["deletion_vs_fresh_m34"]
    m03 = d["stage03_summary"]["estimands"]
    raw03 = m03["raw_choice_coupled"]["confirmatory"]["shared"]
    graded03 = m03["graded_preregistered"]["confirmatory"]["shared"]
    m04 = d["stage04_summary"]["estimands"]
    raw04 = m04["raw_choice_coupled"]
    graded04 = m04["graded_preregistered"]
    raw04_fresh = raw04["fresh_donor_endpoint"]["splits"]["confirmatory"][
        "replacement_only_m34"
    ]
    graded04_fresh = graded04["fresh_donor_endpoint"]["splits"]["confirmatory"][
        "replacement_only_m34"
    ]
    raw04_nested = raw04["nested_calibration"]["splits"]["confirmatory"][
        "paired_squared_error_improvement_item_bootstrap"
    ]
    graded04_nested = graded04["nested_calibration"]["splits"]["confirmatory"][
        "paired_squared_error_improvement_item_bootstrap"
    ]
    m05 = d["stage05_summary"]
    m10 = d["stage10_summary"]
    rank10 = m10["oof_shared_target"]["spearman"]
    m06 = d["stage06_summary"]
    m06_three = d["stage06_three_layer_summary"]
    rank06 = m06["frozen_rank_gate"]["shared_target"]["association"]
    post06 = m06["postquery_report_transfer"][
        "frozen_prediction_vs_postquery_report"
    ]

    rows = [
        _row(
            "R01_RAW",
            "Actual Source Reliance",
            "01 frozen measurement",
            "confirmatory",
            "paired behavioral measurement",
            "deletion vs replacement Spearman",
            n=raw01["n"],
            estimate=raw01["spearman"],
            ci=_metric_ci(raw01),
            gate_bearing=True,
            gate_passed=m01["raw_reliability"]["gate_passed"],
            claim="robust choice-coupled source-side sensitivity",
        ),
        _row(
            "R01_GRADED",
            "Actual Source Reliance",
            "01 frozen measurement",
            "confirmatory",
            "paired behavioral measurement",
            "nuisance-adjusted deletion vs replacement Spearman",
            n=graded01["n"],
            estimate=graded01["spearman"],
            ci=_metric_ci(graded01),
            gate_bearing=True,
            gate_passed=m01["graded_reliability"]["gate_passed"],
            claim="graded target failed its frozen reliability gate",
        ),
        _row(
            "R01_DONOR",
            "Actual Source Reliance",
            "01 frozen measurement",
            "confirmatory",
            "donor replicate",
            "donor1 vs donor2 Spearman",
            n=donor01["n"],
            estimate=donor01["spearman"],
            ci=_metric_ci(donor01),
            gate_bearing=True,
            gate_passed=m01["donor_replicate_reliability"]["gate_passed"],
            claim="original donor gate failed; the overall measurement gate stays failed",
        ),
        _row(
            "R02_SPLIT_HALF",
            "Actual Source Reliance",
            "02 donor3/4 extension",
            "confirmatory",
            "post-confirmatory paired replication",
            "old M12 vs fresh M34 Spearman",
            n=split02["n"],
            estimate=split02["spearman"],
            ci=_metric_ci(split02),
            gate_bearing=False,
            gate_passed=m02["extension_gate_passed"],
            claim="fresh-donor convergence on the same items; cannot reverse Stage 01",
        ),
        _row(
            "R02_DELETE_FRESH",
            "Actual Source Reliance",
            "02 donor3/4 extension",
            "confirmatory",
            "post-confirmatory paired replication",
            "deletion vs fresh M34 Spearman",
            n=fresh02["n"],
            estimate=fresh02["spearman"],
            ci=_metric_ci(fresh02),
            gate_bearing=False,
            gate_passed=m02["extension_gate_passed"],
            claim="supports donor robustness, not retroactive measurement authorization",
        ),
        _row(
            "R03_RAW_READOUT",
            "Internal Reliance candidate",
            "03 dev-fit/frozen-confirm readout",
            "confirmatory",
            "predictive readout",
            "raw shared target Spearman",
            n=raw03["n"],
            estimate=raw03["association"]["spearman"],
            ci=_metric_ci(raw03),
            gate_bearing=True,
            gate_passed=m03["raw_choice_coupled"]["readout_gate_passed"],
            claim="endpoint-coupled diagnostic; not a causal mediator",
        ),
        _row(
            "R03_GRADED_READOUT",
            "Internal Reliance candidate",
            "03 dev-fit/frozen-confirm readout",
            "confirmatory",
            "exploratory predictive readout",
            "graded shared target Spearman",
            n=graded03["n"],
            estimate=graded03["association"]["spearman"],
            ci=_metric_ci(graded03),
            gate_bearing=True,
            gate_passed=False,
            claim=(
                "predictive pattern only; measurement authorization is absent despite the "
                "legacy artifact's positive statistical label"
            ),
        ),
        _row(
            "R04_RAW_FRESH",
            "Internal Reliance candidate",
            "04 no-refit sensitivity",
            "confirmatory",
            "post-hoc target transport",
            "frozen replacement prediction vs M34 Spearman",
            n=raw04_fresh["n"],
            estimate=raw04_fresh["spearman"],
            ci=raw04_fresh["item_cluster_bootstrap"]["spearman"]["ci95"],
            gate_bearing=False,
            gate_passed=None,
            claim="genuinely fresh endpoint, but post-hoc same-item sensitivity",
        ),
        _row(
            "R04_GRADED_FRESH",
            "Internal Reliance candidate",
            "04 no-refit sensitivity",
            "confirmatory",
            "post-hoc target transport",
            "frozen graded replacement prediction vs M34 Spearman",
            n=graded04_fresh["n"],
            estimate=graded04_fresh["spearman"],
            ci=graded04_fresh["item_cluster_bootstrap"]["spearman"]["ci95"],
            gate_bearing=False,
            gate_passed=None,
            claim="rank transport only; R2 bootstrap includes zero",
        ),
        _row(
            "R04_RAW_NESTED",
            "Internal Reliance candidate",
            "04 nested calibration sensitivity",
            "confirmatory",
            "post-hoc paired prediction error",
            "MSE improvement: nuisance+hidden over nuisance",
            n=raw04["nested_calibration"]["splits"]["confirmatory"][
                "nuisance_only"
            ]["n"],
            estimate=raw04_nested["estimate"],
            ci=raw04_nested["ci95"],
            gate_bearing=False,
            gate_passed=None,
            claim="CI includes zero; no clear raw conditional increment",
        ),
        _row(
            "R04_GRADED_NESTED",
            "Internal Reliance candidate",
            "04 nested calibration sensitivity",
            "confirmatory",
            "post-hoc paired prediction error",
            "MSE improvement over intercept in nuisance-residual target space",
            n=graded04["nested_calibration"]["splits"]["confirmatory"][
                "nuisance_only"
            ]["n"],
            estimate=graded04_nested["estimate"],
            ci=graded04_nested["ci95"],
            gate_bearing=False,
            gate_passed=None,
            claim="not incremental R2 on the original behavioral outcome",
        ),
        _row(
            "A10_RANK",
            "Internal Attribution candidate",
            "Stage-10 protocol screen",
            "development",
            "item-OOF report-semantic readout",
            "shared target Spearman",
            n=rank10["n"],
            estimate=rank10["estimate"],
            ci=rank10["ci95"],
            gate_bearing=True,
            gate_passed=m10["rank_gate"]["passed"],
            claim="protocol-transportable semantic rank",
        ),
        _row(
            "A10_COORDINATE",
            "Internal Attribution candidate",
            "Stage-10 protocol screen",
            "development",
            "coordinate invariance",
            "global common+legacy coordinate gate",
            n=m10["n_items"],
            estimate=m10["coordinate_metrics"].get("common_icc_a1"),
            gate_bearing=True,
            gate_passed=m10["coordinate_gate"]["passed"],
            claim="global coordinate invariance failed",
        ),
    ]
    for name, label in (
        ("B_vs_V", "Actual Reliance vs verbal report"),
        ("B_vs_A", "Actual Reliance vs attribution rank readout"),
        ("A_vs_V", "attribution rank readout vs verbal report"),
    ):
        metric = m05["primary_associations"][name]
        rows.append(
            _row(
                f"T05_{name.upper()}",
                "Three-layer comparison",
                "05 descriptive screen",
                "development endpoint-matched",
                "descriptive association",
                label,
                n=metric["n"],
                estimate=metric["spearman"],
                ci=metric["spearman_item_bootstrap"]["ci95"],
                gate_bearing=False,
                gate_passed=None,
                claim=(
                    "construction/transport check, not independent convergence"
                    if name == "A_vs_V"
                    else "no confirmatory item-wise convergence established"
                ),
            )
        )
    rows.extend(
        [
            _row(
                "A06_RANK",
                "Internal Attribution candidate",
                "06 frozen confirmatory panel",
                "confirmatory",
                "frozen report/readout transport",
                "shared target Spearman",
                n=rank06["n"],
                estimate=rank06["spearman"],
                ci=rank06["spearman_ci95"],
                gate_bearing=False,
                gate_passed=m06["frozen_rank_gate"]["passed"],
                claim="held-out report/readout rank transfer; noncausal",
            ),
            _row(
                "A06_COMMON_COORDINATE",
                "Internal Attribution candidate",
                "06 frozen confirmatory panel",
                "confirmatory",
                "scoped coordinate check",
                "seven-common-protocol coordinate gate",
                n=m06["n"],
                estimate=m06["frozen_common_coordinate_gate"]["metrics"].get(
                    "common_icc_a1"
                ),
                gate_bearing=False,
                gate_passed=m06["frozen_common_coordinate_gate"]["passed"],
                claim="cannot overwrite the failed Stage-10 global coordinate gate",
            ),
            _row(
                "A06_POSTQUERY",
                "Verbal SA Report",
                "06 branched post-answer continuation",
                "confirmatory",
                "frozen report transfer",
                "answer-prefix prediction vs later postquery report Spearman",
                n=post06["n"],
                estimate=post06["spearman"],
                ci=post06["spearman_ci95"],
                gate_bearing=False,
                gate_passed=m06["postquery_report_transfer"]["passed"],
                claim="report/readout transfer only; no hidden-state equality or causality",
            ),
            _row(
                "C07_SKIP",
                "Causal divergence tracing",
                "07 gate decision",
                "not run",
                "causal intervention",
                "blockwise patching / steering",
                n=0,
                estimate=None,
                gate_bearing=True,
                gate_passed=gate["causal_divergence_tracing_authorized"],
                claim="skipped by frozen upstream gate; zero planned forwards",
            ),
        ]
    )
    raw_three = m06_three["primary_original_B"]["raw_choice_coupled"]
    graded_three = m06_three["primary_original_B"]["graded_preregistered"]
    for estimand, block in (
        ("RAW", raw_three),
        ("GRADED", graded_three),
    ):
        for centered_name, centered_label in (
            ("uncentered", "uncentered"),
            ("answer_side_centered", "answer-side centered"),
        ):
            associations = block[centered_name]["associations"]
            for pair in ("B_vs_A", "B_vs_V"):
                metric = associations[pair]
                rows.append(
                    _row(
                        f"T06C_{estimand}_{'CENTERED_' if centered_name != 'uncentered' else ''}{pair.upper()}",
                        "Three-layer comparison",
                        "06 confirmatory exact-join analysis",
                        "confirmatory",
                        "descriptive association",
                        f"{estimand.lower()} {pair.replace('_', ' ')} ({centered_label})",
                        n=metric["n"],
                        estimate=metric["spearman"],
                        ci=_metric_ci(metric),
                        gate_bearing=False,
                        gate_passed=None,
                        claim=(
                            "confirmatory item-wise association; observational and noncausal"
                        ),
                    )
                )
        delta = block["uncentered"]["paired_rho_difference"]
        rows.append(
            _row(
                f"T06C_{estimand}_DELTA",
                "Three-layer comparison",
                "06 confirmatory exact-join analysis",
                "confirmatory",
                "paired correlation contrast",
                "rho(B,A) - rho(B,V)",
                n=m06_three["n"],
                estimate=delta["estimate"],
                ci=delta["ci95"],
                gate_bearing=False,
                gate_passed=None,
                claim=(
                    "paired descriptive contrast; A and V are constructively non-independent"
                ),
            )
        )
    confirm_a_v = raw_three["uncentered"]["associations"]["A_vs_V"]
    rows.append(
        _row(
            "T06C_A_V",
            "Three-layer comparison",
            "06 confirmatory exact-join analysis",
            "confirmatory",
            "frozen construction/transport check",
            "attribution rank readout vs verbal report",
            n=confirm_a_v["n"],
            estimate=confirm_a_v["spearman"],
            ci=_metric_ci(confirm_a_v),
            gate_bearing=False,
            gate_passed=None,
            claim="not an independent convergent-measure validation",
        )
    )
    return rows


def build_trace_artifacts(
    gate: Mapping[str, Any],
    *,
    input_aggregate_sha256: str,
    config_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if gate.get("causal_divergence_tracing_authorized"):
        raise ValueError("This runner only materializes the frozen-gate skip decision")
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "experiment": "causal_divergence_tracing",
        "status": "skipped_by_gate",
        "cohort_definition": "none; upstream causal prerequisites failed",
        "n": 0,
        "case_ids": [],
        "planned_forwards": 0,
        "actual_forwards": 0,
        "intervention_attempted": False,
        "gate_fingerprint": gate["gate_fingerprint"],
        "input_aggregate_sha256": input_aggregate_sha256,
        "config_fingerprint": config_fingerprint,
    }
    manifest["manifest_fingerprint"] = stable_hash(manifest)
    summary = {
        "format_version": FORMAT_VERSION,
        "title": "Causal Divergence Tracing",
        "status": "skipped_by_gate",
        "n": 0,
        "planned_forwards": 0,
        "actual_forwards": 0,
        "intervention_attempted": False,
        "gate_fingerprint": gate["gate_fingerprint"],
        "primary_blockers": [
            value
            for value in gate["blockers"]
            if value.startswith("stage01_") or value.startswith("stage03_")
        ],
        "additional_coordinate_limitation": (
            "stage10_global_attribution_coordinate_gate_failed"
            in gate["blockers"]
        ),
        "reason": (
            "The frozen Stage-01 confirmatory measurement gate and Stage-03 "
            "reliance-representation authorization are false. Later evidence cannot "
            "retroactively authorize activation tracing."
        ),
        "claim_limit": (
            "No steering, patching, mediation, or causal-divergence result was produced."
        ),
        "input_aggregate_sha256": input_aggregate_sha256,
        "config_fingerprint": config_fingerprint,
    }
    return manifest, summary


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def _association_text(metric: Mapping[str, Any]) -> str:
    ci = metric.get("spearman_item_bootstrap", {}).get("ci95")
    if ci is None:
        ci = metric.get("fold_item_cluster_bootstrap", {}).get("ci95")
    if ci is None:
        ci = metric.get("spearman_ci95", [None, None])
    return (
        f"rho={_number(metric.get('spearman'))} "
        f"[{_number(ci[0])}, {_number(ci[1])}]"
    )


def build_final_analysis_payload(
    inputs: FinalAnalysisInputs,
    gate: Mapping[str, Any],
    core_table: Sequence[Mapping[str, Any]],
    *,
    config_fingerprint: str,
) -> dict[str, Any]:
    d = inputs.documents
    m01 = d["stage01_confirmatory_summary"]
    m02 = d["stage02_confirmatory_summary"]
    m03 = d["stage03_summary"]
    m04 = d["stage04_summary"]
    m05 = d["stage05_summary"]
    m06 = d["stage06_summary"]
    m06_three = d["stage06_three_layer_summary"]
    m10 = d["stage10_summary"]
    return {
        "format_version": FORMAT_VERSION,
        "title": "Actual Reliance / Internal Attribution / Verbal SA computational bridge",
        "status": "completed",
        "config_fingerprint": config_fingerprint,
        "input_aggregate_sha256": inputs.provenance["aggregate_sha256"],
        "validation_audit": inputs.validation_audit,
        "causal_authorization_gate": dict(gate),
        "causal_divergence_tracing": {
            "status": "skipped_by_gate",
            "planned_forwards": 0,
            "actual_forwards": 0,
        },
        "stage_snapshots": {
            "stage01_confirmatory_measurement_gate": m01["measurement_gate_passed"],
            "stage02_extension_gate": m02["extension_gate_passed"],
            "stage03_causal_mediator_authorized": m03["causal_mediator_authorized"],
            "stage04_gate_bearing": m04["gate_bearing"],
            "stage05_classification": m05["classification"],
            "stage10_rank_gate": m10["rank_gate"]["passed"],
            "stage10_global_coordinate_gate": m10["coordinate_gate"]["passed"],
            "stage06_rank_gate": m06["frozen_rank_gate"]["passed"],
            "stage06_common_coordinate_gate": m06[
                "frozen_common_coordinate_gate"
            ]["passed"],
            "stage06_postquery_transfer": m06["postquery_report_transfer"]["passed"],
            "stage06_three_layer_status": m06_three["status"],
            "stage06_three_layer_n": m06_three["n"],
            "stage06_three_layer_causal": False,
        },
        "conclusions": {
            "actual_reliance": (
                "A robust choice-coupled source-side sensitivity replicates across fresh "
                "donors on the same items, but the original graded/donor confirmatory "
                "measurement gate remains failed."
            ),
            "internal_reliance": (
                "Hidden states carry predictive reliance signal, including post-hoc donor "
                "transport, but no graded source-use representation is authorized."
            ),
            "internal_attribution": (
                "A protocol-transportable report-semantic rank component is supported. The "
                "Stage-10 global coordinate gate remains failed even if Stage 06 confirms "
                "a narrower common-template coordinate."
            ),
            "three_layer_relation": (
                "The exact 76-item confirmatory join shows a modest uncentered raw "
                "Reliance-report association but no raw Reliance-attribution association; "
                "both disappear after answer-side centering, and graded associations are "
                "null. This is coarse shared side structure, not fine-grained convergence."
            ),
            "mechanistic_claim": (
                "The evidence disfavors a simple faithful scalar-readout account, but does "
                "not by itself prove that the three objects are mechanistically distinct."
            ),
            "causal_claim": (
                "No causal mediator, mediation, blockwise patching, or divergence point is "
                "authorized or reported."
            ),
        },
        "core_table_rows": len(core_table),
        "claim_boundary": {
            "confirmatory_causal": False,
            "mediation": False,
            "prospective_external_reliance_readout_validation": False,
            "protocol_transportable_attribution_rank": True,
            "global_protocol_invariant_attribution_coordinate": False,
            "verbal_sa_as_faithful_instancewise_reliance_readout": False,
            "confirmatory_three_layer_association_available": True,
        },
    }


def final_analysis_markdown(
    inputs: FinalAnalysisInputs,
    gate: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    d = inputs.documents
    c01 = d["stage01_confirmatory_summary"]
    c02 = d["stage02_confirmatory_summary"]
    c03 = d["stage03_summary"]["estimands"]
    c04 = d["stage04_summary"]["estimands"]
    c05 = d["stage05_summary"]
    c06 = d["stage06_summary"]
    c06_three = d["stage06_three_layer_summary"]
    c10 = d["stage10_summary"]
    b_v = c05["primary_associations"]["B_vs_V"]
    b_a = c05["primary_associations"]["B_vs_A"]
    a_v = c05["primary_associations"]["A_vs_V"]
    rank06 = c06["frozen_rank_gate"]["shared_target"]["association"]
    post06 = c06["postquery_report_transfer"][
        "frozen_prediction_vs_postquery_report"
    ]
    raw_fresh = c04["raw_choice_coupled"]["fresh_donor_endpoint"]["splits"][
        "confirmatory"
    ]["replacement_only_m34"]
    graded_fresh = c04["graded_preregistered"]["fresh_donor_endpoint"]["splits"][
        "confirmatory"
    ]["replacement_only_m34"]
    confirm_raw = c06_three["primary_original_B"]["raw_choice_coupled"]
    confirm_graded = c06_three["primary_original_B"]["graded_preregistered"]
    confirm_raw_ba = confirm_raw["uncentered"]["associations"]["B_vs_A"]
    confirm_raw_bv = confirm_raw["uncentered"]["associations"]["B_vs_V"]
    confirm_raw_ba_centered = confirm_raw["answer_side_centered"]["associations"][
        "B_vs_A"
    ]
    confirm_raw_bv_centered = confirm_raw["answer_side_centered"]["associations"][
        "B_vs_V"
    ]
    confirm_graded_ba = confirm_graded["uncentered"]["associations"]["B_vs_A"]
    confirm_graded_bv = confirm_graded["uncentered"]["associations"]["B_vs_V"]
    confirm_a_v = confirm_raw["uncentered"]["associations"]["A_vs_V"]
    lines = [
        "# Computational Bridge — Final Analysis",
        "",
        "## Executive conclusion",
        "",
        "The current evidence is inconsistent with a simple model in which verbal Source "
        "Attribution is a faithful item-wise scalar readout of Actual Source Reliance. It "
        "instead supports partial sharing of coarse answer/source-side information followed "
        "by report-conditioned reconstruction. This is evidence against the simple identity "
        "account, not a causal proof that all three computations are distinct.",
        "",
        "Causal divergence tracing was **skipped by gate**, with zero planned and zero actual "
        "forwards. The frozen Stage-01 confirmatory measurement gate and Stage-03 reliance-"
        "representation authorization are false. Stage 02, 04, 05, and 06 are explicitly "
        "non-overriding and cannot reverse those decisions.",
        "",
        "## 1. Actual Source Reliance",
        "",
        f"- Frozen Stage-01 confirmatory overall gate: **{'PASS' if c01['measurement_gate_passed'] else 'FAIL'}**.",
        f"- Raw deletion↔replacement: {_association_text(c01['raw_reliability']['delete_vs_replace'])}; "
        f"graded: {_association_text(c01['graded_reliability']['delete_vs_replace'])}.",
        f"- Fresh donor split-half M12↔M34: {_association_text(c02['donor_split_half']['m12_vs_m34'])}. "
        "This same-item extension is supportive but cannot retroactively flip Stage 01.",
        "",
        "The justified construct is therefore a reproducible **choice-coupled source-side "
        "sensitivity**. Fine-grained graded Actual Reliance remains provisional.",
        "",
        "## 2. Internal Reliance candidate",
        "",
        f"- Raw frozen-confirm readout: R²={_number(c03['raw_choice_coupled']['confirmatory']['shared']['r2'])}, "
        f"{_association_text(c03['raw_choice_coupled']['confirmatory']['shared']['association'])}.",
        f"- Graded frozen-confirm readout: R²={_number(c03['graded_preregistered']['confirmatory']['shared']['r2'])}, "
        f"{_association_text(c03['graded_preregistered']['confirmatory']['shared']['association'])}.",
        f"- No-refit M34 replacement transport: raw rho={_number(raw_fresh['spearman'])} "
        f"[{_number(raw_fresh['item_cluster_bootstrap']['spearman']['ci95'][0])}, "
        f"{_number(raw_fresh['item_cluster_bootstrap']['spearman']['ci95'][1])}]; graded "
        f"rho={_number(graded_fresh['spearman'])} "
        f"[{_number(graded_fresh['item_cluster_bootstrap']['spearman']['ci95'][0])}, "
        f"{_number(graded_fresh['item_cluster_bootstrap']['spearman']['ci95'][1])}].",
        "",
        "These are predictive and post-hoc transport results. They do not authorize an "
        "Internal Reliance mediator. The positive graded nested sensitivity is a hidden "
        "readout versus an intercept in nuisance-residual target space, not incremental R² "
        "on the original behavioral outcome.",
        "",
        "## 3. Internal Attribution candidate and verbal report",
        "",
        f"- Stage-10 semantic rank gate: **{'PASS' if c10['rank_gate']['passed'] else 'FAIL'}**; "
        f"global coordinate gate: **{'PASS' if c10['coordinate_gate']['passed'] else 'FAIL'}**.",
        f"- Stage-06 frozen confirmatory rank: {_association_text(rank06)}; gate "
        f"**{'PASS' if c06['frozen_rank_gate']['passed'] else 'FAIL'}**.",
        f"- Stage-06 seven-common-protocol coordinate gate: "
        f"**{'PASS' if c06['frozen_common_coordinate_gate']['passed'] else 'FAIL'}**. "
        "Its scope cannot overwrite the Stage-10 global common+legacy coordinate failure.",
        f"- Branched post-answer report transfer: {_association_text(post06)}; gate "
        f"**{'PASS' if c06['postquery_report_transfer']['passed'] else 'FAIL'}**.",
        "",
        "This supports a held-out, protocol-transportable **report-semantic rank/readout**. "
        "It does not establish a globally invariant coordinate or a causal attribution state.",
        "",
        "## 4. Three-layer comparison",
        "",
        f"On 67 development-only endpoint-matched cases, B↔V was {_association_text(b_v)}, "
        f"B↔A was {_association_text(b_a)}, and A↔V was {_association_text(a_v)}. "
        "A was trained to predict the same report-semantic target V on other items, so A↔V "
        "is a construction/transport check rather than independent triangulation.",
        "",
        "The Stage-05 development screen itself had no confirmatory overlap. A separately "
        "constructed exact 76-item confirmatory join now gives:",
        "",
        f"- Raw B↔A: {_association_text(confirm_raw_ba)}; raw B↔V: "
        f"{_association_text(confirm_raw_bv)}; A↔V: {_association_text(confirm_a_v)}.",
        f"- After answer-side centering, raw B↔A: "
        f"{_association_text(confirm_raw_ba_centered)} and raw B↔V: "
        f"{_association_text(confirm_raw_bv_centered)}.",
        f"- Graded B↔A: {_association_text(confirm_graded_ba)}; graded B↔V: "
        f"{_association_text(confirm_graded_bv)}.",
        "",
        "Thus raw Reliance and verbal report share some uncentered coarse side information, "
        "but neither raw nor graded Reliance shows fine-grained convergence with the frozen "
        "attribution readout. A was trained on the V-kind target, so the strong A↔V transfer "
        "remains constructively non-independent. All B/A/V associations are observational "
        "across items even though B itself was built from behavioral perturbations.",
        "",
        "## 5. Monotonic causal authorization",
        "",
        f"- Stage-01 measurement gate: `{gate['frozen_upstream']['stage01_confirmatory_measurement_gate']}`.",
        f"- Stage-03 representation authorization: `{gate['frozen_upstream']['stage03_representation_authorization']}`.",
        f"- Stage-10 global coordinate gate: `{gate['attribution_scope']['stage10_global_coordinate_gate']}`.",
        f"- Final causal-divergence authorization: `{gate['causal_divergence_tracing_authorized']}`.",
        "- Planned/actual causal forwards: `0 / 0`.",
        "",
        "No steering, activation patching, mediation, or divergence-point result is claimed.",
        "",
        "## Reproducibility",
        "",
        f"All source artifacts were read-only and covered by aggregate SHA256 "
        f"`{inputs.provenance['aggregate_sha256']}`. Stage-06 hidden/logit artifacts were "
        "verified transitively through its artifact manifest. The detailed evidence table is "
        "`core_table.csv`, and the exact authorization decision is "
        "`causal_authorization_gate.json`.",
        "",
    ]
    return "\n".join(lines)


def trace_summary_markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Causal Divergence Tracing",
            "",
            "- Status: `skipped_by_gate`",
            "- Cohort n: `0`",
            "- Planned forwards: `0`",
            "- Actual forwards: `0`",
            "",
            str(summary["reason"]),
            "",
            str(summary["claim_limit"]),
            "",
        ]
    )


def write_final_analysis_outputs(
    inputs: FinalAnalysisInputs,
    *,
    config_fingerprint: str,
    implementation_files: Sequence[str | Path] = (),
) -> dict[str, Any]:
    trace = inputs.paths.trace_output
    analysis = inputs.paths.analysis_output
    trace.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)
    gate = derive_causal_authorization_gate(
        inputs.documents["stage01_confirmatory_summary"],
        inputs.documents["stage03_summary"],
        inputs.documents["stage03_authorization"],
        inputs.documents["stage10_summary"],
        inputs.documents["stage06_summary"],
    )
    if gate["causal_divergence_tracing_authorized"]:
        raise ValueError("Frozen sources unexpectedly authorize causal tracing")

    provenance = dict(inputs.provenance)
    provenance["implementation"] = {
        str(Path(path).resolve()): {
            "sha256": sha256_file(path),
            "bytes": Path(path).stat().st_size,
        }
        for path in implementation_files
    }
    provenance["config_fingerprint"] = config_fingerprint
    provenance["generated_outputs"] = {
        "stage07": str(trace),
        "analysis": str(analysis),
    }
    manifest, trace_summary = build_trace_artifacts(
        gate,
        input_aggregate_sha256=inputs.provenance["aggregate_sha256"],
        config_fingerprint=config_fingerprint,
    )
    table = build_core_table(inputs, gate)
    final = build_final_analysis_payload(
        inputs, gate, table, config_fingerprint=config_fingerprint
    )

    atomic_write_json(analysis / "causal_authorization_gate.json", gate)
    atomic_write_json(analysis / "aggregate_provenance.json", provenance)
    write_csv_atomic(analysis / "core_table.csv", table)
    atomic_write_json(analysis / "final_analysis.json", final)
    atomic_write_text(
        analysis / "FINAL_ANALYSIS.md",
        final_analysis_markdown(inputs, gate, final),
    )
    atomic_write_json(trace / "cohort_manifest.json", manifest)
    write_jsonl_atomic(trace / "results.jsonl", [])
    atomic_write_json(trace / "summary.json", trace_summary)
    atomic_write_text(trace / "summary.md", trace_summary_markdown(trace_summary))
    return final


__all__ = [
    "ANALYSIS_DIR",
    "BRIDGE_DIR",
    "FORMAT_VERSION",
    "TRACE_DIR",
    "FinalAnalysisInputs",
    "FinalAnalysisPaths",
    "build_core_table",
    "build_final_analysis_payload",
    "build_input_provenance",
    "build_trace_artifacts",
    "derive_causal_authorization_gate",
    "discover_final_analysis_paths",
    "final_analysis_markdown",
    "input_readiness",
    "load_final_analysis_inputs",
    "trace_summary_markdown",
    "validate_output_paths",
    "write_final_analysis_outputs",
]
