"""Run the frozen Stage-09 v2 prospective History response panel."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "layer_metacognition"

from confidence_test.answer_metrics import normalize_answer
from layer_metacognition.hidden_state_store import atomic_write_json, atomic_write_text, load_jsonl

from .sa_formation.confirmatory_attribution_panel import immutable_json
from .sa_formation.core import (
    SAFormationArtifacts,
    atomic_save_npz,
    canonical_message_hash,
    initialize_run,
    sha256_file,
    stable_hash,
    write_jsonl_atomic,
)
from .sa_formation.prospective_history_readouts import (
    AnswerOnlyReadoutMeasurement,
    JointCommon9Measurement,
    ProspectiveReadoutRepository,
    load_prospective_readout_repository,
    audit_target_readout_exclusion,
    measure_answer_only_readouts,
    measure_joint_common9,
    nuisance_row_for_fixed_answer,
    project_answer_hidden,
    restricted_next_answer_distribution,
)
from .sa_formation.prospective_history_response_cohort import (
    BRANCHES,
    EVIDENCE_CONDITIONS,
    HISTORY_BRANCH_FACTORS,
    PRIMARY_N,
    ProspectiveHistoryResponsePlan,
    audit_plan_messages,
    build_joint_history_messages,
    build_messages,
    build_plan,
    choose_structural_replacement,
    cohort_candidate_manifest,
    donor_manifest,
    evidence_condition_sources,
    history_branch_factors,
    protocol_manifest,
)
from .sa_formation.prospective_history_response_stats import (
    build_factorial_rows,
    change_reliability,
    factorial_effect_summaries,
    leave_one_reused_donor_cluster_out,
    qualification_gate,
    shift_alignment,
)
from .sa_formation.runtime import Stage3Runtime


PANEL_DIR = "09_prospective_history_response_panel"
DEFAULT_EXPERIMENT = (
    Path(__file__).resolve().parent
    / "output"
    / "Final_v4_run"
    / "answer_basis_9"
)
PHASES = ("auto", "phase0", "phase1", "phase2")
PHASE0_RESULTS = "phase0_results.jsonl"
PHASE1_BEHAVIOR_RESULTS = "phase1_behavior_readout_results.jsonl"
PHASE1_REPORT_RESULTS = "phase1_report_formation_results.jsonl"
PHASE1_RESULTS = "phase1_results.jsonl"
FORMAL_BEHAVIOR_RESULTS = "behavior_readout_results.jsonl"
FORMAL_REPORT_RESULTS = "report_formation_results.jsonl"
FORMAL_RESULTS = "results.jsonl"


class TimeBudgetExceeded(RuntimeError):
    """The current invocation reached its explicit wall-clock budget."""


class Deadline:
    def __init__(self, minutes: float) -> None:
        if not math.isfinite(minutes) or minutes <= 0:
            raise ValueError("--max-minutes must be a positive finite number")
        self.started = time.monotonic()
        self.seconds = float(minutes) * 60.0

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    def check(self) -> None:
        if self.elapsed_seconds >= self.seconds:
            raise TimeBudgetExceeded(
                f"Stage-09 invocation exhausted {self.elapsed_seconds / 60.0:.1f} minutes"
            )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    value.add_argument("--output-dir")
    value.add_argument("--max-minutes", type=float, default=60.0)
    value.add_argument("--phase", choices=PHASES, default="auto")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--smoke-only", action="store_true")
    value.add_argument("--analyze-only", action="store_true")
    return value


def panel_root(experiment_dir: str | Path) -> Path:
    return (
        Path(experiment_dir).resolve()
        / "stage3_sa_computational_bridge"
        / PANEL_DIR
    )


def validate_output(experiment_dir: str | Path, requested: str | None) -> Path:
    experiment = Path(experiment_dir).resolve()
    expected = panel_root(experiment).resolve()
    output = Path(requested).resolve() if requested else expected
    if output != expected:
        raise ValueError(f"Stage-09 output is fixed to {expected}; got {output}")
    protected = [
        experiment / "results.jsonl",
        experiment / "hidden_states",
        experiment / "stage1_metacognition",
        experiment / "stage3_sa_formation",
        experiment / "stage3_sa_formation_followup",
        experiment / "stage3_sa_mechanism",
        experiment / "stage3_sa_second_order",
        experiment / "stage3_sa_truth_audit",
        *[
            path
            for path in (experiment / "stage3_sa_computational_bridge").glob("*")
            if path.name != PANEL_DIR
        ],
        *experiment.glob("stage2_*")
    ]
    if any(
        output == path.resolve() or path.resolve().is_relative_to(output)
        for path in protected
    ):
        raise ValueError("Stage-09 output overlaps a protected Stage 1/2/3 artifact")
    return output


def _primary_cohort_manifest(
    plan: ProspectiveHistoryResponsePlan, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    rows = [
        {
            key: row[key]
            for key in (
                "case_id",
                "item_id",
                "prior_index",
                "condition",
                "difficulty",
                "fold",
                "text_answer",
                "image_answer",
                "legacy_final_image",
                "prior_strength",
                "selection_sha256",
                "selection_role",
                "selection_stratum",
                "primary_rank",
            )
        }
        for row in plan.primary_rows
    ]
    payload: dict[str, Any] = {
        "format_version": 1,
        "panel_version": 2,
        "selection_time": "before every new Stage-09 model outcome",
        "primary_n": len(rows),
        "unique_item_n": len({row["item_id"] for row in rows}),
        "candidate_manifest_fingerprint": candidate["cohort_fingerprint"],
        "endpoint_replacement_policy": (
            "same frozen selection_stratum; first hash-ranked reserve on an unused item; "
            "only tied/other endpoint or repeated technical failure may replace a row"
        ),
        "rows": rows,
    }
    payload["cohort_fingerprint"] = stable_hash(payload)
    return payload


def _joint_branch_messages(
    plan: ProspectiveHistoryResponsePlan,
    row: Mapping[str, Any],
    branch: str,
    answer_star: str,
) -> tuple[list[dict[str, Any]], str]:
    """Delegate to the protocol-frozen History/common-nine builder."""

    return build_joint_history_messages(
        plan, row, branch, answer_star=str(answer_star)
    )


def _joint_message_manifest(plan: ProspectiveHistoryResponsePlan) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in plan.all_rows:
        final_hashes: set[str] = set()
        branch_rows: dict[str, Any] = {}
        # The historical endpoint is a placeholder only for construction.  It
        # does not select, fit, or measure any new Stage-09 outcome.
        placeholder = str(row["text_answer"])
        for branch in BRANCHES:
            try:
                answer_only = build_messages(plan, row, branch, "full")
                joint, assistant_text = _joint_branch_messages(
                    plan, row, branch, placeholder
                )
                expected_prefix = answer_only[:-2]
                if joint[:-2] != expected_prefix:
                    raise RuntimeError("History prefix differs between answer and joint cells")
                final_hash = canonical_message_hash(joint[-2:])
                final_hashes.add(final_hash)
                branch_rows[branch] = {
                    "history_prefix_hash": canonical_message_hash(expected_prefix),
                    "answer_only_final_hash": canonical_message_hash(answer_only[-2:]),
                    "joint_final_hash": final_hash,
                    "joint_full_hash_placeholder_answer": canonical_message_hash(joint),
                    "assistant_continuation": assistant_text,
                }
            except Exception as exc:
                failures.append(
                    {
                        "case_id": row["case_id"],
                        "branch": branch,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        if len(final_hashes) != 1:
            failures.append(
                {
                    "case_id": row["case_id"],
                    "error_type": "JointFinalTurnMismatch",
                    "error": f"observed {len(final_hashes)} joint final-turn hashes",
                }
            )
        rows.append(
            {
                "case_id": row["case_id"],
                "selection_role": row["selection_role"],
                "joint_final_hash": next(iter(final_hashes)) if len(final_hashes) == 1 else None,
                "branches": branch_rows,
            }
        )
    payload: dict[str, Any] = {
        "format_version": 1,
        "panel_version": 2,
        "row_n": len(rows),
        "branch_n": len(BRANCHES),
        "passed": not failures,
        "failure_n": len(failures),
        "failures": failures,
        "evidence_audit": (
            "History prefixes are copied from answer-only cells; every branch receives the "
            "same target common-nine final user/assistant pair. The placeholder answer is "
            "construction-only and is replaced by the Phase-0 frozen A*."
        ),
        "rows": rows,
    }
    payload["joint_message_fingerprint"] = stable_hash(payload)
    return payload


def configuration(
    artifacts: SAFormationArtifacts,
    output: Path,
    protocol: Mapping[str, Any],
    candidate: Mapping[str, Any],
    cohort: Mapping[str, Any],
    donors: Mapping[str, Any],
    image_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "prospective_history_response_panel_v2",
        "experiment_dir": str(artifacts.experiment_dir),
        "output_dir": str(output),
        "primary": {
            "version": "v4",
            "attribution_mode": "joint",
            "source_prompt_variant": "answer_basis_9",
            "seed": 42,
        },
        "primary_n": PRIMARY_N,
        "branches": list(BRANCHES),
        "evidence_conditions": list(EVIDENCE_CONDITIONS),
        "forwards": {
            "retained_successful_cell_per_branch_both_tracks": 9,
            "retained_successful_cell_total_both_tracks": PRIMARY_N
            * len(BRANCHES)
            * 9,
            "phase0_per_selected_endpoint": 2,
            "phase1_additional_per_item": 7,
            "phase2_per_history_branch": 9,
            "retained_successful_total_both_tracks": 3240,
            "retained_successful_total_behavior_readout_only": 2920,
            "retained_successful_total_report_formation_only": 680,
            "retained_successful_phase1_only_if_neither_authorized": 360,
            "technical_retry_or_replacement_overhead_included": False,
            "smoke_forward_count_included": False,
        },
        "fingerprints": {
            "protocol": protocol["protocol_fingerprint"],
            "candidate": candidate["cohort_fingerprint"],
            "cohort": cohort["cohort_fingerprint"],
            "donor": donors["donor_fingerprint"],
            "images": image_manifest["image_manifest_fingerprint"],
        },
        "causal_scope": {
            "history_prompt_manipulation": True,
            "activation_intervention": False,
            "mediator_authorized": False,
        },
    }


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def image_input_manifest(plan: ProspectiveHistoryResponsePlan) -> dict[str, Any]:
    """Hash every image byte source reachable by a target, donor, or History cell."""

    image_paths: set[Path] = set()
    for row in plan.all_rows:
        for source in evidence_condition_sources(plan, row).values():
            image_paths.add(Path(str(source["image_path"])).resolve())
        for branch in ("relevant_image_ai", "irrelevant_image_ai"):
            messages = build_messages(plan, row, branch, "full")
            for message in messages:
                for content in message.get("content", []):
                    if isinstance(content, Mapping) and content.get("type") == "image":
                        image_paths.add(Path(str(content["image"])).resolve())
    missing = [str(path) for path in sorted(image_paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Frozen Stage-09 image inputs are missing: " + ", ".join(missing)
        )
    rows = [_file_record(path) for path in sorted(image_paths)]
    payload: dict[str, Any] = {
        "format_version": 1,
        "selection_scope": "all candidate/reserve evidence and History image sources",
        "image_n": len(rows),
        "rows": rows,
    }
    payload["image_manifest_fingerprint"] = stable_hash(payload)
    return payload


def provenance(
    artifacts: SAFormationArtifacts,
    readouts: ProspectiveReadoutRepository,
    plan: ProspectiveHistoryResponsePlan,
    image_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    experiment = artifacts.experiment_dir
    source_relatives = (
        "stage3_sa_truth_audit/01_counterfactual_source_use/results.jsonl",
        "stage3_sa_computational_bridge/01_actual_source_reliance/development_cohort_manifest.json",
        "stage3_sa_computational_bridge/01_actual_source_reliance/confirmatory_cohort_manifest.json",
        "stage3_sa_computational_bridge/01_actual_source_reliance/development_analysis.jsonl",
        "stage3_sa_computational_bridge/01_actual_source_reliance/confirmatory_results.jsonl",
        "stage3_sa_computational_bridge/02_donor_replication_extension/development_cohort_manifest.json",
        "stage3_sa_computational_bridge/02_donor_replication_extension/confirmatory_cohort_manifest.json",
        "stage3_sa_computational_bridge/03_reliance_representation_devfit_confirm/summary.json",
        "stage3_sa_computational_bridge/03_reliance_representation_devfit_confirm/raw_choice_coupled/directions/index.json",
        "stage3_sa_computational_bridge/03_reliance_representation_devfit_confirm/raw_choice_coupled/development_oof_predictions.jsonl",
        "stage3_sa_computational_bridge/06_confirmatory_attribution_panel/frozen_rule.json",
        "stage3_sa_computational_bridge/08_fixed_l18_representation_divergence/directions/index.json",
        "stage3_sa_computational_bridge/08_fixed_l18_representation_divergence/provenance.json",
        "stage3_sa_formation/02_history/cohort_manifest.json",
        "stage3_sa_formation_followup/02_history_exact_factorial/cohort_manifest.json",
        "stage3_sa_mechanism/01_old_direction_natural_audit/history_results.jsonl",
        "stage3_sa_mechanism/03_relevant_irrelevant_history/results.jsonl",
        "stage3_sa_second_order/01_history_behavior_dissociation/results.jsonl",
        "stage3_sa_truth_audit/03_history_factorial_reanalysis/results.jsonl",
        "stage3_sa_truth_audit/09_history_conditioned_fixed_answer_deletion/results.jsonl",
    )
    missing_sources = [
        relative for relative in source_relatives if not (experiment / relative).is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "Frozen Stage-09 provenance inputs are missing: " + ", ".join(missing_sources)
        )
    sources = {
        relative: _file_record(experiment / relative) for relative in source_relatives
    }
    implementations = (
        Path(__file__).resolve(),
        repository / "confidence_test/joint_answer_source_extension.py",
        repository / "confidence_test/prompt_utils.py",
        repository / "confidence_test/source_attribution_analyzer.py",
        repository / "confidence_test/source_attribution_prompt_utils.py",
        repository / "confidence_test/source_attribution_schema.py",
        repository / "confidence_test/source_attribution_variants.py",
        repository / "layer_metacognition/conversation_builder.py",
        repository / "layer_metacognition/model_adapter.py",
        repository / "layer_metacognition/prompts.py",
        repository / "layer_metacognition/sa_formation/STAGE09_PROSPECTIVE_HISTORY_RESPONSE_PROTOCOL.md",
        repository / "layer_metacognition/sa_formation/prospective_history_response_cohort.py",
        repository / "layer_metacognition/sa_formation/prospective_history_readouts.py",
        repository / "layer_metacognition/sa_formation/prospective_history_response_stats.py",
        repository / "layer_metacognition/sa_formation/confirmatory_attribution_panel.py",
        repository / "layer_metacognition/sa_formation/reliance_measurement.py",
        repository / "layer_metacognition/sa_formation/runtime.py",
    )
    missing_implementations = [str(path) for path in implementations if not path.is_file()]
    if missing_implementations:
        raise FileNotFoundError(
            "Stage-09 implementation provenance is missing: "
            + ", ".join(missing_implementations)
        )
    model_metadata_names = (
        "config.json",
        "configuration.json",
        "generation_config.json",
        "chat_template.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    model_metadata_paths = [artifacts.model_path / name for name in model_metadata_names]
    missing_model_metadata = [str(path) for path in model_metadata_paths if not path.is_file()]
    if missing_model_metadata:
        raise FileNotFoundError(
            "Model identity metadata is missing: " + ", ".join(missing_model_metadata)
        )
    weight_inventory = [
        {"name": path.name, **_file_record(path)}
        for path in sorted(artifacts.model_path.glob("*.safetensors"))
    ]
    if not weight_inventory:
        raise FileNotFoundError(f"Model has no safetensors weights: {artifacts.model_path}")
    readout_audit = readouts.audit_manifest()
    return {
        "format_version": 1,
        "base_inputs": artifacts.provenance(),
        "external_inputs": {
            "dataset": _file_record(artifacts.dataset),
            "inference": _file_record(artifacts.inference_path),
            "image_manifest_fingerprint": image_manifest[
                "image_manifest_fingerprint"
            ],
        },
        "model_identity": {
            "path": str(artifacts.model_path),
            "metadata": {path.name: _file_record(path) for path in model_metadata_paths},
            "weight_inventory": weight_inventory,
            "weight_inventory_fingerprint": stable_hash(weight_inventory),
        },
        "source_inputs": sources,
        "readout_sources": readout_audit,
        "readout_manifest_fingerprint": stable_hash(readout_audit),
        "implementation": {
            str(path.relative_to(repository)): _file_record(path)
            for path in implementations
        },
    }


def _freeze_cpu_artifacts(
    artifacts: SAFormationArtifacts,
    output: Path,
    plan: ProspectiveHistoryResponsePlan,
    readouts: ProspectiveReadoutRepository,
    *,
    resume: bool,
) -> dict[str, Any]:
    protocol = protocol_manifest()
    candidate = cohort_candidate_manifest(plan)
    cohort = _primary_cohort_manifest(plan, candidate)
    donors = donor_manifest(plan)
    images = image_input_manifest(plan)
    exclusion = audit_target_readout_exclusion(
        plan, readouts, artifacts.experiment_dir
    )
    if exclusion.get("passed") is not True:
        raise RuntimeError("Prospective readout-target exclusion audit failed")
    config = configuration(
        artifacts, output, protocol, candidate, cohort, donors, images
    )
    initialize_run(output, config, resume=resume)
    answer_message_audit = audit_plan_messages(plan, include_reserve=True)
    joint_message_audit = _joint_message_manifest(plan)
    if not answer_message_audit["passed"] or not joint_message_audit["passed"]:
        raise RuntimeError("Frozen Stage-09 message reconstruction audit failed")
    immutable_json(output / "protocol_manifest.json", protocol)
    immutable_json(output / "candidate_manifest.json", candidate)
    immutable_json(output / "cohort_manifest.json", cohort)
    immutable_json(output / "donor_manifest.json", donors)
    immutable_json(output / "image_manifest.json", dict(images))
    immutable_json(output / "readout_target_exclusion_audit.json", exclusion)
    immutable_json(output / "message_manifest.json", answer_message_audit)
    immutable_json(output / "joint_message_manifest.json", joint_message_audit)
    immutable_json(output / "readout_manifest.json", readouts.audit_manifest())
    immutable_json(
        output / "provenance.json", provenance(artifacts, readouts, plan, images)
    )
    atomic_write_json(
        output / "runtime_environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
        },
    )
    return config


def _latest_by_key(path: Path, key: str = "intervention_key") -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(load_jsonl(path, repair_trailing=True)):
        value = str(row.get(key) or row.get("case_id") or index)
        latest[value] = dict(row)
    return latest


def _upsert_jsonl(path: Path, row: dict[str, Any]) -> None:
    key = str(row["intervention_key"])
    existing = _latest_by_key(path)
    existing[key] = row
    write_jsonl_atomic(path, [existing[name] for name in sorted(existing)])


def _safe_name(value: str) -> str:
    return stable_hash(value)[:20]


def _save_hidden_once(path: Path, **arrays: Any) -> dict[str, Any]:
    """Atomically save hidden arrays, or verify an orphan byte-semantically.

    A process can die after the atomic NPZ rename but before its JSONL row is
    committed.  Such an NPZ is safe to reuse only when every key, dtype,
    shape, and value equals the freshly reconstructed payload.  We never
    overwrite an existing archive and never accept a checksum alone as proof
    that it belongs to the current measurement.
    """

    expected = {str(key): np.asarray(value) for key, value in arrays.items()}
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as archive:
                observed_keys = set(archive.files)
                if observed_keys != set(expected):
                    raise ValueError(
                        f"orphan keys {sorted(observed_keys)} != {sorted(expected)}"
                    )
                for key, wanted in expected.items():
                    observed = np.asarray(archive[key])
                    if observed.dtype != wanted.dtype:
                        raise ValueError(
                            f"{key} dtype {observed.dtype} != {wanted.dtype}"
                        )
                    if observed.shape != wanted.shape:
                        raise ValueError(
                            f"{key} shape {observed.shape} != {wanted.shape}"
                        )
                    if not np.array_equal(observed, wanted, equal_nan=True):
                        raise ValueError(f"{key} values differ")
        except Exception as exc:
            raise FileExistsError(
                f"Existing hidden artifact failed strict orphan validation: {path}: {exc}"
            ) from exc
    else:
        atomic_save_npz(path, **expected)
        # Read the archive back before publishing a JSON reference to it.
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise RuntimeError(f"Atomic hidden archive lost arrays: {path}")
            for key, wanted in expected.items():
                observed = np.asarray(archive[key])
                if (
                    observed.dtype != wanted.dtype
                    or observed.shape != wanted.shape
                    or not np.array_equal(observed, wanted, equal_nan=True)
                ):
                    raise RuntimeError(f"Atomic hidden archive failed readback: {path}/{key}")
    return {
        "path": str(path.relative_to(path.parents[1])),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _validate_hidden_reference(output: Path, reference: Mapping[str, Any]) -> None:
    """Validate a previously committed hidden reference before resume skips it."""

    path_value = reference.get("path")
    digest = reference.get("sha256")
    size = reference.get("bytes")
    if not path_value or not digest or size is None:
        raise ValueError("Committed hidden reference is incomplete")
    path = (output / str(path_value)).resolve()
    if not path.is_relative_to(output.resolve()):
        raise ValueError(f"Hidden reference escapes Stage-09 output: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Committed hidden artifact is missing: {path}")
    if path.stat().st_size != int(size) or sha256_file(path) != str(digest):
        raise ValueError(f"Committed hidden artifact checksum changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if not archive.files:
            raise ValueError(f"Committed hidden artifact is empty: {path}")
        for key in archive.files:
            value = np.asarray(archive[key])
            if value.dtype == object or not np.isfinite(value).all():
                raise ValueError(f"Committed hidden array is invalid: {path}/{key}")


def _validate_record_hidden(output: Path, row: Mapping[str, Any]) -> None:
    hidden = row.get("hidden")
    if not isinstance(hidden, Mapping):
        raise ValueError(f"Completed row lacks hidden metadata: {row.get('intervention_key')}")
    if "path" in hidden:
        _validate_hidden_reference(output, hidden)
        return
    references = [value for value in hidden.values() if isinstance(value, Mapping)]
    if not references:
        raise ValueError(f"Completed row has no hidden references: {row.get('intervention_key')}")
    for reference in references:
        _validate_hidden_reference(output, reference)


def _validate_terminal_track(output: Path, row: Mapping[str, Any]) -> None:
    status = row.get("status")
    if status == "completed":
        if row.get("terminal_track_status") not in {None, "completed"}:
            raise ValueError("Completed track has inconsistent terminal status")
        _validate_record_hidden(output, row)
        return
    if status == "failed":
        if row.get("terminal_track_status") != "failed":
            raise ValueError("Failed track is not marked terminal_failed")
        if int(row.get("identical_attempt_n", -1)) != 2:
            raise ValueError("Terminal failed track did not exhaust exactly two attempts")
        return
    raise ValueError(f"Unknown track terminal status: {status!r}")


def _answer_hidden_arrays(measured: AnswerOnlyReadoutMeasurement) -> dict[str, Any]:
    layers = sorted(int(layer) for layer in measured.hidden_by_layer)
    return {
        "answer_layers": np.asarray(layers, dtype=np.int64),
        "answer_hidden": np.stack(
            [np.asarray(measured.hidden_by_layer[layer], dtype=np.float16) for layer in layers]
        ),
    }


def _answer_side(answer: str, row: Mapping[str, Any]) -> str:
    value = str(normalize_answer(answer))
    if value == str(normalize_answer(row["image_answer"])):
        return "image"
    if value == str(normalize_answer(row["text_answer"])):
        return "text"
    return "other"


def _phase0_attempt(
    runtime: Stage3Runtime,
    plan: ProspectiveHistoryResponsePlan,
    row: Mapping[str, Any],
    readouts: ProspectiveReadoutRepository,
    output: Path,
    deadline: Deadline,
    *,
    selection_slot: str,
    attempt_rank: int,
) -> dict[str, Any]:
    key = f"phase0::{row['case_id']}"
    base = {
        "format_version": 1,
        "experiment": "prospective_history_response_panel_v2",
        "phase": "phase0",
        "intervention_key": key,
        "selection_slot": selection_slot,
        "attempt_rank": int(attempt_rank),
        "case_id": str(row["case_id"]),
        "item_id": str(row["item_id"]),
        "fold": int(row["fold"]),
        "difficulty": str(row["difficulty"]),
        "condition": str(row["condition"]),
    }
    started = time.perf_counter()
    messages = build_messages(plan, row, "no_history", "full")
    errors: list[dict[str, str]] = []
    measured: AnswerOnlyReadoutMeasurement | None = None
    for retry in range(2):
        try:
            deadline.check()
            measured = measure_answer_only_readouts(
                runtime,
                messages,
                answer_classes=plan.cases[
                    (str(row["item_id"]), int(row["prior_index"]))
                ].answer_classes,
                fixed_answer=None,
                base_row=row,
                readouts=readouts,
            )
            break
        except TimeBudgetExceeded:
            raise
        except Exception as exc:
            errors.append({"error_type": type(exc).__name__, "error": str(exc)})
            if retry == 0 and "unique restricted top-1" not in str(exc):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            break
    if measured is None:
        return {
            **base,
            "status": "failed" if not any("unique restricted top-1" in e["error"] for e in errors) else "excluded",
            "exclusion_reason": (
                "tied_natural_endpoint"
                if any("unique restricted top-1" in e["error"] for e in errors)
                else None
            ),
            "errors": errors,
            "elapsed_seconds": time.perf_counter() - started,
        }
    side = _answer_side(measured.answer_star, row)
    if side == "other":
        return {
            **base,
            "status": "excluded",
            "exclusion_reason": "answer_star_is_neither_text_nor_image_endpoint",
            "answer_star": measured.answer_star,
            "answer_distribution": measured.answer_distribution,
            "elapsed_seconds": time.perf_counter() - started,
        }
    hidden_path = output / "hidden" / f"phase0_{_safe_name(str(row['case_id']))}.npz"
    hidden = _save_hidden_once(hidden_path, **_answer_hidden_arrays(measured))
    return {
        **base,
        "status": "completed",
        "answer_star": measured.answer_star,
        "answer_star_side": side,
        "measurement": measured.to_payload(),
        "hidden": hidden,
        "retry_errors": errors,
        "formal_forward_count": 2,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_phase0(
    runtime: Stage3Runtime,
    plan: ProspectiveHistoryResponsePlan,
    readouts: ProspectiveReadoutRepository,
    output: Path,
    deadline: Deadline,
) -> dict[str, Any]:
    endpoint_path = output / "endpoint_manifest.json"
    if endpoint_path.is_file():
        return _validate_endpoint_manifest(
            json.loads(endpoint_path.read_text(encoding="utf-8")), plan, output
        )
    results_path = output / PHASE0_RESULTS
    existing = _latest_by_key(results_path)
    completed_slots = [
        str(row.get("selection_slot"))
        for row in existing.values()
        if row.get("status") == "completed"
    ]
    if len(completed_slots) != len(set(completed_slots)):
        raise ValueError("Multiple completed Phase-0 endpoints exist for one slot")
    for key, prior in existing.items():
        if key != f"phase0::{prior.get('case_id')}":
            raise ValueError(f"Phase-0 intervention key/case mismatch: {key}")
    rows_by_case = {str(row["case_id"]): dict(row) for row in plan.all_rows}
    active = {str(row["case_id"]): dict(row) for row in plan.primary_rows}
    chosen: list[dict[str, Any]] = []
    for primary in plan.primary_rows:
        slot = str(primary["case_id"])
        current = dict(active[slot])
        selected: dict[str, Any] | None = None
        # A reserve context is globally single-use.  Pre-block attempts owned
        # by every other slot (including later slots on resume), but replay
        # this slot's own attempt chain in order below.
        attempted_case_ids = {
            str(prior.get("case_id"))
            for prior in existing.values()
            if str(prior.get("selection_slot")) != slot
        }
        prior_attempts = sorted(
            (
                dict(row)
                for row in existing.values()
                if str(row.get("selection_slot")) == slot
            ),
            key=lambda row: int(row.get("attempt_rank", 10**9)),
        )
        expected_rank = 1
        for prior in prior_attempts:
            case_id = str(prior.get("case_id"))
            if int(prior.get("attempt_rank", -1)) != expected_rank:
                raise ValueError(f"Non-contiguous Phase-0 attempt ranks for slot {slot}")
            if case_id != str(current["case_id"]):
                raise ValueError(
                    f"Phase-0 resume replacement order drifted for {slot}: "
                    f"{case_id} != {current['case_id']}"
                )
            attempted_case_ids.add(case_id)
            status = str(prior.get("status"))
            if status == "completed":
                _validate_record_hidden(output, prior)
                selected = prior
                active[slot] = rows_by_case[case_id]
                break
            if status not in {"excluded", "failed"}:
                raise ValueError(f"Unknown Phase-0 status for {case_id}: {status}")
            current = choose_structural_replacement(
                plan,
                current,
                list(active.values()),
                attempted_case_ids,
            )
            active[slot] = current
            expected_rank += 1

        while selected is None:
            measured = _phase0_attempt(
                runtime,
                plan,
                current,
                readouts,
                output,
                deadline,
                selection_slot=slot,
                attempt_rank=expected_rank,
            )
            _upsert_jsonl(results_path, measured)
            existing[measured["intervention_key"]] = measured
            attempted_case_ids.add(str(current["case_id"]))
            if measured.get("status") == "completed":
                selected = measured
                active[slot] = current
                break
            current = choose_structural_replacement(
                plan,
                current,
                list(active.values()),
                attempted_case_ids,
            )
            active[slot] = current
            expected_rank += 1
        if selected is None:
            atomic_write_json(
                output / "phase0_summary.json",
                {
                    "status": "incomplete",
                    "completed_slots": len(chosen),
                    "expected_slots": PRIMARY_N,
                    "failed_slot": slot,
                },
            )
            raise RuntimeError(f"Phase 0 could not establish an endpoint for slot {slot}")
        chosen.append(selected)

    active_items = [str(row["item_id"]) for row in active.values()]
    if len(set(active_items)) != PRIMARY_N:
        raise RuntimeError("Dynamic Phase-0 replacement violated item uniqueness")

    rows = [
        {
            "selection_slot": row["selection_slot"],
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "fold": row["fold"],
            "difficulty": row["difficulty"],
            "condition": row["condition"],
            "answer_star": row["answer_star"],
            "answer_star_side": row["answer_star_side"],
            "phase0_full_margin": float(
                row["measurement"]["answer_distribution"]["top1_top2_logit_margin"]
            ),
            "phase0_full_entropy": _entropy(
                row["measurement"]["answer_distribution"]["answer_class_probabilities"]
            ),
            "phase0_hard_answer_image": 1.0
            if row["answer_star_side"] == "image"
            else 0.0,
            "phase0_hard_answer_other": 0.0,
            "phase0_intervention_key": row["intervention_key"],
            "phase0_hidden_sha256": row["hidden"]["sha256"],
            "phase0_measurement_fingerprint": stable_hash(row["measurement"]),
        }
        for row in chosen
    ]
    payload: dict[str, Any] = {
        "format_version": 1,
        "panel_version": 2,
        "selection_time": "after no-History Full endpoint measurement and before Phase 1/2",
        "n": len(rows),
        "unique_item_n": len({row["item_id"] for row in rows}),
        "fold_counts": {
            str(fold): sum(int(row["fold"]) == fold for row in rows)
            for fold in range(5)
        },
        "rows": rows,
    }
    payload["endpoint_fingerprint"] = stable_hash(payload)
    immutable_json(endpoint_path, payload)
    payload = _validate_endpoint_manifest(payload, plan, output)
    atomic_write_json(
        output / "phase0_summary.json",
        {
            "status": "complete",
            "n": len(rows),
            "unique_item_n": len({row["item_id"] for row in rows}),
            "endpoint_fingerprint": payload["endpoint_fingerprint"],
        },
    )
    return payload


def _plan_rows_from_endpoint(
    plan: ProspectiveHistoryResponsePlan, endpoint: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_case = {str(row["case_id"]): dict(row) for row in plan.all_rows}
    output: list[dict[str, Any]] = []
    for endpoint_row in endpoint["rows"]:
        case_id = str(endpoint_row["case_id"])
        if case_id not in by_case:
            raise ValueError(f"Endpoint manifest contains an unknown case: {case_id}")
        row = {
            **by_case[case_id],
            "answer_star": str(endpoint_row["answer_star"]),
            "answer_star_side": str(endpoint_row["answer_star_side"]),
            "phase0_full_margin": float(endpoint_row["phase0_full_margin"]),
            "selection_slot": str(endpoint_row["selection_slot"]),
        }
        output.append(row)
    if len(output) != PRIMARY_N or len({row["item_id"] for row in output}) != PRIMARY_N:
        raise ValueError("Endpoint manifest is not a 40-item final cohort")
    return output


def _validate_endpoint_manifest(
    endpoint: Mapping[str, Any],
    plan: ProspectiveHistoryResponsePlan,
    output: Path,
) -> dict[str, Any]:
    """Re-derive endpoint identity, shape, and hidden linkage before reuse."""

    payload = dict(endpoint)
    fingerprint = payload.pop("endpoint_fingerprint", None)
    if not isinstance(fingerprint, str) or stable_hash(payload) != fingerprint:
        raise ValueError("Endpoint manifest fingerprint is missing or changed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != PRIMARY_N:
        raise ValueError("Endpoint manifest must contain exactly 40 rows")
    case_ids = [str(row.get("case_id", "")) for row in rows]
    item_ids = [str(row.get("item_id", "")) for row in rows]
    slots = [str(row.get("selection_slot", "")) for row in rows]
    if len(set(case_ids)) != PRIMARY_N or len(set(item_ids)) != PRIMARY_N:
        raise ValueError("Endpoint cases/items are not 40-way unique")
    if len(set(slots)) != PRIMARY_N or set(slots) != {
        str(row["case_id"]) for row in plan.primary_rows
    }:
        raise ValueError("Endpoint selection slots do not equal the frozen primary slots")
    fold_counts = {
        fold: sum(int(row.get("fold", -1)) == fold for row in rows)
        for fold in range(5)
    }
    if fold_counts != {fold: 8 for fold in range(5)}:
        raise ValueError(f"Endpoint fold allocation drifted: {fold_counts}")
    completed = {
        str(row["case_id"]): row
        for row in load_jsonl(output / PHASE0_RESULTS, repair_trailing=True)
        if row.get("status") == "completed"
    }
    plan_by_case = {str(value["case_id"]): value for value in plan.all_rows}
    for row in rows:
        case_id = str(row["case_id"])
        source = completed.get(case_id)
        if source is None:
            raise ValueError(f"Endpoint lacks completed Phase-0 source row: {case_id}")
        frozen = plan_by_case.get(case_id)
        if frozen is None:
            raise ValueError(f"Endpoint case is absent from the frozen plan: {case_id}")
        for field in ("item_id", "fold", "difficulty", "condition"):
            if str(source.get(field)) != str(frozen.get(field)):
                raise ValueError(
                    f"Phase-0 source differs from frozen plan: {case_id}/{field}"
                )
        derived_side = _answer_side(str(source.get("answer_star")), frozen)
        if derived_side == "other" or derived_side != str(source.get("answer_star_side")):
            raise ValueError(f"Phase-0 answer side is not derivable from frozen endpoints: {case_id}")
        if str(source.get("selection_slot")) != str(row["selection_slot"]):
            raise ValueError(f"Endpoint selection slot differs from Phase 0: {case_id}")
        expected_fields = {
            "case_id": str(source["case_id"]),
            "item_id": str(source["item_id"]),
            "fold": int(source["fold"]),
            "difficulty": str(source["difficulty"]),
            "condition": str(source["condition"]),
            "selection_slot": str(source["selection_slot"]),
            "answer_star": str(source["answer_star"]),
            "answer_star_side": str(source["answer_star_side"]),
            "phase0_full_margin": float(
                source["measurement"]["answer_distribution"][
                    "top1_top2_logit_margin"
                ]
            ),
            "phase0_full_entropy": _entropy(
                source["measurement"]["answer_distribution"][
                    "answer_class_probabilities"
                ]
            ),
            "phase0_hard_answer_image": 1.0
            if source["answer_star_side"] == "image"
            else 0.0,
            "phase0_hard_answer_other": 0.0,
        }
        for field, expected in expected_fields.items():
            observed = row.get(field)
            if isinstance(expected, float):
                equal = isinstance(observed, (int, float)) and math.isclose(
                    float(observed), expected, rel_tol=0.0, abs_tol=0.0
                )
            else:
                equal = observed == expected
            if not equal:
                raise ValueError(
                    f"Endpoint field differs from Phase 0: {case_id}/{field}: "
                    f"{observed!r} != {expected!r}"
                )
        if stable_hash(source["measurement"]) != str(
            row.get("phase0_measurement_fingerprint")
        ):
            raise ValueError(f"Endpoint measurement fingerprint changed: {case_id}")
        if str(source["hidden"]["sha256"]) != str(row.get("phase0_hidden_sha256")):
            raise ValueError(f"Endpoint hidden link changed: {case_id}")
        _validate_record_hidden(output, source)
    return {**payload, "endpoint_fingerprint": fingerprint}


def _behavior_effects(
    measurements: Mapping[str, Mapping[str, Any]], answer_star: str
) -> dict[str, float]:
    answer = str(normalize_answer(answer_star))
    logp: dict[str, float] = {}
    for condition in EVIDENCE_CONDITIONS:
        probability = float(measurements[condition]["answer_class_probabilities"][answer])
        if not math.isfinite(probability) or probability <= 0:
            raise ValueError(f"Invalid fixed-answer probability in {condition}: {probability}")
        logp[condition] = math.log(probability)
    deletion = logp["no_text"] - logp["no_image"]
    donor5 = logp["replace_text_d5"] - logp["replace_image_d5"]
    donor6 = logp["replace_text_d6"] - logp["replace_image_d6"]
    return {
        "fixed_answer_logp_full": logp["full"],
        "B_D": deletion,
        "M5": donor5,
        "M6": donor6,
        "B_M56": 0.5 * (donor5 + donor6),
        "replacement_donor_disagreement": 0.5 * (donor5 - donor6),
        "remove_image_drop_logp": logp["full"] - logp["no_image"],
        "remove_text_drop_logp": logp["full"] - logp["no_text"],
    }


def _entropy(probabilities: Mapping[str, Any]) -> float:
    return float(
        -sum(float(value) * math.log(float(value)) for value in probabilities.values() if float(value) > 0)
    )


def _flatten_branch(
    plan: ProspectiveHistoryResponsePlan,
    row: Mapping[str, Any],
    branch: str,
    measurements: Mapping[str, Mapping[str, Any]] | None,
    answer_measurement: AnswerOnlyReadoutMeasurement | None,
    fixed_margin_readouts: Mapping[str, Any] | None,
    joint: JointCommon9Measurement | None,
    hidden: Mapping[str, Any],
    *,
    elapsed_seconds: float,
    new_forward_count: int,
    reused_phase0: bool,
) -> dict[str, Any]:
    history_donor = row["history_donor"]
    donor5 = row["donor5"]
    donor6 = row["donor6"]
    history_exact = bool(row["history_answer_identity"]["ordered_pair_exact"])
    answer_messages = build_messages(plan, row, branch, "full")
    joint_messages, _ = _joint_branch_messages(
        plan, row, branch, str(row["answer_star"])
    )
    answer_history = answer_messages[:-2]
    joint_history = joint_messages[:-2]
    record: dict[str, Any] = {
        "format_version": 1,
        "experiment": "prospective_history_response_panel_v2",
        "phase": "phase1" if branch == "no_history" else "phase2",
        "intervention_key": f"{row['case_id']}::{branch}",
        "status": "completed",
        "case_id": str(row["case_id"]),
        "item_id": str(row["item_id"]),
        "prior_index": int(row["prior_index"]),
        "condition": str(row["condition"]),
        "difficulty": str(row["difficulty"]),
        "fold": int(row["fold"]),
        "text_answer": str(row["text_answer"]),
        "image_answer": str(row["image_answer"]),
        "answer_star": str(row["answer_star"]),
        "answer_star_side": str(row["answer_star_side"]),
        "selection_slot": str(row["selection_slot"]),
        "branch": branch,
        "branch_factors": history_branch_factors(row, branch),
        "history_match_tier": str(row["history_match_tier"]),
        "history_ordered_pair_exact": history_exact,
        "history_donor_item_id": str(history_donor["item_id"]),
        "history_donor_case_id": str(history_donor["case_id"]),
        "donor5_item_id": str(donor5["item_id"]),
        "donor5_case_id": str(donor5["case_id"]),
        "donor6_item_id": str(donor6["item_id"]),
        "donor6_case_id": str(donor6["case_id"]),
        "phase0_full_margin": float(row["phase0_full_margin"]),
        "history_prefix_equal_across_answer_and_joint": answer_history
        == joint_history,
        "answer_history_prefix_hash": canonical_message_hash(answer_history),
        "joint_history_prefix_hash": canonical_message_hash(joint_history),
        # This explicit False guards against later prose claiming that the two
        # final prompts are identical; common-nine intentionally asks for SA.
        "answer_joint_full_prompt_equal": False,
        "hidden": dict(hidden),
        "reused_phase0_full_and_u": bool(reused_phase0),
        "new_forward_count": int(new_forward_count),
        "formal_branch_forward_count_including_reuse": int(new_forward_count)
        + (2 if reused_phase0 else 0),
        "elapsed_seconds": float(elapsed_seconds),
        "causal_intervention": False,
        "history_is_external_prompt_manipulation": branch != "no_history",
        "steering_applied": False,
        "behavior_readout_measured": answer_measurement is not None,
        "report_formation_measured": joint is not None,
    }
    if answer_measurement is not None:
        if measurements is None or fixed_margin_readouts is None:
            raise ValueError("Behavior/readout branch lacks its measurement payload")
        effects = _behavior_effects(measurements, str(row["answer_star"]))
        current_u = answer_measurement.readouts
        fixed_u = fixed_margin_readouts
        full = measurements["full"]
        natural_side = _answer_side(str(full["predicted_answer"]), row)
        answer_only_no_sa = all(
            cell.get("verbal_sa_leakage") is False for cell in measurements.values()
        )
        record.update(
            {
                **effects,
                "B_target_shared": None,
                "U_prediction": float(fixed_u["primary_u"]["frozen_prediction"]),
                "U_coordinate": float(fixed_u["primary_u"]["coordinate"]),
                "U_L18_prediction": float(
                    fixed_u["secondary_u_l18"]["frozen_prediction"]
                ),
                "U_L18_coordinate": float(fixed_u["secondary_u_l18"]["coordinate"]),
                "U_nuisance_prediction": (
                    float(fixed_u["nuisance_only"]["frozen_prediction"])
                    if fixed_u["nuisance_only"]["frozen_prediction"] is not None
                    else None
                ),
                "U_nuisance_margin_source": "phase0_no_history_full_margin",
                "U_current_margin_sensitivity_prediction": float(
                    current_u["primary_u"]["frozen_prediction"]
                ),
                "U_current_margin_sensitivity_coordinate": float(
                    current_u["primary_u"]["coordinate"]
                ),
                "U_current_margin_sensitivity_nuisance_prediction": (
                    float(current_u["nuisance_only"]["frozen_prediction"])
                    if current_u["nuisance_only"]["frozen_prediction"] is not None
                    else None
                ),
                "U_current_margin_sensitivity_is_on_policy": str(
                    normalize_answer(full["predicted_answer"])
                )
                == str(normalize_answer(row["answer_star"]))
                and full.get("unique_top1") is True,
                "U_current_margin_sensitivity_answer_fixed_to_phase0": True,
                "full_logp": float(effects["fixed_answer_logp_full"]),
                "full_margin": float(full["top1_top2_logit_margin"]),
                "full_entropy": _entropy(full["answer_class_probabilities"]),
                "hard_answer_side": natural_side,
                "hard_answer_image": 1.0 if natural_side == "image" else 0.0,
                "hard_answer_other": 1.0 if natural_side == "other" else 0.0,
                "measurements": dict(measurements),
                "answer_readout": answer_measurement.to_payload(),
                "answer_only_no_sa": bool(answer_only_no_sa),
                "causal_prefix_equal": bool(
                    answer_measurement.causal_prefix_audit.get("passed") is True
                ),
                "answer_hook_exactly_once": bool(
                    answer_measurement.hook_audit.get("hook_exactly_once") is True
                    and answer_only_no_sa
                    and answer_measurement.causal_prefix_audit.get("passed") is True
                ),
            }
        )
    else:
        record.update(
            {
                "answer_only_no_sa": None,
                "causal_prefix_equal": None,
                "answer_hook_exactly_once": None,
            }
        )
    if joint is not None:
        attribution = joint.payload
        record.update(
            {
                "A_prediction": float(attribution["attribution_prediction"]),
                "A_coordinate": float(attribution["attribution_coordinate"]),
                "V": float(attribution["semantic_imageward_score"]),
                "V_hard_label": str(attribution["hard_label"]),
                "joint_common9": attribution,
                "joint_hook_exactly_once": bool(
                    attribution["hook_audit"].get("hook_exactly_once") is True
                ),
            }
        )
    else:
        record["joint_hook_exactly_once"] = None
    return record


def _measurement_from_phase0(
    phase0: Mapping[str, Any], hidden_path: Path
) -> AnswerOnlyReadoutMeasurement:
    payload = dict(phase0["measurement"])
    with np.load(hidden_path, allow_pickle=False) as archive:
        layers = [int(value) for value in archive["answer_layers"].tolist()]
        values = np.asarray(archive["answer_hidden"], dtype=np.float32)
    if values.shape[0] != len(layers):
        raise ValueError("Phase-0 hidden layer metadata is inconsistent")
    return AnswerOnlyReadoutMeasurement(
        answer_distribution=dict(payload["answer_distribution"]),
        answer_star=str(payload["answer_star"]),
        nuisance_row=dict(payload["nuisance_row"]),
        readouts=dict(payload["readouts"]),
        hidden_by_layer={layer: values[index] for index, layer in enumerate(layers)},
        hidden_checksums=dict(payload["hidden_checksums"]),
        causal_prefix_audit=dict(payload["causal_prefix_audit"]),
        hook_audit=dict(payload["hook_audit"]),
        teacher_forced_messages_hash=str(payload["teacher_forced_messages_hash"]),
        teacher_forced_rendered_sha256=str(payload["teacher_forced_rendered_sha256"]),
        length_path_replay=dict(payload["length_path_replay"]),
    )


def _measure_branch(
    runtime: Stage3Runtime,
    plan: ProspectiveHistoryResponsePlan,
    row: Mapping[str, Any],
    branch: str,
    readouts: ProspectiveReadoutRepository,
    output: Path,
    deadline: Deadline,
    *,
    phase0_row: Mapping[str, Any] | None = None,
    measure_behavior_readout: bool = True,
    measure_report_formation: bool = True,
) -> dict[str, Any]:
    if not measure_behavior_readout and not measure_report_formation:
        raise ValueError("At least one qualified Stage-09 track must be measured")
    if phase0_row is not None and not measure_behavior_readout:
        raise ValueError("Phase-0 reuse is only defined for the behavior/readout track")
    started = time.perf_counter()
    case = plan.cases[(str(row["item_id"]), int(row["prior_index"]))]
    answer_measurement: AnswerOnlyReadoutMeasurement | None = None
    fixed_margin_readouts: Mapping[str, Any] | None = None
    measurements: dict[str, dict[str, Any]] | None = None
    joint: JointCommon9Measurement | None = None
    hidden: dict[str, Any] = {}
    new_forward_count = 0
    reused_phase0 = False

    if measure_behavior_readout:
        if phase0_row is None:
            deadline.check()
            full_messages = build_messages(plan, row, branch, "full")
            answer_measurement = measure_answer_only_readouts(
                runtime,
                full_messages,
                answer_classes=case.answer_classes,
                fixed_answer=str(row["answer_star"]),
                base_row=row,
                readouts=readouts,
            )
            measurements = {"full": dict(answer_measurement.answer_distribution)}
            new_forward_count = 2
        else:
            _validate_hidden_reference(output, phase0_row["hidden"])
            hidden_path = output / str(phase0_row["hidden"]["path"])
            answer_measurement = _measurement_from_phase0(phase0_row, hidden_path)
            if answer_measurement.answer_star != str(row["answer_star"]):
                raise ValueError("Phase-0 answer differs from endpoint manifest")
            measurements = {"full": dict(answer_measurement.answer_distribution)}
            reused_phase0 = True
            hidden["answer"] = dict(phase0_row["hidden"])

        for condition in EVIDENCE_CONDITIONS[1:]:
            deadline.check()
            messages = build_messages(plan, row, branch, condition)
            measurements[condition] = restricted_next_answer_distribution(
                runtime,
                messages,
                answer_classes=case.answer_classes,
            )
            new_forward_count += 1

        # The primary U estimand holds both answer identity and nuisance margin
        # at their Phase-0 no-History values.  The projection produced inside
        # measure_answer_only_readouts uses the branch-current margin and is
        # retained only as an explicitly labelled sensitivity.
        frozen_nuisance = nuisance_row_for_fixed_answer(
            row,
            fixed_answer=str(row["answer_star"]),
            answer_margin=float(row["phase0_full_margin"]),
        )
        if reused_phase0:
            if not math.isclose(
                float(answer_measurement.nuisance_row["full_margin"]),
                float(row["phase0_full_margin"]),
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError("Stored Phase-0 readout does not use its frozen margin")
            # Preserve the original FP32 projection exactly; the on-disk FP16
            # vector is archival and must not introduce a second quantized U.
            fixed_margin_readouts = answer_measurement.readouts
        else:
            fixed_margin_readouts = project_answer_hidden(
                answer_measurement.hidden_by_layer,
                frozen_nuisance,
                readouts,
            )
        if not reused_phase0:
            answer_path = output / "hidden" / (
                f"phase2_BU_{_safe_name(str(row['case_id']) + '::' + branch)}.npz"
            )
            hidden["answer"] = _save_hidden_once(
                answer_path, **_answer_hidden_arrays(answer_measurement)
            )

    if measure_report_formation:
        joint_messages, assistant_text = _joint_branch_messages(
            plan, row, branch, str(row["answer_star"])
        )
        deadline.check()
        joint = measure_joint_common9(
            runtime,
            joint_messages,
            answer_star=str(row["answer_star"]),
            fold=int(row["fold"]),
            readouts=readouts,
            assistant_text=assistant_text,
        )
        new_forward_count += 1
        phase_label = "phase1" if reused_phase0 else "phase2_AV"
        attribution_path = output / "hidden" / (
            f"{phase_label}_{_safe_name(str(row['case_id']) + '::' + branch)}.npz"
        )
        hidden["attribution"] = _save_hidden_once(
            attribution_path,
            attribution_layer=np.asarray([18], dtype=np.int64),
            attribution_hidden=np.asarray(joint.hidden, dtype=np.float16)[None, :],
        )
    record = _flatten_branch(
        plan,
        row,
        branch,
        measurements,
        answer_measurement,
        fixed_margin_readouts,
        joint,
        hidden,
        elapsed_seconds=time.perf_counter() - started,
        new_forward_count=new_forward_count,
        reused_phase0=reused_phase0,
    )
    if measure_behavior_readout:
        record["B_target_shared"] = readouts.primary_u[
            int(row["fold"])
        ].transform_behavior(record["B_D"], record["B_M56"])
    return record


def _merge_track_rows(
    behavior: Mapping[str, Any] | None,
    report: Mapping[str, Any] | None,
    *,
    behavior_required: bool,
    report_required: bool,
) -> dict[str, Any] | None:
    """Merge independently resumable B/U and A/V technical outcomes.

    A successful track is never discarded because the other track failed.
    The merged row's ``status=completed`` means both track attempts are
    terminal; the explicit per-track statuses say whether measurement itself
    succeeded.  This lets qualification reject one technical track without
    suppressing the other.
    """

    sources = [dict(row) for row in (behavior, report) if isinstance(row, Mapping)]
    if not sources:
        return None
    keys = {str(row.get("intervention_key")) for row in sources}
    if len(keys) != 1:
        raise ValueError(f"Cannot merge different intervention keys: {keys}")
    behavior_status = (
        str(behavior.get("status")) if isinstance(behavior, Mapping) else "not_requested"
    )
    report_status = (
        str(report.get("status")) if isinstance(report, Mapping) else "not_requested"
    )
    merged: dict[str, Any] = {}
    skip = {
        "status",
        "hidden",
        "new_forward_count",
        "formal_branch_forward_count_including_reuse",
        "behavior_readout_measured",
        "report_formation_measured",
        "elapsed_seconds",
        "reused_phase0_full_and_u",
        "phase",
        "error",
        "error_type",
        "track",
        "attempted_forward_count",
        "attempted_forward_count_note",
        "terminal_track_status",
        "identical_attempt_n",
        "retry_errors",
        "attempt_errors",
        "answer_only_no_sa",
        "causal_prefix_equal",
        "causal_prefix_audit_scope",
        "steering_applied",
    }
    for source in sources:
        for key, value in source.items():
            if key in skip or value is None:
                continue
            if key in merged and stable_hash(merged[key]) != stable_hash(value):
                raise ValueError(f"Track metadata drift while merging {keys}: {key}")
            merged[key] = value
    hidden: dict[str, Any] = {}
    for source in sources:
        source_hidden = source.get("hidden")
        if isinstance(source_hidden, Mapping):
            for role, reference in source_hidden.items():
                if role in hidden and stable_hash(hidden[role]) != stable_hash(reference):
                    raise ValueError(f"Track hidden reference drift: {role}")
                hidden[str(role)] = reference
    required_complete = (
        (not behavior_required or behavior_status == "completed")
        and (not report_required or report_status == "completed")
    )
    merged.update(
        {
            # `status=completed` means both requested track attempts have a
            # terminal, auditable record. Measurement success lives in the
            # two explicit track-status fields and is never conflated here.
            "status": "completed",
            "all_requested_tracks_completed": bool(required_complete),
            "behavior_readout_status": behavior_status,
            "report_formation_status": report_status,
            "behavior_readout_required": bool(behavior_required),
            "report_formation_required": bool(report_required),
            "behavior_readout_measured": behavior_status == "completed",
            "report_formation_measured": report_status == "completed",
            "hidden": hidden,
            "new_forward_count": sum(
                int(source.get("new_forward_count", 0))
                for source in sources
                if source.get("status") == "completed"
            ),
            "formal_branch_forward_count_including_reuse": sum(
                int(source.get("formal_branch_forward_count_including_reuse", 0))
                for source in sources
                if source.get("status") == "completed"
            ),
            "track_errors": {
                "behavior_readout": {
                    "error_type": behavior.get("error_type"),
                    "error": behavior.get("error"),
                }
                if isinstance(behavior, Mapping) and behavior_status != "completed"
                else None,
                "report_formation": {
                    "error_type": report.get("error_type"),
                    "error": report.get("error"),
                }
                if isinstance(report, Mapping) and report_status != "completed"
                else None,
            },
            "track_runtime": {
                "behavior_readout": {
                    "elapsed_seconds": behavior.get("elapsed_seconds"),
                    "reused_phase0_full_and_u": behavior.get(
                        "reused_phase0_full_and_u"
                    ),
                    "new_forward_count": behavior.get("new_forward_count"),
                }
                if isinstance(behavior, Mapping)
                else None,
                "report_formation": {
                    "elapsed_seconds": report.get("elapsed_seconds"),
                    "reused_phase0_full_and_u": report.get(
                        "reused_phase0_full_and_u"
                    ),
                    "new_forward_count": report.get("new_forward_count"),
                }
                if isinstance(report, Mapping)
                else None,
            },
            "track_structural_audits": {
                "behavior_readout": {
                    "answer_only_no_sa": behavior.get("answer_only_no_sa"),
                    "causal_prefix_equal": behavior.get("causal_prefix_equal"),
                    "steering_applied": behavior.get("steering_applied"),
                }
                if isinstance(behavior, Mapping)
                else None,
                "report_formation": {
                    "answer_only_no_sa": report.get("answer_only_no_sa"),
                    "causal_prefix_equal": report.get("causal_prefix_equal"),
                    "causal_prefix_audit_scope": report.get(
                        "causal_prefix_audit_scope"
                    ),
                    "steering_applied": report.get("steering_applied"),
                }
                if isinstance(report, Mapping)
                else None,
            },
        }
    )
    # Qualification's common structural keys come from the immutable Phase-0
    # endpoint audit attached to the A/V track. B/U branch-local structural
    # failures are folded into its answer-hook gate above, so they cannot erase
    # an otherwise valid report-formation track.
    common_source = report if isinstance(report, Mapping) else behavior
    for key in ("answer_only_no_sa", "causal_prefix_equal"):
        merged[key] = common_source.get(key) if isinstance(common_source, Mapping) else None
    steering = [
        source.get("steering_applied")
        for source in sources
        if source.get("steering_applied") is not None
    ]
    merged["steering_applied"] = any(value is True for value in steering) if steering else None
    return merged


def _track_failure(
    row: Mapping[str, Any], branch: str, phase: str, track: str, exc: Exception
) -> dict[str, Any]:
    attempt_errors = getattr(exc, "attempt_errors", None)
    return {
        "format_version": 1,
        "experiment": "prospective_history_response_panel_v2",
        "phase": phase,
        "track": track,
        "intervention_key": f"{row['case_id']}::{branch}",
        "status": "failed",
        "case_id": str(row["case_id"]),
        "item_id": str(row["item_id"]),
        "fold": int(row["fold"]),
        "branch": branch,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "terminal_track_status": "failed",
        "identical_attempt_n": len(attempt_errors) if attempt_errors else 2,
        "attempt_errors": attempt_errors,
        "attempted_forward_count": None,
        "attempted_forward_count_note": (
            "the exception boundary cannot prove how many forwards completed; "
            "successful retained-cell totals exclude this technical overhead"
        ),
    }


def _attempt_track_measurement(
    measurement: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run one frozen track with at most one byte-identical technical retry."""

    errors: list[dict[str, str]] = []
    for retry in range(2):
        try:
            row = measurement()
            row["identical_attempt_n"] = retry + 1
            row["retry_errors"] = errors
            row["terminal_track_status"] = "completed"
            return row
        except TimeBudgetExceeded:
            # No terminal record: the same cell is resumable next invocation.
            raise
        except Exception as exc:
            errors.append({"error_type": type(exc).__name__, "error": str(exc)})
            if retry == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            terminal = RuntimeError(
                "track failed after two identical attempts: "
                + json.dumps(errors, ensure_ascii=False)
            )
            setattr(terminal, "attempt_errors", errors)
            raise terminal from exc
    raise AssertionError("unreachable track-attempt loop")


def run_phase1(
    runtime: Stage3Runtime,
    plan: ProspectiveHistoryResponsePlan,
    readouts: ProspectiveReadoutRepository,
    output: Path,
    deadline: Deadline,
    endpoint: Mapping[str, Any],
) -> dict[str, Any]:
    if (output / "qualification_gate.json").is_file():
        return _validated_qualification_gate(output)
    rows = _plan_rows_from_endpoint(plan, endpoint)
    phase0 = {
        str(row["case_id"]): row
        for row in load_jsonl(output / PHASE0_RESULTS, repair_trailing=True)
        if row.get("status") == "completed"
    }
    behavior_rows = _latest_by_key(output / PHASE1_BEHAVIOR_RESULTS)
    report_rows = _latest_by_key(output / PHASE1_REPORT_RESULTS)
    for row in rows:
        key = f"{row['case_id']}::no_history"
        behavior = behavior_rows.get(key)
        if behavior is None:
            try:
                if str(row["case_id"]) not in phase0:
                    raise ValueError(
                        f"Completed Phase-0 row is missing for {row['case_id']}"
                    )
                behavior = _attempt_track_measurement(
                    lambda: _measure_branch(
                        runtime,
                        plan,
                        row,
                        "no_history",
                        readouts,
                        output,
                        deadline,
                        phase0_row=phase0[str(row["case_id"])],
                        measure_behavior_readout=True,
                        measure_report_formation=False,
                    )
                )
            except TimeBudgetExceeded:
                raise
            except Exception as exc:
                behavior = _track_failure(
                    row, "no_history", "phase1", "behavior_readout", exc
                )
            _upsert_jsonl(output / PHASE1_BEHAVIOR_RESULTS, behavior)
            behavior_rows[key] = behavior
        else:
            _validate_terminal_track(output, behavior)

        report = report_rows.get(key)
        if report is None:
            try:
                report = _attempt_track_measurement(
                    lambda: _measure_branch(
                        runtime,
                        plan,
                        row,
                        "no_history",
                        readouts,
                        output,
                        deadline,
                        measure_behavior_readout=False,
                        measure_report_formation=True,
                    )
                )
            except TimeBudgetExceeded:
                raise
            except Exception as exc:
                report = _track_failure(
                    row, "no_history", "phase1", "report_formation", exc
                )
            _upsert_jsonl(output / PHASE1_REPORT_RESULTS, report)
            report_rows[key] = report
        else:
            _validate_terminal_track(output, report)

        # Common structural qualification is independently inherited from the
        # already successful Phase-0 endpoint, never from whether the B/U
        # Phase-1 supplement happened to succeed.
        phase0_source = phase0.get(str(row["case_id"]))
        if phase0_source is None:
            raise ValueError(f"Missing Phase-0 structural source for {key}")
        phase0_measurement = phase0_source["measurement"]
        report = dict(report)
        report.update(
            {
                "answer_only_no_sa": phase0_measurement["answer_distribution"].get(
                    "verbal_sa_leakage"
                )
                is False,
                "causal_prefix_equal": phase0_measurement["causal_prefix_audit"].get(
                    "passed"
                )
                is True,
                "causal_prefix_audit_scope": "phase0_no_history_endpoint",
                "steering_applied": False,
            }
        )
        _upsert_jsonl(output / PHASE1_REPORT_RESULTS, report)
        report_rows[key] = report

        merged = _merge_track_rows(
            behavior,
            report,
            behavior_required=True,
            report_required=True,
        )
        if merged is None:
            raise RuntimeError(f"No Phase-1 track result exists for {key}")
        _upsert_jsonl(output / PHASE1_RESULTS, merged)
        _upsert_jsonl(output / FORMAL_RESULTS, merged)
        if behavior.get("status") == "completed":
            _upsert_jsonl(output / FORMAL_BEHAVIOR_RESULTS, behavior)
        if report.get("status") == "completed":
            _upsert_jsonl(output / FORMAL_REPORT_RESULTS, report)

    completed = [
        row
        for row in _latest_by_key(output / PHASE1_RESULTS).values()
        if row.get("status") == "completed"
    ]
    if len(completed) != PRIMARY_N:
        raise RuntimeError(
            f"Phase 1 did not reach 40 terminal merged rows: {len(completed)}"
        )
    gate = qualification_gate(completed, expected_n=PRIMARY_N)
    behavior_authorized, report_authorized = _gate_authorizations(gate)
    immutable_json(output / "qualification_gate.json", gate)
    atomic_write_json(
        output / "phase1_summary.json",
        {
            "status": "complete" if len(completed) == PRIMARY_N else "technical_incomplete",
            "completed_n": len(completed),
            "expected_n": PRIMARY_N,
            "behavior_readout_completed_n": sum(
                row.get("status") == "completed" for row in behavior_rows.values()
            ),
            "report_formation_completed_n": sum(
                row.get("status") == "completed" for row in report_rows.values()
            ),
            "qualification_passed": bool(gate["passed"]),
            "authorizations": dict(gate["authorizations"]),
            "gate_components": gate["components"],
        },
    )
    if not behavior_authorized:
        immutable_json(
            output / "behavior_readout_history_skipped.json",
            {
                "status": "skipped",
                "track": "behavior_readout_history",
                "reason": "predeclared no-History B/U qualification track failed",
                "authorization": False,
                "qualification_gate_fingerprint": stable_hash(gate),
                "components": gate["tracks"]["behavior_readout"]["components"],
                "activation_intervention_run": False,
            },
        )
    if not report_authorized:
        immutable_json(
            output / "report_formation_history_skipped.json",
            {
                "status": "skipped",
                "track": "report_formation_history",
                "reason": "predeclared no-History A/V qualification track failed",
                "authorization": False,
                "qualification_gate_fingerprint": stable_hash(gate),
                "components": gate["tracks"]["report_formation"]["components"],
                "activation_intervention_run": False,
            },
        )
    if not behavior_authorized and not report_authorized:
        immutable_json(
            output / "phase2_skipped.json",
            {
                "status": "skipped",
                "reason": "both independently frozen qualification tracks failed",
                "authorizations": dict(gate["authorizations"]),
                "activation_intervention_run": False,
            },
        )
    return gate


def run_phase2(
    runtime: Stage3Runtime,
    plan: ProspectiveHistoryResponsePlan,
    readouts: ProspectiveReadoutRepository,
    output: Path,
    deadline: Deadline,
    endpoint: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    behavior_authorized, report_authorized = _gate_authorizations(gate)
    if not behavior_authorized and not report_authorized:
        return {"status": "skipped", "reason": "both qualification tracks failed"}
    rows = _plan_rows_from_endpoint(plan, endpoint)
    behavior_rows = _latest_by_key(output / FORMAL_BEHAVIOR_RESULTS)
    report_rows = _latest_by_key(output / FORMAL_REPORT_RESULTS)
    for row in rows:
        for branch in HISTORY_BRANCH_FACTORS:
            key = f"{row['case_id']}::{branch}"
            behavior = behavior_rows.get(key) if behavior_authorized else None
            if behavior_authorized:
                if behavior is None:
                    try:
                        behavior = _attempt_track_measurement(
                            lambda: _measure_branch(
                                runtime,
                                plan,
                                row,
                                branch,
                                readouts,
                                output,
                                deadline,
                                measure_behavior_readout=True,
                                measure_report_formation=False,
                            )
                        )
                    except TimeBudgetExceeded:
                        raise
                    except Exception as exc:
                        behavior = _track_failure(
                            row, branch, "phase2", "behavior_readout", exc
                        )
                    _upsert_jsonl(output / FORMAL_BEHAVIOR_RESULTS, behavior)
                    behavior_rows[key] = behavior
                else:
                    _validate_terminal_track(output, behavior)

            report = report_rows.get(key) if report_authorized else None
            if report_authorized:
                if report is None:
                    try:
                        report = _attempt_track_measurement(
                            lambda: _measure_branch(
                                runtime,
                                plan,
                                row,
                                branch,
                                readouts,
                                output,
                                deadline,
                                measure_behavior_readout=False,
                                measure_report_formation=True,
                            )
                        )
                    except TimeBudgetExceeded:
                        raise
                    except Exception as exc:
                        report = _track_failure(
                            row, branch, "phase2", "report_formation", exc
                        )
                    _upsert_jsonl(output / FORMAL_REPORT_RESULTS, report)
                    report_rows[key] = report
                else:
                    _validate_terminal_track(output, report)
                report = dict(report)
                report.update(
                    {
                        "answer_only_no_sa": not any(
                            "Source Attribution" in str(message.get("content"))
                            for message in build_messages(
                                plan, row, branch, "full"
                            )
                        ),
                        "history_answer_message_no_sa": True,
                        "causal_prefix_equal": True,
                        "causal_prefix_audit_scope": (
                            "phase0_no_history_endpoint; History A/V-only branch "
                            "did not capture an answer-only hidden state"
                        ),
                        "steering_applied": False,
                    }
                )
                _upsert_jsonl(output / FORMAL_REPORT_RESULTS, report)
                report_rows[key] = report

            merged = _merge_track_rows(
                behavior,
                report,
                behavior_required=behavior_authorized,
                report_required=report_authorized,
            )
            if merged is None:
                raise RuntimeError(f"No authorized track result exists for {key}")
            _upsert_jsonl(output / FORMAL_RESULTS, merged)

    completed = [
        row for row in _latest_by_key(output / FORMAL_RESULTS).values()
        if row.get("status") == "completed"
    ]
    expected = PRIMARY_N * len(BRANCHES)
    behavior_completed = sum(
        row.get("status") == "completed" for row in behavior_rows.values()
    )
    report_completed = sum(
        row.get("status") == "completed" for row in report_rows.values()
    )
    authorized_tracks_complete = bool(
        len(completed) == expected
        and (not behavior_authorized or behavior_completed == expected)
        and (not report_authorized or report_completed == expected)
    )
    payload = {
        "status": "complete" if authorized_tracks_complete else "technical_incomplete",
        "terminal_branch_n": len(completed),
        "expected_branch_n": expected,
        "unique_item_n": len({str(row["item_id"]) for row in completed}),
        "authorizations": dict(gate["authorizations"]),
        "behavior_readout_history_run": behavior_authorized,
        "report_formation_history_run": report_authorized,
        "behavior_readout_completed_branch_n": behavior_completed
        if behavior_authorized
        else 0,
        "report_formation_completed_branch_n": report_completed
        if report_authorized
        else 0,
        "retained_successful_cell_forward_total": sum(
            int(row.get("new_forward_count", 0)) for row in completed
        )
        + 2 * PRIMARY_N,
        "technical_failure_forward_overhead": "not exactly recoverable after an exception",
    }
    atomic_write_json(output / "phase2_summary.json", payload)
    return payload


def analyze(output: Path) -> dict[str, Any]:
    phase1 = [
        row for row in _latest_by_key(output / PHASE1_RESULTS).values()
        if row.get("status") == "completed"
    ]
    gate = (
        qualification_gate(phase1, expected_n=PRIMARY_N)
        if len(phase1) == PRIMARY_N
        else None
    )
    behavior_authorized = report_authorized = False
    if gate is not None:
        behavior_authorized, report_authorized = _gate_authorizations(gate)
        gate_path = output / "qualification_gate.json"
        if gate_path.is_file():
            saved = json.loads(gate_path.read_text(encoding="utf-8"))
            if saved != gate:
                raise ValueError("Analysis gate differs from the finalized gate")
    records = [
        row for row in _latest_by_key(output / FORMAL_RESULTS).values()
        if row.get("status") == "completed"
    ]
    # A merged terminal row can legitimately contain two exhausted technical
    # failures.  It is auditable progress, but it is not a measured branch and
    # therefore has no donor/outcome payload for factorial analysis.  Keep it
    # in terminal counts while excluding it from the analysis input.  Rows
    # with either successful track remain available, and each track's complete
    # item count is checked independently below.
    measured_records = [
        row
        for row in records
        if row.get("behavior_readout_status") == "completed"
        or row.get("report_formation_status") == "completed"
    ]
    contrasts = build_factorial_rows(measured_records) if measured_records else []
    expected_branches = PRIMARY_N * len(BRANCHES)
    behavior_outcomes = ("B_D", "B_M56", "U_prediction")
    report_outcomes = ("A_prediction", "V")
    behavior_secondary_outcomes = (
        "M5",
        "M6",
        "U_coordinate",
        "U_L18_prediction",
        "U_L18_coordinate",
        "full_logp",
        "full_margin",
        "full_entropy",
        "hard_answer_image",
        "hard_answer_other",
    )
    report_secondary_outcomes = ("A_coordinate",)

    def complete_for(outcomes: Sequence[str]) -> list[dict[str, Any]]:
        required = [
            f"{family}_{effect}_{outcome}"
            for family in ("relevant", "irrelevant", "did")
            for effect in ("modality", "replay", "interaction", "history_vs_none")
            for outcome in outcomes
        ]
        return [
            row
            for row in contrasts
            if all(
                row.get(key) is not None and math.isfinite(float(row[key]))
                for key in required
            )
        ]

    behavior_contrasts = complete_for(behavior_outcomes)
    report_contrasts = complete_for(report_outcomes)
    behavior_completed_branches = sum(
        row.get("behavior_readout_status") == "completed" for row in records
    )
    report_completed_branches = sum(
        row.get("report_formation_status") == "completed" for row in records
    )
    authorized_complete = bool(
        gate is not None
        and (
            not behavior_authorized
            or (
                behavior_completed_branches == expected_branches
                and len(behavior_contrasts) == PRIMARY_N
            )
        )
        and (
            not report_authorized
            or (
                report_completed_branches == expected_branches
                and len(report_contrasts) == PRIMARY_N
            )
        )
    )
    summary: dict[str, Any] = {
        "format_version": 1,
        "title": "Stage 09 v2 — Prospective History Response Pilot",
        "status": (
            "complete"
            if authorized_complete
            else "qualification_failed"
            if gate is not None and not behavior_authorized and not report_authorized
            else "partial"
        ),
        "phase1_completed_n": len(phase1),
        "formal_terminal_branch_n": len(records),
        "formal_expected_branch_n": expected_branches,
        "factorial_structural_item_n": len(contrasts),
        "qualification_gate": gate,
        "authorizations": dict(gate["authorizations"]) if gate else None,
        "tracks": {
            "behavior_readout_history": {
                "authorized": behavior_authorized,
                "completed_branch_n": behavior_completed_branches,
                "expected_branch_n_if_authorized": expected_branches,
                "complete_factorial_item_n": len(behavior_contrasts),
                "outcomes": list(behavior_outcomes),
                "status": "complete"
                if behavior_authorized
                and behavior_completed_branches == expected_branches
                and len(behavior_contrasts) == PRIMARY_N
                else "skipped"
                if not behavior_authorized
                else "partial",
            },
            "report_formation_history": {
                "authorized": report_authorized,
                "completed_branch_n": report_completed_branches,
                "expected_branch_n_if_authorized": expected_branches,
                "complete_factorial_item_n": len(report_contrasts),
                "outcomes": list(report_outcomes),
                "status": "complete"
                if report_authorized
                and report_completed_branches == expected_branches
                and len(report_contrasts) == PRIMARY_N
                else "skipped"
                if not report_authorized
                else "partial",
            },
        },
        "claim_boundary": {
            "behavior_history_prompt_causal_effect_allowed": bool(
                behavior_authorized and len(behavior_contrasts) == PRIMARY_N
            ),
            "report_history_prompt_causal_effect_allowed": bool(
                report_authorized and len(report_contrasts) == PRIMARY_N
            ),
            "causal_unit": "complete external History prompt bundle",
            "pure_relevance_effect_allowed": False,
            "internal_mediation_claim_allowed": False,
            "activation_intervention_run": False,
        },
    }
    if behavior_contrasts:
        summary["behavior_readout_factorial_effects"] = factorial_effect_summaries(
            behavior_contrasts, outcomes=behavior_outcomes
        )
        summary["change_reliability"] = change_reliability(behavior_contrasts)
        summary[
            "behavior_readout_leave_one_reused_donor_cluster_out"
        ] = leave_one_reused_donor_cluster_out(
            behavior_contrasts, outcomes=behavior_outcomes
        )
        summary.setdefault("secondary_total_effects", {})[
            "behavior_readout_history"
        ] = factorial_effect_summaries(
            behavior_contrasts, outcomes=behavior_secondary_outcomes
        )
    if report_contrasts:
        summary["report_formation_factorial_effects"] = factorial_effect_summaries(
            report_contrasts, outcomes=report_outcomes
        )
        summary[
            "report_formation_leave_one_reused_donor_cluster_out"
        ] = leave_one_reused_donor_cluster_out(
            report_contrasts, outcomes=report_outcomes
        )
        summary.setdefault("secondary_total_effects", {})[
            "report_formation_history"
        ] = factorial_effect_summaries(
            report_contrasts, outcomes=report_secondary_outcomes
        )
    if len(behavior_contrasts) == PRIMARY_N and len(report_contrasts) == PRIMARY_N:
        # Alignment is explicitly cross-track and is meaningful only after all
        # five outcomes are present for all 40 recipient items.
        summary["shift_alignment"] = shift_alignment(contrasts)
        summary["shift_alignment_scope"] = {
            "effects_included": [
                "relevant_modality",
                "irrelevant_modality",
                "did_modality",
                "relevant_replay",
                "irrelevant_replay",
                "did_replay",
            ],
            "effects_not_included": ["interaction", "history_vs_none"],
            "reason": (
                "the preregistered change-reliability and direct specialization "
                "comparisons were defined for modality/replay shifts only"
            ),
        }
    else:
        summary["shift_alignment_skipped"] = (
            "cross-track alignment requires complete authorized B/U and A/V panels"
        )
    summary["artifact_inputs"] = {
        name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
        for name in (
            PHASE0_RESULTS,
            PHASE1_BEHAVIOR_RESULTS,
            PHASE1_REPORT_RESULTS,
            PHASE1_RESULTS,
            FORMAL_BEHAVIOR_RESULTS,
            FORMAL_REPORT_RESULTS,
            FORMAL_RESULTS,
        )
        if (output / name).is_file()
    }
    atomic_write_json(output / "summary.json", summary)
    lines = [
        "# Stage 09 v2 — Prospective History Response Pilot",
        "",
        f"- Status: `{summary['status']}`",
        f"- Phase 1 complete items: `{len(phase1)}/{PRIMARY_N}`",
        f"- Terminal formal branches: `{len(records)}/{expected_branches}`",
        f"- Complete B/U items: `{len(behavior_contrasts)}/{PRIMARY_N}`",
        f"- Complete A/V items: `{len(report_contrasts)}/{PRIMARY_N}`",
        f"- B/U History authorization: `{behavior_authorized}`",
        f"- A/V History authorization: `{report_authorized}`",
        "- Internal mediation claim authorized: `False`",
        "- Activation intervention executed: `False`",
        "",
        "The saved JSON contains the full fixed-fold bootstrap, factorial-effect, "
        "change-reliability, and B/U/A/V shift-alignment tables.",
        "",
    ]
    atomic_write_text(output / "summary.md", "\n".join(lines))
    return summary


def gpu_smoke(
    runtime: Stage3Runtime,
    plan: ProspectiveHistoryResponsePlan,
    readouts: ProspectiveReadoutRepository,
    output: Path,
    deadline: Deadline,
) -> dict[str, Any]:
    row = dict(plan.primary_rows[0])
    case = plan.cases[(str(row["item_id"]), int(row["prior_index"]))]
    messages = build_messages(plan, row, "no_history", "full")
    deadline.check()
    answer = measure_answer_only_readouts(
        runtime,
        messages,
        answer_classes=case.answer_classes,
        fixed_answer=None,
        base_row=row,
        readouts=readouts,
    )
    side = _answer_side(answer.answer_star, row)
    if side == "other":
        raise RuntimeError("Smoke endpoint is outside the Text/Image conflict pair")
    smoke_row = {
        **row,
        "answer_star": answer.answer_star,
        "answer_star_side": side,
        "phase0_full_margin": float(
            answer.answer_distribution["top1_top2_logit_margin"]
        ),
        "selection_slot": str(row["case_id"]),
    }
    branch = "relevant_image_ai"
    answer_messages = build_messages(plan, smoke_row, branch, "full")
    joint_messages, assistant_text = _joint_branch_messages(
        plan, smoke_row, branch, answer.answer_star
    )
    if len(answer_messages) != 4 or len(joint_messages) != 4:
        raise RuntimeError("History smoke did not render a four-message conversation")
    if answer_messages[:2] != joint_messages[:2]:
        raise RuntimeError("Answer/joint History prefixes differ in smoke")
    deadline.check()
    history_answer = measure_answer_only_readouts(
        runtime,
        answer_messages,
        answer_classes=case.answer_classes,
        fixed_answer=answer.answer_star,
        base_row=smoke_row,
        readouts=readouts,
    )
    frozen_nuisance = nuisance_row_for_fixed_answer(
        smoke_row,
        fixed_answer=answer.answer_star,
        answer_margin=float(smoke_row["phase0_full_margin"]),
    )
    fixed_projection = project_answer_hidden(
        history_answer.hidden_by_layer, frozen_nuisance, readouts
    )
    deadline.check()
    joint = measure_joint_common9(
        runtime,
        joint_messages,
        answer_star=answer.answer_star,
        fold=int(row["fold"]),
        readouts=readouts,
        assistant_text=assistant_text,
    )
    hidden_path = output / "hidden" / "gpu_smoke.npz"
    hidden = _save_hidden_once(
        hidden_path,
        no_history_answer_layers=np.asarray(
            sorted(answer.hidden_by_layer), dtype=np.int64
        ),
        no_history_answer_hidden=np.stack(
            [
                np.asarray(answer.hidden_by_layer[layer], dtype=np.float16)
                for layer in sorted(answer.hidden_by_layer)
            ]
        ),
        history_answer_layers=np.asarray(
            sorted(history_answer.hidden_by_layer), dtype=np.int64
        ),
        history_answer_hidden=np.stack(
            [
                np.asarray(history_answer.hidden_by_layer[layer], dtype=np.float16)
                for layer in sorted(history_answer.hidden_by_layer)
            ]
        ),
        attribution_layer=np.asarray([18], dtype=np.int64),
        attribution_hidden=np.asarray(joint.hidden, dtype=np.float16)[None, :],
    )
    current_side = _answer_side(
        history_answer.answer_distribution["predicted_answer"], smoke_row
    )
    return {
        "status": "passed",
        "case_id": row["case_id"],
        "answer_star": answer.answer_star,
        "answer_star_side": side,
        "history_branch": branch,
        "history_message_count": len(answer_messages),
        "history_prefix_equal_across_answer_and_joint": answer_messages[:2]
        == joint_messages[:2],
        "answer_joint_full_prompt_equal": answer_messages == joint_messages,
        "answer_only_no_verbal_sa": history_answer.answer_distribution[
            "verbal_sa_leakage"
        ]
        is False,
        "phase0_answer_causal_prefix": answer.causal_prefix_audit,
        "history_answer_causal_prefix": history_answer.causal_prefix_audit,
        "history_answer_hook_audit": history_answer.hook_audit,
        "joint_hook_audit": joint.payload["hook_audit"],
        "joint_protocol": joint.payload["protocol"],
        "primary_u_frozen_margin_prediction": fixed_projection["primary_u"][
            "frozen_prediction"
        ],
        "primary_u_current_margin_sensitivity_prediction": history_answer.readouts[
            "primary_u"
        ]["frozen_prediction"],
        "frozen_nuisance_margin": frozen_nuisance["full_margin"],
        "current_history_margin": history_answer.nuisance_row["full_margin"],
        "hard_answer_side": current_side,
        "hard_answer_image": 1.0 if current_side == "image" else 0.0,
        "hard_answer_other": 1.0 if current_side == "other" else 0.0,
        "other_path_representable": current_side in {"text", "image", "other"},
        "hidden": hidden,
        "smoke_forward_count": 5,
        "formal_outcome": False,
    }


def _validate_gpu_smoke(
    payload: Mapping[str, Any],
    plan: ProspectiveHistoryResponsePlan,
    output: Path,
) -> dict[str, Any]:
    expected_case = str(plan.primary_rows[0]["case_id"])
    checks = {
        "status": payload.get("status") == "passed",
        "case": str(payload.get("case_id")) == expected_case,
        "history_branch": payload.get("history_branch") == "relevant_image_ai",
        "history_message_count": payload.get("history_message_count") == 4,
        "history_prefix_equal": payload.get(
            "history_prefix_equal_across_answer_and_joint"
        )
        is True,
        "full_prompt_not_equal": payload.get("answer_joint_full_prompt_equal") is False,
        "no_verbal_sa": payload.get("answer_only_no_verbal_sa") is True,
        "phase0_prefix": payload.get("phase0_answer_causal_prefix", {}).get("passed")
        is True,
        "history_prefix": payload.get("history_answer_causal_prefix", {}).get("passed")
        is True,
        "answer_hook": payload.get("history_answer_hook_audit", {}).get(
            "hook_exactly_once"
        )
        is True,
        "joint_hook": payload.get("joint_hook_audit", {}).get("hook_exactly_once")
        is True,
        "protocol": payload.get("joint_protocol") == "common_9_ordered",
        "other_path": payload.get("other_path_representable") is True,
        "hard_side": payload.get("hard_answer_side") in {"text", "image", "other"},
        "hard_indicators": payload.get("hard_answer_image") in {0.0, 1.0}
        and payload.get("hard_answer_other") in {0.0, 1.0},
        "forward_count": payload.get("smoke_forward_count") == 5,
        "formal_outcome": payload.get("formal_outcome") is False,
    }
    for key in (
        "primary_u_frozen_margin_prediction",
        "primary_u_current_margin_sensitivity_prediction",
        "frozen_nuisance_margin",
        "current_history_margin",
    ):
        try:
            checks[f"finite_{key}"] = math.isfinite(float(payload.get(key)))
        except (TypeError, ValueError):
            checks[f"finite_{key}"] = False
    if not all(checks.values()):
        raise ValueError(f"Saved GPU smoke failed validation: {checks}")
    _validate_hidden_reference(output, payload.get("hidden", {}))
    return dict(payload)


def _progress_counts(output: Path) -> dict[str, int]:
    phase0 = _latest_by_key(output / PHASE0_RESULTS)
    phase1 = _latest_by_key(output / PHASE1_RESULTS)
    formal = _latest_by_key(output / FORMAL_RESULTS)
    behavior = _latest_by_key(output / FORMAL_BEHAVIOR_RESULTS)
    report = _latest_by_key(output / FORMAL_REPORT_RESULTS)
    return {
        "phase0_completed_n": sum(row.get("status") == "completed" for row in phase0.values()),
        "phase1_completed_n": sum(row.get("status") == "completed" for row in phase1.values()),
        "formal_completed_branch_n": sum(row.get("status") == "completed" for row in formal.values()),
        "behavior_readout_completed_branch_n": sum(
            row.get("status") == "completed" for row in behavior.values()
        ),
        "report_formation_completed_branch_n": sum(
            row.get("status") == "completed" for row in report.values()
        ),
    }


def _gate_authorizations(gate: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return the two authoritative, independently frozen Phase-2 gates."""

    authorizations = gate.get("authorizations")
    if not isinstance(authorizations, Mapping):
        raise ValueError("Qualification gate lacks authoritative `authorizations`")
    required = ("behavior_readout_history", "report_formation_history")
    if any(key not in authorizations for key in required):
        raise ValueError(f"Qualification gate lacks exact authorization keys: {required}")
    behavior = authorizations["behavior_readout_history"]
    report = authorizations["report_formation_history"]
    if not isinstance(behavior, bool) or not isinstance(report, bool):
        raise TypeError("Qualification authorizations must be booleans")
    return behavior, report


def _validated_qualification_gate(output: Path) -> dict[str, Any]:
    """Recompute the frozen gate from Phase-1 rows before every downstream use."""

    gate_path = output / "qualification_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError("Phase 2 requires a finalized qualification_gate.json")
    rows = [
        row
        for row in _latest_by_key(output / PHASE1_RESULTS).values()
        if row.get("status") == "completed"
    ]
    if len(rows) != PRIMARY_N:
        raise ValueError(
            f"Final qualification gate requires 40 terminal merged rows; got {len(rows)}"
        )
    recomputed = qualification_gate(rows, expected_n=PRIMARY_N)
    saved = json.loads(gate_path.read_text(encoding="utf-8"))
    if saved != recomputed:
        raise ValueError("Saved qualification gate differs from deterministic recomputation")
    _gate_authorizations(recomputed)
    return recomputed


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    modes = sum(bool(value) for value in (args.dry_run, args.smoke_only, args.analyze_only))
    if modes > 1:
        raise ValueError("--dry-run, --smoke-only, and --analyze-only are mutually exclusive")
    if args.analyze_only and args.phase != "auto":
        raise ValueError("--analyze-only does not accept an explicit --phase")

    artifacts = SAFormationArtifacts.discover(args.experiment_dir)
    output = validate_output(artifacts.experiment_dir, args.output_dir)
    plan = build_plan(artifacts)
    protocol = protocol_manifest()
    candidate = cohort_candidate_manifest(plan)
    donors = donor_manifest(plan)
    cohort = _primary_cohort_manifest(plan, candidate)
    if args.dry_run:
        message_audit = audit_plan_messages(plan, include_reserve=False)
        joint_audit = _joint_message_manifest(
            ProspectiveHistoryResponsePlan(
                artifacts=plan.artifacts,
                cases=plan.cases,
                inventory=plan.inventory,
                primary_rows=plan.primary_rows,
                reserve_rows=tuple(),
                all_rows=plan.primary_rows,
                donor_diagnostics=plan.donor_diagnostics,
            )
        )
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "cuda_available": torch.cuda.is_available(),
                    "output_dir": str(output),
                    "candidate_n": candidate["candidate_n"],
                    "candidate_item_n": candidate["candidate_item_n"],
                    "primary_n": cohort["primary_n"],
                    "branches": list(BRANCHES),
                    "evidence_conditions": list(EVIDENCE_CONDITIONS),
                    "formal_forward_count": PRIMARY_N * len(BRANCHES) * 9,
                    "answer_message_audit_passed": message_audit["passed"],
                    "joint_message_audit_passed": joint_audit["passed"],
                    "fingerprints": {
                        "protocol": protocol["protocol_fingerprint"],
                        "candidate": candidate["cohort_fingerprint"],
                        "cohort": cohort["cohort_fingerprint"],
                        "donor": donors["donor_fingerprint"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    readouts = load_prospective_readout_repository(artifacts.experiment_dir)
    _freeze_cpu_artifacts(
        artifacts,
        output,
        plan,
        readouts,
        resume=args.resume or args.analyze_only,
    )
    if args.analyze_only:
        analyze(output)
        return 0

    atomic_write_json(
        output / "progress.json",
        {
            "status": "preflight_complete",
            "requested_phase": args.phase,
            **_progress_counts(output),
        },
    )
    if not torch.cuda.is_available():
        atomic_write_json(
            output / "gpu_skipped.json",
            {
                "status": "gpu_skipped",
                "reason": "torch.cuda.is_available() is false",
                "formal_forward_count": 0,
                "requested_phase": args.phase,
            },
        )
        atomic_write_json(
            output / "progress.json",
            {"status": "gpu_skipped", "requested_phase": args.phase, **_progress_counts(output)},
        )
        return 0

    deadline = Deadline(args.max_minutes)
    runtime = Stage3Runtime(artifacts)
    try:
        smoke_path = output / "gpu_smoke.json"
        if smoke_path.is_file():
            _validate_gpu_smoke(
                json.loads(smoke_path.read_text(encoding="utf-8")), plan, output
            )
        else:
            smoke = gpu_smoke(runtime, plan, readouts, output, deadline)
            _validate_gpu_smoke(smoke, plan, output)
            immutable_json(smoke_path, smoke)
        if args.smoke_only:
            atomic_write_json(
                output / "progress.json",
                {
                    "status": "smoke_complete",
                    "requested_phase": args.phase,
                    **_progress_counts(output),
                },
            )
            return 0

        endpoint: Mapping[str, Any] | None = None
        gate: Mapping[str, Any] | None = None
        if args.phase in {"auto", "phase0"}:
            endpoint = run_phase0(runtime, plan, readouts, output, deadline)
        elif (output / "endpoint_manifest.json").is_file():
            endpoint = _validate_endpoint_manifest(
                json.loads(
                    (output / "endpoint_manifest.json").read_text(encoding="utf-8")
                ),
                plan,
                output,
            )
        else:
            raise FileNotFoundError("Phase 1/2 requires endpoint_manifest.json from Phase 0")

        if args.phase in {"auto", "phase1"}:
            gate = run_phase1(runtime, plan, readouts, output, deadline, endpoint)
        elif (output / "qualification_gate.json").is_file():
            gate = _validated_qualification_gate(output)

        if args.phase in {"auto", "phase2"}:
            if gate is None:
                raise FileNotFoundError("Phase 2 requires qualification_gate.json from Phase 1")
            run_phase2(runtime, plan, readouts, output, deadline, endpoint, gate)
    except TimeBudgetExceeded as exc:
        atomic_write_json(
            output / "progress.json",
            {
                "status": "budget_exhausted",
                "requested_phase": args.phase,
                "reason": str(exc),
                "resume_command": "rerun the identical command with --resume",
                **_progress_counts(output),
            },
        )
        return 0
    except Exception as exc:
        atomic_write_json(
            output / "progress.json",
            {
                "status": "technical_failure",
                "requested_phase": args.phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                **_progress_counts(output),
            },
        )
        raise

    summary = analyze(output)
    atomic_write_json(
        output / "progress.json",
        {
            "status": summary["status"],
            "requested_phase": args.phase,
            "summary": str(output / "summary.json"),
            **_progress_counts(output),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
