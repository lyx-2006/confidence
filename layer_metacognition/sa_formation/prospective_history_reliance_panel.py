"""Versioned prospective History/reliance panel (Stage 09).

This experiment is deliberately downstream of the completed method-v2
Actual Source Reliance, donor-replication, reliance-representation, and
confirmatory attribution panels.  Every endpoint, donor, protocol, and gate is
frozen before a new model forward.  The module never mutates those inputs and
never re-selects the fixed answer after adding History.

The primary cohort contains 72 previously held-out method-v2 confirmatory
items.  Three ``answer_star_side=other`` cases and one case whose frozen donor
stratum has fewer than three unused items are structurally excluded before any
new outcome.  This is a prospective test of future candidate measurements; it
cannot retroactively alter the conclusions of stages 01, 03, or 07 and cannot
authorize a causal mediator.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from confidence_test.dataset_utils import EvaluationCase, load_evaluation_cases
from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from confidence_test.prompt_utils import STAGE1_TEXT_ANSWER_PROMPT
from layer_metacognition.hidden_state_store import load_jsonl
from layer_metacognition.probe.prompts import IMAGE_ONLY_ANSWER_PROMPT

from .confirmatory_attribution_panel import (
    CORE_PROTOCOL_NAMES,
    FrozenDirectionRepository,
    build_joint_messages,
    load_confirmatory_cohort,
    panel_protocols,
    panel_root as attribution_panel_root,
)
from .core import (
    SAFormationArtifacts,
    atomic_save_npz,
    canonical_message_hash,
    stable_hash,
    write_jsonl_atomic,
)
from .donor_replication_extension import _icc_consistency
from .history_grounding import direct_messages_fixed_answer_distribution
from .reliance_measurement import (
    build_answer_only_messages,
    contains_verbal_sa_request,
)
from .reliance_representation_sensitivity import (
    load_frozen_target_transforms,
    load_sensitivity_inputs,
)
from .runtime import Stage3Runtime, prepare_measurement
from .second_order import ProtocolAnalyzer, build_answer_history_messages


BRIDGE_DIR = "stage3_sa_computational_bridge"
PANEL_DIR = "09_prospective_history_reliance_panel"
PANEL_VERSION = 1
SEED = 42
BOOTSTRAP_ITERATIONS = 1000
PRIMARY_LAYER = 18
PRIMARY_POSITION = "panl"
PRIMARY_N = 72
SOURCE_COMPLETED_N = 76

BRANCHES = (
    "no_history",
    "relevant_text",
    "relevant_image",
    "irrelevant_text",
    "irrelevant_image",
)
HISTORY_BRANCHES = BRANCHES[1:]
FRESH_DONOR_INDICES = (5, 6)
HISTORY_CONDITIONS = (
    "full",
    "no_text",
    "no_image",
    "replace_text_d5",
    "replace_image_d5",
    "replace_text_d6",
    "replace_image_d6",
)
NO_HISTORY_NEW_CONDITIONS = HISTORY_CONDITIONS[3:]
JOINT_PROTOCOL_NAME = "common_9_ordered"

OTHER_ENDPOINT_EXCLUSIONS = (
    "131__prior_2__conflict_hard__v4__joint",
    "159__prior_2__conflict_hard__v4__joint",
    "190__prior_0__conflict_hard__v4__joint",
)
DONOR_EXHAUSTION_EXCLUSION = "197__prior_5__conflict_easy__v4__joint"
STRUCTURAL_EXCLUSIONS = (*OTHER_ENDPOINT_EXCLUSIONS, DONOR_EXHAUSTION_EXCLUSION)


def panel_root(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir).resolve() / BRIDGE_DIR / PANEL_DIR


def _numeric_key(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return int(text), text
    except ValueError:
        return 10**12, text


def _latest_rows(path: str | Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path, repair_trailing=True):
        key = str(row.get("intervention_key") or row.get("case_id") or "")
        if key:
            latest[key] = row
    return list(latest.values())


def append_record_atomic(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append through an atomic whole-file replacement.

    Stage 09 has at most a few hundred branch records.  Rewriting the small
    JSONL file makes every resume boundary recoverable without relying on a
    partially written trailing line.
    """

    destination = Path(path)
    rows = load_jsonl(destination, repair_trailing=True) if destination.exists() else []
    rows.append(dict(row))
    write_jsonl_atomic(destination, rows)


def _case_map(artifacts: SAFormationArtifacts) -> dict[tuple[str, int], EvaluationCase]:
    cases, _ = load_evaluation_cases(artifacts.dataset)
    return {(str(case.item_id), int(case.prior_index)): case for case in cases}


def _history_prompt_length(case: EvaluationCase) -> int:
    text = STAGE1_TEXT_ANSWER_PROMPT.format(
        question=case.question,
        text_clue=case.text_clue,
    )
    image = IMAGE_ONLY_ANSWER_PROMPT.format(question=case.question)
    return len(text) + len(image)


def _prior_bin(case: EvaluationCase) -> str:
    return str(case.prior_bin or "")


def _history_rank(
    target: Mapping[str, Any],
    candidate: Mapping[str, Any],
    cases: Mapping[tuple[str, int], EvaluationCase],
) -> tuple[Any, ...]:
    target_case = cases[(str(target["item_id"]), int(target["prior_index"]))]
    donor_case = cases[(str(candidate["item_id"]), int(candidate["prior_index"]))]
    return (
        int(_prior_bin(donor_case) != _prior_bin(target_case)),
        int(str(candidate["condition"]) != str(target["condition"])),
        abs(_history_prompt_length(donor_case) - _history_prompt_length(target_case)),
        _numeric_key(candidate["item_id"]),
        int(candidate["prior_index"]),
        str(candidate["case_id"]),
    )


def _replacement_rank(
    target: Mapping[str, Any],
    candidate: Mapping[str, Any],
    cases: Mapping[tuple[str, int], EvaluationCase],
) -> tuple[Any, ...]:
    target_case = cases[(str(target["item_id"]), int(target["prior_index"]))]
    donor_case = cases[(str(candidate["item_id"]), int(candidate["prior_index"]))]
    return (
        int(_prior_bin(donor_case) != _prior_bin(target_case)),
        int(str(candidate["condition"]) != str(target["condition"])),
        abs(len(donor_case.text_clue) - len(target_case.text_clue)),
        _numeric_key(candidate["item_id"]),
        int(candidate["prior_index"]),
        str(candidate["case_id"]),
    )


@dataclass
class ProspectivePlan:
    rows: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    endpoint_audit: dict[str, Any]
    cases: dict[tuple[str, int], EvaluationCase]
    method_pool: list[dict[str, Any]]
    source_by_case: dict[str, dict[str, Any]]
    attribution_by_case: dict[str, dict[str, Any]]
    sensitivity_inputs: Any


def _load_attribution_rows(experiment_dir: str | Path) -> dict[str, dict[str, Any]]:
    path = attribution_panel_root(experiment_dir) / "results.jsonl"
    rows = [row for row in _latest_rows(path) if row.get("status") == "completed"]
    if len(rows) != SOURCE_COMPLETED_N:
        raise ValueError(
            f"Stage 09 requires {SOURCE_COMPLETED_N} completed Stage-06 rows; got {len(rows)}"
        )
    output = {str(row["case_id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("Stage-06 confirmatory rows are not case-unique")
    for row in rows:
        protocol = row.get("protocols", {}).get(JOINT_PROTOCOL_NAME)
        if not isinstance(protocol, dict):
            raise ValueError(f"Stage-06 row lacks {JOINT_PROTOCOL_NAME}: {row['case_id']}")
        if protocol.get("hook_exactly_once") is not True:
            raise ValueError(f"Stage-06 hook audit failed: {row['case_id']}")
    return output


def build_plan(artifacts: SAFormationArtifacts) -> ProspectivePlan:
    """Build the deterministic prospective cohort and all donor roles."""

    endpoints, endpoint_audit = load_confirmatory_cohort(
        artifacts.experiment_dir,
        expected_completed=SOURCE_COMPLETED_N,
    )
    sensitivity = load_sensitivity_inputs(artifacts.experiment_dir)
    cases = _case_map(artifacts)
    attribution = _load_attribution_rows(artifacts.experiment_dir)

    source_by_case: dict[str, dict[str, Any]] = {}
    method_pool: list[dict[str, Any]] = []
    for split in ("development", "confirmatory"):
        for joined in sensitivity.joined[split]:
            measurement = dict(joined["measurement"])
            method_pool.append(measurement)
            if split == "confirmatory":
                source_by_case[str(joined["case_id"])] = dict(joined)
    if len(method_pool) != 173 or len({str(row["item_id"]) for row in method_pool}) != 173:
        raise ValueError("Stage 09 requires the 173 unique completed method-v2 rows")
    if set(source_by_case) != {str(row["case_id"]) for row in endpoints}:
        raise ValueError("Method-v2/Stage-03 confirmatory case lineage differs")
    if set(attribution) != set(source_by_case):
        raise ValueError("Stage-06 and method-v2 confirmatory case sets differ")

    planned: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for endpoint in endpoints:
        case_id = str(endpoint["case_id"])
        source = source_by_case[case_id]
        measurement = source["measurement"]
        donor_extension = source["donor"]
        if endpoint["answer_star_side"] == "other":
            if case_id not in OTHER_ENDPOINT_EXCLUSIONS:
                raise ValueError(f"Unexpected other-endpoint case: {case_id}")
            excluded.append(
                {
                    "case_id": case_id,
                    "item_id": str(endpoint["item_id"]),
                    "reason": "frozen_answer_star_side_other",
                    "outcome_observed": False,
                }
            )
            continue

        legacy_items = {
            str(endpoint["item_id"]),
            str(measurement["donor1_item_id"]),
            str(measurement["donor2_item_id"]),
            str(donor_extension["donor3_item_id"]),
            str(donor_extension["donor4_item_id"]),
        }
        eligible = [
            row
            for row in method_pool
            if str(row["item_id"]) not in legacy_items
            and int(row["fold"]) == int(endpoint["fold"])
            and str(row["difficulty"]) == str(endpoint["difficulty"])
            and str(row["answer_star_side"]) == str(endpoint["answer_star_side"])
        ]
        unique: dict[str, dict[str, Any]] = {}
        for row in eligible:
            unique.setdefault(str(row["item_id"]), row)
        eligible = list(unique.values())
        if len(eligible) < 3:
            if case_id != DONOR_EXHAUSTION_EXCLUSION or len(eligible) != 1:
                raise ValueError(
                    f"Unexpected donor-stratum exhaustion for {case_id}: {len(eligible)}"
                )
            excluded.append(
                {
                    "case_id": case_id,
                    "item_id": str(endpoint["item_id"]),
                    "reason": "frozen_donor_stratum_exhaustion_after_target_d1_d4",
                    "eligible_distinct_candidate_n": len(eligible),
                    "outcome_observed": False,
                }
            )
            continue

        history_donor = min(
            eligible,
            key=lambda row: _history_rank(measurement, row, cases),
        )
        remaining = [
            row for row in eligible if str(row["item_id"]) != str(history_donor["item_id"])
        ]
        ranked = sorted(
            remaining,
            key=lambda row: _replacement_rank(measurement, row, cases),
        )
        donor5, donor6 = ranked[:2]
        role_items = {
            str(history_donor["item_id"]),
            str(donor5["item_id"]),
            str(donor6["item_id"]),
        }
        if len(role_items) != 3 or role_items.intersection(legacy_items):
            raise RuntimeError(f"Fresh donor roles collide for {case_id}")

        prediction = source["predictions"]["raw_choice_coupled"]
        no_history_joint = attribution[case_id]["protocols"][JOINT_PROTOCOL_NAME]
        row = {
            **dict(endpoint),
            "text_answer": str(measurement["text_answer"]),
            "image_answer": str(measurement["image_answer"]),
            "prior_strength": float(measurement.get("prior_strength", 0.0)),
            "full_margin": float(measurement["full_margin"]),
            "method_v2_row_fingerprint": stable_hash(measurement),
            "donor_extension_row_fingerprint": stable_hash(donor_extension),
            "stage03_raw_prediction_row_fingerprint": stable_hash(prediction),
            "stage06_common9_row_fingerprint": stable_hash(no_history_joint),
            "behavior_delete_imageward": float(
                measurement["behavior_delete_imageward"]
            ),
            "behavior_replace_imageward_d1": float(
                measurement["behavior_replace_imageward_d1"]
            ),
            "behavior_replace_imageward_d2": float(
                measurement["behavior_replace_imageward_d2"]
            ),
            "behavior_replace_imageward_d12_mean": float(
                measurement["behavior_replace_imageward"]
            ),
            "behavior_replace_imageward_d3": float(
                donor_extension["behavior_replace_imageward_d3"]
            ),
            "behavior_replace_imageward_d4": float(
                donor_extension["behavior_replace_imageward_d4"]
            ),
            "behavior_replace_imageward_d34_mean": float(
                donor_extension["behavior_replace_imageward_d34_mean"]
            ),
            "prediction_replacement": float(prediction["prediction_replacement"]),
            "prediction_shared": float(prediction["prediction_shared"]),
            "prediction_nuisance": float(prediction["prediction_nuisance"]),
            "target_deletion_stage03": float(prediction["target_deletion"]),
            "target_replacement_stage03": float(prediction["target_replacement"]),
            "target_shared_stage03": float(prediction["target_shared"]),
            "no_history_common9": {
                "semantic_imageward_score": float(
                    no_history_joint["semantic_imageward_score"]
                ),
                "hard_label": str(no_history_joint["hard_label"]),
                "frozen_prediction": float(no_history_joint["frozen_prediction"]),
                "frozen_coordinate": float(no_history_joint["frozen_coordinate"]),
                "prefix_hash": str(no_history_joint["prefix_hash"]),
                "hook_exactly_once": bool(no_history_joint["hook_exactly_once"]),
            },
            "eligible_fresh_candidate_n": len(eligible),
            "history_donor": _donor_payload(history_donor),
            "donor5": _donor_payload(donor5),
            "donor6": _donor_payload(donor6),
            "legacy_donors": {
                f"donor{index}": {
                    "case_id": str(
                        measurement[f"donor{index}_case_id"]
                        if index <= 2
                        else donor_extension[f"donor{index}_case_id"]
                    ),
                    "item_id": str(
                        measurement[f"donor{index}_item_id"]
                        if index <= 2
                        else donor_extension[f"donor{index}_item_id"]
                    ),
                }
                for index in (1, 2, 3, 4)
            },
            "method_v2_reused_measurements": {
                name: measurement["measurements"][name]
                for name in ("full", "no_text", "no_image")
            },
        }
        planned.append(row)

    planned.sort(key=lambda row: (int(row["fold"]), _numeric_key(row["item_id"])))
    excluded.sort(key=lambda row: _numeric_key(row["item_id"]))
    if len(planned) != PRIMARY_N:
        raise ValueError(f"Prospective primary cohort drifted: {len(planned)} != {PRIMARY_N}")
    if {row["case_id"] for row in excluded} != set(STRUCTURAL_EXCLUSIONS):
        raise ValueError("Prospective structural exclusion set drifted")
    if len({str(row["item_id"]) for row in planned}) != PRIMARY_N:
        raise ValueError("Prospective cohort is not item-unique")
    return ProspectivePlan(
        rows=planned,
        excluded=excluded,
        endpoint_audit=endpoint_audit,
        cases=cases,
        method_pool=method_pool,
        source_by_case=source_by_case,
        attribution_by_case=attribution,
        sensitivity_inputs=sensitivity,
    )


def _donor_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(row["case_id"]),
        "item_id": str(row["item_id"]),
        "prior_index": int(row["prior_index"]),
        "condition": str(row["condition"]),
        "difficulty": str(row["difficulty"]),
        "fold": int(row["fold"]),
        "answer_star": str(row["answer_star"]),
        "answer_star_side": str(row["answer_star_side"]),
        "text_answer": str(row["text_answer"]),
        "image_answer": str(row["image_answer"]),
    }


def protocol_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "panel_version": PANEL_VERSION,
        "experiment": "prospective_heldout_history_reliance_panel",
        "primary_n": PRIMARY_N,
        "source_completed_n": SOURCE_COMPLETED_N,
        "branches": list(BRANCHES),
        "history_branches": list(HISTORY_BRANCHES),
        "behavior_conditions": {
            "history": list(HISTORY_CONDITIONS),
            "no_history_new": list(NO_HISTORY_NEW_CONDITIONS),
            "no_history_reused": ["full", "no_text", "no_image"],
        },
        "endpoint": {
            "field": "method-v2 confirmatory answer_star",
            "selected_once": "no-history full answer-only context in Stage 01",
            "history_reselection": False,
            "post_treatment_endpoint_filtering": False,
        },
        "history": {
            "relevant": "target source-only historical question",
            "irrelevant": "same-fold/difficulty/A*-side donor historical question",
            "modality": ["text", "image"],
            "replayed_answer": "target A_T in every relevant/irrelevant branch",
            "semantic_limit": (
                "The irrelevant donor question is paired with target A_T to hold the "
                "assistant token fixed; it can therefore be semantically incongruent."
            ),
        },
        "fresh_donors": {
            "indices": list(FRESH_DONOR_INDICES),
            "selected_before_new_outcomes": True,
            "same_fold_difficulty_answer_star_side": True,
            "exclude": ["target", "d1", "d2", "d3", "d4", "history_donor"],
            "same_pair_all_branches": True,
            "text_image_replacement_symmetric": True,
        },
        "behavior_formulas": {
            "D": "log P(A*|no_text)-log P(A*|no_image)",
            "M5": "log P(A*|replace_text_d5)-log P(A*|replace_image_d5)",
            "M6": "log P(A*|replace_text_d6)-log P(A*|replace_image_d6)",
            "M56": "(M5+M6)/2",
            "M12": "frozen Stage-01 donor mean",
            "M34": "frozen Stage-02 donor mean",
        },
        "joint": {
            "protocol": JOINT_PROTOCOL_NAME,
            "answer": "teacher-forced identical A*",
            "layer": PRIMARY_LAYER,
            "position": PRIMARY_POSITION,
            "A_primary": "frozen Stage-10 standardized coordinate",
            "V_primary": "restricted semantic imageward SA score",
            "no_history": "exact reuse from Stage 06",
            "history_is_out_of_distribution": True,
        },
        "primary_contrasts": {
            "within_relevance": "image-history minus text-history",
            "relevance_did": "relevant modality effect minus irrelevant modality effect",
            "layers": ["B_deletion", "B_M56", "A", "V"],
            "alignment": "itemwise pairwise Spearman and sign agreement",
        },
        "prospective_gate": {
            "technical": "complete_case_n>=70 and latest_failed_branch_n=0",
            "replacement_replication": [
                "M12-vs-M56 fixed-fold/item Spearman CI lower>0",
                "M34-vs-M56 fixed-fold/item Spearman CI lower>0",
                "ICC(M5,M6)>=0.60",
            ],
            "replacement_transport": (
                "frozen raw Stage-03 prediction_replacement vs fold-transformed M56 "
                "Spearman CI lower>0"
            ),
            "shared_transport": [
                "frozen raw Stage-03 prediction_shared vs shared(D,M56) R2>0",
                "Spearman CI lower>0",
                ">=4 folds with positive Spearman",
                "paired squared-error improvement over frozen nuisance CI lower>0",
            ],
            "cross_method": "D-vs-M56 Spearman CI lower>0",
        },
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": SEED,
            "unit": "item, resampled within each frozen fold",
        },
        "claim_scope": (
            "Future candidate only; never retroactively flips Stage 01/03/07 and "
            "never authorizes a causal mediator."
        ),
        "formal_new_forward_count": PRIMARY_N * 36,
        "formal_new_behavior_forward_count": PRIMARY_N * 32,
        "formal_new_joint_forward_count": PRIMARY_N * 4,
    }
    payload["protocol_fingerprint"] = stable_hash(payload)
    return payload


def cohort_manifest(plan: ProspectivePlan) -> dict[str, Any]:
    rows = []
    for row in plan.rows:
        rows.append(
            {
                key: row[key]
                for key in (
                    "case_id",
                    "item_id",
                    "prior_index",
                    "condition",
                    "difficulty",
                    "fold",
                    "answer_star",
                    "answer_star_side",
                    "text_answer",
                    "image_answer",
                    "method_v2_intervention_key",
                    "method_v2_manifest_fingerprint",
                    "method_v2_selection_messages_hash",
                    "method_v2_teacher_forced_messages_hash",
                    "method_v2_selection_rendered_hash",
                    "method_v2_answer_star_token_id",
                    "method_v2_row_fingerprint",
                    "donor_extension_row_fingerprint",
                    "stage03_raw_prediction_row_fingerprint",
                    "stage06_common9_row_fingerprint",
                )
            }
        )
    payload: dict[str, Any] = {
        "format_version": 1,
        "panel_version": PANEL_VERSION,
        "n": len(rows),
        "unique_items": len({row["item_id"] for row in rows}),
        "fold_counts": {
            str(fold): sum(int(row["fold"]) == fold for row in rows)
            for fold in range(5)
        },
        "structural_exclusions": plan.excluded,
        "endpoint_audit": plan.endpoint_audit,
        "rows": rows,
    }
    payload["cohort_fingerprint"] = stable_hash(payload)
    return payload


def donor_manifest(plan: ProspectivePlan) -> dict[str, Any]:
    rows = []
    for row in plan.rows:
        rows.append(
            {
                "case_id": row["case_id"],
                "item_id": row["item_id"],
                "fold": row["fold"],
                "difficulty": row["difficulty"],
                "answer_star_side": row["answer_star_side"],
                "eligible_fresh_candidate_n": row["eligible_fresh_candidate_n"],
                "legacy_donors": row["legacy_donors"],
                "history_donor": row["history_donor"],
                "donor5": row["donor5"],
                "donor6": row["donor6"],
            }
        )
    payload: dict[str, Any] = {
        "format_version": 1,
        "panel_version": PANEL_VERSION,
        "selection_time": "before every Stage-09 outcome",
        "candidate_pool": "173 unique completed method-v2 development+confirmatory items",
        "history_rank": "prior-bin, condition, combined Text/Image prompt length, item",
        "replacement_rank": "prior-bin, condition, text-clue length, item",
        "constraints": protocol_manifest()["fresh_donors"],
        "rows": rows,
    }
    payload["donor_fingerprint"] = stable_hash(payload)
    return payload


def _target_case(plan: ProspectivePlan, row: Mapping[str, Any]) -> EvaluationCase:
    return plan.cases[(str(row["item_id"]), int(row["prior_index"]))]


def _donor_case(
    plan: ProspectivePlan, payload: Mapping[str, Any]
) -> EvaluationCase:
    return plan.cases[(str(payload["item_id"]), int(payload["prior_index"]))]


def history_prefix(
    plan: ProspectivePlan,
    row: Mapping[str, Any],
    branch: str,
) -> list[dict[str, Any]]:
    if branch == "no_history":
        return []
    if branch not in HISTORY_BRANCHES:
        raise ValueError(f"Unknown branch: {branch}")
    relevance, modality = branch.split("_", 1)
    target = _target_case(plan, row)
    if relevance == "relevant":
        history = target
        history_condition = str(row["condition"])
    else:
        payload = row["history_donor"]
        history = _donor_case(plan, payload)
        history_condition = str(payload["condition"])
    built = build_answer_history_messages(
        target,
        str(row["condition"]),
        history,
        history_condition,
        modality,
        str(row["text_answer"]),
    )
    prefix = built[:2]
    if len(prefix) != 2 or [message["role"] for message in prefix] != ["user", "assistant"]:
        raise RuntimeError(f"Invalid History prefix for {row['case_id']}/{branch}")
    return prefix


def condition_sources(
    plan: ProspectivePlan,
    row: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    target = _target_case(plan, row)
    target_image = str(target.conditions[str(row["condition"])].resolved_image_path)
    null_image = str(target.conditions["null"].resolved_image_path)
    output: dict[str, dict[str, Any]] = {
        "full": {
            "text_clue": target.text_clue,
            "image_path": target_image,
            "text_source_item": str(row["item_id"]),
            "image_source_item": str(row["item_id"]),
        },
        "no_text": {
            "text_clue": "[No text clue available.]",
            "image_path": target_image,
            "text_source_item": None,
            "image_source_item": str(row["item_id"]),
        },
        "no_image": {
            "text_clue": target.text_clue,
            "image_path": null_image,
            "text_source_item": str(row["item_id"]),
            "image_source_item": None,
        },
    }
    for index in FRESH_DONOR_INDICES:
        payload = row[f"donor{index}"]
        donor = _donor_case(plan, payload)
        donor_image = str(
            donor.conditions[str(payload["condition"])].resolved_image_path
        )
        output[f"replace_text_d{index}"] = {
            "text_clue": donor.text_clue,
            "image_path": target_image,
            "text_source_item": str(payload["item_id"]),
            "image_source_item": str(row["item_id"]),
        }
        output[f"replace_image_d{index}"] = {
            "text_clue": target.text_clue,
            "image_path": donor_image,
            "text_source_item": str(row["item_id"]),
            "image_source_item": str(payload["item_id"]),
        }
    if tuple(output) != HISTORY_CONDITIONS:
        raise RuntimeError(f"Stage-09 condition order drifted: {tuple(output)}")
    for index in FRESH_DONOR_INDICES:
        text = output[f"replace_text_d{index}"]
        image = output[f"replace_image_d{index}"]
        if (
            text["text_source_item"] != image["image_source_item"]
            or text["image_source_item"] != image["text_source_item"]
        ):
            raise RuntimeError(f"Fresh donor {index} is asymmetric")
    return output


def build_behavior_messages(
    plan: ProspectivePlan,
    row: Mapping[str, Any],
    branch: str,
    condition: str,
) -> list[dict[str, Any]]:
    sources = condition_sources(plan, row)
    if condition not in sources:
        raise ValueError(f"Unknown condition: {condition}")
    target = _target_case(plan, row)
    source = sources[condition]
    base = build_answer_only_messages(
        target,
        text_clue=str(source["text_clue"]),
        image_path=str(source["image_path"]),
    )
    messages = [*history_prefix(plan, row, branch), *base]
    if contains_verbal_sa_request(messages):
        raise ValueError("Answer-only Stage-09 branch leaks Source Attribution")
    return messages


def build_joint_history_messages(
    plan: ProspectivePlan,
    row: Mapping[str, Any],
    branch: str,
) -> tuple[list[dict[str, Any]], str, Any]:
    if branch not in HISTORY_BRANCHES:
        raise ValueError("Only History branches require new joint forwards")
    protocol = next(
        value for value in panel_protocols() if value.name == JOINT_PROTOCOL_NAME
    )
    target = _target_case(plan, row)
    base, assistant_text = build_joint_messages(
        target,
        str(row["condition"]),
        protocol,
        answer_star=str(row["answer_star"]),
    )
    messages = [*history_prefix(plan, row, branch), *base]
    return messages, assistant_text, protocol


def audit_plan_messages(plan: ProspectivePlan) -> dict[str, Any]:
    """CPU-only structural audit performed before any GPU authorization."""

    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    for row in plan.rows:
        case_id = str(row["case_id"])
        sources = condition_sources(plan, row)
        expected_full = str(row["method_v2_selection_messages_hash"])
        reconstructed_full = canonical_message_hash(
            build_behavior_messages(plan, row, "no_history", "full")
        )
        if expected_full != reconstructed_full:
            failures.append(f"{case_id}: no-history Full message hash drift")
        prefixes: dict[str, set[str]] = defaultdict(set)
        final_hashes: dict[str, set[str]] = defaultdict(set)
        for branch in BRANCHES:
            conditions = (
                NO_HISTORY_NEW_CONDITIONS
                if branch == "no_history"
                else HISTORY_CONDITIONS
            )
            for condition in conditions:
                messages = build_behavior_messages(plan, row, branch, condition)
                prefix = messages[:-2]
                prefixes[branch].add(canonical_message_hash(prefix))
                final_hashes[condition].add(canonical_message_hash(messages[-2:]))
            if len(prefixes[branch]) != 1:
                failures.append(f"{case_id}/{branch}: History prefix changes by condition")
        for condition in NO_HISTORY_NEW_CONDITIONS:
            if len(final_hashes[condition]) != len(BRANCHES):
                # The same final turn should collapse to one hash across branches.
                failures.append(f"{case_id}/{condition}: final target turn differs by History")
        donor_items = [
            str(row["item_id"]),
            *[
                str(row["legacy_donors"][f"donor{index}"]["item_id"])
                for index in (1, 2, 3, 4)
            ],
            str(row["history_donor"]["item_id"]),
            str(row["donor5"]["item_id"]),
            str(row["donor6"]["item_id"]),
        ]
        if len(donor_items) != len(set(donor_items)):
            failures.append(f"{case_id}: target/donor roles are not item-distinct")
        rows.append(
            {
                "case_id": case_id,
                "no_history_full_hash": reconstructed_full,
                "method_v2_full_hash_equal": reconstructed_full == expected_full,
                "branch_prefix_hashes": {
                    branch: next(iter(values)) for branch, values in prefixes.items()
                },
                "fresh_source_items": {
                    condition: {
                        key: value
                        for key, value in source.items()
                        if key != "text_clue"
                    }
                    for condition, source in sources.items()
                },
            }
        )
    return {
        "status": "passed" if not failures else "failed",
        "n": len(plan.rows),
        "failures": failures,
        "all_answer_star_fixed_before_history": True,
        "all_no_history_full_hashes_equal_method_v2": not any(
            "Full message hash" in value for value in failures
        ),
        "irrelevant_history_semantic_limit": protocol_manifest()["history"][
            "semantic_limit"
        ],
        "rows": rows,
    }


def _safe_record(
    base: Mapping[str, Any], operation: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = operation()
        return {**dict(base), **result, "elapsed_seconds": time.perf_counter() - started}
    except Exception as exc:
        return {
            **dict(base),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }


def _log_probability(measurement: Mapping[str, Any], answer: str) -> float:
    if measurement.get("fixed_answer_log_probability") is not None:
        value = float(measurement["fixed_answer_log_probability"])
    else:
        probability = float(measurement["answer_class_probabilities"][answer])
        if probability <= 0.0 or not math.isfinite(probability):
            raise ValueError("Fixed-answer restricted probability is non-positive")
        value = math.log(probability)
    if not math.isfinite(value):
        raise ValueError("Fixed-answer log probability is non-finite")
    return value


def _behavior_effects(
    measurements: Mapping[str, Mapping[str, Any]], answer: str
) -> dict[str, float]:
    logp = {name: _log_probability(value, answer) for name, value in measurements.items()}
    required = set(HISTORY_CONDITIONS)
    if not required.issubset(logp):
        raise ValueError(f"Behavior branch omits {sorted(required.difference(logp))}")
    deletion = logp["no_text"] - logp["no_image"]
    donor5 = logp["replace_text_d5"] - logp["replace_image_d5"]
    donor6 = logp["replace_text_d6"] - logp["replace_image_d6"]
    return {
        "behavior_delete_imageward": deletion,
        "behavior_replace_imageward_d5": donor5,
        "behavior_replace_imageward_d6": donor6,
        "behavior_replace_imageward_d56_mean": 0.5 * (donor5 + donor6),
        "replacement_d56_disagreement": 0.5 * (donor5 - donor6),
        "fixed_answer_logp_full": logp["full"],
    }


def measure_behavior_branch(
    runtime: Stage3Runtime,
    plan: ProspectivePlan,
    row: Mapping[str, Any],
    branch: str,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    if branch not in BRANCHES:
        raise ValueError(branch)
    answer = str(row["answer_star"])
    target = _target_case(plan, row)
    measurements: dict[str, dict[str, Any]] = {}
    reused: list[str] = []
    if branch == "no_history":
        for condition, source in row["method_v2_reused_measurements"].items():
            measurements[str(condition)] = dict(source)
            reused.append(str(condition))
        conditions = NO_HISTORY_NEW_CONDITIONS
    else:
        conditions = HISTORY_CONDITIONS
    history_hashes: set[str] = set()
    final_hashes: dict[str, str] = {}
    for condition in conditions:
        deadline()
        messages = build_behavior_messages(plan, row, branch, condition)
        measured = direct_messages_fixed_answer_distribution(
            runtime,
            messages,
            answer_classes=target.answer_classes,
            fixed_answer=answer,
        )
        measured["condition"] = condition
        measured["history_prefix_hash"] = canonical_message_hash(messages[:-2])
        measured["final_turn_hash"] = canonical_message_hash(messages[-2:])
        measured["verbal_sa_leakage"] = contains_verbal_sa_request(messages)
        if measured["verbal_sa_leakage"]:
            raise ValueError("Answer-only formal branch contains an SA request")
        history_hashes.add(measured["history_prefix_hash"])
        final_hashes[condition] = measured["final_turn_hash"]
        measurements[condition] = measured
    if len(history_hashes) != 1:
        raise RuntimeError("History prefix changes across evidence conditions")
    effects = _behavior_effects(measurements, answer)
    return {
        "status": "completed",
        "answer_star": answer,
        "answer_star_reused": True,
        "branch": branch,
        "conditions": list(measurements),
        "new_forward_count": len(conditions),
        "reused_conditions": reused,
        "measurements": measurements,
        "history_prefix_hash": next(iter(history_hashes)),
        "final_turn_hashes": final_hashes,
        "hidden_captured": False,
        "verbal_sa_leakage": False,
        **effects,
    }


def measure_joint_branch(
    runtime: Stage3Runtime,
    plan: ProspectivePlan,
    row: Mapping[str, Any],
    branch: str,
    direction: Any,
    hidden_path: str | Path,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    deadline()
    messages, assistant_text, protocol = build_joint_history_messages(
        plan, row, branch
    )
    prepared = prepare_measurement(
        runtime.generator,
        messages,
        assistant_text=assistant_text,
        answer=str(row["answer_star"]),
    )
    analyzer = ProtocolAnalyzer(runtime.generator.tokenizer, protocol.spec)
    measured = runtime.measure(prepared, direction, analyzer=analyzer)
    hidden = np.asarray(measured.hidden, dtype=np.float32)
    if hidden.ndim != 1 or not np.isfinite(hidden).all():
        raise ValueError(f"Invalid L18 PANL hidden state: {hidden.shape}")
    if measured.applied_count != 1 or measured.injection_l2 != 0.0:
        raise RuntimeError("Joint readout hook/alpha=0 audit failed")
    destination = Path(hidden_path)
    atomic_save_npz(
        destination,
        hidden=hidden,
        layer=np.asarray(PRIMARY_LAYER, dtype=np.int64),
        position=np.asarray(PRIMARY_POSITION),
        branch=np.asarray(branch),
    )
    source = measured.source
    payload = {
        "status": "completed",
        "branch": branch,
        "answer_star": str(row["answer_star"]),
        "protocol": JOINT_PROTOCOL_NAME,
        "semantic_imageward_score": float(source["soft_image_score"]),
        "hard_label": str(source["hard_label"]),
        "class_logits": source["class_logits"],
        "class_probabilities": source["class_probabilities"],
        "frozen_prediction": float(direction.predict(hidden)),
        "frozen_coordinate": float(direction.coordinate(hidden)),
        "prefix_hash": prepared.prefix_hash,
        "history_prefix_hash": canonical_message_hash(messages[:2]),
        "final_joint_turn_hash": canonical_message_hash(messages[2:]),
        "panl_position": int(prepared.panl_position),
        "target_position": int(prepared.target_position),
        "input_token_count": int(prepared.inputs.input_ids.shape[1]),
        "hook_call_count": int(measured.hook_call_count),
        "hook_applied_count": int(measured.applied_count),
        "hook_exactly_once": bool(measured.applied_count == 1),
        "steering_applied": False,
        "injection_l2": float(measured.injection_l2),
        "hidden_file": str(destination),
        "new_forward_count": 1,
    }
    runtime.release_inputs(prepared)
    return payload


def run_formal_panel(
    runtime: Stage3Runtime,
    plan: ProspectivePlan,
    output_dir: str | Path,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    output = Path(output_dir)
    behavior_path = output / "behavior_results.jsonl"
    joint_path = output / "joint_results.jsonl"
    hidden_dir = output / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    behavior_done = {
        str(row["intervention_key"])
        for row in _latest_rows(behavior_path)
        if row.get("status") == "completed"
    }
    joint_done = {
        str(row["intervention_key"])
        for row in _latest_rows(joint_path)
        if row.get("status") == "completed"
    }
    directions = FrozenDirectionRepository(attribution_panel_root(output.parents[1]))
    for row in plan.rows:
        for branch in BRANCHES:
            deadline()
            key = f"stage09_behavior|{row['case_id']}|{branch}"
            if key in behavior_done:
                continue
            base = {
                "intervention_key": key,
                "experiment": "prospective_history_reliance_behavior",
                "panel_version": PANEL_VERSION,
                "case_id": row["case_id"],
                "item_id": row["item_id"],
                "fold": int(row["fold"]),
                "branch": branch,
            }
            result = _safe_record(
                base,
                lambda row=row, branch=branch: measure_behavior_branch(
                    runtime, plan, row, branch, deadline=deadline
                ),
            )
            append_record_atomic(behavior_path, result)
            if result.get("status") == "completed":
                behavior_done.add(key)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        direction = directions.get(int(row["fold"]))
        for branch in HISTORY_BRANCHES:
            deadline()
            key = f"stage09_joint|{row['case_id']}|{branch}"
            if key in joint_done:
                continue
            base = {
                "intervention_key": key,
                "experiment": "prospective_history_attribution_readout",
                "panel_version": PANEL_VERSION,
                "case_id": row["case_id"],
                "item_id": row["item_id"],
                "fold": int(row["fold"]),
                "branch": branch,
            }
            hidden_path = hidden_dir / f"{row['case_id']}__{branch}.npz"
            result = _safe_record(
                base,
                lambda row=row, branch=branch, direction=direction, hidden_path=hidden_path: measure_joint_branch(
                    runtime,
                    plan,
                    row,
                    branch,
                    direction,
                    hidden_path,
                    deadline=deadline,
                ),
            )
            append_record_atomic(joint_path, result)
            if result.get("status") == "completed":
                joint_done.add(key)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {
        "behavior_completed_branch_n": len(behavior_done),
        "behavior_expected_branch_n": PRIMARY_N * len(BRANCHES),
        "joint_completed_branch_n": len(joint_done),
        "joint_expected_branch_n": PRIMARY_N * len(HISTORY_BRANCHES),
    }


def fixed_fold_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Resample items within each immutable fold while preserving fold sizes."""

    values = [dict(row) for row in rows]
    if not values:
        return {"estimate": None, "ci95": [None, None], "iterations": iterations, "valid": 0}
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        by_fold[int(row["fold"])].append(row)
    estimate = float(statistic(values))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sample: list[dict[str, Any]] = []
        for fold in sorted(by_fold):
            group = by_fold[fold]
            indices = rng.integers(0, len(group), size=len(group))
            sample.extend(group[int(index)] for index in indices)
        try:
            value = float(statistic(sample))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if math.isfinite(value):
            samples.append(value)
    return {
        "estimate": estimate if math.isfinite(estimate) else None,
        "ci95": (
            [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
            if samples
            else [None, None]
        ),
        "iterations": iterations,
        "valid": len(samples),
        "resampling": "item-within-fixed-fold",
    }


def _spearman(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> float:
    x = np.asarray([float(row[left]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[right]) for row in rows], dtype=np.float64)
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return math.nan
    return float(spearmanr(x, y).statistic)


def association_summary(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, *, seed: int = SEED
) -> dict[str, Any]:
    values = [
        dict(row)
        for row in rows
        if row.get(left) is not None
        and row.get(right) is not None
        and math.isfinite(float(row[left]))
        and math.isfinite(float(row[right]))
    ]
    if len(values) < 3:
        return {"n": len(values), "pearson": None, "spearman": None, "spearman_fixed_fold_bootstrap": None}
    x = np.asarray([float(row[left]) for row in values], dtype=np.float64)
    y = np.asarray([float(row[right]) for row in values], dtype=np.float64)
    pearson = (
        float(pearsonr(x, y).statistic)
        if np.std(x) > 1e-12 and np.std(y) > 1e-12
        else None
    )
    spearman = _spearman(values, left, right)
    return {
        "n": len(values),
        "unique_items": len({str(row["item_id"]) for row in values}),
        "pearson": pearson,
        "spearman": spearman if math.isfinite(spearman) else None,
        "spearman_fixed_fold_bootstrap": fixed_fold_bootstrap(
            values,
            lambda sample: _spearman(sample, left, right),
            seed=seed,
        ),
        "sign_agreement": float(
            statistics.fmean(
                (float(row[left]) > 0) == (float(row[right]) > 0)
                for row in values
                if float(row[left]) != 0 and float(row[right]) != 0
            )
        ),
    }


def paired_effect_summary_fixed_fold(
    rows: Sequence[Mapping[str, Any]], key: str, *, seed: int = SEED
) -> dict[str, Any]:
    values = [dict(row) for row in rows if row.get(key) is not None]
    numbers = np.asarray([float(row[key]) for row in values], dtype=np.float64)
    bootstrap = fixed_fold_bootstrap(
        values,
        lambda sample: float(np.mean([float(row[key]) for row in sample])),
        seed=seed,
    )
    return {
        "n": len(values),
        "unique_items": len({str(row["item_id"]) for row in values}),
        "mean": float(numbers.mean()) if len(numbers) else None,
        "sd": float(numbers.std(ddof=1)) if len(numbers) > 1 else None,
        "ci95": bootstrap["ci95"],
        "direction_rate": float(np.mean(numbers > 0)) if len(numbers) else None,
        "fixed_fold_bootstrap": bootstrap,
    }


def prediction_summary(
    rows: Sequence[Mapping[str, Any]],
    target: str,
    prediction: str,
    *,
    seed: int,
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    y = np.asarray([float(row[target]) for row in values], dtype=np.float64)
    pred = np.asarray([float(row[prediction]) for row in values], dtype=np.float64)
    rho = association_summary(values, target, prediction, seed=seed)
    folds = []
    for fold in range(5):
        selected = [row for row in values if int(row["fold"]) == fold]
        folds.append(
            {
                "fold": fold,
                "n": len(selected),
                "r2": (
                    float(
                        r2_score(
                            [float(row[target]) for row in selected],
                            [float(row[prediction]) for row in selected],
                        )
                    )
                    if len(selected) >= 2
                    else None
                ),
                "spearman": _spearman(selected, target, prediction),
            }
        )
    r2 = float(r2_score(y, pred))
    r2_bootstrap = fixed_fold_bootstrap(
        values,
        lambda sample: float(
            r2_score(
                [float(row[target]) for row in sample],
                [float(row[prediction]) for row in sample],
            )
        ),
        seed=seed + 1,
    )
    return {
        "n": len(values),
        "r2": r2,
        "mse": float(mean_squared_error(y, pred)),
        "association": rho,
        "r2_fixed_fold_bootstrap": r2_bootstrap,
        "fold_metrics": folds,
        "positive_spearman_fold_count": sum(
            value["spearman"] is not None
            and math.isfinite(float(value["spearman"]))
            and float(value["spearman"]) > 0
            for value in folds
        ),
    }


def _ci_lower_positive(summary: Mapping[str, Any]) -> bool:
    bootstrap = summary.get("spearman_fixed_fold_bootstrap")
    if not isinstance(bootstrap, Mapping):
        return False
    low = bootstrap.get("ci95", [None, None])[0]
    return low is not None and float(low) > 0


def _latest_index(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    latest = {
        str(row["intervention_key"]): row for row in _latest_rows(path)
    }
    failed = sum(row.get("status") == "failed" for row in latest.values())
    return latest, failed


def analyze_panel(
    artifacts: SAFormationArtifacts,
    plan: ProspectivePlan,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    behavior, behavior_failed = _latest_index(output / "behavior_results.jsonl")
    joint, joint_failed = _latest_index(output / "joint_results.jsonl")
    complete: list[dict[str, Any]] = []
    raw_transforms = load_frozen_target_transforms(
        artifacts.experiment_dir / BRIDGE_DIR / "03_reliance_representation_devfit_confirm",
        "raw_choice_coupled",
    )
    for row in plan.rows:
        case_id = str(row["case_id"])
        behavior_branches: dict[str, dict[str, Any]] = {}
        joint_branches: dict[str, dict[str, Any]] = {
            "no_history": dict(row["no_history_common9"])
        }
        missing = False
        for branch in BRANCHES:
            key = f"stage09_behavior|{case_id}|{branch}"
            value = behavior.get(key)
            if value is None or value.get("status") != "completed":
                missing = True
                break
            behavior_branches[branch] = value
        if missing:
            continue
        for branch in HISTORY_BRANCHES:
            key = f"stage09_joint|{case_id}|{branch}"
            value = joint.get(key)
            if value is None or value.get("status") != "completed":
                missing = True
                break
            joint_branches[branch] = value
        if missing:
            continue

        source = plan.source_by_case[case_id]
        measurement = source["measurement"]
        transform = raw_transforms[int(row["fold"])]
        deletion = float(behavior_branches["no_history"]["behavior_delete_imageward"])
        m56 = float(
            behavior_branches["no_history"]["behavior_replace_imageward_d56_mean"]
        )
        replay = transform.apply(
            [measurement],
            [float(row["behavior_delete_imageward"])],
            [float(row["behavior_replace_imageward_d12_mean"])],
        )
        original_errors = {
            "deletion": abs(float(replay["deletion"][0]) - float(row["target_deletion_stage03"])),
            "replacement": abs(float(replay["replacement"][0]) - float(row["target_replacement_stage03"])),
            "shared": abs(float(replay["shared"][0]) - float(row["target_shared_stage03"])),
        }
        if max(original_errors.values()) > 1e-10:
            raise ValueError(f"Stage-03 transform replay failed for {case_id}: {original_errors}")
        fresh = transform.apply([measurement], [deletion], [m56])

        record: dict[str, Any] = {
            "case_id": case_id,
            "item_id": str(row["item_id"]),
            "fold": int(row["fold"]),
            "difficulty": row["difficulty"],
            "answer_star": row["answer_star"],
            "answer_star_side": row["answer_star_side"],
            "D": deletion,
            "M5": float(behavior_branches["no_history"]["behavior_replace_imageward_d5"]),
            "M6": float(behavior_branches["no_history"]["behavior_replace_imageward_d6"]),
            "M56": m56,
            "M12": float(row["behavior_replace_imageward_d12_mean"]),
            "M34": float(row["behavior_replace_imageward_d34_mean"]),
            "fresh_target_replacement_m56": float(fresh["replacement"][0]),
            "fresh_target_shared_d_m56": float(fresh["shared"][0]),
            "frozen_prediction_replacement": float(row["prediction_replacement"]),
            "frozen_prediction_shared": float(row["prediction_shared"]),
            "frozen_prediction_nuisance": float(row["prediction_nuisance"]),
            "stage03_transform_replay_max_abs_error": max(original_errors.values()),
        }
        record["paired_mse_improvement_over_frozen_nuisance"] = (
            (record["fresh_target_shared_d_m56"] - record["frozen_prediction_nuisance"]) ** 2
            - (record["fresh_target_shared_d_m56"] - record["frozen_prediction_shared"]) ** 2
        )
        for branch in BRANCHES:
            behavior_value = behavior_branches[branch]
            joint_value = joint_branches[branch]
            record[f"B_D_{branch}"] = float(behavior_value["behavior_delete_imageward"])
            record[f"B_M56_{branch}"] = float(
                behavior_value["behavior_replace_imageward_d56_mean"]
            )
            record[f"A_{branch}"] = float(joint_value["frozen_coordinate"])
            record[f"A_prediction_{branch}"] = float(joint_value["frozen_prediction"])
            record[f"V_{branch}"] = float(joint_value["semantic_imageward_score"])
        for relevance in ("relevant", "irrelevant"):
            for layer in ("B_D", "B_M56", "A", "V"):
                record[f"{relevance}_delta_{layer}"] = (
                    record[f"{layer}_{relevance}_image"]
                    - record[f"{layer}_{relevance}_text"]
                )
        for layer in ("B_D", "B_M56", "A", "V"):
            record[f"did_{layer}"] = (
                record[f"relevant_delta_{layer}"]
                - record[f"irrelevant_delta_{layer}"]
            )
        complete.append(record)

    write_jsonl_atomic(output / "analysis" / "results.jsonl", complete)
    technical = {
        "planned_n": PRIMARY_N,
        "complete_case_n": len(complete),
        "behavior_latest_record_n": len(behavior),
        "joint_latest_record_n": len(joint),
        "behavior_failed_latest_n": behavior_failed,
        "joint_failed_latest_n": joint_failed,
        "expected_behavior_branch_n": PRIMARY_N * len(BRANCHES),
        "expected_joint_branch_n": PRIMARY_N * len(HISTORY_BRANCHES),
    }
    technical["passed"] = bool(
        len(complete) >= 70 and behavior_failed == 0 and joint_failed == 0
    )

    validation: dict[str, Any] = {}
    if complete:
        validation["M12_vs_M56"] = association_summary(complete, "M12", "M56", seed=51)
        validation["M34_vs_M56"] = association_summary(complete, "M34", "M56", seed=61)
        validation["D_vs_M56"] = association_summary(complete, "D", "M56", seed=71)
        validation["donor56_icc_consistency"] = _icc_consistency(
            np.asarray([[row["M5"], row["M6"]] for row in complete], dtype=np.float64)
        )
        replacement_transport = prediction_summary(
            complete,
            "fresh_target_replacement_m56",
            "frozen_prediction_replacement",
            seed=81,
        )
        shared_transport = prediction_summary(
            complete,
            "fresh_target_shared_d_m56",
            "frozen_prediction_shared",
            seed=91,
        )
        mse_improvement = fixed_fold_bootstrap(
            complete,
            lambda sample: float(
                np.mean(
                    [
                        float(row["paired_mse_improvement_over_frozen_nuisance"])
                        for row in sample
                    ]
                )
            ),
            seed=101,
        )
        validation["replacement_transport"] = replacement_transport
        validation["shared_transport"] = shared_transport
        validation["paired_mse_improvement_over_frozen_nuisance"] = mse_improvement
    else:
        replacement_transport = {}
        shared_transport = {}
        mse_improvement = {"ci95": [None, None]}

    history_effects: dict[str, Any] = {}
    alignment: dict[str, Any] = {}
    if complete:
        for contrast in ("relevant_delta", "irrelevant_delta", "did"):
            history_effects[contrast] = {
                layer: paired_effect_summary_fixed_fold(
                    complete, f"{contrast}_{layer}", seed=200 + offset
                )
                for offset, layer in enumerate(("B_D", "B_M56", "A", "V"))
            }
            alignment[contrast] = {}
            for left, right in (
                ("B_D", "A"),
                ("B_M56", "A"),
                ("B_D", "V"),
                ("B_M56", "V"),
                ("A", "V"),
            ):
                alignment[contrast][f"{left}_vs_{right}"] = association_summary(
                    complete,
                    f"{contrast}_{left}",
                    f"{contrast}_{right}",
                    seed=300 + len(alignment[contrast]),
                )
        alignment["no_history"] = {
            "M56_vs_A": association_summary(complete, "M56", "A_no_history", seed=401),
            "M56_vs_V": association_summary(complete, "M56", "V_no_history", seed=411),
            "A_vs_V": association_summary(complete, "A_no_history", "V_no_history", seed=421),
        }

    replication_pass = bool(
        complete
        and _ci_lower_positive(validation["M12_vs_M56"])
        and _ci_lower_positive(validation["M34_vs_M56"])
        and validation["donor56_icc_consistency"] is not None
        and float(validation["donor56_icc_consistency"]) >= 0.60
    )
    replacement_transport_pass = bool(
        complete
        and _ci_lower_positive(replacement_transport.get("association", {}))
    )
    shared_transport_pass = bool(
        complete
        and float(shared_transport.get("r2", -math.inf)) > 0
        and _ci_lower_positive(shared_transport.get("association", {}))
        and int(shared_transport.get("positive_spearman_fold_count", 0)) >= 4
        and mse_improvement.get("ci95", [None, None])[0] is not None
        and float(mse_improvement["ci95"][0]) > 0
    )
    cross_method_pass = bool(
        complete and _ci_lower_positive(validation["D_vs_M56"])
    )
    gate = {
        "technical": technical["passed"],
        "replacement_replication": replication_pass,
        "replacement_transport": replacement_transport_pass,
        "shared_transport": shared_transport_pass,
        "cross_method_D_vs_M56": cross_method_pass,
    }
    gate["passed"] = all(gate.values())
    summary = {
        "title": "Stage 09 — Prospective Held-out History/Reliance Panel",
        "status": "completed" if technical["passed"] else "incomplete_or_technical_failure",
        "panel_version": PANEL_VERSION,
        "technical_gate": technical,
        "prospective_validation": validation,
        "prospective_gate": gate,
        "history_effects": history_effects,
        "three_layer_itemwise_alignment": alignment,
        "classification": (
            "future_reliance_candidate_passed_prospective_gate"
            if gate["passed"]
            else "future_reliance_candidate_did_not_pass_all_prospective_gates"
        ),
        "claim_limit": protocol_manifest()["claim_scope"],
        "causal_mediator_authorized": False,
        "retroactive_stage01_03_07_change": False,
    }
    from layer_metacognition.hidden_state_store import atomic_write_json, atomic_write_text

    atomic_write_json(output / "summary.json", summary)
    atomic_write_text(output / "summary.md", _summary_markdown(summary))
    return summary


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    technical = summary["technical_gate"]
    gate = summary["prospective_gate"]
    lines = [
        "# Stage 09 — Prospective Held-out History/Reliance Panel",
        "",
        f"- Complete cases: {technical['complete_case_n']}/{technical['planned_n']}.",
        f"- Technical gate: {gate['technical']}.",
        f"- Fresh replacement replication gate: {gate['replacement_replication']}.",
        f"- Replacement transport gate: {gate['replacement_transport']}.",
        f"- Shared transport gate: {gate['shared_transport']}.",
        f"- D↔M56 gate: {gate['cross_method_D_vs_M56']}.",
        f"- Overall prospective gate: **{gate['passed']}**.",
        "",
        "This panel is a future candidate test only. It does not revise Stages 01, 03, or 07 and does not authorize a causal mediator.",
        "",
        "The irrelevant History branch intentionally replays target A_T after a donor question to hold the assistant token fixed; this control can be semantically incongruent.",
        "",
    ]
    return "\n".join(lines)


def gpu_smoke(
    runtime: Stage3Runtime,
    plan: ProspectivePlan,
    output_dir: str | Path,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    """Exercise every branch family without writing formal result rows."""

    row = plan.rows[0]
    behavior: dict[str, Any] = {}
    for branch in HISTORY_BRANCHES:
        target = _target_case(plan, row)
        messages = build_behavior_messages(plan, row, branch, "full")
        deadline()
        behavior[branch] = direct_messages_fixed_answer_distribution(
            runtime,
            messages,
            answer_classes=target.answer_classes,
            fixed_answer=str(row["answer_star"]),
        )
    for condition in NO_HISTORY_NEW_CONDITIONS:
        target = _target_case(plan, row)
        messages = build_behavior_messages(plan, row, "no_history", condition)
        deadline()
        behavior[f"no_history/{condition}"] = direct_messages_fixed_answer_distribution(
            runtime,
            messages,
            answer_classes=target.answer_classes,
            fixed_answer=str(row["answer_star"]),
        )
    directions = FrozenDirectionRepository(
        attribution_panel_root(Path(output_dir).parents[1])
    )
    direction = directions.get(int(row["fold"]))
    joint: dict[str, Any] = {}
    smoke_hidden = Path(output_dir) / "gpu_smoke_hidden"
    smoke_hidden.mkdir(parents=True, exist_ok=True)
    for branch in HISTORY_BRANCHES:
        joint[branch] = measure_joint_branch(
            runtime,
            plan,
            row,
            branch,
            direction,
            smoke_hidden / f"{row['case_id']}__{branch}.npz",
            deadline=deadline,
        )
    if any(not value["hook_exactly_once"] for value in joint.values()):
        raise RuntimeError("Smoke joint hook did not apply exactly once")
    return {
        "status": "passed",
        "case_id": row["case_id"],
        "behavior_forward_count": len(behavior),
        "joint_forward_count": len(joint),
        "branches": list(BRANCHES),
        "answer_star": row["answer_star"],
        "all_joint_hooks_exactly_once": True,
        "formal_results_written": False,
    }

