"""History-conditioned fixed-answer source-sensitivity experiment.

This experiment nests evidence deletion/replacement inside the same Text-first
and Image-first histories that previously produced a large verbal-SA contrast.
It measures the probability of one fixed endpoint answer before generating any
answer token.  The answer-only protocol contains no Source Attribution request;
the joint protocol is retained as a report-conditioned comparison.
"""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from confidence_test.answer_metrics import normalize_answer
from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from confidence_test.prompt_utils import V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT
from layer_metacognition.hidden_state_store import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
)
from layer_metacognition.model_adapter import run_logits_forward

from .core import (
    canonical_message_hash,
    paired_effect_summary,
    write_experiment_summary,
)
from .mechanism import build_relevance_history_messages
from .runtime import Stage3Runtime, image_content
from .second_order import build_answer_history_messages
from .truth_audit import (
    BEHAVIOR_DIR,
    _association,
    _latest_rows,
    _prompt_with_text,
    _safe_record,
    _select_grounding_donors,
    canonical_leading_answer_tokens,
)


HISTORY_GROUNDING_DIR = "09_history_conditioned_fixed_answer_deletion"
PROTOCOLS = ("joint_report", "answer_only")
HISTORIES = ("text_first", "image_first")
CONDITIONS = ("full", "no_text", "no_image", "replace_text", "replace_image")


def _numeric_item_key(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return int(text), text
    except ValueError:
        return 10**12, text


def select_history_grounding_cohort(
    experiment_root: Path, truth_root: Path
) -> list[dict[str, Any]]:
    """Return relevant-History TF/IF pairs with the same generated endpoint."""
    mechanism_path = (
        experiment_root
        / "stage3_sa_mechanism"
        / "03_relevant_irrelevant_history"
        / "results.jsonl"
    )
    answer_only_path = (
        experiment_root
        / "stage3_sa_second_order"
        / "01_history_behavior_dissociation"
        / "results.jsonl"
    )
    baseline_path = truth_root / BEHAVIOR_DIR / "results.jsonl"
    oof_path = (
        experiment_root
        / "stage3_sa_formation"
        / "00_natural_state"
        / "oof_predictions.jsonl"
    )
    mechanism = {
        row["case_id"]: row
        for row in _latest_rows(mechanism_path)
        if row.get("status") == "completed"
    }
    answer_only = {
        row["case_id"]: row
        for row in _latest_rows(answer_only_path)
        if row.get("status") == "completed"
    }
    baseline = {
        row["case_id"]: row
        for row in load_jsonl(baseline_path)
        if row.get("status") == "completed"
    }
    oof = {row["case_id"]: row for row in load_jsonl(oof_path)}
    cohort: list[dict[str, Any]] = []
    for case_id, source in mechanism.items():
        text = source["branches"]["relevant_text"]
        image = source["branches"]["relevant_image"]
        if text["normalized_answer"] != image["normalized_answer"]:
            continue
        if case_id not in oof or case_id not in answer_only:
            raise ValueError(f"Missing OOF or answer-only reconstruction for {case_id}")
        answer_row = answer_only[case_id]
        fixed_answer = text["normalized_answer"]
        natural = {**oof[case_id], **baseline.get(case_id, {})}
        row = {
            **natural,
            "prior_answer": source["prior_answer"],
            "fixed_answer": fixed_answer,
            "final_answer": fixed_answer,
            "final_image": int(fixed_answer == natural["image_answer"]),
            "old_delta_sa_if_minus_tf": float(image["pass2_sa"] - text["pass2_sa"]),
            "old_sa_text_first": float(text["pass2_sa"]),
            "old_sa_image_first": float(image["pass2_sa"]),
            "expected_joint_hashes": {
                "text_first": text["messages_hash"],
                "image_first": image["messages_hash"],
            },
            "expected_answer_only_hashes": {
                "text_first": answer_row["branches"]["relevant_text"][
                    "messages_hash"
                ],
                "image_first": answer_row["branches"]["relevant_image"][
                    "messages_hash"
                ],
            },
            "answer_only_natural_endpoints": {
                "text_first": answer_row["branches"]["relevant_text"]["generated"][
                    "normalized_answer"
                ],
                "image_first": answer_row["branches"]["relevant_image"]["generated"][
                    "normalized_answer"
                ],
            },
        }
        cohort.append(row)
    cohort.sort(key=lambda row: (_numeric_item_key(row["item_id"]), row["case_id"]))
    if len(cohort) != 20 or len({row["item_id"] for row in cohort}) != 20:
        raise ValueError(
            f"Expected 20 unique endpoint-matched relevant-History cases, got {len(cohort)}"
        )
    return cohort


def answer_only_prompt_with_text(case: Any, text_clue: str) -> str:
    return V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT.format(
        question=case.question,
        text_clue=text_clue,
    )


def contains_sa_request(messages: Sequence[dict[str, Any]]) -> bool:
    return any(
        "Source Attribution" in str(part.get("text", ""))
        for message in messages
        for part in message.get("content", [])
        if isinstance(part, dict)
    )


def build_history_perturbation_messages(
    *,
    protocol: str,
    target_case: Any,
    target_condition: str,
    modality: str,
    prior_answer: str,
    text_clue: str,
    image_path: str,
) -> list[dict[str, Any]]:
    """Build one branch while keeping the historical two-turn prefix intact."""
    if protocol == "joint_report":
        messages = build_relevance_history_messages(
            target_case,
            target_condition,
            target_case,
            target_condition,
            modality,
            prior_answer,
        )
        prompt = _prompt_with_text(target_case, text_clue)
    elif protocol == "answer_only":
        messages = build_answer_history_messages(
            target_case,
            target_condition,
            target_case,
            target_condition,
            modality,
            prior_answer,
        )
        prompt = answer_only_prompt_with_text(target_case, text_clue)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    messages[-2] = {
        "role": "user",
        "content": image_content(image_path, prompt),
    }
    if protocol == "answer_only" and contains_sa_request(messages):
        raise ValueError("Answer-only behavior measurement leaks a verbal-SA request")
    return messages


def direct_messages_fixed_answer_distribution(
    runtime: Stage3Runtime,
    messages: list[dict[str, Any]],
    *,
    answer_classes: Sequence[str],
    fixed_answer: str,
) -> dict[str, Any]:
    """Read the restricted next-answer distribution for an arbitrary history."""
    rendered, inputs = runtime.generator.prepare_messages(
        messages, assistant_text=ASSISTANT_ANSWER_PREFILL
    )
    position = int(inputs.input_ids.shape[1]) - 1
    vocab_logits = run_logits_forward(
        runtime.model, inputs, [position], runtime.modules
    )[position]
    token_ids = canonical_leading_answer_tokens(
        runtime.generator.tokenizer, answer_classes
    )
    labels = [str(normalize_answer(label)) for label in answer_classes]
    class_logits = torch.stack([vocab_logits[token_ids[label]] for label in labels])
    probabilities = torch.softmax(class_logits.float(), dim=-1)
    logits_map = {
        label: float(class_logits[index].item()) for index, label in enumerate(labels)
    }
    probability_map = {
        label: float(probabilities[index].item()) for index, label in enumerate(labels)
    }
    fixed = str(normalize_answer(fixed_answer))
    if fixed not in probability_map:
        raise ValueError(f"Fixed answer {fixed!r} is not in the candidate set")
    fixed_probability = probability_map[fixed]
    if not math.isfinite(fixed_probability) or fixed_probability <= 0.0:
        raise ValueError(f"Non-positive fixed-answer probability: {fixed_probability}")
    ordered = sorted(logits_map.items(), key=lambda value: value[1], reverse=True)
    margin = float(ordered[0][1] - ordered[1][1])
    result = {
        "messages_hash": canonical_message_hash(messages),
        "history_prefix_hash": canonical_message_hash(messages[:2]),
        "rendered_hash": __import__("hashlib").sha256(rendered.encode()).hexdigest(),
        "fixed_answer": fixed,
        "fixed_answer_probability": fixed_probability,
        "fixed_answer_log_probability": math.log(fixed_probability),
        "predicted_answer": ordered[0][0],
        "unique_top1": bool(margin > 1e-6),
        "top1_top2_logit_margin": margin,
        "answer_class_logits": logits_map,
        "answer_class_probabilities": probability_map,
        "canonical_leading_token_ids": token_ids,
        "input_token_count": int(inputs.input_ids.shape[1]),
    }
    del inputs, vocab_logits, class_logits, probabilities
    return result


def history_difference_in_differences(
    text_first: dict[str, float], image_first: dict[str, float]
) -> dict[str, float]:
    """Compute imageward source sensitivity and its Image-first−Text-first DiD."""
    delete_tf = float(text_first["no_text"] - text_first["no_image"])
    delete_if = float(image_first["no_text"] - image_first["no_image"])
    replace_tf = float(text_first["replace_text"] - text_first["replace_image"])
    replace_if = float(image_first["replace_text"] - image_first["replace_image"])
    return {
        "delete_text_first": delete_tf,
        "delete_image_first": delete_if,
        "replace_text_first": replace_tf,
        "replace_image_first": replace_if,
        "delta_history_delete": delete_if - delete_tf,
        "delta_history_replace": replace_if - replace_tf,
    }


def _sign_agreement(left: Sequence[float], right: Sequence[float]) -> float:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if float(a) != 0.0 and float(b) != 0.0
    ]
    if not pairs:
        return float("nan")
    return statistics.fmean((a > 0) == (b > 0) for a, b in pairs)


def equivalence_to_zero(
    effect: dict[str, Any], natural_scale: float, standardized_bound: float = 0.2
) -> dict[str, Any]:
    band = float(standardized_bound * natural_scale)
    low, high = effect["ci95"]
    passed = bool(
        natural_scale > 0
        and low is not None
        and high is not None
        and float(low) >= -band
        and float(high) <= band
    )
    return {
        "standardized_bound": standardized_bound,
        "natural_scale": natural_scale,
        "raw_band": [-band, band],
        "passed": passed,
    }


def _protocol_summary(
    rows: Sequence[dict[str, Any]], protocol: str
) -> dict[str, Any]:
    contrasts = [
        {
            "item_id": row["item_id"],
            "delta_history_delete": row["protocols"][protocol][
                "delta_history_delete"
            ],
            "delta_history_replace": row["protocols"][protocol][
                "delta_history_replace"
            ],
            "delta_sa": row["old_delta_sa_if_minus_tf"],
        }
        for row in rows
    ]
    contexts = [
        {
            "item_id": row["item_id"],
            "history": history,
            "delete": row["protocols"][protocol][f"delete_{history}"],
            "replace": row["protocols"][protocol][f"replace_{history}"],
        }
        for row in rows
        for history in HISTORIES
    ]
    delete_effect = paired_effect_summary(contrasts, "delta_history_delete")
    replace_effect = paired_effect_summary(contrasts, "delta_history_replace")
    context_reliability = _association(contexts, "delete", "replace")
    delta_reliability = _association(
        contrasts, "delta_history_delete", "delta_history_replace"
    )
    context_sign = _sign_agreement(
        [row["delete"] for row in contexts], [row["replace"] for row in contexts]
    )
    delta_sign = _sign_agreement(
        [row["delta_history_delete"] for row in contrasts],
        [row["delta_history_replace"] for row in contrasts],
    )
    full_endpoint = [
        row
        for row in rows
        if all(
            row["protocols"][protocol]["histories"][history]["measurements"][
                "full"
            ]["predicted_answer"]
            == row["fixed_answer"]
            and row["protocols"][protocol]["histories"][history]["measurements"][
                "full"
            ]["unique_top1"]
            for history in HISTORIES
        )
    ]
    sensitivity = [
        row
        for row in full_endpoint
        if protocol != "answer_only"
        or all(
            row["answer_only_natural_endpoints"][history] == row["fixed_answer"]
            for history in HISTORIES
        )
    ]
    sensitivity_contrasts = [
        {
            "item_id": row["item_id"],
            "delta_history_delete": row["protocols"][protocol][
                "delta_history_delete"
            ],
            "delta_history_replace": row["protocols"][protocol][
                "delta_history_replace"
            ],
        }
        for row in sensitivity
    ]
    scales = {
        method: float(np.std([row[method] for row in contexts], ddof=1))
        for method in ("delete", "replace")
    }
    reliability_pass = bool(
        context_reliability["spearman_item_bootstrap"]["ci95"][0] is not None
        and context_reliability["spearman_item_bootstrap"]["ci95"][0] > 0
        and context_sign >= 0.60
    )
    sensitivity_effects = {
        "deletion": paired_effect_summary(
            sensitivity_contrasts, "delta_history_delete"
        ),
        "replacement": paired_effect_summary(
            sensitivity_contrasts, "delta_history_replace"
        ),
    }
    history_shift_pass = bool(
        len(rows) >= 20
        and all(effect["ci95"][0] is not None and effect["ci95"][0] > 0 for effect in (delete_effect, replace_effect))
        and len(sensitivity) >= 15
        and all(
            effect["ci95"][0] is not None and effect["ci95"][0] > 0
            for effect in sensitivity_effects.values()
        )
    )
    return {
        "n": len(rows),
        "full_unique_endpoint_n": len(full_endpoint),
        "natural_endpoint_sensitivity_n": len(sensitivity),
        "delta_history": {
            "deletion": delete_effect,
            "replacement": replace_effect,
        },
        "delta_history_sensitivity": sensitivity_effects,
        "behavior_target_reliability": {
            "context_delete_vs_replace": context_reliability,
            "context_sign_agreement": context_sign,
            "delta_delete_vs_replace": delta_reliability,
            "delta_sign_agreement": delta_sign,
            "passed": reliability_pass,
        },
        "delta_behavior_vs_delta_sa": {
            "deletion": _association(
                contrasts, "delta_history_delete", "delta_sa"
            ),
            "replacement": _association(
                contrasts, "delta_history_replace", "delta_sa"
            ),
        },
        "equivalence_to_zero": {
            "deletion": equivalence_to_zero(delete_effect, scales["delete"]),
            "replacement": equivalence_to_zero(replace_effect, scales["replace"]),
        },
        "gates": {
            "behavior_target_reliable": reliability_pass,
            "positive_history_shift": history_shift_pass,
            "grounded_history_shift": bool(reliability_pass and history_shift_pass),
        },
    }


def summarize_history_grounding(
    rows: Sequence[dict[str, Any]], failed: int
) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    protocol = {name: _protocol_summary(completed, name) for name in PROTOCOLS}
    sa_effect = paired_effect_summary(completed, "old_delta_sa_if_minus_tf")
    answer_only = protocol["answer_only"]
    joint = protocol["joint_report"]
    protocol_interaction_rows = [
        {
            "item_id": row["item_id"],
            "deletion": row["protocols"]["joint_report"]["delta_history_delete"]
            - row["protocols"]["answer_only"]["delta_history_delete"],
            "replacement": row["protocols"]["joint_report"][
                "delta_history_replace"
            ]
            - row["protocols"]["answer_only"]["delta_history_replace"],
        }
        for row in completed
    ]
    protocol_interaction = {
        method: paired_effect_summary(protocol_interaction_rows, method)
        for method in ("deletion", "replacement")
    }
    sa_positive = bool(sa_effect["ci95"][0] is not None and sa_effect["ci95"][0] > 0)
    answer_only_positive_methods = [
        method
        for method, effect in answer_only["delta_history"].items()
        if effect["ci95"][0] is not None and effect["ci95"][0] > 0
    ]
    answer_only_sensitivity_positive_methods = [
        method
        for method, effect in answer_only["delta_history_sensitivity"].items()
        if effect["ci95"][0] is not None and effect["ci95"][0] > 0
    ]
    partial_answer_only_shift = bool(
        answer_only["gates"]["behavior_target_reliable"]
        and set(answer_only_positive_methods)
        & set(answer_only_sensitivity_positive_methods)
    )
    behavior_equivalent = all(
        value["passed"] for value in answer_only["equivalence_to_zero"].values()
    )
    if answer_only["gates"]["grounded_history_shift"]:
        classification = "History changes a reliable answer-only behavioral source-use target"
    elif partial_answer_only_shift:
        classification = "History produces a partial answer-only behavioral source-sensitivity shift, but deletion and replacement do not both pass"
    elif joint["gates"]["grounded_history_shift"]:
        classification = "Only the joint protocol passes the strict cross-method History-shift gate; this does not by itself establish a report-request interaction"
    elif sa_positive and behavior_equivalent:
        classification = "History changes verbal SA while answer-only behavioral source sensitivity is equivalent to zero"
    elif sa_positive and all(
        effect["ci95"][1] is not None and effect["ci95"][1] < 0
        for effect in answer_only["delta_history"].values()
    ):
        classification = "History verbal-SA and answer-only source-sensitivity effects have opposite signs"
    else:
        classification = "Behavioral source-use measurement remains inconclusive for the History effect"
    return {
        "title": "Truth Audit 9 — History-conditioned Fixed-answer Source Sensitivity",
        "status": "completed",
        "n": len(completed),
        "failed": failed,
        "protocols": protocol,
        "joint_minus_answer_only_history_effect": protocol_interaction,
        "old_verbal_sa_history_effect": sa_effect,
        "technical_checks": {
            "all_joint_full_hashes_match": all(
                row["technical_checks"]["joint_full_hashes_match"] for row in completed
            ),
            "all_answer_only_full_hashes_match": all(
                row["technical_checks"]["answer_only_full_hashes_match"]
                for row in completed
            ),
            "answer_only_sa_leakage_n": sum(
                row["technical_checks"]["answer_only_sa_leakage"]
                for row in completed
            ),
            "nonpositive_probability_n": sum(
                row["technical_checks"]["nonpositive_probability"]
                for row in completed
            ),
        },
        "grounded_gate_passed": bool(
            answer_only["gates"]["grounded_history_shift"]
        ),
        "partial_answer_only_history_shift": partial_answer_only_shift,
        "answer_only_positive_methods": answer_only_positive_methods,
        "answer_only_sensitivity_positive_methods": answer_only_sensitivity_positive_methods,
        "classification": classification,
        "claim_limit": "A null CI is not interpreted as no source-use change unless the predeclared equivalence band also passes; deletion/replacement reliability is a separate gate.",
    }


def write_history_grounding_report(directory: Path, summary: dict[str, Any]) -> None:
    def effect(value: dict[str, Any]) -> str:
        low, high = value["ci95"]
        return f"{value['mean']:+.3f}, 95% CI [{low:+.3f}, {high:+.3f}], n={value['n']}"

    def rho(value: dict[str, Any]) -> str:
        low, high = value["spearman_item_bootstrap"]["ci95"]
        return f"rho={value['spearman']:.3f}, 95% CI [{low:.3f}, {high:.3f}], n={value['n']}"

    lines = [
        "# History-conditioned Fixed-answer Source Sensitivity",
        "",
        f"- Classification: **{summary['classification']}**.",
        f"- Existing endpoint-matched verbal-SA IF−TF effect: {effect(summary['old_verbal_sa_history_effect'])}.",
        f"- Joint−answer-only History effect: deletion {effect(summary['joint_minus_answer_only_history_effect']['deletion'])}; replacement {effect(summary['joint_minus_answer_only_history_effect']['replacement'])}.",
        f"- Strict answer-only grounded gate={summary['grounded_gate_passed']}; partial answer-only shift={summary['partial_answer_only_history_shift']} ({summary['answer_only_positive_methods']}).",
        "",
    ]
    for name in PROTOCOLS:
        value = summary["protocols"][name]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Deletion ΔHistory: {effect(value['delta_history']['deletion'])}.",
                f"- Replacement ΔHistory: {effect(value['delta_history']['replacement'])}.",
                f"- Context deletion↔replacement reliability: {rho(value['behavior_target_reliability']['context_delete_vs_replace'])}; sign agreement={value['behavior_target_reliability']['context_sign_agreement']:.3f}.",
                f"- ΔHistory deletion↔verbal-SA: {rho(value['delta_behavior_vs_delta_sa']['deletion'])}.",
                f"- ΔHistory replacement↔verbal-SA: {rho(value['delta_behavior_vs_delta_sa']['replacement'])}.",
                f"- Full unique endpoint n={value['full_unique_endpoint_n']}; natural-endpoint sensitivity n={value['natural_endpoint_sensitivity_n']}.",
                f"- Gates: {value['gates']}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation rule",
            "",
            f"- {summary['claim_limit']}",
            f"- Technical checks: {summary['technical_checks']}.",
            "",
        ]
    )
    atomic_write_text(directory / "FINAL_ANALYSIS.md", "\n".join(lines))


def run_history_grounding(
    runtime: Stage3Runtime,
    experiment_root: Path,
    truth_root: Path,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = truth_root / HISTORY_GROUNDING_DIR
    directory.mkdir(parents=True, exist_ok=True)
    cohort = select_history_grounding_cohort(experiment_root, truth_root)
    behavior_candidates = [
        row
        for row in load_jsonl(truth_root / BEHAVIOR_DIR / "results.jsonl")
        if row.get("status") == "completed"
    ]
    donors = _select_grounding_donors(runtime, cohort, behavior_candidates)
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "n": len(cohort),
            "case_ids": [row["case_id"] for row in cohort],
            "protocols": list(PROTOCOLS),
            "histories": list(HISTORIES),
            "conditions": list(CONDITIONS),
            "fixed_answer": "common generated endpoint from existing relevant Text-first/Image-first joint branches",
            "answer_scoring": "canonical leading-space one-token restricted next-token distribution; answer token is not teacher-forced",
            "history_rule": "only final evidence is perturbed; historical user/assistant turns remain byte-identical within each protocol/history",
            "donors": {
                row["case_id"]: donors[row["case_id"]]["case_id"] for row in cohort
            },
            "primary_behavior_protocol": "answer_only (contains no SA request)",
            "joint_protocol_role": "report-conditioned comparison",
        },
    )
    result_path = directory / "results.jsonl"
    existing = {
        row["intervention_key"]
        for row in _latest_rows(result_path)
        if row.get("status") == "completed"
    }
    for row in cohort:
        deadline()
        key = f"history_grounding|{row['case_id']}"
        if key in existing:
            continue
        donor = donors[row["case_id"]]
        base = {
            "intervention_key": key,
            "experiment": "history_conditioned_fixed_answer_source_sensitivity",
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "prior_index": row["prior_index"],
            "condition": row["condition"],
            "difficulty": row["difficulty"],
            "fold": int(row["fold"]),
            "prior_answer": row["prior_answer"],
            "fixed_answer": row["fixed_answer"],
            "old_sa_text_first": row["old_sa_text_first"],
            "old_sa_image_first": row["old_sa_image_first"],
            "old_delta_sa_if_minus_tf": row["old_delta_sa_if_minus_tf"],
            "answer_only_natural_endpoints": row["answer_only_natural_endpoints"],
            "donor_case_id": donor["case_id"],
            "donor_item_id": donor["item_id"],
        }

        def execute() -> dict[str, Any]:
            case = runtime.case(row["item_id"], row["prior_index"])
            donor_case = runtime.case(donor["item_id"], donor["prior_index"])
            original_image = str(case.conditions[row["condition"]].resolved_image_path)
            conditions = {
                "full": (case.text_clue, original_image),
                "no_text": ("[No text clue available.]", original_image),
                "no_image": (case.text_clue, str(case.conditions["null"].resolved_image_path)),
                "replace_text": (donor_case.text_clue, original_image),
                "replace_image": (case.text_clue, str(case.conditions["irr"].resolved_image_path)),
            }
            protocol_results: dict[str, Any] = {}
            technical = {
                "joint_full_hashes_match": True,
                "answer_only_full_hashes_match": True,
                "answer_only_sa_leakage": 0,
                "nonpositive_probability": 0,
            }
            for protocol in PROTOCOLS:
                histories: dict[str, Any] = {}
                for history in HISTORIES:
                    modality = "text" if history == "text_first" else "image"
                    measurements: dict[str, Any] = {}
                    for condition_name, (text_clue, image_path) in conditions.items():
                        messages = build_history_perturbation_messages(
                            protocol=protocol,
                            target_case=case,
                            target_condition=row["condition"],
                            modality=modality,
                            prior_answer=row["prior_answer"],
                            text_clue=text_clue,
                            image_path=image_path,
                        )
                        if protocol == "answer_only" and contains_sa_request(messages):
                            technical["answer_only_sa_leakage"] += 1
                        measurements[condition_name] = direct_messages_fixed_answer_distribution(
                            runtime,
                            messages,
                            answer_classes=case.answer_classes,
                            fixed_answer=row["fixed_answer"],
                        )
                    if len(
                        {
                            value["history_prefix_hash"]
                            for value in measurements.values()
                        }
                    ) != 1:
                        raise ValueError(
                            f"History prefix changed across evidence conditions: {protocol}/{history}"
                        )
                    expected = (
                        row["expected_joint_hashes"][history]
                        if protocol == "joint_report"
                        else row["expected_answer_only_hashes"][history]
                    )
                    full_matches = measurements["full"]["messages_hash"] == expected
                    if protocol == "joint_report":
                        technical["joint_full_hashes_match"] &= full_matches
                    else:
                        technical["answer_only_full_hashes_match"] &= full_matches
                    logp = {
                        name: value["fixed_answer_log_probability"]
                        for name, value in measurements.items()
                    }
                    histories[history] = {
                        "measurements": measurements,
                        "expected_full_messages_hash": expected,
                        "full_messages_hash_matches": full_matches,
                        "delete_imageward": logp["no_text"] - logp["no_image"],
                        "replace_imageward": logp["replace_text"] - logp["replace_image"],
                        "remove_image_drop_logp": logp["full"] - logp["no_image"],
                        "remove_text_drop_logp": logp["full"] - logp["no_text"],
                        "replace_image_drop_logp": logp["full"] - logp["replace_image"],
                        "replace_text_drop_logp": logp["full"] - logp["replace_text"],
                    }
                did = history_difference_in_differences(
                    {
                        name: histories["text_first"]["measurements"][name][
                            "fixed_answer_log_probability"
                        ]
                        for name in CONDITIONS
                    },
                    {
                        name: histories["image_first"]["measurements"][name][
                            "fixed_answer_log_probability"
                        ]
                        for name in CONDITIONS
                    },
                )
                protocol_results[protocol] = {
                    "histories": histories,
                    **did,
                }
            if not technical["joint_full_hashes_match"]:
                raise ValueError("Joint full reconstruction hash mismatch")
            if not technical["answer_only_full_hashes_match"]:
                raise ValueError("Answer-only full reconstruction hash mismatch")
            if technical["answer_only_sa_leakage"]:
                raise ValueError("Answer-only protocol leaked Source Attribution text")
            return {
                **base,
                "status": "completed",
                "protocols": protocol_results,
                "technical_checks": technical,
            }

        append_jsonl(result_path, _safe_record(base, execute))
    latest = _latest_rows(result_path)
    completed = [row for row in latest if row.get("status") == "completed"]
    summary = summarize_history_grounding(completed, len(latest) - len(completed))
    write_experiment_summary(directory, summary)
    write_history_grounding_report(directory, summary)
    return summary
