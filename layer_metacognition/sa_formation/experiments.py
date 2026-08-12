"""Gate-driven experiments for the Stage 3 SA formation pilot."""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from confidence_test.source_attribution_schema import ASSISTANT_SOURCE_ATTRIBUTION_PREFILL
from layer_metacognition.hidden_state_store import append_jsonl, atomic_write_json, load_jsonl
from layer_metacognition.steering.decision_side_steering import BaselineHiddenStateRepository

from .core import (
    EXPERIMENT_DIR_NAMES,
    FoldDirection,
    GateDecision,
    SAFormationArtifacts,
    SAOOFDirectionRepository,
    assert_endpoint_evidence_equal,
    assert_policy_no_verbal_sa,
    canonical_message_hash,
    coordinate_delta,
    decide_gate,
    item_cluster_bootstrap,
    natural_projection_summary,
    orthogonal_equal_norm_control,
    paired_effect_summary,
    read_json,
    transplant_delta,
    write_experiment_summary,
    write_jsonl_atomic,
)
from .runtime import (
    SOURCE_CHOICE_PROMPT,
    Stage3Runtime,
    assistant_message,
    build_history_messages,
    full_prompt,
    image_content,
    prepare_measurement,
    prepare_policy_measurement,
    right_pad_measurement_inputs,
    source_prefix_from_generation,
    text_content,
)


def _completed_by_key(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["intervention_key"]): row
        for row in load_jsonl(path, repair_trailing=True)
        if row.get("status") == "completed" and row.get("intervention_key")
    }


def _safe_call(record: dict[str, Any], function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = function()
        result.setdefault("status", "completed")
    except Exception as exc:
        result = dict(record)
        result.update(
            {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
    result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return result


def _round_robin_sample(
    rows: Sequence[dict[str, Any]],
    n: int,
    strata: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda x: (str(x["item_id"]), int(x.get("prior_index", 0)), str(x.get("condition", "")))):
        buckets.setdefault(strata(row), []).append(row)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets, key=str)
    while len(selected) < n and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < n:
                selected.append(buckets[key].pop(0))
    return selected


def _single_turn_messages(case: Any, condition: str, assistant_text: str) -> list[dict[str, Any]]:
    image_path = str(case.conditions[condition].resolved_image_path)
    return [
        {"role": "user", "content": image_content(image_path, full_prompt(case))},
        assistant_message(assistant_text),
    ]


def _source_soft(state: dict[str, Any]) -> float:
    return float(state["source"]["soft_image_score"])


def select_transplant_pairs(
    oof: Sequence[dict[str, Any]],
    directions: SAOOFDirectionRepository,
    *,
    pair_count: int = 20,
) -> list[dict[str, Any]]:
    candidates = [row for row in oof if row.get("final_answer")]
    pairs: list[dict[str, Any]] = []
    used_case_ids: set[str] = set()
    for fold in sorted({int(row["fold"]) for row in candidates}):
        fold_rows = sorted(
            [row for row in candidates if int(row["fold"]) == fold],
            key=lambda row: float(row["z_sa"]),
        )
        low = fold_rows[: max(20, len(fold_rows) // 4)]
        high = list(reversed(fold_rows[-max(20, len(fold_rows) // 4) :]))
        scored: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
        sigma = directions.get(fold).sigma_z
        for left in low:
            for right in high:
                if left["item_id"] == right["item_id"]:
                    continue
                gap = float(right["z_sa"] - left["z_sa"])
                if gap <= 0:
                    continue
                score = (
                    0 if left["condition"] == right["condition"] else 1,
                    0 if left["final_answer"] == right["final_answer"] else 1,
                    0 if gap >= sigma else 1,
                    -gap / sigma,
                    str(left["case_id"]),
                    str(right["case_id"]),
                )
                scored.append((score, left, right))
        target_for_fold = pair_count // 5 + (1 if fold < pair_count % 5 else 0)
        selected_here = 0
        for _score, left, right in sorted(scored, key=lambda value: value[0]):
            if left["case_id"] in used_case_ids or right["case_id"] in used_case_ids:
                continue
            pairs.append(
                {
                    "pair_id": f"pair_{len(pairs):02d}",
                    "fold": fold,
                    "low_case_id": left["case_id"],
                    "high_case_id": right["case_id"],
                    "low_item_id": left["item_id"],
                    "high_item_id": right["item_id"],
                    "low_z": left["z_sa"],
                    "high_z": right["z_sa"],
                    "gap": float(right["z_sa"] - left["z_sa"]),
                    "gap_sigma": float((right["z_sa"] - left["z_sa"]) / directions.get(fold).sigma_z),
                    "condition_matched": left["condition"] == right["condition"],
                    "answer_matched": left["final_answer"] == right["final_answer"],
                }
            )
            used_case_ids.update((str(left["case_id"]), str(right["case_id"])))
            selected_here += 1
            if selected_here >= target_for_fold:
                break
    if len(pairs) < pair_count:
        raise RuntimeError(f"Could select only {len(pairs)}/{pair_count} same-fold transplant pairs")
    return pairs[:pair_count]


def run_experiment_0(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    output_root: Path,
    oof: Sequence[dict[str, Any]],
    directions: SAOOFDirectionRepository,
    *,
    deadline: Callable[[], None],
) -> tuple[dict[str, Any], GateDecision]:
    directory = output_root / EXPERIMENT_DIR_NAMES[0]
    directory.mkdir(parents=True, exist_ok=True)
    natural = natural_projection_summary(oof)
    pairs = select_transplant_pairs(oof, directions)
    atomic_write_json(directory / "cohort_manifest.json", {"base_pair_count": len(pairs), "directed_contrast_count": len(pairs) * 2, "pairs": pairs})
    by_id = {row["case_id"]: row for row in oof}
    hidden_repo = BaselineHiddenStateRepository(artifacts.experiment_dir)
    results_path = directory / "results.jsonl"
    completed = _completed_by_key(results_path)
    for pair in pairs:
        for recipient_label, donor_label in (("low", "high"), ("high", "low")):
            deadline()
            recipient = by_id[pair[f"{recipient_label}_case_id"]]
            donor = by_id[pair[f"{donor_label}_case_id"]]
            key = f"transplant|{recipient['case_id']}|from|{donor['case_id']}"
            if key in completed:
                continue
            base = {
                "intervention_key": key,
                "experiment": "coordinate_transplant",
                "pair_id": pair["pair_id"],
                "case_id": recipient["case_id"],
                "item_id": recipient["item_id"],
                "donor_case_id": donor["case_id"],
                "donor_item_id": donor["item_id"],
                "fold": recipient["fold"],
            }

            def execute() -> dict[str, Any]:
                direction = directions.get(int(recipient["fold"]))
                if int(donor["fold"]) != direction.fold:
                    raise ValueError("Transplant pair crosses held-out folds")
                case = runtime.case(recipient["item_id"], recipient["prior_index"])
                answer = str(recipient["final_answer"])
                assistant_text = f"**Answer**: {answer}\n{ASSISTANT_SOURCE_ATTRIBUTION_PREFILL}"
                messages = _single_turn_messages(case, recipient["condition"], assistant_text)
                prepared = prepare_measurement(runtime.generator, messages, assistant_text=assistant_text, answer=answer)
                sham = runtime.measure(prepared, direction)
                donor_hidden = hidden_repo.get(donor, 18, "panl")
                vector = coordinate_delta(sham.hidden, direction.d_unit, float(donor_hidden @ direction.d_unit))
                transplant = runtime.measure(prepared, direction, steering_vector=vector)
                orthogonal = orthogonal_equal_norm_control(
                    direction.d_unit,
                    float(np.linalg.norm(vector)),
                    seed_material=key,
                )
                control = runtime.measure(prepared, direction, steering_vector=orthogonal)
                sign = 1.0 if float(donor["z_sa"]) > float(recipient["z_sa"]) else -1.0
                delta_sa = _source_soft(transplant.to_dict()) - _source_soft(sham.to_dict())
                orth_delta_sa = _source_soft(control.to_dict()) - _source_soft(sham.to_dict())
                runtime.release_inputs(prepared)
                return {
                    **base,
                    "status": "completed",
                    "recipient_z_oof": recipient["z_sa"],
                    "donor_z_oof": donor["z_sa"],
                    "delta_z": float(donor["z_sa"] - recipient["z_sa"]),
                    "delta_z_sigma": float((donor["z_sa"] - recipient["z_sa"]) / direction.sigma_z),
                    "sham": sham.to_dict(),
                    "transplant": transplant.to_dict(),
                    "orthogonal_control": control.to_dict(),
                    "delta_sa": delta_sa,
                    "aligned_delta_sa": sign * delta_sa,
                    "orthogonal_aligned_delta_sa": sign * orth_delta_sa,
                    "coordinate_minus_control": sign * (delta_sa - orth_delta_sa),
                    "directional": sign * delta_sa > 0,
                    "equal_hidden_l2": abs(transplant.injection_l2 - control.injection_l2) <= max(0.125, transplant.injection_l2 * 0.05),
                }

            record = _safe_call(base, execute)
            append_jsonl(results_path, record)
    records = [row for row in load_jsonl(results_path) if row.get("status") == "completed"]
    coordinate = paired_effect_summary(records, "aligned_delta_sa")
    orthogonal = paired_effect_summary(records, "orthogonal_aligned_delta_sa")
    difference = paired_effect_summary(records, "coordinate_minus_control")
    effective = bool(
        coordinate["n"] >= 30
        and coordinate["ci95"][0] is not None
        and coordinate["ci95"][0] > 0
        and coordinate["direction_rate"] >= 0.60
        and difference["ci95"][0] is not None
        and difference["ci95"][0] > 0
    )
    transplant_summary = {
        "n": coordinate["n"],
        "coordinate": coordinate,
        "orthogonal_control": orthogonal,
        "coordinate_minus_control": difference,
        "coordinate_effective": effective,
        "criterion": "n>=30, aligned CI lower>0, direction rate>=.60, and coordinate-minus-orthogonal CI lower>0",
        "failed": sum(row.get("status") != "completed" for row in load_jsonl(results_path)),
    }
    gate = decide_gate(bool(natural["natural_effective"]), transplant_summary)
    summary = {
        "title": "Experiment 0 — Natural projection and coordinate transplant",
        "status": "completed",
        "n": natural["n"],
        "natural_projection": natural,
        "coordinate_transplant": transplant_summary,
        "gate": gate.to_dict(),
    }
    write_experiment_summary(directory, summary)
    return summary, gate


def _load_unimodal(artifacts: SAFormationArtifacts) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    text = {(str(row["item_id"]), int(row["prior_index"])): row for row in load_jsonl(artifacts.text_labels)}
    image = {(str(row["item_id"]), str(row["condition"])): row for row in load_jsonl(artifacts.image_labels)}
    return text, image


def _logit(probability: float) -> float:
    p = min(1 - 1e-8, max(1e-8, float(probability)))
    return math.log(p / (1 - p))


def _prior_strength(case: Any) -> float:
    value = str(case.prior_bin or "")
    try:
        left, right = value.split("-", 1)
        return abs((float(left) + float(right)) / 2 - 0.5)
    except Exception:
        return 0.0


def _ols(rows: Sequence[dict[str, Any]], outcome: str, predictor: str) -> dict[str, Any]:
    valid = [row for row in rows if row.get(outcome) is not None and row.get(predictor) is not None]
    if len(valid) < 6:
        return {"n": len(valid), "coefficients": None}
    x = np.asarray(
        [[1.0, float(row[predictor]), float(row.get("final_image", 0)), float(row.get("difficulty_hard", 0)), float(row.get("prior_strength", 0))] for row in valid]
    )
    y = np.asarray([float(row[outcome]) for row in valid])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    predicted = x @ coefficients
    return {
        "n": len(valid),
        "terms": ["intercept", predictor, "final_image", "difficulty_hard", "prior_strength"],
        "coefficients": [float(value) for value in coefficients],
        "r2": float(1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def run_experiment_1(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    output_root: Path,
    oof: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    directory = output_root / EXPERIMENT_DIR_NAMES[1]
    directory.mkdir(parents=True, exist_ok=True)
    text_map, image_map = _load_unimodal(artifacts)
    conflicts = [row for row in oof if row["text_answer"] != row["image_answer"] and row["text_answer"] and row["image_answer"]]
    enriched: list[dict[str, Any]] = []
    for row in conflicts:
        case = runtime.case(row["item_id"], row["prior_index"])
        final_side = (
            "image" if row["final_answer"] == row["image_answer"]
            else "text" if row["final_answer"] == row["text_answer"]
            else "other"
        )
        enriched.append(
            {
                **row,
                "selection_final_side": final_side,
                "selection_prior_bin": case.prior_bin,
            }
        )
    selected = _round_robin_sample(
        enriched,
        80,
        lambda row: (
            row["selection_final_side"],
            row["difficulty"],
            row["selection_prior_bin"],
            row["fold"],
        ),
    )
    records: list[dict[str, Any]] = []
    for row in selected:
        text = text_map[(row["item_id"], row["prior_index"])]["generation_result"]
        image = image_map[(row["item_id"], row["condition"])]["generation_result"]
        case = runtime.case(row["item_id"], row["prior_index"])
        p_text = float(text["answer_prob"])
        p_image = float(image["answer_prob"])
        final_side = 1 if row["final_answer"] == row["image_answer"] else 0 if row["final_answer"] == row["text_answer"] else -1
        records.append(
            {
                "intervention_key": f"evidence|{row['case_id']}",
                "experiment": "evidence_balance",
                "status": "completed",
                "case_id": row["case_id"],
                "item_id": row["item_id"],
                "prior_index": row["prior_index"],
                "fold": row["fold"],
                "condition": row["condition"],
                "difficulty": row["difficulty"],
                "final_answer_side": "image" if final_side == 1 else "text" if final_side == 0 else "other",
                "final_image": final_side == 1,
                "difficulty_hard": row["difficulty"] == "hard",
                "prior_strength": _prior_strength(case),
                "prior_bin": case.prior_bin,
                "e_text": p_text,
                "e_image": p_image,
                "e_balance": _logit(p_image) - _logit(p_text),
                "z_sa": row["z_sa"],
                "sa": row["sa"],
                "elapsed_seconds": 0.0,
            }
        )
    # Secondary easy/hard within-item pairs with the same final endpoint.
    secondary: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row in oof:
        prefix = row["condition"].rsplit("_", 1)[0]
        grouped.setdefault((row["item_id"], row["prior_index"], prefix), {})[row["difficulty"]] = row
    for key, values in sorted(grouped.items()):
        if "easy" not in values or "hard" not in values:
            continue
        easy, hard = values["easy"], values["hard"]
        if easy["final_answer"] != hard["final_answer"]:
            continue
        secondary.append(
            {
                "pair_id": "|".join(map(str, key)),
                "item_id": easy["item_id"],
                "easy_case_id": easy["case_id"],
                "hard_case_id": hard["case_id"],
                "delta_z_hard_minus_easy": float(hard["z_sa"] - easy["z_sa"]),
                "delta_sa_hard_minus_easy": float(hard["sa"] - easy["sa"]),
            }
        )
        if len(secondary) >= 40:
            break
    write_jsonl_atomic(directory / "results.jsonl", records)
    balance_counts: dict[str, int] = {}
    for row in records:
        cell = "|".join(
            [
                str(row["final_answer_side"]),
                str(row["difficulty"]),
                str(row["prior_bin"]),
                f"fold_{row['fold']}",
            ]
        )
        balance_counts[cell] = balance_counts.get(cell, 0) + 1
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "primary_n": len(records),
            "primary_unique_items": len({row["item_id"] for row in records}),
            "balance_dimensions": ["final_answer_side", "difficulty", "prior_bin", "fold"],
            "balance_counts": balance_counts,
            "secondary_pair_n": len(secondary),
            "case_ids": [row["case_id"] for row in records],
            "secondary_pairs": secondary,
        },
    )
    z_spearman_bootstrap = item_cluster_bootstrap(
        records,
        lambda sample: spearmanr(
            [r["e_balance"] for r in sample], [r["z_sa"] for r in sample]
        ).statistic,
    )
    sa_spearman_bootstrap = item_cluster_bootstrap(
        records,
        lambda sample: spearmanr(
            [r["e_balance"] for r in sample], [r["sa"] for r in sample]
        ).statistic,
    )
    summary = {
        "title": "Experiment 1 — Evidence Balance",
        "status": "completed",
        "n": len(records),
        "gpu_forwards": 0,
        "definition": "E_balance=logit(unimodal image chosen-answer probability)-logit(unimodal text chosen-answer probability)",
        "e_balance_z": {"pearson": float(pearsonr([r["e_balance"] for r in records], [r["z_sa"] for r in records]).statistic), "spearman": float(spearmanr([r["e_balance"] for r in records], [r["z_sa"] for r in records]).statistic), "spearman_item_bootstrap": z_spearman_bootstrap},
        "e_balance_sa": {"pearson": float(pearsonr([r["e_balance"] for r in records], [r["sa"] for r in records]).statistic), "spearman": float(spearmanr([r["e_balance"] for r in records], [r["sa"] for r in records]).statistic), "spearman_item_bootstrap": sa_spearman_bootstrap},
        "adjusted_z": _ols(records, "z_sa", "e_balance"),
        "adjusted_sa": _ols(records, "sa", "e_balance"),
        "secondary_easy_hard": {"n": len(secondary), "z": paired_effect_summary(secondary, "delta_z_hard_minus_easy"), "sa": paired_effect_summary(secondary, "delta_sa_hard_minus_easy")},
    }
    write_experiment_summary(directory, summary)
    return summary


def select_history_cohort(oof: Sequence[dict[str, Any]], n: int = 60) -> list[dict[str, Any]]:
    eligible = [row for row in oof if row["text_answer"] and row["image_answer"] and row["text_answer"] != row["image_answer"] and row["condition"].startswith("conflict_")]
    return _round_robin_sample(eligible, n, lambda row: (row["difficulty"], row.get("decision_side"), row["fold"]))


def run_experiment_2(
    runtime: Stage3Runtime,
    output_root: Path,
    oof: Sequence[dict[str, Any]],
    directions: SAOOFDirectionRepository,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / EXPERIMENT_DIR_NAMES[2]
    directory.mkdir(parents=True, exist_ok=True)
    cohort = select_history_cohort(oof)
    atomic_write_json(directory / "cohort_manifest.json", {"case_count": len(cohort), "case_ids": [row["case_id"] for row in cohort], "resume_semantics": "latest record per intervention_key is authoritative"})
    results_path = directory / "results.jsonl"
    completed = _completed_by_key(results_path)
    for row in cohort:
        deadline()
        key = f"history|{row['case_id']}"
        if key in completed:
            continue
        base = {"intervention_key": key, "experiment": "history", "case_id": row["case_id"], "item_id": row["item_id"], "fold": row["fold"]}

        def execute() -> dict[str, Any]:
            case = runtime.case(row["item_id"], row["prior_index"])
            direction = directions.get(row["fold"])
            branch_results: dict[str, Any] = {}
            final_users: dict[str, list[dict[str, Any]]] = {}
            for side, initial in (("text_first", row["text_answer"]), ("image_first", row["image_answer"])):
                messages = build_history_messages(case, row["condition"], side, str(initial))
                final_users[side] = messages
                generated = runtime.generator.generate_messages(
                    messages,
                    case.answer_classes,
                    max_new_tokens=48,
                )
                if not generated.parse_success or generated.source_metric_status != "completed" or not generated.normalized_answer or not generated.source_label:
                    raise RuntimeError(f"{side} Pass 1 failed: {generated.error}")
                assistant_text = source_prefix_from_generation(generated.raw_output, generated.source_label)
                pass2_messages = messages[:-1] + [assistant_message(assistant_text)]
                prepared = prepare_measurement(runtime.generator, pass2_messages, assistant_text=assistant_text, answer=generated.normalized_answer)
                measured = runtime.measure(prepared, direction)
                generation_logits = generated.source_attribution["class_logits"]
                pass2_logits = measured.source["class_logits"]
                max_logit_error = max(abs(float(a) - float(b)) for a, b in zip(generation_logits, pass2_logits))
                soft_error = abs(float(generated.source_attribution["soft_image_score"]) - float(measured.source["soft_image_score"]))
                tolerance = 0.25
                branch_results[side] = {
                    "initial_answer": initial,
                    "pass1": generated.to_dict(),
                    "pass2": measured.to_dict(),
                    "pass2_assistant_prefix": assistant_text,
                    "reconstruction": {"max_class_logit_abs_error": max_logit_error, "soft_score_abs_error": soft_error, "bf16_tolerance": tolerance, "passed": max_logit_error <= tolerance},
                    "final_evidence_hash": canonical_message_hash([messages[-2]]),
                }
                runtime.release_inputs(prepared)
            assert_endpoint_evidence_equal(final_users["text_first"], final_users["image_first"])
            tf = branch_results["text_first"]
            imf = branch_results["image_first"]
            tf_answer = tf["pass1"]["normalized_answer"]
            if_answer = imf["pass1"]["normalized_answer"]
            endpoint_matched = tf_answer == if_answer
            delta_z = float(imf["pass2"]["z_sa"] - tf["pass2"]["z_sa"])
            delta_sa = float(imf["pass2"]["source"]["soft_image_score"] - tf["pass2"]["source"]["soft_image_score"])
            def revision(initial: str, final: str) -> str:
                if final == row["text_answer"]:
                    final_side = "T"
                elif final == row["image_answer"]:
                    final_side = "I"
                else:
                    return "other"
                initial_side = "T" if initial == row["text_answer"] else "I"
                return f"{initial_side}→{final_side}"
            return {
                **base,
                "status": "completed",
                "condition": row["condition"],
                "prior_index": row["prior_index"],
                "text_answer": row["text_answer"],
                "image_answer": row["image_answer"],
                "branches": branch_results,
                "endpoint_matched": endpoint_matched,
                "final_evidence_equal": True,
                "delta_z_if_minus_tf": delta_z,
                "delta_sa_if_minus_tf": delta_sa,
                "text_first_revision": revision(str(row["text_answer"]), tf_answer),
                "image_first_revision": revision(str(row["image_answer"]), if_answer),
                "signed_update": delta_z,
            }

        append_jsonl(results_path, _safe_call(base, execute))
    raw_rows = load_jsonl(results_path)
    latest = {str(row["intervention_key"]): row for row in raw_rows}
    all_rows = list(latest.values())
    completed_rows = [row for row in all_rows if row.get("status") == "completed"]
    primary = [row for row in completed_rows if row.get("endpoint_matched") and row.get("final_evidence_equal")]
    reconstruction_values = [
        branch["reconstruction"]
        for row in completed_rows
        for branch in row["branches"].values()
    ]
    reconstruction_pass_rate = (
        sum(value["passed"] for value in reconstruction_values) / len(reconstruction_values)
        if reconstruction_values else None
    )
    soft_reconstruction_pass_rate = (
        sum(value["soft_score_abs_error"] <= 0.125 for value in reconstruction_values) / len(reconstruction_values)
        if reconstruction_values else None
    )
    summary = {
        "title": "Experiment 2 — Revision / History",
        "status": "completed",
        "n": len(primary),
        "attempted": len(all_rows),
        "failed": len(all_rows) - len(completed_rows),
        "pass1_pass2_gpu_smoke": "passed before formal pilot; see ../../gpu_smoke.json",
        "formal_reconstruction_diagnostic_pass_rate": reconstruction_pass_rate,
        "formal_soft_score_within_0.125_rate": soft_reconstruction_pass_rate,
        "formal_reconstruction_max_logit_abs_error": max((value["max_class_logit_abs_error"] for value in reconstruction_values), default=None),
        "formal_reconstruction_max_soft_abs_error": max((value["soft_score_abs_error"] for value in reconstruction_values), default=None),
        "primary_endpoint_matched": True,
        "delta_z_if_minus_tf": paired_effect_summary(primary, "delta_z_if_minus_tf"),
        "delta_sa_if_minus_tf": paired_effect_summary(primary, "delta_sa_if_minus_tf"),
        "revision_counts": {label: sum(row.get("text_first_revision") == label or row.get("image_first_revision") == label for row in completed_rows) for label in ("T→I", "I→T", "T→T", "I→I", "other")},
    }
    write_experiment_summary(directory, summary)
    return summary


def _select_answer_force_cases(artifacts: SAFormationArtifacts, n: int = 50) -> list[dict[str, Any]]:
    cases = read_json(artifacts.answer_force_manifest)["cases"]
    return _round_robin_sample(cases, n, lambda row: (row["decision_side"], row["difficulty"]))


def _mismatch_regression(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    observations: list[tuple[float, float, float]] = []
    for row in rows:
        natural = row["natural_answer"]
        for forced_side, outcome in ((0.0, row["force_text"]["source"]["soft_image_score"]), (1.0, row["force_image"]["source"]["soft_image_score"])):
            forced_answer = row["text_answer"] if forced_side == 0 else row["image_answer"]
            observations.append((forced_side, float(forced_answer != natural), float(outcome)))
    if not observations:
        return {"n": 0}
    x = np.asarray([[1.0, side, mismatch] for side, mismatch, _ in observations])
    y = np.asarray([outcome for _, _, outcome in observations])
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return {"n": len(observations), "terms": ["intercept", "forced_image_side", "natural_forced_mismatch"], "coefficients": [float(value) for value in coef]}


def run_experiment_3(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    output_root: Path,
    directions: SAOOFDirectionRepository,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / EXPERIMENT_DIR_NAMES[3]
    directory.mkdir(parents=True, exist_ok=True)
    cohort = _select_answer_force_cases(artifacts)
    old_results = {str(row["case_id"]): row for row in load_jsonl(artifacts.answer_force_results) if row.get("experiment") == "answer_force"}
    natural_baselines = {
        str(row["case_id"]): row.get("generated", {}).get("current_answer_result", {})
        for row in load_jsonl(artifacts.results)
        if row.get("status") == "completed"
    }
    atomic_write_json(directory / "cohort_manifest.json", {"case_count": len(cohort), "reuse": ["case IDs", "cohort definition", "baseline answers/probabilities", "A_T/A_I", "behavioral results only"], "recaptured": ["z_SA_forceT", "z_SA_forceI"], "cases": cohort})
    results_path = directory / "results.jsonl"
    completed = _completed_by_key(results_path)
    for source_case in cohort:
        deadline()
        key = f"mismatch|{source_case['case_id']}"
        if key in completed:
            continue
        base = {"intervention_key": key, "experiment": "answer_mismatch", "case_id": source_case["case_id"], "item_id": source_case["item_id"]}

        def execute() -> dict[str, Any]:
            case = runtime.case(source_case["item_id"], source_case["prior_index"])
            split = read_json(artifacts.item_split)["item_to_fold"]
            fold = int(split[str(source_case["item_id"])])
            direction = directions.get(fold)
            states: dict[str, Any] = {}
            for name, answer in (("force_text", source_case["text_answer"]), ("force_image", source_case["image_answer"])):
                assistant_text = f"**Answer**: {answer}\n{ASSISTANT_SOURCE_ATTRIBUTION_PREFILL}"
                messages = _single_turn_messages(case, source_case["condition"], assistant_text)
                prepared = prepare_measurement(runtime.generator, messages, assistant_text=assistant_text, answer=answer)
                states[name] = runtime.measure(prepared, direction).to_dict()
                runtime.release_inputs(prepared)
            old = old_results.get(source_case["case_id"], {})
            delta_z = float(states["force_image"]["z_sa"] - states["force_text"]["z_sa"])
            delta_sa = float(states["force_image"]["source"]["soft_image_score"] - states["force_text"]["source"]["soft_image_score"])
            return {
                **base,
                "status": "completed",
                "fold": fold,
                "prior_index": source_case["prior_index"],
                "condition": source_case["condition"],
                "difficulty": source_case["difficulty"],
                "natural_answer": source_case["normalized_answer"],
                "natural_answer_probability": natural_baselines.get(source_case["case_id"], {}).get("answer_prob"),
                "natural_answer_class_probabilities": natural_baselines.get(source_case["case_id"], {}).get("answer_class_probabilities"),
                "text_answer": source_case["text_answer"],
                "image_answer": source_case["image_answer"],
                "force_text": states["force_text"],
                "force_image": states["force_image"],
                "delta_z_forceI_minus_forceT": delta_z,
                "delta_sa_forceI_minus_forceT": delta_sa,
                "old_behavioral_sanity": {"available": bool(old), "old_aligned_delta_sa": old.get("aligned_delta_sa"), "old_directional": old.get("directional")},
            }

        append_jsonl(results_path, _safe_call(base, execute))
    all_rows = load_jsonl(results_path)
    rows = [row for row in all_rows if row.get("status") == "completed"]
    summary = {
        "title": "Experiment 3 — Answer Mismatch",
        "status": "completed",
        "n": len(rows),
        "failed": len(all_rows) - len(rows),
        "delta_z_forceI_minus_forceT": paired_effect_summary(rows, "delta_z_forceI_minus_forceT"),
        "delta_sa_forceI_minus_forceT": paired_effect_summary(rows, "delta_sa_forceI_minus_forceT"),
        "adjusted_model": _mismatch_regression(rows),
        "internal_states_reused": False,
    }
    write_experiment_summary(directory, summary)
    return summary


def _stable_cue(summary: dict[str, Any], z_key: str, sa_key: str) -> bool:
    z = summary.get(z_key, {})
    sa = summary.get(sa_key, {})
    return bool(
        z.get("n", 0) >= 20
        and sa.get("n", 0) >= 20
        and z.get("ci95", [None])[0] is not None
        and sa.get("ci95", [None])[0] is not None
        and z["ci95"][0] > 0
        and sa["ci95"][0] > 0
        and z.get("direction_rate", 0) >= 0.60
        and sa.get("direction_rate", 0) >= 0.60
    )


def _history_oriented_rows(history_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    primary = [row for row in history_rows if row.get("status") == "completed" and row.get("endpoint_matched")]
    if not primary:
        return [], 1.0
    orientation = 1.0 if statistics.fmean(float(row["delta_z_if_minus_tf"]) for row in primary) >= 0 else -1.0
    oriented = [
        {
            **row,
            "aligned_delta_z": orientation * float(row["delta_z_if_minus_tf"]),
            "aligned_delta_sa": orientation * float(row["delta_sa_if_minus_tf"]),
        }
        for row in primary
    ]
    return oriented, orientation


def select_mediation_cue(output_root: Path) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    mismatch_rows = [row for row in load_jsonl(output_root / EXPERIMENT_DIR_NAMES[3] / "results.jsonl") if row.get("status") == "completed"]
    if mismatch_rows:
        z = paired_effect_summary(mismatch_rows, "delta_z_forceI_minus_forceT")
        sa = paired_effect_summary(mismatch_rows, "delta_sa_forceI_minus_forceT")
        candidate = {"cue": "answer_mismatch", "rows": mismatch_rows, "orientation": 1.0, "z": z, "sa": sa}
        candidate["stable"] = _stable_cue({"z": z, "sa": sa}, "z", "sa")
        candidates.append(candidate)
    history_rows = load_jsonl(output_root / EXPERIMENT_DIR_NAMES[2] / "results.jsonl")
    oriented, orientation = _history_oriented_rows(history_rows)
    if oriented:
        z = paired_effect_summary(oriented, "aligned_delta_z")
        sa = paired_effect_summary(oriented, "aligned_delta_sa")
        candidate = {"cue": "history", "rows": oriented, "orientation": orientation, "z": z, "sa": sa}
        candidate["stable"] = _stable_cue({"z": z, "sa": sa}, "z", "sa")
        candidates.append(candidate)
    stable = [candidate for candidate in candidates if candidate["stable"]]
    if not stable:
        return None
    for candidate in stable:
        values = [float(row["delta_z_forceI_minus_forceT"] if candidate["cue"] == "answer_mismatch" else row["aligned_delta_z"]) for row in candidate["rows"]]
        candidate["standardized_effect"] = statistics.fmean(values) / (statistics.stdev(values) or float("inf"))
    return max(stable, key=lambda candidate: abs(candidate["standardized_effect"]))


def _mediation_contexts(
    runtime: Stage3Runtime,
    row: dict[str, Any],
    cue: str,
) -> tuple[Any, Any, float]:
    if cue == "answer_mismatch":
        case = runtime.case(row["item_id"], row["prior_index"])
        contexts = []
        for answer in (row["text_answer"], row["image_answer"]):
            assistant_text = f"**Answer**: {answer}\n{ASSISTANT_SOURCE_ATTRIBUTION_PREFILL}"
            messages = _single_turn_messages(case, row["condition"], assistant_text)
            contexts.append(prepare_measurement(runtime.generator, messages, assistant_text=assistant_text, answer=answer))
        return contexts[0], contexts[1], 1.0
    case = runtime.case(row["item_id"], row["prior_index"])
    contexts = []
    for side, initial in (("text_first", row["text_answer"]), ("image_first", row["image_answer"])):
        branch = row["branches"][side]
        assistant_text = branch["pass2_assistant_prefix"]
        messages = build_history_messages(case, row["condition"], side, str(initial), assistant_text=assistant_text)
        contexts.append(prepare_measurement(runtime.generator, messages, assistant_text=assistant_text, answer=branch["pass1"]["normalized_answer"]))
    return contexts[0], contexts[1], float(row.get("_mediation_orientation", 1.0))


def run_experiment_4(
    runtime: Stage3Runtime,
    output_root: Path,
    directions: SAOOFDirectionRepository,
    gate: GateDecision,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / EXPERIMENT_DIR_NAMES[4]
    directory.mkdir(parents=True, exist_ok=True)
    if not gate.allow_causal_mediator:
        summary = {"title": "Experiment 4 — PANL Mediation", "status": "skipped", "n": 0, "reason": f"Gate Level {gate.level} forbids causal-mediator interpretation"}
        atomic_write_json(directory / "cohort_manifest.json", {"cases": [], "reason": summary["reason"]})
        write_jsonl_atomic(directory / "results.jsonl", [])
        write_experiment_summary(directory, summary)
        return summary
    selected = select_mediation_cue(output_root)
    if selected is None:
        summary = {"title": "Experiment 4 — PANL Mediation", "status": "skipped", "n": 0, "reason": "No Exp 1/2/3 cue had stable paired effects on both z_SA and SA"}
        atomic_write_json(directory / "cohort_manifest.json", {"cases": [], "reason": summary["reason"]})
        write_jsonl_atomic(directory / "results.jsonl", [])
        write_experiment_summary(directory, summary)
        return summary
    cue = selected["cue"]
    cohort = list(selected["rows"])[:30]
    if cue == "history":
        for row in cohort:
            row["_mediation_orientation"] = selected["orientation"]
    atomic_write_json(directory / "cohort_manifest.json", {"cue": cue, "standardized_effect": selected["standardized_effect"], "pair_count": len(cohort), "case_ids": [row["case_id"] for row in cohort]})
    results_path = directory / "results.jsonl"
    completed = _completed_by_key(results_path)
    for row in cohort:
        deadline()
        key = f"mediation|{cue}|{row['case_id']}"
        if key in completed:
            continue
        base = {"intervention_key": key, "experiment": "mediation", "cue": cue, "case_id": row["case_id"], "item_id": row["item_id"], "fold": row["fold"]}

        def execute() -> dict[str, Any]:
            direction = directions.get(row["fold"])
            left_context, right_context, orientation = _mediation_contexts(runtime, row, cue)
            left_clean = runtime.measure(left_context, direction)
            right_clean = runtime.measure(right_context, direction)
            target = (left_clean.z_sa + right_clean.z_sa) / 2
            left_vector = coordinate_delta(left_clean.hidden, direction.d_unit, target)
            right_vector = coordinate_delta(right_clean.hidden, direction.d_unit, target)
            left_clamp = runtime.measure(left_context, direction, steering_vector=left_vector)
            right_clamp = runtime.measure(right_context, direction, steering_vector=right_vector)
            left_sham = runtime.measure(left_context, direction)
            right_sham = runtime.measure(right_context, direction)
            left_control = runtime.measure(left_context, direction, steering_vector=orthogonal_equal_norm_control(direction.d_unit, np.linalg.norm(left_vector), seed_material=key + "|left"))
            right_control = runtime.measure(right_context, direction, steering_vector=orthogonal_equal_norm_control(direction.d_unit, np.linalg.norm(right_vector), seed_material=key + "|right"))
            def contrast(left: Any, right: Any) -> float:
                return orientation * (_source_soft(right.to_dict()) - _source_soft(left.to_dict()))
            clean_effect = contrast(left_clean, right_clean)
            clamp_effect = contrast(left_clamp, right_clamp)
            sham_effect = contrast(left_sham, right_sham)
            control_effect = contrast(left_control, right_control)
            runtime.release_inputs(left_context, right_context)
            return {
                **base,
                "status": "completed",
                "orientation": orientation,
                "target_z": target,
                "target_rule": "branch midpoint",
                "clean_effect": clean_effect,
                "clamp_effect": clamp_effect,
                "sham_effect": sham_effect,
                "orthogonal_effect": control_effect,
                "clamp_attenuation": clean_effect - clamp_effect,
                "sham_attenuation": clean_effect - sham_effect,
                "orthogonal_attenuation": clean_effect - control_effect,
                "clamp_minus_control_attenuation": control_effect - clamp_effect,
                "left": {"clean": left_clean.to_dict(), "clamp": left_clamp.to_dict(), "sham": left_sham.to_dict(), "orthogonal": left_control.to_dict()},
                "right": {"clean": right_clean.to_dict(), "clamp": right_clamp.to_dict(), "sham": right_sham.to_dict(), "orthogonal": right_control.to_dict()},
            }

        append_jsonl(results_path, _safe_call(base, execute))
    all_rows = load_jsonl(results_path)
    rows = [row for row in all_rows if row.get("status") == "completed"]
    attenuation = paired_effect_summary(rows, "clamp_attenuation")
    versus_control = paired_effect_summary(rows, "clamp_minus_control_attenuation")
    evidence = bool(attenuation["ci95"][0] is not None and attenuation["ci95"][0] > 0 and versus_control["ci95"][0] is not None and versus_control["ci95"][0] > 0)
    summary = {
        "title": "Experiment 4 — PANL Mediation",
        "status": "completed",
        "n": len(rows),
        "failed": len(all_rows) - len(rows),
        "selected_cue": cue,
        "clamp_attenuation": attenuation,
        "sham_attenuation": paired_effect_summary(rows, "sham_attenuation"),
        "orthogonal_attenuation": paired_effect_summary(rows, "orthogonal_attenuation"),
        "clamp_minus_control_attenuation": versus_control,
        "causal_mediation_evidence": evidence,
        "claim_limit": "evidence for causal mediation, never complete mediation" if evidence else "no causal mediation claim",
    }
    write_experiment_summary(directory, summary)
    return summary


def _policy_prefix_hash(history_prefix: Sequence[dict[str, Any]], answer: str) -> str:
    return canonical_message_hash([*history_prefix, assistant_message(f"**Answer**: {answer}\n\n")])


def run_experiment_5(
    runtime: Stage3Runtime,
    output_root: Path,
    directions: SAOOFDirectionRepository,
    gate: GateDecision,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / EXPERIMENT_DIR_NAMES[5]
    directory.mkdir(parents=True, exist_ok=True)
    if not gate.run_natural_formation:
        summary = {"title": "Experiment 5 — Branched Post-Answer Continuation", "status": "skipped", "n": 0, "reason": "Natural projection gate failed"}
        atomic_write_json(directory / "cohort_manifest.json", {"contexts": [], "reason": summary["reason"]})
        write_jsonl_atomic(directory / "results.jsonl", [])
        write_experiment_summary(directory, summary)
        return summary
    history_rows = [row for row in load_jsonl(output_root / EXPERIMENT_DIR_NAMES[2] / "results.jsonl") if row.get("status") == "completed" and row.get("endpoint_matched")]
    pairs = history_rows[:20]
    contexts = [(row, side) for row in pairs for side in ("text_first", "image_first")]
    atomic_write_json(directory / "cohort_manifest.json", {"pair_count": len(pairs), "context_count": len(contexts), "gate_level": gate.level, "alphas_sigma_units": [-2, 0, 2] if gate.allow_policy_steering else [0], "authoritative_results_key_prefix": "policy_v3_shape_matched|", "superseded_attempts_retained_for_audit": ["policy|", "policy_v2_shared_double_newline|"], "contexts": [{"case_id": row["case_id"], "history_side": side} for row, side in contexts]})
    results_path = directory / "results.jsonl"
    completed = _completed_by_key(results_path)
    alphas = [-2.0, 0.0, 2.0] if gate.allow_policy_steering else [0.0]
    for row, side in contexts:
        deadline()
        key = f"policy_v3_shape_matched|{row['case_id']}|{side}"
        if key in completed:
            continue
        base = {"intervention_key": key, "experiment": "branched_post_answer", "case_id": row["case_id"], "item_id": row["item_id"], "fold": row["fold"], "history_side": side}

        def execute() -> dict[str, Any]:
            case = runtime.case(row["item_id"], row["prior_index"])
            initial = row["text_answer"] if side == "text_first" else row["image_answer"]
            answer = row["branches"][side]["pass1"]["normalized_answer"]
            history = build_history_messages(case, row["condition"], side, str(initial))
            prefix = history[:-1]
            sa_assistant = f"**Answer**: {answer}\n\n{ASSISTANT_SOURCE_ATTRIBUTION_PREFILL}"
            branch_a_messages = prefix + [assistant_message(sa_assistant)]
            branch_b_messages = prefix + [assistant_message(f"**Answer**: {answer}\n\n"), {"role": "user", "content": text_content(SOURCE_CHOICE_PROMPT)}, assistant_message("**Source Choice**:")]
            assert_policy_no_verbal_sa(branch_b_messages)
            hash_a = _policy_prefix_hash(prefix, answer)
            hash_b = _policy_prefix_hash(prefix, answer)
            if hash_a != hash_b:
                raise ValueError("Branch A/B canonical prefix through Answer differs")
            direction = directions.get(row["fold"])
            branch_a = prepare_measurement(runtime.generator, branch_a_messages, assistant_text=sa_assistant, answer=answer)
            branch_b = prepare_policy_measurement(runtime.generator, branch_b_messages, assistant_text="**Source Choice**:", fixed_answer=answer)
            common_length = max(int(branch_a.inputs.input_ids.shape[1]), int(branch_b.inputs.input_ids.shape[1]))
            pad_token_id = int(runtime.generator.tokenizer.pad_token_id)
            right_pad_measurement_inputs(branch_a, common_length, pad_token_id=pad_token_id)
            right_pad_measurement_inputs(branch_b, common_length, pad_token_id=pad_token_id)
            clean_sa = runtime.measure(branch_a, direction)
            clean_policy = runtime.measure(branch_b, direction, policy=True)
            prefix_hidden_max_error = float(np.max(np.abs(clean_sa.hidden - clean_policy.hidden)))
            prefix_z_error = abs(clean_sa.z_sa - clean_policy.z_sa)
            hidden_difference = clean_sa.hidden - clean_policy.hidden
            relative_l2 = float(np.linalg.norm(hidden_difference) / max(np.linalg.norm(clean_sa.hidden), 1e-12))
            cosine = float(clean_sa.hidden @ clean_policy.hidden / max(np.linalg.norm(clean_sa.hidden) * np.linalg.norm(clean_policy.hidden), 1e-12))
            prefix_passed = bool(
                cosine >= 0.999
                and relative_l2 <= 0.03
                and prefix_z_error / direction.sigma_z <= 0.10
            )
            doses: list[dict[str, Any]] = []
            for alpha in alphas:
                if alpha == 0:
                    measured = clean_policy
                else:
                    vector = alpha * direction.sigma_z * direction.d_unit
                    measured = runtime.measure(branch_b, direction, steering_vector=vector, policy=True)
                doses.append({"alpha_sigma": alpha, "expected_delta_z": alpha * direction.sigma_z, "delta_z_sigma": measured.applied_delta_z / direction.sigma_z, "policy": measured.to_dict(), "p_image": measured.source["soft_image_score"], "entropy": measured.source["source_entropy"], "hard_choice": measured.source["hard_label"]})
            runtime.release_inputs(branch_a, branch_b)
            return {
                **base,
                "status": "completed",
                "answer": answer,
                "canonical_prefix_hash_a": hash_a,
                "canonical_prefix_hash_b": hash_b,
                "policy_assistant_sa_leakage": False,
                "shape_matched_prefill_length": common_length,
                "clean_sa_branch": clean_sa.to_dict(),
                "clean_policy_branch": clean_policy.to_dict(),
                "prefix_hidden_max_abs_error": prefix_hidden_max_error,
                "prefix_z_abs_error": prefix_z_error,
                "prefix_z_error_sigma": prefix_z_error / direction.sigma_z,
                "prefix_hidden_relative_l2": relative_l2,
                "prefix_hidden_cosine": cosine,
                "prefix_reconstruction_passed": prefix_passed,
                "doses": doses,
            }

        append_jsonl(results_path, _safe_call(base, execute))
    raw_rows = load_jsonl(results_path)
    latest = {
        str(row["intervention_key"]): row
        for row in raw_rows
        if str(row.get("intervention_key", "")).startswith("policy_v3_shape_matched|")
    }
    all_rows = list(latest.values())
    rows = [row for row in all_rows if row.get("status") == "completed"]
    natural_rows = [{"item_id": row["item_id"], "z": row["clean_sa_branch"]["z_sa"], "p": next(dose["p_image"] for dose in row["doses"] if dose["alpha_sigma"] == 0)} for row in rows]
    association = None
    if len(natural_rows) >= 3:
        association = {
            "n": len(natural_rows),
            "pearson": float(pearsonr([r["z"] for r in natural_rows], [r["p"] for r in natural_rows]).statistic),
            "spearman": float(spearmanr([r["z"] for r in natural_rows], [r["p"] for r in natural_rows]).statistic),
            "spearman_bootstrap": item_cluster_bootstrap(natural_rows, lambda sample: spearmanr([r["z"] for r in sample], [r["p"] for r in sample]).statistic),
        }
    causal_rows: list[dict[str, Any]] = []
    if gate.allow_policy_steering:
        for row in rows:
            p = {float(dose["alpha_sigma"]): float(dose["p_image"]) for dose in row["doses"]}
            if -2.0 in p and 2.0 in p:
                causal_rows.append({"item_id": row["item_id"], "delta_p_image_plus2_minus_minus2": p[2.0] - p[-2.0]})
    summary = {
        "title": "Experiment 5 — Branched Post-Answer Continuation",
        "status": "completed",
        "n": len(rows),
        "failed": len(all_rows) - len(rows),
        "gate_level": gate.level,
        "natural_policy_association": association,
        "causal_policy_steering": paired_effect_summary(causal_rows, "delta_p_image_plus2_minus_minus2") if causal_rows else {"n": 0, "reason": "Gate Level 2 permits alpha=0 only"},
        "alpha_definition": "alpha units are fold-training natural SD(z_SA)",
        "kv_cache_fork_claimed": False,
        "continuation_definition": "separately reconstructed branched post-answer continuation",
        "prefix_reconstruction_passed": all(
            row.get("prefix_reconstruction_passed", row["prefix_hidden_max_abs_error"] <= 0.25)
            for row in rows
        ),
        "prefix_reconstruction_pass_rate": (
            sum(row.get("prefix_reconstruction_passed", row["prefix_hidden_max_abs_error"] <= 0.25) for row in rows) / len(rows)
            if rows else None
        ),
    }
    write_experiment_summary(directory, summary)
    return summary


def write_skipped_experiments(output_root: Path, start: int, reason: str) -> None:
    titles = {
        1: "Experiment 1 — Evidence Balance",
        2: "Experiment 2 — Revision / History",
        3: "Experiment 3 — Answer Mismatch",
        4: "Experiment 4 — PANL Mediation",
        5: "Experiment 5 — Branched Post-Answer Continuation",
    }
    for index in range(start, 6):
        directory = output_root / EXPERIMENT_DIR_NAMES[index]
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / "cohort_manifest.json", {"cases": [], "reason": reason})
        write_jsonl_atomic(directory / "results.jsonl", [])
        write_experiment_summary(directory, {"title": titles[index], "status": "skipped", "n": 0, "reason": reason})
