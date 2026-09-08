from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch

from confidence_test.answer_metrics import parse_answer_output
from dp_SA.prompts import ANSWER_PREFILL
from dp_SA.unimodal_logit_confidence.score_unimodal import candidate_suffix_ids
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_logits_forward

from .artifacts import MeanArtifacts, evaluation_shape_key
from .config import CONDITIONS, TEMPERATURE
from .hooks import EmbeddingReplacement, EmbeddingReplacementHook, resolve_language_model
from .io_utils import canonical_hash, load_jsonl, upsert_jsonl
from .metrics import restricted_probabilities
from .protocol import locate_evidence, phase0_messages, prepare_phase0


def condition_components(condition: str) -> tuple[str, ...]:
    mapping = {
        "clean": (), "10_text_corrupt": ("text",),
        "01_image_corrupt": ("image",), "00_both_corrupt": ("image", "text"),
    }
    if condition not in mapping:
        raise ValueError(f"Unknown corruption condition: {condition}")
    return mapping[condition]


def scores_from_vocab(logits: torch.Tensor, candidates: Sequence[str],
                      candidate_ids: dict[str, list[int]]) -> list[float]:
    if any(len(candidate_ids[name]) != 1 for name in candidates):
        raise ValueError("Single-token scoring received a multi-token candidate")
    return [float(logits[candidate_ids[name][0]]) for name in candidates]


def teacher_forced_log_probability(logits: dict[int, torch.Tensor], positions: Sequence[int],
                                   suffix: Sequence[int]) -> float:
    if not suffix or len(positions) != len(suffix):
        raise ValueError("Teacher-forced positions and suffix must have equal positive length")
    return sum(float(torch.log_softmax(logits[int(position)].double(), dim=-1)[int(token)])
               for position, token in zip(positions, suffix, strict=True))


def tokenization_preflight(processor: Any, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    audits: dict[str, dict[str, list[int]]] = {}
    lengths: dict[int, int] = {}
    for row in rows:
        _prompt, messages = phase0_messages(row)
        rendered = render_continued_assistant(processor, messages, ANSWER_PREFILL)
        ids = {candidate: candidate_suffix_ids(tokenizer, rendered, candidate)
               for candidate in row["answer_classes"]}
        if len(ids) != 12 or len({tuple(value) for value in ids.values()}) != 12:
            raise ValueError(f"Candidate tokenization is not 12 distinct sequences: {row['case_id']}")
        for value in ids.values():
            lengths[len(value)] = lengths.get(len(value), 0) + 1
        audits[str(row["case_id"])] = ids
    policy = "single_token_next_token_logits" if set(lengths) == {1} else "sequence_score_temperature_1"
    historical_lengths = [len(row["historical_phase0"]["phase0_generated_token_ids"]) for row in rows]
    if policy == "single_token_next_token_logits":
        logical = len(rows) * len(CONDITIONS)
        internal = sum(historical_lengths) + len(rows) * 3
    else:
        logical = len(rows) + len(rows) * len(CONDITIONS) * 12
        internal = sum(historical_lengths) + len(rows) * len(CONDITIONS) * 12
    return {
        "policy": policy, "candidate_length_occurrences": {str(k): v for k, v in sorted(lengths.items())},
        "candidate_token_ids": audits, "logical_evaluations": logical,
        "estimated_internal_model_forwards": internal,
        "historical_generation_length_counts": {
            str(length): historical_lengths.count(length) for length in sorted(set(historical_lengths))
        },
    }


def _candidate_result(row: dict[str, Any], condition: str, scores: Sequence[float],
                      candidate_ids: dict[str, list[int]], details: dict[str, Any],
                      diagnostics: dict[str, Any], artifacts: MeanArtifacts,
                      run_fingerprint: str, *, forward_calls: int) -> dict[str, Any]:
    candidates = list(row["answer_classes"])
    probabilities = restricted_probabilities(scores, TEMPERATURE)
    fixed = str(row["phase0_normalized_answer"])
    fixed_index = candidates.index(fixed)
    value = float(probabilities[fixed_index])
    artifacts_used = condition_components(condition)
    hashes = artifacts.manifest["artifact_sha256"]
    return {
        "score_key": f"{row['case_id']}|{condition}", "run_fingerprint": run_fingerprint,
        "case_id": row["case_id"], "family_id": row["family_id"], "item_id": str(row["item_id"]),
        "dataset_condition": row["condition"], "answer_side": row["answer_side"],
        "corruption_condition": condition, "fixed_answer": fixed,
        "candidate_order": candidates, "candidate_token_ids": candidate_ids,
        "candidate_scores": {name: float(scores[index]) for index, name in enumerate(candidates)},
        "restricted_probabilities": {name: float(probabilities[index]) for index, name in enumerate(candidates)},
        "probability_sum": float(probabilities.sum()), "fixed_answer_probability": value,
        "fixed_answer_log_probability": float(math.log(max(value, torch.finfo(torch.float64).tiny))),
        "condition_argmax_answer": candidates[int(probabilities.argmax())],
        "spans": {"image": details["image_positions"], "text": details["text_positions"]},
        "span_lengths": {"image": len(details["image_positions"]), "text": len(details["text_positions"])},
        "replacement_diagnostics": diagnostics,
        "artifact_sha256": {name: hashes[name] for name in artifacts_used},
        "mean_artifact_fingerprint": artifacts.manifest["fingerprint"],
        "logical_forward_calls": int(forward_calls), "temperature": TEMPERATURE,
    }


def _replacements(condition: str, inputs: Any, details: dict[str, Any], artifacts: MeanArtifacts,
                  hidden_size: int) -> list[EmbeddingReplacement]:
    shape_key = evaluation_shape_key(inputs, details, hidden_size)
    values = {
        "image": EmbeddingReplacement("image", tuple(details["image_positions"]),
                                      artifacts.image_for(shape_key, len(details["image_positions"]))),
        "text": EmbeddingReplacement("text", tuple(details["text_positions"]),
                                     artifacts.text_for(len(details["text_positions"]))),
    }
    return [values[name] for name in condition_components(condition)]


def _generate_parity(inference: Any, row: dict[str, Any], candidate_ids: dict[str, list[int]],
                     run_fingerprint: str) -> dict[str, Any]:
    _rendered, inputs, details = prepare_phase0(inference.processor, row, device=model_input_device(inference))
    with torch.inference_mode():
        generated = inference.model.generate(
            **inputs, max_new_tokens=24, do_sample=False, use_cache=True,
            return_dict_in_generate=True, output_scores=True,
        )
    if not generated.scores:
        raise RuntimeError("Parity generation returned no first-token scores")
    input_length = int(inputs.input_ids.shape[1])
    tokens = [int(value) for value in generated.sequences[0, input_length:].tolist()]
    tokenizer = inference.processor.tokenizer
    continuation = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    raw_output = ANSWER_PREFILL + continuation
    answer, normalized, parse_success = parse_answer_output(raw_output)
    history = row["historical_phase0"]
    checks = {
        "phase0_prompt_exact": details["prompt"] == history["phase0_prompt"],
        "phase0_prompt_hash_exact": details["prompt_hash"] == history["phase0_prompt_hash"],
        "image_sha256_exact": details["image_sha256"] == history["image_sha256"],
        "raw_answer_exact": answer == history["phase0_raw_answer"],
        "normalized_answer_exact": normalized == history["phase0_normalized_answer"],
        "generated_token_ids_exact": tokens == history["phase0_generated_token_ids"],
        "parse_success": bool(parse_success),
    }
    first = generated.scores[0][0].detach().float().cpu()
    scores = scores_from_vocab(first, row["answer_classes"], candidate_ids)
    return {
        "parity_key": str(row["case_id"]), "run_fingerprint": run_fingerprint,
        "case_id": row["case_id"], "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "expected_raw_answer": history["phase0_raw_answer"], "actual_raw_answer": answer,
        "expected_normalized_answer": history["phase0_normalized_answer"], "actual_normalized_answer": normalized,
        "expected_generated_token_ids": history["phase0_generated_token_ids"], "actual_generated_token_ids": tokens,
        "generated_token_count": len(tokens), "candidate_scores": scores,
        "candidate_token_ids": candidate_ids, "input_details": details,
    }


def _single_score(inference: Any, row: dict[str, Any], condition: str, candidate_ids: dict[str, list[int]],
                  artifacts: MeanArtifacts, hidden_size: int, *, interpolation: float = 1.0):
    _rendered, inputs, details = prepare_phase0(inference.processor, row, device=model_input_device(inference))
    position = int(inputs.input_ids.shape[1]) - 1
    replacements = _replacements(condition, inputs, details, artifacts, hidden_size)
    if not replacements:
        logits = run_logits_forward(inference.model, inputs, [position])[position]
        diagnostics: dict[str, Any] = {}
    else:
        hook = EmbeddingReplacementHook(resolve_language_model(inference.model), replacements=replacements,
                                        prefill_sequence_length=int(inputs.input_ids.shape[1]),
                                        hidden_size=hidden_size, interpolation=interpolation)
        with hook:
            logits = run_logits_forward(inference.model, inputs, [position])[position]
        diagnostics = hook.diagnostics()
    scores = scores_from_vocab(logits, row["answer_classes"], candidate_ids)
    return scores, details, diagnostics


def _sequence_scores(inference: Any, row: dict[str, Any], condition: str,
                     candidate_ids: dict[str, list[int]], artifacts: MeanArtifacts, hidden_size: int):
    _prompt, messages = phase0_messages(row)
    rendered = render_continued_assistant(inference.processor, messages, ANSWER_PREFILL)
    base = prepare_multimodal_inputs(inference.processor, messages, rendered, device=model_input_device(inference))
    base_length = int(base.input_ids.shape[1])
    totals: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    first_details: dict[str, Any] | None = None
    for candidate in row["answer_classes"]:
        full_rendered = rendered + candidate
        inputs = prepare_multimodal_inputs(inference.processor, messages, full_rendered, device=model_input_device(inference))
        details = locate_evidence(inference.processor, row, full_rendered, inputs)
        suffix = [int(value) for value in inputs.input_ids[0, base_length:].tolist()]
        if suffix != candidate_ids[candidate]:
            raise ValueError(f"Processed candidate suffix mismatch: {row['case_id']} {candidate}")
        positions = list(range(base_length - 1, int(inputs.input_ids.shape[1]) - 1))
        replacements = _replacements(condition, inputs, details, artifacts, hidden_size)
        if replacements:
            hook = EmbeddingReplacementHook(resolve_language_model(inference.model), replacements=replacements,
                                            prefill_sequence_length=int(inputs.input_ids.shape[1]), hidden_size=hidden_size)
            with hook:
                logits = run_logits_forward(inference.model, inputs, positions)
            diagnostics.append(hook.diagnostics())
        else:
            logits = run_logits_forward(inference.model, inputs, positions)
            diagnostics.append({})
        total = teacher_forced_log_probability(logits, positions, suffix)
        totals.append(total)
        first_details = first_details or details
    return totals, first_details, {"candidate_forwards": diagnostics}


def run_scores(inference: Any, rows: Sequence[dict[str, Any]], artifacts: MeanArtifacts,
               preflight: dict[str, Any], output_dir: Path, run_fingerprint: str,
               *, smoke: bool) -> dict[str, Any]:
    modules = resolve_language_modules(inference.model)
    parity_path = output_dir / "parity_audit.jsonl"
    scores_path = output_dir / "condition_scores.jsonl"
    parity_rows = load_jsonl(parity_path, repair_trailing=True)
    score_rows = load_jsonl(scores_path, repair_trailing=True)
    parity_by = {str(row["case_id"]): row for row in parity_rows}
    score_keys = {str(row["score_key"]) for row in score_rows}
    if len(parity_by) != len(parity_rows) or len(score_keys) != len(score_rows):
        raise ValueError("Resume files contain duplicate keys")
    if any(row.get("run_fingerprint") != run_fingerprint for row in [*parity_rows, *score_rows]):
        raise ValueError("Resume row fingerprint mismatch")
    new_logical = 0
    new_internal = 0
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in parity_by:
            if parity_by[case_id].get("status") != "passed":
                raise RuntimeError(f"Existing Phase 0 parity failure: {case_id}")
            continue
        audit = _generate_parity(inference, row, preflight["candidate_token_ids"][case_id], run_fingerprint)
        parity_rows = upsert_jsonl(parity_path, parity_rows, audit, key="parity_key")
        parity_by[case_id] = audit
        new_logical += 1
        new_internal += int(audit["generated_token_count"])
        if audit["status"] != "passed":
            raise RuntimeError(f"Phase 0 parity failed for {case_id}: {audit['checks']}")
    if set(parity_by) != {str(row["case_id"]) for row in rows}:
        raise RuntimeError("Parity cohort is incomplete; corruption is forbidden")

    for row in rows:
        case_id = str(row["case_id"])
        ids = preflight["candidate_token_ids"][case_id]
        for condition in CONDITIONS:
            score_key = f"{case_id}|{condition}"
            if score_key in score_keys:
                continue
            if preflight["policy"] == "single_token_next_token_logits" and condition == "clean":
                audit = parity_by[case_id]
                scores = audit["candidate_scores"]
                details = audit["input_details"]
                diagnostics = {}
                forwards = 0
            elif preflight["policy"] == "single_token_next_token_logits":
                scores, details, diagnostics = _single_score(
                    inference, row, condition, ids, artifacts, modules.hidden_size,
                )
                forwards = 1
            else:
                scores, details, diagnostics = _sequence_scores(
                    inference, row, condition, ids, artifacts, modules.hidden_size,
                )
                forwards = 12
            result = _candidate_result(row, condition, scores, ids, details, diagnostics,
                                       artifacts, run_fingerprint, forward_calls=forwards)
            score_rows = upsert_jsonl(scores_path, score_rows, result, key="score_key")
            score_keys.add(score_key)
            new_logical += forwards
            new_internal += forwards

    lambda_report = None
    lambda_path = output_dir / "lambda_zero_audit.json"
    if smoke:
        if lambda_path.exists():
            lambda_report = __import__("json").loads(lambda_path.read_text())
            if lambda_report.get("run_fingerprint") != run_fingerprint:
                raise ValueError("Lambda-zero audit fingerprint mismatch")
        else:
            row = rows[0]
            ids = preflight["candidate_token_ids"][str(row["case_id"])]
            values, _details, diagnostics = _single_score(
                inference, row, "00_both_corrupt", ids, artifacts, modules.hidden_size, interpolation=0.0,
            )
            clean = parity_by[str(row["case_id"])]["candidate_scores"]
            exact = values == clean
            lambda_report = {
                "status": "passed" if exact else "failed", "run_fingerprint": run_fingerprint,
                "case_id": row["case_id"], "candidate_logits_exact": exact,
                "clean_candidate_scores": clean, "lambda_zero_candidate_scores": values,
                "hook_diagnostics": diagnostics, "logical_forward_calls": 1,
            }
            from .io_utils import atomic_json
            atomic_json(lambda_path, lambda_report)
            new_logical += 1
            new_internal += 1
            if not exact:
                raise RuntimeError("Lambda=0 logits differ from clean logits")
    expected = len(rows) * len(CONDITIONS)
    actual = sum(str(result["case_id"]) in {str(row["case_id"]) for row in rows} for result in score_rows)
    if actual != expected:
        raise RuntimeError(f"Condition score cohort incomplete: {actual}/{expected}")
    return {
        "status": "passed", "case_count": len(rows), "condition_score_count": actual,
        "new_logical_evaluations": new_logical, "new_internal_model_forwards": new_internal,
        "resumed_noop": new_logical == 0, "lambda_zero": lambda_report,
    }


__all__ = [
    "condition_components", "run_scores", "scores_from_vocab",
    "teacher_forced_log_probability", "tokenization_preflight",
]
