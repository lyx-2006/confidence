from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import AdditiveActivationHook, model_input_device, resolve_language_modules, run_logits_forward

from dp_SA.prompts import SA_PREFILL
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits

from .config import (
    ALPHAS,
    INFERENCE_PATH,
    LAYERS,
    LOGIT_PARITY_TOLERANCE,
    MODEL_PATH,
    POSITIONS,
    PROBABILITY_PARITY_TOLERANCE,
    RESULTS_ROOT,
    SMOKE_ALPHAS,
    SMOKE_LAYERS,
    SOFT_SA_PARITY_TOLERANCE,
)
from .fingerprint import check_or_write_config, experiment_config
from .io_utils import append_jsonl, array_hash, atomic_json, canonical_hash, load_jsonl, sha256_file
from .manifests import load_frozen_manifests
from .positions import locate_checkpoint_positions
from .vectors import build_vectors, load_scaled_vectors


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [
            {"type": "image", "image": str(Path(row["image_path"]).resolve())},
            {"type": "text", "text": str(row["phase1_prompt"])},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]},
    ]


def trial_key(row: dict[str, Any]) -> str:
    return f'{row["case_id"]}|{row["position"]}|L{int(row["layer"])}|a{float(row["alpha"]):g}'


def class_margin(logits: np.ndarray, class_index: int) -> float:
    values = np.asarray(logits, dtype=np.float64)
    return float(values[class_index] - np.max(np.delete(values, class_index)))


def validate_alpha_zero(
    *,
    clean_logits: np.ndarray,
    clean_probabilities: np.ndarray,
    clean_soft_sa: float,
    clean_hard_class: int,
    scored: dict[str, Any],
    before: np.ndarray,
    after: np.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    logit_error = float(np.max(np.abs(np.asarray(scored["class_logits"], dtype=float) - clean_logits)))
    probability_error = float(np.max(np.abs(np.asarray(scored["class_probabilities"], dtype=float) - clean_probabilities)))
    soft_error = abs(float(scored["soft_sa_image_score"]) - float(clean_soft_sa))
    activation_equal = bool(np.array_equal(before, after))
    passed = (
        logit_error <= LOGIT_PARITY_TOLERANCE
        and probability_error <= PROBABILITY_PARITY_TOLERANCE
        and soft_error <= SOFT_SA_PARITY_TOLERANCE
        and int(scored["argmax_hard_class"]) == int(clean_hard_class)
        and int(diagnostics["hook_call_count"]) == 1
        and int(diagnostics["steering_applied_count"]) == 1
        and activation_equal
        and array_hash(before) == array_hash(after)
    )
    result = {
        "passed": passed,
        "logit_max_abs_error": logit_error,
        "probability_max_abs_error": probability_error,
        "soft_sa_abs_error": soft_error,
        "hard_class_equal": int(scored["argmax_hard_class"]) == int(clean_hard_class),
        "activation_equal": activation_equal,
    }
    if not passed:
        raise RuntimeError(f"Alpha-zero parity failed: {result}")
    return result


def _validate_existing_trials(rows: Sequence[dict[str, Any]], config_fingerprint: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        if row.get("config_fingerprint") != config_fingerprint:
            raise ValueError(f"Trial config fingerprint mismatch: {trial_key(row)}")
        key = trial_key(row)
        if key in output:
            raise ValueError(f"Duplicate completed trial: {key}")
        output[key] = row
    return output


def run_steering(*, output_root: Path = RESULTS_ROOT, smoke: bool = False, resume: bool = False) -> dict[str, Any]:
    root = Path(output_root)
    construction, test, _selection = load_frozen_manifests(root)
    base_config = experiment_config(construction, test, smoke=smoke)
    clean_path = root / "artifacts" / "diagnostics" / "clean_capture.jsonl"
    clean_rows = [row for row in load_jsonl(clean_path) if row.get("status") == "completed"]
    clean_by_case = {str(row["case_id"]): row for row in clean_rows}
    expected_clean = {str(row["case_id"]) for row in [*construction, *test]}
    if not expected_clean.issubset(clean_by_case):
        raise ValueError("Clean checkpoint capture is incomplete")
    vector_metadata = build_vectors(root, construction, clean_rows, smoke=smoke, resume=resume)
    run_config = {
        **base_config,
        "capture_artifact_sha256": sha256_file(clean_path),
        "vector_fingerprint": vector_metadata["fingerprint"],
    }
    run_config["fingerprint"] = canonical_hash(run_config)
    check_or_write_config(root / "progress" / "steering_config.json", run_config, resume=resume)
    trial_path = root / "artifacts" / "diagnostics" / "steering_trials.jsonl"
    completed = _validate_existing_trials(load_jsonl(trial_path), run_config["fingerprint"])
    positions = POSITIONS
    layers = SMOKE_LAYERS if smoke else LAYERS
    alphas = SMOKE_ALPHAS if smoke else ALPHAS
    expected = len(test) * len(positions) * len(layers) * len(alphas)
    if len(completed) == expected:
        summary = {"status": "complete", "smoke": smoke, "completed_cells": expected, "expected_cells": expected, "new_gpu_forwards": 0, "resumed_noop": True}
        atomic_json(root / "progress" / "steering_progress.json", summary)
        return summary
    if len(completed) > expected:
        raise ValueError("Trial artifact contains more rows than the configured grid")

    vectors = load_scaled_vectors(root, vector_metadata)
    runtime = load_runtime(INFERENCE_PATH)
    inference = runtime.QwenVLInference(str(MODEL_PATH))
    modules = resolve_language_modules(inference.model)
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    expected_class_ids = class_token_ids(tokenizer)
    device = model_input_device(inference)
    new_forwards = 0
    started = time.time()

    for manifest_row in test:
        case_id = str(manifest_row["case_id"])
        clean = clean_by_case[case_id]
        messages = _messages(manifest_row)
        rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
        if canonical_hash(rendered) != clean["rendered_prompt_hash"]:
            raise ValueError(f"Rendered prompt changed for {case_id}")
        inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=device)
        located = locate_checkpoint_positions(tokenizer, rendered, inputs, str(manifest_row["phase0_raw_answer"]))
        for name in (*POSITIONS, "P1_SAC"):
            if int(located[name]["processed_index"]) != int(clean["positions"][name]["processed_index"]):
                raise ValueError(f"Resume position changed for {case_id} {name}")
        sac = int(located["P1_SAC"]["processed_index"])
        sequence_length = int(inputs.input_ids.shape[1])
        clean_logits = np.asarray(clean["class_logits"], dtype=np.float64)
        clean_probabilities = np.asarray(clean["class_probabilities"], dtype=np.float64)
        clean_soft = float(clean["soft_sa_image_score"])
        clean_hard = int(clean["argmax_hard_class"])
        clean_margin = class_margin(clean_logits, clean_hard)
        if [int(value) for value in clean["class_token_ids"]] != expected_class_ids:
            raise ValueError(f"Class token IDs changed for {case_id}")

        for position in positions:
            target = int(located[position]["processed_index"])
            for layer in layers:
                vector = torch.from_numpy(vectors[(position, int(layer))])
                for alpha in alphas:
                    proto = {
                        "case_id": case_id,
                        "item_id": str(manifest_row["item_id"]),
                        "test_side": str(manifest_row["test_side"]),
                        "position": position,
                        "layer": int(layer),
                        "alpha": float(alpha),
                    }
                    if trial_key(proto) in completed:
                        continue
                    try:
                        hook = AdditiveActivationHook(
                            modules,
                            layer_index=int(layer),
                            target_position=target,
                            steering_vector=vector * float(alpha),
                            prefill_sequence_length=sequence_length,
                            injection_site="block_output",
                        )
                        with hook:
                            logits = run_logits_forward(inference.model, inputs, [sac], modules)[sac]
                        new_forwards += 1
                        diagnostics = hook.diagnostics()
                        scored = soft_sa_from_logits(logits, expected_class_ids)
                        before = hook.h_before.numpy()
                        after = hook.h_after.numpy()
                        before_norm = float(np.linalg.norm(before))
                        after_norm = float(np.linalg.norm(after))
                        cosine = float(np.dot(before, after) / (before_norm * after_norm))
                        ratio = float(after_norm / before_norm)
                        steered_logits = np.asarray(scored["class_logits"], dtype=np.float64)
                        steered_margin = class_margin(steered_logits, clean_hard)
                        alpha_zero = None
                        if float(alpha) == 0.0:
                            alpha_zero = validate_alpha_zero(
                                clean_logits=clean_logits,
                                clean_probabilities=clean_probabilities,
                                clean_soft_sa=clean_soft,
                                clean_hard_class=clean_hard,
                                scored=scored,
                                before=before,
                                after=after,
                                diagnostics=diagnostics,
                            )
                        all_values = np.concatenate([steered_logits, np.asarray(scored["class_probabilities"], dtype=float), [float(scored["soft_sa_image_score"]), cosine, ratio]])
                        finite = bool(np.isfinite(all_values).all())
                        probability_valid = abs(float(scored["probability_sum"]) - 1.0) <= 1e-9
                        if not finite or not probability_valid:
                            raise ValueError("Steering trial produced invalid finite/probability values")
                        result = {
                            "status": "completed",
                            **proto,
                            "processed_position": target,
                            "clean_soft_sa": clean_soft,
                            "steered_soft_sa": float(scored["soft_sa_image_score"]),
                            "delta_soft_sa": float(scored["soft_sa_image_score"]) - clean_soft,
                            "clean_hard_class": clean_hard,
                            "steered_hard_class": int(scored["argmax_hard_class"]),
                            "hard_class_changed": int(scored["argmax_hard_class"]) != clean_hard,
                            "hard_class_delta": int(scored["argmax_hard_class"]) - clean_hard,
                            "clean_class_logits": clean_logits.tolist(),
                            "clean_class_probabilities": clean_probabilities.tolist(),
                            "steered_class_logits": scored["class_logits"],
                            "steered_class_probabilities": scored["class_probabilities"],
                            "probability_sum": float(scored["probability_sum"]),
                            "clean_class_margin": clean_margin,
                            "steered_clean_class_margin": steered_margin,
                            "margin_change": steered_margin - clean_margin,
                            "saturated": float(scored["soft_sa_image_score"]) <= 0.05 + 1e-9 or float(scored["soft_sa_image_score"]) >= 0.95 - 1e-9,
                            "finite_values": finite,
                            "activation_cosine": cosine,
                            "activation_norm_ratio": ratio,
                            "hook_diagnostics": diagnostics,
                            "activation_before_hash": array_hash(before),
                            "activation_after_hash": array_hash(after),
                            "alpha_zero_parity": alpha_zero,
                            "config_fingerprint": run_config["fingerprint"],
                        }
                        append_jsonl(trial_path, result)
                        completed[trial_key(result)] = result
                    except Exception as exc:
                        failure = {"stage": "steering", **proto, "type": type(exc).__name__, "message": str(exc), "timestamp": time.time()}
                        append_jsonl(root / "progress" / "failures.jsonl", failure)
                        atomic_json(root / "progress" / "steering_progress.json", {"status": "failed", "completed_cells": len(completed), "expected_cells": expected, "failure": failure})
                        raise
                    if new_forwards % 10 == 0:
                        atomic_json(root / "progress" / "steering_progress.json", {
                            "status": "running", "completed_cells": len(completed), "expected_cells": expected,
                            "new_gpu_forwards": new_forwards, "elapsed_seconds": time.time() - started, "last": proto,
                        })
    if len(completed) != expected:
        raise RuntimeError(f"Steering grid is incomplete: {len(completed)}/{expected}")
    summary = {
        "status": "complete", "smoke": smoke, "completed_cells": len(completed), "expected_cells": expected,
        "new_gpu_forwards": new_forwards, "resumed_noop": new_forwards == 0, "elapsed_seconds": time.time() - started,
    }
    atomic_json(root / "progress" / "steering_progress.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke and not args.output_root:
        parser.error("--smoke requires an explicit --output-root outside the formal results directory; prefer run_pipeline --smoke")
    root = Path(args.output_root) if args.output_root else RESULTS_ROOT
    if args.smoke:
        try:
            root.resolve().relative_to(RESULTS_ROOT.resolve())
        except ValueError:
            pass
        else:
            parser.error("smoke output cannot be inside the formal results directory")
    print(json.dumps(run_steering(output_root=root, smoke=args.smoke, resume=args.resume), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
