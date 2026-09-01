from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import AdditiveActivationHook, model_input_device, resolve_language_modules, run_logits_forward

from dp_SA.checkpoint_steering.run import class_margin, validate_alpha_zero
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits

from .config import ALPHAS, DIRECTIONS, INFERENCE_PATH, LAYERS, MODEL_PATH, POSITIONS, RESULTS_ROOT, SMOKE_ALPHAS, SMOKE_DIRECTIONS, SMOKE_LAYERS
from .fingerprint import check_or_write, experiment_config
from .io_utils import append_jsonl, array_hash, atomic_json, canonical_hash, load_jsonl, sha256_file
from .prepare import MANIFEST_NAMES
from .vectors import build_vectors, load_vector


def trial_key(row: dict[str, Any]) -> str:
    return f'{row["case_id"]}|{row["position"]}|fold={int(row["fold"])}|{row["direction"]}|L{int(row["layer"])}|a{float(row["alpha"]):g}'


def paired_trial_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in ("case_id", "family_id", "item_id", "fold", "answer", "test_side", "condition", "test_status", "direction", "layer", "alpha"))


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image", "image": str(Path(row["image_path"]).resolve())}, {"type": "text", "text": str(row["phase1_prompt"])}]}, {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]}]


def _manifest_fingerprints(root: Path) -> dict[str, str]:
    directory = root / "artifacts" / "manifests"
    return {name: sha256_file(directory / name) for name in MANIFEST_NAMES}


def run_steering(*, output_root: Path = RESULTS_ROOT, smoke: bool = False, resume: bool = False, position_filter: str | None = None) -> dict[str, Any]:
    root = Path(output_root); split_gate = json.loads((root / "progress" / "split_gate.json").read_text())
    if split_gate.get("status") != "passed": raise RuntimeError("Steering requires a passed split gate")
    test = load_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl")
    clean_rows = [row for row in load_jsonl(root / "artifacts" / "diagnostics" / "clean_capture.jsonl") if row.get("status") == "completed"]
    clean = {str(row["case_id"]): row for row in clean_rows}
    if not {str(row["case_id"]) for row in test}.issubset(clean): raise ValueError("Test clean capture is incomplete")
    selected_positions = [position_filter] if position_filter else list(POSITIONS)
    unknown = set(selected_positions) - set(POSITIONS)
    if unknown:
        raise ValueError(f"Unknown position filter: {sorted(unknown)}")
    vectors = build_vectors(root, smoke=smoke, resume=resume)
    base = experiment_config(smoke=smoke, manifest_fingerprints=_manifest_fingerprints(root))
    run_config = {**base, "clean_capture_sha256": sha256_file(root / "artifacts" / "diagnostics" / "clean_capture.jsonl"), "vector_fingerprint": vectors["fingerprint"]}
    run_config["fingerprint"] = canonical_hash(run_config)
    # Sharding is an execution-only change.  Preserve the established formal
    # run fingerprint so already completed trials remain resumable.
    existing_run_config = root / "progress" / "run_config.json"
    if position_filter and existing_run_config.is_file():
        run_config = json.loads(existing_run_config.read_text())
    check_or_write(root / "progress" / "run_config.json", run_config, resume=resume)
    canonical_trial_path = root / "artifacts" / "diagnostics" / "steering_trials.jsonl"
    trial_path = (root / "artifacts" / "diagnostics" / f"steering_trials.{position_filter}.jsonl") if position_filter else canonical_trial_path
    progress_path = root / "progress" / (f"steering_progress.{position_filter}.json" if position_filter else "steering_progress.json")
    # Seed a position shard from the legacy combined file exactly once.  Each
    # shard then has exclusive ownership of its output file and can run safely
    # in a separate process/GPU.
    if position_filter and not trial_path.exists() and canonical_trial_path.exists():
        for existing in load_jsonl(canonical_trial_path):
            if existing.get("position") == position_filter:
                append_jsonl(trial_path, existing)
    completed = {}
    for row in load_jsonl(trial_path):
        if row.get("status") != "completed": continue
        if row.get("config_fingerprint") != run_config["fingerprint"]: raise ValueError("Existing trial config fingerprint mismatch")
        key = trial_key(row)
        if key in completed: raise ValueError(f"Duplicate trial: {key}")
        completed[key] = row
    layers = SMOKE_LAYERS if smoke else LAYERS; alphas = SMOKE_ALPHAS if smoke else ALPHAS; directions = SMOKE_DIRECTIONS if smoke else DIRECTIONS
    expected = len(test) * len(selected_positions) * len(directions) * len(layers) * len(alphas)
    if len(completed) == expected:
        if not position_filter:
            keys = {position: {paired_trial_key(row) for row in completed.values() if row["position"] == position} for position in POSITIONS}
            if keys[POSITIONS[0]] != keys[POSITIONS[1]]: raise RuntimeError("LAT/PANL trial pairing gate failed")
        summary = {"status": "complete", "smoke_only": smoke, "completed_cells": expected, "expected_cells": expected, "new_gpu_forwards": 0, "resumed_noop": True}
        atomic_json(progress_path, summary); return summary
    if len(completed) > expected: raise ValueError("Trial file exceeds configured grid")
    runtime = load_runtime(INFERENCE_PATH); inference = runtime.QwenVLInference(str(MODEL_PATH)); modules = resolve_language_modules(inference.model)
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor); token_ids = class_token_ids(tokenizer); device = model_input_device(inference)
    started = time.time(); new_forwards = 0
    for manifest_row in test:
        case_id = str(manifest_row["case_id"]); clean_row = clean[case_id]
        messages = _messages(manifest_row); rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
        if canonical_hash(rendered) != clean_row["rendered_prompt_hash"]: raise ValueError(f"Rendered prompt changed: {case_id}")
        inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=device); located = locate_phase1_positions(tokenizer, rendered, inputs, str(manifest_row["phase0_raw_answer"]))
        lat = int(located["P1_LAT"]["processed_index"]); panl = int(located["P1_PANL"]["processed_index"]); sac = int(located["P1_SAC"]["processed_index"]); sequence_length = int(inputs.input_ids.shape[1])
        if not (lat < panl < sac): raise ValueError("LAT/PANL/SAC causal order changed")
        if any(int(located[name]["processed_index"]) != int(clean_row["positions"][name]["processed_index"]) for name in (*POSITIONS, "P1_SAC")): raise ValueError("Clean/test token position changed")
        clean_logits = np.asarray(clean_row["class_logits"], dtype=np.float64); clean_probs = np.asarray(clean_row["class_probabilities"], dtype=np.float64); clean_soft = float(clean_row["soft_sa_image_score"]); clean_hard = int(clean_row["argmax_hard_class"]); clean_margin = class_margin(clean_logits, clean_hard)
        fold = int(manifest_row["fold"]); answer = str(manifest_row["test_answer"])
        for position in selected_positions:
            target = int(located[position]["processed_index"])
            for direction in directions:
                for layer in layers:
                    vector_np, vector_row = load_vector(root, vectors, position=position, fold=fold, answer=answer, layer=int(layer), direction=direction); vector = torch.from_numpy(vector_np)
                    for alpha in alphas:
                        proto = {"case_id": case_id, "family_id": manifest_row["family_id"], "item_id": str(manifest_row["item_id"]), "fold": fold, "answer": answer, "test_side": manifest_row["test_side"], "condition": manifest_row["condition"], "test_status": manifest_row["test_status"], "position": position, "direction": direction, "layer": int(layer), "alpha": float(alpha)}
                        if trial_key(proto) in completed: continue
                        try:
                            hook = AdditiveActivationHook(modules, layer_index=int(layer), target_position=target, steering_vector=vector * float(alpha), prefill_sequence_length=sequence_length, injection_site="block_output")
                            with hook: logits = run_logits_forward(inference.model, inputs, [sac], modules)[sac]
                            new_forwards += 1; diagnostics = hook.diagnostics(); scored = soft_sa_from_logits(logits, token_ids)
                            before = hook.h_before.numpy(); after = hook.h_after.numpy(); before_norm = float(np.linalg.norm(before)); after_norm = float(np.linalg.norm(after))
                            cosine = float(np.dot(before, after) / (before_norm * after_norm)); ratio = float(after_norm / before_norm)
                            alpha_zero = None
                            if float(alpha) == 0.0: alpha_zero = validate_alpha_zero(clean_logits=clean_logits, clean_probabilities=clean_probs, clean_soft_sa=clean_soft, clean_hard_class=clean_hard, scored=scored, before=before, after=after, diagnostics=diagnostics)
                            steered_logits = np.asarray(scored["class_logits"], dtype=np.float64); steered_margin = class_margin(steered_logits, clean_hard)
                            finite = bool(np.isfinite(np.concatenate([steered_logits, np.asarray(scored["class_probabilities"]), [scored["soft_sa_image_score"], cosine, ratio]])).all())
                            if not finite or abs(float(scored["probability_sum"]) - 1.0) > 1e-9: raise ValueError("Non-finite steering result")
                            result = {"status": "completed", **proto, "processed_position": target, "answer_token_span": clean_row["phase1_answer_span"], "answer_token_ids": clean_row["phase1_answer_token_ids"], "clean_class_logits": clean_logits.tolist(), "clean_class_probabilities": clean_probs.tolist(), "clean_soft_sa": clean_soft, "clean_hard_class": clean_hard, "steered_class_logits": scored["class_logits"], "steered_class_probabilities": scored["class_probabilities"], "steered_soft_sa": float(scored["soft_sa_image_score"]), "steered_hard_class": int(scored["argmax_hard_class"]), "delta_soft_sa": float(scored["soft_sa_image_score"]) - clean_soft, "hard_class_changed": int(scored["argmax_hard_class"]) != clean_hard, "hard_class_delta": int(scored["argmax_hard_class"]) - clean_hard, "clean_class_margin": clean_margin, "steered_clean_class_margin": steered_margin, "margin_change": steered_margin - clean_margin, "hook_diagnostics": diagnostics, "activation_before_hash": array_hash(before), "activation_after_hash": array_hash(after), "activation_before_norm": before_norm, "activation_after_norm": after_norm, "activation_cosine": cosine, "activation_norm_ratio": ratio, "finite_values": finite, "saturated": float(scored["soft_sa_image_score"]) <= 0.05 + 1e-9 or float(scored["soft_sa_image_score"]) >= 0.95 - 1e-9, "probability_sum": float(scored["probability_sum"]), "alpha_zero_parity": alpha_zero, "vector_fingerprint": vector_row["vector_fingerprint"], "manifest_fingerprints": base["manifest_fingerprints"], "config_fingerprint": run_config["fingerprint"]}
                            append_jsonl(trial_path, result); completed[trial_key(result)] = result
                        except Exception as exc:
                            failure = {"stage": "steering", **proto, "type": type(exc).__name__, "message": str(exc), "timestamp": time.time()}; append_jsonl(root / "progress" / "failures.jsonl", failure); atomic_json(progress_path, {"status": "failed", "completed_cells": len(completed), "expected_cells": expected, "failure": failure}); raise
                        if new_forwards % 10 == 0: atomic_json(progress_path, {"status": "running", "completed_cells": len(completed), "expected_cells": expected, "new_gpu_forwards": new_forwards, "elapsed_seconds": time.time() - started, "last": proto})
    if len(completed) != expected: raise RuntimeError(f"Steering grid incomplete: {len(completed)}/{expected}")
    summary = {"status": "complete", "smoke_only": smoke, "completed_cells": len(completed), "expected_cells": expected, "new_gpu_forwards": new_forwards, "resumed_noop": new_forwards == 0, "elapsed_seconds": time.time() - started}
    if not position_filter:
        keys = {position: {paired_trial_key(row) for row in completed.values() if row["position"] == position} for position in POSITIONS}
        if keys[POSITIONS[0]] != keys[POSITIONS[1]]: raise RuntimeError("LAT/PANL trial pairing gate failed")
    atomic_json(progress_path, summary); return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--position", choices=POSITIONS)
    args = parser.parse_args(argv); root = Path(args.output_root) if args.output_root else RESULTS_ROOT
    if args.smoke and not args.output_root: parser.error("--smoke requires an explicit output root outside formal results")
    print(json.dumps(run_steering(output_root=root, smoke=args.smoke, resume=args.resume, position_filter=args.position), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
