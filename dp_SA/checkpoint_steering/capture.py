from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_hooked_forward

from dp_SA.prompts import SA_PREFILL
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits

from .config import (
    FLOAT_TOLERANCE,
    HISTORICAL_ROOT,
    INFERENCE_PATH,
    LAYERS,
    MODEL_PATH,
    POSITIONS,
    RESULTS_ROOT,
    SMOKE_LAYERS,
)
from .fingerprint import check_or_write_config, experiment_config
from .io_utils import append_jsonl, atomic_json, atomic_jsonl, atomic_npz, canonical_hash, load_jsonl, sha256_file
from .manifests import prepare_manifests
from .positions import locate_checkpoint_positions


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [
            {"type": "image", "image": str(Path(row["image_path"]).resolve())},
            {"type": "text", "text": str(row["phase1_prompt"])},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]},
    ]


def _parity(
    historical: dict[str, Any],
    located: dict[str, Any],
    score: dict[str, Any],
    new_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    if historical["phase1_inserted_raw_answer"] != historical["phase0_raw_answer"]:
        raise ValueError("Historical raw fixed answer is inconsistent")
    if located["phase1_answer_token_ids"] != historical["phase1_answer_token_ids"]:
        raise ValueError("Phase-1 answer token IDs changed")
    for name in ("P1_LAT", "P1_PANL", "P1_SAC"):
        old = historical["positions"][name]
        new = located[name]
        if int(old["processed_index"]) != int(new["processed_index"]) or int(old["token_id"]) != int(new["token_id"]):
            raise ValueError(f"Historical position parity failed at {name}")
    old_logits = np.asarray(historical["class_logits"], dtype=np.float64)
    old_probs = np.asarray(historical["class_probabilities"], dtype=np.float64)
    new_logits = np.asarray(score["class_logits"], dtype=np.float64)
    new_probs = np.asarray(score["class_probabilities"], dtype=np.float64)
    logit_error = float(np.max(np.abs(new_logits - old_logits)))
    probability_error = float(np.max(np.abs(new_probs - old_probs)))
    soft_error = abs(float(score["soft_sa_image_score"]) - float(historical["soft_sa_image_score"]))
    if logit_error > FLOAT_TOLERANCE or probability_error > FLOAT_TOLERANCE or soft_error > FLOAT_TOLERANCE:
        raise ValueError(f"Clean score parity failed: logits={logit_error}, probs={probability_error}, soft={soft_error}")
    if int(score["argmax_hard_class"]) != int(historical["argmax_hard_class"]):
        raise ValueError("Clean hard-class parity failed")

    hidden_cells = 0
    hidden_max_error = 0.0
    historical_path = HISTORICAL_ROOT / historical["hidden_file"]
    with np.load(historical_path) as old_hidden:
        overlap_layers = [
            layer
            for layer in (10, 14, 18, 20, 24, 26)
            if f"P1_PANL__L{layer}" in new_arrays and f"P1_PANL__L{layer}" in old_hidden
        ]
        for layer in overlap_layers:
            key = f"P1_PANL__L{layer}"
            old_value = np.asarray(old_hidden[key], dtype=np.float16)
            new_value = np.asarray(new_arrays[key], dtype=np.float16)
            error = float(np.max(np.abs(old_value.astype(np.float32) - new_value.astype(np.float32))))
            hidden_max_error = max(hidden_max_error, error)
            hidden_cells += 1
            if not np.array_equal(old_value, new_value):
                raise ValueError(f"PANL hidden parity failed at {key}: max error {error}")
    return {
        "status": "passed",
        "logit_max_abs_error": logit_error,
        "probability_max_abs_error": probability_error,
        "soft_sa_abs_error": soft_error,
        "hard_class_equal": True,
        "panl_hidden_bitwise_cells": hidden_cells,
        "panl_hidden_max_abs_error": hidden_max_error,
    }


def _validate_completed(root: Path, row: dict[str, Any], config_fingerprint: str) -> None:
    if row.get("config_fingerprint") != config_fingerprint:
        raise ValueError(f"Clean capture config mismatch for {row.get('case_id')}")
    hidden = root / row["hidden_file"]
    if not hidden.is_file() or sha256_file(hidden) != row["hidden_sha256"]:
        raise ValueError(f"Clean hidden artifact changed for {row.get('case_id')}")


def run_capture(*, output_root: Path = RESULTS_ROOT, smoke: bool = False, resume: bool = False) -> dict[str, Any]:
    root = Path(output_root)
    failure_path = root / "progress" / "failures.jsonl"
    if not failure_path.exists():
        atomic_jsonl(failure_path, [])
    construction, test, selection = prepare_manifests(root, smoke=smoke, resume=resume)
    config = experiment_config(construction, test, smoke=smoke)
    config_path = root / "progress" / "capture_config.json"
    check_or_write_config(config_path, config, resume=resume)
    clean_path = root / "artifacts" / "diagnostics" / "clean_capture.jsonl"
    existing_rows = [row for row in load_jsonl(clean_path) if row.get("status") == "completed"]
    existing = {str(row["case_id"]): row for row in existing_rows}
    for row in existing.values():
        _validate_completed(root, row, config["fingerprint"])

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in [*construction, *test]:
        case_id = str(row["case_id"])
        if case_id not in seen:
            ordered.append(row)
            seen.add(case_id)
    if set(existing) == seen:
        summary = {
            "status": "complete", "smoke": smoke, "completed": len(existing), "expected": len(ordered),
            "new_gpu_forwards": 0, "resumed_noop": True, "selection": selection,
        }
        atomic_json(root / "progress" / "capture_progress.json", summary)
        return summary

    historical_by_case = {
        str(row["case_id"]): row
        for row in load_jsonl(HISTORICAL_ROOT / "capture" / "results.jsonl")
        if row.get("status") == "completed"
    }
    missing_history = seen.difference(historical_by_case)
    if missing_history:
        raise ValueError(f"Frozen cases are absent from historical capture: {sorted(missing_history)}")

    runtime = load_runtime(INFERENCE_PATH)
    inference = runtime.QwenVLInference(str(MODEL_PATH))
    modules = resolve_language_modules(inference.model)
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    token_ids = class_token_ids(tokenizer)
    device = model_input_device(inference)
    layers = SMOKE_LAYERS if smoke else LAYERS
    new_forwards = 0
    started = time.time()
    for ordinal, manifest_row in enumerate(ordered, 1):
        case_id = str(manifest_row["case_id"])
        if case_id in existing:
            continue
        try:
            historical = historical_by_case[case_id]
            if str(manifest_row["phase0_raw_answer"]) != str(historical["phase0_raw_answer"]):
                raise ValueError("Frozen raw fixed answer differs from historical capture")
            if canonical_hash(manifest_row["phase1_prompt"]) != historical["phase1_prompt_hash"]:
                raise ValueError("Frozen Phase-1 prompt hash differs from historical capture")
            messages = _messages(manifest_row)
            rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
            inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=device)
            located = locate_checkpoint_positions(tokenizer, rendered, inputs, str(manifest_row["phase0_raw_answer"]))
            positions = {name: int(located[name]["processed_index"]) for name in POSITIONS}
            sac = int(located["P1_SAC"]["processed_index"])
            forward = run_hooked_forward(inference.model, inputs, modules, positions, logits_positions=[sac])
            new_forwards += 1
            score = soft_sa_from_logits(forward.logits_by_position[sac], token_ids)
            arrays = {
                f"{position}__L{layer}": forward.hidden_by_name[position][layer].detach().float().cpu().numpy().astype(np.float16)
                for position in POSITIONS for layer in layers
            }
            parity = _parity(historical, located, score, arrays)
            relative = Path("artifacts") / "hidden" / f"{case_id}.npz"
            hidden_path = root / relative
            atomic_npz(hidden_path, arrays)
            result = {
                "status": "completed",
                "case_id": case_id,
                "item_id": str(manifest_row["item_id"]),
                "test_side": manifest_row.get("test_side"),
                "construction_side": manifest_row.get("construction_side"),
                "phase0_raw_answer": manifest_row["phase0_raw_answer"],
                "phase1_prompt_hash": canonical_hash(manifest_row["phase1_prompt"]),
                "rendered_prompt_hash": canonical_hash(rendered),
                "phase1_answer_token_ids": located["phase1_answer_token_ids"],
                "positions": located,
                **score,
                "hidden_file": str(relative),
                "hidden_keys": sorted(arrays),
                "hidden_sha256": sha256_file(hidden_path),
                "hidden_dtype": "float16",
                "hidden_definition": config["hidden_definition"],
                "parity": parity,
                "config_fingerprint": config["fingerprint"],
            }
            append_jsonl(clean_path, result)
            existing[case_id] = result
        except Exception as exc:
            failure = {"stage": "capture", "case_id": case_id, "type": type(exc).__name__, "message": str(exc), "timestamp": time.time()}
            append_jsonl(root / "progress" / "failures.jsonl", failure)
            atomic_json(root / "progress" / "capture_progress.json", {"status": "failed", "completed": len(existing), "expected": len(ordered), "failure": failure})
            raise
        atomic_json(root / "progress" / "capture_progress.json", {
            "status": "running", "completed": len(existing), "expected": len(ordered), "new_gpu_forwards": new_forwards,
            "elapsed_seconds": time.time() - started, "last_case_id": case_id,
        })
    summary = {
        "status": "complete", "smoke": smoke, "completed": len(existing), "expected": len(ordered),
        "new_gpu_forwards": new_forwards, "resumed_noop": new_forwards == 0, "elapsed_seconds": time.time() - started,
        "selection": selection,
    }
    atomic_json(root / "progress" / "capture_progress.json", summary)
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
    print(json.dumps(run_capture(output_root=root, smoke=args.smoke, resume=args.resume), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
