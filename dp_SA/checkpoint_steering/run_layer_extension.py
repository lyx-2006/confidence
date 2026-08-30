from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import (
    AdditiveActivationHook,
    model_input_device,
    resolve_language_modules,
    run_hooked_forward,
    run_logits_forward,
)

from dp_SA.prompts import SA_PREFILL
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits

from .analyze import analyze
from .capture import _messages
from .config import ALPHAS, FLOAT_TOLERANCE, INFERENCE_PATH, MODEL_PATH, RESULTS_ROOT
from .io_utils import (
    append_jsonl,
    array_hash,
    atomic_json,
    atomic_npz,
    canonical_hash,
    load_jsonl,
    sha256_file,
)
from .manifests import load_frozen_manifests, manifest_fingerprint
from .positions import locate_checkpoint_positions
from .run import class_margin, trial_key, validate_alpha_zero
from .vectors import construct_direction


EXTENSION_NAME = "layer_12_16_22"
EXTENSION_POSITIONS = ("P1_LAT", "P1_PANL", "P1_CLASS_LIST_END")
EXTENSION_LAYERS = (12, 16, 22)


def _paths(root: Path) -> dict[str, Path]:
    return {
        "config": root / "progress" / f"config_{EXTENSION_NAME}.json",
        "capture_progress": root / "progress" / f"capture_progress_{EXTENSION_NAME}.json",
        "steering_progress": root / "progress" / f"steering_progress_{EXTENSION_NAME}.json",
        "failures": root / "progress" / f"failures_{EXTENSION_NAME}.jsonl",
        "clean": root / "artifacts" / "diagnostics" / f"clean_capture_{EXTENSION_NAME}.jsonl",
        "trials": root / "artifacts" / "diagnostics" / f"steering_trials_{EXTENSION_NAME}.jsonl",
        "vectors": root / "artifacts" / "vectors" / f"vector_metadata_{EXTENSION_NAME}.json",
    }


def _extension_config(root: Path, construction: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completion_path = root / "progress" / "completion.json"
    if not completion_path.is_file() or json.loads(completion_path.read_text()).get("status") != "complete":
        raise ValueError("Base checkpoint-steering formal run must be complete before extending layers")
    payload = {
        "format_version": 1,
        "extension": EXTENSION_NAME,
        "positions": list(EXTENSION_POSITIONS),
        "layers": list(EXTENSION_LAYERS),
        "alphas": list(ALPHAS),
        "construction_fingerprint": manifest_fingerprint(construction),
        "test_fingerprint": manifest_fingerprint(test),
        "base_completion_sha256": sha256_file(completion_path),
        "source_sha256": sha256_file(Path(__file__)),
        "expected_clean_forwards": len({str(row["case_id"]) for row in [*construction, *test]}),
        "expected_steering_forwards": len(test) * len(EXTENSION_POSITIONS) * len(EXTENSION_LAYERS) * len(ALPHAS),
    }
    payload["fingerprint"] = canonical_hash(payload)
    return payload


def _check_config(path: Path, config: dict[str, Any], *, resume: bool) -> None:
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("fingerprint") != config["fingerprint"]:
            raise ValueError("Layer-extension config fingerprint mismatch")
        if not resume:
            raise FileExistsError(f"Layer extension already exists; use --resume: {path}")
    else:
        atomic_json(path, config)


def _release_model(inference: Any) -> None:
    del inference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _capture(
    root: Path,
    construction: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _paths(root)
    existing_rows = [row for row in load_jsonl(paths["clean"]) if row.get("status") == "completed"]
    existing = {str(row["case_id"]): row for row in existing_rows}
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in [*construction, *test]:
        case_id = str(row["case_id"])
        if case_id not in seen:
            ordered.append(row)
            seen.add(case_id)
    for row in existing.values():
        if row.get("config_fingerprint") != config["fingerprint"]:
            raise ValueError(f"Extension clean config mismatch: {row['case_id']}")
        hidden = root / row["hidden_file"]
        if not hidden.is_file() or sha256_file(hidden) != row["hidden_sha256"]:
            raise ValueError(f"Extension hidden artifact changed: {row['case_id']}")
    if set(existing) == seen:
        summary = {"status": "complete", "completed": len(existing), "expected": len(ordered), "new_gpu_forwards": 0, "resumed_noop": True}
        atomic_json(paths["capture_progress"], summary)
        return list(existing.values()), summary

    base_clean = {
        str(row["case_id"]): row
        for row in load_jsonl(root / "artifacts" / "diagnostics" / "clean_capture.jsonl")
        if row.get("status") == "completed"
    }
    runtime = load_runtime(INFERENCE_PATH)
    inference = runtime.QwenVLInference(str(MODEL_PATH))
    modules = resolve_language_modules(inference.model)
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    class_ids = class_token_ids(tokenizer)
    device = model_input_device(inference)
    started = time.time()
    new_forwards = 0
    try:
        for manifest_row in ordered:
            case_id = str(manifest_row["case_id"])
            if case_id in existing:
                continue
            try:
                base = base_clean[case_id]
                messages = _messages(manifest_row)
                rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
                if canonical_hash(rendered) != base["rendered_prompt_hash"]:
                    raise ValueError("Rendered prompt differs from base clean capture")
                inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=device)
                located = locate_checkpoint_positions(tokenizer, rendered, inputs, str(manifest_row["phase0_raw_answer"]))
                for name in (*EXTENSION_POSITIONS, "P1_SAC"):
                    if int(located[name]["processed_index"]) != int(base["positions"][name]["processed_index"]):
                        raise ValueError(f"Position parity failed at {name}")
                positions = {name: int(located[name]["processed_index"]) for name in EXTENSION_POSITIONS}
                sac = int(located["P1_SAC"]["processed_index"])
                forward = run_hooked_forward(inference.model, inputs, modules, positions, logits_positions=[sac])
                new_forwards += 1
                score = soft_sa_from_logits(forward.logits_by_position[sac], class_ids)
                logit_error = float(np.max(np.abs(np.asarray(score["class_logits"]) - np.asarray(base["class_logits"]))))
                probability_error = float(np.max(np.abs(np.asarray(score["class_probabilities"]) - np.asarray(base["class_probabilities"]))))
                soft_error = abs(float(score["soft_sa_image_score"]) - float(base["soft_sa_image_score"]))
                if logit_error > FLOAT_TOLERANCE or probability_error > FLOAT_TOLERANCE or soft_error > FLOAT_TOLERANCE or int(score["argmax_hard_class"]) != int(base["argmax_hard_class"]):
                    raise ValueError(f"Extension clean parity failed: logits={logit_error}, probs={probability_error}, soft={soft_error}")
                arrays = {
                    f"{position}__L{layer}": forward.hidden_by_name[position][layer].detach().float().cpu().numpy().astype(np.float16)
                    for position in EXTENSION_POSITIONS for layer in EXTENSION_LAYERS
                }
                relative = Path("artifacts") / "hidden" / f"{EXTENSION_NAME}__{case_id}.npz"
                hidden_path = root / relative
                atomic_npz(hidden_path, arrays)
                result = {
                    "status": "completed", "case_id": case_id, "item_id": str(manifest_row["item_id"]),
                    "test_side": manifest_row.get("test_side"), "construction_side": manifest_row.get("construction_side"),
                    "rendered_prompt_hash": canonical_hash(rendered), "phase0_raw_answer": manifest_row["phase0_raw_answer"],
                    "positions": located, **score, "hidden_file": str(relative), "hidden_sha256": sha256_file(hidden_path),
                    "hidden_keys": sorted(arrays), "hidden_dtype": "float16", "parity": {
                        "status": "passed", "logit_max_abs_error": logit_error,
                        "probability_max_abs_error": probability_error, "soft_sa_abs_error": soft_error,
                    }, "config_fingerprint": config["fingerprint"],
                }
                append_jsonl(paths["clean"], result)
                existing[case_id] = result
            except Exception as exc:
                append_jsonl(paths["failures"], {"stage": "capture", "case_id": case_id, "type": type(exc).__name__, "message": str(exc), "timestamp": time.time()})
                raise
            atomic_json(paths["capture_progress"], {"status": "running", "completed": len(existing), "expected": len(ordered), "new_gpu_forwards": new_forwards, "elapsed_seconds": time.time() - started, "last_case_id": case_id})
    finally:
        _release_model(inference)
    summary = {"status": "complete", "completed": len(existing), "expected": len(ordered), "new_gpu_forwards": new_forwards, "resumed_noop": new_forwards == 0, "elapsed_seconds": time.time() - started}
    atomic_json(paths["capture_progress"], summary)
    return list(existing.values()), summary


def _vectors(
    root: Path,
    construction: Sequence[dict[str, Any]],
    clean_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    resume: bool,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    path = _paths(root)["vectors"]
    if path.exists():
        if not resume:
            raise FileExistsError("Extension vectors exist; use --resume")
        metadata = json.loads(path.read_text())
        if metadata.get("config_fingerprint") != config["fingerprint"]:
            raise ValueError("Extension vector config mismatch")
        output = {}
        for cell in metadata["vectors"]:
            vector_path = root / cell["vector_file"]
            if sha256_file(vector_path) != cell["file_sha256"]:
                raise ValueError(f"Extension vector changed: {vector_path}")
            with np.load(vector_path) as payload:
                output[(cell["position"], int(cell["layer"]))] = np.asarray(payload["scaled_vector"], dtype=np.float32)
        return output, metadata
    clean = {str(row["case_id"]): row for row in clean_rows}
    high = [row for row in construction if row["construction_side"] == "high_image"]
    low = [row for row in construction if row["construction_side"] == "high_text"]
    output: dict[tuple[str, int], np.ndarray] = {}
    cells = []
    for position in EXTENSION_POSITIONS:
        for layer in EXTENSION_LAYERS:
            def read(row: dict[str, Any]) -> np.ndarray:
                with np.load(root / clean[str(row["case_id"])]["hidden_file"]) as payload:
                    return np.asarray(payload[f"{position}__L{layer}"], dtype=np.float32)
            arrays, metrics = construct_direction(np.stack([read(row) for row in high]), np.stack([read(row) for row in low]))
            relative = Path("artifacts") / "vectors" / f"{position}__L{layer}__{EXTENSION_NAME}.npz"
            vector_path = root / relative
            atomic_npz(vector_path, arrays)
            output[(position, layer)] = arrays["scaled_vector"]
            cells.append({"position": position, "layer": layer, "vector_file": str(relative), "file_sha256": sha256_file(vector_path), **metrics})
    metadata = {"format_version": 1, "extension": EXTENSION_NAME, "config_fingerprint": config["fingerprint"], "vectors": cells}
    metadata["fingerprint"] = canonical_hash(metadata)
    atomic_json(path, metadata)
    return output, metadata


def _steer(
    root: Path,
    test: Sequence[dict[str, Any]],
    clean_rows: Sequence[dict[str, Any]],
    vectors: dict[tuple[str, int], np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    paths = _paths(root)
    existing_rows = [row for row in load_jsonl(paths["trials"]) if row.get("status") == "completed"]
    existing: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        if row.get("config_fingerprint") != config["fingerprint"]:
            raise ValueError(f"Extension trial config mismatch: {trial_key(row)}")
        key = trial_key(row)
        if key in existing:
            raise ValueError(f"Duplicate extension trial: {key}")
        existing[key] = row
    expected = int(config["expected_steering_forwards"])
    if len(existing) == expected:
        summary = {"status": "complete", "completed_cells": expected, "expected_cells": expected, "new_gpu_forwards": 0, "resumed_noop": True}
        atomic_json(paths["steering_progress"], summary)
        return summary
    clean = {str(row["case_id"]): row for row in clean_rows}
    runtime = load_runtime(INFERENCE_PATH)
    inference = runtime.QwenVLInference(str(MODEL_PATH))
    modules = resolve_language_modules(inference.model)
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    class_ids = class_token_ids(tokenizer)
    device = model_input_device(inference)
    started = time.time()
    new_forwards = 0
    try:
        for manifest_row in test:
            case_id = str(manifest_row["case_id"])
            baseline = clean[case_id]
            messages = _messages(manifest_row)
            rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
            if canonical_hash(rendered) != baseline["rendered_prompt_hash"]:
                raise ValueError(f"Extension rendered prompt changed: {case_id}")
            inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=device)
            located = locate_checkpoint_positions(tokenizer, rendered, inputs, str(manifest_row["phase0_raw_answer"]))
            sac = int(located["P1_SAC"]["processed_index"])
            sequence_length = int(inputs.input_ids.shape[1])
            clean_logits = np.asarray(baseline["class_logits"], dtype=float)
            clean_probabilities = np.asarray(baseline["class_probabilities"], dtype=float)
            clean_soft = float(baseline["soft_sa_image_score"])
            clean_hard = int(baseline["argmax_hard_class"])
            clean_margin = class_margin(clean_logits, clean_hard)
            for position in EXTENSION_POSITIONS:
                target = int(located[position]["processed_index"])
                for layer in EXTENSION_LAYERS:
                    vector = torch.from_numpy(vectors[(position, layer)])
                    for alpha in ALPHAS:
                        proto = {"case_id": case_id, "item_id": str(manifest_row["item_id"]), "test_side": str(manifest_row["test_side"]), "position": position, "layer": layer, "alpha": float(alpha)}
                        if trial_key(proto) in existing:
                            continue
                        try:
                            hook = AdditiveActivationHook(modules, layer_index=layer, target_position=target, steering_vector=vector * float(alpha), prefill_sequence_length=sequence_length, injection_site="block_output")
                            with hook:
                                logits = run_logits_forward(inference.model, inputs, [sac], modules)[sac]
                            new_forwards += 1
                            diagnostics = hook.diagnostics()
                            scored = soft_sa_from_logits(logits, class_ids)
                            before = hook.h_before.numpy()
                            after = hook.h_after.numpy()
                            before_norm = float(np.linalg.norm(before)); after_norm = float(np.linalg.norm(after))
                            cosine = float(np.dot(before, after) / (before_norm * after_norm)); ratio = float(after_norm / before_norm)
                            steered_logits = np.asarray(scored["class_logits"], dtype=float)
                            steered_margin = class_margin(steered_logits, clean_hard)
                            alpha_zero = None
                            if float(alpha) == 0.0:
                                alpha_zero = validate_alpha_zero(clean_logits=clean_logits, clean_probabilities=clean_probabilities, clean_soft_sa=clean_soft, clean_hard_class=clean_hard, scored=scored, before=before, after=after, diagnostics=diagnostics)
                            result = {
                                "status": "completed", **proto, "processed_position": target,
                                "clean_soft_sa": clean_soft, "steered_soft_sa": float(scored["soft_sa_image_score"]), "delta_soft_sa": float(scored["soft_sa_image_score"]) - clean_soft,
                                "clean_hard_class": clean_hard, "steered_hard_class": int(scored["argmax_hard_class"]),
                                "hard_class_changed": int(scored["argmax_hard_class"]) != clean_hard, "hard_class_delta": int(scored["argmax_hard_class"]) - clean_hard,
                                "clean_class_logits": clean_logits.tolist(), "clean_class_probabilities": clean_probabilities.tolist(),
                                "steered_class_logits": scored["class_logits"], "steered_class_probabilities": scored["class_probabilities"], "probability_sum": float(scored["probability_sum"]),
                                "clean_class_margin": clean_margin, "steered_clean_class_margin": steered_margin, "margin_change": steered_margin - clean_margin,
                                "saturated": float(scored["soft_sa_image_score"]) <= 0.05 + 1e-9 or float(scored["soft_sa_image_score"]) >= 0.95 - 1e-9,
                                "finite_values": True, "activation_cosine": cosine, "activation_norm_ratio": ratio,
                                "hook_diagnostics": diagnostics, "activation_before_hash": array_hash(before), "activation_after_hash": array_hash(after),
                                "alpha_zero_parity": alpha_zero, "config_fingerprint": config["fingerprint"], "extension": EXTENSION_NAME,
                            }
                            append_jsonl(paths["trials"], result)
                            existing[trial_key(result)] = result
                        except Exception as exc:
                            append_jsonl(paths["failures"], {"stage": "steering", **proto, "type": type(exc).__name__, "message": str(exc), "timestamp": time.time()})
                            raise
                        if new_forwards % 10 == 0:
                            atomic_json(paths["steering_progress"], {"status": "running", "completed_cells": len(existing), "expected_cells": expected, "new_gpu_forwards": new_forwards, "elapsed_seconds": time.time() - started, "last": proto})
    finally:
        _release_model(inference)
    if len(existing) != expected:
        raise RuntimeError(f"Extension steering grid incomplete: {len(existing)}/{expected}")
    summary = {"status": "complete", "completed_cells": len(existing), "expected_cells": expected, "new_gpu_forwards": new_forwards, "resumed_noop": new_forwards == 0, "elapsed_seconds": time.time() - started}
    atomic_json(paths["steering_progress"], summary)
    return summary


def run_layer_extension(*, output_root: Path = RESULTS_ROOT, resume: bool = False) -> dict[str, Any]:
    root = Path(output_root)
    construction, test, _source = load_frozen_manifests(root)
    config = _extension_config(root, construction, test)
    paths = _paths(root)
    _check_config(paths["config"], config, resume=resume)
    try:
        clean_rows, capture_summary = _capture(root, construction, test, config, resume=resume)
        vectors, vector_metadata = _vectors(root, construction, clean_rows, config, resume=resume)
        steering_summary = _steer(root, test, clean_rows, vectors, config)
        analysis_summary = analyze(output_root=root, smoke=False, repeats=2000, refresh=True)
        completion = {"status": "complete", "extension": EXTENSION_NAME, "capture": capture_summary, "steering": steering_summary, "vector_fingerprint": vector_metadata["fingerprint"], "analysis": analysis_summary}
        atomic_json(root / "progress" / f"completion_{EXTENSION_NAME}.json", completion)
        return completion
    except Exception as exc:
        append_jsonl(paths["failures"], {"stage": "pipeline", "type": type(exc).__name__, "message": str(exc), "timestamp": time.time()})
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add L12/L16/L22 for LAT, PANL, and CLASS_LIST_END to the completed formal results")
    parser.add_argument("--output-root", default=str(RESULTS_ROOT))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run_layer_extension(output_root=Path(args.output_root), resume=args.resume), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
