from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import AdditiveActivationHook, run_hooked_forward, run_logits_forward

from dp_SA.SA_probe.config import HIDDEN_SIZE, MAX_PIXELS, MIN_PIXELS, TRAIN_MANIFEST
from dp_SA.SA_probe.positions import locate_probe_positions
from dp_SA.SA_probe.runtime import load_inference, messages
from dp_SA.config import HIDDEN_DEFINITION, SEED, VECTOR_NORM_FRACTION
from dp_SA.prompts import PHASE1_TEMPLATE, SA_PREFILL
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits

from .analyze import analyze
from .config import HISTORICAL_CONSTRUCTION, HISTORICAL_TEST
from .io_utils import array_hash, atomic_json, atomic_jsonl, atomic_npz, canonical_hash, load_jsonl, sha256_file
from .manifests import manifest_fingerprint, prepare_manifests
from .run import class_margin, trial_key, validate_alpha_zero
from .vectors import construct_direction


PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "output" / "PANL2CLE_position"
SOURCE_CAPTURE_ROOT = PACKAGE_ROOT.parent / "SA_probe" / "output"
SOURCE_CAPTURE = SOURCE_CAPTURE_ROOT / "artifacts" / "diagnostics" / "capture_manifest.jsonl"

POSITIONS = (
    "P1_ATTRIBUTION_QUERY_END_NL",
    "P1_INTEGER_INSTRUCTION_END_NL",
    "P1_IMAGE_POLARITY_SENTENCE_END",
    "P1_SCALE_LINE_END_NL",
    "P1_CLASS4_RULE_END_NL",
)
LAYERS = (16, 18, 20)
ALPHAS = (-10.0, -2.0, 0.0, 2.0, 10.0)
SMOKE_ALPHAS = (-2.0, 0.0, 2.0)
BOOTSTRAP_REPEATS = 2000
SMOKE_BOOTSTRAP_REPEATS = 200
FLOAT_ATOL = 1e-6
EXPECTED_FORMAL_IMPORTS = 105
EXPECTED_FORMAL_CAPTURES = 45


def hidden_key(position: str, layer: int) -> str:
    return f"{position}__L{layer}"


def expected_hidden_keys() -> set[str]:
    return {hidden_key(position, layer) for position in POSITIONS for layer in LAYERS}


def _release(inference: Any) -> None:
    del inference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _config(root: Path, construction: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]], *, smoke: bool) -> dict[str, Any]:
    source_files = [Path(__file__), Path(locate_probe_positions.__code__.co_filename), Path(load_inference.__code__.co_filename)]
    payload = {
        "format_version": 1,
        "experiment": "PANL2CLE_position",
        "smoke": smoke,
        "positions": list(POSITIONS),
        "layers": list(LAYERS),
        "alphas": list(SMOKE_ALPHAS if smoke else ALPHAS),
        "seed": SEED,
        "vector_definition": "mean(high_image)-mean(high_text)",
        "vector_norm_fraction": VECTOR_NORM_FRACTION,
        "hidden_definition": HIDDEN_DEFINITION,
        "processor": {"class": "Qwen2VLImageProcessorFast", "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        "construction_fingerprint": manifest_fingerprint(construction),
        "test_fingerprint": manifest_fingerprint(test),
        "source_inputs": {
            "construction_sha256": sha256_file(HISTORICAL_CONSTRUCTION),
            "test_sha256": sha256_file(HISTORICAL_TEST),
            "probe_manifest_sha256": sha256_file(TRAIN_MANIFEST),
            "probe_capture_sha256": sha256_file(SOURCE_CAPTURE),
            "phase1_template": canonical_hash(PHASE1_TEMPLATE),
            "sa_prefill": canonical_hash(SA_PREFILL),
        },
        "source_code": {path.name: sha256_file(path) for path in source_files},
        "expected_clean_cases": len({str(row["case_id"]) for row in [*construction, *test]}),
        "expected_trials": len(test) * len(POSITIONS) * len(LAYERS) * len(SMOKE_ALPHAS if smoke else ALPHAS),
    }
    payload["fingerprint"] = canonical_hash(payload)
    return payload


def _check_config(root: Path, config: dict[str, Any], *, resume: bool) -> None:
    path = root / "progress" / "config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("fingerprint") != config["fingerprint"]:
            raise ValueError("PANL2CLE config fingerprint mismatch")
        if not resume:
            raise FileExistsError(f"Results already exist; use --resume: {root}")
    else:
        atomic_json(path, config)


def _score_from_probe(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "class_token_ids": row["class_token_ids"],
        "class_logits": row["class_logits"],
        "class_probabilities": row["class_probabilities"],
        "soft_sa_image_score": row["final_soft_sa"],
        "argmax_hard_class": row["final_hard_class"],
    }


def _validate_score(score: dict[str, Any], expected: dict[str, Any], case_id: str) -> dict[str, Any]:
    logits = float(np.max(np.abs(np.asarray(score["class_logits"], dtype=float) - np.asarray(expected["class_logits"], dtype=float))))
    probabilities = float(np.max(np.abs(np.asarray(score["class_probabilities"], dtype=float) - np.asarray(expected["class_probabilities"], dtype=float))))
    soft = abs(float(score["soft_sa_image_score"]) - float(expected["soft_sa_image_score"]))
    hard = int(score["argmax_hard_class"]) == int(expected["argmax_hard_class"])
    passed = logits <= FLOAT_ATOL and probabilities <= FLOAT_ATOL and soft <= FLOAT_ATOL and hard
    result = {"passed": passed, "logits_max_abs_error": logits, "probabilities_max_abs_error": probabilities, "soft_sa_abs_error": soft, "hard_sa_equal": hard}
    if not passed:
        raise ValueError(f"Historical clean parity failed for {case_id}: {result}")
    return result


def _validate_probe_identity(target: dict[str, Any], source_manifest: dict[str, Any], source_capture: dict[str, Any]) -> None:
    case_id = str(target["case_id"])
    prompt_sha = hashlib.sha256(str(target["phase1_prompt"]).encode()).hexdigest()
    image_sha = sha256_file(Path(target["image_path"]))
    checks = {
        "case_id": case_id == str(source_manifest["case_id"]) == str(source_capture["case_id"]),
        "item_id": str(target["item_id"]) == str(source_manifest["item_id"]) == str(source_capture["item_id"]),
        "phase0_raw_answer": str(target["phase0_raw_answer"]) == str(source_manifest["phase0_raw_answer"]),
        "phase1_prompt": (
            str(target["phase1_prompt"]) == str(source_manifest["phase1_prompt"])
            and str(target["phase1_prompt_hash"]) == str(source_manifest["phase1_prompt_hash"])
            and prompt_sha == str(source_capture["phase1_prompt_sha256"])
        ),
        "image": image_sha == str(source_manifest["image_sha256"]),
        "hidden_definition": source_capture.get("hidden_definition") == HIDDEN_DEFINITION,
    }
    if not all(checks.values()):
        raise ValueError(f"SA_probe identity mismatch for {case_id}: {checks}")


def _clean_record(
    root: Path,
    target: dict[str, Any],
    arrays: dict[str, np.ndarray],
    positions: dict[str, Any],
    score: dict[str, Any],
    config: dict[str, Any],
    *,
    source: str,
    rendered_sha: str,
    processor: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(target["case_id"])
    if set(arrays) != expected_hidden_keys() or any(value.shape != (HIDDEN_SIZE,) or value.dtype != np.float16 or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"Invalid 15-key hidden payload for {case_id}")
    relative = Path("artifacts") / "hidden" / f"{case_id}.npz"
    atomic_npz(root / relative, arrays)
    parity = _validate_score(score, target, case_id)
    selected_positions = {name: positions[name] for name in (*POSITIONS, "P1_PANL", "P1_CLASS_LIST_END", "P1_SAC")}
    return {
        "status": "completed", "case_id": case_id, "item_id": str(target["item_id"]),
        "construction_side": target.get("construction_side"), "test_side": target.get("test_side"),
        "capture_source": source, "hidden_file": str(relative), "hidden_sha256": sha256_file(root / relative),
        "hidden_tensor_sha256": {key: array_hash(value) for key, value in arrays.items()},
        "hidden_keys": sorted(arrays), "hidden_definition": HIDDEN_DEFINITION,
        "positions": selected_positions,
        "ordered_processed_indices": positions.get("ordered_processed_indices", {}),
        "rendered_prompt_sha256": rendered_sha,
        "phase1_prompt_sha256": hashlib.sha256(str(target["phase1_prompt"]).encode()).hexdigest(),
        "image_sha256": sha256_file(Path(target["image_path"])),
        "class_token_ids": [int(value) for value in score["class_token_ids"]],
        "class_logits": [float(value) for value in score["class_logits"]],
        "class_probabilities": [float(value) for value in score["class_probabilities"]],
        "soft_sa_image_score": float(score["soft_sa_image_score"]),
        "argmax_hard_class": int(score["argmax_hard_class"]),
        "historical_parity": parity, "processor": processor, "config_fingerprint": config["fingerprint"],
    }


def _load_existing_clean(root: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(root / "artifacts" / "diagnostics" / "clean_capture.jsonl"):
        if row.get("status") != "completed":
            continue
        if row.get("config_fingerprint") != config["fingerprint"]:
            raise ValueError(f"Clean config mismatch: {row.get('case_id')}")
        path = root / row["hidden_file"]
        if not path.is_file() or sha256_file(path) != row["hidden_sha256"]:
            raise ValueError(f"Clean hidden hash mismatch: {path}")
        with np.load(path) as payload:
            if set(payload.files) != expected_hidden_keys():
                raise ValueError(f"Clean hidden key mismatch: {path}")
        case_id = str(row["case_id"])
        if case_id in output:
            raise ValueError(f"Duplicate clean case: {case_id}")
        output[case_id] = row
    return output


def capture_clean(root: Path, construction: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = [*construction, *test]
    expected_ids = {str(row["case_id"]) for row in records}
    existing = _load_existing_clean(root, config)
    if set(existing) == expected_ids:
        result = {"status": "complete", "case_count": len(existing), "imported_cases": 0, "new_gpu_forwards": 0, "resumed_noop": True}
        atomic_json(root / "progress" / "capture.json", result)
        return list(existing.values()), result

    probe_capture = {str(row["case_id"]): row for row in load_jsonl(SOURCE_CAPTURE) if row.get("status") == "completed"}
    probe_manifest = {str(row["case_id"]): row for row in load_jsonl(TRAIN_MANIFEST) if row.get("status") == "completed"}
    imported = 0
    forwards = 0
    started = time.time()
    pending_gpu: list[dict[str, Any]] = []
    for target in records:
        case_id = str(target["case_id"])
        if case_id in existing:
            continue
        if case_id not in probe_capture or case_id not in probe_manifest:
            pending_gpu.append(target)
            continue
        source = probe_capture[case_id]
        _validate_probe_identity(target, probe_manifest[case_id], source)
        source_path = SOURCE_CAPTURE_ROOT / source["hidden_file"]
        if sha256_file(source_path) != source["hidden_sha256"]:
            raise ValueError(f"SA_probe source hidden hash mismatch: {source_path}")
        with np.load(source_path) as payload:
            arrays = {key: np.asarray(payload[key], dtype=np.float16) for key in sorted(expected_hidden_keys())}
        row = _clean_record(root, target, arrays, source["positions"], _score_from_probe(source), config, source="SA_probe_import", rendered_sha=source["rendered_prompt_sha256"], processor=source["processor"])
        existing[case_id] = row
        imported += 1
        atomic_jsonl(root / "artifacts" / "diagnostics" / "clean_capture.jsonl", sorted(existing.values(), key=lambda value: str(value["case_id"])))

    inference = None
    try:
        if pending_gpu:
            inference, modules, tokenizer, device, processor = load_inference()
            if modules.hidden_size != HIDDEN_SIZE or any(layer >= modules.num_hidden_layers for layer in LAYERS):
                raise ValueError("Model hidden size/layer count changed")
            ids = class_token_ids(tokenizer)
            for target in pending_gpu:
                case_id = str(target["case_id"])
                wire = messages(target)
                rendered = render_continued_assistant(inference.processor, wire, SA_PREFILL)
                inputs = prepare_multimodal_inputs(inference.processor, wire, rendered, device=device)
                located = locate_probe_positions(tokenizer, rendered, inputs, str(target["phase0_raw_answer"]))
                position_indices = {name: int(located[name]["processed_index"]) for name in POSITIONS}
                sac = int(located["P1_SAC"]["processed_index"])
                forward = run_hooked_forward(inference.model, inputs, modules, position_indices, logits_positions=[sac])
                forwards += 1
                score = soft_sa_from_logits(forward.logits_by_position[sac], ids)
                arrays = {hidden_key(position, layer): forward.hidden_by_name[position][layer].detach().float().cpu().numpy().astype(np.float16) for position in POSITIONS for layer in LAYERS}
                row = _clean_record(root, target, arrays, located, score, config, source="new_fast_capture", rendered_sha=hashlib.sha256(rendered.encode()).hexdigest(), processor=processor)
                existing[case_id] = row
                atomic_jsonl(root / "artifacts" / "diagnostics" / "clean_capture.jsonl", sorted(existing.values(), key=lambda value: str(value["case_id"])))
                atomic_json(root / "progress" / "capture.json", {"status": "running", "case_count": len(existing), "expected": len(expected_ids), "imported_cases": imported, "new_gpu_forwards": forwards, "last_case_id": case_id})
    finally:
        if inference is not None:
            _release(inference)
    if set(existing) != expected_ids:
        raise RuntimeError(f"Clean capture incomplete: {len(existing)}/{len(expected_ids)}")
    audit = [{"case_id": row["case_id"], "item_id": row["item_id"], "capture_source": row["capture_source"], "positions": row["positions"], "ordered_processed_indices": row["ordered_processed_indices"], "rendered_prompt_sha256": row["rendered_prompt_sha256"]} for row in sorted(existing.values(), key=lambda value: str(value["case_id"]))]
    atomic_jsonl(root / "artifacts" / "positions" / "position_audit.jsonl", audit)
    result = {"status": "complete", "case_count": len(existing), "imported_cases": imported, "new_gpu_forwards": forwards, "resumed_noop": imported == 0 and forwards == 0, "elapsed_seconds": time.time() - started}
    atomic_json(root / "progress" / "capture.json", result)
    return list(existing.values()), result


def build_vectors(root: Path, construction: Sequence[dict[str, Any]], clean_rows: Sequence[dict[str, Any]], config: dict[str, Any], *, resume: bool) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    metadata_path = root / "artifacts" / "vectors" / "vector_metadata.json"
    if metadata_path.exists():
        if not resume:
            raise FileExistsError("Vectors exist; use --resume")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("config_fingerprint") != config["fingerprint"]:
            raise ValueError("Vector config mismatch")
        output = {}
        for cell in metadata["vectors"]:
            path = root / cell["vector_file"]
            if sha256_file(path) != cell["file_sha256"]:
                raise ValueError(f"Vector hash mismatch: {path}")
            with np.load(path) as payload:
                output[(cell["position"], int(cell["layer"]))] = np.asarray(payload["scaled_vector"], dtype=np.float32)
        return output, metadata
    clean = {str(row["case_id"]): row for row in clean_rows}
    high = [row for row in construction if row["construction_side"] == "high_image"]
    low = [row for row in construction if row["construction_side"] == "high_text"]
    output: dict[tuple[str, int], np.ndarray] = {}
    cells = []
    for position in POSITIONS:
        for layer in LAYERS:
            def read(row: dict[str, Any]) -> np.ndarray:
                with np.load(root / clean[str(row["case_id"])]["hidden_file"]) as payload:
                    return np.asarray(payload[hidden_key(position, layer)], dtype=np.float32)
            arrays, metrics = construct_direction(np.stack([read(row) for row in high]), np.stack([read(row) for row in low]))
            relative = Path("artifacts") / "vectors" / f"{position}__L{layer}.npz"
            atomic_npz(root / relative, arrays)
            output[(position, layer)] = arrays["scaled_vector"]
            cells.append({"position": position, "layer": layer, "vector_file": str(relative), "file_sha256": sha256_file(root / relative), "vector_fingerprint": canonical_hash({name: array_hash(value) for name, value in arrays.items()}), **metrics})
    metadata = {"format_version": 1, "config_fingerprint": config["fingerprint"], "construction_fingerprint": manifest_fingerprint(construction), "vectors": cells}
    metadata["fingerprint"] = canonical_hash(metadata)
    atomic_json(metadata_path, metadata)
    return output, metadata


def steer(root: Path, test: Sequence[dict[str, Any]], clean_rows: Sequence[dict[str, Any]], vectors: dict[tuple[str, int], np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    trial_path = root / "artifacts" / "diagnostics" / "steering_trials.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(trial_path):
        if row.get("status") != "completed":
            continue
        if row.get("config_fingerprint") != config["fingerprint"]:
            raise ValueError(f"Trial config mismatch: {trial_key(row)}")
        key = trial_key(row)
        if key in completed:
            raise ValueError(f"Duplicate trial: {key}")
        completed[key] = row
    expected = int(config["expected_trials"])
    if len(completed) == expected:
        result = {"status": "complete", "completed_cells": expected, "expected_cells": expected, "new_gpu_forwards": 0, "resumed_noop": True}
        atomic_json(root / "progress" / "steering.json", result)
        return result
    clean = {str(row["case_id"]): row for row in clean_rows}
    inference, modules, tokenizer, device, _processor = load_inference()
    ids = class_token_ids(tokenizer)
    alphas = tuple(float(value) for value in config["alphas"])
    started = time.time()
    forwards = 0
    try:
        for target in test:
            case_id = str(target["case_id"])
            baseline = clean[case_id]
            wire = messages(target)
            rendered = render_continued_assistant(inference.processor, wire, SA_PREFILL)
            if hashlib.sha256(rendered.encode()).hexdigest() != baseline["rendered_prompt_sha256"]:
                raise ValueError(f"Rendered prompt changed: {case_id}")
            inputs = prepare_multimodal_inputs(inference.processor, wire, rendered, device=device)
            located = locate_probe_positions(tokenizer, rendered, inputs, str(target["phase0_raw_answer"]))
            for name in (*POSITIONS, "P1_SAC"):
                if int(located[name]["processed_index"]) != int(baseline["positions"][name]["processed_index"]):
                    raise ValueError(f"Position changed for {case_id} {name}")
            sac = int(located["P1_SAC"]["processed_index"])
            sequence_length = int(inputs.input_ids.shape[1])
            clean_logits = np.asarray(baseline["class_logits"], dtype=float)
            clean_probabilities = np.asarray(baseline["class_probabilities"], dtype=float)
            clean_soft = float(baseline["soft_sa_image_score"])
            clean_hard = int(baseline["argmax_hard_class"])
            clean_margin = class_margin(clean_logits, clean_hard)
            if [int(value) for value in baseline["class_token_ids"]] != ids:
                raise ValueError(f"Class token IDs changed: {case_id}")
            for position in POSITIONS:
                target_index = int(located[position]["processed_index"])
                for layer in LAYERS:
                    vector = torch.from_numpy(vectors[(position, layer)])
                    for alpha in alphas:
                        proto = {"case_id": case_id, "item_id": str(target["item_id"]), "test_side": str(target["test_side"]), "position": position, "layer": layer, "alpha": alpha}
                        if trial_key(proto) in completed:
                            continue
                        try:
                            hook = AdditiveActivationHook(modules, layer_index=layer, target_position=target_index, steering_vector=vector * alpha, prefill_sequence_length=sequence_length, injection_site="block_output")
                            with hook:
                                logits = run_logits_forward(inference.model, inputs, [sac], modules)[sac]
                            forwards += 1
                            diagnostics = hook.diagnostics()
                            scored = soft_sa_from_logits(logits, ids)
                            before = hook.h_before.numpy(); after = hook.h_after.numpy()
                            before_norm = float(np.linalg.norm(before)); after_norm = float(np.linalg.norm(after))
                            cosine = float(np.dot(before, after) / (before_norm * after_norm))
                            ratio = float(after_norm / before_norm)
                            steered_logits = np.asarray(scored["class_logits"], dtype=float)
                            parity = None
                            if alpha == 0.0:
                                parity = validate_alpha_zero(clean_logits=clean_logits, clean_probabilities=clean_probabilities, clean_soft_sa=clean_soft, clean_hard_class=clean_hard, scored=scored, before=before, after=after, diagnostics=diagnostics)
                            values = np.concatenate([steered_logits, np.asarray(scored["class_probabilities"], dtype=float), [float(scored["soft_sa_image_score"]), cosine, ratio]])
                            if not np.isfinite(values).all() or abs(float(scored["probability_sum"]) - 1.0) > 1e-9:
                                raise ValueError("Invalid steering outputs")
                            result = {
                                "status": "completed", **proto, "processed_position": target_index,
                                "clean_soft_sa": clean_soft, "steered_soft_sa": float(scored["soft_sa_image_score"]), "delta_soft_sa": float(scored["soft_sa_image_score"]) - clean_soft,
                                "clean_hard_class": clean_hard, "steered_hard_class": int(scored["argmax_hard_class"]),
                                "hard_class_changed": int(scored["argmax_hard_class"]) != clean_hard, "hard_class_delta": int(scored["argmax_hard_class"]) - clean_hard,
                                "clean_class_logits": clean_logits.tolist(), "clean_class_probabilities": clean_probabilities.tolist(),
                                "steered_class_logits": scored["class_logits"], "steered_class_probabilities": scored["class_probabilities"], "probability_sum": float(scored["probability_sum"]),
                                "clean_class_margin": clean_margin, "steered_clean_class_margin": class_margin(steered_logits, clean_hard), "margin_change": class_margin(steered_logits, clean_hard) - clean_margin,
                                "saturated": float(scored["soft_sa_image_score"]) <= 0.05 + 1e-9 or float(scored["soft_sa_image_score"]) >= 0.95 - 1e-9,
                                "finite_values": True, "activation_cosine": cosine, "activation_norm_ratio": ratio,
                                "hook_diagnostics": diagnostics, "activation_before_hash": array_hash(before), "activation_after_hash": array_hash(after),
                                "alpha_zero_parity": parity, "config_fingerprint": config["fingerprint"],
                            }
                            completed[trial_key(result)] = result
                            atomic_jsonl(trial_path, sorted(completed.values(), key=trial_key))
                        except Exception as exc:
                            failures = load_jsonl(root / "progress" / "failures.jsonl")
                            failures.append({"stage": "steering", **proto, "type": type(exc).__name__, "message": str(exc), "timestamp": time.time()})
                            atomic_jsonl(root / "progress" / "failures.jsonl", failures)
                            raise
                        if forwards % 10 == 0:
                            atomic_json(root / "progress" / "steering.json", {"status": "running", "completed_cells": len(completed), "expected_cells": expected, "new_gpu_forwards": forwards, "elapsed_seconds": time.time() - started, "last": proto})
    finally:
        _release(inference)
    if len(completed) != expected:
        raise RuntimeError(f"Steering grid incomplete: {len(completed)}/{expected}")
    result = {"status": "complete", "completed_cells": len(completed), "expected_cells": expected, "new_gpu_forwards": forwards, "resumed_noop": forwards == 0, "elapsed_seconds": time.time() - started}
    atomic_json(root / "progress" / "steering.json", result)
    return result


def run_once(root: Path, *, smoke: bool, resume: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "progress" / "failures.jsonl").exists():
        atomic_jsonl(root / "progress" / "failures.jsonl", [])
    construction, test, source = prepare_manifests(root, smoke=smoke, resume=resume)
    config = _config(root, construction, test, smoke=smoke)
    _check_config(root, config, resume=resume)
    clean_rows, capture = capture_clean(root, construction, test, config)
    vectors, vector_metadata = build_vectors(root, construction, clean_rows, config, resume=resume)
    steering = steer(root, test, clean_rows, vectors, config)
    analysis = analyze(output_root=root, smoke=smoke, resume=resume, repeats=SMOKE_BOOTSTRAP_REPEATS if smoke else BOOTSTRAP_REPEATS, positions=POSITIONS, alphas=SMOKE_ALPHAS if smoke else ALPHAS, generic_summary=True)
    completion = {"status": "complete", "smoke": smoke, "capture": capture, "steering": steering, "analysis": analysis, "vector_fingerprint": vector_metadata["fingerprint"], "selection_source": source}
    atomic_json(root / "progress" / "completion.json", completion)
    return completion


def _test_gate() -> dict[str, Any]:
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "dp_SA/checkpoint_steering/tests", "dp_SA/SA_probe/tests"], cwd=PACKAGE_ROOT.parents[1], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    passed = failed = 0
    for line in completed.stdout.splitlines()[-20:]:
        match = re.search(r"(\d+) passed", line)
        if match:
            passed = int(match.group(1))
        match = re.search(r"(\d+) failed", line)
        if match:
            failed = int(match.group(1))
    if completed.returncode:
        raise RuntimeError("CPU test gate failed\n" + completed.stdout)
    return {"status": "passed", "passed": passed, "failed": failed, "output": completed.stdout}


def run_smoke() -> dict[str, Any]:
    smoke_parent = OUTPUT_ROOT / "smoke_tmp"
    rounds = sorted(path for path in smoke_parent.glob("round_*") if path.name.removeprefix("round_").isdigit())
    number = max((int(path.name.removeprefix("round_")) for path in rounds), default=0) + 1
    if number > 2:
        raise RuntimeError("The configured maximum of two smoke rounds has been reached")
    root = smoke_parent / f"round_{number}"
    started = time.time()
    tests = _test_gate()
    first = run_once(root, smoke=True, resume=False)
    second = run_once(root, smoke=True, resume=True)
    capture = first["capture"]
    steering_result = first["steering"]
    if capture["imported_cases"] != 4 or capture["new_gpu_forwards"] != 4 or steering_result["new_gpu_forwards"] != 180:
        raise RuntimeError(f"Unexpected smoke counts: capture={capture}, steering={steering_result}")
    if second["capture"]["new_gpu_forwards"] != 0 or second["steering"]["new_gpu_forwards"] != 0 or not second["analysis"].get("resumed_noop"):
        raise RuntimeError("Smoke resume was not a zero-forward no-op")
    report = {
        "status": "passed", "round": number, "tests_passed": tests["passed"], "tests_failed": tests["failed"],
        "imported_cases": capture["imported_cases"], "capture_forwards": capture["new_gpu_forwards"],
        "steering_forwards": steering_result["new_gpu_forwards"], "total_gpu_forwards": capture["new_gpu_forwards"] + steering_result["new_gpu_forwards"],
        "resume_noop": True, "alpha_zero_parity": first["analysis"]["alpha_zero_parity"],
        "source_sha256": sha256_file(Path(__file__)), "elapsed_seconds": time.time() - started,
    }
    atomic_json(root / "progress" / "smoke_report.json", report)
    return report


def _passed_smoke() -> dict[str, Any]:
    reports = sorted(OUTPUT_ROOT.glob("smoke_tmp/round_*/progress/smoke_report.json"))
    if not reports:
        raise RuntimeError("A successful PANL2CLE smoke is required before the formal run")
    report = json.loads(reports[-1].read_text())
    if report.get("status") != "passed" or report.get("source_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("Latest smoke does not match the current PANL2CLE runner")
    return report


def run_formal(*, resume: bool) -> dict[str, Any]:
    _passed_smoke()
    result = run_once(OUTPUT_ROOT, smoke=False, resume=resume)
    capture = result["capture"]
    if not resume and (capture["imported_cases"] != EXPECTED_FORMAL_IMPORTS or capture["new_gpu_forwards"] != EXPECTED_FORMAL_CAPTURES):
        raise RuntimeError(f"Unexpected formal clean counts: {capture}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PANL→CLE new-position checkpoint steering")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke and args.resume:
        parser.error("--smoke and --resume cannot be combined; smoke performs its own resume audit")
    result = run_smoke() if args.smoke else run_formal(resume=args.resume)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
