from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch

from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import AdditiveActivationHook, model_input_device, resolve_language_modules, run_hooked_forward
from dp_SA.checkpoint_steering.run import class_margin
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits

from .config import (
    ALPHAS, DIRECTIONS, FORMAL_ROOT, HIDDEN_DEFINITION, INFERENCE_PATH,
    MODEL_PATH, NULL_ALPHAS, NULL_INITIAL_REPEATS, PANL_LAYER, PANL_POSITION,
    SMOKE_ALPHAS, SMOKE_LAYERS, STEERING_LAYERS, STEERING_POSITION,
)
from .core import answer_origin
from .io_utils import (
    append_jsonl, array_hash, atomic_json, atomic_jsonl, atomic_npz,
    canonical_hash, ensure_layout, load_jsonl, semantic_fingerprint, sha256_file,
    stable_shard,
)
from .processor import enforce_fast_image_processor
from .run_spec import add_run_spec_arguments, normalize_run_spec, run_spec_cli_args, run_spec_from_args


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image", "image": str(Path(row["image_path"]).resolve())}, {"type": "text", "text": str(row["phase1_prompt"])}]}, {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]}]


def trial_key(row: dict[str, Any]) -> str:
    return f'{row["case_id"]}|{row["direction"]}|L{int(row["layer"])}|a{float(row["alpha"]):g}'


def null_key(row: dict[str, Any]) -> str:
    return f'{row["case_id"]}|null{int(row["null_replicate"]):03d}|L{int(row.get("layer", 14))}|a{float(row["alpha"]):g}'


def _load_probe(root: Path, name: str):
    payload = joblib.load(root / f"artifacts/probes/{name}__full.joblib")
    return payload["model"]


def _load_vectors(root: Path, layers: Sequence[int]):
    metadata = json.loads((root / "artifacts/directions/vector_metadata.json").read_text()); meta = {(int(r["layer"]), r["recipient_answer"], r["direction"]): r for r in metadata["vectors"]}; payloads = {layer: np.load(root / f"artifacts/directions/P1_LAT__L{layer}.npz") for layer in set(layers)}
    return metadata, meta, payloads


def _read_completed(root: Path, prefix: str, key_fn) -> dict[str, dict[str, Any]]:
    output = {}
    for path in sorted((root / "artifacts/trials").glob(f"{prefix}.shard_*.jsonl")):
        for row in load_jsonl(path, repair_trailing=True):
            key = key_fn(row)
            if key in output and row != output[key]: raise ValueError(f"Conflicting trial duplicate: {key}")
            output[key] = row
    return output


def _predict(model, hidden: np.ndarray) -> float:
    return float(model.predict(np.asarray(hidden, np.float32).reshape(1, -1))[0])


def _score(forward, sac: int, class_ids: list[int]) -> dict[str, Any]:
    return soft_sa_from_logits(forward.logits_by_position[sac], class_ids)


def alpha_zero_parity(clean_score: dict[str, Any], zero_score: dict[str, Any], before: np.ndarray, after: np.ndarray, clean_panl: np.ndarray, zero_panl: np.ndarray, panl_probe_error: float) -> dict[str, Any]:
    result = {
        "lat_before_after_bitwise": bool(np.array_equal(before, after)),
        "panl_clean_zero_bitwise": bool(np.array_equal(clean_panl, zero_panl)),
        "panl_probe_error": float(panl_probe_error),
        "sac_logits_bitwise": bool(np.array_equal(np.asarray(clean_score["class_logits"]), np.asarray(zero_score["class_logits"]))),
        "sac_probabilities_bitwise": bool(np.array_equal(np.asarray(clean_score["class_probabilities"]), np.asarray(zero_score["class_probabilities"]))),
        "soft_sa_error": abs(float(clean_score["soft_sa_image_score"]) - float(zero_score["soft_sa_image_score"])),
        "hard_sa_equal": int(clean_score["argmax_hard_class"]) == int(zero_score["argmax_hard_class"]),
    }
    result["passed"] = all((result["lat_before_after_bitwise"], result["panl_clean_zero_bitwise"], result["sac_logits_bitwise"], result["sac_probabilities_bitwise"], result["soft_sa_error"] <= 1e-12, result["hard_sa_equal"], result["panl_probe_error"] <= 1e-8))
    return result


def _trial_row(proto: dict[str, Any], manifest: dict[str, Any], clean_score: dict[str, Any], score: dict[str, Any], before: np.ndarray, after: np.ndarray, clean_panl: np.ndarray, panl: np.ndarray, hook_diag: dict[str, Any], lat_conf, panl_conf, panl_sa, hidden_file: str, hidden_key: str, vector_fp: str, alpha_zero: dict[str, Any] | None, vector_info: dict[str, Any] | None = None) -> dict[str, Any]:
    clean_hard = int(clean_score["argmax_hard_class"]); clean_logits = np.asarray(clean_score["class_logits"], float); steered_logits = np.asarray(score["class_logits"], float)
    before_norm, after_norm = float(np.linalg.norm(before)), float(np.linalg.norm(after)); displacement = float(np.linalg.norm(after - before))
    result = {
        "status": "completed", **proto, "item_id": str(manifest["item_id"]), "family_id": str(manifest["family_id"]), "condition": str(manifest["condition"]),
        "answer_origin": answer_origin(manifest), "fixed_answer": str(manifest["phase0_normalized_answer"]), "hidden_definition": HIDDEN_DEFINITION,
        "activation_before_hash": array_hash(before), "activation_after_hash": array_hash(after), "activation_before_norm": before_norm, "activation_after_norm": after_norm,
        "activation_cosine": float(before @ after / (before_norm * after_norm)), "displacement_norm": displacement, "displacement_ratio": displacement / before_norm,
        "hook_hit_count": int(hook_diag["hook_call_count"]), "steering_applied_count": int(hook_diag["steering_applied_count"]), "only_target_lat_token_modified": True,
        "panl_clean_hidden_hash": array_hash(clean_panl), "panl_steered_hidden_hash": array_hash(panl), "panl_hidden_file": hidden_file, "panl_hidden_key": hidden_key,
        "clean_lat_confidence_probe": _predict(lat_conf, before), "steered_lat_confidence_probe": _predict(lat_conf, after),
        "delta_confidence_LAT_immediate": _predict(lat_conf, after) - _predict(lat_conf, before),
        "clean_panl_confidence_probe": _predict(panl_conf, clean_panl), "steered_panl_confidence_probe": _predict(panl_conf, panl),
        "delta_confidence_PANL_L18": _predict(panl_conf, panl) - _predict(panl_conf, clean_panl),
        "clean_panl_sa_probe": _predict(panl_sa, clean_panl), "steered_panl_sa_probe": _predict(panl_sa, panl), "delta_panl_probe_sa": _predict(panl_sa, panl) - _predict(panl_sa, clean_panl),
        "clean_sa_logits": clean_score["class_logits"], "steered_sa_logits": score["class_logits"], "clean_sa_probabilities": clean_score["class_probabilities"], "steered_sa_probabilities": score["class_probabilities"],
        "clean_soft_sa": float(clean_score["soft_sa_image_score"]), "steered_soft_sa": float(score["soft_sa_image_score"]), "delta_final_soft_sa": float(score["soft_sa_image_score"] - clean_score["soft_sa_image_score"]),
        "clean_hard_sa": clean_hard, "steered_hard_sa": int(score["argmax_hard_class"]), "hard_sa_changed": int(score["argmax_hard_class"]) != clean_hard,
        "clean_class_margin": class_margin(clean_logits, clean_hard), "clean_class_margin_change": class_margin(steered_logits, clean_hard) - class_margin(clean_logits, clean_hard),
        "actual_injection_norm": abs(float(proto["alpha"])) * float(vector_info["scaled_norm"]) if vector_info else displacement,
        "signed_injection_natural_sd": float(proto["alpha"]) * float(vector_info["injection_to_natural_projection_sd"]) if vector_info and vector_info.get("injection_to_natural_projection_sd") is not None else None,
        "saturated": bool(float(score["soft_sa_image_score"]) < 1e-6 or float(score["soft_sa_image_score"]) > 1 - 1e-6), "format_valid": True, "vector_fingerprint": vector_fp, "alpha_zero_parity": alpha_zero,
    }
    floats = [value for value in result.values() if isinstance(value, float)]
    if not np.isfinite(floats).all() or abs(sum(result["steered_sa_probabilities"]) - 1.0) > 1e-8: raise ValueError("Non-finite/invalid trial")
    return result


def _worker(root: Path, worker: int, num_gpus: int, *, smoke: bool, desired_null: int, config_fingerprint: str, run_spec: dict[str, Any]) -> dict[str, Any]:
    manifest = load_jsonl(root / "artifacts/manifests/runtime_manifest.jsonl"); selected = [r for r in manifest if stable_shard(str(r["case_id"]), num_gpus) == worker]
    layers = list(SMOKE_LAYERS if smoke and run_spec["is_default"] else run_spec["layers"]); alphas = tuple(SMOKE_ALPHAS if smoke and run_spec["is_default"] else run_spec["alphas"]); directions = tuple(run_spec["directions"])
    null_layers = ([14] if run_spec["is_default"] else layers) if run_spec["shuffle_requested"] and not smoke else []
    null_alphas = tuple(alpha for alpha in alphas if alpha != 0 and -alpha in alphas)
    completed = _read_completed(root, "main_trials", trial_key); null_completed = _read_completed(root, "null_trials", null_key)
    main_path = root / f"artifacts/trials/main_trials.shard_{worker}.jsonl"; null_path = root / f"artifacts/trials/null_trials.shard_{worker}.jsonl"
    expected_main = {trial_key({"case_id": r["case_id"], "direction": d, "layer": l, "alpha": a}) for r in selected for d in directions for l in layers for a in alphas}
    expected_null = {null_key({"case_id": r["case_id"], "null_replicate": rep, "layer": layer, "alpha": a}) for r in selected for rep in range(1, desired_null + 1) for layer in null_layers for a in null_alphas}
    if expected_main <= set(completed) and expected_null <= set(null_completed): return {"worker": worker, "new_gpu_forwards": 0, "resumed_noop": True}
    runtime = load_runtime(INFERENCE_PATH); inference = runtime.QwenVLInference(str(MODEL_PATH)); enforce_fast_image_processor(inference.processor); modules = resolve_language_modules(inference.model); tokenizer = getattr(inference.processor, "tokenizer", inference.processor); ids = class_token_ids(tokenizer); device = model_input_device(inference)
    metadata, vector_meta, vector_payloads = _load_vectors(root, layers)
    lat_conf = {layer: _load_probe(root, f"confidence_gap__P1_LAT__L{layer}") for layer in layers}; panl_conf = _load_probe(root, "confidence_gap__P1_PANL__L18"); panl_sa = _load_probe(root, "final_sa__P1_PANL__L18")
    forwards = 0
    for row in selected:
        case = str(row["case_id"]); answer = str(row["phase0_normalized_answer"])
        pending_main = [key for key in expected_main if key.startswith(case + "|") and key not in completed]; pending_null = [key for key in expected_null if key.startswith(case + "|") and key not in null_completed]
        if not pending_main and not pending_null: continue
        messages = _messages(row); rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL); inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=device); located = locate_phase1_positions(tokenizer, rendered, inputs, str(row["phase0_raw_answer"])); lat = int(located[STEERING_POSITION]["processed_index"]); panl_pos = int(located[PANL_POSITION]["processed_index"]); sac = int(located["P1_SAC"]["processed_index"]); sequence = int(inputs.input_ids.shape[1])
        if lat != int(row["positions"][STEERING_POSITION]["processed_index"]) or panl_pos != int(row["positions"][PANL_POSITION]["processed_index"]) or sac != int(row["positions"]["P1_SAC"]["processed_index"]): raise ValueError(f"Position parity failed: {case}")
        hidden_path = root / f"artifacts/hidden/{case}.npz"; hidden_arrays = {}
        if hidden_path.is_file():
            with np.load(hidden_path) as old: hidden_arrays.update({k: np.asarray(old[k]) for k in old.files})
        zero_cache = {}
        if pending_main:
            clean_forward = run_hooked_forward(inference.model, inputs, modules, {PANL_POSITION: panl_pos}, logits_positions=[sac]); forwards += 1; clean_score = _score(clean_forward, sac, ids); clean_panl = clean_forward.hidden_by_name[PANL_POSITION][PANL_LAYER].detach().float().cpu().numpy().astype(np.float32)
            hidden_arrays["clean__P1_PANL__L18"] = clean_panl
            for layer in layers:
                vector = torch.zeros(len(vector_payloads[layer][f"{answer}__{directions[0]}__scaled"]), dtype=torch.float32)
                hook = AdditiveActivationHook(modules, layer_index=layer, target_position=lat, steering_vector=vector, prefill_sequence_length=sequence, injection_site="block_output")
                with hook: forward = run_hooked_forward(inference.model, inputs, modules, {PANL_POSITION: panl_pos}, logits_positions=[sac])
                forwards += 1; score = _score(forward, sac, ids); before = hook.h_before.numpy(); after = hook.h_after.numpy(); hp = forward.hidden_by_name[PANL_POSITION][PANL_LAYER].detach().float().cpu().numpy().astype(np.float32); diag = hook.diagnostics()
                if int(diag["hook_call_count"]) != 1 or int(diag["steering_applied_count"]) != 1: raise ValueError(f"Hook did not hit exactly once: {case}/L{layer}")
                parity = alpha_zero_parity(clean_score, score, before, after, clean_panl, hp, abs(_predict(panl_sa, clean_panl) - _predict(panl_sa, hp)))
                if not parity["passed"]: raise ValueError(f"Alpha-zero parity failed: {case}/L{layer}: {parity}")
                key = f"zero__L{layer}__P1_PANL__L18"; hidden_arrays[key] = hp; zero_cache[layer] = (score, before, after, hp, diag, parity, key)
        else:
            source = next(r for r in completed.values() if r["case_id"] == case)
            clean_score = {"class_logits": source["clean_sa_logits"], "class_probabilities": source["clean_sa_probabilities"], "soft_sa_image_score": source["clean_soft_sa"], "argmax_hard_class": source["clean_hard_sa"]}
            clean_panl = np.asarray(hidden_arrays["clean__P1_PANL__L18"], np.float32)
        case_main = []
        for layer in layers:
            for direction in directions:
                meta = vector_meta[layer, answer, direction]; base = np.asarray(vector_payloads[layer][meta["scaled_key"]], np.float32)
                for alpha in alphas:
                    proto = {"case_id": case, "direction": direction, "layer": layer, "alpha": float(alpha), "trial_type": "main", "config_fingerprint": config_fingerprint}
                    if trial_key(proto) in completed: continue
                    if float(alpha) == 0:
                        score, before, after, hp, diag, parity, hidden_key = zero_cache[layer]
                    else:
                        hook = AdditiveActivationHook(modules, layer_index=layer, target_position=lat, steering_vector=torch.from_numpy(base) * float(alpha), prefill_sequence_length=sequence, injection_site="block_output")
                        with hook: forward = run_hooked_forward(inference.model, inputs, modules, {PANL_POSITION: panl_pos}, logits_positions=[sac])
                        forwards += 1; score = _score(forward, sac, ids); before = hook.h_before.numpy(); after = hook.h_after.numpy(); hp = forward.hidden_by_name[PANL_POSITION][PANL_LAYER].detach().float().cpu().numpy().astype(np.float32); diag = hook.diagnostics(); parity = None
                        if int(diag["hook_call_count"]) != 1 or int(diag["steering_applied_count"]) != 1: raise ValueError(f"Hook did not hit exactly once: {case}/{direction}/L{layer}/a{alpha}")
                        hidden_key = f"main__{direction}__L{layer}__a{float(alpha):+g}__P1_PANL__L18"; hidden_arrays[hidden_key] = hp
                    case_main.append(_trial_row(proto, row, clean_score, score, before, after, clean_panl, hp, diag, lat_conf[layer], panl_conf, panl_sa, str(hidden_path.relative_to(root)), hidden_key, meta["vector_fingerprint"], parity, meta))
        case_null = []
        if null_layers:
            for layer in null_layers:
                for rep in range(1, desired_null + 1):
                    base = np.asarray(vector_payloads[layer][f"null_{rep:03d}__{answer}__scaled"], np.float32)
                    vector_fp = canonical_hash({"replicate": rep, "answer": answer, "layer": layer, "hash": array_hash(base)})
                    for alpha in null_alphas:
                        proto = {"case_id": case, "direction": "rebuilt_shuffle_null", "layer": layer, "alpha": float(alpha), "trial_type": "null", "null_replicate": rep, "config_fingerprint": config_fingerprint}
                        if null_key(proto) in null_completed: continue
                        hook = AdditiveActivationHook(modules, layer_index=layer, target_position=lat, steering_vector=torch.from_numpy(base) * float(alpha), prefill_sequence_length=sequence, injection_site="block_output")
                        with hook: forward = run_hooked_forward(inference.model, inputs, modules, {PANL_POSITION: panl_pos}, logits_positions=[sac])
                        forwards += 1; score = _score(forward, sac, ids); before = hook.h_before.numpy(); after = hook.h_after.numpy(); hp = forward.hidden_by_name[PANL_POSITION][PANL_LAYER].detach().float().cpu().numpy().astype(np.float32); diag = hook.diagnostics(); hidden_key = f"null__r{rep:03d}__L{layer}__a{float(alpha):+g}__P1_PANL__L18"; hidden_arrays[hidden_key] = hp
                        if int(diag["hook_call_count"]) != 1 or int(diag["steering_applied_count"]) != 1: raise ValueError(f"Null hook did not hit exactly once: {case}/r{rep}/L{layer}/a{alpha}")
                        case_null.append(_trial_row(proto, row, clean_score, score, before, after, clean_panl, hp, diag, lat_conf[layer], panl_conf, panl_sa, str(hidden_path.relative_to(root)), hidden_key, vector_fp, None))
        atomic_npz(hidden_path, hidden_arrays)
        for result in case_main: append_jsonl(main_path, result); completed[trial_key(result)] = result
        for result in case_null: append_jsonl(null_path, result); null_completed[null_key(result)] = result
    for payload in vector_payloads.values(): payload.close()
    return {"worker": worker, "new_gpu_forwards": forwards, "resumed_noop": forwards == 0}


def _merge(root: Path, *, smoke: bool, desired_null: int, config_fingerprint: str, run_spec: dict[str, Any]) -> dict[str, Any]:
    manifest = load_jsonl(root / "artifacts/manifests/runtime_manifest.jsonl"); layers = list(SMOKE_LAYERS if smoke and run_spec["is_default"] else run_spec["layers"]); alphas = SMOKE_ALPHAS if smoke and run_spec["is_default"] else run_spec["alphas"]
    main = _read_completed(root, "main_trials", trial_key); null = _read_completed(root, "null_trials", null_key)
    null_layers = ([14] if run_spec["is_default"] else layers) if run_spec["shuffle_requested"] and not smoke else []; null_alphas = [a for a in alphas if a != 0 and -a in alphas]
    expected_main = len(manifest) * len(layers) * len(run_spec["directions"]) * len(alphas); expected_null = len(manifest) * desired_null * len(null_layers) * len(null_alphas)
    if len(main) != expected_main or len(null) != expected_null: raise ValueError(f"Trial merge incomplete main={len(main)}/{expected_main} null={len(null)}/{expected_null}")
    atomic_jsonl(root / "artifacts/trials/main_trials.jsonl", [main[k] for k in sorted(main)])
    if expected_null or run_spec["is_default"]: atomic_jsonl(root / "artifacts/trials/null_trials.jsonl", [null[k] for k in sorted(null)])
    return {"main_trial_count": expected_main, "null_trial_count": expected_null}


def run_steering(*, output_root: Path = FORMAL_ROOT, smoke: bool = False, resume: bool = False, num_gpus: int = 1, desired_null: int | None = None, worker_id: int | None = None, run_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    if num_gpus not in (1, 2): raise ValueError("--num-gpus must be 1 or 2")
    run_spec = normalize_run_spec() if run_spec is None else run_spec; desired_null = (NULL_INITIAL_REPEATS if run_spec["shuffle_requested"] else 0) if desired_null is None else int(desired_null)
    if not run_spec["shuffle_requested"] and desired_null: raise ValueError("Null repeats require an explicitly selected shuffle direction")
    root = ensure_layout(output_root); prelock = json.loads((root / "artifacts/diagnostics/prelock_material.json").read_text()); manifest_path = root / "artifacts/manifests/runtime_manifest.jsonl"
    if prelock.get("run_spec") != run_spec: raise ValueError("Runtime run spec does not match prepared artifacts")
    config = {"format_version": 4, "smoke_only": smoke, "prelock_fingerprint": prelock["fingerprint"], "runtime_manifest_sha256": sha256_file(manifest_path), "run_spec": run_spec, "desired_null": desired_null}
    fingerprint = canonical_hash(config)
    if worker_id is not None: return _worker(root, worker_id, num_gpus, smoke=smoke, desired_null=desired_null, config_fingerprint=fingerprint, run_spec=run_spec)
    semantic_fingerprint(root / "progress/run_config.json", config, resume=resume)
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus: raise RuntimeError(f"Requested {num_gpus} GPUs, visible={torch.cuda.device_count()}")
    started = time.time(); processes = []
    for worker in range(num_gpus):
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(worker); command = [sys.executable, "-m", "dp_SA.confidence_steering.run", "--output-root", str(root), "--num-gpus", str(num_gpus), "--worker-id", str(worker), "--desired-null", str(desired_null), *run_spec_cli_args(run_spec)]
        if smoke: command.append("--smoke")
        processes.append(subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], env=env))
    codes = [p.wait() for p in processes]
    if any(codes): raise RuntimeError(f"GPU worker failed: {codes}")
    merged = _merge(root, smoke=smoke, desired_null=0 if smoke else desired_null, config_fingerprint=fingerprint, run_spec=run_spec)
    forwards = sum(json.loads((root / f"progress/worker_{w}.json").read_text()).get("new_gpu_forwards", 0) for w in range(num_gpus))
    result = {"status": "complete", "smoke_only": smoke, **merged, "desired_null": 0 if smoke else desired_null, "new_gpu_forwards": forwards, "resumed_noop": forwards == 0, "config_fingerprint": fingerprint, "elapsed_seconds": time.time() - started}; atomic_json(root / "progress/run.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--num-gpus", type=int, choices=(1, 2), default=1); parser.add_argument("--worker-id", type=int); parser.add_argument("--desired-null", type=int); add_run_spec_arguments(parser)
    args = parser.parse_args(argv); root = Path(args.output_root) if args.output_root else FORMAL_ROOT; result = run_steering(output_root=root, smoke=args.smoke, resume=args.resume, num_gpus=args.num_gpus, desired_null=args.desired_null, worker_id=args.worker_id, run_spec=run_spec_from_args(args))
    if args.worker_id is not None: atomic_json(root / f"progress/worker_{args.worker_id}.json", result)
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
