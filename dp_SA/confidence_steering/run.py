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

import numpy as np
import torch

from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import AdditiveActivationHook, model_input_device, resolve_language_modules, run_logits_forward
from dp_SA.answer_matched_lat_steering.io_utils import canonical_hash
from dp_SA.checkpoint_steering.run import class_margin, validate_alpha_zero
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import PHASE1_TEMPLATE, SA_PREFILL
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits

from .config import ALPHAS, DIRECTIONS, HIDDEN_DEFINITION, INFERENCE_PATH, LAYERS, MODEL_PATH, POSITION, RESULTS_ROOT, SMOKE_ALPHAS, SMOKE_LAYERS
from .io_utils import append_jsonl, array_hash, atomic_json, atomic_jsonl, check_fingerprint, ensure_layout, load_jsonl, sha256_file, stable_shard
from .processor import enforce_frozen_image_processor


def trial_key(row: dict[str, Any]) -> str:
    return f'{row["case_id"]}|{row["direction"]}|L{int(row["layer"])}|a{float(row["alpha"]):g}'


def baseline_key(row: dict[str, Any]) -> str:
    return f'{row["case_id"]}|L{int(row["layer"])}'


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image", "image": str(Path(row["image_path"]).resolve())}, {"type": "text", "text": str(row["phase1_prompt"])}]},
            {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]}]


def _semantic_config(root: Path, *, smoke: bool) -> dict[str, Any]:
    prepare = json.loads((root / "progress/prepare.json").read_text())
    metadata = json.loads((root / "artifacts/vectors/vector_metadata.json").read_text())
    test_path = root / "artifacts/audits/test_manifest.jsonl"
    model_hashes = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.json", "model.safetensors.index.json"):
        path = MODEL_PATH / name
        if path.is_file(): model_hashes[name] = sha256_file(path)
    for layer,digest in metadata["files"].items():
        if sha256_file(root/f"artifacts/vectors/P1_LAT__L{layer}.npz") != digest: raise ValueError("Vector file fingerprint mismatch before run")
    package=Path(__file__).resolve().parent; source_hashes={name:sha256_file(package/name) for name in ("config.py","io_utils.py","processor.py","run.py")}
    return {"format_version": 1, "smoke_only": smoke, "prepare_fingerprint": prepare["config_fingerprint"], "vector_fingerprint": metadata["fingerprint"],
            "test_manifest_sha256": sha256_file(test_path), "position": POSITION, "layers": list(SMOKE_LAYERS if smoke else LAYERS),
            "alphas": list(SMOKE_ALPHAS if smoke else ALPHAS), "directions": list(DIRECTIONS), "phase1_template_hash": canonical_hash(PHASE1_TEMPLATE),
            "sa_prefill_hash": canonical_hash(SA_PREFILL), "position_code_sha256": sha256_file(Path(__file__).resolve().parents[1] / "positions.py"),
            "model_path": str(MODEL_PATH.resolve()), "model_processor_hashes": model_hashes, "inference_sha256": sha256_file(INFERENCE_PATH),"source_code":source_hashes}


def _load_vectors(root: Path) -> tuple[dict[tuple[str, int, str], np.ndarray], dict[tuple[str, int, str], dict[str, Any]]]:
    metadata = json.loads((root / "artifacts/vectors/vector_metadata.json").read_text()); vectors = {}; rows = {}
    by_layer = {}
    for row in metadata["vectors"]:
        layer = int(row["layer"])
        if layer not in by_layer: by_layer[layer] = np.load(root / f"artifacts/vectors/P1_LAT__L{layer}.npz")
        value = np.asarray(by_layer[layer][row["scaled_key"]], dtype=np.float32)
        if array_hash(value) != row["scaled_hash"]: raise ValueError("Vector array hash mismatch")
        key = (str(row["recipient_answer"]), layer, str(row["direction"])); vectors[key] = value; rows[key] = row
    for payload in by_layer.values(): payload.close()
    return vectors, rows


def _completed_from_shards(root: Path, prefix: str, key_fn) -> dict[str, dict[str, Any]]:
    output = {}
    for path in sorted((root / "artifacts/trials").glob(f"{prefix}.shard_*.jsonl")):
        for row in load_jsonl(path, repair_trailing=True):
            if row.get("status") != "completed": continue
            key = key_fn(row)
            if key in output and row != output[key]: raise ValueError(f"Conflicting duplicate artifact: {key}")
            output[key] = row
    return output


def _trial_result(proto: dict[str, Any], *, manifest: dict[str, Any], clean: dict[str, Any], scored: dict[str, Any], before: np.ndarray, after: np.ndarray,
                  diagnostics: dict[str, Any], vector_row: dict[str, Any], config_fingerprint: str, alpha_zero: dict[str, Any] | None, baseline_ref: str | None) -> dict[str, Any]:
    clean_logits = np.asarray(clean["class_logits"], dtype=float); clean_probs = np.asarray(clean["class_probabilities"], dtype=float)
    clean_hard = int(clean["argmax_hard_class"]); clean_margin = class_margin(clean_logits, clean_hard)
    steered_logits = np.asarray(scored["class_logits"], dtype=float); steered_margin = class_margin(steered_logits, clean_hard)
    before_norm, after_norm = float(np.linalg.norm(before)), float(np.linalg.norm(after))
    if before_norm <= 0 or after_norm <= 0: raise ValueError("Zero activation norm")
    cosine = float(np.dot(before, after) / (before_norm * after_norm))
    result = {"status": "completed", **proto, "item_id": str(manifest["item_id"]), "family_id": str(manifest["family_id"]),
              "condition": str(manifest["condition"]), "answer_origin": str(manifest["answer_origin"]), "fixed_answer": str(manifest["fixed_answer_color"]),
              "processed_lat_index": int(clean["processed_lat_index"]), "answer_token_span": manifest["phase1_answer_span"], "answer_token_ids": manifest["phase1_answer_token_ids"],
              "clean_sa_logits": clean_logits.tolist(), "clean_sa_probabilities": clean_probs.tolist(), "clean_soft_sa": float(clean["soft_sa_image_score"]), "clean_hard_sa": clean_hard,
              "steered_sa_logits": scored["class_logits"], "steered_sa_probabilities": scored["class_probabilities"], "steered_soft_sa": float(scored["soft_sa_image_score"]), "steered_hard_sa": int(scored["argmax_hard_class"]),
              "delta_soft_sa": float(scored["soft_sa_image_score"]) - float(clean["soft_sa_image_score"]), "hard_class_changed": int(scored["argmax_hard_class"]) != clean_hard,
              "clean_class_margin": clean_margin, "steered_clean_class_margin": steered_margin, "margin_change": steered_margin-clean_margin,
              "activation_before_norm": before_norm, "activation_after_norm": after_norm, "activation_cosine": cosine,
              "hook_hit_count": int(diagnostics["hook_call_count"]), "steering_applied_count": int(diagnostics["steering_applied_count"]),
              "activation_before_hash": array_hash(before), "activation_after_hash": array_hash(after), "vector_fingerprint": vector_row["vector_fingerprint"],
              "config_fingerprint": config_fingerprint, "alpha_zero_parity": alpha_zero, "baseline_ref": baseline_ref,"clean_definition":"run-local canonical hooked alpha-zero with bitwise-identical activation","hidden_definition":HIDDEN_DEFINITION}
    finite = [result[k] for k in ("steered_soft_sa", "delta_soft_sa", "clean_class_margin", "margin_change", "activation_before_norm", "activation_after_norm", "activation_cosine")]
    if not np.isfinite(finite).all() or abs(sum(result["steered_sa_probabilities"])-1.0) > 1e-9: raise ValueError("Non-finite trial")
    return result


def _worker(root: Path, worker_id: int, num_gpus: int, *, smoke: bool, resume: bool, config_fingerprint: str) -> dict[str, Any]:
    test = load_jsonl(root / "artifacts/audits/test_manifest.jsonl"); selected = [r for r in test if stable_shard(str(r["case_id"]), num_gpus) == worker_id]
    completed = _completed_from_shards(root, "trials", trial_key); baselines = _completed_from_shards(root, "alpha_zero", baseline_key)
    trial_path = root / f"artifacts/trials/trials.shard_{worker_id}.jsonl"; baseline_path = root / f"artifacts/trials/alpha_zero.shard_{worker_id}.jsonl"
    layers = SMOKE_LAYERS if smoke else LAYERS; alphas = SMOKE_ALPHAS if smoke else ALPHAS
    expected_keys = {trial_key({"case_id": r["case_id"], "direction": d, "layer": l, "alpha": a}) for r in selected for d in DIRECTIONS for l in layers for a in alphas}
    if expected_keys.issubset(completed): return {"worker": worker_id, "new_gpu_forwards": 0, "resumed_noop": True}
    runtime = load_runtime(INFERENCE_PATH); inference = runtime.QwenVLInference(str(MODEL_PATH)); enforce_frozen_image_processor(inference.processor); modules = resolve_language_modules(inference.model)
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor); class_ids = class_token_ids(tokenizer); device = model_input_device(inference)
    vectors, vector_rows = _load_vectors(root); new_forwards = 0
    prepare_config=json.loads((root/"progress/prepare_config.json").read_text()); manifest_fingerprint=sha256_file(root/"artifacts/audits/test_manifest.jsonl"); input_fingerprint=canonical_hash(prepare_config["inputs"])
    for manifest in selected:
        case_id = str(manifest["case_id"]); messages = _messages(manifest); rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
        inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=device); located = locate_phase1_positions(tokenizer, rendered, inputs, str(manifest["phase0_raw_answer"]))
        lat, sac, sequence_length = int(located[POSITION]["processed_index"]), int(located["P1_SAC"]["processed_index"]), int(inputs.input_ids.shape[1])
        if lat != int(manifest["positions"][POSITION]["processed_index"]) or lat != int(located["phase1_answer_span"][1])-1: raise ValueError(f"LAT position parity failed: {case_id}")
        if [int(x) for x in manifest["class_token_ids"]] != class_ids: raise ValueError("SA token IDs changed")
        clean: dict[str, Any] | None = None
        canonical_baseline = baselines.get(baseline_key({"case_id": case_id, "layer": int(layers[0])}))
        if canonical_baseline is not None:
            canonical_scored = canonical_baseline["scored"]
            clean = {"class_logits": canonical_scored["class_logits"], "class_probabilities": canonical_scored["class_probabilities"], "soft_sa_image_score": canonical_scored["soft_sa_image_score"], "argmax_hard_class": canonical_scored["argmax_hard_class"], "processed_lat_index": lat}
        answer = str(manifest["fixed_answer_color"])
        for layer in layers:
            bproto = {"case_id": case_id, "layer": int(layer),"manifest_fingerprint":manifest_fingerprint,"input_fingerprint":input_fingerprint}; bkey = baseline_key(bproto)
            if bkey not in baselines:
                first_vector = torch.from_numpy(vectors[answer, int(layer), DIRECTIONS[0]]) * 0.0
                hook = AdditiveActivationHook(modules, layer_index=int(layer), target_position=lat, steering_vector=first_vector, prefill_sequence_length=sequence_length, injection_site="block_output")
                with hook: logits = run_logits_forward(inference.model, inputs, [sac], modules)[sac]
                new_forwards += 1; scored = soft_sa_from_logits(logits, class_ids); before, after = hook.h_before.numpy(), hook.h_after.numpy(); diagnostics = hook.diagnostics()
                if clean is None:
                    # A zero vector plus bitwise-equal before/after activation is
                    # the run-local clean oracle. Historical logits are retained
                    # below as a drift audit because library/runtime upgrades can
                    # change them without changing the frozen prompt or tokens.
                    clean = {"class_logits": scored["class_logits"], "class_probabilities": scored["class_probabilities"], "soft_sa_image_score": scored["soft_sa_image_score"], "argmax_hard_class": scored["argmax_hard_class"], "processed_lat_index": lat}
                parity = validate_alpha_zero(clean_logits=np.asarray(clean["class_logits"]), clean_probabilities=np.asarray(clean["class_probabilities"]), clean_soft_sa=float(clean["soft_sa_image_score"]), clean_hard_class=int(clean["argmax_hard_class"]), scored=scored, before=before, after=after, diagnostics=diagnostics)
                historical_logits=np.asarray(manifest["class_logits"],dtype=float); historical_probs=np.asarray(manifest["class_probabilities"],dtype=float)
                drift={"logit_max_abs_error":float(np.max(np.abs(np.asarray(scored["class_logits"])-historical_logits))),"probability_max_abs_error":float(np.max(np.abs(np.asarray(scored["class_probabilities"])-historical_probs))),"soft_sa_abs_error":abs(float(scored["soft_sa_image_score"])-float(manifest["soft_sa_image_score"]))}
                baseline = {"status": "completed", **bproto, "scored": scored, "before": before.tolist(), "after": after.tolist(), "diagnostics": diagnostics, "alpha_zero_parity": parity, "historical_clean_drift":drift,"clean_definition":"first hooked alpha-zero with bitwise-identical activation", "config_fingerprint": config_fingerprint}
                append_jsonl(baseline_path, baseline); baselines[bkey] = baseline
            baseline = baselines[bkey]
            if baseline.get("config_fingerprint") != config_fingerprint: raise ValueError("Baseline config fingerprint mismatch")
            if clean is None:
                canonical_scored=baseline["scored"]; clean={"class_logits":canonical_scored["class_logits"],"class_probabilities":canonical_scored["class_probabilities"],"soft_sa_image_score":canonical_scored["soft_sa_image_score"],"argmax_hard_class":canonical_scored["argmax_hard_class"],"processed_lat_index":lat}
            for direction in DIRECTIONS:
                vector_row = vector_rows[answer, int(layer), direction]
                zero_proto = {"case_id": case_id, "direction": direction, "layer": int(layer), "alpha": 0.0,"manifest_fingerprint":manifest_fingerprint,"input_fingerprint":input_fingerprint}
                if trial_key(zero_proto) not in completed:
                    row = _trial_result(zero_proto, manifest=manifest, clean=clean, scored=baseline["scored"], before=np.asarray(baseline["before"], dtype=np.float32), after=np.asarray(baseline["after"], dtype=np.float32), diagnostics=baseline["diagnostics"], vector_row=vector_row, config_fingerprint=config_fingerprint, alpha_zero=baseline["alpha_zero_parity"], baseline_ref=bkey)
                    append_jsonl(trial_path, row); completed[trial_key(row)] = row
                for alpha in (a for a in alphas if float(a) != 0.0):
                    proto = {"case_id": case_id, "direction": direction, "layer": int(layer), "alpha": float(alpha),"manifest_fingerprint":manifest_fingerprint,"input_fingerprint":input_fingerprint}
                    if trial_key(proto) in completed: continue
                    vector = torch.from_numpy(vectors[answer, int(layer), direction]) * float(alpha)
                    hook = AdditiveActivationHook(modules, layer_index=int(layer), target_position=lat, steering_vector=vector, prefill_sequence_length=sequence_length, injection_site="block_output")
                    with hook: logits = run_logits_forward(inference.model, inputs, [sac], modules)[sac]
                    new_forwards += 1; scored = soft_sa_from_logits(logits, class_ids); before, after, diagnostics = hook.h_before.numpy(), hook.h_after.numpy(), hook.diagnostics()
                    row = _trial_result(proto, manifest=manifest, clean=clean, scored=scored, before=before, after=after, diagnostics=diagnostics, vector_row=vector_row, config_fingerprint=config_fingerprint, alpha_zero=None, baseline_ref=None)
                    append_jsonl(trial_path, row); completed[trial_key(row)] = row
    return {"worker": worker_id, "new_gpu_forwards": new_forwards, "resumed_noop": new_forwards == 0}


def _merge(root: Path, *, smoke: bool, config_fingerprint: str) -> dict[str, Any]:
    test = load_jsonl(root / "artifacts/audits/test_manifest.jsonl"); layers = SMOKE_LAYERS if smoke else LAYERS; alphas = SMOKE_ALPHAS if smoke else ALPHAS
    trials = _completed_from_shards(root, "trials", trial_key); baselines = _completed_from_shards(root, "alpha_zero", baseline_key)
    expected = len(test)*len(DIRECTIONS)*len(layers)*len(alphas); expected_baselines = len(test)*len(layers)
    if len(trials) != expected or len(baselines) != expected_baselines: raise ValueError(f"Shard merge incomplete: trials={len(trials)}/{expected}, baselines={len(baselines)}/{expected_baselines}")
    if any(row.get("config_fingerprint") != config_fingerprint for row in [*trials.values(), *baselines.values()]): raise ValueError("Merged config fingerprint mismatch")
    atomic_jsonl(root / "artifacts/trials/trials.jsonl", [trials[k] for k in sorted(trials)]); atomic_jsonl(root / "artifacts/trials/alpha_zero_baselines.jsonl", [baselines[k] for k in sorted(baselines)])
    return {"trial_count": expected, "alpha_zero_baseline_count": expected_baselines}


def run_steering(*, output_root: Path = RESULTS_ROOT, smoke: bool = False, resume: bool = False, num_gpus: int = 1, worker_id: int | None = None) -> dict[str, Any]:
    if num_gpus not in (1, 2): raise ValueError("--num-gpus must be 1 or 2")
    root = ensure_layout(output_root); config = _semantic_config(root, smoke=smoke)
    if worker_id is not None:
        stored = json.loads((root / "progress/run_config.json").read_text()); expected = canonical_hash(config)
        if stored.get("fingerprint") != expected: raise ValueError("Worker semantic config fingerprint mismatch")
        return _worker(root, worker_id, num_gpus, smoke=smoke, resume=True, config_fingerprint=expected)
    fingerprint = check_fingerprint(root / "progress/run_config.json", config, resume=resume)
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus: raise RuntimeError(f"Requested {num_gpus} GPUs, visible={torch.cuda.device_count()}")
    started = time.time(); processes = []
    for worker in range(num_gpus):
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(worker)
        command = [sys.executable, "-m", "dp_SA.confidence_steering.run", "--output-root", str(root), "--num-gpus", str(num_gpus), "--worker-id", str(worker)]
        if smoke: command.append("--smoke")
        if resume: command.append("--resume")
        processes.append(subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], env=env))
    codes = [p.wait() for p in processes]
    if any(codes): raise RuntimeError(f"Steering worker failure: {codes}")
    merged = _merge(root, smoke=smoke, config_fingerprint=fingerprint)
    forwards = sum(json.loads((root / f"progress/worker_{w}.json").read_text()).get("new_gpu_forwards", 0) for w in range(num_gpus) if (root / f"progress/worker_{w}.json").is_file())
    # Worker progress is also reconstructed below when subprocess stdout is unavailable.
    if forwards == 0 and not resume: forwards = len(load_jsonl(root / "artifacts/trials/alpha_zero_baselines.jsonl")) + sum(float(r["alpha"]) != 0 for r in load_jsonl(root / "artifacts/trials/trials.jsonl"))
    result = {"status": "complete", "smoke_only": smoke, **merged, "new_gpu_forwards": forwards, "resumed_noop": forwards == 0, "config_fingerprint": fingerprint, "elapsed_seconds": time.time()-started}
    atomic_json(root / "progress/run.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--num-gpus", type=int, choices=(1,2), default=1); parser.add_argument("--worker-id", type=int)
    args = parser.parse_args(argv); root = Path(args.output_root) if args.output_root else RESULTS_ROOT
    result = run_steering(output_root=root, smoke=args.smoke, resume=args.resume, num_gpus=args.num_gpus, worker_id=args.worker_id)
    if args.worker_id is not None: atomic_json(root / f"progress/worker_{args.worker_id}.json", result)
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
