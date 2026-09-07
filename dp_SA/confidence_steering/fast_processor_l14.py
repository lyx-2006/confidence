from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch
import transformers

from confidence_test.runtime_imports import load_runtime
from dp_SA.checkpoint_steering.run import class_margin
from dp_SA.positions import locate_phase1_positions
from dp_SA.soft_score import class_token_ids
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import AdditiveActivationHook, model_input_device, resolve_language_modules, run_hooked_forward

from .analyze import family_draws, summarize
from .config import BOOTSTRAP_REPEATS, HIDDEN_DEFINITION, INFERENCE_PATH, MODEL_PATH, PANL_LAYER, PANL_POSITION, SEED, STEERING_POSITION
from .core import HiddenResolver
from .io_utils import append_jsonl, array_hash, atomic_csv, atomic_json, atomic_jsonl, atomic_text, canonical_hash, load_jsonl, semantic_fingerprint, sha256_file, stable_shard
from .processor import enforce_fast_image_processor, processor_identity
from .random_sa_null import NATURAL_FORMAL_ROOT, model_processor_hashes
from .run import _messages, _predict, _score


EXPERIMENT = "all_fast_l14_reproduction"
LAYER = 14
DOSE = 2.0
DIRECTIONS = ("confidence_raw", "confidence_parallel_sa", "confidence_perp_sa_natural_scale")
ENDPOINTS = ("delta_confidence_LAT_immediate", "delta_panl_probe_sa", "delta_final_soft_sa")
OUTPUT_ROOT = Path(__file__).resolve().parent / "output/all_fast_l14"
SMOKE_ROOT = Path(__file__).resolve().parent / "output/all_fast_l14_smoke"


def paths(root: Path) -> dict[str, Path]:
    return {"artifacts": root / "artifacts", "clean": root / "artifacts/clean", "trials": root / "artifacts/trials", "diagnostics": root / "artifacts/diagnostics", "tables": root / "tables", "figures": root / "figures", "progress": root / "progress"}


def ensure_layout(root: Path) -> dict[str, Path]:
    result = paths(root)
    for value in result.values(): value.mkdir(parents=True, exist_ok=True)
    return result


def code_hashes() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    selected = (Path(__file__).resolve(), Path(__file__).resolve().parent / "processor.py", Path(__file__).resolve().parent / "run.py", repo / "qwen-2.5-vl/inference.py", repo / "layer_metacognition/model_adapter.py", repo / "layer_metacognition/conversation_builder.py", repo / "dp_SA/positions.py", repo / "dp_SA/soft_score.py")
    return {str(path.relative_to(repo)): sha256_file(path) for path in selected}


def _processor_source_audit(source_root: Path, root: Path) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    components = (
        ("panl_information", repo / "dp_SA/panl_information/capture.py"),
        ("checkpoint_steering", repo / "dp_SA/checkpoint_steering/capture.py"),
        ("unimodal_confidence_capture", repo / "dp_SA/unimodal_logit_confidence/capture.py"),
        ("confidence_steering_runtime", repo / "dp_SA/confidence_steering/run.py"),
        ("random_null_runtime", repo / "dp_SA/confidence_steering/random_sa_null.py"),
    )
    rows = []
    for name, path in components:
        text = path.read_text()
        forces_slow = "enforce_frozen_image_processor" in text or "Qwen2VLImageProcessor.from_pretrained" in text
        forces_fast = "enforce_fast_image_processor" in text or "Qwen2VLImageProcessorFast" in text
        policy = "explicit_slow" if forces_slow else "explicit_fast" if forces_fast else "implicit_AutoProcessor"
        rows.append({"component": name, "path": str(path.relative_to(repo)), "sha256": sha256_file(path), "loads_qwen_inference": "QwenVLInference" in text, "explicit_fast": forces_fast, "explicit_slow": forces_slow, "processor_policy": policy, "historical_runtime_class_recorded": False})
    reuse = [json.loads(line) for line in (repo / "dp_SA/unimodal_logit_confidence/output/results/confidence_probe/artifacts/hidden/reuse_manifest.jsonl").open()]
    relevant = Counter()
    for row in reuse:
        for key in ("P1_LAT__L14", "P1_PANL__L18"):
            source = row.get("cell_sources", {}).get(key)
            relevant[(key, source["source"] if source else "capture_delta")] += 1
    source_counts = [{"hidden_key": key, "source": source, "cell_count": count} for (key, source), count in sorted(relevant.items())]
    runtime_policies = {row["component"]: row["processor_policy"] for row in rows}
    audit = {"status": "passed_with_historical_runtime_identity_limitation", "finding": "direction/probe capture paths use implicit AutoProcessor; current runtime policy is reported per component", "runtime_policies": runtime_policies, "transformers_version_now": transformers.__version__, "current_default_is_fast": True, "historical_runtime_image_processor_class_was_not_persisted": True, "therefore_source_fast_status": "source-code inference pending exact hidden parity", "component_rows": rows, "relevant_hidden_source_counts": source_counts, "source_root": str(source_root.resolve())}
    atomic_csv(paths(root)["diagnostics"] / "processor_source_audit.csv", rows + source_counts)
    atomic_json(paths(root)["diagnostics"] / "processor_source_audit.json", audit)
    return audit


def _load_runtime():
    runtime = load_runtime(INFERENCE_PATH); inference = runtime.QwenVLInference(str(MODEL_PATH)); enforce_fast_image_processor(inference.processor)
    identity = processor_identity(inference.processor)
    if not identity["is_fast"] or not identity["image_processor_class"].endswith("Qwen2VLImageProcessorFast"): raise RuntimeError(f"Runtime is not explicitly Fast: {identity}")
    modules = resolve_language_modules(inference.model); tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    return inference, modules, tokenizer, class_token_ids(tokenizer), model_input_device(inference), identity


def _locate(inference: Any, tokenizer: Any, row: dict[str, Any], device: torch.device):
    messages = _messages(row)
    from dp_SA.prompts import SA_PREFILL
    rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
    inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=device)
    located = locate_phase1_positions(tokenizer, rendered, inputs, str(row["phase0_raw_answer"]))
    for name in (STEERING_POSITION, PANL_POSITION, "P1_SAC"):
        if int(located[name]["processed_index"]) != int(row["positions"][name]["processed_index"]): raise ValueError(f"Fast position mismatch: {row['case_id']} {name}")
    return inputs, located, rendered


def _read_shards(directory: Path, pattern: str, key_fn) -> dict[str, dict[str, Any]]:
    output = {}
    for path in sorted(directory.glob(pattern)):
        for row in load_jsonl(path, repair_trailing=True):
            key = key_fn(row)
            if key in output and row != output[key]: raise ValueError(f"Conflicting duplicate: {key}")
            output[key] = row
    return output


def _clean_worker(source_root: Path, root: Path, worker: int, num_gpus: int, fingerprint: str) -> dict[str, Any]:
    manifest = load_jsonl(paths(root)["artifacts"] / "runtime_manifest.jsonl"); selected = [row for row in manifest if stable_shard(str(row["case_id"]), num_gpus) == worker]
    completed = _read_shards(paths(root)["clean"], "clean.shard_*.jsonl", lambda row: row["case_id"]); pending = [row for row in selected if row["case_id"] not in completed]
    if not pending: return {"new_forwards": 0, "resumed_noop": True}
    inference, modules, tokenizer, ids, device, identity = _load_runtime(); resolver = HiddenResolver()
    lat_probe = joblib.load(source_root / "artifacts/probes/confidence_gap__P1_LAT__L14__full.joblib")["model"]; panl_probe = joblib.load(source_root / "artifacts/probes/final_sa__P1_PANL__L18__full.joblib")["model"]
    old_trials = load_jsonl(source_root / "artifacts/trials/main_trials.jsonl"); old_by_case = {}
    for trial in sorted(old_trials, key=lambda value: (str(value["case_id"]), str(value["direction"]), int(value["layer"]), float(value["alpha"]))): old_by_case.setdefault(str(trial["case_id"]), trial)
    out = paths(root)["clean"] / f"clean.shard_{worker}.jsonl"; forwards = 0
    for row in pending:
        inputs, located, rendered = _locate(inference, tokenizer, row, device); lat = int(located[STEERING_POSITION]["processed_index"]); panl = int(located[PANL_POSITION]["processed_index"]); sac = int(located["P1_SAC"]["processed_index"])
        forward = run_hooked_forward(inference.model, inputs, modules, {STEERING_POSITION: lat, PANL_POSITION: panl}, logits_positions=[sac]); forwards += 1
        lat_hidden = forward.hidden_by_name[STEERING_POSITION][LAYER].detach().float().cpu().numpy().astype(np.float32); panl_hidden = forward.hidden_by_name[PANL_POSITION][PANL_LAYER].detach().float().cpu().numpy().astype(np.float32); score = _score(forward, sac, ids)
        stored_lat = resolver.load(str(row["case_id"]), "P1_LAT__L14"); stored_panl = resolver.load(str(row["case_id"]), "P1_PANL__L18")
        lat_equal = bool(np.array_equal(lat_hidden.astype(np.float16), stored_lat.astype(np.float16))); panl_equal = bool(np.array_equal(panl_hidden.astype(np.float16), stored_panl.astype(np.float16)))
        if not lat_equal or not panl_equal: raise ValueError(f"Explicit Fast hidden does not reproduce direction/probe capture: {row['case_id']} LAT={lat_equal} PANL={panl_equal}")
        npz = paths(root)["clean"] / f"{row['case_id']}.npz"; from .io_utils import atomic_npz; atomic_npz(npz, {"lat": lat_hidden, "panl": panl_hidden})
        old = old_by_case[str(row["case_id"])]; fast_lat_conf = _predict(lat_probe, lat_hidden); fast_panl_sa = _predict(panl_probe, panl_hidden); fast_final = float(score["soft_sa_image_score"])
        old_hidden_path = source_root / str(old["panl_hidden_file"]); old_panl = None
        if old_hidden_path.is_file():
            with np.load(old_hidden_path) as payload:
                if "clean__P1_PANL__L18" in payload: old_panl = np.asarray(payload["clean__P1_PANL__L18"], np.float32)
        panl_delta_norm = float(np.linalg.norm(panl_hidden.astype(np.float64) - old_panl.astype(np.float64))) if old_panl is not None else None
        result = {"case_id": str(row["case_id"]), "family_id": str(row["family_id"]), "item_id": str(row["item_id"]), "fixed_answer": str(row["phase0_normalized_answer"]), "processor_identity": identity, "rendered_prompt_sha256": __import__('hashlib').sha256(rendered.encode()).hexdigest(), "lat_hidden_hash": array_hash(lat_hidden), "panl_hidden_hash": array_hash(panl_hidden), "stored_lat_float16_parity": lat_equal, "stored_panl_float16_parity": panl_equal, "clean_lat_confidence": fast_lat_conf, "clean_panl_sa": fast_panl_sa, "clean_final_sa": fast_final, "clean_sa_logits": score["class_logits"], "clean_sa_probabilities": score["class_probabilities"], "clean_hard_sa": int(score["argmax_hard_class"]), "clean_class_margin": class_margin(np.asarray(score["class_logits"]), int(score["argmax_hard_class"])), "old_slow_lat_hidden_hash": old["activation_before_hash"], "old_slow_panl_hidden_hash": old["panl_clean_hidden_hash"], "fast_slow_lat_hash_equal": array_hash(lat_hidden) == old["activation_before_hash"], "fast_slow_panl_hash_equal": array_hash(panl_hidden) == old["panl_clean_hidden_hash"], "fast_minus_slow_lat_confidence": fast_lat_conf - float(old["clean_lat_confidence_probe"]), "fast_minus_slow_panl_sa": fast_panl_sa - float(old["clean_panl_sa_probe"]), "fast_minus_slow_final_sa": fast_final - float(old["clean_soft_sa"]), "fast_slow_panl_hidden_delta_norm": panl_delta_norm, "fast_slow_logits_equal": bool(np.array_equal(np.asarray(score["class_logits"]), np.asarray(old["clean_sa_logits"]))), "hidden_file": str(npz.relative_to(root)), "fingerprint": fingerprint, "status": "passed"}
        append_jsonl(out, result); completed[result["case_id"]] = result
    return {"new_forwards": forwards, "resumed_noop": forwards == 0}


def trial_key(row: dict[str, Any]) -> str: return f'{row["case_id"]}|{row["direction"]}|a{float(row["alpha"]):g}'


def _steer_worker(source_root: Path, root: Path, worker: int, num_gpus: int, fingerprint: str) -> dict[str, Any]:
    manifest = load_jsonl(paths(root)["artifacts"] / "runtime_manifest.jsonl"); selected = [row for row in manifest if stable_shard(str(row["case_id"]), num_gpus) == worker]
    completed = _read_shards(paths(root)["trials"], "trials.shard_*.jsonl", trial_key); expected = {trial_key({"case_id": row["case_id"], "direction": direction, "alpha": alpha}) for row in selected for direction in DIRECTIONS for alpha in (-DOSE, DOSE)}
    if expected <= set(completed): return {"new_forwards": 0, "resumed_noop": True}
    clean = {row["case_id"]: row for row in load_jsonl(paths(root)["clean"] / "clean.jsonl")}; inference, modules, tokenizer, ids, device, identity = _load_runtime()
    lat_probe = joblib.load(source_root / "artifacts/probes/confidence_gap__P1_LAT__L14__full.joblib")["model"]; panl_probe = joblib.load(source_root / "artifacts/probes/final_sa__P1_PANL__L18__full.joblib")["model"]
    metadata = json.loads((source_root / "artifacts/directions/vector_metadata.json").read_text()); meta = {(row["recipient_answer"], row["direction"]): row for row in metadata["vectors"] if int(row["layer"]) == LAYER}
    vectors_file = np.load(source_root / "artifacts/directions/P1_LAT__L14.npz"); out = paths(root)["trials"] / f"trials.shard_{worker}.jsonl"; forwards = 0
    for row in selected:
        case = str(row["case_id"]); baseline = clean[case]; inputs, located, _rendered = _locate(inference, tokenizer, row, device); lat = int(located[STEERING_POSITION]["processed_index"]); panl = int(located[PANL_POSITION]["processed_index"]); sac = int(located["P1_SAC"]["processed_index"]); sequence = int(inputs.input_ids.shape[1]); answer = str(row["phase0_normalized_answer"])
        with np.load(root / baseline["hidden_file"]) as payload: clean_lat = np.asarray(payload["lat"], np.float32); clean_panl = np.asarray(payload["panl"], np.float32)
        for direction in DIRECTIONS:
            info = meta[answer, direction]; vector = np.asarray(vectors_file[info["scaled_key"]], np.float32)
            if array_hash(vector) != info["scaled_hash"]: raise ValueError("Vector hash mismatch")
            for alpha in (-DOSE, DOSE):
                proto = {"case_id": case, "direction": direction, "alpha": alpha}
                if trial_key(proto) in completed: continue
                hook = AdditiveActivationHook(modules, layer_index=LAYER, target_position=lat, steering_vector=torch.from_numpy(vector) * alpha, prefill_sequence_length=sequence, injection_site="block_output")
                with hook: forward = run_hooked_forward(inference.model, inputs, modules, {PANL_POSITION: panl}, logits_positions=[sac])
                forwards += 1; diagnostics = hook.diagnostics(); score = _score(forward, sac, ids); steered_panl = forward.hidden_by_name[PANL_POSITION][PANL_LAYER].detach().float().cpu().numpy().astype(np.float32)
                if diagnostics["hook_call_count"] != 1 or diagnostics["steering_applied_count"] != 1 or array_hash(hook.h_before.numpy()) != baseline["lat_hidden_hash"]: raise ValueError(f"Fast hook/baseline parity failed: {trial_key(proto)}")
                result = {**proto, "layer": LAYER, "family_id": str(row["family_id"]), "item_id": str(row["item_id"]), "condition": str(row["condition"]), "fixed_answer": answer, "processor_mode": "explicit_fast", "processor_identity": identity, "vector_hash": array_hash(vector), "clean_lat_confidence": float(baseline["clean_lat_confidence"]), "steered_lat_confidence": _predict(lat_probe, hook.h_after.numpy()), "delta_confidence_LAT_immediate": _predict(lat_probe, hook.h_after.numpy()) - float(baseline["clean_lat_confidence"]), "clean_panl_sa": float(baseline["clean_panl_sa"]), "steered_panl_sa": _predict(panl_probe, steered_panl), "delta_panl_probe_sa": _predict(panl_probe, steered_panl) - float(baseline["clean_panl_sa"]), "clean_final_soft_sa": float(baseline["clean_final_sa"]), "steered_final_soft_sa": float(score["soft_sa_image_score"]), "delta_final_soft_sa": float(score["soft_sa_image_score"]) - float(baseline["clean_final_sa"]), "steered_sa_logits": score["class_logits"], "hook_hit_count": 1, "actual_injection_norm": float(np.linalg.norm(hook.h_after.numpy().astype(np.float64) - hook.h_before.numpy().astype(np.float64))), "fingerprint": fingerprint, "status": "completed"}
                append_jsonl(out, result); completed[trial_key(result)] = result
    vectors_file.close(); return {"new_forwards": forwards, "resumed_noop": forwards == 0}


def _spawn(stage: str, source_root: Path, root: Path, num_gpus: int, fingerprint: str) -> None:
    repo = Path(__file__).resolve().parents[2]; processes = []
    for worker in range(num_gpus):
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(worker)
        command = [sys.executable, "-m", "dp_SA.confidence_steering.fast_processor_l14", "--worker-stage", stage, "--source-root", str(source_root), "--output-root", str(root), "--worker", str(worker), "--num-gpus", str(num_gpus), "--fingerprint", fingerprint]
        processes.append(subprocess.Popen(command, cwd=repo, env=env))
    codes = [process.wait() for process in processes]
    if any(codes): raise RuntimeError(f"Fast L14 {stage} worker failure: {codes}")


def _analyze(source_root: Path, root: Path, fingerprint: str) -> dict[str, Any]:
    trials = load_jsonl(paths(root)["trials"] / "trials.jsonl"); draws, draw_hash = family_draws(trials, BOOTSTRAP_REPEATS); effects = []
    for endpoint in ENDPOINTS:
        for direction in DIRECTIONS:
            selected = [row for row in trials if row["direction"] == direction]; by_case = {}
            for row in selected: by_case.setdefault(row["case_id"], {})[float(row["alpha"])] = row
            rows = []
            for values in by_case.values():
                plus, minus = values[DOSE], values[-DOSE]; rows.append({**{key: plus[key] for key in ("case_id", "family_id", "item_id", "condition", "fixed_answer")}, "effect": (float(plus[endpoint]) - float(minus[endpoint])) / 2.0})
            summary = summarize(rows, "effect", "answer_equal_macro", draws); effects.append({"endpoint": endpoint, "direction": direction, "dose": DOSE, "group": "answer_equal_macro", **summary})
    atomic_csv(paths(root)["tables"] / "all_fast_l14_symmetric_effects.csv", effects)
    old = load_jsonl(source_root / "artifacts/trials/main_trials.jsonl"); comparisons = []
    for effect in effects:
        endpoint, direction = effect["endpoint"], effect["direction"]; field = endpoint
        selected = [row for row in old if int(row["layer"]) == LAYER and row["direction"] == direction and abs(float(row["alpha"])) == DOSE]; by_case = {}
        for row in selected: by_case.setdefault(row["case_id"], {})[float(row["alpha"])] = row
        rows = [{**{key: values[DOSE][key] for key in ("case_id", "family_id", "item_id", "condition", "fixed_answer")}, "effect": (float(values[DOSE][field]) - float(values[-DOSE][field])) / 2.0} for values in by_case.values()]
        slow = summarize(rows, "effect", "answer_equal_macro", draws)
        comparisons.append({"endpoint": endpoint, "direction": direction, "fast_effect": effect["mean_delta"], "fast_ci_low": effect["ci95_low"], "fast_ci_high": effect["ci95_high"], "old_slow_effect": slow["mean_delta"], "old_slow_ci_low": slow["ci95_low"], "old_slow_ci_high": slow["ci95_high"], "same_sign": int(np.sign(effect["mean_delta"]) == np.sign(slow["mean_delta"]))})
    atomic_csv(paths(root)["tables"] / "all_fast_vs_old_slow.csv", comparisons)
    atomic_text(root / "summary.md", "# 全 Fast L14 复现\n\n本实验显式固定 `Qwen2VLImageProcessorFast`。方向与probe仅在100个正式case的Fast clean hidden与历史capture float16逐case一致后复用。旧Slow结果保留，不立即作废。\n")
    result = {"status": "complete", "fingerprint": fingerprint, "bootstrap_draw_fingerprint": draw_hash, "effect_rows": len(effects), "comparison_rows": len(comparisons)}; atomic_json(paths(root)["progress"] / "analyze.json", result); return result


def _shift_audit(clean_rows: Sequence[dict[str, Any]], root: Path) -> dict[str, Any]:
    fields = ("fast_minus_slow_lat_confidence", "fast_minus_slow_panl_sa", "fast_minus_slow_final_sa", "fast_slow_panl_hidden_delta_norm")
    summaries = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in clean_rows if row[field] is not None], np.float64)
        summaries[field] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "min": float(values.min()), "max": float(values.max()), "positive": int((values > 0).sum()), "negative": int((values < 0).sum()), "zero": int((values == 0).sum()), "unique_rounded_1e12": len(set(np.round(values, 12)))}
    scalar_nonuniform = any(value["sd"] > 1e-8 and value["unique_rounded_1e12"] > 1 for key, value in summaries.items() if key != "fast_slow_panl_hidden_delta_norm")
    hash_patterns_nonuniform = not all(row["fast_slow_lat_hash_equal"] for row in clean_rows) or not all(row["fast_slow_panl_hash_equal"] for row in clean_rows)
    result = {"status": "nonuniform_shift" if scalar_nonuniform or hash_patterns_nonuniform else "uniform_or_no_shift", "case_count": len(clean_rows), "all_match_historical_capture_float16": all(row["stored_lat_float16_parity"] and row["stored_panl_float16_parity"] for row in clean_rows), "fast_slow_lat_hash_equal_count": sum(row["fast_slow_lat_hash_equal"] for row in clean_rows), "fast_slow_panl_hash_equal_count": sum(row["fast_slow_panl_hash_equal"] for row in clean_rows), "fast_slow_logits_equal_count": sum(row["fast_slow_logits_equal"] for row in clean_rows), "scalar_nonuniform": scalar_nonuniform, "hash_patterns_nonuniform": hash_patterns_nonuniform, "field_summaries": summaries, "decision_rule": "rerun full all-Fast L14 when smoke shift is nonuniform"}
    atomic_json(paths(root)["diagnostics"] / "fast_vs_slow_shift_audit.json", result); atomic_csv(paths(root)["diagnostics"] / "fast_vs_slow_case_differences.csv", clean_rows); return result


def run_pipeline(*, source_root: Path = NATURAL_FORMAL_ROOT, output_root: Path = OUTPUT_ROOT, num_gpus: int = 1, resume: bool = False, case_limit: int | None = None) -> dict[str, Any]:
    source_root = source_root.resolve(); output_root = output_root.resolve(); ensure_layout(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not resume and (paths(output_root)["progress"] / "config.json").exists(): raise FileExistsError(output_root)
    audit = _processor_source_audit(source_root, output_root); model_hash, processor_hash = model_processor_hashes(); source_manifest = source_root / "artifacts/manifests/runtime_manifest.jsonl"
    manifest_rows = load_jsonl(source_manifest)
    if case_limit is not None:
        families = sorted({str(row["family_id"]) for row in manifest_rows}); chosen = set(families[: math.ceil(case_limit / 2)]); manifest_rows = [row for row in manifest_rows if str(row["family_id"]) in chosen]
        if len(manifest_rows) != case_limit: raise ValueError(f"Family-complete case limit produced {len(manifest_rows)} rows, expected {case_limit}")
    manifest = paths(output_root)["artifacts"] / "runtime_manifest.jsonl"; atomic_jsonl(manifest, sorted(manifest_rows, key=lambda row: str(row["case_id"])))
    case_count = len(manifest_rows); config = {"format_version": 1, "experiment": EXPERIMENT, "processor_mode": "explicit_fast", "layer": LAYER, "dose": DOSE, "directions": list(DIRECTIONS), "case_count": case_count, "case_selection": "all" if case_limit is None else "first complete families sorted by family_id", "source_runtime_manifest_sha256": sha256_file(source_manifest), "runtime_manifest_sha256": sha256_file(manifest), "source_vector_sha256": sha256_file(source_root / "artifacts/directions/P1_LAT__L14.npz"), "source_vector_metadata_sha256": sha256_file(source_root / "artifacts/directions/vector_metadata.json"), "probe_sha256": {name: sha256_file(source_root / f"artifacts/probes/{name}__full.joblib") for name in ("confidence_gap__P1_LAT__L14", "final_sa__P1_PANL__L18")}, "model_hashes": model_hash, "processor_hashes": processor_hash, "transformers_version": transformers.__version__, "code_hashes": code_hashes(), "seed": SEED, "bootstrap_repeats": BOOTSTRAP_REPEATS}
    fingerprint = semantic_fingerprint(paths(output_root)["progress"] / "config.json", config, resume=resume)
    _spawn("clean", source_root, output_root, num_gpus, fingerprint); clean = _read_shards(paths(output_root)["clean"], "clean.shard_*.jsonl", lambda row: row["case_id"])
    if len(clean) != case_count or not all(row["stored_lat_float16_parity"] and row["stored_panl_float16_parity"] for row in clean.values()): raise RuntimeError("Fast clean/capture parity barrier failed")
    clean_rows = [clean[key] for key in sorted(clean)]; atomic_jsonl(paths(output_root)["clean"] / "clean.jsonl", clean_rows); shift = _shift_audit(clean_rows, output_root); _spawn("steer", source_root, output_root, num_gpus, fingerprint)
    trials = _read_shards(paths(output_root)["trials"], "trials.shard_*.jsonl", trial_key)
    expected_trials = case_count * len(DIRECTIONS) * 2
    if len(trials) != expected_trials: raise RuntimeError(f"Fast trial coverage incomplete: {len(trials)}/{expected_trials}")
    atomic_jsonl(paths(output_root)["trials"] / "trials.jsonl", [trials[key] for key in sorted(trials)]); analyzed = _analyze(source_root, output_root, fingerprint)
    result = {"status": "complete", "fingerprint": fingerprint, "processor_audit": audit["status"], "clean_parity_cases": len(clean), "fast_slow_shift": shift, "trial_count": len(trials), "planned_forwards": case_count * 7, "analyze": analyzed}; atomic_json(paths(output_root)["progress"] / "run.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-root", type=Path, default=NATURAL_FORMAL_ROOT); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--num-gpus", type=int, choices=(1, 2), default=1); parser.add_argument("--resume", action="store_true"); parser.add_argument("--case-limit", type=int); parser.add_argument("--worker-stage", choices=("clean", "steer")); parser.add_argument("--worker", type=int); parser.add_argument("--fingerprint")
    args = parser.parse_args(argv)
    if args.worker_stage: result = _clean_worker(args.source_root, args.output_root, args.worker, args.num_gpus, args.fingerprint) if args.worker_stage == "clean" else _steer_worker(args.source_root, args.output_root, args.worker, args.num_gpus, args.fingerprint)
    else: result = run_pipeline(source_root=args.source_root, output_root=args.output_root, num_gpus=args.num_gpus, resume=args.resume, case_limit=args.case_limit)
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
