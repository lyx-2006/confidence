from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from confidence_test.runtime_imports import load_runtime
from dp_SA.checkpoint_steering.run import class_margin
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from dp_SA.soft_score import class_token_ids
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import AdditiveActivationHook, model_input_device, resolve_language_modules, run_hooked_forward

from .analyze import _mean, family_draws, summarize, symmetric_effect
from .config import (
    BOOTSTRAP_REPEATS, CANONICAL_COLORS, HIDDEN_DEFINITION, HIDDEN_SIZE,
    INFERENCE_PATH, MODEL_PATH, PANL_LAYER, PANL_POSITION, SEED,
    STEERING_POSITION,
)
from .core import HiddenResolver, raw_gradient
from .io_utils import (
    append_jsonl, array_hash, atomic_csv, atomic_json, atomic_jsonl, atomic_npz,
    atomic_text, canonical_hash, load_jsonl, semantic_fingerprint, sha256_file,
    stable_shard,
)
from .processor import PROCESSOR_MODE, enforce_fast_image_processor
from .run import _messages, _predict, _score, trial_key
from .run_spec import normalize_run_spec


RANDOM_NULL_NAME = "random_sa_subspace_null"
RANDOM_NULL_LAYER = 14
RANDOM_NULL_DOSE = 2.0
CANDIDATE_POOL_SIZE = 200
SELECTED_CANDIDATES = 20
FORMAL_REPEATS = 20
SMOKE_REPEATS = 3
RANDOM_SMOKE_PARENT = Path(__file__).resolve().parent / "output/random_sa_subspace_null_smoke"
NATURAL_FORMAL_ROOT = Path(__file__).resolve().parent / "output/natural_decomposition"
NATURAL_SMOKE_PARENT = Path(__file__).resolve().parent / "output/orthogonal_smoke"
NATURAL_RUN_SPEC = normalize_run_spec(
    ["confidence_raw", "confidence_parallel_sa", "confidence_perp_sa_natural_scale"],
    [14, 16], [-2, -1, 0, 1, 2],
)
ENDPOINTS = (
    "delta_confidence_LAT_immediate",
    "delta_confidence_PANL_L18",
    "delta_panl_probe_sa",
    "delta_final_soft_sa",
)
GROUPS = ("answer_equal_macro", "family_micro", "all")
PROTECTED_RELATIVE_PATHS = (
    "artifacts/trials/main_trials.jsonl",
    "artifacts/directions/P1_LAT__L14.npz",
    "artifacts/directions/P1_LAT__L16.npz",
    "artifacts/directions/vector_metadata.json",
    "tables/steering_effects.csv",
    "tables/symmetric_effects.csv",
    "tables/local_slopes.csv",
    "tables/component_additivity.csv",
    "figures/final/confidence_raw.png",
    "figures/final/confidence_parallel_sa.png",
    "figures/final/confidence_perp_sa_natural_scale.png",
    "figures/panl/confidence_raw.png",
    "figures/panl/confidence_parallel_sa.png",
    "figures/panl/confidence_perp_sa_natural_scale.png",
    "summary.md",
    "progress/run.json",
    "progress/run_config.json",
    "progress/analyze.json",
    "progress/analyze_config.json",
)
HISTORICAL_CLEAN_FIELDS = (
    "clean_sa_logits", "clean_sa_probabilities", "clean_soft_sa",
    "clean_hard_sa", "clean_class_margin", "clean_panl_confidence_probe",
    "clean_panl_sa_probe", "panl_clean_hidden_hash", "config_fingerprint",
    "hidden_definition",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _random_paths(root: Path) -> dict[str, Path]:
    return {
        "directions": root / "artifacts/directions/random_sa_subspace_nulls",
        "diagnostics": root / "artifacts/diagnostics/random_sa_subspace_null",
        "trials": root / "artifacts/trials",
        "tables": root / "tables",
        "figures": root / "figures/diagnostics",
        "progress": root / "progress",
    }


def _ensure_random_layout(root: Path) -> dict[str, Path]:
    paths = _random_paths(root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def protected_hashes(root: Path) -> dict[str, str]:
    output = {}
    for relative in PROTECTED_RELATIVE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Protected main artifact is missing: {path}")
        output[relative] = sha256_file(path)
    return output


def _artifact_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in paths}


def model_processor_hashes() -> tuple[dict[str, str], dict[str, str]]:
    model_names = [f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)] + [
        "model.safetensors.index.json", "config.json",
    ]
    processor_names = [
        "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "chat_template.json",
    ]
    model = _artifact_hashes([MODEL_PATH / name for name in model_names])
    processor = _artifact_hashes([MODEL_PATH / name for name in processor_names])
    processor["processor.py"] = sha256_file(Path(__file__).resolve().parent / "processor.py")
    return model, processor


def random_code_hashes() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(), Path(__file__).resolve().parent / "run.py",
        Path(__file__).resolve().parent / "analyze.py",
        Path(__file__).resolve().parent / "run_pipeline.py",
        Path(__file__).resolve().parent / "processor.py",
        repo / "dp_SA/positions.py", repo / "dp_SA/soft_score.py",
        repo / "layer_metacognition/model_adapter.py",
        repo / "layer_metacognition/conversation_builder.py",
        Path(INFERENCE_PATH),
    )
    return {str(path.relative_to(repo)): sha256_file(path) for path in paths}


def random_protocol_material() -> dict[str, Any]:
    model_hash, processor_hash = model_processor_hashes()
    return {
        "format_version": 1, "processor_mode": PROCESSOR_MODE,
        "layer": RANDOM_NULL_LAYER, "dose": RANDOM_NULL_DOSE,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "selected_candidates": SELECTED_CANDIDATES, "seed": SEED,
        "seed_rule": "SeedSequence([42,14,recipient_index,candidate_id])",
        "global_distance_rule": "equal mean across 12 recipients; sort(global_distance,candidate_id)",
        "matching_hidden": "donor-excluded construction family-cells",
        "natural_run_spec_fingerprint": NATURAL_RUN_SPEC["fingerprint"],
        "model_files_sha256": model_hash,
        "processor_files_sha256": processor_hash,
        "code_hashes": random_code_hashes(),
    }


def random_null_key(row: dict[str, Any]) -> str:
    return f'{row["case_id"]}|null{int(row["null_replicate"]):03d}|L{int(row["layer"])}|a{float(row["alpha"]):g}'


def _random_basis_and_vector(
    raw_scaled: np.ndarray,
    retained_ratio: float,
    rank: int,
    *,
    recipient_index: int,
    candidate_id: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    raw = np.asarray(raw_scaled, dtype=np.float64)
    if raw.ndim != 1 or not np.isfinite(raw).all() or np.linalg.norm(raw) <= 0:
        raise ValueError("Invalid raw confidence vector")
    if not 0 < retained_ratio < 1 or rank < 1 or rank >= raw.size:
        raise ValueError("Invalid retained ratio or subspace rank")
    seed_parts = [SEED, RANDOM_NULL_LAYER, int(recipient_index), int(candidate_id)]
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    vhat = raw / np.linalg.norm(raw)
    u = rng.standard_normal(raw.size)
    u -= vhat * float(vhat @ u)
    u /= np.linalg.norm(u)
    q1 = math.sqrt(1.0 - retained_ratio ** 2) * vhat + retained_ratio * u
    columns = [q1]
    extras: list[np.ndarray] = []
    for _ in range(1, rank):
        value = rng.standard_normal(raw.size)
        fixed = np.column_stack([vhat, u, *extras])
        value -= fixed @ (fixed.T @ value)
        value /= np.linalg.norm(value)
        extras.append(value)
        columns.append(value)
    basis, _ = np.linalg.qr(np.column_stack(columns), mode="reduced")
    residual = raw - basis @ (basis.T @ raw)
    residual -= basis @ (basis.T @ residual)
    return basis, residual, int(np.random.SeedSequence(seed_parts).generate_state(1, dtype=np.uint64)[0])


def _principal_angles(true_basis: np.ndarray, random_basis: np.ndarray) -> list[float]:
    singular = np.linalg.svd(np.asarray(true_basis, np.float64).T @ np.asarray(random_basis, np.float64), compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular, -1.0, 1.0))).tolist()


def candidate_metrics(
    raw_scaled: np.ndarray,
    true_vector: np.ndarray,
    true_basis: np.ndarray,
    natural_hidden: np.ndarray,
    confidence_gradient: np.ndarray,
    *,
    recipient_index: int,
    candidate_id: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    raw = np.asarray(raw_scaled, np.float64)
    true = np.asarray(true_vector, np.float64)
    rho = float(np.linalg.norm(true) / np.linalg.norm(raw))
    basis, vector64, candidate_seed = _random_basis_and_vector(
        raw, rho, int(true_basis.shape[1]), recipient_index=recipient_index,
        candidate_id=candidate_id,
    )
    vector32 = vector64.astype(np.float32)
    true_unit = true / np.linalg.norm(true)
    null_unit = vector64 / np.linalg.norm(vector64)
    true_sd = float(np.std(np.asarray(natural_hidden, np.float64) @ true_unit))
    null_sd = float(np.std(np.asarray(natural_hidden, np.float64) @ null_unit))
    true_dot = float(np.asarray(confidence_gradient, np.float64) @ true)
    null_dot = float(np.asarray(confidence_gradient, np.float64) @ vector64)
    retention = float(np.linalg.norm(vector64) / np.linalg.norm(raw))
    retention_error = abs(retention - rho)
    basis_error = float(np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1]))))
    orthogonality = float(np.max(np.abs(basis.T @ null_unit)))
    float32_norm_error = abs(float(np.linalg.norm(vector32)) - float(np.linalg.norm(vector64))) / float(np.linalg.norm(vector64))
    valid = bool(
        basis.shape == (raw.size, true_basis.shape[1])
        and vector32.shape == (raw.size,)
        and np.isfinite(basis).all() and np.isfinite(vector32).all()
        and basis_error <= 1e-8 and orthogonality <= 1e-5
        and retention_error <= 1e-6 and float32_norm_error <= 1e-5
        and true_sd > 0 and null_sd > 0 and true_dot != 0 and null_dot != 0
        and true_dot * null_dot > 0
    )
    distance = (
        abs(math.log(null_sd / true_sd)) + abs(math.log(abs(null_dot) / abs(true_dot)))
        if valid else math.inf
    )
    metrics = {
        "candidate_id": candidate_id, "candidate_seed": candidate_seed,
        "candidate_rank": int(true_basis.shape[1]), "subspace_rank": int(true_basis.shape[1]),
        "true_retained_ratio": rho, "null_retained_ratio": retention,
        "retained_ratio_error": retention_error,
        "true_vector_norm": float(np.linalg.norm(true)), "null_vector_norm": float(np.linalg.norm(vector64)),
        "true_natural_projection_sd": true_sd, "null_natural_projection_sd": null_sd,
        "alpha1_natural_sd": float(np.linalg.norm(vector64) / null_sd),
        "alpha2_natural_sd": float(2.0 * np.linalg.norm(vector64) / null_sd),
        "true_confidence_probe_dot": true_dot, "null_confidence_probe_dot": null_dot,
        "confidence_probe_dot_ratio": null_dot / true_dot,
        "cosine_with_raw_confidence": float(null_unit @ (raw / np.linalg.norm(raw))),
        "principal_angles_degrees": _principal_angles(true_basis, basis),
        "basis_orthogonality_error": basis_error,
        "max_random_basis_absolute_cosine": orthogonality,
        "float32_relative_norm_error": float32_norm_error,
        "basis_sha256": array_hash(basis), "vector_sha256": array_hash(vector32),
        "matching_distance": distance, "valid": valid,
    }
    return metrics, basis, vector32


def select_global_candidates(per_candidate: dict[int, list[dict[str, Any]]], count: int = SELECTED_CANDIDATES) -> tuple[list[int], list[dict[str, Any]]]:
    summaries = []
    expected_answers = set(CANONICAL_COLORS)
    for candidate_id in sorted(per_candidate):
        rows = per_candidate[candidate_id]
        answers = {str(row["recipient_answer"]) for row in rows}
        valid = answers == expected_answers and len(rows) == len(CANONICAL_COLORS) and all(bool(row["valid"]) for row in rows)
        distance = float(np.mean([float(row["matching_distance"]) for row in rows])) if valid else math.inf
        summaries.append({"candidate_id": candidate_id, "all_recipients_valid": valid, "global_matching_distance": distance, "selected_rank": None})
    eligible = sorted((row for row in summaries if row["all_recipients_valid"]), key=lambda row: (row["global_matching_distance"], row["candidate_id"]))
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} globally valid candidates; need {count}")
    selected = [int(row["candidate_id"]) for row in eligible[:count]]
    rank = {candidate_id: index for index, candidate_id in enumerate(selected, 1)}
    for row in summaries:
        row["selected_rank"] = rank.get(int(row["candidate_id"]))
    return selected, summaries


def _historical_clean_baselines(source_root: Path, expected_cases: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main = load_jsonl(source_root / "artifacts/trials/main_trials.jsonl")
    manifest = {str(row["case_id"]): row for row in load_jsonl(source_root / "artifacts/manifests/runtime_manifest.jsonl")}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in main:
        grouped[str(row["case_id"])].append(row)
    if len(grouped) != expected_cases or set(grouped) != set(manifest):
        raise ValueError("Main trials and runtime manifest case sets differ")
    expected_keys = {(direction, layer, alpha) for direction in NATURAL_RUN_SPEC["directions"] for layer in NATURAL_RUN_SPEC["layers"] for alpha in NATURAL_RUN_SPEC["alphas"]}
    baselines = []; audit = []
    for case_id in sorted(grouped):
        rows = grouped[case_id]
        actual_keys = {(row["direction"], int(row["layer"]), float(row["alpha"])) for row in rows}
        if len(rows) != 30 or actual_keys != expected_keys:
            raise ValueError(f"Incomplete main grid for clean reuse: {case_id}")
        unique_counts = {field: len({canonical_hash(row[field]) for row in rows}) for field in HISTORICAL_CLEAN_FIELDS}
        l14_rows = [row for row in rows if int(row["layer"]) == RANDOM_NULL_LAYER]
        unique_counts["clean_lat_confidence_probe_L14"] = len({canonical_hash(row["clean_lat_confidence_probe"]) for row in l14_rows})
        if any(value != 1 for value in unique_counts.values()):
            raise ValueError(f"Historical clean baseline mismatch: {case_id}: {unique_counts}")
        source = min(rows, key=lambda row: (str(row["direction"]), int(row["layer"]), float(row["alpha"])))
        source_l14 = min(l14_rows, key=lambda row: (str(row["direction"]), float(row["alpha"])))
        runtime = manifest[case_id]
        baseline = {
            "case_id": case_id, "canonical_source_key": trial_key(source),
            **{field: source[field] for field in HISTORICAL_CLEAN_FIELDS},
            "clean_lat_confidence_probe": source_l14["clean_lat_confidence_probe"],
            "phase1_prompt_hash": runtime["phase1_prompt_hash"],
            "phase0_answer_fingerprint": runtime["phase0_answer_fingerprint"],
            "manifest_positions": runtime["positions"],
            "manifest_position_sha256": canonical_hash(runtime["positions"]),
        }
        baselines.append(baseline)
        audit.append({"case_id": case_id, "source_row_count": len(rows), **{f"unique_{key}_count": value for key, value in unique_counts.items()}, "historical_parity_status": "passed", "selected_canonical_source_key": trial_key(source)})
    return baselines, audit


def _source_anchor(source_root: Path) -> dict[str, Any]:
    run = json.loads((source_root / "progress/run.json").read_text())
    analyze = json.loads((source_root / "progress/analyze.json").read_text())
    run_config = json.loads((source_root / "progress/run_config.json").read_text())
    material = json.loads((source_root / "artifacts/diagnostics/prelock_material.json").read_text())
    if run.get("status") != "complete" or analyze.get("status") != "complete" or material.get("run_spec") != NATURAL_RUN_SPEC or run_config.get("run_spec") != NATURAL_RUN_SPEC:
        raise ValueError("Source is not a complete natural-scale decomposition run")
    manifest = load_jsonl(source_root / "artifacts/manifests/runtime_manifest.jsonl")
    expected = 24 if bool(run.get("smoke_only")) else 100
    if len(manifest) != expected or int(run.get("main_trial_count", -1)) != expected * 30:
        raise ValueError("Natural-scale source cardinality mismatch")
    vector_path = source_root / "artifacts/directions/P1_LAT__L14.npz"
    probes = [source_root / f"artifacts/probes/{name}__full.joblib" for name in (
        "confidence_gap__P1_LAT__L14", "confidence_gap__P1_PANL__L18", "final_sa__P1_PANL__L18",
    )]
    return {
        "case_count": expected, "smoke_source": bool(run.get("smoke_only")),
        "main_run_fingerprint": run["config_fingerprint"],
        "main_trials_sha256": sha256_file(source_root / "artifacts/trials/main_trials.jsonl"),
        "runtime_manifest_sha256": sha256_file(source_root / "artifacts/manifests/runtime_manifest.jsonl"),
        "vector_file_sha256": sha256_file(vector_path),
        "probe_files_sha256": {path.name: sha256_file(path) for path in probes},
        "construction_cells_sha256": sha256_file(source_root / "artifacts/manifests/construction_cells.jsonl"),
        "construction_records_sha256": sha256_file(source_root / "artifacts/manifests/construction_records.jsonl"),
        "natural_run_spec_fingerprint": NATURAL_RUN_SPEC["fingerprint"],
    }


def prepare_random_null(source_root: Path, output_root: Path, *, resume: bool) -> dict[str, Any]:
    source_root = source_root.resolve(); output_root = output_root.resolve(); paths = _ensure_random_layout(output_root)
    source_anchor = _source_anchor(source_root)
    before = protected_hashes(source_root)
    protocol_material = random_protocol_material()
    static_lock_fingerprint = canonical_hash(protocol_material)
    config = {
        **protocol_material, "static_lock_fingerprint": static_lock_fingerprint,
        "source_anchor": source_anchor, "protected_before": before,
    }
    fingerprint = semantic_fingerprint(paths["progress"] / "random_sa_subspace_null_prepare_config.json", config, resume=resume)
    progress = paths["progress"] / "random_sa_subspace_null_prepare.json"
    if resume and progress.is_file():
        previous = json.loads(progress.read_text())
        if previous.get("status") == "complete" and previous.get("config_fingerprint") == fingerprint:
            if protected_hashes(source_root) != before:
                raise ValueError("Protected source artifacts changed during resume")
            return {**previous, "resumed_noop": True}

    cells = load_jsonl(source_root / "artifacts/manifests/construction_cells.jsonl")
    resolver = HiddenResolver(); hidden = {}
    for cell in cells:
        value = np.stack([resolver.load(case_id, "P1_LAT__L14") for case_id in cell["case_ids"]]).mean(axis=0, dtype=np.float32)
        if array_hash(value) != cell["hidden_hashes"]["L14"]:
            raise ValueError(f"Construction cell hidden hash mismatch: {cell['array_key']}")
        hidden[cell["array_key"]] = value
    vector_metadata = json.loads((source_root / "artifacts/directions/vector_metadata.json").read_text())["vectors"]
    meta = {(int(row["layer"]), row["recipient_answer"], row["direction"]): row for row in vector_metadata}
    confidence_model = joblib.load(source_root / "artifacts/probes/confidence_gap__P1_LAT__L14__full.joblib")["model"]
    gradient = raw_gradient(confidence_model)
    per_candidate: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with np.load(source_root / "artifacts/directions/P1_LAT__L14.npz") as payload:
        for recipient_index, recipient in enumerate(CANONICAL_COLORS):
            raw = np.asarray(payload[f"{recipient}__confidence_raw__scaled"], np.float64)
            true = np.asarray(payload[f"{recipient}__confidence_perp_sa_natural_scale__scaled"], np.float64)
            true_basis = np.asarray(payload[f"{recipient}__basis_sa"], np.float64)
            donors = set(meta[RANDOM_NULL_LAYER, recipient, "confidence_raw"]["included_answers"])
            natural = np.stack([hidden[cell["array_key"]] for cell in cells if cell["fixed_answer_color"] in donors])
            for candidate_id in range(1, CANDIDATE_POOL_SIZE + 1):
                row, _basis, _vector = candidate_metrics(raw, true, true_basis, natural, gradient, recipient_index=recipient_index, candidate_id=candidate_id)
                per_candidate[candidate_id].append({"recipient_answer": recipient, **row})
        selected_ids, candidate_summaries = select_global_candidates(per_candidate)
        arrays = {}; audit_rows = []; all_hashes = []
        selected_rank = {candidate_id: index for index, candidate_id in enumerate(selected_ids, 1)}
        for candidate_id in selected_ids:
            for recipient_index, recipient in enumerate(CANONICAL_COLORS):
                raw = np.asarray(payload[f"{recipient}__confidence_raw__scaled"], np.float64)
                true = np.asarray(payload[f"{recipient}__confidence_perp_sa_natural_scale__scaled"], np.float64)
                true_basis = np.asarray(payload[f"{recipient}__basis_sa"], np.float64)
                donors = set(meta[RANDOM_NULL_LAYER, recipient, "confidence_raw"]["included_answers"])
                natural = np.stack([hidden[cell["array_key"]] for cell in cells if cell["fixed_answer_color"] in donors])
                row, basis, vector = candidate_metrics(raw, true, true_basis, natural, gradient, recipient_index=recipient_index, candidate_id=candidate_id)
                replicate = selected_rank[candidate_id]
                basis_key = f"rep_{replicate:03d}__{recipient}__basis"
                vector_key = f"rep_{replicate:03d}__{recipient}__vector"
                arrays[basis_key] = basis; arrays[vector_key] = vector
                all_hashes.append(row["vector_sha256"])
                summary = next(item for item in candidate_summaries if item["candidate_id"] == candidate_id)
                audit_rows.append({"null_replicate": replicate, "recipient_answer": recipient, "global_matching_distance": summary["global_matching_distance"], "basis_key": basis_key, "vector_key": vector_key, **row})
    if len(set(all_hashes)) != SELECTED_CANDIDATES * len(CANONICAL_COLORS):
        raise ValueError("Selected random-null vector hashes are not globally unique")
    vector_path = paths["directions"] / "P1_LAT__L14.npz"
    atomic_npz(vector_path, arrays)
    vector_fingerprint = canonical_hash({"file_sha256": sha256_file(vector_path), "audit": audit_rows})
    atomic_csv(paths["diagnostics"] / "random_sa_subspace_null_vector_audit.csv", audit_rows)
    atomic_json(paths["diagnostics"] / "candidate_selection.json", {"selected_candidate_ids": selected_ids, "candidates": candidate_summaries, "recipient_metrics": [row for candidate_id in sorted(per_candidate) for row in per_candidate[candidate_id]]})
    atomic_json(paths["directions"] / "manifest.json", {"selected_candidate_ids": selected_ids, "vector_fingerprint": vector_fingerprint, "file_sha256": sha256_file(vector_path), "static_lock_fingerprint": static_lock_fingerprint})

    baselines, historical_audit = _historical_clean_baselines(source_root, source_anchor["case_count"])
    atomic_jsonl(paths["diagnostics"] / "historical_clean_baselines.jsonl", baselines)
    atomic_csv(paths["diagnostics"] / "historical_clean_baseline_audit.csv", historical_audit)
    after = protected_hashes(source_root)
    if after != before:
        raise ValueError("Protected source artifacts changed during random-null preparation")
    atomic_json(paths["diagnostics"] / "protected_artifact_hashes.json", {"before": before, "after_prepare": after, "passed": True})
    result = {"status": "complete", "config_fingerprint": fingerprint, "static_lock_fingerprint": static_lock_fingerprint, "vector_fingerprint": vector_fingerprint, "selected_candidate_ids": selected_ids, "candidate_pool_size": CANDIDATE_POOL_SIZE, "globally_valid_candidates": sum(row["all_recipients_valid"] for row in candidate_summaries), "case_count": source_anchor["case_count"], "resumed_noop": False}
    atomic_json(progress, result)
    return result


def _read_shards(directory: Path, pattern: str, key_field: str | None = None) -> dict[str, dict[str, Any]]:
    output = {}
    for path in sorted(directory.glob(pattern)):
        for row in load_jsonl(path, repair_trailing=True):
            key = str(row[key_field]) if key_field else random_null_key(row)
            if key in output and row != output[key]:
                raise ValueError(f"Conflicting shard duplicate: {key}")
            output[key] = row
    return output


def _load_runtime_objects(source_root: Path):
    runtime = load_runtime(INFERENCE_PATH)
    inference = runtime.QwenVLInference(str(MODEL_PATH))
    enforce_fast_image_processor(inference.processor)
    modules = resolve_language_modules(inference.model)
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    return inference, modules, tokenizer, class_token_ids(tokenizer), model_input_device(inference)


def _frozen_position_object(located: dict[str, Any], manifest_positions: dict[str, Any]) -> dict[str, Any]:
    missing = set(manifest_positions) - set(located)
    extras = set(located) - set(manifest_positions)
    # P1_CLASS_LIST_END was added to the shared locator after this frozen run.
    # It is not a steering/readout position in this experiment and is excluded
    # from the versioned position object rather than mutating the old manifest.
    if missing or extras - {"P1_CLASS_LIST_END"}:
        raise ValueError(f"Position schema mismatch: missing={sorted(missing)} extras={sorted(extras)}")
    frozen = {key: located[key] for key in manifest_positions}
    if canonical_hash(frozen) != canonical_hash(manifest_positions):
        raise ValueError("Frozen position values differ from runtime manifest")
    return frozen


def _render_and_locate(inference, tokenizer, manifest: dict[str, Any], device):
    messages = _messages(manifest)
    rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
    inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=device)
    located = locate_phase1_positions(tokenizer, rendered, inputs, str(manifest["phase0_raw_answer"]))
    frozen = _frozen_position_object(located, manifest["positions"])
    return rendered, inputs, frozen


def _exact_array(left: Any, right: Any) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _clean_worker(source_root: Path, output_root: Path, worker: int, num_gpus: int, run_fingerprint: str) -> dict[str, Any]:
    paths = _random_paths(output_root)
    manifest = load_jsonl(source_root / "artifacts/manifests/runtime_manifest.jsonl")
    selected = [row for row in manifest if stable_shard(str(row["case_id"]), num_gpus) == worker]
    completed = _read_shards(paths["diagnostics"], "validated_clean_baselines.shard_*.jsonl", "case_id")
    expected = {str(row["case_id"]) for row in selected}
    if expected <= set(completed):
        return {"stage": "clean", "worker": worker, "new_gpu_forwards": 0, "resumed_noop": True}
    historical = {row["case_id"]: row for row in load_jsonl(paths["diagnostics"] / "historical_clean_baselines.jsonl")}
    with (paths["diagnostics"] / "historical_clean_baseline_audit.csv").open() as handle:
        historical_audit = {row["case_id"]: row for row in csv.DictReader(handle)}
    config = json.loads((paths["progress"] / "random_sa_subspace_null_run_config.json").read_text())
    inference, modules, tokenizer, class_ids, device = _load_runtime_objects(source_root)
    lat_conf = joblib.load(source_root / "artifacts/probes/confidence_gap__P1_LAT__L14__full.joblib")["model"]
    panl_conf = joblib.load(source_root / "artifacts/probes/confidence_gap__P1_PANL__L18__full.joblib")["model"]
    panl_sa = joblib.load(source_root / "artifacts/probes/final_sa__P1_PANL__L18__full.joblib")["model"]
    baseline_path = paths["diagnostics"] / f"validated_clean_baselines.shard_{worker}.jsonl"
    audit_path = paths["diagnostics"] / f"clean_baseline_reuse_audit.shard_{worker}.jsonl"
    forwards = 0
    for row in selected:
        case_id = str(row["case_id"])
        if case_id in completed:
            continue
        old = historical[case_id]
        rendered, inputs, located = _render_and_locate(inference, tokenizer, row, device)
        lat = int(located[STEERING_POSITION]["processed_index"]); panl_pos = int(located[PANL_POSITION]["processed_index"]); sac = int(located["P1_SAC"]["processed_index"])
        forward = run_hooked_forward(inference.model, inputs, modules, {STEERING_POSITION: lat, PANL_POSITION: panl_pos}, logits_positions=[sac])
        forwards += 1
        score = _score(forward, sac, class_ids)
        clean_lat = forward.hidden_by_name[STEERING_POSITION][RANDOM_NULL_LAYER].detach().float().cpu().numpy().astype(np.float32)
        clean_panl = forward.hidden_by_name[PANL_POSITION][PANL_LAYER].detach().float().cpu().numpy().astype(np.float32)
        current = {
            "clean_sa_logits": score["class_logits"], "clean_sa_probabilities": score["class_probabilities"],
            "clean_soft_sa": float(score["soft_sa_image_score"]), "clean_hard_sa": int(score["argmax_hard_class"]),
            "clean_class_margin": class_margin(np.asarray(score["class_logits"], float), int(score["argmax_hard_class"])),
            "clean_lat_confidence_probe": _predict(lat_conf, clean_lat),
            "clean_panl_confidence_probe": _predict(panl_conf, clean_panl),
            "clean_panl_sa_probe": _predict(panl_sa, clean_panl),
            "clean_lat_hidden_hash": array_hash(clean_lat), "panl_clean_hidden_hash": array_hash(clean_panl),
        }
        exact = {
            "clean_sa_logits": _exact_array(current["clean_sa_logits"], old["clean_sa_logits"]),
            "clean_sa_probabilities": _exact_array(current["clean_sa_probabilities"], old["clean_sa_probabilities"]),
            "clean_hard_sa": current["clean_hard_sa"] == int(old["clean_hard_sa"]),
            "panl_clean_hidden_hash": current["panl_clean_hidden_hash"] == old["panl_clean_hidden_hash"],
        }
        errors = {field: abs(float(current[field]) - float(old[field])) for field in (
            "clean_soft_sa", "clean_class_margin", "clean_lat_confidence_probe",
            "clean_panl_confidence_probe", "clean_panl_sa_probe",
        )}
        parity = all(exact.values()) and all(value <= 1e-12 for value in errors.values())
        if not parity:
            raise ValueError(f"Fresh clean forward differs from historical baseline: {case_id}: exact={exact} errors={errors}")
        rendered_hash = _sha256_text(rendered)
        position_hash = canonical_hash(located)
        context = {
            "phase1_prompt_hash": row["phase1_prompt_hash"], "rendered_prompt_sha256": rendered_hash,
            "position_sha256": position_hash, "model_fingerprint": config["model_fingerprint"],
            "processor_fingerprint": config["processor_fingerprint"],
            "main_config_fingerprint": old["config_fingerprint"],
        }
        baseline = {
            "case_id": case_id, "family_id": str(row["family_id"]), "item_id": str(row["item_id"]),
            "condition": str(row["condition"]), "fixed_answer": str(row["phase0_normalized_answer"]),
            "canonical_source_key": old["canonical_source_key"], **current,
            **context, "runtime_context_fingerprint": canonical_hash(context),
            "random_null_run_fingerprint": run_fingerprint,
        }
        source_audit = historical_audit[case_id]
        audit = {
            "case_id": case_id, "source_row_count": int(source_audit["source_row_count"]),
            **{key: int(value) for key, value in source_audit.items() if key.startswith("unique_")},
            "historical_parity_status": "passed", "fresh_forward_parity_status": "passed",
            "selected_canonical_source_key": old["canonical_source_key"],
            **{f"fresh_error_{key}": value for key, value in errors.items()}, **context,
            "runtime_context_fingerprint": baseline["runtime_context_fingerprint"], "status": "passed",
        }
        append_jsonl(baseline_path, baseline); append_jsonl(audit_path, audit)
        completed[case_id] = baseline
    return {"stage": "clean", "worker": worker, "new_gpu_forwards": forwards, "resumed_noop": forwards == 0}


def _null_trial_row(
    proto: dict[str, Any], manifest: dict[str, Any], baseline: dict[str, Any], score: dict[str, Any],
    before: np.ndarray, after: np.ndarray, panl: np.ndarray, hook_diag: dict[str, Any],
    lat_conf, panl_conf, panl_sa, vector_info: dict[str, Any], fingerprints: dict[str, str],
) -> dict[str, Any]:
    clean_hard = int(baseline["clean_hard_sa"]); clean_logits = np.asarray(baseline["clean_sa_logits"], float); steered_logits = np.asarray(score["class_logits"], float)
    before_norm = float(np.linalg.norm(before)); after_norm = float(np.linalg.norm(after)); displacement = float(np.linalg.norm(after - before))
    result = {
        "status": "completed", **proto, "trial_type": RANDOM_NULL_NAME,
        "direction": RANDOM_NULL_NAME,
        "candidate_id": int(vector_info["candidate_id"]), "candidate_seed": int(vector_info["candidate_seed"]),
        "item_id": str(manifest["item_id"]), "family_id": str(manifest["family_id"]), "condition": str(manifest["condition"]),
        "answer_origin": "follow_text" if manifest["answer_matches_text"] and not manifest["answer_matches_image"] else "follow_image" if manifest["answer_matches_image"] and not manifest["answer_matches_text"] else "other",
        "fixed_answer": str(manifest["phase0_normalized_answer"]), "hidden_definition": HIDDEN_DEFINITION,
        "activation_before_hash": array_hash(before), "activation_after_hash": array_hash(after), "activation_before_norm": before_norm, "activation_after_norm": after_norm,
        "activation_cosine": float(before @ after / (before_norm * after_norm)), "displacement_norm": displacement, "displacement_ratio": displacement / before_norm,
        "hook_hit_count": int(hook_diag["hook_call_count"]), "steering_applied_count": int(hook_diag["steering_applied_count"]), "only_target_lat_token_modified": True,
        "panl_clean_hidden_hash": baseline["panl_clean_hidden_hash"], "panl_steered_hidden_hash": array_hash(panl),
        "clean_lat_confidence_probe": float(baseline["clean_lat_confidence_probe"]), "steered_lat_confidence_probe": _predict(lat_conf, after),
        "delta_confidence_LAT_immediate": _predict(lat_conf, after) - float(baseline["clean_lat_confidence_probe"]),
        "clean_panl_confidence_probe": float(baseline["clean_panl_confidence_probe"]), "steered_panl_confidence_probe": _predict(panl_conf, panl),
        "delta_confidence_PANL_L18": _predict(panl_conf, panl) - float(baseline["clean_panl_confidence_probe"]),
        "clean_panl_sa_probe": float(baseline["clean_panl_sa_probe"]), "steered_panl_sa_probe": _predict(panl_sa, panl),
        "delta_panl_probe_sa": _predict(panl_sa, panl) - float(baseline["clean_panl_sa_probe"]),
        "clean_sa_logits": baseline["clean_sa_logits"], "steered_sa_logits": score["class_logits"],
        "clean_sa_probabilities": baseline["clean_sa_probabilities"], "steered_sa_probabilities": score["class_probabilities"],
        "clean_soft_sa": float(baseline["clean_soft_sa"]), "steered_soft_sa": float(score["soft_sa_image_score"]),
        "delta_final_soft_sa": float(score["soft_sa_image_score"]) - float(baseline["clean_soft_sa"]),
        "clean_hard_sa": clean_hard, "steered_hard_sa": int(score["argmax_hard_class"]), "hard_sa_changed": int(score["argmax_hard_class"]) != clean_hard,
        "clean_class_margin": float(baseline["clean_class_margin"]), "clean_class_margin_change": class_margin(steered_logits, clean_hard) - class_margin(clean_logits, clean_hard),
        "actual_injection_norm": abs(float(proto["alpha"])) * float(vector_info["null_vector_norm"]),
        "signed_injection_natural_sd": float(proto["alpha"]) * float(vector_info["alpha1_natural_sd"]),
        "vector_sha256": vector_info["vector_sha256"], "basis_sha256": vector_info["basis_sha256"],
        "runtime_context_fingerprint": baseline["runtime_context_fingerprint"], **fingerprints,
        "format_valid": True,
    }
    if array_hash(before) != baseline["clean_lat_hidden_hash"]:
        raise ValueError(f"Hook pre-injection hidden differs from validated clean baseline: {proto['case_id']}")
    if int(hook_diag["hook_call_count"]) != 1 or int(hook_diag["steering_applied_count"]) != 1:
        raise ValueError(f"Random-null hook did not hit exactly once: {random_null_key(result)}")
    floats = [value for value in result.values() if isinstance(value, float)]
    if not np.isfinite(floats).all() or abs(sum(result["steered_sa_probabilities"]) - 1.0) > 1e-8:
        raise ValueError("Non-finite/invalid random-null trial")
    return result


def _null_worker(source_root: Path, output_root: Path, worker: int, num_gpus: int, repeats: int, run_fingerprint: str) -> dict[str, Any]:
    paths = _random_paths(output_root); manifest = load_jsonl(source_root / "artifacts/manifests/runtime_manifest.jsonl")
    selected = [row for row in manifest if stable_shard(str(row["case_id"]), num_gpus) == worker]
    completed = _read_shards(paths["trials"], "random_sa_subspace_null_trials.shard_*.jsonl")
    expected = {random_null_key({"case_id": row["case_id"], "null_replicate": replicate, "layer": RANDOM_NULL_LAYER, "alpha": alpha}) for row in selected for replicate in range(1, repeats + 1) for alpha in (-RANDOM_NULL_DOSE, RANDOM_NULL_DOSE)}
    if expected <= set(completed):
        return {"stage": "null", "worker": worker, "new_gpu_forwards": 0, "resumed_noop": True}
    baselines = {row["case_id"]: row for row in load_jsonl(paths["diagnostics"] / "validated_clean_baselines.jsonl")}
    vector_audit = list(csv.DictReader((paths["diagnostics"] / "random_sa_subspace_null_vector_audit.csv").open()))
    vector_info = {(int(row["null_replicate"]), row["recipient_answer"]): row for row in vector_audit}
    prep = json.loads((paths["progress"] / "random_sa_subspace_null_prepare.json").read_text())
    config = json.loads((paths["progress"] / "random_sa_subspace_null_run_config.json").read_text())
    fingerprints = {
        "main_fingerprint": config["source_anchor"]["main_run_fingerprint"],
        "random_null_fingerprint": run_fingerprint,
        "probe_fingerprint": canonical_hash(config["source_anchor"]["probe_files_sha256"]),
        "data_fingerprint": config["source_anchor"]["runtime_manifest_sha256"],
        "random_vector_fingerprint": prep["vector_fingerprint"],
    }
    inference, modules, tokenizer, class_ids, device = _load_runtime_objects(source_root)
    lat_conf = joblib.load(source_root / "artifacts/probes/confidence_gap__P1_LAT__L14__full.joblib")["model"]
    panl_conf = joblib.load(source_root / "artifacts/probes/confidence_gap__P1_PANL__L18__full.joblib")["model"]
    panl_sa = joblib.load(source_root / "artifacts/probes/final_sa__P1_PANL__L18__full.joblib")["model"]
    trial_path = paths["trials"] / f"random_sa_subspace_null_trials.shard_{worker}.jsonl"
    forwards = 0
    with np.load(paths["directions"] / "P1_LAT__L14.npz") as vectors:
        for row in selected:
            case_id = str(row["case_id"]); pending = [key for key in expected if key.startswith(case_id + "|") and key not in completed]
            if not pending: continue
            baseline = baselines[case_id]
            rendered, inputs, located = _render_and_locate(inference, tokenizer, row, device)
            if _sha256_text(rendered) != baseline["rendered_prompt_sha256"] or canonical_hash(located) != baseline["position_sha256"]:
                raise ValueError(f"Runtime context differs from validated clean stage: {case_id}")
            lat = int(located[STEERING_POSITION]["processed_index"]); panl_pos = int(located[PANL_POSITION]["processed_index"]); sac = int(located["P1_SAC"]["processed_index"]); sequence = int(inputs.input_ids.shape[1]); answer = str(row["phase0_normalized_answer"])
            for replicate in range(1, repeats + 1):
                info = vector_info[replicate, answer]
                vector = np.asarray(vectors[info["vector_key"]], np.float32)
                if array_hash(vector) != info["vector_sha256"]: raise ValueError("Random-null vector hash mismatch")
                for alpha in (-RANDOM_NULL_DOSE, RANDOM_NULL_DOSE):
                    proto = {"case_id": case_id, "null_replicate": replicate, "layer": RANDOM_NULL_LAYER, "alpha": alpha}
                    key = random_null_key(proto)
                    if key in completed: continue
                    hook = AdditiveActivationHook(modules, layer_index=RANDOM_NULL_LAYER, target_position=lat, steering_vector=torch.from_numpy(vector) * alpha, prefill_sequence_length=sequence, injection_site="block_output")
                    with hook:
                        forward = run_hooked_forward(inference.model, inputs, modules, {PANL_POSITION: panl_pos}, logits_positions=[sac])
                    forwards += 1
                    score = _score(forward, sac, class_ids); panl = forward.hidden_by_name[PANL_POSITION][PANL_LAYER].detach().float().cpu().numpy().astype(np.float32)
                    result = _null_trial_row(proto, row, baseline, score, hook.h_before.numpy(), hook.h_after.numpy(), panl, hook.diagnostics(), lat_conf, panl_conf, panl_sa, info, fingerprints)
                    append_jsonl(trial_path, result); completed[key] = result
    return {"stage": "null", "worker": worker, "new_gpu_forwards": forwards, "resumed_noop": forwards == 0}


def _spawn_workers(source_root: Path, output_root: Path, stage: str, num_gpus: int, repeats: int, run_fingerprint: str) -> list[dict[str, Any]]:
    processes = []
    repo = Path(__file__).resolve().parents[2]
    for worker in range(num_gpus):
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(worker)
        command = [
            sys.executable, "-m", "dp_SA.confidence_steering.random_sa_null",
            "--worker-stage", stage, "--source-root", str(source_root),
            "--output-root", str(output_root), "--worker-id", str(worker),
            "--num-gpus", str(num_gpus), "--random-sa-null-repeats", str(repeats),
            "--run-fingerprint", run_fingerprint,
        ]
        processes.append(subprocess.Popen(command, cwd=repo, env=env))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"Random-null {stage} worker failed: {codes}")
    paths = _random_paths(output_root)
    results = []
    for worker in range(num_gpus):
        path = paths["progress"] / f"random_sa_subspace_null_{stage}_worker_{worker}.json"
        results.append(json.loads(path.read_text()))
    return results


def run_random_null(source_root: Path, output_root: Path, *, repeats: int, num_gpus: int, resume: bool) -> dict[str, Any]:
    if num_gpus not in (1, 2): raise ValueError("--num-gpus must be 1 or 2")
    if repeats not in (SMOKE_REPEATS, FORMAL_REPEATS): raise ValueError("Random-null repeats must be 3 for smoke or 20 for formal")
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
        raise RuntimeError(f"Requested {num_gpus} GPUs, visible={torch.cuda.device_count()}")
    source_root = source_root.resolve(); output_root = output_root.resolve(); paths = _ensure_random_layout(output_root)
    prepare_result = json.loads((paths["progress"] / "random_sa_subspace_null_prepare.json").read_text())
    prepare_config = json.loads((paths["progress"] / "random_sa_subspace_null_prepare_config.json").read_text())
    source_anchor = _source_anchor(source_root); before = prepare_config["protected_before"]
    if protected_hashes(source_root) != before: raise ValueError("Protected source artifacts changed before random-null runtime")
    config = {
        "format_version": 1, "prepare_fingerprint": prepare_result["config_fingerprint"],
        "static_lock_fingerprint": prepare_result["static_lock_fingerprint"],
        "vector_fingerprint": prepare_result["vector_fingerprint"], "source_anchor": source_anchor,
        "repeats": repeats, "layer": RANDOM_NULL_LAYER, "dose": RANDOM_NULL_DOSE,
        "num_gpus": num_gpus, "clean_validation_required": True,
        "model_fingerprint": canonical_hash(prepare_config["model_files_sha256"]),
        "processor_fingerprint": canonical_hash(prepare_config["processor_files_sha256"]),
    }
    fingerprint = semantic_fingerprint(paths["progress"] / "random_sa_subspace_null_run_config.json", config, resume=resume)
    progress = paths["progress"] / "random_sa_subspace_null_run.json"
    started = time.time()
    clean_results = _spawn_workers(source_root, output_root, "clean", num_gpus, repeats, fingerprint)
    clean = _read_shards(paths["diagnostics"], "validated_clean_baselines.shard_*.jsonl", "case_id")
    clean_audit = _read_shards(paths["diagnostics"], "clean_baseline_reuse_audit.shard_*.jsonl", "case_id")
    if len(clean) != source_anchor["case_count"] or len(clean_audit) != source_anchor["case_count"] or not all(row["status"] == "passed" for row in clean_audit.values()):
        raise ValueError("Clean-validation barrier is incomplete")
    atomic_jsonl(paths["diagnostics"] / "validated_clean_baselines.jsonl", [clean[key] for key in sorted(clean)])
    atomic_csv(paths["diagnostics"] / "clean_baseline_reuse_audit.csv", [clean_audit[key] for key in sorted(clean_audit)])

    null_results = _spawn_workers(source_root, output_root, "null", num_gpus, repeats, fingerprint)
    trials = _read_shards(paths["trials"], "random_sa_subspace_null_trials.shard_*.jsonl")
    expected_trials = source_anchor["case_count"] * repeats * 2
    if len(trials) != expected_trials:
        raise ValueError(f"Random-null merge incomplete: {len(trials)}/{expected_trials}")
    merged_path = paths["trials"] / "random_sa_subspace_null_trials.jsonl"
    atomic_jsonl(merged_path, [trials[key] for key in sorted(trials)])
    after = protected_hashes(source_root)
    if after != before: raise ValueError("Protected source artifacts changed during random-null runtime")
    protection_path = paths["diagnostics"] / "protected_artifact_hashes.json"
    protection = json.loads(protection_path.read_text()); protection["after_run"] = after; protection["passed"] = protection["before"] == after
    atomic_json(protection_path, protection)
    clean_forwards = sum(int(row["new_gpu_forwards"]) for row in clean_results); null_forwards = sum(int(row["new_gpu_forwards"]) for row in null_results)
    result = {
        "status": "complete", "config_fingerprint": fingerprint, "case_count": source_anchor["case_count"],
        "repeats": repeats, "clean_trial_count": len(clean), "null_trial_count": len(trials),
        "new_clean_forwards": clean_forwards, "new_null_forwards": null_forwards,
        "new_gpu_forwards": clean_forwards + null_forwards,
        "resumed_noop": clean_forwards + null_forwards == 0,
        "elapsed_seconds": time.time() - started, "protected_artifacts_unchanged": protection["passed"],
    }
    atomic_json(progress, result)
    return result


def _null_symmetric_effect(rows: Sequence[dict[str, Any]], endpoint: str, replicate: int, dose: float) -> list[dict[str, Any]]:
    selected = [row for row in rows if int(row["null_replicate"]) == replicate]
    grouped: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in selected: grouped[str(row["case_id"])][float(row["alpha"])] = row
    output = []
    for values in grouped.values():
        if dose not in values or -dose not in values: continue
        source = values[dose]
        output.append({
            **{key: source[key] for key in ("case_id", "item_id", "family_id", "condition", "answer_origin", "fixed_answer")},
            "effect": (float(values[dose][endpoint]) - float(values[-dose][endpoint])) / 2.0,
        })
    return output


def random_null_comparison_rows(main: Sequence[dict[str, Any]], null: Sequence[dict[str, Any]], repeats: int, draws: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    output = []
    true_main = [row for row in main if row["direction"] == "confidence_perp_sa_natural_scale" and int(row["layer"]) == RANDOM_NULL_LAYER and abs(float(row["alpha"])) == RANDOM_NULL_DOSE]
    if len(true_main) != len({row["case_id"] for row in true_main}) * 2:
        raise ValueError("True natural-perpendicular ±2 main rows are incomplete")
    for endpoint in ENDPOINTS:
        true_effect = symmetric_effect(true_main, endpoint, RANDOM_NULL_DOSE)
        for group in GROUPS:
            true_summary = summarize(true_effect, "effect", group, draws)
            true_value = float(true_summary["mean_delta"])
            output.append({"endpoint": endpoint, "group": group, "row_type": "true", "null_replicate": None, "effect": true_value, **{key: value for key, value in true_summary.items() if key != "mean_delta"}})
            null_values = []
            for replicate in range(1, repeats + 1):
                effects = _null_symmetric_effect(null, endpoint, replicate, RANDOM_NULL_DOSE)
                if len(effects) != len(true_effect): raise ValueError(f"Null replicate {replicate} is incomplete for {endpoint}")
                summary = summarize(effects, "effect", group, draws); value = float(summary["mean_delta"]); null_values.append(value)
                output.append({"endpoint": endpoint, "group": group, "row_type": "null", "null_replicate": replicate, "effect": value, **{key: item for key, item in summary.items() if key != "mean_delta"}})
            negative_count = sum(value <= true_value for value in null_values)
            two_count = sum(abs(value) >= abs(true_value) for value in null_values)
            output.append({
                "endpoint": endpoint, "group": group, "row_type": "comparison", "null_replicate": None,
                "effect": true_value, "true_effect": true_value, "null_count": len(null_values),
                "null_mean": float(np.mean(null_values)), "null_sd": float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0,
                "null_min": min(null_values), "null_max": max(null_values),
                "true_percentile_negative_cdf": 100.0 * negative_count / len(null_values),
                "empirical_p_negative_one_sided": (1 + negative_count) / (len(null_values) + 1),
                "empirical_p_two_sided": (1 + two_count) / (len(null_values) + 1),
                "minimum_attainable_p": 1.0 / (len(null_values) + 1),
            })
    return output


def _plot_random_null(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, endpoint, title in zip(axes, ("delta_panl_probe_sa", "delta_final_soft_sa"), ("PANL L18 SA", "Final soft SA"), strict=True):
        selected = [row for row in rows if row["endpoint"] == endpoint and row["group"] == "answer_equal_macro"]
        null = sorted((row for row in selected if row["row_type"] == "null"), key=lambda row: int(row["null_replicate"]))
        summary = next(row for row in selected if row["row_type"] == "comparison")
        ax.scatter([int(row["null_replicate"]) for row in null], [float(row["effect"]) for row in null], color="#4C78A8", s=35, label="matched random subspace")
        true = float(summary["true_effect"]); ax.axhline(true, color="#D62728", lw=2, label="true SA subspace")
        ax.axhline(0, color="black", lw=.7); ax.set_xlabel("Global null replicate"); ax.set_ylabel(r"$S^2$"); ax.set_title(title)
        ax.text(.02, .98, f"true={true:+.5f}\npercentile={float(summary['true_percentile_negative_cdf']):.1f}%\np(one-sided)={float(summary['empirical_p_negative_one_sided']):.4f}\nB={int(summary['null_count'])}", transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=300); plt.close(fig)


def analyze_random_null(source_root: Path, output_root: Path, *, repeats: int, resume: bool, bootstrap_repeats: int) -> dict[str, Any]:
    source_root = source_root.resolve(); output_root = output_root.resolve(); paths = _random_paths(output_root)
    main_path = source_root / "artifacts/trials/main_trials.jsonl"; null_path = paths["trials"] / "random_sa_subspace_null_trials.jsonl"
    run_result = json.loads((paths["progress"] / "random_sa_subspace_null_run.json").read_text())
    config = {"format_version": 1, "main_sha256": sha256_file(main_path), "null_sha256": sha256_file(null_path), "run_fingerprint": run_result["config_fingerprint"], "repeats": repeats, "bootstrap_repeats": bootstrap_repeats, "seed": SEED, "negative_tail_prelocked": True}
    fingerprint = semantic_fingerprint(paths["progress"] / "random_sa_subspace_null_analyze_config.json", config, resume=resume)
    progress = paths["progress"] / "random_sa_subspace_null_analyze.json"
    if resume and progress.is_file():
        previous = json.loads(progress.read_text())
        if previous.get("status") == "complete" and previous.get("config_fingerprint") == fingerprint:
            return {**previous, "resumed_noop": True}
    main = load_jsonl(main_path); null = load_jsonl(null_path)
    true_rows = [row for row in main if row["direction"] == "confidence_perp_sa_natural_scale" and int(row["layer"]) == RANDOM_NULL_LAYER and abs(float(row["alpha"])) == RANDOM_NULL_DOSE]
    draws, draw_fingerprint = family_draws(true_rows, bootstrap_repeats)
    rows = random_null_comparison_rows(main, null, repeats, draws)
    atomic_csv(paths["tables"] / "random_sa_subspace_null_effects.csv", [row for row in rows if row["row_type"] != "comparison"])
    atomic_csv(paths["tables"] / "random_sa_subspace_null_comparison.csv", [row for row in rows if row["row_type"] == "comparison"])
    _plot_random_null(rows, paths["figures"] / "random_sa_subspace_null.png")
    comparisons = {(row["endpoint"], row["group"]): row for row in rows if row["row_type"] == "comparison"}
    panl = comparisons["delta_panl_probe_sa", "answer_equal_macro"]; final = comparisons["delta_final_soft_sa", "answer_equal_macro"]
    panl_tail = float(panl["empirical_p_negative_one_sided"]) <= .10; final_tail = float(final["empirical_p_negative_one_sided"]) <= .10
    if panl_tail and final_tail:
        interpretation = "PANL与final的真实结果均位于匹配随机子空间null的负向尾部，提示真实SA子空间相对于匹配随机子空间具有探索性特殊性。"
    elif panl_tail and not final_tail:
        interpretation = "PANL结果位于负向尾部而final不位于负向尾部，提示特殊结构主要出现在LAT→PANL阶段，最终输出存在进一步衰减或重编码。"
    elif not panl_tail and not final_tail:
        interpretation = "PANL与final的真实结果均位于匹配随机子空间null中部，反向效应可能是一般投影几何现象。"
    else:
        interpretation = "final结果位于负向尾部而PANL未呈现相同模式；本次小规模筛查不对此作进一步机制归因。"
    atomic_text(output_root / "random_sa_subspace_null_summary.md", f"# 匹配随机SA子空间null\n\n- null repeats: `{repeats}`\n- family bootstrap: `{bootstrap_repeats}`\n- negative one-sided minimum p: `{1/(repeats+1):.6f}`\n- automatic extension: `disabled`\n\n{interpretation}\n\n本实验是小规模探索性机制筛查，不构成最终确认性证明。`confidence_perp_sa_natural_scale`仅表示删除已测量线性SA子空间后的自然尺度分量，不称为纯confidence。是否扩展null数量必须由用户在看到结果后另行决定。\n")
    prepare_config = json.loads((paths["progress"] / "random_sa_subspace_null_prepare_config.json").read_text()); after = protected_hashes(source_root)
    if after != prepare_config["protected_before"]: raise ValueError("Protected source artifacts changed during random-null analysis")
    protection_path = paths["diagnostics"] / "protected_artifact_hashes.json"; protection = json.loads(protection_path.read_text()); protection["after_analyze"] = after; protection["passed"] = protection["before"] == after; atomic_json(protection_path, protection)
    result = {"status": "complete", "config_fingerprint": fingerprint, "bootstrap_repeats": bootstrap_repeats, "bootstrap_draw_fingerprint": draw_fingerprint, "table_rows": len(rows), "null_repeats": repeats, "expand_null": False, "protected_artifacts_unchanged": protection["passed"], "resumed_noop": False}
    atomic_json(progress, result); return result


def _natural_smoke_source() -> Path:
    candidates: list[Path] = []
    for lock_path in NATURAL_SMOKE_PARENT.glob("config_*/round_*/progress/experiment_lock.json"):
        lock = json.loads(lock_path.read_text())
        if lock.get("status") == "locked" and lock.get("run_spec") == NATURAL_RUN_SPEC:
            candidates.append(lock_path.parents[1])
    if not candidates:
        raise RuntimeError("No locked 24-case natural-decomposition smoke source exists")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _next_random_smoke_root(protocol_fingerprint: str) -> tuple[int, Path]:
    parent = RANDOM_SMOKE_PARENT / f"config_{protocol_fingerprint[:12]}"
    parent.mkdir(parents=True, exist_ok=True)
    for round_number in range(1, 6):
        root = parent / f"round_{round_number}"
        if not root.exists():
            return round_number, root
    raise RuntimeError("All five random-null smoke rounds have been used")


def _matching_random_smoke_lock(protocol_fingerprint: str) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for path in RANDOM_SMOKE_PARENT.glob("config_*/round_*/progress/random_sa_subspace_null_lock.json"):
        lock = json.loads(path.read_text())
        if lock.get("status") == "locked" and lock.get("static_lock_fingerprint") == protocol_fingerprint:
            candidates.append((path, lock))
    if not candidates:
        raise RuntimeError("Formal random null is blocked: no matching successful 3-null smoke lock")
    return max(candidates, key=lambda item: item[0].stat().st_mtime)


def run_random_null_pipeline(
    *, repeats: int, smoke: bool, resume: bool, num_gpus: int,
    output_root: Path | None = None,
) -> dict[str, Any]:
    expected_repeats = SMOKE_REPEATS if smoke else FORMAL_REPEATS
    if repeats != expected_repeats:
        raise ValueError(f"Expected random-null repeats={expected_repeats} for {'smoke' if smoke else 'formal'} mode")
    protocol_fingerprint = canonical_hash(random_protocol_material())
    if smoke:
        source_root = _natural_smoke_source()
        round_number, destination = _next_random_smoke_root(protocol_fingerprint) if output_root is None else (1, Path(output_root))
        if destination.exists() and not resume:
            raise FileExistsError(f"Random-null smoke output already exists: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        formal_before = protected_hashes(NATURAL_FORMAL_ROOT)
        failures = []
        try:
            prepared = prepare_random_null(source_root, destination, resume=resume)
            ran = run_random_null(source_root, destination, repeats=repeats, num_gpus=num_gpus, resume=resume)
            analyzed = analyze_random_null(source_root, destination, repeats=repeats, resume=resume, bootstrap_repeats=BOOTSTRAP_REPEATS)
            resumed_prepare = prepare_random_null(source_root, destination, resume=True)
            resumed_run = run_random_null(source_root, destination, repeats=repeats, num_gpus=num_gpus, resume=True)
            resumed_analyze = analyze_random_null(source_root, destination, repeats=repeats, resume=True, bootstrap_repeats=BOOTSTRAP_REPEATS)
            if ran["new_gpu_forwards"] != 168:
                raise RuntimeError(f"Smoke forward budget mismatch: {ran['new_gpu_forwards']} != 168")
            if not resumed_run["resumed_noop"] or resumed_run["new_gpu_forwards"] != 0:
                raise RuntimeError("Random-null smoke resume performed new GPU forwards")
            if protected_hashes(NATURAL_FORMAL_ROOT) != formal_before:
                raise RuntimeError("Formal natural-decomposition artifacts changed during smoke")
            lock = {
                "status": "locked", "format_version": 1,
                "static_lock_fingerprint": protocol_fingerprint,
                "smoke_round": round_number, "source_root": str(source_root),
                "output_root": str(destination.resolve()), "repeats": repeats,
                "clean_forwards": ran["new_clean_forwards"],
                "null_forwards": ran["new_null_forwards"],
                "total_forwards": ran["new_gpu_forwards"],
                "resume_new_forwards": resumed_run["new_gpu_forwards"],
                "prepare_fingerprint": prepared["config_fingerprint"],
                "run_fingerprint": ran["config_fingerprint"],
                "analyze_fingerprint": analyzed["config_fingerprint"],
                "protected_formal_hashes": formal_before,
            }
            lock["fingerprint"] = canonical_hash(lock)
            atomic_json(destination / "progress/random_sa_subspace_null_lock.json", lock)
            report = {
                "status": "passed", "smoke": True, "round": round_number,
                "output_root": str(destination.resolve()), "source_root": str(source_root.resolve()),
                "prepare": prepared, "run": ran, "analyze": analyzed,
                "resume": {"prepare": resumed_prepare, "run": resumed_run, "analyze": resumed_analyze},
                "lock": lock, "formal_random_null_started": False,
            }
            atomic_json(destination / "progress/random_sa_subspace_null_smoke_report.json", report)
            return report
        except Exception as exc:
            failures.append({"type": type(exc).__name__, "message": str(exc)})
            _ensure_random_layout(destination)
            atomic_json(destination / "progress/random_sa_subspace_null_smoke_failure.json", failures[-1])
            raise

    if not resume:
        raise ValueError("Formal random-null writes into the existing natural-decomposition root; pass --resume explicitly")
    source_root = Path(output_root or NATURAL_FORMAL_ROOT).resolve()
    lock_path, smoke_lock = _matching_random_smoke_lock(protocol_fingerprint)
    prepared = prepare_random_null(source_root, source_root, resume=True)
    if prepared["static_lock_fingerprint"] != smoke_lock["static_lock_fingerprint"]:
        raise RuntimeError("Formal random-null protocol does not match smoke lock")
    ran = run_random_null(source_root, source_root, repeats=repeats, num_gpus=num_gpus, resume=True)
    analyzed = analyze_random_null(source_root, source_root, repeats=repeats, resume=True, bootstrap_repeats=BOOTSTRAP_REPEATS)
    return {
        "status": "complete", "smoke": False, "source_smoke_lock": str(lock_path),
        "prepare": prepared, "run": ran, "analyze": analyzed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-stage", choices=("clean", "null"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, choices=(1, 2), required=True)
    parser.add_argument("--random-sa-null-repeats", type=int, required=True)
    parser.add_argument("--run-fingerprint", required=True)
    args = parser.parse_args(argv)
    if args.worker_stage == "clean":
        result = _clean_worker(args.source_root, args.output_root, args.worker_id, args.num_gpus, args.run_fingerprint)
    else:
        result = _null_worker(args.source_root, args.output_root, args.worker_id, args.num_gpus, args.random_sa_null_repeats, args.run_fingerprint)
    atomic_json(_random_paths(args.output_root)["progress"] / f"random_sa_subspace_null_{args.worker_stage}_worker_{args.worker_id}.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
