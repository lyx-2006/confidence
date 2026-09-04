from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
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
from scipy.stats import pearsonr, spearmanr

from dp_SA.checkpoint_steering.run import class_margin
from dp_SA.config import MIDPOINTS
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits
from layer_metacognition.model_adapter import AdditiveActivationHook, _selected_logits_kwargs, run_hooked_forward

from .analyze import _mean, family_draws, summarize
from .config import BOOTSTRAP_REPEATS, HIDDEN_DEFINITION, PANL_LAYER, PANL_POSITION, SEED, STEERING_POSITION
from .core import raw_gradient
from .io_utils import append_jsonl, array_hash, atomic_csv, atomic_json, atomic_jsonl, atomic_npz, atomic_text, canonical_hash, load_jsonl, semantic_fingerprint, sha256_file, stable_shard
from .random_sa_null import (
    NATURAL_FORMAL_ROOT, RANDOM_SMOKE_PARENT, _load_runtime_objects,
    _matching_random_smoke_lock, _natural_smoke_source, _render_and_locate,
    model_processor_hashes,
)
from .run import _predict, _score


GRADIENT_LAYER = 14
TRUE_DIRECTIONS = ("confidence_raw", "confidence_parallel_sa", "confidence_perp_sa_natural_scale")
ENDPOINTS = ("panl_sa", "final_sa")
GROUPS = ("answer_equal_macro", "family_micro", "all")
DEFAULT_EPSILONS = (0.5,)
DOSE_SELECTION_CANDIDATES = (0.25, 0.5, 1.0)
HISTORICAL_EPSILONS = (1.0, 2.0)
NULL_REPEATS = 20
SMOKE_NULL_REPEATS = 3
GRADIENT_SMOKE_PARENT = Path(__file__).resolve().parent / "output/gradient_validation_smoke"
ALLOWED_RELATIVE_PREFIXES = ("artifacts/gradient_validation/", "tables/gradient_", "figures/diagnostics/gradient_", "progress/gradient_validation_", "gradient_validation_summary.md")


def normalize_epsilons(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(sorted(float(value) for value in values))
    if not result or len(result) != len(set(result)) or any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError("Gradient epsilons must be distinct, finite, and positive")
    return result


def torch_soft_sa(selected_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if selected_logits.ndim != 1 or selected_logits.numel() != 9:
        raise ValueError("selected_logits must contain exactly nine values")
    logits64 = selected_logits.to(torch.float64)
    probabilities = torch.softmax(logits64, dim=0)
    midpoints = torch.as_tensor(MIDPOINTS, dtype=torch.float64, device=logits64.device)
    return torch.dot(probabilities, midpoints), probabilities


def probe_raw_parameters(model: Any) -> tuple[np.ndarray, float]:
    scaler = model.named_steps["scaler"]
    ridge = model.named_steps["ridge"]
    weight = np.asarray(ridge.coef_, np.float64) / np.asarray(scaler.scale_, np.float64)
    intercept = float(np.asarray(ridge.intercept_).reshape(-1)[0] if np.asarray(ridge.intercept_).ndim else ridge.intercept_)
    bias = intercept - float(weight @ np.asarray(scaler.mean_, np.float64))
    return weight, bias


def directional_metrics(gradient: np.ndarray, vector: np.ndarray) -> dict[str, float]:
    g = np.asarray(gradient, np.float32).astype(np.float64)
    v = np.asarray(vector, np.float32).astype(np.float64)
    gn = float(np.linalg.norm(g)); vn = float(np.linalg.norm(v)); dot = float(g @ v)
    return {"directional_derivative": dot, "gradient_norm": gn, "vector_norm": vn, "gradient_vector_cosine": dot / (gn * vn) if gn and vn else math.nan}


def central_difference(plus: float, minus: float, epsilon: float) -> dict[str, float]:
    return {"D": (float(plus) - float(minus)) / (2.0 * epsilon), "S": (float(plus) - float(minus)) / 2.0}


def relative_additivity_error(raw: float, parallel: float, perpendicular: float) -> float:
    return abs(float(raw) - float(parallel) - float(perpendicular)) / max(abs(float(raw)), abs(float(parallel)) + abs(float(perpendicular)), 1e-12)


def prediction_metrics(d: Sequence[float], gv: Sequence[float]) -> dict[str, float | int]:
    actual = np.asarray(d, np.float64); predicted = np.asarray(gv, np.float64); error = actual - predicted
    valid = np.isfinite(actual) & np.isfinite(predicted); actual, predicted, error = actual[valid], predicted[valid], error[valid]
    if not len(actual): raise ValueError("No finite prediction pairs")
    active = np.maximum(np.abs(actual), np.abs(predicted)) >= 1e-8
    pearson = float(pearsonr(predicted, actual).statistic) if len(actual) > 1 and np.std(actual) and np.std(predicted) else math.nan
    spearman = float(spearmanr(predicted, actual).statistic) if len(actual) > 1 else math.nan
    return {
        "mean_signed_error": float(np.mean(error)), "mae": float(np.mean(np.abs(error))),
        "relative_mae": float(np.mean(np.abs(error)) / max(float(np.mean(np.abs(predicted))), 1e-12)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "nrmse": float(np.sqrt(np.mean(error ** 2)) / max(float(np.sqrt(np.mean(predicted ** 2))), 1e-12)),
        "pearson": pearson, "spearman": spearman,
        "sign_agreement": float(np.mean(np.sign(actual[active]) == np.sign(predicted[active]))) if active.any() else math.nan,
        "n": int(len(actual)), "valid_sign_n": int(active.sum()),
    }


def historical_pair_index(rows: Sequence[dict[str, Any]], directions: Sequence[str], epsilons: Sequence[float], *, layer: int = 14) -> dict[tuple[str, str, float], dict[str, Any]]:
    sides: dict[tuple[str, str, float], dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        direction = str(row.get("direction")); alpha = float(row.get("alpha", math.nan))
        epsilon = abs(alpha)
        if direction not in directions or int(row.get("layer", -1)) != layer or epsilon not in epsilons or alpha == 0:
            continue
        if row.get("status") != "completed" or not row.get("format_valid", True) or row.get("hidden_definition") != HIDDEN_DEFINITION:
            continue
        key = (str(row["case_id"]), direction, epsilon)
        sign = math.copysign(1.0, alpha)
        if sign in sides[key] and sides[key][sign] != row: raise ValueError(f"Conflicting historical trial: {key}/{sign}")
        sides[key][sign] = row
    return {key: values for key, values in sides.items() if set(values) == {-1.0, 1.0}}


def missing_paired_cells(case_ids: Sequence[str], directions: Sequence[str], epsilons: Sequence[float], existing: dict[tuple[str, str, float], Any]) -> list[tuple[str, str, float]]:
    return [(case, direction, epsilon) for case in case_ids for direction in directions for epsilon in epsilons if (case, direction, epsilon) not in existing]


def random_null_pair_index(rows: Sequence[dict[str, Any]], epsilon: float = 2.0) -> dict[tuple[str, int], dict[float, dict[str, Any]]]:
    result: dict[tuple[str, int], dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        alpha = float(row.get("alpha", math.nan))
        if row.get("direction") != "random_sa_subspace_null" or abs(alpha) != epsilon or row.get("status") != "completed": continue
        key = (str(row["case_id"]), int(row["null_replicate"])); sign = math.copysign(1.0, alpha)
        if sign in result[key] and result[key][sign] != row: raise ValueError(f"Conflicting random-null trial: {key}/{sign}")
        result[key][sign] = row
    return {key: sides for key, sides in result.items() if set(sides) == {-1.0, 1.0}}


class DifferentiableLatHook:
    def __init__(self, modules: Any, *, layer: int, lat_position: int, panl_position: int, sequence_length: int):
        self.modules = modules; self.layer = int(layer); self.lat_position = int(lat_position); self.panl_position = int(panl_position); self.sequence_length = int(sequence_length)
        self.lat_calls = 0; self.panl_calls = 0; self.h_leaf: torch.Tensor | None = None; self.panl_hidden: torch.Tensor | None = None
        self.original_dtype = ""; self.original_shape: tuple[int, ...] = (); self.original_stride: tuple[int, ...] = (); self.layout_preserved = False; self.refill_exact = False; self.other_tokens_exact = False; self.tuple_tail_preserved = False; self._handles: list[Any] = []

    @staticmethod
    def _split(output: Any) -> tuple[torch.Tensor, tuple[Any, ...] | None]:
        if isinstance(output, torch.Tensor): return output, None
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor): return output[0], output[1:]
        raise TypeError(f"Unsupported decoder output: {type(output)!r}")

    def _lat(self, _module: Any, _args: Any, output: Any) -> Any:
        tensor, tail = self._split(output); self.lat_calls += 1
        if tensor.ndim != 3 or tensor.shape[1] != self.sequence_length or self.h_leaf is not None: return output
        before = tensor.detach(); self.original_dtype = str(tensor.dtype); self.original_shape = tuple(tensor.shape); self.original_stride = tuple(tensor.stride())
        self.h_leaf = tensor[0, self.lat_position, :].detach().clone().requires_grad_(True)
        patched = tensor.clone(); patched[0, self.lat_position, :] = self.h_leaf
        self.layout_preserved = patched.dtype == tensor.dtype and patched.device == tensor.device and patched.shape == tensor.shape and patched.stride() == tensor.stride() and self.h_leaf.dtype == tensor.dtype and self.h_leaf.device == tensor.device
        self.refill_exact = bool(torch.equal(patched[0, self.lat_position, :].detach(), before[0, self.lat_position, :]))
        mask = torch.ones(tensor.shape[1], dtype=torch.bool, device=tensor.device); mask[self.lat_position] = False
        self.other_tokens_exact = bool(torch.equal(patched[0, mask, :].detach(), before[0, mask, :]))
        self.tuple_tail_preserved = tail is None or all(a is b for a, b in zip(tail, output[1:], strict=True))
        return patched if tail is None else (patched, *tail)

    def _panl(self, _module: Any, _args: Any, output: Any) -> None:
        tensor, _tail = self._split(output); self.panl_calls += 1
        if tensor.ndim == 3 and tensor.shape[1] == self.sequence_length and self.panl_hidden is None:
            self.panl_hidden = tensor[0, self.panl_position, :]

    def __enter__(self):
        self._handles = [self.modules.language_layers[self.layer].register_forward_hook(self._lat), self.modules.language_layers[PANL_LAYER].register_forward_hook(self._panl)]
        return self

    def __exit__(self, *_args):
        for handle in self._handles: handle.remove()
        self._handles.clear()

    def validate(self) -> None:
        if self.lat_calls != 1 or self.panl_calls != 1 or self.h_leaf is None or self.panl_hidden is None: raise RuntimeError(f"Gradient hooks not hit exactly once: LAT={self.lat_calls} PANL={self.panl_calls}")
        if not self.layout_preserved or not self.refill_exact or not self.other_tokens_exact or not self.tuple_tail_preserved: raise RuntimeError("Differentiable leaf reinsertion parity failed")


def differentiable_forward(model: torch.nn.Module, inputs: Any, modules: Any, *, lat_position: int, panl_position: int, sac_position: int, class_ids: Sequence[int], panl_probe: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if any(parameter.requires_grad for parameter in model.parameters()): raise ValueError("All model parameters must have requires_grad=False")
    sequence = int(inputs.input_ids.shape[1]); kwargs, selected = _selected_logits_kwargs(model, inputs, [int(sac_position)], modules)
    # A 7B bf16 model plus the L14->output autograd tape is slightly larger than
    # a 24GB card. Offloading saved tensors changes neither the forward values nor
    # the differentiation point; tensors are restored to their original device
    # and dtype by autograd for the two VJPs.
    saved_context = torch.autograd.graph.save_on_cpu(pin_memory=True) if next(model.parameters()).is_cuda else torch.enable_grad()
    with saved_context:
        with DifferentiableLatHook(modules, layer=GRADIENT_LAYER, lat_position=lat_position, panl_position=panl_position, sequence_length=sequence) as hook:
            outputs = model(**kwargs, use_cache=False, output_attentions=False, output_hidden_states=False, return_dict=True)
    hook.validate()
    logits = outputs.logits[0, 0 if selected else int(sac_position)]
    indices = torch.as_tensor(class_ids, dtype=torch.long, device=logits.device)
    selected_logits = logits.index_select(0, indices)
    final_sa, probabilities = torch_soft_sa(selected_logits)
    weight, bias = probe_raw_parameters(panl_probe)
    panl_hidden32 = hook.panl_hidden.float(); w = torch.as_tensor(weight, dtype=torch.float32, device=hook.panl_hidden.device)
    panl_sa = torch.dot(panl_hidden32, w) + torch.as_tensor(bias, dtype=torch.float32, device=hook.panl_hidden.device)
    gradients = {}
    for name, scalar, retain in (("panl_sa", panl_sa, True), ("final_sa", final_sa, False)):
        value = torch.autograd.grad(scalar, hook.h_leaf, retain_graph=retain, create_graph=False, allow_unused=False)[0]
        gradients[name] = value.detach().float().cpu().numpy().astype(np.float32)
    if any(parameter.grad is not None for parameter in model.parameters()): raise RuntimeError("Model parameter gradient was populated")
    raw_prediction = float(weight @ hook.panl_hidden.detach().float().cpu().numpy().astype(np.float64) + bias)
    pipeline_prediction = _predict(panl_probe, hook.panl_hidden.detach().float().cpu().numpy().astype(np.float32))
    diagnostics = {
        "selected_logits": selected_logits.detach().float().cpu().numpy().astype(np.float32),
        "logits_original_dtype": str(selected_logits.dtype),
        "probabilities": probabilities.detach().cpu().numpy().astype(np.float64), "final_soft_sa": float(final_sa.detach().cpu()),
        "hard_class": int(torch.argmax(selected_logits).item()), "panl_sa": float(panl_sa.detach().cpu()),
        "probe_raw_pipeline_error": abs(raw_prediction - pipeline_prediction), "gradient_original_dtype": str(hook.h_leaf.dtype),
        "activation_shape": list(hook.original_shape), "activation_stride": list(hook.original_stride), "leaf_refill_exact": hook.refill_exact,
        "other_tokens_exact": hook.other_tokens_exact, "tuple_tail_preserved": hook.tuple_tail_preserved, "layout_preserved": hook.layout_preserved,
    }
    return gradients, diagnostics


def _gradient_paths(root: Path) -> dict[str, Path]:
    return {"base": root / "artifacts/gradient_validation", "gradients": root / "artifacts/gradient_validation/gradients", "tables": root / "tables", "figures": root / "figures/diagnostics", "progress": root / "progress"}


def _ensure_layout(root: Path) -> dict[str, Path]:
    paths = _gradient_paths(root)
    for path in paths.values(): path.mkdir(parents=True, exist_ok=True)
    return paths


def inventory(root: Path) -> dict[str, str]:
    result = {}
    if not root.exists(): return result
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = str(path.relative_to(root))
        if any(relative.startswith(prefix) for prefix in ALLOWED_RELATIVE_PREFIXES): continue
        result[relative] = sha256_file(path)
    return result


def gradient_code_hashes() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    paths = (Path(__file__).resolve(), Path(__file__).resolve().parent / "run_pipeline.py", Path(__file__).resolve().parent / "run.py", Path(__file__).resolve().parent / "random_sa_null.py", repo / "dp_SA/soft_score.py", repo / "dp_SA/positions.py", repo / "layer_metacognition/model_adapter.py")
    return {str(path.relative_to(repo)): sha256_file(path) for path in paths}


def _source_roots(smoke: bool) -> tuple[Path, Path]:
    if not smoke: return NATURAL_FORMAL_ROOT.resolve(), NATURAL_FORMAL_ROOT.resolve()
    natural = _natural_smoke_source().resolve()
    locks = list(RANDOM_SMOKE_PARENT.glob("config_*/round_*/progress/random_sa_subspace_null_lock.json"))
    valid = [(path, json.loads(path.read_text())) for path in locks]
    valid = [(path, lock) for path, lock in valid if lock.get("status") == "locked" and Path(lock.get("source_root", "")).resolve() == natural]
    if not valid: raise RuntimeError("No random-null smoke matching the natural smoke source")
    return natural, max(valid, key=lambda item: item[0].stat().st_mtime)[0].parents[1].resolve()


def _load_vector_maps(natural_root: Path, random_root: Path) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    metadata = json.loads((natural_root / "artifacts/directions/vector_metadata.json").read_text())
    selected_meta = {(row["recipient_answer"], row["direction"]): row for row in metadata["vectors"] if int(row["layer"]) == GRADIENT_LAYER and row["direction"] in TRUE_DIRECTIONS}
    true = {}
    with np.load(natural_root / "artifacts/directions/P1_LAT__L14.npz") as payload:
        for key, row in selected_meta.items():
            vector = np.asarray(payload[row["scaled_key"]], np.float32)
            if array_hash(vector) != row["scaled_hash"]: raise ValueError(f"True vector hash mismatch: {key}")
            true[key] = vector
    null = {}
    audit = list(csv.DictReader((random_root / "artifacts/diagnostics/random_sa_subspace_null/random_sa_subspace_null_vector_audit.csv").open()))
    info = {(row["recipient_answer"], int(row["null_replicate"])): row for row in audit}
    with np.load(random_root / "artifacts/directions/random_sa_subspace_nulls/P1_LAT__L14.npz") as payload:
        for key, row in info.items():
            vector = np.asarray(payload[row["vector_key"]], np.float32)
            if array_hash(vector) != row["vector_sha256"]: raise ValueError(f"Null vector hash mismatch: {key}")
            null[key] = vector
    if len(null) != 12 * NULL_REPEATS: raise ValueError(f"Expected 240 random vectors, found {len(null)}")
    reconstruction = {}
    for answer in sorted({key[0] for key in true}):
        raw = true[answer, TRUE_DIRECTIONS[0]]; parallel = true[answer, TRUE_DIRECTIONS[1]]; perp = true[answer, TRUE_DIRECTIONS[2]]
        error = float(np.linalg.norm(raw.astype(np.float64) - parallel.astype(np.float64) - perp.astype(np.float64)) / max(np.linalg.norm(raw.astype(np.float64)), 1e-12))
        if error > 1e-5: raise ValueError(f"Float32 vector reconstruction failed: {answer} {error}")
        reconstruction[answer] = error
    return true, null, {"metadata_fingerprint": metadata["fingerprint"], "reconstruction_errors": reconstruction}


def _baseline_path(random_root: Path) -> Path:
    return random_root / "artifacts/diagnostics/random_sa_subspace_null/validated_clean_baselines.jsonl"


def _read_shards(directory: Path, pattern: str, key_fn) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(directory.glob(pattern)):
        for row in load_jsonl(path, repair_trailing=True):
            key = key_fn(row)
            if key in result and row != result[key]: raise ValueError(f"Conflicting shard duplicate: {key}")
            result[key] = row
    return result


def gradient_row_key(row: dict[str, Any]) -> str:
    return f'{row["case_id"]}|{row["endpoint"]}|{row["direction"]}|e{float(row["epsilon"]):g}'


def fd_row_key(row: dict[str, Any]) -> str:
    return f'{row["case_id"]}|{row["direction"]}|e{float(row["epsilon"]):g}'


def prepare_gradient_validation(source_root: Path, random_root: Path, output_root: Path, *, epsilons: Sequence[float], smoke: bool, resume: bool) -> dict[str, Any]:
    epsilons = normalize_epsilons(epsilons); paths = _ensure_layout(output_root)
    manifest_path = source_root / "artifacts/manifests/runtime_manifest.jsonl"; main_path = source_root / "artifacts/trials/main_trials.jsonl"; null_path = random_root / "artifacts/trials/random_sa_subspace_null_trials.jsonl"
    baseline_path = _baseline_path(random_root)
    for path in (manifest_path, main_path, null_path, baseline_path):
        if not path.is_file(): raise FileNotFoundError(path)
    manifest = load_jsonl(manifest_path); case_ids = [str(row["case_id"]) for row in manifest]
    expected_cases = 24 if smoke else 100
    if len(manifest) != expected_cases: raise ValueError(f"Expected {expected_cases} cases, found {len(manifest)}")
    baselines = {str(row["case_id"]): row for row in load_jsonl(baseline_path)}
    if set(baselines) != set(case_ids): raise ValueError("Validated baseline coverage mismatch")
    true, null, vector_audit = _load_vector_maps(source_root, random_root)
    main = load_jsonl(main_path); existing = historical_pair_index(main, TRUE_DIRECTIONS, tuple(epsilons) + HISTORICAL_EPSILONS)
    missing = missing_paired_cells(case_ids, TRUE_DIRECTIONS, epsilons, existing)
    random_rows = load_jsonl(null_path); available_null = random_null_pair_index(random_rows)
    expected_null = expected_cases * (SMOKE_NULL_REPEATS if smoke else NULL_REPEATS)
    if len(available_null) != expected_null: raise ValueError(f"Historical random-null ±2 coverage mismatch: {len(available_null)}/{expected_null}")
    protected_before = inventory(source_root); random_before = inventory(random_root) if random_root != source_root else protected_before
    model_hash, processor_hash = model_processor_hashes()
    config = {
        "format_version": 1, "smoke": smoke, "layer": GRADIENT_LAYER, "epsilons": list(epsilons),
        "directions": list(TRUE_DIRECTIONS), "endpoints": list(ENDPOINTS), "seed": SEED,
        "runtime_manifest_sha256": sha256_file(manifest_path), "main_trials_sha256": sha256_file(main_path),
        "random_trials_sha256": sha256_file(null_path), "validated_baselines_sha256": sha256_file(baseline_path),
        "true_vector_file_sha256": sha256_file(source_root / "artifacts/directions/P1_LAT__L14.npz"),
        "random_vector_file_sha256": sha256_file(random_root / "artifacts/directions/random_sa_subspace_nulls/P1_LAT__L14.npz"),
        "panl_probe_sha256": sha256_file(source_root / "artifacts/probes/final_sa__P1_PANL__L18__full.joblib"),
        "lat_probe_sha256": sha256_file(source_root / "artifacts/probes/confidence_gap__P1_LAT__L14__full.joblib"),
        "model_files_sha256": model_hash, "processor_files_sha256": processor_hash, "code_hashes": gradient_code_hashes(),
    }
    fingerprint = semantic_fingerprint(paths["base"] / "fingerprints.json", config, resume=resume)
    result = {"status": "complete", "config_fingerprint": fingerprint, "case_count": expected_cases, "historical_complete_pairs": len(existing), "missing_paired_cells": len(missing), "planned_fd_forwards": 2 * len(missing), "planned_clean_forwards": expected_cases, "planned_vjp": 2 * expected_cases, "vector_reconstruction_errors": vector_audit["reconstruction_errors"], "protected_before": protected_before, "random_protected_before": random_before, "resumed_noop": resume}
    atomic_json(paths["progress"] / "gradient_validation_prepare.json", result)
    return result


def _gradient_output_row(manifest: dict[str, Any], endpoint: str, direction: str, epsilon: float, metrics: dict[str, float], *, gradient_hash: str, vector_hash: str, run_fingerprint: str, vector_kind: str, null_replicate: int | None = None) -> dict[str, Any]:
    return {
        "case_id": str(manifest["case_id"]), "item_id": str(manifest["item_id"]), "family_id": str(manifest["family_id"]),
        "condition": str(manifest["condition"]), "fixed_answer": str(manifest["phase0_normalized_answer"]),
        "answer_origin": "follow_text" if manifest["answer_matches_text"] and not manifest["answer_matches_image"] else "follow_image" if manifest["answer_matches_image"] and not manifest["answer_matches_text"] else "other",
        "endpoint": endpoint, "direction": direction, "vector_kind": vector_kind, "null_replicate": null_replicate,
        "layer": GRADIENT_LAYER, "epsilon": float(epsilon), **metrics,
        "predicted_S": float(epsilon) * float(metrics["directional_derivative"]),
        "gradient_sha256": gradient_hash, "vector_sha256": vector_hash,
        "gradient_validation_fingerprint": run_fingerprint, "status": "completed",
    }


def _parity_check(diagnostics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    logits = np.asarray(diagnostics["selected_logits"], np.float64); historical_logits = np.asarray(baseline["clean_sa_logits"], np.float64)
    dtype = diagnostics["logits_original_dtype"]
    low_precision = dtype in ("torch.float16", "torch.bfloat16", "float16", "bfloat16")
    logits_atol = 1e-3 if low_precision else 1e-6; soft_atol = 1e-6 if low_precision else 1e-8
    logits_error = float(np.max(np.abs(logits - historical_logits)))
    probabilities_error = float(abs(np.asarray(diagnostics["probabilities"], np.float64).sum() - 1.0))
    soft_error = abs(float(diagnostics["final_soft_sa"]) - float(baseline["clean_soft_sa"]))
    panl_error = abs(float(diagnostics["panl_sa"]) - float(baseline["clean_panl_sa_probe"]))
    result = {
        "logits_dtype": dtype, "max_selected_logits_error": logits_error, "logits_tolerance": logits_atol,
        "soft_sa_error": soft_error, "soft_sa_tolerance": soft_atol, "probability_sum_error": probabilities_error,
        "hard_class_equal": int(diagnostics["hard_class"]) == int(baseline["clean_hard_sa"]),
        "panl_probe_error": panl_error, "probe_raw_pipeline_error": float(diagnostics["probe_raw_pipeline_error"]),
    }
    result["passed"] = bool(logits_error <= logits_atol and soft_error <= soft_atol and probabilities_error <= 1e-8 and result["hard_class_equal"] and panl_error <= 1e-7 and result["probe_raw_pipeline_error"] <= 1e-7)
    return result


def _steered_pair(inference: Any, modules: Any, inputs: Any, positions: dict[str, Any], class_ids: Sequence[int], vector: np.ndarray, epsilon: float, panl_probe: Any, baseline: dict[str, Any]) -> dict[str, Any]:
    lat = int(positions[STEERING_POSITION]["processed_index"]); panl_position = int(positions[PANL_POSITION]["processed_index"]); sac = int(positions["P1_SAC"]["processed_index"]); sequence = int(inputs.input_ids.shape[1])
    sides = {}
    for sign in (-1.0, 1.0):
        hook = AdditiveActivationHook(modules, layer_index=GRADIENT_LAYER, target_position=lat, steering_vector=torch.from_numpy(vector) * (sign * epsilon), prefill_sequence_length=sequence, injection_site="block_output")
        with hook: forward = run_hooked_forward(inference.model, inputs, modules, {PANL_POSITION: panl_position}, logits_positions=[sac])
        score = _score(forward, sac, list(class_ids)); panl = forward.hidden_by_name[PANL_POSITION][PANL_LAYER].detach().float().cpu().numpy().astype(np.float32); diag = hook.diagnostics()
        if int(diag["hook_call_count"]) != 1 or int(diag["steering_applied_count"]) != 1: raise RuntimeError("FD hook did not hit exactly once")
        before = hook.h_before.numpy(); after = hook.h_after.numpy()
        if array_hash(before) != baseline["clean_lat_hidden_hash"]: raise ValueError("FD pre-injection activation differs from validated clean baseline")
        displacement = float(np.linalg.norm(after.astype(np.float64) - before.astype(np.float64)))
        sides[sign] = {
            "panl_sa": _predict(panl_probe, panl), "final_sa": float(score["soft_sa_image_score"]),
            "sa_logits": score["class_logits"], "sa_probabilities": score["class_probabilities"], "hard_class": int(score["argmax_hard_class"]),
            "margin": class_margin(np.asarray(score["class_logits"], float), int(baseline["clean_hard_sa"])),
            "hook_hit_count": int(diag["hook_call_count"]), "steering_applied_count": int(diag["steering_applied_count"]),
            "activation_before_hash": array_hash(before), "activation_after_hash": array_hash(after), "actual_injection_norm": displacement,
        }
    return sides


def _gradient_worker(source_root: Path, random_root: Path, output_root: Path, worker: int, num_gpus: int, epsilons: Sequence[float], run_fingerprint: str) -> dict[str, Any]:
    paths = _gradient_paths(output_root); manifest = load_jsonl(source_root / "artifacts/manifests/runtime_manifest.jsonl")
    selected = [row for row in manifest if stable_shard(str(row["case_id"]), num_gpus) == worker]
    gradient_rows = _read_shards(paths["base"], "gradient_rows.shard_*.jsonl", gradient_row_key)
    fd_rows = _read_shards(paths["base"], "finite_difference.shard_*.jsonl", fd_row_key)
    historical = historical_pair_index(load_jsonl(source_root / "artifacts/trials/main_trials.jsonl"), TRUE_DIRECTIONS, tuple(epsilons) + HISTORICAL_EPSILONS)
    true_vectors, null_vectors, _ = _load_vector_maps(source_root, random_root)
    baselines = {str(row["case_id"]): row for row in load_jsonl(_baseline_path(random_root))}
    all_names = list(TRUE_DIRECTIONS) + [f"random_null_{replicate:03d}" for replicate in range(1, NULL_REPEATS + 1)]
    expected_gradient = {f'{row["case_id"]}|{endpoint}|{direction}|e{epsilon:g}' for row in selected for endpoint in ENDPOINTS for direction in all_names for epsilon in epsilons}
    missing_fd = [(str(row["case_id"]), direction, epsilon) for row in selected for direction in TRUE_DIRECTIONS for epsilon in epsilons if (str(row["case_id"]), direction, epsilon) not in historical and f'{row["case_id"]}|{direction}|e{epsilon:g}' not in fd_rows]
    missing_gradient_cases = [row for row in selected if any(key.startswith(str(row["case_id"]) + "|") and key not in gradient_rows for key in expected_gradient)]
    if not missing_fd and not missing_gradient_cases: return {"worker": worker, "new_differentiable_forwards": 0, "new_vjp": 0, "new_fd_forwards": 0, "resumed_noop": True}
    inference, modules, tokenizer, class_ids, device = _load_runtime_objects(source_root)
    for parameter in inference.model.parameters(): parameter.requires_grad_(False); parameter.grad = None
    panl_probe = joblib.load(source_root / "artifacts/probes/final_sa__P1_PANL__L18__full.joblib")["model"]
    true_meta_payload = json.loads((source_root / "artifacts/directions/vector_metadata.json").read_text())
    true_meta = {(row["recipient_answer"], row["direction"]): row for row in true_meta_payload["vectors"] if int(row["layer"]) == GRADIENT_LAYER}
    gradient_path = paths["base"] / f"gradient_rows.shard_{worker}.jsonl"; fd_path = paths["base"] / f"finite_difference.shard_{worker}.jsonl"
    new_clean = new_vjp = new_fd = 0
    for manifest_row in selected:
        case = str(manifest_row["case_id"]); answer = str(manifest_row["phase0_normalized_answer"]); baseline = baselines[case]
        pending_grad = [key for key in expected_gradient if key.startswith(case + "|") and key not in gradient_rows]
        pending_fd = [(direction, epsilon) for c, direction, epsilon in missing_fd if c == case]
        if not pending_grad and not pending_fd: continue
        rendered, inputs, positions = _render_and_locate(inference, tokenizer, manifest_row, device)
        if hashlib.sha256(rendered.encode()).hexdigest() != baseline["rendered_prompt_sha256"] or canonical_hash(positions) != baseline["position_sha256"]: raise ValueError(f"Runtime context mismatch: {case}")
        vectors = {direction: true_vectors[answer, direction] for direction in TRUE_DIRECTIONS}
        vectors.update({f"random_null_{replicate:03d}": null_vectors[answer, replicate] for replicate in range(1, NULL_REPEATS + 1)})
        gradient_file = paths["gradients"] / f"{case}.npz"
        diagnostics_file = paths["gradients"] / f"{case}.json"
        gradients: dict[str, np.ndarray]
        if pending_grad and gradient_file.is_file() and diagnostics_file.is_file():
            with np.load(gradient_file) as payload: gradients = {endpoint: np.asarray(payload[endpoint], np.float32) for endpoint in ENDPOINTS}
            diagnostics = json.loads(diagnostics_file.read_text())
            for endpoint in ENDPOINTS:
                if array_hash(gradients[endpoint]) != diagnostics[f"{endpoint}_gradient_sha256"]: raise ValueError(f"Saved gradient hash mismatch: {case}/{endpoint}")
        elif pending_grad:
            gradients, raw_diagnostics = differentiable_forward(inference.model, inputs, modules, lat_position=int(positions[STEERING_POSITION]["processed_index"]), panl_position=int(positions[PANL_POSITION]["processed_index"]), sac_position=int(positions["P1_SAC"]["processed_index"]), class_ids=class_ids, panl_probe=panl_probe)
            parity = _parity_check(raw_diagnostics, baseline)
            if not parity["passed"]: raise ValueError(f"Differentiable clean parity failed: {case}: {parity}")
            diagnostics = {k: v for k, v in raw_diagnostics.items() if k not in ("selected_logits", "probabilities")}
            diagnostics.update({"case_id": case, "parity": parity, "gradient_original_dtype": raw_diagnostics["gradient_original_dtype"]})
            for endpoint in ENDPOINTS: diagnostics[f"{endpoint}_gradient_sha256"] = array_hash(gradients[endpoint])
            atomic_npz(gradient_file, gradients); atomic_json(diagnostics_file, diagnostics); new_clean += 1; new_vjp += 2
        if pending_grad:
            for endpoint in ENDPOINTS:
                gradient_hash = array_hash(gradients[endpoint])
                dots = {}
                for direction, vector in vectors.items(): dots[direction] = directional_metrics(gradients[endpoint], vector)["directional_derivative"]
                add_error = relative_additivity_error(dots[TRUE_DIRECTIONS[0]], dots[TRUE_DIRECTIONS[1]], dots[TRUE_DIRECTIONS[2]])
                if add_error > 1e-5: raise ValueError(f"Gradient additivity failed: {case}/{endpoint}: {add_error}")
                for direction, vector in vectors.items():
                    kind = "true" if direction in TRUE_DIRECTIONS else "random_null"; replicate = None if kind == "true" else int(direction.rsplit("_", 1)[1])
                    for epsilon in epsilons:
                        proto = {"case_id": case, "endpoint": endpoint, "direction": direction, "epsilon": epsilon}
                        key = gradient_row_key(proto)
                        if key in gradient_rows: continue
                        result = _gradient_output_row(manifest_row, endpoint, direction, epsilon, directional_metrics(gradients[endpoint], vector), gradient_hash=gradient_hash, vector_hash=array_hash(vector), run_fingerprint=run_fingerprint, vector_kind=kind, null_replicate=replicate)
                        result["gradient_additivity_relative_error"] = add_error
                        append_jsonl(gradient_path, result); gradient_rows[key] = result
        for direction, epsilon in pending_fd:
            vector = vectors[direction]; sides = _steered_pair(inference, modules, inputs, positions, class_ids, vector, epsilon, panl_probe, baseline); new_fd += 2
            meta = true_meta[answer, direction]
            result = {
                "case_id": case, "item_id": str(manifest_row["item_id"]), "family_id": str(manifest_row["family_id"]), "condition": str(manifest_row["condition"]),
                "fixed_answer": answer, "answer_origin": "follow_text" if manifest_row["answer_matches_text"] and not manifest_row["answer_matches_image"] else "follow_image" if manifest_row["answer_matches_image"] and not manifest_row["answer_matches_text"] else "other",
                "direction": direction, "layer": GRADIENT_LAYER, "epsilon": epsilon, "minus": sides[-1.0], "plus": sides[1.0],
                "vector_sha256": array_hash(vector), "vector_fingerprint": meta["vector_fingerprint"], "actual_injection_norm": sides[1.0]["actual_injection_norm"],
                "signed_injection_natural_sd_plus": epsilon * float(meta["injection_to_natural_projection_sd"]), "gradient_validation_fingerprint": run_fingerprint, "status": "completed",
            }
            append_jsonl(fd_path, result); fd_rows[fd_row_key(result)] = result
    return {"worker": worker, "new_differentiable_forwards": new_clean, "new_vjp": new_vjp, "new_fd_forwards": new_fd, "resumed_noop": new_clean + new_fd == 0}


def _spawn_gradient_workers(source_root: Path, random_root: Path, output_root: Path, num_gpus: int, epsilons: Sequence[float], run_fingerprint: str) -> None:
    repo = Path(__file__).resolve().parents[2]; processes = []
    for worker in range(num_gpus):
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(worker); env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        command = [sys.executable, "-m", "dp_SA.confidence_steering.gradient_validation", "--worker", str(worker), "--num-gpus", str(num_gpus), "--source-root", str(source_root), "--random-root", str(random_root), "--output-root", str(output_root), "--run-fingerprint", run_fingerprint, "--gradient-epsilons", *map(str, epsilons)]
        processes.append(subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.DEVNULL))
    codes = [process.wait() for process in processes]
    if any(codes): raise RuntimeError(f"Gradient worker failed: {codes}")


def run_gradient_validation(source_root: Path, random_root: Path, output_root: Path, *, epsilons: Sequence[float], num_gpus: int, smoke: bool, resume: bool) -> dict[str, Any]:
    if num_gpus not in (1, 2): raise ValueError("num_gpus must be 1 or 2")
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus: raise RuntimeError(f"Requested {num_gpus} GPUs, visible={torch.cuda.device_count()}")
    epsilons = normalize_epsilons(epsilons); paths = _gradient_paths(output_root)
    prepared = json.loads((paths["progress"] / "gradient_validation_prepare.json").read_text()); fingerprint_config = json.loads((paths["base"] / "fingerprints.json").read_text())
    semantic = {key: value for key, value in fingerprint_config.items() if key != "fingerprint"}
    if prepared["config_fingerprint"] != canonical_hash(semantic): raise ValueError("Prepare fingerprint mismatch")
    run_fingerprint = prepared["config_fingerprint"]
    previous = json.loads((paths["progress"] / "gradient_validation_run.json").read_text()) if (paths["progress"] / "gradient_validation_run.json").is_file() else None
    if previous and previous.get("config_fingerprint") != run_fingerprint: raise ValueError("Gradient run fingerprint mismatch")
    if previous and not resume: raise FileExistsError("Gradient run exists; use --resume")
    gradient_before = len(list(paths["gradients"].glob("*.npz")))
    fd_before = len(_read_shards(paths["base"], "finite_difference.shard_*.jsonl", fd_row_key))
    started = time.time(); _spawn_gradient_workers(source_root, random_root, output_root, num_gpus, epsilons, run_fingerprint)
    gradient_rows = _read_shards(paths["base"], "gradient_rows.shard_*.jsonl", gradient_row_key); fd_rows = _read_shards(paths["base"], "finite_difference.shard_*.jsonl", fd_row_key)
    manifest = load_jsonl(source_root / "artifacts/manifests/runtime_manifest.jsonl"); case_ids = [str(row["case_id"]) for row in manifest]
    expected_gradient = len(case_ids) * len(ENDPOINTS) * (len(TRUE_DIRECTIONS) + NULL_REPEATS) * len(epsilons)
    if len(gradient_rows) != expected_gradient: raise ValueError(f"Gradient rows incomplete: {len(gradient_rows)}/{expected_gradient}")
    history = historical_pair_index(load_jsonl(source_root / "artifacts/trials/main_trials.jsonl"), TRUE_DIRECTIONS, epsilons)
    expected_fd_keys = {(case, direction, epsilon) for case in case_ids for direction in TRUE_DIRECTIONS for epsilon in epsilons if (case, direction, epsilon) not in history}
    actual_fd_keys = {(row["case_id"], row["direction"], float(row["epsilon"])) for row in fd_rows.values()}
    if actual_fd_keys != expected_fd_keys: raise ValueError(f"FD rows mismatch missing={len(expected_fd_keys-actual_fd_keys)} extra={len(actual_fd_keys-expected_fd_keys)}")
    atomic_jsonl(paths["base"] / "gradient_rows.jsonl", [gradient_rows[key] for key in sorted(gradient_rows)])
    atomic_jsonl(paths["base"] / "finite_difference.jsonl", [fd_rows[key] for key in sorted(fd_rows)])
    new_clean = len(list(paths["gradients"].glob("*.npz"))) - gradient_before; new_fd_pairs = len(fd_rows) - fd_before
    result = {"status": "complete", "config_fingerprint": run_fingerprint, "num_gpus_execution_metadata": num_gpus, "new_differentiable_forwards": new_clean, "new_vjp": 2 * new_clean, "new_fd_forwards": 2 * new_fd_pairs, "new_gpu_forwards": new_clean + 2 * new_fd_pairs, "gradient_rows": len(gradient_rows), "finite_difference_rows": len(fd_rows), "elapsed_seconds": time.time() - started, "resumed_noop": new_clean == 0 and new_fd_pairs == 0}
    atomic_json(paths["progress"] / "gradient_validation_run.json", result); return result


def _paired_rows(source_root: Path, output_root: Path, epsilons: Sequence[float]) -> list[dict[str, Any]]:
    own = {fd_row_key(row): row for row in load_jsonl(_gradient_paths(output_root)["base"] / "finite_difference.jsonl")}
    main_pairs = historical_pair_index(load_jsonl(source_root / "artifacts/trials/main_trials.jsonl"), TRUE_DIRECTIONS, tuple(epsilons) + HISTORICAL_EPSILONS)
    manifest = {str(row["case_id"]): row for row in load_jsonl(source_root / "artifacts/manifests/runtime_manifest.jsonl")}; output = []
    for case, row in manifest.items():
        for direction in TRUE_DIRECTIONS:
            for epsilon in tuple(epsilons) + HISTORICAL_EPSILONS:
                key = (case, direction, epsilon)
                if key in main_pairs:
                    minus = main_pairs[key][-1.0]; plus = main_pairs[key][1.0]
                    values = {endpoint: {"minus": float(minus["delta_panl_probe_sa" if endpoint == "panl_sa" else "delta_final_soft_sa"]), "plus": float(plus["delta_panl_probe_sa" if endpoint == "panl_sa" else "delta_final_soft_sa"])} for endpoint in ENDPOINTS}
                    source = "historical_main"
                else:
                    own_key = f"{case}|{direction}|e{epsilon:g}"
                    if own_key not in own: continue
                    trial = own[own_key]; values = {endpoint: {"minus": float(trial["minus"][endpoint]), "plus": float(trial["plus"][endpoint])} for endpoint in ENDPOINTS}; source = "new_fd"
                for endpoint in ENDPOINTS:
                    deriv = central_difference(values[endpoint]["plus"], values[endpoint]["minus"], epsilon)
                    output.append({"case_id": case, "item_id": str(row["item_id"]), "family_id": str(row["family_id"]), "condition": str(row["condition"]), "fixed_answer": str(row["phase0_normalized_answer"]), "answer_origin": "follow_text" if row["answer_matches_text"] and not row["answer_matches_image"] else "follow_image" if row["answer_matches_image"] and not row["answer_matches_text"] else "other", "direction": direction, "endpoint": endpoint, "epsilon": epsilon, "D": deriv["D"], "S": deriv["S"], "source": source})
    return output


def _bootstrap_value(rows: Sequence[dict[str, Any]], field: str, group: str, draws: Sequence[Sequence[str]]) -> dict[str, Any]:
    return summarize(rows, field, group, draws)


def _gradient_effect_table(gradient_rows: Sequence[dict[str, Any]], source_root: Path, random_root: Path, draws: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    # epsilon is duplicated by design in raw rows; use one copy for local gradients.
    first_epsilon = min(float(row["epsilon"]) for row in gradient_rows); unique = [row for row in gradient_rows if float(row["epsilon"]) == first_epsilon]
    output = []
    for endpoint in ENDPOINTS:
        for direction in TRUE_DIRECTIONS + tuple(f"random_null_{rep:03d}" for rep in range(1, NULL_REPEATS + 1)):
            selected = [row for row in unique if row["endpoint"] == endpoint and row["direction"] == direction]
            for group in GROUPS:
                summary = _bootstrap_value(selected, "directional_derivative", group, draws)
                output.append({"endpoint": endpoint, "diagnostic": "local_gradient", "direction": direction, "group": group, **summary})
    # Frozen LAT confidence is an analytic linear diagnostic, not another VJP.
    manifest = load_jsonl(source_root / "artifacts/manifests/runtime_manifest.jsonl"); true, null, _ = _load_vector_maps(source_root, random_root)
    lat_probe = joblib.load(source_root / "artifacts/probes/confidence_gap__P1_LAT__L14__full.joblib")["model"]; weight = raw_gradient(lat_probe).astype(np.float64)
    case_rows = []
    for row in manifest:
        answer = str(row["phase0_normalized_answer"])
        vectors = {direction: true[answer, direction] for direction in TRUE_DIRECTIONS}; vectors.update({f"random_null_{rep:03d}": null[answer, rep] for rep in range(1, NULL_REPEATS + 1)})
        for direction, vector in vectors.items(): case_rows.append({**{key: row[key] for key in ("case_id", "item_id", "family_id", "condition")}, "fixed_answer": answer, "direction": direction, "directional_derivative": float(weight @ vector.astype(np.float64))})
    for direction in TRUE_DIRECTIONS + tuple(f"random_null_{rep:03d}" for rep in range(1, NULL_REPEATS + 1)):
        selected = [row for row in case_rows if row["direction"] == direction]
        for group in GROUPS: output.append({"endpoint": "lat_confidence", "diagnostic": "linear_probe", "direction": direction, "group": group, **_bootstrap_value(selected, "directional_derivative", group, draws)})
    return output


def _finite_difference_table(paired: Sequence[dict[str, Any]], gradient_rows: Sequence[dict[str, Any]], draws: Sequence[Sequence[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gradient = {(row["case_id"], row["endpoint"], row["direction"]): float(row["directional_derivative"]) for row in gradient_rows}
    joined = []
    for row in paired:
        gv = gradient[row["case_id"], row["endpoint"], row["direction"]]
        joined.append({**row, "g_dot_v": gv, "predicted_S": float(row["epsilon"]) * gv, "signed_error": float(row["D"]) - gv, "absolute_error": abs(float(row["D"]) - gv)})
    output = []
    for endpoint in ENDPOINTS:
        for direction in TRUE_DIRECTIONS:
            for epsilon in sorted({float(row["epsilon"]) for row in joined}):
                selected = [row for row in joined if row["endpoint"] == endpoint and row["direction"] == direction and float(row["epsilon"]) == epsilon]
                if not selected: continue
                for group in GROUPS:
                    group_rows = selected if group == "all" else selected
                    metrics = prediction_metrics([row["D"] for row in group_rows], [row["g_dot_v"] for row in group_rows])
                    dsum = _bootstrap_value(selected, "D", group, draws); gsum = _bootstrap_value(selected, "g_dot_v", group, draws); esum = _bootstrap_value(selected, "signed_error", group, draws)
                    output.append({"endpoint": endpoint, "direction": direction, "epsilon": epsilon, "group": group, "mean_D": dsum["mean_delta"], "D_sem": dsum["sem"], "D_ci95_low": dsum["ci95_low"], "D_ci95_high": dsum["ci95_high"], "mean_g_dot_v": gsum["mean_delta"], "g_ci95_low": gsum["ci95_low"], "g_ci95_high": gsum["ci95_high"], "signed_error_ci95_low": esum["ci95_low"], "signed_error_ci95_high": esum["ci95_high"], **metrics})
    additivity = []
    lookup = {(row["case_id"], row["endpoint"], row["direction"], float(row["epsilon"])): row for row in joined}
    for endpoint in ENDPOINTS:
        for epsilon in sorted({float(row["epsilon"]) for row in joined}):
            rows = []
            for case in sorted({row["case_id"] for row in joined}):
                keys = [(case, endpoint, direction, epsilon) for direction in TRUE_DIRECTIONS]
                if not all(key in lookup for key in keys): continue
                raw, parallel, perp = (lookup[key] for key in keys)
                rows.append({**{k: raw[k] for k in ("case_id", "item_id", "family_id", "condition", "fixed_answer")}, "additivity_error": raw["S"] - parallel["S"] - perp["S"]})
            for group in GROUPS:
                summary = _bootstrap_value(rows, "additivity_error", group, draws)
                additivity.append({"endpoint": endpoint, "direction": "component_additivity", "epsilon": epsilon, "group": group, **summary})
    return output + additivity, joined


def _group_rows(rows: Sequence[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    # Macro weighting is applied to aggregate means/bootstrap; case-level predictive
    # diagnostics retain all paired cases and are explicitly labelled case-level.
    return list(rows)


def _null_case_rows(source_root: Path, random_root: Path, paired: Sequence[dict[str, Any]], gradient_rows: Sequence[dict[str, Any]], endpoint: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    epsilon = min(float(row["epsilon"]) for row in gradient_rows)
    local = [dict(row) for row in gradient_rows if row["endpoint"] == endpoint and float(row["epsilon"]) == epsilon and (row["direction"] == TRUE_DIRECTIONS[2] or row["vector_kind"] == "random_null")]
    true_d2 = [dict(row, value=float(row["D"]), null_replicate=None, vector_kind="true") for row in paired if row["endpoint"] == endpoint and row["direction"] == TRUE_DIRECTIONS[2] and float(row["epsilon"]) == 2.0]
    random_trials = load_jsonl(random_root / "artifacts/trials/random_sa_subspace_null_trials.jsonl")
    field = "delta_panl_probe_sa" if endpoint == "panl_sa" else "delta_final_soft_sa"
    pairs = random_null_pair_index(random_trials)
    manifest = {str(row["case_id"]): row for row in load_jsonl(source_root / "artifacts/manifests/runtime_manifest.jsonl")}; null_d2 = []
    for (case, replicate), sides in pairs.items():
        m = manifest[case]
        null_d2.append({"case_id": case, "item_id": str(m["item_id"]), "family_id": str(m["family_id"]), "condition": str(m["condition"]), "fixed_answer": str(m["phase0_normalized_answer"]), "null_replicate": replicate, "vector_kind": "random_null", "direction": f"random_null_{replicate:03d}", "value": (float(sides[1.0][field]) - float(sides[-1.0][field])) / 4.0})
    return local, true_d2 + null_d2


def _null_tables(source_root: Path, random_root: Path, paired: Sequence[dict[str, Any]], gradient_rows: Sequence[dict[str, Any]], draws: Sequence[Sequence[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparison = []; existing = []
    for endpoint in ENDPOINTS:
        local, d2 = _null_case_rows(source_root, random_root, paired, gradient_rows, endpoint)
        local_values = {(row["case_id"], row["direction"]): float(row["directional_derivative"]) for row in local}
        d2_values = {(row["case_id"], row["direction"]): float(row["value"]) for row in d2}
        for group in GROUPS:
            true_local_rows = [dict(row, value=float(row["directional_derivative"])) for row in local if row["direction"] == TRUE_DIRECTIONS[2]]
            true_d2_rows = [row for row in d2 if row["vector_kind"] == "true"]
            for scale, rows, true_rows in (("local_gradient", local, true_local_rows), ("existing_D2", d2, true_d2_rows)):
                null_effects = []
                for replicate in range(1, NULL_REPEATS + 1):
                    direction = f"random_null_{replicate:03d}"
                    selected = [dict(row, value=float(row["directional_derivative"])) if scale == "local_gradient" else row for row in rows if row["direction"] == direction]
                    if selected: null_effects.append((replicate, _mean(selected, "value", group)))
                true_effect = _mean(true_rows, "value", group); values = [value for _, value in null_effects]
                negative = sum(value <= true_effect for value in values); two = sum(abs(value) >= abs(true_effect) for value in values)
                # Family CI conditions on the frozen true/null directions. Nulls are
                # averaged within case before paired resampling.
                true_by_case = {row["case_id"]: row for row in true_rows}; null_by_case: dict[str, list[float]] = defaultdict(list)
                for row in rows:
                    if row.get("vector_kind") == "random_null": null_by_case[row["case_id"]].append(float(row["directional_derivative"] if scale == "local_gradient" else row["value"]))
                diff_rows = [{**row, "difference": float(row["value"]) - float(np.mean(null_by_case[case]))} for case, row in true_by_case.items() if case in null_by_case]
                diff_summary = summarize(diff_rows, "difference", group, draws)
                comparison.append({"endpoint": endpoint, "group": group, "comparison_scale": scale, "true_effect": true_effect, "null_mean": float(np.mean(values)), "null_sd": float(np.std(values, ddof=1)), "null_min": min(values), "null_max": max(values), "null_count": len(values), "empirical_p_negative_one_sided": (1 + negative) / (len(values) + 1), "empirical_p_two_sided": (1 + two) / (len(values) + 1), "minimum_attainable_p": 1 / (len(values) + 1), "true_minus_null_mean": diff_summary["mean_delta"], "paired_family_ci95_low": diff_summary["ci95_low"], "paired_family_ci95_high": diff_summary["ci95_high"]})
        null_summary = []
        available_for_association = []
        for replicate in range(1, NULL_REPEATS + 1):
            direction = f"random_null_{replicate:03d}"; g_rows = [row for row in local if row["direction"] == direction]; d_rows = [row for row in d2 if row["direction"] == direction]
            if not g_rows: continue
            gv = _mean(g_rows, "directional_derivative", "answer_equal_macro"); derivative = _mean(d_rows, "value", "answer_equal_macro") if d_rows else None
            row = {"endpoint": endpoint, "row_type": "null", "null_replicate": replicate, "g_dot_v": gv, "D2": derivative, "D2_minus_g_dot_v": derivative - gv if derivative is not None else None, "sign_agreement": int(np.sign(derivative) == np.sign(gv)) if derivative is not None else None}
            null_summary.append(row)
            if derivative is not None: available_for_association.append(row)
        if len(available_for_association) >= 2:
            gx = [row["g_dot_v"] for row in available_for_association]; dy = [row["D2"] for row in available_for_association]
            association = {"endpoint": endpoint, "row_type": "association", "null_replicate": None, "g_dot_v": None, "D2": None, "D2_minus_g_dot_v": None, "pearson": float(pearsonr(gx, dy).statistic), "spearman": float(spearmanr(gx, dy).statistic), "sign_agreement": float(np.mean([row["sign_agreement"] for row in available_for_association])), "n": len(available_for_association)}
            existing.extend(null_summary + [association])
    return comparison, existing


def _plot_gradient_null(comparison: Sequence[dict[str, Any]], existing: Sequence[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, endpoint, title in zip(axes, ENDPOINTS, ("PANL L18 SA", "Final soft SA"), strict=True):
        points = [row for row in existing if row["endpoint"] == endpoint and row["row_type"] == "null"]
        summary = next(row for row in comparison if row["endpoint"] == endpoint and row["group"] == "answer_equal_macro" and row["comparison_scale"] == "local_gradient")
        ax.scatter([row["null_replicate"] for row in points], [row["g_dot_v"] for row in points], s=30, color="#4C78A8", label="20 random nulls")
        ax.axhline(float(summary["true_effect"]), color="#D62728", lw=2, label="true natural-perp")
        ax.axhline(0, color="black", lw=.7); ax.set_title(title); ax.set_xlabel("global null replicate"); ax.set_ylabel(r"clean local $g^T v$"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def _plot_fd(joined: Sequence[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7)); colors = plt.cm.viridis(np.linspace(0, 1, len(sorted({row["epsilon"] for row in joined})))); color_map = dict(zip(sorted({row["epsilon"] for row in joined}), colors, strict=True)); markers = dict(zip(TRUE_DIRECTIONS, ("o", "s", "^"), strict=True))
    for ax, endpoint, title in zip(axes, ENDPOINTS, ("PANL L18 SA", "Final soft SA"), strict=True):
        selected = [row for row in joined if row["endpoint"] == endpoint]
        for direction in TRUE_DIRECTIONS:
            for epsilon in sorted(color_map):
                cells = [row for row in selected if row["direction"] == direction and row["epsilon"] == epsilon]
                ax.scatter([row["g_dot_v"] for row in cells], [row["D"] for row in cells], s=10, alpha=.45, marker=markers[direction], color=color_map[epsilon], label=f"{direction}, ε={epsilon:g}")
        limits = ax.get_xlim() + ax.get_ylim(); low, high = min(limits), max(limits); ax.plot([low, high], [low, high], "k--", lw=1); ax.set_xlim(low, high); ax.set_ylim(low, high); ax.set_title(title); ax.set_xlabel(r"case-level $g^T v$"); ax.set_ylabel(r"case-level $D^\epsilon$")
    handles, labels = axes[1].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7); fig.tight_layout(rect=(0, .14, 1, 1)); fig.savefig(path, dpi=300); plt.close(fig)


def analyze_gradient_validation(source_root: Path, random_root: Path, output_root: Path, *, epsilons: Sequence[float], smoke: bool, resume: bool) -> dict[str, Any]:
    paths = _gradient_paths(output_root); run_result = json.loads((paths["progress"] / "gradient_validation_run.json").read_text()); progress = paths["progress"] / "gradient_validation_analyze.json"
    if progress.is_file():
        previous = json.loads(progress.read_text())
        if previous.get("config_fingerprint") != run_result["config_fingerprint"]: raise ValueError("Analyze fingerprint mismatch")
        if resume and previous.get("status") == "complete": return {**previous, "resumed_noop": True}
        if not resume: raise FileExistsError("Gradient analysis exists; use --resume")
    gradient_rows = load_jsonl(paths["base"] / "gradient_rows.jsonl"); paired = _paired_rows(source_root, output_root, epsilons)
    draws, draw_hash = family_draws(load_jsonl(source_root / "artifacts/manifests/runtime_manifest.jsonl"), BOOTSTRAP_REPEATS)
    directional = _gradient_effect_table(gradient_rows, source_root, random_root, draws); fd_table, joined = _finite_difference_table(paired, gradient_rows, draws); comparison, existing = _null_tables(source_root, random_root, paired, gradient_rows, draws)
    atomic_csv(paths["tables"] / "gradient_directional_effects.csv", directional); atomic_csv(paths["tables"] / "gradient_finite_difference_validation.csv", fd_table); atomic_csv(paths["tables"] / "gradient_vs_existing_s2.csv", existing); atomic_csv(paths["tables"] / "gradient_random_null_comparison.csv", comparison)
    _plot_gradient_null(comparison, existing, paths["figures"] / "gradient_directional_effects.png"); _plot_fd(joined, paths["figures"] / "gradient_finite_difference_validation.png")
    gate = {"epsilon": 0.5, "primary_endpoint": "panl_sa", "final_sa_role": "diagnostic_only_due_to_bfloat16_logit_resolution", "endpoints": {}, "passed": True}
    for endpoint in ENDPOINTS:
        selected = [row for row in joined if row["endpoint"] == endpoint and float(row["epsilon"]) == .5]
        metrics = prediction_metrics([row["D"] for row in selected], [row["g_dot_v"] for row in selected]); direction_signs = {}
        for direction in TRUE_DIRECTIONS:
            cells = [row for row in selected if row["direction"] == direction]; d = _mean(cells, "D", "answer_equal_macro"); g = _mean(cells, "g_dot_v", "answer_equal_macro"); direction_signs[direction] = bool((abs(d) <= 1e-8 and abs(g) <= 1e-8) or np.sign(d) == np.sign(g))
        passed = bool(metrics["pearson"] >= .8 and metrics["sign_agreement"] >= .75 and metrics["nrmse"] <= .5 and all(direction_signs.values()))
        participates = endpoint == "panl_sa"
        gate["endpoints"][endpoint] = {**metrics, "direction_aggregate_sign_agreement": direction_signs, "participates_in_lock_gate": participates, "passed": passed}
        if participates: gate["passed"] = gate["passed"] and passed
    summary = f"# LAT→PANL/SAC 局部梯度验证\n\n- 配置：post-hoc 机制诊断；不称为传播路径。\n- natural-perp：删除已测量线性 SA 子空间后的自然尺度分量，不称为纯 confidence。\n- 小剂量固定为 `ε=0.5`，选择仅依据与正式 test 隔离的 audit smoke，并非查看正式结果后挑参。\n- PANL-SA 是小剂量梯度验证的主要 endpoint；final-SA 受 bf16 离散 logits 分辨率限制，仅作诊断，不参与签锁。\n- smoke PANL 门禁：`{'passed' if gate['passed'] else 'failed'}`。\n- 20 个几何 null 的最小经验 p 为 `1/21=0.047619`；它衡量方向在随机几何中的特殊性。\n- paired family-bootstrap CI 衡量给定这些冻结方向后对相似 family 的推广，不用于产生更小的几何 null p 值。\n"
    atomic_text(output_root / "gradient_validation_summary.md", summary)
    result = {"status": "complete", "config_fingerprint": run_result["config_fingerprint"], "bootstrap_draw_fingerprint": draw_hash, "bootstrap_repeats": BOOTSTRAP_REPEATS, "smoke_gate": gate, "directional_rows": len(directional), "finite_difference_rows": len(fd_table), "null_comparison_rows": len(comparison), "resumed_noop": False}
    atomic_json(progress, result); return result


def gradient_protocol_material(epsilons: Sequence[float]) -> dict[str, Any]:
    model_hash, processor_hash = model_processor_hashes()
    return {"format_version": 2, "layer": GRADIENT_LAYER, "epsilons": list(normalize_epsilons(epsilons)), "directions": list(TRUE_DIRECTIONS), "endpoints": list(ENDPOINTS), "null_repeats": NULL_REPEATS, "seed": SEED, "bootstrap_repeats": BOOTSTRAP_REPEATS, "same_point_definition": "P1_LAT L14 decoder block output pre-final-norm; detached cloned leaf reinserted at the same token", "fd_formulas": {"D": "(Y(+e)-Y(-e))/(2e)", "S": "(Y(+e)-Y(-e))/2", "predicted_S": "e*g_dot_v"}, "dose_selection": {"source": "formal-test-isolated audit smoke", "candidates": list(DOSE_SELECTION_CANDIDATES), "rule": "smallest dose passing PANL-SA gate", "selected": .5, "not_post_formal_selection": True}, "smoke_gate": {"epsilon": .5, "primary_endpoint": "panl_sa", "final_sa": "diagnostic_only", "pearson_min": .8, "sign_min": .75, "nrmse_max": .5, "direction_macro_sign": True}, "formal_scope": {"clean_gradient_cases": 100, "new_small_dose": .5, "reuse_existing_dose": 2.0, "epsilon_1_role": "auxiliary_existing_only"}, "model_files_sha256": model_hash, "processor_files_sha256": processor_hash, "code_hashes": gradient_code_hashes()}


def _next_smoke_root(protocol_fingerprint: str) -> tuple[int, Path]:
    parent = GRADIENT_SMOKE_PARENT / f"config_{protocol_fingerprint[:12]}"; parent.mkdir(parents=True, exist_ok=True)
    for number in range(1, 6):
        root = parent / f"round_{number}"
        if not root.exists(): return number, root
    raise RuntimeError("All five gradient-validation smoke rounds have been used")


def _matching_smoke_lock(protocol_fingerprint: str) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for path in GRADIENT_SMOKE_PARENT.glob("config_*/round_*/progress/gradient_validation_lock.json"):
        lock = json.loads(path.read_text())
        if lock.get("status") == "locked" and lock.get("static_lock_fingerprint") == protocol_fingerprint: candidates.append((path, lock))
    if not candidates: raise RuntimeError("Formal gradient validation is blocked: no matching successful smoke lock")
    return max(candidates, key=lambda item: item[0].stat().st_mtime)


def _selection_evidence_smoke() -> Path:
    candidates = []
    for path in GRADIENT_SMOKE_PARENT.glob("config_*/round_*"):
        run_path = path / "progress/gradient_validation_run.json"; gradient_path = path / "artifacts/gradient_validation/gradient_rows.jsonl"; fd_path = path / "artifacts/gradient_validation/finite_difference.jsonl"
        if not all(value.is_file() for value in (run_path, gradient_path, fd_path)): continue
        run = json.loads(run_path.read_text())
        if int(run.get("gradient_rows", 0)) == 3312 and int(run.get("finite_difference_rows", 0)) == 216:
            candidates.append(path)
    if not candidates: raise RuntimeError("No completed formal-test-isolated three-dose audit smoke exists")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _dose_selection_evidence(source_root: Path, smoke_root: Path) -> dict[str, Any]:
    gradients = load_jsonl(smoke_root / "artifacts/gradient_validation/gradient_rows.jsonl"); paired = _paired_rows(source_root, smoke_root, (.25, .5))
    lookup = {(row["case_id"], row["endpoint"], row["direction"]): float(row["directional_derivative"]) for row in gradients}; evidence = {}
    selected_dose = None
    for epsilon in DOSE_SELECTION_CANDIDATES:
        evidence[str(epsilon)] = {}
        for endpoint in ENDPOINTS:
            cells = [row for row in paired if row["endpoint"] == endpoint and float(row["epsilon"]) == epsilon]
            metrics = prediction_metrics([row["D"] for row in cells], [lookup[row["case_id"], endpoint, row["direction"]] for row in cells])
            direction_signs = {}
            for direction in TRUE_DIRECTIONS:
                rows = [row for row in cells if row["direction"] == direction]; d = _mean(rows, "D", "answer_equal_macro"); g = _mean([{**row, "g": lookup[row["case_id"], endpoint, direction]} for row in rows], "g", "answer_equal_macro")
                direction_signs[direction] = bool((abs(d) <= 1e-8 and abs(g) <= 1e-8) or np.sign(d) == np.sign(g))
            passed = bool(metrics["pearson"] >= .8 and metrics["sign_agreement"] >= .75 and metrics["nrmse"] <= .5 and all(direction_signs.values()))
            evidence[str(epsilon)][endpoint] = {**metrics, "direction_aggregate_sign_agreement": direction_signs, "passes_original_numeric_thresholds": passed}
        if selected_dose is None and evidence[str(epsilon)]["panl_sa"]["passes_original_numeric_thresholds"]: selected_dose = epsilon
    if selected_dose != .5: raise RuntimeError(f"Frozen dose-selection rule did not select epsilon=.5: {selected_dose}")
    return {"source_smoke_root": str(smoke_root.resolve()), "source_gradient_sha256": sha256_file(smoke_root / "artifacts/gradient_validation/gradient_rows.jsonl"), "source_fd_sha256": sha256_file(smoke_root / "artifacts/gradient_validation/finite_difference.jsonl"), "candidate_metrics": evidence, "selection_rule": "smallest candidate passing original PANL-SA thresholds", "selected_epsilon": selected_dose, "formal_test_was_not_read": True}


def _reuse_selected_smoke(source_root: Path, random_root: Path, evidence_root: Path, destination: Path, fingerprint: str) -> dict[str, Any]:
    paths = _gradient_paths(destination); gradient_rows = [row for row in load_jsonl(evidence_root / "artifacts/gradient_validation/gradient_rows.jsonl") if float(row["epsilon"]) == .5]; fd_rows = [row for row in load_jsonl(evidence_root / "artifacts/gradient_validation/finite_difference.jsonl") if float(row["epsilon"]) == .5]
    if len(gradient_rows) != 24 * 2 * 23 or len(fd_rows) != 24 * 3: raise ValueError("Selected epsilon=.5 smoke subset is incomplete")
    for row in gradient_rows: row["gradient_validation_fingerprint"] = fingerprint; row["reused_from_audit_smoke"] = str(evidence_root.resolve())
    for row in fd_rows: row["gradient_validation_fingerprint"] = fingerprint; row["reused_from_audit_smoke"] = str(evidence_root.resolve())
    atomic_jsonl(paths["base"] / "gradient_rows.shard_0.jsonl", sorted(gradient_rows, key=gradient_row_key)); atomic_jsonl(paths["base"] / "gradient_rows.jsonl", sorted(gradient_rows, key=gradient_row_key)); atomic_jsonl(paths["base"] / "finite_difference.shard_0.jsonl", sorted(fd_rows, key=fd_row_key)); atomic_jsonl(paths["base"] / "finite_difference.jsonl", sorted(fd_rows, key=fd_row_key))
    source_gradients = evidence_root / "artifacts/gradient_validation/gradients"
    for source in source_gradients.glob("*"): shutil.copy2(source, paths["gradients"] / source.name)
    result = {"status": "complete", "config_fingerprint": fingerprint, "num_gpus_execution_metadata": 0, "new_differentiable_forwards": 0, "new_vjp": 0, "new_fd_forwards": 0, "new_gpu_forwards": 0, "reused_audit_clean_forwards": 24, "reused_audit_vjp": 48, "reused_audit_fd_forwards_at_selected_dose": 144, "gradient_rows": len(gradient_rows), "finite_difference_rows": len(fd_rows), "elapsed_seconds": 0.0, "resumed_noop": False, "reuse_only": True}
    atomic_json(paths["progress"] / "gradient_validation_run.json", result); return result


def run_gradient_pipeline(*, epsilons: Sequence[float], smoke: bool, resume: bool, num_gpus: int, output_root: Path | None = None) -> dict[str, Any]:
    epsilons = normalize_epsilons(epsilons)
    if epsilons != DEFAULT_EPSILONS: raise ValueError(f"Locked gradient protocol requires epsilons={DEFAULT_EPSILONS}")
    protocol_fingerprint = canonical_hash(gradient_protocol_material(epsilons))
    source_root, random_root = _source_roots(smoke)
    if smoke:
        number, destination = _next_smoke_root(protocol_fingerprint) if output_root is None else (1, Path(output_root).resolve())
        if destination.exists() and not resume: raise FileExistsError(destination)
        destination.mkdir(parents=True, exist_ok=True); failures = []
        source_before = inventory(source_root); random_before = inventory(random_root)
        try:
            evidence_root = _selection_evidence_smoke(); selection = _dose_selection_evidence(source_root, evidence_root)
            prepared = prepare_gradient_validation(source_root, random_root, destination, epsilons=epsilons, smoke=True, resume=resume)
            ran = _reuse_selected_smoke(source_root, random_root, evidence_root, destination, prepared["config_fingerprint"])
            analyzed = analyze_gradient_validation(source_root, random_root, destination, epsilons=epsilons, smoke=True, resume=resume)
            if ran["new_gpu_forwards"] != 0 or ran["new_vjp"] != 0: raise RuntimeError(f"Protocol amendment unexpectedly ran GPU work: {ran}")
            if not analyzed["smoke_gate"]["passed"]: raise RuntimeError(f"Real-model local-gradient smoke gate failed: {analyzed['smoke_gate']}")
            resumed_prepare = prepare_gradient_validation(source_root, random_root, destination, epsilons=epsilons, smoke=True, resume=True)
            resumed_run = run_gradient_validation(source_root, random_root, destination, epsilons=epsilons, num_gpus=num_gpus, smoke=True, resume=True)
            resumed_analyze = analyze_gradient_validation(source_root, random_root, destination, epsilons=epsilons, smoke=True, resume=True)
            if not resumed_run["resumed_noop"] or resumed_run["new_gpu_forwards"] or resumed_run["new_vjp"]: raise RuntimeError("Gradient smoke resume was not a zero-work no-op")
            if inventory(source_root) != source_before or inventory(random_root) != random_before: raise RuntimeError("Historical source artifacts changed during gradient smoke")
            lock = {"status": "locked", "format_version": 2, "static_lock_fingerprint": protocol_fingerprint, "smoke_round": number, "source_root": str(source_root), "random_root": str(random_root), "output_root": str(destination.resolve()), "prepare_fingerprint": prepared["config_fingerprint"], "run_fingerprint": ran["config_fingerprint"], "smoke_gate": analyzed["smoke_gate"], "dose_selection_evidence": selection, "protocol_interpretation_boundaries": {"selection_used_only_formal_test_isolated_audit_smoke": True, "selection_is_not_post_formal_result_tuning": True, "panl_sa_is_small_dose_primary": True, "final_sa_is_diagnostic_only": True, "formal_primary_components": ["100-case clean gradients", "existing paired dose 2"], "only_new_formal_dose": .5}, "new_forwards_for_amendment": ran["new_gpu_forwards"], "new_vjp_for_amendment": ran["new_vjp"], "resume_new_forwards": resumed_run["new_gpu_forwards"], "resume_new_vjp": resumed_run["new_vjp"], "formal_forward_budget": 700, "formal_vjp_budget": 200, "source_inventory": source_before, "random_inventory": random_before}
            lock["fingerprint"] = canonical_hash(lock); atomic_json(_gradient_paths(destination)["progress"] / "gradient_validation_lock.json", lock)
            report = {"status": "passed", "smoke": True, "round": number, "output_root": str(destination.resolve()), "prepare": prepared, "run": ran, "analyze": analyzed, "resume": {"prepare": resumed_prepare, "run": resumed_run, "analyze": resumed_analyze}, "lock": lock, "formal_started": False}
            atomic_json(_gradient_paths(destination)["progress"] / "gradient_validation_smoke_report.json", report); return report
        except Exception as exc:
            failures.append({"type": type(exc).__name__, "message": str(exc)}); _ensure_layout(destination); atomic_json(_gradient_paths(destination)["progress"] / "gradient_validation_smoke_failure.json", failures[-1]); raise
    if not resume: raise ValueError("Formal gradient validation adds files to natural_decomposition; pass --resume explicitly")
    destination = Path(output_root or NATURAL_FORMAL_ROOT).resolve(); lock_path, lock = _matching_smoke_lock(protocol_fingerprint)
    prepared = prepare_gradient_validation(source_root, random_root, destination, epsilons=epsilons, smoke=False, resume=True)
    if lock["static_lock_fingerprint"] != protocol_fingerprint: raise RuntimeError("Formal gradient protocol does not match smoke lock")
    ran = run_gradient_validation(source_root, random_root, destination, epsilons=epsilons, num_gpus=num_gpus, smoke=False, resume=True)
    analyzed = analyze_gradient_validation(source_root, random_root, destination, epsilons=epsilons, smoke=False, resume=True)
    if inventory(source_root) != prepared["protected_before"]: raise RuntimeError("Historical formal artifacts changed")
    return {"status": "complete", "smoke": False, "source_smoke_lock": str(lock_path), "prepare": prepared, "run": ran, "analyze": analyzed}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--worker", type=int, required=True); parser.add_argument("--num-gpus", type=int, choices=(1, 2), required=True); parser.add_argument("--source-root", type=Path, required=True); parser.add_argument("--random-root", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--run-fingerprint", required=True); parser.add_argument("--gradient-epsilons", type=float, nargs="+", required=True)
    args = parser.parse_args(argv); result = _gradient_worker(args.source_root, args.random_root, args.output_root, args.worker, args.num_gpus, normalize_epsilons(args.gradient_epsilons), args.run_fingerprint); print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
