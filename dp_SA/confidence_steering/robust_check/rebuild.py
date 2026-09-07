from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dp_SA.confidence_steering.core import (
    HiddenResolver,
    answer_patterns,
    fit_oof_probe,
    loao,
    natural_sa_decomposition,
    prepare_train_rows,
    raw_gradient,
    regression_metrics,
    svd_basis,
    target_norm,
    weighted_sa_probe,
)

from .config import (
    BOOTSTRAP_REPEATS, CANONICAL_COLORS, DIRECTIONS, FROZEN_VECTORS,
    HIDDEN_REUSE, HIDDEN_SIZE, LAYER, PANL_LAYER, RECONSTRUCTION_RTOL,
)
from .io_utils import (
    array_hash, atomic_csv, atomic_joblib, atomic_json, atomic_jsonl,
    atomic_npz, canonical_hash, load_jsonl, sha256_file,
)


def direction_sign_audit(gradient: np.ndarray, scaled: dict[str, np.ndarray]) -> dict[str, Any]:
    """Report target-fixed signs without modifying any supplied vector."""
    before = {name: array_hash(np.asarray(value)) for name, value in scaled.items()}
    dots = {name: float(np.asarray(gradient, np.float64) @ np.asarray(value, np.float64)) for name, value in scaled.items()}
    after = {name: array_hash(np.asarray(value)) for name, value in scaled.items()}
    if before != after:
        raise RuntimeError("Sign audit mutated a direction")
    raw = dots["confidence_raw"]
    return {
        "confidence_probe_dot_raw": raw,
        "confidence_probe_dot_parallel": dots["confidence_parallel_sa"],
        "confidence_probe_dot_perpendicular": dots["confidence_perp_sa_natural_scale"],
        "direction_sign_inconsistent": bool(raw < 0),
        "direction_sign_degenerate": bool(raw == 0),
        "sign_policy": "fixed_by_G_L_no_posthoc_flip",
    }


def enrich_train_rows(manifest: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = {row["case_id"]: row for row in prepare_train_rows(manifest)}
    return [{**raw, **prepared[str(raw["case_id"])]} for raw in manifest]


def load_hidden(rows: Sequence[dict[str, Any]], root: Path) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    resolver = HiddenResolver()
    allowed = {str(row["case_id"]) for row in rows}
    reuse = {str(row["case_id"]): row for row in load_jsonl(HIDDEN_REUSE) if str(row["case_id"]) in allowed}
    hidden: dict[str, dict[str, np.ndarray]] = {}
    audit = []
    for row in rows:
        case = str(row["case_id"])
        lat = resolver.load(case, f"P1_LAT__L{LAYER}")
        panl = resolver.load(case, f"P1_PANL__L{PANL_LAYER}")
        lat_source = reuse[case]["cell_sources"][f"P1_LAT__L{LAYER}"]
        panl_source = reuse[case]["cell_sources"][f"P1_PANL__L{PANL_LAYER}"]
        hidden[case] = {"lat": lat, "panl": panl}
        audit.append({
            "case_id": case,
            "family_id": str(row["family_id"]),
            "lat_key": f"P1_LAT__L{LAYER}",
            "panl_key": f"P1_PANL__L{PANL_LAYER}",
            "lat_tensor_sha256": array_hash(lat.astype(np.float16)),
            "panl_tensor_sha256": array_hash(panl.astype(np.float16)),
            "lat_expected_tensor_sha256": lat_source["tensor_sha256"],
            "panl_expected_tensor_sha256": panl_source["tensor_sha256"],
            "lat_source_file": lat_source["path"],
            "panl_source_file": panl_source["path"],
            "lat_source_file_sha256": lat_source["file_sha256"],
            "panl_source_file_sha256": panl_source["file_sha256"],
            "lat_shape": list(lat.shape),
            "panl_shape": list(panl.shape),
            "prompt_sha256": str(row["phase1_prompt_hash"]),
            "image_sha256": str(row["image_sha256"]),
            "lat_position_index": int(row["positions"]["P1_LAT"]["processed_index"]),
            "lat_token_id": int(row["positions"]["P1_LAT"]["token_id"]),
            "panl_position_index": int(row["positions"]["P1_PANL"]["processed_index"]),
            "panl_token_id": int(row["positions"]["P1_PANL"]["token_id"]),
            "source_component": lat_source["source"],
            "source_files_validated": True,
        })
    atomic_jsonl(root / "artifacts/hidden_reuse/validated_manifest.jsonl", audit)
    atomic_json(root / "artifacts/hidden_reuse/summary.json", {
        "status": "passed", "records": len(audit), "keys_per_record": 2,
        "hidden_definition": "decoder_block_output_pre_final_norm",
        "dtype_on_disk": "float16", "loaded_dtype": "float32",
    })
    return hidden, audit


def _bootstrap_metrics(y: np.ndarray, prediction: np.ndarray, families: Sequence[str], seed: int, repeats: int) -> dict[str, Any]:
    family_array = np.asarray(list(map(str, families)))
    ordered = sorted(set(family_array))
    indices = {family: np.flatnonzero(family_array == family) for family in ordered}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
    samples = {name: [] for name in ("r2", "pearson", "spearman", "mae")}
    for _ in range(repeats):
        take = np.concatenate([indices[ordered[index]] for index in rng.integers(0, len(ordered), len(ordered))])
        for name, value in regression_metrics(y[take], prediction[take]).items():
            if math.isfinite(value):
                samples[name].append(value)
    result: dict[str, Any] = {}
    for name, values in samples.items():
        if values:
            low, high = np.percentile(values, [2.5, 97.5])
            result.update({f"{name}_ci_low": float(low), f"{name}_ci_high": float(high), f"{name}_bootstrap_valid": len(values)})
        else:
            result.update({f"{name}_ci_low": None, f"{name}_ci_high": None, f"{name}_bootstrap_valid": 0})
    return result


def fit_probe(
    root: Path,
    seed: int,
    name: str,
    construction: Sequence[dict[str, Any]],
    audit: Sequence[dict[str, Any]],
    hidden: dict[str, dict[str, np.ndarray]],
    hidden_name: str,
    target: str,
    repeats: int = BOOTSTRAP_REPEATS,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    X = np.stack([hidden[str(row["case_id"])][hidden_name] for row in construction])
    y = np.asarray([float(row[target]) for row in construction])
    folds = [int(row["outer_fold"]) for row in construction]
    oof, fold_models, oof_metrics, model, alpha, trace = fit_oof_probe(X, y, construction, folds)
    probe_root = root / f"artifacts/probes/seed_{seed}"
    fold_hashes = {}
    for fold, fold_model in fold_models.items():
        path = probe_root / f"{name}__fold_{fold}.joblib"
        atomic_joblib(path, {"model": fold_model, "held_out_fold": fold, "seed": seed})
        fold_hashes[str(fold)] = sha256_file(path)
    full_path = probe_root / f"{name}__full.joblib"
    payload = {
        "model": model, "seed": seed, "target": target,
        "position": "P1_LAT" if hidden_name == "lat" else "P1_PANL",
        "layer": LAYER if hidden_name == "lat" else PANL_LAYER,
        "selected_alpha": alpha, "alpha_trace": trace,
        "training_case_ids": [str(row["case_id"]) for row in construction],
        "training_family_ids": sorted({str(row["family_id"]) for row in construction}),
        "audit_case_ids": [], "formal_case_ids": [],
        "raw_gradient": raw_gradient(model),
    }
    atomic_joblib(full_path, payload)
    audit_X = np.stack([hidden[str(row["case_id"])][hidden_name] for row in audit])
    audit_y = np.asarray([float(row[target]) for row in audit])
    audit_prediction = model.predict(audit_X)
    metrics = {
        "seed": seed, "probe": name, "target": target,
        "selected_alpha": alpha,
        "construction_records": len(construction),
        "construction_families": len({row["family_id"] for row in construction}),
        "audit_records": len(audit),
        "audit_families": len({row["family_id"] for row in audit}),
        **{f"construction_oof_{key}": value for key, value in oof_metrics.items()},
        **{f"audit_{key}": value for key, value in regression_metrics(audit_y, audit_prediction).items()},
        **{f"audit_{key}": value for key, value in _bootstrap_metrics(audit_y, audit_prediction, [row["family_id"] for row in audit], seed, repeats).items()},
        "model_sha256": sha256_file(full_path),
        "fold_models_sha256": canonical_hash(fold_hashes),
        "audit_used_for_fit": False,
        "formal_used_for_fit": False,
    }
    predictions = []
    for split, selected, truth, predicted in (
        ("construction_oof", construction, y, oof),
        ("audit", audit, audit_y, audit_prediction),
    ):
        predictions.extend({
            "seed": seed, "probe": name, "split": split,
            "case_id": str(row["case_id"]), "family_id": str(row["family_id"]),
            "true": float(actual), "predicted": float(value),
        } for row, actual, value in zip(selected, truth, predicted, strict=True))
    return model, metrics, predictions


def make_cells(rows: Sequence[dict[str, Any]], hidden: dict[str, dict[str, np.ndarray]]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family_id"]), str(row["fixed_answer_color"])].append(row)
    cells, arrays = [], {}
    for index, ((family, color), members) in enumerate(sorted(grouped.items())):
        folds = {int(row["outer_fold"]) for row in members}
        if len(folds) != 1:
            raise RuntimeError(f"Cell family crosses folds: {family}")
        key = f"cell_{index:04d}"
        value = np.stack([hidden[str(row["case_id"])]["lat"] for row in members]).mean(axis=0, dtype=np.float32)
        arrays[key] = value
        cells.append({
            "array_key": key, "family_id": family, "fixed_answer_color": color,
            "case_ids": sorted(str(row["case_id"]) for row in members),
            "record_count": len(members), "outer_fold": folds.pop(),
            "mean_G_L": float(np.mean([row["G_L"] for row in members])),
            "mean_clean_final_sa": float(np.mean([row["clean_final_sa"] for row in members])),
            "hidden_sha256": array_hash(value),
        })
    return cells, arrays


def build_seed(
    root: Path,
    seed: int,
    construction: Sequence[dict[str, Any]],
    audit: Sequence[dict[str, Any]],
    hidden: dict[str, dict[str, np.ndarray]],
    *,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, Any]:
    models = {}
    probe_metrics, predictions = [], []
    for name, hidden_name, target in (
        ("confidence_gap__P1_LAT__L14", "lat", "G_L"),
        ("confidence_gap__P1_PANL__L18", "panl", "G_L"),
        ("final_sa__P1_PANL__L18", "panl", "clean_final_sa"),
    ):
        model, metrics, rows = fit_probe(root, seed, name, construction, audit, hidden, hidden_name, target, bootstrap_repeats)
        models[name] = model
        probe_metrics.append(metrics)
        predictions.extend(rows)
    atomic_jsonl(root / f"artifacts/probes/seed_{seed}/predictions.jsonl", predictions)
    atomic_json(root / f"artifacts/probes/seed_{seed}/metrics.json", probe_metrics)

    cells, cell_hidden = make_cells(construction, hidden)
    confidence_patterns, confidence_audit = answer_patterns(cells, cell_hidden, "mean_G_L", return_audit=True)
    sa_patterns, sa_audit = answer_patterns(cells, cell_hidden, "mean_clean_final_sa", return_audit=True)
    direction_root = root / f"artifacts/directions/seed_{seed}"
    atomic_jsonl(direction_root / "construction_cells.jsonl", cells)
    arrays: dict[str, np.ndarray] = {}
    metadata, subspaces = [], []
    gradient = raw_gradient(models["confidence_gap__P1_LAT__L14"])
    for recipient in CANONICAL_COLORS:
        confidence, donors = loao(confidence_patterns, recipient)
        sa_pattern, _ = loao(sa_patterns, recipient)
        sa_model, sa_ridge, training_cells, conversion_error, ridge_alpha, ridge_trace = weighted_sa_probe(cells, cell_hidden, recipient)
        sa_path = root / f"artifacts/probes/seed_{seed}/construction_lat_sa__{recipient}__L14.joblib"
        atomic_joblib(sa_path, {
            "model": sa_model, "raw_gradient": sa_ridge,
            "training_cells": training_cells, "recipient_excluded": recipient,
            "conversion_error": conversion_error, "selected_alpha": ridge_alpha,
            "alpha_trace": ridge_trace, "seed": seed,
        })
        basis, basis_meta = svd_basis([sa_pattern, sa_ridge])
        norm = target_norm(cells, cell_hidden, donors)
        decomposition = natural_sa_decomposition(confidence, basis, norm)
        if decomposition["reconstruction_relative_error"] > RECONSTRUCTION_RTOL or not decomposition["raw_matches_existing"]:
            raise RuntimeError(f"Direction reconstruction failed: seed={seed} recipient={recipient}")
        scaled = {
            "confidence_raw": decomposition["raw"],
            "confidence_parallel_sa": decomposition["parallel_scaled"],
            "confidence_perp_sa_natural_scale": decomposition["perpendicular_scaled"],
        }
        components = {
            "confidence_raw": confidence,
            "confidence_parallel_sa": decomposition["parallel"],
            "confidence_perp_sa_natural_scale": decomposition["perpendicular"],
        }
        sign_audit = direction_sign_audit(gradient, scaled)
        for direction in DIRECTIONS:
            vector = np.asarray(scaled[direction], np.float32)
            key = f"{recipient}__{direction}__scaled"
            arrays[key] = vector
            natural = np.stack([cell_hidden[cell["array_key"]] for cell in cells if cell["fixed_answer_color"] in donors])
            unit = vector.astype(np.float64) / np.linalg.norm(vector)
            projection_sd = float(np.std(natural @ unit))
            source_norm = float(np.linalg.norm(confidence))
            component_norm = float(np.linalg.norm(components[direction]))
            metadata.append({
                "seed": seed, "recipient_answer": recipient, "layer": LAYER,
                "direction": direction, "scaled_key": key,
                "scaled_hash": array_hash(vector), "actual_norm": float(np.linalg.norm(vector)),
                "relative_norm": component_norm / source_norm,
                "retained_norm_ratio": component_norm / source_norm,
                "target_norm": norm, "common_scale": float(decomposition["common_scale"]),
                "natural_projection_std": projection_sd,
                "reconstruction_relative_error": float(decomposition["reconstruction_relative_error"]),
                **sign_audit,
                "included_answers": donors,
            })
        arrays[f"{recipient}__basis_sa"] = basis
        subspaces.append({
            "seed": seed, "recipient_answer": recipient, "layer": LAYER,
            "basis_key": f"{recipient}__basis_sa", "basis_sha256": array_hash(basis),
            "rank": basis_meta["rank"], "singular_values": basis_meta["singular_values"],
            "ridge_alpha": ridge_alpha, "ridge_conversion_error": conversion_error,
            "sa_probe_sha256": sha256_file(sa_path),
        })
    vector_path = direction_root / "P1_LAT__L14.npz"
    atomic_npz(vector_path, arrays)
    for row in metadata:
        row["vector_fingerprint"] = canonical_hash({key: row[key] for key in ("seed", "recipient_answer", "direction", "scaled_hash", "target_norm", "sign_policy")})
    atomic_json(direction_root / "vector_metadata.json", {"seed": seed, "vectors": metadata})
    atomic_json(direction_root / "subspaces.json", subspaces)
    atomic_json(direction_root / "pattern_audit.json", {"confidence": confidence_audit, "final_sa": sa_audit})

    reproduction = []
    if seed == 42:
        with np.load(FROZEN_VECTORS) as frozen:
            for recipient in CANONICAL_COLORS:
                for direction in DIRECTIONS:
                    key = f"{recipient}__{direction}__scaled"
                    current, historical = arrays[key], np.asarray(frozen[key])
                    reproduction.append({
                        "recipient_answer": recipient, "direction": direction,
                        "current_sha256": array_hash(current), "historical_sha256": array_hash(historical),
                        "bitwise_equal": bool(np.array_equal(current, historical)),
                        "signed_cosine": signed_cosine(current, historical),
                    })
        atomic_csv(direction_root / "seed42_engineering_reproduction.csv", reproduction)
    return {
        "seed": seed, "probe_metrics": probe_metrics, "vectors": metadata,
        "subspaces": subspaces, "vector_file_sha256": sha256_file(vector_path),
        "direction_sign_inconsistent_count": sum(row["direction_sign_inconsistent"] for row in metadata if row["direction"] == "confidence_raw"),
        "seed42_reproduction": reproduction,
    }


def signed_cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return math.nan if denominator == 0 else float(a @ b / denominator)
