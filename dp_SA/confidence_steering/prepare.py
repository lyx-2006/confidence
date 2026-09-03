from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .audit import cosine, projection_audit_rows
from .config import (
    ALPHAS, BOOTSTRAP_REPEATS, CANONICAL_COLORS, DIRECTIONS, FORMAL_ROOT,
    HIDDEN_SIZE, NULL_MAX_REPEATS, PANL_LAYER, PANL_POSITION,
    PANL_SA_PEARSON_MIN, PANL_SA_R2_MIN, PRIMARY_DIRECTION,
    PROTOCOL_VERSION, REMOVED_COSINE_LIMIT, RETAINED_NORM_MIN, SEED,
    STEERING_LAYERS, TARGET_DEFINITION, TRAIN_MANIFEST,
)
from .core import (
    HiddenResolver, answer_patterns, fit_oof_probe, loao, make_cells,
    natural_sa_decomposition, prelock_inventory, prepare_train_rows, project_out, raw_gradient,
    regression_metrics, scale_vector, shuffled_targets, split_train, svd_basis,
    target_norm, weighted_sa_probe,
)
from .io_utils import (
    array_hash, atomic_csv, atomic_joblib, atomic_json, atomic_jsonl, atomic_npz,
    canonical_hash, ensure_layout, load_jsonl, semantic_fingerprint, sha256_file,
)
from .run_spec import add_run_spec_arguments, normalize_run_spec, run_spec_from_args


def code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    repo = package.parents[1]
    paths = sorted(package.glob("*.py")) + [
        repo / "dp_SA/positions.py", repo / "dp_SA/soft_score.py",
        repo / "layer_metacognition/model_adapter.py",
        repo / "layer_metacognition/conversation_builder.py",
    ]
    return {str(path.relative_to(repo)): sha256_file(path) for path in paths if path.name != "__init__.py"}


def bootstrap_metrics(y: np.ndarray, pred: np.ndarray, families: Sequence[str]) -> dict[str, Any]:
    family_array = np.asarray(list(map(str, families)))
    ordered = sorted(set(family_array)); indices = {f: np.flatnonzero(family_array == f) for f in ordered}
    rng = np.random.default_rng(SEED); samples = {k: [] for k in ("r2", "pearson", "spearman", "mae")}
    for _ in range(BOOTSTRAP_REPEATS):
        take = np.concatenate([indices[ordered[i]] for i in rng.integers(0, len(ordered), len(ordered))])
        for name, value in regression_metrics(y[take], pred[take]).items():
            if math.isfinite(value): samples[name].append(value)
    output = {}
    for name, values in samples.items():
        low, high = np.percentile(values, [2.5, 97.5])
        output.update({f"{name}_ci_low": float(low), f"{name}_ci_high": float(high), f"{name}_bootstrap_valid": len(values)})
    return output


def save_probe(root: Path, name: str, X: np.ndarray, y: np.ndarray,
               construction: Sequence[dict[str, Any]], audit_X: np.ndarray,
               audit_y: np.ndarray, audit: Sequence[dict[str, Any]],
               target: str, position: str, layer: int):
    folds = [int(row["outer_fold"]) for row in construction]
    oof, fold_models, oof_metrics, model, alpha, trace = fit_oof_probe(X, y, construction, folds)
    fold_hashes = {}
    for fold, fold_model in fold_models.items():
        path = root / f"artifacts/probes/{name}__fold_{fold}.joblib"
        atomic_joblib(path, fold_model); fold_hashes[str(fold)] = sha256_file(path)
    prediction = model.predict(audit_X); audit_metrics = regression_metrics(audit_y, prediction)
    path = root / f"artifacts/probes/{name}__full.joblib"
    payload = {"model": model, "target": target, "position": position, "layer": layer,
               "selected_alpha": alpha, "alpha_trace": trace, "target_std": float(np.std(y)),
               "training_case_ids": [r["case_id"] for r in construction], "audit_case_ids": []}
    atomic_joblib(path, payload)
    metrics = {"name": name, "target": target, "position": position, "layer": layer,
               "selected_alpha": alpha, "construction_count": len(construction),
               "construction_family_count": len({r["family_id"] for r in construction}),
               "audit_count": len(audit), "audit_family_count": len({r["family_id"] for r in audit}),
               **{f"construction_oof_{k}": v for k, v in oof_metrics.items()}, **audit_metrics,
               **bootstrap_metrics(audit_y, prediction, [r["family_id"] for r in audit]),
               "model_sha256": sha256_file(path), "fold_models_sha256": canonical_hash(fold_hashes),
               "audit_used_for_fit": False}
    predictions = [{"case_id": row["case_id"], "family_id": row["family_id"], "probe": name,
                    "true": float(truth), "predicted": float(value), "split": "direction_audit"}
                   for row, truth, value in zip(audit, audit_y, prediction, strict=True)]
    return model, metrics, predictions


def smoke_manifest(audit_manifest: Sequence[dict[str, Any]]):
    import itertools
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in audit_manifest: grouped.setdefault(str(row["family_id"]), []).append(row)
    eligible = []
    for combo in itertools.combinations(sorted(grouped), 4):
        rows = [r for family in combo for r in grouped[family]]
        origins = {"follow_text" if r["answer_matches_text"] and not r["answer_matches_image"] else "follow_image" if r["answer_matches_image"] and not r["answer_matches_text"] else "other" for r in rows}
        if len({r["phase0_normalized_answer"] for r in rows}) >= 4 and {"follow_text", "follow_image"} <= origins and {r["condition"] for r in rows} == {"conflict_easy", "conflict_hard"}:
            eligible.append((len(rows), combo, rows))
    if not eligible: raise ValueError("No complete four-family audit smoke selection")
    _, combo, rows = min(eligible, key=lambda value: (value[0], value[1]))
    return sorted(rows, key=lambda r: r["case_id"]), {"source": "direction_audit_only", "families": list(combo), "records": len(rows)}


def build_directions(root: Path, cells: list[dict[str, Any]], cell_arrays: dict[int, dict[str, np.ndarray]], run_spec: dict[str, Any]):
    all_vectors = {}; metadata = []; subspaces = []; pattern_audits = []
    selected_directions = tuple(run_spec["directions"]); shuffle_requested = bool(run_spec["shuffle_requested"])
    for layer in run_spec["layers"]:
        hidden = cell_arrays[layer]; patterns = {}
        for field, label in (("mean_G_L", "confidence"), ("mean_D_t", "difficulty_text"),
                             ("mean_D_i", "difficulty_image"), ("mean_clean_final_sa", "final_sa")):
            patterns[label], audits = answer_patterns(cells, hidden, field, return_audit=True)
            pattern_audits.extend({**row, "layer": layer, "pattern": label} for row in audits)
        shuffled_sets = []
        if shuffle_requested:
            for replicate in range(NULL_MAX_REPEATS + 1):
                values = shuffled_targets(cells, replicate)
                shuffled = [{**cell, "mean_G_L_shuffled": values[cell["array_key"]]} for cell in cells]
                shuffled_sets.append(answer_patterns(shuffled, hidden, "mean_G_L_shuffled"))
        arrays = {}; layer_vectors = {}
        for recipient in CANONICAL_COLORS:
            confidence, donors = loao(patterns["confidence"], recipient)
            d_text, _ = loao(patterns["difficulty_text"], recipient)
            d_image, _ = loao(patterns["difficulty_image"], recipient)
            sa_pattern, _ = loao(patterns["final_sa"], recipient)
            sa_model, sa_raw, training_cells, conversion_error, ridge_alpha, ridge_trace = weighted_sa_probe(cells, hidden, recipient)
            sa_path = root / f"artifacts/probes/construction_lat_sa__{recipient}__L{layer}.joblib"
            atomic_joblib(sa_path, {"model": sa_model, "raw_gradient": sa_raw, "training_cells": training_cells,
                                    "recipient_excluded": recipient, "conversion_error": conversion_error,
                                    "selected_alpha": ridge_alpha, "alpha_trace": ridge_trace})
            q_difficulty, meta_difficulty = svd_basis([d_text, d_image])
            q_sa, meta_sa = svd_basis([sa_pattern, sa_raw])
            q_both, meta_both = svd_basis([d_text, d_image, sa_pattern, sa_raw])
            main_shuffle = loao(shuffled_sets[0], recipient)[0] if shuffle_requested else None
            norm = target_norm(cells, hidden, donors)
            decomposition = natural_sa_decomposition(confidence, q_sa, norm)
            parallel_sa = decomposition["parallel"]; perpendicular_sa = decomposition["perpendicular"]
            common_scale = decomposition["common_scale"]; raw_scaled = decomposition["raw"]
            layer_vectors[f"{recipient}|__raw_reference"] = raw_scaled
            parallel_scaled = decomposition["parallel_scaled"]; perpendicular_natural_scaled = decomposition["perpendicular_scaled"]
            reconstruction_error = decomposition["reconstruction_relative_error"]
            if reconstruction_error > 1e-5 or not decomposition["raw_matches_existing"]:
                raise ValueError(f"Natural SA decomposition reconstruction failed: {recipient}/L{layer}")
            variants = {
                "confidence_raw": confidence,
                "confidence_perp_difficulty": project_out(confidence, q_difficulty),
                "confidence_perp_sa": project_out(confidence, q_sa),
                "confidence_perp_difficulty_sa": project_out(confidence, q_both),
                "confidence_parallel_sa": parallel_sa,
                "confidence_perp_sa_natural_scale": perpendicular_sa,
            }
            if shuffle_requested:
                variants["within_answer_shuffled_perp_difficulty_sa"] = project_out(main_shuffle, q_both)
            removed_columns = {
                "confidence_perp_difficulty": [d_text, d_image],
                "confidence_perp_sa": [sa_pattern, sa_raw],
                "confidence_perp_difficulty_sa": [d_text, d_image, sa_pattern, sa_raw],
                "confidence_perp_sa_natural_scale": [sa_pattern, sa_raw],
                "within_answer_shuffled_perp_difficulty_sa": [d_text, d_image, sa_pattern, sa_raw],
            }
            natural = np.stack([hidden[cell["array_key"]] for cell in cells if cell["fixed_answer_color"] in donors])
            natural_scaled = {"confidence_parallel_sa": parallel_scaled, "confidence_perp_sa_natural_scale": perpendicular_natural_scaled}
            for direction in selected_directions:
                projected = variants[direction]
                source = main_shuffle if direction.startswith("within_answer") else confidence
                scaled = natural_scaled[direction] if direction in natural_scaled else scale_vector(projected, norm); key = f"{recipient}__{direction}__scaled"
                arrays[key] = scaled; layer_vectors[f"{recipient}|{direction}"] = scaled
                retained = float(np.linalg.norm(projected) / np.linalg.norm(source)); unit = scaled.astype(np.float64) / np.linalg.norm(scaled)
                removed = [abs(cosine(projected, column)) for column in removed_columns.get(direction, [])]
                projection_sd = float(np.std(natural @ unit))
                scaled_norm = float(np.linalg.norm(scaled)); valid = bool(scaled.shape == (HIDDEN_SIZE,) and scaled.dtype == np.float32 and np.isfinite(scaled).all() and scaled_norm > 0 and all(value <= REMOVED_COSINE_LIMIT for value in removed) and reconstruction_error <= 1e-5)
                row = {"recipient_answer": recipient, "layer": layer, "direction": direction, "scaled_key": key,
                       "pre_projection_norm": float(np.linalg.norm(source)), "post_projection_norm": float(np.linalg.norm(projected)),
                       "retained_norm_ratio": retained, "relative_norm": retained, "severe_overlap": retained < RETAINED_NORM_MIN,
                       "target_norm": norm, "common_scale": common_scale, "scaled_norm": scaled_norm,
                       "cosine_with_original_confidence": cosine(scaled, confidence),
                       "cosine_difficulty_text": cosine(scaled, d_text), "cosine_difficulty_image": cosine(scaled, d_image),
                       "cosine_sa_pattern": cosine(scaled, sa_pattern), "cosine_sa_ridge": cosine(scaled, sa_raw),
                       "max_removed_absolute_cosine": max(removed, default=0.0),
                       "natural_projection_std": projection_sd,
                       "injection_to_natural_projection_sd": scaled_norm / projection_sd if projection_sd else None,
                       "alpha1_natural_sd": scaled_norm / projection_sd if projection_sd else None,
                       "reconstruction_relative_error": reconstruction_error,
                       "orthogonality_error": max(removed, default=0.0),
                       "scaled_hash": array_hash(scaled), "implementation_invariant_passed": valid,
                       "included_answers": donors}
                row["vector_fingerprint"] = canonical_hash({key: row[key] for key in ("recipient_answer", "layer", "direction", "scaled_hash", "target_norm")})
                metadata.append(row)
            if shuffle_requested:
                for replicate in range(1, NULL_MAX_REPEATS + 1):
                    shuffled, _ = loao(shuffled_sets[replicate], recipient)
                    arrays[f"null_{replicate:03d}__{recipient}__scaled"] = scale_vector(project_out(shuffled, q_both), norm)
            basis_hashes = {"difficulty": array_hash(q_difficulty), "sa": array_hash(q_sa), "difficulty_sa": array_hash(q_both)}
            arrays[f"{recipient}__basis_difficulty"] = q_difficulty
            arrays[f"{recipient}__basis_sa"] = q_sa
            arrays[f"{recipient}__basis_difficulty_sa"] = q_both
            subspaces.append({"recipient_answer": recipient, "layer": layer, "difficulty": meta_difficulty,
                              "sa": meta_sa, "difficulty_sa": meta_both, "basis_hashes": basis_hashes,
                              "sa_model_sha256": sha256_file(sa_path), "ridge_conversion_error": conversion_error})
        path = root / f"artifacts/directions/P1_LAT__L{layer}.npz"; atomic_npz(path, arrays)
        all_vectors[layer] = layer_vectors
    return all_vectors, metadata, subspaces, pattern_audits


def direction_sensitivity(root: Path, vectors, metadata, confidence_models, target_std: float, run_spec: dict[str, Any]):
    meta = {(row["layer"], row["recipient_answer"], row["direction"]): row for row in metadata}
    rows = []; reliability = {}
    for layer in run_spec["layers"]:
        gradient = raw_gradient(confidence_models[layer]); layer_ok = True
        probe_path = root / f"artifacts/probes/confidence_gap__P1_LAT__L{layer}__full.joblib"; probe_hash = sha256_file(probe_path)
        with np.load(root / f"artifacts/directions/P1_LAT__L{layer}.npz") as payload:
            for recipient in CANONICAL_COLORS:
                raw_dot = float(gradient @ vectors[layer][f"{recipient}|__raw_reference"])
                main_dot = float(gradient @ vectors[layer][f"{recipient}|{PRIMARY_DIRECTION}"]) if PRIMARY_DIRECTION in run_spec["directions"] else None
                null = ([float(gradient @ np.asarray(payload[f"null_{rep:03d}__{recipient}__scaled"])) / target_std
                         for rep in range(1, NULL_MAX_REPEATS + 1)] if layer == 14 and run_spec["shuffle_requested"] else [])
                threshold = float(np.percentile(null, 95)) if null else None
                cell_ok = raw_dot > 0 if "confidence_raw" in run_spec["directions"] else True
                if run_spec["analysis_kind"] == "legacy_confirmatory":
                    cell_ok = cell_ok and main_dot is not None and main_dot > 0 and (threshold is None or main_dot / target_std > threshold)
                layer_ok &= cell_ok
                for direction in run_spec["directions"]:
                    dot = float(gradient @ vectors[layer][f"{recipient}|{direction}"])
                    percentile = 100.0 * sum(value < dot / target_std for value in null) / len(null) if null else None
                    if direction == "confidence_raw": direction_status = "passed" if raw_dot > 0 else "failed"
                    elif direction == PRIMARY_DIRECTION and run_spec["analysis_kind"] == "legacy_confirmatory": direction_status = "passed" if main_dot > 0 and (threshold is None or main_dot / target_std > threshold) else "failed"
                    else: direction_status = "reported"
                    for alpha in run_spec["alphas"]:
                        rows.append({"recipient_answer": recipient, "layer": layer, "direction": direction, "alpha": alpha,
                                     "raw_probe_dot": dot, "standardized_sensitivity": dot / target_std,
                                     "standardized_delta_at_alpha": alpha * dot / target_std,
                                     "retained_confidence_sensitivity_ratio": dot / raw_dot if raw_dot else math.nan,
                                     "shuffle_percentile": percentile, "shuffle95_threshold": threshold,
                                     "status": direction_status, "probe_hash": probe_hash,
                                     "vector_hash": meta[layer, recipient, direction]["scaled_hash"]})
        reliability[layer] = layer_ok
    # L14 is the confirmatory readout. Other selected layers retain their own
    # reliability labels but do not prevent an L14 run from proceeding.
    return rows, reliability, reliability.get(14, True)


def prepare_gate_status(
    probe_reliable: dict[int, bool],
    direction_reliable: dict[int, bool],
    *,
    numerical_gate: bool,
    panl_sa_gate: bool,
) -> dict[str, bool]:
    l14_probe_gate = probe_reliable.get(14, True)
    l14_direction_gate = direction_reliable.get(14, True)
    return {
        "numerical_gate": bool(numerical_gate),
        "l14_confidence_probe_gate": bool(l14_probe_gate),
        "all_selected_confidence_probes_reliable": all(probe_reliable.values()),
        "panl_final_sa_probe_gate": bool(panl_sa_gate),
        "direction_sensitivity_gate": bool(l14_direction_gate),
        "formal_eligible": bool(numerical_gate and l14_probe_gate and panl_sa_gate and l14_direction_gate),
    }


def run_prepare(*, output_root: Path = FORMAL_ROOT, smoke: bool = False, resume: bool = False, run_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    run_spec = normalize_run_spec() if run_spec is None else run_spec
    root = ensure_layout(output_root); inventory = prelock_inventory()
    config = {"protocol_version": PROTOCOL_VERSION, "target_definition": TARGET_DEFINITION,
              "test_state": "sealed", "smoke_only": smoke, "seed": SEED,
              "run_spec": run_spec,
              "inputs": {name: value["sha256"] for name, value in inventory.items()}, "source_code": code_hashes()}
    fingerprint = semantic_fingerprint(root / "progress/prepare_config.json", config, resume=resume)
    progress = root / "progress/prepare.json"
    if resume and progress.is_file():
        previous = json.loads(progress.read_text())
        if previous.get("status") == "complete" and previous.get("config_fingerprint") == fingerprint:
            return {**previous, "resumed_noop": True}
    manifest = load_jsonl(TRAIN_MANIFEST)
    construction_manifest, audit_manifest, split_audit = split_train(manifest)
    prepared = prepare_train_rows(manifest); by_case = {row["case_id"]: row for row in prepared}
    construction = [by_case[row["case_id"]] for row in construction_manifest]
    audit = [by_case[row["case_id"]] for row in audit_manifest]
    atomic_json(root / "artifacts/diagnostics/input_and_split_audit.json", {"inventory": inventory, "split": split_audit, "test_manifest_opened": False})
    atomic_jsonl(root / "artifacts/manifests/construction_records.jsonl", construction)
    atomic_jsonl(root / "artifacts/manifests/direction_audit_records.jsonl", audit)
    resolver = HiddenResolver(); cells, cell_arrays = make_cells(construction, resolver)
    atomic_jsonl(root / "artifacts/manifests/construction_cells.jsonl", cells)
    vectors, metadata, subspaces, pattern_audits = build_directions(root, cells, cell_arrays, run_spec)
    vector_files = {str(layer): sha256_file(root / f"artifacts/directions/P1_LAT__L{layer}.npz") for layer in run_spec["layers"]}
    vector_fingerprint = canonical_hash(vector_files)
    atomic_json(root / "artifacts/directions/vector_metadata.json", {"vectors": metadata, "files": vector_files, "fingerprint": vector_fingerprint, "l14_null_repeats": NULL_MAX_REPEATS if run_spec["shuffle_requested"] else 0, "run_spec": run_spec})
    atomic_csv(root / "tables/vector_audit.csv", metadata)
    atomic_csv(root / "tables/direction_orthogonality.csv", metadata)
    atomic_csv(root / "tables/retained_norm.csv", metadata)
    atomic_csv(root / "tables/answer_balance_audit.csv", pattern_audits)
    atomic_json(root / "artifacts/diagnostics/subspace_rank_and_singular_values.json", {"rows": subspaces})

    probe_metrics = []; probe_predictions = []; confidence_models = {}
    audit_lat = {layer: np.stack([resolver.load(row["case_id"], f"P1_LAT__L{layer}") for row in audit]) for layer in run_spec["layers"]}
    for layer in run_spec["layers"]:
        train_x = np.stack([resolver.load(row["case_id"], f"P1_LAT__L{layer}") for row in construction])
        model, metrics, predictions = save_probe(root, f"confidence_gap__P1_LAT__L{layer}", train_x,
            np.asarray([row["G_L"] for row in construction]), construction, audit_lat[layer],
            np.asarray([row["G_L"] for row in audit]), audit, "G_L", "P1_LAT", layer)
        confidence_models[layer] = model; probe_metrics.append(metrics); probe_predictions.extend(predictions)
    train_panl = np.stack([resolver.load(row["case_id"], f"{PANL_POSITION}__L{PANL_LAYER}") for row in construction])
    audit_panl = np.stack([resolver.load(row["case_id"], f"{PANL_POSITION}__L{PANL_LAYER}") for row in audit])
    for name, target in (("confidence_gap__P1_PANL__L18", "G_L"), ("final_sa__P1_PANL__L18", "clean_final_sa")):
        _, metrics, predictions = save_probe(root, name, train_panl, np.asarray([row[target] for row in construction]),
            construction, audit_panl, np.asarray([row[target] for row in audit]), audit, target, PANL_POSITION, PANL_LAYER)
        probe_metrics.append(metrics); probe_predictions.extend(predictions)
    atomic_csv(root / "tables/probe_metrics.csv", probe_metrics)
    atomic_jsonl(root / "artifacts/probes/audit_predictions.jsonl", probe_predictions)

    sensitivity, direction_reliable, sensitivity_gate = direction_sensitivity(
        root, vectors, metadata, confidence_models, float(np.std([row["G_L"] for row in construction])), run_spec)
    atomic_csv(root / "tables/direction_confidence_sensitivity.csv", sensitivity)
    projection_rows = []
    for layer in run_spec["layers"]:
        matrices = {direction: {color: vectors[layer][f"{color}|{direction}"] / np.linalg.norm(vectors[layer][f"{color}|{direction}"])
                                for color in CANONICAL_COLORS} for direction in run_spec["directions"]}
        rows, _ = projection_audit_rows(audit, audit_lat[layer], matrices, layer=layer); projection_rows.extend(rows)
    atomic_csv(root / "tables/heldout_projection_audit.csv", projection_rows)

    metric_index = {row["name"]: row for row in probe_metrics}
    probe_reliable = {layer: metric_index[f"confidence_gap__P1_LAT__L{layer}"]["pearson"] > 0 and metric_index[f"confidence_gap__P1_LAT__L{layer}"]["r2"] > 0 for layer in run_spec["layers"]}
    panl_sa = metric_index["final_sa__P1_PANL__L18"]
    numerical_gate = all(row["implementation_invariant_passed"] for row in metadata)
    panl_sa_gate = panl_sa["pearson"] >= PANL_SA_PEARSON_MIN and panl_sa["r2"] >= PANL_SA_R2_MIN
    gates = prepare_gate_status(probe_reliable, direction_reliable, numerical_gate=numerical_gate, panl_sa_gate=panl_sa_gate)
    formal_eligible = gates["formal_eligible"]
    verdict = {"status": "passed" if formal_eligible else "failed", "formal_eligible": formal_eligible,
               "test_state": "sealed", "test_manifest_opened": False, **gates,
               "selected_probe_gate": gates["l14_confidence_probe_gate"],
               "l14_direction_sensitivity_gate": sensitivity_gate if 14 in run_spec["layers"] else None, "probe_layer_reliable": probe_reliable,
               "direction_layer_reliable": direction_reliable, "analysis_kind": run_spec["analysis_kind"], "run_spec_fingerprint": run_spec["fingerprint"]}
    atomic_json(root / "artifacts/diagnostics/prelock_verdict.json", verdict)
    smoke_rows, smoke_selection = smoke_manifest(audit_manifest)
    atomic_jsonl(root / "artifacts/manifests/smoke_manifest.jsonl", smoke_rows)
    atomic_json(root / "artifacts/diagnostics/smoke_selection.json", smoke_selection)
    probe_files = {str(path.relative_to(root)): sha256_file(path) for path in sorted((root / "artifacts/probes").glob("*.joblib"))}
    material = {"protocol_version": PROTOCOL_VERSION, "target_definition": TARGET_DEFINITION,
                "prepare_fingerprint": fingerprint, "vector_fingerprint": vector_fingerprint,
                "vector_files": vector_files, "probe_files": probe_files, "verdict": verdict,
                "code_hashes": code_hashes(), "layers": run_spec["layers"], "alphas": run_spec["alphas"],
                "directions": run_spec["directions"], "run_spec": run_spec, "seed": SEED, "output_schema": list(("figures", "tables", "progress", "artifacts/manifests", "artifacts/directions", "artifacts/probes", "artifacts/hidden", "artifacts/diagnostics", "artifacts/trials")),
                "null_rule": "legacy confirmatory 20->99 expansion" if run_spec["analysis_kind"] == "legacy_confirmatory" else "disabled for diagnostic run spec"}
    material["fingerprint"] = canonical_hash(material)
    atomic_json(root / "artifacts/diagnostics/prelock_material.json", material)
    result = {"status": "complete", "construction_records": len(construction), "audit_records": len(audit),
              "construction_cells": len(cells), "formal_eligible": formal_eligible,
              "config_fingerprint": fingerprint, "vector_fingerprint": vector_fingerprint,
              "test_manifest_opened": False, "resumed_noop": False}
    atomic_json(progress, result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true"); add_run_spec_arguments(parser)
    args = parser.parse_args(argv); print(json.dumps(run_prepare(output_root=Path(args.output_root) if args.output_root else FORMAL_ROOT, smoke=args.smoke, resume=args.resume, run_spec=run_spec_from_args(args)), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
