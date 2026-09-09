from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib

from .config import (
    AUDIT_MANIFEST, CANDIDATE_MANIFEST, CELL_MANIFEST, CONSTRUCTION_DISTRIBUTION, CONSTRUCTION_MANIFEST,
    EXPECTED_COUNTS, FOLD_MANIFEST, MATCHED_ROOT, PROBE_INDEX, PROBE_LAYERS,
    PROBE_ROOT, POSITIONS, T0_STEERING_TABLE, T0_STEERING_TRIALS,
    T0_CLEAN_CAPTURE, T0_VECTOR_METADATA, TEST_MANIFEST, TRAJECTORY_CONFIG,
)
from .io_utils import canonical_hash, inventory, load_jsonl, sha256_file, verify_inventory


def frozen_files() -> list[Path]:
    files = [AUDIT_MANIFEST, CONSTRUCTION_MANIFEST, PROBE_INDEX, TRAJECTORY_CONFIG,
             CANDIDATE_MANIFEST, CELL_MANIFEST, CONSTRUCTION_DISTRIBUTION, FOLD_MANIFEST, TEST_MANIFEST,
             T0_CLEAN_CAPTURE, T0_VECTOR_METADATA, T0_STEERING_TRIALS, T0_STEERING_TABLE]
    index = load_jsonl(PROBE_INDEX)
    files.extend(PROBE_ROOT / row["probe_file"].removeprefix("artifacts/probes/") for row in index if row.get("target") == "final_soft_sa")
    metadata = json.loads(T0_VECTOR_METADATA.read_text())
    files.extend(MATCHED_ROOT / relative for relative in sorted({row["vector_file"] for row in metadata["vectors"] if row["position"] == "P1_LAT" and row["direction"] == "matched_loao"}))
    return files


def source_inventory() -> dict[str, str]:
    paths = frozen_files()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing: raise FileNotFoundError(f"Frozen files missing: {missing[:5]}")
    return inventory(paths)


def audit_frozen_sources() -> dict[str, Any]:
    audit = load_jsonl(AUDIT_MANIFEST); construction = load_jsonl(CONSTRUCTION_MANIFEST)
    candidates = load_jsonl(CANDIDATE_MANIFEST); cells = load_jsonl(CELL_MANIFEST)
    folds = load_jsonl(FOLD_MANIFEST); test = load_jsonl(TEST_MANIFEST)
    observed = {
        "audit": len(audit), "audit_families": len({r["family_id"] for r in audit}),
        "probe_construction": len(construction), "candidates": len(candidates),
        "candidate_families": len({r["family_id"] for r in candidates}), "cells": len(cells),
        "folds": len({int(r["fold"]) for r in folds}), "test": len(test),
        "confirmatory_test": sum(r["test_status"] == "confirmatory" for r in test),
    }
    for key, expected in EXPECTED_COUNTS.items():
        if key == "probes": continue
        if observed[key] != expected: raise ValueError(f"Frozen count changed: {key}={observed[key]}, expected {expected}")
    if any(r.get("sa_side") not in {"high_text", "high_image"} for r in candidates): raise ValueError("Candidate sa_side is not frozen binary T0 verbal-SA side")
    if any(r.get("test_side") not in {"high_text", "high_image"} for r in test): raise ValueError("Test test_side is not frozen binary T0 verbal-SA side")
    audit_ids = {str(r["case_id"]) for r in audit}; construction_ids = [str(r["case_id"]) for r in construction]
    if audit_ids.intersection(construction_ids): raise ValueError("Probe construction/audit case leakage")
    trajectory_config = json.loads(TRAJECTORY_CONFIG.read_text()); expected_fingerprint = trajectory_config["fingerprint"]
    selected = [r for r in load_jsonl(PROBE_INDEX) if r.get("target") == "final_soft_sa"]
    expected_keys = {(position, layer) for position in POSITIONS for layer in PROBE_LAYERS}
    if len(selected) != EXPECTED_COUNTS["probes"] or {(r["position"], int(r["layer"])) for r in selected} != expected_keys: raise ValueError("Frozen final_soft_sa probe grid changed")
    probe_rows = []; probe_config_fingerprints=set()
    for row in selected:
        path = PROBE_ROOT / row["probe_file"].removeprefix("artifacts/probes/")
        digest = sha256_file(path)
        if digest != row["probe_sha256"]: raise ValueError(f"Probe hash mismatch: {path}")
        payload = joblib.load(path)
        if payload.get("target") != "final_soft_sa" or payload.get("position") != row["position"] or int(payload.get("layer", -1)) != int(row["layer"]): raise ValueError(f"Probe identity mismatch: {path}")
        if list(map(str, payload.get("construction_case_ids", []))) != construction_ids: raise ValueError(f"Probe training IDs changed: {path}")
        if not payload.get("config_fingerprint"): raise ValueError(f"Probe config fingerprint missing: {path}")
        probe_config_fingerprints.add(str(payload["config_fingerprint"]))
        model = payload.get("model")
        if not hasattr(model, "named_steps") or set(model.named_steps) != {"scale", "ridge"}: raise ValueError(f"Probe is not frozen StandardScaler+Ridge: {path}")
        probe_rows.append({"position": row["position"], "layer": int(row["layer"]), "probe_file": str(path.resolve()), "probe_sha256": digest, "alpha": float(payload["alpha"]), "construction_case_count": len(construction_ids), "audit_used_for_fit": False})
    if len(probe_config_fingerprints)!=1: raise ValueError(f"Frozen probes have mixed config fingerprints: {sorted(probe_config_fingerprints)}")
    vector_metadata = json.loads(T0_VECTOR_METADATA.read_text())
    selected_vectors = [r for r in vector_metadata["vectors"] if r["position"] == "P1_LAT" and r["direction"] == "matched_loao" and 9 <= int(r["layer"]) <= 15]
    if not selected_vectors: raise ValueError("No frozen T0 matched LOAO LAT vectors")
    return {
        "status": "passed", "counts": observed, "audit_used_for_fit": False,
        "audit_description": "frozen held-out diagnostic set", "probe_rows": probe_rows,
        "trajectory_fingerprint": expected_fingerprint, "probe_training_config_fingerprint": next(iter(probe_config_fingerprints)),
        "model_processor_hashes": trajectory_config.get("model_processor_identity", {}), "inventory": source_inventory(),
        "inventory_fingerprint": canonical_hash(source_inventory()), "vector_count": len(selected_vectors),
        "side_definition": "Frozen T0 verbal-SA grouping label; not Real SA or objective modality dependence.",
    }


def verify_frozen_sources(before: dict[str, str]) -> None:
    verify_inventory(before)


def load_frozen_probes() -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for row in load_jsonl(PROBE_INDEX):
        if row.get("target") != "final_soft_sa": continue
        path = PROBE_ROOT / row["probe_file"].removeprefix("artifacts/probes/")
        if sha256_file(path) != row["probe_sha256"]: raise ValueError(f"Probe hash mismatch: {path}")
        output[row["position"], int(row["layer"])] = joblib.load(path)
    return output
