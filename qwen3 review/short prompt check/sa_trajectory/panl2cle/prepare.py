from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from Steering.steering import build_vectors, _save_torch_atomic

from .config import (
    CAPTURE_ROOT, CLE_LAYERS, EXPECTED_CAPTURE_CASES, EXPECTED_PROBE_AUDIT,
    EXPECTED_PROBE_CONSTRUCTION, EXPECTED_PROBE_ELIGIBLE, EXPECTED_TEST_CASES,
    EXPECTED_TEST_SIDES, MODEL_PATH, PANL_LAYERS, SEED, STEERING_ROOT,
)
from .contracts import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file, validate_layer_design


def split_probe_manifests(capture: list[dict[str, Any]], test: list[dict[str, Any]]):
    test_items = {str(r["item_id"]) for r in test}
    test_images = {str(r["image_sha256"]) for r in test}
    eligible = [r for r in capture if str(r["item_id"]) not in test_items and str(r["image_sha256"]) not in test_images]
    cardinality = (len(eligible), len({str(r["item_id"]) for r in eligible}))
    if cardinality != EXPECTED_PROBE_ELIGIBLE:
        raise ValueError(f"Probe eligible population changed: {cardinality}")
    groups = np.asarray([str(r["item_id"]) for r in eligible])
    splitter = GroupKFold(5, shuffle=True, random_state=SEED)
    fold_by_index: dict[int, int] = {}
    for fold, (_train, held) in enumerate(splitter.split(np.zeros(len(eligible)), groups=groups)):
        for index in held:
            fold_by_index[int(index)] = fold
    audit = [{**row, "outer_fold": 0} for i, row in enumerate(eligible) if fold_by_index[i] == 0]
    construction = [{**row, "outer_fold": fold_by_index[i]} for i, row in enumerate(eligible) if fold_by_index[i] != 0]
    actual = (
        (len(construction), len({str(r["item_id"]) for r in construction})),
        (len(audit), len({str(r["item_id"]) for r in audit})),
    )
    if actual != (EXPECTED_PROBE_CONSTRUCTION, EXPECTED_PROBE_AUDIT):
        raise ValueError(f"Probe split changed: {actual}")
    sets = []
    for rows in (construction, audit, test):
        sets.append(({str(r["item_id"]) for r in rows}, {str(r["image_sha256"]) for r in rows}))
    overlap = [
        (a, b, kind) for a in range(3) for b in range(a + 1, 3) for kind in (0, 1)
        if sets[a][kind] & sets[b][kind]
    ]
    if overlap:
        raise ValueError(f"Probe construction/audit/test leakage: {overlap}")
    summary = {
        "seed": SEED, "eligible_cases": len(eligible), "eligible_items": len(set(groups)),
        "construction_cases": len(construction), "construction_items": len(sets[0][0]),
        "audit_cases": len(audit), "audit_items": len(sets[1][0]),
        "test_cases": len(test), "test_items": len(sets[2][0]),
        "item_overlap": 0, "image_overlap": 0,
    }
    return construction, audit, summary


def build_short_panl_vectors(capture_root: Path, steering_root: Path):
    construction = load_jsonl(steering_root / "construction_manifest.jsonl")
    counts = Counter(r["construction_side"] for r in construction)
    if counts != Counter({"high_image": 25, "high_text": 25}):
        raise ValueError(f"Short steering construction changed: {counts}")
    if len({str(r["item_id"]) for r in construction}) != 50:
        raise ValueError("Short steering construction is not item-disjoint")
    vectors, metadata, artifacts = build_vectors(
        capture_root, construction, positions=("PANL",), layers=PANL_LAYERS,
    )
    old = torch.load(steering_root / "short_vectors.pt", map_location="cpu", weights_only=False)
    parity = {}
    for layer in (14, 16):
        reference = old[f"PANL__L{layer}"]["scaled_vector"].float()
        current = artifacts[f"PANL__L{layer}"]["scaled_vector"].float()
        error = float((reference - current).abs().max().item())
        if error != 0.0:
            raise ValueError(f"Short PANL L{layer} vector parity failed: {error}")
        parity[str(layer)] = error
    metadata["existing_short_vector_max_abs_error"] = parity
    return vectors, metadata, artifacts, construction


def prepare(
    *, output_root: Path, capture_root: Path = CAPTURE_ROOT,
    steering_root: Path = STEERING_ROOT, model_path: Path = MODEL_PATH,
    smoke: bool = False, resume: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    capture_root = Path(capture_root).resolve()
    steering_root = Path(steering_root).resolve()
    model_path = Path(model_path).resolve()
    validate_layer_design()
    capture = [r for r in load_jsonl(capture_root / "results.jsonl") if r.get("status") == "completed"]
    test = load_jsonl(steering_root / "test_manifest.jsonl")
    if len(capture) != EXPECTED_CAPTURE_CASES or len(test) != EXPECTED_TEST_CASES:
        raise ValueError(f"Short capture/test cardinality changed: {len(capture)}/{len(test)}")
    if Counter(r["test_side"] for r in test) != Counter(EXPECTED_TEST_SIDES):
        raise ValueError("Short test-side counts changed")
    probe_construction, probe_audit, split_summary = split_probe_manifests(capture, test)
    vectors, vector_metadata, vector_artifacts, steering_construction = build_short_panl_vectors(capture_root, steering_root)
    source_files = [
        capture_root / "config.json", capture_root / "results.jsonl",
        steering_root / "construction_manifest.jsonl", steering_root / "test_manifest.jsonl",
        steering_root / "short_vectors.pt", model_path / "config.json",
    ]
    code_files = sorted(Path(__file__).parent.glob("*.py"))
    payload = {
        "format_version": 1, "experiment": "qwen3_short_panl_to_next_cle_four_cell",
        "model": str(model_path), "capture_root": str(capture_root), "steering_root": str(steering_root),
        "hidden_definition": "decoder_block_output_pre_final_norm",
        "panl_layers": list(PANL_LAYERS), "cle_layers": list(CLE_LAYERS),
        "pairs": [[14, 15], [16, 17], [18, 19]], "alphas": [-5.0, 5.0],
        "seed": SEED, "attention_implementation": "sdpa",
        "source_hashes": {str(p): sha256_file(p) for p in source_files},
        "code_hashes": {str(p): sha256_file(p) for p in code_files},
        "split_audit": split_summary, "smoke": bool(smoke),
    }
    fingerprint = canonical_hash(payload)
    path = output_root / "fingerprint.json"
    existed = path.exists()
    if existed:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise ValueError("Short trajectory output fingerprint mismatch")
        if not resume:
            raise FileExistsError(f"Prepared output exists; use --resume: {output_root}")
    atomic_jsonl(output_root / "artifacts/manifests/probe_construction.jsonl", probe_construction)
    atomic_jsonl(output_root / "artifacts/manifests/probe_audit.jsonl", probe_audit)
    atomic_jsonl(output_root / "artifacts/manifests/test_manifest.jsonl", test)
    atomic_jsonl(output_root / "artifacts/manifests/steering_construction.jsonl", steering_construction)
    atomic_json(output_root / "artifacts/manifests/split_audit.json", split_summary)
    _save_torch_atomic(output_root / "artifacts/vectors/panl_vectors.pt", vector_artifacts)
    atomic_json(output_root / "artifacts/vectors/vector_metadata.json", vector_metadata)
    atomic_json(path, {**payload, "fingerprint": fingerprint})
    result = {
        "status": "complete", "fingerprint": fingerprint, "split_audit": split_summary,
        "vector_count": len(vectors), "resumed": existed,
    }
    atomic_json(output_root / "progress/prepare.json", result)
    return result
