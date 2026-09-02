from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dp_SA.answer_matched_lat_steering.vectors import scale_direction
from .config import CANONICAL_COLORS, EXPECTED_INPUTS, HIDDEN_CAPTURE, HIDDEN_DEFINITION, HIDDEN_REUSE, LAYERS, SEED, SOURCE_ROOT, VECTOR_NORM_FRACTION
from .io_utils import array_hash, canonical_hash, load_jsonl, sha256_file


def answer_origin(row: dict[str, Any]) -> str:
    text, image = bool(row["answer_matches_text"]), bool(row["answer_matches_image"])
    if text and not image: return "follow_text"
    if image and not text: return "follow_image"
    if not text and not image: return "neither_match"
    return "both_match"


def input_inventory() -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for name, (path, expected_rows, expected_hash) in EXPECTED_INPUTS.items():
        if not path.is_file(): raise FileNotFoundError(path)
        rows = load_jsonl(path); digest = sha256_file(path)
        if len(rows) != expected_rows or digest != expected_hash:
            raise ValueError(f"Frozen input mismatch: {name} rows={len(rows)} sha256={digest}")
        inventory[name] = {"path": str(path.resolve()), "record_count": len(rows), "sha256": digest}
    return inventory


def _unique(rows: Sequence[dict[str, Any]], key, label: str) -> dict[Any, dict[str, Any]]:
    output = {}
    for row in rows:
        value = key(row)
        if value in output: raise ValueError(f"Duplicate {label}: {value}")
        output[value] = row
    return output


def validate_frozen_design(train: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(train) != 1112 or len(test) != 100 or len({r["family_id"] for r in test}) != 50:
        raise ValueError("Frozen train/test cardinality changed")
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test: families[str(row["family_id"])].append(row)
    if any(sorted(r["condition"] for r in rows) != ["conflict_easy", "conflict_hard"] for rows in families.values()):
        raise ValueError("Every test family must contain its frozen easy/hard records")
    definitions = {
        "case_id": lambda r: str(r["case_id"]), "item_id": lambda r: str(r["item_id"]),
        "family_id": lambda r: str(r["family_id"]), "image_hash": lambda r: str(r["image_sha256"]),
        "text_unique_key": lambda r: (str(r["item_id"]), int(r["prior_index"])),
        "image_unique_key": lambda r: (str(r["item_id"]), str(r["condition"]), str(r["image_sha256"])),
    }
    overlaps = {name: len({fn(r) for r in train} & {fn(r) for r in test}) for name, fn in definitions.items()}
    if any(overlaps.values()): raise ValueError(f"Train/test leakage: {overlaps}")
    fold_sets: dict[str, set[int]] = defaultdict(set)
    for row in train: fold_sets[str(row["family_id"])].add(int(row["outer_fold"]))
    if any(len(value) != 1 for value in fold_sets.values()) or set().union(*fold_sets.values()) != set(range(5)):
        raise ValueError("outer_fold is not a frozen five-fold family split")
    return {"status": "passed", "train_records": len(train), "train_families": len(fold_sets), "test_records": len(test), "test_families": len(families), "overlaps": overlaps, "test_records_per_family": 2}


def prepare_rows(train: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]], joined: Sequence[dict[str, Any]], scores: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    manifests = _unique([*train, *test], lambda r: str(r["case_id"]), "manifest case")
    confidence = _unique(joined, lambda r: str(r["case_id"]), "confidence case")
    score_map = _unique(scores, lambda r: (str(r["modality"]), tuple(r["unique_key"])), "score key")
    if set(manifests) != set(confidence): raise ValueError("Manifest/confidence case sets differ")
    output = []
    for case_id in sorted(manifests):
        m, c = manifests[case_id], confidence[case_id]
        if any(str(m[k]) != str(c[k]) for k in ("item_id", "family_id", "condition")): raise ValueError(f"Identity mismatch: {case_id}")
        if str(m["phase0_normalized_answer"]) != str(c["fixed_answer"]): raise ValueError(f"Fixed-answer mismatch: {case_id}")
        if str(m["image_sha256"]) != str(c["image_hash"]): raise ValueError(f"Image hash mismatch: {case_id}")
        tk, ik = ("text", tuple(c["text_score_unique_key"])), ("image", tuple(c["image_score_unique_key"]))
        if tk not in score_map or ik not in score_map: raise ValueError(f"Missing difficulty score: {case_id}")
        lt, li, gl = float(c["text_fixed_answer_log_odds"]), float(c["image_fixed_answer_log_odds"]), float(c["G_L"])
        if not math.isclose(gl, li - lt, rel_tol=0.0, abs_tol=1e-12): raise ValueError(f"G_L mismatch: {case_id}")
        row = {
            "case_id": case_id, "item_id": str(m["item_id"]), "family_id": str(m["family_id"]), "split": str(m["split"]),
            "condition": str(m["condition"]), "outer_fold": int(m["outer_fold"]), "prior_bin": str(m["prior_bin"]),
            "fixed_answer_color": str(c["fixed_answer"]), "answer_origin": answer_origin(m), "Hard": int(m["condition"] == "conflict_hard"),
            "C_t": float(c["text_fixed_answer_confidence"]), "C_i": float(c["image_fixed_answer_confidence"]), "L_t": lt, "L_i": li, "G_L": gl,
            "D_t": float(score_map[tk]["entropy_difficulty"]), "D_i": float(score_map[ik]["entropy_difficulty"]),
            "text_score_unique_key": list(c["text_score_unique_key"]), "image_score_unique_key": list(c["image_score_unique_key"]),
            "confidence_join_fingerprint": str(c["join_fingerprint"]), "score_fingerprint": str(c["score_fingerprint"]),
        }
        if row["answer_origin"] == "both_match": raise ValueError(f"Unexpected both-match answer: {case_id}")
        if not all(math.isfinite(row[k]) for k in ("C_t", "C_i", "L_t", "L_i", "G_L", "D_t", "D_i")): raise ValueError(f"Non-finite row: {case_id}")
        output.append(row)
    return output


def smoke_subset(rows: Sequence[dict[str, Any]], manifests: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train = [r for r in rows if r["split"] == "train"]; test_manifest = [r for r in manifests if r["split"] == "test"]
    cells: dict[str, set[str]] = defaultdict(set)
    for row in train: cells[row["fixed_answer_color"]].add(row["family_id"])
    colors = sorted((c for c in cells if len(cells[c]) >= 8), key=lambda c: (-len(cells[c]), CANONICAL_COLORS.index(c)))[:9]
    selected_families = set()
    for color in colors: selected_families.update(sorted(cells[color])[:8])
    smoke_train = [row for row in train if row["family_id"] in selected_families]
    chosen_test_families = sorted({str(r["family_id"]) for r in test_manifest})[:2]
    smoke_test = [r for r in test_manifest if str(r["family_id"]) in chosen_test_families]
    return smoke_train, smoke_test, {"colors": colors, "train_family_count": len(selected_families), "train_families": sorted(selected_families), "test_families": chosen_test_families, "test_record_count": len(smoke_test)}


NUISANCE_COLUMNS = ("D_t", "D_i", "Hard", "prior_bin", "answer_origin", "fixed_answer_color")


def nuisance_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[row[column] for column in NUISANCE_COLUMNS] for row in rows], dtype=object)


def nuisance_pipeline() -> Pipeline:
    transform = ColumnTransformer([
        ("continuous", StandardScaler(), [0, 1]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), [2, 3, 4, 5]),
    ])
    return Pipeline([("features", transform), ("ridge", Ridge(alpha=1.0, solver="lsqr"))])


def oof_residualize(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, Pipeline]]:
    if not rows: raise ValueError("No training rows")
    predictions: dict[str, float] = {}; models = {}
    for fold in range(5):
        fit = [r for r in rows if int(r["outer_fold"]) != fold]; held = [r for r in rows if int(r["outer_fold"]) == fold]
        if not fit or not held: raise ValueError(f"Empty nuisance fold: {fold}")
        model = nuisance_pipeline(); model.fit(nuisance_matrix(fit), np.asarray([r["G_L"] for r in fit], dtype=float))
        values = model.predict(nuisance_matrix(held)); models[fold] = model
        for row, value in zip(held, values, strict=True):
            if row["case_id"] in predictions: raise ValueError("Duplicate OOF prediction")
            predictions[row["case_id"]] = float(value)
    if set(predictions) != {r["case_id"] for r in rows}: raise ValueError("Incomplete OOF residuals")
    output = [{**r, "predicted_G_L_oof": predictions[r["case_id"]], "R_C": float(r["G_L"] - predictions[r["case_id"]])} for r in rows]
    return output, models


def family_answer_cells(rows: Sequence[dict[str, Any]], *, layers: Sequence[int]) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    reuse={str(r["case_id"]):r for r in load_jsonl(HIDDEN_REUSE)}; capture={str(r["case_id"]):r for r in load_jsonl(HIDDEN_CAPTURE)}; file_hash_cache={}
    def hidden(case_id: str, key: str) -> np.ndarray:
        delta=capture.get(case_id)
        if delta and key in delta["delta_keys"]:
            path=SOURCE_ROOT/delta["delta_file"]; expected_file=delta["delta_file_sha256"]; expected_tensor=None
        else:
            source=reuse[case_id]["cell_sources"].get(key)
            if source is None: raise KeyError(f"Missing hidden cell: {case_id} {key}")
            path=Path(source["path"]); expected_file=source["file_sha256"]; expected_tensor=source["tensor_sha256"]
        path_text=str(path.resolve())
        if path_text not in file_hash_cache: file_hash_cache[path_text]=sha256_file(path)
        if file_hash_cache[path_text] != expected_file: raise ValueError(f"Hidden file fingerprint mismatch: {path}")
        with np.load(path) as payload: raw=np.asarray(payload[key])
        if raw.dtype != np.float16 or raw.shape != (3584,) or not np.isfinite(raw).all(): raise ValueError(f"Invalid frozen hidden: {case_id} {key}")
        if expected_tensor is not None and array_hash(raw) != expected_tensor: raise ValueError(f"Hidden tensor fingerprint mismatch: {case_id} {key}")
        return raw.astype(np.float32)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[row["family_id"], row["fixed_answer_color"]].append(row)
    arrays: dict[int, dict[str, np.ndarray]] = {int(layer): {} for layer in layers}; metadata = []
    for index, ((family, answer), members) in enumerate(sorted(grouped.items())):
        key = f"cell_{index:04d}"; hashes = {}
        for layer in layers:
            value = np.stack([hidden(r["case_id"], f"P1_LAT__L{layer}") for r in members]).mean(axis=0, dtype=np.float32)
            if value.shape != (3584,) or not np.isfinite(value).all(): raise ValueError(f"Invalid cell hidden: {family}/{answer}/L{layer}")
            arrays[int(layer)][key] = value; hashes[f"L{layer}"] = array_hash(value)
        metadata.append({"array_key": key, "family_id": family, "fixed_answer_color": answer, "case_ids": sorted(r["case_id"] for r in members), "record_count": len(members), "mean_residual": float(np.mean([r["R_C"] for r in members])), "hidden_hashes": hashes})
    return metadata, arrays


def tail_assignments(cells: Sequence[dict[str, Any]]) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells: by_answer[cell["fixed_answer_color"]].append(cell)
    rng = np.random.default_rng(SEED); assignments = {}; audit = []
    for answer in CANONICAL_COLORS:
        ordered = sorted(by_answer.get(answer, []), key=lambda r: (float(r["mean_residual"]), str(r["family_id"])))
        n = len(ordered); k = max(2, int(math.floor(.3 * n))) if n else 0; eligible = n >= 8 and 2 * k <= n
        true = {}
        if eligible:
            for i, cell in enumerate(ordered): true[cell["array_key"]] = "low" if i < k else "high" if i >= n-k else "middle"
            labels = [true[cell["array_key"]] for cell in sorted(ordered, key=lambda r: str(r["family_id"]))]
            permuted = [str(value) for value in rng.permutation(labels)]; shuffled = {cell["array_key"]: label for cell, label in zip(sorted(ordered, key=lambda r: str(r["family_id"])), permuted, strict=True)}
            assignments[answer] = {f"true:{key}": value for key, value in true.items()} | {f"shuffled:{key}": value for key, value in shuffled.items()}
        audit.append({"fixed_answer_color": answer, "cell_count": n, "tail_count": k, "eligible": eligible,
                      "true_high_families": sorted(c["family_id"] for c in ordered if true.get(c["array_key"]) == "high"),
                      "true_low_families": sorted(c["family_id"] for c in ordered if true.get(c["array_key"]) == "low"),
                      "true_middle_families": sorted(c["family_id"] for c in ordered if true.get(c["array_key"]) == "middle"),
                      "shuffled_high_families": sorted(c["family_id"] for c in ordered if eligible and shuffled.get(c["array_key"]) == "high"),
                      "shuffled_low_families": sorted(c["family_id"] for c in ordered if eligible and shuffled.get(c["array_key"]) == "low"),
                      "shuffled_middle_families": sorted(c["family_id"] for c in ordered if eligible and shuffled.get(c["array_key"]) == "middle")})
    return assignments, audit


def build_vectors(cells: Sequence[dict[str, Any]], arrays: dict[int, dict[str, np.ndarray]], assignments: dict[str, dict[str, str]], eligibility: Sequence[dict[str, Any]], recipients: Sequence[str], *, layers: Sequence[int]) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]]]:
    eligible = [r["fixed_answer_color"] for r in eligibility if r["eligible"]]
    by_answer = defaultdict(list)
    for cell in cells: by_answer[cell["fixed_answer_color"]].append(cell)
    output: dict[int, dict[str, np.ndarray]] = {int(layer): {} for layer in layers}; metadata = []
    for layer in layers:
        true_d, shuffled_d = {}, {}
        for answer in eligible:
            answer_cells = by_answer[answer]; mapping = assignments[answer]
            def mean(label: str, prefix: str) -> np.ndarray:
                chosen = [arrays[int(layer)][c["array_key"]] for c in answer_cells if mapping[f"{prefix}:{c['array_key']}"] == label]
                if len(chosen) < 2: raise ValueError(f"Tail gate failed: {answer}/{prefix}/{label}")
                return np.stack(chosen).mean(axis=0, dtype=np.float32)
            true_d[answer] = mean("high", "true") - mean("low", "true")
            shuffled_d[answer] = mean("high", "shuffled") - mean("low", "shuffled")
        for recipient in sorted(set(recipients)):
            included = [answer for answer in eligible if answer != recipient]
            if len(included) < 8: raise ValueError(f"LOAO gate failed for {recipient}: {len(included)}")
            norm_cells = [arrays[int(layer)][c["array_key"]] for answer in included for c in by_answer[answer]]
            target = float(VECTOR_NORM_FRACTION * np.mean([np.linalg.norm(v) for v in norm_cells]))
            for direction, directions in (("residual_confidence_loao", true_d), ("within_answer_shuffled", shuffled_d)):
                raw = np.stack([directions[a] for a in included]).mean(axis=0, dtype=np.float32); scaled = scale_direction(raw, target)
                raw_key = f"{recipient}__{direction}__raw"; scaled_key = f"{recipient}__{direction}__scaled"
                output[int(layer)][raw_key] = raw.astype(np.float32); output[int(layer)][scaled_key] = scaled.astype(np.float32)
                raw_norm, scaled_norm = float(np.linalg.norm(raw)), float(np.linalg.norm(scaled))
                if not all(math.isfinite(x) and x > 0 for x in (raw_norm, target, scaled_norm)) or not math.isclose(scaled_norm, target, rel_tol=2e-6, abs_tol=1e-6): raise ValueError("Vector norm gate failed")
                metadata.append({"recipient_answer": recipient, "layer": int(layer), "direction": direction, "raw_key": raw_key, "scaled_key": scaled_key, "raw_norm": raw_norm, "target_norm": target, "scaled_norm": scaled_norm, "included_answers": included, "excluded_answer": recipient,
                                 "construction_families_by_answer": {a: sorted(c["family_id"] for c in by_answer[a]) for a in included},
                                 "raw_hash": array_hash(raw), "scaled_hash": array_hash(scaled), "vector_fingerprint": canonical_hash({"layer": int(layer), "recipient": recipient, "direction": direction, "scaled_hash": array_hash(scaled)})})
    return output, metadata
