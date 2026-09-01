from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import DIRECTIONS, LAYERS, POSITIONS, SEED, SMOKE_DIRECTIONS, SMOKE_LAYERS, VECTOR_NORM_FRACTION
from .io_utils import array_hash, atomic_json, atomic_npz, canonical_hash, load_jsonl, sha256_file


def family_equal_mean(values: dict[str, Sequence[np.ndarray]]) -> tuple[np.ndarray, int]:
    if not values: raise ValueError("Family mean requires at least one family")
    family_means = []
    for family in sorted(values):
        arrays = np.stack([np.asarray(value, dtype=np.float32) for value in values[family]])
        family_means.append(arrays.mean(axis=0, dtype=np.float32))
    return np.stack(family_means).mean(axis=0, dtype=np.float32), len(family_means)


def scale_direction(raw: np.ndarray, target_norm: float) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float32); norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0 or not math.isfinite(target_norm) or target_norm <= 0: raise ValueError("Direction norm is invalid")
    return np.asarray(value / norm * target_norm, dtype=np.float32)


def loao_direction(answer_directions: dict[str, np.ndarray], recipient_answer: str) -> tuple[np.ndarray, list[str]]:
    included = sorted(answer for answer in answer_directions if answer != recipient_answer)
    if len(included) < 3: raise ValueError(f"LOAO leaves only {len(included)} eligible answers for {recipient_answer}")
    return np.stack([answer_directions[answer] for answer in included]).mean(axis=0, dtype=np.float32), included


def recipient_target_norm(cell_values: dict[tuple[str, str, str], np.ndarray], included_answers: Sequence[str]) -> float:
    included = set(included_answers)
    values = [np.asarray(value, dtype=np.float32) for (_family, answer, _side), value in cell_values.items() if answer in included]
    if not values: raise ValueError("Recipient target norm has no included construction cells")
    target = float(VECTOR_NORM_FRACTION * np.mean([np.linalg.norm(value) for value in values]))
    if not math.isfinite(target) or target <= 0: raise ValueError("Recipient target norm is invalid")
    return target


def shuffled_answer_direction(cells: Sequence[tuple[str, str, np.ndarray]], *, seed: str) -> tuple[np.ndarray, dict[str, int]]:
    ordered = sorted(cells, key=lambda value: (value[0], value[1])); labels = [side for _family, side, _value in ordered]
    random.Random(seed).shuffle(labels)
    groups: dict[str, dict[str, list[np.ndarray]]] = {"high_text": defaultdict(list), "high_image": defaultdict(list)}
    for (family, _old_side, value), new_side in zip(ordered, labels): groups[new_side][family].append(value)
    high, high_count = family_equal_mean(groups["high_image"]); low, low_count = family_equal_mean(groups["high_text"])
    return np.asarray(high - low, dtype=np.float32), {"high_text": low_count, "high_image": high_count}


def _hidden(root: Path, clean: dict[str, dict[str, Any]], case_id: str, position: str, layer: int) -> np.ndarray:
    row = clean[case_id]
    with np.load(root / row["hidden_file"]) as payload: return np.asarray(payload[f"{position}__L{layer}"], dtype=np.float32)


def _metadata_fingerprint(rows: Sequence[dict[str, Any]]) -> str:
    return canonical_hash([{k: row[k] for k in ("position", "fold", "recipient_answer", "layer", "direction", "vector_file", "scaled_key", "vector_fingerprint")} for row in rows])


def build_vectors(root: Path, *, smoke: bool, resume: bool) -> dict[str, Any]:
    metadata_path = root / "artifacts" / "vectors" / "vector_metadata.json"
    if metadata_path.exists():
        if not resume: raise FileExistsError("Vector artifacts exist; use --resume")
        metadata = json.loads(metadata_path.read_text())
        for row in metadata["vectors"]:
            path = root / row["vector_file"]
            if not path.is_file() or sha256_file(path) != row["vector_file_sha256"]: raise ValueError("Vector artifact fingerprint mismatch")
        if metadata.get("fingerprint") != _metadata_fingerprint(metadata["vectors"]): raise ValueError("Vector metadata fingerprint mismatch")
        return {**metadata, "resumed_noop": True}
    manifests = root / "artifacts" / "manifests"
    candidates = load_jsonl(manifests / "candidate_manifest.jsonl"); test = load_jsonl(manifests / "test_manifest.jsonl")
    cells = load_jsonl(manifests / "construction_family_cells.jsonl"); distribution = load_jsonl(manifests / "construction_distribution.jsonl")
    clean_rows = [row for row in load_jsonl(root / "artifacts" / "diagnostics" / "clean_capture.jsonl") if row.get("status") == "completed"]
    clean = {str(row["case_id"]): row for row in clean_rows}
    if {str(row["case_id"]) for row in candidates} != set(clean): raise ValueError("Vector construction requires complete clean hidden capture")
    candidate_by_case = {str(row["case_id"]): row for row in candidates}
    test_answers_by_fold: dict[int, set[str]] = defaultdict(set)
    for row in test: test_answers_by_fold[int(row["fold"])].add(str(row["test_answer"]))
    layers = SMOKE_LAYERS if smoke else LAYERS; directions = SMOKE_DIRECTIONS if smoke else DIRECTIONS; vector_rows = []
    for fold in sorted({int(row["fold"]) for row in cells}):
        fold_cells = [row for row in cells if int(row["fold"]) == fold]
        eligible = sorted(row["answer"] for row in distribution if int(row["fold"]) == fold and bool(row["eligible_for_direction"]))
        if len(eligible) < 4: raise ValueError(f"Fold {fold} has insufficient eligible answers")
        heldout = {row["family_id"] for row in load_jsonl(manifests / "fold_assignments.jsonl") if int(row["fold"]) == fold}
        if any(row["family_id"] in heldout for row in fold_cells): raise ValueError("Held-out family entered vector construction")
        for position in POSITIONS:
            for layer in layers:
                cell_values: dict[tuple[str, str, str], np.ndarray] = {}
                for cell in fold_cells:
                    values = [_hidden(root, clean, str(case_id), position, layer) for case_id in cell["case_ids"]]
                    cell_values[str(cell["family_id"]), str(cell["answer"]), str(cell["sa_side"])] = np.stack(values).mean(axis=0, dtype=np.float32)
                true_directions = {}; shuffled_directions = {}; answer_counts = {}
                for answer in eligible:
                    side_groups = {"high_text": defaultdict(list), "high_image": defaultdict(list)}; shuffled_cells = []
                    for (family, candidate_answer, side), value in cell_values.items():
                        if candidate_answer != answer: continue
                        side_groups[side][family].append(value); shuffled_cells.append((family, side, value))
                    high, high_count = family_equal_mean(side_groups["high_image"]); low, low_count = family_equal_mean(side_groups["high_text"])
                    if min(high_count, low_count) < (2 if smoke else 15): raise ValueError(f"Direction family gate failed: position={position} fold={fold} answer={answer}")
                    true_directions[answer] = np.asarray(high - low, dtype=np.float32)
                    shuffled_directions[answer], shuffled_counts = shuffled_answer_direction(shuffled_cells, seed=f"{SEED}|fold={fold}|answer={answer}")
                    answer_counts[answer] = {"high_text": low_count, "high_image": high_count, "shuffled": shuffled_counts}
                global_groups = {"high_text": defaultdict(list), "high_image": defaultdict(list)}
                for (family, _answer, side), value in cell_values.items(): global_groups[side][family].append(value)
                global_high, global_high_count = family_equal_mean(global_groups["high_image"]); global_low, global_low_count = family_equal_mean(global_groups["high_text"])
                global_raw = np.asarray(global_high - global_low, dtype=np.float32)
                arrays = {}; pending = []
                for recipient in sorted(test_answers_by_fold[fold]):
                    matched_raw, included = loao_direction(true_directions, recipient)
                    shuffled_raw, shuffled_included = loao_direction(shuffled_directions, recipient)
                    if included != shuffled_included: raise AssertionError("True/shuffled LOAO answers differ")
                    target_norm = recipient_target_norm(cell_values, included)
                    raw_by_direction = {"matched_loao": matched_raw, "unmatched_global": global_raw, "within_answer_shuffled": shuffled_raw}
                    for direction in directions:
                        raw = raw_by_direction[direction]; scaled = scale_direction(raw, target_norm)
                        prefix = f"{position}__{recipient}__{direction}"; raw_key = prefix + "__raw"; scaled_key = prefix + "__scaled"
                        arrays[raw_key] = raw.astype(np.float32); arrays[scaled_key] = scaled
                        fingerprint = canonical_hash({"position": position, "scaled_sha256": array_hash(scaled)})
                        pending.append({"position": position, "fold": fold, "recipient_answer": recipient, "layer": int(layer), "direction": direction, "raw_key": raw_key, "scaled_key": scaled_key, "raw_norm": float(np.linalg.norm(raw)), "target_norm": target_norm, "scaled_norm": float(np.linalg.norm(scaled)), "eligible_answers_after_loao": included, "eligible_answer_count_after_loao": len(included), "answer_family_counts": answer_counts, "global_family_counts": {"high_text": global_low_count, "high_image": global_high_count}, "vector_fingerprint": fingerprint})
                relative = Path("artifacts") / "vectors" / f"{position}__fold_{fold:02d}__L{layer}.npz"; destination = root / relative; atomic_npz(destination, arrays); file_hash = sha256_file(destination)
                for row in pending: vector_rows.append({**row, "vector_file": str(relative), "vector_file_sha256": file_hash})
    metadata = {"status": "complete", "smoke_only": smoke, "vectors": vector_rows, "vector_count": len(vector_rows), "fingerprint": _metadata_fingerprint(vector_rows), "resumed_noop": False}
    atomic_json(metadata_path, metadata); return metadata


def load_vector(root: Path, metadata: dict[str, Any], *, position: str, fold: int, answer: str, layer: int, direction: str) -> tuple[np.ndarray, dict[str, Any]]:
    matches = [row for row in metadata["vectors"] if row["position"] == position and int(row["fold"]) == fold and row["recipient_answer"] == answer and int(row["layer"]) == layer and row["direction"] == direction]
    if len(matches) != 1: raise ValueError(f"Missing/duplicate vector: {position} fold={fold} answer={answer} L{layer} {direction}")
    row = matches[0]
    with np.load(root / row["vector_file"]) as payload: value = np.asarray(payload[row["scaled_key"]], dtype=np.float32)
    if canonical_hash({"position": position, "scaled_sha256": array_hash(value)}) != row["vector_fingerprint"]: raise ValueError("Scaled vector fingerprint mismatch")
    return value, row
