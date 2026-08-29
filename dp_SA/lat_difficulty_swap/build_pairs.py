from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from .config import (
    DATASET_PATH, DECISION_OOF_PATH, DIFFICULTY_OOF_PATH, HISTORICAL_SA_OOF_PATH,
    JOINED_PATH, MIN_DIFFICULTY_GAP, PANL_CAPTURE_PATH, PHASE0_PATH, RESULTS_ROOT,
    SPLIT_PATH,
)
from .io_utils import atomic_json, atomic_jsonl, canonical_hash, ensure_layout, load_jsonl, sha256_file, stage_update, text_hash


def _same_bytes(left: Any, right: Any) -> bool:
    return str(left).encode("utf-8") == str(right).encode("utf-8")


def _answer_ids(row: dict[str, Any], phase0: dict[str, dict[str, Any]]) -> tuple[list[int], list[int]]:
    p0 = phase0[str(row["case_id"])]
    return list(map(int, p0.get("phase0_answer_token_ids", []))), list(map(int, row["phase1_answer_token_ids"]))


def _audit_pair(left: dict[str, Any], right: dict[str, Any], phase0: dict[str, dict[str, Any]], arm: str) -> dict[str, bool]:
    left_p0, left_p1 = _answer_ids(left, phase0)
    right_p0, right_p1 = _answer_ids(right, phase0)
    common = {
        "question_bytes_equal": _same_bytes(left["question"], right["question"]),
        "phase0_raw_answer_bytes_equal": _same_bytes(left["phase0_raw_answer"], right["phase0_raw_answer"]),
        "normalized_answer_equal": left["phase0_normalized_answer"] == right["phase0_normalized_answer"],
        "phase0_answer_token_ids_equal": left_p0 == right_p0 and bool(left_p0),
        "phase1_answer_token_ids_equal": left_p1 == right_p1 and bool(left_p1),
        "outer_fold_equal": int(left["outer_fold"]) == int(right["outer_fold"]),
    }
    if arm == "A":
        common.update({
            "prior_index_equal": int(left["prior_index"]) == int(right["prior_index"]),
            "text_clue_bytes_equal": _same_bytes(left["text_clue"], right["text_clue"]),
            "image_condition_changed": left["condition"] != right["condition"],
            "image_path_changed": str(left["image_path"]) != str(right["image_path"]),
            "image_hash_changed": str(left["image_sha256"]) != str(right["image_sha256"]),
        })
    else:
        common.update({
            "prior_index_changed": int(left["prior_index"]) != int(right["prior_index"]),
            "text_clue_bytes_changed": not _same_bytes(left["text_clue"], right["text_clue"]),
            "image_condition_equal": left["condition"] == right["condition"],
            "image_path_equal": str(left["image_path"]) == str(right["image_path"]),
            "image_hash_equal": str(left["image_sha256"]) == str(right["image_sha256"]),
        })
    return common


def _candidate(arm: str, easy: dict[str, Any], hard: dict[str, Any], gap: float, audit: dict[str, bool]) -> dict[str, Any]:
    return {
        "arm": arm, "item_id": str(easy["item_id"]), "outer_fold": int(easy["outer_fold"]),
        "easy_case_id": str(easy["case_id"]), "hard_case_id": str(hard["case_id"]),
        "easy_prior_index": int(easy["prior_index"]), "hard_prior_index": int(hard["prior_index"]),
        "easy_condition": str(easy["condition"]), "hard_condition": str(hard["condition"]),
        "easy_difficulty": float(easy[f"{'image' if arm == 'A' else 'text'}_model_perceived_difficulty"]),
        "hard_difficulty": float(hard[f"{'image' if arm == 'A' else 'text'}_model_perceived_difficulty"]),
        "difficulty_gap": float(gap), "audit": audit,
        "question_sha256": text_hash(easy["question"]),
        "easy_text_clue_sha256": text_hash(easy["text_clue"]), "hard_text_clue_sha256": text_hash(hard["text_clue"]),
        "easy_image_sha256": str(easy["image_sha256"]), "hard_image_sha256": str(hard["image_sha256"]),
        "phase0_raw_answer_sha256": text_hash(easy["phase0_raw_answer"]),
        "phase0_raw_answer": str(easy["phase0_raw_answer"]),
        "phase0_normalized_answer": str(easy["phase0_normalized_answer"]),
        "answer_token_ids": list(map(int, easy["phase1_answer_token_ids"])),
    }


def build_pairs(rows: Sequence[dict[str, Any]], phase0_rows: Sequence[dict[str, Any]], *, min_gap: float = MIN_DIFFICULTY_GAP) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid = [row for row in rows if row.get("status") == "completed"]
    phase0 = {str(row["case_id"]): row for row in phase0_rows if row.get("status") == "completed"}
    missing = sorted({str(row["case_id"]) for row in valid} - set(phase0))
    if missing:
        raise ValueError(f"Missing completed Phase 0 rows: {missing[:5]} ({len(missing)})")
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    by_image_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_image_key[(str(row["item_id"]), int(row["prior_index"]))].append(row)
    eligible_a: list[dict[str, Any]] = []
    for key, group in sorted(by_image_key.items()):
        easy = [row for row in group if row["condition"] == "conflict_easy"]
        hard = [row for row in group if row["condition"] == "conflict_hard"]
        if len(easy) != 1 or len(hard) != 1:
            excluded.append({"arm": "A", "item_id": key[0], "prior_index": key[1], "reasons": ["missing_or_duplicate_image_condition"]})
            continue
        e, h = easy[0], hard[0]
        audit = _audit_pair(e, h, phase0, "A")
        gap = float(h["image_model_perceived_difficulty"]) - float(e["image_model_perceived_difficulty"])
        row = _candidate("A", e, h, gap, audit)
        reasons = [name for name, passed in audit.items() if not passed]
        if gap < min_gap:
            reasons.append("difficulty_gap_below_threshold")
        row.update({"eligible": not reasons, "reasons": reasons})
        candidates.append(row)
        if reasons:
            excluded.append(row)
        else:
            eligible_a.append(row)
    selected_a: list[dict[str, Any]] = []
    for item in sorted({row["item_id"] for row in eligible_a}, key=lambda value: (int(value) if value.isdigit() else value)):
        options = [row for row in eligible_a if row["item_id"] == item]
        chosen = sorted(options, key=lambda row: (row["easy_prior_index"], row["easy_case_id"], row["hard_case_id"]))[0]
        chosen = {**chosen, "pair_id": f"A__{item}", "selected": True}
        selected_a.append(chosen)
        for row in options:
            if row["easy_case_id"] != chosen["easy_case_id"]:
                excluded.append({**row, "reasons": ["eligible_not_selected_item_limit"]})

    by_text_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_text_key[(str(row["item_id"]), str(row["condition"]), str(row["image_path"]), str(row["image_sha256"]))].append(row)
    eligible_b: list[dict[str, Any]] = []
    for _key, group in sorted(by_text_key.items()):
        ordered = sorted(group, key=lambda row: (int(row["prior_index"]), str(row["case_id"])))
        for index, first in enumerate(ordered):
            for second in ordered[index + 1:]:
                easy, hard = sorted((first, second), key=lambda row: (float(row["text_model_perceived_difficulty"]), int(row["prior_index"]), str(row["case_id"])))
                audit = _audit_pair(easy, hard, phase0, "B")
                gap = float(hard["text_model_perceived_difficulty"]) - float(easy["text_model_perceived_difficulty"])
                row = _candidate("B", easy, hard, gap, audit)
                reasons = [name for name, passed in audit.items() if not passed]
                if gap < min_gap:
                    reasons.append("difficulty_gap_below_threshold")
                row.update({"eligible": not reasons, "reasons": reasons})
                candidates.append(row)
                if reasons:
                    excluded.append(row)
                else:
                    eligible_b.append(row)
    selected_b: list[dict[str, Any]] = []
    for item in sorted({row["item_id"] for row in eligible_b}, key=lambda value: (int(value) if value.isdigit() else value)):
        options = [row for row in eligible_b if row["item_id"] == item]
        chosen = sorted(options, key=lambda row: (
            -row["difficulty_gap"], 0 if row["easy_condition"] == "conflict_easy" else 1,
            row["easy_prior_index"], row["hard_prior_index"], row["easy_case_id"], row["hard_case_id"],
        ))[0]
        chosen = {**chosen, "pair_id": f"B__{item}", "selected": True}
        selected_b.append(chosen)
        chosen_key = (chosen["easy_case_id"], chosen["hard_case_id"])
        for row in options:
            if (row["easy_case_id"], row["hard_case_id"]) != chosen_key:
                excluded.append({**row, "reasons": ["eligible_not_selected_max_gap_or_tie_break"]})
    if len({row["item_id"] for row in selected_a}) != len(selected_a) or len({row["item_id"] for row in selected_b}) != len(selected_b):
        raise AssertionError("An item contributes more than one pair within an arm")
    return candidates, selected_a, selected_b, excluded


def source_fingerprints(rows: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    from .config import MODEL_PATH, PACKAGE_ROOT, PANL_READOUT_BY_SWAP_LAYER, SWAP_LAYERS
    repo = PACKAGE_ROOT.parents[1]
    data_paths = (DATASET_PATH, JOINED_PATH, PHASE0_PATH, PANL_CAPTURE_PATH, HISTORICAL_SA_OOF_PATH, DIFFICULTY_OOF_PATH, DECISION_OOF_PATH, SPLIT_PATH)
    model_files = sorted(path for pattern in ("*.json", "tokenizer*", "*.txt", "*.safetensors") for path in MODEL_PATH.glob(pattern) if path.is_file())
    source_files = sorted(PACKAGE_ROOT.glob("*.py")) + [repo / path for path in (
        Path("dp_SA/prompts.py"), Path("dp_SA/positions.py"), Path("dp_SA/soft_score.py"),
        Path("dp_SA/patching/protocol.py"), Path("layer_metacognition/model_adapter.py"),
        Path("layer_metacognition/conversation_builder.py"),
    )]
    image_hashes = {}
    for row in rows or load_jsonl(JOINED_PATH):
        path = Path(str(row["image_path"])).resolve()
        if path.is_file() and str(path) not in image_hashes:
            actual = sha256_file(path)
            if actual != str(row["image_sha256"]): raise ValueError(f"Image hash changed: {path}")
            image_hashes[str(path)] = actual
    return {
        "inputs": {str(path.resolve()): sha256_file(path) for path in data_paths},
        "model_processor_tokenizer": {str(path.relative_to(MODEL_PATH)): sha256_file(path) for path in model_files},
        "images": image_hashes,
        "source_code": {str(path.relative_to(repo)): sha256_file(path) for path in source_files},
        "registered_grid": {"swap_layers": list(SWAP_LAYERS), "panl_readout_mapping": PANL_READOUT_BY_SWAP_LAYER},
        "output_schema_version": 1,
    }


def build_pair_artifacts(root: Path, *, resume: bool, min_gap: float = MIN_DIFFICULTY_GAP) -> dict[str, Any]:
    ensure_layout(root, resume=resume)
    rows, phase0 = load_jsonl(JOINED_PATH), load_jsonl(PHASE0_PATH)
    candidates, image_pairs, text_pairs, excluded = build_pairs(rows, phase0, min_gap=min_gap)
    if len(image_pairs) < 2 or len(text_pairs) < 2:
        raise RuntimeError(f"Insufficient eligible pairs at gap {min_gap}: A={len(image_pairs)} B={len(text_pairs)}")
    payload = {
        "format_version": 1, "experiment": "delayed_sa_unimodal_difficulty_lat_swap",
        "min_difficulty_gap": float(min_gap), "source_fingerprints": source_fingerprints(rows),
        "pair_manifest_hashes": {"image": canonical_hash(image_pairs), "text": canonical_hash(text_pairs)},
        "tie_break": {"A": "minimum_prior_index_then_case_id", "B": "maximum_gap_then_easy_condition_prior_indices_case_ids"},
    }
    payload["fingerprint"] = canonical_hash(payload)
    config_path = root / "progress" / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != payload["fingerprint"]:
            raise ValueError("Pair/config fingerprint mismatch; refusing resume")
        if not resume:
            raise FileExistsError(f"Output exists; pass --resume: {root}")
    else:
        atomic_json(config_path, payload)
    atomic_json(root / "progress" / "input_fingerprints.json", payload["source_fingerprints"])
    if not (root / "progress" / "failures.jsonl").exists():
        atomic_jsonl(root / "progress" / "failures.jsonl", [])
    atomic_jsonl(root / "artifacts" / "all_candidate_pairs.jsonl", candidates)
    atomic_jsonl(root / "artifacts" / "image_pair_manifest.jsonl", image_pairs)
    atomic_jsonl(root / "artifacts" / "text_pair_manifest.jsonl", text_pairs)
    atomic_jsonl(root / "artifacts" / "excluded_pairs.jsonl", excluded)
    audit = {"status": "passed", "candidate_count": len(candidates), "image_pair_count": len(image_pairs), "text_pair_count": len(text_pairs), "image_item_count": len(image_pairs), "text_item_count": len(text_pairs), "min_gap": min_gap, "item_unique_within_arm": True, "split_leakage": False, "selection_uses_outcome": False}
    atomic_json(root / "artifacts" / "pairing_audit.json", audit)
    stage_update(root, "pairing", "complete", **{key: value for key, value in audit.items() if key != "status"})
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build frozen same-answer LAT difficulty-swap pairs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-gap", type=float, default=MIN_DIFFICULTY_GAP)
    args = parser.parse_args(argv)
    print(json.dumps(build_pair_artifacts(RESULTS_ROOT, resume=args.resume, min_gap=args.min_gap), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
