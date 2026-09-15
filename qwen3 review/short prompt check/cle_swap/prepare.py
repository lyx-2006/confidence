from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dp_SA.io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file

from .config import (DIRECTIONS, EXPECTED_CASES, LAYERS, MODEL_PATH, OUTPUT_ROOT,
                     REVERSE_CAPTURE, SEED, SHORT_CAPTURE, SIDE_COUNTS)


def select_manifest(short_rows: list[dict], reverse_rows: list[dict]) -> list[dict]:
    short = {r["case_id"]: r for r in short_rows if r.get("status") == "completed"}
    reverse = {r["case_id"]: r for r in reverse_rows if r.get("status") == "completed"}
    if len(short) != 500 or set(short) != set(reverse):
        raise ValueError(f"Expected matching 500-case captures: {len(short)}/{len(reverse)}")
    used: set[str] = set()
    chosen: list[dict] = []
    for side in ("text_side", "image_side"):
        candidates = [r for r in short.values() if (float(r["soft_sa_image_score"]) < .5) == (side == "text_side")]
        candidates.sort(key=lambda r: (
            float(r["soft_sa_image_score"]) if side == "text_side" else -float(r["soft_sa_image_score"]),
            str(r["case_id"]),
        ))
        selected = []
        for row in candidates:
            item = str(row["item_id"])
            if item in used:
                continue
            other = reverse[row["case_id"]]
            if row["phase0_answer_fingerprint"] != other["phase0_answer_fingerprint"] or row["image_sha256"] != other["image_sha256"]:
                raise ValueError(f"Cross-prompt case mismatch: {row['case_id']}")
            selected.append({
                "case_id": row["case_id"], "item_id": item, "test_side": side,
                "selection_rank": len(selected) + 1,
                "short_clean_sa": float(row["soft_sa_image_score"]),
                "reverse_clean_sa": float(other["soft_sa_image_score"]),
                "distance_from_midpoint": abs(float(row["soft_sa_image_score"]) - .5),
                "phase0_answer_fingerprint": row["phase0_answer_fingerprint"],
                "answer": row["phase0_normalized_answer"],
                "image_sha256": row["image_sha256"],
            })
            used.add(item)
            if len(selected) == SIDE_COUNTS[side]:
                break
        if len(selected) != SIDE_COUNTS[side]:
            raise ValueError(f"Insufficient {side}: {len(selected)}")
        chosen.extend(selected)
    if len(chosen) != EXPECTED_CASES or len({r["item_id"] for r in chosen}) != EXPECTED_CASES:
        raise ValueError("Swap manifest cardinality/item-disjoint gate failed")
    return chosen


def prepare(output_root: Path = OUTPUT_ROOT, *, resume: bool = False):
    root = Path(output_root).resolve()
    short_path = SHORT_CAPTURE / "results.jsonl"
    reverse_path = REVERSE_CAPTURE / "results.jsonl"
    manifest = select_manifest(load_jsonl(short_path), load_jsonl(reverse_path))
    payload = {
        "format_version": 1, "experiment": "short_reverse_cle_bidirectional_swap",
        "layers": list(LAYERS), "directions": list(DIRECTIONS), "case_count": len(manifest),
        "side_counts": dict(Counter(r["test_side"] for r in manifest)),
        "selection": "short_soft_sa_extremes_text_first_global_item_disjoint",
        "seed": SEED, "attention_implementation": "sdpa",
        "model": str(MODEL_PATH.resolve()),
        "source_hashes": {str(p.resolve()): sha256_file(p) for p in (
            SHORT_CAPTURE / "config.json", short_path,
            REVERSE_CAPTURE / "config.json", reverse_path,
            MODEL_PATH / "config.json",
        )},
    }
    fingerprint = canonical_hash(payload)
    old_path = root / "fingerprint.json"
    if old_path.exists():
        old = json.loads(old_path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise ValueError("CLE swap fingerprint mismatch")
        if not resume:
            raise FileExistsError(f"Output exists; use --resume: {root}")
    atomic_jsonl(root / "artifacts/manifests/test_manifest.jsonl", manifest)
    atomic_json(root / "artifacts/manifests/selection_summary.json", {
        "case_count": 50, "item_count": 50,
        "side_counts": dict(Counter(r["test_side"] for r in manifest)),
        "minimum_distance_by_side": {
            side: min(r["distance_from_midpoint"] for r in manifest if r["test_side"] == side)
            for side in SIDE_COUNTS
        },
    })
    atomic_json(old_path, {**payload, "fingerprint": fingerprint})
    result = {"status": "complete", "fingerprint": fingerprint, "case_count": 50}
    atomic_json(root / "progress/prepare.json", result)
    return result
