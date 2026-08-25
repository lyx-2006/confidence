from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .config import CONSTRUCTION_PER_SIDE, RECIPIENTS_PER_SIDE, SMOKE_RECIPIENTS_PER_SIDE
from .matching import build_swap_pairs
from .utils import canonical_hash, load_jsonl, stable_key


def _required(row: dict[str, Any], *, side_field: str) -> None:
    required = ("case_id", "item_id", "phase0_raw_answer", "phase1_inserted_raw_answer", "phase1_prompt",
                "phase1_prompt_hash", "phase1_answer_span", "positions", "class_logits", "argmax_hard_class",
                "phase0_normalized_answer", "image_path", "image_sha256", "hidden_file")
    missing = [key for key in required if row.get(key) is None]
    if missing:
        raise ValueError(f"{row.get('case_id')}: missing manifest fields {missing}")
    if row.get(side_field) is None:
        raise ValueError(f"{row.get('case_id')}: missing {side_field}")
    if row["phase0_raw_answer"] != row["phase1_inserted_raw_answer"]:
        raise ValueError(f"{row['case_id']}: Phase 0/Phase 1 answer mismatch")
    if int(row["argmax_hard_class"]) == 4:
        raise ValueError(f"{row['case_id']}: class 4 is not eligible")
    if len(row["class_logits"]) != 9:
        raise ValueError(f"{row['case_id']}: expected nine class logits")


def load_frozen_manifests(source_root: Path, *, max_recipients_per_side: int, smoke: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    steering = source_root / "steering"
    construction = load_jsonl(steering / "construction_manifest.jsonl")
    test = load_jsonl(steering / "test_manifest.jsonl")
    if len(construction) != 2 * CONSTRUCTION_PER_SIDE:
        raise ValueError(f"frozen construction manifest must have 50 rows, found {len(construction)}")
    if len(test) != 2 * RECIPIENTS_PER_SIDE:
        raise ValueError(f"frozen test manifest must have 100 rows, found {len(test)}")
    for row in construction:
        _required(row, side_field="construction_side")
        if row["construction_side"] not in {"high_image", "high_text"}:
            raise ValueError("invalid construction side")
    for row in test:
        _required(row, side_field="test_side")
        if row["test_side"] not in {"image_side", "text_side"}:
            raise ValueError("invalid test side")
    if sum(row["construction_side"] == "high_image" for row in construction) != 25 or sum(row["construction_side"] == "high_text" for row in construction) != 25:
        raise ValueError("construction side counts are not 25/25")
    if sum(row["test_side"] == "image_side" for row in test) != 50 or sum(row["test_side"] == "text_side" for row in test) != 50:
        raise ValueError("test side counts are not 50/50")
    construction_items = {str(row["item_id"]) for row in construction}
    test_items = {str(row["item_id"]) for row in test}
    if len(construction_items) != len(construction) or len(test_items) != len(test) or construction_items & test_items:
        raise ValueError("construction/test item leakage or duplicate items")
    if max_recipients_per_side < 1 or max_recipients_per_side > RECIPIENTS_PER_SIDE:
        raise ValueError("max recipients per side must be in [1, 50]")
    if smoke:
        max_recipients_per_side = SMOKE_RECIPIENTS_PER_SIDE
    selected = []
    for side in ("image_side", "text_side"):
        side_rows = sorted((row for row in test if row["test_side"] == side), key=stable_key)
        selected.extend(side_rows[:max_recipients_per_side])
    selected.sort(key=stable_key)
    if len(selected) != 2 * max_recipients_per_side:
        raise ValueError("recipient selection is not balanced")
    return [dict(row) for row in construction], [dict(row) for row in selected]


def enrich_length(row: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    spans = details["spans"]
    output = dict(row)
    output["question_token_length"] = int(spans["QUESTION"][1] - spans["QUESTION"][0])
    output["answer_token_length"] = int(spans["ANSWER"][1] - spans["ANSWER"][0])
    output["position_fingerprint"] = canonical_hash(details["located"])
    output["input_fingerprint"] = canonical_hash({
        "messages_hash": details["messages_hash"], "rendered_hash": details["rendered_hash"],
        "image_sha256": details["image_sha256"], "question_hash": details["question_hash"],
        "text_clue_hash": details["text_clue_hash"], "answer_hash": details["answer_hash"],
        "located": details["located"], "token_ids": details["located"].get("phase1_answer_token_ids", []),
    })
    return output


def assert_frozen_output(path: Path, rows: Sequence[dict[str, Any]], *, key: str) -> None:
    expected = {str(row["case_id"]): row for row in rows}
    if path.exists():
        previous = {str(row["case_id"]): row for row in load_jsonl(path, repair_trailing=True)}
        if canonical_hash(previous) != canonical_hash(expected):
            raise ValueError(f"frozen {key} differs; refusing resume")


__all__ = ["assert_frozen_output", "enrich_length", "load_frozen_manifests"]
