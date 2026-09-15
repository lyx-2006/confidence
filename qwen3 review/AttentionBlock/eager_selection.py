from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from dp_SA.attention_block.run import _forward
from dp_SA.io_utils import append_jsonl, atomic_json, atomic_jsonl, canonical_hash, load_jsonl
from dp_SA.selection import record_key
from dp_SA.soft_score import class_token_ids
from Steering.runtime import load_qwen3_inference

from .config import (
    CAPTURE_ROOT, CLEAN_LOGIT_TOLERANCE, CLEAN_SOFT_SA_TOLERANCE,
    MODEL_PATH, OUTPUT_ROOT, SEED, TEST_PER_SIDE,
)
from .run import _case_context
from .selection import sa_side


SHARED_ROOT = OUTPUT_ROOT / "shared_eager_selection"


def ensure_eager_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = SHARED_ROOT / "test_manifest.jsonl"
    completion_path = SHARED_ROOT / "completion.json"
    if completion_path.exists() and manifest_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        rows = load_jsonl(manifest_path)
        if completion.get("status") == "complete" and len(rows) == 2 * TEST_PER_SIDE:
            return rows, completion

    capture_config = json.loads((CAPTURE_ROOT / "config.json").read_text(encoding="utf-8"))
    eligible = [row for row in load_jsonl(CAPTURE_ROOT / "results.jsonl")
                if row.get("status") == "completed" and row.get("valid_class", True) and sa_side(row)]
    eligible.sort(key=record_key)
    permutation = list(range(len(eligible)))
    random.Random(SEED).shuffle(permutation)
    assessments_path = SHARED_ROOT / "assessments.jsonl"
    assessments = {row["case_id"]: row for row in load_jsonl(assessments_path)}
    selected: list[dict[str, Any]] = []
    runtime = load_qwen3_inference(MODEL_PATH, attn_implementation="eager")
    token_ids = class_token_ids(runtime.processor.tokenizer)
    try:
        for permutation_index in permutation:
            row = eligible[permutation_index]
            assessment = assessments.get(row["case_id"])
            if assessment is None:
                inputs, _located, _positions = _case_context(runtime, row)
                logits, score = _forward(runtime.model, inputs, _positions["P1_SAC"], token_ids)
                logit_error = max(abs(float(a) - float(b)) for a, b in zip(logits, row["class_logits"]))
                soft_error = abs(float(score["soft_sa_image_score"]) - float(row["soft_sa_image_score"]))
                hard_equal = int(score["argmax_hard_class"]) == int(row["argmax_hard_class"])
                passed = hard_equal and logit_error <= CLEAN_LOGIT_TOLERANCE and soft_error <= CLEAN_SOFT_SA_TOLERANCE
                assessment = {
                    "case_id": row["case_id"], "passed": passed,
                    "logit_max_abs_error": logit_error, "soft_sa_abs_error": soft_error,
                    "hard_class_equal": hard_equal, "class_logits": logits,
                    "soft_sa_image_score": float(score["soft_sa_image_score"]),
                    "argmax_hard_class": int(score["argmax_hard_class"]),
                }
                append_jsonl(assessments_path, assessment)
                assessments[row["case_id"]] = assessment
    finally:
        del runtime
    stable: dict[str, dict[str, tuple[int, dict[str, Any], dict[str, Any]]]] = {
        "text_side": {}, "image_side": {},
    }
    for permutation_index in permutation:
        row = eligible[permutation_index]
        assessment = assessments[row["case_id"]]
        side, item = sa_side(row), str(row["item_id"])
        if assessment["passed"] and item not in stable[side]:
            stable[side][item] = (permutation_index, row, assessment)
    text_items = list(stable["text_side"])
    text_items.sort(key=lambda item: (item in stable["image_side"], stable["text_side"][item][0]))
    chosen_text = text_items[:TEST_PER_SIDE]
    chosen_image = sorted(
        (item for item in stable["image_side"] if item not in set(chosen_text)),
        key=lambda item: stable["image_side"][item][0],
    )[:TEST_PER_SIDE]
    chosen = (("text_side", chosen_text), ("image_side", chosen_image))
    counts: Counter[str] = Counter()
    used: set[str] = set()
    for side, items in chosen:
        for rank, item in enumerate(items, 1):
            permutation_index, row, assessment = stable[side][item]
            selected.append({
                **row, "family_id": item, "test_answer": str(row["phase0_normalized_answer"]),
                "test_side": side, "selection_rank": rank,
                "random_permutation_index": permutation_index, "eager_clean": assessment,
            })
            counts[side] += 1; used.add(item)
    if counts != Counter({"text_side": TEST_PER_SIDE, "image_side": TEST_PER_SIDE}):
        raise RuntimeError(f"Could not select 50/50 eager-stable cases: {dict(counts)}")
    selected.sort(key=record_key)
    atomic_jsonl(manifest_path, selected)
    completion = {
        "status": "complete", "case_count": len(selected), "unique_item_count": len(used),
        "counts": dict(counts), "seed": SEED,
        "capture_fingerprint": capture_config["fingerprint"],
        "manifest_fingerprint": canonical_hash(selected),
        "assessed_case_count": len(assessments),
        "rejected_case_count": sum(not row["passed"] for row in assessments.values()),
        "gate": {"logit": CLEAN_LOGIT_TOLERANCE, "soft_sa": CLEAN_SOFT_SA_TOLERANCE,
                 "hard_class_equal": True},
    }
    atomic_json(completion_path, completion)
    return selected, completion
