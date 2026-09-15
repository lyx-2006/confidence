from __future__ import annotations

import random
from collections import Counter
from typing import Any, Sequence

from dp_SA.selection import record_key

from .config import SEED, TEST_PER_SIDE


def sa_side(row: dict[str, Any]) -> str | None:
    hard = int(row["argmax_hard_class"])
    if hard in (0, 1, 2, 3):
        return "text_side"
    if hard in (5, 6, 7, 8):
        return "image_side"
    return None


def select_manifest(
    rows: Sequence[dict[str, Any]], *, per_side: int = TEST_PER_SIDE, seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [
        row for row in rows
        if row.get("status") == "completed" and row.get("valid_class", True) and sa_side(row)
    ]
    eligible.sort(key=record_key)
    permutation = list(range(len(eligible)))
    random.Random(seed).shuffle(permutation)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    counts: Counter[str] = Counter()
    # Reserve the scarcer text-side pool before filling image-side.
    for side in ("text_side", "image_side"):
        for permutation_index in permutation:
            row = eligible[permutation_index]
            item = str(row["item_id"])
            if sa_side(row) != side or item in used:
                continue
            counts[side] += 1
            used.add(item)
            selected.append({
                **row,
                "family_id": item,
                "test_answer": str(row["phase0_normalized_answer"]),
                "test_side": side,
                "selection_rank": counts[side],
                "random_permutation_index": permutation_index,
            })
            if counts[side] == per_side:
                break
    if counts != Counter({"image_side": per_side, "text_side": per_side}):
        available = {side: len({str(row["item_id"]) for row in eligible if sa_side(row) == side})
                     for side in ("image_side", "text_side")}
        raise ValueError(f"Insufficient item-disjoint cases: selected={dict(counts)}, available={available}")
    if len(selected) != 2 * per_side or len(used) != len(selected):
        raise AssertionError("Attention-block manifest contains duplicate items")
    selected.sort(key=record_key)
    summary = {
        "seed": seed,
        "counts": dict(Counter(row["test_side"] for row in selected)),
        "case_count": len(selected),
        "unique_item_count": len(used),
        "answer_counts": dict(Counter(row["test_answer"] for row in selected)),
        "condition_counts": dict(Counter(row["condition"] for row in selected)),
        "class_counts": dict(Counter(str(row["argmax_hard_class"]) for row in selected)),
    }
    return selected, summary

