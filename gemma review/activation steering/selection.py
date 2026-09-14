from __future__ import annotations

import random
from collections import Counter
from typing import Any, Sequence

from experiment_config import CONSTRUCTION_PER_SIDE, SEED, TEST_PER_SIDE


def record_key(row: dict[str, Any]) -> tuple[Any, ...]:
    item = str(row["item_id"])
    item_key = (0, int(item)) if item.isdigit() else (1, item)
    return (*item_key, int(row["prior_index"]), str(row["condition"]), str(row.get("version", "v4")))


def _take_extreme(rows: Sequence[dict[str, Any]], count: int, reverse: bool, used: set[str]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: ((-1 if reverse else 1) * float(row["soft_sa_image_score"]), record_key(row)),
    )
    selected = []
    for row in ordered:
        item = str(row["item_id"])
        if item in used:
            continue
        selected.append(dict(row))
        used.add(item)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Could not select {count} item-disjoint extreme records; found {len(selected)}")
    return selected


def select_manifests(
    rows: Sequence[dict[str, Any]],
    construction_per_side: int = CONSTRUCTION_PER_SIDE,
    test_per_side: int = TEST_PER_SIDE,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eligible = [row for row in rows if row.get("status") == "completed" and row.get("valid_class", True)]
    used: set[str] = set()
    high = _take_extreme(eligible, construction_per_side, True, used)
    low = _take_extreme(eligible, construction_per_side, False, used)
    construction = [
        {**row, "construction_side": side, "selection_rank": rank}
        for side, group in (("high_image", high), ("high_text", low))
        for rank, row in enumerate(group, 1)
    ]
    candidates = []
    for row in sorted(eligible, key=record_key):
        if str(row["item_id"]) in used:
            continue
        hard = int(row["argmax_hard_class"])
        side = "image_side" if hard in (5, 6, 7, 8) else "text_side" if hard in (0, 1, 2, 3) else None
        if side:
            candidates.append({**row, "test_side": side})
    random.Random(seed).shuffle(candidates)
    counts: Counter[str] = Counter()
    selected = []
    test_items: set[str] = set()
    for permutation_index, row in enumerate(candidates):
        side, item = row["test_side"], str(row["item_id"])
        if counts[side] >= test_per_side or item in test_items:
            continue
        selected.append({**row, "random_permutation_index": permutation_index, "selection_rank": counts[side] + 1})
        counts[side] += 1
        test_items.add(item)
        if counts["image_side"] == test_per_side and counts["text_side"] == test_per_side:
            break
    if counts["image_side"] != test_per_side or counts["text_side"] != test_per_side:
        available = {
            side: len({str(row["item_id"]) for row in candidates if row["test_side"] == side})
            for side in ("image_side", "text_side")
        }
        raise ValueError(f"Insufficient item-disjoint Gemma test records: selected={dict(counts)}, available={available}")
    return construction, sorted(selected, key=record_key), {
        "seed": seed,
        "construction_counts": dict(Counter(row["construction_side"] for row in construction)),
        "test_counts": dict(counts),
        "construction_item_count": len(used),
        "test_item_count": len(test_items),
        "test_selection": "seeded_random_permutation_without_soft_score_sorting",
    }


def select_centered_manifests(
    rows: Sequence[dict[str, Any]],
    construction_per_side: int = CONSTRUCTION_PER_SIDE,
    test_count: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Construct vectors from extremes and test on item-disjoint SA≈0.5 cases."""
    eligible = [row for row in rows if row.get("status") == "completed" and row.get("valid_class", True)]
    used: set[str] = set()
    high_image = _take_extreme(eligible, construction_per_side, True, used)
    high_text = _take_extreme(eligible, construction_per_side, False, used)
    construction = [
        {**row, "construction_side": side, "selection_rank": rank}
        for side, group in (("high_image", high_image), ("high_text", high_text))
        for rank, row in enumerate(group, 1)
    ]
    best_by_item: dict[str, dict[str, Any]] = {}
    for row in eligible:
        item = str(row["item_id"])
        if item in used:
            continue
        distance = abs(float(row["soft_sa_image_score"]) - 0.5)
        previous = best_by_item.get(item)
        candidate_key = (distance, record_key(row))
        if previous is None or candidate_key < (previous["center_distance"], record_key(previous)):
            best_by_item[item] = {**row, "center_distance": distance}
    ordered = sorted(best_by_item.values(), key=lambda row: (float(row["center_distance"]), record_key(row)))
    if len(ordered) < test_count:
        raise ValueError(f"Insufficient item-disjoint center test records: {len(ordered)} < {test_count}")
    test = [
        {**row, "test_side": "center", "center_rank": rank}
        for rank, row in enumerate(ordered[:test_count], 1)
    ]
    scores = [float(row["soft_sa_image_score"]) for row in test]
    return construction, test, {
        "selection": "item_disjoint_nearest_to_soft_sa_0.5",
        "target_soft_sa": 0.5,
        "construction_counts": dict(Counter(row["construction_side"] for row in construction)),
        "construction_item_count": len(used),
        "test_count": len(test),
        "test_item_count": len({str(row["item_id"]) for row in test}),
        "test_soft_sa_min": min(scores),
        "test_soft_sa_mean": sum(scores) / len(scores),
        "test_soft_sa_max": max(scores),
        "test_soft_sa_max_distance": max(abs(score - 0.5) for score in scores),
    }
