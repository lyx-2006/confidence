from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from .io import load_jsonl_strict, stable_seed


def stable_case_key(row: dict[str, Any]) -> tuple[Any, ...]:
    raw = str(row["item_id"])
    item = (0, int(raw)) if raw.isdigit() else (1, raw)
    return (*item, int(row.get("prior_index", 0)), str(row.get("condition", "")),
            str(row.get("version", "v4")), str(row.get("case_id", "")))


def _required_record(row: dict[str, Any]) -> str | None:
    required = (
        "case_id", "item_id", "question", "text_clue", "image_path",
        "phase0_raw_answer", "phase1_inserted_raw_answer", "phase1_prompt",
        "phase1_answer_span", "positions", "class_logits", "argmax_hard_class",
        "soft_sa_image_score",
    )
    missing = [name for name in required if row.get(name) is None]
    if missing:
        return "missing_fields:" + ",".join(missing)
    if row.get("status") != "completed" or not row.get("valid_class", True):
        return "not_completed_or_invalid_class"
    if row["phase0_raw_answer"] != row["phase1_inserted_raw_answer"]:
        return "fixed_answer_byte_parity_failed"
    logits = row.get("class_logits")
    if not isinstance(logits, list) or len(logits) != 9:
        return "invalid_class_logits"
    if not Path(str(row["image_path"])).is_file():
        return "missing_image"
    return None


def select_evaluation_manifest(rows: Sequence[dict[str, Any]], *, eval_cases: int) -> dict[str, Any]:
    if eval_cases < 2 or eval_cases % 2:
        raise ValueError("eval_cases must be a positive even number")
    per_side = eval_cases // 2
    candidates: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for side in ("image_side", "text_side"):
        side_rows = [row for row in rows if row.get("test_side") == side]
        if side == "image_side":
            side_rows.sort(key=lambda row: (-int(row.get("argmax_hard_class", -1)),
                                            -float(row.get("soft_sa_image_score", float("-inf"))),
                                            stable_case_key(row)))
        else:
            side_rows.sort(key=lambda row: (int(row.get("argmax_hard_class", 99)),
                                            float(row.get("soft_sa_image_score", float("inf"))),
                                            stable_case_key(row)))
        side_selected = 0
        for rank, row in enumerate(side_rows, 1):
            reason = _required_record(row)
            if reason is None and int(row["argmax_hard_class"]) == 4:
                reason = "class_4_excluded"
            item = str(row.get("item_id"))
            if reason is None and item in used:
                reason = "duplicate_item"
            choose = reason is None and side_selected < per_side
            if choose:
                side_selected += 1
                used.add(item)
                selected.append({**row, "evaluation_rank": side_selected})
            elif reason is None:
                reason = "beyond_side_quota"
            candidates.append({
                "case_id": row.get("case_id"), "item_id": row.get("item_id"),
                "test_side": side, "candidate_rank": rank,
                "hard_class": row.get("argmax_hard_class"),
                "soft_sa": row.get("soft_sa_image_score"),
                "selected": choose, "exclusion_reason": None if choose else reason,
                "stable_case_key": list(stable_case_key(row)),
            })
        if side_selected != per_side:
            raise ValueError(f"Insufficient eligible unique {side}: {side_selected}/{per_side}")
    if len(selected) != eval_cases or len(used) != eval_cases:
        raise AssertionError("Evaluation cohort is not balanced and item-unique")
    return {
        "format_version": 1,
        "selection_policy": "hard_extreme_then_soft_extreme_then_stable_case_key",
        "eval_cases": eval_cases,
        "per_side": per_side,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": sorted(selected, key=stable_case_key),
        "selected_item_count": len(used),
        "side_counts": dict(Counter(row["test_side"] for row in selected)),
    }


def _seeded_unique(
    rows: Iterable[dict[str, Any]], *, count: int, seed: int, pool: str,
    stratum: str, used: set[str], records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=stable_case_key)
    random.Random(stable_seed(seed, pool, stratum)).shuffle(ordered)
    output: list[dict[str, Any]] = []
    for permutation_index, row in enumerate(ordered):
        item = str(row["item_id"])
        reason = _required_record(row)
        if reason is None and item in used:
            reason = "duplicate_item_in_pool"
        choose = reason is None and len(output) < count
        if choose:
            used.add(item)
            output.append(row)
        elif reason is None:
            reason = "beyond_stratum_quota"
        records.append({
            "pool": pool, "stratum": stratum, "case_id": row.get("case_id"),
            "item_id": row.get("item_id"), "permutation_index": permutation_index,
            "selected": choose, "exclusion_reason": None if choose else reason,
        })
    if len(output) != count:
        raise ValueError(f"Insufficient {pool}/{stratum} donors: {len(output)}/{count}")
    return output


def select_calibration_manifest(
    rows: Sequence[dict[str, Any]], *, evaluation_items: set[str], seed: int,
) -> dict[str, Any]:
    eligible = [row for row in rows if str(row.get("item_id")) not in evaluation_items]
    audit: list[dict[str, Any]] = []
    pools: dict[str, list[dict[str, Any]]] = {}
    for pool in ("image", "text"):
        used: set[str] = set()
        easy = _seeded_unique((r for r in eligible if r.get("condition") == "conflict_easy"),
                              count=50, seed=seed, pool=pool, stratum="conflict_easy", used=used, records=audit)
        hard = _seeded_unique((r for r in eligible if r.get("condition") == "conflict_hard"),
                              count=50, seed=seed, pool=pool, stratum="conflict_hard", used=used, records=audit)
        pools[pool] = [*easy, *hard]
    answer_used: set[str] = set()
    answer_image = _seeded_unique((r for r in eligible if int(r.get("argmax_hard_class", 4)) in {5, 6, 7, 8}),
                                  count=50, seed=seed, pool="answer", stratum="image_side", used=answer_used, records=audit)
    answer_text = _seeded_unique((r for r in eligible if int(r.get("argmax_hard_class", 4)) in {0, 1, 2, 3}),
                                 count=50, seed=seed, pool="answer", stratum="text_side", used=answer_used, records=audit)
    pools["answer"] = [*answer_image, *answer_text]
    serialized = {
        name: [{key: row.get(key) for key in (
            "case_id", "item_id", "prior_index", "condition", "version", "question",
            "text_clue", "image_path", "image_sha256", "phase0_raw_answer",
            "phase1_inserted_raw_answer",
            "phase0_answer_fingerprint", "phase1_prompt", "phase1_prompt_hash",
            "phase1_answer_span", "phase1_answer_token_ids", "positions",
            "argmax_hard_class", "soft_sa_image_score",
        )} for row in group]
        for name, group in pools.items()
    }
    for name, group in serialized.items():
        items = {str(row["item_id"]) for row in group}
        if len(group) != 100 or len(items) != 100 or items & evaluation_items:
            raise AssertionError(f"Invalid calibration pool {name}")
    return {
        "format_version": 1, "seed": seed,
        "selection_policy": "stable_sort_then_stratum_seeded_shuffle_unique_item",
        "evaluation_item_overlap": False, "pools": serialized, "candidate_audit": audit,
    }


def load_and_select(capture_dir: Path, *, eval_cases: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = load_jsonl_strict(capture_dir.parent / "steering" / "test_manifest.jsonl")
    capture = load_jsonl_strict(capture_dir / "results.jsonl")
    evaluation = select_evaluation_manifest(frozen, eval_cases=eval_cases)
    items = {str(row["item_id"]) for row in evaluation["selected"]}
    calibration = select_calibration_manifest(capture, evaluation_items=items, seed=seed)
    return evaluation, calibration
