from __future__ import annotations

import random
from collections import Counter
from typing import Any, Sequence

from .config import CONSTRUCTION_PER_SIDE, SEED, TEST_PER_SIDE

def record_key(row: dict[str, Any]) -> tuple[Any, ...]:
    item = str(row["item_id"])
    item_key = (0, int(item)) if item.isdigit() else (1, item)
    return (*item_key, int(row["prior_index"]), str(row["condition"]), str(row.get("version", "v4")))

def _take_extreme(rows: Sequence[dict[str, Any]], count: int, reverse: bool, used: set[str]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: ((-1 if reverse else 1) * float(r["soft_sa_image_score"]), record_key(r)))
    selected=[]
    for row in ordered:
        item=str(row["item_id"])
        if item in used: continue
        selected.append(dict(row)); used.add(item)
        if len(selected)==count: break
    if len(selected)!=count: raise ValueError(f"Could not select {count} item-disjoint extreme records; found {len(selected)}")
    return selected

def select_manifests(rows: Sequence[dict[str, Any]], *, construction_per_side: int = CONSTRUCTION_PER_SIDE,
                     test_per_side: int = TEST_PER_SIDE, seed: int = SEED) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eligible=[r for r in rows if r.get("status")=="completed" and r.get("valid_class", True)]
    used:set[str]=set()
    high=_take_extreme(eligible,construction_per_side,True,used)
    low=_take_extreme(eligible,construction_per_side,False,used)
    construction=[]
    for side, group in (("high_image",high),("high_text",low)):
        for rank,row in enumerate(group,1): construction.append({**row,"construction_side":side,"selection_rank":rank})
    candidates=[]
    for row in sorted(eligible,key=record_key):
        if str(row["item_id"]) in used: continue
        hard=int(row["argmax_hard_class"])
        side="image_side" if hard in (5,6,7,8) else "text_side" if hard in (0,1,2,3) else None
        if side: candidates.append({**row,"test_side":side})
    random.Random(seed).shuffle(candidates)
    counts=Counter(); selected=[]; test_items=set()
    for permutation_index,row in enumerate(candidates):
        side=row["test_side"]; item=str(row["item_id"])
        if counts[side]>=test_per_side or item in test_items: continue
        selected.append({**row,"random_permutation_index":permutation_index,"selection_rank":counts[side]+1})
        counts[side]+=1; test_items.add(item)
        if counts["image_side"]==test_per_side and counts["text_side"]==test_per_side: break
    if counts["image_side"]!=test_per_side or counts["text_side"]!=test_per_side:
        unique_by_side={side:len({str(r["item_id"]) for r in candidates if r["test_side"]==side}) for side in ("image_side","text_side")}
        raise ValueError(f"Insufficient item-disjoint test records: selected={dict(counts)}, candidates={unique_by_side}")
    construction_items={str(r["item_id"]) for r in construction}
    if construction_items & test_items or len(construction_items)!=2*construction_per_side or len(test_items)!=2*test_per_side:
        raise AssertionError("Construction/test item leakage or duplicate items")
    def distributions(records: Sequence[dict[str,Any]], group_field: str) -> dict[str,Any]:
        output={}
        for group in sorted({str(r[group_field]) for r in records}):
            values=[r for r in records if str(r[group_field])==group]
            output[group]={"condition_counts":dict(Counter(str(r["condition"]) for r in values)),
                           "argmax_class_counts":dict(Counter(str(r["argmax_hard_class"]) for r in values)),
                           "correct_count":sum(bool(r.get("phase0_correct")) for r in values),
                           "answer_length_mean":sum(int(r.get("answer_length",0)) for r in values)/len(values),
                           "soft_sa_min":min(float(r["soft_sa_image_score"]) for r in values),"soft_sa_mean":sum(float(r["soft_sa_image_score"]) for r in values)/len(values),
                           "soft_sa_max":max(float(r["soft_sa_image_score"]) for r in values)}
        return output
    summary={"seed":seed,"construction_counts":dict(Counter(r["construction_side"] for r in construction)),
             "test_counts":dict(counts),"construction_item_count":len(construction_items),"test_item_count":len(test_items),
             "test_selection":"seeded_random_permutation_without_soft_score_sorting",
             "test_class_counts":dict(Counter(str(r["argmax_hard_class"]) for r in selected)),
             "construction_distributions":distributions(construction,"construction_side"),"test_distributions":distributions(selected,"test_side")}
    return construction,sorted(selected,key=record_key),summary
