from __future__ import annotations

import numpy as np
import pytest

from dp_SA.analysis import bh_fdr, select_probe_candidates
from dp_SA.capture import _parse_layers as parse_capture_layers, _parse_positions as parse_capture_positions
from dp_SA.config import LAYERS, POSITIONS
from dp_SA.prompts import SA_INSTRUCTION_START, SA_PREFILL, phase0_prompt, phase1_prompt
from dp_SA.selection import select_manifests
from dp_SA.soft_score import soft_sa_from_logits

def _row(item: int, score: float, hard: int, prior: int=0) -> dict:
    return {"status":"completed","valid_class":True,"case_id":f"c{item}_{prior}","item_id":str(item),"prior_index":prior,"condition":"conflict_easy","version":"v4",
            "soft_sa_image_score":score,"argmax_hard_class":hard,"phase0_correct":True,"answer_length":4}

def test_prompts_delayed_instruction_and_exact_answer():
    p0=phase0_prompt("q","c"); assert "Source attribution classes:" not in p0
    p1=phase1_prompt("q","c","blue"); assert "**Answer**: blue\n\n" in p1
    assert p1.index("**Answer**: blue") < p1.index(SA_INSTRUCTION_START)
    assert not p1.endswith(SA_PREFILL)


def test_capture_grid_validation_preserves_historical_defaults_and_supports_new_positions():
    assert parse_capture_positions(POSITIONS)==POSITIONS
    assert parse_capture_layers(LAYERS)==LAYERS
    assert parse_capture_positions(["P1_LAT","P1_CLASS_LIST_END"])==("P1_LAT","P1_CLASS_LIST_END")
    with pytest.raises(ValueError,match="unique"): parse_capture_positions(["P1_LAT","P1_LAT"])
    with pytest.raises(ValueError,match="Unsupported"): parse_capture_positions(["UNKNOWN"])
    with pytest.raises(ValueError,match="non-negative"): parse_capture_layers([-1])

def test_soft_score_stable_and_midpoint():
    out=soft_sa_from_logits(np.asarray([0.0]*9),list(range(9)))
    assert out["probability_sum"]==pytest.approx(1.0); assert out["soft_sa_image_score"]==pytest.approx(sum((.05,.175,.325,.4375,.5,.5625,.675,.825,.95))/9)

def test_random_side_selection_is_item_disjoint_and_excludes_class4():
    rows=[]
    for item in range(180):
        hard=8 if item%2==0 else 0
        rows.append(_row(item,item/180,hard))
    rows.extend(_row(1000+i,.5,4) for i in range(20))
    construction,test,summary=select_manifests(rows,seed=42)
    assert len(construction)==50 and len(test)==100
    assert {r["test_side"] for r in test}=={"image_side","text_side"}
    assert all(r["argmax_hard_class"]!=4 for r in test)
    assert not ({r["item_id"] for r in construction}&{r["item_id"] for r in test})
    assert summary["test_selection"].startswith("seeded_random")
    _,test2,_=select_manifests(rows,seed=42); assert [r["case_id"] for r in test]==[r["case_id"] for r in test2]

def test_bh_and_bidirectional_candidate_gate():
    assert bh_fdr([.01,.02,.5])==pytest.approx([.03,.03,.5])
    rows=[]
    for item in range(100):
        for alpha in (-10,-2,0,2,10):
            rows.append({"status":"completed","item_id":str(item),"position":"P1_PANL","layer":18,"direction_type":"true","alpha":alpha,"delta_soft_sa":alpha*.01})
    selected,metrics=select_probe_candidates(rows,repeats=200,seed=1)
    assert [(r["position"],r["layer"]) for r in selected]==[("P1_PANL",18)]
    assert metrics[0]["point_direction_valid"]
