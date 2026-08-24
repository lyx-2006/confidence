from __future__ import annotations

import pytest

from dp_SA.attention_block.config import COARSE_WINDOWS, GLOBAL_CONDITIONS, MATCHED_PAIRS, REFINE_WINDOWS, WINDOW_CONDITIONS, parse_windows
from dp_SA.attention_block.run import _atomic_config, _phase_grid
from dp_SA.attention_block.sources import select_joint_manifest
from dp_SA.attention_block.spans import edges_for_condition


def spans():
    return {"EVIDENCE":[1,2,4],"ANSWER":[5,7],"ALL_CONTENT":[0,1,2,4,5,6],"PANL":7,"PANL_PLUS_1":8,"SAC":12,
            "ALL_DOWNSTREAM_OF_PANL":list(range(8,13))}


def test_registered_windows_and_conditions():
    assert COARSE_WINDOWS == ((0,11),(4,15),(8,19),(12,23),(16,27))
    assert REFINE_WINDOWS == tuple((x,x+5) for x in range(0,23,2))
    assert len(REFINE_WINDOWS)==12 and all(end-start+1==6 for start,end in REFINE_WINDOWS)
    assert len(WINDOW_CONDITIONS)==9 and len(GLOBAL_CONDITIONS)==8 and len(MATCHED_PAIRS)==3


def test_window_cli_and_refine_pair_dimension():
    assert parse_windows("0-11,4-15") == ((0,11),(4,15))
    with pytest.raises(ValueError): parse_windows("0-28")
    selected={"joint":["panl_cache","jit_all_content"],"delayed":[]}
    grid=_phase_grid("refine",selected,"joint",False)
    assert len(grid)==48
    assert sum(condition=="sac_to_panl_plus_1" for condition, *_ in grid)==24
    assert {pair for *_rest,pair in grid}=={"panl_cache","jit_all_content"}


def test_resume_config_rejects_fingerprint_change(tmp_path):
    path=tmp_path/"run_config.json"
    _atomic_config(path,{"coarse_windows":COARSE_WINDOWS},resume=False)
    _atomic_config(path,{"coarse_windows":COARSE_WINDOWS},resume=True)
    with pytest.raises(ValueError):
        _atomic_config(path,{"coarse_windows":((0,11),)},resume=True)


def test_evidence_answer_union_and_adjacent_control():
    value=spans()
    assert set(edges_for_condition(value,"panl_to_evidence_answer").pairs)=={(7,x) for x in [1,2,4,5,6]}
    assert set(edges_for_condition(value,"panl_plus_1_to_evidence_answer").pairs)=={(8,x) for x in [1,2,4,5,6]}


def test_all_later_and_keep_panl_semantics():
    value=spans(); full=set(edges_for_condition(value,"all_later_to_answer").pairs); keep=set(edges_for_condition(value,"all_later_to_answer_keep_panl").pairs)
    assert (7,5) in full and (7,6) in full and (8,5) in full
    assert (7,5) not in keep and (7,6) not in keep and (8,5) in keep


def test_joint_selection_balanced_unique_and_excludes_class4():
    rows=[]
    for item in range(1,31):
        rows.append({"status":"completed","item_id":str(item),"prior_index":0,"condition":"conflict_easy","version":"v4",
                     "case_id":str(item),"argmax_hard_class":6 if item<=15 else 2})
        rows.append({**rows[-1],"prior_index":1,"case_id":f"{item}b","argmax_hard_class":4})
    selected=select_joint_manifest(rows,per_side=10,seed=42)
    assert len(selected)==20 and len({r["item_id"] for r in selected})==20
    assert sum(r["test_side"]=="image_side" for r in selected)==10
    assert sum(r["test_side"]=="text_side" for r in selected)==10
    assert all(r["argmax_hard_class"]!=4 for r in selected)
