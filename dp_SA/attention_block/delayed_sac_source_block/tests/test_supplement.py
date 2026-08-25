from __future__ import annotations
import pytest
from dp_SA.attention_block.delayed_sac_source_block.run import CONDITIONS,NEW_CONDITION,REUSED_CONDITIONS,WINDOWS,ensure_config,evidence_answer_edges


def spans():return {"EVIDENCE":[1,2,3,7],"ANSWER":[8,10],"SAC":12,"PANL":10,"PANL_PLUS_1":11,"ALL_CONTENT":[0,1,2,3,7,8,9]}


def test_design_is_five_conditions_five_windows():
    assert len(CONDITIONS)==5 and len(WINDOWS)==5 and NEW_CONDITION not in REUSED_CONDITIONS


def test_evidence_answer_edges_are_exact_and_exclude_question():
    edges=set(evidence_answer_edges(spans()).pairs)
    assert edges=={(12,x) for x in (1,2,3,7,8,9)} and (12,0) not in edges


def test_resume_fingerprint(tmp_path):
    path=tmp_path/"config.json";value={"fingerprint":"a"};ensure_config(path,value,resume=False);ensure_config(path,value,resume=True)
    with pytest.raises(ValueError):ensure_config(path,{"fingerprint":"b"},resume=True)
