from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from dp_SA.config import MODEL_PATH
from dp_SA.prompt_check.positions import locate_template_positions
from dp_SA.prompt_check.scoring import numeric_score, parse_t3_greedy, t3_score
from dp_SA.prompt_check.stats import hard_agreement, retention_eligibility, shared_family_draws
from dp_SA.prompt_check.io_utils import atomic_json, atomic_jsonl, atomic_npz, load_jsonl
from dp_SA.prompt_check.sampling import validation_audit, validation_geometry_cells, validation_test
from dp_SA.prompt_check.templates import TEMPLATES
from dp_SA.prompts import PHASE1_TEMPLATE


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)


def test_t0_is_direct_reference():
    assert TEMPLATES["T0"].template is PHASE1_TEMPLATE


@pytest.mark.parametrize("name", ["T0", "T1", "T2", "T3"])
def test_dynamic_semantic_positions(tokenizer, name):
    spec=TEMPLATES[name];prompt=spec.render(question="What color?",text_clue="A clue containing **Answer**: fake.",answer="light blue")
    rendered="<|im_start|>user\n"+prompt+"<|im_end|>\n<|im_start|>assistant\n**Source Attribution**:"
    ids=tokenizer.encode(rendered,add_special_tokens=False);inputs={"input_ids":torch.tensor([ids]),"attention_mask":torch.ones((1,len(ids)),dtype=torch.long)}
    located=locate_template_positions(tokenizer,rendered,inputs,"light blue",spec)
    assert located["P1_LAT"]["processed_index"]==located["phase1_answer_span"][1]-1
    assert located["P1_LAT"]["processed_index"]<located["P1_PANL"]["processed_index"]<located["P1_CLASS_LIST_END"]["processed_index"]<located["P1_SAC"]["processed_index"]
    assert located["P1_CLASS_LIST_END"]["anchor_text"]==spec.last_class_description


def test_t2_canonical_mapping_is_reversed():
    score=numeric_score([9,8,7,6,5,4,3,2,1],reversed_scale=True)
    assert score["raw_hard_class"]==0 and score["canonical_hard_class"]==8
    reverse=numeric_score([1,2,3,4,5,6,7,8,9],reversed_scale=True)
    assert reverse["raw_hard_class"]==8 and reverse["canonical_hard_class"]==0
    assert score["canonical_soft_sa"]>reverse["canonical_soft_sa"]


def test_t3_total_and_length_normalized_are_both_preserved(tokenizer):
    labels=["STRONG_TEXT","SLIGHT_TEXT","BALANCED","SLIGHT_IMAGE","STRONG_IMAGE"]
    ids=[tokenizer.encode(x,add_special_tokens=False) for x in labels]
    assert [len(x) for x in ids]==[3,3,2,3,3]
    result=t3_score([-3,-4,-2.5,-4,-5],ids)
    assert result["canonical_hard_label"]=="BALANCED"
    assert len(result["t3_candidates"])==5
    assert np.isclose(sum(x["probability"] for x in result["t3_candidates"]),1)
    assert "length_normalized_soft_sa" in result


def test_t3_greedy_parser_is_strict():
    assert parse_t3_greedy("BALANCED")["greedy_parse_status"]=="valid"
    assert parse_t3_greedy(" BALANCED")["greedy_parse_status"]=="invalid"
    assert parse_t3_greedy("BALANCED\n")["greedy_parse_status"]=="invalid"


def test_bootstrap_draws_are_deterministic_and_global():
    ordered,a=shared_family_draws(["b","a","c","a"],20,42);ordered2,b=shared_family_draws(["c","b","a"],20,42)
    assert ordered==ordered2==["a","b","c"] and np.array_equal(a,b)


def test_retention_requires_stable_nonzero_denominator():
    assert retention_eligibility(.2,np.linspace(.1,.3,2000))["eligible"]
    assert not retention_eligibility(.01,np.linspace(-.1,.1,2000))["eligible"]
    assert not retention_eligibility(.2,np.r_[np.ones(1900)*.2,np.ones(100)*-.2])["eligible"]


def test_quadratic_hard_agreement():
    result=hard_agreement([0,1,2,3],[0,2,2,3],within_one=True)
    assert result["exact_agreement"]==.75 and result["within_one_agreement"]==1
    assert result["quadratic_weighted_kappa"]>0


def test_atomic_outputs_replace_complete_files(tmp_path):
    path=tmp_path/"nested/value.json";atomic_json(path,{"version":1});atomic_json(path,{"version":2})
    assert __import__("json").loads(path.read_text())=={"version":2}
    rows=tmp_path/"rows.jsonl";atomic_jsonl(rows,[{"x":1},{"x":2}]);assert load_jsonl(rows)==[{"x":1},{"x":2}]
    arrays=tmp_path/"a.npz";atomic_npz(arrays,{"v":np.asarray([1,2,3])})
    with np.load(arrays) as payload:assert payload["v"].tolist()==[1,2,3]


def test_validation_100_sampling_is_deterministic_and_covered():
    audit=validation_audit(100);test=validation_test(100);cells=validation_geometry_cells(100)
    assert len(audit)==len({r["case_id"] for r in audit})==100
    assert len(test)==len({r["case_id"] for r in test})==100
    assert {r["test_status"] for r in test}=={"confirmatory","exploratory_sparse"}
    case_ids={str(case) for cell in cells for case in cell["case_ids"]}
    assert 50<=len(case_ids)<=100
    answers={r["answer"] for r in cells}
    assert all({r["sa_side"] for r in cells if r["answer"]==answer}=={"high_text","high_image"} for answer in answers)
