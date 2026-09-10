from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from dp_SA.real_sa.random_corruption_validation.analysis import build_case_metrics
from dp_SA.real_sa.random_corruption_validation.inputs import apply_text_tokens
from dp_SA.real_sa.random_corruption_validation.io_utils import load_jsonl, upsert
from dp_SA.real_sa.random_corruption_validation.randomization import (
    TokenPool, build_token_pool, ensure_gaussian_image, gaussian_png_bytes, sample_text_tokens, stable_seed,
)
from dp_SA.real_sa.random_corruption_validation.run import _validate_complete
from dp_SA.real_sa.random_corruption_validation.scoring import score_condition


class Batch(dict):
    __getattr__=dict.__getitem__


class TinyTokenizer:
    all_special_ids=[99]
    def __init__(self): self.vocab={"alpha":1," beta":2,"red":3,"<|x|>":4," ":5,"rare":6,"Question":7}
    def get_vocab(self): return self.vocab
    def encode(self,text,add_special_tokens=False):
        if text in self.vocab: return [self.vocab[text]]
        result=[]
        for token,value in self.vocab.items():
            if token.strip() and token.strip() in text: result.append(value)
        return result
    def decode(self,ids,**kwargs):
        reverse={value:key for key,value in self.vocab.items()}; return "".join(reverse.get(int(value),"?") for value in ids)


def test_stable_seed_is_canonical_sha256():
    result=stable_seed("case",3,"text")
    raw=json.dumps([42,"case",3,"text"],ensure_ascii=False,sort_keys=True,separators=(",",":"))
    assert result["sha256"]==hashlib.sha256(raw.encode()).hexdigest()
    assert result==stable_seed("case",3,"text") and result!=stable_seed("case",4,"text")


def test_gaussian_png_is_deterministic_and_pixel_independent(tmp_path):
    a=tmp_path/"a.png"; b=tmp_path/"b.png"
    Image.new("RGB",(12,9),(0,0,0)).save(a); Image.new("RGB",(12,9),(255,0,2)).save(b)
    one=ensure_gaussian_image(a,tmp_path/"one.png","case",0); two=ensure_gaussian_image(b,tmp_path/"two.png","case",0)
    assert one["sha256"]==two["sha256"]
    assert (tmp_path/"one.png").read_bytes()==(tmp_path/"two.png").read_bytes()
    payload,pixels=gaussian_png_bytes((12,9),one["seed"]["uint64"])
    assert hashlib.sha256(payload).hexdigest()==one["sha256"] and pixels.shape==(9,12,3)


def test_token_pool_exclusions_frequency_and_case_specific_sampling():
    tokenizer=TinyTokenizer(); answers=["red"]*12
    train=[{"text_clue":"alpha beta rare","answer_classes":answers} for _ in range(4)]
    train += [{"text_clue":"alpha beta","answer_classes":answers} for _ in range(2)]
    pool=build_token_pool(tokenizer,train)
    assert pool.token_ids.tolist()==[1,2]
    assert pool.probabilities.sum()==pytest.approx(1) and pool.probabilities[0]==pytest.approx(pool.probabilities[1])
    sampled,audit=sample_text_tokens(pool,tokenizer,[1,1,1],"case",0)
    assert sampled==[2,2,2] and audit["available_token_count"]==1


def test_text_replacement_preserves_every_non_target_input():
    inputs=Batch(input_ids=torch.tensor([[10,11,12,13]]),attention_mask=torch.ones(1,4,dtype=torch.long),pixel_values=torch.tensor([7]))
    changed,audit=apply_text_tokens(inputs,{"text_positions":[1,2]},[20,21],frozenset({99}))
    assert changed["input_ids"].tolist()==[[10,20,21,13]]
    assert torch.equal(changed["attention_mask"],inputs.attention_mask) and torch.equal(changed["pixel_values"],inputs.pixel_values)
    assert audit["outside_input_ids_equal"] and audit["attention_mask_equal"]
    with pytest.raises(ValueError,match="banned"): apply_text_tokens(inputs,{"text_positions":[1,2]},[20,99],frozenset({99}))


def _source(case="c"):
    return {"case_id":case,"family_id":"f","item_id":"1","condition":"conflict_easy","answer_side":"follow_image",
            "phase0_normalized_answer":"red","soft_sa_image_score":.75}


def test_replicate_then_condition_aggregation_uses_shared_clean():
    source=_source(); scores=[{"case_id":"c","replicate":None,"corruption_condition":"clean","fixed_answer_probability":.8}]
    for rep in range(5):
        scores.extend([{"case_id":"c","replicate":rep,"corruption_condition":"10_random_text","fixed_answer_probability":.6+rep*.01},
                       {"case_id":"c","replicate":rep,"corruption_condition":"01_gaussian_image","fixed_answer_probability":.5},
                       {"case_id":"c","replicate":rep,"corruption_condition":"00_both_random","fixed_answer_probability":.2}])
    replicate,aggregate=build_case_metrics(scores,[source])
    assert len(replicate)==5 and aggregate[0]["v10"]==pytest.approx(.62)
    assert aggregate[0]["signed_verbal_sa"]==pytest.approx(.5)
    assert all(row["efficiency_error"]<=1e-10 for row in [*replicate,*aggregate])


def test_jsonl_upsert_rejects_duplicate_and_preserves_schema(tmp_path):
    path=tmp_path/"rows.jsonl"; rows=upsert(path,[],{"score_key":"a","value":1},"score_key")
    assert load_jsonl(path)==rows
    with pytest.raises(ValueError,match="Duplicate"): upsert(path,rows,{"score_key":"a","value":2},"score_key")


def test_multitoken_teacher_forcing_reuses_exact_corruption(monkeypatch):
    import dp_SA.real_sa.random_corruption_validation.scoring as scoring
    candidates=[f"c{i}" for i in range(12)]; candidate_ids={name:[20+i,40+i] for i,name in enumerate(candidates)}
    calls=[]
    def prepare(_processor,_row,**kwargs):
        suffix=kwargs.get("rendered_suffix",""); ids=[1,2,3,4]+(candidate_ids[suffix] if suffix else [])
        calls.append((suffix,list(kwargs["replacement_ids"])))
        return "rendered",Batch(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long)),{"text_positions":[1,2],"image_positions":[0]}, {"replacement_text_token_ids":list(kwargs["replacement_ids"])}
    def logits(_model,_inputs,positions):
        return {position:torch.arange(80,dtype=torch.float32) for position in positions}
    monkeypatch.setattr(scoring,"prepare_condition_inputs",prepare); monkeypatch.setattr(scoring,"model_input_device",lambda _:torch.device("cpu")); monkeypatch.setattr(scoring,"run_logits_forward",logits)
    row={**_source(),"answer_classes":candidates}
    values,_details,audit,count=score_condition(SimpleNamespace(processor=object(),model=object()),row,candidate_ids,image_path=None,replacement_ids=[8,9],banned_ids=frozenset())
    assert count==12 and len(values)==12 and len(audit["candidate_forwards"])==12
    assert len(calls)==13 and all(replacement==[8,9] for _,replacement in calls)


def test_new_resume_validation_requires_complete_paired_triplet():
    row={"case_id":"c"}; fingerprint="fp"
    def score(condition):
        text={"replacement_text_token_ids":[8,9]} if condition!="01_gaussian_image" else {}
        return {"run_fingerprint":fingerprint,"text_corruption_audit":text,"input_details":{"image_sha256":"noise" if condition!="10_random_text" else "original"}}
    existing={f"c|r0|{condition}":score(condition) for condition in ("10_random_text","01_gaussian_image","00_both_random")}
    assert _validate_complete(existing,row,0,fingerprint)
    assert not _validate_complete({},row,0,fingerprint)
