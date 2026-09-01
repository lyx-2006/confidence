from __future__ import annotations
from types import SimpleNamespace
import torch
import pytest
from dp_SA.positions import P1_CLASS_LIST_END_ANCHOR, locate_phase1_positions
from dp_SA.prompts import SA_PREFILL, phase1_prompt

class CharTokenizer:
    def encode(self,text,add_special_tokens=False): return [ord(x) for x in text]
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        out={"input_ids":self.encode(text)}
        if return_offsets_mapping: out["offset_mapping"]=[(i,i+1) for i in range(len(text))]
        return out
    def decode(self,ids,**kwargs): return "".join(chr(x) for x in ids)
    def convert_tokens_to_ids(self,token): return -999

def test_all_positions_and_causal_order():
    tokenizer=CharTokenizer(); user=phase1_prompt("q","c","blue"); rendered=user+"\nassistant\n"+SA_PREFILL
    ids=tokenizer.encode(rendered); inputs=SimpleNamespace(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long))
    p=locate_phase1_positions(tokenizer,rendered,inputs,"blue")
    assert p["P1_LAT"]["processed_index"] < p["P1_PANL"]["processed_index"] < p["SA_INSTRUCTION_START"]["processed_index"] < p["P1_CLASS_LIST_END"]["processed_index"] < p["P1_SAC"]["processed_index"]
    assert p["P1_PANL"]["token_text"]=="\n"
    class_end=p["P1_CLASS_LIST_END"]
    assert class_end["anchor_occurrence_count"]==1
    assert class_end["rendered_index"]==rendered.index(P1_CLASS_LIST_END_ANCHOR)+len(P1_CLASS_LIST_END_ANCHOR)
    assert "\n" in class_end["token_text"]

def test_fixed_answer_locator_uses_bounded_last_field_when_clue_duplicates_it():
    tokenizer=CharTokenizer(); user=phase1_prompt("q","clue says **Answer**: white","white"); rendered=user+"\nassistant\n"+SA_PREFILL
    ids=tokenizer.encode(rendered); inputs=SimpleNamespace(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long))
    p=locate_phase1_positions(tokenizer,rendered,inputs,"white")
    assert p["P1_PANL"]["processed_index"] > rendered.index("clue says **Answer**: white")

@pytest.mark.parametrize(("rendered_transform","message"),[
    (lambda value:value.replace(P1_CLASS_LIST_END_ANCHOR,"missing"),"found 0"),
    (lambda value:value.replace(P1_CLASS_LIST_END_ANCHOR+"\n",P1_CLASS_LIST_END_ANCHOR+"x",1),"not immediately followed"),
])
def test_class_list_anchor_missing_or_not_followed_by_newline_fails(rendered_transform,message):
    tokenizer=CharTokenizer(); rendered=phase1_prompt("q","c","blue")+"\nassistant\n"+SA_PREFILL
    rendered=rendered_transform(rendered); ids=tokenizer.encode(rendered)
    inputs=SimpleNamespace(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long))
    with pytest.raises(ValueError,match=message): locate_phase1_positions(tokenizer,rendered,inputs,"blue")

def test_duplicate_class_list_anchor_fails():
    tokenizer=CharTokenizer(); rendered=phase1_prompt("q",P1_CLASS_LIST_END_ANCHOR,"blue")+"\nassistant\n"+SA_PREFILL
    ids=tokenizer.encode(rendered); inputs=SimpleNamespace(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long))
    with pytest.raises(ValueError,match="found 2"): locate_phase1_positions(tokenizer,rendered,inputs,"blue")
