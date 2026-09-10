from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch

from layer_metacognition.model_adapter import model_input_device, run_logits_forward

from dp_SA.real_sa.metrics import restricted_probabilities
from dp_SA.real_sa.scoring import scores_from_vocab, teacher_forced_log_probability
from dp_SA.unimodal_logit_confidence.score_unimodal import candidate_suffix_ids

from .config import REPLICATES, TEMPERATURE
from .inputs import prepare_condition_inputs


def score_condition(inference: Any, row: dict[str, Any], candidate_ids: dict[str,list[int]], *,
                    image_path: Path | None, replacement_ids: Sequence[int] | None,
                    banned_ids: frozenset[int]) -> tuple[list[float],dict[str,Any],dict[str,Any],int]:
    device=model_input_device(inference)
    candidates=list(row["answer_classes"])
    single=all(len(candidate_ids[name])==1 for name in candidates)
    if single:
        _rendered,inputs,details,text_audit=prepare_condition_inputs(inference.processor,row,device=device,image_path=image_path,
                                                                     replacement_ids=replacement_ids,banned_ids=banned_ids)
        position=int(inputs.input_ids.shape[1])-1
        logits=run_logits_forward(inference.model,inputs,[position])[position]
        return scores_from_vocab(logits,candidates,candidate_ids),details,text_audit,1

    base_rendered,base_inputs,_details,_audit=prepare_condition_inputs(inference.processor,row,device=device,image_path=image_path,
                                                                       replacement_ids=replacement_ids,banned_ids=banned_ids)
    base_length=int(base_inputs.input_ids.shape[1]); totals=[]; audits=[]; first_details=None
    for candidate in candidates:
        _rendered,inputs,details,text_audit=prepare_condition_inputs(inference.processor,row,device=device,image_path=image_path,
                                                                     replacement_ids=replacement_ids,banned_ids=banned_ids,
                                                                     rendered_suffix=candidate)
        suffix=[int(value) for value in inputs.input_ids[0,base_length:].tolist()]
        if suffix!=candidate_ids[candidate]: raise ValueError(f"Candidate suffix mismatch: {row['case_id']} {candidate}")
        positions=list(range(base_length-1,int(inputs.input_ids.shape[1])-1))
        logits=run_logits_forward(inference.model,inputs,positions)
        totals.append(teacher_forced_log_probability(logits,positions,suffix))
        audits.append(text_audit); first_details=first_details or details
    if replacement_ids is not None and any(audit.get("replacement_text_token_ids")!=list(replacement_ids) for audit in audits):
        raise ValueError("Teacher-forcing forwards did not share the same text corruption")
    return totals,first_details,{"candidate_forwards":audits},len(candidates)


def result_row(row:dict[str,Any],condition:str,replicate:int|None,scores:Sequence[float],candidate_ids:dict[str,list[int]],
               details:dict[str,Any],text_audit:dict[str,Any],run_fingerprint:str,forward_count:int,
               randomization:dict[str,Any]|None=None)->dict[str,Any]:
    candidates=list(row["answer_classes"]); probabilities=restricted_probabilities(scores,TEMPERATURE)
    fixed=str(row["phase0_normalized_answer"]); index=candidates.index(fixed)
    order=sorted(range(len(candidates)),key=lambda i:(-float(scores[i]),i)); rank=order.index(index)+1
    key=f"{row['case_id']}|clean" if replicate is None else f"{row['case_id']}|r{replicate}|{condition}"
    return {"score_key":key,"run_fingerprint":run_fingerprint,"case_id":row["case_id"],"family_id":row["family_id"],
            "item_id":str(row["item_id"]),"dataset_condition":row["condition"],"answer_side":row["answer_side"],
            "replicate":replicate,"corruption_condition":condition,"fixed_answer":fixed,"candidate_order":candidates,
            "candidate_token_ids":candidate_ids,"candidate_scores":{name:float(scores[i]) for i,name in enumerate(candidates)},
            "restricted_probabilities":{name:float(probabilities[i]) for i,name in enumerate(candidates)},
            "probability_sum":float(probabilities.sum()),"fixed_answer_probability":float(probabilities[index]),
            "fixed_answer_log_probability":float(math.log(max(float(probabilities[index]),torch.finfo(torch.float64).tiny))),
            "fixed_answer_rank":rank,"condition_argmax_answer":candidates[order[0]],"temperature":TEMPERATURE,
            "span_lengths":{"image":len(details["image_positions"]),"text":len(details["text_positions"])},
            "spans":{"image":details["image_positions"],"text":details["text_positions"]},
            "input_details":{"prompt_hash":details["prompt_hash"],"rendered_hash":details["rendered_hash"],
                             "image_path":details["image_path"],"image_sha256":details["image_sha256"],
                             "input_ids_sha256":details["input_ids_sha256"],"sequence_length":details["sequence_length"],
                             "text_positions":details["text_positions"]},
            "text_corruption_audit":text_audit,"randomization":randomization or {},"internal_model_forwards":forward_count}


def preflight_for_rows(processor:Any,rows:Sequence[dict[str,Any]])->dict[str,Any]:
    from dp_SA.real_sa.scoring import tokenization_preflight
    base=tokenization_preflight(processor,rows)
    multiplier=1+REPLICATES*3
    per_condition=1 if base["policy"]=="single_token_next_token_logits" else 12
    return {**base,"random_replicates":REPLICATES,"conditions_per_case":multiplier,
            "expected_internal_model_forwards":len(rows)*multiplier*per_condition}


__all__=["preflight_for_rows","result_row","score_condition"]
