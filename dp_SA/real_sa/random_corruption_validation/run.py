from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from dp_SA.real_sa.data import load_frozen_cohort
from dp_SA.real_sa.run import smoke_records
from dp_SA.real_sa.runtime import load_strict_inference
from dp_SA.unimodal_logit_confidence.score_unimodal import candidate_suffix_ids

from .analysis import analyze
from .config import (CLEAN_PROBABILITY_TOLERANCE, CORRUPT_CONDITIONS, MEAN_CONDITION_SCORES,
                     MEAN_PROCESSOR_AUDIT, OUTPUT_ROOT, PACKAGE_ROOT, REPLICATES, SEED,
                     TEMPERATURE, TRAIN_MANIFEST)
from .inputs import prepare_condition_inputs
from .io_utils import (atomic_json, atomic_jsonl, canonical_hash, ensure_layout, load_jsonl,
                       sha256_file, tree_hashes)
from .randomization import build_token_pool, ensure_gaussian_image, sample_text_tokens, stable_seed
from .scoring import preflight_for_rows, result_row, score_condition


def _old_snapshot() -> dict[str,Any]:
    parent=PACKAGE_ROOT.parent
    return {"source":{path.name:sha256_file(path) for path in sorted(parent.glob("*.py"))},
            "output":tree_hashes(parent/"output")}


def _new_code_hashes() -> dict[str,str]:
    return {path.name:sha256_file(path) for path in sorted(PACKAGE_ROOT.glob("*.py"))}


def _mode_base(root:Path,mode:str)->Path:
    base=root/"progress/smoke" if mode=="smoke" else root
    for name in ("artifacts/gaussian_images","tables","figures"): (base/name).mkdir(parents=True,exist_ok=True)
    return base


def _validate_runtime(identity:dict[str,Any])->None:
    prior=json.loads(MEAN_PROCESSOR_AUDIT.read_text(encoding="utf-8"))
    if identity!=prior: raise RuntimeError("Runtime processor identity differs from the completed mean-embedding Real SA run")
    if identity["image_processor_class"].split(".")[-1]!="Qwen2VLImageProcessorFast" or identity["is_fast"] is not True:
        raise RuntimeError("Qwen2VLImageProcessorFast is mandatory")


def _run_payload(cohort:Any,identity:dict[str,Any],old:dict[str,Any])->dict[str,Any]:
    return {"format_version":1,"experiment":"random_corruption_intervention_robustness","seed":SEED,
        "replicates":REPLICATES,"temperature":TEMPERATURE,"noise":{"mean":127.5,"std":63.75,"format":"RGB PNG"},
        "token_sampling":{"minimum_frequency":5,"weighted_by_empirical_frequency":True,"exclude_case_clue_ids":True},
        "cohort_audit_fingerprint":canonical_hash(cohort.audit),"runtime_identity_fingerprint":identity["fingerprint"],
        "frozen_input_sha256":cohort.audit["source_sha256"],"mean_real_sa_condition_scores_sha256":sha256_file(MEAN_CONDITION_SCORES),
        "mean_processor_audit_sha256":sha256_file(MEAN_PROCESSOR_AUDIT),"new_implementation_sha256":_new_code_hashes(),
        "original_real_sa_snapshot":old}


def _set_fingerprint(path:Path,payload:dict[str,Any],resume:bool)->str:
    fingerprint=canonical_hash(payload); document={"fingerprint":fingerprint,"payload":payload}
    if path.exists():
        previous=json.loads(path.read_text(encoding="utf-8"))
        if previous!=document: raise RuntimeError("Existing output has a different run fingerprint")
        if not resume: raise RuntimeError("Output already exists; pass --resume to reuse it")
    else: atomic_json(path,document)
    return fingerprint


def _old_clean_rows()->dict[str,dict[str,Any]]:
    rows={str(row["case_id"]):row for row in load_jsonl(MEAN_CONDITION_SCORES) if row["corruption_condition"]=="clean"}
    if len(rows)!=100: raise ValueError(f"Expected 100 prior clean rows, found {len(rows)}")
    return rows


def _clean_checks(current:dict[str,Any],prior:dict[str,Any])->dict[str,bool]:
    return {"candidate_order_exact":current["candidate_order"]==prior["candidate_order"],
        "candidate_token_ids_exact":current["candidate_token_ids"]==prior["candidate_token_ids"],
        "candidate_scores_exact":current["candidate_scores"]==prior["candidate_scores"],
        "argmax_exact":current["condition_argmax_answer"]==prior["condition_argmax_answer"],
        "fixed_probability_within_1e-12":abs(current["fixed_answer_probability"]-prior["fixed_answer_probability"])<=CLEAN_PROBABILITY_TOLERANCE}


def _score_clean_parity(inference:Any,rows:Sequence[dict[str,Any]],candidate_map:dict[str,Any],fingerprint:str,
                        score_path:Path,parity_path:Path,resume:bool,identity:dict[str,Any])->tuple[list[dict[str,Any]],int]:
    scores=load_jsonl(score_path); existing={row["score_key"]:row for row in scores}
    for row in scores:
        if row.get("run_fingerprint")!=fingerprint: raise RuntimeError("Score fingerprint mismatch")
    if resume and parity_path.exists() and json.loads(parity_path.read_text()).get("status")=="passed":
        missing=[row["case_id"] for row in rows if f"{row['case_id']}|clean" not in existing]
        if not missing: return scores,0
    prior=_old_clean_rows(); pending=[]; forwards=0
    # Do not persist any newly computed clean row until every selected case passes.
    for row in rows:
        case=str(row["case_id"]); ids=candidate_map[case]
        values,details,text_audit,count=score_condition(inference,row,ids,image_path=None,replacement_ids=None,banned_ids=frozenset())
        current=result_row(row,"clean",None,values,ids,details,text_audit,fingerprint,count); checks=_clean_checks(current,prior[case])
        pending.append((current,checks)); forwards+=count
    report={"status":"passed" if all(all(c.values()) for _,c in pending) else "failed","case_count":len(rows),
            "run_fingerprint":fingerprint,"runtime_identity":identity,
            "tolerance":CLEAN_PROBABILITY_TOLERANCE,"cases":[{"case_id":r["case_id"],"checks":c} for r,c in pending]}
    atomic_json(parity_path,report)
    if report["status"]!="passed": raise RuntimeError(f"Clean parity failed; see {parity_path}")
    keep=[r for r in scores if not (r["corruption_condition"]=="clean" and str(r["case_id"]) in {str(x["case_id"]) for x in rows})]
    scores=sorted(keep+[r for r,_ in pending],key=lambda r:r["score_key"]); atomic_jsonl(score_path,scores)
    return scores,forwards


def _validate_complete(existing:dict[str,dict[str,Any]],row:dict[str,Any],rep:int,fingerprint:str)->bool:
    keys=[f"{row['case_id']}|r{rep}|{condition}" for condition in CORRUPT_CONDITIONS]
    present=[key in existing for key in keys]
    if any(present) and not all(present): return False
    if all(present):
        for key in keys:
            if existing[key].get("run_fingerprint")!=fingerprint: raise RuntimeError("Existing corruption score fingerprint mismatch")
        v10,v01,v00=(existing[key] for key in keys)
        if _replacement_ids(v10)!=_replacement_ids(v00): raise RuntimeError("Resumed v10/v00 token pairing failed")
        if v01["input_details"]["image_sha256"]!=v00["input_details"]["image_sha256"]: raise RuntimeError("Resumed v01/v00 image pairing failed")
        return True
    return False


def _replacement_ids(score:dict[str,Any])->list[int]:
    audit=score["text_corruption_audit"]
    if "replacement_text_token_ids" in audit: return list(map(int,audit["replacement_text_token_ids"]))
    forwards=audit.get("candidate_forwards",[])
    if not forwards: raise RuntimeError("Text-corrupted score has no replacement audit")
    values=[entry["replacement_text_token_ids"] for entry in forwards]
    if any(value!=values[0] for value in values[1:]): raise RuntimeError("Teacher-forced text corruption changed by candidate")
    return list(map(int,values[0]))


def _score_corruptions(inference:Any,rows:Sequence[dict[str,Any]],candidate_map:dict[str,Any],pool:Any,
                       base:Path,fingerprint:str,score_rows:list[dict[str,Any]])->tuple[list[dict[str,Any]],int]:
    score_path=base/"artifacts/condition_scores.jsonl"; audit_path=base/"artifacts/corruption_audit.jsonl"
    audits=load_jsonl(audit_path); audit_by={row["audit_key"]:row for row in audits}; existing={row["score_key"]:row for row in score_rows}; forwards=0
    from layer_metacognition.model_adapter import model_input_device
    device=model_input_device(inference)
    for row in rows:
        case=str(row["case_id"])
        for rep in range(REPLICATES):
            if _validate_complete(existing,row,rep,fingerprint):
                audit=audit_by.get(f"{case}|r{rep}")
                if not audit or audit.get("run_fingerprint")!=fingerprint or sha256_file(audit["gaussian_image"]["path"])!=audit["gaussian_image"]["sha256"]:
                    raise RuntimeError(f"Resume audit invalid: {case} replicate {rep}")
                continue
            _rendered,original_inputs,details,_audit=prepare_condition_inputs(inference.processor,row,device=device)
            original_ids=[int(v) for v in original_inputs.input_ids[0,details["text_positions"]].tolist()]
            replacement,text_random=sample_text_tokens(pool,inference.processor.tokenizer,original_ids,case,rep)
            image_path=base/"artifacts/gaussian_images"/f"{case}__r{rep}.png"
            image_random=ensure_gaussian_image(Path(row["image_path"]),image_path,case,rep)
            randomization={"text":text_random,"image":image_random}
            made={}
            specs=(("10_random_text",None,replacement),("01_gaussian_image",image_path,None),("00_both_random",image_path,replacement))
            for condition,path,tokens in specs:
                key=f"{case}|r{rep}|{condition}"
                if key in existing: made[condition]=existing[key]; continue
                values,input_details,text_audit,count=score_condition(inference,row,candidate_map[case],image_path=path,replacement_ids=tokens,banned_ids=pool.banned_ids)
                made[condition]=result_row(row,condition,rep,values,candidate_map[case],input_details,text_audit,fingerprint,count,randomization); forwards+=count
                score_rows=sorted([*score_rows,made[condition]],key=lambda value:value["score_key"]); existing[key]=made[condition]; atomic_jsonl(score_path,score_rows)
            if _replacement_ids(made["10_random_text"])!=_replacement_ids(made["00_both_random"]): raise RuntimeError("v10/v00 text pairing failed")
            if made["01_gaussian_image"]["input_details"]["image_sha256"]!=made["00_both_random"]["input_details"]["image_sha256"]: raise RuntimeError("v01/v00 image pairing failed")
            audit={"audit_key":f"{case}|r{rep}","run_fingerprint":fingerprint,"case_id":case,"family_id":row["family_id"],"replicate":rep,
                "text":text_random,"gaussian_image":image_random,"original_text_token_ids":original_ids,
                "v10_v00_token_ids_equal":True,"v01_v00_image_sha256_equal":True,
                "expected_text_seed":stable_seed(case,rep,"text"),"expected_image_seed":stable_seed(case,rep,"image")}
            old=audit_by.get(audit["audit_key"])
            if old is not None and old!=audit: raise RuntimeError("Existing corruption audit changed")
            if old is None: audits=sorted([*audits,audit],key=lambda value:value["audit_key"]); audit_by[audit["audit_key"]]=audit; atomic_jsonl(audit_path,audits)
    return score_rows,forwards


def run(mode:str,output_root:Path,*,resume:bool)->dict[str,Any]:
    if mode not in {"smoke","formal"}: raise ValueError("mode must be smoke or formal")
    root=ensure_layout(output_root); base=_mode_base(root,mode); progress=base if mode=="smoke" else root/"progress"; old_before=_old_snapshot(); cohort=load_frozen_cohort(); rows=smoke_records(cohort.tests) if mode=="smoke" else cohort.tests
    if not torch.cuda.is_available() or torch.cuda.device_count()<1:
        failure={"status":"blocked","reason":"GPU_REQUIRED","mode":mode,"cuda_device_count":torch.cuda.device_count()}; atomic_json(progress/"environment_gate.json",failure); raise RuntimeError("GPU execution requires one visible CUDA device")
    inference,identity=load_strict_inference(); _validate_runtime(identity)
    payload=_run_payload(cohort,identity,old_before); fingerprint=_set_fingerprint(root/"progress/run_config.json",payload,resume)
    if mode=="formal":
        path=root/"progress/smoke/smoke_report.json"
        if not path.exists(): raise RuntimeError("Formal run requires a passing smoke")
        smoke=json.loads(path.read_text());
        if smoke.get("status")!="passed" or smoke.get("run_fingerprint")!=fingerprint: raise RuntimeError("Formal run requires a passing smoke with the same fingerprint")
    preflight=preflight_for_rows(inference.processor,rows); atomic_json(progress/"forward_budget.json",preflight)
    train=load_jsonl(TRAIN_MANIFEST); pool=build_token_pool(inference.processor.tokenizer,train)
    candidate_map=preflight["candidate_token_ids"]; score_path=base/"artifacts/condition_scores.jsonl"; parity_path=progress/"processor_and_clean_parity.json"
    started=time.time(); score_rows,clean_forwards=_score_clean_parity(inference,rows,candidate_map,fingerprint,score_path,parity_path,resume,identity)
    selected={str(row["case_id"]) for row in rows}; clean_selected=[r for r in score_rows if r["corruption_condition"]=="clean" and str(r["case_id"]) in selected]
    case_pool_counts={str(r["case_id"]):int(sum(int(token) not in set(r["text_corruption_audit"]["original_text_token_ids"]) for token in pool.token_ids)) for r in clean_selected}
    if len(case_pool_counts)!=len(rows) or min(case_pool_counts.values())<=0: raise RuntimeError("Case-specific random token coverage failed")
    atomic_json(base/"artifacts/token_pool_audit.json",{**pool.audit,"case_available_token_counts":case_pool_counts,"minimum_case_available_token_count":min(case_pool_counts.values())})
    score_rows,corruption_forwards=_score_corruptions(inference,rows,candidate_map,pool,base,fingerprint,score_rows)
    analysis=analyze(base,score_rows,rows); new_forwards=clean_forwards+corruption_forwards
    result={"status":"passed" if mode=="smoke" else "complete","mode":mode,"run_fingerprint":fingerprint,"case_count":len(rows),
        "replicates":REPLICATES,"new_internal_model_forwards":new_forwards,"expected_total_internal_model_forwards":preflight["expected_internal_model_forwards"],
        "analysis":analysis,"elapsed_seconds":time.time()-started}
    if mode=="smoke":
        again_rows,again_clean=_score_clean_parity(inference,rows,candidate_map,fingerprint,score_path,parity_path,True,identity)
        _again_rows,again_corrupt=_score_corruptions(inference,rows,candidate_map,pool,base,fingerprint,again_rows)
        result["resume"]={"new_internal_model_forwards":again_clean+again_corrupt,"resumed_noop":again_clean+again_corrupt==0}
        if not result["resume"]["resumed_noop"]: raise RuntimeError("Second smoke resume performed a model forward")
    old_after=_old_snapshot(); result["original_real_sa_unchanged"]=old_after==old_before
    if not result["original_real_sa_unchanged"]: raise RuntimeError("Original real_sa source or historical output changed")
    report=progress/("smoke_report.json" if mode=="smoke" else "formal_report.json"); atomic_json(report,result)
    del inference; gc.collect(); torch.cuda.empty_cache(); return result


def build_parser()->argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description="Random-corruption intervention robustness validation")
    parser.add_argument("--mode",choices=("smoke","formal"),required=True); parser.add_argument("--output-root",type=Path,default=OUTPUT_ROOT); parser.add_argument("--resume",action="store_true"); return parser


def main(argv:Sequence[str]|None=None)->int:
    args=build_parser().parse_args(argv); print(json.dumps(run(args.mode,args.output_root,resume=args.resume),ensure_ascii=False)); return 0

if __name__=="__main__": raise SystemExit(main())
