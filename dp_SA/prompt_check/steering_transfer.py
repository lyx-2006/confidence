from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from layer_metacognition.model_adapter import AdditiveActivationHook, run_logits_forward

from .capture import stable_shard
from .config import (BOOTSTRAPS, MATCHED_ROOT, SEED, SMOKE_BOOTSTRAPS, VALIDATION_BOOTSTRAPS, T0_STEERING_TABLE,
                     T0_STEERING_TRIALS, T0_VECTOR_METADATA, TEST_MANIFEST)
from .io_utils import append_jsonl, array_hash, atomic_csv, atomic_json, atomic_jsonl, load_jsonl, sha256_file
from .runtime import _append_candidate, class_token_ids, load_inference, prepare_case
from .scoring import conditional_sequence_log_likelihood, numeric_score, t3_score
from .stats import retention_eligibility
from .templates import TEMPLATES
from .sampling import validation_test


def _vector(metadata: dict[str,Any], fold: int, answer: str, layer: int) -> tuple[np.ndarray,dict[str,Any]]:
    matches=[r for r in metadata["vectors"] if r["position"]=="P1_LAT" and r["direction"]=="matched_loao" and int(r["fold"])==fold and r["recipient_answer"]==answer and int(r["layer"])==layer]
    if len(matches)!=1:raise ValueError(f"Unique T0 vector not found: fold={fold} {answer} L{layer}")
    row=matches[0];path=MATCHED_ROOT/row["vector_file"]
    if sha256_file(path)!=row["vector_file_sha256"]:raise ValueError("T0 vector file hash changed")
    with np.load(path) as payload:value=np.asarray(payload[row["scaled_key"]],dtype=np.float32)
    return value,row


@torch.inference_mode()
def _steered_score(model:Any,modules:Any,tokenizer:Any,inputs:Any,located:dict[str,Any],template:str,layer:int,alpha:float,vector:np.ndarray)->tuple[dict[str,Any],dict[str,Any],np.ndarray]:
    spec=TEMPLATES[template];lat=int(located["P1_LAT"]["processed_index"]);sac=int(located["P1_SAC"]["processed_index"])
    if spec.kind!="labels":
        hook=AdditiveActivationHook(modules,layer_index=layer,target_position=lat,steering_vector=torch.from_numpy(vector)*alpha,prefill_sequence_length=int(inputs.input_ids.shape[1]),injection_site="block_output")
        with hook:logits=run_logits_forward(model,inputs,[sac],modules)[sac]
        ids=class_token_ids(tokenizer);score=numeric_score([float(logits[i]) for i in ids],reversed_scale=spec.kind=="numeric_reversed",token_ids=ids)
        return score,hook.diagnostics(),hook.h_before.numpy()
    labels=__import__("dp_SA.prompt_check.config",fromlist=["T3_LABELS"]).T3_LABELS;token_ids=[list(map(int,tokenizer.encode(label,add_special_tokens=False))) for label in labels]
    prefix=int(inputs.input_ids.shape[1]);likelihoods=[];diagnostics=[];before=None
    for ids in token_ids:
        candidate=_append_candidate(inputs,ids);hook=AdditiveActivationHook(modules,layer_index=layer,target_position=lat,steering_vector=torch.from_numpy(vector)*alpha,prefill_sequence_length=int(candidate["input_ids"].shape[1]),injection_site="block_output")
        with hook:output=model(**candidate,use_cache=False,return_dict=True)
        likelihoods.append(conditional_sequence_log_likelihood(output.logits,prefix,ids));diagnostics.append(hook.diagnostics())
        if before is None:before=hook.h_before.numpy()
    assert before is not None
    return t3_score(likelihoods,token_ids),{"candidate_hooks":diagnostics,"hook_count":len(diagnostics)},before


def _trial_key(template:str,case:str,layer:int,alpha:float)->str:return f"{template}|{case}|L{layer}|a{alpha:g}"


def steering_worker(root:Path,*,worker:int,num_gpus:int,templates:Sequence[str],layers:Sequence[int],alphas:Sequence[float],resume:bool,smoke:bool,validation_cases:int|None=None)->dict[str,Any]:
    test=validation_test(validation_cases) if validation_cases else load_jsonl(TEST_MANIFEST)
    if smoke:test=test[:4]
    test=[r for r in test if stable_shard(str(r["case_id"]),num_gpus)==worker]
    path=root/f"artifacts/steering/steering.worker_{worker}.jsonl";existing={_trial_key(r["template"],str(r["case_id"]),int(r["layer"]),float(r["alpha"])) for r in load_jsonl(path) if r.get("status")=="completed"}
    expected={_trial_key(template,str(record["case_id"]),int(layer),float(alpha)) for template in templates for record in test for layer in layers for alpha in alphas}
    if expected.issubset(existing):return {"status":"complete","worker":worker,"new_gpu_forwards":0,"resumed_noop":True,"elapsed_seconds":0.0}
    metadata=json.loads(T0_VECTOR_METADATA.read_text());inference,modules,tokenizer,device,processor=load_inference();forwards=0;started=time.time()
    clean={(r["template"],str(r["case_id"])):r for r in load_jsonl(root/"artifacts/clean/capture.jsonl") if r.get("is_candidate")}
    for template in templates:
        for record in test:
            case=str(record["case_id"]);inputs,rendered,located=prepare_case(inference.processor,tokenizer,device,record,TEMPLATES[template]);clean_row=clean[template,case]
            for layer in layers:
                vector,vrow=_vector(metadata,int(record["fold"]),str(record["test_answer"]),int(layer));vector_hash=array_hash(vector)
                for alpha in alphas:
                    key=_trial_key(template,case,int(layer),float(alpha))
                    if key in existing:
                        if not resume:raise FileExistsError(f"Steering result exists; use --resume: {key}")
                        continue
                    score,diagnostics,before=_steered_score(inference.model,modules,tokenizer,inputs,located,template,int(layer),float(alpha),vector);forwards += 5 if template=="T3" else 1
                    clean_soft=float(clean_row["canonical_soft_sa"]);hard_field="canonical_hard_label" if template=="T3" else "canonical_hard_class";clean_hard=clean_row[hard_field];steered_hard=score[hard_field]
                    alpha0_error=abs(float(score["canonical_soft_sa"])-clean_soft) if float(alpha)==0 else None
                    if alpha0_error is not None and (alpha0_error>1e-6 or steered_hard!=clean_hard):raise ValueError(f"Alpha=0 parity failed: {key} error={alpha0_error}")
                    unit=vector/np.linalg.norm(vector);projection=float(np.dot(before.astype(np.float32),unit.astype(np.float32)))
                    result={"status":"completed","template":template,"case_id":case,"family_id":record["family_id"],"item_id":str(record["item_id"]),"fold":int(record["fold"]),"answer":record["test_answer"],"test_side":record["test_side"],"test_status":record["test_status"],"condition":record["condition"],"layer":int(layer),"alpha":float(alpha),"clean_soft_sa":clean_soft,"steered_soft_sa":float(score["canonical_soft_sa"]),"delta_soft_sa":float(score["canonical_soft_sa"])-clean_soft,"clean_hard":clean_hard,"steered_hard":steered_hard,"hard_label_changed":steered_hard!=clean_hard,"alpha_zero_abs_error":alpha0_error,"alpha_zero_parity":alpha0_error is None or alpha0_error<=1e-6,"natural_unit_projection":projection,"vector_norm":float(np.linalg.norm(vector)),"vector_array_sha256":vector_hash,"vector_fingerprint":vrow["vector_fingerprint"],"vector_file_sha256":vrow["vector_file_sha256"],"hook_diagnostics":diagnostics,"processor":processor,"scoring":score}
                    append_jsonl(path,result);existing.add(key)
                    if forwards%10==0:atomic_json(root/f"progress/steering_worker_{worker}.json",{"status":"running","new_gpu_forwards":forwards,"last":key,"elapsed_seconds":time.time()-started})
    result={"status":"complete","worker":worker,"new_gpu_forwards":forwards,"resumed_noop":forwards==0,"elapsed_seconds":time.time()-started};atomic_json(root/f"progress/steering_worker_{worker}.json",result);return result


def run_steering(root:Path,*,num_gpus:int,templates:Sequence[str],layers:Sequence[int],alphas:Sequence[float],resume:bool,smoke:bool,validation_cases:int|None=None)->dict[str,Any]:
    if set(map(float,alphas))!={-2.,0.,2.}:raise ValueError("Formal S^2 transfer requires exactly alphas -2, 0, +2")
    if num_gpus==1:workers=[steering_worker(root,worker=0,num_gpus=1,templates=templates,layers=layers,alphas=alphas,resume=resume,smoke=smoke,validation_cases=validation_cases)]
    else:
        processes=[]
        for worker in range(num_gpus):
            env=dict(os.environ);env["CUDA_VISIBLE_DEVICES"]=str(worker);cmd=[sys.executable,"-m","dp_SA.prompt_check.steering_transfer","--worker",str(worker),"--num-gpus",str(num_gpus),"--output-root",str(root),"--templates",*templates,"--layers",*map(str,layers),"--alphas",*map(str,alphas)]
            if resume:cmd.append("--resume")
            if smoke:cmd.append("--smoke")
            if validation_cases:cmd.extend(["--validation-cases",str(validation_cases)])
            processes.append(subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[2],env=env))
        codes=[p.wait() for p in processes]
        if any(codes):raise RuntimeError(f"Steering workers failed: {codes}")
        workers=[json.loads((root/f"progress/steering_worker_{i}.json").read_text()) for i in range(num_gpus)]
    rows=[r for i in range(num_gpus) for r in load_jsonl(root/f"artifacts/steering/steering.worker_{i}.jsonl") if r.get("template") in templates and int(r["layer"]) in layers and float(r["alpha"]) in alphas]
    expected=(validation_cases or (4 if smoke else 174))*len(templates)*len(layers)*len(alphas);keys=[_trial_key(r["template"],str(r["case_id"]),int(r["layer"]),float(r["alpha"])) for r in rows]
    if len(rows)!=expected or len(keys)!=len(set(keys)):raise ValueError(f"Steering merge incomplete/duplicate: {len(rows)}/{expected}")
    rows.sort(key=lambda r:(r["template"],r["case_id"],r["layer"],r["alpha"]));atomic_jsonl(root/"artifacts/steering_trials.jsonl",rows);result={"status":"complete","trial_count":len(rows),"workers":workers};atomic_json(root/"progress/steering.json",result);return result


class SteeringBootstrap:
    def __init__(self,test:Sequence[dict[str,Any]],repeats:int):
        self.test={str(r["case_id"]):r for r in test};self.repeats=repeats;self.rng=np.random.default_rng(SEED+400)
        self.confirmatory_answers=sorted({r["test_answer"] for r in test if r["test_status"]=="confirmatory"})
        self.by_answer={a:sorted(str(r["case_id"]) for r in test if r["test_status"]=="confirmatory" and r["test_answer"]==a) for a in self.confirmatory_answers}
        self.answer_draws={a:self.rng.integers(0,len(ids),size=(repeats,len(ids))) for a,ids in self.by_answer.items()};self.all_ids=sorted(self.test);self.all_draws=self.rng.integers(0,len(self.all_ids),size=(repeats,len(self.all_ids)))

    def aggregate(self,values:dict[str,float],mode:str)->tuple[float,np.ndarray,list[str]]:
        if mode=="answer_equal":
            observed=[];boots=[];used=[]
            for answer in self.confirmatory_answers:
                ids=self.by_answer[answer];vector=np.asarray([values[i] for i in ids]);observed.append(vector.mean());boots.append(vector[self.answer_draws[answer]].mean(axis=1));used.extend(ids)
            return float(np.mean(observed)),np.stack(boots).mean(axis=0),used
        vector=np.asarray([values[i] for i in self.all_ids]);return float(vector.mean()),vector[self.all_draws].mean(axis=1),self.all_ids


def analyze_steering(root:Path,*,templates:Sequence[str],layers:Sequence[int],smoke:bool,validation_cases:int|None=None)->dict[str,Any]:
    trials=load_jsonl(root/"artifacts/steering_trials.jsonl");test=validation_test(validation_cases) if validation_cases else (load_jsonl(TEST_MANIFEST)[:4] if smoke else load_jsonl(TEST_MANIFEST));repeats=SMOKE_BOOTSTRAPS if smoke else (VALIDATION_BOOTSTRAPS if validation_cases else BOOTSTRAPS);bootstrap=SteeringBootstrap(test,repeats)
    historical=[r for r in load_jsonl(T0_STEERING_TRIALS) if r["direction"]=="matched_loao" and r["position"]=="P1_LAT" and int(r["layer"]) in layers and float(r["alpha"]) in {-2.,0.,2.} and str(r["case_id"]) in bootstrap.test]
    def keyed(rows:Sequence[dict[str,Any]],template:str|None=None):return {(template or r.get("template","T0"),str(r["case_id"]),int(r["layer"]),float(r["alpha"])):r for r in rows}
    all_rows={**keyed(historical,"T0"),**keyed(trials)};effects=[];contrasts=[];historical_point_parity=[]
    with T0_STEERING_TABLE.open(newline="",encoding="utf-8") as handle:
        historical_table=list(csv.DictReader(handle))
    for layer in layers:
        for mode in ("answer_equal","family_micro"):
            t0_delta={alpha:{case:float(all_rows["T0",case,layer,alpha]["delta_soft_sa"]) for case in bootstrap.test} for alpha in (-2.,0.,2.)}
            t0_s2={case:(t0_delta[2.][case]-t0_delta[-2.][case])/2 for case in bootstrap.test};t0_point,t0_boot,used=bootstrap.aggregate(t0_s2,mode)
            table_point=math.nan
            if not smoke and not validation_cases:
                table_rows=[r for r in historical_table if r["position"]=="P1_LAT" and r["direction"]=="matched_loao" and int(r["layer"])==layer]
                if mode=="answer_equal":
                    values={float(r["symmetric_effect_2"]) for r in table_rows}
                    if len(values)!=1:raise ValueError(f"Ambiguous historical T0 S2 table value: L{layer}")
                    table_point=values.pop()
                else:
                    alpha_rows={float(r["alpha"]):r for r in table_rows}
                    table_point=(float(alpha_rows[2.]["family_micro_delta_sa"])-float(alpha_rows[-2.]["family_micro_delta_sa"]))/2
                error=abs(t0_point-table_point)
                if error>1e-12:raise ValueError(f"Historical T0 steering point parity failed: L{layer} {mode} error={error}")
                historical_point_parity.append({"layer":layer,"aggregation":mode,"reconstructed_s2":t0_point,"historical_table_s2":table_point,"absolute_error":error,"passed":True})
            for template in templates:
                deltas={alpha:{case:float(all_rows[template,case,layer,alpha]["delta_soft_sa"]) for case in bootstrap.test} for alpha in (-2.,0.,2.)};s2={case:(deltas[2.][case]-deltas[-2.][case])/2 for case in bootstrap.test}
                projections={case:float(all_rows[template,case,layer,0.]["natural_unit_projection"]) for case in bootstrap.test};projection_sd=float(np.std([projections[c] for c in used],ddof=1));vector_norm=float(np.mean([all_rows[template,c,layer,0.]["vector_norm"] for c in used]))
                for alpha in (-2.,0.,2.):
                    point,boot,_=bootstrap.aggregate(deltas[alpha],mode);low,high=np.quantile(boot,[.025,.975]);changes={case:float(all_rows[template,case,layer,alpha]["hard_label_changed"]) for case in bootstrap.test};change_point,change_boot,_=bootstrap.aggregate(changes,mode)
                    effects.append({"template":template,"layer":layer,"alpha":alpha,"aggregation":mode,"mean_delta_sa":point,"ci_low":float(low),"ci_high":float(high),"hard_label_change_rate":change_point,"hard_change_ci_low":float(np.quantile(change_boot,.025)),"hard_change_ci_high":float(np.quantile(change_boot,.975)),"natural_projection_sd":projection_sd,"mean_vector_norm":vector_norm,"dose_in_projection_sd":abs(alpha)*vector_norm/projection_sd if projection_sd>0 else math.nan,"case_count":len(used)})
                tx_point,tx_boot,_=bootstrap.aggregate(s2,mode);contrast=tx_point-t0_point;contrast_boot=tx_boot-t0_boot;elig=retention_eligibility(t0_point,t0_boot);ratio=tx_point/t0_point if elig["eligible"] else math.nan;ratio_boot=np.full_like(tx_boot,np.nan,dtype=float);np.divide(tx_boot,t0_boot,out=ratio_boot,where=t0_boot!=0);valid_ratio=ratio_boot[np.isfinite(ratio_boot)&(np.sign(t0_boot)==np.sign(t0_point))]
                sign=float(np.mean([np.sign(s2[c])==np.sign(t0_s2[c]) for c in used]));contrasts.append({"template":template,"layer":layer,"aggregation":mode,"s2_tx":tx_point,"s2_t0":t0_point,"t0_historical_table_s2":table_point,"absolute_contrast":contrast,"contrast_ci_low":float(np.quantile(contrast_boot,.025)),"contrast_ci_high":float(np.quantile(contrast_boot,.975)),"retention_ratio":ratio,"retention_ratio_ci_low":float(np.quantile(valid_ratio,.025)) if elig["eligible"] else math.nan,"retention_ratio_ci_high":float(np.quantile(valid_ratio,.975)) if elig["eligible"] else math.nan,"retention_eligible":elig["eligible"],"t0_denominator_ci_low":elig["ci_low"],"t0_denominator_ci_high":elig["ci_high"],"t0_denominator_same_sign_fraction":elig["same_sign_fraction"],"case_s2_sign_agreement":sign,"case_count":len(used)})
    atomic_csv(root/"tables/steering_transfer_effects.csv",effects);atomic_csv(root/"tables/steering_transfer_vs_t0.csv",contrasts);atomic_json(root/"artifacts/diagnostics/steering_bootstrap_design.json",{"seed":SEED+400,"repeats":repeats,"confirmatory_answers":bootstrap.confirmatory_answers,"answer_equal_case_count":sum(map(len,bootstrap.by_answer.values())),"family_micro_case_count":len(bootstrap.all_ids)});atomic_json(root/"artifacts/diagnostics/t0_steering_point_parity.json",{"status":"not_applicable_subset" if (smoke or validation_cases) else "passed","rows":historical_point_parity})
    return {"status":"complete","effect_rows":len(effects),"contrast_rows":len(contrasts)}


def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser();p.add_argument("--worker",type=int,required=True);p.add_argument("--num-gpus",type=int,choices=(1,2),required=True);p.add_argument("--output-root",required=True);p.add_argument("--templates",nargs="+",required=True);p.add_argument("--layers",nargs="+",type=int,required=True);p.add_argument("--alphas",nargs="+",type=float,required=True);p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true");p.add_argument("--validation-cases",type=int);a=p.parse_args(argv);print(json.dumps(steering_worker(Path(a.output_root),worker=a.worker,num_gpus=a.num_gpus,templates=a.templates,layers=a.layers,alphas=a.alphas,resume=a.resume,smoke=a.smoke,validation_cases=a.validation_cases),ensure_ascii=False));return 0


if __name__=="__main__":raise SystemExit(main())
