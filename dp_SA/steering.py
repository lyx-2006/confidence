from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import AdditiveActivationHook, model_input_device, resolve_language_modules, run_logits_forward

from .config import ALPHAS, BOOTSTRAP_REPEATS, INFERENCE_PATH, LAYERS, MODEL_PATH, OUTPUT_ROOT, POSITIONS, SEED, VECTOR_NORM_FRACTION
from .io_utils import append_jsonl, atomic_json, atomic_jsonl, canonical_hash, load_jsonl
from .positions import locate_phase1_positions
from .prompts import SA_PREFILL
from .selection import record_key, select_manifests
from .soft_score import class_token_ids, soft_sa_from_logits

def _load_vector(root: Path, row: dict[str,Any], position: str, layer: int) -> np.ndarray:
    with np.load(root/row["hidden_file"]) as payload: return np.asarray(payload[f"{position}__L{layer}"],dtype=np.float32)

def _messages(row: dict[str,Any]) -> list[dict[str,Any]]:
    return [{"role":"user","content":[{"type":"image","image":str(Path(row["image_path"]).resolve())},{"type":"text","text":row["phase1_prompt"]}]},
            {"role":"assistant","content":[{"type":"text","text":SA_PREFILL}]}]

def _save_torch_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent); os.close(fd)
    try: torch.save(value,tmp); os.replace(tmp,path)
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise

def build_vectors(root: Path, construction: Sequence[dict[str,Any]], *, positions: Sequence[str]=POSITIONS, layers: Sequence[int]=LAYERS,
                  shuffled: bool=True, seed: int=SEED) -> tuple[dict[tuple[str,int,str],torch.Tensor],dict[str,Any],dict[str,Any]]:
    high=[r for r in construction if r["construction_side"]=="high_image"]; low=[r for r in construction if r["construction_side"]=="high_text"]
    vectors={}; metadata=[]; artifacts={}
    for position in positions:
        for layer in layers:
            h=np.stack([_load_vector(root,r,position,layer) for r in high]); l=np.stack([_load_vector(root,r,position,layer) for r in low]); all_h=np.concatenate([h,l])
            raw=h.mean(0)-l.mean(0); raw_norm=float(np.linalg.norm(raw)); mean_residual=float(np.linalg.norm(all_h,axis=1).mean()); target=VECTOR_NORM_FRACTION*mean_residual
            if not all(math.isfinite(x) and x>0 for x in (raw_norm,mean_residual,target)): raise ValueError(f"Invalid vector norm at {position} L{layer}")
            scaled=(raw/raw_norm*target).astype(np.float32); vectors[(position,int(layer),"true")]=torch.from_numpy(scaled)
            artifact_key=f"{position}__L{layer}__true"; artifacts[artifact_key]={"raw_vector":torch.from_numpy(raw.astype(np.float32)),"scaled_vector":torch.from_numpy(scaled),"high_mean":torch.from_numpy(h.mean(0).astype(np.float32)),"low_mean":torch.from_numpy(l.mean(0).astype(np.float32))}
            metadata.append({"position":position,"layer":int(layer),"direction_type":"true","raw_vector_norm":raw_norm,"mean_residual_norm":mean_residual,"target_vector_norm":target,
                             "high_mean_norm":float(np.linalg.norm(h.mean(0))),"low_mean_norm":float(np.linalg.norm(l.mean(0)))})
            if shuffled and position=="P1_PANL" and layer in (18,20):
                combined=all_h.copy(); labels=np.asarray([1]*len(h)+[0]*len(l)); random.Random(seed+layer).shuffle(labels)
                sh=combined[labels==1].mean(0)-combined[labels==0].mean(0); sh_norm=float(np.linalg.norm(sh))
                if not math.isfinite(sh_norm) or sh_norm<=0: raise ValueError("Invalid shuffled vector")
                sv=(sh/sh_norm*target).astype(np.float32); vectors[(position,int(layer),"shuffled")]=torch.from_numpy(sv)
                artifacts[f"{position}__L{layer}__shuffled"]={"raw_vector":torch.from_numpy(sh.astype(np.float32)),"scaled_vector":torch.from_numpy(sv)}
                metadata.append({"position":position,"layer":int(layer),"direction_type":"shuffled","raw_vector_norm":sh_norm,"mean_residual_norm":mean_residual,"target_vector_norm":target})
    return vectors,{"normalization_fraction":VECTOR_NORM_FRACTION,"vectors":metadata,"construction_fingerprint":canonical_hash([r["case_id"] for r in construction])},artifacts

def _smoke_manifests(rows: Sequence[dict[str,Any]]) -> tuple[list[dict[str,Any]],list[dict[str,Any]],dict[str,Any]]:
    ordered=sorted(rows,key=lambda r:(float(r["soft_sa_image_score"]),record_key(r))); used=set(); low=[]; high=[]
    for row in ordered:
        if row["item_id"] not in used: low.append({**row,"construction_side":"high_text"}); used.add(row["item_id"])
        if len(low)==5: break
    for row in reversed(ordered):
        if row["item_id"] not in used: high.append({**row,"construction_side":"high_image"}); used.add(row["item_id"])
        if len(high)==5: break
    test=[]
    for row in sorted(rows,key=record_key):
        if row["item_id"] in used: continue
        test.append({**row,"test_side":"smoke"}); used.add(row["item_id"])
        if len(test)==10: break
    if len(low)<2 or len(high)<2 or not test: raise ValueError("Smoke capture needs more item-disjoint records")
    return high+low,test,{"smoke":True,"construction":len(high)+len(low),"test":len(test)}

def prepare_manifests(root: Path, *, smoke: bool=False) -> tuple[list[dict[str,Any]],list[dict[str,Any]],dict[str,Any]]:
    rows=[r for r in load_jsonl(root/"capture"/"results.jsonl") if r.get("status")=="completed"]
    construction,test,summary=_smoke_manifests(rows) if smoke else select_manifests(rows)
    steering=root/"steering"; atomic_jsonl(steering/"construction_manifest.jsonl",construction); atomic_jsonl(steering/"test_manifest.jsonl",test); atomic_json(steering/"selection_summary.json",summary)
    return construction,test,summary

def _key(row: dict[str,Any]) -> str:
    return f'{row["case_id"]}|{row["position"]}|L{row["layer"]}|{row["direction_type"]}|a{row["alpha"]:g}'

def run_steering(*, output_root: Path=OUTPUT_ROOT, smoke: bool=False, resume: bool=False) -> dict[str,Any]:
    steering_dir=output_root/"steering"; steering_dir.mkdir(parents=True,exist_ok=True); pid_path=steering_dir/"active.pid"
    if pid_path.exists():
        try: pid=int(pid_path.read_text()); os.kill(pid,0); raise RuntimeError(f"Steering already active: PID {pid}")
        except ProcessLookupError: pid_path.unlink()
    pid_path.write_text(str(os.getpid()))
    try:
        construction,test,selection=prepare_manifests(output_root,smoke=smoke)
        positions=("P1_PANL",) if smoke else POSITIONS; layers=(18,) if smoke else LAYERS; alphas=(-2.0,0.0,2.0) if smoke else ALPHAS
        vectors,metadata,artifacts=build_vectors(output_root,construction,positions=positions,layers=layers,shuffled=not smoke); _save_torch_atomic(steering_dir/"vectors.pt",artifacts); atomic_json(steering_dir/"vector_metadata.json",metadata)
        config={"format_version":1,"smoke":smoke,"positions":list(positions),"layers":list(layers),"alphas":list(alphas),"test_count":len(test),"seed":SEED,
                "construction_fingerprint":metadata["construction_fingerprint"],"test_fingerprint":canonical_hash([r["case_id"] for r in test])}; config["fingerprint"]=canonical_hash(config)
        config_path=steering_dir/"config.json"
        if config_path.exists():
            old=json.loads(config_path.read_text());
            if old.get("fingerprint")!=config["fingerprint"]: raise ValueError("Steering config fingerprint changed")
            if not resume: raise FileExistsError("Steering output exists; use --resume")
        else: atomic_json(config_path,config)
        pred_path=steering_dir/"predictions.jsonl"; existing={_key(r) for r in load_jsonl(pred_path) if r.get("status")=="completed"}
        runtime=load_runtime(INFERENCE_PATH); inference=runtime.QwenVLInference(str(MODEL_PATH)); modules=resolve_language_modules(inference.model); tokenizer=getattr(inference.processor,"tokenizer",inference.processor); ids=class_token_ids(tokenizer); device=model_input_device(inference)
        total=len(test)*(len(positions)*len(layers)*len(alphas)+(0 if smoke else 2*len(alphas))); done=len(existing); started=time.time()
        for sample_index,row in enumerate(test):
            messages=_messages(row); rendered=render_continued_assistant(inference.processor,messages,SA_PREFILL); inputs=prepare_multimodal_inputs(inference.processor,messages,rendered,device=device)
            located=locate_phase1_positions(tokenizer,rendered,inputs,row["phase0_raw_answer"]); seq=int(inputs.input_ids.shape[1]); clean_logits=np.asarray(row["class_logits"],float); clean_probs=np.asarray(row["class_probabilities"],float); clean=float(row["soft_sa_image_score"])
            directions=[(p,l,"true") for p in positions for l in layers]
            if not smoke: directions += [("P1_PANL",l,"shuffled") for l in (18,20)]
            for position,layer,direction_type in directions:
                target=int(located[position]["processed_index"]); base=vectors[(position,layer,direction_type)]
                for alpha in alphas:
                    proto={"case_id":row["case_id"],"item_id":row["item_id"],"test_side":row["test_side"],"position":position,"layer":layer,"direction_type":direction_type,"alpha":float(alpha)}
                    if _key(proto) in existing: continue
                    hook=AdditiveActivationHook(modules,layer_index=layer,target_position=target,steering_vector=base*float(alpha),prefill_sequence_length=seq)
                    with hook: logits=run_logits_forward(inference.model,inputs,[int(located["P1_SAC"]["processed_index"])],modules)[int(located["P1_SAC"]["processed_index"])]
                    diag=hook.diagnostics(); scored=soft_sa_from_logits(logits,ids)
                    if alpha==0 and (np.max(np.abs(np.asarray(scored["class_logits"])-clean_logits))>1e-6 or np.max(np.abs(np.asarray(scored["class_probabilities"])-clean_probs))>1e-6): raise RuntimeError(f"Alpha-zero parity failed: {row['case_id']} {position} L{layer}")
                    before=hook.h_before.numpy(); after=hook.h_after.numpy(); cosine=float(np.dot(before,after)/(np.linalg.norm(before)*np.linalg.norm(after))); ratio=float(np.linalg.norm(after)/np.linalg.norm(before))
                    clean_hard=int(row["argmax_hard_class"]); clean_sorted=np.sort(clean_logits); steered_logits=np.asarray(scored["class_logits"],float); steered_sorted=np.sort(steered_logits)
                    result={"status":"completed",**proto,"clean_soft_sa":clean,"steered_soft_sa":scored["soft_sa_image_score"],"delta_soft_sa":scored["soft_sa_image_score"]-clean,
                            "clean_argmax_class":row["argmax_hard_class"],"steered_argmax_class":scored["argmax_hard_class"],"class_logits":scored["class_logits"],"class_probabilities":scored["class_probabilities"],
                            "clean_class_logits":clean_logits.tolist(),"clean_argmax_logit_margin":float(clean_sorted[-1]-clean_sorted[-2]),"steered_argmax_logit_margin":float(steered_sorted[-1]-steered_sorted[-2]),
                            "clean_class_logit_margin_after_steering":float(steered_logits[clean_hard]-max(np.delete(steered_logits,clean_hard))),
                            "probability_sum":scored["probability_sum"],"hard_class_changed":clean_hard!=int(scored["argmax_hard_class"]),"hard_class_delta":int(scored["argmax_hard_class"])-clean_hard,
                            "ceiling_saturated":scored["soft_sa_image_score"]>=0.95-1e-9,"floor_saturated":scored["soft_sa_image_score"]<=0.05+1e-9,
                            "hook_diagnostics":diag,"activation_cosine":cosine,"activation_norm_ratio":ratio}
                    append_jsonl(pred_path,result); existing.add(_key(result)); done+=1
                    if done%25==0: atomic_json(steering_dir/"progress.json",{"completed_cells":done,"total_cells":total,"fraction":done/total,"elapsed_seconds":time.time()-started,"last":proto})
        rows=load_jsonl(pred_path); summary={"status":"complete","completed_cells":len([r for r in rows if r.get("status")=="completed"]),"expected_cells":total,"selection":selection}
        atomic_json(steering_dir/"summary.json",summary); return summary
    finally:
        if pid_path.exists() and pid_path.read_text().strip()==str(os.getpid()): pid_path.unlink()

def main(argv: Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(); p.add_argument("--output-root",default=str(OUTPUT_ROOT)); p.add_argument("--smoke",action="store_true"); p.add_argument("--resume",action="store_true")
    a=p.parse_args(argv); run_steering(output_root=Path(a.output_root),smoke=a.smoke,resume=a.resume); return 0
if __name__=="__main__": raise SystemExit(main())
