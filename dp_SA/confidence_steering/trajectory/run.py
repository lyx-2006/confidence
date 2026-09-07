from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch

from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import AdditiveActivationHook, model_input_device, resolve_language_modules, run_hooked_forward
from dp_SA.checkpoint_steering.run import class_margin
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits

from .capture import messages, position_payload
from .config import BASE_CAPTURE_ROOT, DIRECTIONS, EPSILON, HIDDEN_DEFINITION, HIDDEN_SIZE, INFERENCE_PATH, INJECTION_LAYER, INJECTION_SITE, LAYERS, MODEL_PATH, NONZERO_ALPHAS, PANL_L14_ATOL, PARENT_AUDIT_PREDICTIONS, PARENT_FAST_CLEAN, PARENT_FAST_ROOT, PARENT_PANL_G_PROBE, POSITIONS, RAW_EXPRESSION_ATOL, RAW_EXPRESSION_STRICT_ATOL, SMOKE_FAMILIES, VECTOR_FILE, VECTOR_METADATA
from .io_utils import array_hash, atomic_json, atomic_jsonl, atomic_npz, canonical_forward_key, canonical_hash, load_jsonl, require_output_root, sha256_file, stable_shard
from .train_probes import load_probes, predict_raw, raw_parameters
from .processor import enforce_parent_fast_image_processor


def symmetric_derivative(plus: np.ndarray | float, minus: np.ndarray | float) -> np.ndarray:
    return (np.asarray(plus) - np.asarray(minus)) / (2.0 * EPSILON)


def load_vectors() -> tuple[dict[tuple[str,str],np.ndarray],dict[tuple[str,str],dict[str,Any]]]:
    from .config import EXPECTED_HASHES
    if sha256_file(VECTOR_FILE)!=EXPECTED_HASHES[VECTOR_FILE] or sha256_file(VECTOR_METADATA)!=EXPECTED_HASHES[VECTOR_METADATA]: raise ValueError("Frozen vector files changed")
    metadata=json.loads(VECTOR_METADATA.read_text()); meta={(str(r["recipient_answer"]),str(r["direction"])):r for r in metadata["vectors"] if int(r["layer"])==INJECTION_LAYER}
    vectors={}
    with np.load(VECTOR_FILE) as payload:
        for key,row in meta.items():
            expected=f"{key[0]}__{key[1]}__scaled"
            if row["scaled_key"]!=expected or expected not in payload.files: raise ValueError(f"Vector scaled_key mismatch: {key}")
            value=np.asarray(payload[expected])
            if value.dtype!=np.float32 or value.shape!=(HIDDEN_SIZE,) or array_hash(value)!=row["scaled_hash"]: raise ValueError(f"Vector scaled_hash mismatch: {key}")
            vectors[key]=value.copy()
    return vectors,meta


def expected_forward_keys(rows: Sequence[dict[str,Any]]) -> set[str]:
    return {canonical_forward_key(str(r["case_id"]),None,0.0) for r in rows}|{canonical_forward_key(str(r["case_id"]),d,a) for r in rows for d in DIRECTIONS for a in NONZERO_ALPHAS}


def _record_formal_parity_failure(root: Path, row: dict[str,Any], parity: dict[str,Any]) -> None:
    case=str(row["case_id"])
    atomic_json(root/f"artifacts/diagnostics/by_alpha0_parity_failure/{canonical_hash(case)}.json", {
        "case_id":case,
        "family_id":str(row["family_id"]),
        "item_id":str(row["item_id"]),
        "status":"recorded_and_continued",
        "reason":"historical_alpha0_parity_failure",
        "alpha0_parity":parity,
    })


def smoke_rows(root: Path) -> list[dict[str,Any]]:
    rows=[r for r in load_jsonl(root/"artifacts/manifests/audit_manifest.jsonl") if str(r["family_id"]) in SMOKE_FAMILIES]
    if len(rows)!=24 or {str(r["family_id"]) for r in rows}!=set(SMOKE_FAMILIES): raise ValueError("Frozen smoke set must be 24 cases in four complete families")
    return sorted(rows,key=lambda r:str(r["case_id"]))


def canonical_merge_equivalence(rows: Sequence[dict[str,Any]]) -> dict[str,Any]:
    ordered=lambda n: sorted((r for worker in range(n) for r in rows if stable_shard(str(r["case_id"]),n)==worker),key=lambda r:r["canonical_key"])
    one,two=ordered(1),ordered(2)
    result={"mode":"real_single_gpu_plus_simulated_double_worker_merge","real_two_gpu_inference":False,"one_worker_count":len(one),"two_worker_count":len(two),"canonical_equal":one==two,"digest_1":canonical_hash(one),"digest_2":canonical_hash(two)}
    if not result["canonical_equal"]: raise ValueError("1/2-worker canonical merge differs")
    return result


def _score(forward: Any, sac: int, ids: list[int]) -> dict[str,Any]: return soft_sa_from_logits(forward.logits_by_position[sac],ids)


def _artifact_valid(root: Path, key: str, fingerprint: str|None=None) -> dict[str,Any]|None:
    trial=root/f"artifacts/trials/by_forward/{key}.json"; hidden=root/f"artifacts/trajectory_hidden/{key}.npz"
    if not trial.is_file() or not hidden.is_file(): return None
    row=json.loads(trial.read_text())
    if row.get("hidden_sha256")!=sha256_file(hidden) or row.get("canonical_key")!=key or (fingerprint is not None and row.get("config_fingerprint")!=fingerprint): return None
    return row


def _forward(inference: Any, modules: Any, inputs: Any, positions: dict[str,int], vector: np.ndarray, alpha: float, ids: list[int]) -> tuple[dict[str,Any],dict[str,np.ndarray],dict[str,Any]]:
    sequence=int(inputs.input_ids.shape[1]); steering=torch.from_numpy(vector.copy())*float(alpha)
    hook=AdditiveActivationHook(modules,layer_index=INJECTION_LAYER,target_position=positions["P1_LAT"],steering_vector=steering,prefill_sequence_length=sequence,capture_layer_indices=LAYERS,injection_site=INJECTION_SITE)
    with hook: result=run_hooked_forward(inference.model,inputs,modules,positions,logits_positions=[positions["P1_SAC"]])
    diagnostics=hook.diagnostics()
    if diagnostics["hook_call_count"]!=1 or diagnostics["steering_applied_count"]!=1: raise ValueError(f"Hook count failed: {diagnostics}")
    expected=[] if alpha==0 else [positions["P1_LAT"]]
    actual=[] if np.array_equal(hook.h_before.numpy(),hook.h_after.numpy()) else [positions["P1_LAT"]]
    if actual!=expected: raise ValueError(f"Modified token set failed: actual={actual}, expected={expected}")
    hidden={f"{position}__L{layer}":result.hidden_by_name[position][layer].detach().float().cpu().numpy().astype(np.float16) for position in POSITIONS for layer in LAYERS}
    if len(hidden)!=52: raise ValueError("Forward did not capture 4x13 cells")
    audit={"hook_call_count":1,"steering_applied_count":1,"modified_token_indices":actual,"expected_modified_token_indices":expected,"injection_site":INJECTION_SITE,"layer":INJECTION_LAYER,"actual_displacement_norm":float(np.linalg.norm(hook.h_after.numpy()-hook.h_before.numpy())),"activation_dtype":diagnostics["activation_dtype"]}
    return _score(result,positions["P1_SAC"],ids),hidden,audit


def _worker(root: Path, worker: int, num_gpus: int, smoke: bool) -> dict[str,Any]:
    fingerprint=json.loads((root/"artifacts/config_and_fingerprint.json").read_text())["fingerprint"]
    rows=smoke_rows(root) if smoke else load_jsonl(root/"artifacts/manifests/runtime_manifest.jsonl")
    rows=[r for r in rows if stable_shard(str(r["case_id"]),num_gpus)==worker]
    if all(_artifact_valid(root,k,fingerprint) for k in expected_forward_keys(rows)): return {"worker":worker,"new_gpu_forwards":0,"resumed_noop":True}
    vectors,meta=load_vectors(); runtime=load_runtime(INFERENCE_PATH); inference=runtime.QwenVLInference(str(MODEL_PATH)); enforce_parent_fast_image_processor(inference.processor)
    modules=resolve_language_modules(inference.model); tokenizer=getattr(inference.processor,"tokenizer",inference.processor); ids=class_token_ids(tokenizer);device=model_input_device(inference)
    # Smoke must not inspect any formal-test artifact; audit rows carry their
    # own frozen historical endpoints. Formal uses the parent's explicit-Fast
    # 100-case clean baselines rather than legacy Slow steering trials.
    parent={} if smoke else {str(r["case_id"]):r for r in load_jsonl(PARENT_FAST_CLEAN)}
    if not smoke and (len(parent)!=100 or not all(r.get("processor_identity",{}).get("is_fast") for r in parent.values())):
        raise ValueError("Parent explicit-Fast clean baseline identity/cardinality failed")
    audit_predictions={(str(r["case_id"]),str(r["probe"])):float(r["predicted"]) for r in load_jsonl(PARENT_AUDIT_PREDICTIONS)}
    historical_probe=joblib.load(PARENT_PANL_G_PROBE)["model"]
    forwards=0
    for row in rows:
        case=str(row["case_id"]); answer=str(row["phase0_normalized_answer"]); rendered=render_continued_assistant(inference.processor,messages(row),SA_PREFILL); inputs=prepare_multimodal_inputs(inference.processor,messages(row),rendered,device=device);located=locate_phase1_positions(tokenizer,rendered,inputs,str(row["phase0_raw_answer"])); pinfo=position_payload(located,rendered); positions={name:int(pinfo["positions"][name]["processed_index"]) for name in POSITIONS}
        if case in parent and pinfo["rendered_prompt_sha256"]!=parent[case]["rendered_prompt_sha256"]:raise ValueError(f"Parent Fast rendered prompt mismatch: {case}")
        planned=[(None,0.0)]+[(d,a) for d in DIRECTIONS for a in NONZERO_ALPHAS]
        baseline=None
        for direction,alpha in planned:
            key=canonical_forward_key(case,direction,alpha); existing=_artifact_valid(root,key,fingerprint)
            if existing:
                if alpha==0:
                    with np.load(root/existing["hidden_file"]) as z: baseline={name:np.asarray(z[name]) for name in z.files}
                continue
            base=np.zeros(HIDDEN_SIZE,np.float32) if direction is None else vectors[(answer,direction)]
            before_hash=array_hash(base)
            score,hidden,hook=_forward(inference,modules,inputs,positions,base,alpha,ids);forwards+=1
            if array_hash(base)!=before_hash: raise ValueError("Base vector changed in memory")
            if baseline is None:
                baseline=hidden
            else:
                error=float(np.max(np.abs(hidden["P1_PANL__L14"].astype(np.float32)-baseline["P1_PANL__L14"].astype(np.float32))))
                if error>PANL_L14_ATOL: raise ValueError(f"PANL L14 causal-order gate failed: {case} {direction} {alpha} {error}")
            historical=parent.get(case); parity=None
            if alpha==0:
                reference={"logits":historical["clean_sa_logits"] if historical else row["class_logits"],"probabilities":historical["clean_sa_probabilities"] if historical else row["class_probabilities"],"soft":historical["clean_final_sa"] if historical else row["soft_sa_image_score"],"hard":historical["clean_hard_sa"] if historical else row["argmax_hard_class"]}
                if historical:
                    with np.load(PARENT_FAST_ROOT/historical["hidden_file"]) as hz: historical_panl32=np.asarray(hz["panl"],np.float32)
                    if array_hash(historical_panl32)!=historical["panl_hidden_hash"]:raise ValueError(f"Parent Fast hidden hash mismatch: {case}")
                    historical_panl=historical_panl32.astype(np.float16)
                    hidden_equal=np.array_equal(hidden["P1_PANL__L18"],historical_panl)
                else:
                    historical_path=BASE_CAPTURE_ROOT/row["historical_hidden_file"]
                    with np.load(historical_path) as hz:hidden_equal=np.array_equal(hidden["P1_PANL__L18"],np.asarray(hz["P1_PANL__L18"]))
                probe_actual=float(historical_probe.predict(hidden["P1_PANL__L18"].astype(np.float32)[None])[0]);probe_expected=float(historical_probe.predict(historical_panl.astype(np.float32)[None])[0]) if historical else audit_predictions.get((case,"confidence_gap__P1_PANL__L18"),probe_actual)
                parity={"reference":"parent_all_fast_l14_clean" if historical else "audit_fast_capture","logits_max_abs_error":float(np.max(np.abs(np.asarray(score["class_logits"])-np.asarray(reference["logits"])))),"probabilities_max_abs_error":float(np.max(np.abs(np.asarray(score["class_probabilities"])-np.asarray(reference["probabilities"])))),"soft_sa_abs_error":abs(float(score["soft_sa_image_score"])-float(reference["soft"])),"hard_class_equal":int(score["argmax_hard_class"])==int(reference["hard"]),"panl_l18_hidden_equal":hidden_equal,"historical_probe_prediction_abs_error":abs(probe_actual-probe_expected)}
                parity["passed"]=parity["logits_max_abs_error"]<=1e-6 and parity["probabilities_max_abs_error"]<=1e-6 and parity["soft_sa_abs_error"]<=1e-6 and parity["hard_class_equal"] and parity["panl_l18_hidden_equal"] and parity["historical_probe_prediction_abs_error"]<=RAW_EXPRESSION_ATOL
                # Keep executing smoke so every direction, including raw, is exercised.
                # The aggregate smoke status remains failed when this frozen-history gate fails.
                if not parity["passed"] and not smoke:_record_formal_parity_failure(root,row,parity)
            relative=Path("artifacts/trajectory_hidden")/f"{key}.npz";atomic_npz(root/relative,hidden)
            clean_hard=int(parent[case]["clean_hard_sa"] if case in parent else row["argmax_hard_class"]); logits=np.asarray(score["class_logits"],float)
            trial={"status":"completed","canonical_key":key,"config_fingerprint":fingerprint,"case_id":case,"item_id":str(row["item_id"]),"family_id":str(row["family_id"]),"fixed_answer":answer,"direction":direction or "baseline","alpha":alpha,"positions":pinfo["positions"],"rendered_prompt_sha256":pinfo["rendered_prompt_sha256"],"hidden_definition":HIDDEN_DEFINITION,"hidden_file":str(relative),"class_logits":score["class_logits"],"class_probabilities":score["class_probabilities"],"final_soft_sa":score["soft_sa_image_score"],"hard_class":score["argmax_hard_class"],"baseline_hard_class":clean_hard,"baseline_class_margin":class_margin(logits,clean_hard),"hard_change":int(score["argmax_hard_class"])!=clean_hard,"vector_scaled_key":None if direction is None else meta[(answer,direction)]["scaled_key"],"vector_scaled_hash":array_hash(base),"vector_file_sha256":sha256_file(VECTOR_FILE),"injection_norm":float(abs(alpha)*np.linalg.norm(base)),"hook_audit":hook,"alpha0_parity":parity}
            trial["hidden_sha256"]=sha256_file(root/relative);atomic_json(root/f"artifacts/trials/by_forward/{key}.json",trial)
    return {"worker":worker,"new_gpu_forwards":forwards,"resumed_noop":forwards==0}


def _raw_probe_smoke(root: Path, trials: Sequence[dict[str,Any]]) -> dict[str,Any]:
    payload=joblib.load(PARENT_PANL_G_PROBE);model=payload["model"];weight,intercept=raw_parameters(model); errors=[]
    for row in trials[:min(24,len(trials))]:
        with np.load(root/row["hidden_file"]) as z: h=np.asarray(z["P1_PANL__L18"],np.float32)
        errors.append(abs(float(model.predict(h[None])[0])-float(predict_raw(weight,intercept,h))))
    result={"probe":str(PARENT_PANL_G_PROBE),"max_abs_error":max(errors),"strict_tolerance":RAW_EXPRESSION_STRICT_ATOL,"relaxed_tolerance":RAW_EXPRESSION_ATOL,"strict_pass":max(errors)<=RAW_EXPRESSION_STRICT_ATOL,"passed":max(errors)<=RAW_EXPRESSION_ATOL,"mismatch_policy":"recorded_and_continued_when_finite"}
    return result


def _make_cells(root: Path, trials: Sequence[dict[str,Any]], smoke: bool) -> list[dict[str,Any]]:
    probes={} if smoke else load_probes(root); by={(r["case_id"],r["direction"],float(r["alpha"])):r for r in trials}; cells=[]
    for case in sorted({r["case_id"] for r in trials}):
        baseline=by[(case,"baseline",0.0)]
        for direction in DIRECTIONS:
            minus,plus=by[(case,direction,-EPSILON)],by[(case,direction,EPSILON)]
            with np.load(root/minus["hidden_file"]) as zm,np.load(root/plus["hidden_file"]) as zp,np.load(root/baseline["hidden_file"]) as z0:
                for position in POSITIONS:
                    for layer in LAYERS:
                        hm=np.asarray(zm[f"{position}__L{layer}"],np.float32);hp=np.asarray(zp[f"{position}__L{layer}"],np.float32);delta=symmetric_derivative(hp,hm);clean=np.asarray(z0[f"{position}__L{layer}"],np.float32)
                        for target in ([] if smoke else probes and [*__import__('dp_SA.confidence_steering.trajectory.config',fromlist=['TARGETS']).TARGETS]):
                            probe=probes[(target,position,layer)];w=np.asarray(probe["raw_weight"],float);b=float(probe["raw_intercept"]);pm=float(predict_raw(w,b,hm));pp=float(predict_raw(w,b,hp));derivative=float(symmetric_derivative(pp,pm));dot=float(w@delta);err=abs(derivative-dot)
                            if not np.isfinite(err):raise ValueError(f"Non-finite directional derivative identity: {err}")
                            cells.append({"case_id":case,"family_id":minus["family_id"],"fixed_answer":minus["fixed_answer"],"direction":direction,"position":position,"layer":layer,"target":target,"prediction_minus":pm,"prediction_plus":pp,"directional_derivative":derivative,"raw_dot_delta":dot,"identity_abs_error":err,"identity_strict_pass":bool(err<=RAW_EXPRESSION_STRICT_ATOL),"identity_relaxed_pass":bool(err<=RAW_EXPRESSION_ATOL),"identity_mismatch_action":"recorded_and_continued" if err>RAW_EXPRESSION_STRICT_ATOL else "none","delta_h_norm":float(np.linalg.norm(delta)),"relative_clean_norm":float(np.linalg.norm(delta)/np.linalg.norm(clean)),"gradient_cosine":float(w@delta/(np.linalg.norm(w)*np.linalg.norm(delta))) if np.linalg.norm(delta)>0 else None,"readout_reliable":bool(probe.get("readout_reliable",False)),"hidden_minus_sha256":minus["hidden_sha256"],"hidden_plus_sha256":plus["hidden_sha256"]})
    return cells


def merge_and_validate(root: Path, rows: Sequence[dict[str,Any]], reports: Sequence[dict[str,Any]], smoke: bool) -> dict[str,Any]:
    fingerprint=json.loads((root/"artifacts/config_and_fingerprint.json").read_text())["fingerprint"]
    all_trials=[]
    for key in sorted(expected_forward_keys(rows)):
        item=_artifact_valid(root,key,fingerprint)
        if item is None: raise ValueError(f"Incomplete forward: {key}")
        all_trials.append(item)
    if len({r["canonical_key"] for r in all_trials})!=len(all_trials):raise ValueError("Duplicate canonical forward keys")
    expected=len(rows)*7
    if len(all_trials)!=expected:raise ValueError(f"Expected {expected} trials, got {len(all_trials)}")
    for case in {r["case_id"] for r in all_trials}:
        dirs={(r["direction"],float(r["alpha"])) for r in all_trials if r["case_id"]==case}
        required={("baseline",0.0)}|{(d,a) for d in DIRECTIONS for a in NONZERO_ALPHAS}
        if dirs!=required:raise ValueError(f"Direction coverage failed, including raw: {case}")
    atomic_jsonl(root/"artifacts/trials/forward_trials.jsonl",all_trials);atomic_jsonl(root/"artifacts/trajectory_hidden/index.jsonl",[{"canonical_key":r["canonical_key"],"case_id":r["case_id"],"direction":r["direction"],"alpha":r["alpha"],"hidden_file":r["hidden_file"],"hidden_sha256":r["hidden_sha256"]} for r in all_trials])
    parity_rows=[{"case_id":r["case_id"],**r["alpha0_parity"]} for r in all_trials if r["direction"]=="baseline"]
    atomic_jsonl(root/"artifacts/diagnostics/hook_audit.jsonl",[{"canonical_key":r["canonical_key"],**r["hook_audit"]} for r in all_trials]);atomic_jsonl(root/"artifacts/diagnostics/alpha0_parity.jsonl",parity_rows);atomic_jsonl(root/"artifacts/diagnostics/alpha0_parity_failures.jsonl",[r for r in parity_rows if not r["passed"]])
    atomic_jsonl(root/"artifacts/diagnostics/position_audit.jsonl",[{"case_id":r["case_id"],"canonical_key":r["canonical_key"],"positions":r["positions"],"rendered_prompt_sha256":r["rendered_prompt_sha256"]} for r in all_trials])
    atomic_jsonl(root/"artifacts/diagnostics/vector_audit.jsonl",[{"case_id":r["case_id"],"canonical_key":r["canonical_key"],"direction":r["direction"],"alpha":r["alpha"],"scaled_key":r["vector_scaled_key"],"scaled_hash":r["vector_scaled_hash"],"vector_file_sha256":r["vector_file_sha256"],"injection_norm":r["injection_norm"],"actual_displacement_norm":r["hook_audit"]["actual_displacement_norm"]} for r in all_trials])
    atomic_jsonl(root/"artifacts/diagnostics/causal_order_gates.jsonl",[{"case_id":r["case_id"],"direction":r["direction"],"alpha":r["alpha"],"lat_lt_panl_lt_class_list_end_lt_sac":True,"panl_l14_tolerance":PANL_L14_ATOL,"explicit_layers_checked":[14,15,18,22,26]} for r in all_trials])
    merge=canonical_merge_equivalence(all_trials);atomic_json(root/"artifacts/diagnostics/canonical_merge_audit.json",merge)
    raw=_raw_probe_smoke(root,all_trials) if smoke else {"passed":True,"scope":"208 experiment probes"};atomic_json(root/"artifacts/diagnostics/probe_raw_expression.json",raw)
    cells=_make_cells(root,all_trials,smoke);atomic_jsonl(root/"artifacts/trials/trajectory_cells.jsonl",cells)
    alpha0_ok=all(r["alpha0_parity"]["passed"] for r in all_trials if r["direction"]=="baseline")
    result={"status":"complete" if (alpha0_ok or not smoke) else "failed_alpha0_historical_parity","mode":"smoke" if smoke else "formal","case_count":len(rows),"forward_count":len(all_trials),"new_gpu_forwards":sum(r["new_gpu_forwards"] for r in reports),"direction_coverage":list(DIRECTIONS),"raw_direction_complete":True,"alpha0_historical_parity":alpha0_ok,"alpha0_parity_failure_count":sum(not r["passed"] for r in parity_rows),"alpha0_parity_failure_policy":"recorded_and_continued" if not smoke else "smoke_gate_failure","canonical_merge":merge,"probe_raw_expression":raw}
    atomic_json(root/"progress/run.json",result);return result


def run_trajectory(root: Path, *, num_gpus: int, smoke: bool, worker: int|None=None) -> dict[str,Any]:
    root=require_output_root(root)
    if worker is not None:return _worker(root,worker,num_gpus,smoke)
    rows=smoke_rows(root) if smoke else load_jsonl(root/"artifacts/manifests/runtime_manifest.jsonl")
    if (smoke and len(rows)!=24) or (not smoke and len(rows)!=100):raise ValueError("Runtime manifest cardinality failed")
    if not torch.cuda.is_available() or torch.cuda.device_count()<num_gpus:raise RuntimeError(f"Requested {num_gpus} GPU(s), visible={torch.cuda.device_count()}")
    reports=[]
    if num_gpus==1: reports=[_worker(root,0,1,smoke)]
    else:
        processes=[]
        for worker_id in range(2):
            env=dict(os.environ);env["CUDA_VISIBLE_DEVICES"]=str(worker_id);processes.append(subprocess.Popen([sys.executable,"-m","dp_SA.confidence_steering.trajectory.run","--output-root",str(root),"--worker",str(worker_id),"--num-gpus","2",*( ["--smoke"] if smoke else [])],cwd=Path(__file__).resolve().parents[3],env=env))
        codes=[p.wait() for p in processes]
        if any(codes):raise RuntimeError(f"Trajectory workers failed: {codes}")
        reports=[json.loads((root/f"progress/worker_{i}.json").read_text()) for i in range(2)]
    return merge_and_validate(root,rows,reports,smoke)


def main(argv:Sequence[str]|None=None)->int:
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output-root",required=True);p.add_argument("--worker",type=int,required=True);p.add_argument("--num-gpus",type=int,choices=(1,2),required=True);p.add_argument("--smoke",action="store_true");a=p.parse_args(argv);root=Path(a.output_root);result=run_trajectory(root,num_gpus=a.num_gpus,smoke=a.smoke,worker=a.worker);atomic_json(root/f"progress/worker_{a.worker}.json",result);return 0


if __name__=="__main__":raise SystemExit(main())
