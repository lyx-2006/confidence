from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dp_SA.config import MODEL_PATH, ROOT
from dp_SA.confidence_steering.config import RIDGE_ALPHA_GRID

from .capture import stable_shard
from .io_utils import (array_hash, atomic_csv, atomic_json, atomic_jsonl, atomic_npz,
                       canonical_hash, inventory, load_jsonl, sha256_file, verify_inventory)
from .runtime import capture_and_score, load_inference, prepare_case
from .templates import TEMPLATES


PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_PARENT = PACKAGE_ROOT / "output"
DEFAULT_ROOT = OUTPUT_PARENT / "template_sa_probes"
DEFAULT_SMOKE_ROOT = OUTPUT_PARENT / "template_sa_probes_smoke"
TRAJECTORY_ROOT = ROOT / "dp_SA/confidence_steering/trajectory/output/results"
CONSTRUCTION_SOURCE = TRAJECTORY_ROOT / "artifacts/manifests/construction_manifest.jsonl"
AUDIT_SOURCE = TRAJECTORY_ROOT / "artifacts/manifests/audit_manifest.jsonl"
T0_PROBE_INDEX = TRAJECTORY_ROOT / "artifacts/probes/probe_index.jsonl"
T0_AUDIT_PREDICTIONS = TRAJECTORY_ROOT / "artifacts/probes/audit_predictions.jsonl"
SPLIT_AUDIT_SOURCE = ROOT / "dp_SA/unimodal_logit_confidence/output/results/shared/split_audit.json"

DEFAULT_TEMPLATES = ("T1", "T2", "T3")
DEFAULT_POSITIONS = ("P1_LAT", "P1_PANL")
DEFAULT_LAYERS = (10, 12, 14, 15, 16, 17)
T0_NEW_LAYERS = (10, 12)
T0_FROZEN_LAYERS = (14, 15, 16, 17)
RIDGE_ALPHAS = tuple(float(x) for x in RIDGE_ALPHA_GRID)
SEED = 42
BOOTSTRAPS = 2000
SMOKE_BOOTSTRAPS = 200
HIDDEN_SIZE = 3584


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, float); prediction = np.asarray(prediction, float)
    return {
        "r2": float(r2_score(y, prediction)),
        "pearson": float(pearsonr(y, prediction).statistic) if len(y) > 1 and np.ptp(y) and np.ptp(prediction) else math.nan,
        "spearman": float(spearmanr(y, prediction).statistic) if len(np.unique(y)) > 1 and len(np.unique(prediction)) > 1 else math.nan,
        "mae": float(mean_absolute_error(y, prediction)),
    }


def _pipeline(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=float(alpha), solver="lsqr"))])


def choose_alpha(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[float, np.ndarray, list[dict[str, float]]]:
    if sorted(set(map(int, folds))) != [1, 2, 3, 4]:
        raise ValueError("Construction folds must be exactly 1..4")
    traces=[]; predictions={}
    for alpha in RIDGE_ALPHAS:
        oof=np.full(len(y), np.nan, float)
        for fold in (1, 2, 3, 4):
            train=folds != fold; validation=~train
            model=_pipeline(alpha); model.fit(x[train], y[train]); oof[validation]=model.predict(x[validation])
        score=float(r2_score(y, oof)); traces.append({"alpha":alpha,"oof_r2":score});predictions[alpha]=oof
    # This is the historical trajectory rule: smaller alpha wins an exact tie.
    selected=float(max(traces,key=lambda row:(row["oof_r2"],-row["alpha"]))["alpha"])
    return selected,predictions[selected],traces


def _raw_parameters(model: Pipeline) -> tuple[np.ndarray, float]:
    scaler=model.named_steps["scale"];ridge=model.named_steps["ridge"]
    weight=np.asarray(ridge.coef_,np.float64).reshape(-1)/np.asarray(scaler.scale_,np.float64)
    intercept=float(np.asarray(ridge.intercept_).reshape(-1)[0] if np.asarray(ridge.intercept_).ndim else ridge.intercept_)-float(weight@np.asarray(scaler.mean_,np.float64))
    return weight,intercept


def _atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True);fd,temporary=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent);os.close(fd)
    try:joblib.dump(payload,temporary);os.replace(temporary,path)
    except Exception:
        try:os.unlink(temporary)
        except FileNotFoundError:pass
        raise


def _family_design(rows: Sequence[dict[str, Any]], repeats: int) -> tuple[list[str], np.ndarray]:
    families=sorted({str(r["family_id"]) for r in rows})
    return families,np.random.default_rng(SEED).integers(0,len(families),size=(repeats,len(families)))


def _bootstrap(y: np.ndarray, prediction: np.ndarray, row_families: Sequence[str], families: Sequence[str], draws: np.ndarray) -> dict[str, tuple[float,float]]:
    family_array=np.asarray(list(map(str,row_families)));by={f:np.flatnonzero(family_array==f) for f in families};values={k:[] for k in ("r2","pearson","spearman","mae")}
    for draw in draws:
        idx=np.concatenate([by[families[int(i)]] for i in draw]); measured=_metrics(y[idx],prediction[idx])
        for name,value in measured.items():
            if np.isfinite(value):values[name].append(value)
    return {name:tuple(map(float,np.quantile(vals,[.025,.975]))) if vals else (math.nan,math.nan) for name,vals in values.items()}


def _source_inventory() -> dict[str,str]:
    paths=[CONSTRUCTION_SOURCE,AUDIT_SOURCE,SPLIT_AUDIT_SOURCE,T0_PROBE_INDEX,T0_AUDIT_PREDICTIONS,MODEL_PATH/"config.json",MODEL_PATH/"model.safetensors.index.json",MODEL_PATH/"tokenizer.json",MODEL_PATH/"tokenizer_config.json",MODEL_PATH/"preprocessor_config.json"]
    for row in load_jsonl(T0_PROBE_INDEX):
        if row.get("target")=="final_soft_sa" and row.get("position") in DEFAULT_POSITIONS and int(row.get("layer",-1)) in T0_FROZEN_LAYERS:
            paths.append(TRAJECTORY_ROOT/row["probe_file"])
    return inventory(paths)


def _select_smoke(construction: Sequence[dict[str,Any]],audit: Sequence[dict[str,Any]]) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    selected=[]
    for fold in (1,2,3,4):
        family=sorted({str(r["family_id"]) for r in construction if int(r["outer_fold"])==fold})[0]
        selected.extend(r for r in construction if str(r["family_id"])==family)
    audit_families=sorted({str(r["family_id"]) for r in audit})[:2]
    return selected,[r for r in audit if str(r["family_id"]) in audit_families]


def prepare(root: Path, *, templates: Sequence[str], positions: Sequence[str], layers: Sequence[int], smoke: bool, resume: bool) -> tuple[list[dict[str,Any]],list[dict[str,Any]],str,dict[str,str]]:
    construction=load_jsonl(CONSTRUCTION_SOURCE);audit=load_jsonl(AUDIT_SOURCE)
    if (len(construction),len(audit),len({r['family_id'] for r in construction}),len({r['family_id'] for r in audit}))!=(882,230,103,25):raise ValueError("Frozen T0 882/230 split changed")
    overlaps={field:len({str(r[field]) for r in construction}&{str(r[field]) for r in audit}) for field in ("case_id","family_id","item_id","image_sha256")}
    if any(overlaps.values()):raise ValueError(f"Construction/audit leakage: {overlaps}")
    by_family={}
    for row in construction+audit:by_family.setdefault(str(row["family_id"]),set()).add(int(row["outer_fold"]))
    if any(len(v)!=1 for v in by_family.values()):raise ValueError("Family spans outer folds")
    full_counts={"construction_cases":882,"construction_families":103,"audit_cases":230,"audit_families":25,"overlaps":overlaps,"audit_used_for_fit":False}
    if smoke:construction,audit=_select_smoke(construction,audit)
    sources=_source_inventory();config={"format_version":1,"experiment":"t0_t3_lat_panl_sa_probes","templates":list(templates),"positions":list(positions),"layers":list(map(int,layers)),"t0_new_layers":list(T0_NEW_LAYERS),"t0_frozen_layers":list(T0_FROZEN_LAYERS),"ridge_alphas":list(RIDGE_ALPHAS),"seed":SEED,"bootstrap_repeats":SMOKE_BOOTSTRAPS if smoke else BOOTSTRAPS,"smoke_only":smoke,"target":"template_own_canonical_soft_sa","hidden_definition":"decoder_block_output_pre_final_norm","source_hashes":sources,"template_hashes":{t:hashlib.sha256(TEMPLATES[t].template.encode()).hexdigest() for t in ("T0",*templates)}};fingerprint=canonical_hash(config)
    config_path=root/"artifacts/config_and_fingerprint.json"
    if config_path.exists():
        old=json.loads(config_path.read_text())
        if old.get("fingerprint")!=fingerprint:raise ValueError("Resume configuration fingerprint mismatch")
        if not resume:raise FileExistsError(config_path)
    root.mkdir(parents=True,exist_ok=True);atomic_json(config_path,{**config,"fingerprint":fingerprint});atomic_jsonl(root/"artifacts/manifests/construction_manifest.jsonl",construction);atomic_jsonl(root/"artifacts/manifests/audit_manifest.jsonl",audit);atomic_json(root/"artifacts/manifests/split_audit.json",{**full_counts,"written_construction_cases":len(construction),"written_audit_cases":len(audit),"smoke_only":smoke})
    return construction,audit,fingerprint,sources


def _hidden_key(position: str, layer: int) -> str:return f"{position}__L{int(layer)}"


def capture(root: Path, records: Sequence[dict[str,Any]], *, templates: Sequence[str], positions: Sequence[str], layers: Sequence[int], fingerprint: str, resume: bool, on_template_complete: Callable[[str],None]|None=None) -> dict[str,Any]:
    path=root/"artifacts/diagnostics/capture.worker_0.jsonl";existing={(r["template"],str(r["case_id"])):r for r in load_jsonl(path) if r.get("status")=="completed" and r.get("config_fingerprint")==fingerprint}
    expected={(t,str(r["case_id"])) for t in ("T0",*templates) for r in records}
    if expected.issubset(existing):
        rows=[]
        for template in ("T0",*templates):
            template_rows=[existing[(template,str(record["case_id"]))] for record in records];rows.extend(template_rows);atomic_jsonl(root/f"artifacts/diagnostics/capture.{template}.jsonl",template_rows)
            if on_template_complete is not None:on_template_complete(template)
        atomic_jsonl(root/"artifacts/diagnostics/capture_manifest.jsonl",rows)
        return {"status":"complete","row_count":len(expected),"new_gpu_forwards":0,"resumed_noop":True}
    inference,modules,tokenizer,device,processor=load_inference();started=time.time();forwards=0
    for template in ("T0",*templates):
        wanted_layers=tuple(l for l in layers if template!="T0" or l in T0_NEW_LAYERS)
        requested={p:wanted_layers for p in positions}
        for record in records:
            identity=(template,str(record["case_id"]));required={_hidden_key(p,l) for p in positions for l in wanted_layers};old=existing.get(identity)
            if old and required==set(old.get("hidden_keys",[])):
                if not resume:raise FileExistsError(identity)
                continue
            inputs,rendered,located=prepare_case(inference.processor,tokenizer,device,record,TEMPLATES[template]);hidden,score=capture_and_score(inference.model,modules,tokenizer,inputs,located,TEMPLATES[template],requested,score_required=template!="T0");forwards+=6 if template=="T3" else 1
            if any(v.shape!=(HIDDEN_SIZE,) or v.dtype!=np.float16 or not np.isfinite(v).all() for v in hidden.values()):raise ValueError(f"Invalid hidden: {identity}")
            relative=Path("artifacts/hidden")/template/f"{record['case_id']}.npz";atomic_npz(root/relative,hidden)
            own_sa=float(record["final_soft_sa"]) if template=="T0" else float(score["canonical_soft_sa"])
            row={"status":"completed","template":template,"case_id":str(record["case_id"]),"family_id":str(record["family_id"]),"item_id":str(record["item_id"]),"outer_fold":int(record["outer_fold"]),"canonical_soft_sa":own_sa,"positions":located,"hidden_file":str(relative),"hidden_sha256":sha256_file(root/relative),"hidden_keys":sorted(hidden),"hidden_tensor_sha256":{k:array_hash(v) for k,v in hidden.items()},"rendered_prompt_sha256":hashlib.sha256(rendered.encode()).hexdigest(),"processor":processor,"config_fingerprint":fingerprint,**({} if template=="T0" else score)}
            existing[identity]=row;atomic_jsonl(path,sorted(existing.values(),key=lambda r:(r["template"],r["case_id"])))
            if forwards%10==0:atomic_json(root/"progress/capture_worker_0.json",{"status":"running","new_gpu_forwards":forwards,"completed_rows":len(existing),"last":identity,"elapsed_seconds":time.time()-started})
        template_rows=[existing[(template,str(record["case_id"]))] for record in records]
        atomic_jsonl(root/f"artifacts/diagnostics/capture.{template}.jsonl",template_rows)
        atomic_json(root/f"progress/capture_{template}.json",{"status":"complete","template":template,"row_count":len(template_rows),"completed_at_unix":time.time()})
        if on_template_complete is not None:on_template_complete(template)
    rows=[existing[k] for k in sorted(expected)];atomic_jsonl(root/"artifacts/diagnostics/capture_manifest.jsonl",rows);result={"status":"complete","row_count":len(rows),"new_gpu_forwards":forwards,"resumed_noop":forwards==0,"elapsed_seconds":time.time()-started};atomic_json(root/"progress/capture.json",result);return result


def _load_hidden(root: Path,row: dict[str,Any],position: str,layer: int) -> np.ndarray:
    with np.load(root/row["hidden_file"]) as data:return np.asarray(data[_hidden_key(position,layer)],np.float32)


def _frozen_t0() -> dict[tuple[str,int],dict[str,Any]]:
    selected={}
    for row in load_jsonl(T0_PROBE_INDEX):
        key=(str(row.get("position")),int(row.get("layer",-1)))
        if row.get("target")=="final_soft_sa" and key[0] in DEFAULT_POSITIONS and key[1] in T0_FROZEN_LAYERS:
            path=TRAJECTORY_ROOT/row["probe_file"]
            if sha256_file(path)!=row["probe_sha256"]:raise ValueError(f"Frozen T0 probe hash mismatch: {path}")
            payload=joblib.load(path)
            if payload.get("target")!="final_soft_sa" or payload.get("position")!=key[0] or int(payload.get("layer",-1))!=key[1]:raise ValueError("Frozen T0 probe identity mismatch")
            selected[key]={"index":row,"payload":payload,"path":path}
    if len(selected)!=8:raise ValueError(f"Expected 8 frozen T0 references, got {len(selected)}")
    return selected


def _ensure_bootstrap_draws(root: Path, audit: Sequence[dict[str,Any]], smoke: bool) -> tuple[list[str],np.ndarray]:
    path=root/"artifacts/bootstrap_draws.json";families,draws=_family_design(audit,SMOKE_BOOTSTRAPS if smoke else BOOTSTRAPS)
    payload={"seed":SEED,"repeats":len(draws),"ordered_families":families,"draws":draws.tolist(),"shared_by_all_probes":True,"fingerprint":canonical_hash(draws.tolist())}
    if path.exists():
        old=json.loads(path.read_text())
        if old.get("fingerprint")!=payload["fingerprint"]:raise ValueError("Bootstrap draw fingerprint mismatch")
    else:atomic_json(path,payload)
    return families,draws


def train_template_shard(root: Path, template: str, *, positions: Sequence[str], layers: Sequence[int], fingerprint: str, smoke: bool, resume: bool) -> dict[str,Any]:
    """Fit one template's new probes without touching another template's outputs."""
    construction=load_jsonl(root/"artifacts/manifests/construction_manifest.jsonl");audit=load_jsonl(root/"artifacts/manifests/audit_manifest.jsonl")
    capture_path=root/f"artifacts/diagnostics/capture.{template}.jsonl"
    capture_rows={(r["template"],str(r["case_id"])):r for r in load_jsonl(capture_path)}
    expected={(template,str(r["case_id"])) for r in (*construction,*audit)}
    if set(capture_rows)!=expected:raise ValueError(f"Incomplete {template} capture shard: {len(capture_rows)}/{len(expected)}")
    families,draws=_ensure_bootstrap_draws(root,audit,smoke);folds=np.asarray([int(r["outer_fold"]) for r in construction]);audit_family_ids=[str(r["family_id"]) for r in audit]
    construction_ids=[str(r["case_id"]) for r in construction];audit_ids=[str(r["case_id"]) for r in audit]
    grids=[(template,p,l) for p in positions for l in layers if template!="T0" or l in T0_NEW_LAYERS]
    shard=root/"artifacts/probe_shards"/template;done=shard/"completion.json"
    if resume and done.exists():
        prior=json.loads(done.read_text())
        if prior.get("config_fingerprint")==fingerprint and prior.get("cell_count")==len(grids):return {**prior,"resumed_noop":True}
    index=[];metrics=[];audit_predictions=[];oof_predictions=[];new_count=0
    for _,position,layer in grids:
        name=f"canonical_soft_sa__{template}__{position}__L{layer}";path=root/f"artifacts/probes/{name}.joblib"
        x_train=np.stack([_load_hidden(root,capture_rows[template,str(r["case_id"])],position,layer) for r in construction]);x_audit=np.stack([_load_hidden(root,capture_rows[template,str(r["case_id"])],position,layer) for r in audit]);y_train=np.asarray([capture_rows[template,str(r["case_id"])]["canonical_soft_sa"] for r in construction],float);y_audit=np.asarray([capture_rows[template,str(r["case_id"])]["canonical_soft_sa"] for r in audit],float)
        if resume and path.exists():
            payload=joblib.load(path)
            if payload.get("config_fingerprint")!=fingerprint or payload.get("construction_case_ids")!=construction_ids:raise ValueError(f"Existing probe identity mismatch: {path}")
            model=payload["model"];alpha=float(payload["alpha"]);trace=payload["alpha_trace"]
            selected,oof,_=choose_alpha(x_train,y_train,folds)
            if selected!=alpha:raise ValueError(f"Resumed alpha mismatch: {path}")
        else:
            alpha,oof,trace=choose_alpha(x_train,y_train,folds);model=_pipeline(alpha);model.fit(x_train,y_train);weight,intercept=_raw_parameters(model);payload={"model":model,"target":"canonical_soft_sa","template":template,"position":position,"layer":layer,"alpha":alpha,"alpha_trace":trace,"raw_weight":weight,"raw_intercept":intercept,"raw_weight_sha256":array_hash(weight),"construction_case_ids":construction_ids,"audit_case_ids":audit_ids,"audit_used_for_fit":False,"config_fingerprint":fingerprint};_atomic_joblib(path,payload);new_count+=1
        prediction=np.asarray(model.predict(x_audit),float);measured=_metrics(y_audit,prediction);oof_metrics=_metrics(y_train,oof);ci=_bootstrap(y_audit,prediction,audit_family_ids,families,draws)
        metric={"template":template,"position":position,"layer":layer,"artifact_role":"newly_fitted","alpha":alpha,"construction_case_count":len(construction),"audit_case_count":len(audit),"audit_used_for_fit":False,**{f"construction_oof_{k}":v for k,v in oof_metrics.items()},**{f"audit_{k}":v for k,v in measured.items()}}
        for k,(lo,hi) in ci.items():metric[f"audit_{k}_ci_low"]=lo;metric[f"audit_{k}_ci_high"]=hi
        metrics.append(metric);index.append({**metric,"probe_file":str(path.relative_to(root)),"probe_sha256":sha256_file(path)})
        audit_predictions.extend({"case_id":r["case_id"],"family_id":r["family_id"],"template":template,"position":position,"layer":layer,"actual_canonical_soft_sa":float(y),"predicted_canonical_soft_sa":float(p)} for r,y,p in zip(audit,y_audit,prediction));oof_predictions.extend({"case_id":r["case_id"],"family_id":r["family_id"],"outer_fold":r["outer_fold"],"template":template,"position":position,"layer":layer,"actual_canonical_soft_sa":float(y),"predicted_canonical_soft_sa":float(p)} for r,y,p in zip(construction,y_train,oof))
        atomic_json(root/f"progress/train_{template}.json",{"status":"running","template":template,"completed_cells":len(index),"expected_cells":len(grids),"newly_fitted":new_count})
    atomic_jsonl(shard/"probe_index.jsonl",index);atomic_jsonl(shard/"metrics.jsonl",metrics);atomic_jsonl(shard/"audit_predictions.jsonl",audit_predictions);atomic_jsonl(shard/"oof_predictions.jsonl",oof_predictions)
    result={"status":"complete","template":template,"cell_count":len(grids),"newly_fitted_this_run":new_count,"config_fingerprint":fingerprint,"completed_at_unix":time.time(),"resumed_noop":False};atomic_json(done,result);atomic_json(root/f"progress/train_{template}.json",result);return result


def _aggregate_shards(root: Path, construction: Sequence[dict[str,Any]], audit: Sequence[dict[str,Any]], *, templates: Sequence[str], positions: Sequence[str], layers: Sequence[int], smoke: bool) -> dict[str,Any]:
    families,draws=_ensure_bootstrap_draws(root,audit,smoke);audit_ids=[str(r["case_id"]) for r in audit];audit_family_ids=[str(r["family_id"]) for r in audit]
    index=[];metrics=[];audit_predictions=[];oof_predictions=[]
    for template in ("T0",*templates):
        shard=root/"artifacts/probe_shards"/template
        index.extend(load_jsonl(shard/"probe_index.jsonl"));metrics.extend(load_jsonl(shard/"metrics.jsonl"));audit_predictions.extend(load_jsonl(shard/"audit_predictions.jsonl"));oof_predictions.extend(load_jsonl(shard/"oof_predictions.jsonl"))
    frozen=_frozen_t0();historical_predictions=load_jsonl(T0_AUDIT_PREDICTIONS)
    for position in positions:
        for layer in layers:
            if layer not in T0_FROZEN_LAYERS:continue
            source=frozen[position,layer];row=source["index"];pred=[r for r in historical_predictions if r["target"]=="final_soft_sa" and r["position"]==position and int(r["layer"])==layer];keyed={str(r["case_id"]):r for r in pred};ordered=[keyed[x] for x in audit_ids];y=np.asarray([float(r["actual"]) for r in ordered]);p=np.asarray([float(r["predicted"]) for r in ordered]);measured=_metrics(y,p);ci=_bootstrap(y,p,audit_family_ids,families,draws);trace=source["payload"]["alpha_trace"];selected=float(source["payload"]["alpha"]);oof_r2=next(float(r["oof_r2"]) for r in trace if float(r["alpha"])==selected)
            metric={"template":"T0","position":position,"layer":layer,"artifact_role":"frozen_t0_reference","alpha":selected,"construction_case_count":len(construction),"audit_case_count":len(audit),"audit_used_for_fit":False,"construction_oof_r2":oof_r2,"construction_oof_pearson":math.nan,"construction_oof_spearman":math.nan,"construction_oof_mae":math.nan,**{f"audit_{k}":v for k,v in measured.items()}}
            for k,(lo,hi) in ci.items():metric[f"audit_{k}_ci_low"]=lo;metric[f"audit_{k}_ci_high"]=hi
            metrics.append(metric);index.append({**metric,"probe_file":str(source["path"].resolve()),"probe_sha256":row["probe_sha256"]});audit_predictions.extend({"case_id":r["case_id"],"family_id":r["family_id"],"template":"T0","position":position,"layer":layer,"actual_canonical_soft_sa":float(a),"predicted_canonical_soft_sa":float(b)} for r,a,b in zip(audit,y,p))
    index.sort(key=lambda r:(r["template"],r["position"],int(r["layer"])));metrics.sort(key=lambda r:(r["template"],r["position"],int(r["layer"])));atomic_jsonl(root/"artifacts/probes/probe_index.jsonl",index);atomic_csv(root/"tables/probe_metrics.csv",metrics);atomic_csv(root/"artifacts/audit_predictions.csv",audit_predictions);atomic_csv(root/"artifacts/construction_oof_predictions.csv",oof_predictions)
    comparisons=[];by={(r["template"],r["position"],int(r["layer"])):r for r in metrics}
    for template in templates:
        for position in positions:
            for layer in layers:
                t0=by["T0",position,layer];tx=by[template,position,layer];comparisons.append({"template":template,"position":position,"layer":layer,**{f"audit_{k}_difference_vs_t0":float(tx[f'audit_{k}'])-float(t0[f'audit_{k}']) for k in ("r2","pearson","spearman","mae")}})
    atomic_csv(root/"tables/t0_tx_probe_comparison.csv",comparisons);return {"status":"complete","index_cells":len(index),"new_probe_cells":40,"frozen_t0_references":8,"bootstrap_repeats":len(draws)}


def train(root: Path, construction: Sequence[dict[str,Any]], audit: Sequence[dict[str,Any]], *, templates: Sequence[str], positions: Sequence[str], layers: Sequence[int], fingerprint: str, smoke: bool, resume: bool) -> dict[str,Any]:
    capture_rows={(r["template"],str(r["case_id"])):r for r in load_jsonl(root/"artifacts/diagnostics/capture_manifest.jsonl")};folds=np.asarray([int(r["outer_fold"]) for r in construction]);families,draws=_family_design(audit,SMOKE_BOOTSTRAPS if smoke else BOOTSTRAPS);atomic_json(root/"artifacts/bootstrap_draws.json",{"seed":SEED,"repeats":len(draws),"ordered_families":families,"draws":draws.tolist(),"shared_by_all_probes":True,"fingerprint":canonical_hash(draws.tolist())})
    index=[];metrics=[];audit_predictions=[];oof_predictions=[];new_count=0
    construction_ids=[str(r["case_id"]) for r in construction];audit_ids=[str(r["case_id"]) for r in audit];audit_family_ids=[str(r["family_id"]) for r in audit]
    frozen=_frozen_t0()
    # New T0 L10/L12 plus every requested Tx cell.
    grids=[("T0",p,l) for p in positions for l in layers if l in T0_NEW_LAYERS]+[(t,p,l) for t in templates for p in positions for l in layers]
    for template,position,layer in grids:
        name=f"canonical_soft_sa__{template}__{position}__L{layer}";path=root/f"artifacts/probes/{name}.joblib"
        x_train=np.stack([_load_hidden(root,capture_rows[template,str(r["case_id"])],position,layer) for r in construction]);x_audit=np.stack([_load_hidden(root,capture_rows[template,str(r["case_id"])],position,layer) for r in audit]);y_train=np.asarray([capture_rows[template,str(r["case_id"])]["canonical_soft_sa"] for r in construction],float);y_audit=np.asarray([capture_rows[template,str(r["case_id"])]["canonical_soft_sa"] for r in audit],float)
        if resume and path.exists():
            payload=joblib.load(path)
            if payload.get("config_fingerprint")!=fingerprint or payload.get("construction_case_ids")!=construction_ids:raise ValueError(f"Existing probe identity mismatch: {path}")
            model=payload["model"];alpha=float(payload["alpha"]);trace=payload["alpha_trace"]
            # Recreate OOF deterministically for complete output tables.
            _,oof,_=choose_alpha(x_train,y_train,folds)
        else:
            alpha,oof,trace=choose_alpha(x_train,y_train,folds);model=_pipeline(alpha);model.fit(x_train,y_train);weight,intercept=_raw_parameters(model);payload={"model":model,"target":"canonical_soft_sa","template":template,"position":position,"layer":layer,"alpha":alpha,"alpha_trace":trace,"raw_weight":weight,"raw_intercept":intercept,"raw_weight_sha256":array_hash(weight),"construction_case_ids":construction_ids,"audit_case_ids":audit_ids,"audit_used_for_fit":False,"config_fingerprint":fingerprint};_atomic_joblib(path,payload);new_count+=1
        prediction=np.asarray(model.predict(x_audit),float);measured=_metrics(y_audit,prediction);oof_metrics=_metrics(y_train,oof);ci=_bootstrap(y_audit,prediction,audit_family_ids,families,draws)
        metric={"template":template,"position":position,"layer":layer,"artifact_role":"newly_fitted","alpha":alpha,"construction_case_count":len(construction),"audit_case_count":len(audit),"audit_used_for_fit":False,**{f"construction_oof_{k}":v for k,v in oof_metrics.items()},**{f"audit_{k}":v for k,v in measured.items()}}
        for k,(lo,hi) in ci.items():metric[f"audit_{k}_ci_low"]=lo;metric[f"audit_{k}_ci_high"]=hi
        metrics.append(metric);index.append({**metric,"probe_file":str(path.relative_to(root)),"probe_sha256":sha256_file(path)})
        audit_predictions.extend({"case_id":r["case_id"],"family_id":r["family_id"],"template":template,"position":position,"layer":layer,"actual_canonical_soft_sa":float(y),"predicted_canonical_soft_sa":float(p)} for r,y,p in zip(audit,y_audit,prediction));oof_predictions.extend({"case_id":r["case_id"],"family_id":r["family_id"],"outer_fold":r["outer_fold"],"template":template,"position":position,"layer":layer,"actual_canonical_soft_sa":float(y),"predicted_canonical_soft_sa":float(p)} for r,y,p in zip(construction,y_train,oof))
        atomic_json(root/"progress/train_probes.json",{"status":"running","completed_new_cells":len(index),"expected_new_cells":len(grids),"newly_fitted":new_count})
    # Register eight immutable historical T0 probes and reconstruct audit CIs from frozen predictions.
    historical_predictions=load_jsonl(T0_AUDIT_PREDICTIONS)
    for position in positions:
        for layer in layers:
            if layer not in T0_FROZEN_LAYERS:continue
            source=frozen[position,layer];row=source["index"];pred=[r for r in historical_predictions if r["target"]=="final_soft_sa" and r["position"]==position and int(r["layer"])==layer]
            keyed={str(r["case_id"]):r for r in pred};ordered=[keyed[x] for x in audit_ids];y=np.asarray([float(r["actual"]) for r in ordered]);p=np.asarray([float(r["predicted"]) for r in ordered]);measured=_metrics(y,p);ci=_bootstrap(y,p,audit_family_ids,families,draws);trace=source["payload"]["alpha_trace"];selected=float(source["payload"]["alpha"]);oof_r2=next(float(r["oof_r2"]) for r in trace if float(r["alpha"])==selected)
            metric={"template":"T0","position":position,"layer":layer,"artifact_role":"frozen_t0_reference","alpha":selected,"construction_case_count":len(construction),"audit_case_count":len(audit),"audit_used_for_fit":False,"construction_oof_r2":oof_r2,"construction_oof_pearson":math.nan,"construction_oof_spearman":math.nan,"construction_oof_mae":math.nan,**{f"audit_{k}":v for k,v in measured.items()}}
            for k,(lo,hi) in ci.items():metric[f"audit_{k}_ci_low"]=lo;metric[f"audit_{k}_ci_high"]=hi
            metrics.append(metric);index.append({**metric,"probe_file":str(source["path"].resolve()),"probe_sha256":row["probe_sha256"]});audit_predictions.extend({"case_id":r["case_id"],"family_id":r["family_id"],"template":"T0","position":position,"layer":layer,"actual_canonical_soft_sa":float(a),"predicted_canonical_soft_sa":float(b)} for r,a,b in zip(audit,y,p))
    index.sort(key=lambda r:(r["template"],r["position"],int(r["layer"])));metrics.sort(key=lambda r:(r["template"],r["position"],int(r["layer"])));atomic_jsonl(root/"artifacts/probes/probe_index.jsonl",index);atomic_csv(root/"tables/probe_metrics.csv",metrics);atomic_csv(root/"artifacts/audit_predictions.csv",audit_predictions);atomic_csv(root/"artifacts/construction_oof_predictions.csv",oof_predictions)
    comparisons=[];by={(r["template"],r["position"],int(r["layer"])):r for r in metrics}
    for template in templates:
        for position in positions:
            for layer in layers:
                t0=by["T0",position,layer];tx=by[template,position,layer];comparisons.append({"template":template,"position":position,"layer":layer,**{f"audit_{k}_difference_vs_t0":float(tx[f'audit_{k}'])-float(t0[f'audit_{k}']) for k in ("r2","pearson","spearman","mae")}})
    atomic_csv(root/"tables/t0_tx_probe_comparison.csv",comparisons);result={"status":"complete","index_cells":len(index),"new_probe_cells":len(grids),"newly_fitted_this_run":new_count,"frozen_t0_references":8,"bootstrap_repeats":len(draws)};atomic_json(root/"progress/train_probes.json",result);return result


def plot(root: Path) -> list[str]:
    rows=list(csv.DictReader(open(root/"tables/probe_metrics.csv")));created=[];colors={"T0":"black","T1":"#2166ac","T2":"#4daf4a","T3":"#b2182b"}
    for metric in ("r2","pearson","spearman","mae"):
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,position in zip(axes,DEFAULT_POSITIONS):
            for template in ("T0",*DEFAULT_TEMPLATES):
                selected=sorted([r for r in rows if r["template"]==template and r["position"]==position],key=lambda r:int(r["layer"]));x=[int(r["layer"]) for r in selected];y=[float(r[f"audit_{metric}"]) for r in selected];lo=[float(r[f"audit_{metric}_ci_low"]) for r in selected];hi=[float(r[f"audit_{metric}_ci_high"]) for r in selected];ax.plot(x,y,marker="o",label=template,color=colors[template]);ax.fill_between(x,lo,hi,color=colors[template],alpha=.1)
            ax.set(title=position,xlabel="Layer");ax.set_xticks(DEFAULT_LAYERS);ax.grid(alpha=.2)
        axes[0].set_ylabel(f"Audit {metric} (95% family-bootstrap CI)");axes[-1].legend();fig.tight_layout();out=root/f"figures/audit_{metric}_by_layer.png";out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=250);plt.close(fig);created.append(str(out))
    return created


def _readme(root: Path, capture_result: dict[str,Any]) -> None:
    text=f"""# T0–T3 LAT/PANL SA Linear Probe

本实验严格复用 T0 的 882 construction / 230 frozen held-out audit 数据划分。T1–T3 的目标是各模板自身的 canonical soft SA；T2 已转换为 image-positive，T3 主目标使用完整序列 log-likelihood。

每个新 probe 均为 `StandardScaler + Ridge(lsqr)`。alpha 只通过 construction 四折 OOF R²选择，audit 不参与拟合、选参或截距调整。T0 L10/L12为本轮新训练；T0 L14–L17为冻结历史 probe 引用。

probe 结果只说明 hidden 中可以线性解码 SA，不证明该位置或层对 SA 有因果作用。

本轮 capture 新增 GPU forward：{capture_result.get('new_gpu_forwards',0)}。
""";(root/"README_zh.md").write_text(text)


def verify(root: Path, *, smoke: bool, templates: Sequence[str], positions: Sequence[str], layers: Sequence[int], construction: Sequence[dict[str,Any]], audit: Sequence[dict[str,Any]]) -> dict[str,bool]:
    capture_rows=load_jsonl(root/"artifacts/diagnostics/capture_manifest.jsonl");index=load_jsonl(root/"artifacts/probes/probe_index.jsonl");expected_capture=(len(construction)+len(audit))*(1+len(templates));expected_grid=12+len(templates)*len(positions)*len(layers)
    gates={"capture_count":len(capture_rows)==expected_capture,"capture_unique":len({(r["template"],r["case_id"]) for r in capture_rows})==expected_capture,"probe_grid":len(index)==expected_grid==48,"new_probe_count":sum(r["artifact_role"]=="newly_fitted" for r in index)==40,"frozen_reference_count":sum(r["artifact_role"]=="frozen_t0_reference" for r in index)==8,"audit_not_fit":all(not r["audit_used_for_fit"] for r in index),"outputs":all((root/p).is_file() and (root/p).stat().st_size for p in ("tables/probe_metrics.csv","tables/t0_tx_probe_comparison.csv","artifacts/audit_predictions.csv","artifacts/construction_oof_predictions.csv","artifacts/bootstrap_draws.json","README_zh.md"))}
    if not all(gates.values()):raise ValueError(f"Acceptance failed: {gates}")
    return gates


def run(*, root: Path, templates: Sequence[str], positions: Sequence[str], layers: Sequence[int], num_gpus: int, smoke: bool, resume: bool) -> dict[str,Any]:
    if num_gpus!=1:raise ValueError("This registered experiment is single-GPU only")
    if tuple(positions)!=DEFAULT_POSITIONS or tuple(map(int,layers))!=DEFAULT_LAYERS or tuple(templates)!=DEFAULT_TEMPLATES:raise ValueError("Formal probe grid must match the preregistered templates/positions/layers")
    construction,audit,fingerprint,sources=prepare(root,templates=templates,positions=positions,layers=layers,smoke=smoke,resume=resume);atomic_json(root/"artifacts/source_hashes_before.json",sources);_ensure_bootstrap_draws(root,audit,smoke)
    workers:dict[str,subprocess.Popen]={};events=[]
    def launch(template: str) -> None:
        done=root/"artifacts/probe_shards"/template/"completion.json"
        if done.exists() and json.loads(done.read_text()).get("config_fingerprint")==fingerprint:
            events.append({"event":"cpu_probe_reused","template":template,"time":time.time()});return
        command=[sys.executable,"-m","dp_SA.prompt_check.train_template_sa_probes","--cpu-worker",template,"--output-root",str(root)]
        environment=os.environ.copy();environment["CUDA_VISIBLE_DEVICES"]="";environment.setdefault("OMP_NUM_THREADS","4");log=root/f"logs/cpu_probe_{template}.log";log.parent.mkdir(parents=True,exist_ok=True)
        with open(log,"ab") as stream:workers[template]=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        events.append({"event":"cpu_probe_started","template":template,"pid":workers[template].pid,"time":time.time()});atomic_json(root/"progress/pipeline_events.json",events)
    capture_result=capture(root,[*construction,*audit],templates=templates,positions=positions,layers=layers,fingerprint=fingerprint,resume=resume,on_template_complete=launch)
    # Fully captured resume runs return before callbacks, so schedule any absent shards here.
    for template in ("T0",*templates):
        if template not in workers and not any(e["template"]==template for e in events):launch(template)
    for template,worker in workers.items():
        code=worker.wait();events.append({"event":"cpu_probe_finished","template":template,"pid":worker.pid,"returncode":code,"time":time.time()});atomic_json(root/"progress/pipeline_events.json",events)
        if code:raise RuntimeError(f"CPU probe worker {template} failed with exit code {code}; see logs/cpu_probe_{template}.log")
    probe_result=_aggregate_shards(root,construction,audit,templates=templates,positions=positions,layers=layers,smoke=smoke);figures=plot(root);_readme(root,capture_result);verify_inventory(sources);atomic_json(root/"artifacts/source_hashes_after.json",_source_inventory());gates=verify(root,smoke=smoke,templates=templates,positions=positions,layers=layers,construction=construction,audit=audit);completion={"status":"complete","smoke_only":smoke,"num_gpus":1,"fingerprint":fingerprint,"capture":capture_result,"probes":probe_result,"pipeline_events":events,"figures":figures,"gates":gates,"historical_sources_unchanged":True,"completed_at_unix":time.time()};atomic_json(root/"completion.json",completion);return completion


def main(argv: Sequence[str]|None=None) -> int:
    p=argparse.ArgumentParser(description="Train T0–T3 LAT/PANL canonical-SA Ridge probes");p.add_argument("--templates",nargs="+",choices=DEFAULT_TEMPLATES,default=list(DEFAULT_TEMPLATES));p.add_argument("--positions",nargs="+",choices=DEFAULT_POSITIONS,default=list(DEFAULT_POSITIONS));p.add_argument("--layers",nargs="+",type=int,default=list(DEFAULT_LAYERS));p.add_argument("--num-gpus",type=int,choices=(1,),default=1);p.add_argument("--smoke",action="store_true");p.add_argument("--resume",action="store_true");p.add_argument("--output-root");p.add_argument("--cpu-worker",choices=("T0",*DEFAULT_TEMPLATES),help=argparse.SUPPRESS);a=p.parse_args(argv);root=Path(a.output_root).resolve() if a.output_root else (DEFAULT_SMOKE_ROOT if a.smoke else DEFAULT_ROOT);allowed=OUTPUT_PARENT.resolve()
    if allowed not in root.parents:raise ValueError("Output root must be under dp_SA/prompt_check/output")
    if a.cpu_worker:
        config=json.loads((root/"artifacts/config_and_fingerprint.json").read_text());train_template_shard(root,a.cpu_worker,positions=config["positions"],layers=config["layers"],fingerprint=config["fingerprint"],smoke=bool(config["smoke_only"]),resume=True);return 0
    print(json.dumps(run(root=root,templates=a.templates,positions=a.positions,layers=a.layers,num_gpus=a.num_gpus,smoke=a.smoke,resume=a.resume),ensure_ascii=False,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
