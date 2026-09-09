from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import (BOOTSTRAPS, CANDIDATE_MANIFEST, CELL_MANIFEST, CONSTRUCTION_DISTRIBUTION,
                     GEOMETRY_LAYERS, MATCHED_ROOT, SEED, SMOKE_BOOTSTRAPS, T0_CLEAN_CAPTURE,
                     T0_VECTOR_METADATA, TEST_MANIFEST, VALIDATION_BOOTSTRAPS)
from .io_utils import array_hash, atomic_csv, atomic_json, atomic_npz, canonical_hash, load_jsonl, sha256_file
from .stats import shared_family_draws
from .capture import smoke_geometry_cells
from .sampling import validation_geometry_cells


@lru_cache(maxsize=None)
def _load_hidden_file(path: str, key: str) -> np.ndarray:
    with np.load(path) as payload:
        value = np.asarray(payload[key], dtype=np.float32)
    value.setflags(write=False)
    return value


def _load_hidden(root: Path, row: dict[str, Any], layer: int) -> np.ndarray:
    return _load_hidden_file(str(root / row["hidden_file"]), f"P1_LAT__L{layer}")


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator=float(np.linalg.norm(left)*np.linalg.norm(right))
    return float(np.dot(left,right)/denominator) if denominator>0 else math.nan


def _weighted_side(cell_values: dict[tuple[str,str,str],np.ndarray], answer: str, side: str, weights: dict[str,int]) -> np.ndarray:
    rows=[(family,value) for (family,a,s),value in cell_values.items() if a==answer and s==side and weights.get(family,0)>0]
    if not rows: raise ValueError(f"Bootstrap has no {answer}/{side} families")
    numerator=np.zeros_like(rows[0][1],dtype=np.float32);denominator=0
    for family,value in sorted(rows):
        weight=int(weights[family]);numerator += value*np.float32(weight);denominator += weight
    return numerator/np.float32(denominator)


def _directions(cell_values: dict[tuple[str,str,str],np.ndarray], eligible: Sequence[str], recipients: Sequence[str], weights: dict[str,int]) -> dict[str,np.ndarray]:
    by_answer={answer:np.asarray(_weighted_side(cell_values,answer,"high_image",weights)-_weighted_side(cell_values,answer,"high_text",weights),dtype=np.float32) for answer in eligible}
    output={}
    for recipient in recipients:
        included=sorted(a for a in eligible if a!=recipient)
        if len(included)<3: raise ValueError("LOAO has fewer than three donor answers")
        output[recipient]=np.stack([by_answer[a] for a in included]).mean(axis=0,dtype=np.float32)
    return output


def _cell_values(cells: Sequence[dict[str,Any]], hidden: dict[str,dict[str,Any]], root: Path, layer: int) -> dict[tuple[str,str,str],np.ndarray]:
    output={}
    for cell in cells:
        vectors=[_load_hidden(root,hidden[str(case)],layer) for case in cell["case_ids"]]
        output[str(cell["family_id"]),str(cell["answer"]),str(cell["sa_side"])]=np.stack(vectors).mean(axis=0,dtype=np.float32)
    return output


def _t0_vector(metadata: dict[str,Any], fold: int, recipient: str, layer: int) -> tuple[np.ndarray,dict[str,Any]]:
    matches=[r for r in metadata["vectors"] if r["position"]=="P1_LAT" and r["direction"]=="matched_loao" and int(r["fold"])==fold and r["recipient_answer"]==recipient and int(r["layer"])==layer]
    if len(matches)!=1: raise ValueError(f"Frozen T0 vector missing: fold={fold} answer={recipient} L{layer}")
    row=matches[0];path=MATCHED_ROOT/row["vector_file"]
    if sha256_file(path)!=row["vector_file_sha256"]: raise ValueError("Frozen T0 vector file hash changed")
    with np.load(path) as payload:value=np.asarray(payload[row["raw_key"]],dtype=np.float32)
    return value,row


def analyze_geometry(root:Path,*,templates:Sequence[str],smoke:bool,validation_cases:int|None=None)->dict[str,Any]:
    candidate=load_jsonl(CANDIDATE_MANIFEST);cells=load_jsonl(CELL_MANIFEST);distribution=load_jsonl(CONSTRUCTION_DISTRIBUTION);test=load_jsonl(TEST_MANIFEST)
    if validation_cases:
        cells=validation_geometry_cells(validation_cases);case_ids={str(case) for cell in cells for case in cell["case_ids"]};candidate=[r for r in candidate if str(r["case_id"]) in case_ids];folds=[0];test=[r for r in test if int(r["fold"])==0]
    elif smoke:
        cells=smoke_geometry_cells();case_ids={str(case) for cell in cells for case in cell["case_ids"]};candidate=[r for r in candidate if str(r["case_id"]) in case_ids];folds=[0];test=[r for r in test if int(r["fold"])==0][:2]
    else:folds=list(range(15))
    capture={(r["template"],str(r["case_id"])):r for r in load_jsonl(root/"artifacts/clean/capture.jsonl") if r.get("is_candidate")}
    t0_clean={str(r["case_id"]):r for r in load_jsonl(T0_CLEAN_CAPTURE) if r.get("status")=="completed"};metadata=json.loads(T0_VECTOR_METADATA.read_text())
    universe=sorted({str(r["family_id"]) for r in cells}) if (smoke or validation_cases) else sorted({str(r["family_id"]) for r in load_jsonl(CANDIDATE_MANIFEST)});repeats=SMOKE_BOOTSTRAPS if smoke else (VALIDATION_BOOTSTRAPS if validation_cases else BOOTSTRAPS)
    ordered,draws=shared_family_draws(universe,repeats,SEED+300);multiplicities=np.asarray([np.bincount(draw,minlength=len(ordered)) for draw in draws],dtype=np.int16)
    detail=[];summary=[];parity=[];fold_boots:dict[tuple[str,int,int],np.ndarray]={}
    layers=GEOMETRY_LAYERS[:1] if smoke else GEOMETRY_LAYERS
    for fold in folds:
        fold_cells=[r for r in cells if int(r["fold"])==fold];eligible=sorted({r["answer"] for r in fold_cells}) if (smoke or validation_cases) else sorted(r["answer"] for r in distribution if int(r["fold"])==fold and r["eligible_for_direction"]);recipients=sorted({str(r["test_answer"]) for r in test if int(r["fold"])==fold})
        if not fold_cells or not recipients:continue
        for layer in layers:
            t0_values=_cell_values(fold_cells,t0_clean,MATCHED_ROOT,layer);point_weights={family:1 for family in universe};t0_point=_directions(t0_values,eligible,recipients,point_weights)
            for recipient,value in t0_point.items():
                if smoke or validation_cases:parity.append({"fold":fold,"recipient_answer":recipient,"layer":layer,"subset":True,"formal_vector_parity":"not_applicable"})
                else:
                    frozen,row=_t0_vector(metadata,fold,recipient,layer);error=float(np.max(np.abs(value-frozen)));equal=array_hash(value)==array_hash(frozen)
                    if error>1e-6:raise ValueError(f"T0 raw vector parity failed: fold={fold} {recipient} L{layer} max={error}")
                    parity.append({"fold":fold,"recipient_answer":recipient,"layer":layer,"max_abs_error":error,"array_hash_equal":equal,"frozen_vector_fingerprint":row["vector_fingerprint"]})
            for template in templates:
                hidden={str(r["case_id"]):capture[template,str(r["case_id"])] for r in candidate};tx_values=_cell_values(fold_cells,hidden,root,layer);tx_point=_directions(tx_values,eligible,recipients,point_weights)
                arrays={f"{recipient}__raw":value for recipient,value in tx_point.items()};relative=Path("artifacts/vectors")/template/f"fold_{fold:02d}__L{layer}.npz";atomic_npz(root/relative,arrays)
                boot_cos={recipient:np.full(repeats,np.nan,dtype=float) for recipient in recipients}
                for draw_index,multiplicity in enumerate(multiplicities):
                    weights={family:int(multiplicity[index]) for index,family in enumerate(ordered)}
                    try:t0_boot=_directions(t0_values,eligible,recipients,weights);tx_boot=_directions(tx_values,eligible,recipients,weights)
                    except ValueError:continue
                    for recipient in recipients:boot_cos[recipient][draw_index]=_cosine(t0_boot[recipient],tx_boot[recipient])
                point_cos=[]
                for recipient in recipients:
                    cosine=_cosine(t0_point[recipient],tx_point[recipient]);point_cos.append(cosine);values=boot_cos[recipient][np.isfinite(boot_cos[recipient])]
                    if not len(values):raise ValueError(f"No valid geometry bootstrap draws: {template} fold={fold} {recipient} L{layer}")
                    low,high=np.quantile(values,[.025,.975])
                    detail.append({"row_type":"recipient","template":template,"fold":fold,"recipient_answer":recipient,"layer":layer,"raw_cosine":cosine,"absolute_cosine":abs(cosine),"norm_ratio":float(np.linalg.norm(tx_point[recipient])/np.linalg.norm(t0_point[recipient])),"cosine_ci_low":float(low),"cosine_ci_high":float(high),"valid_bootstrap_repeats":len(values),"vector_file":str(relative)})
                matrix=np.stack([boot_cos[r] for r in recipients]);valid=np.all(np.isfinite(matrix),axis=0);aggregate_full=np.full(repeats,np.nan,dtype=float);aggregate_full[valid]=matrix[:,valid].mean(axis=0);aggregate_boot=aggregate_full[valid]
                low,high=np.quantile(aggregate_boot,[.025,.975]);fold_boots[template,layer,fold]=aggregate_full
                summary.append({"row_type":"fold_recipient_equal_summary","template":template,"fold":fold,"recipient_answer":"__all__","layer":layer,"raw_cosine":float(np.mean(point_cos)),"absolute_cosine":float(np.mean(np.abs(point_cos))),"recipient_min_cosine":float(np.min(point_cos)),"recipient_max_cosine":float(np.max(point_cos)),"fold_min_cosine":math.nan,"fold_max_cosine":math.nan,"cosine_ci_low":float(low),"cosine_ci_high":float(high),"valid_bootstrap_repeats":len(aggregate_boot)})
    # The primary geometry summary gives every fold and every recipient equal weight.
    for template in templates:
        for layer in layers:
            selected=[r for r in detail if r["template"]==template and int(r["layer"])==layer]
            fold_rows=[r for r in summary if r["row_type"]=="fold_recipient_equal_summary" and r["template"]==template and int(r["layer"])==layer]
            if not selected or not fold_rows:continue
            boot_matrix=np.stack([fold_boots[template,layer,int(r["fold"])] for r in fold_rows]);valid=np.all(np.isfinite(boot_matrix),axis=0);global_boot=boot_matrix[:,valid].mean(axis=0)
            if not len(global_boot):raise ValueError(f"No globally paired geometry draws: {template} L{layer}")
            low,high=np.quantile(global_boot,[.025,.975]);fold_points=np.asarray([r["raw_cosine"] for r in fold_rows],float);recipient_points=np.asarray([r["raw_cosine"] for r in selected],float)
            summary.append({"row_type":"global_recipient_equal_summary","template":template,"fold":-1,"recipient_answer":"__all__","layer":layer,"raw_cosine":float(fold_points.mean()),"absolute_cosine":float(np.mean([r["absolute_cosine"] for r in fold_rows])),"recipient_min_cosine":float(recipient_points.min()),"recipient_max_cosine":float(recipient_points.max()),"fold_min_cosine":float(fold_points.min()),"fold_max_cosine":float(fold_points.max()),"cosine_ci_low":float(low),"cosine_ci_high":float(high),"valid_bootstrap_repeats":len(global_boot)})
    rows=detail+summary;atomic_csv(root/"tables/vector_geometry.csv",rows);atomic_json(root/"artifacts/diagnostics/t0_vector_parity.json",{"status":"passed","rows":parity});atomic_json(root/"artifacts/diagnostics/geometry_bootstrap_draws.json",{"seed":SEED+300,"families":ordered,"multiplicity_draws":multiplicities.tolist()})
    return {"status":"complete","detail_rows":len(detail),"summary_rows":len(summary),"bootstrap_repeats":repeats}
