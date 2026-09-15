from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from .config import (
    CAPTURE_ROOT, CAUSAL_PAIRS, CLE_LAYERS, EXPECTED_AUDIT, EXPECTED_CAPTURE_CASES, EXPECTED_CONSTRUCTION,
    EXPECTED_PROBE_ELIGIBLE, EXPECTED_TEST_CASES, EXPECTED_TEST_SIDES, HIDDEN_DEFINITION,
    MODEL_PATH, PANL_LAYERS, SEED, STEERING_ROOT, VECTOR_NORM_FRACTION,
)
from .contracts import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file, validate_layer_design


def _hidden(capture_root: Path, row: dict[str, Any], layer: int) -> np.ndarray:
    with np.load(capture_root / row["hidden_file"], allow_pickle=False) as archive:
        value = np.asarray(archive[f"PANL__L{layer}"], np.float32)
    if value.shape != (4096,) or not np.isfinite(value).all(): raise ValueError(f"Invalid PANL L{layer}: {row['case_id']}")
    return value


def _save_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, temporary=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent);os.close(fd)
    try: torch.save(value,temporary);os.replace(temporary,path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def build_vectors(capture_root: Path, steering_root: Path) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    construction=load_jsonl(steering_root/"construction_manifest.jsonl")
    high=[r for r in construction if r["construction_side"]=="high_image"];low=[r for r in construction if r["construction_side"]=="high_text"]
    if (len(high),len(low))!=(25,25) or len({str(r["item_id"]) for r in construction})!=50: raise ValueError("Frozen extreme construction must be 25+25 item-disjoint")
    old=torch.load(steering_root/"vectors.pt",map_location="cpu",weights_only=False)
    vectors={};metadata=[];artifacts={}
    for layer in PANL_LAYERS:
        hi=np.stack([_hidden(capture_root,r,layer) for r in high]);lo=np.stack([_hidden(capture_root,r,layer) for r in low]);combined=np.concatenate([hi,lo])
        raw=hi.mean(0)-lo.mean(0);raw_norm=float(np.linalg.norm(raw));mean_norm=float(np.linalg.norm(combined,axis=1).mean());target=VECTOR_NORM_FRACTION*mean_norm
        scaled=(raw/raw_norm*target).astype(np.float32)
        parity=None
        if layer in (14,16):
            reference=old[f"PANL__L{layer}"]["scaled_vector"].float().numpy();parity=float(np.max(np.abs(reference-scaled)))
            if parity!=0: raise ValueError(f"PANL L{layer} direction differs from frozen Steering vector: {parity}")
        vectors[layer]=torch.from_numpy(scaled);artifacts[f"PANL__L{layer}"]={"raw_vector":torch.from_numpy(raw.astype(np.float32)),"scaled_vector":vectors[layer]}
        metadata.append({"position":"PANL","layer":layer,"raw_vector_norm":raw_norm,"mean_residual_norm":mean_norm,"target_vector_norm":target,"actual_vector_norm":float(np.linalg.norm(scaled)),"existing_vector_max_abs_error":parity})
    return vectors,metadata,artifacts


def split_manifests(capture: list[dict[str, Any]], test: list[dict[str, Any]]) -> tuple[list[dict[str, Any]],list[dict[str,Any]],dict[str,Any]]:
    test_items={str(r["item_id"]) for r in test};test_images={str(r["image_sha256"]) for r in test}
    eligible=[r for r in capture if str(r["item_id"]) not in test_items and str(r["image_sha256"]) not in test_images]
    if (len(eligible),len({str(r['item_id']) for r in eligible}))!=EXPECTED_PROBE_ELIGIBLE: raise ValueError("Probe eligible population changed")
    groups=np.asarray([str(r["item_id"]) for r in eligible]);splitter=GroupKFold(5,shuffle=True,random_state=SEED)
    fold_by_index={}
    for fold,(_,held) in enumerate(splitter.split(np.zeros(len(eligible)),groups=groups)):
        for index in held: fold_by_index[int(index)]=fold
    audit=[{**row,"outer_fold":0} for i,row in enumerate(eligible) if fold_by_index[i]==0]
    construction=[{**row,"outer_fold":fold_by_index[i]} for i,row in enumerate(eligible) if fold_by_index[i]!=0]
    cardinalities=((len(construction),len({str(r['item_id']) for r in construction})),(len(audit),len({str(r['item_id']) for r in audit})))
    if cardinalities!=(EXPECTED_CONSTRUCTION,EXPECTED_AUDIT): raise ValueError(f"Probe split changed: {cardinalities}")
    sets=[]
    for rows in (construction,audit,test): sets.append(({str(r["item_id"]) for r in rows},{str(r["image_sha256"]) for r in rows}))
    if any(sets[a][kind]&sets[b][kind] for a in range(3) for b in range(a+1,3) for kind in (0,1)): raise ValueError("Construction/audit/test leakage")
    return construction,audit,{"seed":SEED,"eligible_cases":len(eligible),"eligible_items":len(set(groups)),"construction_cases":len(construction),"construction_items":len(sets[0][0]),"audit_cases":len(audit),"audit_items":len(sets[1][0]),"test_cases":len(test),"test_items":len(sets[2][0]),"item_overlap":0,"image_overlap":0}


def prepare(*, output_root: Path, capture_root: Path=CAPTURE_ROOT, steering_root: Path=STEERING_ROOT, model_path: Path=MODEL_PATH, smoke: bool=False, resume: bool=False) -> dict[str,Any]:
    output_root=Path(output_root).resolve();capture_root=Path(capture_root).resolve();steering_root=Path(steering_root).resolve();model_path=Path(model_path).resolve();validate_layer_design()
    capture=load_jsonl(capture_root/"results.jsonl");test=load_jsonl(steering_root/"test_manifest.jsonl")
    if len(capture)!=EXPECTED_CAPTURE_CASES or len(test)!=EXPECTED_TEST_CASES: raise ValueError("Frozen capture/test cardinality changed")
    if Counter(r["test_side"] for r in test)!=Counter(EXPECTED_TEST_SIDES): raise ValueError("Frozen 50/31 test split changed")
    construction,audit,split_audit=split_manifests(capture,test)
    vectors,vector_metadata,artifacts=build_vectors(capture_root,steering_root)
    source_files=[capture_root/"config.json",capture_root/"results.jsonl",steering_root/"construction_manifest.jsonl",steering_root/"test_manifest.jsonl",steering_root/"vectors.pt",model_path/"config.json"]
    code_files=sorted(Path(__file__).parent.glob("*.py"))
    payload={"format_version":1,"experiment":"qwen3_panl_to_cle_four_cell","model":str(model_path),"capture_root":str(capture_root),"steering_root":str(steering_root),"hidden_definition":HIDDEN_DEFINITION,"panl_layers":list(PANL_LAYERS),"cle_layers":list(CLE_LAYERS),"causal_pairs":[list(x) for x in CAUSAL_PAIRS],"alphas":[-5.0,5.0],"seed":SEED,"attention_implementation":"sdpa","source_hashes":{str(p):sha256_file(p) for p in source_files},"code_hashes":{str(p):sha256_file(p) for p in code_files},"split_audit":split_audit,"smoke":bool(smoke)}
    fingerprint=canonical_hash(payload);path=output_root/"fingerprint.json";existed=path.exists()
    if existed:
        previous=json.loads(path.read_text())
        if previous.get("fingerprint")!=fingerprint: raise ValueError("Output fingerprint mismatch")
        if not resume: raise FileExistsError(f"Prepared output exists; use --resume: {output_root}")
    atomic_jsonl(output_root/"artifacts/manifests/probe_construction.jsonl",construction);atomic_jsonl(output_root/"artifacts/manifests/probe_audit.jsonl",audit);atomic_jsonl(output_root/"artifacts/manifests/test_manifest.jsonl",test)
    atomic_json(output_root/"artifacts/manifests/split_audit.json",split_audit);_save_torch(output_root/"artifacts/vectors/panl_vectors.pt",artifacts);atomic_json(output_root/"artifacts/vectors/vector_metadata.json",{"normalization_fraction":VECTOR_NORM_FRACTION,"vectors":vector_metadata})
    atomic_json(path,{**payload,"fingerprint":fingerprint});result={"status":"complete","fingerprint":fingerprint,"split_audit":split_audit,"vector_count":len(vectors),"resumed":existed};atomic_json(output_root/"progress/prepare.json",result);return result
