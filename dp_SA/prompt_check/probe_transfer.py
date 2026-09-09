from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import AUDIT_MANIFEST, BOOTSTRAPS, POSITIONS, PROBE_LAYERS, SEED, SMOKE_BOOTSTRAPS, VALIDATION_BOOTSTRAPS
from .io_utils import atomic_csv, atomic_json, load_jsonl
from .sources import load_frozen_probes
from .stats import add_ci, bootstrap_rows, regression_metrics, shared_family_draws
from .capture import select_smoke_audit
from .sampling import validation_audit


def _hidden(root: Path, row: dict[str, Any], position: str, layer: int) -> np.ndarray:
    with np.load(root / row["hidden_file"]) as payload: return np.asarray(payload[f"{position}__L{layer}"], dtype=np.float32)


def analyze_probe_transfer(root:Path,*,templates:Sequence[str],smoke:bool,validation_cases:int|None=None)->dict[str,Any]:
    audit=validation_audit(validation_cases) if validation_cases else load_jsonl(AUDIT_MANIFEST)
    if smoke: audit=select_smoke_audit(audit,4)
    captured={(r["template"],str(r["case_id"])):r for r in load_jsonl(root/"artifacts/clean/capture.jsonl") if r.get("is_audit")}
    probes=load_frozen_probes(); layers=PROBE_LAYERS[:2] if smoke else PROBE_LAYERS; positions=POSITIONS
    families=[str(r["family_id"]) for r in audit]; ordered,draws=shared_family_draws(families,SMOKE_BOOTSTRAPS if smoke else (VALIDATION_BOOTSTRAPS if validation_cases else BOOTSTRAPS),SEED+200)
    metrics=[];predictions=[]
    t0=np.asarray([float(r["soft_sa_image_score"]) for r in audit])
    for template in templates:
        rows=[captured[template,str(r["case_id"])] for r in audit];tx=np.asarray([float(r["canonical_soft_sa"]) for r in rows]);hard=np.asarray([float(r.get("canonical_hard_score",np.nan)) for r in rows])
        for position in positions:
            for layer in layers:
                payload=probes[position,int(layer)];model=payload["model"]
                before={f"{step}.{name}":value.copy() for step,estimator in model.named_steps.items() for name,value in estimator.__dict__.items() if isinstance(value,np.ndarray)}
                x=np.stack([_hidden(root,row,position,int(layer)) for row in rows]);prediction=np.asarray(model.predict(x),float)
                after={f"{step}.{name}":value for step,estimator in model.named_steps.items() for name,value in estimator.__dict__.items() if isinstance(value,np.ndarray)}
                if before.keys()!=after.keys() or any(not np.array_equal(before[k],after[k]) for k in before): raise RuntimeError("Frozen probe mutated during predict")
                for meta,row,p in zip(audit,rows,prediction): predictions.append({"template":template,"case_id":meta["case_id"],"family_id":meta["family_id"],"position":position,"layer":int(layer),"prediction":float(p),"sa_tx":float(row["canonical_soft_sa"]),"sa_t0":float(meta["soft_sa_image_score"]),"t3_hard_score":row.get("canonical_hard_score")})
                targets=[("tx_sa",tx),("t0_sa",t0)]+([("t3_hard_score",hard)] if template=="T3" else [])
                for reference,target in targets:
                    observed=regression_metrics(target,prediction);boots,valid=bootstrap_rows(lambda idx:regression_metrics(target[idx],prediction[idx]),families,ordered,draws)
                    metrics.append({"template":template,"position":position,"layer":int(layer),"target_reference":reference,"alpha":float(payload["alpha"]),**add_ci(observed,boots,list(observed)),"case_count":len(target),"family_count":len(ordered),"valid_bootstrap_repeats":valid})
    atomic_csv(root/"tables/t0_probe_transfer_metrics.csv",metrics);atomic_csv(root/"artifacts/t0_probe_transfer_predictions.csv",predictions)
    atomic_json(root/"artifacts/diagnostics/probe_bootstrap_draws.json",{"seed":SEED+200,"repeats":len(draws),"families":ordered,"draws":draws.tolist()})
    return {"status":"complete","metric_cells":len(metrics),"prediction_rows":len(predictions)}
