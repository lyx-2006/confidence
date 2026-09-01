from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import BOOTSTRAP_REPEATS, OUTPUT_ROOT, SEED, SPLIT_PATH, SUPPORTED_CAPTURE_POSITIONS
from .io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl

def _load_hidden(root: Path, row: dict[str,Any], position: str, layer: int) -> np.ndarray:
    with np.load(root/row["hidden_file"]) as payload:
        return np.asarray(payload[f"{position}__L{layer}"],dtype=np.float32)

def parse_cells(values: Sequence[str]) -> list[tuple[str, int]]:
    if not values:
        raise ValueError("--cell must be provided at least once")
    output: list[tuple[str, int]]=[]; seen: set[tuple[str, int]]=set()
    for raw in values:
        position, separator, layer_text=str(raw).partition(":")
        if not separator or not position or not layer_text or ":" in layer_text:
            raise ValueError(f"Invalid --cell {raw!r}; expected POSITION:LAYER")
        if position not in SUPPORTED_CAPTURE_POSITIONS:
            raise ValueError(f"Unsupported probe position in --cell: {position!r}")
        try: layer=int(layer_text)
        except ValueError as exc: raise ValueError(f"Invalid layer in --cell {raw!r}") from exc
        if layer < 0:
            raise ValueError(f"Layer in --cell must be non-negative: {raw!r}")
        cell=(position,layer)
        if cell not in seen:
            output.append(cell); seen.add(cell)
    return output

def bootstrap_seed(seed: int, position: str, layer: int) -> int:
    return int(canonical_hash({"seed":int(seed),"position":str(position),"layer":int(layer)})[:16],16) % (2**32)

def _validate_hidden_keys(root: Path, records: Sequence[dict[str,Any]], cells: Sequence[tuple[str,int]]) -> None:
    for row in records:
        path=root/row["hidden_file"]
        if not path.is_file():
            raise FileNotFoundError(f"Capture hidden NPZ is missing for {row['case_id']}: {path}")
        with np.load(path) as payload:
            available=set(payload.files)
        for position,layer in cells:
            key=f"{position}__L{layer}"
            if key not in available:
                raise KeyError(f"Capture hidden key missing for case {row['case_id']}: {key} in {path}")

def run_probe(output_root: Path, candidate_path: Path|None, split_path: Path, bootstrap: int=BOOTSTRAP_REPEATS,
              *, cells: Sequence[tuple[str,int]]|None=None, seed: int=SEED) -> dict[str,Any]:
    if cells is not None and candidate_path is not None:
        raise ValueError("Explicit cells and candidate manifest are mutually exclusive")
    candidates=([{"position":position,"layer":layer} for position,layer in cells]
                if cells is not None else load_jsonl(candidate_path))
    out=output_root/"probe"
    if not candidates:
        summary={"status":"probe_skipped_no_significant_cells","candidate_count":0}; atomic_json(out/"metrics.json",summary); return summary
    records=[r for r in load_jsonl(output_root/"capture"/"results.jsonl") if r.get("status")=="completed"]
    split=json.loads(split_path.read_text()); item_to_fold={str(k):int(v) for k,v in split["item_to_fold"].items()}
    if {str(r["item_id"]) for r in records} != set(item_to_fold): raise ValueError("Probe records do not exactly cover split items")
    requested=[(str(candidate["position"]),int(candidate["layer"])) for candidate in candidates]
    _validate_hidden_keys(output_root,records,requested)
    predictions=[]; metrics=[]
    for candidate in candidates:
        position=str(candidate["position"]); layer=int(candidate["layer"])
        X=np.stack([_load_hidden(output_root,r,position,layer) for r in records]); y=np.asarray([r["soft_sa_image_score"] for r in records],float)
        pred=np.empty(len(records),float)
        for fold in range(5):
            train=np.asarray([item_to_fold[str(r["item_id"])]!=fold for r in records]); test=~train
            train_items={str(records[i]["item_id"]) for i in np.flatnonzero(train)}
            test_items={str(records[i]["item_id"]) for i in np.flatnonzero(test)}
            if not train.any() or not test.any() or train_items & test_items:
                raise ValueError(f"Invalid item-level OOF split for fold {fold}")
            model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=1.0,solver="lsqr"))]); model.fit(X[train],y[train]); pred[test]=model.predict(X[test])
            for i in np.flatnonzero(test): predictions.append({"case_id":records[i]["case_id"],"item_id":records[i]["item_id"],"position":position,"layer":layer,"fold":fold,"target":float(y[i]),"prediction":float(pred[i])})
        observed=[float(r2_score(y,pred)),float(spearmanr(y,pred).statistic),float(pearsonr(y,pred).statistic),float(mean_absolute_error(y,pred))]
        by_item={item:np.flatnonzero([str(r["item_id"])==item for r in records]) for item in sorted(set(str(r["item_id"]) for r in records))}; items=list(by_item); rng=np.random.default_rng(bootstrap_seed(seed,position,layer)); boot=[]
        for _ in range(bootstrap):
            chosen=rng.choice(items,size=len(items),replace=True); indices=np.concatenate([by_item[str(item)] for item in chosen]); yy=y[indices]; pp=pred[indices]
            values=[r2_score(yy,pp),spearmanr(yy,pp).statistic,pearsonr(yy,pp).statistic,mean_absolute_error(yy,pp)]
            if np.isfinite(values).all(): boot.append(values)
        ci=np.percentile(np.asarray(boot),[2.5,97.5],axis=0)
        metrics.append({"position":position,"layer":layer,"r2":observed[0],"r2_ci_low":ci[0,0],"r2_ci_high":ci[1,0],"spearman":observed[1],"spearman_ci_low":ci[0,1],"spearman_ci_high":ci[1,1],
                        "pearson":observed[2],"pearson_ci_low":ci[0,2],"pearson_ci_high":ci[1,2],"mae":observed[3],"mae_ci_low":ci[0,3],"mae_ci_high":ci[1,3],"valid_bootstrap_repeats":len(boot),"sample_count":len(y),"item_count":len(items)})
    atomic_jsonl(out/"oof_predictions.jsonl",predictions); summary={"status":"complete","candidate_count":len(candidates),"metrics":metrics}; atomic_json(out/"metrics.json",summary)
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    import matplotlib.pyplot as plt
    for metric,filename in (("r2","r2_across_layers.png"),("spearman","spearman_across_layers.png")):
        fig,ax=plt.subplots(figsize=(7,4))
        for position in sorted({m["position"] for m in metrics}):
            data=sorted([m for m in metrics if m["position"]==position],key=lambda m:m["layer"]); ax.errorbar([m["layer"] for m in data],[m[metric] for m in data],yerr=[[max(0.0,m[metric]-m[f"{metric}_ci_low"]) for m in data],[max(0.0,m[f"{metric}_ci_high"]-m[metric]) for m in data]],marker="o",label=position)
        ax.axhline(0,color="black",ls="--",lw=.8); ax.set_xlabel("zero-based decoder layer"); ax.set_ylabel(metric); ax.legend(); fig.tight_layout(); fig.savefig(out/filename,dpi=160); plt.close(fig)
    return summary

def main(argv: Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(); p.add_argument("--output-root",default=str(OUTPUT_ROOT)); group=p.add_mutually_exclusive_group(); group.add_argument("--candidates"); group.add_argument("--cell",action="append")
    p.add_argument("--split",default=str(SPLIT_PATH)); p.add_argument("--bootstrap",type=int,default=BOOTSTRAP_REPEATS); p.add_argument("--seed",type=int,default=SEED)
    a=p.parse_args(argv); root=Path(a.output_root); cells=parse_cells(a.cell) if a.cell is not None else None
    candidate_path=None if cells is not None else Path(a.candidates) if a.candidates else root/"steering"/"probe_candidate_manifest.jsonl"
    run_probe(root,candidate_path,Path(a.split),a.bootstrap,cells=cells,seed=a.seed); return 0
if __name__=="__main__": raise SystemExit(main())
