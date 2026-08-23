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

from .config import BOOTSTRAP_REPEATS, OUTPUT_ROOT, SPLIT_PATH
from .io_utils import atomic_json, atomic_jsonl, load_jsonl

def _load_hidden(root: Path, row: dict[str,Any], position: str, layer: int) -> np.ndarray:
    payload=np.load(root/row["hidden_file"])
    return np.asarray(payload[f"{position}__L{layer}"],dtype=np.float32)

def run_probe(output_root: Path, candidate_path: Path, split_path: Path, bootstrap: int=BOOTSTRAP_REPEATS) -> dict[str,Any]:
    candidates=load_jsonl(candidate_path); out=output_root/"probe"
    if not candidates:
        summary={"status":"probe_skipped_no_significant_cells","candidate_count":0}; atomic_json(out/"metrics.json",summary); return summary
    records=[r for r in load_jsonl(output_root/"capture"/"results.jsonl") if r.get("status")=="completed"]
    split=json.loads(split_path.read_text()); item_to_fold={str(k):int(v) for k,v in split["item_to_fold"].items()}
    if {str(r["item_id"]) for r in records} != set(item_to_fold): raise ValueError("Probe records do not exactly cover split items")
    predictions=[]; metrics=[]
    for candidate in candidates:
        position=str(candidate["position"]); layer=int(candidate["layer"])
        X=np.stack([_load_hidden(output_root,r,position,layer) for r in records]); y=np.asarray([r["soft_sa_image_score"] for r in records],float)
        pred=np.empty(len(records),float)
        for fold in range(5):
            train=np.asarray([item_to_fold[str(r["item_id"])]!=fold for r in records]); test=~train
            model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=1.0,solver="lsqr"))]); model.fit(X[train],y[train]); pred[test]=model.predict(X[test])
            for i in np.flatnonzero(test): predictions.append({"case_id":records[i]["case_id"],"item_id":records[i]["item_id"],"position":position,"layer":layer,"fold":fold,"target":float(y[i]),"prediction":float(pred[i])})
        observed=[float(r2_score(y,pred)),float(spearmanr(y,pred).statistic),float(pearsonr(y,pred).statistic),float(mean_absolute_error(y,pred))]
        by_item={item:np.flatnonzero([str(r["item_id"])==item for r in records]) for item in sorted(set(str(r["item_id"]) for r in records))}; items=list(by_item); rng=np.random.default_rng(42+layer); boot=[]
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
            data=sorted([m for m in metrics if m["position"]==position],key=lambda m:m["layer"]); ax.errorbar([m["layer"] for m in data],[m[metric] for m in data],yerr=[[m[metric]-m[f"{metric}_ci_low"] for m in data],[m[f"{metric}_ci_high"]-m[metric] for m in data]],marker="o",label=position)
        ax.axhline(0,color="black",ls="--",lw=.8); ax.set_xlabel("zero-based decoder layer"); ax.set_ylabel(metric); ax.legend(); fig.tight_layout(); fig.savefig(out/filename,dpi=160); plt.close(fig)
    return summary

def main(argv: Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(); p.add_argument("--output-root",default=str(OUTPUT_ROOT)); p.add_argument("--candidates"); p.add_argument("--split",default=str(SPLIT_PATH)); p.add_argument("--bootstrap",type=int,default=BOOTSTRAP_REPEATS)
    a=p.parse_args(argv); root=Path(a.output_root); run_probe(root,Path(a.candidates) if a.candidates else root/"steering"/"probe_candidate_manifest.jsonl",Path(a.split),a.bootstrap); return 0
if __name__=="__main__": raise SystemExit(main())
