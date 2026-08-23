from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import BOOTSTRAP_REPEATS, OUTPUT_ROOT, PROBE_POSITIONS, SEED
from .io_utils import atomic_json, atomic_jsonl, load_jsonl

def bh_fdr(pvalues: Sequence[float]) -> list[float]:
    p=np.asarray(pvalues,dtype=float); n=len(p)
    if n==0:return []
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)): raise ValueError("Invalid p-values")
    order=np.argsort(p); ranked=p[order]; adjusted=np.minimum.accumulate((ranked*n/np.arange(1,n+1))[::-1])[::-1]
    out=np.empty(n); out[order]=np.minimum(adjusted,1.0); return out.tolist()

def _one_sided_p(values: np.ndarray, positive: bool) -> float:
    n=len(values)
    bad=np.count_nonzero(values<=0 if positive else values>=0)
    return float((bad+1)/(n+1))

def _bootstrap_cell(rows: Sequence[dict[str, Any]], repeats: int, seed: int) -> dict[str, Any]:
    by_item=defaultdict(list)
    for r in rows: by_item[str(r["item_id"])].append(r)
    items=sorted(by_item); rng=np.random.default_rng(seed)
    def stats(sample_items: Sequence[str]) -> tuple[float,float,float]:
        sampled=[r for item in sample_items for r in by_item[item]]
        x=np.asarray([float(r["alpha"]) for r in sampled]); y=np.asarray([float(r["delta_soft_sa"]) for r in sampled])
        slope=float(np.polyfit(x,y,1)[0]); plus=float(np.mean([float(r["delta_soft_sa"]) for r in sampled if float(r["alpha"])==10])); minus=float(np.mean([float(r["delta_soft_sa"]) for r in sampled if float(r["alpha"])==-10]))
        return slope,plus,minus
    observed=stats(items); boot=np.asarray([stats(rng.choice(items,size=len(items),replace=True).tolist()) for _ in range(repeats)])
    p=[_one_sided_p(boot[:,0],True),_one_sided_p(boot[:,1],True),_one_sided_p(boot[:,2],False)]
    return {"slope":observed[0],"plus10_mean_delta":observed[1],"minus10_mean_delta":observed[2],
            "slope_ci":np.percentile(boot[:,0],[2.5,97.5]).tolist(),"plus10_ci":np.percentile(boot[:,1],[2.5,97.5]).tolist(),
            "minus10_ci":np.percentile(boot[:,2],[2.5,97.5]).tolist(),"component_pvalues":p,"intersection_union_p":max(p),
            "bootstrap_repeats":repeats,"item_count":len(items)}

def select_probe_candidates(rows: Sequence[dict[str, Any]], *, repeats: int=BOOTSTRAP_REPEATS, seed: int=SEED) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    groups=defaultdict(list)
    for row in rows:
        if row.get("status")=="completed" and row.get("position") in PROBE_POSITIONS and row.get("direction_type","true")=="true":
            groups[(str(row["position"]),int(row["layer"]))].append(row)
    metrics=[]
    for index,key in enumerate(sorted(groups)):
        cell=_bootstrap_cell(groups[key],repeats,seed+index)
        metrics.append({"position":key[0],"layer":key[1],**cell})
    q=bh_fdr([r["intersection_union_p"] for r in metrics])
    selected=[]
    for row,value in zip(metrics,q):
        row["q_value"]=value
        row["point_direction_valid"]=row["slope"]>0 and row["plus10_mean_delta"]>0 and row["minus10_mean_delta"]<0
        row["selected_for_probe"]=bool(row["point_direction_valid"] and value<0.05)
        if row["selected_for_probe"]: selected.append(dict(row))
    return selected,metrics

def _mean_ci(rows: Sequence[dict[str,Any]], repeats: int, seed: int) -> tuple[float,list[float]]:
    by_item=defaultdict(list)
    for row in rows: by_item[str(row["item_id"])].append(float(row["delta_soft_sa"]))
    items=sorted(by_item); value=float(np.mean([v for values in by_item.values() for v in values])); rng=np.random.default_rng(seed)
    boot=[]
    for _ in range(repeats):
        chosen=rng.choice(items,size=len(items),replace=True); boot.append(float(np.mean([v for item in chosen for v in by_item[str(item)]])))
    return value,np.percentile(boot,[2.5,97.5]).tolist()

def build_steering_metrics(rows: Sequence[dict[str,Any]], repeats: int, seed: int) -> list[dict[str,Any]]:
    groups=defaultdict(list)
    completed=[r for r in rows if r.get("status")=="completed"]
    for row in completed:
        for group in ("all",str(row.get("test_side","unknown"))): groups[(row["position"],int(row["layer"]),row["direction_type"],float(row["alpha"]),group)].append(row)
    output=[]
    for index,(key,values) in enumerate(sorted(groups.items(),key=lambda pair:str(pair[0]))):
        mean,ci=_mean_ci(values,repeats,seed+index); output.append({"position":key[0],"layer":key[1],"direction_type":key[2],"alpha":key[3],"group":key[4],"mean_delta_soft_sa":mean,"ci_low":ci[0],"ci_high":ci[1],
            "hard_class_change_rate":sum(bool(r.get("hard_class_changed")) for r in values)/len(values),"hard_class_mean_delta":float(np.mean([r.get("hard_class_delta",0) for r in values])),
            "mean_clean_class_margin_after_steering":float(np.mean([r.get("clean_class_logit_margin_after_steering",float("nan")) for r in values])),
            "invalid_probability_count":sum(not math.isfinite(float(r.get("probability_sum",float("nan")))) or abs(float(r.get("probability_sum",0))-1)>1e-6 for r in values),
            "ceiling_saturation_rate":sum(bool(r.get("ceiling_saturated")) for r in values)/len(values),"floor_saturation_rate":sum(bool(r.get("floor_saturated")) for r in values)/len(values),
            "sample_count":len(values),"item_count":len({r["item_id"] for r in values})})
    return output

def build_dose_metrics(rows: Sequence[dict[str,Any]], repeats: int, seed: int) -> list[dict[str,Any]]:
    groups=defaultdict(list)
    for row in rows:
        if row.get("status")!="completed": continue
        for group in ("all",str(row.get("test_side","unknown"))): groups[(row["position"],int(row["layer"]),row["direction_type"],group)].append(row)
    output=[]
    for index,(key,values) in enumerate(sorted(groups.items(),key=lambda pair:str(pair[0]))):
        cell=_bootstrap_cell(values,repeats,seed+5000+index); output.append({"position":key[0],"layer":key[1],"direction_type":key[2],"group":key[3],**cell})
    return output

def _write_csv(path: Path, rows: Sequence[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text(""); return
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def _plots(root: Path, metrics: Sequence[dict[str,Any]]) -> None:
    import matplotlib.pyplot as plt
    true_all=[r for r in metrics if r["direction_type"]=="true" and r["group"]=="all"]
    positions=sorted({r["position"] for r in true_all}); fig,axes=plt.subplots(1,len(positions),figsize=(5*len(positions),4),squeeze=False)
    for ax,position in zip(axes[0],positions):
        for alpha in sorted({r["alpha"] for r in true_all}):
            data=sorted([r for r in true_all if r["position"]==position and r["alpha"]==alpha],key=lambda r:r["layer"])
            ax.plot([r["layer"] for r in data],[r["mean_delta_soft_sa"] for r in data],marker="o",label=f"a={alpha:g}")
        ax.axhline(0,color="black",ls="--",lw=.8); ax.set_title(position); ax.set_xlabel("zero-based layer")
    axes[0][0].set_ylabel("mean delta soft SA"); axes[0][-1].legend(fontsize=7); fig.tight_layout(); fig.savefig(root/"steering_delta_soft_by_layer.png",dpi=160); plt.close(fig)
    panl=[r for r in metrics if r["direction_type"]=="true" and r["position"]=="P1_PANL" and r["layer"] in (18,20)]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,layer in zip(axes,(18,20)):
        for group in ("all","image_side","text_side"):
            data=sorted([r for r in panl if r["layer"]==layer and r["group"]==group],key=lambda r:r["alpha"])
            if data: ax.plot([r["alpha"] for r in data],[r["mean_delta_soft_sa"] for r in data],marker="o",label=group)
        ax.axhline(0,color="black",ls="--",lw=.8); ax.set_title(f"P1_PANL L{layer}"); ax.set_xlabel("alpha")
    axes[0].set_ylabel("mean delta soft SA"); axes[1].legend(); fig.tight_layout(); fig.savefig(root/"steering_dose_response_panl.png",dpi=160); plt.close(fig)
    extremes=[r for r in true_all if abs(r["alpha"])==10]; best=[]
    for position in positions:
        candidates=[r for r in extremes if r["position"]==position]; best.append(max(candidates,key=lambda r:abs(r["mean_delta_soft_sa"])))
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar([r["position"] for r in best],[r["mean_delta_soft_sa"] for r in best]); ax.axhline(0,color="black",ls="--"); ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(root/"steering_position_comparison.png",dpi=160); plt.close(fig)
    shuffled=[r for r in metrics if r["position"]=="P1_PANL" and r["layer"] in (18,20) and r["group"]=="all"]
    fig,ax=plt.subplots(figsize=(7,4))
    for layer in (18,20):
        for direction in ("true","shuffled"):
            data=sorted([r for r in shuffled if r["layer"]==layer and r["direction_type"]==direction],key=lambda r:r["alpha"])
            if data: ax.plot([r["alpha"] for r in data],[r["mean_delta_soft_sa"] for r in data],marker="o",label=f"L{layer} {direction}")
    ax.axhline(0,color="black",ls="--"); ax.legend(); fig.tight_layout(); fig.savefig(root/"steering_shuffled_control.png",dpi=160); plt.close(fig)

def _natural_ood(output_root: Path, intervention_rows: Sequence[dict[str,Any]]) -> dict[str,Any]:
    manifests=load_jsonl(output_root/"steering"/"test_manifest.jsonl")
    summaries=[]
    for position in ("P1_PANL","P1_PANL_PLUS_1","P1_SAC"):
        for layer in sorted({int(r["layer"]) for r in intervention_rows if r.get("position")==position}):
            vectors=[]
            for row in manifests:
                with np.load(output_root/row["hidden_file"]) as payload: vectors.append(np.asarray(payload[f"{position}__L{layer}"],dtype=np.float64))
            natural_cos=[]; natural_ratio=[]
            for left,right in zip(vectors[::2],vectors[1::2]):
                natural_cos.append(float(np.dot(left,right)/(np.linalg.norm(left)*np.linalg.norm(right)))); natural_ratio.append(float(np.linalg.norm(right)/np.linalg.norm(left)))
            steered=[r for r in intervention_rows if r.get("status")=="completed" and r["position"]==position and int(r["layer"])==layer]
            summaries.append({"position":position,"layer":layer,"natural_pair_count":len(natural_cos),"natural_cosine_percentiles":np.percentile(natural_cos,[5,50,95]).tolist(),
                              "natural_norm_ratio_percentiles":np.percentile(natural_ratio,[5,50,95]).tolist(),"steered_cosine_percentiles":np.percentile([r["activation_cosine"] for r in steered],[5,50,95]).tolist(),
                              "steered_norm_ratio_percentiles":np.percentile([r["activation_norm_ratio"] for r in steered],[5,50,95]).tolist()})
    return {"definition":"paired clean test activations in stable manifest order","cells":summaries}

def main(argv: Sequence[str]|None=None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",default=str(OUTPUT_ROOT)); parser.add_argument("--bootstrap",type=int,default=BOOTSTRAP_REPEATS)
    args=parser.parse_args(argv); root=Path(args.output_root); rows=load_jsonl(root/"steering"/"predictions.jsonl")
    selected,metrics=select_probe_candidates(rows,repeats=args.bootstrap)
    steering_metrics=build_steering_metrics(rows,args.bootstrap,SEED); dose_metrics=build_dose_metrics(rows,args.bootstrap,SEED); _write_csv(root/"steering"/"metrics.csv",steering_metrics); atomic_json(root/"steering"/"metrics.json",{"cells":steering_metrics,"dose_response":dose_metrics})
    _write_csv(root/"steering"/"dose_response_metrics.csv",dose_metrics)
    ood=[{"position":r["position"],"layer":r["layer"],"alpha":r["alpha"],"direction_type":r["direction_type"],"case_id":r["case_id"],"activation_cosine":r["activation_cosine"],"activation_norm_ratio":r["activation_norm_ratio"]} for r in rows if r.get("status")=="completed"]
    _write_csv(root/"steering"/"ood_diagnostics.csv",ood); atomic_json(root/"steering"/"ood_summary.json",_natural_ood(root,rows))
    _plots(root/"steering",steering_metrics)
    atomic_jsonl(root/"steering"/"probe_candidate_manifest.jsonl",selected); atomic_json(root/"steering"/"candidate_metrics.json",{"cells":metrics,"selected_count":len(selected),"sac_excluded":True})
    return 0

if __name__=="__main__": raise SystemExit(main())
