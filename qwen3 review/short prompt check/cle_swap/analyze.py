from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from dp_SA.io_utils import atomic_json, load_jsonl
from .config import BOOTSTRAP_REPEATS, DIRECTIONS, EXPECTED_CASES, LAYERS, OUTPUT_ROOT, SEED

GROUPS=("overall","text_side","image_side","answer_equal_macro")
METRICS=("delta_sa","abs_delta_sa","logit_change_diff","token_changed","hard_class_changed",
         "semantic_movement","semantic_distance_change","moved_toward_source",
         "raw_label_movement","raw_label_distance_change","semantic_preference_contrast")

def _csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True); fields=sorted({k for r in rows for k in r})
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise

def _select(rows,group):
    return rows if group in ("overall","answer_equal_macro") else [r for r in rows if r["test_side"]==group]

def _aggregate(rows,field,group):
    rows=_select(rows,group)
    if group=="answer_equal_macro":
        values=[]
        for answer in sorted({r["answer"] for r in rows}):
            values.append(np.mean([float(r[field]) for r in rows if r["answer"]==answer]))
        return float(np.mean(values))
    return float(np.mean([float(r[field]) for r in rows]))

def _bootstrap(rows,field,group,seed,repeats):
    selected=_select(rows,group); rng=np.random.default_rng(seed); point=_aggregate(rows,field,group); vals=[]
    for _ in range(repeats):
        sample=list(rng.choice(selected,len(selected),replace=True))
        vals.append(_aggregate(sample,field,group))
    lo,hi=np.quantile(vals,[.025,.975]);return point,float(lo),float(hi)

def analyze(output_root:Path=OUTPUT_ROOT,repeats:int=BOOTSTRAP_REPEATS):
    root=Path(output_root); rows=load_jsonl(root/"artifacts/trials.jsonl")
    answers={r["case_id"]:r["answer"] for r in load_jsonl(root/"artifacts/manifests/test_manifest.jsonl")}
    rows=[{**r,"answer":answers[r["case_id"]]} for r in rows]
    expected=EXPECTED_CASES*len(LAYERS)*len(DIRECTIONS)
    if len(rows)!=expected or len({(r["case_id"],r["direction"],r["layer"]) for r in rows})!=expected:
        raise ValueError(f"Swap grid incomplete/duplicated: {len(rows)}/{expected}")
    summary=[];counter=0
    for direction in DIRECTIONS:
        for layer in LAYERS:
            cell=[r for r in rows if r["direction"]==direction and int(r["layer"])==layer]
            for group in GROUPS:
                for metric in METRICS:
                    mean,lo,hi=_bootstrap(cell,metric,group,SEED+counter,repeats);counter+=1
                    summary.append({"direction":direction,"layer":layer,"group":group,"metric":metric,
                                    "mean":mean,"ci95_low":lo,"ci95_high":hi,"case_count":len(_select(cell,group)),"bootstrap_repeats":repeats})
    paired=[]
    by={(r["case_id"],int(r["layer"]),r["direction"]):r for r in rows}
    for layer in LAYERS:
        cell=[]
        for case in sorted({r["case_id"] for r in rows}):
            a=by[(case,layer,"reverse_to_short")];b=by[(case,layer,"short_to_reverse")]
            cell.append({"case_id":case,"test_side":a["test_side"],"target_clean_raw_class":a["target_clean_raw_class"],
                         "answer":answers[case],
                         "bidirectional_semantic_movement":(float(a["semantic_movement"])+float(b["semantic_movement"]))/2,
                         "both_toward_source":int(a["moved_toward_source"] and b["moved_toward_source"]),
                         "bidirectional_semantic_preference":(float(a["semantic_preference_contrast"])+float(b["semantic_preference_contrast"]))/2})
        for group in GROUPS:
            for metric in ("bidirectional_semantic_movement","both_toward_source","bidirectional_semantic_preference"):
                mean,lo,hi=_bootstrap(cell,metric,group,SEED+50000+counter,repeats);counter+=1
                paired.append({"layer":layer,"group":group,"metric":metric,"mean":mean,"ci95_low":lo,"ci95_high":hi,
                               "case_count":len(_select(cell,group)),"bootstrap_repeats":repeats})
    associations=[]
    for direction in DIRECTIONS:
        for layer in LAYERS:
            cell=[r for r in rows if r["direction"]==direction and int(r["layer"])==layer]
            delta=np.asarray([r["delta_sa"] for r in cell],float); semantic=np.asarray([r["semantic_gap"] for r in cell],float); raw=np.asarray([r["raw_label_gap"] for r in cell],float)
            def stats(gap):
                slope=float((gap@delta)/(gap@gap)) if float(gap@gap)>0 else math.nan
                corr=float(np.corrcoef(gap,delta)[0,1]) if np.std(gap)>0 and np.std(delta)>0 else math.nan
                return slope,corr
            ss,sc=stats(semantic);rs,rc=stats(raw)
            associations.append({"direction":direction,"layer":layer,"semantic_gap_slope":ss,"semantic_gap_pearson":sc,
                                 "raw_label_gap_slope":rs,"raw_label_gap_pearson":rc})
    _csv(root/"tables/direction_layer_metrics.csv",summary);_csv(root/"tables/bidirectional_paired_metrics.csv",paired);_csv(root/"tables/gap_associations.csv",associations)
    _plots(root,summary,paired)
    result={"status":"complete","case_count":EXPECTED_CASES,"swap_trial_count":len(rows),"bootstrap_repeats":repeats,
            "summary_rows":len(summary),"paired_rows":len(paired)}
    atomic_json(root/"analysis_summary.json",result);return result

def _plots(root,summary,paired):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    out=root/"figures";out.mkdir(parents=True,exist_ok=True)
    labels={"reverse_to_short":"Reverse CLE → short","short_to_reverse":"Short CLE → reverse"}
    def plot_metrics(filename,metrics,titles):
        fig,axes=plt.subplots(1,len(metrics),figsize=(6*len(metrics),4.5))
        if len(metrics)==1:axes=[axes]
        for ax,metric,title in zip(axes,metrics,titles):
            for direction in DIRECTIONS:
                p=sorted([r for r in summary if r["direction"]==direction and r["group"]=="overall" and r["metric"]==metric],key=lambda r:r["layer"])
                x=[r["layer"] for r in p];y=np.asarray([r["mean"] for r in p]);err=np.asarray([[r["mean"]-r["ci95_low"] for r in p],[r["ci95_high"]-r["mean"] for r in p]])
                ax.errorbar(x,y,yerr=err,marker="o",capsize=3,label=labels[direction])
            ax.axhline(0,color="black",lw=.8);ax.set_xticks(LAYERS);ax.set_xlabel("CLE swap layer");ax.set_title(title);ax.grid(axis="y",alpha=.2)
        axes[-1].legend(frameon=False);fig.tight_layout();fig.savefig(out/filename,dpi=220);plt.close(fig)
    plot_metrics("delta_sa_by_layer.png",("delta_sa",),("Canonical ΔSA",))
    plot_metrics("logit_and_token_change.png",("logit_change_diff","token_changed"),("Clean-class logit margin loss","Token change rate"))
    plot_metrics("semantic_vs_raw_attraction.png",("semantic_movement","raw_label_movement"),("Movement toward source semantics","Movement toward source raw-label code"))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,metric,title in zip(axes,("bidirectional_semantic_movement","both_toward_source"),("Mean bidirectional semantic movement","Both directions move toward source")):
        p=sorted([r for r in paired if r["group"]=="overall" and r["metric"]==metric],key=lambda r:r["layer"])
        x=[r["layer"] for r in p];y=np.asarray([r["mean"] for r in p]);err=np.asarray([[r["mean"]-r["ci95_low"] for r in p],[r["ci95_high"]-r["mean"] for r in p]])
        ax.errorbar(x,y,yerr=err,marker="o",capsize=3);ax.axhline(0,color="black",lw=.8);ax.set_xticks(LAYERS);ax.set_xlabel("CLE swap layer");ax.set_title(title);ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(out/"bidirectional_paired_effects.png",dpi=220);plt.close(fig)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--output-root",type=Path,default=OUTPUT_ROOT);p.add_argument("--repeats",type=int,default=BOOTSTRAP_REPEATS);a=p.parse_args(argv)
    print(json.dumps(analyze(a.output_root,a.repeats),ensure_ascii=False));return 0
if __name__=="__main__":raise SystemExit(main())
