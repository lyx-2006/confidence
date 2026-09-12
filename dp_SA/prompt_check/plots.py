from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def _read(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))


def make_plots(root:Path)->dict[str,Any]:
    import matplotlib.pyplot as plt
    root.joinpath("figures").mkdir(parents=True,exist_ok=True);created=[]
    path=root/"tables/behavior_pairwise_correlations.csv"
    if path.is_file():
        case=_read(root/"artifacts/behavior_case_level.csv");by={(r["template"],r["case_id"]):float(r["canonical_soft_sa"]) for r in case};ids=sorted({r["case_id"] for r in case if r["template"]=="T0"});fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,template in zip(axes,("T1","T2","T3")):
            if not all((template,i) in by for i in ids):continue
            x=[by["T0",i] for i in ids];y=[by[template,i] for i in ids];ax.scatter(x,y,s=10,alpha=.55);ax.plot([0,1],[0,1],color="black",lw=.8);ax.set(title=f"T0 vs {template}",xlabel="T0 canonical soft SA",ylabel=f"{template} canonical soft SA",xlim=(0,1),ylim=(0,1));ax.grid(alpha=.2)
        fig.tight_layout();out=root/"figures/behavior_pairwise_scatter.png";fig.savefig(out,dpi=250);plt.close(fig);created.append(out)
    path=root/"tables/t0_probe_transfer_metrics.csv"
    if path.is_file():
        rows=_read(path)
        for metric,filename in (("r2","t0_probe_transfer_r2.png"),("pearson","t0_probe_transfer_correlations.png")):
            fig,axes=plt.subplots(1,4,figsize=(15,3.7),sharey=False)
            r2_ticks=None
            if metric=="r2":
                # R² can be strongly negative while the interpretable positive range is compact.
                # Compress only the negative branch so positive values remain readable at 0.1 spacing.
                selected_all=[r for r in rows if r["target_reference"]=="tx_sa"]
                # Keep the negative display range bounded for readability; values below -9 are clipped.
                negative_ticks=[-9.0,-6.0,-3.0]
                maximum=max(float(r["r2_ci_high"]) for r in selected_all)
                positive_ticks=list(np.arange(0.0,0.1*(int(np.ceil(maximum/0.1))+1),0.1))
                r2_ticks=negative_ticks+positive_ticks[1:]
                positive_limit=max(0.1, max(positive_ticks))
                negative_scale=positive_limit/9.0
                transform=lambda value: value if value>=0 else value*negative_scale
            for ax,position in zip(axes,("P1_LAT","P1_PANL","P1_CLASS_LIST_END","P1_SAC")):
                for template,color in zip(("T1","T2","T3"),("#2166ac","#4daf4a","#b2182b")):
                    selected=sorted([r for r in rows if r["template"]==template and r["position"]==position and r["target_reference"]=="tx_sa"],key=lambda r:int(r["layer"]))
                    x=[int(r["layer"]) for r in selected]; y=[float(r[metric]) for r in selected]
                    if metric=="r2":
                        y_plot=[transform(v) for v in y]
                        low=[transform(float(r["r2_ci_low"])) for r in selected]; high=[transform(float(r["r2_ci_high"])) for r in selected]
                        ax.fill_between(x,low,high,color=color,alpha=.14,linewidth=0)
                    else: y_plot=y
                    ax.plot(x,y_plot,marker="o",label=template,color=color)
                if metric=="r2":ax.axhline(0,color="black",lw=.7)
                ax.set_title(position);ax.set_xlabel("Layer");ax.grid(alpha=.2)
                if metric=="r2" and r2_ticks is not None:
                    ax.set_yticks([transform(v) for v in r2_ticks]);ax.set_yticklabels([f"{v:g}" for v in r2_ticks],fontsize=5,rotation=45,ha="right");ax.set_ylabel("R² (compressed negative axis; 95% CI)")
                    ax.set_ylim(-positive_limit, positive_limit)
            if metric!="r2": axes[0].set_ylabel(metric)
            axes[-1].legend();fig.tight_layout();out=root/"figures"/filename;fig.savefig(out,dpi=250);plt.close(fig);created.append(out)
    path=root/"tables/vector_geometry.csv"
    if path.is_file():
        rows=[r for r in _read(path) if r["row_type"]=="global_recipient_equal_summary"];fig,ax=plt.subplots(figsize=(7,4.5))
        for template,color in zip(("T1","T2","T3"),("#2166ac","#4daf4a","#b2182b")):
            grouped={layer:[] for layer in sorted({int(r["layer"]) for r in rows})}
            for r in rows:
                if r["template"]==template:grouped[int(r["layer"])].append(float(r["raw_cosine"]))
            layer_values=list(grouped);ax.plot(layer_values,[np.mean(grouped[x]) for x in layer_values],marker="o",label=template,color=color)
        ax.axhline(0,color="black",lw=.7);ax.set(xlabel="Layer",ylabel="T0–Tx signed cosine",title="Independent vector geometry");ax.legend();ax.grid(alpha=.2);fig.tight_layout();out=root/"figures/vector_cosine_by_layer.png";fig.savefig(out,dpi=250);plt.close(fig);created.append(out)
    path=root/"tables/steering_transfer_effects.csv"
    if path.is_file():
        rows=[r for r in _read(path) if r["aggregation"]=="answer_equal" and float(r["alpha"])!=0];fig,ax=plt.subplots(figsize=(8,4.8))
        for template,color in zip(("T1","T2","T3"),("#2166ac","#4daf4a","#b2182b")):
            for alpha,style in ((-2.,"--"),(2.,"-")):
                selected=sorted([r for r in rows if r["template"]==template and float(r["alpha"])==alpha],key=lambda r:int(r["layer"]));ax.plot([int(r["layer"]) for r in selected],[float(r["mean_delta_sa"]) for r in selected],style,marker="o",color=color,label=f"{template} a={alpha:+g}")
        ax.axhline(0,color="black",lw=.7);ax.set(xlabel="Layer",ylabel="Mean canonical ΔSA",title="T0 vector transfer");ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2);fig.tight_layout();out=root/"figures/steering_transfer_by_template.png";fig.savefig(out,dpi=250);plt.close(fig);created.append(out)
    return {"status":"complete","figures":[str(p) for p in created]}
