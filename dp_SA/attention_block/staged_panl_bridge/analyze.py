from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any,Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dp_SA.attention_block.analyze import _mean_ci
from dp_SA.io_utils import atomic_json,canonical_hash,load_jsonl

from .core import G_WINDOWS,R_WINDOWS,WINDOW_PAIRS,Cell,cells,effects,iut_q,layer_edges,one_sided_sign_flip

EFFECT_NAMES=("interaction","bridge_gain","matched_gain")
COLORS={"C00":"#d62728","C10":"#1f77b4","C01":"#2ca02c","C11":"#e377c2","CTRL":"#7f7f7f"}


def _atomic_csv(path:Path,rows:list[dict[str,Any]])->None:
    fields=sorted({k for row in rows for k in row});path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent);os.close(fd)
    try:
        with open(temp,"w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows);handle.flush();os.fsync(handle.fileno())
        os.replace(temp,path)
    except Exception:
        try:os.unlink(temp)
        except FileNotFoundError:pass
        raise


def _cell_key(family:str,g:tuple[int,int],r:tuple[int,int])->str:
    if family=="C00":return "C00"
    if family=="C10":return f"C10_G{g[0]}_{g[1]}"
    if family=="C01":return f"C01_R{r[0]}_{r[1]}"
    return f"{family}_G{g[0]}_{g[1]}_R{r[0]}_{r[1]}"


def _item_effects(blocked:list[dict[str,Any]])->list[dict[str,Any]]:
    lookup={(row["case_id"],row["cell_key"]):row for row in blocked};case_ids=sorted({row["case_id"] for row in blocked})
    output=[]
    for case_id in case_ids:
        exemplar=next(row for row in blocked if row["case_id"]==case_id)
        for g,r in WINDOW_PAIRS:
            rows={family:lookup[(case_id,_cell_key(family,g,r))] for family in ("C00","C10","C01","C11","CTRL")}
            margin=effects(*(float(rows[c]["margin"]) for c in ("C00","C10","C01","C11","CTRL")))
            soft=effects(*(float(rows[c]["soft_sa"]) for c in ("C00","C10","C01","C11","CTRL")))
            output.append({"case_id":case_id,"item_id":exemplar["item_id"],"test_side":exemplar["test_side"],
                "g_start":g[0],"g_end":g[1],"r_start":r[0],"r_end":r[1],"pair_label":f"G{g[0]}–{g[1]}→R{r[0]}–{r[1]}",
                **dict(zip(EFFECT_NAMES,margin)),**{f"soft_{k}":v for k,v in zip(EFFECT_NAMES,soft)},
                **{f"margin_{c.lower()}":float(rows[c]["margin"]) for c in rows},
                **{f"disruption_{c.lower()}":float(rows[c]["clean_margin_disruption"]) for c in rows},
                **{f"soft_delta_{c.lower()}":float(rows[c]["soft_sa_delta"]) for c in rows},
                **{f"token_change_{c.lower()}":int(rows[c]["token_changed"]) for c in rows}})
    return output


def _validate(output:Path,blocked:list[dict[str,Any]],expected_n:int)->dict[str,Any]:
    parity=load_jsonl(output/"clean_parity.jsonl");failures=load_jsonl(output/"failures.jsonl");span_rows=load_jsonl(output/"token_spans.jsonl")
    try:
        if len(blocked)!=expected_n*21 or len(parity)!=expected_n or failures:raise ValueError("count/failure gate")
        if len({(r["case_id"],r["cell_key"]) for r in blocked})!=expected_n*21:raise ValueError("duplicate/missing cells")
        span_by_case={r["case_id"]:{k:v for k,v in r.items() if k not in {"case_id","item_id","test_side"}} for r in span_rows};result_lookup={(r["case_id"],r["cell_key"]):r for r in blocked}
        cell_by_key={c.key:c for c in cells()}
        for row in blocked:
            if len(row["attention_diagnostics"]["by_layer"])!=28:raise ValueError("not all layers diagnosed")
            expected=layer_edges(span_by_case[row["case_id"]],cell_by_key[row["cell_key"]])
            for layer in range(28):
                diag=row["attention_diagnostics"]["by_layer"][str(layer)]
                if diag["max_blocked_weight"]!=0 or not diag["finite"] or diag["head_count"]!=28 or diag.get("hook_call_count")!=1 or diag["max_row_sum_error"]>.01:
                    raise ValueError(f"attention diagnostic failed layer {layer}")
                spec=row["edge_specification"][str(layer)]
                if spec["count"]!=len(expected[layer].pairs) or spec["sha256"]!=canonical_hash(expected[layer].pairs):raise ValueError("edge fingerprint mismatch")
        if not all(r["hard_equal"] and r["max_abs_logit_difference"]<=.125 and r["abs_soft_sa_difference"]<=1e-6 for r in parity):raise ValueError("clean parity")
        from .run import c10_parity
        case_ids=sorted({r["case_id"] for r in blocked})
        for case_id in case_ids:
            c00=result_lookup[(case_id,"C00")]
            for key in ("C10_G10_15","C10_G12_17"):c10_parity(c00,result_lookup[(case_id,key)])
    except Exception as exc:return {"passed":False,"error":str(exc),"blocked":len(blocked),"clean":len(parity),"failures":len(failures)}
    return {"passed":True,"error":None,"blocked":len(blocked),"clean":len(parity),"failures":0,
            "all_28_layers_masked":True,"c10_no_sac_panl_leakage":True,"c10_numeric_parity_to_c00":True,
            "c00_single_cached_cell_pair_invariant":True,"c11_sac_answer_blocked_all_layers":True}


def _summaries(items:list[dict[str,Any]],*,bootstrap:int,sign_repeats:int,seed:int):
    all_tests=[];summary=[]
    for index,(g,r) in enumerate(WINDOW_PAIRS):
        pair=[x for x in items if x["g_start"]==g[0] and x["r_start"]==r[0]]
        test={"g_start":g[0],"g_end":g[1],"r_start":r[0],"r_end":r[1],"pair_label":pair[0]["pair_label"]}
        for ei,effect in enumerate(EFFECT_NAMES):
            values=[float(x[effect]) for x in pair];stats=_mean_ci(values,repeats=bootstrap,seed=seed+index*101+ei)
            test.update({f"{effect}_{k}":v for k,v in stats.items()});test[f"p_{effect}"]=one_sided_sign_flip(values,seed=seed+2000+index*101+ei,repeats=sign_repeats)
        all_tests.append(test)
        for group in ("all","image_side","text_side"):
            subset=pair if group=="all" else [x for x in pair if x["test_side"]==group]
            row={"g_start":g[0],"g_end":g[1],"r_start":r[0],"r_end":r[1],"pair_label":pair[0]["pair_label"],"group":group,"n":len(subset)}
            metrics=list(EFFECT_NAMES)+[f"soft_{x}" for x in EFFECT_NAMES]
            metrics += [f"{prefix}_{c}" for prefix in ("margin","disruption","soft_delta","token_change") for c in ("c00","c10","c01","c11","ctrl")]
            for mi,metric in enumerate(metrics):
                stats=_mean_ci([float(x[metric]) for x in subset],repeats=bootstrap,seed=seed+5000+index*307+mi*7+len(group))
                for key,value in stats.items():row[f"{metric}_{key}"]=value
            summary.append(row)
    for test,q in zip(all_tests,iut_q(all_tests)):
        test["p_iut"]=max(test[f"p_{effect}"] for effect in EFFECT_NAMES);test["q_iut_bh"]=q
        test["supported"]=bool(all(test[f"{effect}_mean"]>0 and test[f"{effect}_ci_low"]>0 for effect in EFFECT_NAMES) and q<.05)
        all_row=next(row for row in summary if row["group"]=="all" and row["g_start"]==test["g_start"] and row["r_start"]==test["r_start"])
        all_row.update({k:v for k,v in test.items() if k.startswith("p_") or k.startswith("q_") or k=="supported"})
    return all_tests,summary


def _plots(output:Path,summary:list[dict[str,Any]])->list[str]:
    plot_dir=output/"plots";plot_dir.mkdir(exist_ok=True);all_rows=[r for r in summary if r["group"]=="all"]
    labels=[r["pair_label"] for r in all_rows];x=np.arange(len(labels));files=[]
    def save(fig,name):
        fig.tight_layout()
        for suffix in ("png","pdf"):
            path=plot_dir/f"{name}.{suffix}";fig.savefig(path,dpi=220);files.append(str(path.relative_to(output)))
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5));ax.plot(x,[r["interaction_mean"] for r in all_rows],marker="o",color="#6a3d9a");ax.axhline(0,color="black",lw=.8);ax.set_xticks(x,labels,rotation=30,ha="right");ax.set_ylabel("Factorial interaction");ax.set_title("Staged PANL factorial interaction");ax.grid(axis="y",alpha=.2);save(fig,"factorial_interaction")
    fig,ax=plt.subplots(figsize=(10,4.7))
    for c in ("C00","C10","C01","C11"):ax.plot(x,[r[f"margin_{c.lower()}_mean"] for r in all_rows],marker="o",label=c,color=COLORS[c])
    ax.set_xticks(x,labels,rotation=30,ha="right");ax.set_ylabel("Fixed-clean-class margin");ax.set_title("Factorial condition margins");ax.legend(ncol=4,frameon=False);ax.grid(axis="y",alpha=.2);save(fig,"factorial_condition_margins")
    fig,ax=plt.subplots(figsize=(10,4.7));ax.plot(x,[r["bridge_gain_mean"] for r in all_rows],marker="o",label="PANL bridge",color=COLORS["C11"]);ax.plot(x,[r["margin_ctrl_mean"]-r["margin_c00_mean"] for r in all_rows],marker="o",label="PANL+1 control",color=COLORS["CTRL"]);ax.axhline(0,color="black",lw=.8);ax.set_xticks(x,labels,rotation=30,ha="right");ax.set_ylabel("Gain over C00");ax.set_title("PANL bridge vs PANL+1 control");ax.legend(frameon=False);ax.grid(axis="y",alpha=.2);save(fig,"bridge_vs_control")
    fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=True)
    for c in ("C00","C10","C01","C11"):
        axes[0].plot(x,[r[f"soft_delta_{c.lower()}_mean"] for r in all_rows],marker="o",label=c,color=COLORS[c]);axes[1].plot(x,[r[f"token_change_{c.lower()}_mean"] for r in all_rows],marker="o",label=c,color=COLORS[c])
    axes[0].axhline(0,color="black",lw=.8);axes[0].set_ylabel("Soft-SA delta");axes[0].legend(ncol=4,frameon=False);axes[1].set_ylabel("Token change rate");axes[1].set_xticks(x,labels,rotation=30,ha="right");axes[0].set_title("Soft-SA and hard token change");save(fig,"soft_sa_and_token_change")
    for metric,title,ylabel in (("disruption","Logit diff change by factorial condition","Logit diff change (clean margin − blocked margin)"),
                                ("token_change","Token change rate by factorial condition","Token change rate")):
        fig,ax=plt.subplots(figsize=(10,4.8))
        plotted=[]
        for offset,c in zip((-.09,-.03,.03,.09),("C00","C10","C01","C11")):
            values=[r[f"{metric}_{c.lower()}_mean"] for r in all_rows];plotted.extend(values)
            ax.plot(x+offset,values,marker="o",linewidth=2,label=c,color=COLORS[c])
        low,high=min(plotted),max(plotted);padding=max((high-low)*.18,.005)
        ax.set_ylim(low-padding,high+padding)
        ax.set_xticks(x,labels,rotation=30,ha="right");ax.set_xlabel("Gathering → readout window pair");ax.set_ylabel(ylabel);ax.set_title(title)
        ax.legend(ncol=4,frameon=False);ax.grid(axis="y",alpha=.2);save(fig,"four_conditions_logit_diff_change" if metric=="disruption" else "four_conditions_token_change_rate")
    matrix=np.full((len(G_WINDOWS),len(R_WINDOWS)),np.nan)
    for row in all_rows:matrix[G_WINDOWS.index((row["g_start"],row["g_end"])),R_WINDOWS.index((row["r_start"],row["r_end"]))]=row["interaction_mean"]
    fig,ax=plt.subplots(figsize=(7,3.8));image=ax.imshow(matrix,cmap="RdBu_r",aspect="auto",vmin=-np.nanmax(abs(matrix)),vmax=np.nanmax(abs(matrix)))
    ax.set_xticks(range(len(R_WINDOWS)),[f"R{a}–{b}" for a,b in R_WINDOWS]);ax.set_yticks(range(len(G_WINDOWS)),[f"G{a}–{b}" for a,b in G_WINDOWS]);ax.set_title("Interaction heatmap")
    for i in range(len(G_WINDOWS)):
        for j in range(len(R_WINDOWS)):
            if np.isfinite(matrix[i,j]):ax.text(j,i,f"{matrix[i,j]:.3f}",ha="center",va="center",fontsize=9)
    fig.colorbar(image,ax=ax,label="Interaction");save(fig,"interaction_heatmap")
    return files


def analyze(output:Path)->dict[str,Any]:
    output=output.resolve();config=json.loads((output/"run_config.json").read_text());blocked=load_jsonl(output/"blocked_results.jsonl");expected_n=4 if config["smoke"] else 100
    technical=_validate(output,blocked,expected_n)
    if not technical["passed"]:raise RuntimeError(f"Technical validation failed: {technical['error']}")
    items=_item_effects(blocked);_atomic_csv(output/"item_level_effects.csv",items)
    tests,summary=_summaries(items,bootstrap=int(config["bootstrap_repeats"]),sign_repeats=int(config["sign_flip_repeats"]),seed=int(config["seed"]));_atomic_csv(output/"window_pair_summary.csv",summary)
    supported=any(row["supported"] for row in tests);plots=_plots(output,summary)
    interpretation=("Staged direct PANL bridge supported in at least one preregistered pair; this supports a position-specific sufficient path, not uniqueness or natural-computation necessity." if supported else "不支持该分阶段直接两跳路径；不能否定 PANL 经 instruction/downstream tokens 的多跳中继。")
    result={"experiment":"staged_panl_bridge","n":expected_n,"technical_validation":technical,"tests":tests,"staged_direct_panl_bridge_supported":supported,"interpretation":interpretation,"plots":plots};atomic_json(output/"analysis.json",result)
    lines=["# Staged PANL bridge","",f"- n: {expected_n}","- Technical validation: PASS",f"- Conclusion: {interpretation}","","| Gathering → Readout | I [95% CI] | Bridge [95% CI] | Matched [95% CI] | p_IUT | q_IUT | Supported |","|---|---:|---:|---:|---:|---:|:---:|"]
    for row in tests:
        vals=[f"{row[f'{e}_mean']:.4f} [{row[f'{e}_ci_low']:.4f}, {row[f'{e}_ci_high']:.4f}]" for e in EFFECT_NAMES]
        lines.append(f"| {row['pair_label']} | {' | '.join(vals)} | {row['p_iut']:.4g} | {row['q_iut_bh']:.4g} | {'yes' if row['supported'] else 'no'} |")
    lines += ["","Side results in `window_pair_summary.csv` are descriptive only. Positive clean-margin disruption means blocking reduced the fixed clean-class margin.",""]
    (output/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    atomic_json(output/"analysis_completion.json",{"status":"complete","technical_validation_passed":True,"staged_direct_panl_bridge_supported":supported,"outputs":["item_level_effects.csv","window_pair_summary.csv","analysis.json","summary.md",*plots]})
    return result


def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args(argv);analyze(a.output_dir);return 0


if __name__=="__main__":raise SystemExit(main())
