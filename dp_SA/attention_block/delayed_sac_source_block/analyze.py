from __future__ import annotations

import argparse,csv,json,os,tempfile
from pathlib import Path
from typing import Any,Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dp_SA.attention_block.analyze import _mean_ci
from dp_SA.io_utils import atomic_json,load_jsonl
from .run import CONDITIONS,NEW_CONDITION,WINDOWS

LABELS={"sac_to_evidence":"SAC→Evidence","sac_to_answer":"SAC→Answer","sac_to_evidence_answer":"SAC→E+A","sac_to_panl":"SAC→PANL","sac_to_panl_plus_1":"SAC→PANL+1"}
COLORS={"sac_to_evidence":"#d62728","sac_to_answer":"#1f77b4","sac_to_evidence_answer":"#2ca02c","sac_to_panl":"#e377c2","sac_to_panl_plus_1":"#9467bd"}


def _csv(path:Path,rows:list[dict[str,Any]])->None:
    fields=sorted({k for row in rows for k in row});fd,temp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent);os.close(fd)
    try:
        with open(temp,"w",newline="",encoding="utf-8") as handle:w=csv.DictWriter(handle,fieldnames=fields);w.writeheader();w.writerows(rows);handle.flush();os.fsync(handle.fileno())
        os.replace(temp,path)
    except Exception:
        try:os.unlink(temp)
        except FileNotFoundError:pass
        raise


def _validate(output:Path,rows:list[dict[str,Any]],n:int)->dict[str,Any]:
    parity=load_jsonl(output/"clean_parity.jsonl");failures=load_jsonl(output/"failures.jsonl");expected=n*len(CONDITIONS)*len(WINDOWS)
    try:
        if len(rows)!=expected or len({(r["case_id"],r["condition"],r["window_start"]) for r in rows})!=expected:raise ValueError(f"blocked cells {len(rows)}/{expected}")
        if len(parity)!=n or failures:raise ValueError("clean/failure gate")
        if not all(r["hard_equal"] and r["max_abs_logit_difference"]<=.125 and r["abs_soft_sa_difference"]<=1e-6 for r in parity):raise ValueError("clean parity")
        new=[r for r in rows if r["condition"]==NEW_CONDITION]
        if len(new)!=n*len(WINDOWS) or any(r.get("provenance")!="new_forward" for r in new):raise ValueError("new E+A provenance/count")
        for row in new:
            diag=row["attention_diagnostics"]
            if len(diag["by_layer"])!=12:raise ValueError("diagnostic layer count")
            for layer in diag["by_layer"].values():
                if layer["max_blocked_weight"]!=0 or not layer["finite"] or layer["head_count"]!=28 or layer.get("hook_call_count")!=1 or layer["max_row_sum_error"]>.01:raise ValueError("attention diagnostic")
    except Exception as exc:return {"passed":False,"error":str(exc),"blocked":len(rows),"clean":len(parity),"failures":len(failures)}
    return {"passed":True,"error":None,"blocked":len(rows),"clean":len(parity),"failures":0,"reused_cells":n*4*5,"new_evidence_answer_cells":n*5}


def _summaries(rows:list[dict[str,Any]],repeats:int,seed:int)->list[dict[str,Any]]:
    output=[]
    for wi,(start,end) in enumerate(WINDOWS):
        for ci,condition in enumerate(CONDITIONS):
            cell=[r for r in rows if r["condition"]==condition and int(r["window_start"])==start]
            for group in ("all","image_side","text_side"):
                subset=cell if group=="all" else [r for r in cell if r["test_side"]==group]
                for mi,(metric,source) in enumerate((("logit_diff_change","logit_margin_disruption"),("token_change_rate","first_token_changed"),("soft_sa_delta","delta_soft_sa"))):
                    output.append({"condition":condition,"condition_label":LABELS[condition],"window_start":start,"window_end":end,"window_center":(start+end)/2,"group":group,"metric":metric,"n":len(subset),
                                   **_mean_ci([float(r[source]) for r in subset],repeats=repeats,seed=seed+wi*307+ci*31+mi*7+len(group))})
    return output


def _plots(output:Path,summary:list[dict[str,Any]])->list[str]:
    plot=output/"plots";plot.mkdir(exist_ok=True);files=[];allrows=[r for r in summary if r["group"]=="all"]
    for metric,ylabel,name in (("logit_diff_change","Logit diff change (clean margin − blocked margin)","logit_diff_change"),("token_change_rate","Token change rate","token_change_rate")):
        fig,ax=plt.subplots(figsize=(8,4.8))
        for condition in CONDITIONS:
            values=sorted((r for r in allrows if r["condition"]==condition and r["metric"]==metric),key=lambda r:r["window_start"])
            ax.plot([r["window_center"] for r in values],[r["mean"] for r in values],marker="o",linewidth=2,color=COLORS[condition],label=LABELS[condition])
        if metric=="logit_diff_change":ax.axhline(0,color="black",lw=.8,alpha=.6)
        ax.set_xticks([(a+b)/2 for a,b in WINDOWS],[f"L{a}–{b}" for a,b in WINDOWS]);ax.set_xlabel("Center of selectively blocked 12-layer window");ax.set_ylabel(ylabel);ax.set_title(f"Delayed-SA: {ylabel}");ax.legend(frameon=False,ncol=2);ax.grid(axis="y",alpha=.2);fig.tight_layout()
        for suffix in ("png","pdf"):
            path=plot/f"{name}.{suffix}";fig.savefig(path,dpi=220);files.append(str(path.relative_to(output)))
        plt.close(fig)
    return files


def analyze(output:Path)->dict[str,Any]:
    output=output.resolve();config=json.loads((output/"run_config.json").read_text());rows=load_jsonl(output/"blocked_results.jsonl");n=4 if config["smoke"] else 100;technical=_validate(output,rows,n)
    if not technical["passed"]:raise RuntimeError(technical["error"])
    summary=_summaries(rows,int(config["bootstrap_repeats"]),int(config["seed"]));_csv(output/"summary.csv",summary);plots=_plots(output,summary)
    result={"experiment":"delayed_sac_source_block","n":n,"technical_validation":technical,"conditions":list(CONDITIONS),"windows":list(WINDOWS),"summary":summary,"plots":plots};atomic_json(output/"analysis.json",result)
    lookup={(r["window_start"],r["condition"],r["metric"]):r for r in summary if r["group"]=="all"}
    lines=["# Delayed-SA SAC source blocking supplement","",f"- n: {n}","- Technical validation: PASS","- Logit diff change = clean fixed-class margin − blocked margin.","","| Window | Evidence | Answer | E+A | PANL | PANL+1 |","|---|---:|---:|---:|---:|---:|"]
    for start,end in WINDOWS:lines.append(f"| L{start}–{end} | "+" | ".join(f"{lookup[(start,c,'logit_diff_change')]['mean']:.4f}" for c in CONDITIONS)+" |")
    lines += ["","Token-change rate, soft-SA delta, side means and 95% CIs are in `summary.csv`.",""];(output/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    atomic_json(output/"analysis_completion.json",{"status":"complete","technical_validation_passed":True,"outputs":["summary.csv","analysis.json","summary.md",*plots]});return result


def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args(argv);analyze(a.output_dir);return 0


if __name__=="__main__":raise SystemExit(main())
