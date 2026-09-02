from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PROBE_LAYERS, PROBE_POSITIONS, RESULTS_ROOT, TARGETS
from .io_utils import atomic_json, atomic_text, ensure_layout, load_jsonl


def plot_probe_metrics(root: Path, rows: Sequence[dict[str,Any]]) -> list[str]:
    styles={position:(color,style) for position,color,style in zip(PROBE_POSITIONS,("#1f77b4","#ff7f0e","#2ca02c","#d62728"),("-","--","-.",":"),strict=True)}; files=[]
    titles={"text_chosen_confidence":"Text chosen confidence","image_chosen_confidence":"Image chosen confidence","text_fixed_answer_confidence":"Text fixed-answer confidence","image_fixed_answer_confidence":"Image fixed-answer confidence"}
    for metric,filename,ylabel in (("r2","probe_r2.png","R²"),("spearman","probe_spearman.png","Spearman"),("pearson","probe_pearson.png","Pearson")):
        fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True); axes=axes.ravel()
        for axis,target in zip(axes,TARGETS,strict=True):
            for position in PROBE_POSITIONS:
                cells=sorted((r for r in rows if r["target"]==target and r["position"]==position),key=lambda r:int(r["layer"])); x=[int(r["layer"]) for r in cells]; y=[float(r[metric]) for r in cells]; low=[max(0,float(r[metric])-float(r[f"{metric}_ci_low"])) for r in cells]; high=[max(0,float(r[f"{metric}_ci_high"])-float(r[metric])) for r in cells]; color,style=styles[position]
                axis.errorbar(x,y,yerr=[low,high],color=color,linestyle=style,marker="o",capsize=2,label=position)
            axis.axhline(0,color="black",lw=.8,ls="--"); axis.set_title(titles[target]); axis.set_xticks(PROBE_LAYERS); axis.set_xlabel("Zero-based decoder layer"); axis.set_ylabel(ylabel); axis.legend(fontsize=7)
        fig.tight_layout(); path=root/"confidence_probe/figures"/filename; fig.savefig(path,dpi=300); plt.close(fig); files.append(str(path.relative_to(root)))
    return files


def analyze(root: Path) -> dict[str,Any]:
    scores=load_jsonl(root/"unimodal_confidence/artifacts/calibrated_scores/unimodal_scores.jsonl"); joined=load_jsonl(root/"unimodal_confidence/artifacts/predictions/phase1_confidence_joined.jsonl")
    forbidden={"fixed_answer","fixed_answer_confidence","fixed_answer_log_odds","G_C","G_L"}
    if any(forbidden & set(row) for row in scores): raise AssertionError("Unique score table contains record-level fixed-answer data")
    if any(not {"fixed_answer","text_fixed_answer_confidence","image_fixed_answer_confidence","G_C","G_L"} <= set(row) for row in joined): raise AssertionError("Joined table misses record-level confidence")
    path=root/"confidence_probe/tables/probe_metrics.csv"
    with path.open(newline="",encoding="utf-8") as handle: metrics=list(csv.DictReader(handle))
    figures=plot_probe_metrics(root,metrics)
    lines=["# Unimodal logit confidence and Phase-1 probe summary","","- Confidence is an external readout of the restricted candidate-output distribution.","- Chosen confidence and fixed-answer confidence are different quantities.","- Fixed-answer confidence is computed only after joining a unique unimodal distribution to each Phase 1 record's own fixed answer.","- Entropy difficulty and confidence are different quantities.","- A successful linear probe only shows that confidence is linearly decodable from hidden state.","- Probe performance and correlation do not establish that confidence causally drives verbal source attribution.","- Establishing causality still requires confidence-matched intervention, steering, or relay-rescue experiments.",""]
    atomic_text(root/"summary.md","\n".join(lines)); summary={"status":"complete","unique_score_count":len(scores),"joined_record_count":len(joined),"probe_cell_count":len(metrics),"figures":figures,"summary":"summary.md"}; atomic_json(root/"confidence_probe/progress/analyze.json",summary); return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",default=str(RESULTS_ROOT)); parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv); root=ensure_layout(args.output_root,resume=True); print(json.dumps(analyze(root),ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
