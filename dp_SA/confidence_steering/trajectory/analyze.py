from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import BOOTSTRAP_REPEATS, DIRECTIONS, GROUPS, LAYERS, POSITIONS, SEED, TARGETS
from .io_utils import atomic_csv, atomic_json, atomic_text, load_jsonl, require_output_root
from .train_probes import cluster_bootstrap_draws


def persistent_onset(rows: Sequence[dict[str,Any]]) -> int|None:
    by={int(r["layer"]):r for r in rows}
    for layer in sorted(by):
        current,nxt=by[layer],by.get(layer+1)
        if not nxt or not current.get("readout_reliable"):continue
        def signed(r):return r["ci_low"]>0 or r["ci_high"]<0
        if signed(current) and signed(nxt) and np.sign(current["mean"])==np.sign(nxt["mean"]):return layer
    return None


def component_additivity(raw: float, parallel: float, perpendicular: float) -> float:
    return float(raw-parallel-perpendicular)


def _case_aggregate(rows: Sequence[dict[str,Any]], group: str) -> float:
    if group=="all":return float(np.mean([r["value"] for r in rows]))
    field="fixed_answer" if group=="answer_equal_macro" else "family_id"
    grouped=defaultdict(list)
    for r in rows:grouped[str(r[field])].append(float(r["value"]))
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def summarize(rows: Sequence[dict[str,Any]], group: str, draws: Sequence[Sequence[str]]) -> dict[str,float]:
    mean=_case_aggregate(rows,group);by=defaultdict(list)
    for r in rows:by[str(r["family_id"])].append(r)
    values=[]
    for draw in draws:
        sampled=[]
        for index,family in enumerate(draw):
            for row in by[family]:sampled.append({**row,"family_id":f"draw_{index}"})
        values.append(_case_aggregate(sampled,group))
    lo,hi=np.quantile(values,[.025,.975]);return {"mean":mean,"ci_low":float(lo),"ci_high":float(hi)}


def _plot(root:Path, summary:list[dict[str,Any]])->None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    primary=[r for r in summary if r["group"]=="answer_equal_macro"]
    fig,ax=plt.subplots(figsize=(10,5))
    for direction in DIRECTIONS:
        values=[r for r in primary if r["direction"]==direction and r["position"]=="P1_PANL" and r["target"]=="G_L"]
        ax.plot([r["layer"] for r in values],[r["mean"] for r in values],marker="o",label=direction,alpha=.85)
    ax.axhline(0,color="black",lw=.7);ax.set(xlabel="Layer (zero-based)",ylabel="symmetric derivative",title="PANL trajectory — answer-equal macro");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(root/"figures/panl_trajectory.png",dpi=180);plt.close(fig)
    for filename,target,title in (("confidence_trajectory_heatmap.png","G_L","Confidence trajectory"),("sa_trajectory_heatmap.png","final_soft_sa","SA trajectory")):
        matrix=np.full((len(POSITIONS)*len(DIRECTIONS),len(LAYERS)),np.nan);reliable=np.zeros_like(matrix,bool);labels=[]
        for i,(position,direction) in enumerate((p,d) for p in POSITIONS for d in DIRECTIONS):
            labels.append(f"{position}/{direction}")
            for j,layer in enumerate(LAYERS):
                hit=[r for r in primary if r["position"]==position and r["direction"]==direction and r["layer"]==layer and r["target"]==target]
                if hit:matrix[i,j]=hit[0]["mean"];reliable[i,j]=hit[0]["readout_reliable"]
        # Use the full available colour range.  A fixed midpoint (as in the
        # previous ``coolwarm`` plot) made positive-only G_L values look nearly
        # indistinguishable.  Keep zero as the midpoint when both signs are
        # present; otherwise use a high-contrast sequential map.
        finite=matrix[np.isfinite(matrix)]
        lo,hi=float(np.nanpercentile(finite,2)),float(np.nanpercentile(finite,98))
        if lo==hi: lo,hi=float(np.nanmin(finite)),float(np.nanmax(finite))
        if lo==hi: lo,hi=lo-1.0,hi+1.0
        if lo<0<hi:
            bound=max(abs(lo),abs(hi)); norm=TwoSlopeNorm(vmin=-bound,vcenter=0.0,vmax=bound); cmap="RdBu_r"
        else:
            norm=Normalize(vmin=lo,vmax=hi); cmap="magma" if hi>=0 else "viridis_r"
        fig,ax=plt.subplots(figsize=(11,8));m=ax.imshow(matrix,aspect="auto",cmap=cmap,norm=norm,interpolation="nearest");ax.set_xticks(range(len(LAYERS)),LAYERS);ax.set_yticks(range(len(labels)),labels,fontsize=6);ax.set_title(title+" — answer-equal macro")
        ax.set_xlabel("Layer (zero-based)"); ax.set_ylabel("Position / direction")
        ax.set_xticks(np.arange(-.5,len(LAYERS),1),minor=True); ax.set_yticks(np.arange(-.5,len(labels),1),minor=True)
        ax.grid(which="minor",color="white",linewidth=.35,alpha=.55); ax.tick_params(which="minor",bottom=False,left=False)
        for i,j in np.argwhere(~reliable):ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fill=False,hatch="///",edgecolor="gray",lw=0))
        fig.colorbar(m,ax=ax,label="symmetric derivative");fig.tight_layout();fig.savefig(root/f"figures/{filename}",dpi=180);plt.close(fig)


def analyze(root:Path,*,formal:bool=True)->dict[str,Any]:
    root=require_output_root(root)
    cells=load_jsonl(root/"artifacts/trials/trajectory_cells.jsonl");trials=load_jsonl(root/"artifacts/trials/forward_trials.jsonl")
    if formal and not cells:raise ValueError("Formal trajectory cells missing")
    families=sorted({r["family_id"] for r in trials});draws=cluster_bootstrap_draws(families,BOOTSTRAP_REPEATS,SEED);atomic_json(root/"artifacts/diagnostics/formal_bootstrap_draws.json",{"seed":SEED,"repeats":BOOTSTRAP_REPEATS,"draws":draws})
    summary=[]
    grouped=defaultdict(list)
    for row in cells:grouped[(row["direction"],row["position"],int(row["layer"]),row["target"])].append({**row,"value":row["directional_derivative"]})
    for (direction,position,layer,target),rows in sorted(grouped.items()):
        for group in GROUPS:
            stats=summarize(rows,group,draws);summary.append({"group":group,"direction":direction,"position":position,"layer":layer,"target":target,**stats,"readout_reliable":all(r["readout_reliable"] for r in rows)})
    atomic_csv(root/"tables/trajectory_readouts.csv",summary)
    transport=[{"case_id":r["case_id"],"family_id":r["family_id"],"fixed_answer":r["fixed_answer"],"direction":r["direction"],"position":r["position"],"layer":r["layer"],"target":r["target"],"delta_h_norm":r["delta_h_norm"],"relative_clean_norm":r["relative_clean_norm"],"gradient_cosine":r["gradient_cosine"],"readout_reliable":r["readout_reliable"]} for r in cells]
    atomic_csv(root/"tables/hidden_transport.csv",transport)
    onset=[]
    for direction in DIRECTIONS:
        for position in POSITIONS:
            for target in TARGETS:
                selected=[r for r in summary if r["group"]=="answer_equal_macro" and r["direction"]==direction and r["position"]==position and r["target"]==target]
                layer=persistent_onset(selected);onset.append({"direction":direction,"position":position,"target":target,"persistent_onset_layer":layer,"interpretation":"未检测到稳定 onset" if layer is None else f"L{layer}"})
    atomic_csv(root/"tables/onset_summary.csv",onset)
    # Direct endpoints and their symmetric final-soft-SA derivative.
    by={(r["case_id"],r["direction"],float(r["alpha"])):r for r in trials};direct=[]
    for case in sorted({r["case_id"] for r in trials}):
        for direction in DIRECTIONS:
            m,p=by[(case,direction,-.5)],by[(case,direction,.5)]
            direct.append({"case_id":case,"family_id":m["family_id"],"fixed_answer":m["fixed_answer"],"direction":direction,"minus_logits":m["class_logits"],"plus_logits":p["class_logits"],"minus_probabilities":m["class_probabilities"],"plus_probabilities":p["class_probabilities"],"minus_soft_sa":m["final_soft_sa"],"plus_soft_sa":p["final_soft_sa"],"symmetric_soft_sa_derivative":p["final_soft_sa"]-m["final_soft_sa"],"minus_hard_class":m["hard_class"],"plus_hard_class":p["hard_class"],"minus_hard_change":m["hard_change"],"plus_hard_change":p["hard_change"],"minus_margin":m["baseline_class_margin"],"plus_margin":p["baseline_class_margin"]})
    atomic_csv(root/"tables/direct_final_endpoints.csv",direct)
    additions=[]
    sources=defaultdict(dict)
    for r in cells:sources[(r["case_id"],r["position"],r["layer"],r["target"])][r["direction"]]=r["directional_derivative"]
    for key,v in sources.items():
        if set(v)==set(DIRECTIONS):
            source=next(r for r in cells if r["case_id"]==key[0] and r["position"]==key[1] and r["layer"]==key[2] and r["target"]==key[3])
            additions.append({"case_id":key[0],"family_id":source["family_id"],"fixed_answer":source["fixed_answer"],"position":key[1],"layer":key[2],"target":key[3],"additivity_residual":component_additivity(v[DIRECTIONS[0]],v[DIRECTIONS[1]],v[DIRECTIONS[2]])})
    direct_sources=defaultdict(dict)
    for r in direct:direct_sources[r["case_id"]][r["direction"]]=r
    for case,v in direct_sources.items():
        additions.append({"case_id":case,"family_id":v[DIRECTIONS[0]]["family_id"],"fixed_answer":v[DIRECTIONS[0]]["fixed_answer"],"position":"P1_SAC","layer":"direct","target":"direct_final_soft_sa","additivity_residual":component_additivity(*[v[d]["symmetric_soft_sa_derivative"] for d in DIRECTIONS])})
    addition_summary=[]
    addition_groups=defaultdict(list)
    for r in additions:addition_groups[(r["position"],r["layer"],r["target"])].append({**r,"value":r["additivity_residual"]})
    for (position,layer,target),rows in addition_groups.items():
        for group in GROUPS:addition_summary.append({"group":group,"position":position,"layer":layer,"target":target,**summarize(rows,group,draws),"interpretation":"low-dose approximation"})
    atomic_csv(root/"tables/component_additivity.csv",addition_summary)
    if formal:_plot(root,summary)
    atomic_text(root/"README_RESULTS_zh.md","# LAT→PANL/SAC 逐层 trajectory 结果\n\n主统计为 answer-equal macro：先按 fixed-answer 颜色求均值，再对颜色等权平均。误差区间为 seed 42、2000 次 50-family cluster bootstrap。\n\n- `probe_metrics.csv`：四位置×十三层×四目标的 audit 指标；`readout_reliable` 要求 R²、Pearson 和 Pearson CI 下界均大于零。\n- `trajectory_readouts.csv`：横轴 layer，符号表示对称方向导数；CI 跨零不能解释为稳定效应。\n- `hidden_transport.csv`：每个 probe target 的 hidden 运输量、相对范数和 gradient cosine。\n- `onset_summary.csv`：连续两层同号、CI 不跨零且 readout 可靠才报告 persistent onset。\n- `component_additivity.csv`：raw−parallel−SA-subspace-orthogonal confidence-related component；仅是低剂量近似。\n- `direct_final_endpoints.csv`：九类 logits/probability、soft/hard SA、margin 与 hard-change。\n\n图均使用主统计；空心/淡化或斜线表示 readout 不可靠。原始入口为 `artifacts/trials/forward_trials.jsonl` 和 `trajectory_cells.jsonl`。本实验只支持局部因果传播描述，不支持把 probe readout 当作机制等价或把低剂量可加性外推到其他剂量。\n")
    result={"status":"complete","trajectory_cell_count":len(cells),"trial_count":len(trials),"formal":formal};atomic_json(root/"progress/analyze.json",result);return result
