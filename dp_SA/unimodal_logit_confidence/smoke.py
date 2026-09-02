from __future__ import annotations

import argparse
import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .build_split import build_split, record_sort_key
from .capture import build_reuse_manifest, capture
from .config import MAX_SMOKE_ROUNDS, RESULTS_ROOT, SMOKE_ROOT
from .io_utils import atomic_json, atomic_jsonl, ensure_layout, load_jsonl, stable_shard
from .score_unimodal import score_unimodal, spec_key, unique_specs


def smoke_records(rows: Sequence[dict[str,Any]]) -> list[dict[str,Any]]:
    by_item={}
    for row in rows: by_item.setdefault(str(row["item_id"]),[]).append(row)
    selected=[]
    for item in sorted(by_item,key=lambda x:int(x) if x.isdigit() else x):
        values=sorted(by_item[item],key=record_sort_key); prior=min(int(r["prior_index"]) for r in values); pair=[r for r in values if int(r["prior_index"])==prior]
        if {r["condition"] for r in pair}=={"conflict_easy","conflict_hard"}: selected.extend(pair)
        if len({r["item_id"] for r in selected})>=3:
            specs=unique_specs(selected)
            if {stable_shard(spec_key(s),2) for s in specs}=={0,1} and {stable_shard(("hidden",r["case_id"]),2) for r in selected}=={0,1}: break
    if len(selected)<4: raise ValueError("Could not form smoke cohort")
    return sorted(selected,key=record_sort_key)


def _prepare(root: Path) -> list[dict[str,Any]]:
    ensure_layout(root,resume=False); build_split(root); selected=smoke_records(load_jsonl(root/"shared/manifests/probe_manifest.jsonl")); atomic_jsonl(root/"shared/manifests/probe_manifest.jsonl",selected); build_reuse_manifest(root,selected); return selected


def _hidden_payload(root: Path) -> dict[tuple[str,str],np.ndarray]:
    output={}
    for row in load_jsonl(root/"confidence_probe/artifacts/hidden/capture_results.jsonl"):
        with np.load(root/row["delta_file"]) as payload:
            for key in row["delta_keys"]: output[row["case_id"],key]=np.asarray(payload[key])
    return output


def run_smoke(base: Path = SMOKE_ROOT) -> dict[str,Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count()<2: raise RuntimeError("GPU smoke requires two visible GPUs")
    base=base.resolve()
    if base == RESULTS_ROOT.resolve() or "smoke" not in base.name: raise ValueError("Smoke root is not safely isolated")
    reports=[]
    for round_id in range(1,MAX_SMOKE_ROUNDS+1):
        round_root=base/f"round_{round_id}"
        if round_root.exists(): shutil.rmtree(round_root)
        single=round_root/"single_gpu"; dual=round_root/"dual_gpu"
        try:
            cohort1=_prepare(single); cohort2=_prepare(dual)
            if [r["case_id"] for r in cohort1] != [r["case_id"] for r in cohort2]: raise AssertionError("Smoke cohorts differ")
            score1=score_unimodal(single,num_gpus=1,resume=False); hidden1=capture(single,num_gpus=1,resume=False)
            score2=score_unimodal(dual,num_gpus=2,resume=False); hidden2=capture(dual,num_gpus=2,resume=False)
            raw1=load_jsonl(single/"unimodal_confidence/artifacts/raw_scores/unimodal_scores.jsonl"); raw2=load_jsonl(dual/"unimodal_confidence/artifacts/raw_scores/unimodal_scores.jsonl")
            if raw1 != raw2: raise AssertionError("Single/dual score outputs differ")
            h1=_hidden_payload(single); h2=_hidden_payload(dual)
            if set(h1)!=set(h2) or any(not np.array_equal(h1[key],h2[key]) for key in h1): raise AssertionError("Single/dual hidden outputs differ")
            resume_score1=score_unimodal(single,num_gpus=1,resume=True); resume_hidden1=capture(single,num_gpus=1,resume=True); resume_score2=score_unimodal(dual,num_gpus=2,resume=True); resume_hidden2=capture(dual,num_gpus=2,resume=True)
            report={"round":round_id,"status":"passed","cohort_count":len(cohort1),"single":{"score":score1,"hidden":hidden1,"resume_score":resume_score1,"resume_hidden":resume_hidden1},"dual":{"score":score2,"hidden":hidden2,"resume_score":resume_score2,"resume_hidden":resume_hidden2},"numeric_parity":"exact","resume_noop":True}; reports.append(report); atomic_json(base/"smoke_report.json",{"status":"passed","rounds":reports}); return report
        except Exception as exc:
            reports.append({"round":round_id,"status":"failed","error":{"type":type(exc).__name__,"message":str(exc)},"traceback":traceback.format_exc()}); atomic_json(base/"smoke_report.json",{"status":"running" if round_id<MAX_SMOKE_ROUNDS else "failed","rounds":reports})
    raise RuntimeError(f"GPU smoke failed after {MAX_SMOKE_ROUNDS} rounds")


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--smoke-root",default=str(SMOKE_ROOT)); args=parser.parse_args(argv); print(json.dumps(run_smoke(Path(args.smoke_root)),ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
