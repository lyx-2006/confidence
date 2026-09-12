from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any,Sequence

import torch

from .behavior import analyze_behavior
from .capture import run_capture
from .config import DEFAULT_ALPHAS, DEFAULT_STEERING_LAYERS, OUTPUT_ROOT, RESULTS_ROOT, SMOKE_ROOT, TEMPLATE_NAMES, validation_root
from .geometry import analyze_geometry
from .io_utils import atomic_json, inventory, verify_inventory
from .plots import make_plots
from .preflight import run_preflight
from .probe_transfer import analyze_probe_transfer
from .sources import source_inventory, verify_frozen_sources
from .steering_transfer import analyze_steering, run_steering
from .validation import validate_outputs

STAGES=("behavior","probe","geometry","steering","analyze")


def run_cpu_tests()->dict[str,Any]:
    env=dict(os.environ);env.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
    completed=subprocess.run([sys.executable,"-m","pytest","-q","dp_SA/prompt_check/tests"],cwd=Path(__file__).resolve().parents[2],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if completed.returncode:raise RuntimeError("prompt_check CPU tests failed\n"+completed.stdout)
    match=re.search(r"(\d+) passed",completed.stdout);return {"status":"passed","passed":int(match.group(1)) if match else None,"output":completed.stdout}


def _validate_args(templates:Sequence[str],num_gpus:int,layers:Sequence[int],alphas:Sequence[float])->None:
    if not templates or len(set(templates))!=len(templates):raise ValueError("Templates must be non-empty and unique")
    if set(templates)-set(TEMPLATE_NAMES):raise ValueError("Only T1, T2 and T3 are selectable; T0 is implicit")
    if num_gpus not in (1,2):raise ValueError("num_gpus must be 1 or 2")
    if torch.cuda.device_count()<num_gpus:raise RuntimeError(f"Requested {num_gpus} GPUs, found {torch.cuda.device_count()}")
    if any(layer<9 or layer>15 for layer in layers):raise ValueError("Steering layers must be within L9–L15")
    if set(map(float,alphas))!={-2.,0.,2.}:raise ValueError("Alphas must be exactly -2 0 +2")


def run_pipeline(*,templates:Sequence[str],num_gpus:int,resume:bool,smoke:bool,stages:Sequence[str],steering_layers:Sequence[int],alphas:Sequence[float],validation_cases:int|None=None,output_name:str|None=None)->dict[str,Any]:
    _validate_args(templates,num_gpus,steering_layers,alphas);selected=set(STAGES if not stages or "all" in stages else stages);unknown=selected-set(STAGES)
    if unknown:raise ValueError(f"Unknown stages: {sorted(unknown)}")
    effective_layers=tuple(dict.fromkeys((steering_layers[0],steering_layers[-1]))) if smoke and len(steering_layers)>2 else tuple(steering_layers)
    if smoke and validation_cases:raise ValueError("--smoke and --validation-cases are mutually exclusive")
    if validation_cases is not None and not 20<=validation_cases<=174:raise ValueError("--validation-cases must be between 20 and 174")
    if output_name is not None and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",output_name) or output_name in {".",".."}):
        raise ValueError("--output-name must be a safe directory name under prompt_check/output")
    root=OUTPUT_ROOT/output_name if output_name else (SMOKE_ROOT if smoke else (validation_root(validation_cases) if validation_cases else RESULTS_ROOT))
    if root.exists() and any(root.iterdir()) and not resume:raise FileExistsError(f"Output exists; use --resume: {root}")
    root.mkdir(parents=True,exist_ok=True);before=source_inventory();atomic_json(root/"artifacts/diagnostics/source_hashes_before.json",before)
    tests=run_cpu_tests();preflight=run_preflight(root,smoke_case_count=2 if smoke else 4,resume=resume)
    results:dict[str,Any]={"tests":tests,"preflight":preflight};needs_capture=bool(selected&{"behavior","probe","geometry","steering"})
    if needs_capture:
        results["capture"]=run_capture(root,num_gpus=num_gpus,templates=templates,resume=resume,smoke=smoke,validation_cases=validation_cases,include_audit=bool(selected&{"behavior","probe"}),include_geometry=bool(selected&{"geometry","steering"}))
    if "behavior" in selected:results["behavior"]=analyze_behavior(root,templates=templates,smoke=smoke,validation_cases=validation_cases)
    if "probe" in selected:results["probe"]=analyze_probe_transfer(root,templates=templates,smoke=smoke,validation_cases=validation_cases)
    if "geometry" in selected:results["geometry"]=analyze_geometry(root,templates=templates,smoke=smoke,validation_cases=validation_cases)
    if "steering" in selected:
        results["steering_run"]=run_steering(root,num_gpus=num_gpus,templates=templates,layers=effective_layers,alphas=alphas,resume=resume,smoke=smoke,validation_cases=validation_cases)
        results["steering_analysis"]=analyze_steering(root,templates=templates,layers=effective_layers,smoke=smoke,validation_cases=validation_cases)
    if "analyze" in selected or selected&{"behavior","probe","geometry","steering"}:results["figures"]=make_plots(root)
    results["validation"]=validate_outputs(root,templates=templates,layers=effective_layers,smoke=smoke,stages=selected,validation_cases=validation_cases)
    verify_frozen_sources(before);atomic_json(root/"artifacts/diagnostics/source_hashes_after.json",source_inventory())
    completion={"status":"complete","smoke_only":smoke,"validation_cases":validation_cases,"output_name":output_name,"templates":list(templates),"stages":sorted(selected),"num_gpus":num_gpus,"steering_layers":list(effective_layers),"requested_steering_layers":list(steering_layers),"alphas":list(map(float,alphas)),"completed_at_unix":time.time(),"historical_sources_unchanged":True,"results":results};atomic_json(root/"completion.json",completion);return completion


def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description="Phase-1 cross-prompt construct and representation transfer")
    p.add_argument("--templates",nargs="+",choices=TEMPLATE_NAMES,default=list(TEMPLATE_NAMES));p.add_argument("--num-gpus",type=int,choices=(1,2),default=1);p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true");p.add_argument("--validation-cases",type=int);p.add_argument("--output-name");p.add_argument("--stages",nargs="+",choices=(*STAGES,"all"),default=["all"]);p.add_argument("--steering-layers",nargs="+",type=int,default=list(DEFAULT_STEERING_LAYERS));p.add_argument("--alphas",nargs="+",type=float,default=list(DEFAULT_ALPHAS));a=p.parse_args(argv)
    result=run_pipeline(templates=a.templates,num_gpus=a.num_gpus,resume=a.resume,smoke=a.smoke,stages=a.stages,steering_layers=a.steering_layers,alphas=a.alphas,validation_cases=a.validation_cases,output_name=a.output_name);print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
