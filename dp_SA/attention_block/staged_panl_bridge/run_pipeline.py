from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from dp_SA.io_utils import atomic_json

from .analyze import analyze
from .run import DEFAULT_BASE,DEFAULT_OUTPUT_PARENT,run


def _test_gate(output:Path)->None:
    root=Path(__file__).resolve().parents[3]
    command=[sys.executable,"-m","pytest","-q","dp_SA/attention_block/staged_panl_bridge/tests","dp_SA/attention_block/tests/test_masking.py"]
    result=subprocess.run(command,cwd=root,capture_output=True,text=True);(output/"test_gate.log").write_text(result.stdout+result.stderr,encoding="utf-8")
    atomic_json(output/"test_gate.json",{"status":"passed" if result.returncode==0 else "failed","returncode":result.returncode,"command":command})
    if result.returncode:raise RuntimeError(f"CPU test gate failed; see {output/'test_gate.log'}")


def pipeline(output:Path,base:Path=DEFAULT_BASE,*,smoke_only:bool=False,resume:bool=False)->dict:
    output=output.resolve();output.mkdir(parents=True,exist_ok=True);state={"status":"testing","started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())};atomic_json(output/"pipeline_state.json",state);_test_gate(output)
    smoke=output/"preflight_smoke";state["status"]="smoke";atomic_json(output/"pipeline_state.json",state)
    if not (smoke/"analysis_completion.json").exists():run(smoke,base,smoke=True,resume=resume or (smoke/"run_config.json").exists());analyze(smoke)
    marker=json.loads((smoke/"analysis_completion.json").read_text())
    if not marker.get("technical_validation_passed"):raise RuntimeError("GPU smoke failed")
    smoke_analysis=json.loads((smoke/"analysis.json").read_text())
    if not smoke_analysis["technical_validation"].get("c10_no_sac_panl_leakage"):raise RuntimeError("C10 leakage smoke gate failed")
    if smoke_only:
        state.update(status="complete",mode="smoke_only",finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()));atomic_json(output/"pipeline_state.json",state);return state
    state["status"]="formal";atomic_json(output/"pipeline_state.json",state);run(output,base,smoke=False,resume=resume or (output/"run_config.json").exists());result=analyze(output)
    state.update(status="complete",mode="formal",staged_direct_panl_bridge_supported=result["staged_direct_panl_bridge_supported"],finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()));atomic_json(output/"pipeline_state.json",state);return state


def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path);p.add_argument("--base-output",type=Path,default=DEFAULT_BASE);p.add_argument("--smoke",action="store_true");p.add_argument("--resume",action="store_true");a=p.parse_args(argv)
    output=a.output_dir or DEFAULT_OUTPUT_PARENT/time.strftime(("pipeline_smoke" if a.smoke else "formal")+"_seed42_%Y%m%dT%H%M%SZ",time.gmtime());pipeline(output,a.base_output.resolve(),smoke_only=a.smoke,resume=a.resume);return 0


if __name__=="__main__":raise SystemExit(main())
