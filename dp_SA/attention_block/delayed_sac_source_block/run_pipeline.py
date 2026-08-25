from __future__ import annotations

import argparse,json,subprocess,sys,time
from pathlib import Path
from typing import Sequence
from dp_SA.io_utils import atomic_json
from .analyze import analyze
from .run import DEFAULT_BASE,DEFAULT_OUTPUT_PARENT,run


def _gate(output:Path)->None:
    command=[sys.executable,"-m","pytest","-q","dp_SA/attention_block/delayed_sac_source_block/tests","dp_SA/attention_block/tests/test_masking.py","dp_SA/attention_block/tests/test_design.py"]
    result=subprocess.run(command,cwd=Path(__file__).resolve().parents[3],capture_output=True,text=True);(output/"test_gate.log").write_text(result.stdout+result.stderr,encoding="utf-8");atomic_json(output/"test_gate.json",{"status":"passed" if result.returncode==0 else "failed","returncode":result.returncode,"command":command})
    if result.returncode:raise RuntimeError(f"CPU gate failed: {output/'test_gate.log'}")


def pipeline(output:Path,base:Path=DEFAULT_BASE,*,smoke_only:bool=False,resume:bool=False):
    output=output.resolve();output.mkdir(parents=True,exist_ok=True);state={"status":"testing","started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())};atomic_json(output/"pipeline_state.json",state);_gate(output)
    smoke=output/"preflight_smoke";state["status"]="smoke";atomic_json(output/"pipeline_state.json",state)
    if not (smoke/"analysis_completion.json").exists():run(smoke,base,smoke=True,resume=resume or (smoke/"run_config.json").exists());analyze(smoke)
    if not json.loads((smoke/"analysis_completion.json").read_text())["technical_validation_passed"]:raise RuntimeError("Smoke failed")
    if smoke_only:state.update(status="complete",mode="smoke_only",finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()));atomic_json(output/"pipeline_state.json",state);return state
    state["status"]="formal";atomic_json(output/"pipeline_state.json",state);run(output,base,smoke=False,resume=resume or (output/"run_config.json").exists());analyze(output);state.update(status="complete",mode="formal",finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()));atomic_json(output/"pipeline_state.json",state);return state


def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path);p.add_argument("--base-output",type=Path,default=DEFAULT_BASE);p.add_argument("--smoke",action="store_true");p.add_argument("--resume",action="store_true");a=p.parse_args(argv);output=a.output_dir or DEFAULT_OUTPUT_PARENT/time.strftime(("pipeline_smoke" if a.smoke else "formal")+"_seed42_%Y%m%dT%H%M%SZ",time.gmtime());pipeline(output,a.base_output.resolve(),smoke_only=a.smoke,resume=a.resume);return 0


if __name__=="__main__":raise SystemExit(main())
