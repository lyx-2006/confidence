from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .analyze import analyze
from .config import MAX_SMOKE_ROUNDS, RESULTS_ROOT, SMOKE_PARENT
from .io_utils import atomic_json
from .prepare import run_prepare
from .run import run_steering


TEST_PATHS = (
    "dp_SA/confidence_steering/tests",
    "dp_SA/answer_matched_lat_steering/tests/test_position_hook.py",
    "dp_SA/answer_matched_lat_steering/tests/test_vectors.py",
    "dp_SA/unimodal_logit_confidence/tests/test_explanatory_comparison.py",
)


def run_cpu_tests() -> dict[str, Any]:
    started=time.time(); completed=subprocess.run([sys.executable,"-m","pytest","-q",*TEST_PATHS],cwd=Path(__file__).resolve().parents[2],check=False)
    if completed.returncode: raise RuntimeError(f"CPU tests failed with exit code {completed.returncode}")
    return {"status":"passed","exit_code":0,"elapsed_seconds":time.time()-started,"paths":list(TEST_PATHS)}


def _next_round() -> tuple[int, Path]:
    SMOKE_PARENT.mkdir(parents=True,exist_ok=True)
    for number in range(1,MAX_SMOKE_ROUNDS+1):
        path=SMOKE_PARENT/f"round_{number:02d}"
        if not path.exists(): return number,path
    raise RuntimeError(f"All {MAX_SMOKE_ROUNDS} smoke rounds have already been used")


def run_pipeline(*, output_root: Path | None = None, smoke: bool = False, resume: bool = False, num_gpus: int = 1) -> dict[str, Any]:
    if not smoke:
        root=Path(output_root or RESULTS_ROOT); prepared=run_prepare(output_root=root,smoke=False,resume=resume)
        ran=run_steering(output_root=root,smoke=False,resume=resume,num_gpus=num_gpus); analyzed=analyze(output_root=root,smoke=False,resume=resume)
        return {"status":"complete","smoke_only":False,"output_root":str(root.resolve()),"prepare":prepared,"run":ran,"analyze":analyzed}
    tests=run_cpu_tests(); failures=[]
    for _ in range(MAX_SMOKE_ROUNDS):
        if output_root is None: number,root=_next_round()
        else: number,root=1,Path(output_root)
        try:
            prepared=run_prepare(output_root=root,smoke=True,resume=resume)
            ran=run_steering(output_root=root,smoke=True,resume=resume,num_gpus=num_gpus)
            analyzed=analyze(output_root=root,smoke=True,resume=resume)
            resumed=run_steering(output_root=root,smoke=True,resume=True,num_gpus=num_gpus)
            if not resumed.get("resumed_noop") or int(resumed.get("new_gpu_forwards",-1)) != 0: raise RuntimeError("Smoke resume was not a zero-forward no-op")
            report={"status":"passed","smoke_only":True,"round":number,"output_root":str(root.resolve()),"cpu_tests":tests,"prepare":prepared,"run":ran,"analyze":analyzed,"resume":resumed,"failures":failures}
            atomic_json(root/"progress/smoke_report.json",report); return report
        except Exception as exc:
            failures.append({"round":number,"output_root":str(root.resolve()),"type":type(exc).__name__,"message":str(exc)})
            if output_root is not None: break
    raise RuntimeError(f"Smoke failed after {len(failures)} round(s): {failures}")


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root"); parser.add_argument("--resume",action="store_true"); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--num-gpus",type=int,choices=(1,2),default=1)
    args=parser.parse_args(argv); result=run_pipeline(output_root=Path(args.output_root) if args.output_root else None,smoke=args.smoke,resume=args.resume,num_gpus=args.num_gpus)
    print(json.dumps(result,ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
