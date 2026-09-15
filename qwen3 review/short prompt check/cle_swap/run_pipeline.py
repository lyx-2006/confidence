from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import analyze
from .config import OUTPUT_ROOT, SMOKE_ROOT
from .prepare import prepare
from .run import run


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("prepare","smoke","run","analyze","all"),default="all")
    p.add_argument("--resume",action="store_true");p.add_argument("--run-formal",action="store_true");a=p.parse_args(argv)
    results={}
    if a.stage in ("prepare","all"):
        results["prepare"]=prepare(OUTPUT_ROOT,resume=a.resume)
    if a.stage in ("smoke","all"):
        prepare(SMOKE_ROOT,resume=a.resume);results["smoke"]=run(SMOKE_ROOT,resume=a.resume,smoke=True)
    if a.stage in ("run","all"):
        if not a.run_formal: raise ValueError("Formal swap requires --run-formal")
        results["run"]=run(OUTPUT_ROOT,resume=True,smoke=False)
    if a.stage in ("analyze","all"):
        results["analyze"]=analyze(OUTPUT_ROOT)
    print(json.dumps(results,ensure_ascii=False));return 0
if __name__=="__main__":raise SystemExit(main())

