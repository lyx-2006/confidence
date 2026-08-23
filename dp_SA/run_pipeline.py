from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Sequence

from .analysis import main as analysis_main
from .capture import run_capture
from .config import BOOTSTRAP_REPEATS, OUTPUT_ROOT, SPLIT_PATH
from .io_utils import atomic_json
from .probe import run_probe
from .steering import run_steering

def main(argv: Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(); p.add_argument("--output-root",default=str(OUTPUT_ROOT)); p.add_argument("--resume",action="store_true"); p.add_argument("--smoke",action="store_true"); p.add_argument("--max-items",type=int); p.add_argument("--max-samples",type=int); p.add_argument("--bootstrap",type=int,default=BOOTSTRAP_REPEATS); p.add_argument("--split",default=str(SPLIT_PATH))
    a=p.parse_args(argv); root=Path(a.output_root); state=root/"pipeline_state.json"
    try:
        atomic_json(state,{"status":"running","stage":"capture"}); run_capture(output_root=root,max_items=a.max_items,max_samples=a.max_samples,resume=a.resume)
        atomic_json(state,{"status":"running","stage":"steering"}); run_steering(output_root=root,smoke=a.smoke,resume=a.resume)
        if a.smoke:
            atomic_json(state,{"status":"complete","stage":"smoke_complete"}); return 0
        atomic_json(state,{"status":"running","stage":"candidate_analysis"}); analysis_main(["--output-root",str(root),"--bootstrap",str(a.bootstrap)])
        atomic_json(state,{"status":"running","stage":"probe"}); probe=run_probe(root,root/"steering"/"probe_candidate_manifest.jsonl",Path(a.split),a.bootstrap)
        atomic_json(state,{"status":"complete","stage":"complete","probe_status":probe["status"]}); (root/"COMPLETED").write_text("complete\n"); return 0
    except Exception as exc:
        atomic_json(state,{"status":"failed","error":{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}}); raise
if __name__=="__main__": raise SystemExit(main())
