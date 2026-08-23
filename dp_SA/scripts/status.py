from __future__ import annotations
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]/"outputs"
for relative in ("pipeline_state.json","capture/progress.json","capture/summary.json","steering/progress.json","steering/summary.json","steering/candidate_metrics.json","probe/metrics.json"):
    path=root/relative
    if path.exists():
        print(f"## {relative}")
        print(json.dumps(json.loads(path.read_text()),ensure_ascii=False,indent=2)[:12000])
