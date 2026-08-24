from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args()
    for name in ("pipeline_state.json","progress.json","refine_selection.json","technical_diagnostics.json","completion.json"):
        path=args.output_dir/name
        if path.exists():
            print(f"## {name}"); print(json.dumps(json.loads(path.read_text()),ensure_ascii=False,indent=2)[:20000])
    progress=args.output_dir/"progress.json"
    if progress.exists():
        value=json.loads(progress.read_text()); remaining=float(value.get("estimated_remaining_seconds",0))
        print(f"ETA remaining: {timedelta(seconds=round(remaining))}")
        print(f"Estimated completion UTC: {(datetime.now(timezone.utc)+timedelta(seconds=remaining)).isoformat()}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
