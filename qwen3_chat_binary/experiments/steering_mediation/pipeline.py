from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .analyze import analyze
from .config import default_output
from .prepare import prepare
from .run import run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare/run/analyze five-class steering mediation")
    parser.add_argument("--stage", choices=("prepare", "run", "analyze", "all"), default="all")
    parser.add_argument("--output-root", type=Path); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true"); args = parser.parse_args(argv)
    root = args.output_root or default_output(args.smoke); result = {}
    if args.stage in ("prepare", "all"): result["prepare"] = prepare(output_root=root, smoke=args.smoke, resume=args.resume)
    if args.stage in ("run", "all"): result["run"] = run(output_root=root, resume=args.resume)
    if args.stage in ("analyze", "all"): result["analyze"] = analyze(output_root=root)
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
