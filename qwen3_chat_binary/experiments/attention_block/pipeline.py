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
    parser = argparse.ArgumentParser(description="Prepare/run/analyze five-class attention blocking")
    parser.add_argument("--stage", choices=("prepare", "run", "analyze", "all"), default="all")
    parser.add_argument("--output-root", type=Path); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steered-block", action="store_true")
    args = parser.parse_args(argv)
    if args.steered_block:
        from .steered_block import default_output as enhanced_output, prepare as selected_prepare, run as selected_run
        from .steered_block_analysis import analyze as selected_analyze
        root = args.output_root or enhanced_output(args.smoke)
    else:
        selected_prepare, selected_run, selected_analyze = prepare, run, analyze
        root = args.output_root or default_output(args.smoke)
    result = {}
    if args.stage in ("prepare", "all"): result["prepare"] = selected_prepare(output_root=root, smoke=args.smoke, resume=args.resume)
    if args.stage in ("run", "all"): result["run"] = selected_run(output_root=root, resume=args.resume)
    if args.stage in ("analyze", "all"): result["analyze"] = selected_analyze(output_root=root)
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
