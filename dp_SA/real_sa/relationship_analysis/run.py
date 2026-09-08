from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .analysis import BOOTSTRAP_REPEATS, DEFAULT_INPUT, DEFAULT_OUTPUT, SEED, run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only bidirectional Real SA/verbal SA relationship analysis")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_repeats <= 0:
        raise ValueError("bootstrap-repeats must be positive")
    result = run_analysis(args.input, args.output_root, args.bootstrap_repeats, args.seed)
    payload = asdict(result)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
