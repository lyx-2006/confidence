from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .analysis import BOOTSTRAP_REPEATS, OUTPUT_ROOT, SEED, run_analysis


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU-only held-out confidence model comparison")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.bootstrap_repeats <= 0:
        raise ValueError("bootstrap-repeats must be positive")
    result = run_analysis(args.output_root, bootstrap_repeats=args.bootstrap_repeats, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
