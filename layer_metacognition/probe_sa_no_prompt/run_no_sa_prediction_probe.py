"""Run no-SA OOF training followed by conflict-only analysis."""

from __future__ import annotations

import json

from .analyze_no_sa_probe_results import run_analysis
from .train_no_sa_probes import build_parser, run_training


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    training = run_training(args)
    analysis = run_analysis(args.output_dir)
    print(json.dumps({"status": "complete", "training": training, "analysis": analysis}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
