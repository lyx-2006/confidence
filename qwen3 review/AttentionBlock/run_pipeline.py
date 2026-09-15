from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    review = Path(__file__).resolve().parents[1]
    for path in (review.parent, review):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

from dp_SA.io_utils import atomic_json

from AttentionBlock.analyze import analyze
from AttentionBlock.config import EXPERIMENTS, default_output
from AttentionBlock.prepare import prepare
from AttentionBlock.run import run


def pipeline(*, experiment: str, output_root: Path | None = None,
             smoke: bool = False, resume: bool = False) -> dict:
    root = Path(output_root or default_output(experiment, smoke)).resolve()
    started = time.time()
    prepared = prepare(experiment=experiment, output_root=root, smoke=smoke, resume=resume)
    executed = run(experiment=experiment, output_root=root, resume=resume)
    analyzed = analyze(experiment=experiment, output_root=root)
    if smoke:
        repeated = run(experiment=experiment, output_root=root, resume=True)
        if repeated["new_gpu_forwards"] != 0:
            raise RuntimeError("Smoke resume repeated completed forward passes")
    result = {"status": "complete", "experiment": experiment, "smoke": smoke,
              "output_root": str(root), "prepare": prepared, "run": executed,
              "analysis": analyzed, "elapsed_seconds": time.time() - started}
    atomic_json(root / "completion.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3-VL attention-edge blocking experiment")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(pipeline(experiment=args.experiment, output_root=args.output_root,
                              smoke=args.smoke, resume=args.resume), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

