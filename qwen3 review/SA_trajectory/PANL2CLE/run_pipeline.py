from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    review = Path(__file__).resolve().parents[2]
    for path in (review, review.parent):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    __package__ = "SA_trajectory.PANL2CLE"

from .analyze import analyze
from .config import CAPTURE_ROOT, MODEL_PATH, OUTPUT_ROOT, SMOKE_ROOT, STEERING_ROOT
from .prepare import prepare
from .probes import train_probes
from .run import alpha_zero_gate, run


def pipeline(*, stage: str, output_root: Path, capture_root: Path, steering_root: Path, model_path: Path, smoke: bool, resume: bool, num_gpus: int, run_formal: bool) -> dict:
    root = Path(output_root).resolve(); result = {"stage": stage, "output_root": str(root), "smoke": smoke}
    if stage in ("prepare", "all"): result["prepare"] = prepare(output_root=root, capture_root=capture_root, steering_root=steering_root, model_path=model_path, smoke=smoke, resume=resume)
    if stage in ("train-probes", "all"): result["train_probes"] = train_probes(root, Path(capture_root).resolve(), resume=resume, repeats=200 if smoke else 2000)
    if stage in ("alpha-zero", "all"): result["alpha_zero"] = alpha_zero_gate(root, model_path=model_path, resume=resume)
    if stage in ("run", "all"):
        if not smoke and not run_formal: raise RuntimeError("Formal GPU execution requires --run-formal")
        result["run"] = run(output_root=root, model_path=model_path, resume=resume, num_gpus=num_gpus, smoke=smoke)
    if stage in ("analyze", "all"): result["analysis"] = analyze(output_root=root, repeats=200 if smoke else 2000)
    result["status"] = "complete"; return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3 PANL-to-CLE four-cell mediation")
    parser.add_argument("--stage", choices=("prepare", "train-probes", "alpha-zero", "run", "analyze", "all"), default="prepare"); parser.add_argument("--output-root", type=Path); parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT); parser.add_argument("--steering-root", type=Path, default=STEERING_ROOT); parser.add_argument("--model-path", type=Path, default=MODEL_PATH); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--num-gpus", type=int, choices=(1, 2), default=1); parser.add_argument("--run-formal", action="store_true"); args = parser.parse_args(argv)
    root = args.output_root or (SMOKE_ROOT if args.smoke else OUTPUT_ROOT); print(json.dumps(pipeline(stage=args.stage, output_root=root, capture_root=args.capture_root, steering_root=args.steering_root, model_path=args.model_path, smoke=args.smoke, resume=args.resume, num_gpus=args.num_gpus, run_formal=args.run_formal), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
