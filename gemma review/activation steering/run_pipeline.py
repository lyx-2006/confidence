from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Sequence

from analysis import run_analysis
from capture import run_capture
from experiment_config import (
    BOOTSTRAP_REPEATS,
    CAPTURE_LAYERS,
    RESULTS_ROOT,
    SMOKE_LAYER,
    SMOKE_MAX_SAMPLES,
    SMOKE_ROOT,
)
from io_utils import atomic_json
from steering import parse_steering_layers, run_steering


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gemma delayed-SA activation-steering pipeline")
    parser.add_argument("--output-root")
    parser.add_argument("--steering-layers", nargs="+", type=int)
    parser.add_argument("--shuffled-layers", nargs="*", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPEATS)
    args = parser.parse_args(argv)
    # Validate the formal layer selection before loading the model or writing outputs.
    parse_steering_layers(args.steering_layers, args.smoke)
    root = Path(args.output_root) if args.output_root else (SMOKE_ROOT if args.smoke else RESULTS_ROOT)
    state_path = root / "pipeline_state.json"
    capture_layers = (SMOKE_LAYER,) if args.smoke else CAPTURE_LAYERS
    max_samples = SMOKE_MAX_SAMPLES if args.smoke else args.max_samples
    try:
        atomic_json(state_path, {"status": "running", "stage": "capture", "smoke": args.smoke})
        capture_summary = run_capture(
            output_root=root,
            max_items=args.max_items,
            max_samples=max_samples,
            resume=args.resume,
            layers=capture_layers,
        )
        atomic_json(state_path, {"status": "running", "stage": "steering", "smoke": args.smoke})
        steering_summary = run_steering(
            output_root=root,
            steering_layers=args.steering_layers,
            shuffled_layers=args.shuffled_layers,
            smoke=args.smoke,
            resume=args.resume,
        )
        if args.smoke:
            final = {
                "status": "complete",
                "stage": "smoke_complete",
                "capture": capture_summary,
                "steering": steering_summary,
            }
        else:
            atomic_json(state_path, {"status": "running", "stage": "analysis", "smoke": False})
            analysis_summary = run_analysis(root, args.bootstrap)
            final = {
                "status": "complete",
                "stage": "complete",
                "capture": capture_summary,
                "steering": steering_summary,
                "analysis": analysis_summary,
            }
        atomic_json(state_path, final)
        (root / "COMPLETED").write_text("complete\n")
        return 0
    except Exception as exc:
        atomic_json(
            state_path,
            {
                "status": "failed",
                "stage": "failed",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

