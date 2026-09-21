from __future__ import annotations

import argparse
import gc
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import torch
import numpy as np

from .capture import run_capture
from .config import DEFAULT_STEERING_POSITIONS, MODEL_PATH, OUTPUT_ROOT, VARIANTS
from .contracts import all_hidden_keys
from .layout import capture_results_path
from .steering import run_steering
from dp_SA.io_utils import load_jsonl


def _validate_capture_smoke(capture_root: Path, summary: dict[str, object]) -> None:
    expected_keys = set(all_hidden_keys())
    if summary.get("total_cases") != 20 or summary.get("shared_completed") != 20:
        raise RuntimeError(f"Smoke capture has incomplete shared results: {summary}")
    completed = summary.get("completed")
    if not isinstance(completed, dict) or any(completed.get(variant) != 20 for variant in VARIANTS):
        raise RuntimeError(f"Smoke capture has incomplete variant results: {summary}")
    for variant in VARIANTS:
        rows = [
            row for row in load_jsonl(capture_results_path(capture_root, variant))
            if row.get("status") == "completed"
        ]
        if len(rows) != 20:
            raise RuntimeError(f"Smoke {variant} has {len(rows)} completed result rows, expected 20")
        for row in rows:
            hidden_path = capture_root / variant / str(row["hidden_file"])
            with np.load(hidden_path) as payload:
                if set(payload.files) != expected_keys:
                    raise RuntimeError(f"Smoke hidden-key mismatch: {hidden_path}")


def run_smoke(
    *,
    output_root: Path = OUTPUT_ROOT,
    model_path: Path = MODEL_PATH,
    capture_only: bool = False,
) -> None:
    smoke_parent = output_root.resolve() / "_smoke"
    smoke_parent.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="run-", dir=smoke_parent))
    capture_root = run_root / "Capture"
    steering_root = run_root / "Steering"
    try:
        capture_summary = run_capture(
            model_path=model_path,
            output_root=capture_root,
            variants=VARIANTS,
            max_samples=20,
        )
        if capture_summary.get("status") != "complete":
            raise RuntimeError(f"Smoke capture did not complete: {capture_summary}")
        _validate_capture_smoke(capture_root, capture_summary)
        if not capture_only:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            steering_summary = run_steering(
                capture_root=capture_root,
                output_root=steering_root,
                model_path=model_path,
                variants=VARIANTS,
                positions=DEFAULT_STEERING_POSITIONS,
                layers=(8, 35),
                alphas=(-2.0, 0.0, 2.0),
                smoke=True,
            )
            if steering_summary.get("status") != "complete":
                raise RuntimeError(f"Smoke steering did not complete: {steering_summary}")
    except Exception:
        print(f"Smoke failed; retained diagnostics at {run_root}")
        raise
    else:
        shutil.rmtree(run_root)
        try:
            smoke_parent.rmdir()
        except OSError:
            pass
        stages = "capture" if capture_only else "capture and steering"
        print(f"Smoke {stages} passed; temporary output was deleted.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run capture and steering smoke checks, deleting output only after full success"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--capture-only", action="store_true")
    args = parser.parse_args(argv)
    run_smoke(
        output_root=args.output_root,
        model_path=args.model_path,
        capture_only=args.capture_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
