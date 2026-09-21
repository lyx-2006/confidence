from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from dp_SA.io_utils import atomic_json

from .analyze_answer_matched import analyze
from .config import ANSWER_MATCHED_OUTPUT_ROOT, CAPTURE_ROOT, MODEL_PATH
from .prepare_answer_matched import prepare
from .run_answer_matched import run


def run_smoke(
    *, capture_root: Path, output_root: Path, model_path: Path,
) -> dict[str, Any]:
    parent = output_root / "_smoke"
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    started = time.time()
    try:
        prepared = prepare(
            capture_root=capture_root, output_root=root,
            positions=("LAT", "PANL", "CLE"), layers=(8, 35), smoke=True,
        )
        result = run(
            capture_root=capture_root, output_root=root, model_path=model_path,
            positions=("LAT", "PANL", "CLE"), layers=(8, 35),
            alphas=(-2.0, 0.0, 2.0), smoke=True,
        )
        analysis = analyze(root)
        report = {
            "status": "passed", "expected_predictions": 360,
            "prepared": prepared, "run": result,
            "alpha_zero_max_abs_delta": analysis["alpha_zero_max_abs_delta"],
            "elapsed_seconds": time.time() - started,
            "formal_forward_estimate": 7110,
            "formal_estimated_seconds_from_smoke": (
                7110 / result["forwards_per_second"] if result.get("forwards_per_second") else None
            ),
        }
        atomic_json(output_root / "progress" / "smoke_report.json", report)
        shutil.rmtree(root)
        try:
            parent.rmdir()
        except OSError:
            pass
        return report
    except Exception:
        atomic_json(root / "progress" / "smoke_failure.json", {
            "status": "failed", "retained_output": str(root),
            "elapsed_seconds": time.time() - started,
        })
        raise


def pipeline(
    *, capture_root: Path = CAPTURE_ROOT, output_root: Path = ANSWER_MATCHED_OUTPUT_ROOT,
    model_path: Path = MODEL_PATH, smoke_only: bool = False, skip_smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    capture_root, output_root, model_path = capture_root.resolve(), output_root.resolve(), model_path.resolve()
    smoke = None if skip_smoke else run_smoke(
        capture_root=capture_root, output_root=output_root, model_path=model_path,
    )
    if smoke_only:
        return {"status": "smoke_complete", "smoke": smoke}
    prepared = prepare(capture_root=capture_root, output_root=output_root, resume=resume)
    result = run(
        capture_root=capture_root, output_root=output_root, model_path=model_path, resume=resume,
    )
    analysis = analyze(output_root)
    summary = {"status": "complete", "smoke": smoke, "prepare": prepared, "run": result, "analysis": analysis}
    atomic_json(output_root / "progress" / "pipeline_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Native answer-matched steering pipeline")
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=ANSWER_MATCHED_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(pipeline(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
