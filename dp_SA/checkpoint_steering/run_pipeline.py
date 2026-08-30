from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .analyze import analyze
from .capture import run_capture
from .config import (
    FORMAL_CAPTURE_FORWARDS,
    FORMAL_STEERING_FORWARDS,
    FORMAL_TOTAL_FORWARDS,
    RESULTS_ROOT,
    SMOKE_ROOT,
)
from .io_utils import append_jsonl, atomic_json, atomic_jsonl, atomic_text, sha256_file
from .run import run_steering


TEST_PATHS = ("dp_SA/checkpoint_steering/tests", "dp_SA/tests")


def _test_gate() -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *TEST_PATHS],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    passed = failed = 0
    for line in completed.stdout.splitlines()[-15:]:
        match = re.search(r"(\d+) passed", line)
        if match:
            passed = int(match.group(1))
        match = re.search(r"(\d+) failed", line)
        if match:
            failed = int(match.group(1))
    result = {"status": "passed" if completed.returncode == 0 else "failed", "return_code": completed.returncode, "passed": passed, "failed": failed, "output": completed.stdout}
    if completed.returncode:
        raise RuntimeError("CPU test gate failed\n" + completed.stdout)
    return result


def _next_smoke_round() -> tuple[int, Path]:
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name.removeprefix("round_")) for path in SMOKE_ROOT.glob("round_*" ) if path.name.removeprefix("round_").isdigit()]
    number = max(numbers, default=0) + 1
    if number > 5:
        raise RuntimeError("Five GPU smoke rounds have already been attempted")
    return number, SMOKE_ROOT / f"round_{number}"


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _required_smoke(root: Path) -> dict[str, bool]:
    names = [
        *(f"figures/{position}_delta_sa_by_layer.png" for position in (
            "P1_LAT", "P1_PANL", "P1_ATTRIBUTION_DEFINITION_END", "P1_CLASS_LIST_END", "P1_FORMAT_DESCRIPTION_END",
        )),
        "figures/position_slope_comparison.png",
        "tables/steering_delta_sa_long.csv", "tables/steering_delta_sa_wide.csv",
        "tables/dose_response_by_position_layer.csv", "tables/position_contrasts.csv",
        "tables/run_audit.csv", "tables/README.md", "progress/completion.json",
    ]
    return {name: (root / name).is_file() and (root / name).stat().st_size > 0 for name in names}


def _smoke_report_markdown(report: dict[str, Any]) -> str:
    return (
        "# Checkpoint steering GPU smoke\n\n"
        f"- Round: {report['round']}\n"
        f"- Status: {report['status']}\n"
        f"- CPU tests: {report['tests_passed']} passed, {report['tests_failed']} failed\n"
        f"- Clean forwards: {report['capture_forwards']}\n"
        f"- Steering forwards: {report['steering_forwards']}\n"
        f"- Total forwards: {report['gpu_forward_count']}\n"
        f"- Alpha-zero parity: {report['alpha_zero_parity']}\n"
        f"- Resume no-op: {report['resume_noop']}\n"
        f"- Changed files since prior round: {', '.join(report.get('changed_files', [])) or 'none'}\n"
        f"- Elapsed seconds: {report['elapsed_seconds']:.1f}\n"
        f"- Failure: {report.get('failure_reason') or 'none'}\n"
    )


def _latest_passed_smoke_report() -> Path | None:
    reports = sorted(SMOKE_ROOT.glob("round_*/progress/smoke_report.md"), key=lambda path: int(path.parents[1].name.removeprefix("round_")))
    return reports[-1] if reports else None


def _source_fingerprints() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {path.name: sha256_file(path) for path in sorted(package.glob("*.py"))}


def _changed_files(round_number: int, current: dict[str, str]) -> list[str]:
    previous_path = SMOKE_ROOT / f"round_{round_number - 1}" / "progress" / "smoke_round.json"
    if not previous_path.is_file():
        return sorted(current)
    previous = json.loads(previous_path.read_text()).get("source_fingerprints", {})
    return sorted(name for name in set(current) | set(previous) if current.get(name) != previous.get(name))


def _run_smoke(root: Path, round_number: int) -> dict[str, Any]:
    started = time.time()
    failure_path = root / "progress" / "failures.jsonl"
    if not failure_path.exists():
        atomic_jsonl(failure_path, [])
    test: dict[str, Any] = {"passed": 0, "failed": 0}
    capture: dict[str, Any] = {}
    steering: dict[str, Any] = {}
    resume_capture: dict[str, Any] = {}
    resume_steering: dict[str, Any] = {}
    resume_analysis: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    status = "failed"
    error: str | None = None
    try:
        test = _test_gate()
        capture = run_capture(output_root=root, smoke=True, resume=False)
        steering = run_steering(output_root=root, smoke=True, resume=False)
        analysis = analyze(output_root=root, smoke=True, resume=False)
        resume_capture = run_capture(output_root=root, smoke=True, resume=True)
        resume_steering = run_steering(output_root=root, smoke=True, resume=True)
        resume_analysis = analyze(output_root=root, smoke=True, resume=True)
        checks = _required_smoke(root)
        if capture.get("new_gpu_forwards") != 8 or steering.get("new_gpu_forwards") != 120:
            raise RuntimeError(f"Unexpected smoke forward counts: capture={capture}, steering={steering}")
        if resume_capture.get("new_gpu_forwards") != 0 or resume_steering.get("new_gpu_forwards") != 0 or not resume_analysis.get("resumed_noop"):
            raise RuntimeError("Smoke resume was not a zero-forward no-op")
        if not all(checks.values()) or analysis.get("alpha_zero_parity") != "passed":
            raise RuntimeError(f"Smoke schema/parity gate failed: {checks}")
        status = "passed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        append_jsonl(root / "progress" / "failures.jsonl", {"stage": "gpu_smoke", "round": round_number, "message": error, "timestamp": time.time()})
    source_fingerprints = _source_fingerprints()
    gpu_seconds = float(capture.get("elapsed_seconds", 0.0)) + float(steering.get("elapsed_seconds", 0.0))
    gpu_forwards = capture.get("new_gpu_forwards", 0) + steering.get("new_gpu_forwards", 0)
    bytes_written = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    report = {
        "round": round_number,
        "status": status,
        "tests_passed": test.get("passed", 0),
        "tests_failed": test.get("failed", 0),
        "capture_forwards": capture.get("new_gpu_forwards", 0),
        "steering_forwards": steering.get("new_gpu_forwards", 0),
        "gpu_forward_count": capture.get("new_gpu_forwards", 0) + steering.get("new_gpu_forwards", 0),
        "alpha_zero_parity": "passed" if status == "passed" else "failed_or_not_reached",
        "resume_noop": bool(resume_capture.get("resumed_noop") and resume_steering.get("resumed_noop") and resume_analysis.get("resumed_noop")),
        "artifact_checks": checks,
        "failure_reason": error,
        "elapsed_seconds": time.time() - started,
        "output_root": str(root.resolve()),
        "source_fingerprints": source_fingerprints,
        "changed_files": _changed_files(round_number, source_fingerprints),
        "formal_estimate": {
            "clean_forwards": FORMAL_CAPTURE_FORWARDS,
            "steering_forwards": FORMAL_STEERING_FORWARDS,
            "total_forwards": FORMAL_TOTAL_FORWARDS,
            "seconds_from_smoke_rate": gpu_seconds / max(gpu_forwards, 1) * FORMAL_TOTAL_FORWARDS,
            "bytes_from_smoke_linear_scale": int(bytes_written / max(gpu_forwards, 1) * FORMAL_TOTAL_FORWARDS),
            "planned_hours": [3.0, 3.5],
            "planned_disk_mb": [70, 100],
        },
    }
    atomic_json(root / "progress" / "smoke_round.json", report)
    atomic_text(root / "progress" / "smoke_report.md", _smoke_report_markdown(report))
    if status != "passed":
        raise RuntimeError(f"GPU smoke round {round_number} failed: {error}")
    return report


def run_pipeline(*, output_root: Path | None = None, smoke: bool = False, resume: bool = False) -> dict[str, Any]:
    if smoke:
        if output_root is None:
            round_number, root = _next_smoke_round()
        else:
            round_number, _default = _next_smoke_round()
            root = Path(output_root)
        if _inside(root, RESULTS_ROOT):
            raise ValueError("Smoke output must remain outside the formal results directory")
        if root.exists() and any(root.iterdir()) and not resume:
            raise FileExistsError(f"Smoke round output exists: {root}")
        return _run_smoke(root, round_number)

    root = Path(output_root or RESULTS_ROOT)
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"Formal results exist; use --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    smoke_report = _latest_passed_smoke_report()
    if smoke_report is not None and not (root / "progress" / "smoke_report.md").exists():
        (root / "progress").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(smoke_report, root / "progress" / "smoke_report.md")
    estimate = {
        "clean_forwards": FORMAL_CAPTURE_FORWARDS,
        "steering_forwards": FORMAL_STEERING_FORWARDS,
        "total_forwards": FORMAL_TOTAL_FORWARDS,
        "estimated_hours": [3.0, 3.5],
        "estimated_disk_mb": [70, 100],
    }
    atomic_json(root / "progress" / "formal_estimate.json", estimate)
    try:
        capture = run_capture(output_root=root, smoke=False, resume=resume)
        steering = run_steering(output_root=root, smoke=False, resume=resume)
        analysis = analyze(output_root=root, smoke=False, resume=resume)
        return {"status": "complete", "capture": capture, "steering": steering, "analysis": analysis, "estimate": estimate}
    except Exception as exc:
        append_jsonl(root / "progress" / "failures.jsonl", {"stage": "pipeline", "type": type(exc).__name__, "message": str(exc), "timestamp": time.time()})
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint steering pipeline")
    parser.add_argument("--output-root")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    result = run_pipeline(output_root=Path(args.output_root) if args.output_root else None, smoke=args.smoke, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
