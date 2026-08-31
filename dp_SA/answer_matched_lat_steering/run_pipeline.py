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
from .config import (
    FORMAL_CAPTURE_FORWARDS, FORMAL_STEERING_FORWARDS, FORMAL_TOTAL_FORWARDS,
    HISTORICAL_CAPTURE, HISTORICAL_CONSTRUCTION, HISTORICAL_TEST,
    MAX_SMOKE_ROUNDS, RESULTS_ROOT, SMOKE_ROOT,
)
from .io_utils import append_jsonl, atomic_json, atomic_text, sha256_file
from .prepare import run_prepare
from .run import run_steering


TEST_PATHS = ("dp_SA/answer_matched_lat_steering/tests", "dp_SA/checkpoint_steering/tests", "dp_SA/tests")


def run_cpu_tests() -> dict[str, Any]:
    environment = dict(os.environ); environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", *TEST_PATHS], cwd=Path(__file__).resolve().parents[2], env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    passed = failed = 0
    for line in completed.stdout.splitlines():
        match = re.search(r"(\d+) passed", line)
        if match: passed = int(match.group(1))
        match = re.search(r"(\d+) failed", line)
        if match: failed = int(match.group(1))
    result = {"status": "passed" if completed.returncode == 0 else "failed", "return_code": completed.returncode, "passed": passed, "failed": failed, "output": completed.stdout}
    if completed.returncode: raise RuntimeError("CPU test gate failed\n" + completed.stdout)
    return result


def _next_round() -> tuple[int, Path]:
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name.removeprefix("round_")) for path in SMOKE_ROOT.glob("round_*") if path.name.removeprefix("round_").isdigit()]
    number = max(numbers, default=0) + 1
    if number > MAX_SMOKE_ROUNDS: raise RuntimeError("Five GPU smoke rounds have already been attempted")
    return number, SMOKE_ROOT / f"round_{number}"


def _inside(path: Path, parent: Path) -> bool:
    try: path.resolve().relative_to(parent.resolve()); return True
    except ValueError: return False


def _historical_hashes() -> dict[str, str]:
    return {str(path): sha256_file(path) for path in (HISTORICAL_CAPTURE, HISTORICAL_CONSTRUCTION, HISTORICAL_TEST)}


def _smoke_report(report: dict[str, Any]) -> str:
    return f"""# Answer-matched LAT steering smoke

- round: {report['round']}
- status: {report['status']}
- next_state: {report['next_state']}
- CPU tests: {report['tests_passed']} passed, {report['tests_failed']} failed
- clean GPU forwards: {report['capture_forwards']}
- steering GPU forwards: {report['steering_forwards']}
- total GPU forwards: {report['gpu_forward_count']}
- alpha=0 parity: {report['alpha_zero_parity']}
- resume no-op: {report['resume_noop']}
- historical artifacts unchanged: {report['historical_unchanged']}
- elapsed seconds: {report['elapsed_seconds']:.1f}
- failure: {report.get('failure_reason') or 'none'}
"""


def run_smoke(root: Path, round_number: int) -> dict[str, Any]:
    started = time.time(); before = _historical_hashes(); failure = None; status = "failed"; tests = {"passed": 0, "failed": 0}; prepare = {}; steering = {}; analysis = {}; resume_prepare = {}; resume_steering = {}; resume_analysis = {}
    try:
        tests = run_cpu_tests(); prepare = run_prepare(output_root=root, smoke=True, resume=False); steering = run_steering(output_root=root, smoke=True, resume=False); analysis = analyze(output_root=root, smoke=True, resume=False)
        resume_prepare = run_prepare(output_root=root, smoke=True, resume=True); resume_steering = run_steering(output_root=root, smoke=True, resume=True); resume_analysis = analyze(output_root=root, smoke=True, resume=True)
        capture_forwards = int(prepare["capture"]["new_gpu_forwards"]); steering_forwards = int(steering["new_gpu_forwards"])
        if capture_forwards != 20 or steering_forwards != 72: raise RuntimeError(f"Unexpected smoke forwards: capture={capture_forwards}, steering={steering_forwards}")
        if not resume_prepare["capture"].get("resumed_noop") or not resume_steering.get("resumed_noop") or not resume_analysis.get("resumed_noop"): raise RuntimeError("Resume was not a zero-forward no-op")
        required = [root / "tables" / name for name in ("delta_sa_by_layer_alpha_and_answer.csv", "dose_response_and_controls.csv", "split_and_selection_audit.csv", "README.md")] + [root / "figures" / "P1_LAT_delta_sa_by_layer.png", root / "summary.md", root / "progress" / "completion.json"]
        if not all(path.is_file() and path.stat().st_size > 0 for path in required) or analysis.get("alpha_zero_parity") != "passed": raise RuntimeError("Smoke schema/parity audit failed")
        if before != _historical_hashes(): raise RuntimeError("Historical artifacts changed during smoke")
        status = "passed"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"; append_jsonl(root / "progress" / "failures.jsonl", {"stage": "gpu_smoke", "round": round_number, "message": failure, "timestamp": time.time()})
    report = {"round": round_number, "status": status, "next_state": "awaiting_formal_confirmation" if status == "passed" else "smoke_failed", "tests_passed": tests.get("passed", 0), "tests_failed": tests.get("failed", 0), "capture_forwards": prepare.get("capture", {}).get("new_gpu_forwards", 0), "steering_forwards": steering.get("new_gpu_forwards", 0), "gpu_forward_count": prepare.get("capture", {}).get("new_gpu_forwards", 0) + steering.get("new_gpu_forwards", 0), "alpha_zero_parity": "passed" if analysis.get("alpha_zero_parity") == "passed" else "failed_or_not_reached", "resume_noop": bool(resume_prepare.get("capture", {}).get("resumed_noop") and resume_steering.get("resumed_noop") and resume_analysis.get("resumed_noop")), "historical_unchanged": before == _historical_hashes(), "failure_reason": failure, "elapsed_seconds": time.time() - started, "output_root": str(root.resolve()), "formal_estimate": {"reused_hidden_cases": 150, "clean_forwards": FORMAL_CAPTURE_FORWARDS, "steering_forwards": FORMAL_STEERING_FORWARDS, "total_forwards": FORMAL_TOTAL_FORWARDS, "estimated_hours": [2.0, 2.5], "estimated_disk_mb": [130, 180], "command": "python -m dp_SA.answer_matched_lat_steering.run_pipeline --output-root dp_SA/answer_matched_lat_steering/output/results"}}
    atomic_json(root / "progress" / "smoke_report.json", report); atomic_text(root / "progress" / "smoke_report.md", _smoke_report(report)); atomic_json(SMOKE_ROOT / "latest_status.json", report)
    if status != "passed": raise RuntimeError(f"GPU smoke round {round_number} failed: {failure}")
    return report


def run_pipeline(*, output_root: Path | None = None, smoke: bool = False, resume: bool = False) -> dict[str, Any]:
    if smoke:
        round_number, default_root = _next_round(); root = Path(output_root) if output_root else default_root
        if _inside(root, RESULTS_ROOT): raise ValueError("Smoke output must remain outside formal results")
        if root.exists() and any(root.iterdir()) and not resume: raise FileExistsError(f"Smoke output exists: {root}")
        return run_smoke(root, round_number)
    root = Path(output_root or RESULTS_ROOT)
    if root.exists() and any(root.iterdir()) and not resume: raise FileExistsError(f"Formal results exist; use --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    latest = SMOKE_ROOT / "latest_status.json"
    if latest.is_file():
        report = json.loads(latest.read_text())
        if report.get("status") != "passed": raise RuntimeError("Formal run requires a passed smoke")
        (root / "progress").mkdir(parents=True, exist_ok=True); shutil.copyfile(Path(report["output_root"]) / "progress" / "smoke_report.md", root / "progress" / "smoke_report.md")
    prepare = run_prepare(output_root=root, smoke=False, resume=resume); steering = run_steering(output_root=root, smoke=False, resume=resume); analysis = analyze(output_root=root, smoke=False, resume=resume)
    return {"status": "complete", "prepare": prepare, "steering": steering, "analysis": analysis}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv); result = run_pipeline(output_root=Path(args.output_root) if args.output_root else None, smoke=args.smoke, resume=args.resume); print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
