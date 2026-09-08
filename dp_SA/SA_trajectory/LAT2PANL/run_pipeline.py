from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .analyze import analyze
from .config import (
    ANSWER_MATCHED_ROOT, EXPECTED_SHA256, FORMAL_FORWARD_COUNT, MANIFEST_PATH,
    PROBE_CONSTRUCTION_PATH, RESULTS_ROOT, SMOKE_FORWARD_COUNT, SMOKE_ROOT,
    VECTOR_FILE_SHA256,
)
from .io_utils import atomic_json, sha256_file
from .prepare import prepare
from .run import run


def _history_hashes() -> dict[str, str]:
    paths = [MANIFEST_PATH, PROBE_CONSTRUCTION_PATH]
    paths.extend(ANSWER_MATCHED_ROOT / "artifacts" / "vectors" / f"P1_LAT__fold_{fold:02d}__L14.npz" for fold in VECTOR_FILE_SHA256)
    return {str(path): sha256_file(path) for path in paths}


def run_cpu_tests() -> dict[str, Any]:
    environment = dict(os.environ); environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    command = [sys.executable, "-m", "pytest", "-q", "dp_SA/SA_trajectory/LAT2PANL/tests"]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[3], env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    matches = re.findall(r"(\d+) passed", completed.stdout); passed = int(matches[-1]) if matches else 0
    result = {"status": "passed" if completed.returncode == 0 else "failed", "passed": passed,
              "return_code": completed.returncode, "output": completed.stdout}
    if completed.returncode: raise RuntimeError("CPU tests failed\n" + completed.stdout)
    return result


def _next_smoke_root() -> tuple[int, Path]:
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name.removeprefix("round_")) for path in SMOKE_ROOT.glob("round_*") if path.name.removeprefix("round_").isdigit()]
    number = max(numbers, default=0) + 1
    if number > 5: raise RuntimeError("Maximum five smoke rounds reached")
    return number, SMOKE_ROOT / f"round_{number}"


def smoke_pipeline(*, output_root: Path | None = None, num_gpus: int = 1, resume: bool = False) -> dict[str, Any]:
    round_number, default_root = _next_smoke_root(); root = Path(output_root or default_root)
    started = time.time(); before = _history_hashes(); tests = run_cpu_tests()
    prepared = prepare(output_root=root, smoke=True, resume=resume)
    executed = run(output_root=root, smoke=True, resume=resume, num_gpus=num_gpus)
    analyzed = analyze(output_root=root, smoke=True, resume=resume)
    resumed = run(output_root=root, smoke=True, resume=True, num_gpus=num_gpus)
    resumed_analysis = analyze(output_root=root, smoke=True, resume=True)
    if executed["new_gpu_forwards"] != SMOKE_FORWARD_COUNT: raise RuntimeError(f"Smoke forward count {executed['new_gpu_forwards']} != {SMOKE_FORWARD_COUNT}")
    if not resumed["resumed_noop"] or resumed["new_gpu_forwards"] != 0 or not resumed_analysis["resumed_noop"]:
        raise RuntimeError("Resume zero-forward gate failed")
    after = _history_hashes()
    if before != after: raise RuntimeError("A frozen historical artifact changed")
    manip = (root / "tables" / "manipulation_checks.csv").read_text()
    if ",False," in manip or not all((root / "figures" / name).is_file() for name in ("fig1_final_sa_delta.png", "fig2_cle_probe_sa_delta.png", "fig3_final_sa_attenuation.png")):
        raise RuntimeError("Smoke manipulation/figure gate failed")
    result = {"status": "passed", "round": round_number, "output_root": str(root.resolve()),
              "cpu_tests_passed": tests["passed"], "gpu_forwards": executed["new_gpu_forwards"],
              "c0_parity": executed["c0_gate"], "replacement_gate": "passed", "l14_negative_control": "passed",
              "resume_zero_forward": True, "historical_unchanged": True, "elapsed_seconds": time.time() - started,
              "prepare": prepared, "analysis": analyzed}
    atomic_json(root / "progress" / "smoke_report.json", result); atomic_json(SMOKE_ROOT / "latest_status.json", result)
    return result


def pipeline(*, output_root: Path | None = None, smoke: bool = False, resume: bool = False,
             num_gpus: int = 1, run_formal: bool = False) -> dict[str, Any]:
    if smoke: return smoke_pipeline(output_root=output_root, num_gpus=num_gpus, resume=resume)
    output_root = Path(output_root or RESULTS_ROOT)
    prepared = prepare(output_root=output_root, smoke=False, resume=resume)
    if not run_formal:
        return {"status": "prepared_only", "formal_run_started": False, "prepare": prepared,
                "expected_gpu_forwards": FORMAL_FORWARD_COUNT,
                "next_command": f"python -m dp_SA.SA_trajectory.LAT2PANL.run_pipeline --run-formal --num-gpus {num_gpus} --resume"}
    latest = SMOKE_ROOT / "latest_status.json"
    if not latest.is_file() or json.loads(latest.read_text()).get("status") != "passed":
        raise RuntimeError("Formal execution requires a passed smoke")
    before = _history_hashes(); executed = run(output_root=output_root, smoke=False, resume=resume, num_gpus=num_gpus)
    analyzed = analyze(output_root=output_root, smoke=False, resume=resume)
    if before != _history_hashes(): raise RuntimeError("A frozen historical artifact changed")
    result = {"status": "complete", "formal_run_started": True, "prepare": prepared, "run": executed,
              "analysis": analyzed, "historical_unchanged": True}
    atomic_json(output_root / "progress" / "completion.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-formal", action="store_true"); parser.add_argument("--num-gpus", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    print(json.dumps(pipeline(output_root=args.output_root, smoke=args.smoke, resume=args.resume,
                              num_gpus=args.num_gpus, run_formal=args.run_formal), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
