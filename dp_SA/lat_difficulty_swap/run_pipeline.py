from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .analyze import analyze
from .build_pairs import build_pair_artifacts
from .config import RESULTS_ROOT, SEED, SMOKE_BOOTSTRAP_REPEATS, SMOKE_ROOT, SWAP_LAYERS
from .io_utils import atomic_json, atomic_jsonl, atomic_text, load_jsonl
from .probe_runtime import prepare_probe_models
from .run import run_experiment


TEST_PATHS = ("dp_SA/lat_difficulty_swap/tests", "dp_SA/tests", "dp_SA/activation_swap/tests", "dp_SA/panl_information/tests")


def _test_gate() -> dict[str, Any]:
    environment = dict(os.environ); environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", *TEST_PATHS], cwd=Path(__file__).resolve().parents[2], env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = completed.stdout.splitlines(); passed = 0; failed = 0
    for line in lines[-10:]:
        import re
        match = re.search(r"(\d+) passed", line); passed = int(match.group(1)) if match else passed
        match = re.search(r"(\d+) failed", line); failed = int(match.group(1)) if match else failed
    result = {"status": "passed" if completed.returncode == 0 else "failed", "return_code": completed.returncode, "passed": passed, "failed": failed, "output": completed.stdout}
    if completed.returncode: raise RuntimeError("CPU test gate failed\n" + completed.stdout)
    return result


def _next_round() -> int:
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    existing = [int(path.name.split("_")[-1]) for path in SMOKE_ROOT.glob("round_*") if path.name.split("_")[-1].isdigit()]
    return max(existing, default=0) + 1


def _round_history() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(SMOKE_ROOT.glob("round_*/progress/smoke_round.json"), key=lambda value: int(value.parents[1].name.split("_")[-1])):
        rows.append(json.loads(path.read_text()))
    return rows


def _source_changes(round_number: int, smoke: Path) -> tuple[str, list[str]]:
    current = json.loads((smoke / "progress" / "run_config.json").read_text())
    current_code = current["source_fingerprints"]["source_code"]
    previous_path = SMOKE_ROOT / f"round_{round_number - 1}" / "progress" / "run_config.json"
    if not previous_path.is_file(): return current["fingerprint"], sorted(current_code)
    previous = json.loads(previous_path.read_text())["source_fingerprints"]["source_code"]
    changed = sorted(path for path in set(current_code) | set(previous) if current_code.get(path) != previous.get(path))
    return current["fingerprint"], changed


def _required_smoke_artifacts(root: Path) -> dict[str, bool]:
    names = (
        "figures/target_directed_absolute_delta_sa.png", "figures/oriented_and_raw_absolute_delta_sa.png",
        "figures/logit_and_token_change.png", "figures/panl_probe_changes.png", "figures/difficulty_gap_dose_response.png",
        "tables/delta_sa.csv", "tables/panl_probe_results.csv", "tables/logit_token_metrics.csv",
        "tables/difficulty_gap_regression.csv", "summary.md", "artifacts/swap_audit.json", "artifacts/clean_parity_audit.json",
    )
    return {name: (root / name).is_file() and (root / name).stat().st_size > 0 for name in names}


def run_pipeline(*, resume: bool) -> dict[str, Any]:
    if RESULTS_ROOT.exists() and any(RESULTS_ROOT.iterdir()) and not resume:
        raise FileExistsError(f"Formal results exist; pass --resume: {RESULTS_ROOT}")
    if resume and (RESULTS_ROOT / "progress" / "smoke_report.json").is_file():
        build_pair_artifacts(RESULTS_ROOT, resume=True); prepare_probe_models(RESULTS_ROOT, resume=True)
        report = json.loads((RESULTS_ROOT / "progress" / "smoke_report.json").read_text()); command = (RESULTS_ROOT / "progress" / "formal_command.txt").read_text(); print(command, end=""); return {"status": "awaiting_formal_run", "resumed_noop": True, "smoke_report": report}
    test = _test_gate(); round_number = _next_round()
    if round_number > 5: raise RuntimeError("Five GPU smoke rounds have already been attempted")
    smoke = SMOKE_ROOT / f"round_{round_number}"; started = time.time()
    try:
        build_pair_artifacts(smoke, resume=False); prepare_probe_models(smoke, resume=False)
        first = run_experiment(smoke, resume=False, smoke=True, layers=(14, 18)); analysis = analyze(smoke, repeats=SMOKE_BOOTSTRAP_REPEATS, seed=SEED, resume=False)
        second = run_experiment(smoke, resume=True, smoke=True, layers=(14, 18)); checks = _required_smoke_artifacts(smoke)
        if second.get("new_gpu_forwards") != 0 or not second.get("resumed_noop") or not all(checks.values()): raise RuntimeError("Smoke no-op resume or artifact gate failed")
        status, error = "passed", None
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"; first = {}; second = {}; analysis = {}
        atomic_jsonl(smoke / "progress" / "failures.jsonl", [{"round": round_number, "error": error, "timestamp": time.time()}])
    source_fingerprint, changed_files = _source_changes(round_number, smoke)
    round_report = {"round": round_number, "status": status, "source_fingerprint": source_fingerprint, "changed_files": changed_files, "tests_passed": test["passed"], "tests_failed": test["failed"], "gpu_forward_count": first.get("new_gpu_forwards", 0), "parity": {"clean": (smoke / "artifacts" / "clean_parity_audit.json").is_file(), "swap": (smoke / "artifacts" / "swap_audit.json").is_file(), "resume_noop": second.get("resumed_noop", False)}, "failure_reason": error, "elapsed_seconds": time.time() - started}
    atomic_json(smoke / "progress" / "smoke_round.json", round_report)
    if status != "passed": raise RuntimeError(f"GPU smoke round {round_number} failed: {error}")

    build_pair_artifacts(RESULTS_ROOT, resume=False); probe_audit = prepare_probe_models(RESULTS_ROOT, resume=False)
    atomic_json(RESULTS_ROOT / "progress" / "test_report.json", {key: value for key, value in test.items() if key != "output"}); atomic_text(RESULTS_ROOT / "progress" / "pipeline.log", test["output"])
    image_count = len(load_jsonl(RESULTS_ROOT / "artifacts" / "image_pair_manifest.jsonl")); text_count = len(load_jsonl(RESULTS_ROOT / "artifacts" / "text_pair_manifest.jsonl")); clean_ids = {case for pair in load_jsonl(RESULTS_ROOT / "artifacts" / "image_pair_manifest.jsonl") + load_jsonl(RESULTS_ROOT / "artifacts" / "text_pair_manifest.jsonl") for case in (pair["easy_case_id"], pair["hard_case_id"])}
    intervention = (image_count + text_count) * 4 * len(SWAP_LAYERS); total = intervention + len(clean_ids); seconds_per_forward = first["elapsed_seconds"] / max(first["new_gpu_forwards"], 1); estimated_seconds = total * seconds_per_forward
    smoke_variable = sum(path.stat().st_size for path in (smoke / "artifacts" / "hidden").glob("*") if path.is_file()) + sum((smoke / "artifacts" / name).stat().st_size for name in ("clean_results.jsonl", "swap_results.jsonl") if (smoke / "artifacts" / name).is_file()); formal_fixed = sum(path.stat().st_size for path in RESULTS_ROOT.rglob("*") if path.is_file()); variable_smoke = max(first["new_gpu_forwards"], 1); estimated_bytes = int(formal_fixed + smoke_variable * total / variable_smoke)
    history = _round_history()
    report = {"status": "passed", "rounds": history, "image_eligible_pairs": image_count, "text_eligible_pairs": text_count, "smoke": first, "analysis": analysis, "resume": second, "formal_estimate": {"clean_forwards": len(clean_ids), "intervention_forwards": intervention, "total_forwards": total, "seconds": estimated_seconds, "hours": estimated_seconds / 3600, "disk_bytes": estimated_bytes, "disk_gib": estimated_bytes / 2**30}, "probe_audit_status": probe_audit["status"]}
    atomic_json(RESULTS_ROOT / "progress" / "smoke_report.json", report); atomic_json(RESULTS_ROOT / "progress" / "completion.json", {"status": "smoke_complete_awaiting_formal", "formal_run_started": False})
    failures = [{"round": row["round"], "stage": "gpu_smoke", "error": row["failure_reason"]} for row in history if row["status"] != "passed"]
    atomic_jsonl(RESULTS_ROOT / "progress" / "failures.jsonl", failures)
    commands = f"{sys.executable} -m dp_SA.lat_difficulty_swap.run --resume\n{sys.executable} -m dp_SA.lat_difficulty_swap.analyze --resume --bootstrap 2000 --seed 42\n"
    atomic_text(RESULTS_ROOT / "progress" / "formal_command.txt", commands); print(commands, end=""); return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU gate and GPU smoke; never launches formal scale"); parser.add_argument("--resume", action="store_true"); args = parser.parse_args(argv)
    print(json.dumps(run_pipeline(resume=args.resume), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
