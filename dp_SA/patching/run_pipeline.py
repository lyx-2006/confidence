from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from .analyze import analyze
from .config import DEFAULT_OUTPUT_PARENT
from .io import atomic_json, load_jsonl_strict
from .run import _default_output, build_parser as build_run_parser


TEST_PATHS = (
    "layer_metacognition/tests/test_sa_patching.py",
    "dp_SA/tests",
    "dp_SA/attention_block/tests",
    "dp_SA/patching/tests",
)


def _test_gate(output: Path) -> dict:
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", *TEST_PATHS],
                               cwd=Path(__file__).resolve().parents[2], env=environment,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "test_gate.log").write_text(completed.stdout, encoding="utf-8")
    gate = {"status": "passed" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "summary_tail": completed.stdout.splitlines()[-3:]}
    atomic_json(output / "test_gate.json", gate)
    if completed.returncode:
        raise RuntimeError(f"Test gate failed; see {output / 'test_gate.log'}")
    return gate


def _namespace_for(args: argparse.Namespace, output: Path, *, smoke: bool, resume: bool) -> argparse.Namespace:
    return argparse.Namespace(
        capture_dir=args.capture_dir, output_dir=output, model_path=args.model_path,
        dataset=args.dataset, image_root=args.image_root, positions=args.positions,
        layers=args.layers, eval_cases=args.eval_cases, seed=args.seed,
        bootstrap=args.bootstrap, resume=resume, smoke=smoke,
        corruption=args.corruption,
    )


def _smoke_gate(output: Path, args: argparse.Namespace) -> dict:
    smoke = output / "smoke"
    log_path = smoke / "smoke.log"
    smoke.mkdir(parents=True, exist_ok=True)
    first_command = _formal_command(args, smoke, smoke=True,
                                    force_resume=(smoke / "run_config.json").exists())
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(first_command, cwd=Path(__file__).resolve().parents[2],
                                   env=environment, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"GPU smoke execution failed; see {log_path}")
    first = json.loads((smoke / "run_completion.json").read_text())
    analyze(smoke, repeats=100, seed=args.seed, final=True)
    before = {
        "baselines": len(load_jsonl_strict(smoke / "baselines.jsonl")),
        "results": len(load_jsonl_strict(smoke / "results.jsonl")),
        "completion_mtime": (smoke / "run_completion.json").stat().st_mtime_ns,
    }
    resume_command = _formal_command(args, smoke, smoke=True, force_resume=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(resume_command, cwd=Path(__file__).resolve().parents[2],
                                   env=environment, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"GPU smoke resume failed; see {log_path}")
    second = json.loads((smoke / "run_completion.json").read_text())
    after = {
        "baselines": len(load_jsonl_strict(smoke / "baselines.jsonl")),
        "results": len(load_jsonl_strict(smoke / "results.jsonl")),
    }
    checks = {
        "four_balanced_cases": first["baseline_count"] == 4,
        "eight_patch_cells": first["patch_cell_count"] == 8,
        "resume_baseline_count_unchanged": before["baselines"] == after["baselines"],
        "resume_result_count_unchanged": before["results"] == after["results"],
        "resume_reports_same_grid": second["baseline_count"] == first["baseline_count"] and second["patch_cell_count"] == first["patch_cell_count"],
        "completion_exists": (smoke / "completion.json").is_file(),
    }
    gate = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "first_run": first, "resume_run": second}
    atomic_json(output / "smoke_gate.json", gate)
    if gate["status"] != "passed":
        raise RuntimeError(f"Smoke gate failed: {gate}")
    return gate


def _copy_shared_smoke_inputs(output: Path) -> None:
    smoke = output / "smoke"
    for name in ("evaluation_manifest.json", "calibration_manifest.json"):
        destination = output / name
        if destination.exists():
            if destination.read_bytes() != (smoke / name).read_bytes():
                raise ValueError(f"Formal/smoke manifest differs: {name}")
        else:
            shutil.copy2(smoke / name, destination)
    source_artifacts = smoke / "corruption_embeddings"
    target_artifacts = output / "corruption_embeddings"
    if target_artifacts.exists():
        source_manifest = json.loads((source_artifacts / "artifact_manifest.json").read_text())
        target_manifest = json.loads((target_artifacts / "artifact_manifest.json").read_text())
        if source_manifest["fingerprint"] != target_manifest["fingerprint"]:
            raise ValueError("Formal/smoke corruption artifacts differ")
    else:
        shutil.copytree(source_artifacts, target_artifacts)


def _formal_command(args: argparse.Namespace, output: Path, *, smoke: bool = False,
                    force_resume: bool | None = None) -> list[str]:
    command = [sys.executable, "-m", "dp_SA.patching.run",
               "--capture-dir", str(args.capture_dir), "--output-dir", str(output),
               "--model-path", str(args.model_path), "--dataset", str(args.dataset),
               "--positions", *map(str, args.positions), "--layers", *map(str, args.layers),
               "--eval-cases", str(args.eval_cases), "--seed", str(args.seed),
               "--bootstrap", str(args.bootstrap), "--corruption", str(args.corruption)]
    if args.image_root:
        command += ["--image-root", str(args.image_root)]
    if smoke:
        command.append("--smoke")
    use_resume = (args.resume or (output / "run_config.json").exists()) if force_resume is None else force_resume
    if use_resume:
        command.append("--resume")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_run_parser()
    args = parser.parse_args(argv)
    output = (args.output_dir or _default_output()).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        command = _formal_command(args, output, smoke=True, force_resume=args.resume)
        subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=True)
        analyze(output, repeats=100, seed=args.seed, final=True)
        return 0
    atomic_json(output / "pipeline_state.json", {"status": "running", "stage": "tests"})
    _test_gate(output)
    atomic_json(output / "pipeline_state.json", {"status": "running", "stage": "gpu_smoke"})
    _smoke_gate(output, args)
    _copy_shared_smoke_inputs(output)
    log_path = output / "formal.log"
    command = _formal_command(args, output)
    atomic_json(output / "pipeline_state.json", {"status": "running", "stage": "formal", "command": command})
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(output / "formal_process.json", {"pid": process.pid, "log": str(log_path),
                                                     "command": command, "started_at_unix": time.time()})
        while process.poll() is None:
            time.sleep(5)
        return_code = int(process.returncode)
    if return_code:
        atomic_json(output / "pipeline_state.json", {"status": "failed", "stage": "formal", "return_code": return_code,
                                                     "pid": process.pid, "log": str(log_path)})
        raise RuntimeError(f"Formal patching failed with code {return_code}; see {log_path}")
    atomic_json(output / "pipeline_state.json", {"status": "running", "stage": "analysis", "formal_pid": process.pid})
    summary = analyze(output, repeats=args.bootstrap, seed=args.seed, final=True)
    atomic_json(output / "pipeline_state.json", {"status": "complete", "stage": "complete", "formal_pid": process.pid,
                                                 "log": str(log_path), "claim_supported": summary["any_layer_supports_functional_information_claim"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
