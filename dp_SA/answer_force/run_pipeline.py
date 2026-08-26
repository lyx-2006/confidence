from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from dp_SA.io_utils import atomic_json

from .analyze import analyze
from .config import BOOTSTRAP_REPEATS, DATASET_PATH, MODEL_PATH, OUTPUT_PARENT, SEED, SOURCE_ROOT, SPLIT_PATH, default_run_name
from .run import run_experiment


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _test_gate(output: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    paths = ["dp_SA/answer_force/tests", "dp_SA/tests", "dp_SA/patching/tests", "dp_SA/activation_swap/tests", "layer_metacognition/tests/test_sa_patching.py"]
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", *paths], cwd=Path(__file__).resolve().parents[2], env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _atomic_text(output / "test_gate.log", completed.stdout)
    gate = {"status": "passed" if completed.returncode == 0 else "failed", "return_code": completed.returncode, "summary_tail": completed.stdout.splitlines()[-5:]}
    atomic_json(output / "test_gate.json", gate)
    if completed.returncode:
        raise RuntimeError(f"CPU test gate failed; see {output / 'test_gate.log'}")
    return gate


def _run_smoke(output: Path, args: argparse.Namespace, *, resume: bool) -> dict[str, Any]:
    return run_experiment(output_root=output, source_root=args.source_root, dataset=args.dataset, split_path=args.split, model_path=args.model_path, seed=args.seed, resume=resume, smoke=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU gate and 2+2 GPU smoke for Answer-force; never launches formal scale")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = (args.output_root or OUTPUT_PARENT / (args.run_name or default_run_name("pipeline_seed42"))).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Answer-force pipeline output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "pipeline_state.json", {"status": "running", "stage": "tests"})
    test_gate = _test_gate(output)
    smoke = output / "smoke"
    atomic_json(output / "pipeline_state.json", {"status": "running", "stage": "gpu_smoke", "test_gate": str(output / "test_gate.json")})
    first = _run_smoke(smoke, args, resume=(smoke / "run_config.json").exists())
    analyze(smoke, repeats=BOOTSTRAP_REPEATS, seed=args.seed)
    before = {"trials": int(first["trial_count"]), "fingerprint": str(first["run_fingerprint"])}
    second = _run_smoke(smoke, args, resume=True)
    after = {"trials": int(second["trial_count"]), "fingerprint": str(second["run_fingerprint"])}
    required = [
        "run_config.json", "input_fingerprint.json", "input_fingerprints.json", "probe_reconstruction_audit.json", "probe_leakage_audit.json", "clean_parity_audit.json", "recipient_manifest.jsonl", "excluded_records.jsonl", "unrelated_answer_manifest.jsonl", "unrelated_matching_diagnostics.json", "failures.jsonl", "results.jsonl", "item_level_metrics.csv", "aggregate_metrics.csv", "bootstrap_results.csv", "regression_results.csv", "correlations.csv", "clean_force_soft_sa_correlations.csv", "specificity_metrics.csv", "hard_class_directional_proportions.csv", "token_matched_aggregate_metrics.csv", "summary.json", "summary.md", "progress.json", "completion.json", "panl_final_signed_delta.png", "panl_final_absolute_delta.png", "delta_sa_absolute_overall.png", "panl_final_absolute_delta_overall.png", "opposite_directional_effect.png", "hard_label_change_rate.png", "panl_vs_final_delta_scatter.png", "original_vs_forced_sa.png",
    ]
    artifact_checks = {
        name: (smoke / name).is_file() and (name == "failures.jsonl" or (smoke / name).stat().st_size > 0)
        for name in required
    }
    smoke_gate = {
        "status": "passed" if all(artifact_checks.values()) and first["recipient_count"] == 4 and first["trial_count"] == 12 and before["trials"] == after["trials"] else "failed",
        "checks": {"cpu_tests_passed": test_gate["status"] == "passed", "four_balanced_recipients": first["recipient_count"] == 4, "three_conditions_each": first["trial_count"] == 12, "resume_noop_trial_count": before["trials"] == after["trials"], "resume_fingerprint_unchanged": before["fingerprint"] == after["fingerprint"], "artifact_gate": all(artifact_checks.values())},
        "first_run": first, "resume_run": second, "artifact_checks": artifact_checks,
    }
    atomic_json(output / "smoke_gate.json", smoke_gate)
    if smoke_gate["status"] != "passed":
        raise RuntimeError(f"Answer-force GPU smoke gate failed: {smoke_gate}")
    formal = output / "formal"
    formal_command = [sys.executable, "-m", "dp_SA.answer_force.run", "--output-root", str(formal), "--source-root", str(args.source_root), "--dataset", str(args.dataset), "--split", str(args.split), "--model-path", str(args.model_path), "--seed", str(args.seed)]
    analysis_command = [sys.executable, "-m", "dp_SA.answer_force.analyze", "--output-root", str(formal), "--bootstrap", str(BOOTSTRAP_REPEATS), "--seed", str(args.seed)]
    command_text = " ".join(formal_command) + "\n" + " ".join(analysis_command) + "\n"
    _atomic_text(output / "formal_command.txt", command_text)
    atomic_json(output / "pipeline_state.json", {"status": "awaiting_formal_confirmation", "stage": "awaiting_formal_confirmation", "formal_command": command_text, "smoke_gate": str(output / "smoke_gate.json")})
    print(command_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
