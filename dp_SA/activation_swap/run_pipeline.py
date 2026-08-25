from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

from .analyze import analyze
from .run import _default_output, build_parser as build_run_parser, run_experiment
from .utils import atomic_json, atomic_jsonl, load_jsonl


def _test_gate(output: Path) -> dict[str, object]:
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    paths = ["dp_SA/activation_swap/tests", "dp_SA/tests", "dp_SA/patching/tests", "layer_metacognition/tests/test_sa_patching.py"]
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", *paths], cwd=Path(__file__).resolve().parents[2],
                               env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "test_gate.log").write_text(completed.stdout, encoding="utf-8")
    gate = {"status": "passed" if completed.returncode == 0 else "failed", "return_code": completed.returncode,
            "summary_tail": completed.stdout.splitlines()[-5:]}
    atomic_json(output / "test_gate.json", gate)
    if completed.returncode:
        raise RuntimeError(f"CPU test gate failed; see {output / 'test_gate.log'}")
    return gate


def _smoke_args(args: argparse.Namespace, output: Path) -> argparse.Namespace:
    return argparse.Namespace(source_root=args.source_root, output_root=output, model_path=args.model_path,
                              dataset=args.dataset, positions=args.positions, layers=args.layers,
                              bootstrap=100, seed=args.seed, resume=bool((output / "run_config.json").exists()),
                              smoke=True, max_recipients_per_side=2)


def _run_smoke_checked(args: argparse.Namespace, output: Path) -> dict[str, object]:
    try:
        return run_experiment(_smoke_args(args, output))
    except Exception as exc:
        failures = output / "failures.jsonl"
        rows = load_jsonl(failures, repair_trailing=True)
        rows.append({"status": "failed", "stage": "pipeline_gpu_smoke",
                     "error_type": type(exc).__name__, "error": str(exc), "timestamp": time.time()})
        atomic_jsonl(failures, rows)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = build_run_parser()
    parser.description = "CPU gate and GPU smoke for delayed-SA activation swap; never launches formal scale"
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = (args.output_root or _default_output()).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "pipeline_state.json", {"status": "running", "stage": "tests"})
    test_gate = _test_gate(output)
    smoke = output / "smoke"
    atomic_json(output / "pipeline_state.json", {"status": "running", "stage": "gpu_smoke"})
    first = _run_smoke_checked(args, smoke)
    analyze(smoke, repeats=100, seed=int(args.seed))
    with tempfile.TemporaryDirectory(prefix="jsonl_tail_probe.", dir=output) as probe_dir:
        probe = Path(probe_dir) / "rows.jsonl"
        atomic_jsonl(probe, [{"probe": 1}])
        with probe.open("a", encoding="utf-8") as handle:
            handle.write("{truncated")
        jsonl_tail_repair = load_jsonl(probe, repair_trailing=True) == [{"probe": 1}]
    before = {"clean": first["clean_forward_count"], "swap": first["swap_forward_count"]}
    second = _run_smoke_checked(args, smoke)
    after = {"clean": second["clean_forward_count"], "swap": second["swap_forward_count"]}
    smoke_summary = json.loads((smoke / "summary.json").read_text()) if (smoke / "summary.json").exists() else {}
    required_artifacts = [
        "run_config.json", "input_fingerprints.json", "recipient_manifest.jsonl", "donor_manifest.jsonl",
        "swap_pair_manifest.jsonl", "matching_diagnostics.json", "clean_predictions.jsonl",
        "swap_predictions.jsonl", "progress.json", "failures.jsonl", "swap_metrics.csv",
        "position_contrasts.csv", "bootstrap_results.csv", "donor_sensitivity.csv",
        "activation_diagnostics.csv", "activation_diagnostic_comparisons.csv",
        "normalized_answer_sensitivity.csv", "logit_change_diff_by_position.csv",
        "token_change_rate_by_position.csv", "summary.json", "summary.md",
        "soft_sa_swap_by_layer.png", "oriented_swap_effect_by_layer.png", "panl_vs_control_contrast.png",
        "supporting_metrics.png", "logit_change_diff_by_position.png", "token_change_rate_by_position.png",
    ]
    artifact_checks = {
        name: (smoke / name).is_file() and (name == "failures.jsonl" or (smoke / name).stat().st_size > 0)
        for name in required_artifacts
    }
    atomic_json(output / "artifact_gate.json", {"status": "passed" if all(artifact_checks.values()) else "failed",
                                                "checks": artifact_checks})
    checks = {
        "cpu_tests_passed": test_gate["status"] == "passed",
        "four_balanced_recipients": first["recipient_count"] == 4,
        "smoke_swap_count": first["swap_forward_count"] == 24,
        "resume_clean_count_unchanged": before["clean"] == after["clean"],
        "resume_swap_count_unchanged": before["swap"] == after["swap"],
        "resume_fingerprint_same": first["run_fingerprint"] == second["run_fingerprint"],
        "noop_bitwise": bool(json.loads((smoke / "smoke_gate.json").read_text()).get("all_noop_bitwise")),
        "analysis_complete": smoke_summary.get("status") == "complete",
        "artifact_gate": all(artifact_checks.values()),
        "jsonl_tail_repair": jsonl_tail_repair,
    }
    smoke_gate = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
                  "first_run": first, "resume_run": second}
    atomic_json(output / "smoke_gate.json", smoke_gate)
    if smoke_gate["status"] != "passed":
        raise RuntimeError(f"GPU smoke gate failed: {smoke_gate}")
    formal_output = output.parent / "formal_delayed_sa_swap_seed42"
    formal = [sys.executable, "-m", "dp_SA.activation_swap.run",
              "--source-root", str(args.source_root.resolve()), "--output-root", str(formal_output),
              "--positions", "P1_PANL,P1_PANL_PLUS_1,P1_SAC", "--layers", "12,14,16,18,22,26",
              "--bootstrap", "2000", "--seed", str(args.seed), "--max-recipients-per-side", "50"]
    command_text = " ".join(formal) + "\n" + " ".join([sys.executable, "-m", "dp_SA.activation_swap.analyze", "--output-root", str(formal_output), "--bootstrap", "2000", "--seed", str(args.seed)]) + "\n"
    (output / "formal_command.txt").write_text(command_text, encoding="utf-8")
    atomic_json(output / "pipeline_state.json", {"status": "smoke_complete", "stage": "awaiting_formal_run",
                                                 "formal_command": command_text, "smoke_gate": str(output / "smoke_gate.json")})
    print(command_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
