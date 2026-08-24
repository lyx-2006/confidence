from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from dp_SA.io_utils import atomic_json, load_jsonl

from .analyze import analyze
from .config import (BOOTSTRAP_REPEATS, COARSE_WINDOWS, DEFAULT_DELAYED_SOURCE,
                     DEFAULT_JOINT_SOURCE, MAX_CASES_PER_SIDE, REFINE_Q_THRESHOLD,
                     REFINE_WINDOWS, SEED, SOFT_PARITY_TOLERANCE, parse_windows)
from .run import _default_output, run_experiment


KNOWN_UNRELATED_FAILURES = {
    "confidence_test/tests/test_four_version_evaluation.py::test_prompt_utils_is_byte_for_byte_unchanged",
    "layer_metacognition/tests/test_v3_v4_source.py::PersistenceAndResumeTests::test_answer_validation_appends_six_columns_to_compact_rows",
    "layer_metacognition/tests/test_v3_v4_source.py::PersistenceAndResumeTests::test_minimal_keeps_nulls_and_head_order",
    "layer_metacognition/tests/test_v3_v4_source.py::PersistenceAndResumeTests::test_six_column_semantic_value_is_null_when_not_collected",
}
SCOPED_TEST_PATHS = (
    "confidence_test/tests",
    "layer_metacognition/tests",
    "dp_SA/tests",
    "dp_SA/attention_block/tests",
)


def _test_gate(output: Path) -> None:
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *SCOPED_TEST_PATHS],
        cwd=Path(__file__).resolve().parents[2], env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    (output / "test_gate.log").write_text(completed.stdout, encoding="utf-8")
    failures = set(re.findall(r"^FAILED (\S+)", completed.stdout, flags=re.MULTILINE))
    errors = re.findall(r"^ERROR (\S+)", completed.stdout, flags=re.MULTILINE)
    gate = {
        "status": "passed" if failures == KNOWN_UNRELATED_FAILURES and not errors else "failed",
        "return_code": completed.returncode,
        "known_unrelated_failures": sorted(KNOWN_UNRELATED_FAILURES),
        "observed_failures": sorted(failures), "observed_errors": errors,
        "new_failures": sorted(failures - KNOWN_UNRELATED_FAILURES),
        "missing_known_failures": sorted(KNOWN_UNRELATED_FAILURES - failures),
        "summary_tail": completed.stdout.splitlines()[-1:] or [],
    }
    atomic_json(output / "test_gate.json", gate)
    if gate["status"] != "passed":
        raise RuntimeError(f"Incremental test gate failed: {gate}")


def _smoke_gate(output: Path, args: argparse.Namespace) -> None:
    smoke_output = output / "preflight_smoke"
    completion = smoke_output / "completion.json"
    if not completion.exists():
        run_experiment(
            output_dir=smoke_output, arm=args.arm, phase="coarse",
            joint_source=args.joint_source_dir, delayed_source=args.delayed_source_dir,
            bootstrap_repeats=args.bootstrap_repeats, seed=args.seed,
            max_cases_per_side=args.max_cases_per_side,
            resume=bool(args.resume or (smoke_output / "run_config.json").exists()), smoke=True,
            coarse_windows=args.coarse_windows, refine_windows=args.refine_windows,
            refine_q_threshold=args.refine_q_threshold, auto_refine=args.auto_refine,
        )
        analyze(smoke_output, repeats=args.bootstrap_repeats, seed=args.seed, final=True)
    clean = load_jsonl(smoke_output / "clean_baselines.jsonl")
    blocked = load_jsonl(smoke_output / "blocked_results.jsonl")
    failures = load_jsonl(smoke_output / "failures.jsonl")
    diagnostics = json.loads((smoke_output / "technical_diagnostics.json").read_text())
    expected_arms = 2 if args.arm == "both" else 1
    joint_clean = [row for row in clean if row["arm"] == "joint"]
    checks = {
        "clean_count": len(clean) == 10 * expected_arms,
        "blocked_count": len(blocked) == 210 * expected_arms,
        "no_failures": not failures,
        "blocked_weights_zero": diagnostics["all_blocked_weights_zero"],
        "attention_finite": diagnostics["all_attention_finite"],
        "all_28_query_heads": diagnostics["head_counts"] == [28],
        "joint_argmax_parity": all(row["parity_diagnostics"]["argmax_exact"] for row in joint_clean),
        "joint_soft_parity": all(
            row["parity_diagnostics"]["abs_soft_sa_difference"] <= SOFT_PARITY_TOLERANCE
            for row in joint_clean
        ),
    }
    gate = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "clean_forwards": len(clean), "blocked_forwards": len(blocked), "failure_count": len(failures)}
    atomic_json(output / "smoke_gate.json", gate)
    if gate["status"] != "passed":
        raise RuntimeError(f"GPU smoke gate failed: {gate}")


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--arm",choices=("joint","delayed","both"),default="both")
    parser.add_argument("--joint-source-dir",type=Path,default=DEFAULT_JOINT_SOURCE); parser.add_argument("--delayed-source-dir",type=Path,default=DEFAULT_DELAYED_SOURCE)
    parser.add_argument("--output-dir",type=Path,default=None); parser.add_argument("--bootstrap-repeats",type=int,default=BOOTSTRAP_REPEATS)
    parser.add_argument("--seed",type=int,default=SEED); parser.add_argument("--max-cases-per-side",type=int,default=MAX_CASES_PER_SIDE)
    parser.add_argument("--coarse-windows",type=parse_windows,default=COARSE_WINDOWS); parser.add_argument("--refine-windows",type=parse_windows,default=REFINE_WINDOWS)
    parser.add_argument("--refine-q-threshold",type=float,default=REFINE_Q_THRESHOLD); parser.add_argument("--no-auto-refine",dest="auto_refine",action="store_false")
    parser.add_argument("--resume",action="store_true"); parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(argv); output=(args.output_dir or _default_output()).resolve(); output.mkdir(parents=True,exist_ok=True)
    if not args.smoke:
        atomic_json(output/"pipeline_state.json",{"status":"running","stage":"test_gate"})
        _test_gate(output)
        atomic_json(output/"pipeline_state.json",{"status":"running","stage":"gpu_smoke"})
        _smoke_gate(output, args)
    atomic_json(output/"pipeline_state.json",{"status":"running","stage":"coarse"})
    run_experiment(output_dir=output,arm=args.arm,phase="coarse",joint_source=args.joint_source_dir,delayed_source=args.delayed_source_dir,
                   bootstrap_repeats=args.bootstrap_repeats,seed=args.seed,max_cases_per_side=args.max_cases_per_side,resume=args.resume,smoke=args.smoke,
                   coarse_windows=args.coarse_windows,refine_windows=args.refine_windows,
                   refine_q_threshold=args.refine_q_threshold,auto_refine=args.auto_refine)
    atomic_json(output/"pipeline_state.json",{"status":"running","stage":"coarse_analysis"})
    coarse=analyze(output,repeats=args.bootstrap_repeats,seed=args.seed,final=False)
    if coarse["selection"]["any_selected"] and args.auto_refine and not args.smoke:
        atomic_json(output/"pipeline_state.json",{"status":"running","stage":"refine","selected_pairs":coarse["selection"]["selected_pairs"]})
        run_experiment(output_dir=output,arm=args.arm,phase="refine",joint_source=args.joint_source_dir,delayed_source=args.delayed_source_dir,
                       bootstrap_repeats=args.bootstrap_repeats,seed=args.seed,max_cases_per_side=args.max_cases_per_side,resume=True,smoke=False,
                       selected_pairs_path=output/"refine_selection.json", coarse_windows=args.coarse_windows,
                       refine_windows=args.refine_windows, refine_q_threshold=args.refine_q_threshold, auto_refine=args.auto_refine)
    atomic_json(output/"pipeline_state.json",{"status":"running","stage":"final_analysis"})
    final=analyze(output,repeats=args.bootstrap_repeats,seed=args.seed,final=True)
    atomic_json(output/"pipeline_state.json",{"status":"complete","stage":"complete","selected_pairs":final["selection"]["selected_pairs"]})
    return 0


if __name__ == "__main__": raise SystemExit(main())
