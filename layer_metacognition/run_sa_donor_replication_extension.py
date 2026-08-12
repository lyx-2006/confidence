"""Run the prospective donor-3/4 replication extension."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "layer_metacognition"

from layer_metacognition.hidden_state_store import atomic_write_json

from .sa_formation.core import SAFormationArtifacts, initialize_run, sha256_file
from .sa_formation.donor_replication_extension import (
    EXTENSION_DIR,
    EXTENSION_METHOD_VERSION,
    NEW_CONDITIONS,
    build_extension_manifest,
    build_extension_plan,
    extension_root,
    measure_extension_case,
    method_v2_root,
    run_extension_panel,
    verify_development_allowed,
)
from .sa_formation.reliance_measurement import BRIDGE_DIR, RELIANCE_DIR
from .sa_formation.runtime import Stage3Runtime


DEFAULT_EXPERIMENT = (
    Path(__file__).resolve().parent
    / "output"
    / "Final_v4_run"
    / "answer_basis_9"
)


class Deadline:
    def __init__(self, minutes: float) -> None:
        if minutes <= 0.0:
            raise ValueError("--max-minutes must be positive")
        self.started = time.monotonic()
        self.seconds = float(minutes) * 60.0

    def check(self) -> None:
        if time.monotonic() - self.started >= self.seconds:
            raise RuntimeError("Donor-replication extension time budget exhausted")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    value.add_argument("--output-dir")
    value.add_argument(
        "--split",
        choices=["confirmatory", "development"],
        default="confirmatory",
    )
    value.add_argument("--max-minutes", type=float, default=60.0)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--smoke-only", action="store_true")
    return value


def validate_output(experiment: str | Path, requested: str | None = None) -> Path:
    root = Path(experiment).resolve()
    expected = extension_root(root).resolve()
    output = Path(requested).resolve() if requested else expected
    if output != expected:
        raise ValueError(f"Donor extension output is fixed to {expected}; got {output}")
    protected = [
        root / "results.jsonl",
        root / "hidden_states",
        root / "stage1_metacognition",
        root / "stage3_sa_formation",
        root / "stage3_sa_formation_followup",
        root / "stage3_sa_mechanism",
        root / "stage3_sa_second_order",
        root / "stage3_sa_truth_audit",
        root / BRIDGE_DIR / RELIANCE_DIR,
        *root.glob("stage2_*"),
    ]
    if any(
        output == path.resolve() or path.resolve().is_relative_to(output)
        for path in protected
    ):
        raise ValueError("Donor extension output overlaps a protected input artifact")
    return output


def configuration(artifacts: SAFormationArtifacts, output: Path) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "prospective_donor_replication_extension",
        "extension_method_version": EXTENSION_METHOD_VERSION,
        "experiment_dir": str(artifacts.experiment_dir),
        "output_dir": str(output),
        "method_v2_input": str(method_v2_root(artifacts.experiment_dir)),
        "prompt": "exact method-v2 V4 answer-only reconstruction; no History or SA request",
        "answer_star": "reuse completed method-v2 A* without a new Full forward",
        "conditions": list(NEW_CONDITIONS),
        "forwards_per_item": 4,
        "hidden_capture": False,
        "donor_selection": (
            "next two distinct donors under method-v2 rank; same "
            "fold/difficulty/final-side; target and d1-d4 distinct"
        ),
        "gate": {
            "technical": "n>=70 confirmatory (>=90 development), no failures or audit drift",
            "split_half": "M12-vs-M34 bootstrap lower>0, sign>=.70, ICC>=.60",
            "raw": "D-vs-M34 bootstrap lower>0, sign>=.70, alpha>=.60, >=4/5 positive folds",
            "graded": "secondary only",
        },
        "bootstrap": {
            "primary": "target-item cluster",
            "sensitivity": [
                "multi-membership donor-cluster",
                "leave-one-donor-out",
            ],
        },
        "claim_scope": (
            "post-confirmatory choice-coupled intervention replication on existing items; "
            "not a reversal of method-v2 and not new-item confirmation"
        ),
        "seed": 42,
    }


def provenance(artifacts: SAFormationArtifacts) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    old = method_v2_root(artifacts.experiment_dir)
    sources = {
        "method_v2_development_analysis": old / "development_analysis.jsonl",
        "method_v2_confirmatory_analysis": old / "confirmatory_analysis.jsonl",
        "method_v2_development_summary": old / "development_summary.json",
        "method_v2_confirmatory_summary": old / "confirmatory_summary.json",
        "method_v2_frozen_rule": old / "frozen_measurement_rule.json",
        "truth_behavior_pool": artifacts.experiment_dir
        / "stage3_sa_truth_audit"
        / "01_counterfactual_source_use"
        / "results.jsonl",
        "item_split": artifacts.item_split,
        "dataset": artifacts.dataset,
    }
    implementation = [
        Path(__file__).resolve(),
        repository
        / "layer_metacognition"
        / "sa_formation"
        / "donor_replication_extension.py",
        repository
        / "layer_metacognition"
        / "sa_formation"
        / "reliance_measurement.py",
    ]
    return {
        "base_inputs": artifacts.provenance(),
        "source_inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in sources.items()
        },
        "implementation": {
            str(path.relative_to(repository)): {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in implementation
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }


def gpu_smoke(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    split: str,
) -> dict[str, Any]:
    cohort, method_v2, donors, cases = build_extension_plan(artifacts, split)
    target = cohort[0]
    case_id = str(target["case_id"])
    measured = measure_extension_case(
        runtime,
        target,
        method_v2[case_id],
        donors[case_id],
        cases,
    )
    if tuple(measured["measurements"]) != NEW_CONDITIONS:
        raise RuntimeError("Smoke did not execute exactly the four extension conditions")
    if measured["hidden_captured"]:
        raise RuntimeError("Smoke unexpectedly captured hidden state")
    return {
        "status": "passed",
        "split": split,
        "case_id": case_id,
        "answer_star": measured["answer_star"],
        "conditions": list(measured["measurements"]),
        "answer_star_reused": measured["answer_star_reused"],
        "full_messages_hash_equal": measured["full_messages_hash_equal"],
        "selection_reused_without_forward": measured[
            "selection_reused_without_forward"
        ],
        "verbal_sa_leakage": measured["verbal_sa_leakage"],
        "hidden_captured": measured["hidden_captured"],
        "donor3_item_id": donors[case_id][0]["item_id"],
        "donor4_item_id": donors[case_id][1]["item_id"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    artifacts = SAFormationArtifacts.discover(args.experiment_dir)
    output = validate_output(artifacts.experiment_dir, args.output_dir)
    cohort, method_v2, donors, _ = build_extension_plan(artifacts, args.split)
    method_v2_path = method_v2_root(artifacts.experiment_dir) / f"{args.split}_analysis.jsonl"
    manifest = build_extension_manifest(
        args.split,
        cohort,
        method_v2,
        donors,
        method_v2_sha256=sha256_file(method_v2_path),
    )
    config = configuration(artifacts, output)
    development_state: dict[str, Any] | None = None
    if args.split == "development" and output.exists():
        rule = output / "frozen_extension_rule.json"
        if rule.is_file():
            value = json.loads(rule.read_text(encoding="utf-8"))
            development_state = {
                "rule_fingerprint": value.get("rule_fingerprint"),
                "development_allowed": value.get("development_allowed"),
            }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "split": args.split,
                    "cuda_available": torch.cuda.is_available(),
                    "output_dir": str(output),
                    "cohort_n": len(cohort),
                    "new_forward_count": 4 * len(cohort),
                    "hidden_capture": False,
                    "manifest_fingerprint": manifest["manifest_fingerprint"],
                    "development_state": development_state,
                    "configuration": config,
                },
                indent=2,
            )
        )
        return 0

    initialize_run(output, config, resume=args.resume)
    atomic_write_json(output / "provenance.json", provenance(artifacts))
    atomic_write_json(output / "progress.json", {"status": "running", "split": args.split})
    if args.split == "development":
        verify_development_allowed(output)
    if not torch.cuda.is_available():
        atomic_write_json(
            output / "gpu_skipped.json",
            {
                "status": "gpu_skipped",
                "reason": "torch.cuda.is_available() is false",
                "split": args.split,
            },
        )
        return 0

    runtime = Stage3Runtime(artifacts)
    smoke_path = output / f"gpu_smoke_{args.split}.json"
    if not smoke_path.is_file():
        atomic_write_json(smoke_path, gpu_smoke(runtime, artifacts, args.split))
    if args.smoke_only:
        atomic_write_json(
            output / "progress.json",
            {"status": "smoke_complete", "split": args.split},
        )
        return 0
    summary = run_extension_panel(
        runtime,
        artifacts,
        output,
        split=args.split,
        deadline=Deadline(args.max_minutes).check,
    )
    atomic_write_json(
        output / "progress.json",
        {
            "status": "complete",
            "split": args.split,
            "extension_gate_passed": summary["extension_gate_passed"],
            "summary": str(output / f"{args.split}_summary.json"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

