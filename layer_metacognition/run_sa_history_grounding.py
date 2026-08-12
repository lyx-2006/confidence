"""Run History-conditioned fixed-answer Source Attribution grounding."""

from __future__ import annotations

import argparse
import json
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
from .sa_formation.history_grounding import (
    HISTORY_GROUNDING_DIR,
    build_history_perturbation_messages,
    direct_messages_fixed_answer_distribution,
    run_history_grounding,
    select_history_grounding_cohort,
)
from .sa_formation.runtime import Stage3Runtime


DEFAULT_EXPERIMENT = (
    Path(__file__).resolve().parent
    / "output"
    / "Final_v4_run"
    / "answer_basis_9"
)


class Deadline:
    def __init__(self, minutes: float) -> None:
        if minutes <= 0:
            raise ValueError("--max-minutes must be positive")
        self.started = time.monotonic()
        self.seconds = minutes * 60.0

    def check(self) -> None:
        if time.monotonic() - self.started >= self.seconds:
            raise RuntimeError("History-grounding time budget exhausted")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    value.add_argument("--output-dir")
    value.add_argument("--max-minutes", type=float, default=60.0)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--smoke-only", action="store_true")
    return value


def _configuration(experiment: Path, output: Path) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "history_conditioned_fixed_answer_source_sensitivity",
        "experiment_dir": str(experiment),
        "output_dir": str(output),
        "cohort_n": 20,
        "protocols": ["joint_report", "answer_only"],
        "histories": ["text_first", "image_first"],
        "conditions": ["full", "no_text", "no_image", "replace_text", "replace_image"],
        "primary": "answer-only fixed-answer source sensitivity",
        "seed": 42,
    }


def _provenance(experiment: Path) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    sources = {
        "mechanism_history": experiment / "stage3_sa_mechanism" / "03_relevant_irrelevant_history" / "results.jsonl",
        "answer_only_history": experiment / "stage3_sa_second_order" / "01_history_behavior_dissociation" / "results.jsonl",
        "behavior_rows": experiment / "stage3_sa_truth_audit" / "01_counterfactual_source_use" / "results.jsonl",
    }
    implementation = [
        Path(__file__).resolve(),
        repository / "layer_metacognition" / "sa_formation" / "history_grounding.py",
        repository / "layer_metacognition" / "sa_formation" / "runtime.py",
    ]
    return {
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "implementation": {
            str(path.relative_to(repository)): sha256_file(path)
            for path in implementation
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    experiment = Path(args.experiment_dir).resolve()
    truth_root = experiment / "stage3_sa_truth_audit"
    expected = (truth_root / HISTORY_GROUNDING_DIR).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else expected
    if output != expected:
        raise ValueError(f"Output is fixed to {expected}; got {output}")
    cohort = select_history_grounding_cohort(experiment, truth_root)
    configuration = _configuration(experiment, output)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "cuda_available": torch.cuda.is_available(),
                    "cohort_n": len(cohort),
                    "output_dir": str(output),
                    "configuration": configuration,
                },
                indent=2,
            )
        )
        return 0
    initialize_run(output, configuration, resume=args.resume)
    atomic_write_json(output / "provenance.json", _provenance(experiment))
    atomic_write_json(output / "progress.json", {"status": "running"})
    if not torch.cuda.is_available():
        atomic_write_json(
            output / "gpu_skipped.json",
            {"status": "gpu_skipped", "reason": "torch.cuda.is_available() is false"},
        )
        return 0
    artifacts = SAFormationArtifacts.discover(experiment)
    runtime = Stage3Runtime(artifacts)
    smoke_path = output / "gpu_smoke.json"
    if not smoke_path.is_file():
        row = cohort[0]
        case = runtime.case(row["item_id"], row["prior_index"])
        smoke: dict[str, Any] = {"status": "passed", "case_id": row["case_id"], "protocols": {}}
        for protocol in ("joint_report", "answer_only"):
            messages = build_history_perturbation_messages(
                protocol=protocol,
                target_case=case,
                target_condition=row["condition"],
                modality="text",
                prior_answer=row["prior_answer"],
                text_clue=case.text_clue,
                image_path=str(case.conditions[row["condition"]].resolved_image_path),
            )
            measured = direct_messages_fixed_answer_distribution(
                runtime,
                messages,
                answer_classes=case.answer_classes,
                fixed_answer=row["fixed_answer"],
            )
            expected_hash = row[
                "expected_joint_hashes" if protocol == "joint_report" else "expected_answer_only_hashes"
            ]["text_first"]
            if measured["messages_hash"] != expected_hash:
                raise ValueError(f"Smoke full-message hash mismatch for {protocol}")
            smoke["protocols"][protocol] = {
                "messages_hash_matches": True,
                "predicted_answer": measured["predicted_answer"],
                "fixed_answer": measured["fixed_answer"],
                "unique_top1": measured["unique_top1"],
                "canonical_tokens": measured["canonical_leading_token_ids"],
            }
        atomic_write_json(smoke_path, smoke)
    if args.smoke_only:
        return 0
    summary = run_history_grounding(
        runtime,
        experiment,
        truth_root,
        deadline=Deadline(args.max_minutes).check,
    )
    atomic_write_json(
        output / "progress.json",
        {
            "status": "complete",
            "summary": summary,
            "final_analysis": str(output / "FINAL_ANALYSIS.md"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

