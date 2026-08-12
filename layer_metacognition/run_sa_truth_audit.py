"""Run the behavior-grounded Source Attribution truth audit."""

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

from .sa_formation.core import (
    SAFormationArtifacts,
    SAOOFDirectionRepository,
    initialize_run,
    sha256_file,
)
from .sa_formation.truth_audit import (
    ANSWER_PROTOCOL_DIR,
    BEHAVIOR_DIR,
    FACTORIAL_DIR,
    GRANULARITY_DIR,
    MATCHED_GROUNDING_DIR,
    _common_protocol_context,
    canonical_leading_answer_tokens,
    common_protocol_specs,
    direct_fixed_answer_distribution,
    run_answer_only_protocol_robustness,
    run_counterfactual_source_use,
    run_history_factorial_reanalysis,
    run_matched_prompt_grounding,
    run_protocol_granularity_bridge,
    write_truth_audit_report,
    write_truth_gate_and_controlled_skips,
)
from .sa_formation.runtime import Stage3Runtime, full_prompt
from .sa_formation.second_order import ProtocolAnalyzer


DEFAULT_EXPERIMENT = (
    Path(__file__).resolve().parent
    / "output"
    / "Final_v4_run"
    / "answer_basis_9"
)


class TimeBudgetExceeded(RuntimeError):
    pass


class Deadline:
    def __init__(self, minutes: float) -> None:
        if minutes <= 0:
            raise ValueError("--max-minutes must be positive")
        self.started = time.monotonic()
        self.seconds = minutes * 60.0

    def check(self) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed >= self.seconds:
            raise TimeBudgetExceeded(
                f"Truth-audit budget exhausted after {elapsed / 60.0:.1f} minutes"
            )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    value.add_argument("--output-dir")
    value.add_argument(
        "--phase",
        choices=[
            "all",
            "cpu",
            "grounding",
            "answer_protocol",
            "granularity",
            "report",
        ],
        default="all",
    )
    value.add_argument("--grounding-items", type=int, default=100)
    value.add_argument("--bridge-items", type=int, default=80)
    value.add_argument("--max-minutes", type=float, default=180.0)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--smoke-only", action="store_true")
    return value


def _validate_output(experiment: Path, requested: str | None) -> Path:
    expected = (experiment / "stage3_sa_truth_audit").resolve()
    output = Path(requested).resolve() if requested else expected
    if output != expected:
        raise ValueError(f"Truth-audit output is fixed to {expected}; got {output}")
    protected = [
        experiment / "results.jsonl",
        experiment / "hidden_states",
        experiment / "stage1_metacognition",
        experiment / "stage3_sa_formation",
        experiment / "stage3_sa_formation_followup",
        experiment / "stage3_sa_mechanism",
        experiment / "stage3_sa_second_order",
        *experiment.glob("stage2_*"),
    ]
    if any(
        output == path.resolve() or path.resolve().is_relative_to(output)
        for path in protected
    ):
        raise ValueError("Truth-audit output would contain a protected input artifact")
    return output


def _configuration(
    artifacts: SAFormationArtifacts, output: Path, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "stage3_sa_behavior_grounded_truth_audit",
        "experiment_dir": str(artifacts.experiment_dir),
        "output_dir": str(output),
        "primary": {
            "version": "v4",
            "attribution_mode": "joint",
            "source_prompt_variant": "answer_basis_9",
            "position": "panl",
            "layer": 18,
            "seed": 42,
        },
        "counts": {
            "matched_grounding_unique_items": args.grounding_items,
            "protocol_bridge_unique_items": args.bridge_items,
            "history_factorial_items": 30,
        },
        "counterfactual_target": "fixed-answer deletion/replacement relative log support",
        "protocol_gate": "common-template 9/3/2 collapse, mapping, lexeme, grammar, decoder, and calibration",
        "causal_rule": "no blockwise/subspace intervention without behavior, verbal-alignment, and protocol gates",
    }


def _provenance(artifacts: SAFormationArtifacts) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    experiment = artifacts.experiment_dir
    implementation = [
        Path(__file__).resolve(),
        repository / "layer_metacognition" / "sa_formation" / "truth_audit.py",
        repository / "layer_metacognition" / "sa_formation" / "second_order.py",
        repository / "layer_metacognition" / "sa_formation" / "runtime.py",
        repository / "confidence_test" / "joint_answer_source_extension.py",
    ]
    source_paths = {
        "exact_factorial": experiment
        / "stage3_sa_formation_followup"
        / "02_history_exact_factorial"
        / "results_nocache.jsonl",
        "answer_only_at": experiment
        / "stage3_sa_second_order"
        / "01_history_behavior_dissociation"
        / "results.jsonl",
        "protocol_previous": experiment
        / "stage3_sa_second_order"
        / "03_protocol_invariant_semantic_sa"
        / "results.jsonl",
        "sa_direction": experiment / "stage3_sa_formation" / "directions" / "index.json",
    }
    return {
        "base_inputs": artifacts.provenance(),
        "source_inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "implementation": {
            str(path.relative_to(repository)): {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in implementation
        },
    }


def _gpu_smoke(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    output: Path,
) -> dict[str, Any]:
    behavior_rows = [
        row
        for row in __import__(
            "layer_metacognition.hidden_state_store", fromlist=["load_jsonl"]
        ).load_jsonl(output / BEHAVIOR_DIR / "results.jsonl")
        if row.get("status") == "completed"
    ]
    row = behavior_rows[0]
    case = runtime.case(row["item_id"], row["prior_index"])
    direct = direct_fixed_answer_distribution(
        runtime,
        prompt=full_prompt(case),
        image_path=str(case.conditions[row["condition"]].resolved_image_path),
        answer_classes=case.answer_classes,
        fixed_answer=row["final_answer"],
    )
    spec = common_protocol_specs()[2]
    analyzer = ProtocolAnalyzer(runtime.generator.tokenizer, spec)
    prepared = _common_protocol_context(runtime, row, spec)
    direction = SAOOFDirectionRepository(
        artifacts.experiment_dir / "stage3_sa_formation" / "directions"
    ).get(row["fold"])
    measured = runtime.measure(prepared, direction, analyzer=analyzer)
    runtime.release_inputs(prepared)
    return {
        "status": "passed",
        "case_id": row["case_id"],
        "canonical_answer_tokens": canonical_leading_answer_tokens(
            runtime.generator.tokenizer, case.answer_classes
        ),
        "direct_full": {
            "fixed_answer": direct["fixed_answer"],
            "predicted_answer": direct["predicted_answer"],
            "unique_top1": direct["unique_top1"],
        },
        "common_binary": {
            "score": measured.source["soft_image_score"],
            "labels": spec.labels_by_semantic,
            "hook_exactly_once": measured.applied_count == 1,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.grounding_items < 80:
        raise ValueError("--grounding-items must be at least 80")
    if args.bridge_items < 70:
        raise ValueError("--bridge-items must be at least 70")
    deadline = Deadline(args.max_minutes)
    artifacts = SAFormationArtifacts.discover(args.experiment_dir)
    experiment = artifacts.experiment_dir
    second_progress = experiment / "stage3_sa_second_order" / "progress.json"
    if (
        not second_progress.is_file()
        or json.loads(second_progress.read_text()).get("status") != "complete"
    ):
        raise ValueError("Completed stage3_sa_second_order input is required")
    output = _validate_output(experiment, args.output_dir)
    configuration = _configuration(artifacts, output, args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "cuda_available": torch.cuda.is_available(),
                    "output_dir": str(output),
                    "configuration": configuration,
                },
                indent=2,
            )
        )
        return 0
    initialize_run(output, configuration, resume=args.resume)
    atomic_write_json(output / "provenance.json", _provenance(artifacts))
    atomic_write_json(output / "progress.json", {"status": "running", "phase": args.phase})
    if args.phase == "report":
        final = write_truth_audit_report(output)
        atomic_write_json(
            output / "progress.json",
            {
                "status": "complete",
                "final_analysis": str(output / "analysis" / "FINAL_ANALYSIS.md"),
                "summary": final,
            },
        )
        return 0
    if args.phase in {"all", "cpu"}:
        run_counterfactual_source_use(artifacts, output)
        run_history_factorial_reanalysis(experiment, output)
    if args.phase == "cpu":
        return 0
    if not (output / BEHAVIOR_DIR / "summary.json").is_file():
        raise ValueError("CPU counterfactual source-use phase must complete first")
    if not torch.cuda.is_available():
        atomic_write_json(
            output / "gpu_skipped.json",
            {
                "status": "gpu_skipped",
                "reason": "torch.cuda.is_available() is false",
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
        )
        return 0
    runtime = Stage3Runtime(artifacts)
    smoke_path = output / "gpu_smoke.json"
    if not smoke_path.is_file():
        atomic_write_json(smoke_path, _gpu_smoke(runtime, artifacts, output))
    if args.smoke_only:
        return 0
    if args.phase in {"all", "grounding"}:
        run_matched_prompt_grounding(
            runtime,
            experiment,
            output,
            n_items=args.grounding_items,
            deadline=deadline.check,
        )
    if args.phase == "grounding":
        return 0
    if args.phase in {"all", "answer_protocol"}:
        run_answer_only_protocol_robustness(
            runtime, experiment, output, deadline=deadline.check
        )
    if args.phase == "answer_protocol":
        return 0
    if args.phase in {"all", "granularity"}:
        run_protocol_granularity_bridge(
            runtime,
            experiment,
            output,
            n_items=args.bridge_items,
            deadline=deadline.check,
        )
    if args.phase == "granularity":
        return 0
    if args.phase == "all":
        gate = write_truth_gate_and_controlled_skips(output)
        final = write_truth_audit_report(output)
        atomic_write_json(
            output / "progress.json",
            {
                "status": "complete",
                "grounded_gate": gate,
                "final_analysis": str(output / "analysis" / "FINAL_ANALYSIS.md"),
                "summary": final,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
