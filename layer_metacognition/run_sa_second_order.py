"""Run the gated second-order Source Attribution formation experiments."""

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
    load_baseline_rows,
    sha256_file,
)
from .sa_formation.second_order import (
    ProtocolAnalyzer,
    _protocol_context,
    _source_history_rows,
    build_answer_history_messages,
    generate_answer_messages,
    protocol_specs,
    register_runtime_artifacts,
    run_history_behavior,
    run_priming_decomposition,
    run_protocol_invariant_semantics,
    write_gate_controlled_skips,
    write_second_order_report,
)
from .sa_formation.runtime import Stage3Runtime


DEFAULT_EXPERIMENT = Path(__file__).resolve().parent / "output" / "Final_v4_run" / "answer_basis_9"


class TimeBudgetExceeded(RuntimeError):
    pass


class Deadline:
    def __init__(self, minutes: float) -> None:
        if minutes <= 0:
            raise ValueError("--max-minutes must be positive")
        self.started = time.monotonic()
        self.seconds = float(minutes) * 60.0

    def check(self) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed >= self.seconds:
            raise TimeBudgetExceeded(
                f"Second-order experiment budget exhausted after {elapsed / 60:.1f} minutes"
            )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    value.add_argument("--output-dir")
    value.add_argument(
        "--phase",
        choices=["all", "exp1", "exp2", "exp3", "report"],
        default="all",
    )
    value.add_argument("--semantic-items", type=int, default=30)
    value.add_argument("--max-minutes", type=float, default=180.0)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--smoke-only", action="store_true")
    return value


def _validate_output(root: Path, requested: str | None) -> Path:
    expected = (root / "stage3_sa_second_order").resolve()
    output = Path(requested).resolve() if requested else expected
    if output != expected:
        raise ValueError(f"Second-order output is fixed to {expected}; got {output}")
    protected = [
        root / "results.jsonl",
        root / "hidden_states",
        root / "stage1_metacognition",
        root / "stage3_sa_formation",
        root / "stage3_sa_formation_followup",
        root / "stage3_sa_mechanism",
        *root.glob("stage2_*"),
    ]
    if any(output == path.resolve() or path.resolve().is_relative_to(output) for path in protected):
        raise ValueError("Second-order output would contain a protected input artifact")
    return output


def _configuration(
    artifacts: SAFormationArtifacts, output: Path, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "stage3_sa_second_order_formation",
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
        "semantic_items": args.semantic_items,
        "history_source": "30 completed Relevant/Irrelevant History cases from stage3_sa_mechanism",
        "semantic_gate": "all cross-protocol decoder Spearman CI lower >0 plus positive shared PC1",
        "downstream_rule": "block tracing and subspace tests require a validated semantic target",
    }


def _provenance(artifacts: SAFormationArtifacts) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    mechanism = artifacts.experiment_dir / "stage3_sa_mechanism"
    stage3 = artifacts.experiment_dir / "stage3_sa_formation"
    implementation = [
        Path(__file__).resolve(),
        repository / "layer_metacognition" / "sa_formation" / "second_order.py",
        repository / "layer_metacognition" / "sa_formation" / "mechanism.py",
        repository / "layer_metacognition" / "sa_formation" / "runtime.py",
        repository / "confidence_test" / "joint_answer_source_extension.py",
        repository / "confidence_test" / "source_attribution_analyzer.py",
    ]
    return {
        "base_inputs": artifacts.provenance(),
        "mechanism_inputs": {
            "history_results": {
                "path": str(mechanism / "03_relevant_irrelevant_history" / "results.jsonl"),
                "sha256": sha256_file(
                    mechanism / "03_relevant_irrelevant_history" / "results.jsonl"
                ),
            },
            "history_summary": {
                "path": str(mechanism / "03_relevant_irrelevant_history" / "summary.json"),
                "sha256": sha256_file(
                    mechanism / "03_relevant_irrelevant_history" / "summary.json"
                ),
            },
            "normal_oof_decoder": {
                "path": str(stage3 / "directions" / "index.json"),
                "sha256": sha256_file(stage3 / "directions" / "index.json"),
            },
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
    mechanism_root: Path,
    stage3_root: Path,
) -> dict[str, Any]:
    source = _source_history_rows(mechanism_root)[0]
    baseline = {row["case_id"]: row for row in load_baseline_rows(artifacts)}
    target = runtime.case(source["item_id"], source["prior_index"])
    donor_id = source["donor_case_id"]
    donor_row = baseline[donor_id]
    donor = runtime.case(donor_row["item_id"], donor_row["prior_index"])
    messages = build_answer_history_messages(
        target,
        source["condition"],
        donor,
        donor_row["condition"],
        "image",
        source["prior_answer"],
    )
    answer = generate_answer_messages(runtime, messages, target.answer_classes)
    if not answer.parse_success or answer.answer_metric_status != "completed":
        raise RuntimeError(f"Answer-only smoke failed: {answer.error}")
    spec = protocol_specs()[-1]
    analyzer = ProtocolAnalyzer(runtime.generator.tokenizer, spec)
    row = baseline[source["case_id"]]
    prepared = _protocol_context(runtime, row, spec)
    direction = SAOOFDirectionRepository(stage3_root / "directions").get(row["fold"])
    measured = runtime.measure(prepared, direction, analyzer=analyzer)
    runtime.release_inputs(prepared)
    return {
        "status": "passed",
        "case_id": source["case_id"],
        "answer_only": answer.to_dict(),
        "answer_prompt_has_no_sa": True,
        "binary_protocol": {
            "labels": spec.labels_by_semantic,
            "token_ids": analyzer.encodings,
            "semantic_score": measured.source["soft_image_score"],
            "hook_exactly_once": measured.applied_count == 1,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.semantic_items < 3:
        raise ValueError("--semantic-items must be at least 3")
    deadline = Deadline(args.max_minutes)
    artifacts = SAFormationArtifacts.discover(args.experiment_dir)
    mechanism = artifacts.experiment_dir / "stage3_sa_mechanism"
    stage3 = artifacts.experiment_dir / "stage3_sa_formation"
    mechanism_progress = mechanism / "progress.json"
    if (
        not mechanism_progress.is_file()
        or json.loads(mechanism_progress.read_text()).get("status") != "complete"
    ):
        raise ValueError("Completed stage3_sa_mechanism input is required")
    output = _validate_output(artifacts.experiment_dir, args.output_dir)
    configuration = _configuration(artifacts, output, args)
    if args.dry_run:
        rows = _source_history_rows(mechanism)
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "cuda_available": torch.cuda.is_available(),
                    "history_cases": len(rows),
                    "semantic_items": args.semantic_items,
                    "output_dir": str(output),
                    "configuration": configuration,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    initialize_run(output, configuration, resume=args.resume)
    atomic_write_json(output / "provenance.json", _provenance(artifacts))
    atomic_write_json(output / "progress.json", {"status": "running", "phase": args.phase})
    if args.phase == "report":
        final = write_second_order_report(output)
        atomic_write_json(
            output / "progress.json",
            {
                "status": "complete",
                "final_analysis": str(output / "analysis" / "FINAL_ANALYSIS.md"),
                "summary": final,
            },
        )
        return 0
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
    register_runtime_artifacts(runtime, artifacts)
    smoke_path = output / "gpu_smoke.json"
    if not smoke_path.is_file():
        atomic_write_json(smoke_path, _gpu_smoke(runtime, artifacts, mechanism, stage3))
    if args.smoke_only:
        return 0
    if args.phase in {"all", "exp1"}:
        run_history_behavior(runtime, mechanism, output, deadline=deadline.check)
    if args.phase == "exp1":
        return 0
    if args.phase in {"all", "exp2"}:
        run_priming_decomposition(runtime, mechanism, output, deadline=deadline.check)
    if args.phase == "exp2":
        return 0
    if args.phase in {"all", "exp3"}:
        run_protocol_invariant_semantics(
            runtime,
            artifacts,
            stage3,
            output,
            n_items=args.semantic_items,
            deadline=deadline.check,
        )
    if args.phase == "exp3":
        return 0
    if args.phase == "all":
        gate = json.loads((output / "semantic_target_gate.json").read_text())
        if gate["passed"]:
            raise RuntimeError(
                "Semantic target gate passed; blockwise tracing implementation is required before continuing"
            )
        write_gate_controlled_skips(output)
        final = write_second_order_report(output)
        atomic_write_json(
            output / "progress.json",
            {
                "status": "complete",
                "final_analysis": str(output / "analysis" / "FINAL_ANALYSIS.md"),
                "summary": final,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
