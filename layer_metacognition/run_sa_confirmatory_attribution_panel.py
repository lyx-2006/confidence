"""Run the frozen 76-item confirmatory Source Attribution protocol panel."""

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

from .sa_formation.confirmatory_attribution_panel import (
    ALL_PROTOCOL_NAMES,
    EXPECTED_COMPLETED_ITEMS,
    JOINT_PROTOCOL_NAMES,
    PANEL_DIR,
    POSTQUERY_PROTOCOL_NAME,
    analyze_confirmatory_panel,
    build_cohort_manifest,
    build_endpoint_audit,
    freeze_stage10_rule,
    gpu_smoke,
    immutable_json,
    inspect_stage10_rule,
    load_confirmatory_cohort,
    method_v2_root,
    panel_root,
    protocol_freeze_payload,
    run_confirmatory_panel,
    stage10_root,
)
from .sa_formation.core import SAFormationArtifacts, initialize_run, sha256_file
from .sa_formation.runtime import Stage3Runtime


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
        self.seconds = float(minutes) * 60.0

    def check(self) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed >= self.seconds:
            raise TimeBudgetExceeded(
                f"Confirmatory attribution budget exhausted after {elapsed / 60.0:.1f} minutes"
            )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    value.add_argument("--output-dir")
    value.add_argument("--max-minutes", type=float, default=60.0)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--smoke-only", action="store_true")
    value.add_argument("--analyze-only", action="store_true")
    return value


def validate_output(experiment_dir: str | Path, requested: str | None = None) -> Path:
    experiment = Path(experiment_dir).resolve()
    expected = panel_root(experiment).resolve()
    output = Path(requested).resolve() if requested else expected
    if output != expected:
        raise ValueError(f"Confirmatory attribution output is fixed to {expected}; got {output}")
    protected = [
        experiment / "results.jsonl",
        experiment / "hidden_states",
        experiment / "stage1_metacognition",
        experiment / "stage3_sa_formation",
        experiment / "stage3_sa_formation_followup",
        experiment / "stage3_sa_mechanism",
        experiment / "stage3_sa_second_order",
        experiment / "stage3_sa_truth_audit",
        *[
            path
            for path in (experiment / "stage3_sa_computational_bridge").glob("*")
            if path.name != PANEL_DIR
        ],
        *experiment.glob("stage2_*"),
    ]
    if any(
        output == path.resolve() or path.resolve().is_relative_to(output)
        for path in protected
    ):
        raise ValueError("Confirmatory output overlaps a protected Stage 1/2/3 artifact")
    return output


def configuration(
    artifacts: SAFormationArtifacts,
    output: Path,
    source_rule: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "frozen_confirmatory_attribution_panel",
        "experiment_dir": str(artifacts.experiment_dir),
        "output_dir": str(output),
        "primary": {
            "version": "v4",
            "attribution_mode": "joint",
            "source_prompt_variant": "answer_basis_9",
            "position": "PANL",
            "layer": 18,
            "seed": 42,
        },
        "cohort": {
            "split": "method-v2 confirmatory",
            "completed_n": EXPECTED_COMPLETED_ITEMS,
            "endpoint": "answer_star",
            "stage10_development_item_overlap": 0,
        },
        "protocols": protocol_freeze_payload(),
        "joint_protocol_order": list(JOINT_PROTOCOL_NAMES),
        "postquery_protocol": POSTQUERY_PROTOCOL_NAME,
        "forwards_per_item": len(ALL_PROTOCOL_NAMES),
        "formal_forward_count": EXPECTED_COMPLETED_ITEMS * len(ALL_PROTOCOL_NAMES),
        "frozen_source": {
            "stage10": str(stage10_root(artifacts.experiment_dir)),
            "summary_sha256": source_rule["source_summary_sha256"],
            "direction_index_sha256": source_rule["source_direction_index_sha256"],
        },
        "causal_intervention": False,
        "causal_mediator_authorized": False,
    }


def provenance(artifacts: SAFormationArtifacts) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    stage10 = stage10_root(artifacts.experiment_dir)
    method_v2 = method_v2_root(artifacts.experiment_dir)
    direction_index = json.loads(
        (stage10 / "directions" / "index.json").read_text(encoding="utf-8")
    )
    sources = {
        "method_v2_confirmatory_results": method_v2 / "confirmatory_results.jsonl",
        "method_v2_confirmatory_manifest": method_v2
        / "confirmatory_cohort_manifest.json",
        "method_v2_frozen_rule": method_v2 / "frozen_measurement_rule.json",
        "stage10_summary": stage10 / "summary.json",
        "stage10_cohort": stage10 / "cohort_manifest.json",
        "stage10_direction_index": stage10 / "directions" / "index.json",
        "item_split": artifacts.item_split,
        "dataset": artifacts.dataset,
    }
    sources.update(
        {
            f"stage10_direction_fold_{entry['fold']}": stage10
            / "directions"
            / str(entry["file"])
            for entry in direction_index["folds"]
        }
    )
    implementation = [
        Path(__file__).resolve(),
        repository
        / "layer_metacognition"
        / "sa_formation"
        / "confirmatory_attribution_panel.py",
        repository / "layer_metacognition" / "sa_formation" / "attribution_component.py",
        repository / "layer_metacognition" / "sa_formation" / "runtime.py",
        repository / "layer_metacognition" / "sa_formation" / "reliance_measurement.py",
        repository / "layer_metacognition" / "sa_formation" / "truth_audit.py",
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
                "bytes": path.stat().st_size,
            }
            for path in implementation
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }


def verify_analysis_rerun_provenance(
    output: str | Path, current: dict[str, Any]
) -> dict[str, Any]:
    """Permit CPU re-analysis drift while keeping every scientific input fixed.

    Formal-forward resume still requires byte-identical implementation through
    ``immutable_json``.  Analyze-only is different: it records the new analysis
    implementation and verifies that all base/source inputs remain identical to
    the formal run before allowing summaries to be regenerated.
    """

    directory = Path(output).resolve()
    original_path = directory / "provenance.json"
    if not original_path.is_file():
        raise FileNotFoundError(f"Formal-run provenance is missing: {original_path}")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for section in ("base_inputs", "source_inputs"):
        if original.get(section) != current.get(section):
            mismatches.append(section)
    if mismatches:
        raise ValueError(
            "Analyze-only scientific input provenance drift: " + ", ".join(mismatches)
        )
    audit = {
        "format_version": 1,
        "mode": "analyze_only",
        "scientific_inputs_identical": True,
        "verified_sections": ["base_inputs", "source_inputs"],
        "formal_provenance_path": str(original_path),
        "formal_provenance_sha256": sha256_file(original_path),
        "formal_implementation": original["implementation"],
        "analysis_implementation": current["implementation"],
        "implementation_identical": bool(
            original["implementation"] == current["implementation"]
        ),
        "environment": current["environment"],
    }
    atomic_write_json(directory / "analysis_rerun_provenance.json", audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.smoke_only and args.analyze_only:
        raise ValueError("--smoke-only and --analyze-only are mutually exclusive")
    artifacts = SAFormationArtifacts.discover(args.experiment_dir)
    output = validate_output(artifacts.experiment_dir, args.output_dir)
    rows, endpoint_preflight = load_confirmatory_cohort(artifacts.experiment_dir)
    cohort = build_cohort_manifest(rows, endpoint_preflight)
    source_rule = inspect_stage10_rule(artifacts.experiment_dir)
    config = configuration(artifacts, output, source_rule)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "cuda_available": torch.cuda.is_available(),
                    "output_dir": str(output),
                    "cohort_n": len(rows),
                    "unique_items": len({row["item_id"] for row in rows}),
                    "stage10_item_overlap": endpoint_preflight["stage10_item_overlap"],
                    "answer_field": endpoint_preflight["answer_field"],
                    "joint_protocols": list(JOINT_PROTOCOL_NAMES),
                    "postquery_protocol": POSTQUERY_PROTOCOL_NAME,
                    "forwards_per_item": len(ALL_PROTOCOL_NAMES),
                    "formal_forward_count": len(rows) * len(ALL_PROTOCOL_NAMES),
                    "smoke_forward_count": len(ALL_PROTOCOL_NAMES),
                    "frozen_source_direction_index_sha256": source_rule[
                        "source_direction_index_sha256"
                    ],
                    "configuration": config,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.analyze_only and not (output / "results.jsonl").is_file():
        raise FileNotFoundError(
            f"--analyze-only requires formal results: {output / 'results.jsonl'}"
        )

    initialize_run(output, config, resume=args.resume or args.analyze_only)
    freeze_stage10_rule(artifacts.experiment_dir, output)
    immutable_json(output / "cohort_manifest.json", cohort)
    current_provenance = provenance(artifacts)
    if args.analyze_only:
        verify_analysis_rerun_provenance(output, current_provenance)
    else:
        immutable_json(output / "provenance.json", current_provenance)
    if not (output / "endpoint_audit.json").exists():
        atomic_write_json(output / "endpoint_audit.json", build_endpoint_audit(cohort))

    if args.analyze_only:
        summary = analyze_confirmatory_panel(output)
        atomic_write_json(
            output / "progress.json",
            {
                "status": "complete"
                if summary["technical_gate"]["passed"]
                else "technical_gate_failed",
                "analyze_only": True,
                "summary": str(output / "summary.json"),
                "artifact_aggregate_sha256": summary["artifact_aggregate_sha256"],
            },
        )
        return 0

    atomic_write_json(
        output / "progress.json",
        {
            "status": "running",
            "completed_n": 0,
            "expected_n": len(rows),
            "formal_forward_count_expected": len(rows) * len(ALL_PROTOCOL_NAMES),
        },
    )
    if not torch.cuda.is_available():
        atomic_write_json(
            output / "gpu_skipped.json",
            {
                "status": "gpu_skipped",
                "reason": "torch.cuda.is_available() is false",
                "formal_forward_count": 0,
            },
        )
        atomic_write_json(
            output / "progress.json",
            {"status": "gpu_skipped", "completed_n": 0, "expected_n": len(rows)},
        )
        return 0

    deadline = Deadline(args.max_minutes)
    runtime = Stage3Runtime(artifacts)
    smoke_path = output / "gpu_smoke.json"
    if not smoke_path.is_file():
        atomic_write_json(
            smoke_path,
            gpu_smoke(runtime, rows[0], output, deadline=deadline.check),
        )
    if args.smoke_only:
        atomic_write_json(
            output / "progress.json",
            {
                "status": "smoke_complete",
                "completed_n": 0,
                "expected_n": len(rows),
                "smoke": str(smoke_path),
            },
        )
        return 0
    try:
        run_state = run_confirmatory_panel(
            runtime, rows, output, deadline=deadline.check
        )
    except TimeBudgetExceeded as exc:
        terminal = [
            row
            for row in __import__(
                "layer_metacognition.hidden_state_store", fromlist=["load_jsonl"]
            ).load_jsonl(output / "results.jsonl", repair_trailing=True)
            if row.get("status") == "completed"
        ]
        atomic_write_json(
            output / "progress.json",
            {
                "status": "budget_exhausted",
                "completed_n": len(terminal),
                "expected_n": len(rows),
                "reason": str(exc),
                "resume_command": "rerun the identical command with --resume",
            },
        )
        return 0
    summary = analyze_confirmatory_panel(output)
    atomic_write_json(
        output / "progress.json",
        {
            "status": "complete"
            if summary["technical_gate"]["passed"]
            else "technical_gate_failed",
            **run_state,
            "summary": str(output / "summary.json"),
            "artifact_aggregate_sha256": summary["artifact_aggregate_sha256"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
