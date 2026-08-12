"""Run the three core Stage 3 Source Attribution follow-up experiments."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from layer_metacognition.hidden_state_store import atomic_write_json

from .sa_formation.core import (
    SAFormationArtifacts,
    SAOOFDirectionRepository,
    initialize_run,
    load_baseline_rows,
    sha256_file,
)
from .sa_formation.followup import (
    DIRECTION_DIR,
    EVIDENCE_DIR,
    HISTORY_DIR,
    _direction_eval_context,
    _run_history_branch,
    fit_oof_old_mean_directions,
    run_direction_comparison,
    run_evidence_reanalysis,
    run_history_factorial,
    select_history_factorial_cohort,
    write_followup_report,
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
                f"Follow-up wall-clock budget exhausted after {elapsed / 60:.1f} minutes"
            )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    value.add_argument("--output-dir")
    value.add_argument(
        "--phase",
        choices=["all", "evidence", "history", "directions", "report"],
        default="all",
    )
    value.add_argument("--history-items", type=int, default=30)
    value.add_argument("--direction-items", type=int, default=30)
    value.add_argument("--max-minutes", type=float, default=120.0)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--smoke-only", action="store_true")
    return value


def _validate_output(root: Path, requested: str | None) -> Path:
    expected = root / "stage3_sa_formation_followup"
    output = Path(requested).resolve() if requested else expected.resolve()
    if output != expected.resolve():
        raise ValueError(f"Follow-up output is fixed to {expected}; got {output}")
    protected = [root / "results.jsonl", root / "hidden_states"] + list(root.glob("stage[12]_*")) + [root / "stage3_sa_formation"]
    if any(output == path.resolve() or path.resolve().is_relative_to(output) for path in protected):
        raise ValueError("Follow-up output would contain a protected artifact")
    return output


def _configuration(
    artifacts: SAFormationArtifacts,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "stage3_sa_formation_core_followup",
        "experiment_dir": str(artifacts.experiment_dir),
        "stage3_input": str(artifacts.experiment_dir / "stage3_sa_formation"),
        "output_dir": str(output),
        "primary": {
            "version": "v4",
            "attribution_mode": "joint",
            "source_prompt_variant": "answer_basis_9",
            "position": "panl",
            "layer": 18,
            "seed": 42,
        },
        "history_items": args.history_items,
        "history_branches": ["text_at", "text_ai", "image_at", "image_ai", "no_history"],
        "direction_items": args.direction_items,
        "direction_doses_sigma_units": [-2, -1, 0, 1, 2],
        "old_direction": "25 strong SA cases per side, rebuilt item-OOF per fold",
    }


def _provenance(artifacts: SAFormationArtifacts) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    implementation = [
        repository / "confidence_test" / "joint_answer_source_extension.py",
        repository / "layer_metacognition" / "sa_formation" / "runtime.py",
        repository / "layer_metacognition" / "sa_formation" / "followup.py",
        Path(__file__).resolve(),
    ]
    stage3 = artifacts.experiment_dir / "stage3_sa_formation"
    return {
        "base_inputs": artifacts.provenance(),
        "stage3_inputs": {
            "oof_predictions": {
                "path": str(stage3 / "00_natural_state" / "oof_predictions.jsonl"),
                "sha256": sha256_file(stage3 / "00_natural_state" / "oof_predictions.jsonl"),
            },
            "ridge_direction_index": {
                "path": str(stage3 / "directions" / "index.json"),
                "sha256": sha256_file(stage3 / "directions" / "index.json"),
            },
        },
        "implementation": {
            str(path.relative_to(repository)): {"path": str(path), "sha256": sha256_file(path)}
            for path in implementation
        },
    }


def _gpu_smoke(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    output: Path,
) -> dict[str, Any]:
    stage3 = artifacts.experiment_dir / "stage3_sa_formation"
    oof_path = stage3 / "00_natural_state" / "oof_predictions.jsonl"
    from layer_metacognition.hidden_state_store import load_jsonl

    row = select_history_factorial_cohort(load_jsonl(oof_path), 1)[0]
    ridge = SAOOFDirectionRepository(stage3 / "directions").get(row["fold"])
    branch = _run_history_branch(
        runtime,
        row,
        "text_at",
        ridge,
        generation_use_cache=False,
    )
    reconstruction = branch["reconstruction"]
    if not reconstruction["raw_logit_within_bf16_tolerance"] or not reconstruction["soft_within_0.01"]:
        raise RuntimeError(
            "Exact-token History smoke exceeds reconstruction tolerance: "
            f"{reconstruction}"
        )
    old_repo, _, _ = fit_oof_old_mean_directions(artifacts, stage3, output)
    baseline = next(
        value
        for value in load_baseline_rows(artifacts)
        if value["case_id"] == row["case_id"]
    )
    prepared = _direction_eval_context(runtime, baseline)
    old = old_repo.get(row["fold"])
    measured = runtime.measure(
        prepared,
        old,
        steering_vector=old.sigma_z * old.d_unit,
    )
    runtime.release_inputs(prepared)
    if abs(measured.applied_delta_z - old.sigma_z) > max(0.125, 0.05 * old.sigma_z):
        raise RuntimeError("Direction smoke did not realize +1 sigma coordinate shift")
    return {
        "status": "passed",
        "case_id": row["case_id"],
        "history_exact_reconstruction": reconstruction,
        "direction_plus_one_sigma": {
            "expected_delta_z": measured.expected_delta_z,
            "applied_delta_z": measured.applied_delta_z,
            "hook_exactly_once": measured.applied_count == 1,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.history_items < 1 or args.direction_items < 1:
        raise ValueError("History/direction item counts must be positive")
    deadline = Deadline(args.max_minutes)
    artifacts = SAFormationArtifacts.discover(args.experiment_dir)
    stage3 = artifacts.experiment_dir / "stage3_sa_formation"
    if not (stage3 / "progress.json").is_file():
        raise FileNotFoundError("Completed Stage 3 input is missing")
    output = _validate_output(artifacts.experiment_dir, args.output_dir)
    configuration = _configuration(artifacts, output, args)
    if args.dry_run:
        rows = load_baseline_rows(artifacts)
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "cuda_available": torch.cuda.is_available(),
                    "baseline_n": len(rows),
                    "unique_items": len({row["item_id"] for row in rows}),
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
    atomic_write_json(
        output / "progress.json",
        {"status": "running", "phase": args.phase},
    )
    if args.phase in {"all", "evidence"}:
        run_evidence_reanalysis(stage3, output)
    if args.phase == "evidence":
        return 0
    if args.phase == "report":
        final = write_followup_report(output)
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
    smoke_path = output / "gpu_smoke_nocache.json"
    if not smoke_path.is_file():
        atomic_write_json(smoke_path, _gpu_smoke(runtime, artifacts, output))
    if args.smoke_only:
        return 0
    if args.phase in {"all", "history"}:
        run_history_factorial(
            runtime,
            stage3,
            output,
            n_items=args.history_items,
            deadline=deadline.check,
        )
    if args.phase in {"all", "directions"}:
        run_direction_comparison(
            runtime,
            artifacts,
            stage3,
            output,
            n_items=args.direction_items,
            deadline=deadline.check,
        )
    if args.phase == "all":
        final = write_followup_report(output)
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
