"""Run the gated Source Attribution natural-formation mechanism experiments."""

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
from .sa_formation.followup import _balanced_unique_cases
from .sa_formation.mechanism import (
    CUE_DIRECTIONS_DIR,
    HISTORY_RELEVANCE_DIR,
    MEDIATION_DIR,
    OLD_AUDIT_DIR,
    POLICY_DIR,
    REMAP_DIR,
    SingleTokenSemanticAnalyzer,
    _remap_context,
    finalize_old_audit,
    label_mappings,
    load_baseline_geometry,
    mechanism_gate,
    run_label_remapping,
    run_natural_cue_directions,
    run_old_history_replay,
    run_old_mediation,
    run_old_natural_cpu,
    run_policy_transfer,
    run_relevant_irrelevant_history,
    write_mechanism_report,
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
                f"Mechanism experiment budget exhausted after {elapsed / 60:.1f} minutes"
            )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    value.add_argument("--output-dir")
    value.add_argument(
        "--phase",
        choices=["all", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "report"],
        default="all",
    )
    value.add_argument("--remap-items", type=int, default=30)
    value.add_argument("--history-items", type=int, default=30)
    value.add_argument("--cue-items", type=int, default=30)
    value.add_argument("--mediation-pairs", type=int, default=30)
    value.add_argument("--policy-items", type=int, default=20)
    value.add_argument("--max-minutes", type=float, default=180.0)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--smoke-only", action="store_true")
    return value


def _validate_output(root: Path, requested: str | None) -> Path:
    expected = (root / "stage3_sa_mechanism").resolve()
    output = Path(requested).resolve() if requested else expected
    if output != expected:
        raise ValueError(f"Mechanism output is fixed to {expected}; got {output}")
    protected = [
        root / "results.jsonl",
        root / "hidden_states",
        root / "stage1_metacognition",
        root / "stage3_sa_formation",
        root / "stage3_sa_formation_followup",
        *root.glob("stage2_*"),
    ]
    if any(output == path.resolve() or path.resolve().is_relative_to(output) for path in protected):
        raise ValueError("Mechanism output would contain a protected input artifact")
    return output


def _configuration(
    artifacts: SAFormationArtifacts, output: Path, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "stage3_sa_natural_mechanism",
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
            "remap_items": args.remap_items,
            "history_items": args.history_items,
            "cue_items": args.cue_items,
            "mediation_pairs": args.mediation_pairs,
            "policy_items": args.policy_items,
        },
        "old_direction": "five item-OOF 25-per-side mean-difference directions from completed core follow-up",
        "gate": "Exp5 and Exp6 require both Exp1 natural and Exp2 semantic gates",
    }


def _provenance(artifacts: SAFormationArtifacts) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    stage3 = artifacts.experiment_dir / "stage3_sa_formation"
    followup = artifacts.experiment_dir / "stage3_sa_formation_followup"
    implementation = [
        Path(__file__).resolve(),
        repository / "layer_metacognition" / "sa_formation" / "mechanism.py",
        repository / "layer_metacognition" / "sa_formation" / "runtime.py",
        repository / "layer_metacognition" / "sa_formation" / "followup.py",
        repository / "confidence_test" / "joint_answer_source_extension.py",
        repository / "confidence_test" / "source_attribution_analyzer.py",
    ]
    return {
        "base_inputs": artifacts.provenance(),
        "stage3_inputs": {
            "ridge_index": {
                "path": str(stage3 / "directions" / "index.json"),
                "sha256": sha256_file(stage3 / "directions" / "index.json"),
            },
            "old_index": {
                "path": str(followup / "directions" / "old_oof" / "index.json"),
                "sha256": sha256_file(followup / "directions" / "old_oof" / "index.json"),
            },
            "history_exact": {
                "path": str(followup / "02_history_exact_factorial" / "results_nocache.jsonl"),
                "sha256": sha256_file(
                    followup / "02_history_exact_factorial" / "results_nocache.jsonl"
                ),
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
    followup: Path,
) -> dict[str, Any]:
    row = _balanced_unique_cases(load_baseline_rows(artifacts), 1)[0]
    direction = SAOOFDirectionRepository(followup / "directions" / "old_oof").get(
        row["fold"]
    )
    mappings = label_mappings()
    diagnostics: dict[str, Any] = {}
    for name, labels in mappings.items():
        analyzer = SingleTokenSemanticAnalyzer(runtime.generator.tokenizer, labels)
        prepared = _remap_context(runtime, row, labels)
        measured = runtime.measure(
            prepared,
            direction,
            steering_vector=direction.sigma_z * direction.d_unit,
            analyzer=analyzer,
        )
        runtime.release_inputs(prepared)
        if measured.applied_count != 1:
            raise RuntimeError(f"Smoke hook count failed for {name}")
        if abs(measured.applied_delta_z / direction.sigma_z - 1.0) > 0.05:
            raise RuntimeError(f"Smoke +1 sigma failed for {name}")
        diagnostics[name] = {
            "labels": labels,
            "token_ids": analyzer.encodings,
            "semantic_score": measured.source["soft_image_score"],
            "applied_delta_sigma": measured.applied_delta_z / direction.sigma_z,
        }
    return {"status": "passed", "case_id": row["case_id"], "mappings": diagnostics}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    counts = [
        args.remap_items,
        args.history_items,
        args.cue_items,
        args.mediation_pairs,
        args.policy_items,
    ]
    if any(value < 1 for value in counts):
        raise ValueError("All experiment counts must be positive")
    deadline = Deadline(args.max_minutes)
    artifacts = SAFormationArtifacts.discover(args.experiment_dir)
    stage3 = artifacts.experiment_dir / "stage3_sa_formation"
    followup = artifacts.experiment_dir / "stage3_sa_formation_followup"
    read_status = followup / "progress.json"
    if not read_status.is_file():
        raise FileNotFoundError("Completed core follow-up is missing")
    if json.loads(read_status.read_text(encoding="utf-8")).get("status") != "complete":
        raise ValueError("Core follow-up input is not complete")
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
    atomic_write_json(output / "progress.json", {"status": "running", "phase": args.phase})

    geometry = None
    if args.phase in {"all", "exp1"}:
        geometry, _ = run_old_natural_cpu(artifacts, stage3, followup, output)
    elif args.phase in {"exp3", "exp4", "exp5"}:
        geometry = load_baseline_geometry(artifacts, stage3, followup)

    if args.phase == "report":
        final = write_mechanism_report(output)
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
    smoke_path = output / "gpu_smoke.json"
    if not smoke_path.is_file():
        atomic_write_json(smoke_path, _gpu_smoke(runtime, artifacts, followup))
    if args.smoke_only:
        return 0

    if args.phase in {"all", "exp1"}:
        assert geometry is not None
        run_old_history_replay(
            runtime,
            stage3,
            followup,
            output,
            geometry,
            deadline=deadline.check,
        )
        finalize_old_audit(output)
    if args.phase == "exp1":
        return 0

    if args.phase in {"all", "exp2"}:
        run_label_remapping(
            runtime,
            artifacts,
            followup,
            output,
            n_items=args.remap_items,
            deadline=deadline.check,
        )
    if args.phase == "exp2":
        return 0

    if args.phase in {"all", "exp3"}:
        assert geometry is not None
        run_relevant_irrelevant_history(
            runtime,
            artifacts,
            followup,
            output,
            geometry,
            n_items=args.history_items,
            deadline=deadline.check,
        )
    if args.phase == "exp3":
        return 0

    if args.phase in {"all", "exp4"}:
        assert geometry is not None
        run_natural_cue_directions(
            runtime,
            artifacts,
            stage3,
            followup,
            output,
            geometry,
            n_items=args.cue_items,
            deadline=deadline.check,
        )
    if args.phase == "exp4":
        return 0

    if args.phase in {"all", "exp5"}:
        assert geometry is not None
        run_old_mediation(
            runtime,
            stage3,
            followup,
            output,
            geometry,
            n_pairs=args.mediation_pairs,
            deadline=deadline.check,
        )
    if args.phase == "exp5":
        return 0

    if args.phase in {"all", "exp6"}:
        run_policy_transfer(
            runtime,
            artifacts,
            stage3,
            followup,
            output,
            n_items=args.policy_items,
            deadline=deadline.check,
        )
    if args.phase == "exp6":
        return 0

    if args.phase == "all":
        mechanism_gate(output)
        final = write_mechanism_report(output)
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
