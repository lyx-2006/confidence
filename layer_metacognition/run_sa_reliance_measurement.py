"""Run clean answer-only Actual Source Reliance measurement."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "layer_metacognition"

from layer_metacognition.hidden_state_store import atomic_write_json

from .sa_formation.core import SAFormationArtifacts, initialize_run, sha256_file
from .sa_formation.reliance_measurement import (
    BRIDGE_DIR,
    CONFIRMATORY_N,
    CORE_CONDITIONS,
    DEVELOPMENT_N,
    MEASUREMENT_METHOD_VERSION,
    PANEL_CONDITIONS,
    POSITIONS,
    RELIANCE_DIR,
    build_split_manifest,
    measure_case,
    plan_split,
    run_reliance_panel,
    verify_confirmatory_allowed,
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
        self.seconds = float(minutes) * 60.0

    def check(self) -> None:
        if time.monotonic() - self.started >= self.seconds:
            raise RuntimeError("Actual Reliance measurement time budget exhausted")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    value.add_argument("--output-dir")
    value.add_argument(
        "--split",
        choices=["development", "confirmatory"],
        default="development",
    )
    value.add_argument("--max-minutes", type=float, default=90.0)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--smoke-only", action="store_true")
    return value


def validate_output(experiment: str | Path, requested: str | None = None) -> Path:
    root = Path(experiment).resolve()
    expected = (root / BRIDGE_DIR / RELIANCE_DIR).resolve()
    output = Path(requested).resolve() if requested else expected
    if output != expected:
        raise ValueError(f"Actual Reliance output is fixed to {expected}; got {output}")
    protected = [
        root / "results.jsonl",
        root / "hidden_states",
        root / "stage1_metacognition",
        root / "stage3_sa_formation",
        root / "stage3_sa_formation_followup",
        root / "stage3_sa_mechanism",
        root / "stage3_sa_second_order",
        root / "stage3_sa_truth_audit",
        *root.glob("stage2_*"),
    ]
    if any(
        output == path.resolve() or path.resolve().is_relative_to(output)
        for path in protected
    ):
        raise ValueError("Actual Reliance output overlaps a protected input artifact")
    return output


def configuration(artifacts: SAFormationArtifacts, output: Path) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "clean_actual_source_reliance_measurement",
        "measurement_method_version": MEASUREMENT_METHOD_VERSION,
        "experiment_dir": str(artifacts.experiment_dir),
        "output_dir": str(output),
        "development_unique_items": DEVELOPMENT_N,
        "confirmatory_unique_items": CONFIRMATORY_N,
        "confirmatory_exclusion": {
            "remaining_after_development": 78,
            "excluded_item_id": "34",
            "reason": "no conflict_* eligible row",
        },
        "prompt": "V4 full-evidence answer-only; no History and no Source Attribution request",
        "answer_star": "natural restricted top-1 from answer-only full context",
        "behavior_probability_readout": "causal prefix-only forward before A* is appended",
        "hidden_capture": "pre-answer from prefix-only forward; post-answer from exact one-token teacher-forced continuation",
        "ambiguous_endpoint_rule": "exclude Full restricted ties at margin<=1e-6 before perturbations; no replacement item",
        "core_conditions": list(CORE_CONDITIONS),
        "panel_conditions": list(PANEL_CONDITIONS),
        "donor_replicates": 2,
        "hidden": {
            "context": "answer-only full evidence only",
            "layers": "all decoder blocks",
            "positions": list(POSITIONS),
            "dtype": "float16",
        },
        "target": "cross-fit equal-family deletion/replacement shared component",
        "verbal_sa_used": False,
        "seed": 42,
    }


def provenance(artifacts: SAFormationArtifacts) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    sources = {
        "truth_behavior_pool": artifacts.experiment_dir
        / "stage3_sa_truth_audit"
        / "01_counterfactual_source_use"
        / "results.jsonl",
        "authoritative_development_cohort": artifacts.experiment_dir
        / "stage3_sa_truth_audit"
        / "02_matched_prompt_source_perturbation"
        / "cohort_manifest.json",
        "item_split": artifacts.item_split,
        "dataset": artifacts.dataset,
    }
    implementation = [
        Path(__file__).resolve(),
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
    output: Path,
) -> dict[str, Any]:
    cohort, donors, case_by_key = plan_split(artifacts, "development")
    row = cohort[0]
    d1, d2 = donors[str(row["case_id"])]
    hidden_path = output / "gpu_smoke_hidden.npz"
    measured = measure_case(
        runtime,
        row,
        (d1, d2),
        case_by_key,
        hidden_path,
    )
    payload = np.load(hidden_path)
    expected_shape = (2, runtime.modules.num_hidden_layers, runtime.modules.hidden_size)
    if tuple(payload["hidden"].shape) != expected_shape:
        raise RuntimeError(
            f"Smoke hidden shape {payload['hidden'].shape} != {expected_shape}"
        )
    if payload["positions"].tolist() != list(POSITIONS):
        raise RuntimeError("Smoke hidden positions are not pre/post-answer")
    if payload["layers"].tolist() != list(range(runtime.modules.num_hidden_layers)):
        raise RuntimeError("Smoke did not capture every decoder block")
    return {
        "status": "passed",
        "case_id": row["case_id"],
        "answer_only_answer": measured["answer_only_answer"],
        "full_margin": measured["full_margin"],
        "conditions": list(measured["measurements"]),
        "donor1_item_id": d1["item_id"],
        "donor2_item_id": d2["item_id"],
        "verbal_sa_leakage": measured["verbal_sa_leakage"],
        "selection_measurement_same_forward": measured[
            "selection_measurement_same_forward"
        ],
        "teacher_forced_causal_prefix_equal": measured[
            "teacher_forced_causal_prefix_equal"
        ],
        "teacher_forced_length_path_max_logit_error_diagnostic": measured[
            "teacher_forced_length_path_max_logit_error"
        ],
        "teacher_forced_length_path_probability_tv_diagnostic": measured[
            "teacher_forced_length_path_probability_tv"
        ],
        "measurement_method_version": measured["measurement_method_version"],
        "hidden_shape": list(payload["hidden"].shape),
        "hidden_dtype": str(payload["hidden"].dtype),
        "positions": payload["positions"].tolist(),
        "layers": payload["layers"].tolist(),
        "symmetric_donor_sources": all(
            measured["condition_sources"][f"replace_text_d{index}"][
                "text_source_item"
            ]
            == measured["condition_sources"][f"replace_image_d{index}"][
                "image_source_item"
            ]
            for index in (1, 2)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    artifacts = SAFormationArtifacts.discover(args.experiment_dir)
    output = validate_output(artifacts.experiment_dir, args.output_dir)
    cohort, donors, _ = plan_split(artifacts, args.split)
    manifest = build_split_manifest(args.split, cohort, donors)
    config = configuration(artifacts, output)
    confirmatory_state: dict[str, Any] | None = None
    if args.split == "confirmatory" and output.exists():
        frozen = output / "frozen_measurement_rule.json"
        if frozen.is_file():
            value = json.loads(frozen.read_text(encoding="utf-8"))
            confirmatory_state = {
                "rule_fingerprint": value.get("rule_fingerprint"),
                "confirmatory_allowed": value.get("confirmatory_allowed"),
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
                    "cohort_manifest_fingerprint": manifest["manifest_fingerprint"],
                    "distinct_donor_pairs": len(
                        {
                            (str(pair[0]["item_id"]), str(pair[1]["item_id"]))
                            for pair in donors.values()
                        }
                    ),
                    "confirmatory_state": confirmatory_state,
                    "configuration": config,
                },
                indent=2,
            )
        )
        return 0
    initialize_run(output, config, resume=args.resume)
    atomic_write_json(output / "provenance.json", provenance(artifacts))
    atomic_write_json(
        output / "progress.json",
        {"status": "running", "split": args.split},
    )
    if args.split == "confirmatory":
        verify_confirmatory_allowed(output)
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
    smoke_path = output / "gpu_smoke.json"
    if not smoke_path.is_file():
        atomic_write_json(smoke_path, gpu_smoke(runtime, artifacts, output))
    if args.smoke_only:
        atomic_write_json(
            output / "progress.json",
            {"status": "smoke_complete", "split": args.split},
        )
        return 0
    summary = run_reliance_panel(
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
            "measurement_gate_passed": summary["measurement_gate_passed"],
            "summary": str(output / f"{args.split}_summary.json"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
