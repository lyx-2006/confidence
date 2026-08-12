"""Run development-fit, frozen-confirm Actual Source Reliance readouts."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "layer_metacognition"

from layer_metacognition.hidden_state_store import atomic_write_json

from .sa_formation.core import initialize_run, sha256_file, stable_hash
from .sa_formation.reliance_external_representation import (
    BRIDGE_DIR,
    ESTIMANDS,
    EXTERNAL_REPRESENTATION_DIR,
    MEASUREMENT_DIR,
    build_measurement_authorization,
    fit_external_reliance_representation,
    load_measurement_rows,
)
from .sa_formation.reliance_representation import (
    RELIANCE_LAYERS,
    RELIANCE_POSITIONS,
)
from .sa_formation.core import RIDGE_ALPHAS


DEFAULT_EXPERIMENT = (
    Path(__file__).resolve().parent
    / "output"
    / "Final_v4_run"
    / "answer_basis_9"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    value.add_argument("--output-dir")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--bootstrap-iterations", type=int, default=1000)
    return value


def validate_output(experiment_dir: str | Path, requested: str | None = None) -> Path:
    root = Path(experiment_dir).resolve()
    expected = (root / BRIDGE_DIR / EXTERNAL_REPRESENTATION_DIR).resolve()
    output = Path(requested).resolve() if requested else expected
    if output != expected:
        raise ValueError(
            f"External reliance representation output is fixed to {expected}; got {output}"
        )
    measurement = (root / BRIDGE_DIR / MEASUREMENT_DIR).resolve()
    if output == measurement or measurement.is_relative_to(output):
        raise ValueError("External representation output overlaps measurement inputs")
    protected = [
        root / "results.jsonl",
        root / "hidden_states",
        root / "stage1_metacognition",
        root / "stage3_sa_formation",
        root / "stage3_sa_truth_audit",
        *root.glob("stage2_*"),
    ]
    if any(
        output == path.resolve() or path.resolve().is_relative_to(output)
        for path in protected
    ):
        raise ValueError("External representation output overlaps a protected artifact")
    return output


def configuration(
    experiment_dir: Path,
    output: Path,
    measurement_root: Path,
    authorization: dict[str, Any],
    input_manifest: dict[str, Any],
    bootstrap_iterations: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "external_actual_source_reliance_representation",
        "experiment_dir": str(experiment_dir),
        "measurement_root": str(measurement_root),
        "output_dir": str(output),
        "measurement_method_version": 2,
        "estimands": list(ESTIMANDS),
        "selection": "development-only inner existing-fold CV",
        "confirmation": "same fold-specific transform/model applied unchanged",
        "layers": list(RELIANCE_LAYERS),
        "positions": list(RELIANCE_POSITIONS),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "graded_nuisance": [
            "answer_side",
            "answer_identity",
            "difficulty",
            "prior_strength",
            "full_margin",
        ],
        "bootstrap_iterations": int(bootstrap_iterations),
        "authorization_fingerprint": authorization["authorization_fingerprint"],
        "input_aggregate_sha256": input_manifest["aggregate_sha256"],
        "causal_mediator_authorized": False,
    }


def build_input_manifest(
    measurement_root: Path,
    development: list[dict[str, Any]],
    confirmatory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fingerprint every behavior table and production hidden-state input."""

    root = measurement_root.resolve()
    fixed = [
        root / "development_results.jsonl",
        root / "confirmatory_results.jsonl",
        root / "development_summary.json",
        root / "confirmatory_summary.json",
        root / "frozen_measurement_rule.json",
        root / "development_cohort_manifest.json",
        root / "confirmatory_cohort_manifest.json",
    ]
    hidden: dict[str, Path] = {}
    for row in [*development, *confirmatory]:
        raw_hidden = str(row.get("hidden_file", "")).strip()
        if not raw_hidden:
            raise ValueError(f"Measurement row {row.get('case_id')!r} lacks hidden_file")
        relative = Path(raw_hidden)
        path = relative.resolve() if relative.is_absolute() else (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Hidden input escapes measurement root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        hidden[str(path.relative_to(root))] = path
    entries = [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "kind": "hidden" if path.suffix == ".npz" else "table",
        }
        for path in [*fixed, *(hidden[key] for key in sorted(hidden))]
    ]
    payload = {
        "measurement_root": str(root),
        "file_count": len(entries),
        "hidden_file_count": len(hidden),
        "files": entries,
    }
    payload["aggregate_sha256"] = stable_hash(payload)
    return payload


def provenance(
    measurement_root: Path,
    input_manifest: dict[str, Any],
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    inputs = [
        measurement_root / "development_results.jsonl",
        measurement_root / "confirmatory_results.jsonl",
        measurement_root / "development_summary.json",
        measurement_root / "confirmatory_summary.json",
        measurement_root / "frozen_measurement_rule.json",
        measurement_root / "development_cohort_manifest.json",
        measurement_root / "confirmatory_cohort_manifest.json",
    ]
    implementation = [
        Path(__file__).resolve(),
        repository
        / "layer_metacognition"
        / "sa_formation"
        / "reliance_external_representation.py",
    ]
    return {
        "inputs": {
            path.name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in inputs
        },
        "input_manifest": input_manifest,
        "implementation": {
            str(path.relative_to(repository)): {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in implementation
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.bootstrap_iterations < 10:
        raise ValueError("--bootstrap-iterations must be at least 10")
    experiment = Path(args.experiment_dir).resolve()
    measurement_root = experiment / BRIDGE_DIR / MEASUREMENT_DIR
    output = validate_output(experiment, args.output_dir)
    development = load_measurement_rows(measurement_root, "development")
    confirmatory = load_measurement_rows(measurement_root, "confirmatory")
    authorization = build_measurement_authorization(measurement_root)
    input_manifest = build_input_manifest(
        measurement_root, development, confirmatory
    )
    config = configuration(
        experiment,
        output,
        measurement_root,
        authorization,
        input_manifest,
        args.bootstrap_iterations,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "output_dir": str(output),
                    "development_n": len(development),
                    "confirmatory_n": len(confirmatory),
                    "measurement_authorization": authorization,
                    "input_aggregate_sha256": input_manifest["aggregate_sha256"],
                    "input_file_count": input_manifest["file_count"],
                    "hidden_file_count": input_manifest["hidden_file_count"],
                    "configuration": config,
                },
                indent=2,
            )
        )
        return 0
    initialize_run(output, config, resume=args.resume)
    atomic_write_json(
        output / "provenance.json", provenance(measurement_root, input_manifest)
    )
    atomic_write_json(output / "measurement_authorization.json", authorization)
    atomic_write_json(
        output / "progress.json",
        {
            "status": "running",
            "development_n": len(development),
            "confirmatory_n": len(confirmatory),
        },
    )
    summary = fit_external_reliance_representation(
        development,
        confirmatory,
        output,
        measurement_authorization=authorization,
        hidden_root=measurement_root,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    atomic_write_json(
        output / "progress.json",
        {
            "status": "complete",
            "causal_mediator_authorized": False,
            "raw_readout_gate_passed": summary["estimands"]["raw_choice_coupled"][
                "readout_gate_passed"
            ],
            "graded_candidate_source_use_representation": summary["estimands"][
                "graded_preregistered"
            ]["candidate_source_use_representation"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
