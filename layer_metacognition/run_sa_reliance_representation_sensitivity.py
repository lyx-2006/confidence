"""Run post-hoc sensitivities for the frozen reliance representation."""

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

from .sa_formation.core import initialize_run, sha256_file
from .sa_formation.reliance_external_representation import (
    BRIDGE_DIR,
    EXTERNAL_REPRESENTATION_DIR,
    MEASUREMENT_DIR,
)
from .sa_formation.donor_replication_extension import EXTENSION_DIR
from .sa_formation.reliance_representation_sensitivity import (
    FORMAT_VERSION,
    SENSITIVITY_DIR,
    input_provenance,
    load_sensitivity_inputs,
    run_representation_sensitivities,
)


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
    expected = (root / BRIDGE_DIR / SENSITIVITY_DIR).resolve()
    output = Path(requested).resolve() if requested else expected
    if output != expected:
        raise ValueError(f"Sensitivity output is fixed to {expected}; got {output}")
    protected = [
        root / "results.jsonl",
        root / "hidden_states",
        root / "stage1_metacognition",
        root / "stage3_sa_formation",
        root / "stage3_sa_truth_audit",
        root / BRIDGE_DIR / MEASUREMENT_DIR,
        root / BRIDGE_DIR / EXTENSION_DIR,
        root / BRIDGE_DIR / EXTERNAL_REPRESENTATION_DIR,
        *root.glob("stage2_*"),
    ]
    if any(
        output == path.resolve()
        or path.resolve().is_relative_to(output)
        or output.is_relative_to(path.resolve())
        for path in protected
    ):
        raise ValueError("Sensitivity output overlaps a protected source artifact")
    return output


def configuration(
    experiment_dir: Path,
    output: Path,
    *,
    bootstrap_iterations: int,
    aggregate_input_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "experiment": "posthoc_reliance_representation_sensitivities",
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output),
        "source_representation_dir": str(
            experiment_dir / BRIDGE_DIR / EXTERNAL_REPRESENTATION_DIR
        ),
        "analyses": [
            "fresh_donor_target_transport_without_readout_refit",
            "development_fit_nested_linear_calibration",
        ],
        "bootstrap_iterations": int(bootstrap_iterations),
        "aggregate_input_sha256": aggregate_input_sha256,
        "post_hoc": True,
        "gate_bearing": False,
        "original_03_gate_modified": False,
        "causal_mediator_authorized": False,
    }


def provenance_payload(
    experiment_dir: Path, input_files: dict[str, Any]
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    implementation = [
        Path(__file__).resolve(),
        repository
        / "layer_metacognition"
        / "sa_formation"
        / "reliance_representation_sensitivity.py",
    ]
    return {
        "inputs": input_files,
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
        "experiment_dir": str(experiment_dir),
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.bootstrap_iterations < 10:
        raise ValueError("--bootstrap-iterations must be at least 10")
    experiment = Path(args.experiment_dir).resolve()
    output = validate_output(experiment, args.output_dir)
    source_provenance = input_provenance(experiment)
    inputs = load_sensitivity_inputs(experiment)
    config = configuration(
        experiment,
        output,
        bootstrap_iterations=args.bootstrap_iterations,
        aggregate_input_sha256=source_provenance["aggregate_sha256"],
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "output_dir": str(output),
                    "development_n": len(inputs.joined["development"]),
                    "confirmatory_n": len(inputs.joined["confirmatory"]),
                    "lineage_audit": inputs.lineage_audit,
                    "aggregate_input_sha256": source_provenance[
                        "aggregate_sha256"
                    ],
                    "configuration": config,
                },
                indent=2,
            )
        )
        return 0

    initialize_run(output, config, resume=args.resume)
    atomic_write_json(
        output / "provenance.json",
        provenance_payload(experiment, source_provenance),
    )
    atomic_write_json(
        output / "progress.json",
        {
            "status": "running",
            "development_n": len(inputs.joined["development"]),
            "confirmatory_n": len(inputs.joined["confirmatory"]),
            "gate_bearing": False,
            "causal_mediator_authorized": False,
        },
    )
    run_representation_sensitivities(
        inputs,
        output,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    atomic_write_json(
        output / "progress.json",
        {
            "status": "complete",
            "original_03_gate_modified": False,
            "gate_bearing": False,
            "causal_mediator_authorized": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
