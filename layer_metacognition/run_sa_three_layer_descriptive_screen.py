"""Run the fixed development-only three-layer descriptive screen."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "layer_metacognition"

from layer_metacognition.hidden_state_store import atomic_write_json

from .sa_formation.core import initialize_run, sha256_file
from .sa_formation.three_layer_descriptive_screen import (
    BRIDGE_DIR,
    SCREEN_DIR,
    analyze_three_layer_panel,
    load_three_layer_panel,
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
    expected = (root / BRIDGE_DIR / SCREEN_DIR).resolve()
    output = Path(requested).resolve() if requested else expected
    if output != expected:
        raise ValueError(f"Three-layer screen output is fixed to {expected}; got {output}")
    protected = [
        root / "results.jsonl",
        root / "hidden_states",
        root / "stage1_metacognition",
        root / "stage3_sa_formation",
        root / "stage3_sa_truth_audit",
        *root.glob("stage2_*"),
        root / BRIDGE_DIR / "01_actual_source_reliance",
        root / BRIDGE_DIR / "02_donor_replication_extension",
        root / BRIDGE_DIR / "03_reliance_representation_devfit_confirm",
        root / BRIDGE_DIR / "04_reliance_representation_sensitivities",
    ]
    if any(
        output == path.resolve() or path.resolve().is_relative_to(output)
        for path in protected
    ):
        raise ValueError("Three-layer screen output overlaps a protected input")
    return output


def configuration(
    experiment_dir: Path,
    output: Path,
    *,
    input_aggregate_sha256: str,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "three_layer_development_descriptive_screen",
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output),
        "analysis_split": "development_only",
        "join_key": "case_id",
        "item_id_fallback": False,
        "primary_endpoint_rule": "final_answer == answer_star",
        "expected_primary_n": 67,
        "expected_endpoint_mismatch_n": 7,
        "B": "formal raw shared D+M12",
        "A_rank": "attribution shared_prediction_oof",
        "V": "attribution shared_target_oof",
        "fixed_nuisance": [
            "answer_side",
            "difficulty",
            "prior_strength",
            "full_margin",
        ],
        "answer_identity_sensitivity": True,
        "bootstrap_iterations": int(bootstrap_iterations),
        "input_aggregate_sha256": input_aggregate_sha256,
        "causal_mediator_authorized": False,
        "confirmatory_claim_authorized": False,
    }


def provenance(panel_input: dict[str, Any]) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    implementation = [
        Path(__file__).resolve(),
        repository
        / "layer_metacognition"
        / "sa_formation"
        / "three_layer_descriptive_screen.py",
    ]
    return {
        **panel_input,
        "implementation": {
            str(path.relative_to(repository)): {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in implementation
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.bootstrap_iterations < 1000:
        raise ValueError("Formal three-layer screen requires at least 1000 bootstraps")
    experiment = Path(args.experiment_dir).resolve()
    output = validate_output(experiment, args.output_dir)
    panel = load_three_layer_panel(
        experiment, expected_primary_n=67, expected_mismatch_n=7
    )
    config = configuration(
        experiment,
        output,
        input_aggregate_sha256=panel.input_provenance["input_aggregate_sha256"],
        bootstrap_iterations=args.bootstrap_iterations,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "output_dir": str(output),
                    "primary_n": len(panel.primary_rows),
                    "endpoint_mismatch_n": len(panel.endpoint_mismatch_rows),
                    "confirmatory_overlap": panel.join_audit[
                        "confirmatory_case_overlap"
                    ],
                    "item_only_nonjoin_n": panel.join_audit["item_only_nonjoin_n"],
                    "input_aggregate_sha256": panel.input_provenance[
                        "input_aggregate_sha256"
                    ],
                    "configuration": config,
                },
                indent=2,
            )
        )
        return 0
    fingerprint = initialize_run(output, config, resume=args.resume)
    atomic_write_json(output / "provenance.json", provenance(panel.input_provenance))
    atomic_write_json(
        output / "progress.json",
        {
            "status": "running",
            "primary_n": len(panel.primary_rows),
            "endpoint_mismatch_n": len(panel.endpoint_mismatch_rows),
            "development_only": True,
        },
    )
    summary = analyze_three_layer_panel(
        panel,
        output,
        bootstrap_iterations=args.bootstrap_iterations,
        config_fingerprint=fingerprint,
    )
    atomic_write_json(
        output / "progress.json",
        {
            "status": "complete",
            "primary_n": summary["n"],
            "endpoint_mismatch_n": summary["endpoint_mismatch_n"],
            "development_only": True,
            "causal_mediator_authorized": False,
            "confirmatory_claim_authorized": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

