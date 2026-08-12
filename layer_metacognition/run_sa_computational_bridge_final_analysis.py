"""Materialize the final monotonic gate, Stage-07 skip, and bridge report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "layer_metacognition"

from layer_metacognition.hidden_state_store import atomic_write_json

from .sa_formation.computational_bridge_final_analysis import (
    FORMAT_VERSION,
    derive_causal_authorization_gate,
    discover_final_analysis_paths,
    input_readiness,
    load_final_analysis_inputs,
    validate_output_paths,
    write_final_analysis_outputs,
)
from .sa_formation.core import stable_hash


DEFAULT_EXPERIMENT = (
    Path(__file__).resolve().parent
    / "output"
    / "Final_v4_run"
    / "answer_basis_9"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    value.add_argument("--trace-output-dir")
    value.add_argument("--analysis-output-dir")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    return value


def configuration(
    experiment_dir: Path,
    trace_output: Path,
    analysis_output: Path,
    *,
    input_aggregate_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "experiment": "computational_bridge_final_analysis",
        "experiment_dir": str(experiment_dir),
        "trace_output_dir": str(trace_output),
        "analysis_output_dir": str(analysis_output),
        "input_aggregate_sha256": input_aggregate_sha256,
        "gate": (
            "monotonic frozen Stage-01 measurement + Stage-03 representation "
            "authorization; Stage-10 global coordinate scope preserved"
        ),
        "non_overriding_downstream_stages": ["02", "04", "05", "06"],
        "planned_causal_forwards_when_gate_fails": 0,
    }


def _existing_files(path: Path) -> list[Path]:
    return sorted(value for value in path.rglob("*") if value.is_file()) if path.exists() else []


def initialize_outputs(
    trace: Path,
    analysis: Path,
    config: dict[str, Any],
    *,
    resume: bool,
) -> str:
    fingerprint = stable_hash(config)
    existing = [*_existing_files(trace), *_existing_files(analysis)]
    config_path = analysis / "run_config.json"
    if existing and not resume:
        raise FileExistsError(
            "Final bridge outputs already exist; pass --resume: "
            + ", ".join(str(value) for value in existing[:5])
        )
    if existing:
        if not config_path.is_file():
            raise ValueError("Existing final outputs lack analysis/run_config.json")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("config_fingerprint") != fingerprint:
            raise ValueError("Final bridge resume configuration fingerprint mismatch")
    trace.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)
    payload = dict(config)
    payload["config_fingerprint"] = fingerprint
    atomic_write_json(config_path, payload)
    return fingerprint


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    experiment = Path(args.experiment_dir).resolve()
    trace, analysis = validate_output_paths(
        experiment, args.trace_output_dir, args.analysis_output_dir
    )
    paths = discover_final_analysis_paths(experiment)
    readiness = input_readiness(paths)
    if not readiness["ready"]:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "waiting_for_inputs",
                        "ready": False,
                        "trace_output_dir": str(trace),
                        "analysis_output_dir": str(analysis),
                        **readiness,
                    },
                    indent=2,
                )
            )
            return 0
        raise FileNotFoundError(
            "Final bridge inputs are incomplete: "
            + ", ".join(readiness["missing_paths"])
        )

    inputs = load_final_analysis_inputs(experiment)
    gate = derive_causal_authorization_gate(
        inputs.documents["stage01_confirmatory_summary"],
        inputs.documents["stage03_summary"],
        inputs.documents["stage03_authorization"],
        inputs.documents["stage10_summary"],
        inputs.documents["stage06_summary"],
    )
    config = configuration(
        experiment,
        trace,
        analysis,
        input_aggregate_sha256=inputs.provenance["aggregate_sha256"],
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "ready": True,
                    "trace_output_dir": str(trace),
                    "analysis_output_dir": str(analysis),
                    "input_aggregate_sha256": inputs.provenance[
                        "aggregate_sha256"
                    ],
                    "validation_audit": inputs.validation_audit,
                    "gate": gate,
                    "configuration": config,
                },
                indent=2,
            )
        )
        return 0

    fingerprint = initialize_outputs(
        trace, analysis, config, resume=bool(args.resume)
    )
    module = (
        Path(__file__).resolve().parent
        / "sa_formation"
        / "computational_bridge_final_analysis.py"
    )
    result = write_final_analysis_outputs(
        inputs,
        config_fingerprint=fingerprint,
        implementation_files=(Path(__file__).resolve(), module),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "causal_divergence_tracing": result[
                    "causal_divergence_tracing"
                ],
                "trace_output_dir": str(trace),
                "analysis_output_dir": str(analysis),
                "config_fingerprint": fingerprint,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
