"""Run the CPU-only Stage08 fixed-L18 representation divergence audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "layer_metacognition"

from .sa_formation.core import stable_hash
from .sa_formation.fixed_l18_representation_divergence import (
    OUTPUT_DIR,
    analyze_fixed_l18_representation_divergence,
    build_input_provenance,
    configuration_payload,
    load_stage08_panel,
    output_root,
)


DEFAULT_EXPERIMENT = (
    Path(__file__).resolve().parent
    / "output"
    / "Final_v4_run"
    / "answer_basis_9"
)
REQUIRED_OUTPUTS = (
    "cohort_manifest.json",
    "results.jsonl",
    "summary.json",
    "summary.md",
    "provenance.json",
    "run_config.json",
    "development_oof_predictions.jsonl",
    "fold_audit.json",
    "directions/index.json",
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    value.add_argument("--output-dir")
    value.add_argument("--bootstrap-iterations", type=int, default=1000)
    value.add_argument(
        "--analyze-only",
        action="store_true",
        help="Document explicitly that this CPU command performs no model forwards.",
    )
    value.add_argument("--dry-run", action="store_true")
    return value


def validate_output(experiment_dir: str | Path, requested: str | None = None) -> Path:
    expected = output_root(experiment_dir).resolve()
    output = Path(requested).resolve() if requested else expected
    if output != expected:
        raise ValueError(f"Stage08 output is fixed to {expected}; got {output}")
    return output


def _existing_complete(
    output: Path, *, config_fingerprint: str, input_sha256: str
) -> dict | None:
    existing = [output.joinpath(name).is_file() for name in REQUIRED_OUTPUTS]
    if not any(existing):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"Unrecognized existing Stage08 output is protected: {output}"
            )
        return None
    if not all(existing):
        missing = [
            name for name, present in zip(REQUIRED_OUTPUTS, existing) if not present
        ]
        raise FileExistsError(
            "Partial existing Stage08 output is protected; missing "
            + ", ".join(missing)
        )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("status") == "complete"
        and summary.get("config_fingerprint") == config_fingerprint
        and summary.get("input_aggregate_sha256") == input_sha256
    ):
        return summary
    raise FileExistsError(
        "Existing Stage08 output has a different config/input fingerprint and is protected: "
        f"{output}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    experiment = Path(args.experiment_dir).resolve()
    output = validate_output(experiment, args.output_dir)
    configuration = configuration_payload(
        bootstrap_iterations=args.bootstrap_iterations
    )
    config_fingerprint = stable_hash(configuration)
    panel = load_stage08_panel(experiment)
    provenance_before = build_input_provenance(panel)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "experiment": OUTPUT_DIR,
                    "cpu_only": True,
                    "new_model_forwards": 0,
                    "development_n": len(panel.development_rows),
                    "confirmatory_n": len(panel.confirmatory_rows),
                    "output_dir": str(output),
                    "config_fingerprint": config_fingerprint,
                    "input_aggregate_sha256": provenance_before[
                        "input_aggregate_sha256"
                    ],
                    "claim_scope": "post-hoc descriptive/OOD, non-gate-bearing, noncausal",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    existing = _existing_complete(
        output,
        config_fingerprint=config_fingerprint,
        input_sha256=provenance_before["input_aggregate_sha256"],
    )
    if existing is not None:
        print(
            json.dumps(
                {
                    "status": "reused",
                    "cpu_only": True,
                    "n": existing["n"],
                    "summary": str(output / "summary.json"),
                    "input_aggregate_sha256": existing["input_aggregate_sha256"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".stage08_fixed_l18_", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / OUTPUT_DIR
        summary = analyze_fixed_l18_representation_divergence(
            panel,
            staging,
            bootstrap_iterations=args.bootstrap_iterations,
            config_fingerprint=config_fingerprint,
            input_provenance=provenance_before,
        )
        provenance_after = build_input_provenance(panel)
        if provenance_after["input_aggregate_sha256"] != provenance_before[
            "input_aggregate_sha256"
        ]:
            raise RuntimeError(
                "A read-only Stage01/03/06/10/07 input changed during Stage08; "
                "staged outputs were discarded"
            )
        if output.exists():
            raise FileExistsError(
                f"Stage08 output appeared during analysis and is protected: {output}"
            )
        os.replace(staging, output)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "cpu_only": True,
                "analyze_only": bool(args.analyze_only),
                "new_model_forwards": 0,
                "development_n": summary["development_n"],
                "n": summary["n"],
                "summary": str(output / "summary.json"),
                "input_aggregate_sha256": summary["input_aggregate_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

