"""Run the CPU-only confirmatory behavior/attribution/report join."""

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

from .sa_formation.confirmatory_three_layer_analysis import (
    analysis_root,
    analyze_confirmatory_three_layer_panel,
    build_confirmatory_input_provenance,
    discover_confirmatory_three_layer_paths,
    load_confirmatory_three_layer_panel,
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
    value.add_argument("--bootstrap-iterations", type=int, default=1000)
    value.add_argument(
        "--analyze-only",
        action="store_true",
        help="Explicitly document that this CPU command performs no model forwards.",
    )
    value.add_argument("--dry-run", action="store_true")
    return value


def validate_output(experiment_dir: str | Path, requested: str | None = None) -> Path:
    expected = analysis_root(experiment_dir).resolve()
    output = Path(requested).resolve() if requested else expected
    if output != expected:
        raise ValueError(f"Confirmatory three-layer output is fixed to {expected}; got {output}")
    return output


def _existing_complete(output: Path, input_sha256: str) -> dict | None:
    targets = (output / "results.jsonl", output / "summary.json", output / "summary.md")
    existing = [path.is_file() for path in targets]
    if not any(existing):
        return None
    if not all(existing):
        raise FileExistsError(
            f"Partial existing analysis is protected; inspect manually before retrying: {output}"
        )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("status") == "complete"
        and summary.get("input_aggregate_sha256") == input_sha256
    ):
        return summary
    raise FileExistsError(
        "Existing three-layer outputs have a different input fingerprint and are protected: "
        f"{output}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    experiment = Path(args.experiment_dir).resolve()
    output = validate_output(experiment, args.output_dir)
    if args.dry_run:
        try:
            paths = discover_confirmatory_three_layer_paths(experiment)
        except FileNotFoundError as exc:
            print(
                json.dumps(
                    {
                        "status": "waiting_for_stage06",
                        "cpu_only": True,
                        "output_dir": str(output),
                        "reason": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        print(
            json.dumps(
                {
                    "status": "ready",
                    "cpu_only": True,
                    "output_dir": str(output),
                    "inputs": {name: str(path) for name, path in paths.files().items()},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    panel = load_confirmatory_three_layer_panel(experiment)
    existing = _existing_complete(
        output, panel.input_provenance["input_aggregate_sha256"]
    )
    if existing is not None:
        print(
            json.dumps(
                {
                    "status": "reused",
                    "cpu_only": True,
                    "n": existing["n"],
                    "input_aggregate_sha256": existing[
                        "input_aggregate_sha256"
                    ],
                    "summary": str(output / "summary.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".three_layer_analysis_", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        summary = analyze_confirmatory_three_layer_panel(
            panel,
            staging,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        final_provenance = build_confirmatory_input_provenance(
            discover_confirmatory_three_layer_paths(experiment)
        )
        if final_provenance["input_aggregate_sha256"] != summary[
            "input_aggregate_sha256"
        ]:
            raise RuntimeError(
                "A Stage-06/03/04 input changed during analysis; staged outputs were "
                "discarded without modifying the analysis destination"
            )
        output.mkdir(parents=True, exist_ok=True)
        for name in ("results.jsonl", "summary.json", "summary.md"):
            os.replace(staging / name, output / name)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "cpu_only": True,
                "analyze_only": bool(args.analyze_only),
                "n": summary["n"],
                "input_aggregate_sha256": summary["input_aggregate_sha256"],
                "summary": str(output / "summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
