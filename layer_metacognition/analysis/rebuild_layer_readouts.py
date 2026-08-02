#!/usr/bin/env python3
"""Rebuild compact V3/V4 layer readouts from an existing results.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from layer_metacognition.analyze_source_sink_results import (
    COMPACT_LAYER_COLUMNS,
    COMPACT_LAYER_COLUMNS_WITH_ANSWER_VAL,
    build_source_sink_minimal,
    split_analysis_by_version,
    write_layer_readout_minimal,
)
from layer_metacognition.hidden_state_store import atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        default="layer_metacognition/output/v3_v4_newprompt",
    )
    parser.add_argument(
        "--output-base",
        default="analysis_layer_readout_minimal.json",
        help="Base filename; _v3/_v4 are inserted before the suffix.",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Synchronize config.json compact_layer_columns after rebuilding.",
    )
    return parser


def _version_path(experiment_dir: Path, output_base: str, version: str) -> Path:
    base = Path(output_base)
    if base.name != output_base:
        raise ValueError("--output-base must be a filename, not a path")
    return experiment_dir / f"{base.stem}_{version}{base.suffix}"


def _update_config(experiment_dir: Path) -> None:
    path = experiment_dir / "config.json"
    config: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    columns = (
        COMPACT_LAYER_COLUMNS_WITH_ANSWER_VAL
        if bool(config.get("answer_val"))
        else COMPACT_LAYER_COLUMNS
    )
    config["compact_layer_columns"] = list(columns)
    atomic_write_json(path, config)


def _build_streaming(results_path: Path) -> tuple[list[dict[str, Any]], int]:
    """Compact records one at a time to avoid retaining the large raw JSONL."""
    analysis: list[dict[str, Any]] = []
    record_count = 0
    with results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {results_path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at {results_path}:{line_number}"
                )
            compact, _statistics = build_source_sink_minimal([record])
            analysis.extend(compact)
            record_count += 1
    return analysis, record_count


def main() -> int:
    args = _parser().parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    results_path = experiment_dir / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    analysis, record_count = _build_streaming(results_path)
    split = split_analysis_by_version(analysis)
    outputs: dict[str, str] = {}
    for version in ("v3", "v4"):
        path = _version_path(experiment_dir, args.output_base, version)
        write_layer_readout_minimal(path, split[version])
        outputs[version] = str(path)
    if args.update_config:
        _update_config(experiment_dir)
    print(
        json.dumps(
            {
                "record_count": record_count,
                "columns": list(COMPACT_LAYER_COLUMNS),
                "outputs": outputs,
                "config_updated": bool(args.update_config),
            },
            indent=2,
        )
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
