#!/usr/bin/env python3
"""Create the compact per-layer answer/confidence JSON requested for inspection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_metacognition.hidden_state_store import atomic_write_json, atomic_write_text, load_jsonl  # noqa: E402


def build_minimal_analysis(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    skipped_layers = 0
    for record in records:
        readout = record.get("direct_readout", {})
        answers = {
            int(layer["layer_index"]): layer
            for layer in readout.get("ac_layers", [])
            if isinstance(layer, dict) and "layer_index" in layer
        }
        confidences = {
            int(layer["layer_index"]): layer
            for layer in readout.get("cc_layers", [])
            if isinstance(layer, dict) and "layer_index" in layer
        }
        layers: dict[str, list[Any]] = {}
        for layer_index in sorted(set(answers) | set(confidences)):
            answer = answers.get(layer_index)
            confidence = confidences.get(layer_index)
            if answer is None or confidence is None:
                skipped_layers += 1
                continue
            values = [
                answer.get("predicted_answer"),
                answer.get("predicted_answer_probability"),
                answer.get("answer_entropy"),
                confidence.get("soft_confidence"),
            ]
            if not isinstance(values[0], str) or any(
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in values[1:]
            ):
                skipped_layers += 1
                continue
            answer_probability = float(values[1])
            answer_entropy = float(values[2])
            soft_confidence = float(values[3])
            if not 0.0 <= answer_probability <= 1.0 or answer_entropy < 0.0 or not 0.0 <= soft_confidence <= 1.0:
                skipped_layers += 1
                continue
            layers[str(layer_index)] = [
                values[0],
                answer_probability,
                answer_entropy,
                soft_confidence,
            ]
        output.append({"case_id": record.get("case_id"), "layers": layers})
    return output, skipped_layers


def write_minimal_analysis(path: str | Path, analysis: list[dict[str, Any]]) -> None:
    """Write outer JSON prettily while keeping every layer tuple on one line."""
    lines = ["["]
    for case_index, record in enumerate(analysis):
        lines.append("  {")
        lines.append(f"    \"case_id\": {json.dumps(record.get('case_id'), ensure_ascii=False)},")
        lines.append("    \"layers\": {")
        layer_items = list(record.get("layers", {}).items())
        for layer_offset, (layer, values) in enumerate(layer_items):
            suffix = "," if layer_offset + 1 < len(layer_items) else ""
            compact_values = (
                "["
                + json.dumps(values[0], ensure_ascii=False)
                + ","
                + ",".join(f"{float(value):.3f}" for value in values[1:])
                + "]"
            )
            lines.append(f"      {json.dumps(str(layer))}: {compact_values}{suffix}")
        lines.append("    }")
        lines.append("  }" + ("," if case_index + 1 < len(analysis) else ""))
    lines.append("]")
    atomic_write_text(path, "\n".join(lines) + "\n")


def build_summary(analysis: list[dict[str, Any]], skipped_layers: int) -> dict[str, Any]:
    """Aggregate the compact per-case readouts by decoder layer."""
    aggregates: dict[str, dict[str, Any]] = {}
    for record in analysis:
        for layer, values in record.get("layers", {}).items():
            aggregate = aggregates.setdefault(
                str(layer),
                {
                    "case_count": 0,
                    "answer_distribution": {},
                    "answer_probability_sum": 0.0,
                    "answer_entropy_sum": 0.0,
                    "soft_confidence_sum": 0.0,
                },
            )
            answer, answer_probability, answer_entropy, soft_confidence = values
            aggregate["case_count"] += 1
            distribution = aggregate["answer_distribution"]
            distribution[answer] = distribution.get(answer, 0) + 1
            aggregate["answer_probability_sum"] += float(answer_probability)
            aggregate["answer_entropy_sum"] += float(answer_entropy)
            aggregate["soft_confidence_sum"] += float(soft_confidence)
    for aggregate in aggregates.values():
        count = aggregate["case_count"]
        aggregate["mean_answer_probability"] = round(aggregate.pop("answer_probability_sum") / count, 3)
        aggregate["mean_answer_entropy"] = round(aggregate.pop("answer_entropy_sum") / count, 3)
        aggregate["mean_soft_confidence"] = round(aggregate.pop("soft_confidence_sum") / count, 3)
    return {
        "case_count": len(analysis),
        "layer_count": len(aggregates),
        "skipped_layers": skipped_layers,
        "layers": aggregates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="layer_metacognition/output/main/results.jsonl")
    parser.add_argument("--output", default="layer_metacognition/output/main/analysis_minimal.json")
    parser.add_argument("--summary-output", default="layer_metacognition/output/main/summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_jsonl(args.results, repair_trailing=False)
    analysis, skipped = build_minimal_analysis(records)
    write_minimal_analysis(args.output, analysis)
    atomic_write_json(args.summary_output, build_summary(analysis, skipped))
    print(
        f"[INFO] Wrote {len(analysis)} cases to {args.output}; "
        f"summary to {args.summary_output}; skipped_layers={skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
