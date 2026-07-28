#!/usr/bin/env python3
"""Create compact seven-field layer readouts and per-head source sinks."""

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

from layer_metacognition.hidden_state_store import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
)
from confidence_test.answer_metrics import normalize_answer  # noqa: E402
from confidence_test.dataset_utils import iter_dataset_items  # noqa: E402


def load_case_metadata(dataset: str | Path) -> dict[str, dict[str, Any]]:
    """Recover readout labels for legacy result records that lack them."""
    payload = json.loads(Path(dataset).read_text(encoding="utf-8"))
    metadata: dict[str, dict[str, Any]] = {}
    for _group, item in iter_dataset_items(payload):
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            continue
        metadata[item_id] = {
            "ground_truths": {
                "answer": normalize_answer(item.get("answer")),
                "conflict_answer": normalize_answer(item.get("conflict_ans")),
            },
            "text_answer": normalize_answer(item.get("text_ans")),
        }
    return metadata


def group_records_by_version(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep stable case/mode order while placing every V3 record before V4."""

    def version_rank(record: dict[str, Any]) -> int:
        version = record.get("version")
        if version == "v3":
            return 0
        if version == "v4":
            return 1
        case_id = str(record.get("case_id", ""))
        if "__v3__" in case_id:
            return 0
        if "__v4__" in case_id:
            return 1
        return 2

    return sorted(records, key=version_rank)


def split_analysis_by_version(
    analysis: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output = {"v3": [], "v4": []}
    for record in analysis:
        version = record.get("version")
        if version not in output:
            case_id = str(record.get("case_id", ""))
            if "__v3__" in case_id:
                version = "v3"
            elif "__v4__" in case_id:
                version = "v4"
        if version in output:
            output[version].append(record)
    return output


def _layer_map(values: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(values, list):
        return {}
    return {
        int(record["layer_index"]): record
        for record in values
        if isinstance(record, dict) and "layer_index" in record
    }


def _source_layer_maps(readout: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    """Read the new mode mapping, with legacy sac_layers as LMhead fallback."""
    raw_by_mode = readout.get("sac_layers_by_mode")
    if isinstance(raw_by_mode, dict):
        return {
            mode: _layer_map(raw_by_mode.get(mode))
            for mode in ("LMhead", "Identity", "Semantic")
        }
    return {
        "LMhead": _layer_map(readout.get("sac_layers")),
        "Identity": {},
        "Semantic": {},
    }


def _finite_or_none(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> Any:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    number = float(value)
    if minimum is not None and number < minimum:
        return None
    if maximum is not None and number > maximum:
        return None
    return number


def _finite_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    if not all(
        isinstance(number, (int, float)) and math.isfinite(float(number))
        for number in value
    ):
        return None
    return [float(number) for number in value]


def build_source_sink_minimal(
    records: list[dict[str, Any]],
    case_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    statistics = {"invalid_layer_values": 0, "invalid_sink_arrays": 0}
    for record in group_records_by_version(records):
        readout = record.get("direct_readout") or {}
        answers = _layer_map(readout.get("ac_layers"))
        confidences = _layer_map(readout.get("cc_layers"))
        sources_by_mode = _source_layer_maps(readout)
        layers: dict[str, list[Any]] = {}
        source_layers = set().union(
            *(set(values) for values in sources_by_mode.values())
        )
        for layer_index in sorted(set(answers) | set(confidences) | source_layers):
            answer = answers.get(layer_index) or {}
            confidence = confidences.get(layer_index) or {}
            values: list[Any] = [
                answer.get("predicted_answer")
                if isinstance(answer.get("predicted_answer"), str)
                else None,
                _finite_or_none(
                    answer.get("predicted_answer_probability"),
                    minimum=0.0,
                    maximum=1.0,
                ),
                _finite_or_none(answer.get("answer_entropy"), minimum=0.0),
                _finite_or_none(
                    confidence.get("soft_confidence"),
                    minimum=0.0,
                    maximum=1.0,
                ),
                *[
                    _finite_or_none(
                        (sources_by_mode[mode].get(layer_index) or {}).get(
                            "soft_image_score"
                        ),
                        minimum=0.0,
                        maximum=1.0,
                    )
                    for mode in ("LMhead", "Identity", "Semantic")
                ],
            ]
            if any(value is None for value in values):
                statistics["invalid_layer_values"] += 1
            layers[str(layer_index)] = values

        sinks: dict[str, Any] = {}
        for target, target_value in (record.get("attention_sinks") or {}).items():
            if not isinstance(target_value, dict):
                continue
            target_output: dict[str, Any] = {}
            for source_name, source_value in target_value.items():
                source_layers: dict[str, dict[str, list[float]]] = {}
                raw_layers = (
                    source_value.get("layers", {})
                    if isinstance(source_value, dict)
                    else {}
                )
                for layer in sorted(raw_layers, key=lambda value: int(value)):
                    metrics = raw_layers[layer]
                    sink_values = _finite_float_list(
                        metrics.get("sink_score_by_head")
                    )
                    mass_values = _finite_float_list(
                        metrics.get("attention_mass_by_head")
                    )
                    if (
                        sink_values is None
                        or mass_values is None
                        or len(sink_values) != len(mass_values)
                    ):
                        statistics["invalid_sink_arrays"] += 1
                        continue
                    source_layers[str(layer)] = {
                        "sink_score_by_head": sink_values,
                        "attention_mass_by_head": mass_values,
                    }
                target_output[str(source_name)] = source_layers
            sinks[str(target)] = target_output
        fallback_metadata = (case_metadata or {}).get(str(record.get("item_id")), {})
        ground_truths = record.get("ground_truths", fallback_metadata.get("ground_truths"))
        if not isinstance(ground_truths, dict):
            ground_truths = None
        compact_record: dict[str, Any] = {
            "case_id": record.get("case_id"),
            "version": record.get("version"),
            "ground_truths": ground_truths,
            "layers": layers,
            "sinks": sinks,
        }
        if record.get("version") == "v3":
            text_answer = record.get("text_answer", fallback_metadata.get("text_answer"))
            compact_record["text_answer"] = (
                text_answer if isinstance(text_answer, str) else None
            )
        output.append(compact_record)
    return output, statistics


def _compact_layer_values(values: list[Any]) -> str:
    encoded: list[str] = []
    for index, value in enumerate(values):
        if value is None:
            encoded.append("null")
        elif index == 0:
            encoded.append(json.dumps(value, ensure_ascii=False))
        else:
            encoded.append(f"{float(value):.3f}")
    return "[" + ",".join(encoded) + "]"


def write_source_sink_minimal(
    path: str | Path,
    analysis: list[dict[str, Any]],
) -> None:
    lines = ["["]
    for case_index, record in enumerate(analysis):
        lines.extend(
            [
                "  {",
                f'    "case_id": {json.dumps(record.get("case_id"), ensure_ascii=False)},',
                '    "layers": {',
            ]
        )
        layer_items = list(record.get("layers", {}).items())
        for offset, (layer, values) in enumerate(layer_items):
            comma = "," if offset + 1 < len(layer_items) else ""
            lines.append(
                f"      {json.dumps(str(layer))}: {_compact_layer_values(values)}{comma}"
            )
        lines.extend(["    },", '    "sinks": {'])
        target_items = list(record.get("sinks", {}).items())
        for target_offset, (target, sources) in enumerate(target_items):
            lines.append(f"      {json.dumps(target)}: {{")
            source_items = list(sources.items())
            for source_offset, (source, layers) in enumerate(source_items):
                lines.append(f"        {json.dumps(source)}: {{")
                sink_items = list(layers.items())
                for layer_offset, (layer, metrics) in enumerate(sink_items):
                    comma = "," if layer_offset + 1 < len(sink_items) else ""
                    sink_values = metrics["sink_score_by_head"]
                    mass_values = metrics["attention_mass_by_head"]
                    compact_sink = "[" + ",".join(
                        f"{float(value):.8f}" for value in sink_values
                    ) + "]"
                    compact_mass = "[" + ",".join(
                        f"{float(value):.8f}" for value in mass_values
                    ) + "]"
                    lines.extend(
                        [
                            f"          {json.dumps(str(layer))}: {{",
                            f'            "sink_score_by_head": {compact_sink},',
                            f'            "attention_mass_by_head": {compact_mass}',
                            f"          }}{comma}",
                        ]
                    )
                source_comma = "," if source_offset + 1 < len(source_items) else ""
                lines.append(f"        }}{source_comma}")
            target_comma = "," if target_offset + 1 < len(target_items) else ""
            lines.append(f"      }}{target_comma}")
        lines.append("    }")
        lines.append("  }" + ("," if case_index + 1 < len(analysis) else ""))
    lines.append("]")
    atomic_write_text(path, "\n".join(lines) + "\n")
    json.loads(Path(path).read_text(encoding="utf-8"))


def write_layer_readout_minimal(
    path: str | Path,
    analysis: list[dict[str, Any]],
) -> None:
    """Write only case IDs and seven-field per-layer readouts, without sinks."""
    lines = ["["]
    for case_index, record in enumerate(analysis):
        lines.extend(
            [
                "  {",
                f'    "case_id": {json.dumps(record.get("case_id"), ensure_ascii=False)},',
                f'    "ground_truths": {json.dumps(record.get("ground_truths"), ensure_ascii=False)},',
            ]
        )
        if record.get("version") == "v3":
            lines.append(
                f'    "text_answer": {json.dumps(record.get("text_answer"), ensure_ascii=False)},'
            )
        lines.append('    "layers": {')
        layer_items = list(record.get("layers", {}).items())
        for offset, (layer, values) in enumerate(layer_items):
            comma = "," if offset + 1 < len(layer_items) else ""
            lines.append(
                f"      {json.dumps(str(layer))}: {_compact_layer_values(values)}{comma}"
            )
        lines.extend(["    }", "  }" + ("," if case_index + 1 < len(analysis) else "")])
    lines.append("]")
    atomic_write_text(path, "\n".join(lines) + "\n")
    json.loads(Path(path).read_text(encoding="utf-8"))


def build_source_sink_summary(
    records: list[dict[str, Any]],
    analysis: list[dict[str, Any]],
    statistics: dict[str, int],
) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    readout_coverage = {"ac": 0, "cc": 0, "sac": 0}
    validation = {"passed": 0, "failed": 0, "not_run": 0}
    sac_coverage_by_mode = {
        "LMhead": 0,
        "Identity": 0,
        "Semantic": 0,
    }
    sac_validation_by_mode = {
        mode: {"passed": 0, "failed": 0, "not_run": 0}
        for mode in ("LMhead", "Identity", "Semantic")
    }
    source_scores: list[float] = []
    sink_arrays = 0
    for record in records:
        status = str(record.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
        direct = record.get("direct_readout") or {}
        for target in readout_coverage:
            if direct.get(f"{target}_layers"):
                readout_coverage[target] += 1
        source_maps = _source_layer_maps(direct)
        for mode, layers in source_maps.items():
            if layers:
                sac_coverage_by_mode[mode] += 1
        record_validation = record.get("validation") or {}
        for key in ("ac_last_layer", "cc_last_layer", "sac_last_layer"):
            check = record_validation.get(key)
            if check is None:
                validation["not_run"] += 1
            elif check.get("passed"):
                validation["passed"] += 1
            else:
                validation["failed"] += 1
        raw_sac_by_mode = record_validation.get("sac_by_mode")
        for mode in ("LMhead", "Identity", "Semantic"):
            if isinstance(raw_sac_by_mode, dict):
                check = raw_sac_by_mode.get(mode)
            elif mode == "LMhead":
                check = record_validation.get("sac_last_layer")
            else:
                check = None
            if check is None:
                sac_validation_by_mode[mode]["not_run"] += 1
            elif check.get("passed"):
                sac_validation_by_mode[mode]["passed"] += 1
            else:
                sac_validation_by_mode[mode]["failed"] += 1
        source = (record.get("generated") or {}).get("source_attribution") or {}
        score = source.get("soft_image_score")
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            source_scores.append(float(score))
        for target in (record.get("attention_sinks") or {}).values():
            for source_value in target.values():
                sink_arrays += len(source_value.get("layers", {}))
    return {
        "case_count": len(records),
        "minimal_case_count": len(analysis),
        "statuses": statuses,
        "readout_case_coverage": readout_coverage,
        "sac_readout_coverage_by_mode": sac_coverage_by_mode,
        "last_layer_validation": validation,
        "sac_validation_by_mode": sac_validation_by_mode,
        "source_score_range": (
            [min(source_scores), max(source_scores)] if source_scores else None
        ),
        "sink_layer_arrays": sink_arrays,
        **statistics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="layer_metacognition/output/v3_v4_source/results.jsonl",
    )
    parser.add_argument(
        "--output",
        default=(
            "layer_metacognition/output/v3_v4_source/"
            "analysis_source_sink_minimal.json"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default="layer_metacognition/output/v3_v4_source/summary.json",
    )
    parser.add_argument(
        "--layer-output",
        help=(
            "Optional base path for pure layer readouts. The writer appends "
            "_v3 and _v4 before the suffix. Defaults to "
            "analysis_layer_readout_minimal.json beside --output."
        ),
    )
    parser.add_argument(
        "--dataset",
        help="Optional dataset JSON used to add labels to legacy results.jsonl.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_jsonl(args.results, repair_trailing=False)
    metadata = load_case_metadata(args.dataset) if args.dataset else None
    analysis, statistics = build_source_sink_minimal(records, metadata)
    write_source_sink_minimal(args.output, analysis)
    layer_output_base = (
        Path(args.layer_output)
        if args.layer_output
        else Path(args.output).with_name("analysis_layer_readout_minimal.json")
    )
    split_analysis = split_analysis_by_version(analysis)
    layer_outputs: dict[str, Path] = {}
    for version in ("v3", "v4"):
        layer_output = layer_output_base.with_name(
            f"{layer_output_base.stem}_{version}{layer_output_base.suffix}"
        )
        write_layer_readout_minimal(layer_output, split_analysis[version])
        layer_outputs[version] = layer_output
    atomic_write_json(
        args.summary_output,
        build_source_sink_summary(records, analysis, statistics),
    )
    print(
        f"[INFO] Wrote {len(analysis)} source/sink cases to {args.output}; "
        f"pure layer readouts to {layer_outputs['v3']} and {layer_outputs['v4']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
