#!/usr/bin/env python3
"""Analyze raw and baseline-corrected Semantic Patchscope robustness."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_metacognition.hidden_state_store import atomic_write_json  # noqa: E402


def average_ranks(values: Sequence[float]) -> list[float]:
    """Assign one-based average ranks, including exact-value ties."""
    indexed = sorted(enumerate(float(value) for value in values), key=lambda x: x[1])
    ranks = [0.0] * len(indexed)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        rank = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[indexed[position][0]] = rank
        cursor = end
    return ranks


def pearson_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    left_mean = statistics.fmean(left_values)
    right_mean = statistics.fmean(right_values)
    left_delta = [value - left_mean for value in left_values]
    right_delta = [value - right_mean for value in right_values]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0.0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_delta, right_delta)
    ) / denominator


def spearman_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson_correlation(average_ranks(left), average_ranks(right))


def _finite_score(value: Any, *, context: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{context} is not a finite number: {value!r}")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{context} is outside [0, 1]: {score}")
    return score


def _score_maps(
    payload: dict[str, Any],
) -> tuple[list[str], list[dict[int, dict[str, float]]]]:
    columns = payload.get("columns")
    if (
        not isinstance(columns, list)
        or columns[:2] != ["answer", "answer_probability"]
        or "base" not in columns[2:]
    ):
        raise ValueError("Result columns must start with answer/probability and include base")
    variants = [str(value) for value in columns[2:]]
    score_maps: list[dict[int, dict[str, float]]] = []
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Result payload has no records array")
    for record in records:
        case_id = str(record.get("case_id"))
        layers = record.get("layers")
        if not isinstance(layers, dict):
            raise ValueError(f"{case_id} has no layers object")
        case_layers: dict[int, dict[str, float]] = {}
        for raw_layer, row in layers.items():
            layer = int(raw_layer)
            if not isinstance(row, list) or len(row) != len(columns):
                raise ValueError(
                    f"{case_id} layer {layer} length does not match columns"
                )
            case_layers[layer] = {
                variant: _finite_score(
                    row[index + 2],
                    context=f"{case_id}/{layer}/{variant}",
                )
                for index, variant in enumerate(variants)
            }
        score_maps.append(case_layers)
    return variants, score_maps


def _paired_layer_values(
    cases: list[dict[int, dict[str, float]]],
    layer: int,
    variant: str,
) -> tuple[list[float], list[float]]:
    base: list[float] = []
    compared: list[float] = []
    for case in cases:
        values = case.get(layer)
        if values is None or "base" not in values or variant not in values:
            continue
        base.append(values["base"])
        compared.append(values[variant])
    return base, compared


def _comparison_layers(config: dict[str, Any]) -> list[int]:
    specification = config.get("comparison_layers")
    if not isinstance(specification, dict):
        raise ValueError("config.json does not define comparison_layers")
    start = int(specification["start"])
    end = int(specification["end"])
    if start < 0 or end < start:
        raise ValueError(f"Invalid comparison layer range: {start}..{end}")
    return list(range(start, end + 1))


def _group_members(
    config: dict[str, Any],
    selected_variants: list[str],
) -> dict[str, list[str]]:
    groups = {"synonym": ["base"], "reverse": ["base"], "order": ["base"]}
    definitions = config.get("semantic_variants")
    if not isinstance(definitions, list):
        raise ValueError("config.json does not define semantic_variants")
    selected = set(selected_variants)
    for definition in definitions:
        variant_id = str(definition.get("variant_id"))
        group = str(definition.get("group"))
        if variant_id in selected and group in groups and variant_id != "base":
            groups[group].append(variant_id)
    return groups


def analyze_result_set(
    payload: dict[str, Any],
    config: dict[str, Any],
    *,
    analysis_start_layer: int,
    analysis_end_layer: int,
) -> dict[str, Any]:
    variants, cases = _score_maps(payload)
    layers = _comparison_layers(config)
    if analysis_start_layer not in layers or analysis_end_layer not in layers:
        raise ValueError(
            "Analysis start/end must be inside the configured comparison range "
            f"{layers[0]}..{layers[-1]}"
        )
    if analysis_start_layer >= analysis_end_layer:
        raise ValueError("--analysis-start-layer must be less than --analysis-end-layer")

    variant_summaries: dict[str, Any] = {}
    for variant in variants:
        if variant == "base":
            continue
        layer_metrics: dict[str, Any] = {}
        for layer in layers:
            base, compared = _paired_layer_values(cases, layer, variant)
            layer_result = {
                "n": len(base),
                "mean_absolute_error": (
                    statistics.fmean(
                        abs(left - right)
                        for left, right in zip(base, compared)
                    )
                    if base
                    else None
                ),
                "spearman_vs_base": spearman_correlation(base, compared),
            }
            if variant == "reverse_direction":
                one_minus = [1.0 - value for value in compared]
                layer_result["one_minus_variant_alignment"] = {
                    "mean_absolute_error": (
                        statistics.fmean(
                            abs(left - right)
                            for left, right in zip(base, one_minus)
                        )
                        if base
                        else None
                    ),
                    "spearman_vs_base": spearman_correlation(
                        base,
                        one_minus,
                    ),
                }
            layer_metrics[str(layer)] = layer_result

        transitions: dict[str, Any] = {}
        for previous, current in zip(layers, layers[1:]):
            base_delta: list[float] = []
            variant_delta: list[float] = []
            for case in cases:
                previous_values = case.get(previous)
                current_values = case.get(current)
                if (
                    previous_values is None
                    or current_values is None
                    or variant not in previous_values
                    or variant not in current_values
                ):
                    continue
                base_delta.append(
                    current_values["base"] - previous_values["base"]
                )
                variant_delta.append(
                    current_values[variant] - previous_values[variant]
                )
            transition_result = {
                "n": len(base_delta),
                "mean_base_delta": (
                    statistics.fmean(base_delta) if base_delta else None
                ),
                "mean_variant_delta": (
                    statistics.fmean(variant_delta) if variant_delta else None
                ),
                "spearman_delta_vs_base": spearman_correlation(
                    base_delta,
                    variant_delta,
                ),
            }
            if variant == "reverse_direction":
                one_minus_delta = [-value for value in variant_delta]
                transition_result["one_minus_variant_alignment"] = {
                    "mean_variant_delta": (
                        statistics.fmean(one_minus_delta)
                        if one_minus_delta
                        else None
                    ),
                    "mean_absolute_error_vs_base": (
                        statistics.fmean(
                            abs(left - right)
                            for left, right in zip(
                                base_delta,
                                one_minus_delta,
                            )
                        )
                        if base_delta
                        else None
                    ),
                    "spearman_delta_vs_base": spearman_correlation(
                        base_delta,
                        one_minus_delta,
                    ),
                }
            transitions[f"{previous}->{current}"] = transition_result

        base_cumulative: list[float] = []
        variant_cumulative: list[float] = []
        for case in cases:
            start = case.get(analysis_start_layer)
            end = case.get(analysis_end_layer)
            if (
                start is None
                or end is None
                or variant not in start
                or variant not in end
            ):
                continue
            base_cumulative.append(end["base"] - start["base"])
            variant_cumulative.append(end[variant] - start[variant])
        cumulative = {
            "start_layer": analysis_start_layer,
            "end_layer": analysis_end_layer,
            "n": len(base_cumulative),
            "mean_base_change": (
                statistics.fmean(base_cumulative)
                if base_cumulative
                else None
            ),
            "mean_variant_change": (
                statistics.fmean(variant_cumulative)
                if variant_cumulative
                else None
            ),
            "mean_absolute_error_vs_base": (
                statistics.fmean(
                    abs(left - right)
                    for left, right in zip(
                        base_cumulative,
                        variant_cumulative,
                    )
                )
                if base_cumulative
                else None
            ),
            "spearman_vs_base": spearman_correlation(
                base_cumulative,
                variant_cumulative,
            ),
        }
        if variant == "reverse_direction":
            one_minus_cumulative = [
                -value for value in variant_cumulative
            ]
            cumulative["one_minus_variant_alignment"] = {
                "mean_variant_change": (
                    statistics.fmean(one_minus_cumulative)
                    if one_minus_cumulative
                    else None
                ),
                "mean_absolute_error_vs_base": (
                    statistics.fmean(
                        abs(left - right)
                        for left, right in zip(
                            base_cumulative,
                            one_minus_cumulative,
                        )
                    )
                    if base_cumulative
                    else None
                ),
                "spearman_vs_base": spearman_correlation(
                    base_cumulative,
                    one_minus_cumulative,
                ),
            }
        variant_summaries[variant] = {
            "layers": layer_metrics,
            "layer_transitions": transitions,
            "cumulative_change": cumulative,
        }

    group_variance: dict[str, Any] = {}
    for group, members in _group_members(config, variants).items():
        layer_values: dict[str, Any] = {}
        all_variances: list[float] = []
        for layer in layers:
            variances: list[float] = []
            if len(members) >= 2:
                for case in cases:
                    values = case.get(layer)
                    if values is None or any(
                        member not in values for member in members
                    ):
                        continue
                    variances.append(
                        statistics.pvariance(
                            [values[member] for member in members]
                        )
                    )
            all_variances.extend(variances)
            layer_values[str(layer)] = {
                "n": len(variances),
                "mean_population_variance": (
                    statistics.fmean(variances) if variances else None
                ),
            }
        group_variance[group] = {
            "members": members,
            "layers": layer_values,
            "case_layer_count": len(all_variances),
            "mean_population_variance": (
                statistics.fmean(all_variances)
                if all_variances
                else None
            ),
        }

    return {
        "source_value_definition": payload.get("source_value_definition"),
        "case_count": len(cases),
        "comparison_layers": {
            "start": layers[0],
            "end": layers[-1],
            "excluded_final_transformer_layer": layers[-1] + 1,
        },
        "cumulative_change_layers": {
            "start": analysis_start_layer,
            "end": analysis_end_layer,
        },
        "variants": variant_summaries,
        "group_variance": group_variance,
    }


def build_validation_summary(
    input_dir: Path,
    *,
    analysis_start_layer: int,
    analysis_end_layer: int,
) -> dict[str, Any]:
    config_path = input_dir / "config.json"
    raw_path = input_dir / "validation_results.json"
    corrected_path = input_dir / "validation_results_corrected.json"
    for path in (config_path, raw_path, corrected_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required validation file does not exist: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    corrected = json.loads(corrected_path.read_text(encoding="utf-8"))
    if raw.get("columns") != corrected.get("columns"):
        raise ValueError("Raw and corrected result columns differ")
    return {
        "analysis_start_layer": analysis_start_layer,
        "analysis_end_layer": analysis_end_layer,
        "raw": analyze_result_set(
            raw,
            config,
            analysis_start_layer=analysis_start_layer,
            analysis_end_layer=analysis_end_layer,
        ),
        "corrected": analyze_result_set(
            corrected,
            config,
            analysis_start_layer=analysis_start_layer,
            analysis_end_layer=analysis_end_layer,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--analysis-start-layer", type=int, default=18)
    parser.add_argument("--analysis-end-layer", type=int, default=24)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_dir = Path(args.input_dir).resolve()
        summary = build_validation_summary(
            input_dir,
            analysis_start_layer=args.analysis_start_layer,
            analysis_end_layer=args.analysis_end_layer,
        )
        output = input_dir / "validation_summary.json"
        atomic_write_json(output, summary)
        print(f"[INFO] Wrote validation summary to {output}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
