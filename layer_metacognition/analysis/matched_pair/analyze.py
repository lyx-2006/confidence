#!/usr/bin/env python3
"""Analyze existing V3/V4 source-attribution results with strict matched pairs.

This module is read-only with respect to the completed experiment. It reads the
existing JSON/JSONL artifacts and writes derived tables, statistics, and plots
to a separate output directory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
from scipy.stats import spearmanr, wilcoxon

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


FINAL_LAYER = 27
LATE_LAYERS = (24, 25, 26)
FOCUS_MARGIN_LAYERS = (22, 23, 24, 25, 26)
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_SEED = 42


@dataclass(frozen=True)
class PairSpecification:
    family: str
    condition_a: str
    condition_b: str
    expected_direction: str
    image_label_condition: str | None = None
    analyze_confidence: bool = False


# Every delta is condition_b - condition_a. For directional hypotheses,
# condition_b is the condition expected to have the larger value.
PAIR_SPECIFICATIONS = (
    PairSpecification(
        "null_vs_conflict_hard",
        "null",
        "conflict_hard",
        "positive",
        image_label_condition="conflict_easy",
    ),
    PairSpecification(
        "irr_vs_conflict_hard",
        "irr",
        "conflict_hard",
        "positive",
        image_label_condition="conflict_easy",
    ),
    PairSpecification(
        "conflict_easy_vs_conflict_hard",
        "conflict_hard",
        "conflict_easy",
        "positive",
        image_label_condition="conflict_easy",
    ),
    PairSpecification(
        "consistent_easy_vs_consistent_hard",
        "consistent_hard",
        "consistent_easy",
        "positive",
        analyze_confidence=True,
    ),
    PairSpecification(
        "null_vs_consistent_easy",
        "null",
        "consistent_easy",
        "positive",
    ),
    PairSpecification(
        "null_vs_irr",
        "null",
        "irr",
        "none",
    ),
)


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        for record in records
    ]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _indexed_layers(
    value: Any,
    *,
    value_field: str,
) -> dict[int, float]:
    if not isinstance(value, list):
        return {}
    output: dict[int, float] = {}
    for layer in value:
        if not isinstance(layer, dict):
            continue
        index = layer.get("layer_index")
        number = _finite_float(layer.get(value_field))
        if not isinstance(index, int) or number is None:
            continue
        if index in output:
            raise ValueError(f"Duplicate layer {index} for {value_field}")
        output[index] = number
    return output


def _answer_probabilities(value: Any) -> dict[int, dict[str, float]]:
    if not isinstance(value, list):
        return {}
    output: dict[int, dict[str, float]] = {}
    for layer in value:
        if not isinstance(layer, dict) or not isinstance(layer.get("layer_index"), int):
            continue
        probabilities = layer.get("answer_class_probabilities")
        if not isinstance(probabilities, dict):
            continue
        clean: dict[str, float] = {}
        for answer, raw_probability in probabilities.items():
            probability = _finite_float(raw_probability)
            if isinstance(answer, str) and probability is not None and probability >= 0.0:
                clean[answer] = probability
        index = int(layer["layer_index"])
        if index in output:
            raise ValueError(f"Duplicate Answer Patchscope layer {index}")
        output[index] = clean
    return output


def _extract_record(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if record.get("status") != "completed":
        return None, "status_not_completed"
    item_id = record.get("item_id")
    prior_index = record.get("prior_index")
    version = record.get("version")
    condition = record.get("condition")
    generated = record.get("generated")
    direct = record.get("direct_readout")
    if not isinstance(item_id, (str, int)):
        return None, "invalid_item_id"
    if not isinstance(prior_index, int):
        return None, "invalid_prior_index"
    if version not in {"v3", "v4"}:
        return None, "unsupported_version"
    if not isinstance(condition, str):
        return None, "invalid_condition"
    if not isinstance(generated, dict) or not isinstance(direct, dict):
        return None, "missing_generated_or_readout"
    current_answer = generated.get("current_answer")
    if not isinstance(current_answer, str) or not current_answer:
        return None, "missing_current_answer"

    source_modes = direct.get("sac_layers_by_mode")
    semantic_layers = (
        source_modes.get("Semantic") if isinstance(source_modes, dict) else None
    )
    sa_by_layer = _indexed_layers(semantic_layers, value_field="soft_image_score")
    if FINAL_LAYER not in sa_by_layer:
        return None, "missing_sa_final"
    if any(layer not in sa_by_layer for layer in LATE_LAYERS):
        return None, "missing_sa_late"

    confidence_by_layer = _indexed_layers(
        direct.get("cc_layers"),
        value_field="soft_confidence",
    )
    probabilities_by_layer = _answer_probabilities(
        direct.get("answer_patchscope_layers")
    )
    return (
        {
            "item_id": str(item_id),
            "prior_index": int(prior_index),
            "version": version,
            "condition": condition,
            "current_answer": current_answer,
            "sa_by_layer": sa_by_layer,
            "confidence_by_layer": confidence_by_layer,
            "answer_probabilities_by_layer": probabilities_by_layer,
        },
        None,
    )


def load_results(path: Path) -> tuple[dict[tuple[str, int, str, str, str], dict[str, Any]], dict[str, Any]]:
    records: dict[tuple[str, int, str, str, str], dict[str, Any]] = {}
    total = 0
    statuses: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    for raw in _iter_jsonl(path):
        total += 1
        statuses[str(raw.get("status", "missing"))] += 1
        record, reason = _extract_record(raw)
        if record is None:
            exclusions[reason or "unknown"] += 1
            continue
        key = (
            record["item_id"],
            record["prior_index"],
            record["version"],
            record["condition"],
            record["current_answer"],
        )
        if key in records:
            raise ValueError(f"Duplicate strict match key in {path}: {key}")
        records[key] = record
    return records, {
        "raw_record_count": total,
        "status_counts": dict(sorted(statuses.items())),
        "eligible_record_count": len(records),
        "record_exclusions": dict(sorted(exclusions.items())),
    }


def _load_text_labels(path: Path) -> dict[tuple[str, int], str]:
    labels: dict[tuple[str, int], str] = {}
    for record in _iter_jsonl(path):
        if record.get("parse_success") is not True:
            continue
        item_id = record.get("item_id")
        prior_index = record.get("prior_index")
        answer = record.get("text_only_answer")
        if not isinstance(item_id, (str, int)) or not isinstance(prior_index, int):
            continue
        if not isinstance(answer, str) or not answer:
            continue
        key = (str(item_id), int(prior_index))
        if key in labels:
            raise ValueError(f"Duplicate text-only label key in {path}: {key}")
        labels[key] = answer
    return labels


def _load_image_labels(path: Path) -> dict[tuple[str, str], str]:
    labels: dict[tuple[str, str], str] = {}
    for record in _iter_jsonl(path):
        if record.get("parse_success") is not True:
            continue
        item_id = record.get("item_id")
        condition = record.get("condition")
        answer = record.get("image_only_answer")
        if not isinstance(item_id, (str, int)) or not isinstance(condition, str):
            continue
        if not isinstance(answer, str) or not answer:
            continue
        key = (str(item_id), condition)
        if key in labels:
            raise ValueError(f"Duplicate image-only label key in {path}: {key}")
        labels[key] = answer
    return labels


def _late_mean(values: Mapping[int, float]) -> float:
    return float(np.mean([values[layer] for layer in LATE_LAYERS]))


def _margin_by_layer(
    record: Mapping[str, Any],
    *,
    image_answer: str,
    text_answer: str,
) -> dict[int, float]:
    margins: dict[int, float] = {}
    for layer, probabilities in record["answer_probabilities_by_layer"].items():
        image_probability = _finite_float(probabilities.get(image_answer))
        text_probability = _finite_float(probabilities.get(text_answer))
        if (
            image_probability is None
            or text_probability is None
            or image_probability <= 0.0
            or text_probability <= 0.0
        ):
            continue
        margins[int(layer)] = math.log(image_probability) - math.log(text_probability)
    return margins


def build_matched_pairs(
    records: Mapping[tuple[str, int, str, str, str], Mapping[str, Any]],
    text_labels: Mapping[tuple[str, int], str],
    image_labels: Mapping[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostics: dict[str, Any] = {}
    key_prefixes = {
        (item, prior, version, answer)
        for item, prior, version, _condition, answer in records
    }

    for specification in PAIR_SPECIFICATIONS:
        family_candidates = 0
        current_answer_mismatches = 0
        competition_counts: Counter[str] = Counter()
        pair_records: list[dict[str, Any]] = []
        for item_id, prior_index, version, current_answer in sorted(key_prefixes):
            key_a = (
                item_id,
                prior_index,
                version,
                specification.condition_a,
                current_answer,
            )
            key_b = (
                item_id,
                prior_index,
                version,
                specification.condition_b,
                current_answer,
            )
            record_a = records.get(key_a)
            record_b = records.get(key_b)
            if record_a is None or record_b is None:
                continue
            family_candidates += 1
            sa_a = float(record_a["sa_by_layer"][FINAL_LAYER])
            sa_b = float(record_b["sa_by_layer"][FINAL_LAYER])
            sa_late_a = _late_mean(record_a["sa_by_layer"])
            sa_late_b = _late_mean(record_b["sa_by_layer"])
            pair: dict[str, Any] = {
                "pair_family": specification.family,
                "item_id": item_id,
                "prior_index": prior_index,
                "version": version,
                "condition_a": specification.condition_a,
                "condition_b": specification.condition_b,
                "delta_definition": "condition_b_minus_condition_a",
                "current_answer": current_answer,
                "SA_a": sa_a,
                "SA_b": sa_b,
                "delta_SA": sa_b - sa_a,
                "SA_late_a": sa_late_a,
                "SA_late_b": sa_late_b,
                "delta_SA_late": sa_late_b - sa_late_a,
                "layer_margin": {},
                "competition_status": "not_applicable",
            }

            if specification.analyze_confidence:
                confidence_a = record_a["confidence_by_layer"]
                confidence_b = record_b["confidence_by_layer"]
                needed = set(LATE_LAYERS) | {FINAL_LAYER}
                if needed.issubset(confidence_a) and needed.issubset(confidence_b):
                    final_a = float(confidence_a[FINAL_LAYER])
                    final_b = float(confidence_b[FINAL_LAYER])
                    late_a = _late_mean(confidence_a)
                    late_b = _late_mean(confidence_b)
                    pair.update(
                        {
                            "confidence_a": final_a,
                            "confidence_b": final_b,
                            "delta_confidence": final_b - final_a,
                            "confidence_late_a": late_a,
                            "confidence_late_b": late_b,
                            "delta_confidence_late": late_b - late_a,
                        }
                    )

            if specification.image_label_condition is not None:
                text_answer = text_labels.get((item_id, prior_index))
                image_answer = image_labels.get(
                    (item_id, specification.image_label_condition)
                )
                if text_answer is None:
                    pair["competition_status"] = "missing_text_only_label"
                elif image_answer is None:
                    pair["competition_status"] = "missing_image_only_label"
                elif text_answer == image_answer:
                    pair["competition_status"] = "unidentifiable_same_unimodal_answer"
                else:
                    margin_a = _margin_by_layer(
                        record_a,
                        image_answer=image_answer,
                        text_answer=text_answer,
                    )
                    margin_b = _margin_by_layer(
                        record_b,
                        image_answer=image_answer,
                        text_answer=text_answer,
                    )
                    common_layers = sorted(set(margin_a).intersection(margin_b))
                    pair["layer_margin"] = {
                        str(layer): margin_b[layer] - margin_a[layer]
                        for layer in common_layers
                    }
                    pair["competition_status"] = (
                        "available" if common_layers else "missing_patchscope_probabilities"
                    )
                    pair["text_only_answer"] = text_answer
                    pair["image_only_answer"] = image_answer
                competition_counts[pair["competition_status"]] += 1

            pair["_sa_trajectory_a"] = dict(record_a["sa_by_layer"])
            pair["_sa_trajectory_b"] = dict(record_b["sa_by_layer"])
            pair_records.append(pair)

        # Quantify strict current-answer losses relative to matching without it.
        loose_a: dict[tuple[str, int, str], set[str]] = defaultdict(set)
        loose_b: dict[tuple[str, int, str], set[str]] = defaultdict(set)
        for item_id, prior_index, version, condition, answer in records:
            loose_key = (item_id, prior_index, version)
            if condition == specification.condition_a:
                loose_a[loose_key].add(answer)
            elif condition == specification.condition_b:
                loose_b[loose_key].add(answer)
        for loose_key in set(loose_a).intersection(loose_b):
            if not loose_a[loose_key].intersection(loose_b[loose_key]):
                current_answer_mismatches += 1

        by_family[specification.family] = pair_records
        diagnostics[specification.family] = {
            "strict_pair_count": family_candidates,
            "loose_keys_excluded_for_current_answer_mismatch": current_answer_mismatches,
            "competition_status_counts": dict(sorted(competition_counts.items())),
        }

    ordered = [
        pair
        for specification in PAIR_SPECIFICATIONS
        for pair in sorted(
            by_family[specification.family],
            key=lambda row: (
                row["version"],
                row["item_id"],
                row["prior_index"],
                row["current_answer"],
            ),
        )
    ]
    return ordered, diagnostics, by_family


def _underpowered(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_pairs: int,
    minimum_items: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if len(rows) < minimum_pairs:
        reasons.append(f"pair_count<{minimum_pairs}")
    item_count = len({str(row["item_id"]) for row in rows})
    if item_count < minimum_items:
        reasons.append(f"unique_item_count<{minimum_items}")
    return bool(reasons), reasons


def _cluster_bootstrap_mean_ci(
    rows: Sequence[Mapping[str, Any]],
    value_field: str,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    by_item: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite_float(row.get(value_field))
        if value is not None:
            by_item[str(row["item_id"])].append(value)
    item_ids = sorted(by_item)
    if not item_ids:
        return []
    generator = np.random.default_rng(seed)
    item_sums = np.asarray(
        [sum(by_item[item_id]) for item_id in item_ids],
        dtype=np.float64,
    )
    item_counts = np.asarray(
        [len(by_item[item_id]) for item_id in item_ids],
        dtype=np.float64,
    )
    sampled_indices = generator.integers(
        0,
        len(item_ids),
        size=(samples, len(item_ids)),
    )
    estimates = (
        np.sum(item_sums[sampled_indices], axis=1)
        / np.sum(item_counts[sampled_indices], axis=1)
    )
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return [float(lower), float(upper)]


def _wilcoxon_summary(values: Sequence[float]) -> dict[str, Any]:
    nonzero = [float(value) for value in values if value != 0.0]
    if not nonzero:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "analysis_unit": "strict_matched_pair",
            "nonzero_pair_count": 0,
        }
    result = wilcoxon(
        values,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "alternative": "two-sided",
        "zero_method": "wilcox",
        "analysis_unit": "strict_matched_pair",
        "nonzero_pair_count": len(nonzero),
    }


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    value_field: str,
    *,
    underpowered: bool,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    values = [
        value
        for row in rows
        if (value := _finite_float(row.get(value_field))) is not None
    ]
    if not values:
        return {
            "value_count": 0,
            "mean": None,
            "median": None,
            "positive_direction_proportion": None,
            "bootstrap_95_ci": None,
            "wilcoxon_signed_rank": None,
        }
    return {
        "value_count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "positive_direction_proportion": float(np.mean(np.asarray(values) > 0.0)),
        "bootstrap_95_ci": (
            None
            if underpowered
            else _cluster_bootstrap_mean_ci(
                rows,
                value_field,
                samples=bootstrap_samples,
                seed=seed,
            )
        ),
        "wilcoxon_signed_rank": (
            None if underpowered else _wilcoxon_summary(values)
        ),
    }


def build_pair_summary(
    by_family: Mapping[str, Sequence[Mapping[str, Any]]],
    diagnostics: Mapping[str, Any],
    *,
    input_metadata: Mapping[str, Any],
    minimum_pairs: int,
    minimum_items: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for family_index, specification in enumerate(PAIR_SPECIFICATIONS):
        rows = list(by_family[specification.family])
        is_underpowered, reasons = _underpowered(
            rows,
            minimum_pairs=minimum_pairs,
            minimum_items=minimum_items,
        )
        final_summary = _metric_summary(
            rows,
            "delta_SA",
            underpowered=is_underpowered,
            bootstrap_samples=bootstrap_samples,
            seed=seed + family_index * 100,
        )
        family_summary: dict[str, Any] = {
            "condition_a": specification.condition_a,
            "condition_b": specification.condition_b,
            "delta_definition": "condition_b_minus_condition_a",
            "expected_direction": specification.expected_direction,
            "pair_count": len(rows),
            "unique_item_count": len({str(row["item_id"]) for row in rows}),
            "status": "underpowered" if is_underpowered else "ok",
            "underpowered": is_underpowered,
            "underpowered_reasons": reasons,
            "delta_SA_mean": final_summary["mean"],
            "delta_SA_median": final_summary["median"],
            "positive_direction_proportion": final_summary[
                "positive_direction_proportion"
            ],
            "bootstrap_95_ci": final_summary["bootstrap_95_ci"],
            "wilcoxon_signed_rank": final_summary["wilcoxon_signed_rank"],
            "delta_SA_final": final_summary,
            "delta_SA_late": _metric_summary(
                rows,
                "delta_SA_late",
                underpowered=is_underpowered,
                bootstrap_samples=bootstrap_samples,
                seed=seed + family_index * 100 + 1,
            ),
            "matching_diagnostics": diagnostics[specification.family],
        }
        if specification.analyze_confidence:
            family_summary["delta_confidence_final"] = _metric_summary(
                rows,
                "delta_confidence",
                underpowered=is_underpowered,
                bootstrap_samples=bootstrap_samples,
                seed=seed + family_index * 100 + 2,
            )
            family_summary["delta_confidence_late"] = _metric_summary(
                rows,
                "delta_confidence_late",
                underpowered=is_underpowered,
                bootstrap_samples=bootstrap_samples,
                seed=seed + family_index * 100 + 3,
            )
        pairs[specification.family] = family_summary
    return {
        "analysis": "strict_matched_pair_source_attribution",
        "interpretation": "associational_only_not_causal",
        "matching_fields": [
            "item_id",
            "prior_index",
            "version",
            "current_answer",
        ],
        "delta_definition": "condition_b_minus_condition_a",
        "SA_definition": {
            "source": (
                "direct_readout.sac_layers_by_mode.Semantic[*].soft_image_score"
            ),
            "raw_only": True,
            "SA_final_layer": FINAL_LAYER,
            "SA_late_layers": list(LATE_LAYERS),
            "SA_late_aggregation": "arithmetic_mean",
        },
        "margin_definition": (
            "log(P(image_only_answer))-log(P(text_only_answer)); "
            "delta_M=condition_b-condition_a"
        ),
        "confidence_definition": {
            "source": "direct_readout.cc_layers[*].soft_confidence",
            "final_layer": FINAL_LAYER,
            "late_layers": list(LATE_LAYERS),
        },
        "inferential_settings": {
            "bootstrap_unit": "item_id",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "bootstrap_interval": "percentile_95",
            "minimum_pairs": minimum_pairs,
            "minimum_unique_items": minimum_items,
            "wilcoxon_unit": "strict_matched_pair",
            "wilcoxon_alternative": "two-sided",
        },
        "input_metadata": dict(input_metadata),
        "pairs": pairs,
    }


def build_layerwise_statistics(
    by_family: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    minimum_pairs: int,
    minimum_items: int,
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for specification in PAIR_SPECIFICATIONS:
        if specification.image_label_condition is None:
            continue
        rows = [
            row
            for row in by_family[specification.family]
            if row.get("competition_status") == "available"
        ]
        layer_values: dict[str, Any] = {}
        all_layers = sorted(
            {
                int(layer)
                for row in rows
                for layer in (row.get("layer_margin") or {}).keys()
            }
        )
        for layer in all_layers:
            eligible = [
                row
                for row in rows
                if str(layer) in (row.get("layer_margin") or {})
                and _finite_float(row.get("delta_SA")) is not None
            ]
            underpowered, reasons = _underpowered(
                eligible,
                minimum_pairs=minimum_pairs,
                minimum_items=minimum_items,
            )
            x = np.asarray(
                [float(row["layer_margin"][str(layer)]) for row in eligible],
                dtype=np.float64,
            )
            y = np.asarray(
                [float(row["delta_SA"]) for row in eligible],
                dtype=np.float64,
            )
            rho: float | None = None
            p_value: float | None = None
            status = "underpowered" if underpowered else "ok"
            if not underpowered:
                if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
                    status = "constant_input"
                else:
                    result = spearmanr(x, y)
                    if math.isfinite(float(result.statistic)):
                        rho = float(result.statistic)
                        p_value = float(result.pvalue)
                    else:
                        status = "undefined"
            layer_values[str(layer)] = {
                "layer": layer,
                "pair_count": len(eligible),
                "unique_item_count": len(
                    {str(row["item_id"]) for row in eligible}
                ),
                "delta_M_mean": float(np.mean(x)) if x.size else None,
                "delta_M_median": float(np.median(x)) if x.size else None,
                "delta_M_positive_proportion": (
                    float(np.mean(x > 0.0)) if x.size else None
                ),
                "delta_M_wilcoxon_signed_rank": (
                    None
                    if underpowered or not x.size
                    else _wilcoxon_summary(x.tolist())
                ),
                "spearman_rho": rho,
                "p_value": p_value,
                "status": status,
                "underpowered": underpowered,
                "underpowered_reasons": reasons,
                "focus_layer": layer in FOCUS_MARGIN_LAYERS,
            }
        families[specification.family] = {
            "condition_a": specification.condition_a,
            "condition_b": specification.condition_b,
            "delta_M_definition": "condition_b_minus_condition_a",
            "delta_SA_definition": "SA_final_condition_b_minus_condition_a",
            "competition_pair_count": len(rows),
            "layers": layer_values,
            "focus_L22_L26": {
                str(layer): layer_values.get(str(layer))
                for layer in FOCUS_MARGIN_LAYERS
            },
        }
    return {
        "statistic": "Spearman(delta_M_l, delta_SA_final)",
        "interpretation": "associational_only_not_causal",
        "families": families,
    }


def _public_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in pair.items()
        if not key.startswith("_")
    }


def plot_delta_sa_boxplots(
    by_family: Mapping[str, Sequence[Mapping[str, Any]]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    generator = np.random.default_rng(DEFAULT_SEED)
    for axis, specification in zip(axes.flat, PAIR_SPECIFICATIONS):
        rows = by_family[specification.family]
        values = np.asarray([float(row["delta_SA"]) for row in rows])
        if values.size:
            axis.boxplot(
                values,
                vert=True,
                widths=0.45,
                patch_artist=True,
                boxprops={"facecolor": "#8ecae6", "alpha": 0.8},
                medianprops={"color": "#d1495b", "linewidth": 2},
            )
            jitter = generator.normal(1.0, 0.035, size=values.size)
            axis.scatter(jitter, values, s=8, alpha=0.22, color="#023047")
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(
            f"{specification.family}\n"
            f"{specification.condition_b} − {specification.condition_a}; n={len(rows)}",
            fontsize=10,
        )
        axis.set_xticks([])
        axis.set_ylabel("ΔSA final (L27)")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Strict matched-pair ΔSA distributions", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_layerwise_spearman(statistics: Mapping[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for family, family_data in statistics["families"].items():
        layers = []
        values = []
        for layer, row in family_data["layers"].items():
            if row["spearman_rho"] is not None:
                layers.append(int(layer))
                values.append(float(row["spearman_rho"]))
        if layers:
            axis.plot(layers, values, marker="o", markersize=3, label=family)
    axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.axvspan(22, 26, color="#ffb703", alpha=0.12, label="focus L22–L26")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Spearman ρ(ΔM_l, ΔSA_final)")
    axis.set_title("Layer-wise answer-competition / source-attribution association")
    axis.set_xticks(range(0, FINAL_LAYER + 1, 2))
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_conflict_trajectory(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    for axis, version in zip(axes, ("v3", "v4")):
        version_rows = [row for row in rows if row["version"] == version]
        for suffix, label, color in (
            ("a", "conflict_hard", "#457b9d"),
            ("b", "conflict_easy", "#e63946"),
        ):
            trajectories = [row[f"_sa_trajectory_{suffix}"] for row in version_rows]
            layers = sorted(set.intersection(*(set(value) for value in trajectories))) if trajectories else []
            if not layers:
                continue
            matrix = np.asarray(
                [[trajectory[layer] for layer in layers] for trajectory in trajectories],
                dtype=np.float64,
            )
            mean = np.mean(matrix, axis=0)
            standard_error = (
                np.std(matrix, axis=0, ddof=1) / math.sqrt(matrix.shape[0])
                if matrix.shape[0] > 1
                else np.zeros(matrix.shape[1])
            )
            axis.plot(layers, mean, color=color, linewidth=2, label=label)
            axis.fill_between(
                layers,
                mean - 1.96 * standard_error,
                mean + 1.96 * standard_error,
                color=color,
                alpha=0.16,
            )
        axis.axvspan(22, 26, color="#ffb703", alpha=0.1)
        axis.set_title(f"{version}; strict matched n={len(version_rows)}")
        axis.set_xlabel("Layer")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Raw semantic soft_image_score")
    figure.suptitle(
        "Conflict easy/hard SA trajectory\nmean ± 1.96 SE over strict matched pairs",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")
    analysis_modes = config.get("analysis_modes")
    if not isinstance(analysis_modes, list) or "Semantic" not in analysis_modes:
        raise ValueError("config.json does not declare Semantic analysis mode")
    runtime = config.get("model_runtime")
    num_layers = runtime.get("num_hidden_layers") if isinstance(runtime, dict) else None
    if not isinstance(num_layers, int) or num_layers <= FINAL_LAYER:
        raise ValueError(
            f"config.json does not support requested final layer {FINAL_LAYER}"
        )
    return {
        "format_version": config.get("format_version"),
        "analysis_modes": analysis_modes,
        "num_hidden_layers": num_layers,
        "conditions": config.get("conditions"),
        "versions": config.get("versions"),
        "compact_layer_columns": config.get("compact_layer_columns"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        default="layer_metacognition/output/v3_v4",
    )
    parser.add_argument(
        "--output-dir",
        default="layer_metacognition/output/v3_v4/matched_pair",
    )
    parser.add_argument("--minimum-pairs", type=int, default=20)
    parser.add_argument("--minimum-items", type=int, default=10)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.minimum_pairs < 1 or args.minimum_items < 1:
        raise ValueError("minimum power thresholds must be positive")
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")

    experiment_dir = Path(args.experiment_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    results_path = experiment_dir / "results.jsonl"
    config_path = experiment_dir / "config.json"
    probe_dir = experiment_dir / "probe"
    text_labels_path = probe_dir / "text_only_labels.jsonl"
    image_labels_path = probe_dir / "image_only_labels.jsonl"
    for path in (
        results_path,
        config_path,
        text_labels_path,
        image_labels_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_metadata = _validate_config(config)
    records, result_metadata = load_results(results_path)
    text_labels = _load_text_labels(text_labels_path)
    image_labels = _load_image_labels(image_labels_path)
    pairs, diagnostics, by_family = build_matched_pairs(
        records,
        text_labels,
        image_labels,
    )

    input_metadata = {
        "experiment_dir": str(experiment_dir),
        "results": str(results_path),
        "config": str(config_path),
        "text_only_labels": str(text_labels_path),
        "image_only_labels": str(image_labels_path),
        "config_validation": config_metadata,
        "results_loading": result_metadata,
        "text_only_label_count": len(text_labels),
        "image_only_label_count": len(image_labels),
        "prohibited_label_fields_used": [],
    }
    summary = build_pair_summary(
        by_family,
        diagnostics,
        input_metadata=input_metadata,
        minimum_pairs=args.minimum_pairs,
        minimum_items=args.minimum_items,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    layerwise = build_layerwise_statistics(
        by_family,
        minimum_pairs=args.minimum_pairs,
        minimum_items=args.minimum_items,
    )

    _write_jsonl(
        output_dir / "matched_pairs.jsonl",
        [_public_pair(pair) for pair in pairs],
    )
    _write_json(output_dir / "pair_summary.json", summary)
    _write_json(output_dir / "layerwise_statistics.json", layerwise)
    plot_delta_sa_boxplots(by_family, output_dir / "delta_sa_boxplots.png")
    plot_layerwise_spearman(layerwise, output_dir / "layerwise_spearman.png")
    plot_conflict_trajectory(
        by_family["conflict_easy_vs_conflict_hard"],
        output_dir / "conflict_easy_hard_sa_trajectory.png",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "matched_pair_count": len(pairs),
                "pair_counts": {
                    specification.family: len(by_family[specification.family])
                    for specification in PAIR_SPECIFICATIONS
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
