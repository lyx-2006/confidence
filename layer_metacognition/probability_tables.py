"""Standalone per-layer restricted probability tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from .hidden_state_store import atomic_write_json


def _by_layer(values: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(values, list):
        return {}
    return {
        int(value["layer_index"]): value
        for value in values
        if isinstance(value, dict) and "layer_index" in value
    }


def _probability_map(
    value: Any,
    labels: Sequence[str] | None = None,
) -> dict[str, float] | None:
    if isinstance(value, dict):
        return {str(label): float(probability) for label, probability in value.items()}
    if isinstance(value, list) and labels is not None and len(value) == len(labels):
        return {
            str(label): float(value[index])
            for index, label in enumerate(labels)
        }
    return None


def _confidence_classes(record: dict[str, Any]) -> list[str] | None:
    generated = record.get("generated") or {}
    for field in ("current_confidence", "initial_confidence"):
        probabilities = (generated.get(field) or {}).get("class_probabilities")
        if isinstance(probabilities, dict) and probabilities:
            return [str(label) for label in probabilities]
    return None


def build_probability_tables(
    records: Iterable[dict[str, Any]],
    *,
    source_classes: Sequence[str],
    confidence_classes: Sequence[str] | None = None,
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    for record in records:
        readout = record.get("direct_readout") or {}
        answers = _by_layer(readout.get("ac_layers"))
        raw_answer_variants = readout.get("answer_patchscope_layers_by_variant")
        answer_variants = (
            {
                str(variant): _by_layer(values)
                for variant, values in raw_answer_variants.items()
            }
            if isinstance(raw_answer_variants, dict)
            else {}
        )
        if "original" not in answer_variants:
            answer_variants["original"] = _by_layer(
                readout.get("answer_patchscope_layers")
            )
        confidences = _by_layer(readout.get("cc_layers"))
        raw_sources = readout.get("sac_layers_by_mode")
        sources = {
            mode: _by_layer((raw_sources or {}).get(mode))
            for mode in ("LMhead", "Identity", "Semantic")
        }
        if not sources["LMhead"]:
            sources["LMhead"] = _by_layer(readout.get("sac_layers"))
        layer_indices = set(answers) | set(confidences)
        for values in answer_variants.values():
            layer_indices.update(values)
        for values in sources.values():
            layer_indices.update(values)
        record_confidence_classes = (
            [str(label) for label in confidence_classes]
            if confidence_classes is not None
            else _confidence_classes(record)
        )
        layers: dict[str, Any] = {}
        for layer_index in sorted(layer_indices):
            answer = answers.get(layer_index) or {}
            confidence = confidences.get(layer_index) or {}
            layers[str(layer_index)] = {
                "answer": {
                    "LMhead": _probability_map(
                        answer.get("answer_class_probabilities")
                    ),
                    "SemanticPatchscope": {
                        variant: _probability_map(
                            (values.get(layer_index) or {}).get(
                                "answer_class_probabilities"
                            )
                        )
                        for variant, values in answer_variants.items()
                    },
                },
                "confidence": _probability_map(
                    confidence.get("confidence_class_probabilities"),
                    record_confidence_classes,
                ),
                "source_attribution": {
                    mode: _probability_map(
                        (values.get(layer_index) or {}).get("class_probabilities"),
                        source_classes,
                    )
                    for mode, values in sources.items()
                },
            }
        output.append(
            {
                "case_id": record.get("case_id"),
                "item_id": record.get("item_id"),
                "prior_index": record.get("prior_index"),
                "condition": record.get("condition"),
                "version": record.get("version"),
                "attribution_mode": record.get("attribution_mode"),
                "status": record.get("status"),
                "layers": layers,
            }
        )
    return {
        "schema_version": 1,
        "probability_scope": "restricted_classes",
        "source_attribution_classes": [str(label) for label in source_classes],
        "records": output,
    }


def write_probability_tables(
    path: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    source_classes: Sequence[str],
    confidence_classes: Sequence[str] | None = None,
) -> None:
    atomic_write_json(
        path,
        build_probability_tables(
            records,
            source_classes=source_classes,
            confidence_classes=confidence_classes,
        ),
    )
