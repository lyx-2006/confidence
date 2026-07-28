"""Layer-wise confidence/source readout and final-layer validation."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from confidence_test.source_attribution_schema import (
    SOURCE_ATTRIBUTION_CLASSES,
    SOURCE_ATTRIBUTION_MIDPOINTS,
    gather_source_class_logits,
    source_distribution,
)

from .direct_readout import project_hidden_to_vocab


def restricted_logits(
    vocab_logits: torch.Tensor,
    labels: Sequence[str],
    class_token_ids: dict[str, Sequence[int]],
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for label in labels:
        ids = list(class_token_ids[label])
        if not ids:
            raise ValueError(f"Class {label!r} has no token variants")
        index = torch.tensor(ids, dtype=torch.long, device=vocab_logits.device)
        values.append(torch.max(vocab_logits.index_select(0, index)))
    return torch.stack(values).float()


def confidence_layer_readout_runtime(
    layer_index: int,
    hidden: torch.Tensor,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    labels: Sequence[str],
    midpoints: Sequence[float],
    class_token_ids: dict[str, Sequence[int]],
) -> dict[str, Any]:
    vocab_logits = project_hidden_to_vocab(hidden, final_norm, lm_head)
    logits = restricted_logits(vocab_logits, labels, class_token_ids)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_index = int(torch.argmax(probabilities).item())
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * torch.log(positive)).sum().item())
    soft = float(
        torch.sum(probabilities * torch.tensor(midpoints, dtype=torch.float32)).item()
    )
    return {
        "layer_index": int(layer_index),
        "hard_confidence_label": labels[predicted_index],
        "hard_confidence_index": predicted_index,
        "soft_confidence": soft,
        "confidence_entropy": entropy,
        "confidence_class_logits": [
            float(value) for value in logits.detach().cpu().tolist()
        ],
        "confidence_class_probabilities": [
            float(value) for value in probabilities.detach().cpu().tolist()
        ],
    }


def source_layer_readout(
    layer_index: int,
    hidden: torch.Tensor,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    class_token_ids: dict[str, Sequence[int]],
) -> dict[str, Any]:
    vocab_logits = project_hidden_to_vocab(hidden, final_norm, lm_head)
    return source_vocab_readout(
        vocab_logits,
        class_token_ids,
        layer_index=layer_index,
        analysis_mode="LMhead",
    )


def source_vocab_readout(
    vocab_logits: torch.Tensor,
    class_token_ids: dict[str, Sequence[int]],
    *,
    layer_index: int | None = None,
    analysis_mode: str | None = None,
) -> dict[str, Any]:
    """Build the common SAC readout schema from one vocabulary-logit row."""

    logits = gather_source_class_logits(vocab_logits, class_token_ids)
    result = source_distribution(
        logits,
        class_token_ids=class_token_ids,
        raw_output="",
        parsed_label=None,
    ).to_dict()
    for key in (
        "raw_output",
        "hard_label_parsed",
        "parsed_label",
        "class_token_ids",
        "token_diagnostics",
    ):
        result.pop(key, None)
    if layer_index is not None:
        result["layer_index"] = int(layer_index)
    if analysis_mode is not None:
        result["analysis_mode"] = str(analysis_mode)
    return result


def validate_restricted_reconstruction(
    reconstructed_vocab_logits: torch.Tensor,
    reference_vocab_logits: torch.Tensor,
    *,
    labels: Sequence[str],
    class_token_ids: dict[str, Sequence[int]],
    midpoints: Sequence[float] | None,
    tolerance: float,
) -> dict[str, Any]:
    reconstructed = reconstructed_vocab_logits.detach().float().cpu()
    reference = reference_vocab_logits.detach().float().cpu()
    max_abs_error = float((reconstructed - reference).abs().max().item())
    reconstructed_logits = restricted_logits(
        reconstructed, labels, class_token_ids
    ).cpu()
    reference_logits = restricted_logits(reference, labels, class_token_ids).cpu()
    reconstructed_probs = torch.softmax(reconstructed_logits, dim=-1)
    reference_probs = torch.softmax(reference_logits, dim=-1)
    probability_error = float(
        (reconstructed_probs - reference_probs).abs().max().item()
    )
    reconstructed_index = int(torch.argmax(reconstructed_probs).item())
    reference_index = int(torch.argmax(reference_probs).item())
    reconstructed_soft = None
    reference_soft = None
    soft_error = None
    if midpoints is not None:
        midpoint_tensor = torch.tensor(midpoints, dtype=torch.float32)
        reconstructed_soft = float(
            torch.sum(reconstructed_probs * midpoint_tensor).item()
        )
        reference_soft = float(torch.sum(reference_probs * midpoint_tensor).item())
        soft_error = abs(reconstructed_soft - reference_soft)
    passed = (
        max_abs_error <= tolerance
        and reconstructed_index == reference_index
        and probability_error <= tolerance
        and (soft_error is None or soft_error <= tolerance)
    )
    return {
        "passed": bool(passed),
        "tolerance": float(tolerance),
        "max_abs_error": max_abs_error,
        "restricted_probability_max_abs_error": probability_error,
        "reference_label": labels[reference_index],
        "reconstructed_label": labels[reconstructed_index],
        "reference_soft_score": reference_soft,
        "reconstructed_soft_score": reconstructed_soft,
        "soft_score_abs_error": soft_error,
    }


def source_readout_is_valid(record: dict[str, Any]) -> bool:
    probabilities = record.get("class_probabilities")
    return bool(
        isinstance(probabilities, list)
        and len(probabilities) == len(SOURCE_ATTRIBUTION_CLASSES)
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in probabilities)
        and abs(sum(float(value) for value in probabilities) - 1.0) < 1e-5
        and 0.0 <= float(record.get("soft_image_score", -1.0)) <= 1.0
        and 0.0 <= float(record.get("soft_text_score", -1.0)) <= 1.0
    )
