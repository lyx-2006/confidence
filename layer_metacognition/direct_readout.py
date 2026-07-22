"""Restricted answer and confidence readouts from decoder residual states."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as functional

from .confidence_schema import CLASS_MIDPOINTS, CONFIDENCE_CLASSES
from .metrics import entropy_from_probabilities, top1_top2_probability_margin


def _encode(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(value) for value in ids]


def build_first_token_collision_report(tokenizer: Any, labels: Sequence[str]) -> dict[str, Any]:
    label_records: dict[str, Any] = {}
    token_to_labels: dict[int, list[str]] = {}
    for label in labels:
        raw_ids = _encode(tokenizer, label)
        space_text = f" {label}"
        space_ids = _encode(tokenizer, space_text)
        first_ids = list(dict.fromkeys(
            ([raw_ids[0]] if raw_ids else []) + ([space_ids[0]] if space_ids else [])
        ))
        if not first_ids:
            raise ValueError(f"Tokenizer produced no tokens for label {label!r}")
        label_records[label] = {
            "raw_text": label,
            "raw_token_ids": raw_ids,
            "space_text": space_text,
            "space_token_ids": space_ids,
            "first_token_variants": first_ids,
        }
        for token_id in first_ids:
            token_to_labels.setdefault(token_id, []).append(label)
    collisions = [
        {"token_id": token_id, "labels": collided}
        for token_id, collided in sorted(token_to_labels.items())
        if len(collided) > 1
    ]
    return {"labels": label_records, "collisions": collisions}


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def project_hidden_to_vocab(
    hidden: torch.Tensor,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
) -> torch.Tensor:
    with torch.inference_mode():
        vector = hidden.detach().reshape(1, 1, -1).to(_module_device(final_norm))
        normalized = final_norm(vector)
        normalized = normalized.to(_module_device(lm_head))
        return lm_head(normalized)[0, 0].detach().float().cpu()


def _restricted_logits(
    vocab_logits: torch.Tensor,
    labels: Sequence[str],
    report: dict[str, Any],
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for label in labels:
        token_ids = report["labels"][label]["first_token_variants"]
        values.append(torch.max(vocab_logits[torch.tensor(token_ids, dtype=torch.long)]))
    return torch.stack(values)


def answer_layer_readout(
    layer_index: int,
    hidden: torch.Tensor,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    candidates: list[str],
    collision_report: dict[str, Any],
    dataset_answer: str | None,
    stage1_answer: str | None,
    image_target: str | None,
    text_target: str | None,
) -> dict[str, Any]:
    vocab_logits = project_hidden_to_vocab(hidden, final_norm, lm_head)
    class_logits = _restricted_logits(vocab_logits, candidates, collision_report)
    probabilities = torch.softmax(class_logits, dim=-1)
    predicted_index = int(torch.argmax(probabilities).item())
    predicted = candidates[predicted_index]
    sorted_indices = torch.argsort(class_logits, descending=True).tolist()
    probability_map = {label: float(probabilities[index].item()) for index, label in enumerate(candidates)}
    logit_map = {label: float(class_logits[index].item()) for index, label in enumerate(candidates)}
    target_rank = None
    if dataset_answer in candidates:
        target_rank = sorted_indices.index(candidates.index(dataset_answer)) + 1
    margin = None
    if image_target in candidates and text_target in candidates and image_target != text_target:
        margin = logit_map[image_target] - logit_map[text_target]
    return {
        "layer_index": layer_index,
        "predicted_answer": predicted,
        "predicted_answer_probability": probability_map[predicted],
        "answer_class_logits": logit_map,
        "answer_class_probabilities": probability_map,
        "answer_entropy": entropy_from_probabilities(probabilities),
        "answer_top1_top2_margin": top1_top2_probability_margin(probabilities),
        "answer_target_rank": target_rank,
        "dataset_answer_probability": probability_map.get(dataset_answer),
        "stage1_answer_probability": probability_map.get(stage1_answer),
        "answer_image_text_margin": margin,
    }


def confidence_layer_readout(
    layer_index: int,
    hidden: torch.Tensor,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    collision_report: dict[str, Any],
) -> dict[str, Any]:
    vocab_logits = project_hidden_to_vocab(hidden, final_norm, lm_head)
    class_logits = _restricted_logits(vocab_logits, CONFIDENCE_CLASSES, collision_report)
    probabilities = torch.softmax(class_logits, dim=-1)
    predicted_index = int(torch.argmax(probabilities).item())
    midpoint_tensor = torch.tensor(CLASS_MIDPOINTS, dtype=probabilities.dtype)
    return {
        "layer_index": layer_index,
        "confidence_class_logits": {
            label: float(class_logits[index].item()) for index, label in enumerate(CONFIDENCE_CLASSES)
        },
        "confidence_class_probabilities": {
            label: float(probabilities[index].item()) for index, label in enumerate(CONFIDENCE_CLASSES)
        },
        "hard_confidence_label": CONFIDENCE_CLASSES[predicted_index],
        "soft_confidence": float(torch.sum(probabilities * midpoint_tensor).item()),
        "confidence_entropy": entropy_from_probabilities(probabilities),
        "confidence_top1_top2_margin": top1_top2_probability_margin(probabilities),
    }


def reconstruction_metrics(
    reconstructed_logits: torch.Tensor,
    model_logits: torch.Tensor,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    reconstructed = reconstructed_logits.detach().float().cpu()
    reference = model_logits.detach().float().cpu()
    delta = (reconstructed - reference).abs()
    return {
        "allclose": bool(torch.allclose(reconstructed, reference, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": float(delta.max().item()),
        "mean_abs_error": float(delta.mean().item()),
        "cosine_similarity": float(functional.cosine_similarity(reconstructed, reference, dim=0).item()),
    }
