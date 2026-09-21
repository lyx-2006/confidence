from __future__ import annotations

import math
from typing import Any

import torch

from .prompts import LABELS, LABEL_WEIGHTS


def label_token_ids(tokenizer: Any) -> tuple[int, ...]:
    ids: list[int] = []
    for label in LABELS:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded, skip_special_tokens=False) != label:
            raise ValueError(f"Attribution label must be exactly one token: {label!r} -> {encoded}")
        ids.append(int(encoded[0]))
    if len(ids) != len(LABEL_WEIGHTS) or len(set(ids)) != len(ids):
        raise ValueError("Five-class source-attribution labels collide or have invalid cardinality")
    return tuple(ids)


def attribution_score(logits: torch.Tensor, token_ids: tuple[int, ...]) -> dict[str, Any]:
    if len(token_ids) != len(LABELS):
        raise ValueError(f"Expected {len(LABELS)} attribution token IDs, found {len(token_ids)}")
    vocab = logits.detach().float().cpu().reshape(-1)
    selected = vocab[list(token_ids)]
    restricted = torch.softmax(selected, dim=-1)
    log_total = torch.logsumexp(vocab, dim=-1)
    label_mass = float(torch.exp(torch.logsumexp(selected, dim=-1) - log_total).item())
    label_logits = {label: float(selected[index].item()) for index, label in enumerate(LABELS)}
    label_probabilities = {
        label: float(restricted[index].item()) for index, label in enumerate(LABELS)
    }
    image_score = sum(label_probabilities[label] * LABEL_WEIGHTS[index] for index, label in enumerate(LABELS))
    signed_score = 2.0 * image_score - 1.0
    maximum = max(label_logits.values())
    maxima = [label for label, value in label_logits.items() if value == maximum]
    predicted = maxima[0] if len(maxima) == 1 else "tie"
    if image_score == 0.5:
        predicted_side = "tie"
    else:
        predicted_side = "image" if image_score > 0.5 else "text"
    if not all(math.isfinite(value) for value in (*label_logits.values(), image_score, signed_score, label_mass)):
        raise ValueError("Non-finite five-class attribution score")
    return {
        "label_token_ids": dict(zip(LABELS, token_ids)),
        "label_logits": label_logits,
        "label_probabilities": label_probabilities,
        "label_count": len(LABELS),
        "image_attribution_score": image_score,
        "signed_attribution_score": signed_score,
        "label_probability_mass": label_mass,
        "predicted_label": predicted,
        "predicted_side": predicted_side,
    }
