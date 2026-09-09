from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .config import MIDPOINTS_9, T3_LABELS, T3_VALUES


def _softmax(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all(): raise ValueError("Scores must be a finite vector")
    probabilities = np.exp(array - array.max()); probabilities /= probabilities.sum()
    return probabilities


def numeric_score(class_logits: Sequence[float], *, reversed_scale: bool = False, token_ids: Sequence[int] | None = None) -> dict[str, Any]:
    logits = np.asarray(class_logits, dtype=np.float64)
    if logits.shape != (9,) or not np.isfinite(logits).all(): raise ValueError("Expected nine finite class logits")
    probabilities = _softmax(logits); raw_hard = int(np.argmax(logits)); canonical_hard = 8 - raw_hard if reversed_scale else raw_hard
    values = np.asarray(MIDPOINTS_9[::-1] if reversed_scale else MIDPOINTS_9)
    soft = float(probabilities @ values)
    return {
        "class_token_ids": list(map(int, token_ids or [])), "class_logits": logits.tolist(), "class_probabilities": probabilities.tolist(),
        "probability_sum": float(probabilities.sum()), "raw_hard_class": raw_hard, "canonical_hard_class": canonical_hard,
        "canonical_soft_sa": soft, "signed_sa": 2.0 * soft - 1.0, "scoring_status": "completed",
    }


def t3_score(sequence_log_likelihoods: Sequence[float], token_ids: Sequence[Sequence[int]]) -> dict[str, Any]:
    if len(sequence_log_likelihoods) != 5 or len(token_ids) != 5: raise ValueError("T3 requires exactly five candidates")
    lengths = np.asarray([len(ids) for ids in token_ids], dtype=np.int64)
    if np.any(lengths <= 0): raise ValueError("T3 labels must contain tokens")
    total = np.asarray(sequence_log_likelihoods, dtype=np.float64)
    probabilities = _softmax(total); normalized_ll = total / lengths; normalized_probabilities = _softmax(normalized_ll)
    values = np.asarray(T3_VALUES, dtype=np.float64); hard = int(np.argmax(total)); normalized_hard = int(np.argmax(normalized_ll))
    soft = float(probabilities @ values); normalized_soft = float(normalized_probabilities @ values)
    candidates = [{"label": label, "token_ids": list(map(int, ids)), "token_count": int(len(ids)), "sequence_log_likelihood": float(ll), "probability": float(p), "length_normalized_log_likelihood": float(nll), "length_normalized_probability": float(np)} for label, ids, ll, p, nll, np in zip(T3_LABELS, token_ids, total, probabilities, normalized_ll, normalized_probabilities)]
    return {
        "t3_candidates": candidates, "canonical_hard_label": T3_LABELS[hard], "canonical_hard_group": hard,
        "canonical_hard_score": float(values[hard]), "canonical_soft_sa": soft, "signed_sa": 2.0 * soft - 1.0,
        "length_normalized_hard_label": T3_LABELS[normalized_hard], "length_normalized_hard_group": normalized_hard,
        "length_normalized_soft_sa": normalized_soft, "length_definition_agrees": hard == normalized_hard,
        "probability_sum": float(probabilities.sum()), "scoring_status": "completed",
    }


def conditional_sequence_log_likelihood(logits: Any, prefix_length: int, candidate_ids: Sequence[int]) -> float:
    """Score candidate tokens in a full prefix+candidate teacher-forcing forward."""
    import torch
    if not candidate_ids: raise ValueError("Candidate cannot be empty")
    if logits.ndim != 3 or logits.shape[0] != 1: raise ValueError("Expected logits [1, sequence, vocabulary]")
    total = 0.0
    for offset, token_id in enumerate(candidate_ids):
        position = prefix_length - 1 + offset
        total += float(torch.log_softmax(logits[0, position].float(), dim=-1)[int(token_id)].item())
    if not math.isfinite(total): raise ValueError("Non-finite sequence log-likelihood")
    return total


def parse_t3_greedy(text: str) -> dict[str, Any]:
    exact = text.strip()
    valid = exact in T3_LABELS and text == exact
    return {"greedy_text": text, "greedy_parse_status": "valid" if valid else "invalid", "greedy_label": exact if valid else None}
