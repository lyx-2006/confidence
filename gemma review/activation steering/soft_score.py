from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from experiment_config import MIDPOINTS


def class_token_ids(tokenizer: Any) -> list[int]:
    ids = []
    for label in map(str, range(9)):
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"SA class {label!r} is not one token: {encoded}")
        ids.append(int(encoded[0]))
    if len(set(ids)) != 9:
        raise ValueError(f"SA class token IDs are not unique: {ids}")
    return ids


def soft_sa_from_logits(logits: Any, token_ids: Sequence[int]) -> dict[str, Any]:
    values = np.asarray([float(logits[index]) for index in token_ids], dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("Class logits must contain nine finite values")
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    score = float(np.dot(probabilities, np.asarray(MIDPOINTS)))
    hard = int(np.argmax(values))
    if not math.isfinite(score) or abs(float(probabilities.sum()) - 1.0) > 1e-9:
        raise ValueError("Invalid soft-SA score")
    return {
        "class_token_ids": list(map(int, token_ids)),
        "class_logits": values.tolist(),
        "class_probabilities": probabilities.tolist(),
        "probability_sum": float(probabilities.sum()),
        "soft_sa_image_score": score,
        "argmax_hard_class": hard,
        "argmax_midpoint": float(MIDPOINTS[hard]),
    }
