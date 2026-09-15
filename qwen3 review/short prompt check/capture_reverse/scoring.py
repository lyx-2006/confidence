from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from dp_SA.config import MIDPOINTS


REVERSE_MIDPOINTS = tuple(reversed(MIDPOINTS))


def reverse_soft_sa_from_logits(logits: Any, token_ids: Sequence[int]) -> dict[str, Any]:
    """Score raw labels 0..8 on the canonical image-side scale.

    The reverse prompt defines raw label 0 as strongest image contribution and
    raw label 8 as strongest text contribution.  Logits/probabilities remain in
    raw label order; canonical hard class is mapped back to the ordinary
    image-increasing 0..8 orientation for direct comparison with short capture.
    """
    values = np.asarray([float(logits[i]) for i in token_ids], dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("Reverse class logits must be nine finite values")
    shifted = values - float(values.max())
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    score = float(np.dot(probabilities, np.asarray(REVERSE_MIDPOINTS)))
    raw_hard = int(np.argmax(values))
    canonical_hard = 8 - raw_hard
    if not math.isfinite(score) or abs(float(probabilities.sum()) - 1.0) > 1e-9:
        raise ValueError("Invalid reverse soft-SA result")
    return {
        "class_token_ids": list(map(int, token_ids)),
        "class_logits": values.tolist(),
        "class_probabilities": probabilities.tolist(),
        "class_score_midpoints": list(REVERSE_MIDPOINTS),
        "probability_sum": float(probabilities.sum()),
        "soft_sa_image_score": score,
        "raw_argmax_class": raw_hard,
        "argmax_hard_class": canonical_hard,
        "argmax_midpoint": float(REVERSE_MIDPOINTS[raw_hard]),
        "class_orientation": "raw_0_image_to_raw_8_text; canonical_hard_0_text_to_8_image",
    }

