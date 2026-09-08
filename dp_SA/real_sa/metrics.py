from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np
from scipy.special import logsumexp

from .config import EFFICIENCY_TOLERANCE, PRIMARY_EFFECT_THRESHOLD


def restricted_probabilities(scores: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (12,) or not np.isfinite(values).all() or temperature <= 0:
        raise ValueError("Expected twelve finite scores and a positive temperature")
    probabilities = np.exp(values / temperature - logsumexp(values / temperature))
    if abs(float(probabilities.sum()) - 1.0) > 1e-10:
        raise ValueError("Restricted candidate probabilities do not sum to one")
    return probabilities


def sign_type(phi_image: float, phi_text: float, *, threshold: float = PRIMARY_EFFECT_THRESHOLD) -> str:
    if abs(phi_image + phi_text) < threshold:
        return "weak_total_effect"
    if phi_image >= 0 and phi_text >= 0:
        return "both_support"
    if phi_image >= 0 and phi_text < 0:
        return "image_support_text_suppress"
    if phi_image < 0 and phi_text >= 0:
        return "text_support_image_suppress"
    return "both_suppress"


def case_metrics(v11: float, v10: float, v01: float, v00: float,
                 *, threshold: float = PRIMARY_EFFECT_THRESHOLD) -> dict[str, Any]:
    values = [float(v11), float(v10), float(v01), float(v00)]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError("Condition support values must be finite probabilities")
    d_image = v11 - v01
    d_text = v11 - v10
    phi_image = ((v10 - v00) + (v11 - v01)) / 2.0
    phi_text = ((v01 - v00) + (v11 - v10)) / 2.0
    interaction = v11 - v10 - v01 + v00
    total = phi_image + phi_text
    efficiency_error = abs(total - (v11 - v00))
    if efficiency_error > EFFICIENCY_TOLERANCE:
        raise ValueError(f"Shapley efficiency failed: {efficiency_error}")
    eligible = phi_image >= 0 and phi_text >= 0 and total >= threshold
    ratio = phi_image / total if eligible else None
    return {
        "D_I": d_image, "D_T": d_text, "phi_I": phi_image, "phi_T": phi_text,
        "J": interaction, "phi_sum": total, "efficiency_error": efficiency_error,
        "sign_type": sign_type(phi_image, phi_text, threshold=threshold),
        "total_effect": total, "R_I": ratio, "R_I_eligible": eligible,
        "G_R": phi_image - phi_text,
    }


def family_cluster_bootstrap(
    rows: Sequence[dict[str, Any]], value: Callable[[Sequence[dict[str, Any]]], float | None],
    *, repeats: int, seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["family_id"])].append(row)
    families = sorted(groups)
    if not families:
        return {"low": None, "high": None, "valid": 0}
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(repeats):
        chosen = rng.choice(families, len(families), replace=True)
        sample = [row for family in chosen for row in groups[str(family)]]
        result = value(sample)
        if result is not None and math.isfinite(float(result)):
            samples.append(float(result))
    if not samples:
        return {"low": None, "high": None, "valid": 0}
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "valid": len(samples)}


__all__ = ["case_metrics", "family_cluster_bootstrap", "restricted_probabilities", "sign_type"]
