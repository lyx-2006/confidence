from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np

from .config import MIDPOINTS_K8
from .utils import finite_vector, stable_seed


def probabilities(logits: Sequence[float]) -> np.ndarray:
    values = finite_vector(logits)
    shifted = values - values.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    return probs


def score_logits(logits: Sequence[float], *, clean_class: int | None = None) -> dict[str, Any]:
    values = finite_vector(logits)
    probs = probabilities(values)
    hard = int(np.argmax(values))
    target = hard if clean_class is None else int(clean_class)
    if target < 0 or target >= 9:
        raise ValueError("clean class outside 0..8")
    margin = float(values[target] - np.delete(values, target).mean())
    output = {
        "class_logits": values.tolist(), "class_probabilities": probs.tolist(),
        "hard_class": hard, "hard_midpoint": float(MIDPOINTS_K8[hard]),
        "soft_sa": float(np.dot(probs, np.asarray(MIDPOINTS_K8))),
        "fixed_clean_class_margin": margin,
    }
    if not all(math.isfinite(float(value)) for value in (output["soft_sa"], margin)):
        raise ValueError("metric contains NaN/Inf")
    return output


def condition_side(recipient_side: str, donor_side: str) -> str:
    if recipient_side == "image_side" and donor_side == "image":
        return "I_from_I"
    if recipient_side == "image_side" and donor_side == "text":
        return "I_from_T"
    if recipient_side == "text_side" and donor_side == "text":
        return "T_from_T"
    if recipient_side == "text_side" and donor_side == "image":
        return "T_from_I"
    raise ValueError(f"invalid side pair: {recipient_side}, {donor_side}")


def paired_effect(same: float, cross: float, side: str, metric: str) -> float:
    if metric in {"first_token_change", "first_token_changed"}:
        return float(cross - same)
    if metric == "fixed_clean_class_margin":
        return float(same - cross)
    if side == "image_side":
        return float(same - cross)
    if side == "text_side":
        return float(cross - same)
    raise ValueError(f"unknown side: {side}")


def paired_rows(rows: Sequence[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["recipient_case_id"])][str(row["swap_kind"])] = row
    output = []
    for case_id, pair in sorted(grouped.items()):
        if set(pair) != {"same", "cross"}:
            raise ValueError(f"recipient {case_id} lacks same/cross pair")
        same = pair["same"]
        cross = pair["cross"]
        side = str(same["recipient_side"])
        if side != str(cross["recipient_side"]):
            raise ValueError("same/cross recipient side mismatch")
        output.append({
            "recipient_case_id": case_id, "recipient_side": side,
            "same_case": same, "cross_case": cross,
            "same": float(same[metric]), "cross": float(cross[metric]),
            "effect": paired_effect(float(same[metric]), float(cross[metric]), side, metric),
        })
    return output


def bootstrap_values(values: Sequence[float], *, repeats: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be non-empty and finite")
    rng = np.random.default_rng(seed)
    samples = np.asarray([array[rng.integers(0, len(array), len(array))].mean() for _ in range(repeats)], dtype=np.float64)
    return {
        "mean": float(array.mean()), "sem": float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else None,
        "ci_low": float(np.percentile(samples, 2.5)), "ci_high": float(np.percentile(samples, 97.5)),
        "bootstrap_repeats": int(repeats), "sample_count": int(len(array)),
    }


def stratified_effect_summary(pairs: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> dict[str, Any]:
    if not pairs:
        raise ValueError("no paired rows")
    sides = {str(row["recipient_side"]) for row in pairs}
    if sides == {"image_side", "text_side"}:
        groups = [np.asarray([row["effect"] for row in pairs if row["recipient_side"] == side], dtype=np.float64)
                  for side in ("image_side", "text_side")]
        if any(len(group) == 0 for group in groups):
            raise ValueError("both side groups are required")
        rng = np.random.default_rng(seed)
        samples = np.asarray([
            0.5 * (groups[0][rng.integers(0, len(groups[0]), len(groups[0]))].mean() +
                    groups[1][rng.integers(0, len(groups[1]), len(groups[1]))].mean())
            for _ in range(repeats)
        ], dtype=np.float64)
        values = np.asarray([0.5 * (np.mean(groups[0]) + np.mean(groups[1]))], dtype=np.float64)
        observed = float(values[0])
        sem = float(np.sqrt(np.var(groups[0], ddof=1) / len(groups[0]) + np.var(groups[1], ddof=1) / len(groups[1])) / 2) if len(groups[0]) > 1 and len(groups[1]) > 1 else None
        return {"mean": observed, "sem": sem, "ci_low": float(np.percentile(samples, 2.5)), "ci_high": float(np.percentile(samples, 97.5)),
                "bootstrap_repeats": repeats, "sample_count": len(pairs), "image_count": len(groups[0]), "text_count": len(groups[1])}
    return bootstrap_values([float(row["effect"]) for row in pairs], repeats=repeats, seed=seed)


def sign_flip_p(values: Sequence[float], *, repeats: int, seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("sign flip values must be finite and non-empty")
    observed = abs(float(array.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(repeats):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(array))
        if abs(float(np.mean(array * signs))) >= observed:
            extreme += 1
    return float((extreme + 1) / (repeats + 1))


def bh_fdr(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("invalid p-values")
    order = np.argsort(array)
    ranked = array[order]
    adjusted = np.minimum.accumulate((ranked * len(array) / np.arange(1, len(array) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output.tolist()


def condition_summary(rows: Sequence[dict[str, Any]], metric: str, *, repeats: int, seed: int) -> list[dict[str, Any]]:
    output = []
    for condition in ("I_from_I", "I_from_T", "T_from_T", "T_from_I"):
        subset = [row for row in rows if row["condition"] == condition]
        if not subset:
            continue
        stats = bootstrap_values([float(row[metric]) for row in subset], repeats=repeats,
                                 seed=stable_seed(seed, "condition", condition, metric))
        output.append({"condition": condition, "metric": metric, **stats})
    return output


__all__ = ["bh_fdr", "bootstrap_values", "condition_side", "condition_summary", "paired_effect",
           "paired_rows", "probabilities", "score_logits", "sign_flip_p", "stratified_effect_summary"]
