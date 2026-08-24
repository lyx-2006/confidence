from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from dp_SA.config import MIDPOINTS
from .config import DENOMINATOR_EPSILON
from .io import stable_seed


def probabilities(logits: Sequence[float]) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("Expected nine finite class logits")
    values = values - values.max()
    probs = np.exp(values)
    probs /= probs.sum()
    return probs


def score_logits(logits: Sequence[float], *, clean_class: int | None = None) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    probs = probabilities(values)
    hard = int(np.argmax(values))
    target = hard if clean_class is None else int(clean_class)
    if target < 0 or target > 8:
        raise ValueError("clean_class is outside 0-8")
    margin = float(values[target] - np.delete(values, target).mean())
    entropy = float(-np.sum(probs * np.log(probs)))
    return {
        "class_logits": values.tolist(), "class_probabilities": probs.tolist(),
        "hard_class": hard, "hard_midpoint": float(MIDPOINTS[hard]),
        "soft_sa": float(np.dot(probs, np.asarray(MIDPOINTS, dtype=np.float64))),
        "entropy": entropy, "fixed_clean_class_margin": margin,
    }


def oriented(value: float, side: str) -> float:
    sigma = 1.0 if side == "image_side" else -1.0 if side == "text_side" else None
    if sigma is None:
        raise ValueError(f"Unknown test side: {side}")
    return sigma * (float(value) - float(MIDPOINTS[4]))


def recovery(clean: float, corrupt: float, patched: float, *, epsilon: float = DENOMINATOR_EPSILON) -> float | None:
    denominator = float(clean) - float(corrupt)
    if not math.isfinite(denominator) or abs(denominator) <= epsilon:
        return None
    value = (float(patched) - float(corrupt)) / denominator
    return value if math.isfinite(value) else None


def bh_fdr(p_values: Sequence[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite in [0,1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output.tolist()


def sign_flip_p(gains: Sequence[float], *, repeats: int, seed: int) -> float:
    values = np.asarray(gains, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or repeats < 1:
        raise ValueError("Invalid paired sign-flip input")
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(repeats):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(values))
        extreme += abs(float(np.mean(values * signs))) >= observed
    return float((extreme + 1) / (repeats + 1))


def bootstrap_ratio(
    clean: Sequence[float], corrupt: Sequence[float], patched: Sequence[float],
    *, repeats: int, seed: int, epsilon: float = DENOMINATOR_EPSILON,
) -> tuple[dict[str, Any], list[dict[str, float | int | None]]]:
    c = np.asarray(clean, dtype=np.float64)
    x = np.asarray(corrupt, dtype=np.float64)
    p = np.asarray(patched, dtype=np.float64)
    if not (len(c) and c.shape == x.shape == p.shape and np.isfinite(np.concatenate([c, x, p])).all()):
        raise ValueError("Bootstrap vectors must be equally-sized and finite")
    disruption = c - x
    gain = p - x
    observed_recovery = recovery(float(c.mean()), float(x.mean()), float(p.mean()), epsilon=epsilon)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | None]] = []
    for repeat in range(repeats):
        indices = rng.integers(0, len(c), size=len(c))
        cm, xm, pm = float(c[indices].mean()), float(x[indices].mean()), float(p[indices].mean())
        rows.append({"repeat": repeat, "clean": cm, "corrupt": xm, "patched": pm,
                     "disruption": cm - xm, "patch_gain": pm - xm,
                     "recovery": recovery(cm, xm, pm, epsilon=epsilon)})
    def stats(name: str, observed: float | None) -> dict[str, Any]:
        vals = np.asarray([float(row[name]) for row in rows if row[name] is not None and math.isfinite(float(row[name]))])
        return {
            "value": observed, "sem": float(vals.std(ddof=1)) if len(vals) > 1 else None,
            "ci_low": float(np.percentile(vals, 2.5)) if len(vals) else None,
            "ci_high": float(np.percentile(vals, 97.5)) if len(vals) else None,
            "valid_bootstrap_repeats": int(len(vals)),
        }
    summary = {
        "clean": stats("clean", float(c.mean())), "corrupt": stats("corrupt", float(x.mean())),
        "patched": stats("patched", float(p.mean())),
        "disruption": stats("disruption", float(disruption.mean())),
        "patch_gain": stats("patch_gain", float(gain.mean())),
        "recovery": stats("recovery", observed_recovery),
        "sample_count": int(len(c)), "item_count": int(len(c)),
        "undefined_item_recovery_count": int(sum(recovery(a, b, d, epsilon=epsilon) is None for a, b, d in zip(c, x, p))),
    }
    return summary, rows
