from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import t as student_t

from .config import MIDPOINTS, SOFT_SA_NO_CHANGE_TOLERANCE


def score_logits(logits: Sequence[float], *, clean_class: int | None = None) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("Expected nine finite class logits")
    shifted = values - values.max()
    probabilities = np.exp(shifted); probabilities /= probabilities.sum()
    hard = int(np.argmax(values)); target = hard if clean_class is None else int(clean_class)
    if target not in range(9):
        raise ValueError("clean class outside 0..8")
    margin = float(values[target] - np.delete(values, target).mean())
    return {
        "class_logits": values.tolist(), "class_probabilities": probabilities.tolist(),
        "soft_sa": float(np.dot(probabilities, np.asarray(MIDPOINTS))), "hard_class": hard,
        "hard_midpoint": float(MIDPOINTS[hard]), "fixed_clean_class_margin": margin,
    }


def directional_metrics(delta_sa: float, target_sign: int, *, tolerance: float = SOFT_SA_NO_CHANGE_TOLERANCE) -> dict[str, Any]:
    delta = float(delta_sa); sign = int(target_sign)
    if sign not in (-1, 1) or not math.isfinite(delta) or tolerance < 0:
        raise ValueError("Invalid directional metric input")
    effective = 0.0 if abs(delta) <= tolerance else delta
    oriented = float(sign * effective)
    toward = float(max(oriented, 0.0)); wrong = float(max(-oriented, 0.0))
    if not math.isclose(toward - wrong, oriented, rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError("Directional decomposition failed")
    return {
        "delta_sa": delta, "effective_delta_sa": effective, "raw_absolute_delta_sa": abs(delta),
        "oriented_delta_sa": oriented, "toward_target_absolute_delta_sa": toward,
        "wrong_direction_absolute_delta_sa": wrong, "toward_target": bool(oriented > 0),
        "wrong_way": bool(oriented < 0), "no_change": bool(oriented == 0),
    }


def hard_direction(clean_class: int, swap_class: int, target_sign: int) -> dict[str, Any]:
    signed = int(target_sign) * (int(swap_class) - int(clean_class))
    return {"hard_class_changed": bool(swap_class != clean_class), "hard_class_toward_target": bool(signed > 0), "hard_class_wrong_way": bool(signed < 0)}


def stable_seed(seed: int, *parts: object) -> int:
    import hashlib
    digest = hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def item_bootstrap(rows: Sequence[dict[str, Any]], value: Callable[[dict[str, Any]], float], *, repeats: int, seed: int) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["item_id"])].append(row)
    items = sorted(buckets)
    if not items or repeats < 1:
        raise ValueError("Item bootstrap needs rows and positive repeats")
    observed_values = np.asarray([value(row) for row in rows], dtype=np.float64)
    if not np.isfinite(observed_values).all():
        raise ValueError("Non-finite bootstrap values")
    rng = np.random.default_rng(seed); samples = []
    for _ in range(repeats):
        selected = rng.choice(items, len(items), replace=True)
        sampled = [row for item in selected for row in buckets[str(item)]]
        result = float(np.mean([value(row) for row in sampled]))
        if math.isfinite(result):
            samples.append(result)
    if not samples:
        raise RuntimeError("No valid bootstrap repetitions")
    low, high = np.percentile(np.asarray(samples), [2.5, 97.5])
    sample_array = np.asarray(samples)
    return {
        "mean": float(observed_values.mean()),
        "sem": float(sample_array.std(ddof=1)) if len(sample_array) > 1 else None,
        "ci_low": float(low), "ci_high": float(high), "valid_bootstrap_repeats": len(samples),
        "pair_count": len({str(row.get("pair_id")) for row in rows}), "item_count": len(items), "observation_count": len(rows),
    }


def bh_fdr(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(values); adjusted = np.empty(len(values)); running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = int(order[reverse_rank]); rank = reverse_rank + 1
        running = min(running, float(values[index]) * len(values) / rank); adjusted[index] = running
    return adjusted.tolist()


def sign_flip_p(rows: Sequence[dict[str, Any]], field: str, *, repeats: int, seed: int) -> float:
    by_item: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_item[str(row["item_id"])].append(float(row[field]))
    values = np.asarray([np.mean(part) for part in by_item.values()], dtype=float)
    observed = abs(float(values.mean())); rng = np.random.default_rng(seed); extreme = 0
    for _ in range(repeats):
        if abs(float(np.mean(values * rng.choice((-1.0, 1.0), len(values))))) >= observed:
            extreme += 1
    return float((extreme + 1) / (repeats + 1))


def clustered_ols(design: np.ndarray, target: np.ndarray, groups: Sequence[str]) -> dict[str, Any]:
    X, y = np.asarray(design, float), np.asarray(target, float)
    if X.ndim != 2 or y.shape != (len(X),) or np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError("Invalid or rank-deficient OLS design")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]; residual = y - X @ beta
    bread = np.linalg.pinv(X.T @ X); meat = np.zeros((X.shape[1], X.shape[1])); unique = sorted(set(map(str, groups)))
    group_array = np.asarray(list(map(str, groups)), object)
    for group in unique:
        selected = group_array == group; score = X[selected].T @ residual[selected]; meat += np.outer(score, score)
    n, k, clusters = len(X), X.shape[1], len(unique)
    correction = (clusters / (clusters - 1)) * ((n - 1) / (n - k)) if clusters > 1 and n > k else 1.0
    covariance = correction * bread @ meat @ bread; se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    statistic = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p = 2 * student_t.sf(np.abs(statistic), df=max(clusters - 1, 1))
    sst = float(np.square(y - y.mean()).sum()); sse = float(np.square(residual).sum())
    return {"coefficient": beta, "standard_error": se, "covariance": covariance, "p_value": p, "r2": float(1 - sse / sst) if sst > 0 else float("nan"), "cluster_count": clusters}
