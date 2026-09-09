from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, r2_score


def shared_family_draws(families: Sequence[str], repeats: int, seed: int) -> tuple[list[str], np.ndarray]:
    ordered = sorted(set(map(str, families)))
    if len(ordered) < 2: raise ValueError("At least two families are required")
    return ordered, np.random.default_rng(seed).integers(0, len(ordered), size=(repeats, len(ordered)))


def sampled_indices(families: Sequence[str], ordered: Sequence[str], draw: Sequence[int]) -> np.ndarray:
    family_array = np.asarray(families, dtype=str); by_family = {name: np.flatnonzero(family_array == name) for name in ordered}
    return np.concatenate([by_family[ordered[int(index)]] for index in draw])


def regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if len(y) != len(p) or len(y) < 2: raise ValueError("Metric vectors must be paired")
    pearson = math.nan if np.ptp(y) == 0 or np.ptp(p) == 0 else float(pearsonr(y, p).statistic)
    spearman = math.nan if len(np.unique(y)) < 2 or len(np.unique(p)) < 2 else float(spearmanr(y, p).statistic)
    return {"r2": float(r2_score(y, p)), "pearson": pearson, "spearman": spearman, "mae": float(mean_absolute_error(y, p))}


def behavior_metrics(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    x, y = np.asarray(left, float), np.asarray(right, float)
    base = regression_metrics(x, y)
    if np.ptp(x)==0:slope,intercept=math.nan,math.nan
    else:
        slope=float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2));intercept=float(y.mean()-slope*x.mean())
    base.update({"mean_signed_difference": float(np.mean(y - x)), "slope": float(slope), "intercept": float(intercept), "direction_agreement": float(np.mean(np.sign(x - .5) == np.sign(y - .5)))})
    base.pop("r2")
    return base


def hard_agreement(left: Sequence[int], right: Sequence[int], *, within_one: bool) -> dict[str, float]:
    x, y = np.asarray(left, int), np.asarray(right, int)
    result = {"exact_agreement": float(np.mean(x == y)), "quadratic_weighted_kappa": float(cohen_kappa_score(x, y, weights="quadratic"))}
    if within_one: result["within_one_agreement"] = float(np.mean(np.abs(x-y) <= 1))
    return result


def bootstrap_rows(metric_fn: Callable[[np.ndarray], dict[str, float]], families: Sequence[str], ordered: Sequence[str], draws: np.ndarray) -> tuple[list[dict[str, float]], int]:
    values = []
    for draw in draws:
        result = metric_fn(sampled_indices(families, ordered, draw))
        if result and all(np.isfinite(list(result.values()))): values.append(result)
    return values, len(values)


def add_ci(row: dict[str, float], boots: Sequence[dict[str, float]], metrics: Sequence[str]) -> dict[str, float]:
    output = dict(row)
    for metric in metrics:
        finite = [float(value[metric]) for value in boots if metric in value and np.isfinite(value[metric])]
        low, high = np.quantile(finite, [.025, .975]) if finite else (math.nan, math.nan)
        output[f"{metric}_ci_low"] = float(low); output[f"{metric}_ci_high"] = float(high)
    return output


def retention_eligibility(point: float, denominators: Sequence[float]) -> dict[str, float | bool]:
    values = np.asarray(denominators, float); finite = values[np.isfinite(values)]
    if not len(finite): return {"eligible": False, "ci_low": math.nan, "ci_high": math.nan, "same_sign_fraction": math.nan}
    low, high = np.quantile(finite, [.025, .975]); same = float(np.mean(np.sign(finite) == np.sign(point)))
    eligible = bool((low > 0 or high < 0) and same >= .975 and point != 0)
    return {"eligible": eligible, "ci_low": float(low), "ci_high": float(high), "same_sign_fraction": same}
