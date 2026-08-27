from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr, t as student_t
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def restricted_distribution(sequence_log_probabilities: Sequence[float]) -> np.ndarray:
    scores = np.asarray(sequence_log_probabilities, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("Candidate sequence scores must be a finite vector")
    probabilities = np.exp(scores - logsumexp(scores))
    if abs(float(probabilities.sum()) - 1.0) > 1e-10:
        raise ValueError("Restricted probabilities do not sum to one")
    return probabilities


def model_perceived_difficulty(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("Probabilities must be a finite non-negative vector")
    if abs(float(values.sum()) - 1.0) > 1e-9:
        raise ValueError("Probabilities must sum to one")
    positive = values[values > 0]
    return float(100.0 * (-np.sum(positive * np.log(positive))) / math.log(len(values)))


def candidate_metrics(candidates: Sequence[str], scores: Sequence[float], target: str) -> dict[str, Any]:
    names = [str(value) for value in candidates]
    if len(names) != len(set(names)) or target not in names:
        raise ValueError("Candidates must be unique and contain the target")
    probabilities = restricted_distribution(scores)
    target_index = names.index(target)
    one_hot = np.zeros(len(names), dtype=np.float64)
    one_hot[target_index] = 1.0
    predicted_index = int(np.argmax(probabilities))
    return {
        "candidate_sequence_log_probabilities": {name: float(scores[i]) for i, name in enumerate(names)},
        "candidate_restricted_probabilities": {name: float(probabilities[i]) for i, name in enumerate(names)},
        "probability_sum": float(probabilities.sum()),
        "predicted_answer": names[predicted_index],
        "target_answer": target,
        "correct": bool(predicted_index == target_index),
        "multiclass_nll": float(-math.log(max(float(probabilities[target_index]), np.finfo(float).tiny))),
        "brier_score": float(np.square(probabilities - one_hot).sum()),
        "model_perceived_difficulty": model_perceived_difficulty(probabilities),
        "max_probability": float(probabilities[predicted_index]),
        "max_probability_percent": float(100.0 * probabilities[predicted_index]),
    }


def expected_calibration_error(confidence: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    conf = np.asarray(confidence, dtype=np.float64)
    outcome = np.asarray(correct, dtype=np.float64)
    if conf.shape != outcome.shape or conf.ndim != 1 or not len(conf):
        raise ValueError("ECE inputs must be non-empty vectors of equal length")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        selected = (conf >= edges[index]) & (conf < edges[index + 1] if index < bins - 1 else conf <= edges[index + 1])
        if selected.any():
            total += float(selected.mean()) * abs(float(outcome[selected].mean()) - float(conf[selected].mean()))
    return float(total)


def calibration_metrics(rows: Sequence[dict[str, Any]], prefix: str, bins: int = 10) -> dict[str, Any]:
    if not rows:
        raise ValueError("Calibration requires records")
    difficulty = np.asarray([float(row[f"{prefix}_model_perceived_difficulty"]) for row in rows])
    correct = np.asarray([bool(row[f"{prefix}_correct"]) for row in rows])
    confidence = np.asarray([float(row[f"{prefix}_max_probability"]) for row in rows])
    error = ~correct
    auroc = None if len(set(error.tolist())) < 2 else float(roc_auc_score(error, difficulty))
    rho = None if len(set(difficulty.tolist())) < 2 else float(spearmanr(difficulty, correct.astype(float)).statistic)
    order = np.argsort(difficulty, kind="stable")
    deciles = []
    for decile, indices in enumerate(np.array_split(order, 10), 1):
        deciles.append({"decile": decile, "count": int(len(indices)), "difficulty_mean": float(difficulty[indices].mean()), "error_rate": float(error[indices].mean())})
    return {
        "sample_count": len(rows),
        "restricted_top1_accuracy": float(correct.mean()),
        "multiclass_nll": float(np.mean([row[f"{prefix}_multiclass_nll"] for row in rows])),
        "brier_score": float(np.mean([row[f"{prefix}_brier_score"] for row in rows])),
        "max_probability_ece": expected_calibration_error(confidence, correct, bins),
        "difficulty_error_auroc": auroc,
        "difficulty_correctness_spearman": rho,
        "difficulty_deciles": deciles,
        "calibrated_difficulty_supported": bool(auroc is not None and auroc > 0.5 and rho is not None and rho < 0),
    }


def difficulty_factors(text_difficulty: float, image_difficulty: float) -> dict[str, float]:
    dt, di = float(text_difficulty) / 100.0, float(image_difficulty) / 100.0
    return {"d_text": dt, "d_image": di, "G": dt - di, "U": (dt + di) / 2.0}


def item_bootstrap(
    rows: Sequence[dict[str, Any]], statistic: Callable[[list[dict[str, Any]]], float], *, repeats: int, seed: int
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["item_id"])].append(row)
    items = sorted(buckets)
    if not items:
        return {"lower": None, "upper": None, "valid_repeats": 0, "item_count": 0}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repeats):
        chosen = rng.choice(items, len(items), replace=True)
        sample = [row for item in chosen for row in buckets[str(item)]]
        try:
            value = float(statistic(sample))
        except (ValueError, FloatingPointError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return {"lower": None, "upper": None, "valid_repeats": 0, "item_count": len(items)}
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"lower": float(lower), "upper": float(upper), "valid_repeats": len(values), "item_count": len(items)}


def pooled_r2(y: Sequence[float], prediction: Sequence[float]) -> float:
    truth = np.asarray(y, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.square(truth - truth.mean()).sum())
    if denominator <= 0:
        raise ValueError("R2 target is constant")
    return float(1.0 - np.square(truth - pred).sum() / denominator)


def bh_fdr(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0,1]")
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, float(values[index]) * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def clustered_ols(X: np.ndarray, y: np.ndarray, groups: Sequence[str]) -> dict[str, Any]:
    design = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    group_values = np.asarray(groups, dtype=object)
    if design.ndim != 2 or target.shape != (len(design),) or len(group_values) != len(design):
        raise ValueError("Invalid OLS shapes")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("OLS design matrix is rank deficient")
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = design @ beta
    residual = target - fitted
    bread = np.linalg.pinv(design.T @ design)
    unique = sorted(set(str(value) for value in group_values))
    meat = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
    for group in unique:
        selected = np.asarray([str(value) == group for value in group_values])
        score = design[selected].T @ residual[selected]
        meat += np.outer(score, score)
    n, k, cluster_count = len(design), design.shape[1], len(unique)
    correction = (cluster_count / (cluster_count - 1)) * ((n - 1) / (n - k)) if cluster_count > 1 and n > k else 1.0
    covariance = correction * bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    statistic = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_value = 2.0 * student_t.sf(np.abs(statistic), df=max(cluster_count - 1, 1))
    sst = float(np.square(target - target.mean()).sum())
    sse = float(np.square(residual).sum())
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    adjusted = float(1.0 - (1.0 - r2) * (n - 1) / (n - k)) if n > k else float("nan")
    return {"coefficient": beta, "fitted": fitted, "residual": residual, "standard_error": se, "p_value": p_value, "r2": r2, "adjusted_r2": adjusted, "cluster_count": cluster_count}


def safe_correlation(kind: str, left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        raise ValueError("Correlation is undefined")
    result = pearsonr(x, y) if kind == "pearson" else spearmanr(x, y)
    return float(result.statistic)


def decision_direction(y: Sequence[int], probability: Sequence[float]) -> float:
    truth, score = np.asarray(y), np.asarray(probability)
    if len(set(truth.tolist())) < 2:
        raise ValueError("Decision direction requires both classes")
    return float(score[truth == 1].mean() - score[truth == 0].mean())
