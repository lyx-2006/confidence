from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


def restricted_probabilities(scores: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) != 12 or not np.isfinite(values).all() or temperature <= 0:
        raise ValueError("Expected twelve finite scores and positive temperature")
    scaled = values / float(temperature)
    result = np.exp(scaled - logsumexp(scaled))
    if abs(float(result.sum()) - 1.0) > 1e-10: raise ValueError("Probability sum failure")
    return result


def entropy_difficulty(probabilities: Sequence[float]) -> float:
    p = np.asarray(probabilities, dtype=np.float64); positive = p[p > 0]
    return float(100.0 * -np.sum(positive * np.log(positive)) / math.log(len(p)))


def ece(confidence: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    c = np.asarray(confidence, dtype=float); y = np.asarray(correct, dtype=float)
    if c.shape != y.shape or c.ndim != 1 or not len(c): raise ValueError("Invalid ECE inputs")
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for index in range(bins):
        chosen = (c >= edges[index]) & (c < edges[index + 1] if index < bins - 1 else c <= edges[index + 1])
        if chosen.any(): total += float(chosen.mean()) * abs(float(y[chosen].mean()) - float(c[chosen].mean()))
    return total


def score_metrics(candidates: Sequence[str], scores: Sequence[float], target: str, temperature: float = 1.0) -> dict[str, Any]:
    names = list(candidates); p = restricted_probabilities(scores, temperature); ti = names.index(target); pi = int(np.argmax(p)); one = np.zeros(12); one[ti] = 1
    return {"probabilities": {name: float(p[i]) for i, name in enumerate(names)}, "probability_sum": float(p.sum()),
            "chosen_answer": names[pi], "chosen_confidence": float(p[pi]), "correct": pi == ti,
            "nll": float(-math.log(max(float(p[ti]), np.finfo(float).tiny))), "brier": float(np.square(p-one).sum())}


def calibration_summary(rows: Sequence[dict[str, Any]], probabilities_field: str = "calibrated_probabilities") -> dict[str, Any]:
    correct = np.asarray([bool(row["correct"]) for row in rows]); confidence = np.asarray([float(row["chosen_confidence"]) for row in rows])
    nll = np.asarray([float(row["nll"]) for row in rows]); brier = np.asarray([float(row["brier"]) for row in rows])
    return {"count": len(rows), "ece": ece(confidence, correct), "nll": float(nll.mean()), "brier": float(brier.mean()),
            "accuracy": float(correct.mean()), "auroc": None if len(set(correct.tolist())) < 2 else float(roc_auc_score(correct, confidence))}


def regression_values(y: Sequence[float], prediction: Sequence[float]) -> dict[str, float]:
    a=np.asarray(y,float); b=np.asarray(prediction,float); denominator=float(np.square(a-a.mean()).sum())
    if denominator <= 0: raise ValueError("Constant target")
    return {"r2": float(1-np.square(a-b).sum()/denominator), "spearman": float(spearmanr(a,b).statistic), "pearson": float(pearsonr(a,b).statistic)}


def family_bootstrap(rows: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> dict[str, dict[str, Any]]:
    groups=defaultdict(list)
    for row in rows: groups[str(row["family_id"])].append(row)
    families=sorted(groups); rng=np.random.default_rng(seed); values=defaultdict(list)
    for _ in range(repeats):
        sample=[row for family in rng.choice(families,len(families),replace=True) for row in groups[str(family)]]
        try: cell=regression_values([r["true"] for r in sample],[r["predicted"] for r in sample])
        except (ValueError,FloatingPointError): continue
        for name,value in cell.items():
            if math.isfinite(value): values[name].append(value)
    result={}
    for name in ("r2","spearman","pearson"):
        if values[name]: low,high=np.percentile(values[name],[2.5,97.5]); result[name]={"low":float(low),"high":float(high),"valid":len(values[name])}
        else: result[name]={"low":None,"high":None,"valid":0}
    return result
