from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 100.0
TEMPERATURE_GRID_SIZE = 4096
LOG_ODDS_EPSILON = 1e-6


def softmax_temperature(logits: Sequence[float], temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= np.max(values)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    return probabilities


def multiclass_metrics(
    rows: Sequence[dict[str, Any]], temperature: float, *, ece_bins: int = 10,
) -> dict[str, float]:
    if not rows:
        raise ValueError("temperature calibration requires at least one row")
    nll: list[float] = []
    brier: list[float] = []
    confidence: list[float] = []
    correct: list[float] = []
    for row in rows:
        classes = list(row["answer_classes"])
        target = str(row["target_answer"])
        if target not in classes:
            raise ValueError(f"target outside answer classes: {target}")
        probabilities = softmax_temperature(
            [float(row["raw_candidate_scores"][name]) for name in classes], temperature,
        )
        target_index = classes.index(target)
        prediction = int(np.argmax(probabilities))
        confidence.append(float(probabilities[prediction]))
        correct.append(float(prediction == target_index))
        nll.append(float(-math.log(max(float(probabilities[target_index]), 1e-300))))
        one_hot = np.zeros(len(classes), dtype=np.float64); one_hot[target_index] = 1.0
        brier.append(float(np.sum((probabilities - one_hot) ** 2)))
    confidence_array = np.asarray(confidence); correct_array = np.asarray(correct)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    for index in range(ece_bins):
        if index == ece_bins - 1:
            mask = (confidence_array >= edges[index]) & (confidence_array <= edges[index + 1])
        else:
            mask = (confidence_array >= edges[index]) & (confidence_array < edges[index + 1])
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(confidence_array[mask])) - float(np.mean(correct_array[mask])))
    return {
        "nll": float(np.mean(nll)), "ece": float(ece),
        "brier": float(np.mean(brier)), "accuracy": float(np.mean(correct)),
    }


def temperature_grid() -> np.ndarray:
    return np.unique(np.r_[
        np.geomspace(TEMPERATURE_MIN, TEMPERATURE_MAX, TEMPERATURE_GRID_SIZE), 1.0,
    ])


def fit_nll_temperature(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, float]]]:
    trace = []
    for temperature in temperature_grid():
        trace.append({"temperature": float(temperature), **multiclass_metrics(rows, float(temperature))})
    best = min(trace, key=lambda row: (row["nll"], row["ece"], abs(math.log(row["temperature"]))))
    baseline = next(row for row in trace if row["temperature"] == 1.0)
    if best["nll"] > baseline["nll"] + 1e-12:
        raise AssertionError("NLL-optimal temperature is worse than T=1")
    return dict(best), trace


def calibrated_probability(
    logit_map: dict[str, float], answer_classes: Sequence[str], answer: str, temperature: float,
) -> float:
    classes = list(answer_classes)
    if answer not in classes:
        raise ValueError(f"fixed answer outside candidate set: {answer}")
    probabilities = softmax_temperature([float(logit_map[name]) for name in classes], temperature)
    return float(probabilities[classes.index(answer)])


def clipped_log_odds(probability: float, epsilon: float = LOG_ODDS_EPSILON) -> tuple[float, str | None]:
    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"invalid probability: {value}")
    clipped = float(np.clip(value, epsilon, 1.0 - epsilon))
    side = "low" if value < epsilon else "high" if value > 1.0 - epsilon else None
    return float(math.log(clipped / (1.0 - clipped))), side


def ols_fit(rows: Sequence[dict[str, Any]], outcome: str, features: Sequence[str]) -> dict[str, Any]:
    if not rows:
        raise ValueError("OLS requires rows")
    y = np.asarray([float(row[outcome]) for row in rows], dtype=np.float64)
    x = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float64)
    design = np.column_stack([np.ones(len(rows)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ beta
    residual = y - prediction
    sse = float(np.sum(residual ** 2)); total = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - sse / total) if total > 0 else float("nan")
    p = len(features); n = len(rows)
    adjusted = float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else float("nan")
    y_std = float(np.std(y, ddof=0))
    standardized = {
        name: float(beta[index + 1] * np.std(x[:, index], ddof=0) / y_std) if y_std > 0 else float("nan")
        for index, name in enumerate(features)
    }
    return {
        "intercept": float(beta[0]),
        "coefficients": {name: float(beta[index + 1]) for index, name in enumerate(features)},
        "standardized_coefficients": standardized,
        "prediction": prediction,
        "r2": r2, "adjusted_r2": adjusted,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
    }


def partial_r2(full_r2: float, reduced_r2: float) -> float:
    denominator = 1.0 - float(reduced_r2)
    return float((float(full_r2) - float(reduced_r2)) / denominator) if denominator > 1e-12 else float("nan")


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in values for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise
