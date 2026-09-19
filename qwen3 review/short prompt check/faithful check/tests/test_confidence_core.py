from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from confidence_core import (  # noqa: E402
    calibrated_probability,
    clipped_log_odds,
    multiclass_metrics,
    ols_fit,
    partial_r2,
    softmax_temperature,
)


def _rows() -> list[dict]:
    return [
        {
            "answer_classes": ["red", "blue", "green"],
            "target_answer": target,
            "raw_candidate_scores": {"red": 3.0 if target == "red" else 0.0, "blue": 3.0 if target == "blue" else 0.0, "green": 3.0 if target == "green" else 0.0},
        }
        for target in ("red", "blue", "green", "red", "blue", "green")
    ]


def test_temperature_softmax_and_calibrated_probability() -> None:
    probabilities = softmax_temperature([0.0, 2.0], 2.0)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.isclose(calibrated_probability({"red": 0.0, "blue": 2.0}, ["red", "blue"], "blue", 2.0), probabilities[1])


def test_nll_temperature_includes_baseline_and_metrics_are_finite() -> None:
    result = multiclass_metrics(_rows(), 1.0)
    assert result["nll"] > 0
    assert 0 <= result["ece"] <= 1
    assert math.isfinite(result["brier"])


def test_log_odds_clip_and_regression_contrasts() -> None:
    value, side = clipped_log_odds(1.0)
    assert side == "high" and np.isclose(value, math.log((1 - 1e-6) / 1e-6))
    rows = [{"y": float(i), "x": float(i), "z": float(2 * i)} for i in range(1, 8)]
    fit = ols_fit(rows, "y", ("x",))
    assert np.isclose(fit["r2"], 1.0)
    assert np.isclose(partial_r2(0.5, 0.25), 1 / 3)
