from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[2]
for candidate in (REPOSITORY_ROOT, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from analyze import _multiple_fit, _robustness, _simple_fit


def test_simple_fit_and_robustness() -> None:
    fit = _simple_fit(np.asarray([-1.0, 0.0, 1.0]), np.asarray([-.5, 0.0, .5]))
    assert abs(fit["slope"] - .5) < 1e-12
    assert fit["r2"] == 1.0
    rows = [
        {"cma_signed": -.8, "cma_log_probability_signed": -.7},
        {"cma_signed": .3, "cma_log_probability_signed": .4},
        {"cma_signed": .8, "cma_log_probability_signed": .9},
    ]
    result = _robustness(rows)
    assert result["n"] == 3
    assert result["sign_agreement_rate"] == 1.0


def test_difficulty_interaction_and_adjustment_designs() -> None:
    rows = []
    for index, (difficulty, cma) in enumerate((
        ("easy", -.8), ("easy", .8), ("hard", -.7), ("hard", .7),
        ("easy", -.3), ("hard", .3), ("easy", .2), ("hard", -.2),
    )):
        rows.append({
            "difficulty": difficulty, "cma_signed": cma, "sa_signed": .5 * cma,
            "original_text_entropy": .1 + index * .01,
            "original_image_entropy": .2 + index * .01,
            "text_entropy_match_error": .01 + index * .001,
            "text_probability_match_error": .02 + index * .001,
        })
    assert _multiple_fit(rows, adjusted=False)["n"] == 8
    assert _multiple_fit(rows, adjusted=True)["n"] == 8
