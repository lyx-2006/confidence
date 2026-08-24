from __future__ import annotations

import pytest

from dp_SA.attention_block.plot_figure7 import build_summary_rows, logit_difference


def test_logit_difference_uses_mean_of_other_eight_digits():
    assert logit_difference([8, 0, 0, 0, 0, 0, 0, 0, 0], 0) == pytest.approx(8.0)
    assert logit_difference([0, 8, 0, 0, 0, 0, 0, 0, 0], 0) == pytest.approx(-1.0)


def test_summary_change_sign_matches_figure7_convention():
    rows = []
    for case, changed, change in (("a", 1, -2.0), ("b", 0, 0.0)):
        rows.append({
            "arm": "joint", "phase": "coarse", "source_run": "x", "condition": "sac_to_panl",
            "condition_label": "SAC→PANL", "window_start": 0, "window_end": 11, "window_center": 5.5,
            "blocked_layer_count": 12, "is_sliding_window": 1, "test_side": "image_side",
            "case_id": case, "token_changed": changed, "logit_difference_change": change,
            "logit_margin_disruption": -change,
        })
    summary = [row for row in build_summary_rows(rows) if row["group"] == "all_100"][0]
    assert summary["token_change_rate_pct"] == pytest.approx(50.0)
    assert summary["logit_difference_change_mean"] == pytest.approx(-1.0)
