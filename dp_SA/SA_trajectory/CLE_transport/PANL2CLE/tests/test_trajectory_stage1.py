from __future__ import annotations

from dp_SA.SA_trajectory.CLE_transport.PANL2CLE.trajectory_stage1 import NEW_CELLS, select_layers


def _row(window, layer, contrast, high, low, toward):
    return {"window": window, "layer": layer, "donor_contrast": contrast,
            "delta_high_recipient": high, "delta_low_recipient": low,
            "mean_toward_score": toward}


def test_exact_new_cell_contract():
    assert len(NEW_CELLS) == 10
    assert {layer for _window, layer in NEW_CELLS} == {13, 14, 16, 17, 19}
    assert {window for window, _layer in NEW_CELLS} == {"W2", "W5"}


def test_selection_requires_continuity_and_prefers_two_positive_strata():
    rows = []
    for window in ("W2", "W5"):
        rows.extend([
            _row(window, 13, .01, .02, .00, .01),
            _row(window, 14, .02, .02, .02, .01),
            _row(window, 15, -.01, -.01, -.01, .01),
            _row(window, 16, .05, .05, .05, .02),  # isolated peak: reject
            _row(window, 17, -.01, -.01, -.01, -.01),
            _row(window, 18, .01, .02, -.001, .01),
            _row(window, 19, .015, .01, .02, .01),
        ])
    result = select_layers(rows)
    for window in ("W2", "W5"):
        assert result["windows"][window]["qualifying_layers"] == [13, 14, 18, 19]
        assert result["windows"][window]["recommended_layer"] in (14, 19)
        isolated = next(row for row in result["windows"][window]["layers"] if row["layer"] == 16)
        assert isolated["run_length"] == 1

