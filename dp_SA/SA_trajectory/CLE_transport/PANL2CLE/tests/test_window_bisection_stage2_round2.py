from __future__ import annotations

import torch

from dp_SA.SA_trajectory.CLE_transport.PANL2CLE.window_bisection_stage2_round2 import (
    EXPECTED_NEW_CLEAN, EXPECTED_NEW_PATCHED, SELECTED_CELLS, SPLITS, load_layer_cle_probes, split_positions,
)


def test_round2_contract_and_counts():
    assert SELECTED_CELLS == (("W2", 17), ("W5", 16))
    assert set(SPLITS[("W2", 17)]) == {"right4_left2", "right4_right2"}
    assert len(SPLITS[("W5", 16)]) == 4
    assert EXPECTED_NEW_PATCHED == 6 * 50 * 2
    assert EXPECTED_NEW_CLEAN == 50


def test_round2_positions_are_two_token_children():
    positions = list(range(200, 208))
    assert split_positions(positions, (4, 6)) == [204, 205]
    assert split_positions(positions, (6, 8)) == [206, 207]
    try:
        split_positions(positions, (0, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("non-two-token split accepted")


def test_layer_specific_cle_probes_are_reliable_and_not_cross_layer():
    probes = load_layer_cle_probes()
    assert set(probes) == {16, 17}
    for layer, (_model, record) in probes.items():
        assert record["target"] == "final_soft_sa"
        assert record["position"] == "P1_CLASS_LIST_END"
        assert record["layer"] == layer
        assert record["readout_reliable"] and record["raw_expression_strict_pass"]
