from __future__ import annotations

from dp_SA.SA_trajectory.CLE_transport.PANL2CLE.window_bisection_w2_singleton_stage3 import (
    EXPECTED_NEW_CLEAN, EXPECTED_NEW_PATCHED, NEXT_CLE_LAYER, SINGLETONS, load_next_layer_cle_probe, singleton_positions,
)


def test_singleton_contract():
    assert SINGLETONS == {"right4_left2_token1": (4, 5), "right4_left2_token2": (5, 6)}
    assert EXPECTED_NEW_PATCHED == 2 * 50 * 2 == 200
    assert EXPECTED_NEW_CLEAN == 50
    assert singleton_positions(list(range(100, 108)), (4, 5)) == [104]
    assert singleton_positions(list(range(100, 108)), (5, 6)) == [105]


def test_next_layer_cle_probe_is_strict_and_reliable():
    _probe, record = load_next_layer_cle_probe()
    assert record["layer"] == NEXT_CLE_LAYER == 18
    assert record["position"] == "P1_CLASS_LIST_END"
    assert record["target"] == "final_soft_sa"
    assert record["readout_reliable"] and record["raw_expression_strict_pass"]
