from __future__ import annotations

import pytest

from dp_SA.attention_block.relay_rescue import CONDITIONS, relay_edges, rescue_values


def _spans():
    return {
        "ANSWER": [5, 7], "PANL": 7, "PANL_PLUS_1": 8, "SAC": 10,
        "EVIDENCE": [1, 2], "ALL_CONTENT": [1, 2, 5, 6], "ALL_DOWNSTREAM_OF_PANL": [8, 9, 10],
    }


def test_relay_edges_restore_only_registered_query_answer_edges():
    spans = _spans(); answer = {5, 6}
    full = set(relay_edges(spans, CONDITIONS[0]).pairs)
    keep = set(relay_edges(spans, CONDITIONS[1]).pairs)
    control = set(relay_edges(spans, CONDITIONS[2]).pairs)
    assert full - keep == {(7, source) for source in answer}
    assert full - control == {(8, source) for source in answer}
    assert len(full) == len(keep) + 2 == len(control) + 2


def test_relay_rescue_formula_and_matched_difference():
    relay, control, contrast = rescue_values(0.8, 0.3, 0.6)
    assert relay == pytest.approx(0.5)
    assert control == pytest.approx(0.2)
    assert contrast == pytest.approx(0.3)
