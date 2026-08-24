from __future__ import annotations

import pytest

from dp_SA.attention_block.relay_metrics import recovery_proportion


def test_recovery_proportion_is_fraction_of_full_disruption_removed():
    assert recovery_proportion(0.2, 0.1) == pytest.approx(50.0)
    assert recovery_proportion(0.2, 0.25) == pytest.approx(-25.0)
    with pytest.raises(ZeroDivisionError):
        recovery_proportion(0.0, 0.0)
