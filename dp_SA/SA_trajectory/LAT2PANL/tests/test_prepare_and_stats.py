from __future__ import annotations

import numpy as np

from dp_SA.SA_trajectory.LAT2PANL.analyze import SharedBootstrap, ratio_summary
from dp_SA.SA_trajectory.LAT2PANL.prepare import prepare


def test_frozen_inputs_vector_isolation_and_cle_eligibility(tmp_path):
    result = prepare(output_root=tmp_path, smoke=True, resume=False)
    assert result["formal_case_count"] == 174
    assert result["case_count"] == result["vector_coverage"] == 24
    assert result["cle_eligible_formal"] == 73
    assert result["fold_errors"] == result["own_family_construction_leaks"] == 0
    resumed = prepare(output_root=tmp_path, smoke=True, resume=True)
    assert resumed["resumed"]


def test_ratio_requires_bootstrap_sign_stability():
    total = np.linspace(.1, .2, 2000); numerator = total * .5
    stable = ratio_summary(.15, .075, total, numerator)
    assert stable["ratio_reportable"] and np.isclose(stable["ratio"], .5)
    crossing = np.linspace(-.1, .1, 2000)
    unstable = ratio_summary(.001, .0005, crossing, crossing * .5)
    assert not unstable["ratio_reportable"] and np.isnan(unstable["ratio"])


def test_bootstrap_draws_are_shared_across_conditions():
    rows = [{"family_id": f"f{i}", "answer": "red" if i < 3 else "green"} for i in range(6)]
    design = SharedBootstrap(rows, 50, seed=42)
    values = {f"f{i}": float(i) for i in range(6)}
    _, first = design.aggregate(values, "family_micro")
    _, second = design.aggregate(values, "family_micro")
    assert np.array_equal(first, second)
