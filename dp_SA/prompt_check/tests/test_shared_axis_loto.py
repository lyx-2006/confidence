from __future__ import annotations

import numpy as np

from dp_SA.prompt_check.shared_axis_loto import (
    DEFAULT_NODES, equal_dose_vector, matched_random_directions,
    parse_nodes, retention_eligible, uncentered_shared_axis,
)


def test_uncentered_svd_and_sign_rule():
    units = [np.array([1., 0., 0.]), np.array([.8, .6, 0.]), np.array([.8, 0., .6])]
    axis, singular = uncentered_shared_axis(units)
    assert axis.shape == (3,) and singular.shape == (3,)
    assert np.mean(np.stack(units) @ axis) > 0
    assert np.all(np.stack(units) @ axis > 0)


def test_equal_dose_has_exact_shared_scale_for_any_unit_direction():
    rng = np.random.default_rng(2)
    for _ in range(5):
        unit = rng.normal(size=64); unit /= np.linalg.norm(unit)
        displacement = equal_dose_vector(unit, 1.7, -2)
        assert abs(np.linalg.norm(displacement.astype(float)) - 3.4) / 3.4 < 1e-6


def test_retention_denominator_gate():
    assert retention_eligible(.2, np.linspace(.1, .3, 2000))["eligible"] is True
    assert retention_eligible(.01, np.linspace(-.2, .2, 2000))["eligible"] is False


def test_random_directions_are_unit_and_sd_matched():
    rng = np.random.default_rng(4); hidden = rng.normal(size=(300, 64))
    directions, rows = matched_random_directions(hidden, 1.0, seed=8, count=3)
    assert directions.shape == (3, 64)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert all(row["relative_sd_error"] <= .05 for row in rows)


def test_preregistered_node_parser_is_exact():
    values = [f"{position}:{layer}" for position, layer in DEFAULT_NODES]
    assert parse_nodes(values) == DEFAULT_NODES
