from __future__ import annotations

import numpy as np

from dp_SA.confidence_steering.core import continuous_pattern, natural_sa_decomposition

from ..rebuild import direction_sign_audit, signed_cosine


def test_continuous_pattern_positive_direction_tracks_target_increase():
    target = np.asarray([-1.0, 0.0, 1.0])
    hidden = np.zeros((3, 3584))
    hidden[:, 0] = target
    hidden[:, 1] = -target
    pattern = continuous_pattern(hidden, target)
    assert pattern[0] > 0 and pattern[1] < 0


def test_negative_probe_dot_is_reported_without_flip():
    vectors = {
        "confidence_raw": np.asarray([-1.0, 0.0], np.float32),
        "confidence_parallel_sa": np.asarray([0.0, 0.0], np.float32),
        "confidence_perp_sa_natural_scale": np.asarray([-1.0, 0.0], np.float32),
    }
    before = vectors["confidence_raw"].copy()
    audit = direction_sign_audit(np.asarray([1.0, 0.0]), vectors)
    assert audit["direction_sign_inconsistent"] is True
    assert np.array_equal(vectors["confidence_raw"], before)


def test_zero_probe_dot_is_degenerate_not_negative():
    vectors = {name: np.asarray([0.0, 1.0], np.float32) for name in (
        "confidence_raw", "confidence_parallel_sa", "confidence_perp_sa_natural_scale")}
    audit = direction_sign_audit(np.asarray([1.0, 0.0]), vectors)
    assert audit["direction_sign_degenerate"] and not audit["direction_sign_inconsistent"]


def test_natural_decomposition_uses_common_scale():
    vector = np.asarray([3.0, 4.0])
    basis = np.asarray([[1.0], [0.0]])
    result = natural_sa_decomposition(vector, basis, 10.0)
    assert result["common_scale"] == 2.0
    assert np.allclose(result["parallel_scaled"], result["parallel"] * 2.0)
    assert np.allclose(result["perpendicular_scaled"], result["perpendicular"] * 2.0)


def test_natural_components_reconstruct_raw():
    result = natural_sa_decomposition(np.asarray([3.0, 4.0]), np.asarray([[1.0], [0.0]]), 10.0)
    assert np.allclose(result["raw"], result["parallel_scaled"] + result["perpendicular_scaled"])
    assert result["reconstruction_relative_error"] < 1e-7


def test_signed_cosine_retains_opposite_sign():
    assert np.isclose(signed_cosine(np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0])), -1.0)
