from __future__ import annotations

import inspect

import numpy as np
import pytest

from dp_SA.confidence_steering.config import (
    EXPECTED_FORMAL_20_NULL_FORWARDS, EXPECTED_FORMAL_99_NULL_FORWARDS,
    EXPECTED_FORMAL_MAIN_FORWARDS, EXPECTED_FORMAL_TRIALS, FORMAL_ROOT,
)
from dp_SA.confidence_steering.core import (
    answer_patterns, continuous_pattern, inner_fold, loao, project_out, shuffled_targets,
    svd_basis, weighted_sa_probe,
)
from dp_SA.confidence_steering.prepare import run_prepare
from dp_SA.confidence_steering.io_utils import create_output_root, semantic_fingerprint
from dp_SA.confidence_steering.run_pipeline import unseal_formal_test


def test_formal_test_is_not_referenced_by_prelock_prepare() -> None:
    source = inspect.getsource(run_prepare)
    assert "SEALED_TEST_MANIFEST" not in source and "TEST_MANIFEST" not in source


def test_fixed_forward_accounting() -> None:
    assert EXPECTED_FORMAL_TRIALS == 10_000
    assert EXPECTED_FORMAL_MAIN_FORWARDS == 8_500
    assert EXPECTED_FORMAL_20_NULL_FORWARDS == 16_500
    assert EXPECTED_FORMAL_99_NULL_FORWARDS == 48_100
    assert FORMAL_ROOT.name == "orthogonal_results"


def test_continuous_pattern_and_answer_equal_loao() -> None:
    x = np.zeros((4, 3584)); x[:, 0] = (0, 1, 2, 3); y = np.asarray((0, 2, 4, 6))
    direction = continuous_pattern(x, y)
    assert direction[0] == pytest.approx(0.5) and np.linalg.norm(direction) == pytest.approx(0.5)
    colors = ("black", "blue", "brown", "cyan", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow")
    patterns = {c: np.eye(12, 3584)[i] for i, c in enumerate(colors)}
    value, included = loao(patterns, "cyan")
    assert len(included) == 11 and "cyan" not in included and np.count_nonzero(value) == 11


def test_answer_pattern_requires_ten_independent_family_cells() -> None:
    hidden = {}; cells = []
    for color, count in (("black", 10), ("blue", 9)):
        for index in range(count):
            key = f"{color}-{index}"; value = np.zeros(3584); value[0] = index
            hidden[key] = value
            cells.append({"array_key": key, "fixed_answer_color": color, "family_id": key, "mean_G_L": float(index)})
    patterns, audit = answer_patterns(cells, hidden, "mean_G_L", return_audit=True)
    assert "black" in patterns and "blue" not in patterns
    status = {row["fixed_answer_color"]: row["valid"] for row in audit}
    assert status["black"] and not status["blue"]


def test_joint_projection_is_order_independent_and_ranked() -> None:
    rng = np.random.default_rng(4); v = rng.normal(size=3584); a = rng.normal(size=3584); b = rng.normal(size=3584)
    q1, meta1 = svd_basis([a, b]); q2, meta2 = svd_basis([b, a])
    left, right = project_out(v, q1), project_out(v, q2)
    assert meta1["rank"] == meta2["rank"] == 2 and np.allclose(left, right, atol=1e-10)
    assert abs(left @ a / np.linalg.norm(a) / np.linalg.norm(left)) < 1e-12


def test_ridge_standardized_coefficient_converts_to_raw_gradient() -> None:
    rng = np.random.default_rng(2); cells = []; hidden = {}; colors = ("black", "blue", "brown", "cyan")
    for i in range(20):
        key = f"c{i}"; value = rng.normal(size=3584).astype(np.float32); hidden[key] = value
        cells.append({"array_key": key, "fixed_answer_color": colors[i % 4], "mean_clean_final_sa": float(value[0] * .1 + value[1] * .2), "outer_fold": 1 + i % 4})
    _model, raw, selected, error, alpha, trace = weighted_sa_probe(cells, hidden, "cyan")
    assert len(selected) == 15 and raw.shape == (3584,) and error < 1e-7 and alpha > 0 and len(trace) == 9


def test_family_hash_fold_is_stable_and_complete() -> None:
    values = [inner_fold(f"family-{i}") for i in range(200)]
    assert set(values) == set(range(5)) and values == [inner_fold(f"family-{i}") for i in range(200)]


def test_rebuilt_shuffle_is_deterministic_and_independent() -> None:
    colors = ("black", "blue", "brown", "cyan", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow")
    cells = [{"array_key": f"{color}-{i}", "fixed_answer_color": color, "family_id": f"f-{color}-{i}", "mean_G_L": float(i)} for color in colors for i in range(4)]
    first = shuffled_targets(cells, 1)
    assert first == shuffled_targets(cells, 1) and first != shuffled_targets(cells, 2)


def test_output_refuses_overwrite_and_resume_checks_full_fingerprint(tmp_path) -> None:
    root = tmp_path / "output"; create_output_root(root, resume=False)
    with pytest.raises(FileExistsError): create_output_root(root, resume=False)
    config = root / "progress/config.json"; semantic_fingerprint(config, {"a": 1}, resume=False)
    with pytest.raises(ValueError, match="fingerprint mismatch"): semantic_fingerprint(config, {"a": 2}, resume=True)
    assert (root / "artifacts/directions").is_dir() and (root / "artifacts/diagnostics").is_dir()


def test_invalid_lock_cannot_open_formal_test(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="before experiment lock"):
        unseal_formal_test(tmp_path, {"status": "audit_failed"})
