from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from layer_metacognition.probe.probe_models import build_hidden_state_probe
from layer_metacognition.probe.torch_logistic_probe import (
    TorchProbeNumericalError,
    balanced_sample_weights,
    fit_torch_logistic_probe,
    sklearn_l2_strength,
)


def _classification_data(classes: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12 + classes)
    labels = np.arange(120, dtype=np.int64) % classes
    centers = np.eye(classes, 10, dtype=np.float32)[labels] * 1.5
    features = (rng.normal(size=(len(labels), 10)) + centers).astype(np.float32)
    return features, labels


def test_balanced_weights_and_sklearn_l2_scaling() -> None:
    labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
    weights = balanced_sample_weights(labels, 2)
    np.testing.assert_allclose(weights, [2 / 3, 2 / 3, 2 / 3, 2.0])
    assert float(weights.sum()) == pytest.approx(4.0)
    assert sklearn_l2_strength(2.0, 4.0) == pytest.approx(0.125)


@pytest.mark.parametrize(
    ("class_count", "binary_single_logit"),
    [(2, True), (3, False)],
)
def test_fixed_c_torch_matches_sklearn_probabilities(
    class_count: int,
    binary_single_logit: bool,
) -> None:
    X, y = _classification_data(class_count)
    sklearn_model = build_hidden_state_probe(1.0).fit(X, y)
    torch_model = fit_torch_logistic_probe(
        X,
        y,
        C=1.0,
        device="cpu",
        seed=42,
        binary_single_logit=binary_single_logit,
    )
    sklearn_probabilities = sklearn_model.predict_proba(X)
    torch_probabilities = torch_model.predict_proba(X)
    agreement = np.mean(sklearn_model.predict(X) == torch_model.predict(X))
    sklearn_prediction = sklearn_model.predict(X)
    torch_prediction = torch_model.predict(X)
    assert agreement >= 0.95
    assert np.mean(np.abs(sklearn_probabilities - torch_probabilities)) <= 0.02
    assert abs(
        accuracy_score(y, sklearn_prediction) - accuracy_score(y, torch_prediction)
    ) <= 0.02
    assert abs(
        balanced_accuracy_score(y, sklearn_prediction)
        - balanced_accuracy_score(y, torch_prediction)
    ) <= 0.02
    np.testing.assert_allclose(torch_probabilities.sum(axis=1), 1.0, atol=1e-7)
    assert torch_model.diagnostics["binary_single_logit"] is binary_single_logit


def test_two_phase_retry_and_non_converged_prediction_is_retained() -> None:
    X, y = _classification_data(3)
    model = fit_torch_logistic_probe(
        X,
        y,
        C=1.0,
        device="cpu",
        max_iter_per_phase=1,
        max_phases=2,
        grad_tolerance=0.0,
        loss_tolerance=0.0,
    )
    assert model.diagnostics["retry_count"] == 1
    assert model.diagnostics["iterations"] <= 2
    assert model.predict(X).shape == y.shape


def test_non_finite_input_is_invalid() -> None:
    X, y = _classification_data(2)
    X[0, 0] = np.nan
    with pytest.raises(TorchProbeNumericalError, match="NaN or Inf"):
        fit_torch_logistic_probe(X, y, C=1.0, device="cpu")


def test_cuda_smoke_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    X, y = _classification_data(2)
    model = fit_torch_logistic_probe(
        X, y, C=1.0, device="cuda", binary_single_logit=True
    )
    assert model.diagnostics["device"] == "cuda"
    assert model.predict_proba(X).shape == (len(y), 2)
