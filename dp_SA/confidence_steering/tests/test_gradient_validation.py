from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dp_SA.soft_score import soft_sa_from_logits
from dp_SA.confidence_steering.gradient_validation import (
    DifferentiableLatHook, central_difference, directional_metrics,
    historical_pair_index, missing_paired_cells, normalize_epsilons,
    prediction_metrics, probe_raw_parameters, random_null_pair_index,
    relative_additivity_error, torch_soft_sa,
)


def test_torch_soft_sa_matches_numpy_float64():
    logits = np.linspace(-3.25, 2.75, 9, dtype=np.float64)
    torch_score, probabilities = torch_soft_sa(torch.from_numpy(logits))
    expected = soft_sa_from_logits(logits, list(range(9)))
    assert abs(torch_score.item() - expected["soft_sa_image_score"]) <= 1e-12
    assert np.max(np.abs(probabilities.numpy() - expected["class_probabilities"])) <= 1e-12


def test_probe_raw_conversion_matches_pipeline():
    rng = np.random.default_rng(42); x = rng.normal(size=(40, 7)); y = rng.normal(size=40)
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=3.0))]).fit(x, y)
    weight, bias = probe_raw_parameters(model); query = rng.normal(size=(8, 7))
    np.testing.assert_allclose(query @ weight + bias, model.predict(query), atol=1e-12, rtol=1e-12)


class _TupleBlock(torch.nn.Module):
    def forward(self, hidden):
        return hidden * 2, {"tail": 1}


def test_leaf_reinsertion_preserves_tuple_and_routes_gradient():
    blocks = torch.nn.ModuleList([_TupleBlock() for _ in range(19)])
    modules = SimpleNamespace(language_layers=blocks)
    hidden = torch.randn(1, 5, 4, dtype=torch.float64, requires_grad=True)
    with DifferentiableLatHook(modules, layer=14, lat_position=2, panl_position=3, sequence_length=5) as hook:
        out14 = blocks[14](hidden); out18 = blocks[18](out14[0])
    hook.validate(); assert out14[1] == {"tail": 1}; assert hook.layout_preserved
    gradient = torch.autograd.grad(out18[0][0, 3].sum() + out14[0][0, 2].sum(), hook.h_leaf)[0]
    assert gradient.shape == (4,); assert torch.isfinite(gradient).all()


def test_formulas_and_float64_directional_metrics():
    result = central_difference(3.0, 1.0, .5); assert result == {"D": 2.0, "S": 1.0}
    metrics = directional_metrics(np.array([1, 2], np.float32), np.array([3, 4], np.float32))
    assert metrics["directional_derivative"] == 11
    assert relative_additivity_error(5, 2, 3) == 0


def test_dynamic_history_requires_complete_pair():
    base = {"case_id": "c", "direction": "d", "layer": 14, "status": "completed", "format_valid": True, "hidden_definition": "decoder_block_output_pre_final_norm"}
    rows = [{**base, "alpha": -0.1}, {**base, "alpha": 0.1}, {**base, "alpha": 0.25}]
    index = historical_pair_index(rows, ("d",), (.1, .25)); assert set(index) == {("c", "d", .1)}
    assert missing_paired_cells(("c",), ("d",), (.1, .25), index) == [("c", "d", .25)]


def test_random_null_pairing_keeps_replicates_distinct():
    rows = []
    for replicate in (1, 2):
        for alpha in (-2.0, 2.0): rows.append({"case_id": "c", "direction": "random_sa_subspace_null", "null_replicate": replicate, "alpha": alpha, "status": "completed"})
    assert set(random_null_pair_index(rows)) == {("c", 1), ("c", 2)}


def test_epsilon_validation_and_prediction_metrics():
    assert normalize_epsilons((.5, .1, .25)) == (.1, .25, .5)
    with pytest.raises(ValueError): normalize_epsilons((.1, .1))
    metrics = prediction_metrics([1, -2, 3], [1, -2, 3]); assert metrics["pearson"] == pytest.approx(1); assert metrics["sign_agreement"] == 1; assert metrics["nrmse"] == 0
