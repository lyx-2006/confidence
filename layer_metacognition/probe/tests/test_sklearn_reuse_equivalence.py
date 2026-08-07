from __future__ import annotations

import numpy as np

from layer_metacognition.probe.probe_metrics import compute_metrics
from layer_metacognition.probe.probe_models import build_hidden_state_probe


def test_reused_fixed_c_model_matches_independent_sklearn_fits() -> None:
    rng = np.random.default_rng(91)
    train_X = rng.normal(size=(60, 7)).astype(np.float32)
    train_y = np.arange(60, dtype=np.int64) % 3
    test_sets = [
        rng.normal(size=(13, 7)).astype(np.float32),
        rng.normal(size=(17, 7)).astype(np.float32),
    ]
    reused = build_hidden_state_probe(1.0).fit(train_X, train_y)
    classes = ["a", "b", "c"]
    for index, test_X in enumerate(test_sets):
        independent = build_hidden_state_probe(1.0).fit(train_X, train_y)
        np.testing.assert_array_equal(reused.predict(test_X), independent.predict(test_X))
        np.testing.assert_allclose(
            reused.predict_proba(test_X),
            independent.predict_proba(test_X),
            atol=1e-8,
            rtol=0.0,
        )
        truth = [classes[value] for value in independent.predict(test_X)]
        reused_metrics = compute_metrics(
            truth,
            [classes[value] for value in reused.predict(test_X)],
            reused.predict_proba(test_X),
            classes=classes,
            item_ids=[f"{index}-{row}" for row in range(len(test_X))],
            majority_class="a",
            selected_C=1.0,
        )
        independent_metrics = compute_metrics(
            truth,
            [classes[value] for value in independent.predict(test_X)],
            independent.predict_proba(test_X),
            classes=classes,
            item_ids=[f"{index}-{row}" for row in range(len(test_X))],
            majority_class="a",
            selected_C=1.0,
        )
        assert reused_metrics == independent_metrics
