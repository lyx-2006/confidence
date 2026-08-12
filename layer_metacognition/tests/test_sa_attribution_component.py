from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from layer_metacognition.sa_formation.attribution_component import (
    ALL_PROTOCOLS,
    COMMON_PROTOCOLS,
    ComponentScreenConfig,
    absolute_agreement_icc,
    classify_component,
    coordinate_invariance_metrics,
    fit_shared_target,
    load_protocol_panel,
    run_attribution_component_screen,
)


def _small_config(**overrides) -> ComponentScreenConfig:
    values = {
        "alphas": (0.1, 1.0, 10.0),
        "bootstrap_iterations": 60,
        "control_iterations": 30,
        "random_direction_iterations": 30,
        "expected_items": None,
        "seed": 42,
    }
    values.update(overrides)
    return ComponentScreenConfig(**values)


def test_shared_target_is_training_only_and_semantically_oriented() -> None:
    rng = np.random.default_rng(4)
    latent = np.linspace(-2, 2, 30)
    scores = np.column_stack(
        [offset + scale * latent + rng.normal(0, 0.01, len(latent)) for offset, scale in [(3, 1), (-8, 2), (1, 0.5)]]
    )
    model = fit_shared_target(scores[:20])
    target = model.transform(scores)
    assert model.mean.shape == (3,)
    assert model.explained_variance > 0.99
    assert np.sum(model.loading) > 0
    assert np.corrcoef(target, latent)[0, 1] > 0.99

    # A held-out outlier cannot alter any fitted target parameter.
    changed = scores.copy()
    changed[20:] += 1_000_000
    repeated = fit_shared_target(changed[:20])
    assert np.array_equal(model.mean, repeated.mean)
    assert np.array_equal(model.scale, repeated.scale)
    assert np.array_equal(model.loading, repeated.loading)


def test_absolute_coordinate_gate_is_stricly_stronger_than_rank() -> None:
    rng = np.random.default_rng(9)
    latent = np.linspace(-2, 2, 80)
    invariant = np.column_stack(
        [latent + rng.normal(0, 0.005, len(latent)) for _ in ALL_PROTOCOLS]
    )
    config = _small_config(bootstrap_iterations=100)
    clean = coordinate_invariance_metrics(invariant, config=config)
    assert absolute_agreement_icc(invariant[:, : len(COMMON_PROTOCOLS)]) > 0.99
    assert all(clean["basic_components"].values())

    shifted = invariant.copy()
    shifted[:, 1:] += np.linspace(0.4, 1.2, len(ALL_PROTOCOLS) - 1)
    assert all(
        np.corrcoef(shifted[:, 0], shifted[:, index])[0, 1] > 0.99
        for index in range(1, len(ALL_PROTOCOLS))
    )
    result = coordinate_invariance_metrics(shifted, config=config)
    assert not result["common_pairwise_equivalence_passed"]
    assert not result["legacy_holdout_equivalence_passed"]
    assert not result["basic_components"]["common_equivalence"]


def test_rank_only_classification_never_claims_coordinate_invariance() -> None:
    assert classify_component(False, False) == "no_validated_shared_attribution_component_on_existing_panel"
    rank_only = classify_component(True, False)
    assert rank_only == "rank_transport_candidate_only_no_coordinate_invariance"
    assert classify_component(True, True).startswith("existing_panel_coordinate_invariant_candidate")


def _write_synthetic_panel(root: Path, n_items: int = 25) -> Path:
    bridge = root / "05_protocol_granularity_bridge"
    hidden_dir = bridge / "hidden"
    hidden_dir.mkdir(parents=True)
    rng = np.random.default_rng(123)
    latent = rng.normal(size=n_items)
    # Make the latent independent of the simple covariate baseline in this finite sample.
    covariates = np.column_stack(
        [np.arange(n_items) % 2, (np.arange(n_items) // 2) % 2, np.arange(n_items) % 3]
    ).astype(float)
    design = np.column_stack([np.ones(n_items), covariates])
    latent -= design @ np.linalg.lstsq(design, latent, rcond=None)[0]
    latent /= latent.std(ddof=1)
    rows = []
    for item in range(n_items):
        hidden = np.zeros((len(ALL_PROTOCOLS), 8), dtype=np.float32)
        protocols = {}
        for protocol_index, name in enumerate(ALL_PROTOCOLS):
            hidden[protocol_index, 0] = latent[item] + rng.normal(0, 0.003)
            hidden[protocol_index, 1] = (protocol_index - 4.5) * 2.0
            hidden[protocol_index, 2:] = rng.normal(0, 0.10, 6)
            semantic = 0.5 + 0.12 * latent[item] + 0.02 * protocol_index + rng.normal(0, 0.001)
            protocols[name] = {
                "semantic_imageward_score": float(semantic),
                "ridge_sa_prediction": float(semantic + rng.normal(0, 0.005)),
                "ridge_sa_coordinate": float(latent[item] + rng.normal(0, 0.01)),
                "ridge_behavior_prediction": float(0.3 * latent[item] + rng.normal(0, 0.1)),
                "ridge_behavior_coordinate": float(latent[item] + rng.normal(0, 0.05)),
            }
        case_id = f"{item}__prior_{item % 3}__conflict_{'hard' if item % 2 else 'easy'}__v4__joint"
        hidden_path = hidden_dir / f"{case_id}.npz"
        np.savez(hidden_path, protocols=np.asarray(ALL_PROTOCOLS), hidden=hidden)
        rows.append(
            {
                "intervention_key": f"bridge|{case_id}",
                "case_id": case_id,
                "item_id": str(item),
                "prior_index": item % 3,
                "condition": "conflict_hard" if item % 2 else "conflict_easy",
                "difficulty": "hard" if item % 2 else "easy",
                "fold": item % 5,
                "final_image": bool(item % 2),
                "behavior_use_residual": float(rng.normal()),
                "status": "completed",
                "protocols": protocols,
                "hidden_file": str(Path("05_protocol_granularity_bridge") / "hidden" / hidden_path.name),
            }
        )
    (bridge / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return bridge


def test_panel_loader_and_oof_screen_are_isolated_and_auditable(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    bridge = _write_synthetic_panel(input_root)
    panel = load_protocol_panel(bridge, expected_items=None)
    assert panel.hidden.shape == (25, 10, 8)
    assert panel.protocols == ALL_PROTOCOLS
    source_before = (bridge / "results.jsonl").read_bytes()

    output = tmp_path / "outputs" / "component"
    summary = run_attribution_component_screen(
        bridge, output, config=_small_config()
    )
    assert summary["status"] == "completed"
    assert summary["n_items"] == 25
    assert summary["protocols"]["training"] == list(COMMON_PROTOCOLS)
    assert summary["classification"] == classify_component(
        summary["rank_gate"]["passed"], summary["coordinate_gate"]["passed"]
    )
    assert (output / "summary.json").is_file()
    assert (output / "results.jsonl").is_file()
    assert (output / "directions" / "index.json").is_file()
    direction_index = json.loads((output / "directions" / "index.json").read_text())
    assert len(direction_index["folds"]) == 5
    assert all(not fold["item_overlap"] for fold in direction_index["folds"])
    assert all(
        set(fold["train_items"]).isdisjoint(fold["test_items"])
        for fold in direction_index["folds"]
    )
    assert direction_index["protocol_specific_calibration"] is False
    assert (bridge / "results.jsonl").read_bytes() == source_before
    assert not any(path.name == "summary.json" for path in bridge.rglob("summary.json"))

    with pytest.raises(FileExistsError):
        run_attribution_component_screen(bridge, output, config=_small_config())
    with pytest.raises(ValueError, match="separate sibling"):
        run_attribution_component_screen(bridge, bridge / "bad", config=_small_config())

