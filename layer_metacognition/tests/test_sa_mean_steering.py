from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from layer_metacognition.model_adapter import AdditiveActivationHook, LanguageModules
from layer_metacognition.steering.decision_side_steering import (
    DecisionDirection,
    MEAN_DIFFERENCE_STEERING_SCALE,
    SteeringCase,
    build_steering_vector,
    manipulation_diagnostics,
)
from layer_metacognition.steering.source_attribution_mean_steering import (
    MeanSADirectionRepository,
    build_mean_directions,
    persist_direction_artifacts,
    select_heldout_evaluation_cases,
    select_strong_sa_sources,
    source_manifest_payload,
)


def _candidate(item: int, side: str, score: float) -> dict:
    return {
        "case_id": f"{item}__prior_0__conflict_easy__v4__joint",
        "item_id": str(item),
        "prior_index": 0,
        "condition": "conflict_easy",
        "decision_side": side,
        "SA_soft_image_score": score,
        "SA_hard_label": "8" if side == "follows_image" else "0",
        "SA_parsed_label": "8" if side == "follows_image" else "0",
        "hidden_state_reference": {"shard_path": "x.pt", "offset": item},
    }


def test_strong_source_selection_is_extreme_unique_and_cross_group_disjoint() -> None:
    candidates = [
        _candidate(item, "follows_text", 0.01 + item / 1000.0)
        for item in range(30)
    ] + [
        _candidate(item, "follows_image", 0.99 - item / 1000.0)
        for item in range(50)
    ]
    groups = select_strong_sa_sources(candidates, cases_per_side=5)
    text_items = {row["item_id"] for row in groups["follows_text"]}
    image_items = {row["item_id"] for row in groups["follows_image"]}
    assert text_items == {"0", "1", "2", "3", "4"}
    assert not text_items.intersection(image_items)
    assert len(image_items) == 5
    assert [row["SA_soft_image_score"] for row in groups["follows_text"]] == sorted(
        row["SA_soft_image_score"] for row in groups["follows_text"]
    )
    assert [row["SA_soft_image_score"] for row in groups["follows_image"]] == sorted(
        (row["SA_soft_image_score"] for row in groups["follows_image"]),
        reverse=True,
    )


def test_mean_direction_uses_image_mean_minus_text_mean() -> None:
    class Hidden:
        def get(self, record, layer, position):
            assert layer == 20 and position == "ac"
            return np.asarray(record["hidden"], dtype=np.float64)

    groups = {
        "follows_image": [{"hidden": [3.0, 4.0]}, {"hidden": [5.0, 6.0]}],
        "follows_text": [{"hidden": [1.0, 2.0]}, {"hidden": [1.0, 0.0]}],
    }
    directions, metadata = build_mean_directions(
        groups=groups,
        hidden_states=Hidden(),
        layers=[20],
        positions=["ac"],
    )
    direction = directions[(20, "ac")]
    expected = np.asarray([3.0, 4.0])
    assert np.allclose(direction.steering_vector, expected)
    assert np.allclose(direction.d_raw, expected / 5.0)
    assert np.allclose(
        build_steering_vector(
            direction,
            alpha=0.5,
            steering_scale=MEAN_DIFFERENCE_STEERING_SCALE,
        ),
        [1.5, 2.0],
    )
    assert metadata[0]["difference_l2"] == 5.0


def test_mean_difference_manipulation_diagnostic_checks_realized_projection() -> None:
    layer = torch.nn.Identity()
    modules = LanguageModules(
        language_layers=[layer],
        final_norm=torch.nn.Identity(),
        lm_head=torch.nn.Linear(2, 3, bias=False),
        hidden_size=2,
        num_hidden_layers=1,
    )
    difference = np.asarray([3.0, 4.0])
    direction = DecisionDirection(
        file="mean.npz",
        fold=0,
        layer=0,
        position="ac",
        d_raw=difference / 5.0,
        d_K=difference / 5.0,
        raw_intercept=0.0,
        steering_vector=difference,
        direction_kind="strong_sa_mean_difference",
    )
    vector = build_steering_vector(
        direction,
        alpha=0.5,
        steering_scale=MEAN_DIFFERENCE_STEERING_SCALE,
    )
    hook = AdditiveActivationHook(
        modules,
        layer_index=0,
        target_position=1,
        steering_vector=torch.from_numpy(vector),
        prefill_sequence_length=2,
    )
    with hook:
        layer(torch.zeros(1, 2, 2))
    diagnostics = manipulation_diagnostics(
        hook,
        direction,
        alpha=0.5,
        steering_scale=MEAN_DIFFERENCE_STEERING_SCALE,
    )
    assert diagnostics["expected_decision_logit_delta"] == 2.5
    assert diagnostics["decision_logit_delta"] == 2.5
    assert diagnostics["passed"] is True


def test_mean_repository_exposes_downstream_trajectory_layers() -> None:
    class Hidden:
        def get(self, record, layer, position):
            base = 1.0 if record["decision_side"] == "follows_image" else 0.0
            return np.asarray([base + layer, base], dtype=np.float64)

    groups = {
        side: [{"decision_side": side} for _ in range(2)]
        for side in ("follows_image", "follows_text")
    }
    directions, _ = build_mean_directions(
        groups=groups,
        hidden_states=Hidden(),
        layers=[20, 24],
        positions=["panl"],
    )
    repository = MeanSADirectionRepository(
        directions=directions,
        manifest_path=__file__,
    )
    repository.validate_requested_grid([20, 24], ["panl"])
    assert repository.trajectory_layers(20, "panl") == [20, 24]
    assert repository.get(0, 24, "panl").layer == 24


def _steering_case(item: int, side: str, condition: str) -> SteeringCase:
    return SteeringCase(
        manifest={
            "case_id": f"{item}__prior_0__{condition}__v4__joint",
            "item_id": str(item),
            "prior_index": 0,
            "condition": condition,
            "decision_side": side,
        },
        evaluation=SimpleNamespace(),
        baseline={},
        fold=0,
    )


def test_heldout_evaluation_is_side_difficulty_balanced_and_item_disjoint() -> None:
    cases = []
    item = 0
    for side in ("follows_text", "follows_image"):
        for condition in ("conflict_easy", "conflict_hard"):
            for _ in range(4):
                cases.append(_steering_case(item, side, condition))
                item += 1
    selected = select_heldout_evaluation_cases(
        cases,
        excluded_item_ids={"0", "8"},
        cases_per_side=2,
    )
    assert len(selected) == 4
    assert not {case.manifest["item_id"] for case in selected}.intersection({"0", "8"})
    assert len({case.manifest["item_id"] for case in selected}) == 4
    for side in ("follows_text", "follows_image"):
        side_cases = [case for case in selected if case.manifest["decision_side"] == side]
        assert len(side_cases) == 2
        assert {case.manifest["condition"] for case in side_cases} == {
            "conflict_easy",
            "conflict_hard",
        }
    smoke = select_heldout_evaluation_cases(
        cases,
        excluded_item_ids={"0", "8"},
        cases_per_side=2,
        max_cases=4,
    )
    assert len(smoke) == 4
    assert {case.manifest["condition"] for case in smoke} == {
        "conflict_easy",
        "conflict_hard",
    }


def test_direction_artifacts_round_trip(tmp_path) -> None:
    groups = {
        "follows_text": [_candidate(1, "follows_text", 0.1)],
        "follows_image": [_candidate(2, "follows_image", 0.9)],
    }
    manifest = source_manifest_payload(groups)
    entry = {
        "file": "layer_20_ac.npz",
        "layer": 20,
        "position": "ac",
        "difference_l2": 1.0,
        "image_mean": np.asarray([2.0, 3.0]),
        "text_mean": np.asarray([1.0, 1.0]),
        "difference": np.asarray([1.0, 2.0]),
        "unit_direction": np.asarray([1.0, 2.0]) / np.sqrt(5.0),
        "midpoint": np.asarray([1.5, 2.0]),
    }
    source_path, index_path = persist_direction_artifacts(
        tmp_path,
        source_manifest=manifest,
        direction_metadata=[entry],
    )
    assert source_path.is_file()
    assert index_path.is_file()
    persist_direction_artifacts(
        tmp_path,
        source_manifest=manifest,
        direction_metadata=[entry],
    )
