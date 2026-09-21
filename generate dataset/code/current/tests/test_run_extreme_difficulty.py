from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "run_extreme_difficulty.py"
SPEC = importlib.util.spec_from_file_location("run_extreme_difficulty_under_test", PATH)
assert SPEC and SPEC.loader
EXTREME = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXTREME
SPEC.loader.exec_module(EXTREME)


def test_five_extreme_rounds_cover_requested_range() -> None:
    specs = EXTREME.extreme_specs()
    assert len(specs) == 5
    assert {spec.semantic_shape_count for spec in specs} == {10, 11, 12}
    assert max(spec.occlusion_target for spec in specs) == pytest.approx(0.8)
    assert all(spec.occluder_count >= 2 for spec in specs)
    assert all(spec.candidate_count == 216 for spec in specs)


@pytest.mark.parametrize("shape", EXTREME.SHAPES)
@pytest.mark.parametrize("spec", EXTREME.extreme_specs(), ids=lambda value: value.name)
def test_multiple_occluders_reach_target_and_all_contribute(spec, shape: str) -> None:
    seed = EXTREME.LEGACY.derive_seed(
        20260919, "extreme", spec.semantic_shape_count, "red", shape, 1
    )
    base = EXTREME.build_semantic_layout(seed, spec.semantic_shape_count, shape, "red")
    layout, measured = EXTREME.add_multiple_occluders(
        base, spec.occlusion_target, spec.occluder_count, seed
    )
    semantic = [obj for obj in layout["objects"] if obj["role"] in {"target", "distractor"}]
    occluders = [obj for obj in layout["objects"] if obj["role"] == "occluder"]
    assert len(semantic) == spec.semantic_shape_count
    assert len({obj["shape"] for obj in semantic}) == spec.semantic_shape_count
    assert len(occluders) == spec.occluder_count
    assert all(obj["expected_mask_occlusion_ratio"] > 0 for obj in occluders)
    assert measured == pytest.approx(spec.occlusion_target, abs=0.02)


def _trial(passed: bool, entropy: float) -> dict:
    return {
        "gate_passed": passed,
        "normalized_entropy": entropy,
        "target_probability": 0.5,
        "prediction": "red" if passed else "blue",
        "restricted_top1": "red" if passed else "blue",
    }


def test_two_of_three_gate_and_median_entropy() -> None:
    accepted = EXTREME.aggregate_trials([_trial(True, 0.1), _trial(False, 0.9), _trial(True, 0.2)])
    rejected = EXTREME.aggregate_trials([_trial(True, 0.1), _trial(False, 0.2), _trial(False, 0.3)])
    assert accepted["gate_passed"] is True
    assert accepted["correct_trials"] == 2
    assert accepted["normalized_entropy"] == pytest.approx(0.2)
    assert rejected["gate_passed"] is False
    assert rejected["correct_trials"] == 1
