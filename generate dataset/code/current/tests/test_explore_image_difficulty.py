from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "explore_image_difficulty.py"
SPEC = importlib.util.spec_from_file_location("explore_image_difficulty_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
EXPLORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPLORE
SPEC.loader.exec_module(EXPLORE)


def test_main_matrix_has_22_rounds_and_4752_candidates() -> None:
    specs = EXPLORE.main_round_specs()
    assert len(specs) == 22
    assert [spec.shape_count for spec in specs[:4]] == [1, 4, 7, 10]
    assert all(spec.occlusion_target == 0 for spec in specs[:4])
    assert sum(spec.candidate_count for spec in specs) == 4752
    assert {
        (spec.shape_count, spec.occlusion_target) for spec in specs[4:]
    } == {
        (count, occlusion)
        for count in (4, 7, 10)
        for occlusion in EXPLORE.OCCLUSION_LEVELS
    }


@pytest.mark.parametrize("count", (4, 7, 10))
@pytest.mark.parametrize("shape", EXPLORE.SHAPES)
def test_occlusion_steps_keep_semantic_layout_and_direction(count: int, shape: str) -> None:
    seed = EXPLORE.base_seed(42, count, "red", shape, 1)
    base = EXPLORE.build_semantic_layout(seed, count, shape, "red")
    semantic_hash = None
    direction = None
    for target in EXPLORE.OCCLUSION_LEVELS:
        layout, measured, current_direction = EXPLORE.add_fixed_direction_occluder(base, target, seed)
        semantic = [obj for obj in layout["objects"] if obj["role"] in {"target", "distractor"}]
        current_hash = EXPLORE.POOL.canonical_hash(semantic)
        semantic_hash = semantic_hash or current_hash
        direction = direction or current_direction
        assert current_hash == semantic_hash
        assert current_direction == direction
        assert measured == pytest.approx(target, abs=0.02)


def test_threshold_search_uses_only_correct_rows_and_meets_gates() -> None:
    rows = []
    for index in range(160):
        level = index // 40
        entropy = (1e-8, 1e-6, 1e-4, 1e-2)[level] * (1 + (index % 7) / 20)
        rows.append({
            "status": "completed",
            "color": EXPLORE.COLORS[index % 3],
            "shape": EXPLORE.SHAPES[(index // 3) % 3],
            "blur_radius": EXPLORE.BLUR_RADII[level + 1],
            "validation": {"gate_passed": True, "normalized_entropy": entropy},
        })
    rows.append({
        "status": "completed", "color": "red", "shape": "circle", "blur_radius": 32.0,
        "validation": {"gate_passed": False, "normalized_entropy": 0.99},
    })
    result = EXPLORE.propose_thresholds(rows)
    assert result["eligible"] is True
    assert all(result["levels"][level]["count"] >= 20 for level in EXPLORE.LEVELS)
    assert all(result["levels"][level]["combination_count"] >= 6 for level in EXPLORE.LEVELS)


def test_round_summary_separates_incorrect_measurements() -> None:
    spec = EXPLORE.RoundSpec(1, 1, 0.0)
    rows = [
        {
            "status": "completed", "base_scene_id": "a", "color": "red", "shape": "circle",
            "blur_radius": 0.0, "occlusion_ratio": 0.0,
            "validation": {"gate_passed": True, "normalized_entropy": 0.1},
        },
        {
            "status": "completed", "base_scene_id": "a", "color": "red", "shape": "circle",
            "blur_radius": 2.0, "occlusion_ratio": 0.0,
            "validation": {"gate_passed": False, "normalized_entropy": 0.9},
        },
    ]
    summary = EXPLORE.summarize_round(spec, rows)
    assert summary["candidate_count"] == 2
    assert summary["correct_count"] == 1
    assert summary["entropy_max"] == pytest.approx(0.1)


def test_completed_round_fast_path_requires_all_terminal_candidates(tmp_path: Path) -> None:
    spec = EXPLORE.RoundSpec(1, 1, 0.0)
    runner = object.__new__(EXPLORE.ExperimentRunner)
    runner.root = tmp_path
    runner.manifest = {"completed_rounds": [1]}
    directory = tmp_path / spec.name
    directory.mkdir()
    for name in ("config.json", "summary.json"):
        (directory / name).write_text("{}", encoding="utf-8")
    (directory / "round_001.md").write_text("complete", encoding="utf-8")
    (directory / "contact_sheet.png").write_bytes(b"png")
    result_path = directory / "candidate_results.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for index in range(spec.candidate_count):
            handle.write(f'{{"candidate_id":"{index}","status":"completed"}}\n')
    assert runner.round_is_complete(spec) is True
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write('{"candidate_id":"extra","status":"completed"}\n')
    assert runner.round_is_complete(spec) is False
