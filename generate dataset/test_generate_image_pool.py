from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image


SCRIPT = Path(__file__).with_name("generate_image_pool.py")
SPEC = importlib.util.spec_from_file_location("generate_image_pool", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
POOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POOL
SPEC.loader.exec_module(POOL)


class FakeRunner:
    def __init__(self, *, generated: str = "red", restricted: str = "red", entropy: float = 0.4):
        self.generated = generated
        self.restricted = restricted
        self.entropy = entropy
        self.calls: list[dict[str, Any]] = []

    def image_only(self, question: str, image_path: str, expected: str) -> dict[str, Any]:
        self.calls.append({"question": question, "image_path": image_path, "expected": expected})
        probabilities = {color: 0.01 for color in POOL.COLORS}
        probabilities[self.restricted] = 0.89
        return {
            "parse_success": True,
            "normalized_answer": self.generated,
            "restricted_top1": self.restricted,
            "answer_metric_status": "completed",
            "gate_passed": self.generated == expected and self.restricted == expected,
            "target_probability": probabilities[expected],
            "target_margin": 1.0,
            "normalized_entropy": self.entropy,
            "answer_class_logits": {color: (3.0 if color == self.restricted else 0.0) for color in POOL.COLORS},
            "answer_class_probabilities": probabilities,
            "actual_output": f"**Answer**: {self.generated}",
            "prompt_hash": "prompt",
            "rendered_hash": "rendered",
            "model_fingerprint": "model",
        }


@pytest.mark.parametrize("profile", POOL.PROFILES, ids=lambda value: value.name)
def test_pool_profiles_have_exact_semantic_counts_and_occlusion(profile: Any) -> None:
    layout = POOL.build_pool_layout(812, profile.name, "star", "red")
    semantic = [obj for obj in layout["objects"] if obj["role"] in {"target", "distractor"}]
    targets = [obj for obj in semantic if obj["role"] == "target"]
    distractors = [obj for obj in semantic if obj["role"] == "distractor"]
    assert len(semantic) == profile.semantic_shape_count
    assert len(targets) == 1
    assert len({obj["shape"] for obj in semantic}) == len(semantic)
    assert all(obj["color"] != "red" for obj in distractors)
    ratio = float(layout["expected_occlusion_ratio"])
    if profile.occlusion_range is None:
        assert ratio == 0.0
    else:
        assert profile.occlusion_range[0] <= ratio <= profile.occlusion_range[1]


def test_layout_and_render_are_deterministic_and_blur_changes_hash(tmp_path: Path) -> None:
    first = POOL.build_pool_layout(99, "medium_occluded", "triangle", "black")
    second = POOL.build_pool_layout(99, "medium_occluded", "triangle", "black")
    assert first == second
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_geometry = POOL.render_base_scene(first, first_dir)
    second_geometry = POOL.render_base_scene(second, second_dir)
    assert first_geometry["valid"] and second_geometry["valid"]
    assert POOL.sha256_file(first_dir / "sharp.png") == POOL.sha256_file(second_dir / "sharp.png")
    clear, blurred = tmp_path / "clear.png", tmp_path / "blurred.png"
    POOL.render_blur_variant(first_dir / "sharp.png", clear, 0)
    POOL.render_blur_variant(first_dir / "sharp.png", blurred, 8)
    assert POOL.sha256_file(clear) == POOL.sha256_file(first_dir / "sharp.png")
    assert POOL.sha256_file(blurred) != POOL.sha256_file(clear)


def test_qwen_gate_requires_generated_and_restricted_answers() -> None:
    correct = FakeRunner().image_only("q", __file__, "red")
    assert POOL.validation_from_qwen(correct, "red")["gate_passed"] is True
    generated_wrong = FakeRunner(generated="blue").image_only("q", __file__, "red")
    value = POOL.validation_from_qwen(generated_wrong, "red")
    assert value["gate_passed"] is False
    assert "generated_answer_incorrect" in value["failure_reasons"]
    restricted_wrong = FakeRunner(restricted="blue").image_only("q", __file__, "red")
    value = POOL.validation_from_qwen(restricted_wrong, "red")
    assert value["gate_passed"] is False
    assert "restricted_top1_incorrect" in value["failure_reasons"]


def test_entropy_boundaries_are_half_open() -> None:
    thresholds = (0.2, 0.4, 0.6)
    assert POOL.entropy_level(0.0, thresholds) == "very_easy"
    assert POOL.entropy_level(0.1999, thresholds) == "very_easy"
    assert POOL.entropy_level(0.2, thresholds) == "easy"
    assert POOL.entropy_level(0.4, thresholds) == "medium"
    assert POOL.entropy_level(0.6, thresholds) == "hard"
    assert POOL.entropy_level(1.0, thresholds) == "hard"


def test_shape_json_is_array_named_by_shape_color_and_resume_is_idempotent(tmp_path: Path) -> None:
    runner = FakeRunner(entropy=0.4)
    pipeline = POOL.ImagePoolPipeline(
        mode="pilot",
        output_root=tmp_path / "pool",
        model_path=POOL.DEFAULT_MODEL,
        seed=42,
        thresholds=None,
        quota_per_level=10,
        resume=False,
        runner=runner,
    )
    pipeline._run_scene("red", "circle", POOL.PROFILE_BY_NAME["single_clear"], 1)
    path = tmp_path / "pool" / "red" / "circle.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(records, list) and len(records) == len(POOL.BLUR_RADII)
    assert records[0]["image"] == "circle_red_000001.png"
    assert all(record["Answer"] == "red" for record in records)
    assert all(record["difficulty"]["level"] == "pending" for record in records)
    call_count = len(runner.calls)
    resumed = POOL.ImagePoolPipeline(
        mode="pilot",
        output_root=tmp_path / "pool",
        model_path=POOL.DEFAULT_MODEL,
        seed=42,
        thresholds=None,
        quota_per_level=10,
        resume=True,
        runner=runner,
    )
    resumed._run_scene("red", "circle", POOL.PROFILE_BY_NAME["single_clear"], 1)
    assert len(runner.calls) == call_count
    assert len(json.loads(path.read_text(encoding="utf-8"))) == len(POOL.BLUR_RADII)


def test_formal_build_and_legacy_import_share_fingerprint(tmp_path: Path) -> None:
    first = POOL.ImagePoolPipeline(
        mode="build",
        output_root=tmp_path / "pool",
        model_path=POOL.DEFAULT_MODEL,
        seed=42,
        thresholds=(0.2, 0.4, 0.6),
        quota_per_level=10,
        resume=False,
        runner=FakeRunner(),
    )
    second = POOL.ImagePoolPipeline(
        mode="import-legacy",
        output_root=tmp_path / "pool",
        model_path=POOL.DEFAULT_MODEL,
        seed=42,
        thresholds=(0.2, 0.4, 0.6),
        quota_per_level=10,
        resume=True,
        runner=FakeRunner(),
    )
    assert first.fingerprint == second.fingerprint


def test_legacy_collection_maps_branches_and_deduplicates_sha(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    consistent = images / "consistent.png"
    conflict = images / "conflict.png"
    Image.new("RGB", (16, 16), "red").save(consistent)
    Image.new("RGB", (16, 16), "blue").save(conflict)
    payload = [{
        "category": "colour",
        "items": [{
            "id": "1",
            "question": {"text": POOL.LEGACY.QUESTION_TEMPLATE.format(shape="circle")},
            "answer": "red",
            "conflict_ans": "blue",
            "image_clue": {
                "consistent": {"easy": "images/consistent.png", "hard": "images/consistent.png"},
                "conflict": {"easy": "images/conflict.png", "hard": "images/conflict.png"},
            },
        }],
    }]
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(payload), encoding="utf-8")
    rows = POOL.collect_legacy_candidates([dataset])
    assert len(rows) == 2
    assert {(row["color"], row["shape"]) for row in rows} == {("red", "circle"), ("blue", "circle")}


def test_pilot_analysis_requires_324_records_and_writes_report(tmp_path: Path) -> None:
    result_path = tmp_path / "candidate_results.jsonl"
    image = tmp_path / "candidate.png"
    Image.new("RGB", (24, 24), "red").save(image)
    for index in range(324):
        entropy = 0.01 + 0.98 * index / 323
        row = {
            "candidate_id": f"candidate-{index}",
            "accepted": True,
            "image_path": str(image),
            "color": POOL.PILOT_COLORS[index % 3],
            "shape": POOL.PILOT_SHAPES[(index // 3) % 3],
            "generation": {
                "base_scene_id": f"scene-{index // 6}",
                "profile": POOL.PROFILES[(index // 6) % 6].name,
                "blur_radius": POOL.BLUR_RADII[index % 6],
            },
            "validation": {"gate_passed": True, "normalized_entropy": entropy},
        }
        POOL.append_jsonl(result_path, row)
    report = POOL.analyze_pilot(tmp_path)
    assert report["candidate_count"] == 324
    assert len(report["proposed_global_thresholds"]) == 3
    assert (tmp_path / "pilot_report.json").is_file()
    assert (tmp_path / "pilot_contact_sheet.png").is_file()
