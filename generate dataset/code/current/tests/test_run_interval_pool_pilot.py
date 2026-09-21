from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "run_interval_pool_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_interval_pool_pilot_under_test", PATH)
assert SPEC and SPEC.loader
PILOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PILOT
SPEC.loader.exec_module(PILOT)


@pytest.mark.parametrize(
    ("entropy", "expected"),
    [
        (0.0, "0.0-0.1"), (0.099999, "0.0-0.1"),
        (0.1, "0.1-0.2"), (0.2, "0.2-0.3"),
        (0.3, "0.3-0.4"), (0.4, "0.4-0.5"),
        (0.5, "0.5-0.6"), (0.6, ">=0.6"), (1.0, ">=0.6"),
    ],
)
def test_entropy_boundaries_are_disjoint(entropy: float, expected: str) -> None:
    assert PILOT.interval_for_entropy(entropy) == expected


def test_matrix_has_three_pairs_and_every_interval_has_constructions() -> None:
    assert PILOT.PAIRS == (("red", "circle"), ("white", "triangle"), ("black", "star"))
    assert set(PILOT.CONSTRUCTIONS) == {item[0] for item in PILOT.INTERVALS}
    assert all(PILOT.CONSTRUCTIONS[name] for name, _lower, _upper in PILOT.INTERVALS)
    assert max(
        construction.occlusion_target
        for values in PILOT.CONSTRUCTIONS.values()
        for construction in values
    ) == pytest.approx(0.8)
    assert {construction.semantic_shape_count for construction in PILOT.CONSTRUCTIONS[">=0.6"]} >= {10, 11, 12}


def test_stop_policy_caps_every_interval_at_quota_or_budget(tmp_path: Path) -> None:
    pilot = PILOT.IntervalPilot(
        tmp_path,
        PILOT.DEFAULT_MODEL,
        seed=1,
        quota=5,
        max_attempts=200,
        workers=1,
        resume=False,
        runner=object(),
    )
    pair = PILOT.PAIRS[0]
    pilot.records[pair].extend(
        {"candidate_id": f"accepted-{index}", "actual_interval": "0.1-0.2"}
        for index in range(5)
    )
    assert pilot.target_done(pair, "0.1-0.2") is True

    pilot.records[pair].extend(
        {"candidate_id": f"high-{index}", "actual_interval": "0.5-0.6"}
        for index in range(25)
    )
    assert pilot.target_done(pair, "0.5-0.6") is True
    pilot.result_rows.extend(
        {
            "candidate_id": f"attempt-{index}", "status": "completed",
            "color": pair[0], "shape": pair[1], "target_interval": "0.5-0.6",
            "target_attempt_index": index + 1,
        }
        for index in range(200)
    )
    assert pilot.target_done(pair, "0.5-0.6") is True


def test_candidate_ids_and_seeds_are_reproducible(tmp_path: Path) -> None:
    pilot = PILOT.IntervalPilot(
        tmp_path,
        PILOT.DEFAULT_MODEL,
        seed=99,
        quota=5,
        max_attempts=200,
        workers=1,
        resume=False,
        runner=object(),
    )
    construction = PILOT.CONSTRUCTIONS["0.3-0.4"][0]
    first = pilot._task(PILOT.PAIRS[0], "0.3-0.4", 17, construction)
    second = pilot._task(PILOT.PAIRS[0], "0.3-0.4", 17, construction)
    assert first["candidate_id"] == second["candidate_id"]
    assert first["seed"] == second["seed"]


class _PassingRunner:
    def image_only(self, _question: str, _image_path: str, expected: str) -> dict:
        logits = {color: 0.0 for color in PILOT.POOL.COLORS}
        probabilities = {color: 1.0 / len(PILOT.POOL.COLORS) for color in PILOT.POOL.COLORS}
        return {
            "parse_success": True, "normalized_answer": expected,
            "restricted_top1": expected, "gate_passed": True,
            "answer_metric_status": "completed", "normalized_entropy": 0.05,
            "target_probability": probabilities[expected], "target_margin": 0.0,
            "answer_class_logits": logits,
            "answer_class_probabilities": probabilities,
        }


def test_terminal_candidate_keeps_final_data_but_removes_staging(tmp_path: Path) -> None:
    pilot = PILOT.IntervalPilot(
        tmp_path / "pool", PILOT.DEFAULT_MODEL, seed=5, quota=5,
        max_attempts=200, workers=1, resume=False, runner=_PassingRunner(),
        retain_staging=False,
    )
    construction = PILOT.CONSTRUCTIONS["0.0-0.1"][0]
    task = pilot._task(PILOT.PAIRS[0], "0.0-0.1", 1, construction)
    output = Path(task["output_dir"])
    output.mkdir(parents=True)
    image = output / "candidate.png"
    image.write_bytes(b"synthetic-png")
    layout_path = output / "layout.json"
    layout_path.write_text(
        '{"objects": [{"role": "target", "shape": "circle", "color": "red"}]}',
        encoding="utf-8",
    )
    generated = {
        **task,
        "image_path": str(image),
        "layout_path": str(layout_path),
        "image_sha256": PILOT.POOL.sha256_file(image),
        "geometry": {"occlusion_ratio": 0.0},
    }
    row = pilot.score(generated)
    assert row["accepted"] is True
    assert row["staging_retained"] is False
    assert not output.exists()
    published = Path(row["published_image"])
    assert published.is_file()
    record = pilot.records[PILOT.PAIRS[0]][0]
    assert set(record["validation"]["answer_class_logits"]) == set(PILOT.POOL.COLORS)
    assert set(record["validation"]["answer_class_probabilities"]) == set(PILOT.POOL.COLORS)
    assert record["layout"]["objects"][0]["role"] == "target"
    assert record["layout_sha256"] == PILOT.POOL.canonical_hash(record["layout"])


def test_resume_cleans_terminal_staging_left_by_interruption(tmp_path: Path) -> None:
    root = tmp_path / "pool"
    first = PILOT.IntervalPilot(
        root, PILOT.DEFAULT_MODEL, seed=8, quota=5, max_attempts=200,
        workers=1, resume=False, runner=object(), retain_staging=False,
    )
    construction = PILOT.CONSTRUCTIONS["0.1-0.2"][0]
    task = first._task(PILOT.PAIRS[0], "0.1-0.2", 1, construction)
    output = Path(task["output_dir"])
    output.mkdir(parents=True)
    image = output / "candidate.png"
    image.write_bytes(b"leftover")
    PILOT.EXP.append_jsonl(first.results_path, {
        **{key: task[key] for key in (
            "candidate_id", "color", "shape", "target_interval",
            "target_attempt_index", "construction",
        )},
        "status": "completed", "accepted": False,
        "actual_interval": "invalid", "image_path": str(image),
    })
    assert output.exists()
    PILOT.IntervalPilot(
        root, PILOT.DEFAULT_MODEL, seed=8, quota=5, max_attempts=200,
        workers=1, resume=True, runner=object(), retain_staging=False,
    )
    assert not output.exists()


def test_resume_can_explicitly_reduce_max_attempts(tmp_path: Path) -> None:
    root = tmp_path / "pool"
    PILOT.IntervalPilot(
        root, PILOT.DEFAULT_MODEL, seed=8, quota=5, max_attempts=200,
        workers=1, resume=False, runner=object(), retain_staging=False,
    )
    resumed = PILOT.IntervalPilot(
        root, PILOT.DEFAULT_MODEL, seed=8, quota=5, max_attempts=100,
        workers=1, resume=True, runner=object(), retain_staging=False,
        migrate_max_attempts_from=200,
    )
    assert resumed.max_attempts == 100
    assert resumed.manifest["config_fingerprint"] == resumed.fingerprint
    assert resumed.manifest["config_migrations"][-1] == {
        "field": "max_attempts_per_pair_interval", "from": 200, "to": 100,
    }
    reduced_quota = PILOT.IntervalPilot(
        root, PILOT.DEFAULT_MODEL, seed=8, quota=3, max_attempts=100,
        workers=1, resume=True, runner=object(), retain_staging=False,
        migrate_quota_from=5,
    )
    assert reduced_quota.quota == 3
    assert reduced_quota.manifest["config_fingerprint"] == reduced_quota.fingerprint
    assert reduced_quota.manifest["config_migrations"][-1] == {
        "field": "quota", "from": 5, "to": 3,
    }


def test_score_batch_uses_multi_runner_and_commits_in_order(tmp_path: Path) -> None:
    class FakeMultiRunner:
        def __init__(self):
            self.requests = []

        def image_only_many(self, requests):
            self.requests = list(requests)
            return [{"sequence": index} for index in range(len(requests))]

    runner = FakeMultiRunner()
    pilot = PILOT.IntervalPilot(
        tmp_path / "pool", PILOT.DEFAULT_MODEL, seed=9, quota=5,
        max_attempts=200, workers=1, resume=False, runner=runner,
    )
    generated = [
        {
            "candidate_id": f"candidate-{index}", "shape": "circle",
            "color": "red", "image_path": f"/tmp/{index}.png",
        }
        for index in range(4)
    ]
    committed = []

    def fake_record(item, raw):
        committed.append((item["candidate_id"], raw["sequence"]))
        return {"candidate_id": item["candidate_id"]}

    pilot._record_score = fake_record
    result = pilot.score_batch(generated)
    assert [value["candidate_id"] for value in result] == [
        f"candidate-{index}" for index in range(4)
    ]
    assert committed == [(f"candidate-{index}", index) for index in range(4)]
    assert len(runner.requests) == 4
