from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "run_interval_pool_full.py"
SPEC = importlib.util.spec_from_file_location("run_interval_pool_full_under_test", PATH)
assert SPEC and SPEC.loader
FULL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FULL
SPEC.loader.exec_module(FULL)


def test_full_matrix_and_default_concurrency_plan() -> None:
    value = FULL.plan()
    assert value["colors"] == 12
    assert value["shapes"] == 17
    assert value["pair_count"] == 204
    assert value["active_pairs"] == 12
    assert value["generation_workers"] == 24
    assert value["bounded_queue"] == 48
    assert value["qwen_workers"] == 1
    assert value["gpu_devices"] == ["0"]
    assert value["retain_staging"] is False


def test_dual_gpu_plan_and_device_parser() -> None:
    assert FULL.parse_gpu_devices("0,1") == ("0", "1")
    value = FULL.plan(("0", "1"))
    assert value["qwen_workers"] == 2
    assert value["gpu_devices"] == ["0", "1"]


def test_existing_pilot_is_imported_without_inference(tmp_path: Path) -> None:
    runner = object()
    experiment = FULL.FullIntervalPool(
        root=tmp_path / "full",
        model_path=FULL.PILOT.DEFAULT_MODEL,
        workers=24,
        active_pairs=12,
        queue_size=48,
        reuse_roots=[FULL.DEFAULT_REUSE_ROOT],
        runner=runner,
    )
    assert experiment._runner_value is runner
    assert len(experiment.result_rows) == 2377
    assert sum(len(values) for values in experiment.records.values()) == 111
    assert all(
        isinstance(record.get("layout"), dict) and record.get("layout_sha256")
        for values in experiment.records.values() for record in values
    )
    source_rows = {
        row["candidate_id"]: row
        for row in FULL.PILOT.EXP.read_jsonl(FULL.DEFAULT_REUSE_ROOT / "candidate_results.jsonl")
        if row.get("accepted")
    }
    for values in experiment.records.values():
        for record in values:
            source_layout_path = (
                Path(source_rows[record["candidate_id"]]["image_path"]).parent
                / "scene" / "layout.json"
            )
            source_layout = json.loads(source_layout_path.read_text(encoding="utf-8"))
            assert record["layout_sha256"] == FULL.POOL.canonical_hash(source_layout)
    assert experiment.accepted_count(("red", "circle"), "0.0-0.1") == 5
    assert experiment.target_done(("red", "circle"), "0.0-0.1") is True
    assert experiment.target_done(("red", "circle"), ">=0.6") is True
    assert experiment.target_done(("orange", "circle"), "0.0-0.1") is False
    assert experiment.import_existing(FULL.DEFAULT_REUSE_ROOT) == {
        "candidate_rows": 0, "accepted_images": 0,
    }


class _ImmediateFuture:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _RecordingExecutor:
    def __init__(self):
        self.tasks = []

    def submit(self, _function, task):
        self.tasks.append(task)
        return _ImmediateFuture(task)


def test_scheduler_first_wave_covers_twelve_distinct_pairs(tmp_path: Path) -> None:
    pairs = FULL.ALL_PAIRS[:13]
    experiment = FULL.PILOT.IntervalPilot(
        root=tmp_path / "scheduler",
        model_path=FULL.PILOT.DEFAULT_MODEL,
        seed=3, quota=1, max_attempts=1, workers=24, resume=False,
        runner=object(), pairs=pairs, active_pair_limit=12, queue_size=48,
    )

    def fake_score(generated):
        row = {
            "candidate_id": generated["candidate_id"], "status": "completed",
            "accepted": False, "color": generated["color"], "shape": generated["shape"],
            "target_interval": generated["target_interval"], "actual_interval": "invalid",
            "target_attempt_index": generated["target_attempt_index"],
            "construction": generated["construction"],
        }
        experiment.result_rows.append(row)
        experiment.completed[row["candidate_id"]] = row
        return row

    experiment.score = fake_score
    executor = _RecordingExecutor()
    experiment.run_interval(executor, "0.0-0.1")
    first_wave = executor.tasks[:12]
    assert [(task["color"], task["shape"]) for task in first_wave] == list(pairs[:12])
    assert len(executor.tasks) == 13


def test_supplement_phase_only_targets_missing_high_bins_with_fresh_budget(tmp_path: Path) -> None:
    experiment = FULL.FullIntervalPool(
        root=tmp_path / "supplement", model_path=FULL.PILOT.DEFAULT_MODEL,
        quota=1, max_attempts=100, workers=1, active_pairs=1, queue_size=1,
        reuse_roots=[], runner=object(),
    )
    existing_pair = FULL.ALL_PAIRS[0]
    missing_pair = FULL.ALL_PAIRS[1]
    experiment.records[existing_pair].append({
        "candidate_id": "already-present", "actual_interval": "0.5-0.6",
    })
    experiment.result_rows.extend({
        "candidate_id": f"old-{index}", "status": "completed",
        "color": missing_pair[0], "shape": missing_pair[1],
        "target_interval": "0.5-0.6", "target_attempt_index": index + 1,
    } for index in range(100))
    experiment._configure_supplement_phase(("0.5-0.6", ">=0.6"), 100)

    assert experiment.target_done(existing_pair, "0.5-0.6") is True
    assert experiment.target_done(missing_pair, "0.1-0.2") is True
    assert experiment.attempt_limit(missing_pair, "0.5-0.6") == 200
    assert experiment.target_done(missing_pair, "0.5-0.6") is False
    experiment.records[missing_pair].append({
        "candidate_id": "new-hit", "actual_interval": "0.5-0.6",
    })
    assert experiment.target_done(missing_pair, "0.5-0.6") is True
