from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).with_name("generate_shape_color_dataset.py")
SPEC = importlib.util.spec_from_file_location("generate_shape_color_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _load_dataset() -> Any:
    return json.loads((ROOT / "datasets/dataset_test.json").read_text(encoding="utf-8"))


def _masks(occlusion: float = 0.0) -> tuple[Image.Image, Image.Image]:
    target = Image.new("L", (1024, 1024), 0)
    occluder = Image.new("L", (1024, 1024), 0)
    ImageDraw.Draw(target).rectangle((400, 400, 549, 549), fill=255)
    if occlusion:
        width = round(150 * occlusion)
        ImageDraw.Draw(occluder).rectangle((400, 400, 399 + width, 549), fill=255)
    return target, occluder


def _result(answer: str, entropy: float = 0.4) -> Any:
    return SimpleNamespace(
        normalized_answer=answer,
        answer_prob=0.7,
        answer_class_probabilities={color: (0.7 if color == answer else 0.3 / 11) for color in MODULE.COLORS},
        raw_answer_entropy=0.9,
        answer_entropy=entropy,
        parse_success=True,
        error=None,
        elapsed_seconds=0.01,
    )


class FakeInference:
    def __init__(self, answers: list[str], entropies: list[float] | None = None):
        self.answers = list(answers)
        self.entropies = list(entropies or [0.4] * len(answers))
        self.calls: list[dict[str, Any]] = []

    def generate_answer_with_metrics(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        return _result(self.answers[index], self.entropies[index])


def test_shape_order_color_order_dedup_and_current_counts() -> None:
    payload = _load_dataset()
    manifest = MODULE.build_manifest(payload, 1234)
    expected_existing = len(MODULE.extract_existing_combinations(payload))
    expected_missing = len(MODULE.SHAPES) * len(MODULE.COLORS) - expected_existing
    assert manifest["existing_count"] == expected_existing
    assert manifest["missing_count"] == manifest["planned_count"] == expected_missing
    assert len(manifest["shape_order"]) == len(set(manifest["shape_order"])) == len(MODULE.SHAPES)
    assert manifest["shape_order"] != MODULE.SHAPES
    existing = MODULE.extract_existing_combinations(payload)
    by_shape: dict[str, list[str]] = {}
    for combo in manifest["combinations"]:
        assert (combo["shape"], combo["text_color"]) not in existing
        by_shape.setdefault(combo["shape"], []).append(combo["text_color"])
    for shape, colors in by_shape.items():
        assert colors == [color for color in MODULE.COLORS if (shape, color) not in existing]


def test_conflict_maps_are_per_shape_derangements_and_globally_balanced() -> None:
    manifest = MODULE.build_manifest(_load_dataset(), 44)
    for mapping in manifest["conflict_maps"].values():
        assert set(mapping) == set(MODULE.COLORS)
        assert set(mapping.values()) == set(MODULE.COLORS)
        assert all(text != conflict for text, conflict in mapping.items())
    counts = list(manifest["conflict_counts"].values())
    assert max(counts) - min(counts) <= 1
    assert sum(counts) == manifest["planned_count"]
    assert manifest == MODULE.build_manifest(_load_dataset(), 44)


def test_similar_shape_groups_are_excluded() -> None:
    layout = MODULE.build_easy_layout(111, "consistent", "rectangle", "red")
    shapes = {obj["shape"] for obj in layout["objects"]}
    assert not ({"square", "parallelogram", "trapezoid", "diamond"} & shapes)


def test_easy_has_seven_distinct_distractors_and_zero_occlusion() -> None:
    layout = MODULE.build_easy_layout(812, "consistent", "star", "red")
    target, occluder = _masks(0.0)
    result = MODULE.validate_layout(layout, "star", "red", "easy", target, occluder)
    assert result["valid"], result["issues"]
    assert result["distinct_distractor_shapes"] >= 7
    assert result["occlusion_ratio"] == 0.0


def test_easy_objects_fill_a_meaningful_fraction_of_canvas() -> None:
    layout = MODULE.build_easy_layout(1812, "consistent", "star", "red")
    bbox_area = sum(
        (obj["bbox"][2] - obj["bbox"][0]) * (obj["bbox"][3] - obj["bbox"][1])
        for obj in layout["objects"]
    )
    assert bbox_area / (MODULE.CANVAS_SIZE ** 2) >= 0.075
    target = next(obj for obj in layout["objects"] if obj["role"] == "target")
    distractors = [obj for obj in layout["objects"] if obj["role"] == "distractor"]
    assert MODULE.EASY_TARGET_SIZE_RANGE[0] <= target["size"] <= MODULE.EASY_TARGET_SIZE_RANGE[1]
    assert min(obj["size"] for obj in distractors) >= MODULE.EASY_DISTRACTOR_SIZE_RANGE[0]
    for obj in layout["objects"]:
        width = obj["bbox"][2] - obj["bbox"][0]
        height = obj["bbox"][3] - obj["bbox"][1]
        assert max(width, height) >= MODULE.EASY_TARGET_SIZE_RANGE[0]
        assert width <= MODULE.MAX_OBJECT_BBOX_SIDE
        assert height <= MODULE.MAX_OBJECT_BBOX_SIDE


def test_hard_has_eleven_shapes_and_70_to_80_percent_occlusion() -> None:
    easy = MODULE.build_easy_layout(922, "conflict", "heart", "blue")
    hard = MODULE.build_hard_layout(923, easy)
    target, occluder = _masks(0.75)
    result = MODULE.validate_layout(hard, "heart", "blue", "hard", target, occluder)
    assert result["valid"], result["issues"]
    assert result["distinct_distractor_shapes"] >= 10
    assert 0.70 <= result["occlusion_ratio"] <= 0.80
    for obj in hard["objects"]:
        assert max(
            obj["bbox"][2] - obj["bbox"][0],
            obj["bbox"][3] - obj["bbox"][1],
        ) >= MODULE.HARD_EXTRA_DISTRACTOR_SIZE_RANGE[0]
        assert obj["bbox"][2] - obj["bbox"][0] <= MODULE.MAX_OBJECT_BBOX_SIDE
        assert obj["bbox"][3] - obj["bbox"][1] <= MODULE.MAX_OBJECT_BBOX_SIDE


def test_local_occluder_solver_covers_every_target_shape() -> None:
    for index, shape in enumerate(MODULE.SHAPES):
        easy = MODULE.build_easy_layout(5000 + index, "conflict", shape, "blue")
        hard = MODULE.build_hard_layout(6000 + index, easy)
        target = MODULE._object_mask(next(obj for obj in hard["objects"] if obj["role"] == "target"))
        occluder = MODULE._object_mask(next(obj for obj in hard["objects"] if obj["role"] == "occluder"))
        ratio = MODULE._mask_overlap_ratio(target, occluder)
        assert 0.70 <= ratio <= 0.80, (shape, ratio)


def test_recreated_hard_attempts_materially_change_layout_and_stay_valid() -> None:
    easy = MODULE.build_easy_layout(6020, "conflict", "star", "blue")
    layouts = [MODULE.build_recreated_hard_layout(seed, easy) for seed in (6021, 6022, 6023)]
    signatures = []
    for layout in layouts:
        target = MODULE._object_mask(next(obj for obj in layout["objects"] if obj["role"] == "target"))
        occluder = MODULE._object_mask(next(obj for obj in layout["objects"] if obj["role"] == "occluder"))
        result = MODULE.validate_layout(layout, "star", "blue", "hard", target, occluder)
        assert result["valid"], result["issues"]
        signatures.append(tuple(tuple(obj["center"]) for obj in layout["objects"]))
    assert len(set(signatures)) == 3
    assert all(layout["target_geometry"] == easy["target_geometry"] for layout in layouts)


def test_layout_rejects_any_bbox_larger_than_configured_limit() -> None:
    layout = MODULE.build_easy_layout(1912, "consistent", "star", "red")
    layout["objects"][0]["bbox"][2] = (
        layout["objects"][0]["bbox"][0] + MODULE.MAX_OBJECT_BBOX_SIDE + 0.001
    )
    result = MODULE.validate_layout(layout, "star", "red", "easy")
    assert "object_0_bbox_too_large" in result["issues"]


def test_layout_rejects_rendered_target_mask_larger_than_configured_limit() -> None:
    layout = MODULE.build_easy_layout(2012, "consistent", "star", "red")
    target = Image.new("L", (1024, 1024), 0)
    occluder = Image.new("L", (1024, 1024), 0)
    side = int(MODULE.MAX_OBJECT_BBOX_SIDE)
    ImageDraw.Draw(target).rectangle((400, 400, 400 + side, 399 + side), fill=255)
    result = MODULE.validate_layout(layout, "star", "red", "easy", target, occluder)
    assert "target_mask_too_large" in result["issues"]


def test_trusted_renderer_has_a_nonempty_bounded_mask_for_every_shape() -> None:
    for index, shape in enumerate(MODULE.SHAPES):
        obj = {
            "shape": shape,
            "color": "red",
            "center": [300.0, 300.0],
            "bbox": [190.0, 190.0, 410.0, 410.0],
            "rotation": float(index * 17),
            "size": MODULE.MAX_OBJECT_BBOX_SIDE,
            "role": "target",
        }
        bounds = MODULE._object_mask(obj).getbbox()
        assert bounds is not None, shape
        assert bounds[2] - bounds[0] <= MODULE.MAX_OBJECT_BBOX_SIDE, shape
        assert bounds[3] - bounds[1] <= MODULE.MAX_OBJECT_BBOX_SIDE, shape


def test_repeated_irrelevant_shape_is_allowed_after_minimum_diversity() -> None:
    layout = MODULE.build_easy_layout(218, "consistent", "arrow", "cyan")
    distractors = [obj for obj in layout["objects"] if obj["role"] == "distractor"]
    for distractor in distractors[-3:]:
        distractor["shape"] = distractors[0]["shape"]
    target, occluder = _masks()
    result = MODULE.validate_layout(layout, "arrow", "cyan", "easy", target, occluder)
    assert result["distinct_distractor_shapes"] == 7
    assert result["valid"], result["issues"]


@pytest.mark.parametrize(
    "centers, expected",
    [
        ([(100, 100), (100, 250), (100, 400), (300, 100), (300, 250), (300, 400), (500, 100), (500, 250), (500, 400)], "alignment"),
        ([(100, 512), (924, 512), (250, 300), (774, 300), (400, 700), (624, 700), (512, 150)], "symmetry"),
        ([(812, 512), (724, 724), (512, 812), (300, 724), (212, 512), (300, 300), (512, 212), (724, 300)], "ring"),
    ],
)
def test_obvious_regular_layouts_are_rejected(centers: list[tuple[int, int]], expected: str) -> None:
    objects = [
        {
            "shape": MODULE.SHAPES[index], "color": MODULE.COLORS[index % 12],
            "center": list(center), "bbox": [center[0] - 20, center[1] - 20, center[0] + 20, center[1] + 20],
            "rotation": index * 17 + 1, "size": 40 + index * 3, "role": "distractor",
        }
        for index, center in enumerate(centers)
    ]
    issues = MODULE.detect_layout_patterns(objects)
    assert any(expected in issue for issue in issues), issues


def test_deepseek_json_cleanup_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    assert MODULE.parse_json_response("```json\n{\"ok\": true}\n```") == {"ok": True}

    class Completions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_kwargs: Any) -> Any:
            self.calls += 1
            content = "not json" if self.calls == 1 else '{"ok": true}'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    completions = Completions()
    agent = MODULE.DeepSeekAgents.__new__(MODULE.DeepSeekAgents)
    agent.api_key = "secret-never-log"
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)
    assert agent.call("prompt", "planner") == {"ok": True}
    assert completions.calls == 2


def test_deepseek_code_roles_use_low_temperature() -> None:
    class Completions:
        def __init__(self) -> None:
            self.temperatures: list[float] = []

        def create(self, **kwargs: Any) -> Any:
            self.temperatures.append(kwargs["temperature"])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
            )

    completions = Completions()
    agent = MODULE.DeepSeekAgents.__new__(MODULE.DeepSeekAgents)
    agent.api_key = "secret"
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent.call("prompt", "code_generator")
    agent.call("prompt", "code_generator_syntax_repair")
    agent.call("prompt", "code_validator")
    assert completions.temperatures == [0.1, 0.1, 0.1]


def test_code_generator_prompt_contains_runtime_correctness_checklist() -> None:
    captured: dict[str, Any] = {}
    agent = MODULE.DeepSeekAgents.__new__(MODULE.DeepSeekAgents)

    def fake_call(prompt: str, role: str, **_kwargs: Any) -> dict[str, Any]:
        captured.update({"prompt": prompt, "role": role})
        return {"code": "def render_scene():\n    return {}\n"}

    agent.call = fake_call
    agent.generate_code({"canvas": [1024, 1024], "objects": []})
    assert captured["role"] == "code_generator"
    assert "undefined names" in captured["prompt"]
    assert "three integer values in 0..255" in captured["prompt"]
    assert "Copy PLAN_DICT literally and completely" in captured["prompt"]
    assert "do not use imagechops.logical_or on l images" in captured["prompt"].lower()


def test_render_style_json_is_strictly_sanitized() -> None:
    assert MODULE.sanitize_render_style({
        "background_rgb": [235, 238, 242],
        "outline_rgb": [10, 20, 30],
        "outline_width": 3,
    }) == {
        "background_rgb": [235, 238, 242],
        "outline_rgb": [10, 20, 30],
        "outline_width": 3,
    }
    assert MODULE.sanitize_render_style({
        "background_rgb": [255, 0, 0],
        "outline_rgb": [255, 255, 255],
        "outline_width": 99,
    }) == MODULE.DEFAULT_RENDER_STYLE


def test_incomplete_old_scale_checkpoint_is_preserved_for_resume() -> None:
    state = {
        "status": "failed",
        "easy_attempts": [{"attempt": 1}],
        "hard_attempts": [{"attempt": 1}],
        "easy": {"status": "accepted", "result": {}},
        "failure_reason": "old failure",
    }
    assert MODULE._upgrade_incomplete_branch_layout_scale(state) is True
    assert state["status"] == "failed"
    assert state["layout_scale_version"] == MODULE.LAYOUT_SCALE_VERSION
    assert state["easy_attempts"] == [{"attempt": 1}]
    assert state["hard_attempts"] == [{"attempt": 1}]
    assert state["easy"] == {"status": "accepted", "result": {}}
    assert state["failure_reason"] == "old failure"
    assert state["layout_scale_migration"]["preserved_accepted_easy"] is True

    completed = {"status": "completed", "result": {"kept": True}}
    assert MODULE._upgrade_incomplete_branch_layout_scale(completed) is False
    assert completed["result"] == {"kept": True}


def test_syntax_self_check_requests_only_minimal_code_generator_patch() -> None:
    original = '''from PIL import Image
UNCHANGED = "keep me"
def render_scene():
    broken = 01
    return {}
'''

    class RepairAgent:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def repair_syntax(self, plan: dict[str, Any], code: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
            self.calls.append({"plan": plan, "code": code, "diagnostics": diagnostics})
            return {
                "patched_code": code.replace("broken = 01", "broken = 1"),
                "changes": ["removed one leading zero"],
            }

    agent = RepairAgent()
    repaired, history = MODULE.ensure_syntax_with_minimal_repairs(agent, {"case_seed": 1}, original)
    assert MODULE.syntax_diagnostics(repaired)["valid"] is True
    assert 'UNCHANGED = "keep me"' in repaired
    assert len(agent.calls) == 1
    assert agent.calls[0]["diagnostics"]["line"] == 4
    assert history[0]["diagnostics"]["valid"] is False
    assert history[-1]["diagnostics"]["valid"] is True


def test_syntax_repair_rejects_whole_program_rewrite_before_minimal_patch() -> None:
    original = "def render_scene():\n    value = 01\n    return value\n" + "\n".join(
        f"KEEP_{index} = {index}" for index in range(20)
    )

    class RepairAgent:
        def __init__(self) -> None:
            self.calls = 0

        def repair_syntax(self, _plan: dict[str, Any], code: str, _diagnostics: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "patched_code": "def render_scene():\n    return {}\n",
                    "changes": ["rewrote everything"],
                }
            return {
                "patched_code": code.replace("value = 01", "value = 1"),
                "changes": ["removed leading zero"],
            }

    agent = RepairAgent()
    repaired, history = MODULE.ensure_syntax_with_minimal_repairs(agent, {}, original)
    assert "KEEP_19 = 19" in repaired
    assert history[0]["patch_rejected"] == "syntax repair changed too many line positions"
    assert agent.calls == 2


def test_gpu_json_queue_is_fifo_and_recovers_running_jobs(tmp_path: Path) -> None:
    queue = MODULE.PersistentGPUQueue(tmp_path / "gpu_queue.json", poll_seconds=0.001)
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(image)
    queue.enqueue("first", image, "q1", "red", {"order": 1})
    queue.enqueue("second", image, "q2", "blue", {"order": 2})
    first = queue.claim_next()
    assert first is not None and first["job_id"] == "first"
    queue.reset_inflight()
    recovered = queue.claim_next()
    assert recovered is not None and recovered["job_id"] == "first"
    queue.complete("first", {"all_correct": True}, None)
    second = queue.claim_next()
    assert second is not None and second["job_id"] == "second"
    queue.complete("second", {"all_correct": False}, None)
    assert queue.wait("first", 1)["all_correct"] is True
    assert queue.wait("second", 1)["all_correct"] is False


def test_gpu_job_cancelled_by_ctrl_c_can_be_requeued(tmp_path: Path) -> None:
    queue = MODULE.PersistentGPUQueue(tmp_path / "gpu_queue.json", poll_seconds=0.001)
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    Image.new("RGB", (8, 8), "red").save(first_image)
    Image.new("RGB", (8, 8), "blue").save(second_image)
    queue.enqueue("same-job", first_image, "old", "red", {"attempt": 1})
    queue.fail_unfinished({"type": "KeyboardInterrupt", "message": "cancelled"})
    queue.enqueue("same-job", second_image, "new", "blue", {"attempt": 2})
    recovered = queue.claim_next()
    assert recovered is not None
    assert recovered["job_id"] == "same-job"
    assert recovered["image_path"] == str(second_image.resolve())
    assert recovered["question"] == "new"
    assert recovered["target"] == "blue"


def test_gpu_consumer_runs_jobs_serially_in_queue_order(tmp_path: Path) -> None:
    queue = MODULE.PersistentGPUQueue(tmp_path / "gpu_queue.json", poll_seconds=0.001)
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    Image.new("RGB", (8, 8), "red").save(first_image)
    Image.new("RGB", (8, 8), "blue").save(second_image)
    queue.enqueue("first", first_image, "first question", "red", {})
    queue.enqueue("second", second_image, "second question", "red", {})

    class SerialInference(FakeInference):
        def __init__(self) -> None:
            super().__init__(["red"] * 6)
            self.active = 0
            self.max_active = 0

        def generate_answer_with_metrics(self, **kwargs: Any) -> Any:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            time.sleep(0.002)
            try:
                return super().generate_answer_with_metrics(**kwargs)
            finally:
                self.active -= 1

    inference = SerialInference()
    done = threading.Event()
    done.set()
    errors: list[BaseException] = []
    MODULE.consume_gpu_queue(queue, inference, done, errors)
    assert not errors
    assert inference.max_active == 1
    assert [Path(call["image_path"]).name for call in inference.calls] == ["first.png"] * 3 + ["second.png"] * 3


def test_planner_receives_previous_gpu_test_feedback() -> None:
    layout = MODULE.build_easy_layout(919, "consistent", "star", "red")
    captured: list[str] = []
    agent = MODULE.DeepSeekAgents.__new__(MODULE.DeepSeekAgents)

    def fake_call(prompt: str, role: str, retries: int = 3) -> dict[str, Any]:
        captured.append(prompt)
        return {"plan": layout}

    agent.call = fake_call
    feedback = {
        "test_result": {"top1_answers": ["blue", "red", "blue"], "correct_count": 1},
        "retry_instructions": ["make the target clearer"],
    }
    result = agent.plan(layout, retry_feedback=feedback)
    assert result["plan"] == layout
    assert result["render_style"] == MODULE.DEFAULT_RENDER_STYLE
    assert "make the target clearer" in captured[0]
    assert "never write Python" in captured[0]


def test_worker_cli_accepts_64_and_rejects_more() -> None:
    assert MODULE.parse_args([]).workers == 16
    assert MODULE.parse_args(["--workers", "64"]).workers == 64
    with pytest.raises(SystemExit):
        MODULE.parse_args(["--workers", "65"])


def test_concurrent_runner_submits_one_process_task_per_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generator = _generator_shell(tmp_path)
    combo = {
        "id": "121", "shape": "star", "text_color": "red", "conflict_color": "blue",
        "case_seeds": {
            "consistent_easy": 1, "consistent_hard": 2,
            "conflict_easy": 3, "conflict_hard": 4,
        },
    }
    generator.state.update({"manifest": {"combinations": [combo]}, "combinations": {}})
    generator.gpu_queue_path = tmp_path / "gpu_queue.json"
    generator.branch_checkpoint_dir = tmp_path / "branches"
    generator.model_path = tmp_path / "model"
    generator.worker_count = 64
    generator.gpu_wait_timeout = 2.0
    submitted: list[dict[str, Any]] = []
    executor_workers: list[int] = []
    calibration = {"normalized_entropy": 0.2, "runs": []}

    def branch_result(spec: dict[str, Any]) -> dict[str, Any]:
        value = {
            "easy": {"image_path": "easy.png", "calibration": calibration},
            "hard": {"image_path": "hard.png", "calibration": calibration},
            "entropy_check": {"hard_success": True},
        }
        return {
            "status": "completed", "item_id": "121", "branch": spec["branch"],
            "result": value, "state": {"status": "completed"},
        }

    class FakeExecutor:
        def __init__(self, max_workers: int, mp_context: Any):
            executor_workers.append(max_workers)

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def submit(self, _function: Any, spec: dict[str, Any]) -> Any:
            submitted.append(spec)
            future: concurrent.futures.Future[Any] = concurrent.futures.Future()
            future.set_result(branch_result(spec))
            return future

        def shutdown(self, wait: bool, cancel_futures: bool) -> None:
            assert wait is True
            assert cancel_futures is True

    import concurrent.futures
    monkeypatch.setattr(MODULE.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(MODULE, "_load_extended_inference", lambda _path: object())
    generator._run_concurrent()
    assert executor_workers == [2]
    assert [spec["branch"] for spec in submitted] == ["consistent", "conflict"]
    assert len(generator.output[0]["items"]) == 1


def test_process_pool_cleanup_terminates_and_reaps_workers() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.alive = True
            self.terminated = False
            self.killed = False
            self.joined = False

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True

        def join(self, timeout: float) -> None:
            self.joined = True
            self.alive = False

        def kill(self) -> None:
            self.killed = True
            self.alive = False

    processes = {index: FakeProcess() for index in range(3)}

    class FakeExecutor:
        _processes = processes

        def shutdown(self, wait: bool, cancel_futures: bool) -> None:
            assert wait is False
            assert cancel_futures is True

    MODULE._terminate_process_pool(FakeExecutor())
    assert all(process.terminated for process in processes.values())
    assert all(process.joined and not process.is_alive() for process in processes.values())


def test_concurrent_keyboard_interrupt_invokes_pool_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generator = _generator_shell(tmp_path)
    combo = {
        "id": "121", "shape": "star", "text_color": "red", "conflict_color": "blue",
        "case_seeds": {
            "consistent_easy": 1, "consistent_hard": 2,
            "conflict_easy": 3, "conflict_hard": 4,
        },
    }
    generator.state.update({"manifest": {"combinations": [combo]}, "combinations": {}})
    generator.gpu_queue_path = tmp_path / "gpu_queue.json"
    generator.branch_checkpoint_dir = tmp_path / "branches"
    generator.model_path = tmp_path / "model"
    generator.worker_count = 2
    generator.gpu_wait_timeout = 2.0
    executor_instance: list[Any] = []
    cleaned: list[Any] = []

    class InterruptingExecutor:
        def __init__(self, **_kwargs: Any) -> None:
            executor_instance.append(self)

        def submit(self, _function: Any, _spec: dict[str, Any]) -> Any:
            future: concurrent.futures.Future[Any] = concurrent.futures.Future()
            future.set_exception(KeyboardInterrupt())
            return future

    import concurrent.futures
    monkeypatch.setattr(MODULE.concurrent.futures, "ProcessPoolExecutor", InterruptingExecutor)
    monkeypatch.setattr(MODULE, "_load_extended_inference", lambda _path: object())
    monkeypatch.setattr(MODULE, "_terminate_process_pool", lambda executor: cleaned.append(executor))
    with pytest.raises(KeyboardInterrupt):
        generator._run_concurrent()
    assert cleaned == executor_instance


def test_sandbox_process_group_can_be_terminated() -> None:
    import subprocess

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    MODULE._terminate_subprocess_group(process, grace_seconds=0.2)
    assert process.poll() is not None


def test_worker_sigterm_handler_unwinds_as_base_exception() -> None:
    with pytest.raises(MODULE.WorkerShutdown):
        MODULE._raise_worker_shutdown(15, None)


def test_dangerous_code_is_rejected_and_safe_code_runs_in_child(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        MODULE.validate_generated_code(
            "import os\ndef render_scene():\n    os.system('id')\n"
        )
    with pytest.raises(ValueError, match="Unsafe"):
        MODULE.validate_generated_code(
            "from PIL import Image\ndef render_scene():\n    return Image.open('/etc/passwd')\n"
        )
    layout = {"canvas": [1024, 1024], "objects": []}
    code = f'''from PIL import Image
def render_scene():
    image = Image.new("RGB", (1024, 1024), "white")
    target = Image.new("L", (1024, 1024), 0)
    occluder = Image.new("L", (1024, 1024), 0)
    return {{"image": image, "target_mask": target, "occluder_mask": occluder, "layout": {layout!r}}}
'''
    output = MODULE.execute_generated_code(code, tmp_path)
    assert (output / "image.png").is_file()
    assert (output / "layout.json").is_file()


def test_candidate_pipeline_uses_json_planner_and_never_generated_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    layout = MODULE.build_easy_layout(3030, "consistent", "star", "red")

    class JsonOnlyAgents:
        def plan(self, local_layout: dict[str, Any], retry_feedback: Any = None) -> dict[str, Any]:
            assert local_layout is layout
            return {
                "plan": local_layout,
                "render_style": MODULE.sanitize_render_style({}),
                "planner_json": {"render_style": {}},
                "scene_notes": [],
                "planner_issues": [],
            }

        def generate_code(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("Code Generator must not be called")

        def validate_code(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("Code Validator must not be called")

        def repair_syntax(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("Syntax Repair must not be called")

    generator = MODULE.DatasetGenerator.__new__(MODULE.DatasetGenerator)
    generator.agents = JsonOnlyAgents()
    generator.inference = object()
    monkeypatch.setattr(
        MODULE,
        "execute_generated_code",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("generated code exec must not be called")),
    )
    calibration = {
        "top1_answer": "red",
        "top1_answers": ["red", "red", "red"],
        "ground_truth_answer": "red",
        "answer_prob": 0.9,
        "answer_class_probabilities": {"red": 0.9},
        "entropy": 0.1,
        "normalized_entropy": 0.1,
        "parse_success": True,
        "correct_count": 3,
        "all_correct": True,
        "elapsed_seconds": 0.01,
        "runs": [{"top1_answer": "red"}] * 3,
    }
    result = generator._candidate_pipeline(
        layout,
        tmp_path,
        MODULE.QUESTION_TEMPLATE.format(shape="star"),
        "red",
        1,
        None,
        model_test=lambda *_args: calibration,
    )
    assert result.failure_reason == "pass"
    assert result.geometry_valid is True
    assert result.agent_results["renderer"]["type"] == "trusted_local_renderer"
    assert "code_generator" not in result.agent_results


def test_same_image_is_tested_three_times_with_identical_prompt(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(image)
    inference = FakeInference(["red", "red", "red"], [0.1, 0.2, 0.3])
    question = MODULE.QUESTION_TEMPLATE.format(shape="star")
    aggregate = MODULE.test_image_three_times(inference, image, question, "red")
    assert len(inference.calls) == 3
    assert {call["prompt"] for call in inference.calls} == {MODULE.IMAGE_TEST_PROMPT.format(question=question)}
    assert all(call["image_path"] == str(image) for call in inference.calls)
    assert aggregate["all_correct"] is True
    assert aggregate["normalized_entropy"] == pytest.approx(0.2)


def test_any_wrong_answer_fails_three_of_three() -> None:
    runs = [
        {"top1_answer": answer, "parse_success": True, "normalized_entropy": 0.2, "entropy": 0.5,
         "answer_prob": 0.5, "answer_class_probabilities": {}, "elapsed_seconds": 0.1}
        for answer in ("red", "blue", "red")
    ]
    result = MODULE.aggregate_model_runs(runs, "red")
    assert result["correct_count"] == 2
    assert result["all_correct"] is False


def test_entropy_gap_boundary_is_strict() -> None:
    candidate = MODULE.CandidateResult(
        attempt=1, seed=1, difficulty="hard", geometry_valid=True,
        all_correct=True, entropy_gap=0.25,
    )
    assert MODULE.hard_candidate_passes(candidate) is False
    candidate.entropy_gap = 0.2500001
    assert MODULE.hard_candidate_passes(candidate) is True


def test_best_hard_candidate_selection_order() -> None:
    candidates = [
        MODULE.CandidateResult(attempt=1, seed=1, difficulty="hard", geometry_valid=True,
                               calibration={"normalized_entropy": 0.9}, correct_count=2, entropy_gap=0.8),
        MODULE.CandidateResult(attempt=2, seed=2, difficulty="hard", geometry_valid=True,
                               calibration={"normalized_entropy": 0.7}, correct_count=3, entropy_gap=0.2),
        MODULE.CandidateResult(attempt=3, seed=3, difficulty="hard", geometry_valid=True,
                               calibration={"normalized_entropy": 0.8}, correct_count=3, entropy_gap=0.3),
    ]
    best, reason = MODULE.select_best_hard(candidates)
    assert best is candidates[2]
    assert "correct_count=3" in reason


def _generator_shell(tmp_path: Path) -> Any:
    generator = MODULE.DatasetGenerator.__new__(MODULE.DatasetGenerator)
    generator.state = {"combinations": {}, "failures": []}
    generator.output_path = tmp_path / "out.json"
    generator.state_path = tmp_path / "out.state.json"
    generator.image_dir = tmp_path / "images"
    generator.output = [{"category": "colour", "items": []}]
    generator.irr_path = "irr.png"
    generator.null_path = "null.png"
    generator.agents = object()
    generator.inference = object()
    generator._persist_state = lambda: MODULE.atomic_write_json(generator.state_path, generator.state)
    return generator


def test_easy_stops_after_five_and_hard_after_ten(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    generator = _generator_shell(tmp_path)
    combo = {
        "id": "121", "shape": "star", "text_color": "red", "conflict_color": "blue",
        "case_seeds": {"consistent_easy": 1, "consistent_hard": 2},
    }
    calls: list[str] = []

    def failed_pipeline(layout: dict[str, Any], *_args: Any, **_kwargs: Any) -> Any:
        calls.append(layout["difficulty"])
        return MODULE.CandidateResult(attempt=len(calls), seed=layout["case_seed"], difficulty=layout["difficulty"])

    monkeypatch.setattr(generator, "_candidate_pipeline", failed_pipeline)
    with pytest.raises(MODULE.GenerationStopped, match="easy failed"):
        generator._run_branch(combo, "consistent", "red")
    assert calls == ["easy"] * 5

    # Lock a synthetic easy result so resume goes directly to exactly ten hard attempts.
    layout = MODULE.build_easy_layout(77, "consistent", "star", "red")
    generator.image_dir.mkdir(exist_ok=True)
    layout_path = generator.image_dir / "easy.layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    rel = Path("images/easy.layout.json").as_posix()
    generator.state["combinations"]["121"] = {
        "status": "in_progress",
        "branches": {"consistent": {"status": "in_progress", "easy": {
            "status": "accepted", "result": {
                "layout_path": rel,
                "calibration": {"normalized_entropy": 0.1},
            },
        }}},
    }
    calls.clear()
    with pytest.raises(MODULE.GenerationStopped, match="no valid hard"):
        generator._run_branch(combo, "consistent", "red")
    assert calls == ["hard"] * 10


def test_locked_easy_is_not_regenerated_on_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    generator = _generator_shell(tmp_path)
    combo = {
        "id": "122", "shape": "cross", "text_color": "green", "conflict_color": "pink",
        "case_seeds": {"consistent_easy": 10, "consistent_hard": 20},
    }
    easy = MODULE.build_easy_layout(88, "consistent", "cross", "green")
    generator.image_dir.mkdir(exist_ok=True)
    (generator.image_dir / "locked.layout.json").write_text(json.dumps(easy), encoding="utf-8")
    generator.state["combinations"]["122"] = {
        "status": "in_progress", "branches": {"consistent": {"status": "in_progress", "easy": {
            "status": "accepted", "result": {
                "layout_path": "images/locked.layout.json",
                "calibration": {"normalized_entropy": 0.1},
            },
        }}},
    }
    seen: list[str] = []

    def hard_only(layout: dict[str, Any], *_args: Any, **_kwargs: Any) -> Any:
        seen.append(layout["difficulty"])
        return MODULE.CandidateResult(attempt=len(seen), seed=layout["case_seed"], difficulty="hard")

    monkeypatch.setattr(generator, "_candidate_pipeline", hard_only)
    with pytest.raises(MODULE.GenerationStopped):
        generator._run_branch(combo, "consistent", "green")
    assert seen == ["hard"] * 10


def test_atomic_state_creation_and_resume(tmp_path: Path) -> None:
    output = tmp_path / "generated.json"
    args = argparse.Namespace(
        input_dataset=str(ROOT / "datasets/dataset_test.json"),
        prior_pool=str(ROOT / "datasets/color_prior_pool.json"),
        output_dataset=str(output), image_dir=str(tmp_path / "images"),
        model_path=str(ROOT / "qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct"),
        seed=555, resume=False, dry_run=False,
    )
    first = MODULE.DatasetGenerator(args)
    assert output.is_file() and first.state_path.is_file()
    assert not list(tmp_path.glob(".*.tmp"))
    saved = json.loads(first.state_path.read_text(encoding="utf-8"))
    assert saved["seed"] == 555
    resumed_args = argparse.Namespace(**{**vars(args), "seed": None, "resume": True})
    resumed = MODULE.DatasetGenerator(resumed_args)
    assert resumed.state["seed"] == 555
    mismatch_args = argparse.Namespace(**{**vars(args), "seed": 556, "resume": True})
    with pytest.raises(ValueError, match="seed mismatch"):
        MODULE.DatasetGenerator(mismatch_args)


def test_resume_migrates_pre_concurrency_branch_state(tmp_path: Path) -> None:
    output = tmp_path / "generated.json"
    args = argparse.Namespace(
        input_dataset=str(ROOT / "datasets/dataset_test.json"),
        prior_pool=str(ROOT / "datasets/color_prior_pool.json"),
        output_dataset=str(output), image_dir=str(tmp_path / "images"),
        model_path=str(ROOT / "qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct"),
        seed=777, workers=4, gpu_queue=None, gpu_wait_timeout=30.0,
        resume=False, dry_run=False,
    )
    first = MODULE.DatasetGenerator(args)
    state = json.loads(first.state_path.read_text(encoding="utf-8"))
    state["config"].pop("gpu_queue")
    state["config"].pop("branch_checkpoint_dir")
    combo = state["manifest"]["combinations"][0]
    state["combinations"] = {
        combo["id"]: {
            "status": "in_progress",
            "branches": {"consistent": {
                "status": "in_progress",
                "easy_attempts": [{"attempt": 1, "failure_reason": "legacy failure"}],
            }},
        }
    }
    MODULE.atomic_write_json(first.state_path, state)
    resumed_args = argparse.Namespace(**{**vars(args), "seed": None, "resume": True})
    resumed = MODULE.DatasetGenerator(resumed_args)
    checkpoint = MODULE._branch_checkpoint_path(
        resumed.branch_checkpoint_dir, combo["id"], "consistent"
    )
    migrated = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert migrated["easy_attempts"][0]["failure_reason"] == "legacy failure"


def test_staging_output_omits_priors_and_current_loader_rejects(tmp_path: Path) -> None:
    generator = _generator_shell(tmp_path)
    calibration = {"normalized_entropy": 0.1, "runs": []}
    branch = {
        "easy": {"image_path": "images/easy.png", "calibration": calibration},
        "hard": {"image_path": "images/hard.png", "calibration": calibration},
        "entropy_check": {"hard_success": False},
    }
    combo = {"id": "121", "shape": "star", "text_color": "red", "conflict_color": "blue"}
    item = generator._build_item(combo, branch, branch)
    assert "selected_text_priors" not in item
    dataset = tmp_path / "staging.json"
    dataset.write_text(json.dumps([{"category": "colour", "items": [item]}]), encoding="utf-8")
    from confidence_test.dataset_utils import load_evaluation_cases
    with pytest.raises(ValueError, match="selected_text_priors"):
        load_evaluation_cases(dataset)


def test_recreate_cli_and_current_invalid_architecture_preflight() -> None:
    assert MODULE.parse_args(["--recreate"]).recreate is True
    with pytest.raises(SystemExit):
        MODULE.parse_args(["--recreate", "--dry-run"])

    generator = MODULE.RecreateDatasetGenerator.__new__(MODULE.RecreateDatasetGenerator)
    generator.datasets_root = SCRIPT.parent / "datasets"
    generator.invalid_root = generator.datasets_root / "invalid_datasets"
    generator.valid_root = generator.datasets_root / "valid_datasets"
    generator.invalid_path = generator.invalid_root / "generated_shape_color_dataset.json"
    generator.source_summary_path = generator.datasets_root / "generated_shape_color_dataset.summary.json"
    generator.invalid = json.loads(generator.invalid_path.read_text(encoding="utf-8"))
    records, recovered = generator._resolve_source_records()
    assert len(records) == len(generator.invalid["items"])
    assert recovered == sum(record["source_kind"] == "original_fallback" for record in records)
    assert all(record["easy_layout_path"].is_file() for record in records)


def test_recreate_failure_analyst_receives_reason_and_complete_layout() -> None:
    layout = MODULE.build_hard_layout(
        9202, MODULE.build_easy_layout(9201, "conflict", "star", "blue")
    )
    captured: dict[str, str] = {}
    agent = MODULE.DeepSeekAgents.__new__(MODULE.DeepSeekAgents)

    def fake_call(prompt: str, role: str, retries: int = 3) -> dict[str, Any]:
        captured["prompt"] = prompt
        captured["role"] = role
        return {"analysis": ["answer mismatch"], "retry_instructions": ["vary the hard presentation"]}

    agent.call = fake_call
    result = agent.review_recreate_failure(
        layout,
        "image_not_correct_in_all_three_runs",
        {
            "top1_answer": "red", "top1_answers": ["red"] * 3,
            "correct_count": 0, "normalized_entropy": 0.2,
        },
        0.1,
        "blue",
    )
    assert captured["role"] == "recreate_failure_analyst"
    assert "image_not_correct_in_all_three_runs" in captured["prompt"]
    assert json.dumps(layout, ensure_ascii=False) in captured["prompt"]
    assert result["retry_instructions"] == ["vary the hard presentation"]


def test_recreate_worker_stops_after_twenty_and_feeds_analysis_forward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    easy_layout = MODULE.build_easy_layout(9301, "conflict", "star", "blue")
    easy_layout_path = tmp_path / "easy.layout.json"
    easy_layout_path.write_text(json.dumps(easy_layout), encoding="utf-8")
    seen_feedback: list[Any] = []
    analyses: list[str] = []

    class FakeAgents:
        def __init__(self, _path: Path):
            pass

        def review_recreate_failure(
            self, _layout: dict[str, Any], reason: str, _calibration: Any,
            _easy_entropy: float, _target: str,
        ) -> dict[str, Any]:
            analyses.append(reason)
            return {"analysis": [reason], "retry_instructions": [f"advice-{len(analyses)}"]}

    def fake_hard(seed: int, _easy: dict[str, Any]) -> dict[str, Any]:
        return {"case_seed": seed, "difficulty": "hard", "objects": [], "canvas": [1024, 1024]}

    def fake_pipeline(
        _self: Any, layout: dict[str, Any], _work: Path, _question: str, _target: str,
        attempt: int, _easy_entropy: float, **kwargs: Any,
    ) -> Any:
        seen_feedback.append(kwargs.get("retry_feedback"))
        return MODULE.CandidateResult(
            attempt=attempt,
            seed=layout["case_seed"],
            difficulty="hard",
            failure_reason="image_not_correct_in_all_three_runs",
        )

    monkeypatch.setattr(MODULE, "DeepSeekAgents", FakeAgents)
    monkeypatch.setattr(MODULE, "PersistentGPUQueue", lambda _path: object())
    monkeypatch.setattr(MODULE, "build_hard_layout", fake_hard)
    monkeypatch.setattr(MODULE.DatasetGenerator, "_candidate_pipeline", fake_pipeline)
    monkeypatch.setattr(MODULE.signal, "signal", lambda *_args: None)
    result = MODULE._run_recreate_process({
        "source_id": "001",
        "target": "blue",
        "question": MODULE.QUESTION_TEMPLATE.format(shape="star"),
        "easy_entropy": 0.1,
        "easy_layout_path": str(easy_layout_path),
        "case_seed": 1,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "gpu_queue_path": str(tmp_path / "queue.json"),
        "gpu_wait_timeout": 1.0,
        "api_config_path": str(tmp_path / "api.json"),
    })
    assert result["status"] == "failed"
    assert result["state"]["attempt_count"] == MODULE.RECREATE_HARD_MAX_ATTEMPTS == 20
    assert len(analyses) == 20
    assert seen_feedback[0] is None
    assert seen_feedback[1]["retry_instructions"] == ["advice-1"]


def test_recreate_uses_one_worker_per_pending_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generator = MODULE.RecreateDatasetGenerator.__new__(MODULE.RecreateDatasetGenerator)
    generator.records = [
        {
            "source_id": f"{index:03d}", "question": "q", "target": "blue",
            "easy_entropy": 0.1, "easy_layout_path": tmp_path / f"{index}.json",
        }
        for index in range(1, 4)
    ]
    generator.invalid = {"items": []}
    generator.state = {"seed": 1, "items": {}, "failures": []}
    generator.gpu_queue_path = tmp_path / "queue.json"
    generator.model_path = tmp_path / "model"
    generator.gpu_wait_timeout = 1.0
    generator.checkpoint_dir = tmp_path / "checkpoints"
    published: list[str] = []
    generator._publish_success = lambda record, _payload: published.append(record["source_id"])
    generator._finalize_recreate_cycle = lambda _failures: None
    worker_counts: list[int] = []

    class FakeQueue:
        def __init__(self, _path: Path):
            pass

        def reset_inflight(self) -> None:
            pass

        def fail_unfinished(self, _error: Any) -> None:
            pass

    class FakeExecutor:
        def __init__(self, max_workers: int, mp_context: Any):
            worker_counts.append(max_workers)

        def submit(self, _function: Any, spec: dict[str, Any]) -> Any:
            future: Any = MODULE.concurrent.futures.Future()
            future.set_result({
                "status": "completed", "source_id": spec["source_id"],
                "result": {"candidate": {}, "attempt_count": 1},
            })
            return future

        def shutdown(self, wait: bool, cancel_futures: bool) -> None:
            assert wait is True and cancel_futures is True

    monkeypatch.setattr(MODULE, "PersistentGPUQueue", FakeQueue)
    monkeypatch.setattr(MODULE, "_load_extended_inference", lambda _path: object())
    monkeypatch.setattr(MODULE, "consume_gpu_queue", lambda *_args: None)
    monkeypatch.setattr(MODULE.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    generator.run()
    assert worker_counts == [3]
    assert published == ["001", "002", "003"]


def test_recreate_publish_renumbers_all_artifacts_and_appends_valid_item(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_images = source_root / "images"
    source_images.mkdir(parents=True)
    question = MODULE.QUESTION_TEMPLATE.format(shape="star")
    layouts = {
        "consistent_easy": MODULE.build_easy_layout(9401, "consistent", "star", "red"),
        "conflict_easy": MODULE.build_easy_layout(9402, "conflict", "star", "blue"),
    }
    layouts["consistent_hard"] = MODULE.build_hard_layout(9403, layouts["consistent_easy"])
    layouts["conflict_hard"] = MODULE.build_hard_layout(9404, layouts["conflict_easy"])
    source_item = {
        "id": "001",
        "question": question,
        "answer": "red",
        "conflict_answer": "blue",
        "groups": {},
    }
    stem_names = {
        "consistent_easy": "001_consist_easy",
        "consistent_hard": "001_consist_hard",
        "conflict_easy": "001_conflict_easy",
        "conflict_hard": "001_conflict_hard",
    }
    for group_name, layout in layouts.items():
        rendered = MODULE.render_scene_locally(layout, tmp_path / f"render-{group_name}")
        MODULE._publish_artifacts(rendered, source_images, stem_names[group_name])
        expected = "red" if group_name.startswith("consistent") else "blue"
        source_item["groups"][group_name] = {
            "image": f"images/{stem_names[group_name]}.png",
            "answer": expected,
            "ground_truth_answer": expected,
            "entropy": 0.1,
            "normalized_entropy": 0.1,
            "correct_count": 3,
            "all_correct": True,
            "parse_success": True,
            "runs": [],
        }

    accepted_layout = MODULE.build_hard_layout(9501, layouts["conflict_easy"])
    accepted_artifacts = MODULE.render_scene_locally(accepted_layout, tmp_path / "accepted")
    runs = [
        {
            "top1_answer": "blue", "ground_truth_answer": "blue",
            "entropy": 1.2, "normalized_entropy": 0.6,
            "parse_success": True, "error": None,
        }
        for _ in range(3)
    ]
    calibration = {
        "top1_answer": "blue", "top1_answers": ["blue"] * 3,
        "ground_truth_answer": "blue", "entropy": 1.2,
        "normalized_entropy": 0.6, "correct_count": 3,
        "all_correct": True, "parse_success": True, "runs": runs,
    }
    generator = MODULE.RecreateDatasetGenerator.__new__(MODULE.RecreateDatasetGenerator)
    generator.valid_root = tmp_path / "valid"
    generator.image_dir = generator.valid_root / "images"
    generator.output_path = generator.valid_root / "generated_shape_color_dataset.json"
    generator.state_path = generator.valid_root / "generated_shape_color_dataset.recreate.state.json"
    generator.invalid_root = source_root
    generator.invalid_path = source_root / "generated_shape_color_dataset.json"
    generator.invalid = {
        "dataset_type": "invalid", "items": [source_item],
        "item_count": 1, "group_count": 4,
    }
    generator.output = {
        "dataset_type": "valid", "items": [{"id": "038"}],
        "item_count": 1, "group_count": 4,
    }
    generator.state = {"seed": 1, "items": {}, "failures": []}
    record = {
        "source_id": "001",
        "invalid_item": source_item,
        "source_item": source_item,
        "source_root": source_root,
        "source_kind": "invalid",
        "easy_entropy": 0.1,
        "target": "blue",
    }
    generator._publish_success(record, {
        "status": "completed",
        "result": {
            "candidate": {"artifact_dir": str(accepted_artifacts), "calibration": calibration},
            "attempt_count": 2,
        },
    })
    appended = generator.output["items"][-1]
    assert appended["id"] == "039"
    assert appended["groups"]["conflict_hard"]["answer"] == "blue"
    assert appended["groups"]["conflict_hard"]["normalized_entropy"] == 0.6
    assert len(list(generator.image_dir.iterdir())) == 16
    assert all(value["image"].startswith("images/039_") for value in appended["groups"].values())
    assert generator.state["items"]["001"]["status"] == "published"
    assert generator.invalid["items"] == []
    assert not list(source_images.iterdir())
    saved = json.loads(generator.output_path.read_text(encoding="utf-8"))
    assert saved["items"][-1] == appended


def test_recreate_cycle_renumbers_remaining_invalid_and_resets_resume_state(tmp_path: Path) -> None:
    invalid_root = tmp_path / "invalid_datasets"
    images = invalid_root / "images"
    images.mkdir(parents=True)
    suffixes = (
        ".png", ".layout.json", ".target_mask.png", ".occluder_mask.png",
    )
    stems = ("consist_easy", "consist_hard", "conflict_easy", "conflict_hard")
    items = []
    for old_id in ("002", "005"):
        groups = {}
        for stem in stems:
            for suffix in suffixes:
                (images / f"{old_id}_{stem}{suffix}").write_bytes(b"artifact")
            group_name = stem.replace("consist", "consistent")
            groups[group_name] = {"image": f"images/{old_id}_{stem}.png"}
        items.append({"id": old_id, "groups": groups})

    generator = MODULE.RecreateDatasetGenerator.__new__(MODULE.RecreateDatasetGenerator)
    generator.invalid_root = invalid_root
    generator.invalid_path = invalid_root / "generated_shape_color_dataset.json"
    generator.invalid = {"dataset_type": "invalid", "items": items}
    generator.valid_root = tmp_path / "valid_datasets"
    generator.valid_root.mkdir()
    generator.output_path = generator.valid_root / "generated_shape_color_dataset.json"
    generator.image_dir = generator.valid_root / "images"
    generator.model_path = tmp_path / "model"
    generator.gpu_queue_path = generator.valid_root / "queue.json"
    generator.state_path = generator.valid_root / "state.json"
    generator.state = {
        "cycle": 1,
        "items": {"002": {"status": "failed"}, "005": {"status": "failed"}},
        "failures": [{"source_id": "002"}, {"source_id": "005"}],
    }
    generator._finalize_recreate_cycle([{"source_id": "002"}, {"source_id": "005"}])
    assert [item["id"] for item in generator.invalid["items"]] == ["001", "002"]
    assert all(
        group["image"].split("/")[-1].startswith(item["id"] + "_")
        for item in generator.invalid["items"] for group in item["groups"].values()
    )
    assert len(list(images.glob("001_*"))) == 16
    assert len(list(images.glob("002_*"))) == 16
    assert not list(images.glob("005_*"))
    assert generator.state["cycle"] == 2
    assert generator.state["items"] == {}
    assert generator.state["failures"] == []
    assert generator.state["cycles"][-1]["renumber_mapping"] == {"002": "001", "005": "002"}
