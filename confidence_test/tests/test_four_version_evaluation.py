from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from confidence_test.answer_metrics import (
    compute_answer_metrics,
    parse_answer_classes,
    parse_answer_output,
)
from confidence_test.dataset_utils import CONDITIONS, load_evaluation_cases
from confidence_test.four_version_evaluation import (
    EvaluationRunner,
    canonical_conditions,
    canonical_variants,
    validate_shared_checkpoint_consistency,
)
from confidence_test.inference_extension import AnswerGenerationResult
from confidence_test.io_utils import (
    full_to_simplified,
    load_json,
    write_compact_simplified_json,
    write_result_pair,
)


TEST_ROOT = Path(__file__).resolve().parent / ".runtime"
PROMPT_SHA256 = "e7f8b7a2b3fff2447c4b5708934dd540032abbe24da612bfd25a6290f356c802"
QUESTION = "What is the color? Choose from: red, blue."


@pytest.fixture(autouse=True)
def clean_test_root() -> Any:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    TEST_ROOT.mkdir(parents=True)
    yield TEST_ROOT
    shutil.rmtree(TEST_ROOT, ignore_errors=True)


def _write_image(path: Path, color: str = "white") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _dataset(question: str | dict[str, str] = QUESTION, missing: str | None = None) -> Path:
    root = TEST_ROOT / "datasets"
    image_clue = {
        "null": "images/null.png",
        "irr": "images/1_irr.png",
        "consistent": {
            "easy": "images/1_consist_easy.png",
            "hard": "images/1_consist_hard.png",
        },
        "conflict": {
            "easy": "images/1_conflict_easy.png",
            "hard": "images/1_conflict_hard.png",
        },
    }
    names = {
        "null": "null.png",
        "irr": "1_irr.png",
        "consistent_easy": "1_consist_easy.png",
        "consistent_hard": "1_consist_hard.png",
        "conflict_easy": "1_conflict_easy.png",
        "conflict_hard": "1_conflict_hard.png",
    }
    for condition, name in names.items():
        if condition != missing:
            _write_image(root / "images" / name)
    payload = [
        {
            "category": "colour",
            "items": [
                {
                    "id": "1",
                    "question": question,
                    "answer": "red",
                    "text_ans": "red",
                    "conflict_ans": "blue",
                    "selected_text_priors": [
                        {"clue": "A weak red clue.", "confidence_bin": "0.4-0.6"},
                        {"clue": "A second clue.", "confidence_bin": "0.6-0.8"},
                    ],
                    "image_clue": image_clue,
                }
            ],
        }
    ]
    path = root / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@dataclass
class FakeConfidenceResult:
    confidence_label: str = "Likely(0.6-0.7)"
    hard_confidence_midpoint: float = 0.65
    soft_confidence: float = 0.67
    class_logits: dict[str, float] | None = None
    class_probabilities: dict[str, float] | None = None
    class_token_variants: dict[str, list[int]] | None = None
    raw_output: str = "Likely(0.6-0.7)"
    rendered_prompt: str = "must-not-be-saved"
    hard_label_parsed: bool = True
    hidden_state_collected: bool = False

    def __post_init__(self) -> None:
        self.class_logits = self.class_logits or {"Likely(0.6-0.7)": 1.0}
        self.class_probabilities = self.class_probabilities or {"Likely(0.6-0.7)": 1.0}
        self.class_token_variants = self.class_token_variants or {"Likely(0.6-0.7)": [1]}


class FakeInference:
    instances = 0

    def __init__(self, interrupt_at: int | None = None):
        type(self).instances += 1
        self.model = object()
        self.processor = object()
        self.calls: list[dict[str, Any]] = []
        self.interrupt_at = interrupt_at

    def generate_answer_with_metrics(
        self,
        prompt: str,
        answer_classes: list[str],
        image_path: str | None = None,
        max_new_tokens: int = 24,
    ) -> AnswerGenerationResult:
        self.calls.append({"prompt": prompt, "image": image_path, "tokens": max_new_tokens})
        if self.interrupt_at is not None and len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt()
        answer = answer_classes[0] if answer_classes else "red"
        probabilities = {value: (0.7 if index == 0 else 0.3) for index, value in enumerate(answer_classes)}
        return AnswerGenerationResult(
            raw_output=f"**Answer**: {answer}",
            answer=answer,
            normalized_answer=answer,
            parse_success=True,
            answer_prob=probabilities.get(answer),
            raw_answer_entropy=0.61,
            answer_entropy=0.88,
            answer_class_logits={value: float(-index) for index, value in enumerate(answer_classes)},
            answer_class_probabilities=probabilities,
            answer_metric_status="completed",
            candidate_count=len(answer_classes),
            elapsed_seconds=0.01,
            error=None,
        )


class FakeConfidenceAnalyzer:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def analyze_prompt(self, prompt: str, image_path: str | None = None) -> FakeConfidenceResult:
        self.calls.append({"prompt": prompt, "image": image_path})
        return FakeConfidenceResult()


def _runner(
    variants: list[str] | None = None,
    conditions: list[str] | None = None,
    results: dict[str, list[dict[str, Any]]] | None = None,
    checkpoint_writer: Any = write_result_pair,
    inference: FakeInference | None = None,
) -> tuple[EvaluationRunner, FakeInference, FakeConfidenceAnalyzer]:
    cases, metadata = load_evaluation_cases(
        _dataset({"text": QUESTION}),
        item_limit=1,
        prior_limit=1,
        fallback_null_path=TEST_ROOT / "assets" / "null.png",
    )
    selected_variants = variants or ["v1", "v2", "v3", "v4"]
    all_results = results or {version: [] for version in ("v1", "v2", "v3", "v4")}
    fake_inference = inference or FakeInference()
    fake_confidence = FakeConfidenceAnalyzer()
    runner = EvaluationRunner(
        inference=fake_inference,
        confidence_analyzer=fake_confidence,
        confidence_class_text="Confidence classes:\n- Low\n- High",
        cases=cases,
        variants=selected_variants,
        conditions=conditions or list(CONDITIONS),
        results_by_version=all_results,
        output_dir=TEST_ROOT / "output",
        run_config={"null_image": metadata["null_image"]},
        logger=logging.getLogger("confidence_test.tests"),
        checkpoint_writer=checkpoint_writer,
    )
    return runner, fake_inference, fake_confidence


def test_prompt_utils_is_byte_for_byte_unchanged() -> None:
    prompt_path = Path(__file__).resolve().parents[1] / "prompt_utils.py"
    assert hashlib.sha256(prompt_path.read_bytes()).hexdigest() == PROMPT_SHA256


@pytest.mark.parametrize("question", [QUESTION, {"text": QUESTION}])
def test_question_string_and_dictionary_are_supported(question: Any) -> None:
    cases, _ = load_evaluation_cases(
        _dataset(question),
        prior_limit=1,
        fallback_null_path=TEST_ROOT / "assets" / "null.png",
    )
    assert cases[0].question == QUESTION
    assert cases[0].record_key == "1::0"


def test_dataset_has_six_conditions_and_uses_existing_null_and_irr() -> None:
    cases, metadata = load_evaluation_cases(
        _dataset(),
        prior_limit=1,
        fallback_null_path=TEST_ROOT / "assets" / "unused.png",
    )
    assert tuple(cases[0].conditions) == CONDITIONS
    assert cases[0].conditions["irr"].relative_image_path == "images/1_irr.png"
    assert cases[0].conditions["null"].relative_image_path == "images/null.png"
    assert metadata["null_image"]["shared"] is True
    assert not (TEST_ROOT / "assets" / "unused.png").exists()


def test_fallback_null_is_created_once() -> None:
    path = _dataset()
    data = json.loads(path.read_text())
    del data[0]["items"][0]["image_clue"]["null"]
    path.write_text(json.dumps(data), encoding="utf-8")
    fallback = TEST_ROOT / "assets" / "null.png"
    load_evaluation_cases(path, fallback_null_path=fallback)
    first_mtime = fallback.stat().st_mtime_ns
    load_evaluation_cases(path, fallback_null_path=fallback)
    assert fallback.stat().st_mtime_ns == first_mtime


def test_answer_class_parsing_and_exact_answer_format() -> None:
    assert parse_answer_classes(QUESTION) == ["red", "blue"]
    assert parse_answer_output("**Answer**: RED") == ("RED", "red", True)
    with pytest.raises(ValueError, match="empty"):
        parse_answer_classes("Choose from: red, , blue.")
    with pytest.raises(ValueError, match="duplicate"):
        parse_answer_classes("Choose from: red, red.")


class TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [2] if text.startswith(" ") else [1]


def test_answer_probability_and_normalized_entropy_use_candidate_distribution() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0, 100.0])
    metric = compute_answer_metrics(logits, ["red"], "red", TinyTokenizer())
    assert metric.answer_prob == pytest.approx(1.0)
    assert metric.raw_answer_entropy == pytest.approx(0.0)
    assert metric.answer_entropy == pytest.approx(0.0)
    assert set(metric.answer_class_probabilities) == {"red"}


def test_cli_aliases_and_validation() -> None:
    assert canonical_variants(["v1_visible_previous_confidence,v4"]) == ["v1", "v4"]
    assert canonical_conditions(["all"]) == list(CONDITIONS)
    with pytest.raises(ValueError):
        canonical_variants(["all", "v1"])


def test_all_variants_make_38_calls_and_share_text_stages() -> None:
    FakeInference.instances = 0
    runner, inference, confidence = _runner()
    counts = runner.run()
    assert FakeInference.instances == 1
    assert counts == {
        "shared_stage1": 1,
        "shared_stage2": 1,
        "v1_stage3": 6,
        "v2_stage3": 6,
        "v3_stage3": 6,
        "v3_stage4": 6,
        "v4_stage1": 6,
        "v4_stage2": 6,
        "total": 38,
    }
    shared = runner.results_by_version["v1"][0]["priors"][0]["text_stage"]
    assert shared == runner.results_by_version["v2"][0]["priors"][0]["text_stage"]
    assert shared == runner.results_by_version["v3"][0]["priors"][0]["text_stage"]
    assert runner.results_by_version["v4"][0]["priors"][0]["text_stage"] is None
    assert len(inference.calls) == 13
    assert len(confidence.calls) == 25


def test_v1_v2_metrics_come_from_shared_answer_and_v2_hides_confidence() -> None:
    runner, _, confidence = _runner()
    runner.run()
    for version in ("v1", "v2"):
        prior = runner.results_by_version[version][0]["priors"][0]
        shared_answer = prior["text_stage"]["answer_result"]
        assert all(
            condition["answer_result"] == shared_answer
            for condition in prior["conditions"].values()
        )
    v2_prompts = [
        call["prompt"] for call in confidence.calls if "independently assess your confidence" in call["prompt"]
    ]
    assert len(v2_prompts) == 6
    assert all("Previous Confidence" not in prompt for prompt in v2_prompts)


def test_v3_and_v4_reuse_the_same_image_within_their_stage_pairs() -> None:
    runner, inference, confidence = _runner()
    runner.run()
    v3_answer_images = [call["image"] for call in inference.calls[1:7]]
    v3_confidence_images = [call["image"] for call in confidence.calls[13:19]]
    v4_answer_images = [call["image"] for call in inference.calls[7:13]]
    v4_confidence_images = [call["image"] for call in confidence.calls[19:25]]
    assert v3_answer_images == v3_confidence_images
    assert v4_answer_images == v4_confidence_images


def test_v4_only_does_not_run_shared_stages() -> None:
    runner, _, _ = _runner(variants=["v4"])
    counts = runner.run()
    assert counts["shared_stage1"] == counts["shared_stage2"] == 0
    assert counts["v4_stage1"] == counts["v4_stage2"] == 6
    simplified = full_to_simplified(runner.results_by_version["v4"], "v4")
    assert simplified[0]["priors"][0]["text_answer"] is None
    assert simplified[0]["priors"][0]["text_conf"] is None


def test_missing_image_only_fails_that_condition() -> None:
    cases, metadata = load_evaluation_cases(
        _dataset(missing="irr"),
        prior_limit=1,
        fallback_null_path=TEST_ROOT / "assets" / "null.png",
    )
    runner, inference, confidence = _runner(variants=["v1"])
    runner.cases = cases
    runner.run_config = {"null_image": metadata["null_image"]}
    runner.run()
    prior = runner.results_by_version["v1"][0]["priors"][0]
    assert prior["conditions"]["irr"]["status"] == "failed"
    assert prior["conditions"]["irr"]["answer_result"] is not None
    assert prior["conditions"]["consistent_easy"]["status"] == "completed"
    simplified = full_to_simplified(runner.results_by_version["v1"], "v1")
    assert simplified[0]["priors"][0]["conditions"]["irr"] == [None, None, None, None]
    assert len(inference.calls) == 1
    assert len(confidence.calls) == 6


def test_checkpoint_occurs_only_after_the_whole_prior_case() -> None:
    checkpoints: list[tuple[str, int]] = []
    holder: dict[str, EvaluationRunner] = {}

    def writer(output: Path, version: str, results: list[dict[str, Any]]) -> tuple[Path, Path]:
        checkpoints.append((version, holder["runner"].call_counts["total"]))
        return write_result_pair(output, version, results)

    runner, _, _ = _runner(checkpoint_writer=writer)
    holder["runner"] = runner
    runner.run()
    assert checkpoints == [("v1", 38), ("v2", 38), ("v3", 38), ("v4", 38)]


def test_interrupted_case_has_no_checkpoint() -> None:
    checkpoints: list[str] = []

    def writer(_output: Path, version: str, _results: list[dict[str, Any]]) -> tuple[Path, Path]:
        checkpoints.append(version)
        return Path("full"), Path("simple")

    runner, _, _ = _runner(
        checkpoint_writer=writer,
        inference=FakeInference(interrupt_at=2),
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    assert checkpoints == []
    assert all(not runner.results_by_version[version] for version in runner.variants)


def test_resume_skips_completed_stages_and_is_stable() -> None:
    runner, _, _ = _runner()
    runner.run()
    first_results = deepcopy_results = {
        version: json.loads(json.dumps(values)) for version, values in runner.results_by_version.items()
    }
    resumed, _, _ = _runner(results=deepcopy_results)
    counts = resumed.run()
    assert counts["total"] == 0
    for version in resumed.variants:
        assert full_to_simplified(resumed.results_by_version[version], version) == full_to_simplified(
            first_results[version], version
        )


def test_cross_version_shared_stage_conflict_is_rejected() -> None:
    runner, _, _ = _runner(variants=["v1", "v2"])
    runner.run()
    results = runner.results_by_version
    results["v2"][0]["priors"][0]["text_stage"]["confidence_result"]["soft_confidence"] = 0.1
    with pytest.raises(ValueError, match="Conflicting"):
        validate_shared_checkpoint_consistency(results)


def test_four_full_and_simplified_outputs_are_independent_and_valid() -> None:
    runner, _, _ = _runner()
    runner.run()
    for version in ("v1", "v2", "v3", "v4"):
        full_path = TEST_ROOT / "output" / f"{version}_results.json"
        simple_path = TEST_ROOT / "output" / f"{version}_simplified.json"
        assert full_path.is_file() and simple_path.is_file()
        full = load_json(full_path)
        simple = load_json(simple_path)
        assert isinstance(full, list) and isinstance(simple, list)
        assert len(full) == len(simple) == 1
        assert len(full[0]["priors"]) == 1
        assert tuple(simple[0]["priors"][0]["conditions"]) == CONDITIONS
        text = simple_path.read_text(encoding="utf-8")
        for condition in CONDITIONS:
            line = next(line for line in text.splitlines() if f'"{condition}"' in line)
            assert line.count("[") == line.count("]") == 1


def test_compact_writer_round_trip_and_failed_condition() -> None:
    data = [
        {
            "id": "1",
            "ground_truth_answer": "red",
            "priors": [
                {
                    "prior_index": 0,
                    "prior_bin": "0.4-0.6",
                    "text_answer": "red",
                    "text_conf": 0.5,
                    "conditions": {condition: [None, None, None, None] for condition in CONDITIONS},
                }
            ],
        }
    ]
    path = TEST_ROOT / "compact.json"
    write_compact_simplified_json(path, data)
    assert json.loads(path.read_text()) == data
    assert '"conflict_hard": [null, null, null, null]' in path.read_text()
