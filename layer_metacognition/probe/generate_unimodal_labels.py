#!/usr/bin/env python3
"""Extract/generate model-behavior text-only and selected image-only labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from confidence_test.answer_metrics import normalize_answer
from confidence_test.dataset_utils import EvaluationCase, load_evaluation_cases
from confidence_test.prompt_utils import STAGE1_TEXT_ANSWER_PROMPT
from layer_metacognition.hidden_state_store import append_jsonl

from . import (
    DEFAULT_PROBE_CONDITIONS,
    PROBE_CONDITIONS,
    normalize_ordered_choices,
)
from .common import (
    atomic_write_jsonl,
    atomic_write_keyed_jsonl,
    iter_jsonl,
    load_optional_jsonl,
    probe_output_dir,
    sortable_item_id,
)
from .prompts import IMAGE_ONLY_ANSWER_PROMPT


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "datasets" / "datasets.json"
)


def _result_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Unsupported inference result: {type(value)!r}")


def _text_key(case: EvaluationCase) -> tuple[str, int]:
    return str(case.item_id), int(case.prior_index)


def _image_key(case: EvaluationCase, condition: str) -> tuple[str, str]:
    return str(case.item_id), condition


def _case_maps(
    cases: list[EvaluationCase],
    probe_conditions: tuple[str, ...] = DEFAULT_PROBE_CONDITIONS,
) -> tuple[
    dict[tuple[str, int], EvaluationCase],
    dict[tuple[str, str], tuple[EvaluationCase, str]],
]:
    text_cases: dict[tuple[str, int], EvaluationCase] = {}
    image_cases: dict[tuple[str, str], tuple[EvaluationCase, str]] = {}
    for case in cases:
        key = _text_key(case)
        previous = text_cases.get(key)
        if previous is not None and (
            previous.question != case.question
            or previous.text_clue != case.text_clue
            or previous.answer_classes != case.answer_classes
        ):
            raise ValueError(f"Inconsistent dataset definition for text label key {key}")
        text_cases[key] = case
        for condition in probe_conditions:
            if condition not in case.conditions:
                raise ValueError(
                    f"Dataset item {case.item_id!r} has no condition {condition!r}"
                )
            image_key = _image_key(case, condition)
            candidate = case.conditions[condition]
            previous_image = image_cases.get(image_key)
            if previous_image is not None:
                old_case, old_condition = previous_image
                old_input = old_case.conditions[old_condition]
                if (
                    old_case.question != case.question
                    or old_case.answer_classes != case.answer_classes
                    or old_input.resolved_image_path != candidate.resolved_image_path
                ):
                    raise ValueError(
                        f"Inconsistent dataset definition for image label key {image_key}"
                    )
            image_cases[image_key] = (case, condition)
    return text_cases, image_cases


def _valid_existing_initial(
    record: dict[str, Any],
    answer_classes: list[str],
) -> tuple[str, str] | None:
    generated = record.get("generated")
    if not isinstance(generated, dict):
        return None
    raw = generated.get("initial_answer")
    detail = generated.get("initial_answer_result")
    if not isinstance(raw, str) or not raw.strip() or not isinstance(detail, dict):
        return None
    if not detail.get("parse_success") or detail.get("answer_metric_status") != "completed":
        return None
    normalized = normalize_answer(raw)
    if normalized is None or normalized not in answer_classes:
        return None
    return raw.strip(), normalized


def extract_existing_text_labels(
    results_path: str | Path,
    text_cases: dict[tuple[str, int], EvaluationCase],
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate valid V3 initial answers and fail on any normalized conflict."""

    values: dict[tuple[str, int], dict[str, set[str]]] = defaultdict(
        lambda: {"raw": set(), "normalized": set()}
    )
    for record in iter_jsonl(results_path):
        if record.get("version") != "v3":
            continue
        key = (str(record.get("item_id")), int(record.get("prior_index")))
        case = text_cases.get(key)
        if case is None:
            raise ValueError(f"Results contain unknown dataset text label key {key}")
        valid = _valid_existing_initial(record, case.answer_classes)
        if valid is None:
            continue
        raw, normalized = valid
        values[key]["raw"].add(raw)
        values[key]["normalized"].add(normalized)

    conflicts: list[dict[str, Any]] = []
    labels: dict[tuple[str, int], dict[str, Any]] = {}
    for key, grouped in values.items():
        if len(grouped["normalized"]) != 1:
            conflicts.append(
                {
                    "label_type": "text_only",
                    "item_id": key[0],
                    "prior_index": key[1],
                    "error": {
                        "type": "TextOnlyLabelConflict",
                        "message": "V3 initial answers disagree after normalization",
                    },
                    "raw_answers": sorted(grouped["raw"]),
                    "normalized_answers": sorted(grouped["normalized"]),
                }
            )
            continue
        case = text_cases[key]
        normalized = next(iter(grouped["normalized"]))
        labels[key] = {
            "item_id": key[0],
            "prior_index": key[1],
            "question": case.question,
            "text_clue": case.text_clue,
            "answer_classes": list(case.answer_classes),
            "text_only_answer": normalized,
            "text_only_answer_raw": sorted(grouped["raw"])[0],
            "parse_success": True,
            "label_source": "existing_v3_initial_answer",
            "error": None,
        }
    return labels, conflicts


def _load_keyed_labels(
    path: Path,
    kind: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in load_optional_jsonl(path):
        if kind == "text":
            key = (str(record["item_id"]), int(record["prior_index"]))
        else:
            key = (str(record["item_id"]), str(record["condition"]))
        if key in records:
            raise ValueError(f"Duplicate {kind} label key in {path}: {key}")
        records[key] = record
    return records


def _answer_is_valid(result: dict[str, Any], answer_classes: list[str]) -> tuple[bool, str | None]:
    normalized = normalize_answer(result.get("normalized_answer") or result.get("answer"))
    valid = bool(
        result.get("parse_success")
        and result.get("answer_metric_status") == "completed"
        and normalized in answer_classes
    )
    return valid, normalized if valid else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--inference-path")
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--probe-conditions",
        nargs="+",
        choices=list(PROBE_CONDITIONS),
        default=list(DEFAULT_PROBE_CONDITIONS),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    probe_conditions = normalize_ordered_choices(
        args.probe_conditions,
        PROBE_CONDITIONS,
        "--probe-conditions",
    )
    experiment_dir = Path(args.experiment_dir).resolve()
    dataset = Path(args.dataset).resolve()
    model_path = Path(args.model_path).resolve()
    results_path = experiment_dir / "results.jsonl"
    output_dir = probe_output_dir(experiment_dir, args.output_dir)
    text_path = output_dir / "text_only_labels.jsonl"
    image_path = output_dir / "image_only_labels.jsonl"
    failure_path = output_dir / "label_failures.jsonl"
    if args.max_answer_tokens < 1:
        raise ValueError("--max-answer-tokens must be positive")
    for path, label in ((results_path, "results"), (dataset, "dataset")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not args.resume and (text_path.exists() or image_path.exists() or failure_path.exists()):
        raise FileExistsError(
            f"Probe labels already exist under {output_dir}; pass --resume to continue"
        )

    cases, _metadata = load_evaluation_cases(
        dataset,
        fallback_null_path=output_dir / ".runtime" / "null.png",
    )
    text_cases, image_cases = _case_maps(cases, probe_conditions)
    extracted, conflicts = extract_existing_text_labels(results_path, text_cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not failure_path.exists():
        atomic_write_jsonl(failure_path, [])
    if conflicts:
        for conflict in conflicts:
            append_jsonl(failure_path, conflict)
        raise ValueError(
            f"Found {len(conflicts)} conflicting V3 text-only label key(s); "
            f"details written to {failure_path}"
        )

    text_labels = _load_keyed_labels(text_path, "text") if args.resume else {}
    for key, record in extracted.items():
        previous = text_labels.get(key)
        if (
            previous is not None
            and previous.get("parse_success")
            and normalize_answer(previous.get("text_only_answer"))
            != record["text_only_answer"]
        ):
            raise ValueError(f"Saved text label conflicts with existing V3 output for {key}")
        text_labels[key] = record
    atomic_write_keyed_jsonl(
        text_path,
        text_labels,
        sort_key=lambda key: (sortable_item_id(key[0]), key[1]),
    )

    image_labels = _load_keyed_labels(image_path, "image") if args.resume else {}
    missing_text = [
        key
        for key in text_cases
        if not text_labels.get(key, {}).get("parse_success")
    ]
    pending_images = [
        key
        for key in image_cases
        if not image_labels.get(key, {}).get("parse_success")
    ]
    inference = None
    if missing_text or pending_images:
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {model_path}")
        from confidence_test.inference_extension import build_extended_inference_class
        from confidence_test.runtime_imports import DEFAULT_INFERENCE_PATH, load_runtime

        inference_source = Path(args.inference_path or DEFAULT_INFERENCE_PATH).resolve()
        runtime = load_runtime(inference_source)
        inference_class = build_extended_inference_class(runtime.QwenVLInference)
        inference = inference_class(model_path=str(model_path))

    for key in sorted(missing_text, key=lambda item: (sortable_item_id(item[0]), item[1])):
        case = text_cases[key]
        prompt = STAGE1_TEXT_ANSWER_PROMPT.format(
            question=case.question,
            text_clue=case.text_clue,
        )
        assert inference is not None
        result = _result_dict(
            inference.generate_answer_with_metrics(
                prompt=prompt,
                answer_classes=case.answer_classes,
                image_path=None,
                max_new_tokens=args.max_answer_tokens,
            )
        )
        valid, normalized = _answer_is_valid(result, case.answer_classes)
        record = {
            "item_id": key[0],
            "prior_index": key[1],
            "question": case.question,
            "text_clue": case.text_clue,
            "answer_classes": list(case.answer_classes),
            "text_only_answer": normalized,
            "text_only_answer_raw": result.get("answer") if valid else None,
            "parse_success": valid,
            "label_source": "generated_missing_label",
            "error": result.get("error") if not valid else None,
            "generation_result": result,
        }
        text_labels[key] = record
        atomic_write_keyed_jsonl(
            text_path,
            text_labels,
            sort_key=lambda item: (sortable_item_id(item[0]), item[1]),
        )
        if not valid:
            append_jsonl(
                failure_path,
                {
                    "label_type": "text_only",
                    "item_id": key[0],
                    "prior_index": key[1],
                    "error": record["error"],
                    "generation_result": result,
                },
            )

    condition_order = {
        condition: index for index, condition in enumerate(PROBE_CONDITIONS)
    }
    for key in sorted(
        pending_images,
        key=lambda item: (sortable_item_id(item[0]), condition_order[item[1]]),
    ):
        case, condition = image_cases[key]
        condition_input = case.conditions[condition]
        result: dict[str, Any]
        if condition_input.error or not condition_input.resolved_image_path:
            result = {
                "answer": None,
                "normalized_answer": None,
                "parse_success": False,
                "answer_metric_status": "failed",
                "error": condition_input.error
                or {"type": "MissingImagePath", "message": f"No image for {key}"},
            }
        else:
            assert inference is not None
            result = _result_dict(
                inference.generate_answer_with_metrics(
                    prompt=IMAGE_ONLY_ANSWER_PROMPT.format(question=case.question),
                    answer_classes=case.answer_classes,
                    image_path=condition_input.resolved_image_path,
                    max_new_tokens=args.max_answer_tokens,
                )
            )
        valid, normalized = _answer_is_valid(result, case.answer_classes)
        record = {
            "item_id": key[0],
            "condition": condition,
            "question": case.question,
            "image_path": condition_input.resolved_image_path,
            "answer_classes": list(case.answer_classes),
            "image_only_answer": normalized,
            "image_only_answer_raw": result.get("answer") if valid else None,
            "parse_success": valid,
            "error": result.get("error") if not valid else None,
            "generation_result": result,
        }
        image_labels[key] = record
        atomic_write_keyed_jsonl(
            image_path,
            image_labels,
            sort_key=lambda item: (
                sortable_item_id(item[0]),
                condition_order[item[1]],
            ),
        )
        if not valid:
            append_jsonl(
                failure_path,
                {
                    "label_type": "image_only",
                    "item_id": key[0],
                    "condition": condition,
                    "error": record["error"],
                    "generation_result": result,
                },
            )

    print(
        json.dumps(
            {
                "status": "complete",
                "text_label_count": len(text_labels),
                "valid_text_label_count": sum(
                    bool(value.get("parse_success")) for value in text_labels.values()
                ),
                "image_label_count": len(image_labels),
                "valid_image_label_count": sum(
                    bool(value.get("parse_success")) for value in image_labels.values()
                ),
                "probe_conditions": list(probe_conditions),
                "selected_missing_or_failed_image_label_count": sum(
                    not bool(image_labels.get(key, {}).get("parse_success"))
                    for key in image_cases
                ),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
