"""Dataset traversal, normalization, and six-condition image resolution."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from confidence_test.answer_metrics import normalize_answer, parse_answer_classes


CONDITIONS = (
    "null",
    "irr",
    "consistent_easy",
    "consistent_hard",
    "conflict_easy",
    "conflict_hard",
)


@dataclass(frozen=True)
class ConditionInput:
    name: str
    relative_image_path: str | None
    resolved_image_path: str | None
    error: dict[str, str] | None


@dataclass(frozen=True)
class EvaluationCase:
    item_id: str
    item_order: int
    ground_truth_answer: str | None
    text_answer: str | None
    conflict_answer: str | None
    question: str
    answer_classes: list[str]
    answer_class_error: dict[str, str] | None
    prior_index: int
    prior_bin: str | None
    text_clue: str
    record_key: str
    conditions: dict[str, ConditionInput]


def question_text(item: dict[str, Any]) -> str:
    question = item.get("question")
    if isinstance(question, dict):
        question = question.get("text")
    text = str(question or "").strip()
    if not text:
        raise ValueError(f"Item {item.get('id')!r} has no question text")
    return text


def iter_dataset_items(payload: Any) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    if not isinstance(payload, list):
        raise ValueError("Dataset root must be an array")
    for group in payload:
        if not isinstance(group, dict):
            continue
        items = group.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                yield group, item


def _raw_condition_path(
    group: dict[str, Any],
    item: dict[str, Any],
    condition: str,
) -> str | None:
    image_clue = item.get("image_clue")
    if not isinstance(image_clue, dict):
        image_clue = {}
    if condition == "null":
        raw = image_clue.get("null", group.get("null_image"))
    elif condition == "irr":
        raw = image_clue.get("irr", image_clue.get("irrelevant"))
    else:
        branch_name, difficulty = condition.split("_", 1)
        branch = image_clue.get(branch_name)
        raw = branch.get(difficulty) if isinstance(branch, dict) else None
    if isinstance(raw, dict):
        raw = raw.get("image")
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _resolve_condition(
    name: str,
    raw_path: str | None,
    dataset_path: Path,
) -> ConditionInput:
    if raw_path is None:
        return ConditionInput(
            name=name,
            relative_image_path=None,
            resolved_image_path=None,
            error={"type": "MissingImageField", "message": f"No image path for {name}"},
        )
    path = Path(raw_path)
    resolved = path.resolve() if path.is_absolute() else (dataset_path.parent / path).resolve()
    error = None
    if not resolved.is_file():
        error = {
            "type": "FileNotFoundError",
            "message": f"Image does not exist: {resolved}",
        }
    return ConditionInput(
        name=name,
        relative_image_path=raw_path,
        resolved_image_path=str(resolved),
        error=error,
    )


def _atomic_create_white_image(path: Path, size: tuple[int, int] = (1024, 1024)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(fd)
    try:
        Image.new("RGB", size, (255, 255, 255)).save(temporary, format="PNG")
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_evaluation_cases(
    dataset: str | Path,
    item_limit: int | None = None,
    prior_limit: int | None = None,
    fallback_null_path: str | Path | None = None,
) -> tuple[list[EvaluationCase], dict[str, Any]]:
    dataset_path = Path(dataset).resolve()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    records = list(iter_dataset_items(payload))
    if item_limit is not None:
        records = records[:item_limit]

    fallback = Path(fallback_null_path).resolve() if fallback_null_path else None
    needs_fallback = any(_raw_condition_path(group, item, "null") is None for group, item in records)
    if needs_fallback:
        if fallback is None:
            raise ValueError("Dataset has no null image and no fallback path was provided")
        _atomic_create_white_image(fallback)

    cases: list[EvaluationCase] = []
    null_paths: list[Path] = []
    null_sources: set[str] = set()
    for item_order, (group, item) in enumerate(records):
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            raise ValueError("Dataset item has no id")
        question = question_text(item)
        try:
            answer_classes = parse_answer_classes(question)
            class_error = None
        except Exception as exc:
            answer_classes = []
            class_error = {"type": type(exc).__name__, "message": str(exc)}
        priors = item.get("selected_text_priors")
        if not isinstance(priors, list):
            raise ValueError(f"Item {item_id!r} has no selected_text_priors array")
        selected_priors = priors[:prior_limit] if prior_limit is not None else priors
        condition_map: dict[str, ConditionInput] = {}
        for condition in CONDITIONS:
            raw_path = _raw_condition_path(group, item, condition)
            if condition == "null" and raw_path is None and fallback is not None:
                raw_path = str(fallback)
                null_sources.add("generated_fallback")
            elif condition == "null":
                null_sources.add("dataset")
            resolved = _resolve_condition(condition, raw_path, dataset_path)
            condition_map[condition] = resolved
            if condition == "null" and resolved.resolved_image_path:
                null_paths.append(Path(resolved.resolved_image_path))
        for prior_index, prior in enumerate(selected_priors):
            if not isinstance(prior, dict):
                raise ValueError(f"Item {item_id!r} prior {prior_index} is not an object")
            clue = prior.get("clue")
            if not isinstance(clue, str) or not clue.strip():
                raise ValueError(f"Item {item_id!r} prior {prior_index} has no clue")
            cases.append(
                EvaluationCase(
                    item_id=item_id,
                    item_order=item_order,
                    ground_truth_answer=normalize_answer(item.get("answer")),
                    text_answer=normalize_answer(item.get("text_ans")),
                    conflict_answer=normalize_answer(item.get("conflict_ans")),
                    question=question,
                    answer_classes=answer_classes,
                    answer_class_error=class_error,
                    prior_index=prior_index,
                    prior_bin=str(prior.get("confidence_bin")).strip()
                    if prior.get("confidence_bin") is not None
                    else None,
                    text_clue=clue.strip(),
                    record_key=f"{item_id}::{prior_index}",
                    conditions=dict(condition_map),
                )
            )

    unique_nulls = list(dict.fromkeys(path.resolve() for path in null_paths))
    null_metadata: dict[str, Any] = {
        "sources": sorted(null_sources),
        "paths": [str(path) for path in unique_nulls],
        "shared": len(unique_nulls) == 1,
        "size": None,
        "mode": None,
    }
    if len(unique_nulls) == 1 and unique_nulls[0].is_file():
        with Image.open(unique_nulls[0]) as image:
            null_metadata["size"] = list(image.size)
            null_metadata["mode"] = image.mode
    return cases, {
        "dataset_path": str(dataset_path),
        "selected_item_count": len(records),
        "case_count": len(cases),
        "null_image": null_metadata,
    }
