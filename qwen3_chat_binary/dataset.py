from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .config import DATASET_PATH


COLORS = (
    "red", "orange", "yellow", "green", "blue", "cyan",
    "purple", "pink", "brown", "white", "black", "gray",
)
PAIR_TYPES = ("hard_text_easy_image", "hard_image_easy_text", "balanced")
REQUIRED_FIELDS = {
    "text", "image", "shape", "pair_type", "text_entropy", "image_entropy",
    "text_answer", "image_answer",
}
QUESTION_TEMPLATE = "What is the color of the {shape}? Choose from: " + ", ".join(COLORS) + "."


@dataclass(frozen=True)
class ConflictCase:
    case_id: str
    source_index: int
    question: str
    text_clue: str
    image_path: Path
    image_reference: str
    shape: str
    pair_type: str
    text_entropy: float
    image_entropy: float
    text_answer: str
    image_answer: str


def _required_text(row: dict[str, object], field: str, index: int) -> str:
    value = row[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Case {index} field {field!r} must be a non-empty string")
    return value.strip()


def _entropy(row: dict[str, object], field: str, index: int) -> float:
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Case {index} field {field!r} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"Case {index} field {field!r} must be finite and in [0, 1]")
    return result


def load_conflict_cases(
    path: Path = DATASET_PATH,
    *,
    max_samples: int | None = None,
) -> list[ConflictCase]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Conflict dataset must be a non-empty JSON array: {path}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive or None")

    cases: list[ConflictCase] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict) or set(raw) != REQUIRED_FIELDS:
            keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
            raise ValueError(f"Case {index} has invalid schema: {keys}")
        text = _required_text(raw, "text", index)
        image_reference = _required_text(raw, "image", index)
        shape = _required_text(raw, "shape", index)
        pair_type = _required_text(raw, "pair_type", index)
        text_answer = _required_text(raw, "text_answer", index)
        image_answer = _required_text(raw, "image_answer", index)
        if pair_type not in PAIR_TYPES:
            raise ValueError(f"Case {index} has unknown pair_type: {pair_type}")
        if text_answer not in COLORS or image_answer not in COLORS:
            raise ValueError(f"Case {index} has an answer outside the configured color labels")
        if text_answer == image_answer:
            raise ValueError(f"Case {index} is not a conflict case")
        image_path = (path.parent / image_reference).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Case {index} image does not exist: {image_path}")
        pair = (text, image_reference)
        if pair in seen_pairs:
            raise ValueError(f"Case {index} duplicates a text/image pair")
        seen_pairs.add(pair)
        cases.append(ConflictCase(
            case_id=f"case_{index:04d}",
            source_index=index,
            question=QUESTION_TEMPLATE.format(shape=shape),
            text_clue=text,
            image_path=image_path,
            image_reference=image_reference,
            shape=shape,
            pair_type=pair_type,
            text_entropy=_entropy(raw, "text_entropy", index),
            image_entropy=_entropy(raw, "image_entropy", index),
            text_answer=text_answer,
            image_answer=image_answer,
        ))
    return cases if max_samples is None else cases[:max_samples]


__all__ = [
    "COLORS", "PAIR_TYPES", "QUESTION_TEMPLATE", "ConflictCase", "load_conflict_cases",
]
