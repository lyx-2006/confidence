"""Dataset expansion and stable experiment-case identifiers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    item_id: str
    prior_index: int
    question: str
    text_clue: str
    image_path: str
    dataset_answer: str | None
    text_target: str | None
    image_target: str | None
    image_condition: str
    difficulty: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().casefold()
    return text or None


def parse_choice_colors(question: str) -> list[str]:
    match = re.search(r"Choose\s+from\s*:\s*(.+?)(?:\.|$)", question, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Question has no 'Choose from:' colour set: {question!r}")
    colors = [part.strip().casefold() for part in match.group(1).split(",") if part.strip()]
    if not colors:
        raise ValueError(f"Question has an empty colour set: {question!r}")
    if len(colors) != len(set(colors)):
        raise ValueError(f"Question has duplicate colour candidates: {question!r}")
    return colors


def parse_stage1_answer(raw_output: str, candidates: list[str]) -> str | None:
    """Return the first legal candidate mentioned in a concise model output."""
    cleaned = re.sub(r"^\s*\*\*answer\*\*\s*:\s*", "", raw_output, flags=re.IGNORECASE)
    matches: list[tuple[int, int, str]] = []
    for candidate in candidates:
        match = re.search(rf"(?<![\w]){re.escape(candidate)}(?![\w])", cleaned, flags=re.IGNORECASE)
        if match:
            matches.append((match.start(), -len(candidate), candidate))
    return min(matches)[2] if matches else None


def _question_text(item: dict[str, Any]) -> str:
    question = item.get("question")
    if isinstance(question, dict):
        question = question.get("text")
    text = str(question or "").strip()
    if not text:
        raise ValueError(f"Item {item.get('id')!r} has no question text")
    return text


def _resolve_image_path(raw_path: str, dataset_path: Path, image_dir: Path | None) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        candidates = [path]
    else:
        roots = [image_dir] if image_dir is not None else [dataset_path.parent]
        candidates = [root / path for root in roots if root is not None]
    existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if not existing:
        raise FileNotFoundError(f"Image path cannot be resolved: {raw_path!r}; tried {candidates}")
    if len(existing) != 1:
        raise ValueError(f"Image path is ambiguous: {raw_path!r}; matches {sorted(map(str, existing))}")
    return next(iter(existing))


def _image_variants(item: dict[str, Any]) -> Iterable[tuple[str, str | None, str, str | None]]:
    image_clue = item.get("image_clue")
    if not isinstance(image_clue, dict):
        raise ValueError(f"Item {item.get('id')!r} has no image_clue mapping")
    answer = normalize_label(item.get("answer") or item.get("text_ans"))
    conflict = normalize_label(item.get("conflict_ans"))
    specifications = [
        ("consistent", "easy", answer),
        ("consistent", "hard", answer),
        ("conflict", "easy", conflict),
        ("conflict", "hard", conflict),
    ]
    for condition, difficulty, target in specifications:
        branch = image_clue.get(condition)
        raw_path = branch.get(difficulty) if isinstance(branch, dict) else None
        if not isinstance(raw_path, str):
            raise ValueError(f"Item {item.get('id')!r} lacks {condition}/{difficulty} image")
        yield condition, difficulty, raw_path, target
    irrelevant = image_clue.get("irrelevant")
    raw_irrelevant = irrelevant.get("image") if isinstance(irrelevant, dict) else irrelevant
    if not isinstance(raw_irrelevant, str):
        raise ValueError(f"Item {item.get('id')!r} lacks irrelevant image")
    yield "irrelevant", None, raw_irrelevant, None


def load_experiment_cases(
    dataset: str | Path,
    image_dir: str | Path | None = None,
    max_items: int | None = None,
    case_id: str | None = None,
) -> tuple[list[ExperimentCase], dict[str, Any]]:
    dataset_path = Path(dataset).resolve()
    image_root = Path(image_dir).resolve() if image_dir is not None else None
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Dataset root must be an array")
    items = [item for group in payload if isinstance(group, dict) for item in group.get("items", [])]
    if max_items is not None:
        if max_items < 1:
            raise ValueError("--max-items must be positive")
        items = items[:max_items]

    cases: list[ExperimentCase] = []
    for item in items:
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            raise ValueError("Dataset item is missing id")
        question = _question_text(item)
        dataset_answer = normalize_label(item.get("answer"))
        text_target = normalize_label(item.get("text_ans"))
        priors = item.get("selected_text_priors")
        if not isinstance(priors, list):
            raise ValueError(f"Item {item_id!r} has no selected_text_priors")
        variants = list(_image_variants(item))
        for prior_index, prior in enumerate(priors):
            clue = prior.get("clue") if isinstance(prior, dict) else None
            if not isinstance(clue, str) or not clue.strip():
                raise ValueError(f"Item {item_id!r} prior {prior_index} has no clue")
            for condition, difficulty, raw_image, image_target in variants:
                suffix = condition if difficulty is None else f"{condition}_{difficulty}"
                stable_id = f"{item_id}__prior_{prior_index}__{suffix}"
                if case_id is not None and stable_id != case_id:
                    continue
                image_path = _resolve_image_path(raw_image, dataset_path, image_root)
                cases.append(
                    ExperimentCase(
                        case_id=stable_id,
                        item_id=item_id,
                        prior_index=prior_index,
                        question=question,
                        text_clue=clue.strip(),
                        image_path=str(image_path),
                        dataset_answer=dataset_answer,
                        text_target=text_target,
                        image_target=image_target,
                        image_condition=condition,
                        difficulty=difficulty,
                    )
                )
    if case_id is not None and not cases:
        raise ValueError(f"Unknown case_id within selected items: {case_id}")
    metadata = {
        "dataset_path": str(dataset_path),
        "image_dir": str(image_root) if image_root else str(dataset_path.parent),
        "selected_item_count": len(items),
        "case_count": len(cases),
        "unavailable_fields": {
            "irrelevant.image_target": "not reliably defined by the dataset",
            "irrelevant.difficulty": "not present in the dataset",
        },
    }
    return cases, metadata
