"""Restricted candidate-answer parsing, probabilities, and entropy."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from layer_metacognition.direct_readout import (
    _restricted_logits,
    build_first_token_collision_report,
)
from layer_metacognition.metrics import entropy_from_probabilities


CHOOSE_FROM_PATTERN = re.compile(
    r"Choose\s+from\s*:\s*(.+?)(?:\.(?:\s|$)|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
ANSWER_PREFIX_PATTERN = re.compile(
    r"^\s*\*\*Answer\*\*\s*:\s*([^\r\n]+)",
    flags=re.IGNORECASE,
)


def normalize_answer(value: Any) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized or None


def parse_answer_classes(question: str) -> list[str]:
    """Parse and validate the literal ordered candidate set in a question."""
    match = CHOOSE_FROM_PATTERN.search(question)
    if match is None:
        raise ValueError("Question has no 'Choose from:' candidate set")
    raw_parts = match.group(1).split(",")
    if any(not part.strip() for part in raw_parts):
        raise ValueError("Question contains an empty answer candidate")
    candidates = [normalize_answer(part) for part in raw_parts]
    if not candidates or any(candidate is None for candidate in candidates):
        raise ValueError("Question contains no usable answer candidates")
    values = [str(candidate) for candidate in candidates]
    if len(values) != len(set(values)):
        raise ValueError("Question contains duplicate answer candidates")
    return values


def parse_answer_output(raw_output: str) -> tuple[str | None, str | None, bool]:
    match = ANSWER_PREFIX_PATTERN.match(raw_output)
    if match is None:
        return None, None, False
    answer = match.group(1).strip()
    normalized = normalize_answer(answer)
    return (answer or None), normalized, bool(answer and normalized)


@dataclass(frozen=True)
class AnswerMetricResult:
    answer_prob: float | None
    raw_answer_entropy: float | None
    answer_entropy: float | None
    answer_class_logits: dict[str, float]
    answer_class_probabilities: dict[str, float]
    answer_metric_status: str
    candidate_count: int
    error: dict[str, str] | None = None
    # V2 aliases.  ``answer_entropy`` remains the historical normalized
    # 0--1 value used by the confidence experiments; the generation pipeline
    # consumes the explicit 0--100 score below.
    entropy_score: float | None = None
    raw_entropy: float | None = None
    restricted_top1: str | None = None


def failed_answer_metrics(
    candidate_count: int,
    error_type: str,
    message: str,
) -> AnswerMetricResult:
    return AnswerMetricResult(
        answer_prob=None,
        raw_answer_entropy=None,
        answer_entropy=None,
        answer_class_logits={},
        answer_class_probabilities={},
        answer_metric_status="failed",
        candidate_count=candidate_count,
        entropy_score=None,
        raw_entropy=None,
        restricted_top1=None,
        error={"type": error_type, "message": message},
    )


def compute_answer_metrics(
    first_token_logits: torch.Tensor,
    answer_classes: Sequence[str],
    normalized_answer: str | None,
    tokenizer: Any,
) -> AnswerMetricResult:
    """Softmax only over candidate-class first-token scores."""
    candidates = [normalize_answer(value) for value in answer_classes]
    if not candidates or any(value is None for value in candidates):
        return failed_answer_metrics(len(candidates), "CandidateSetError", "No usable candidates")
    labels = [str(value) for value in candidates]
    if len(labels) != len(set(labels)):
        return failed_answer_metrics(len(labels), "CandidateSetError", "Duplicate candidates")
    try:
        collision_report = build_first_token_collision_report(tokenizer, labels)
        vocab_logits = first_token_logits.detach().float().cpu()
        class_logits = _restricted_logits(vocab_logits, labels, collision_report)
        probabilities = torch.softmax(class_logits, dim=-1)
        raw_entropy = entropy_from_probabilities(probabilities)
        candidate_count = len(labels)
        normalized_entropy = 0.0 if candidate_count == 1 else raw_entropy / math.log(candidate_count)
        normalized_entropy = min(1.0, max(0.0, float(normalized_entropy)))
        logit_map = {
            label: float(class_logits[index].item()) for index, label in enumerate(labels)
        }
        probability_map = {
            label: float(probabilities[index].item()) for index, label in enumerate(labels)
        }
        restricted_top1 = labels[int(torch.argmax(class_logits).item())]
        answer_probability = probability_map.get(normalized_answer or "")
        status = "completed" if answer_probability is not None else "failed"
        error = None
        if answer_probability is None:
            error = {
                "type": "AnswerOutsideCandidateSet",
                "message": f"Generated answer is not a legal candidate: {normalized_answer!r}",
            }
        return AnswerMetricResult(
            answer_prob=answer_probability,
            raw_answer_entropy=float(raw_entropy),
            answer_entropy=normalized_entropy,
            answer_class_logits=logit_map,
            answer_class_probabilities=probability_map,
            answer_metric_status=status,
            candidate_count=candidate_count,
            error=error,
            entropy_score=float(normalized_entropy * 100.0),
            raw_entropy=float(raw_entropy),
            restricted_top1=restricted_top1,
        )
    except Exception as exc:
        return failed_answer_metrics(len(labels), type(exc).__name__, str(exc))
