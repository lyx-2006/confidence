"""Canonical source-attribution classes and restricted-logit math."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch


SOURCE_ATTRIBUTION_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

SOURCE_ATTRIBUTION_MIDPOINTS = [
    0.05,
    0.175,
    0.325,
    0.4375,
    0.5,
    0.5625,
    0.675,
    0.825,
    0.95,
]

SOURCE_ATTRIBUTION_CLASS_TEXT = """Source attribution classes:
0: Strongly text dominant — Image 0.000-0.100, Text 0.900-1.000
1: Moderately text dominant — Image 0.100-0.250, Text 0.750-0.900
2: Slightly text dominant — Image 0.250-0.400, Text 0.600-0.750
3: Weakly text dominant — Image 0.400-0.475, Text 0.525-0.600
4: Balanced contribution — Image 0.475-0.525, Text 0.475-0.525
5: Weakly image dominant — Image 0.525-0.600, Text 0.400-0.475
6: Slightly image dominant — Image 0.600-0.750, Text 0.250-0.400
7: Moderately image dominant — Image 0.750-0.900, Text 0.100-0.250
8: Strongly image dominant — Image 0.900-1.000, Text 0.000-0.100"""

ASSISTANT_SOURCE_ATTRIBUTION_PREFILL = "**Source Attribution**:"


@dataclass(frozen=True)
class SourceTokenSpecification:
    """Tokenizer diagnostics for the deliberately no-space SA wire format."""

    class_token_ids: dict[str, list[int]]
    raw_encodings: dict[str, list[int]]
    leading_space_encodings: dict[str, list[int]]
    shared_leading_token_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceAttributionResult:
    hard_label: str
    hard_index: int
    hard_image_score: float
    hard_text_score: float
    soft_image_score: float
    soft_text_score: float
    source_entropy: float
    normalized_source_entropy: float
    class_logits: list[float]
    class_probabilities: list[float]
    class_token_ids: dict[str, list[int]]
    raw_output: str
    hard_label_parsed: bool
    parsed_label: str | None = None
    token_diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(token_id) for token_id in encoded]


def build_source_token_specification(
    tokenizer: Any,
    classes: Sequence[str] = SOURCE_ATTRIBUTION_CLASSES,
) -> SourceTokenSpecification:
    """Build disjoint class-bearing IDs and inspect leading-space spellings.

    Qwen tokenizes ``" 6"`` as a structural whitespace token followed by the
    digit token.  The SA protocol intentionally emits ``":6"`` so the
    class-bearing next token is the raw digit.  Leading-space spellings are
    still checked and recorded, but their common structural token is not a
    class token.
    """

    raw_encodings: dict[str, list[int]] = {}
    spaced_encodings: dict[str, list[int]] = {}
    class_token_ids: dict[str, list[int]] = {}
    leading_tokens: list[int] = []
    labels = [str(label) for label in classes]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Source classes must be non-empty and distinct")
    for label in labels:
        raw = _encode(tokenizer, label)
        spaced = _encode(tokenizer, f" {label}")
        if not raw or not spaced:
            raise RuntimeError(f"Tokenizer produced no tokens for source class {label!r}")
        if len(raw) != 1:
            raise RuntimeError(
                f"No-space source class must be one token: label={label!r}, ids={raw}"
            )
        if spaced[-len(raw) :] != raw:
            raise RuntimeError(
                "Leading-space source spelling does not end in the raw class token: "
                f"label={label!r}, raw={raw}, leading_space={spaced}"
            )
        prefix = spaced[: -len(raw)]
        if not prefix:
            raise RuntimeError(
                f"Tokenizer did not expose a leading-space form for source class {label!r}"
            )
        raw_encodings[label] = raw
        spaced_encodings[label] = spaced
        class_token_ids[label] = list(dict.fromkeys(raw))
        leading_tokens.extend(prefix)

    for left_index, left in enumerate(labels):
        left_ids = set(class_token_ids[left])
        for right in labels[left_index + 1 :]:
            overlap = left_ids.intersection(class_token_ids[right])
            if overlap:
                raise RuntimeError(
                    f"Source class token collision: {left!r} vs {right!r}: {sorted(overlap)}"
                )

    return SourceTokenSpecification(
        class_token_ids=class_token_ids,
        raw_encodings=raw_encodings,
        leading_space_encodings=spaced_encodings,
        shared_leading_token_ids=sorted(set(leading_tokens)),
    )


def gather_source_class_logits(
    vocab_logits: torch.Tensor,
    class_token_ids: dict[str, Sequence[int]],
    classes: Sequence[str] = SOURCE_ATTRIBUTION_CLASSES,
) -> torch.Tensor:
    """Apply ConfidenceAnalyzer's strongest-variant aggregation."""

    values: list[torch.Tensor] = []
    for label in classes:
        ids = list(class_token_ids[label])
        if not ids:
            raise ValueError(f"Source class {label!r} has no token IDs")
        index = torch.tensor(ids, dtype=torch.long, device=vocab_logits.device)
        values.append(torch.max(vocab_logits.index_select(0, index)))
    return torch.stack(values).float()


def source_distribution(
    class_logits: torch.Tensor,
    *,
    class_token_ids: dict[str, Sequence[int]],
    raw_output: str,
    parsed_label: str | None,
    token_diagnostics: dict[str, Any] | None = None,
    classes: Sequence[str] = SOURCE_ATTRIBUTION_CLASSES,
    midpoints: Sequence[float] = SOURCE_ATTRIBUTION_MIDPOINTS,
) -> SourceAttributionResult:
    labels = [str(label) for label in classes]
    midpoint_values = [float(value) for value in midpoints]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Source classes must be non-empty and distinct")
    if len(labels) != len(midpoint_values):
        raise ValueError(
            "Source classes and midpoints must have the same length: "
            f"{len(labels)} != {len(midpoint_values)}"
        )
    logits = class_logits.detach().float().cpu()
    if logits.ndim != 1 or logits.numel() != len(labels):
        raise ValueError(
            f"Expected {len(labels)} source logits, got shape {tuple(logits.shape)}"
        )
    probabilities = torch.softmax(logits, dim=-1)
    hard_index = int(torch.argmax(probabilities).item())
    hard_label = labels[hard_index]
    hard_image = midpoint_values[hard_index]
    midpoint_tensor = torch.tensor(midpoint_values, dtype=torch.float32)
    soft_image = float(torch.sum(probabilities * midpoint_tensor).item())
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * torch.log(positive)).sum().item())
    normalized_entropy = float(entropy / math.log(len(labels)))
    hard_text = 1.0 - hard_image
    soft_text = 1.0 - soft_image
    for image_score, text_score in ((hard_image, hard_text), (soft_image, soft_text)):
        if not (0.0 <= image_score <= 1.0 and 0.0 <= text_score <= 1.0):
            raise RuntimeError("Source scores are outside [0, 1]")
        if abs(image_score + text_score - 1.0) >= 1e-6:
            raise RuntimeError("Image and text source scores do not sum to one")
    return SourceAttributionResult(
        hard_label=hard_label,
        hard_index=hard_index,
        hard_image_score=float(hard_image),
        hard_text_score=float(hard_text),
        soft_image_score=soft_image,
        soft_text_score=soft_text,
        source_entropy=entropy,
        normalized_source_entropy=normalized_entropy,
        class_logits=[float(value) for value in logits.tolist()],
        class_probabilities=[float(value) for value in probabilities.tolist()],
        class_token_ids={
            label: [int(token_id) for token_id in class_token_ids[label]]
            for label in labels
        },
        raw_output=raw_output,
        hard_label_parsed=parsed_label is not None,
        parsed_label=parsed_label,
        token_diagnostics=token_diagnostics,
    )
