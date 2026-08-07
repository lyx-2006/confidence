"""Prompt and class definitions for joint source-attribution experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .source_attribution_prompt_utils import (
    V3_STAGE3_REANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT,
    V4_STAGE1_FULL_EVIDENCE_ANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT,
)
from .source_attribution_schema import (
    SOURCE_ATTRIBUTION_CLASSES,
    SOURCE_ATTRIBUTION_CLASS_TEXT,
    SOURCE_ATTRIBUTION_MIDPOINTS,
)


V3_ANSWER_BASIS_9_PROMPT = """You previously answered this question using limited information.
You now have both the text clue and the image.

Question:
{question}

Text clue:
{text_clue}

**Previous Answer**: {previous_answer}
**Previous Confidence**: {previous_confidence}

Answer the question again using all currently available evidence.

Then report what you believe your answer was mainly based on.

Source attribution asks what you believe your answer was based on: the text clue, the image, or both. Report which source you relied on more when selecting your answer.

The previous answer is not fixed. You may keep it or replace it.

{source_classes}

Output exactly:

**Answer**: <your new answer>
**Source Attribution**:<CLASS>

Do not include reasoning, confidence, or any additional text."""


V4_ANSWER_BASIS_9_PROMPT = """Question:
{question}

Text clue:
{text_clue}

Answer the question using the text clue and the image.

Then report what you believe your answer was mainly based on.

Source attribution asks what you believe your answer was based on: the text clue, the image, or both. Report which source you relied on more when selecting your answer.

{source_classes}

Output exactly:

**Answer**: <your answer>
**Source Attribution**:<CLASS>

Do not include reasoning, confidence, or any additional text."""


ANSWER_BASIS_9_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

ANSWER_BASIS_9_MIDPOINTS = [
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

ANSWER_BASIS_9_CLASS_TEXT = """Source attribution classes:
0: The answer was based almost entirely on the text clue.
1: The answer was based mainly on the text clue.
2: The answer was based more on the text clue than on the image.
3: The answer was based slightly more on the text clue.
4: The answer was based on the text clue and the image to a similar extent.
5: The answer was based slightly more on the image.
6: The answer was based more on the image than on the text clue.
7: The answer was based mainly on the image.
8: The answer was based almost entirely on the image."""


V3_ANSWER_BASIS_10_PROMPT = """You previously answered this question using limited information.
You now have both the text clue and the image.

Question:
{question}

Text clue:
{text_clue}

**Previous Answer**: {previous_answer}
**Previous Confidence**: {previous_confidence}

Answer the question again using all currently available evidence.

Then report what you believe your answer was mainly based on.

Source attribution asks what you believe your answer was based on: the text clue or the image. Report which source you relied on more when selecting your answer.

The previous answer is not fixed. You may keep it or replace it.

{source_classes}


Output exactly:

**Answer**: <your new answer>
**Source Attribution**:<CLASS>

Do not include reasoning, confidence, or any additional text."""


V4_ANSWER_BASIS_10_PROMPT = """Question:
{question}

Text clue:
{text_clue}

Answer the question using the text clue and the image.

Then report what you believe your answer was mainly based on.

Source attribution asks what you believe your answer was based on: the text clue or the image. Report which source you relied on more when selecting your answer.

{source_classes}


Output exactly:

**Answer**: <your answer>
**Source Attribution**:<CLASS>

Do not include reasoning, confidence, or any additional text."""


ANSWER_BASIS_10_CLASSES = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9",
]

ANSWER_BASIS_10_MIDPOINTS = [
    0.05,
    0.15,
    0.25,
    0.35,
    0.45,
    0.55,
    0.65,
    0.75,
    0.85,
    0.95,
]

ANSWER_BASIS_10_CLASS_TEXT = """Source attribution classes:
0: The answer was clearly based more on the text clue.
1: The answer was based more on the text clue.
2: The answer was moderately based more on the text clue.
3: The answer was somewhat based more on the text clue.
4: The answer was only slightly based more on the text clue.
5: The answer was only slightly based more on the image.
6: The answer was somewhat based more on the image.
7: The answer was moderately based more on the image.
8: The answer was based more on the image.
9: The answer was clearly based more on the image."""


@dataclass(frozen=True)
class SourcePromptVariant:
    name: str
    v3_joint_prompt: str
    v4_joint_prompt: str
    classes: tuple[str, ...]
    midpoints: tuple[float, ...]
    class_text: str

    def __post_init__(self) -> None:
        if not self.classes or len(self.classes) != len(set(self.classes)):
            raise ValueError(f"{self.name}: source classes must be non-empty and distinct")
        if len(self.classes) != len(self.midpoints):
            raise ValueError(
                f"{self.name}: source classes and midpoints must have equal length"
            )


SOURCE_PROMPT_VARIANT_ORDER = (
    "baseline",
    "answer_basis_9",
    "answer_basis_10",
)

SOURCE_PROMPT_VARIANTS = {
    "baseline": SourcePromptVariant(
        name="baseline",
        v3_joint_prompt=V3_STAGE3_REANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT,
        v4_joint_prompt=V4_STAGE1_FULL_EVIDENCE_ANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT,
        classes=tuple(SOURCE_ATTRIBUTION_CLASSES),
        midpoints=tuple(SOURCE_ATTRIBUTION_MIDPOINTS),
        class_text=SOURCE_ATTRIBUTION_CLASS_TEXT,
    ),
    "answer_basis_9": SourcePromptVariant(
        name="answer_basis_9",
        v3_joint_prompt=V3_ANSWER_BASIS_9_PROMPT,
        v4_joint_prompt=V4_ANSWER_BASIS_9_PROMPT,
        classes=tuple(ANSWER_BASIS_9_CLASSES),
        midpoints=tuple(ANSWER_BASIS_9_MIDPOINTS),
        class_text=ANSWER_BASIS_9_CLASS_TEXT,
    ),
    "answer_basis_10": SourcePromptVariant(
        name="answer_basis_10",
        v3_joint_prompt=V3_ANSWER_BASIS_10_PROMPT,
        v4_joint_prompt=V4_ANSWER_BASIS_10_PROMPT,
        classes=tuple(ANSWER_BASIS_10_CLASSES),
        midpoints=tuple(ANSWER_BASIS_10_MIDPOINTS),
        class_text=ANSWER_BASIS_10_CLASS_TEXT,
    ),
}


def get_source_prompt_variant(name: str) -> SourcePromptVariant:
    try:
        return SOURCE_PROMPT_VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown source prompt variant: {name}") from exc
