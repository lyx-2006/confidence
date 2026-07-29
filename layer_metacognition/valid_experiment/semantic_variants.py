"""Canonical Semantic Patchscope prompt variants and stable result ordering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SemanticVariant:
    variant_id: str
    group: str
    instruction: str
    class_rows: dict[int, str]
    class_order: tuple[int, ...]
    image_midpoints: tuple[float, ...]


BASE_INSTRUCTION = (
    "Classify the relative contribution of the text clue and the image into "
    "exactly one source attribution class."
)

CANONICAL_CLASS_ROWS = {
    0: "Strongly text dominant — Image 0.000-0.100, Text 0.900-1.000",
    1: "Moderately text dominant — Image 0.100-0.250, Text 0.750-0.900",
    2: "Slightly text dominant — Image 0.250-0.400, Text 0.600-0.750",
    3: "Weakly text dominant — Image 0.400-0.475, Text 0.525-0.600",
    4: "Balanced contribution — Image 0.475-0.525, Text 0.475-0.525",
    5: "Weakly image dominant — Image 0.525-0.600, Text 0.400-0.475",
    6: "Slightly image dominant — Image 0.600-0.750, Text 0.250-0.400",
    7: "Moderately image dominant — Image 0.750-0.900, Text 0.100-0.250",
    8: "Strongly image dominant — Image 0.900-1.000, Text 0.000-0.100",
}

CANONICAL_CLASS_ORDER = (0, 1, 2, 3, 4, 5, 6, 7, 8)

CANONICAL_IMAGE_MIDPOINTS = (
    0.05,
    0.175,
    0.325,
    0.4375,
    0.5,
    0.5625,
    0.675,
    0.825,
    0.95,
)

SYNONYM_INSTRUCTIONS = {
    "synonym_reliance": (
        "Assess the relative reliance on the text clue and the image and assign "
        "exactly one source attribution class."
    ),
    "synonym_evidential_weight": (
        "Determine the relative evidential weight of the text clue and the image "
        "and select exactly one source attribution class."
    ),
    "synonym_support_balance": (
        "Select the source attribution class that best represents the balance of "
        "support provided by the text clue and the image."
    ),
    "synonym_evidence_distribution": (
        "Classify how the supporting evidence is distributed between the text "
        "clue and the image into exactly one source attribution class."
    ),
    "synonym_dominant_support": (
        "Identify whether the relevant support is more text-based, more "
        "image-based, or balanced, and select exactly one source attribution class."
    ),
}

REVERSED_CLASS_ROWS = {
    0: "Strongly image dominant — Image 0.900-1.000, Text 0.000-0.100",
    1: "Moderately image dominant — Image 0.750-0.900, Text 0.100-0.250",
    2: "Slightly image dominant — Image 0.600-0.750, Text 0.250-0.400",
    3: "Weakly image dominant — Image 0.525-0.600, Text 0.400-0.475",
    4: "Balanced contribution — Image 0.475-0.525, Text 0.475-0.525",
    5: "Weakly text dominant — Image 0.400-0.475, Text 0.525-0.600",
    6: "Slightly text dominant — Image 0.250-0.400, Text 0.600-0.750",
    7: "Moderately text dominant — Image 0.100-0.250, Text 0.750-0.900",
    8: "Strongly text dominant — Image 0.000-0.100, Text 0.900-1.000",
}

REVERSED_IMAGE_MIDPOINTS = (
    0.95,
    0.825,
    0.675,
    0.5625,
    0.5,
    0.4375,
    0.325,
    0.175,
    0.05,
)

CLASS_ORDERS = {
    "order_shuffle_1": (4, 8, 1, 6, 0, 3, 7, 2, 5),
    "order_shuffle_2": (7, 3, 0, 5, 2, 8, 4, 1, 6),
    "order_shuffle_3": (2, 6, 4, 0, 8, 5, 1, 7, 3),
}

SEMANTIC_PROMPT_TEMPLATE = """{instruction}

{source_classes}

Output exactly:

**Source Attribution**:<CLASS>

CLASS must exactly match one of the source attribution class numbers listed above.
Do not output reasoning, explanation, an answer, confidence, or any additional text."""


def format_source_classes(variant: SemanticVariant) -> str:
    """Render class rows in prompt order without changing numeric logit order."""
    return "\n".join(
        ["Source attribution classes:"]
        + [
            f"{class_index}: {variant.class_rows[class_index]}"
            for class_index in variant.class_order
        ]
    )


def build_semantic_prompt(variant: SemanticVariant) -> str:
    return SEMANTIC_PROMPT_TEMPLATE.format(
        instruction=variant.instruction,
        source_classes=format_source_classes(variant),
    )


def _build_variants() -> tuple[SemanticVariant, ...]:
    values = [
        SemanticVariant(
            variant_id="base",
            group="base",
            instruction=BASE_INSTRUCTION,
            class_rows=dict(CANONICAL_CLASS_ROWS),
            class_order=CANONICAL_CLASS_ORDER,
            image_midpoints=CANONICAL_IMAGE_MIDPOINTS,
        )
    ]
    values.extend(
        SemanticVariant(
            variant_id=variant_id,
            group="synonym",
            instruction=instruction,
            class_rows=dict(CANONICAL_CLASS_ROWS),
            class_order=CANONICAL_CLASS_ORDER,
            image_midpoints=CANONICAL_IMAGE_MIDPOINTS,
        )
        for variant_id, instruction in SYNONYM_INSTRUCTIONS.items()
    )
    values.append(
        SemanticVariant(
            variant_id="reverse_direction",
            group="reverse",
            instruction=BASE_INSTRUCTION,
            class_rows=dict(REVERSED_CLASS_ROWS),
            class_order=CANONICAL_CLASS_ORDER,
            image_midpoints=REVERSED_IMAGE_MIDPOINTS,
        )
    )
    values.extend(
        SemanticVariant(
            variant_id=variant_id,
            group="order",
            instruction=BASE_INSTRUCTION,
            class_rows=dict(CANONICAL_CLASS_ROWS),
            class_order=class_order,
            image_midpoints=CANONICAL_IMAGE_MIDPOINTS,
        )
        for variant_id, class_order in CLASS_ORDERS.items()
    )
    return tuple(values)


SEMANTIC_VARIANTS = _build_variants()
SEMANTIC_VARIANT_IDS = tuple(value.variant_id for value in SEMANTIC_VARIANTS)
SEMANTIC_VARIANT_BY_ID = {
    value.variant_id: value for value in SEMANTIC_VARIANTS
}

RESULT_COLUMNS = [
    "answer",
    "answer_probability",
    *SEMANTIC_VARIANT_IDS,
]


def select_semantic_variants(values: Iterable[str]) -> tuple[SemanticVariant, ...]:
    """Validate a CLI selection and return canonical experiment order."""
    raw = [
        part.strip()
        for value in values
        for part in str(value).split(",")
        if part.strip()
    ]
    if not raw:
        raise ValueError("--semantic-variants requires at least one value")
    if "all" in raw:
        if raw != ["all"]:
            raise ValueError("'all' cannot be combined with variant IDs")
        return SEMANTIC_VARIANTS
    duplicates = sorted({value for value in raw if raw.count(value) > 1})
    if duplicates:
        raise ValueError(
            "--semantic-variants contains duplicate value(s): "
            + ", ".join(duplicates)
        )
    invalid = [value for value in raw if value not in SEMANTIC_VARIANT_BY_ID]
    if invalid:
        raise ValueError(f"Unknown semantic variant(s): {', '.join(invalid)}")
    if "base" not in raw:
        raise ValueError("--semantic-variants must include base")
    selected = set(raw)
    return tuple(
        variant for variant in SEMANTIC_VARIANTS if variant.variant_id in selected
    )


def variant_to_dict(variant: SemanticVariant) -> dict[str, object]:
    return {
        "variant_id": variant.variant_id,
        "group": variant.group,
        "instruction": variant.instruction,
        "prompt": build_semantic_prompt(variant),
        "class_rows": {
            str(index): variant.class_rows[index]
            for index in CANONICAL_CLASS_ORDER
        },
        "class_order": list(variant.class_order),
        "image_midpoints": list(variant.image_midpoints),
    }


for _variant in SEMANTIC_VARIANTS:
    if set(_variant.class_rows) != set(CANONICAL_CLASS_ORDER):
        raise RuntimeError(f"{_variant.variant_id} does not define classes 0-8")
    if tuple(sorted(_variant.class_order)) != CANONICAL_CLASS_ORDER:
        raise RuntimeError(f"{_variant.variant_id} class_order is not a permutation")
    if len(_variant.image_midpoints) != len(CANONICAL_CLASS_ORDER):
        raise RuntimeError(f"{_variant.variant_id} must define nine midpoints")
