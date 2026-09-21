from __future__ import annotations

from dp_SA.prompts import ANSWER_PREFILL, PHASE0_TEMPLATE, phase0_prompt


ATTRIBUTION_TEMPLATE = """State what you believe your previous answer was mainly based on.

Source attribution refers to the relative contribution of the text clue and the image to the formation of your previous answer. Report whether your previous answer was based mainly on the text clue, more on the text clue, on both sources to a similar extent, more on the image, or mainly on the image.

Do not change, repeat, or regenerate your previous answer.

Choose 0 or 4 only when the answer was based almost entirely on the corresponding source. Choose 2 only when the two sources contributed to a similar extent. Otherwise choose 1 or 3 according to which source contributed more.

The labels form an ordered scale. A higher number indicates stronger image contribution and weaker text-clue contribution.

Source attribution labels:
0: The answer was based mainly on the text clue, with little or no meaningful contribution from the image.
1: The answer was based more on the text clue than on the image, while the image also contributed.
2: The answer was based on the text clue and the image to a similar extent.
3: The answer was based more on the image than on the text clue, while the text clue also contributed.
4: The answer was based mainly on the image, with little or no meaningful contribution from the text clue.

Output exactly one of the following:
**Source Attribution**:<$Class>

Do not add a space after the colon.
Do not output reasoning, explanation, confidence, or any additional text."""

SA_PREFILL = "**Source Attribution**:"
ATTRIBUTION_INSTRUCTION_START = "State what you believe your previous answer was mainly based on."
CLE_ANCHOR = "4: The answer was based mainly on the image, with little or no meaningful contribution from the text clue."
LABELS = ("0", "1", "2", "3", "4")
LABEL_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def attribution_prompt() -> str:
    return ATTRIBUTION_TEMPLATE


__all__ = [
    "ANSWER_PREFILL", "PHASE0_TEMPLATE", "phase0_prompt", "ATTRIBUTION_TEMPLATE",
    "ATTRIBUTION_INSTRUCTION_START", "CLE_ANCHOR", "LABELS", "LABEL_WEIGHTS", "SA_PREFILL",
    "attribution_prompt",
]
