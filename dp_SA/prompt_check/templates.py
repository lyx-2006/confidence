from __future__ import annotations

from dataclasses import dataclass

from dp_SA.prompts import PHASE1_TEMPLATE, SA_PREFILL

T1_TEMPLATE = """You will be shown a question, together with a text clue, an image, and an answer you previously provided before.

Question:
{question}

Text clue:
{text_clue}

**Answer**: {answer}

State what you believe the fixed answer above was mainly based on.

Source attribution refers to the relative contribution of the text clue and the image to the formation of the fixed answer. Report whether the fixed answer was based more on the text clue, more on the image, or on both sources to a similar extent.

State your source attribution as exactly one integer between 0 and 8 using the classes below.

A higher class indicates stronger image contribution. A lower class indicates stronger text contribution.

Do not choose class 4 merely because both sources were shown. Choose class 4 only if you believe the text clue and the image contributed to the fixed answer to a similar extent.

Source attribution classes:
0: The text clue contributed almost exclusively; the image contributed almost nothing.
1: The text clue contributed much more than the image.
2: The text clue contributed clearly more than the image.
3: The text clue contributed slightly more than the image.
4: The text clue and the image contributed to a similar extent.
5: The image contributed slightly more than the text clue.
6: The image contributed clearly more than the text clue.
7: The image contributed much more than the text clue.
8: The image contributed almost exclusively; the text clue contributed almost nothing.

At the very end of your output, format your source attribution as:
**Source Attribution**:$CLASS
where CLASS is exactly one integer between 0 and 8.
Do NOT leave a space before the class digit or you will be scored WRONG.
Do not output the answer again.
Do not output reasoning, explanation, confidence, punctuation, or any additional text."""

T2_TEMPLATE = """You will be shown a question, together with a text clue, an image, and an answer you previously provided before.

Question:
{question}

Text clue:
{text_clue}

**Answer**: {answer}

State what you believe the fixed answer above was mainly based on.

Source attribution refers to the relative contribution of the text clue and the image to the formation of the fixed answer. Report whether the fixed answer was based more on the text clue, more on the image, or on both sources to a similar extent.

The answer shown above is fixed. Do not change, repeat, or regenerate the answer.

State your source attribution as exactly one integer between 0 and 8 using the classes below.

A lower class indicates stronger image contribution. A higher class indicates stronger text contribution.

Do not choose class 4 merely because both sources were shown. Choose class 4 only if you believe the text clue and the image contributed to the fixed answer to a similar extent.

Source attribution classes:
0: The answer was based almost entirely on the image.
1: The answer was based mainly on the image.
2: The answer was based more on the image than on the text clue.
3: The answer was based slightly more on the image.
4: The answer was based on the text clue and the image to a similar extent.
5: The answer was based slightly more on the text clue.
6: The answer was based more on the text clue than on the image.
7: The answer was based mainly on the text clue.
8: The answer was based almost entirely on the text clue.

At the very end of your output, format your source attribution as:
**Source Attribution**:$CLASS
where CLASS is exactly one integer between 0 and 8.
Do NOT leave a space before the class digit or you will be scored WRONG.
Do not output the answer again.
Do not output reasoning, explanation, confidence, punctuation, or any additional text."""

T3_TEMPLATE = """You will be shown a question, together with a text clue, an image, and an answer you previously provided before.

Question:
{question}

Text clue:
{text_clue}

**Answer**: {answer}

State what you believe the fixed answer above was mainly based on.

Source attribution refers to the relative contribution of the text clue and the image to the formation of the fixed answer. Report whether the fixed answer was based more on the text clue, more on the image, or on both sources to a similar extent.

The answer shown above is fixed. Do not change, repeat, or regenerate the answer.

Select exactly one of the following five source-attribution labels:

STRONG_TEXT: The answer depended predominantly on the text clue.
SLIGHT_TEXT: The answer depended somewhat more on the text clue than on the image.
BALANCED: The text clue and the image contributed to a similar extent.
SLIGHT_IMAGE: The answer depended somewhat more on the image than on the text clue.
STRONG_IMAGE: The answer depended predominantly on the image.

At the very end of your output, write:
**Source Attribution**:$LABEL
where LABEL is exactly one of:
STRONG_TEXT
SLIGHT_TEXT
BALANCED
SLIGHT_IMAGE
STRONG_IMAGE

Do not place a space before the label. Do not output the answer, reasoning, explanation, confidence, punctuation, or any other text."""


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    template: str
    last_class_description: str
    post_answer_anchor: str
    kind: str

    def render(self, *, question: str, text_clue: str, answer: str) -> str:
        if not answer or "\n" in answer or "\r" in answer:
            raise ValueError("Fixed answer must be non-empty and single-line")
        return self.template.format(question=question, text_clue=text_clue, answer=answer)


TEMPLATES = {
    "T0": TemplateSpec("T0", PHASE1_TEMPLATE, "8: The answer was based almost entirely on the image.", "State what you believe the fixed answer above was mainly based on.", "numeric"),
    "T1": TemplateSpec("T1", T1_TEMPLATE, "8: The image contributed almost exclusively; the text clue contributed almost nothing.", "State what you believe the fixed answer above was mainly based on.", "numeric"),
    "T2": TemplateSpec("T2", T2_TEMPLATE, "8: The answer was based almost entirely on the text clue.", "State what you believe the fixed answer above was mainly based on.", "numeric_reversed"),
    "T3": TemplateSpec("T3", T3_TEMPLATE, "STRONG_IMAGE: The answer depended predominantly on the image.", "State what you believe the fixed answer above was mainly based on.", "labels"),
}

assert TEMPLATES["T0"].template is PHASE1_TEMPLATE
assert SA_PREFILL == "**Source Attribution**:"
