from __future__ import annotations


PHASE0_TEXT_ONLY_TEMPLATE = """Question:
{question}

Text clue:
{text_clue}

Answer the question using the text clue.

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, source attribution, or any additional text.

Output exactly:

**Answer**: <your answer>"""

PHASE0_IMAGE_ONLY_TEMPLATE = """Question:
{question}

Answer the question using the image.

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, source attribution, or any additional text.

Output exactly:

**Answer**: <your answer>"""

ANSWER_PREFILL = "**Answer**:"


def text_only_prompt(question: str, text_clue: str) -> str:
    return PHASE0_TEXT_ONLY_TEMPLATE.format(question=question, text_clue=text_clue)


def image_only_prompt(question: str) -> str:
    return PHASE0_IMAGE_ONLY_TEMPLATE.format(question=question)

