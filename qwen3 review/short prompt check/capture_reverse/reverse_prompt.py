from __future__ import annotations


PHASE1_TEMPLATE_SHORT_REVERSE = """You will be shown a question, together with a text clue, an image, and an answer you previously provided before.

Question:
{question}

Text clue:
{text_clue}

**Answer**: {answer}

Source attribution refers to the relative contribution of the text clue and the image to the formation of the fixed answer. Report whether the fixed answer was based more on the text clue, more on the image, or on both sources to a similar extent.

State your source attribution as exactly one integer between 0 and 8 using the classes below.

A higher class indicates stronger text contribution. A lower class indicates stronger image contribution.

Source attribution classes:
0,1,2,3,4,5,6,7,8

At the very end of your output, format your source attribution as:
**Source Attribution**:$CLASS
where CLASS is exactly one integer between 0 and 8.
Do NOT leave a space before the class digit or you will be scored WRONG.
Do not output the answer again.
Do not output reasoning, explanation, confidence, punctuation, or any additional text."""


def phase1_prompt_short_reverse(question: str, text_clue: str, answer: str) -> str:
    if not answer or "\n" in answer or "\r" in answer:
        raise ValueError("Reverse short Phase 1 fixed answer must be non-empty and single-line")
    return PHASE1_TEMPLATE_SHORT_REVERSE.format(
        question=question,
        text_clue=text_clue,
        answer=answer,
    )

