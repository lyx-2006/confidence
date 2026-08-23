from __future__ import annotations

PHASE0_TEMPLATE = """Question:
{question}

Text clue:
{text_clue}

Answer the question using the text clue and the image.

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, source attribution, or any additional text.

Output exactly:

**Answer**: <your answer>"""

PHASE1_TEMPLATE = """You will be shown a question, together with a text clue, an image, and an answer you previously provided before.

Question:
{question}

Text clue:
{text_clue}

**Answer**: {answer}

State what you believe the fixed answer above was mainly based on.

Source attribution refers to the relative contribution of the text clue and the image to the formation of the fixed answer. Report whether the fixed answer was based more on the text clue, more on the image, or on both sources to a similar extent.

The answer shown above is fixed. Do not change, repeat, or regenerate the answer.

State your source attribution as exactly one integer between 0 and 8 using the classes below.

A higher class indicates stronger image contribution. A lower class indicates stronger text contribution.

Do not choose class 4 merely because both sources were shown. Choose class 4 only if you believe the text clue and the image contributed to the fixed answer to a similar extent.

Source attribution classes:
0: The answer was based almost entirely on the text clue.
1: The answer was based mainly on the text clue.
2: The answer was based more on the text clue than on the image.
3: The answer was based slightly more on the text clue.
4: The answer was based on the text clue and the image to a similar extent.
5: The answer was based slightly more on the image.
6: The answer was based more on the image than on the text clue.
7: The answer was based mainly on the image.
8: The answer was based almost entirely on the image.

At the very end of your output, format your source attribution as:
**Source Attribution**:$CLASS
where CLASS is exactly one integer between 0 and 8.
Do NOT leave a space before the class digit or you will be scored WRONG.
Do not output the answer again.
Do not output reasoning, explanation, confidence, punctuation, or any additional text."""

ANSWER_PREFILL = "**Answer**:"
SA_PREFILL = "**Source Attribution**:"
SA_INSTRUCTION_START = "State what you believe the fixed answer above was mainly based on."
FORBIDDEN_BEFORE_PANL = (
    "source attribution", "mainly based", "text contribution", "image contribution",
    "class 0", "class 1", "class 2", "class 3", "class 4", "class 5",
    "class 6", "class 7", "class 8",
)

def phase0_prompt(question: str, text_clue: str) -> str:
    return PHASE0_TEMPLATE.format(question=question, text_clue=text_clue)

def phase1_prompt(question: str, text_clue: str, answer: str) -> str:
    if not answer or "\n" in answer or "\r" in answer:
        raise ValueError("Phase 1 fixed answer must be non-empty and single-line")
    return PHASE1_TEMPLATE.format(question=question, text_clue=text_clue, answer=answer)
