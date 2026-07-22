"""Exact prompts used by the main layer-metacognition experiment."""

ASSISTANT_ANSWER_PREFILL = "**Answer**:"
ASSISTANT_CONFIDENCE_PREFILL = "**Confidence**:"

STAGE1_MULTIMODAL_ANSWER_PROMPT = """Question:
{question}

Text clue:
{text_clue}

Answer the question using the question, text clue, and image.

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, or any additional text.

Output exactly:

**Answer**: <your answer>

Do not include any additional text."""

STAGE2_CONFIDENCE_PROMPT = """You will be shown a question together with an answer you previously provided before.

Question:
{question}

Text clue:
{text_clue}

**Answer**: {answer}

Classify your confidence in the correctness of the previous answer into one of
the following classes.

Confidence refers to the probability that the previous answer is correct
given the question, text clue, and image.

The previous answer is fixed. Do not change the answer.

The fact that an answer was selected does not mean the confidence must be high.
The answer may be the most likely choice while still having low confidence
when multiple alternatives remain plausible.

{classes}

Output exactly:

**Confidence**: <CLASS>

CLASS must be exactly one of the confidence class names listed above.
Do not include reasoning, explanation, or any additional text."""
