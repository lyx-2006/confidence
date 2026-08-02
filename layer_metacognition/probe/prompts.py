"""Probe-only prompts."""

IMAGE_ONLY_ANSWER_PROMPT = """Question:
{question}

Answer the question using only the question and the image.

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, hedging, or any additional text.

Output exactly:

**Answer**: <your answer>

Do not include any additional text."""
