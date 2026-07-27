"""Source-attribution prompts for the V3/V4 extension.

The only intentional deviation from the supplied prose is the confirmed Qwen
wire format: the class digit immediately follows the SA colon.
"""

V3_STAGE4_META_SOURCE_ATTRIBUTION_PROMPT = """You have already answered this question after considering all currently available evidence.

The same complete evidence, including the image, is available again in this stage.

Question:
{question}

Text clue:
{text_clue}

Your previous result based on limited information:
**Initial Answer**: {initial_answer}
**Initial Confidence**: {initial_confidence}

Your most recent answer based on the complete evidence:
**Current Answer**: {stage3_answer}

Your task is to reassess all currently available evidence and evaluate the relative contribution of the text clue and the image to the formation of the current answer.

Important requirements:
1. The current answer is fixed. You must not change, repeat, restate, or replace it.
2. The initial answer and initial confidence are historical information.
3. You must carefully evaluate all currently available evidence and your most recent answer.
4. The final source attribution must represent the relative contribution of the text clue and the image.
5. A higher class indicates stronger image contribution, while a lower class indicates stronger text contribution.
6. Do not default to the middle classes without carefully evaluating the evidence.
7. You must perform the assessment internally, but do not output reasoning, explanation, analysis, the answer, confidence values, or any additional text.
8. Choose exactly one source attribution class from the list below.

{source_classes}

Output exactly:

**Source Attribution**:<CLASS>

CLASS must exactly match one of the source attribution class numbers listed above."""


V4_STAGE2_FULL_EVIDENCE_SOURCE_ATTRIBUTION_PROMPT = """You will be shown a question together with an answer you previously provided before.

Question:
{question}

Text clue:
{text_clue}

**Answer**: {answer}

Classify the relative contribution of the text clue and the image to the formation of the previous answer into one of
the following classes.

Source attribution refers to the relative information contribution of the text clue and the image.

The previous answer is fixed. Do not change the answer.

A higher class indicates stronger image contribution, while a lower class indicates stronger text contribution.
The answer may rely mainly on either source or on both sources.

{source_classes}

Output exactly:

**Source Attribution**:<CLASS>

CLASS must be exactly one of the source attribution class numbers listed above.
Do not include reasoning, explanation, or any additional text."""


V3_STAGE3_REANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT = """You previously answered this question when the available information was limited.

You now have richer and more complete evidence than you had when answering it for the first time.

Question:
{question}

Text clue:
{text_clue}

Your previous result based only on limited information:
**Previous Answer**: {previous_answer}
**Previous Confidence**: {previous_confidence}

Answer the question again after considering all currently available evidence.
Then classify the relative contribution of the text clue and the image to the formation of your new answer.

Important requirements:
1. The previous answer is not fixed.
2. You may retain the previous answer or replace it with a different answer.
3. Do not mechanically retain either the previous answer or the previous confidence.
4. You must carefully integrate all the information provided in this stage.
5. A higher source attribution class indicates stronger image contribution, while a lower class indicates stronger text contribution.
6. Do not default to the middle classes without carefully evaluating the evidence.
7. Do not provide reasoning, explanation, analysis, hedging, confidence, or any additional text.
8. Choose exactly one source attribution class from the list below.

{source_classes}

Output exactly:

**Answer**: <your new answer>
**Source Attribution**:<CLASS>

CLASS must exactly match one of the source attribution class numbers listed above.
Do not include any additional text."""


V4_STAGE1_FULL_EVIDENCE_ANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT = """Question:
{question}

Text clue:
{text_clue}

Answer the question using the information.
Then classify the relative contribution of the text clue and the image to the formation of your answer.

Answer as concisely as possible.

A higher source attribution class indicates stronger image contribution, while a lower class indicates stronger text contribution.

Do not default to the middle classes without carefully evaluating the evidence.

{source_classes}

Do not provide reasoning, explanation, confidence, or any additional text.

Output exactly:

**Answer**: <your answer>
**Source Attribution**:<CLASS>

CLASS must exactly match one of the source attribution class numbers listed above.
Do not include any additional text."""

