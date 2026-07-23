"""English prompt definitions for the four evaluation variants."""

STAGE1_TEXT_ANSWER_PROMPT = """Question:
{question}

Text clue:
{text_clue}

Answer the question using only the question and text clue above.

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, hedging, or any additional text.

Output exactly:

**Answer**: <your answer>

Do not include any additional text."""

STAGE2_TEXT_CONFIDENCE_PROMPT = """You will be shown a question together with an answer you previously provided before.

Question:
{question}

Text clue:
{text_clue}

**Answer**: {answer}

Classify your confidence in the correctness of the previous answer into one of
the following classes.

Confidence refers to the probability that the previous answer is correct
given only the question and text clue.

The previous answer is fixed. Do not change the answer.

The fact that an answer was selected does not mean the confidence must be high.
The answer may be the most likely choice while still having low confidence
when multiple alternatives remain plausible.

{classes}

Output exactly:

**Confidence**: <CLASS>

CLASS must be exactly one of the confidence class names listed above.
Do not include reasoning, explanation, or any additional text."""

V1_STAGE3_IMAGE_CONFIDENCE_PROMPT = """You previously answered this question with limited information.
You now have richer and more complete evidence than you had when answering it for the first time.

Question:
{question}

Text clue:
{text_clue}

Your previous answer and confidence:
**Answer**: {answer}
**Previous Confidence**: {previous_confidence}

Your task is to reassess your confidence in this answer after considering all currently available evidence.

Important requirements:
1. The previous answer is fixed. You must not change, repeat, restate, or replace it.
2. The previous confidence was only an initial estimate made when the available evidence was limited.
3. The previous confidence represents only your belief state at that time. It is not a result that must be preserved in this reassessment.
4. You must not directly copy, mechanically repeat, or automatically retain the previous confidence.
5. You must reassess your confidence in the answer based on the richer evidence currently available.
6. During the reassessment, you should consider:
   - the question;
   - the fixed previous answer;
   - the initial belief represented by the previous confidence;
   - the more complete evidence currently available.
7. You must not select the same confidence class merely because that confidence was previously given.
8. Even if you ultimately select the same confidence class as before, it must result from a new assessment based on all currently available evidence, rather than from copying the previous result.
9. The final confidence must reflect your confidence in the previous answer after considering all currently available evidence.
10. Do not provide reasoning, explanations, analysis, hedging, or any additional text.
11. Choose exactly one confidence class from the list below.
12. Some of the information provided may have no relevance to the target shape's color. Under this circumstance, you should assess your confidence based on the relevant information only.

{classes}

Each confidence class represents the probability that the fixed previous answer is correct after considering all currently available evidence.

Output exactly in the following format:
**Confidence**: $CLASS

CLASS must exactly match one of the confidence class names listed above.
Do not include quotation marks, probability ranges, the previous answer, or any additional text."""

V2_STAGE3_IMAGE_CONFIDENCE_PROMPT = """You previously answered this question when the available information was limited.
You now have richer and more complete evidence than you had when answering it for the first time.

Question:
{question}

Text clue:
{text_clue}

Your previous answer:
**Answer**: {answer}

Your task is to independently assess your confidence in the previous answer after considering all currently available evidence.

Important requirements:
1. The previous answer is fixed. You must not change, repeat, restate, or replace it.
2. You must make an independent assessment based only on the currently available evidence.
3. You must carefully evaluate the information.
4. Do not select a middle or neutral confidence class merely to avoid making a clear judgment.
6. Do not ignore the currently available evidence and directly output a default confidence value.
7. The final confidence must represent your confidence that the fixed answer is correct after considering all currently available evidence.
8. You must perform the assessment internally, but do not output reasoning, explanation, analysis, hedging, the previous answer, or any additional text.
9. Choose exactly one confidence class from the list below.
10. Some of the information provided may have no relevance to the target shape's color. Under this circumstance, you should assess your confidence based on the relevant information only.

{classes}

Output exactly:

**Confidence**: <CLASS>

CLASS must exactly match one of the confidence class names listed above."""

V3_STAGE3_REANSWER_PROMPT = """You previously answered this question when the available information was limited.
You now have richer and more complete evidence than you had when answering it for the first time.

Question:
{question}

Text clue:
{text_clue}

Your previous result based only on limited information:
**Previous Answer**: {previous_answer}
**Previous Confidence**: {previous_confidence}

Answer the question again after considering all currently available evidence.

Important requirements:
1. The previous answer is not fixed.
2. You may retain the previous answer or replace it with a different answer.
3. Do not mechanically retain either the previous answer or the previous confidence.
4. You must carefully integrate all the information provided in this stage.
5. Do not provide reasoning, explanation, analysis, hedging, confidence, or any additional text.

Output exactly:

**Answer**: <your new answer>

Do not include any additional text."""

V3_STAGE4_META_CONFIDENCE_PROMPT = """You have already answered this question after considering all currently available evidence.

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

Your task is to reassess all currently available evidence and evaluate your confidence in the correctness of the current answer.

Important requirements:
1. The current answer is fixed. You must not change, repeat, restate, or replace it.
2. The initial answer and initial confidence are historical information.
3. You must carefully evaluate all currently available evidence and your most recent answer.
4. The final confidence must represent your confidence that the current answer is correct.
5. Do not default to a middle or neutral confidence class without carefully evaluating the evidence.
6. You must perform the assessment internally, but do not output reasoning, explanation, analysis, the answer, previous confidence values, or any additional text.
7. Choose exactly one confidence class from the list below.

{classes}

Output exactly:

**Confidence**: <CLASS>

CLASS must exactly match one of the confidence class names listed above."""

V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT = """Question:
{question}

Text clue:
{text_clue}

Answer the question using the information.

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, or any additional text.

Output exactly:

**Answer**: <your answer>

Do not include any additional text."""

V4_STAGE2_FULL_EVIDENCE_CONFIDENCE_PROMPT = """You will be shown a question together with an answer you previously provided before.

Question:
{question}

Text clue:
{text_clue}

**Answer**: {answer}

Classify your confidence in the correctness of the previous answer into one of
the following classes.

Confidence refers to the probability that the previous answer is correct
given only the question and text clue.

The previous answer is fixed. Do not change the answer.

The fact that an answer was selected does not mean the confidence must be high.
The answer may be the most likely choice while still having low confidence
when multiple alternatives remain plausible.

{classes}

Output exactly:

**Confidence**: <CLASS>

CLASS must be exactly one of the confidence class names listed above.
Do not include reasoning, explanation, or any additional text."""

EVALUATION_VARIANTS = [
    {
        "version": "v1_visible_previous_confidence",
        "stage1_text_answer_prompt": STAGE1_TEXT_ANSWER_PROMPT,
        "stage2_text_confidence_prompt": STAGE2_TEXT_CONFIDENCE_PROMPT,
        "stage3_image_confidence_prompt": V1_STAGE3_IMAGE_CONFIDENCE_PROMPT,
    },
    {
        "version": "v2_hidden_previous_confidence",
        "stage1_text_answer_prompt": STAGE1_TEXT_ANSWER_PROMPT,
        "stage2_text_confidence_prompt": STAGE2_TEXT_CONFIDENCE_PROMPT,
        "stage3_image_confidence_prompt": V2_STAGE3_IMAGE_CONFIDENCE_PROMPT,
    },
    {
        "version": "v3_reanswer_then_confidence",
        "stage1_text_answer_prompt": STAGE1_TEXT_ANSWER_PROMPT,
        "stage2_text_confidence_prompt": STAGE2_TEXT_CONFIDENCE_PROMPT,
        "stage3_reanswer_prompt": V3_STAGE3_REANSWER_PROMPT,
        "stage4_meta_confidence_prompt": V3_STAGE4_META_CONFIDENCE_PROMPT,
    },
    {
        "version": "v4_full_evidence_baseline",
        "stage1_full_evidence_answer_prompt": V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT,
        "stage2_full_evidence_confidence_prompt": V4_STAGE2_FULL_EVIDENCE_CONFIDENCE_PROMPT,
    },
]

