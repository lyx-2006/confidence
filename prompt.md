STAGE1 Text only + Answer

```python
STAGE1_TEXT_ANSWER_PROMPT = """Question:
{question}

Text clue:
{text_clue}

Answer the question using only the question and text clue above.

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, or any additional text.

Output exactly:

**Answer**: <your answer>

Do not include any additional text."""
```

STAGE2 Confidence

```python
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
```

STAGE3 image new confidence

```python
STAGE3_IMAGE_CONFIDENCE_PROMPT = """You previously answered this question with limited information.
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

{classes}

Each confidence class represents the probability that the fixed previous answer is correct after considering all currently available evidence.

Output exactly in the following format:
**Confidence**: $CLASS

CLASS must exactly match one of the confidence class names listed above.
Do not include quotation marks, probability ranges, the previous answer, or any additional text."""
```

