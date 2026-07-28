"""Content-free target prompts for SAC Patchscope readouts."""

IDENTITY_PATCHSCOPE_USER_PROMPT = """The assistant prefix contains identity examples.
Continue the final item by emitting exactly one class digit from 0 to 8.
Do not output reasoning, explanation, punctuation, or any additional text."""

IDENTITY_PATCHSCOPE_ASSISTANT_PREFILL = """0 -> 0
1 -> 1
2 -> 2
3 -> 3
4 -> 4
5 -> 5
6 -> 6
7 -> 7
8 -> 8
?"""

SEMANTIC_PATCHSCOPE_USER_PROMPT = """Classify the relative contribution of the text clue and the image into exactly one source attribution class.

{source_classes}

Output exactly:

**Source Attribution**:<CLASS>

CLASS must exactly match one of the source attribution class numbers listed above.
Do not output reasoning, explanation, an answer, confidence, or any additional text."""
