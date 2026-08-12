"""Generation and restricted first-token scoring for source attribution."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import torch

from layer_metacognition.conversation_builder import (
    prepare_multimodal_inputs,
    render_continued_assistant,
)

from .source_attribution_schema import (
    ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
    SOURCE_ATTRIBUTION_CLASSES,
    SOURCE_ATTRIBUTION_MIDPOINTS,
    SourceAttributionResult,
    build_source_token_specification,
    gather_source_class_logits,
    source_distribution,
)


PARALLEL_SOURCE_PATTERN = re.compile(
    r"\*\*Source Attribution\*\*:[ \t]*([0-9])(?:\*\*)?\s*\Z"
)
JOINT_SOURCE_PATTERN = re.compile(
    r"\*\*Answer\*\*:[ \t]+([^\r\n]*?\S)[ \t]*\r?\n+"
    r"[ \t]*\*\*Source Attribution\*\*:[ \t]*([0-9])(?:\*\*)?\s*\Z"
)
JOINT_ADD_CRITERION_PATTERN = re.compile(
    r"\*\*Answer\*\*:[ \t]+([^\r\n]*?\S)[ \t]*\r?\n"
    r"[ \t]*addCriterion[ \t]*\r?\n[ \t]*([0-9])(?:\*\*)?\s*\Z"
)
JOINT_COLON_IN_BOLD_PATTERN = re.compile(
    r"\*\*Answer:\*\*[ \t]+([^\r\n]*?\S)[ \t]*\r?\n+"
    r"[ \t]*\*\*Source Attribution:\*\*[ \t]*([0-9])[.]?\s*\Z"
)
JOINT_PLAIN_FIELDS_PATTERN = re.compile(
    r"Answer:[ \t]+([^\r\n]*?\S)[ \t]*\r?\n+"
    r"[ \t]*Source Attribution:[ \t]*([0-9])[.]?\s*\Z"
)
JOINT_SINGLE_LINE_PATTERN = re.compile(
    r"\*\*Answer\*\*:[ \t]+(.+?\S)[ \t]+"
    r"\*\*Source Attribution\*\*:[ \t]*([0-9])(?:\*\*)?[.]?\s*\Z"
)
JOINT_ANGLE_CLASS_PATTERN = re.compile(
    r"\*\*Answer\*\*:[ \t]+([^\r\n]*?\S)[ \t]*\r?\n+"
    r"[ \t]*\*\*Source Attribution\*\*:[ \t]*<([0-9])>?\s*\Z"
)
JOINT_BOLD_VALUE_PATTERN = re.compile(
    r"\*\*Answer\*\*:[ \t]+([^\r\n]*?\S)[ \t]*\r?\n+"
    r"[ \t]*\*\*Source Attribution\*\*:[ \t]*\*\*([0-9])\*\*[ \t]*\Z"
)
JOINT_ADD_CRITERION_FIELD_PATTERN = re.compile(
    r"\*\*Answer\*\*:[ \t]+([^\r\n]*?\S)[ \t]*\r?\n+"
    r"[ \t]*addCriterion\*\*:[ \t]*([0-9])(?:\*\*)?[ \t]*\Z"
)
JOINT_CODE_FENCE_PATTERN = re.compile(
    r"[ \t]*```(?:text|markdown)?[ \t]*\r?\n"
    r"(?P<body>.*?)\r?\n```[ \t]*\s*\Z",
    re.DOTALL | re.IGNORECASE,
)

JOINT_PARSE_PATTERNS = (
    JOINT_SOURCE_PATTERN,
    JOINT_ADD_CRITERION_PATTERN,
    JOINT_COLON_IN_BOLD_PATTERN,
    JOINT_PLAIN_FIELDS_PATTERN,
    JOINT_SINGLE_LINE_PATTERN,
    JOINT_ANGLE_CLASS_PATTERN,
    JOINT_BOLD_VALUE_PATTERN,
    JOINT_ADD_CRITERION_FIELD_PATTERN,
)


def parse_parallel_source_output(
    raw_output: str,
    source_classes: Sequence[str] = SOURCE_ATTRIBUTION_CLASSES,
) -> str | None:
    match = PARALLEL_SOURCE_PATTERN.fullmatch(raw_output)
    if match is None or match.group(1) not in source_classes:
        return None
    return match.group(1)


def parse_joint_answer_source_output(
    raw_output: str,
    source_classes: Sequence[str] = SOURCE_ATTRIBUTION_CLASSES,
) -> tuple[str | None, str | None, bool]:
    candidates = [raw_output]
    fenced = JOINT_CODE_FENCE_PATTERN.fullmatch(raw_output)
    if fenced is not None:
        candidates.append(fenced.group("body"))

    # Retry only known, bounded wire-format deviations. Each expression still
    # consumes the complete candidate, so explanations or additional fields
    # cannot be silently accepted.
    for candidate in candidates:
        for pattern in JOINT_PARSE_PATTERNS:
            match = pattern.fullmatch(candidate)
            if match is None or match.group(2) not in source_classes:
                continue
            answer = match.group(1)
            # Qwen sometimes closes the answer value as Markdown even though
            # only the field name was requested in bold.
            if answer.endswith("**"):
                answer = answer[:-2].rstrip()
            if answer:
                return answer, match.group(2), True
    return None, None, False


def _user_content(prompt: str, image_path: str | None) -> list[dict[str, str]]:
    if image_path is None:
        return [{"type": "text", "text": prompt}]
    resolved = Path(image_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return [
        {"type": "image", "image": str(resolved)},
        {"type": "text", "text": prompt},
    ]


class SourceAttributionAnalyzer:
    def __init__(
        self,
        inference: Any,
        max_new_tokens: int = 4,
        *,
        source_classes: Sequence[str] = SOURCE_ATTRIBUTION_CLASSES,
        source_midpoints: Sequence[float] = SOURCE_ATTRIBUTION_MIDPOINTS,
    ):
        self.inference = inference
        self.model = inference.model
        self.processor = inference.processor
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.max_new_tokens = max_new_tokens
        self.source_classes = [str(label) for label in source_classes]
        self.source_midpoints = [float(value) for value in source_midpoints]
        self.token_specification = build_source_token_specification(
            self.tokenizer,
            self.source_classes,
        )

    def score_vocab_logits(
        self,
        vocab_logits: torch.Tensor,
        *,
        raw_output: str,
        parsed_label: str | None,
    ) -> SourceAttributionResult:
        class_logits = gather_source_class_logits(
            vocab_logits.float(),
            self.token_specification.class_token_ids,
            self.source_classes,
        )
        return source_distribution(
            class_logits,
            class_token_ids=self.token_specification.class_token_ids,
            raw_output=raw_output,
            parsed_label=parsed_label,
            token_diagnostics=self.token_specification.to_dict(),
            classes=self.source_classes,
            midpoints=self.source_midpoints,
        )

    def build_messages(
        self,
        prompt: str,
        image_path: str | None,
    ) -> tuple[list[dict[str, Any]], str]:
        if not prompt.strip():
            raise ValueError("Source-attribution prompt must be non-empty")
        messages = [
            {"role": "user", "content": _user_content(prompt, image_path)},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ASSISTANT_SOURCE_ATTRIBUTION_PREFILL}],
            },
        ]
        rendered = render_continued_assistant(
            self.processor,
            messages,
            ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
        )
        return messages, rendered

    def analyze_prompt(
        self,
        prompt: str,
        image_path: str | None,
    ) -> SourceAttributionResult:
        messages, rendered = self.build_messages(prompt, image_path)
        inputs = prepare_multimodal_inputs(
            self.processor,
            messages,
            rendered,
            device=self.inference._get_inputs_device(),
        )
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        if not generated.scores:
            raise RuntimeError("Source generation returned no first-token logits")
        input_length = int(inputs.input_ids.shape[1])
        continuation = self.tokenizer.decode(
            generated.sequences[0, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw_output = ASSISTANT_SOURCE_ATTRIBUTION_PREFILL + continuation
        parsed_label = parse_parallel_source_output(
            raw_output,
            self.source_classes,
        )
        return self.score_vocab_logits(
            generated.scores[0][0],
            raw_output=raw_output,
            parsed_label=parsed_label,
        )
