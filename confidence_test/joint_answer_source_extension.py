"""Joint answer + source-attribution generation with strict two-line parsing."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from layer_metacognition.conversation_builder import (
    prepare_multimodal_inputs,
    render_continued_assistant,
)

from .answer_metrics import compute_answer_metrics, failed_answer_metrics, normalize_answer
from .inference_extension import ASSISTANT_ANSWER_PREFILL
from .source_attribution_analyzer import parse_joint_answer_source_output
from .source_attribution_schema import SOURCE_ATTRIBUTION_CLASSES


@dataclass
class JointAnswerSourceGenerationResult:
    raw_output: str = ""
    answer: str | None = None
    normalized_answer: str | None = None
    source_label: str | None = None
    parse_success: bool = False
    answer_prob: float | None = None
    raw_answer_entropy: float | None = None
    answer_entropy: float | None = None
    answer_class_logits: dict[str, float] = field(default_factory=dict)
    answer_class_probabilities: dict[str, float] = field(default_factory=dict)
    answer_metric_status: str = "failed"
    candidate_count: int = 0
    elapsed_seconds: float = 0.0
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


class JointAnswerSourceGenerator:
    def __init__(self, inference: Any):
        self.inference = inference
        self.model = inference.model
        self.processor = inference.processor
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    def generate(
        self,
        prompt: str,
        answer_classes: list[str],
        image_path: str | None,
        max_new_tokens: int = 32,
        source_classes: Sequence[str] = SOURCE_ATTRIBUTION_CLASSES,
    ) -> JointAnswerSourceGenerationResult:
        started = time.perf_counter()
        result = JointAnswerSourceGenerationResult(candidate_count=len(answer_classes))
        try:
            messages = [
                {"role": "user", "content": _user_content(prompt, image_path)},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ASSISTANT_ANSWER_PREFILL}],
                },
            ]
            rendered = render_continued_assistant(
                self.processor,
                messages,
                ASSISTANT_ANSWER_PREFILL,
            )
            inputs = prepare_multimodal_inputs(
                self.processor,
                messages,
                rendered,
                device=self.inference._get_inputs_device(),
            )
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            input_length = int(inputs.input_ids.shape[1])
            continuation = self.tokenizer.decode(
                generated.sequences[0, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            result.raw_output = ASSISTANT_ANSWER_PREFILL + continuation
            answer, source_label, parsed = parse_joint_answer_source_output(
                result.raw_output,
                source_classes,
            )
            result.answer = answer
            result.normalized_answer = normalize_answer(answer)
            result.source_label = source_label
            result.parse_success = parsed and result.normalized_answer is not None
            if generated.scores:
                metrics = compute_answer_metrics(
                    generated.scores[0][0],
                    answer_classes,
                    result.normalized_answer,
                    self.tokenizer,
                )
            else:
                metrics = failed_answer_metrics(
                    len(answer_classes),
                    "MissingGenerationScores",
                    "Joint generation returned no first-token scores",
                )
            for key, value in asdict(metrics).items():
                if key != "error":
                    setattr(result, key, value)
            if metrics.error is not None:
                result.error = metrics.error
            if not result.parse_success:
                result.answer_metric_status = "failed"
                result.error = {
                    "type": "JointParseError",
                    "message": f"Could not parse exact joint output: {result.raw_output!r}",
                }
        except Exception as exc:
            result.error = {"type": type(exc).__name__, "message": str(exc)}
            result.answer_metric_status = "failed"
        result.elapsed_seconds = round(time.perf_counter() - started, 6)
        return result
