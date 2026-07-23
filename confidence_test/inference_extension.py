"""Answer generation extension using the original Qwen model and processor."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from layer_metacognition.conversation_builder import (
    prepare_multimodal_inputs,
    render_continued_assistant,
)

from confidence_test.answer_metrics import (
    compute_answer_metrics,
    failed_answer_metrics,
    parse_answer_output,
)
from confidence_test.runtime_imports import QwenVLInference


ASSISTANT_ANSWER_PREFILL = "**Answer**:"


@dataclass
class AnswerGenerationResult:
    raw_output: str = ""
    answer: str | None = None
    normalized_answer: str | None = None
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


class AnswerMetricsInferenceMixin:
    """Mixin so custom --inference-path classes retain their own loader."""

    def generate_answer_with_metrics(
        self,
        prompt: str,
        answer_classes: list[str],
        image_path: str | None = None,
        max_new_tokens: int = 24,
    ) -> AnswerGenerationResult:
        started = time.perf_counter()
        result = AnswerGenerationResult(candidate_count=len(answer_classes))
        try:
            if not prompt.strip():
                raise ValueError("prompt must be non-empty")
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
                device=self._get_inputs_device(),
            )
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": True,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            with torch.inference_mode():
                generated = self.model.generate(**inputs, **generation_kwargs)
            input_length = int(inputs.input_ids.shape[1])
            generated_tokens = generated.sequences[0, input_length:]
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            continuation = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            result.raw_output = ASSISTANT_ANSWER_PREFILL + continuation
            result.answer, result.normalized_answer, result.parse_success = parse_answer_output(
                result.raw_output
            )
            if not generated.scores:
                metrics = failed_answer_metrics(
                    len(answer_classes),
                    "MissingGenerationScores",
                    "Model generation returned no first-token scores",
                )
            else:
                metrics = compute_answer_metrics(
                    generated.scores[0][0],
                    answer_classes,
                    result.normalized_answer,
                    tokenizer,
                )
            for key, value in asdict(metrics).items():
                if key != "error":
                    setattr(result, key, value)
            if metrics.error is not None:
                result.error = metrics.error
            if not result.parse_success:
                result.error = {
                    "type": "AnswerParseError",
                    "message": f"Could not parse exact answer output: {result.raw_output!r}",
                }
                result.answer_metric_status = "failed"
        except Exception as exc:
            result.error = {"type": type(exc).__name__, "message": str(exc)}
            result.answer_metric_status = "failed"
        result.elapsed_seconds = round(time.perf_counter() - started, 6)
        return result


def build_extended_inference_class(base_class: type[Any]) -> type[Any]:
    if issubclass(base_class, AnswerMetricsInferenceMixin):
        return base_class
    return type(
        "ExtendedQwenVLInference",
        (AnswerMetricsInferenceMixin, base_class),
        {"__module__": __name__},
    )


ExtendedQwenVLInference = build_extended_inference_class(QwenVLInference)
