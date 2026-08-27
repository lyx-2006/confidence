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
    entropy_score: float | None = None
    raw_entropy: float | None = None
    restricted_top1: str | None = None
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
    try:
        from PIL import Image
        with Image.open(resolved) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"Image cannot be read: {resolved}: {exc}") from exc
    return [
        {"type": "image", "image": str(resolved)},
        {"type": "text", "text": prompt},
    ]


class AnswerMetricsInferenceMixin:
    """Mixin so custom --inference-path classes retain their own loader."""

    def __init__(self, *args: Any, **kwargs: Any):
        # Decoder-only Qwen2.5-VL batches must pad on the left for generation;
        # right padding triggers the transformers warning and produces wrong
        # padded-position outputs.  Applied here so every batched
        # generate_answer_with_metrics_batch path inherits it without touching
        # the original qwen-2.5-vl/inference.py loader.
        super().__init__(*args, **kwargs)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tokenizer.padding_side = "left"

    def generate_answer_with_metrics(
        self,
        prompt: str,
        answer_classes: list[str],
        image_path: str | None = None,
        max_new_tokens: int = 24,
    ) -> AnswerGenerationResult:
        request = {
            "prompt": prompt,
            "answer_classes": list(answer_classes),
            "image_path": image_path,
            "max_new_tokens": max_new_tokens,
        }
        return self.generate_answer_with_metrics_batch([request])[0]

    @staticmethod
    def _failed_batch_result(answer_classes: list[str], exc: Exception) -> AnswerGenerationResult:
        return AnswerGenerationResult(
            candidate_count=len(answer_classes),
            error={"type": type(exc).__name__, "message": str(exc)},
            answer_metric_status="failed",
        )

    def generate_answer_with_metrics_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[AnswerGenerationResult]:
        """Run one genuine ``model.generate`` call for a homogeneous batch.

        Requests are prepared independently first.  A malformed image or
        prompt therefore receives its own failed result while valid members
        retain their original order in the single model call and response.
        Text-only and image requests must not be mixed in one invocation.
        """
        if not requests:
            return []
        results: list[AnswerGenerationResult] = [
            AnswerGenerationResult(candidate_count=len(list(req.get("answer_classes", []))))
            for req in requests
        ]
        valid_indices: list[int] = []
        messages_by_index: dict[int, list[dict[str, Any]]] = {}
        rendered_by_index: dict[int, str] = {}
        kind: str | None = None
        for index, request in enumerate(requests):
            started = time.perf_counter()
            try:
                prompt = str(request.get("prompt", ""))
                classes = list(request.get("answer_classes", []))
                image_path = request.get("image_path")
                if not prompt.strip():
                    raise ValueError("prompt must be non-empty")
                current_kind = "image" if image_path is not None else "text"
                messages = [
                    {"role": "user", "content": _user_content(prompt, image_path)},
                    {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_ANSWER_PREFILL}]},
                ]
                rendered = render_continued_assistant(self.processor, messages, ASSISTANT_ANSWER_PREFILL)
                if kind is None:
                    kind = current_kind
                elif current_kind != kind:
                    raise ValueError("Text-only and image requests cannot share a Qwen batch")
                messages_by_index[index] = messages
                rendered_by_index[index] = rendered
                valid_indices.append(index)
            except Exception as exc:
                results[index] = self._failed_batch_result(
                    list(request.get("answer_classes", [])), exc
                )
                results[index].elapsed_seconds = round(time.perf_counter() - started, 6)

        if not valid_indices:
            return results
        try:
            from qwen_vl_utils import process_vision_info

            conversations = [messages_by_index[index] for index in valid_indices]
            image_inputs, video_inputs = process_vision_info(conversations)
            rendered_prompts = [rendered_by_index[index] for index in valid_indices]
            processor_kwargs: dict[str, Any] = {
                "text": rendered_prompts,
                "padding": True,
                "return_tensors": "pt",
            }
            # Text-only processors reject an explicit empty images/videos
            # argument on some transformers versions; vision processors need
            # those values.  Keep the two paths semantically identical.
            if image_inputs is not None and (not hasattr(image_inputs, "__len__") or len(image_inputs) > 0):
                processor_kwargs["images"] = image_inputs
            if video_inputs is not None and (not hasattr(video_inputs, "__len__") or len(video_inputs) > 0):
                processor_kwargs["videos"] = video_inputs
            inputs = self.processor(**processor_kwargs).to(self._get_inputs_device())
            max_tokens = max(int(requests[index].get("max_new_tokens", 24)) for index in valid_indices)
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": False,
                "use_cache": True,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            started = time.perf_counter()
            with torch.inference_mode():
                generated = self.model.generate(**inputs, **generation_kwargs)
            if not generated.scores:
                raise RuntimeError("Model generation returned no first-token scores")
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            input_length = int(inputs.input_ids.shape[1])
            score_rows = generated.scores[0]
            for row, index in enumerate(valid_indices):
                result = results[index]
                continuation = tokenizer.decode(
                    generated.sequences[row, input_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                result.raw_output = ASSISTANT_ANSWER_PREFILL + continuation
                result.answer, result.normalized_answer, result.parse_success = parse_answer_output(result.raw_output)
                metrics = compute_answer_metrics(
                    score_rows[row],
                    list(requests[index].get("answer_classes", [])),
                    result.normalized_answer,
                    tokenizer,
                )
                for key, value in asdict(metrics).items():
                    if key != "error":
                        setattr(result, key, value)
                result.error = metrics.error
                if not result.parse_success:
                    result.error = {
                        "type": "AnswerParseError",
                        "message": f"Could not parse exact answer output: {result.raw_output!r}",
                    }
                    result.answer_metric_status = "failed"
                result.elapsed_seconds = round(time.perf_counter() - started, 6)
        except Exception as exc:
            for index in valid_indices:
                results[index] = self._failed_batch_result(
                    list(requests[index].get("answer_classes", [])), exc
                )
        return results


def build_extended_inference_class(base_class: type[Any]) -> type[Any]:
    if issubclass(base_class, AnswerMetricsInferenceMixin):
        return base_class
    return type(
        "ExtendedQwenVLInference",
        (AnswerMetricsInferenceMixin, base_class),
        {"__module__": __name__},
    )


ExtendedQwenVLInference = build_extended_inference_class(QwenVLInference)
