"""Joint answer + source-attribution generation with strict two-line parsing."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Sequence

import torch

from layer_metacognition.conversation_builder import (
    prepare_multimodal_inputs,
    render_continued_assistant,
)

from .answer_metrics import compute_answer_metrics, failed_answer_metrics, normalize_answer
from .inference_extension import ASSISTANT_ANSWER_PREFILL
from .source_attribution_analyzer import parse_joint_answer_source_output
from .source_attribution_schema import SOURCE_ATTRIBUTION_CLASSES
from .source_attribution_schema import (
    SOURCE_ATTRIBUTION_MIDPOINTS,
    build_source_token_specification,
    gather_source_class_logits,
    source_distribution,
)


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
    source_attribution: dict[str, Any] | None = None
    source_metric_status: str = "failed"
    source_token_step: int | None = None
    generated_token_ids: list[int] = field(default_factory=list)
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

    def prepare_inputs(
        self,
        prompt: str,
        image_path: str | None,
        *,
        assistant_text: str = ASSISTANT_ANSWER_PREFILL,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Build the exact joint conversation inputs used by generation."""
        messages = [
            {"role": "user", "content": _user_content(prompt, image_path)},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]
        rendered = render_continued_assistant(
            self.processor,
            messages,
            assistant_text,
        )
        inputs = prepare_multimodal_inputs(
            self.processor,
            messages,
            rendered,
            device=self.inference._get_inputs_device(),
        )
        return messages, rendered, inputs

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        assistant_text: str = ASSISTANT_ANSWER_PREFILL,
    ) -> tuple[str, Any]:
        """Prepare an arbitrary real multi-turn conversation.

        The final message must be the assistant continuation represented by
        ``assistant_text``.  The caller retains the messages so History
        experiments can hash and reconstruct the exact causal prefix.
        """
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("messages must end in an assistant continuation")
        final_content = messages[-1].get("content")
        if not (
            isinstance(final_content, list)
            and len(final_content) == 1
            and final_content[0].get("type") == "text"
            and final_content[0].get("text") == assistant_text
        ):
            raise ValueError("final assistant content must exactly match assistant_text")
        rendered = render_continued_assistant(self.processor, messages, assistant_text)
        inputs = prepare_multimodal_inputs(
            self.processor,
            messages,
            rendered,
            device=self.inference._get_inputs_device(),
        )
        return rendered, inputs

    def generate_messages(
        self,
        messages: list[dict[str, Any]],
        answer_classes: list[str],
        *,
        assistant_text: str = ASSISTANT_ANSWER_PREFILL,
        max_new_tokens: int = 32,
        use_cache: bool = True,
        source_classes: Sequence[str] = SOURCE_ATTRIBUTION_CLASSES,
        source_midpoints: Sequence[float] = SOURCE_ATTRIBUTION_MIDPOINTS,
        generation_context_factory: (
            Callable[[Any, str], ContextManager[Any]] | None
        ) = None,
    ) -> JointAnswerSourceGenerationResult:
        """Generate from multi-turn messages and retain Answer and SA step scores."""
        started = time.perf_counter()
        result = JointAnswerSourceGenerationResult(candidate_count=len(answer_classes))
        try:
            rendered, inputs = self.prepare_messages(
                messages,
                assistant_text=assistant_text,
            )
            generation_context = (
                nullcontext()
                if generation_context_factory is None
                else generation_context_factory(inputs, rendered)
            )
            with generation_context, torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=use_cache,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_logits=True,
                )
            input_length = int(inputs.input_ids.shape[1])
            generated_ids = [
                int(value) for value in generated.sequences[0, input_length:].tolist()
            ]
            result.generated_token_ids = generated_ids
            continuation = self.tokenizer.decode(
                generated.sequences[0, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            result.raw_output = assistant_text + continuation
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

            if source_label is not None and generated.scores:
                specification = build_source_token_specification(
                    self.tokenizer, source_classes
                )
                source_ids = set(specification.class_token_ids[source_label])
                source_steps = [
                    index for index, token_id in enumerate(generated_ids)
                    if token_id in source_ids and index < len(generated.scores)
                ]
                if source_steps:
                    source_step = source_steps[-1]
                    raw_generation_logits = getattr(generated, "logits", None)
                    source_vocab_logits = (
                        raw_generation_logits[source_step][0]
                        if raw_generation_logits is not None
                        else generated.scores[source_step][0]
                    )
                    class_logits = gather_source_class_logits(
                        source_vocab_logits,
                        specification.class_token_ids,
                        source_classes,
                    )
                    source_result = source_distribution(
                        class_logits,
                        class_token_ids=specification.class_token_ids,
                        raw_output=result.raw_output,
                        parsed_label=source_label,
                        token_diagnostics=specification.to_dict(),
                        classes=source_classes,
                        midpoints=source_midpoints,
                    )
                    result.source_attribution = source_result.to_dict()
                    result.source_metric_status = "completed"
                    result.source_token_step = source_step
            if not result.parse_success:
                result.answer_metric_status = "failed"
                result.error = {
                    "type": "JointParseError",
                    "message": f"Could not parse exact joint output: {result.raw_output!r}",
                }
            elif result.source_metric_status != "completed":
                result.error = {
                    "type": "SourceScoreStepError",
                    "message": "Parsed SA class but could not locate its generation score step",
                }
        except Exception as exc:
            result.error = {"type": type(exc).__name__, "message": str(exc)}
            result.answer_metric_status = "failed"
            result.source_metric_status = "failed"
        result.elapsed_seconds = round(time.perf_counter() - started, 6)
        return result

    def generate(
        self,
        prompt: str,
        answer_classes: list[str],
        image_path: str | None,
        max_new_tokens: int = 32,
        source_classes: Sequence[str] = SOURCE_ATTRIBUTION_CLASSES,
        generation_context_factory: (
            Callable[[Any, str], ContextManager[Any]] | None
        ) = None,
    ) -> JointAnswerSourceGenerationResult:
        messages = [
            {"role": "user", "content": _user_content(prompt, image_path)},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ASSISTANT_ANSWER_PREFILL}],
            },
        ]
        return self.generate_messages(
            messages,
            answer_classes,
            max_new_tokens=max_new_tokens,
            source_classes=source_classes,
            generation_context_factory=generation_context_factory,
        )
