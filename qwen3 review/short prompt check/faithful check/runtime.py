from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SHORT_ROOT = HERE.parent
REVIEW_ROOT = SHORT_ROOT.parent
REPOSITORY_ROOT = REVIEW_ROOT.parent
for candidate in (REPOSITORY_ROOT, REVIEW_ROOT, SHORT_ROOT, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from confidence_test.answer_metrics import compute_answer_metrics, normalize_answer, parse_answer_output
from dp_SA.io_utils import canonical_hash, sha256_file
from dp_SA.prompts import phase0_prompt
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device
from Steering.capture import _generate, _model_fingerprint
from Steering.runtime import load_qwen3_inference
from capture.short_prompt import SA_PREFILL, phase1_prompt_short

from config import COLORS, MODEL_PATH
from core import normalize, signed_soft_sa
from prompts import ANSWER_PREFILL, image_only_prompt, text_only_prompt


def _messages(prompt: str, image_path: str | None, prefill: str) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = []
    if image_path is not None:
        path = Path(image_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        content.append({"type": "image", "image": str(path)})
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": prefill}]},
    ]


class FaithfulQwenRunner:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path.resolve()
        self.inference = load_qwen3_inference(self.model_path)
        self.processor = self.inference.processor
        self.tokenizer = self.processor.tokenizer
        self.device = model_input_device(self.inference)
        self.sa_class_ids = class_token_ids(self.tokenizer)
        self.model_fingerprint = _model_fingerprint(self.model_path)

    def _inputs(self, prompt: str, image_path: str | None, prefill: str):
        messages = _messages(prompt, image_path, prefill)
        rendered = render_continued_assistant(self.processor, messages, prefill)
        inputs = prepare_multimodal_inputs(self.processor, messages, rendered, device=self.device)
        return messages, rendered, inputs

    def _vocab_logits(self, inputs: Any) -> torch.Tensor:
        with torch.inference_mode():
            outputs = self.inference.model(**inputs, use_cache=False)
        return outputs.logits[0, -1].detach().float().cpu()

    def answer_run(
        self,
        *, prompt: str, image_path: str | None, expected_color: str | None,
        run_generation: bool = True,
    ) -> dict[str, Any]:
        _messages_value, rendered, inputs = self._inputs(prompt, image_path, ANSWER_PREFILL)
        generated_ids: list[int] = []
        continuation = ""
        parsed_answer = normalized_answer = None
        parse_success = False
        if run_generation:
            generated_ids, continuation, _eos = _generate(self.inference, inputs, 24)
            parsed_answer, normalized_answer, parse_success = parse_answer_output(
                ANSWER_PREFILL + continuation
            )
        logits = self._vocab_logits(inputs)
        metric_target = normalize(expected_color) if expected_color is not None else normalized_answer
        metrics = compute_answer_metrics(logits, COLORS, metric_target, self.tokenizer)
        value = asdict(metrics)
        logit_map = value["answer_class_logits"]
        expected = normalize(expected_color) if expected_color is not None else normalized_answer
        other = [float(score) for color, score in logit_map.items() if color != expected]
        target_logit = logit_map.get(expected or "")
        margin = float(target_logit - max(other)) if target_logit is not None and other else None
        gate_passed = None
        if expected_color is not None:
            gate_passed = bool(
                parse_success
                and normalized_answer == expected
                and metrics.restricted_top1 == expected
                and metrics.answer_metric_status == "completed"
            )
        return {
            "model_fingerprint": self.model_fingerprint,
            "prompt": prompt,
            "prompt_hash": canonical_hash(prompt),
            "rendered_hash": canonical_hash(rendered),
            "image_path": str(Path(image_path).resolve()) if image_path else None,
            "image_sha256": sha256_file(image_path) if image_path else None,
            "actual_output": ANSWER_PREFILL + continuation if run_generation else None,
            "actual_answer": parsed_answer,
            "normalized_answer": normalized_answer,
            "parse_success": parse_success,
            "generated_token_ids": generated_ids,
            "expected_color": expected,
            "gate_passed": gate_passed,
            "target_probability": value["answer_class_probabilities"].get(expected or ""),
            "target_logit": target_logit,
            "target_margin": margin,
            "normalized_entropy": value["answer_entropy"],
            **value,
        }

    def text_only(self, question: str, clue: str, expected_color: str) -> dict[str, Any]:
        result = self.answer_run(
            prompt=text_only_prompt(question, clue), image_path=None,
            expected_color=expected_color, run_generation=True,
        )
        result.update({"modality": "text", "question": question, "text_clue": clue})
        return result

    def image_only(self, question: str, image_path: str, expected_color: str) -> dict[str, Any]:
        result = self.answer_run(
            prompt=image_only_prompt(question), image_path=image_path,
            expected_color=expected_color, run_generation=True,
        )
        result.update({"modality": "image", "question": question})
        return result

    def original_answer(self, question: str, clue: str, image_path: str) -> dict[str, Any]:
        result = self.answer_run(
            prompt=phase0_prompt(question, clue), image_path=image_path,
            expected_color=None, run_generation=True,
        )
        result.update({"modality": "multimodal", "question": question, "text_clue": clue})
        result["legal_color_answer"] = result["normalized_answer"] in COLORS
        return result

    def multimodal_score(
        self, question: str, clue: str, image_path: str, fixed_answer: str,
    ) -> dict[str, Any]:
        result = self.answer_run(
            prompt=phase0_prompt(question, clue), image_path=image_path,
            expected_color=fixed_answer, run_generation=False,
        )
        probability = result["answer_class_probabilities"].get(normalize(fixed_answer))
        result["fixed_answer"] = normalize(fixed_answer)
        result["fixed_answer_log_probability"] = (
            math.log(max(float(probability), 1e-45)) if probability is not None else None
        )
        return result

    def short_sa(self, question: str, clue: str, image_path: str, fixed_answer: str) -> dict[str, Any]:
        prompt = phase1_prompt_short(question, clue, fixed_answer)
        _messages_value, rendered, inputs = self._inputs(prompt, image_path, SA_PREFILL)
        logits = self._vocab_logits(inputs)
        score = soft_sa_from_logits(logits, self.sa_class_ids)
        generated_ids, generated_text, _eos = _generate(
            self.inference, inputs, 1, self.sa_class_ids
        )
        if generated_text not in set(map(str, range(9))):
            raise ValueError(f"Invalid constrained short-SA output: {generated_text!r}")
        if int(generated_text) != int(score["argmax_hard_class"]):
            raise ValueError("Short-SA forward argmax differs from constrained generation")
        return {
            "prompt": prompt,
            "prompt_hash": canonical_hash(prompt),
            "rendered_hash": canonical_hash(rendered),
            "fixed_answer": fixed_answer,
            **score,
            "soft_sa_signed": signed_soft_sa(float(score["soft_sa_image_score"])),
            "raw_generated_class": generated_text,
            "generated_token_ids": generated_ids,
        }
