"""Semantic answer and source-attribution Patchscope decoding."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Any, Sequence

import torch

from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from confidence_test.source_attribution_schema import (
    ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
    SOURCE_ATTRIBUTION_CLASSES,
    SOURCE_ATTRIBUTION_CLASS_TEXT,
    SOURCE_ATTRIBUTION_MIDPOINTS,
)

from .conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from .layer_stage_analyzer import restricted_logits, source_vocab_readout
from .model_adapter import (
    LanguageModules,
    run_logits_forward,
    run_patched_logits_forward,
)
from .source_patchscope_prompts import (
    IDENTITY_PATCHSCOPE_USER_PROMPT,
    SEMANTIC_ANSWER_PATCHSCOPE_USER_PROMPT,
    SEMANTIC_PATCHSCOPE_USER_PROMPT,
)
from .token_positions import (
    encode_without_special_tokens,
    locate_marker_in_assistant,
    unique_subsequence,
)
from .token_spans import build_rendered_alignment


PATCHSCOPE_ANALYSIS_MODES = ("Identity", "Semantic")
ANSWER_PATCHSCOPE_VARIANTS = (
    "original",
    "shuffle_1",
    "shuffle_2",
    "shuffle_3",
)
ANSWER_PATCHSCOPE_SHUFFLE_SEEDS = {
    "shuffle_1": 17,
    "shuffle_2": 29,
    "shuffle_3": 43,
}


def answer_candidate_orders(
    candidates: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Return one canonical and three deterministic, distinct prompt orders."""
    original = tuple(str(candidate) for candidate in candidates)
    if not original or len(original) != len(set(original)):
        raise ValueError(
            "Answer Patchscope candidates must be non-empty and distinct"
        )
    orders = {"original": original}
    used = {original}
    for variant, seed in ANSWER_PATCHSCOPE_SHUFFLE_SEEDS.items():
        for attempt in range(100):
            shuffled = list(original)
            random.Random(seed + attempt * 1009).shuffle(shuffled)
            candidate_order = tuple(shuffled)
            if candidate_order not in used:
                orders[variant] = candidate_order
                used.add(candidate_order)
                break
        else:
            raise ValueError(
                "Candidate set cannot produce three distinct shuffled orders"
            )
    return orders


@dataclass
class PreparedSourceTarget:
    name: str
    inputs: Any
    target_position: int
    rendered_prompt: str
    baseline: dict[str, Any]


@dataclass
class PreparedAnswerTarget:
    candidates: tuple[str, ...]
    display_candidates: tuple[str, ...]
    inputs: Any
    target_position: int
    rendered_prompt: str


class AnswerPatchscopeDecoder:
    """Decode AC hidden states with a content-free semantic answer target."""

    def __init__(
        self,
        *,
        inference: Any,
        modules: LanguageModules,
    ):
        self.inference = inference
        self.model = inference.model
        self.modules = modules
        self.processor = inference.processor
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.targets: dict[
            tuple[tuple[str, ...], tuple[str, ...]],
            PreparedAnswerTarget,
        ] = {}
        self.call_counts = {"target_prepare": 0, "patched": 0}

    @staticmethod
    def _text_content(text: str) -> list[dict[str, str]]:
        return [{"type": "text", "text": text}]

    def _prepare_target(
        self,
        candidates: tuple[str, ...],
        display_candidates: tuple[str, ...],
    ) -> PreparedAnswerTarget:
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError(
                "Answer Patchscope candidates must be non-empty and distinct"
            )
        if (
            len(display_candidates) != len(set(display_candidates))
            or set(display_candidates) != set(candidates)
        ):
            raise ValueError(
                "Answer Patchscope display order must be a permutation of candidates"
            )
        user_prompt = SEMANTIC_ANSWER_PATCHSCOPE_USER_PROMPT.format(
            answer_classes="\n".join(
                f"- {candidate}" for candidate in display_candidates
            )
        )
        prefill = ASSISTANT_ANSWER_PREFILL
        messages = [
            {"role": "user", "content": self._text_content(user_prompt)},
            {"role": "assistant", "content": self._text_content(prefill)},
        ]
        rendered = render_continued_assistant(
            self.processor,
            messages,
            prefill,
        )
        inputs = prepare_multimodal_inputs(
            self.processor,
            messages,
            rendered,
            device=self.inference._get_inputs_device(),
        )
        alignment = build_rendered_alignment(
            self.tokenizer,
            rendered,
            inputs.input_ids,
            inputs.attention_mask,
        )
        location = locate_marker_in_assistant(
            self.tokenizer,
            alignment.rendered_ids,
            prefill,
            prefill,
            name="semantic answer target ac",
            position_map=alignment.rendered_to_processed,
            processed_ids=alignment.processed_ids,
        )
        target_position = int(location["position"])
        if target_position != len(alignment.processed_ids) - 1:
            raise ValueError(
                f"Answer target position {target_position} is not the final valid "
                f"input token {len(alignment.processed_ids) - 1}"
            )
        self.call_counts["target_prepare"] += 1
        return PreparedAnswerTarget(
            candidates=candidates,
            display_candidates=display_candidates,
            inputs=inputs,
            target_position=target_position,
            rendered_prompt=rendered,
        )

    def target_for(
        self,
        candidates: Sequence[str],
        display_candidates: Sequence[str] | None = None,
    ) -> PreparedAnswerTarget:
        canonical = tuple(str(candidate) for candidate in candidates)
        display = (
            canonical
            if display_candidates is None
            else tuple(str(candidate) for candidate in display_candidates)
        )
        key = (canonical, display)
        if key not in self.targets:
            self.targets[key] = self._prepare_target(canonical, display)
        return self.targets[key]

    def run_patched_answer_readout(
        self,
        *,
        layer_index: int,
        source_hidden: torch.Tensor,
        candidates: Sequence[str],
        collision_report: dict[str, Any],
        display_candidates: Sequence[str] | None = None,
        variant_id: str = "original",
    ) -> tuple[dict[str, Any], torch.Tensor]:
        target = self.target_for(candidates, display_candidates)
        logits = run_patched_logits_forward(
            self.model,
            target.inputs,
            self.modules,
            layer_index=layer_index,
            target_position=target.target_position,
            source_hidden=source_hidden,
        )
        self.call_counts["patched"] += 1
        class_token_ids = {
            label: collision_report["labels"][label]["first_token_variants"]
            for label in target.candidates
        }
        class_logits = restricted_logits(
            logits,
            target.candidates,
            class_token_ids,
        )
        probabilities = torch.softmax(class_logits, dim=-1)
        predicted_index = int(torch.argmax(probabilities).item())
        result = {
            "layer_index": int(layer_index),
            "analysis_mode": "Semantic",
            "target_name": "semantic_answer",
            "target_variant": str(variant_id),
            "target_candidate_order": list(target.display_candidates),
            "target_position": target.target_position,
            "predicted_answer": target.candidates[predicted_index],
            "predicted_answer_probability": float(
                probabilities[predicted_index].item()
            ),
            "answer_class_logits": {
                label: float(class_logits[index].item())
                for index, label in enumerate(target.candidates)
            },
            "answer_class_probabilities": {
                label: float(probabilities[index].item())
                for index, label in enumerate(target.candidates)
            },
        }
        del class_logits, probabilities
        return result, logits


class SourcePatchscopeDecoder:
    """Prepare content-free targets once and patch SAC states layer by layer."""

    def __init__(
        self,
        *,
        inference: Any,
        modules: LanguageModules,
        class_token_ids: dict[str, Sequence[int]],
        analysis_modes: Sequence[str],
        source_classes: Sequence[str] = SOURCE_ATTRIBUTION_CLASSES,
        source_midpoints: Sequence[float] = SOURCE_ATTRIBUTION_MIDPOINTS,
        source_class_text: str = SOURCE_ATTRIBUTION_CLASS_TEXT,
    ):
        selected = [
            mode for mode in PATCHSCOPE_ANALYSIS_MODES if mode in set(analysis_modes)
        ]
        self.inference = inference
        self.model = inference.model
        self.modules = modules
        self.processor = inference.processor
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.class_token_ids = {
            label: [int(token_id) for token_id in ids]
            for label, ids in class_token_ids.items()
        }
        self.source_classes = [str(label) for label in source_classes]
        self.source_midpoints = [float(value) for value in source_midpoints]
        self.source_class_text = str(source_class_text)
        self.targets: dict[str, PreparedSourceTarget] = {}
        self.call_counts = {"baseline": 0, "patched": 0}
        for mode in selected:
            if mode == "Identity":
                self.targets[mode] = self.prepare_identity_target()
            else:
                self.targets[mode] = self.prepare_semantic_target()

    @staticmethod
    def _text_content(text: str) -> list[dict[str, str]]:
        return [{"type": "text", "text": text}]

    def _prepare_target(
        self,
        *,
        mode: str,
        user_prompt: str,
        assistant_prefill: str,
    ) -> PreparedSourceTarget:
        messages = [
            {"role": "user", "content": self._text_content(user_prompt)},
            {"role": "assistant", "content": self._text_content(assistant_prefill)},
        ]
        rendered = render_continued_assistant(
            self.processor,
            messages,
            assistant_prefill,
        )
        inputs = prepare_multimodal_inputs(
            self.processor,
            messages,
            rendered,
            device=self.inference._get_inputs_device(),
        )
        processed_ids = [int(value) for value in inputs.input_ids[0].tolist()]
        alignment = build_rendered_alignment(
            self.tokenizer,
            rendered,
            inputs.input_ids,
            inputs.attention_mask,
        )
        if mode == "Identity":
            assistant_ids = encode_without_special_tokens(
                self.tokenizer,
                assistant_prefill,
            )
            assistant_start, _assistant_end = unique_subsequence(
                alignment.rendered_ids,
                assistant_ids,
                name="identity target assistant output",
            )
            placeholder_ids = encode_without_special_tokens(self.tokenizer, "?")
            placeholder_start, placeholder_end = unique_subsequence(
                assistant_ids,
                placeholder_ids,
                name="identity target placeholder",
            )
            if placeholder_end != len(assistant_ids):
                raise ValueError("Identity target placeholder is not the final assistant token")
            raw_position = assistant_start + placeholder_end - 1
            try:
                target_position = int(alignment.rendered_to_processed[raw_position])
            except KeyError as exc:
                raise ValueError(
                    f"Identity placeholder maps through an unavailable token: {exc}"
                ) from exc
            if assistant_ids[placeholder_start:placeholder_end] != placeholder_ids:
                raise RuntimeError("Identity placeholder token validation failed")
        else:
            location = locate_marker_in_assistant(
                self.tokenizer,
                alignment.rendered_ids,
                assistant_prefill,
                ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
                name="semantic target sac",
                position_map=alignment.rendered_to_processed,
                processed_ids=processed_ids,
            )
            target_position = int(location["position"])
        if target_position != len(processed_ids) - 1:
            raise ValueError(
                f"{mode} target position {target_position} is not the final valid "
                f"input token {len(processed_ids) - 1}"
            )

        logits = run_logits_forward(
            self.model,
            inputs,
            [target_position],
            self.modules,
        )[target_position]
        self.call_counts["baseline"] += 1
        baseline = source_vocab_readout(
            logits,
            self.class_token_ids,
            analysis_mode=mode,
            source_classes=self.source_classes,
            source_midpoints=self.source_midpoints,
        )
        baseline["target_name"] = mode.casefold()
        baseline["target_position"] = target_position
        del logits
        return PreparedSourceTarget(
            name=mode.casefold(),
            inputs=inputs,
            target_position=target_position,
            rendered_prompt=rendered,
            baseline=baseline,
        )

    def prepare_identity_target(self) -> PreparedSourceTarget:
        assistant_prefill = "\n".join(
            [*(f"{label} -> {label}" for label in self.source_classes), "?"]
        )
        user_prompt = IDENTITY_PATCHSCOPE_USER_PROMPT.replace(
            "from 0 to 8",
            f"from {self.source_classes[0]} to {self.source_classes[-1]}",
        )
        return self._prepare_target(
            mode="Identity",
            user_prompt=user_prompt,
            assistant_prefill=assistant_prefill,
        )

    def prepare_semantic_target(self) -> PreparedSourceTarget:
        return self._prepare_target(
            mode="Semantic",
            user_prompt=SEMANTIC_PATCHSCOPE_USER_PROMPT.format(
                source_classes=self.source_class_text
            ),
            assistant_prefill=ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
        )

    def run_target_baseline(self, analysis_mode: str) -> dict[str, Any]:
        try:
            return deepcopy(self.targets[analysis_mode].baseline)
        except KeyError as exc:
            raise ValueError(
                f"Patchscope target {analysis_mode!r} was not prepared"
            ) from exc

    def baselines(self) -> dict[str, dict[str, Any]]:
        return {
            mode: self.run_target_baseline(mode)
            for mode in PATCHSCOPE_ANALYSIS_MODES
            if mode in self.targets
        }

    def run_patched_source_readout(
        self,
        *,
        analysis_mode: str,
        layer_index: int,
        source_hidden: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        try:
            target = self.targets[analysis_mode]
        except KeyError as exc:
            raise ValueError(
                f"Patchscope target {analysis_mode!r} was not prepared"
            ) from exc
        logits = run_patched_logits_forward(
            self.model,
            target.inputs,
            self.modules,
            layer_index=layer_index,
            target_position=target.target_position,
            source_hidden=source_hidden,
        )
        self.call_counts["patched"] += 1
        result = source_vocab_readout(
            logits,
            self.class_token_ids,
            layer_index=layer_index,
            analysis_mode=analysis_mode,
            source_classes=self.source_classes,
            source_midpoints=self.source_midpoints,
        )
        result["target_name"] = target.name
        result["target_position"] = target.target_position
        return result, logits
