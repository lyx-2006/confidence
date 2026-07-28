"""Identity and semantic Patchscope decoding for SAC hidden states."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from confidence_test.source_attribution_schema import (
    ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
    SOURCE_ATTRIBUTION_CLASS_TEXT,
)

from .conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from .layer_stage_analyzer import source_vocab_readout
from .model_adapter import (
    LanguageModules,
    run_logits_forward,
    run_patched_logits_forward,
)
from .source_patchscope_prompts import (
    IDENTITY_PATCHSCOPE_ASSISTANT_PREFILL,
    IDENTITY_PATCHSCOPE_USER_PROMPT,
    SEMANTIC_PATCHSCOPE_USER_PROMPT,
)
from .token_positions import (
    encode_without_special_tokens,
    locate_marker_in_assistant,
    unique_subsequence,
)
from .token_spans import build_rendered_alignment


PATCHSCOPE_ANALYSIS_MODES = ("Identity", "Semantic")


@dataclass
class PreparedSourceTarget:
    name: str
    inputs: Any
    target_position: int
    rendered_prompt: str
    baseline: dict[str, Any]


class SourcePatchscopeDecoder:
    """Prepare content-free targets once and patch SAC states layer by layer."""

    def __init__(
        self,
        *,
        inference: Any,
        modules: LanguageModules,
        class_token_ids: dict[str, Sequence[int]],
        analysis_modes: Sequence[str],
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
        return self._prepare_target(
            mode="Identity",
            user_prompt=IDENTITY_PATCHSCOPE_USER_PROMPT,
            assistant_prefill=IDENTITY_PATCHSCOPE_ASSISTANT_PREFILL,
        )

    def prepare_semantic_target(self) -> PreparedSourceTarget:
        return self._prepare_target(
            mode="Semantic",
            user_prompt=SEMANTIC_PATCHSCOPE_USER_PROMPT.format(
                source_classes=SOURCE_ATTRIBUTION_CLASS_TEXT
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
        )
        result["target_name"] = target.name
        result["target_position"] = target.target_position
        return result, logits
