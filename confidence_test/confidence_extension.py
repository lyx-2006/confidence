"""Single-image support that delegates confidence math to the base analyzer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from layer_metacognition.conversation_builder import (
    prepare_multimodal_inputs,
    render_continued_assistant,
)

from confidence_test.runtime_imports import (
    ASSISTANT_CONFIDENCE_PREFILL,
    ConfidenceAnalyzer,
    ConfidenceResult,
)


class _PromptAnalyzerAdapter:
    def __init__(
        self,
        base_analyzer: Any,
        inference: Any,
        prompt: str,
        image_path: str | None,
    ):
        self._base = base_analyzer
        self._inference = inference
        self._prompt = prompt
        self._image_path = image_path
        self._messages: list[dict[str, Any]] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def build_prompt(self, _question: str, _text_clue: str, _answer: str) -> tuple[list[dict[str, Any]], str]:
        prefill = ASSISTANT_CONFIDENCE_PREFILL
        if self._image_path is None:
            user_content = [{"type": "text", "text": self._prompt}]
        else:
            resolved = Path(self._image_path).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Image not found: {self._image_path}")
            user_content = [
                {"type": "image", "image": str(resolved)},
                {"type": "text", "text": self._prompt},
            ]
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": prefill}]},
        ]
        rendered = render_continued_assistant(self._base.processor, messages, prefill)
        self._messages = messages
        return messages, rendered

    def _prepare_inputs(self, rendered_prompt: str) -> Any:
        if self._messages is None:
            raise RuntimeError("Confidence messages were not built before input preparation")
        return prepare_multimodal_inputs(
            self._base.processor,
            self._messages,
            rendered_prompt,
            device=self._inference._get_inputs_device(),
        )


class MultimodalConfidenceAnalyzer:
    def __init__(self, base_analyzer: ConfidenceAnalyzer, inference: Any):
        if base_analyzer.model is not inference.model or base_analyzer.processor is not inference.processor:
            raise ValueError("base_analyzer and inference must share model and processor instances")
        self.base_analyzer = base_analyzer
        self.inference = inference

    def analyze_prompt(
        self,
        prompt: str,
        image_path: str | None = None,
    ) -> ConfidenceResult:
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")
        adapter = _PromptAnalyzerAdapter(
            self.base_analyzer,
            self.inference,
            prompt,
            image_path,
        )
        # Call the original method body with an adapter that only replaces
        # prompt construction and input preparation. All logits math, parsing,
        # token variants, and fallback behavior remain the original code.
        return type(self.base_analyzer).analyze(adapter, "prompt", "prompt", "answer")
