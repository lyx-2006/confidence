#!/usr/bin/env python3
"""Stage-2 confidence analysis for the local Qwen2.5-VL model.

The module is deliberately independent from the colour-pool orchestration.  A
caller should pass an already loaded ``QwenVLInference`` object to
``ConfidenceAnalyzer`` so the model, processor, and tokenizer are not loaded a
second time.  The CLI is useful for one-off diagnostics and loads the model
once for that process.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


CONFIDENCE_CLASSES = [
    "No chance(0.0-0.1)",
    "Really unlikely(0.1-0.2)",
    "Chances are slight(0.2-0.3)",
    "Unlikely(0.3-0.4)",
    "Less than even(0.4-0.5)",
    "Better than even(0.5-0.6)",
    "Likely(0.6-0.7)",
    "Very good chance(0.7-0.8)",
    "Highly likely(0.8-0.9)",
    "Almost certain(0.9-1.0)",
]

CLASS_MIDPOINTS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

CONFIDENCE_CLASS_TEXT = """Confidence classes:
- No chance(0.0-0.1)
- Really unlikely(0.1-0.2)
- Chances are slight(0.2-0.3)
- Unlikely(0.3-0.4)
- Less than even(0.4-0.5)
- Better than even(0.5-0.6)
- Likely(0.6-0.7)
- Very good chance(0.7-0.8)
- Highly likely(0.8-0.9)
- Almost certain(0.9-1.0)"""

STAGE2_TEXT_CONFIDENCE_PROMPT = """You will be shown a question together with an answer you previously provided before.

Question:
{question}

Text clue:
{text_clue}

**Answer**: {answer}

Classify your confidence in the correctness of the previous answer into one of
the following classes.

Confidence refers to the probability that the previous answer is correct
given only the question and text clue.

The previous answer is fixed. Do not change the answer.

The fact that an answer was selected does not mean the confidence must be high.
The answer may be the most likely choice while still having low confidence
when multiple alternatives remain plausible.

{classes}

Output exactly:

**Confidence**: <CLASS>

CLASS must be exactly one of the confidence class names listed above.
Do not include reasoning, explanation, or any additional text."""

ASSISTANT_CONFIDENCE_PREFILL = "**confidence**:"


@dataclass
class ConfidenceResult:
    confidence_label: str
    hard_confidence_midpoint: float
    soft_confidence: float
    class_logits: dict[str, float]
    class_probabilities: dict[str, float]
    class_token_variants: dict[str, list[int]]
    raw_output: str
    rendered_prompt: str
    hard_label_parsed: bool
    hidden_state_collected: bool = False


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class ConfidenceAnalyzer:
    """Compute hard and soft confidence without collecting hidden states."""

    def __init__(self, inference: Any, max_new_tokens: int = 12):
        self.inference = inference
        self.model = inference.model
        self.processor = inference.processor
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.max_new_tokens = max_new_tokens
        self._class_token_variants = self._build_class_token_variants()

    def _encode_without_special_tokens(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.tolist()
        return [int(token_id) for token_id in encoded]

    def _build_class_token_variants(self) -> dict[str, list[int]]:
        """Collect unique first-token variants used after the assistant prefill.

        Qwen tokenization differs for a phrase at a word boundary.  We retain
        raw and leading-space forms, including multi-token
        labels, then score the first token of every usable variant.
        """
        variants: dict[str, list[int]] = {}
        for label in CONFIDENCE_CLASSES:
            first_ids: list[int] = []
            for text in (label, f" {label}"):
                token_ids = self._encode_without_special_tokens(text)
                if token_ids and token_ids[0] not in first_ids:
                    first_ids.append(token_ids[0])
            if not first_ids:
                raise RuntimeError(f"Tokenizer produced no tokens for confidence class: {label}")
            variants[label] = first_ids
        return variants

    @staticmethod
    def _message_content(text: str) -> list[dict[str, str]]:
        return [{"type": "text", "text": text}]

    def build_prompt(self, question: str, text_clue: str, answer: str) -> tuple[list[dict[str, Any]], str]:
        stage2_prompt = STAGE2_TEXT_CONFIDENCE_PROMPT.format(
            question=question,
            text_clue=text_clue,
            answer=answer,
            classes=CONFIDENCE_CLASS_TEXT,
        )
        messages = [
            {"role": "user", "content": self._message_content(stage2_prompt)},
            {"role": "assistant", "content": self._message_content(ASSISTANT_CONFIDENCE_PREFILL)},
        ]

        rendered: str | None = None
        try:
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        except (TypeError, ValueError):
            rendered = None

        if not rendered or not rendered.endswith(ASSISTANT_CONFIDENCE_PREFILL):
            # Compatibility path: render the user turn plus the assistant-start
            # marker, then append the exact assistant prefill ourselves.  This
            # avoids an EOS/im_end token after the prefill.
            user_messages = messages[:1]
            rendered = self.processor.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            rendered = rendered + ASSISTANT_CONFIDENCE_PREFILL

        if not rendered.endswith(ASSISTANT_CONFIDENCE_PREFILL):
            raise RuntimeError("Stage-2 rendered prompt does not end with the exact confidence prefill")
        return messages, rendered

    def _prepare_inputs(self, rendered_prompt: str) -> Any:
        inputs = self.processor(
            text=[rendered_prompt],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.inference._get_inputs_device())

    @staticmethod
    def _parse_hard_label(raw_output: str) -> str | None:
        cleaned = raw_output.strip()
        cleaned = re.sub(r"^\*\*confidence\*\*\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        for label in sorted(CONFIDENCE_CLASSES, key=len, reverse=True):
            if re.match(re.escape(label) + r"(?:\b|$)", cleaned, flags=re.IGNORECASE):
                return label
        return None

    def analyze(self, question: str, text_clue: str, answer: str) -> ConfidenceResult:
        if not question.strip() or not text_clue.strip() or not answer.strip():
            raise ValueError("question, text_clue, and answer must all be non-empty")

        _messages, rendered = self.build_prompt(question, text_clue, answer)
        inputs = self._prepare_inputs(rendered)

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)

        if not generated.scores:
            raise RuntimeError("Model generation returned no logits for the first confidence token")
        first_token_logits = generated.scores[0][0].float()

        class_logits_list: list[torch.Tensor] = []
        for label in CONFIDENCE_CLASSES:
            ids = self._class_token_variants[label]
            id_tensor = torch.tensor(ids, dtype=torch.long, device=first_token_logits.device)
            # Use the strongest tokenizer spelling for the class.  logsumexp
            # would reward classes merely for having more token variants.
            class_logits_list.append(torch.max(first_token_logits.index_select(0, id_tensor)))
        class_logits_tensor = torch.stack(class_logits_list)
        class_probs_tensor = torch.softmax(class_logits_tensor, dim=-1)
        midpoint_tensor = torch.tensor(
            CLASS_MIDPOINTS,
            dtype=class_probs_tensor.dtype,
            device=class_probs_tensor.device,
        )
        soft_confidence = torch.sum(class_probs_tensor * midpoint_tensor).item()

        input_length = int(inputs.input_ids.shape[1])
        generated_tokens = generated.sequences[0, input_length:]
        raw_output = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        parsed_label = self._parse_hard_label(raw_output)
        hard_label_parsed = parsed_label is not None
        if parsed_label is None:
            parsed_label = CONFIDENCE_CLASSES[int(torch.argmax(class_logits_tensor).item())]
        hard_index = CONFIDENCE_CLASSES.index(parsed_label)

        return ConfidenceResult(
            confidence_label=parsed_label,
            hard_confidence_midpoint=CLASS_MIDPOINTS[hard_index],
            soft_confidence=float(soft_confidence),
            class_logits={
                label: float(class_logits_tensor[index].item())
                for index, label in enumerate(CONFIDENCE_CLASSES)
            },
            class_probabilities={
                label: float(class_probs_tensor[index].item())
                for index, label in enumerate(CONFIDENCE_CLASSES)
            },
            class_token_variants={label: list(ids) for label, ids in self._class_token_variants.items()},
            raw_output=raw_output,
            rendered_prompt=rendered,
            hard_label_parsed=hard_label_parsed,
        )


def _load_inference_class(inference_path: Path) -> type[Any]:
    specification = importlib.util.spec_from_file_location("qwen_vl_inference", inference_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load inference module from {inference_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module.QwenVLInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Qwen Stage-2 confidence for a fixed answer")
    parser.add_argument("--question", required=True)
    parser.add_argument("--text-clue", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument(
        "--model-path",
        default="qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--inference-path",
        default="qwen-2.5-vl/inference.py",
    )
    parser.add_argument("--output")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        inference_class = _load_inference_class(Path(args.inference_path).resolve())
        inference = inference_class(model_path=str(Path(args.model_path).resolve()))
        analyzer = ConfidenceAnalyzer(inference, max_new_tokens=args.max_new_tokens)
        result = asdict(analyzer.analyze(args.question, args.text_clue, args.answer))
    except Exception as exc:
        print(f"[ERROR] Confidence analysis failed: {exc}", file=sys.stderr)
        return 1

    if args.output:
        _atomic_write_json(Path(args.output), result)
        print(f"[INFO] Result saved to {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
