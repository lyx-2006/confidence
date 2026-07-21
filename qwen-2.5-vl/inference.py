#!/usr/bin/env python3
"""
Qwen2.5-VL-7B-Instruct native Transformers inference.

Supports:
  - Pure text inference
  - Single-image inference
  - Hidden states / attention extraction for research (forward_analysis)

Usage:
  # Text only
  python inference.py --prompt "介绍一下你自己" --max-new-tokens 128

  # Image + text
  python inference.py --image /path/to/image.jpg --prompt "描述这张图片" --max-new-tokens 128

  # Save output
  python inference.py --image test.jpg --prompt "图中有什么？" --output outputs/result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Silence the temperature warning when do_sample=False — we intentionally omit
# temperature/top_p/top_k since they are irrelevant for greedy decoding.
warnings.filterwarnings(
    "ignore",
    message=".*temperature.*do_sample.*",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    model_path: str
    image_path: str | None
    prompt: str
    response: str
    generation_config: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core inference class
# ---------------------------------------------------------------------------

class QwenVLInference:
    """Load a Qwen2.5-VL model once and run repeated inference calls.

    Parameters
    ----------
    model_path : str
        Path to the local model checkpoint directory.
    min_pixels : int
        Minimum pixels for vision preprocessing (default: 256*28*28).
    max_pixels : int
        Maximum pixels for vision preprocessing (default: 1280*28*28).
    """

    def __init__(
        self,
        model_path: str,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # -- Load processor --------------------------------------------------
        print(f"[INFO] Loading processor from {model_path} ...")
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            local_files_only=True,
        )
        print("[INFO] Processor loaded.")

        # -- Load model ------------------------------------------------------
        print(f"[INFO] Loading model from {model_path} ...")
        dtype, dtype_name, load_error = self._load_model()
        if load_error:
            print(f"[ERROR] Model load failed: {load_error}")
            raise RuntimeError(load_error)

        self.dtype = dtype
        self.dtype_name = dtype_name
        print(f"[INFO] Model loaded. dtype={self.dtype_name}, device_map=auto")

        # Report device placement
        self._report_device_map()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> tuple[torch.dtype, str, str | None]:
        from transformers import Qwen2_5_VLForConditionalGeneration

        # Try BF16 first
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
                local_files_only=True,
            )
            self.model = model.eval()
            return torch.bfloat16, "bfloat16", None
        except Exception as e_bf16:
            bf16_error = str(e_bf16)
            print(f"[WARN] BF16 load failed: {bf16_error}")

        # Fallback to FP16
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="eager",
                local_files_only=True,
            )
            self.model = model.eval()
            return torch.float16, "float16", None
        except Exception as e_fp16:
            return (
                torch.float32,
                "float32 (fallback)",
                f"BF16: {bf16_error}; FP16: {str(e_fp16)}",
            )

    def _report_device_map(self) -> None:
        """Print a compact summary of where model parameters live."""
        if hasattr(self.model, "hf_device_map"):
            dm = self.model.hf_device_map
            print(f"[INFO] Model device_map: {dm}")
        else:
            try:
                device = next(self.model.parameters()).device
                print(f"[INFO] All model parameters on: {device}")
            except StopIteration:
                print("[WARN] Could not determine model device.")

    def _get_inputs_device(self) -> torch.device:
        """Return the device of the model's input embeddings.

        When using device_map='auto', the first (embedding) layer may be on a
        different device than other layers.  We place input tensors there.
        """
        # model.model.embed_tokens is the standard for Qwen2.5-VL
        try:
            embed = self.model.model.embed_tokens
            return next(embed.parameters()).device
        except (AttributeError, StopIteration):
            pass
        # Fallback: use the first parameter's device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        image_path: str | None = None,
        max_new_tokens: int = 128,
    ) -> GenerationResult:
        """Run text or image+text inference.

        Parameters
        ----------
        prompt : str
            The text prompt.
        image_path : str or None
            Path to an image file, or None for text-only inference.
        max_new_tokens : int
            Maximum number of tokens to generate.

        Returns
        -------
        GenerationResult
            Contains response text, timing, and GPU memory info.
        """
        t_start = time.perf_counter()

        # Build messages in Qwen2.5-VL official format
        if image_path:
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            # Validate image readability
            from PIL import Image

            try:
                img = Image.open(image_path_obj)
                img.verify()
            except Exception as e:
                raise ValueError(f"Image cannot be read: {image_path} — {e}")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path_obj.resolve())},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        # Apply chat template
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, _video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        # Place tensors on the embedding device (safe for device_map="auto")
        embed_device = self._get_inputs_device()
        inputs = inputs.to(embed_device)

        # Free unused GPU memory before generation (only if OOM-prone)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # Trim input tokens — only decode the newly generated portion
        input_ids_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = [
            output_ids[input_ids_len:]
            for output_ids in generated_ids
        ]

        response_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        elapsed = time.perf_counter() - t_start

        peak_gpu = 0.0
        if torch.cuda.is_available():
            peak_gpu = torch.cuda.max_memory_allocated() / (1024**3)

        return GenerationResult(
            model_path=self.model_path,
            image_path=image_path,
            prompt=prompt,
            response=response_text,
            generation_config={
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
            },
            runtime={
                "device": str(embed_device),
                "dtype": self.dtype_name,
                "elapsed_seconds": round(elapsed, 3),
                "peak_gpu_memory_gb": round(peak_gpu, 3),
            },
        )

    def forward_analysis(
        self,
        prompt: str,
        image_path: str | None = None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
    ) -> dict[str, Any]:
        """Run a single forward pass and return logits/hidden_states/attentions.

        This is intended for research use (PANL, confidence probes, etc.).
        By default only hidden_states are collected; attentions are opt-in
        because of the large memory footprint.

        Parameters
        ----------
        prompt : str
            The text prompt.
        image_path : str or None
            Path to an image file, or None for text-only.
        output_hidden_states : bool
            If True (default), collect hidden states from each layer.
        output_attentions : bool
            If True, collect attention weights.  Requires eager attention
            and can consume significant GPU memory.  Default: False.

        Returns
        -------
        dict
            Contains keys: logits, hidden_states, attentions, shapes, metadata.
            The raw tensors are on the model's device — caller is responsible
            for moving/copying them.  `shapes` provides lightweight metadata
            for inspection.
        """
        if output_attentions and self.model.config._attn_implementation != "eager":
            raise RuntimeError(
                "output_attentions=True requires eager attention, "
                f"but model uses {self.model.config._attn_implementation}."
            )

        from qwen_vl_utils import process_vision_info

        if image_path:
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path_obj.resolve())},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, _video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        embed_device = self._get_inputs_device()
        inputs = inputs.to(embed_device)

        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )

        # Build lightweight shape metadata
        shapes: dict[str, Any] = {}
        if hasattr(outputs, "logits") and outputs.logits is not None:
            shapes["logits"] = list(outputs.logits.shape)

        if output_hidden_states and outputs.hidden_states is not None:
            shapes["hidden_states"] = {
                "num_layers": len(outputs.hidden_states),
                "per_layer_shape": list(outputs.hidden_states[0].shape),
            }

        if output_attentions and outputs.attentions is not None:
            shapes["attentions"] = {
                "num_layers": len(outputs.attentions),
                "per_layer_shape": list(outputs.attentions[0].shape),
            }

        return {
            "logits": getattr(outputs, "logits", None),
            "hidden_states": getattr(outputs, "hidden_states", None),
            "attentions": getattr(outputs, "attentions", None),
            "shapes": shapes,
            "metadata": {
                "dtype": self.dtype_name,
                "device": str(embed_device),
                "output_hidden_states": output_hidden_states,
                "output_attentions": output_attentions,
            },
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL-7B-Instruct native Transformers inference",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct",
        help="Path to local model checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请简单介绍一下你自己。",
        help="Text prompt for the model",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file for vision-language inference",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the result as JSON (optional)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=256 * 28 * 28,
        help="Minimum pixels for vision preprocessing",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1280 * 28 * 28,
        help="Maximum pixels for vision preprocessing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not Path(args.model_path).exists():
        print(f"[ERROR] Model path does not exist: {args.model_path}")
        return 1

    # Load model once
    try:
        inference = QwenVLInference(
            model_path=args.model_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize inference: {e}")
        return 1

    # Run inference
    try:
        result = inference.generate(
            prompt=args.prompt,
            image_path=args.image,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")

        # Provide OOM-specific guidance
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
                print(f"  GPU memory reserved:  {torch.cuda.memory_reserved() / 1e9:.1f} GB")
            if args.image:
                from PIL import Image

                try:
                    img = Image.open(args.image)
                    print(f"  Image size: {img.size}")
                except Exception:
                    pass
            print(f"  max_pixels: {args.max_pixels}")
            print(f"  max_new_tokens: {args.max_new_tokens}")
            print("  Suggestion: try --max-pixels 256x28x28 or --max-new-tokens 64")
        return 1

    # Print result
    print()
    print("=" * 60)
    print(f"PROMPT: {result.prompt}")
    if result.image_path:
        print(f"IMAGE:  {result.image_path}")
    print(f"RESPONSE:\n{result.response}")
    print("=" * 60)
    print(f"Time: {result.runtime['elapsed_seconds']:.2f}s")
    print(f"Peak GPU: {result.runtime['peak_gpu_memory_gb']:.2f} GB")
    print(f"Dtype: {result.runtime['dtype']}")
    print(f"Device: {result.runtime['device']}")
    print("=" * 60)

    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_path": result.model_path,
            "image_path": result.image_path,
            "prompt": result.prompt,
            "response": result.response,
            "generation_config": result.generation_config,
            "runtime": result.runtime,
        }
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] Result saved to {output_path}")

    return 0



if __name__ == "__main__":
    sys.exit(main())
