from __future__ import annotations

from typing import Any

from transformers import AutoProcessor, Qwen2VLImageProcessor

from .config import MODEL_PATH

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


def load_frozen_processor() -> Any:
    """Load the fast tokenizer paired with the capture-era slow image processor."""
    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH), local_files_only=True, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    processor.image_processor = Qwen2VLImageProcessor.from_pretrained(
        str(MODEL_PATH), local_files_only=True, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    return processor


def enforce_frozen_image_processor(processor: Any) -> Any:
    processor.image_processor = Qwen2VLImageProcessor.from_pretrained(
        str(MODEL_PATH), local_files_only=True, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    return processor

