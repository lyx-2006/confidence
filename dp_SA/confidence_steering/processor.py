from __future__ import annotations

from typing import Any

from transformers import AutoProcessor, Qwen2VLImageProcessorFast

from .config import MODEL_PATH

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
PROCESSOR_MODE = "explicit_fast"


def load_fast_processor() -> Any:
    """Load the processor with an explicitly selected Fast image processor."""
    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH), local_files_only=True, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    return enforce_fast_image_processor(processor)


def enforce_fast_image_processor(processor: Any) -> Any:
    """Explicitly select the fast image processor; never rely on library defaults."""
    processor.image_processor = Qwen2VLImageProcessorFast.from_pretrained(
        str(MODEL_PATH), local_files_only=True, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    if not bool(getattr(processor.image_processor, "is_fast", False)):
        raise RuntimeError("Explicit Fast image processor selection failed")
    return processor


def processor_identity(processor: Any) -> dict[str, Any]:
    image = processor.image_processor
    return {
        "processor_class": f"{type(processor).__module__}.{type(processor).__name__}",
        "image_processor_class": f"{type(image).__module__}.{type(image).__name__}",
        "is_fast": bool(getattr(image, "is_fast", False)),
        "min_pixels": int(getattr(image, "min_pixels")),
        "max_pixels": int(getattr(image, "max_pixels")),
    }
