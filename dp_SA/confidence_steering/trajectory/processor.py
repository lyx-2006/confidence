"""Processor policy shared with the current parent experiment."""
from __future__ import annotations

from typing import Any

from dp_SA.confidence_steering.processor import (
    PROCESSOR_MODE, enforce_fast_image_processor, processor_identity,
)


def enforce_parent_fast_image_processor(processor: Any) -> Any:
    """Use the exact explicit-Fast policy exported by the parent experiment."""
    result = enforce_fast_image_processor(processor)
    identity = processor_identity(result)
    if PROCESSOR_MODE != "explicit_fast" or not identity["is_fast"]:
        raise RuntimeError(f"Parent Fast processor policy failed: {identity}")
    return result
