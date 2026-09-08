from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

import torch

from confidence_test.runtime_imports import load_runtime

from .config import INFERENCE_PATH, MAX_PIXELS, MIN_PIXELS, MODEL_CONFIG_FILES, MODEL_PATH
from .io_utils import canonical_hash, sha256_file


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def install_fast_processor(inference: Any, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    from transformers import Qwen2VLImageProcessorFast

    processor = inference.processor
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor has no tokenizer")
    tokenizer_id = id(tokenizer)
    chat_template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)
    fast = Qwen2VLImageProcessorFast.from_pretrained(
        model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, local_files_only=True,
    )
    processor.image_processor = fast
    if id(processor.tokenizer) != tokenizer_id:
        raise RuntimeError("Installing the fast image processor changed the tokenizer")
    current_template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)
    if current_template != chat_template:
        raise RuntimeError("Installing the fast image processor changed the chat template")
    if type(fast).__name__ != "Qwen2VLImageProcessorFast" or getattr(fast, "is_fast", None) is not True:
        raise RuntimeError("Explicit Qwen2VLImageProcessorFast installation failed")
    if int(fast.min_pixels) != MIN_PIXELS or int(fast.max_pixels) != MAX_PIXELS:
        raise RuntimeError("Fast processor pixel bounds changed")
    processor_config = {
        "processor": _jsonable(processor.to_dict()),
        "image_processor": _jsonable(fast.to_dict()),
        "video_processor": _jsonable(processor.video_processor.to_dict())
        if getattr(processor, "video_processor", None) is not None else None,
        "tokenizer_init_kwargs": _jsonable(getattr(tokenizer, "init_kwargs", {})),
        "chat_template": current_template,
    }
    return {
        "processor_class": f"{type(processor).__module__}.{type(processor).__name__}",
        "image_processor_class": f"{type(fast).__module__}.{type(fast).__name__}",
        "is_fast": bool(fast.is_fast), "min_pixels": int(fast.min_pixels),
        "max_pixels": int(fast.max_pixels), "chat_template_hash": canonical_hash(current_template),
        "processor_config": processor_config,
        "image_processor_config": _jsonable(fast.to_dict()),
    }


def load_strict_inference(model_path: Path = MODEL_PATH) -> tuple[Any, dict[str, Any]]:
    runtime = load_runtime(INFERENCE_PATH)
    inference = runtime.QwenVLInference(str(model_path), min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    if getattr(inference, "dtype", None) != torch.bfloat16:
        raise RuntimeError(f"BF16 is mandatory; loader returned {getattr(inference, 'dtype', None)}")
    implementation = getattr(inference.model.config, "_attn_implementation", None)
    if implementation != "eager":
        raise RuntimeError(f"Eager attention is mandatory; got {implementation!r}")
    processor = install_fast_processor(inference, model_path)
    import transformers
    try:
        qwen_version = importlib.metadata.version("qwen-vl-utils")
    except importlib.metadata.PackageNotFoundError:
        qwen_version = "unavailable"
    identity = {
        **processor, "model_path": str(model_path.resolve()), "model_dtype": "bfloat16",
        "attention_implementation": implementation, "transformers_version": transformers.__version__,
        "qwen_vl_utils_version": qwen_version,
        "config_sha256": {name: sha256_file(model_path / name) for name in MODEL_CONFIG_FILES},
    }
    identity["fingerprint"] = canonical_hash(identity)
    return inference, identity


__all__ = ["install_fast_processor", "load_strict_inference"]
