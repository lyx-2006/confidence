from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import INFERENCE_PATH


_MODULE_CACHE: dict[Path, ModuleType] = {}


def load_interface(path: str | Path = INFERENCE_PATH) -> ModuleType:
    source = Path(path).resolve()
    if source in _MODULE_CACHE:
        return _MODULE_CACHE[source]
    if not source.is_file():
        raise FileNotFoundError(f"Qwen3 inference source does not exist: {source}")
    name = f"qwen3_chat_binary_interface_{abs(hash(str(source)))}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Qwen3 inference source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "Qwen3VLInference"):
        raise AttributeError(f"{source} does not define Qwen3VLInference")
    _MODULE_CACHE[source] = module
    return module


def load_qwen3_inference(model_path: str | Path, *, attn_implementation: str = "sdpa") -> Any:
    return load_interface().Qwen3VLInference(
        str(Path(model_path).resolve()), attn_implementation=attn_implementation, dtype="auto"
    )

