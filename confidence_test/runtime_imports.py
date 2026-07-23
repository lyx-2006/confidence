"""Dynamic imports for the repository's original inference/confidence code."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INFERENCE_PATH = REPOSITORY_ROOT / "qwen-2.5-vl" / "inference.py"
CONFIDENCE_ANALYSIS_PATH = REPOSITORY_ROOT / "confidence_analysis.py"

_MODULE_CACHE: dict[Path, ModuleType] = {}


def _load_module(path: str | Path, prefix: str) -> ModuleType:
    resolved = Path(path).resolve()
    if resolved in _MODULE_CACHE:
        return _MODULE_CACHE[resolved]
    if not resolved.is_file():
        raise FileNotFoundError(f"Python source does not exist: {resolved}")
    module_name = f"confidence_test_{prefix}_{abs(hash(str(resolved)))}"
    specification = importlib.util.spec_from_file_location(module_name, resolved)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import Python source: {resolved}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    _MODULE_CACHE[resolved] = module
    return module


@dataclass(frozen=True)
class RuntimeImports:
    QwenVLInference: type[Any]
    GenerationResult: type[Any]
    ConfidenceAnalyzer: type[Any]
    ConfidenceResult: type[Any]
    CONFIDENCE_CLASSES: list[str]
    CLASS_MIDPOINTS: list[float]
    CONFIDENCE_CLASS_TEXT: str
    ASSISTANT_CONFIDENCE_PREFILL: str
    inference_path: Path
    confidence_path: Path


def load_runtime(inference_path: str | Path = DEFAULT_INFERENCE_PATH) -> RuntimeImports:
    """Load original classes without constructing a model or processor."""
    inference_source = Path(inference_path).resolve()
    inference_module = _load_module(inference_source, "qwen_inference")
    confidence_module = _load_module(CONFIDENCE_ANALYSIS_PATH, "confidence_analysis")
    return RuntimeImports(
        QwenVLInference=inference_module.QwenVLInference,
        GenerationResult=inference_module.GenerationResult,
        ConfidenceAnalyzer=confidence_module.ConfidenceAnalyzer,
        ConfidenceResult=confidence_module.ConfidenceResult,
        CONFIDENCE_CLASSES=list(confidence_module.CONFIDENCE_CLASSES),
        CLASS_MIDPOINTS=list(confidence_module.CLASS_MIDPOINTS),
        CONFIDENCE_CLASS_TEXT=str(confidence_module.CONFIDENCE_CLASS_TEXT),
        ASSISTANT_CONFIDENCE_PREFILL=str(confidence_module.ASSISTANT_CONFIDENCE_PREFILL),
        inference_path=inference_source,
        confidence_path=CONFIDENCE_ANALYSIS_PATH.resolve(),
    )


_DEFAULT_RUNTIME = load_runtime()
QwenVLInference = _DEFAULT_RUNTIME.QwenVLInference
GenerationResult = _DEFAULT_RUNTIME.GenerationResult
ConfidenceAnalyzer = _DEFAULT_RUNTIME.ConfidenceAnalyzer
ConfidenceResult = _DEFAULT_RUNTIME.ConfidenceResult
CONFIDENCE_CLASSES = _DEFAULT_RUNTIME.CONFIDENCE_CLASSES
CLASS_MIDPOINTS = _DEFAULT_RUNTIME.CLASS_MIDPOINTS
CONFIDENCE_CLASS_TEXT = _DEFAULT_RUNTIME.CONFIDENCE_CLASS_TEXT
ASSISTANT_CONFIDENCE_PREFILL = _DEFAULT_RUNTIME.ASSISTANT_CONFIDENCE_PREFILL
