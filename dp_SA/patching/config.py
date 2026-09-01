from __future__ import annotations

from pathlib import Path

from dp_SA.config import BOOTSTRAP_REPEATS, DATASET_PATH, MIDPOINTS, MODEL_PATH, ROOT, SEED

FORMAT_VERSION = 1
DEFAULT_CAPTURE_DIR = ROOT / "dp_SA" / "outputs" / "capture"
DEFAULT_OUTPUT_PARENT = ROOT / "dp_SA" / "patching" / "outputs"
DEFAULT_POSITIONS = ("P1_PANL", "P1_PANL_PLUS_1")
SUPPORTED_POSITIONS = (*DEFAULT_POSITIONS, "P1_CLASS_LIST_END")
DEFAULT_LAYERS = (12, 14, 16, 18, 20)
DEFAULT_EVAL_CASES = 50
CORRUPTIONS = ("all", "answer_only")
DENOMINATOR_EPSILON = 1e-8
LOGIT_PARITY_TOLERANCE = 0.125
SOFT_PARITY_TOLERANCE = 1e-6
HISTORICAL_LAYERS = (14, 18, 20)
MODEL_CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)


def parse_positions(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    output = tuple(str(value) for value in values)
    if not output or len(set(output)) != len(output):
        raise ValueError("--positions must be non-empty and unique")
    invalid = sorted(set(output) - set(SUPPORTED_POSITIONS))
    if invalid:
        raise ValueError(f"Unsupported delayed patch positions: {invalid}")
    return output


def parse_layers(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    output = tuple(int(value) for value in values)
    if not output or len(set(output)) != len(output):
        raise ValueError("--layers must be non-empty and unique")
    if min(output) < 0:
        raise ValueError("--layers must be zero-based non-negative indices")
    return output


__all__ = [name for name in globals() if name.isupper()]
