from __future__ import annotations

from pathlib import Path

from dp_SA.config import DATASET_PATH, INFERENCE_PATH, MODEL_PATH, OUTPUT_ROOT, ROOT, SEED

SOURCE_ROOT = OUTPUT_ROOT
DEFAULT_OUTPUT_PARENT = ROOT / "dp_SA" / "activation_swap" / "outputs"
DEFAULT_POSITIONS = ("P1_PANL", "P1_PANL_PLUS_1", "P1_SAC")
DEFAULT_LAYERS = (12, 14, 16, 18, 22, 26)
SMOKE_LAYERS = (14,)
BOOTSTRAP_REPEATS = 2000
SMOKE_BOOTSTRAP_REPEATS = 100
RECIPIENTS_PER_SIDE = 50
SMOKE_RECIPIENTS_PER_SIDE = 2
CONSTRUCTION_PER_SIDE = 25
POSITION_NAMES = {"P1_PANL": "PANL", "P1_PANL_PLUS_1": "PANL_PLUS_1", "P1_SAC": "SAC"}
HIDDEN_DEFINITION = "decoder_block_output_post_mlp_residual"
FORMAL_FORWARD_COUNT = 3600
HISTORY_LAYERS = (14, 18, 26)
CLASS_COUNT = 9
MIDPOINTS_K8 = tuple(index / 8 for index in range(CLASS_COUNT))
OLD_MIDPOINTS = (0.05, 0.175, 0.325, 0.4375, 0.5, 0.5625, 0.675, 0.825, 0.95)
MODEL_CONFIG_FILES = (
    "config.json", "generation_config.json", "preprocessor_config.json",
    "processor_config.json", "chat_template.json", "tokenizer.json",
    "tokenizer_config.json", "model.safetensors.index.json",
)


def parse_csv_strings(value: str, *, allowed: set[str] | None = None) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("comma-separated values must be non-empty and unique")
    if allowed is not None:
        invalid = sorted(set(values) - allowed)
        if invalid:
            raise ValueError(f"unsupported values: {invalid}")
    return values


def parse_layers(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not values or len(set(values)) != len(values) or min(values) < 0:
        raise ValueError("layers must be unique, non-negative comma-separated integers")
    return values


__all__ = [name for name in globals() if name.isupper()]
