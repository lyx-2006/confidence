from __future__ import annotations

from pathlib import Path

from dp_SA.config import DATASET_PATH, INFERENCE_PATH, MODEL_PATH, ROOT, SPLIT_PATH

SOURCE_ROOT = ROOT / "dp_SA" / "outputs"
OUTPUT_PARENT = ROOT / "dp_SA" / "answer_force" / "outputs"
PRIMARY_POSITION = "P1_PANL"
PRIMARY_LAYER = 14
CONDITIONS = ("clean", "force_opposite", "force_unrelated")
ORIGINS = ("text", "image")
DIFFICULTIES = ("easy", "hard")
CELLS = tuple((origin, difficulty) for origin in ORIGINS for difficulty in DIFFICULTIES)
RECIPIENTS_PER_CELL = 25
SMOKE_PER_ORIGIN = 2
BOOTSTRAP_REPEATS = 2000
SEED = 42
FLOAT_TOLERANCE = 1e-6
LOGIT_TOLERANCE = 0.125
DELTA_E76_THRESHOLD = 40.0
MIDPOINTS = (0.05, 0.175, 0.325, 0.4375, 0.5, 0.5625, 0.675, 0.825, 0.95)

# These are the canonical color names used by datasets/datasets.json.  They
# are deliberately local constants so the unrelated-answer rule has no
# dependency on a color package or a display profile.
COLOR_HEX = {
    "red": "FF0000",
    "orange": "FFA500",
    "yellow": "FFFF00",
    "green": "008000",
    "blue": "0000FF",
    "cyan": "00FFFF",
    "purple": "800080",
    "pink": "FFC0CB",
    "brown": "A52A2A",
    "white": "FFFFFF",
    "black": "000000",
    "gray": "808080",
}


def default_run_name(prefix: str = "answer_force_seed42") -> str:
    import time

    return f"{prefix}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


__all__ = [name for name in globals() if name.isupper()] + ["default_run_name"]
