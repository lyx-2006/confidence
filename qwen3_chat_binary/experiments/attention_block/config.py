from __future__ import annotations

from pathlib import Path

from qwen3_chat_binary.config import CAPTURE_ROOT, MODEL_PATH, OUTPUT_ROOT

OUTPUT_ROOT = OUTPUT_ROOT / "AttentionBlockFiveway"
VARIANT = "native_boundary"
TEST_PER_SIDE = 50
WINDOWS = ((12, 16), (16, 20), (20, 24), (24, 28))
CONDITIONS = ("SAC_to_PANL", "SAC_to_PANL_plus_1", "SAC_to_CLE", "SAC_to_CLE_plus_1")
PAIRS = {
    "PANL": ("SAC_to_PANL", "SAC_to_PANL_plus_1"),
    "CLE": ("SAC_to_CLE", "SAC_to_CLE_plus_1"),
}
ROW_SUM_TOLERANCE = 0.01
CLEAN_LOGIT_TOLERANCE = 1.0
CLEAN_SCORE_TOLERANCE = 0.01
BOOTSTRAP_REPEATS = 2000
SEED = 42


def default_output(smoke: bool = False) -> Path:
    return OUTPUT_ROOT / ("smoke" if smoke else "formal")

