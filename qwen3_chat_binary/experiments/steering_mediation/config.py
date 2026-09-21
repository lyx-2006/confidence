from __future__ import annotations

from pathlib import Path

from qwen3_chat_binary.config import CAPTURE_ROOT, MODEL_PATH, OUTPUT_ROOT

OUTPUT_ROOT = OUTPUT_ROOT / "SteeringMediationFiveway"
VARIANT = "native_boundary"
CHAINS = {"LAT_to_PANL": ("LAT", "PANL"), "PANL_to_CLE": ("PANL", "CLE")}
PAIRS = ((14, 15), (16, 17), (18, 19))
ALPHAS = (-5.0, 5.0)
CONDITIONS = ("C0", "C1", "C2", "C3")
CONSTRUCTION_PER_SIDE = 25
TEST_PER_SIDE = 50
VECTOR_NORM_FRACTION = 0.03
LOGIT_PARITY_TOLERANCE = 1e-5
BOOTSTRAP_REPEATS = 2000
SEED = 42


def default_output(smoke: bool = False) -> Path:
    return OUTPUT_ROOT / ("smoke" if smoke else "formal")

