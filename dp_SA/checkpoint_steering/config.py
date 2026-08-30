from __future__ import annotations

from pathlib import Path

from dp_SA.config import (
    BOOTSTRAP_REPEATS,
    FLOAT_TOLERANCE,
    HIDDEN_DEFINITION,
    INFERENCE_PATH,
    MODEL_PATH,
    ROOT,
    SEED,
    VECTOR_NORM_FRACTION,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_PARENT = PACKAGE_ROOT / "output"
RESULTS_ROOT = OUTPUT_PARENT / "results"
SMOKE_ROOT = OUTPUT_PARENT / "smoke_tmp"

HISTORICAL_ROOT = ROOT / "dp_SA" / "outputs"
HISTORICAL_CAPTURE = HISTORICAL_ROOT / "capture" / "results.jsonl"
HISTORICAL_CAPTURE_CONFIG = HISTORICAL_ROOT / "capture" / "config.json"
HISTORICAL_CONSTRUCTION = HISTORICAL_ROOT / "steering" / "construction_manifest.jsonl"
HISTORICAL_TEST = HISTORICAL_ROOT / "steering" / "test_manifest.jsonl"

POSITIONS = (
    "P1_LAT",
    "P1_PANL",
    "P1_ATTRIBUTION_DEFINITION_END",
    "P1_CLASS_LIST_END",
    "P1_FORMAT_DESCRIPTION_END",
)
LAYERS = (8, 10, 14, 18, 20, 24, 26)
ALPHAS = (-10.0, -2.0, 0.0, 2.0, 10.0)

SMOKE_LAYERS = (8, 14)
SMOKE_ALPHAS = (-2.0, 0.0, 2.0)
SMOKE_BOOTSTRAP_REPEATS = 200
TOKEN_WINDOW_RADIUS = 4

ANCHORS = {
    "P1_ATTRIBUTION_DEFINITION_END": (
        "Source attribution refers to the relative contribution of the text clue and the image "
        "to the formation of the fixed answer. Report whether the fixed answer was based more "
        "on the text clue, more on the image, or on both sources to a similar extent."
    ),
    "P1_CLASS_LIST_END": "8: The answer was based almost entirely on the image.",
    "P1_FORMAT_DESCRIPTION_END": "where CLASS is exactly one integer between 0 and 8.",
}

POSITION_ORDER = (*POSITIONS, "P1_SAC")
POSITION_DEFINITION_VERSION = 1
LOGIT_PARITY_TOLERANCE = FLOAT_TOLERANCE
PROBABILITY_PARITY_TOLERANCE = FLOAT_TOLERANCE
SOFT_SA_PARITY_TOLERANCE = FLOAT_TOLERANCE

FORMAL_CAPTURE_FORWARDS = 150
FORMAL_STEERING_FORWARDS = 100 * len(POSITIONS) * len(LAYERS) * len(ALPHAS)
FORMAL_TOTAL_FORWARDS = FORMAL_CAPTURE_FORWARDS + FORMAL_STEERING_FORWARDS

