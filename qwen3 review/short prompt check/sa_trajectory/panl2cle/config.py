from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SHORT_ROOT = PACKAGE_ROOT.parents[1]
REVIEW_ROOT = SHORT_ROOT.parent
REPOSITORY_ROOT = REVIEW_ROOT.parent

CAPTURE_ROOT = SHORT_ROOT / "output" / "capture"
STEERING_ROOT = SHORT_ROOT / "output" / "steering"
MODEL_PATH = REPOSITORY_ROOT / "qwen-3-vl" / "model"
OUTPUT_ROOT = SHORT_ROOT / "output" / "sa_trajectory" / "PANL2CLE_four_cell"
SMOKE_ROOT = OUTPUT_ROOT / "smoke"

PANL_LAYERS = (14, 16, 18)
CLE_LAYERS = (15, 17, 19)
PAIRS = ((14, 15), (16, 17), (18, 19))
ALPHAS = (-5.0, 5.0)
RIDGE_ALPHAS = tuple(10.0**x for x in range(-4, 5))
GROUPS = ("answer_equal_macro", "overall_micro", "image_side", "text_side")
SEED = 42
BOOTSTRAP_REPEATS = 2000
VECTOR_NORM_FRACTION = 0.03
HIDDEN_SIZE = 4096
NUM_LAYERS = 36
EXPECTED_CAPTURE_CASES = 500
EXPECTED_TEST_CASES = 80
EXPECTED_TEST_SIDES = {"image_side": 74, "text_side": 6}
EXPECTED_PROBE_ELIGIBLE = (322, 98)
EXPECTED_PROBE_CONSTRUCTION = (267, 78)
EXPECTED_PROBE_AUDIT = (55, 20)
LOGIT_PARITY_ATOL = 1e-6
RAW_EXPRESSION_ATOL = 1e-4
