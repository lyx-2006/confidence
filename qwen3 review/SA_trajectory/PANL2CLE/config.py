from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REVIEW_ROOT = PACKAGE_ROOT.parents[1]
REPOSITORY_ROOT = REVIEW_ROOT.parent

CAPTURE_ROOT = REVIEW_ROOT / "capture"
STEERING_ROOT = REVIEW_ROOT / "Steering" / "output" / "panl_lat_cle_asym31"
MODEL_PATH = REPOSITORY_ROOT / "qwen-3-vl" / "model"
OUTPUT_ROOT = PACKAGE_ROOT / "output" / "results"
SMOKE_ROOT = PACKAGE_ROOT / "output" / "smoke"

PANL_LAYERS = (14, 16, 18)
CLE_LAYERS = (15, 17, 19, 21)
CAUSAL_PAIRS = tuple((panl, cle) for panl in PANL_LAYERS for cle in CLE_LAYERS if cle > panl)
ALPHAS = (-5.0, 5.0)
RIDGE_ALPHAS = tuple(10.0**exponent for exponent in range(-4, 5))
SEED = 42
BOOTSTRAP_REPEATS = 2000
VECTOR_NORM_FRACTION = 0.03
HIDDEN_SIZE = 4096
NUM_LAYERS = 36
HIDDEN_DEFINITION = "decoder_block_output_pre_final_norm"
EXPECTED_CAPTURE_CASES = 500
EXPECTED_TEST_CASES = 81
EXPECTED_TEST_SIDES = {"image_side": 50, "text_side": 31}
EXPECTED_PROBE_ELIGIBLE = (321, 97)
EXPECTED_CONSTRUCTION = (252, 77)
EXPECTED_AUDIT = (69, 20)
LOGIT_PARITY_ATOL = 1e-6
RAW_EXPRESSION_ATOL = 1e-4

