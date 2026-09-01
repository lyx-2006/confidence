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
LEGACY_RESULTS_ROOT = OUTPUT_PARENT / "results"
RESULTS_ROOT = OUTPUT_PARENT / "lat_panl_comparison"
SMOKE_ROOT = OUTPUT_PARENT / "lat_panl_comparison_smoke"

HISTORICAL_CAPTURE = ROOT / "dp_SA" / "outputs" / "capture" / "results.jsonl"
HISTORICAL_CAPTURE_CONFIG = ROOT / "dp_SA" / "outputs" / "capture" / "config.json"
HISTORICAL_CONSTRUCTION = ROOT / "dp_SA" / "outputs" / "steering" / "construction_manifest.jsonl"
HISTORICAL_TEST = ROOT / "dp_SA" / "outputs" / "steering" / "test_manifest.jsonl"
CHECKPOINT_ROOT = ROOT / "dp_SA" / "checkpoint_steering" / "output" / "results"
CHECKPOINT_CLEAN = CHECKPOINT_ROOT / "artifacts" / "diagnostics" / "clean_capture.jsonl"
CHECKPOINT_EXTENSION_CLEAN = CHECKPOINT_ROOT / "artifacts" / "diagnostics" / "clean_capture_layer_12_16_22.jsonl"
CHECKPOINT_COMPLETION = CHECKPOINT_ROOT / "progress" / "completion.json"

EXPECTED_SOURCE_SHA256 = {
    "historical_capture": "356e73b66fd483676b952f4b78ce41ee9404a1d0e52f4174a1a34ec4fdef7332",
    "historical_capture_config": "6d25e20881b3433607823cc00939248f1de8958dd5ed489a6a82ef2994e6c64f",
    "checkpoint_clean": "e7ee2041c3cfa75a4d62effaf2f3043aff42cfd21cd4ad56b93d74eebaa4d6b3",
    "checkpoint_extension_clean": "cc528a1a316821723b01ef58f84b1017439f95372da6f3a5075c066247e200ab",
    "checkpoint_completion": "31465ddd0791dbb4860e71e7cb6196bcc8a002ce023d873fff5935d08568ea1a",
}

CANONICAL_ANSWERS = (
    "black", "blue", "brown", "cyan", "gray", "green",
    "orange", "pink", "purple", "red", "white", "yellow",
)
POSITIONS = ("P1_LAT", "P1_PANL")
LAYERS = tuple(range(9, 16))
ALPHAS = (-10.0, -2.0, 0.0, 2.0, 10.0)
DIRECTIONS = ("matched_loao", "unmatched_global", "within_answer_shuffled")
FOLD_COUNTS = (5, 8, 10, 12, 15)
FINAL_FOLD_COUNT = 15
CONSTRUCTION_MIN_FAMILIES = 15
TEST_MIN_FAMILIES = 10
TEST_MAX_FAMILIES = 15
FOLD_SEARCH_REPEATS = 5000

SMOKE_LAYERS = (9, 12, 14, 18)
SMOKE_ALPHAS = (-2.0, 0.0, 2.0)
SMOKE_DIRECTIONS = ("matched_loao", "within_answer_shuffled")
SMOKE_CONSTRUCTION_MIN = 2
SMOKE_TEST_COUNT = 4
SMOKE_BOOTSTRAP_REPEATS = 200
MAX_SMOKE_ROUNDS = 5

CANDIDATE_CASE_FINGERPRINT = "3bf867012d18f5d23645e2d7268c53e933ac690de0c568af691e0f17e08a0c9f"
CANDIDATE_IDENTITY_FINGERPRINT = "a03e7e917917ae555ed1dd7547bbf9354b52bed3a587036a6366377410c03e39"
FAMILY_FINGERPRINT = "d0f35cf81df8ced33629d0463f41bb4d0aa386836840df7b4901bbe0aa96b081"

FORMAL_REUSED_HIDDEN = 0
FORMAL_CAPTURE_FORWARDS = 1625
FORMAL_TEST_FAMILIES = 174
FORMAL_STEERING_FORWARDS = FORMAL_TEST_FAMILIES * len(POSITIONS) * len(DIRECTIONS) * len(LAYERS) * len(ALPHAS)
FORMAL_TOTAL_FORWARDS = FORMAL_CAPTURE_FORWARDS + FORMAL_STEERING_FORWARDS
