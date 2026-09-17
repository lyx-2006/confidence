from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
SHORT_ROOT = HERE.parent
REVIEW_ROOT = SHORT_ROOT.parent
REPOSITORY_ROOT = REVIEW_ROOT.parent

MODEL_PATH = REPOSITORY_ROOT / "qwen-3-vl" / "model"
SOURCE_DATASET = (
    REPOSITORY_ROOT
    / "generate dataset"
    / "datasets"
    / "valid_datasets"
    / "generated_shape_color_dataset.json"
)
SOURCE_IMAGE_ROOT = SOURCE_DATASET.parent
TEXT_POOL = REPOSITORY_ROOT / "merged_color_prior_pool.json"
OUTPUT_ROOT = SHORT_ROOT / "output" / "faithful_check"

COLORS = (
    "red", "orange", "yellow", "green", "blue", "cyan",
    "purple", "pink", "brown", "white", "black", "gray",
)

SEED = 42
EASY_QUOTA = 70
HARD_QUOTA = 30
TEXT_TARGET_PER_COLOR = 10
TEXT_ENTROPY_TOLERANCE = 0.05
TEXT_PROBABILITY_TOLERANCE = 0.10
ATTRIBUTION_EPSILON = 1e-8
BOOTSTRAP_REPEATS = 2000

