from __future__ import annotations

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
MODEL_PATH = ROOT / "gemma-3-4b-it" / "models"
DATASET_PATH = ROOT / "datasets" / "datasets.json"
OUTPUT_BASE = EXPERIMENT_DIR / "output"
RESULTS_ROOT = OUTPUT_BASE / "results"
SMOKE_ROOT = OUTPUT_BASE / "smoke"

CONDITIONS = ("conflict_easy", "conflict_hard")
POSITIONS = (
    "P1_AC",
    "P1_LAT",
    "P1_PANL",
    "P1_PANL_PLUS_1",
    "P1_CLE",
    "P1_SAC",
)
CAPTURE_LAYERS = tuple(range(6, 34))
DEFAULT_SHUFFLED_LAYERS = (22, 24)
ALPHAS = (-10.0, -2.0, 0.0, 2.0, 10.0)
SMOKE_ALPHAS = (-2.0, 0.0, 2.0)
SMOKE_LAYER = 22
SMOKE_MAX_SAMPLES = 30

MIDPOINTS = (0.05, 0.175, 0.325, 0.4375, 0.5, 0.5625, 0.675, 0.825, 0.95)
SEED = 42
BOOTSTRAP_REPEATS = 2000
VECTOR_NORM_FRACTION = 0.03
CONSTRUCTION_PER_SIDE = 25
TEST_PER_SIDE = 50
ERROR_RATE_LIMIT = 0.05
HIDDEN_SIZE = 2560
NUM_HIDDEN_LAYERS = 34
HIDDEN_DEFINITION = "gemma_decoder_block_output_pre_final_norm"

