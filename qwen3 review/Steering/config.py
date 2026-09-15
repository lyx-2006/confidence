from __future__ import annotations

from pathlib import Path


REVIEW_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REVIEW_ROOT.parent

MODEL_PATH = REPOSITORY_ROOT / "qwen-3-vl" / "model"
INFERENCE_PATH = REPOSITORY_ROOT / "qwen-3-vl" / "interface.py"
DATASET_PATH = REPOSITORY_ROOT / "datasets" / "datasets.json"
CAPTURE_ROOT = REVIEW_ROOT / "capture"
STEERING_OUTPUT_ROOT = REVIEW_ROOT / "Steering" / "output"

CONDITIONS = ("conflict_easy", "conflict_hard")
POSITIONS = ("LAT", "PANL", "CLE", "PANL+1", "SAC")
POSITION_KEYS = {
    "LAT": "P1_LAT",
    "PANL": "P1_PANL",
    "CLE": "P1_CLASS_LIST_END",
    "PANL+1": "P1_PANL_PLUS_1",
    "SAC": "P1_SAC",
}

# Qwen3-VL-8B-Instruct has 36 zero-based language layers.  The experiment
# intentionally starts at layer 8 and captures every remaining block output.
EXPECTED_NUM_HIDDEN_LAYERS = 36
EXPECTED_HIDDEN_SIZE = 4096
CAPTURE_LAYERS = tuple(range(8, EXPECTED_NUM_HIDDEN_LAYERS))

SEED = 42
VECTOR_NORM_FRACTION = 0.03
CONSTRUCTION_PER_SIDE = 25
IMAGE_TEST_COUNT = 50
TEXT_TEST_COUNT = 31
ERROR_RATE_LIMIT = 0.05
LOGIT_PARITY_TOLERANCE = 1e-6
HIDDEN_DEFINITION = "decoder_block_output_pre_final_norm"
