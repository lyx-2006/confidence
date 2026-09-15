from pathlib import Path

HERE = Path(__file__).resolve().parent
SHORT_ROOT = HERE.parent
REVIEW_ROOT = SHORT_ROOT.parent
REPOSITORY_ROOT = REVIEW_ROOT.parent

MODEL_PATH = REPOSITORY_ROOT / "qwen-3-vl" / "model"
SHORT_CAPTURE = SHORT_ROOT / "output" / "capture"
REVERSE_CAPTURE = SHORT_ROOT / "output" / "capture_reverse"
OUTPUT_ROOT = SHORT_ROOT / "output" / "cle_swap"
SMOKE_ROOT = OUTPUT_ROOT / "smoke"

LAYERS = (12, 16, 18, 20, 22, 24, 26, 28, 30)
DIRECTIONS = ("reverse_to_short", "short_to_reverse")
SEED = 42
BOOTSTRAP_REPEATS = 2000
EXPECTED_CASES = 50
SIDE_COUNTS = {"text_side": 25, "image_side": 25}
LOGIT_PARITY_ATOL = 1e-6
HIDDEN_SIZE = 4096
NUM_LAYERS = 36

