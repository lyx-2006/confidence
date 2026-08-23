from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "dp_SA" / "outputs"
MODEL_PATH = ROOT / "qwen-2.5-vl" / "models" / "Qwen2.5-VL-7B-Instruct"
DATASET_PATH = ROOT / "datasets" / "datasets.json"
INFERENCE_PATH = ROOT / "qwen-2.5-vl" / "inference.py"
SPLIT_PATH = ROOT / "layer_metacognition" / "output" / "Final_v4_run_no_sa" / "baseline" / "stage_no_sa_prediction_probe" / "split_assignments.json"

CONDITIONS = ("conflict_easy", "conflict_hard")
POSITIONS = ("P1_AC", "P1_PANL", "P1_PANL_PLUS_1", "P1_SAC")
PROBE_POSITIONS = ("P1_AC", "P1_PANL", "P1_PANL_PLUS_1")
LAYERS = (10, 14, 18, 20, 24, 26)
ALPHAS = (-10.0, -2.0, 0.0, 2.0, 10.0)
MIDPOINTS = (0.05, 0.175, 0.325, 0.4375, 0.5, 0.5625, 0.675, 0.825, 0.95)
SEED = 42
BOOTSTRAP_REPEATS = 2000
VECTOR_NORM_FRACTION = 0.03
CONSTRUCTION_PER_SIDE = 25
TEST_PER_SIDE = 50
FLOAT_TOLERANCE = 1e-6
ERROR_RATE_LIMIT = 0.05
HIDDEN_DEFINITION = "decoder_block_output_pre_final_norm"
