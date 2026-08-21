"""Answer-fixed Source Attribution hidden-state steering."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_DIR = (
    ROOT
    / "layer_metacognition"
    / "output"
    / "Final_v4_run_sa_prediction"
    / "answer_basis_9"
)
DEFAULT_PROBE_DIR = DEFAULT_EXPERIMENT_DIR / "stage_sa_prediction_probe"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "layer_metacognition"
    / "output"
    / "Final_v4_run"
    / "answer_basis_9"
    / "stage_sa_steering"
)
DEFAULT_MODEL_PATH = ROOT / "qwen-2.5-vl" / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_DATASET = ROOT / "datasets" / "datasets.json"

POSITIONS = ("ac", "lat", "panl", "sac")
LAYERS = (12, 16, 18, 20, 24, 26)
METHODS = ("mean_difference", "probe_weight")
DIRECTIONS = ("high", "low")
ALPHAS = (2.0, 10.0)
LOW_SA_CLASSES = ("0", "1", "2")
HIGH_SA_CLASSES = ("5", "6", "8")
HIDDEN_STATE_DEFINITION = "decoder_block_output_pre_final_norm"
VECTOR_NORM_FRACTION = 0.03


__all__ = [
    "ALPHAS",
    "DEFAULT_DATASET",
    "DEFAULT_EXPERIMENT_DIR",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PROBE_DIR",
    "DIRECTIONS",
    "HIGH_SA_CLASSES",
    "HIDDEN_STATE_DEFINITION",
    "LAYERS",
    "LOW_SA_CLASSES",
    "METHODS",
    "POSITIONS",
    "ROOT",
    "VECTOR_NORM_FRACTION",
]
