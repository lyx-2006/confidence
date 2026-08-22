"""Answer-fixed Source Attribution embedding corruption and activation patching."""

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
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "stage_sa_patching"
DEFAULT_MODEL_PATH = ROOT / "qwen-2.5-vl" / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_DATASET = ROOT / "datasets" / "datasets.json"

POSITIONS = ("ac", "panl", "sac")
CORRUPTIONS = ("image_only", "text_only", "image_text", "answer_only", "all")
POSITION_LAYERS = {
    "ac": (12, 16, 20),
    "panl": (16, 18, 20),
    "sac": (18, 20, 24),
}
EVALUATION_CONDITIONS = ("conflict_easy", "conflict_hard")
LOW_SA_CLASSES = ("0", "1", "2")
HIGH_SA_CLASSES = ("5", "6", "8")
FORMAT_VERSION = 1


__all__ = [
    "CORRUPTIONS",
    "DEFAULT_DATASET",
    "DEFAULT_EXPERIMENT_DIR",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "EVALUATION_CONDITIONS",
    "HIGH_SA_CLASSES",
    "LOW_SA_CLASSES",
    "POSITIONS",
    "POSITION_LAYERS",
    "ROOT",
]
