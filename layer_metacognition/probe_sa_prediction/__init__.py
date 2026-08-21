"""OOF probes for predicting final Source Attribution from hidden states."""

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
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "stage_sa_prediction_probe"

DEFAULT_LAYERS = (10, 12, 14, 16, 18, 20, 22, 24, 26, 27)
DEFAULT_POSITIONS = ("ac", "lat", "panl", "sac")
SA_CLASSES = tuple(str(index) for index in range(9))
TASKS = ("hard_label", "soft_score")
HIDDEN_STATE_DEFINITION = "decoder_block_output_pre_final_norm"


def job_id(task: str, position: str, layer: int, fold: int) -> str:
    return f"{task}|{position}|{int(layer)}|{int(fold)}"


def prediction_key(record: dict[str, object]) -> tuple[str, str, int, int, str]:
    return (
        str(record["task"]),
        str(record["position"]),
        int(record["layer"]),
        int(record["fold"]),
        str(record["case_id"]),
    )
