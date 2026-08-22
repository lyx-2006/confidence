"""Early Source Attribution probes from an answer-only (no-SA) prompt."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOINT_EXPERIMENT_DIR = (
    ROOT
    / "layer_metacognition"
    / "output"
    / "Final_v4_run_sa_prediction"
    / "answer_basis_9"
)
DEFAULT_NO_SA_EXPERIMENT_DIR = (
    ROOT
    / "layer_metacognition"
    / "output"
    / "Final_v4_run_no_sa"
    / "baseline"
)
DEFAULT_SPLIT_ASSIGNMENTS = (
    DEFAULT_JOINT_EXPERIMENT_DIR / "stage_sa_prediction_probe" / "split_assignments.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_NO_SA_EXPERIMENT_DIR / "stage_no_sa_prediction_probe"

DEFAULT_LAYERS = (10, 12, 14, 16, 18, 20, 22, 24, 26, 27)
DEFAULT_POSITIONS = ("ptnl", "pit", "ac", "lat", "panl")
DEFAULT_COHORTS = ("answer_matched", "all_joined")
CONFLICT_CONDITIONS = ("conflict_easy", "conflict_hard")
SA_CLASSES = tuple(str(index) for index in range(9))
TASKS = ("hard_label", "soft_score")
HIDDEN_STATE_DEFINITION = "decoder_block_output_pre_final_norm"
FORBIDDEN_SA_TEXT = ("source attribution", "sa class")


def join_key(record: dict[str, object]) -> tuple[str, int, str, str]:
    return (
        str(record["item_id"]),
        int(record["prior_index"]),
        str(record["condition"]),
        str(record["version"]),
    )


def prediction_key(record: dict[str, object]) -> tuple[str, str, str, int, int, int, str]:
    return (
        str(record["cohort"]),
        str(record["task"]),
        str(record["position"]),
        int(record["layer"]),
        int(record["fold"]),
        int(record.get("prior_index", 0)),
        str(record["no_sa_case_id"]),
    )


__all__ = [
    "CONFLICT_CONDITIONS",
    "DEFAULT_COHORTS",
    "DEFAULT_JOINT_EXPERIMENT_DIR",
    "DEFAULT_LAYERS",
    "DEFAULT_NO_SA_EXPERIMENT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_POSITIONS",
    "DEFAULT_SPLIT_ASSIGNMENTS",
    "FORBIDDEN_SA_TEXT",
    "HIDDEN_STATE_DEFINITION",
    "SA_CLASSES",
    "TASKS",
    "join_key",
    "prediction_key",
]
