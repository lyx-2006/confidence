from __future__ import annotations

from pathlib import Path

from dp_SA.config import BOOTSTRAP_REPEATS, DATASET_PATH, INFERENCE_PATH, MIDPOINTS, MODEL_PATH, SEED

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOINT_SOURCE = ROOT / "layer_metacognition" / "output" / "Final_v4_run" / "answer_basis_9"
DEFAULT_DELAYED_SOURCE = ROOT / "dp_SA" / "outputs"
DEFAULT_OUTPUT_PARENT = ROOT / "dp_SA" / "attention_block" / "outputs"

COARSE_WINDOWS = tuple((start, start + 11) for start in (0, 4, 8, 12, 16))
REFINE_WINDOWS = tuple((start, start + 5) for start in range(0, 23, 2))
ALL_LAYERS = tuple(range(28))

WINDOW_CONDITIONS = (
    "panl_to_evidence",
    "panl_to_answer",
    "panl_to_evidence_answer",
    "panl_plus_1_to_evidence_answer",
    "sac_to_panl",
    "sac_to_panl_plus_1",
    "sac_to_evidence",
    "sac_to_answer",
    "sac_to_all_content",
)

GLOBAL_CONDITIONS = (
    "all_downstream_to_panl",
    "all_downstream_to_panl_plus_1",
    "all_later_to_evidence",
    "all_later_to_evidence_keep_panl",
    "all_later_to_answer",
    "all_later_to_answer_keep_panl",
    "all_later_to_evidence_answer",
    "all_later_to_evidence_answer_keep_panl",
)

SMOKE_CONDITIONS = (
    "panl_to_evidence",
    "panl_to_answer",
    "panl_to_evidence_answer",
    "panl_plus_1_to_evidence_answer",
    "sac_to_panl",
    "sac_to_panl_plus_1",
    "sac_to_all_content",
)

MATCHED_PAIRS = {
    "panl_cache": ("sac_to_panl", "sac_to_panl_plus_1"),
    "panl_gather": ("panl_to_evidence_answer", "panl_plus_1_to_evidence_answer"),
    "jit_all_content": ("sac_to_all_content", "sac_to_panl_plus_1"),
}

MAX_CASES_PER_SIDE = 50
# The historical joint teacher stage is reproducible on the same eager full
# forward.  Keep a small float-conversion tolerance for the derived soft score.
SOFT_PARITY_TOLERANCE = 1e-6
LOGIT_PARITY_TOLERANCE = 0.125
ROW_SUM_TOLERANCE = 0.01
FAILURE_RATE_LIMIT = 0.05
REFINE_Q_THRESHOLD = 0.05


def parse_windows(value: str) -> tuple[tuple[int, int], ...]:
    windows = []
    for cell in value.split(","):
        left, separator, right = cell.strip().partition("-")
        if not separator:
            raise ValueError(f"Window must use START-END syntax: {cell!r}")
        start, end = int(left), int(right)
        if start < 0 or end < start or end >= 28:
            raise ValueError(f"Window is outside layers 0-27: {(start, end)}")
        windows.append((start, end))
    if not windows or len(set(windows)) != len(windows):
        raise ValueError("Windows must be non-empty and unique")
    return tuple(windows)

__all__ = [name for name in globals() if name.isupper()]
