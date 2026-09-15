from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REVIEW_ROOT = PACKAGE_ROOT.parent
REPOSITORY_ROOT = REVIEW_ROOT.parent
CAPTURE_ROOT = REVIEW_ROOT / "capture"
MODEL_PATH = REPOSITORY_ROOT / "qwen-3-vl" / "model"
OUTPUT_ROOT = PACKAGE_ROOT / "output"

SEED = 42
TEST_PER_SIDE = 50
BOOTSTRAP_REPEATS = 2000
ROW_SUM_TOLERANCE = 0.01
CLEAN_LOGIT_TOLERANCE = 1.0
CLEAN_SOFT_SA_TOLERANCE = 1e-2

CONDITIONS = ("C1_main_block", "C2_source_plus_1_control")
PRIMARY_GROUP = "answer_equal_macro"
GROUPS = (PRIMARY_GROUP, "overall_micro", "image_side", "text_side")
METRICS = ("delta_soft_sa", "token_change_rate", "logit_change_diff")


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    query: str
    main_source: str
    control_source: str
    windows: tuple[tuple[int, int], ...]


EXPERIMENTS = {
    "CLE2SAC": ExperimentSpec(
        "CLE2SAC", "P1_SAC", "P1_CLASS_LIST_END", "P1_CLASS_LIST_END_PLUS_1",
        ((12, 16), (16, 20), (20, 24), (24, 28)),
    ),
    "PANL2CLE": ExperimentSpec(
        "PANL2CLE", "P1_CLASS_LIST_END", "P1_PANL", "P1_PANL_PLUS_1",
        ((8, 12), (12, 16), (16, 20), (20, 24)),
    ),
}


def default_output(experiment: str, smoke: bool = False) -> Path:
    return OUTPUT_ROOT / ("smoke" if smoke else "") / experiment
