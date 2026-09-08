from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dp_SA.config import MODEL_PATH, ROOT

PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "output"
MANIFEST_PATH = (
    ROOT / "dp_SA" / "answer_matched_lat_steering" / "output"
    / "lat_panl_comparison" / "artifacts" / "manifests" / "test_manifest.jsonl"
)
HISTORICAL_CLEAN_PATH = (
    ROOT / "dp_SA" / "answer_matched_lat_steering" / "output"
    / "lat_panl_comparison" / "artifacts" / "diagnostics" / "clean_capture.jsonl"
)
PREPROCESSOR_CONFIG_PATH = MODEL_PATH / "preprocessor_config.json"

EXPECTED_MANIFEST_SHA256 = "6a6d6c88b8b1b120490a5723b6c3ee6e7ec40af4047b43096637a44dcf12bca4"
EXPECTED_HISTORICAL_CLEAN_SHA256 = "c137d2d8325e8c18a250f781aa5982b24850a78fbe509a09a94404a79562906c"
EXPECTED_PREPROCESSOR_SHA256 = "f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0"

WINDOWS = ((8, 12), (13, 17), (18, 22), (23, 26))
WINDOW_NAMES = {window: f"W{index}" for index, window in enumerate(WINDOWS, 1)}
SEED = 42
BOOTSTRAP_REPEATS = 2000
SMOKE_BOOTSTRAP_REPEATS = 200
EXPECTED_FORMAL_CASES = 174
SMOKE_CASES = 2
LOGIT_PARITY_TOLERANCE = 0.125
SOFT_PARITY_TOLERANCE = 1e-6
ROW_SUM_TOLERANCE = 0.01
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
PRIMARY_GROUP = "answer_equal_macro"
GROUPS = (PRIMARY_GROUP, "family_micro", "all", "image_side", "text_side")
PRIMARY_METRICS = ("delta_soft_sa", "token_change_rate", "logit_change_diff")
ALL_METRICS = (*PRIMARY_METRICS, "abs_delta_soft_sa")
CONFIRMATORY_ANSWERS = (
    "black", "brown", "cyan", "gray", "green", "orange",
    "pink", "purple", "red", "white", "yellow",
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    query: str
    main_source: str
    control_source: str


EXPERIMENTS = {
    "PANL2CLE": ExperimentSpec(
        "PANL2CLE", "P1_CLASS_LIST_END", "P1_PANL", "P1_PANL_PLUS_1"
    ),
    "CLE2SAC": ExperimentSpec(
        "CLE2SAC", "P1_SAC", "P1_CLASS_LIST_END", "P1_CLASS_LIST_END_PLUS_1"
    ),
}
CONDITIONS = ("C1_main_block", "C2_source_plus_1_control")


def parse_windows(value: str | None) -> tuple[tuple[int, int], ...]:
    if value is None:
        return WINDOWS
    parsed: list[tuple[int, int]] = []
    for cell in value.split(","):
        left, separator, right = cell.strip().partition("-")
        if not separator:
            raise ValueError(f"Window must use START-END syntax: {cell!r}")
        window = (int(left), int(right))
        if window not in WINDOWS:
            raise ValueError(f"Window is not one of the frozen ranges {WINDOWS}: {window}")
        parsed.append(window)
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError("Windows must be non-empty and unique")
    return tuple(parsed)


def default_output(experiment: str) -> Path:
    return OUTPUT_ROOT / experiment
