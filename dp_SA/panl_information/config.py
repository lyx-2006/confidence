from __future__ import annotations

from pathlib import Path

from dp_SA.config import DATASET_PATH, INFERENCE_PATH, MODEL_PATH, ROOT, SPLIT_PATH

PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_PARENT = PACKAGE_ROOT / "output"
RESULTS_ROOT = OUTPUT_PARENT / "results"
TEST_TMP_ROOT = OUTPUT_PARENT / "test_tmp"
SMOKE_TMP_ROOT = OUTPUT_PARENT / "smoke_tmp"
SOURCE_CAPTURE_ROOT = ROOT / "dp_SA" / "outputs"
SOURCE_RESULTS = SOURCE_CAPTURE_ROOT / "capture" / "results.jsonl"
SOURCE_PHASE0 = SOURCE_CAPTURE_ROOT / "capture" / "phase0_results.jsonl"
SOURCE_CONFIG = SOURCE_CAPTURE_ROOT / "capture" / "config.json"
SOURCE_OOF = SOURCE_CAPTURE_ROOT / "probe" / "oof_predictions.jsonl"

CONDITIONS = ("conflict_easy", "conflict_hard")
POSITIONS = ("P1_AC", "P1_LAT", "P1_PANL", "P1_PANL_PLUS_1", "P1_SAC")
LAYERS = (10, 12, 14, 16, 18, 20, 22, 24, 26, 27)
HISTORICAL_POSITIONS = ("P1_AC", "P1_PANL", "P1_PANL_PLUS_1", "P1_SAC")
HISTORICAL_LAYERS = (10, 14, 18, 20, 24, 26)

SEED = 42
N_SPLITS = 5
INNER_SPLITS = 3
BOOTSTRAP_REPEATS = 2000
PERMUTATION_REPEATS = 2000
SMOKE_BOOTSTRAP_REPEATS = 100
RIDGE_ALPHA = 1.0
C_GRID = (0.01, 0.1, 1.0, 10.0)
ECE_BINS = 10
SOFT_SA_TOLERANCE = 1e-6
LOGIT_TOLERANCE = 0.125
PARAMETERIZATION_TOLERANCE = 1e-10
HIDDEN_DEFINITION = "decoder_block_output_pre_final_norm"

ARTIFACT_NAMES = {
    "unimodal": "unimodal_scores.jsonl",
    "manifest": "manifest.jsonl",
    "joined": "joined_records.jsonl",
    "excluded": "excluded_records.jsonl",
    "capture": "capture_results.jsonl",
    "difficulty_oof": "difficulty_probe_oof_predictions.jsonl",
    "decision_oof": "decision_probe_oof_predictions.jsonl",
}

PROGRESS_FILES = (
    "run_config.json",
    "input_fingerprints.json",
    "progress.json",
    "stage_status.json",
    "test_report.json",
    "failures.jsonl",
    "completion.json",
    "pipeline.log",
)

__all__ = [name for name in globals() if name.isupper()]
