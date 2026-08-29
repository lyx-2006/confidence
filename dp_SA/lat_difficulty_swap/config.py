from __future__ import annotations

from pathlib import Path

from dp_SA.config import DATASET_PATH, INFERENCE_PATH, MODEL_PATH, ROOT, SPLIT_PATH

PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_PARENT = PACKAGE_ROOT / "output"
RESULTS_ROOT = OUTPUT_PARENT / "results"
SMOKE_ROOT = OUTPUT_PARENT / "smoke_tmp"

PANL_ROOT = ROOT / "dp_SA" / "panl_information" / "output" / "results"
PANL_ARTIFACTS = PANL_ROOT / "artifacts"
DELAYED_ROOT = ROOT / "dp_SA" / "outputs"
JOINED_PATH = PANL_ARTIFACTS / "joined_records.jsonl"
PANL_CAPTURE_PATH = PANL_ARTIFACTS / "capture_results.jsonl"
PHASE0_PATH = DELAYED_ROOT / "capture" / "phase0_results.jsonl"
HISTORICAL_SA_OOF_PATH = DELAYED_ROOT / "probe" / "oof_predictions.jsonl"
DIFFICULTY_OOF_PATH = PANL_ARTIFACTS / "difficulty_probe_oof_predictions.jsonl"
DECISION_OOF_PATH = PANL_ARTIFACTS / "decision_probe_oof_predictions.jsonl"
PROBE_MODEL_ROOT = PANL_ARTIFACTS / "probe_models"

SWAP_LAYERS = (10, 14, 16, 18, 22, 24, 26)
PANL_READOUT_BY_SWAP_LAYER = {10: 12, 14: 16, 16: 18, 18: 20, 22: 24, 24: 26, 26: 27}
SA_RECONSTRUCTION_LAYERS = tuple(sorted(set(PANL_READOUT_BY_SWAP_LAYER.values()) | {14}))
POSITIONS = ("P1_LAT", "P1_PANL", "P1_PANL_PLUS_1", "P1_SAC")

MIDPOINTS = (0.05, 0.175, 0.325, 0.4375, 0.5, 0.5625, 0.675, 0.825, 0.95)
MIN_DIFFICULTY_GAP = 20.0
SEED = 42
BOOTSTRAP_REPEATS = 2000
SMOKE_BOOTSTRAP_REPEATS = 100
SOFT_SA_NO_CHANGE_TOLERANCE = 1e-6
SOFT_SA_PARITY_TOLERANCE = 1e-6
LOGIT_PARITY_TOLERANCE = 0.125
PROBE_PARITY_TOLERANCE = 1e-10
HIDDEN_DEFINITION = "decoder_block_output_pre_final_norm"

PROGRESS_FILES = (
    "run_config.json", "input_fingerprints.json", "progress.json", "stage_status.json",
    "test_report.json", "smoke_report.json", "failures.jsonl", "completion.json", "pipeline.log",
)

__all__ = [name for name in globals() if name.isupper()]
