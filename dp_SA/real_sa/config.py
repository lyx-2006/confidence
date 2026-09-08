from __future__ import annotations

from pathlib import Path

from dp_SA.config import INFERENCE_PATH, MODEL_PATH, ROOT

PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "output"

FROZEN_ROOT = ROOT / "dp_SA" / "unimodal_logit_confidence" / "output" / "results"
TEST_MANIFEST = FROZEN_ROOT / "shared" / "manifests" / "test_manifest.jsonl"
TRAIN_MANIFEST = FROZEN_ROOT / "shared" / "manifests" / "probe_train_manifest.jsonl"
IMAGE_DONOR_MANIFEST = FROZEN_ROOT / "shared" / "manifests" / "image_calibration_manifest.jsonl"
TEXT_DONOR_MANIFEST = FROZEN_ROOT / "shared" / "manifests" / "text_calibration_manifest.jsonl"
SPLIT_AUDIT = FROZEN_ROOT / "shared" / "split_audit.json"
HISTORICAL_PHASE0 = ROOT / "dp_SA" / "outputs" / "capture" / "phase0_results.jsonl"

SEED = 42
BOOTSTRAP_REPEATS = 2000
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
TEMPERATURE = 1.0
EFFICIENCY_TOLERANCE = 1e-10
PRIMARY_EFFECT_THRESHOLD = 0.02
SENSITIVITY_THRESHOLDS = (0.01, 0.02, 0.05)
CONDITIONS = ("clean", "10_text_corrupt", "01_image_corrupt", "00_both_corrupt")

MODEL_CONFIG_FILES = ("config.json", "tokenizer.json", "preprocessor_config.json")

__all__ = [
    "BOOTSTRAP_REPEATS", "CONDITIONS", "EFFICIENCY_TOLERANCE", "FROZEN_ROOT",
    "HISTORICAL_PHASE0", "IMAGE_DONOR_MANIFEST", "INFERENCE_PATH", "MAX_PIXELS",
    "MIN_PIXELS", "MODEL_CONFIG_FILES", "MODEL_PATH", "OUTPUT_ROOT",
    "PRIMARY_EFFECT_THRESHOLD", "SEED", "SENSITIVITY_THRESHOLDS", "SPLIT_AUDIT",
    "TEMPERATURE", "TEST_MANIFEST", "TEXT_DONOR_MANIFEST", "TRAIN_MANIFEST",
]
