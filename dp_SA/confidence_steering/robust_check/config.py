from __future__ import annotations

from pathlib import Path

from dp_SA.config import INFERENCE_PATH, MODEL_PATH, ROOT, VECTOR_NORM_FRACTION

PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "output"
RESULTS_ROOT = OUTPUT_ROOT / "results"

SOURCE_ROOT = ROOT / "dp_SA/unimodal_logit_confidence/output/results"
TRAIN_MANIFEST = SOURCE_ROOT / "shared/manifests/probe_train_manifest.jsonl"
FORMAL_MANIFEST = SOURCE_ROOT / "shared/manifests/test_manifest.jsonl"
CONFIDENCE_JOINED = SOURCE_ROOT / "unimodal_confidence/artifacts/predictions/phase1_confidence_joined.jsonl"
UNIMODAL_SCORES = SOURCE_ROOT / "unimodal_confidence/artifacts/calibrated_scores/unimodal_scores.jsonl"
HIDDEN_REUSE = SOURCE_ROOT / "confidence_probe/artifacts/hidden/reuse_manifest.jsonl"
HIDDEN_CAPTURE = SOURCE_ROOT / "confidence_probe/artifacts/hidden/capture_results.jsonl"

PARENT_ROOT = PACKAGE_ROOT.parent
NATURAL_ROOT = PARENT_ROOT / "output/natural_decomposition"
FAST_REFERENCE_ROOT = PARENT_ROOT / "output/all_fast_l14"
FROZEN_LAT_PROBE = NATURAL_ROOT / "artifacts/probes/confidence_gap__P1_LAT__L14__full.joblib"
FROZEN_PANL_SA_PROBE = NATURAL_ROOT / "artifacts/probes/final_sa__P1_PANL__L18__full.joblib"
FROZEN_VECTORS = NATURAL_ROOT / "artifacts/directions/P1_LAT__L14.npz"
FROZEN_VECTOR_METADATA = NATURAL_ROOT / "artifacts/directions/vector_metadata.json"

SEEDS = (42, 43, 44, 45)
ALTERNATIVE_SEEDS = (43, 44, 45)
SMOKE_SEED = 45
N_SPLITS = 5
AUDIT_FOLD = 0
LAYER = 14
PANL_LAYER = 18
HIDDEN_SIZE = 3584
POSITION = "P1_LAT"
PANL_POSITION = "P1_PANL"
SAC_POSITION = "P1_SAC"
DIRECTIONS = (
    "confidence_raw",
    "confidence_parallel_sa",
    "confidence_perp_sa_natural_scale",
)
ALPHAS = (-0.5, 0.0, 0.5)
NONZERO_ALPHAS = (-0.5, 0.5)
CANONICAL_COLORS = (
    "black", "blue", "brown", "cyan", "gray", "green",
    "orange", "pink", "purple", "red", "white", "yellow",
)
RIDGE_ALPHA_GRID = tuple(10.0**exponent for exponent in range(-4, 5))
BOOTSTRAP_REPEATS = 2000
SMOKE_BOOTSTRAP_REPEATS = 200
MAX_SMOKE_ROUNDS = 5
VECTOR_NORM_FRACTION = float(VECTOR_NORM_FRACTION)
RECONSTRUCTION_RTOL = 1e-5
PROTOCOL_VERSION = 1
ANALYSIS_NAME = "fixed-evaluation-set split-seed stability analysis"

EXPECTED_TRAIN_SHA256 = "7b498813302b1c2223aeb8f0eb4a10f0c3ed110e15ac1d0fe905d93d0b67102f"
EXPECTED_FORMAL_SHA256 = "97e7726e40b6df67661d1649306f39e76a4dc8bfedfe9771451ac458964adf68"
EXPECTED_ASSIGNMENT_HASHES = {
    42: "a6cd067008d0bf26d16127d88772e65a6deed0bc02d5bc6411297fa7cfb2ca5e",
    43: "de077f35a2285c2ba3258a9aac822d3e91922af5008d355a40c80f2a8fd6b171",
    44: "472e0b2871cbd50ab4ca725d1de7a1dbfabf411f75bdafed81c7aa4625551c7c",
    45: "33f3e74c45b9f11cfa4b068607a55d7d2fb04ade6063a7d37454f2a71fe3890e",
}

