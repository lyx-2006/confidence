from __future__ import annotations

from pathlib import Path

from dp_SA.config import BOOTSTRAP_REPEATS, INFERENCE_PATH, MIDPOINTS, MODEL_PATH, ROOT, SEED

PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "output"
RESULTS_ROOT = OUTPUT_ROOT / "results"
SMOKE_ROOT = OUTPUT_ROOT / "smoke"

def validation_root(case_count: int) -> Path:
    return OUTPUT_ROOT / f"validation_{case_count}"

TRAJECTORY_ROOT = ROOT / "dp_SA/confidence_steering/trajectory/output/results"
AUDIT_MANIFEST = TRAJECTORY_ROOT / "artifacts/manifests/audit_manifest.jsonl"
CONSTRUCTION_MANIFEST = TRAJECTORY_ROOT / "artifacts/manifests/construction_manifest.jsonl"
PROBE_ROOT = TRAJECTORY_ROOT / "artifacts/probes"
PROBE_INDEX = PROBE_ROOT / "probe_index.jsonl"
TRAJECTORY_CONFIG = TRAJECTORY_ROOT / "artifacts/config_and_fingerprint.json"

MATCHED_ROOT = ROOT / "dp_SA/answer_matched_lat_steering/output/lat_panl_comparison"
MATCHED_MANIFEST_ROOT = MATCHED_ROOT / "artifacts/manifests"
CANDIDATE_MANIFEST = MATCHED_MANIFEST_ROOT / "candidate_manifest.jsonl"
CELL_MANIFEST = MATCHED_MANIFEST_ROOT / "construction_family_cells.jsonl"
CONSTRUCTION_DISTRIBUTION = MATCHED_MANIFEST_ROOT / "construction_distribution.jsonl"
FOLD_MANIFEST = MATCHED_MANIFEST_ROOT / "fold_assignments.jsonl"
TEST_MANIFEST = MATCHED_MANIFEST_ROOT / "test_manifest.jsonl"
T0_VECTOR_METADATA = MATCHED_ROOT / "artifacts/vectors/vector_metadata.json"
T0_CLEAN_CAPTURE = MATCHED_ROOT / "artifacts/diagnostics/clean_capture.jsonl"
T0_STEERING_TRIALS = MATCHED_ROOT / "artifacts/diagnostics/steering_trials.P1_LAT.jsonl"
T0_STEERING_TABLE = MATCHED_ROOT / "tables/table1_lat_panl_steering.csv"

POSITIONS = ("P1_LAT", "P1_PANL", "P1_CLASS_LIST_END", "P1_SAC")
PROBE_LAYERS = tuple(range(14, 27))
GEOMETRY_LAYERS = tuple(range(9, 16))
DEFAULT_STEERING_LAYERS = GEOMETRY_LAYERS
DEFAULT_ALPHAS = (-2.0, 0.0, 2.0)
TEMPLATE_NAMES = ("T1", "T2", "T3")
MIDPOINTS_9 = tuple(float(x) for x in MIDPOINTS)
T3_LABELS = ("STRONG_TEXT", "SLIGHT_TEXT", "BALANCED", "SLIGHT_IMAGE", "STRONG_IMAGE")
T3_VALUES = (0.1125, 0.38125, 0.5, 0.61875, 0.8875)
FLOAT_ATOL = 1e-6
BOOTSTRAPS = BOOTSTRAP_REPEATS
SMOKE_BOOTSTRAPS = 200
VALIDATION_BOOTSTRAPS = 500
SMOKE_CASES_PER_SOURCE = 4

EXPECTED_COUNTS = {
    "audit": 230,
    "audit_families": 25,
    "probe_construction": 882,
    "probes": 52,
    "candidates": 1625,
    "candidate_families": 178,
    "cells": 8792,
    "folds": 15,
    "test": 174,
    "confirmatory_test": 165,
}
