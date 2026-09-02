from __future__ import annotations

from pathlib import Path

from dp_SA.config import DATASET_PATH, INFERENCE_PATH, MODEL_PATH, ROOT, SPLIT_PATH

PACKAGE_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PACKAGE_ROOT / "output" / "results"
SMOKE_ROOT = PACKAGE_ROOT / "output" / "smoke_tmp"

SOURCE_RESULTS = ROOT / "dp_SA" / "outputs" / "capture" / "results.jsonl"
SOURCE_CONFIG = ROOT / "dp_SA" / "outputs" / "capture" / "config.json"
SOURCE_MANIFEST = ROOT / "dp_SA" / "panl_information" / "output" / "results" / "artifacts" / "manifest.jsonl"
SOURCE_CAPTURE = ROOT / "dp_SA" / "panl_information" / "output" / "results" / "artifacts" / "capture_results.jsonl"
SOURCE_HIDDEN_ROOT = ROOT / "dp_SA" / "panl_information" / "output" / "results"
SOURCE_CAPTURE_AUDIT = SOURCE_HIDDEN_ROOT / "artifacts" / "capture_parity_audit.json"
SUPPLEMENT_CAPTURE = ROOT / "dp_SA" / "checkpoint_steering" / "output" / "results" / "artifacts" / "diagnostics" / "clean_capture.jsonl"
SUPPLEMENT_HIDDEN_ROOT = ROOT / "dp_SA" / "checkpoint_steering" / "output" / "results"
FAMILY_MANIFEST = ROOT / "dp_SA" / "answer_matched_lat_steering" / "output" / "results" / "artifacts" / "manifests" / "family_manifest.jsonl"

SEED = 42
CONDITIONS = ("conflict_easy", "conflict_hard")
CANONICAL_COLORS = ("black", "blue", "brown", "cyan", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow")
PROBE_POSITIONS = ("P1_AC", "P1_LAT", "P1_PANL", "P1_PANL_PLUS_1")
PROBE_LAYERS = (6, 8, 10, 12, 14, 16, 18, 22, 24, 26)
HISTORICAL_PRIMARY_LAYERS = (10, 12, 14, 16, 18, 22, 24, 26)
HIDDEN_DEFINITION = "decoder_block_output_pre_final_norm"
TARGETS = ("text_chosen_confidence", "image_chosen_confidence", "text_fixed_answer_confidence", "image_fixed_answer_confidence")

TEXT_PHASE0_TEMPLATE = """Question:
{question}

Text clue:
{text_clue}

Answer the question using the text clue.

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, source attribution, or any additional text.

Output exactly:

**Answer**: <your answer>"""

IMAGE_PHASE0_TEMPLATE = """Question:
{question}

Answer the question using the image.

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, source attribution, or any additional text.

Output exactly:

**Answer**: <your answer>"""

ANSWER_PREFILL = "**Answer**:"
TEMPERATURE_GRID_SIZE = 4096
TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 100.0
ECE_BINS = 10
BOOTSTRAP_REPEATS = 2000
RIDGE_ALPHA = 1.0
FIXED_EPSILON = 1e-12
MAX_SMOKE_ROUNDS = 5

