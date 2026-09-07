from __future__ import annotations

from pathlib import Path

from dp_SA.config import INFERENCE_PATH, MODEL_PATH, ROOT
from dp_SA.confidence_steering.config import RIDGE_ALPHA_GRID

PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_PARENT = PACKAGE_ROOT / "output"
RESULTS_ROOT = OUTPUT_PARENT / "results"
SMOKE_ROOT = OUTPUT_PARENT / "smoke"

SOURCE_ROOT = ROOT / "dp_SA" / "unimodal_logit_confidence" / "output" / "results"
TRAIN_MANIFEST = SOURCE_ROOT / "shared/manifests/probe_train_manifest.jsonl"
SEALED_TEST_MANIFEST = ROOT / "dp_SA/confidence_steering/output/natural_decomposition/artifacts/manifests/runtime_manifest.jsonl"
SPLIT_AUDIT = SOURCE_ROOT / "shared/split_audit.json"
JOINED_CONFIDENCE = SOURCE_ROOT / "unimodal_confidence/artifacts/predictions/phase1_confidence_joined.jsonl"
CALIBRATED_SCORES = SOURCE_ROOT / "unimodal_confidence/artifacts/calibrated_scores/unimodal_scores.jsonl"
TEXT_TEMPERATURE = SOURCE_ROOT / "unimodal_confidence/artifacts/temperature/text_temperature.json"
IMAGE_TEMPERATURE = SOURCE_ROOT / "unimodal_confidence/artifacts/temperature/image_temperature.json"
CONFIDENCE_REUSE = SOURCE_ROOT / "confidence_probe/artifacts/hidden/reuse_manifest.jsonl"
CONFIDENCE_CAPTURE = SOURCE_ROOT / "confidence_probe/artifacts/hidden/capture_results.jsonl"

NATURAL_ROOT = ROOT / "dp_SA/confidence_steering/output/natural_decomposition"
PARENT_FAST_ROOT = ROOT / "dp_SA/confidence_steering/output/all_fast_l14"
PARENT_FAST_CLEAN = PARENT_FAST_ROOT / "artifacts/clean/clean.jsonl"
PARENT_FAST_CONFIG = PARENT_FAST_ROOT / "progress/config.json"
PARENT_PROCESSOR_FILE = ROOT / "dp_SA/confidence_steering/processor.py"
VECTOR_FILE = NATURAL_ROOT / "artifacts/directions/P1_LAT__L14.npz"
VECTOR_METADATA = NATURAL_ROOT / "artifacts/directions/vector_metadata.json"
PARENT_PANL_G_PROBE = NATURAL_ROOT / "artifacts/probes/confidence_gap__P1_PANL__L18__full.joblib"
PARENT_PANL_SA_PROBE = NATURAL_ROOT / "artifacts/probes/final_sa__P1_PANL__L18__full.joblib"
PARENT_AUDIT_PREDICTIONS = NATURAL_ROOT / "artifacts/probes/audit_predictions.jsonl"

BASE_CAPTURE_ROOT = ROOT / "dp_SA/outputs"
BASE_CAPTURE_CONFIG = BASE_CAPTURE_ROOT / "capture/config.json"
BASE_CAPTURE_ROWS = BASE_CAPTURE_ROOT / "capture/results.jsonl"
CLASS_CAPTURE_ROOT = ROOT / "dp_SA/outputs/probe_class_list_end_20260831"
CLASS_CAPTURE_CONFIG = CLASS_CAPTURE_ROOT / "capture/config.json"
CLASS_CAPTURE_ROWS = CLASS_CAPTURE_ROOT / "capture/results.jsonl"
CHECKPOINT_ROOT = ROOT / "dp_SA/checkpoint_steering/output/results"

EXPECTED_HASHES = {
    TRAIN_MANIFEST: "7b498813302b1c2223aeb8f0eb4a10f0c3ed110e15ac1d0fe905d93d0b67102f",
    SEALED_TEST_MANIFEST: "97e7726e40b6df67661d1649306f39e76a4dc8bfedfe9771451ac458964adf68",
    SPLIT_AUDIT: "6a3b500fd292eeff0e8add619d189760f30796a9c918b483b2931c824c9f9ca5",
    JOINED_CONFIDENCE: "0264871a0cecaccf37a58b00ac34e059090ad641a8bdfe2769dea953b7cad7cc",
    CALIBRATED_SCORES: "a23e9912ef672490937c7e80f7ef3f520a10899ecebb04904d20f7bc66818fec",
    TEXT_TEMPERATURE: "26774e64ed4f54c89c60543d996794dad6743c249e9f056213a938a30786d89b",
    IMAGE_TEMPERATURE: "adf2257291218f2b96ac536d37cdf0dabbee1cdefe168e3a818f0b7db1029697",
    VECTOR_FILE: "38dc50cffb9803ab844f22436d761bb3284059a6ce38c6b10c30554f65e78431",
    VECTOR_METADATA: "e86b85a820f5813c72cb39aeba5490ce513e8f9d166283a2761c5b27e8069e58",
    PARENT_FAST_CLEAN: "740930e22f51c9b6caae916a5975578d5771f3f2b06c9b50de9f0345b1dfc33d",
    PARENT_FAST_CONFIG: "5fc64430f987a9c01c730f91085d387da8f6aa9eb17e28a7226d186dacf4bd4b",
}

FORMAL_ONLY_SOURCES = frozenset((SEALED_TEST_MANIFEST, PARENT_FAST_CLEAN, PARENT_FAST_CONFIG))

TEXT_TAU = 0.7389907404044218
IMAGE_TAU = 0.8847730645925013
SEED = 42
BOOTSTRAP_REPEATS = 2000
HIDDEN_SIZE = 3584
HIDDEN_DEFINITION = "decoder_block_output_pre_final_norm"
INJECTION_SITE = "block_output"
INJECTION_LAYER = 14
EPSILON = 0.5
ALPHAS = (-0.5, 0.0, 0.5)
NONZERO_ALPHAS = (-0.5, 0.5)
POSITIONS = ("P1_LAT", "P1_PANL", "P1_CLASS_LIST_END", "P1_SAC")
LAYERS = tuple(range(14, 27))
TARGETS = ("C_i", "C_t", "G_L", "final_soft_sa")
DIRECTIONS = (
    "confidence_raw",
    "confidence_parallel_sa",
    "confidence_perp_sa_natural_scale",
)
SMOKE_DIRECTIONS = DIRECTIONS
SMOKE_FAMILIES = (
    "family_00b329339659ae6f",
    "family_479f016b66f001c1",
    "family_b64c6a771503d359",
    "family_ccd18627d4e12186",
)
CANONICAL_COLORS = ("black", "blue", "brown", "cyan", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow")
GROUPS = ("answer_equal_macro", "family_micro", "all")

TABLE_SCHEMAS = {
    "probe_metrics.csv": ("target", "position", "layer", "r2", "pearson", "pearson_ci_low", "pearson_ci_high", "readout_reliable"),
    "trajectory_readouts.csv": ("group", "direction", "position", "layer", "target", "mean", "ci_low", "ci_high", "readout_reliable"),
    "hidden_transport.csv": ("case_id", "direction", "position", "layer", "target", "delta_h_norm", "relative_clean_norm", "gradient_cosine", "readout_reliable"),
    "component_additivity.csv": ("group", "position", "layer", "target", "mean", "ci_low", "ci_high"),
}

PANL_L14_ATOL = 1e-6
RAW_EXPRESSION_STRICT_ATOL = 1e-5
RAW_EXPRESSION_ATOL = 1e-4
FLOAT_PARITY_ATOL = 1e-6
MAX_SMOKE_ROUNDS = 5

POSITION_DEFINITIONS = {
    "P1_LAT": "last processed token intersecting the fixed-answer character span before PANL",
    "P1_PANL": "first processed token intersecting the newline immediately after the fixed answer",
    "P1_CLASS_LIST_END": "first newline character immediately after the unique class-8 anchor",
    "P1_SAC": "last processed token intersecting the final colon of the assistant SA prefill",
}

RIDGE_ALPHAS = tuple(float(value) for value in RIDGE_ALPHA_GRID)
