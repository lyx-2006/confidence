from __future__ import annotations

from pathlib import Path

from dp_SA.config import INFERENCE_PATH, MODEL_PATH, ROOT, VECTOR_NORM_FRACTION

PACKAGE_ROOT = Path(__file__).resolve().parent
FORMAL_ROOT = PACKAGE_ROOT / "output" / "orthogonal_results"
SMOKE_PARENT = PACKAGE_ROOT / "output" / "orthogonal_smoke"
SOURCE_ROOT = ROOT / "dp_SA" / "unimodal_logit_confidence" / "output" / "results"
SOURCE_SPLIT_AUDIT = SOURCE_ROOT / "shared" / "split_audit.json"

TRAIN_MANIFEST = SOURCE_ROOT / "shared" / "manifests" / "probe_train_manifest.jsonl"
# Deliberately not included in any pre-lock inventory. Formal runtime imports it lazily.
SEALED_TEST_MANIFEST = SOURCE_ROOT / "shared" / "manifests" / "test_manifest.jsonl"
CONFIDENCE_JOINED = SOURCE_ROOT / "unimodal_confidence" / "artifacts" / "predictions" / "phase1_confidence_joined.jsonl"
UNIMODAL_SCORES = SOURCE_ROOT / "unimodal_confidence" / "artifacts" / "calibrated_scores" / "unimodal_scores.jsonl"
HIDDEN_REUSE = SOURCE_ROOT / "confidence_probe" / "artifacts" / "hidden" / "reuse_manifest.jsonl"
HIDDEN_CAPTURE = SOURCE_ROOT / "confidence_probe" / "artifacts" / "hidden" / "capture_results.jsonl"

EXPECTED_PRELOCK_INPUTS = {
    "probe_train_manifest": (TRAIN_MANIFEST, 1112, "7b498813302b1c2223aeb8f0eb4a10f0c3ed110e15ac1d0fe905d93d0b67102f"),
    "phase1_confidence_joined": (CONFIDENCE_JOINED, 1212, "0264871a0cecaccf37a58b00ac34e059090ad641a8bdfe2769dea953b7cad7cc"),
    "unimodal_scores": (UNIMODAL_SCORES, 962, "a23e9912ef672490937c7e80f7ef3f520a10899ecebb04904d20f7bc66818fec"),
    "hidden_reuse_index": (HIDDEN_REUSE, 1212, "b61ff6db591dd1998bc5d620d8a7f84f0d2ab1a649e2bf5d0ff64252a8dedaff"),
    "hidden_capture_index": (HIDDEN_CAPTURE, 1212, "f975a7dc2cfd0a47306c446a9a92c1fcb7d53d3ef969e7823b83c9b0ab5585c0"),
}
EXPECTED_SPLIT_AUDIT_SHA256 = "6a3b500fd292eeff0e8add619d189760f30796a9c918b483b2931c824c9f9ca5"
EXPECTED_TEST_ROWS = 100
EXPECTED_TEST_FAMILIES = 50
EXPECTED_TEST_SHA256 = "97e7726e40b6df67661d1649306f39e76a4dc8bfedfe9771451ac458964adf68"

SEED = 42
AUDIT_OUTER_FOLD = 0
STEERING_POSITION = "P1_LAT"
PANL_POSITION = "P1_PANL"
SAC_POSITION = "P1_SAC"
STEERING_LAYERS = (10, 12, 14, 16)
PANL_LAYER = 18
ALPHAS = (-10.0, -2.0, 0.0, 2.0, 10.0)
DIRECTIONS = (
    "confidence_raw",
    "confidence_perp_difficulty",
    "confidence_perp_sa",
    "confidence_perp_difficulty_sa",
    "within_answer_shuffled_perp_difficulty_sa",
)
TRUE_DIRECTIONS = DIRECTIONS[:4]
SA_ORTHOGONAL_DIRECTIONS = DIRECTIONS[2:]
PRIMARY_DIRECTION = "confidence_perp_sa"
SMOKE_LAYERS = (12, 14)
SMOKE_ALPHAS = (-2.0, 0.0, 2.0)
NULL_ALPHAS = (-10.0, -2.0, 2.0, 10.0)
BOOTSTRAP_REPEATS = 2000
SMOKE_BOOTSTRAP_REPEATS = 200
PERMUTATION_REPEATS = 2000
MAX_SMOKE_ROUNDS = 5
SMOKE_FAMILY_COUNT = 4
HIDDEN_SIZE = 3584
HIDDEN_DEFINITION = "decoder_block_output_pre_final_norm"
CANONICAL_COLORS = ("black", "blue", "brown", "cyan", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow")

RIDGE_ALPHA_GRID = tuple(10.0 ** exponent for exponent in range(-4, 5))
PROTOCOL_VERSION = 3
TARGET_DEFINITION = "G_L=log((C_i+1e-12)/(1-C_i+1e-12))-log((C_t+1e-12)/(1-C_t+1e-12))"
REMOVED_COSINE_LIMIT = 1e-5
SA_NUMERICAL_CHANGE_LIMIT = 1e-6
RETAINED_NORM_MIN = 0.20
CLEANLINESS_PASS_LIMIT = 0.20
CLEANLINESS_WARNING_LIMIT = 0.25
NUISANCE_MEAN_REDUCTION_MIN = 0.20
PANL_SA_PEARSON_MIN = 0.5
PANL_SA_R2_MIN = 0.2
CONFIDENCE_PEARSON_MIN = 0.30
CONFIDENCE_R2_MIN = 0.0
NULL_INITIAL_REPEATS = 20
NULL_MAX_REPEATS = 99
NULL_EXPAND_P_THRESHOLD = 0.10
VECTOR_NORM_FRACTION = float(VECTOR_NORM_FRACTION)

EXPECTED_FORMAL_TRIALS = 100 * len(STEERING_LAYERS) * len(DIRECTIONS) * len(ALPHAS)
EXPECTED_FORMAL_MAIN_FORWARDS = 100 + 100 * len(STEERING_LAYERS) + 100 * len(STEERING_LAYERS) * len(DIRECTIONS) * (len(ALPHAS) - 1)
EXPECTED_FORMAL_20_NULL_FORWARDS = EXPECTED_FORMAL_MAIN_FORWARDS + 100 * NULL_INITIAL_REPEATS * len(NULL_ALPHAS)
EXPECTED_FORMAL_99_NULL_FORWARDS = EXPECTED_FORMAL_MAIN_FORWARDS + 100 * NULL_MAX_REPEATS * len(NULL_ALPHAS)
