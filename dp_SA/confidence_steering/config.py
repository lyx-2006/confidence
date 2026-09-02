from __future__ import annotations

from pathlib import Path

from dp_SA.config import FLOAT_TOLERANCE, INFERENCE_PATH, MODEL_PATH, ROOT, VECTOR_NORM_FRACTION

PACKAGE_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PACKAGE_ROOT / "output" / "results"
SMOKE_PARENT = PACKAGE_ROOT / "output" / "smoke_tmp"
SOURCE_ROOT = ROOT / "dp_SA" / "unimodal_logit_confidence" / "output" / "results"

TRAIN_MANIFEST = SOURCE_ROOT / "shared" / "manifests" / "probe_train_manifest.jsonl"
TEST_MANIFEST = SOURCE_ROOT / "shared" / "manifests" / "test_manifest.jsonl"
CONFIDENCE_JOINED = SOURCE_ROOT / "unimodal_confidence" / "artifacts" / "predictions" / "phase1_confidence_joined.jsonl"
UNIMODAL_SCORES = SOURCE_ROOT / "unimodal_confidence" / "artifacts" / "calibrated_scores" / "unimodal_scores.jsonl"
HIDDEN_REUSE = SOURCE_ROOT / "confidence_probe" / "artifacts" / "hidden" / "reuse_manifest.jsonl"
HIDDEN_CAPTURE = SOURCE_ROOT / "confidence_probe" / "artifacts" / "hidden" / "capture_results.jsonl"

EXPECTED_INPUTS = {
    "probe_train_manifest": (TRAIN_MANIFEST, 1112, "7b498813302b1c2223aeb8f0eb4a10f0c3ed110e15ac1d0fe905d93d0b67102f"),
    "test_manifest": (TEST_MANIFEST, 100, "97e7726e40b6df67661d1649306f39e76a4dc8bfedfe9771451ac458964adf68"),
    "phase1_confidence_joined": (CONFIDENCE_JOINED, 1212, "0264871a0cecaccf37a58b00ac34e059090ad641a8bdfe2769dea953b7cad7cc"),
    "unimodal_scores": (UNIMODAL_SCORES, 962, "a23e9912ef672490937c7e80f7ef3f520a10899ecebb04904d20f7bc66818fec"),
    "hidden_reuse_index": (HIDDEN_REUSE, 1212, "b61ff6db591dd1998bc5d620d8a7f84f0d2ab1a649e2bf5d0ff64252a8dedaff"),
    "hidden_capture_index": (HIDDEN_CAPTURE, 1212, "f975a7dc2cfd0a47306c446a9a92c1fcb7d53d3ef969e7823b83c9b0ab5585c0"),
}

SEED = 42
POSITION = "P1_LAT"
LAYERS = (8, 10, 12, 14, 16)
ALPHAS = (-10.0, -2.0, 0.0, 2.0, 10.0)
DIRECTIONS = ("residual_confidence_loao", "within_answer_shuffled")
SMOKE_LAYERS = (8, 14)
SMOKE_ALPHAS = (-2.0, 0.0, 2.0)
BOOTSTRAP_REPEATS = 2000
SMOKE_BOOTSTRAP_REPEATS = 200
MAX_SMOKE_ROUNDS = 5
HIDDEN_DEFINITION = "decoder_block_output_pre_final_norm"
CANONICAL_COLORS = ("black", "blue", "brown", "cyan", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow")
FORBIDDEN_DIRECTION_FIELDS = {
    "soft_sa_image_score", "argmax_hard_class", "class_logits", "class_probabilities",
    "final_sa", "hard_sa", "panl_probe_sa", "prediction", "predicted_confidence",
    "delta_soft_sa", "steering", "swap", "patching",
}
EXPECTED_FORMAL_FORWARDS = 4500

