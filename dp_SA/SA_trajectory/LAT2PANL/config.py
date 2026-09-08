from __future__ import annotations

from pathlib import Path

from dp_SA.config import INFERENCE_PATH, MODEL_PATH, ROOT

PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "output"
RESULTS_ROOT = OUTPUT_ROOT / "results"
SMOKE_ROOT = OUTPUT_ROOT / "smoke"

ANSWER_MATCHED_ROOT = ROOT / "dp_SA" / "answer_matched_lat_steering" / "output" / "lat_panl_comparison"
MANIFEST_PATH = ANSWER_MATCHED_ROOT / "artifacts" / "manifests" / "test_manifest.jsonl"
CANDIDATE_MANIFEST_PATH = ANSWER_MATCHED_ROOT / "artifacts" / "manifests" / "candidate_manifest.jsonl"
FOLD_ASSIGNMENTS_PATH = ANSWER_MATCHED_ROOT / "artifacts" / "manifests" / "fold_assignments.jsonl"
CONSTRUCTION_CELLS_PATH = ANSWER_MATCHED_ROOT / "artifacts" / "manifests" / "construction_family_cells.jsonl"
VECTOR_METADATA_PATH = ANSWER_MATCHED_ROOT / "artifacts" / "vectors" / "vector_metadata.json"
HISTORICAL_CLEAN_PATH = ANSWER_MATCHED_ROOT / "artifacts" / "diagnostics" / "clean_capture.jsonl"
HISTORICAL_LOG_PATH = ANSWER_MATCHED_ROOT / "progress" / "steering_gpu0.log"

PROBE_ROOT = ROOT / "dp_SA" / "confidence_steering" / "trajectory" / "output" / "results"
PROBE_INDEX_PATH = PROBE_ROOT / "artifacts" / "probes" / "probe_index.jsonl"
PROBE_CONSTRUCTION_PATH = PROBE_ROOT / "artifacts" / "manifests" / "construction_manifest.jsonl"
PROBE_CONFIG_PATH = PROBE_ROOT / "artifacts" / "config_and_fingerprint.json"
PREPROCESSOR_CONFIG_PATH = MODEL_PATH / "preprocessor_config.json"

EXPECTED_SHA256 = {
    "manifest": "6a6d6c88b8b1b120490a5723b6c3ee6e7ec40af4047b43096637a44dcf12bca4",
    "vector_metadata": "8744df9091cd8136843fbfd963a00d4d19bc7d09bf3516927ce8236f01bcab16",
    "probe": "aa78bd70cbabf8ce0c2d9cd4d9ec7c9aeb75eb25c7ba46a0eb0d54809ca116da",
    "probe_construction": "525432e11a7ae135ce1a362e00ff3cbaa4660a5affafbe49d46576a75b4c0fd5",
    "preprocessor_config": "f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0",
    "historical_log": "4186ac6268209f304035351304ec711bf701abeb83d043ebd24c772b210f5642",
    "historical_clean": "c137d2d8325e8c18a250f781aa5982b24850a78fbe509a09a94404a79562906c",
}

VECTOR_FILE_SHA256 = {
    0: "3eb2b57c8f1952f92681d770d14763f5cba3c2a0f875e083f8371f1bf2a069e6",
    1: "d9bf9b3e7c0337ff912ce442b650bb66277c6fcaacfb2548b148effd618e150e",
    2: "7266ebb3984cacc8ac7d01a1f992cc393504fd93f8b5d68f31b6fe6a19023e80",
    3: "39fd347aca8067c5b883d925cfebe9a3b0a7cf14bb4e365472f1c041a7cf74bf",
    4: "53b3fcac1b8a958187d1bee9d10ed5b8ac6c555d2e9dd8ce90d09314d1ec6654",
    5: "852e16c763387475d2ffe2575df5ddf6bedb291a57aff73e5c2009898a334c6a",
    6: "acc3329b055006710945ee5dd3448ae129e6dddf1fe0bd24737e4f5d95ee067b",
    7: "4e91ae564a756c414f3df5dd51723f96115305f6cd740e4c7da8c03a9622b06d",
    8: "689972f6ca13e84e81438d8f82b9084a54a49afec909fda88809d060cb5a1236",
    9: "de3402cbf3f42b0d1cfc3c6211fb686790e07043971a6ba4b24c6ecfed18b212",
    10: "6e62af8eef364dc6f2907cd6528b3ffd3843b25463feec7ca0f1ad18588f52a2",
    11: "82aa1ef554810ab589c8f87d698a42609abc2507e33c24e3f75b0e97c85f10b3",
    12: "ed6bca614b88bd81231df841de28f31c76b72560ba13f2a7be884c64dae35f7f",
    13: "1185163a7cdb89291f0048c92af7197936493a241a506ea61abb3a2872ec9e15",
    14: "4f044f354a0b729ec8920d815baed7d519d615cbd48d3b4fabd0c2952c0ddb73",
}

LAT_LAYER = 14
PANL_MEDIATOR_LAYERS = (14, 15, 16, 17, 18)
ALPHAS = (-10.0, -2.0, 0.0, 2.0, 10.0)
NONZERO_ALPHAS = tuple(value for value in ALPHAS if value != 0)
DOSES = (2.0, 10.0)
SMOKE_LAYERS = (14, 15, 18)
SMOKE_ALPHAS = (-2.0, 0.0, 2.0)
SMOKE_FAMILIES = (
    "family_479f016b66f001c1",
    "family_4d6def316db95c14",
    "family_b64c6a771503d359",
    "family_ccd18627d4e12186",
)
CANONICAL_ANSWERS = (
    "black", "blue", "brown", "cyan", "gray", "green",
    "orange", "pink", "purple", "red", "white", "yellow",
)
CONFIRMATORY_ANSWERS = tuple(value for value in CANONICAL_ANSWERS if value != "blue")
BOOTSTRAP_REPEATS = 2000
SMOKE_BOOTSTRAP_REPEATS = 200
SEED = 42
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
HIDDEN_DEFINITION = "decoder_block_output_pre_final_norm"
EXPECTED_FORMAL_CASES = 174
EXPECTED_CLE_ELIGIBLE = 73
FORMAL_FORWARD_COUNT = 7830
SMOKE_FORWARD_COUNT = 360
