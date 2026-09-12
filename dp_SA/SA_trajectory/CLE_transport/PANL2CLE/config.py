from __future__ import annotations

from pathlib import Path

from dp_SA.config import MODEL_PATH, ROOT

PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_PARENT = PACKAGE_ROOT / "output"
FORMAL_ROOT = OUTPUT_PARENT / "formal"

PROBE_ROOT = ROOT / "dp_SA" / "SA_probe" / "output"
AUDIT_MANIFEST = PROBE_ROOT / "artifacts/manifests/audit_manifest.jsonl"
CONSTRUCTION_MANIFEST = PROBE_ROOT / "artifacts/manifests/construction_manifest.jsonl"
CANDIDATE_MANIFEST = ROOT / "dp_SA/answer_matched_lat_steering/output/lat_panl_comparison/artifacts/manifests/candidate_manifest.jsonl"
HISTORICAL_CAPTURE = ROOT / "dp_SA/outputs/capture/results.jsonl"

CLE_PROBE_ROOT = ROOT / "dp_SA/confidence_steering/trajectory/output/results"
CLE_PROBE_INDEX = CLE_PROBE_ROOT / "artifacts/probes/probe_index.jsonl"
CLE_PROBE_TRAIN = CLE_PROBE_ROOT / "artifacts/manifests/construction_manifest.jsonl"
CLE_PROBE_CONFIG = CLE_PROBE_ROOT / "artifacts/config_and_fingerprint.json"

LAYERS = (12, 15, 18, 21)
CLE_LAYER = 22
WINDOW_SIZE = 8
WINDOWS = ("W1", "W2", "W3", "W4", "W5", "W6")
SMOKE_CELLS = tuple((window, 18) for window in WINDOWS) + (("W5", 12), ("W5", 15), ("W5", 21))
RECIPIENTS_PER_SIDE = 25
SEED = 42
BOOTSTRAP_REPEATS = 2000
SMOKE_BOOTSTRAP_REPEATS = 200
HIDDEN_SIZE = 3584
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
PARITY_ATOL = 1e-6

HIDDEN_DEFINITION = "decoder_block_output_pre_final_norm"
SWAP_SITE = "decoder_block_output_post_mlp_residual"


def require_output_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    try:
        root.relative_to(OUTPUT_PARENT.resolve())
    except ValueError as exc:
        raise ValueError(f"Output must remain inside {OUTPUT_PARENT.resolve()}: {root}") from exc
    return root

