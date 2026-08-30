from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from dp_SA.prompts import PHASE1_TEMPLATE, SA_PREFILL

from .config import (
    ALPHAS,
    ANCHORS,
    HIDDEN_DEFINITION,
    HISTORICAL_CAPTURE,
    HISTORICAL_CAPTURE_CONFIG,
    INFERENCE_PATH,
    LAYERS,
    MODEL_PATH,
    POSITION_DEFINITION_VERSION,
    POSITIONS,
    SEED,
    SMOKE_ALPHAS,
    SMOKE_LAYERS,
    VECTOR_NORM_FRACTION,
)
from .io_utils import canonical_hash, sha256_file
from .manifests import manifest_fingerprint


def _existing_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in paths if path.is_file()}


def input_fingerprints(construction: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]]) -> dict[str, Any]:
    model_files = [MODEL_PATH / name for name in (
        "config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
        "chat_template.json", "model.safetensors.index.json",
    )]
    source_paths = [
        Path(__file__).resolve().parent / name
        for name in ("config.py", "positions.py", "manifests.py", "capture.py", "vectors.py", "run.py", "analyze.py")
    ]
    return {
        "model_path": str(MODEL_PATH.resolve()),
        "model_processor_files": _existing_hashes(model_files),
        "inference_sha256": sha256_file(INFERENCE_PATH),
        "historical_capture_sha256": sha256_file(HISTORICAL_CAPTURE),
        "historical_capture_config_sha256": sha256_file(HISTORICAL_CAPTURE_CONFIG),
        "construction_fingerprint": manifest_fingerprint(construction),
        "test_fingerprint": manifest_fingerprint(test),
        "phase1_template_hash": canonical_hash(PHASE1_TEMPLATE),
        "sa_prefill_hash": canonical_hash(SA_PREFILL),
        "source_code": {path.name: sha256_file(path) for path in source_paths if path.is_file()},
    }


def experiment_config(construction: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]], *, smoke: bool) -> dict[str, Any]:
    payload = {
        "format_version": 1,
        "smoke": smoke,
        "positions": list(POSITIONS),
        "position_order_with_sac": [*POSITIONS, "P1_SAC"],
        "position_definition_version": POSITION_DEFINITION_VERSION,
        "anchors": ANCHORS,
        "layers": list(SMOKE_LAYERS if smoke else LAYERS),
        "alphas": list(SMOKE_ALPHAS if smoke else ALPHAS),
        "seed": SEED,
        "vector_norm_fraction": VECTOR_NORM_FRACTION,
        "hidden_definition": HIDDEN_DEFINITION,
        "construction_count": len(construction),
        "test_count": len(test),
        "inputs": input_fingerprints(construction, test),
    }
    payload["fingerprint"] = canonical_hash(payload)
    return payload


def check_or_write_config(path: Path, payload: dict[str, Any], *, resume: bool) -> None:
    import json

    from .io_utils import atomic_json

    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("fingerprint") != payload.get("fingerprint"):
            raise ValueError(f"Config fingerprint mismatch: {path}")
        if not resume:
            raise FileExistsError(f"Stage config exists; use --resume: {path}")
    else:
        atomic_json(path, payload)

