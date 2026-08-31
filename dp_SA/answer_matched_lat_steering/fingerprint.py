from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from dp_SA.prompts import PHASE1_TEMPLATE, SA_PREFILL

from .config import (
    ALPHAS, BOOTSTRAP_REPEATS, DIRECTIONS, EXPECTED_SOURCE_SHA256,
    HIDDEN_DEFINITION, INFERENCE_PATH, LAYERS, MODEL_PATH, POSITION,
    SEED, SMOKE_ALPHAS, SMOKE_LAYERS, VECTOR_NORM_FRACTION,
)
from .io_utils import atomic_json, canonical_hash, sha256_file


def package_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {path.name: sha256_file(path) for path in sorted(directory.glob("*.py")) if path.name != "__init__.py"}


def model_hashes() -> dict[str, str]:
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.json", "model.safetensors.index.json")
    return {name: sha256_file(MODEL_PATH / name) for name in names if (MODEL_PATH / name).is_file()}


def experiment_config(*, smoke: bool, manifest_fingerprints: dict[str, str]) -> dict[str, Any]:
    payload = {
        "format_version": 1, "smoke_only": smoke, "position": POSITION,
        "layers": list(SMOKE_LAYERS if smoke else LAYERS),
        "alphas": list(SMOKE_ALPHAS if smoke else ALPHAS),
        "directions": list(DIRECTIONS), "seed": SEED,
        "bootstrap_repeats": 200 if smoke else BOOTSTRAP_REPEATS,
        "vector_norm_fraction": VECTOR_NORM_FRACTION,
        "hidden_definition": HIDDEN_DEFINITION,
        "phase1_template_hash": canonical_hash(PHASE1_TEMPLATE),
        "sa_prefill_hash": canonical_hash(SA_PREFILL),
        "model_path": str(MODEL_PATH.resolve()), "model_processor_hashes": model_hashes(),
        "inference_sha256": sha256_file(INFERENCE_PATH),
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
        "manifest_fingerprints": manifest_fingerprints,
        "source_code": package_hashes(),
    }
    payload["fingerprint"] = canonical_hash(payload)
    return payload


def check_or_write(path: Path, payload: dict[str, Any], *, resume: bool) -> None:
    import json
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("fingerprint") != payload.get("fingerprint"):
            raise ValueError(f"Resume fingerprint mismatch: {path}")
        if not resume: raise FileExistsError(f"Stage output exists; use --resume: {path}")
    else:
        atomic_json(path, payload)
