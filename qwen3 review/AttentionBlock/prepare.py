from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from dp_SA.io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file
from Steering.contracts import ensure_fingerprinted_config

from .config import (
    BOOTSTRAP_REPEATS, CAPTURE_ROOT, EXPERIMENTS, MODEL_PATH, SEED,
    TEST_PER_SIDE, default_output,
)
from .eager_selection import ensure_eager_manifest


MODEL_FILES = (
    "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.json",
    "preprocessor_config.json", "model.safetensors.index.json",
)


def implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        path.name: sha256_file(path)
        for path in sorted(root.glob("*.py"))
        if path.name != "__init__.py"
    }


def prepare(
    *, experiment: str, output_root: Path | None = None, smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    root = Path(output_root or default_output(experiment, smoke)).resolve()
    capture_config = __import__("json").loads((CAPTURE_ROOT / "config.json").read_text(encoding="utf-8"))
    formal, selection = ensure_eager_manifest()
    if smoke:
        selected = [
            next(row for row in formal if row["test_side"] == "text_side"),
            next(row for row in formal if row["test_side"] == "image_side"),
        ]
    else:
        selected = formal
    spec = EXPERIMENTS[experiment]
    model_hashes = {name: sha256_file(MODEL_PATH / name) for name in MODEL_FILES}
    if model_hashes != capture_config.get("model_fingerprint"):
        raise ValueError("Current Qwen3 model/processor files differ from capture fingerprint")
    payload = {
        "format_version": 1,
        "experiment": experiment,
        "smoke": bool(smoke),
        "model_path": str(MODEL_PATH.resolve()),
        "capture_root": str(CAPTURE_ROOT.resolve()),
        "capture_fingerprint": capture_config["fingerprint"],
        "model_fingerprint": model_hashes,
        "selected_case_count": len(selected),
        "formal_case_count": len(formal),
        "manifest_fingerprint": canonical_hash(selected),
        "formal_manifest_fingerprint": canonical_hash(formal),
        "selection": selection,
        "seed": SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "attention_implementation": "eager",
        "conditions": ["C1_main_block", "C2_source_plus_1_control"],
        "spec": {
            "query": spec.query, "main_source": spec.main_source,
            "control_source": spec.control_source,
            "windows": [list(window) for window in (spec.windows[:1] if smoke else spec.windows)],
            "window_semantics": "inclusive",
        },
        "implementation_sha256": implementation_hashes(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip(),
    }
    root.mkdir(parents=True, exist_ok=True)
    config = ensure_fingerprinted_config(root / "run_config.json", payload, resume=resume, label="AttentionBlock")
    manifest_path = root / "artifacts" / "manifests" / "test_manifest.jsonl"
    if not manifest_path.exists():
        atomic_jsonl(manifest_path, selected)
        atomic_json(root / "artifacts" / "manifests" / "selection_summary.json", selection)
    elif canonical_hash(load_jsonl(manifest_path)) != canonical_hash(selected):
        raise ValueError("Existing test manifest differs from frozen selection")
    return {"status": "complete", "output_root": str(root), "case_count": len(selected),
            "fingerprint": config["fingerprint"], "resumed": resume}
