from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dp_SA.positions import locate_phase1_positions as locate_t0_historical
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import run_hooked_forward

from .config import AUDIT_MANIFEST, FLOAT_ATOL, POSITIONS, PROBE_LAYERS, TRAJECTORY_ROOT
from .io_utils import array_hash, atomic_json, canonical_hash, load_jsonl, sha256_file
from .runtime import class_token_ids, load_inference, messages
from .scoring import numeric_score
from .sources import audit_frozen_sources
from .templates import TEMPLATES
from .capture import select_smoke_audit


def _historical_hidden_sources(case_id: str) -> dict[str, dict[str, Any]]:
    rows = {str(row["case_id"]): row for row in load_jsonl(TRAJECTORY_ROOT / "artifacts/clean_hidden/reuse_manifest.jsonl")}
    if case_id not in rows: raise ValueError(f"T0 hidden reuse manifest lacks {case_id}")
    sources = dict(rows[case_id].get("cell_sources", {}))
    capture_rows = {str(row["case_id"]): row for row in load_jsonl(TRAJECTORY_ROOT / "artifacts/clean_hidden/capture_manifest.jsonl")}
    if case_id in capture_rows:
        relative = capture_rows[case_id].get("hidden_file") or capture_rows[case_id].get("delta_file")
        if not relative: raise ValueError(f"T0 capture row has no hidden/delta file: {case_id}")
        path = TRAJECTORY_ROOT / relative
        with np.load(path) as payload:
            for key in payload.files: sources.setdefault(key, {"path": str(path.resolve()), "tensor_sha256": array_hash(payload[key])})
    return sources


def _load_source_tensor(info: dict[str, Any], key: str) -> np.ndarray:
    path = Path(info["path"])
    if not path.is_file(): raise FileNotFoundError(path)
    with np.load(path) as payload:
        if key not in payload.files: raise KeyError(f"{key} absent from {path}")
        value = np.asarray(payload[key], dtype=np.float16)
    expected = info.get("tensor_sha256")
    if expected and array_hash(value) != expected: raise ValueError(f"Historical hidden tensor hash changed: {key}")
    return value


def run_preflight(root: Path, *, smoke_case_count: int = 4, resume: bool = False) -> dict[str, Any]:
    path = root / "artifacts/diagnostics/preflight.json"
    frozen = audit_frozen_sources(); fingerprint = canonical_hash({"inventory": frozen["inventory_fingerprint"], "smoke_case_count": smoke_case_count})
    if path.is_file():
        old = json.loads(path.read_text())
        if old.get("fingerprint") != fingerprint: raise ValueError("Preflight resume fingerprint mismatch")
        if resume and old.get("status") == "passed": return {**old, "resumed_noop": True}
        if not resume: raise FileExistsError(f"Preflight exists; use --resume: {path}")
    inference, modules, tokenizer, device, processor = load_inference()
    audit = select_smoke_audit(load_jsonl(AUDIT_MANIFEST),smoke_case_count); results = []
    requested = {position: PROBE_LAYERS for position in POSITIONS}
    ids = class_token_ids(tokenizer)
    for record in audit:
        wire, answer = messages(record, TEMPLATES["T0"])
        rendered = render_continued_assistant(inference.processor, wire, "**Source Attribution**:")
        inputs = prepare_multimodal_inputs(inference.processor, wire, rendered, device=device)
        current = __import__("dp_SA.prompt_check.positions", fromlist=["locate_template_positions"]).locate_template_positions(tokenizer, rendered, inputs, answer, TEMPLATES["T0"])
        historical_locator = locate_t0_historical(tokenizer, rendered, inputs, answer)
        for position in POSITIONS:
            for field in ("processed_index", "token_id"):
                if int(current[position][field]) != int(historical_locator[position][field]): raise ValueError(f"T0 locator parity failed: {record['case_id']} {position} {field}")
                if position in record["positions"] and int(current[position][field]) != int(record["positions"][position][field]): raise ValueError(f"T0 historical position parity failed: {record['case_id']} {position} {field}")
        positions = {name: int(current[name]["processed_index"]) for name in POSITIONS}
        forward = run_hooked_forward(inference.model, inputs, modules, positions, logits_positions=[positions["P1_SAC"]])
        score = numeric_score([float(forward.logits_by_position[positions["P1_SAC"]][i]) for i in ids], token_ids=ids)
        logit_error = float(np.max(np.abs(np.asarray(score["class_logits"]) - np.asarray(record["class_logits"]))))
        soft_error = abs(float(score["canonical_soft_sa"]) - float(record["soft_sa_image_score"]))
        if logit_error > FLOAT_ATOL or soft_error > FLOAT_ATOL or int(score["canonical_hard_class"]) != int(record["argmax_hard_class"]): raise ValueError(f"T0 behavior parity failed: {record['case_id']}")
        sources = _historical_hidden_sources(str(record["case_id"])); hidden_max = 0.0
        for position, layers in requested.items():
            for layer in layers:
                key = f"{position}__L{layer}"
                historical = _load_source_tensor(sources[key], key)
                current_hidden = forward.hidden_by_name[position][layer].detach().float().cpu().numpy().astype(np.float16)
                hidden_max = max(hidden_max, float(np.max(np.abs(current_hidden.astype(np.float32) - historical.astype(np.float32)))))
        if hidden_max > FLOAT_ATOL: raise ValueError(f"T0 hidden parity failed: {record['case_id']} max={hidden_max}")
        results.append({"case_id": record["case_id"], "position_parity": True, "logits_max_abs_error": logit_error, "soft_sa_abs_error": soft_error, "hidden_max_abs_error": hidden_max, "passed": True})
    result = {"status": "passed", "fingerprint": fingerprint, "processor": processor, "cases": results, "frozen_audit": {k: v for k, v in frozen.items() if k not in {"inventory", "probe_rows"}}, "probe_rows": frozen["probe_rows"], "resumed_noop": False}
    atomic_json(path, result); return result
