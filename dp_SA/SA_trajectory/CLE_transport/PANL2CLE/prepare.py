from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch

from dp_SA.confidence_steering.processor import load_fast_processor, processor_identity
from dp_SA.prompts import PHASE1_TEMPLATE, SA_PREFILL, phase1_prompt
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.token_positions import locate_image_pad_span

from .config import (
    AUDIT_MANIFEST, CANDIDATE_MANIFEST, CLE_LAYER, CLE_PROBE_CONFIG, CLE_PROBE_INDEX,
    CLE_PROBE_ROOT, CLE_PROBE_TRAIN, CONSTRUCTION_MANIFEST, HISTORICAL_CAPTURE, LAYERS,
    MODEL_PATH, OUTPUT_PARENT, WINDOWS, require_output_root,
)
from .io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file
from .matching import match_donors
from .selection import select_recipients, side_from_class
from .windows import locate_swap_windows


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image", "image": str(Path(row["image_path"]).resolve())},
                                              {"type": "text", "text": str(row["phase1_prompt"])}]},
            {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]}]


def package_hashes() -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(Path(__file__).resolve().parent.glob("*.py"))}


def source_hashes() -> dict[str, str]:
    paths = (AUDIT_MANIFEST, CONSTRUCTION_MANIFEST, CANDIDATE_MANIFEST, HISTORICAL_CAPTURE,
             CLE_PROBE_INDEX, CLE_PROBE_TRAIN, CLE_PROBE_CONFIG, MODEL_PATH / "config.json",
             MODEL_PATH / "tokenizer.json", MODEL_PATH / "preprocessor_config.json")
    for path in paths:
        if not path.is_file(): raise FileNotFoundError(path)
    return {str(path.resolve()): sha256_file(path) for path in paths}


def _raw_answer(row: dict[str, Any]) -> str:
    value = str(row.get("phase1_inserted_raw_answer") or row.get("phase0_raw_answer") or "")
    if not value or "\n" in value or "\r" in value: raise ValueError(f"Invalid fixed answer: {row.get('case_id')}")
    return value


def _validate_prompt(row: dict[str, Any]) -> None:
    answer = _raw_answer(row)
    expected = phase1_prompt(str(row["question"]), str(row["text_clue"]), answer)
    if str(row["phase1_prompt"]) != expected: raise ValueError(f"Frozen Phase-1 prompt mismatch: {row['case_id']}")


def enrich_token_metadata(processor: Any, row: dict[str, Any]) -> dict[str, Any]:
    _validate_prompt(row); wire = _messages(row)
    rendered = render_continued_assistant(processor, wire, SA_PREFILL)
    inputs = prepare_multimodal_inputs(processor, wire, rendered, device=torch.device("cpu"))
    tokenizer = processor.tokenizer; windows = locate_swap_windows(tokenizer, rendered, inputs)
    ids = inputs.input_ids.detach().cpu().reshape(-1).tolist()
    image = locate_image_pad_span(tokenizer, ids)
    return {
        **row, "phase0_raw_answer": _raw_answer(row), "template_sha256": hashlib.sha256(PHASE1_TEMPLATE.encode()).hexdigest(),
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(), "sequence_length": len(ids),
        "image_token_count": int(image["span"][1] - image["span"][0]), "windows": windows,
    }


def _resolve_probe() -> dict[str, Any]:
    matches = [row for row in load_jsonl(CLE_PROBE_INDEX) if row.get("target") == "final_soft_sa"
               and row.get("position") == "P1_CLASS_LIST_END" and int(row.get("layer", -1)) == CLE_LAYER]
    if len(matches) != 1: raise ValueError(f"Expected one L{CLE_LAYER} CLE probe, found {len(matches)}")
    row = matches[0]; path = CLE_PROBE_ROOT / str(row["probe_file"])
    if sha256_file(path) != row.get("probe_sha256") or not row.get("readout_reliable"):
        raise ValueError("CLE probe hash/reliability check failed")
    config = json.loads(CLE_PROBE_CONFIG.read_text())
    if config.get("hidden_definition") != "decoder_block_output_pre_final_norm": raise ValueError("CLE probe hidden definition mismatch")
    return {**row, "absolute_path": str(path.resolve())}


def _cle_eligibility(recipients: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    training = load_jsonl(CLE_PROBE_TRAIN)
    pools = {field: {str(row[field]) for row in training} for field in ("family_id", "item_id", "image_sha256")}
    output = []
    for row in recipients:
        reasons = [f"{field}_overlap" for field, values in pools.items() if str(row[field]) in values]
        output.append({"case_id": str(row["case_id"]), "eligible": not reasons, "exclusion_reasons": reasons})
    return output


def _ensure_layout(root: Path) -> None:
    for relative in ("artifacts/manifests", "artifacts/positions", "artifacts/donor_hidden", "artifacts/trials",
                     "artifacts/diagnostics", "tables", "figures", "progress", "logs"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def prepare(root: str | Path, *, resume: bool = False) -> dict[str, Any]:
    root = require_output_root(root); _ensure_layout(root)
    config_path = root / "run_config.json"
    sources = source_hashes(); code = package_hashes()
    semantic = {"source_hashes": sources, "code_hashes": code, "layers": list(LAYERS), "windows": list(WINDOWS),
                "template_sha256": hashlib.sha256(PHASE1_TEMPLATE.encode()).hexdigest(), "recipient_counts": {"high_image": 25, "high_text": 25}}
    fingerprint = canonical_hash(semantic)
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old.get("fingerprint") != fingerprint: raise ValueError("Resume fingerprint mismatch")
        if not resume: raise FileExistsError(f"Output exists; use --resume: {root}")
        return {"status": "complete", "resumed": True, "fingerprint": fingerprint, "root": str(root)}

    audit = load_jsonl(AUDIT_MANIFEST); construction_raw = load_jsonl(CONSTRUCTION_MANIFEST); candidates = load_jsonl(CANDIDATE_MANIFEST)
    if (len(audit), len(construction_raw)) != (230, 882): raise ValueError("Frozen SA-probe split cardinality changed")
    candidate_by_id = {str(row["case_id"]): row for row in candidates}
    construction = []
    for original in construction_raw:
        case_id = str(original["case_id"])
        if case_id not in candidate_by_id: continue  # class 4 or invalid/noncanonical
        row = dict(candidate_by_id[case_id]); row["sa_side"] = str(row.get("sa_side") or side_from_class(row["argmax_hard_class"]))
        construction.append(row)
    donor_cells = Counter((str(row["phase0_raw_answer"]), str(row["sa_side"])) for row in construction)
    feasible_answers = {answer for answer, _side in donor_cells if donor_cells[answer, "high_image"] and donor_cells[answer, "high_text"]}
    selected, selection_audit = select_recipients(candidates, audit, construction_raw, allowed_answers=feasible_answers)
    processor = load_fast_processor(); identity = processor_identity(processor)
    if not identity.get("is_fast") or not str(identity.get("image_processor_class", "")).endswith("Qwen2VLImageProcessorFast"):
        raise ValueError(f"Fast processor gate failed: {identity}")
    recipients = [enrich_token_metadata(processor, row) for row in selected]
    needed_answers = {row["phase0_raw_answer"] for row in recipients}
    donor_candidates = [row for row in construction if _raw_answer(row) in needed_answers]
    donors = [enrich_token_metadata(processor, row) for row in donor_candidates]
    pairs, matching_audit = match_donors(recipients, donors)
    used_ids = {row["donor_case_id"] for row in pairs}; used_donors = [row for row in donors if str(row["case_id"]) in used_ids]
    eligibility = _cle_eligibility(recipients); probe = _resolve_probe()
    by_id = {row["case_id"]: row for row in recipients}
    window_audit = []
    for row in recipients + used_donors:
        for name in WINDOWS:
            window_audit.append({"case_id": row["case_id"], "role": "recipient" if row["case_id"] in by_id else "donor", **row["windows"][name]})
    atomic_jsonl(root / "artifacts/manifests/recipient_manifest.jsonl", recipients)
    atomic_jsonl(root / "artifacts/manifests/donor_manifest.jsonl", used_donors)
    atomic_jsonl(root / "artifacts/manifests/donor_matching.jsonl", pairs)
    atomic_jsonl(root / "artifacts/manifests/cle_probe_eligibility.jsonl", eligibility)
    atomic_jsonl(root / "artifacts/positions/window_audit.jsonl", window_audit)
    atomic_json(root / "artifacts/diagnostics/selection_audit.json", selection_audit)
    atomic_json(root / "artifacts/diagnostics/matching_audit.json", matching_audit)
    atomic_json(root / "artifacts/diagnostics/cle_probe.json", probe)
    config = {**semantic, "fingerprint": fingerprint, "processor_identity": identity, "selection": selection_audit,
              "matching": matching_audit, "cle_probe": probe, "formal_recipient_forwards": 2450,
              "donor_cache_forwards": len(used_donors), "actual_formal_forwards": 2450 + len(used_donors)}
    atomic_json(config_path, config)
    result = {"status": "complete", "resumed": False, "fingerprint": fingerprint, "root": str(root),
              "recipient_count": len(recipients), "donor_count": len(used_donors), "pair_count": len(pairs)}
    atomic_json(root / "progress/prepare.json", result)
    return result
