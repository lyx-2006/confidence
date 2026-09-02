from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL

from .config import (
    CANONICAL_COLORS, HIDDEN_DEFINITION, LAYERS, MODEL_PATH, POSITION, RESULTS_ROOT,
    SEED, SMOKE_LAYERS, SOURCE_ROOT,
)
from .core import (
    answer_origin, build_vectors, family_answer_cells, input_inventory, oof_residualize,
    prepare_rows, smoke_subset, tail_assignments, validate_frozen_design,
)
from .io_utils import (
    array_hash, atomic_joblib, atomic_json, atomic_jsonl, atomic_npz, canonical_hash,
    check_fingerprint, ensure_layout, load_jsonl, sha256_file,
)
from .processor import load_frozen_processor


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "image", "image": str(Path(row["image_path"]).resolve())}, {"type": "text", "text": str(row["phase1_prompt"])}]},
        {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]},
    ]


def validate_positions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    # Preserve the fast tokenizer (needed for offset mappings) while replacing
    # the newer default fast image processor with the capture-era slow one.
    processor = load_frozen_processor()
    tokenizer = getattr(processor, "tokenizer", processor)
    for row in rows:
        messages = _messages(row); rendered = render_continued_assistant(processor, messages, SA_PREFILL)
        inputs = prepare_multimodal_inputs(processor, messages, rendered, device="cpu")
        located = locate_phase1_positions(tokenizer, rendered, inputs, str(row["phase0_raw_answer"]))
        lat = located[POSITION]
        frozen = row["positions"][POSITION]
        if any(lat[field] != frozen[field] for field in ("processed_index", "token_id", "token_text")):
            raise ValueError(f"LAT position changed: {row['case_id']}")
        if int(lat["processed_index"]) != int(located["phase1_answer_span"][1]) - 1:
            raise ValueError(f"LAT is not the last real answer token: {row['case_id']}")
    return {"status": "passed", "record_count": len(rows), "position": POSITION, "definition": "last real token of fixed answer"}


def _prepare_config(inventory: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    model_files = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.json"):
        path = MODEL_PATH / name
        if path.is_file(): model_files[name] = sha256_file(path)
    package=Path(__file__).resolve().parent; source_hashes={name:sha256_file(package/name) for name in ("config.py","core.py","io_utils.py","prepare.py","processor.py")}
    return {"format_version": 1, "experiment": "confidence_steering", "smoke_only": smoke, "seed": SEED,
            "position": POSITION, "layers": list(SMOKE_LAYERS if smoke else LAYERS), "hidden_definition": HIDDEN_DEFINITION,
            "inputs": {name: row["sha256"] for name, row in inventory.items()}, "model_processor_hashes": model_files,"source_code":source_hashes}


def _resume_complete(root: Path, fingerprint: str) -> dict[str, Any] | None:
    progress = root / "progress" / "prepare.json"
    required = [root / "artifacts/residualization/oof_predictions.jsonl", root / "artifacts/family_answer_cells/cells.jsonl", root / "artifacts/vectors/vector_metadata.json"]
    if not progress.is_file(): return None
    value = json.loads(progress.read_text())
    if value.get("config_fingerprint") != fingerprint: raise ValueError("Prepare resume fingerprint mismatch")
    if value.get("status") == "complete" and all(path.is_file() and path.stat().st_size for path in required):
        residual=json.loads((root/"artifacts/residualization/residualization_audit.json").read_text())
        for fold,digest in residual["models"].items():
            if sha256_file(root/f"artifacts/residualization/fold_models/fold_{fold}.joblib") != digest: raise ValueError("Residual model fingerprint mismatch")
        vectors=json.loads((root/"artifacts/vectors/vector_metadata.json").read_text())
        for layer,digest in vectors["files"].items():
            path=root/f"artifacts/vectors/P1_LAT__L{layer}.npz"
            if sha256_file(path) != digest: raise ValueError("Vector file fingerprint mismatch")
            import numpy as np
            with np.load(path) as payload:
                for row in (r for r in vectors["vectors"] if str(r["layer"])==str(layer)):
                    if array_hash(payload[row["scaled_key"]]) != row["scaled_hash"]: raise ValueError("Vector tensor fingerprint mismatch")
        return {**value, "resumed_noop": True}
    return None


def run_prepare(*, output_root: Path = RESULTS_ROOT, smoke: bool = False, resume: bool = False, revalidate_positions: bool = True) -> dict[str, Any]:
    root = ensure_layout(output_root); inventory = input_inventory()
    train_manifest = load_jsonl(Path(inventory["probe_train_manifest"]["path"])); test_manifest = load_jsonl(Path(inventory["test_manifest"]["path"]))
    joined = load_jsonl(Path(inventory["phase1_confidence_joined"]["path"])); scores = load_jsonl(Path(inventory["unimodal_scores"]["path"]))
    design = validate_frozen_design(train_manifest, test_manifest)
    mapping = {
        "C_t": "phase1_confidence_joined.text_fixed_answer_confidence", "C_i": "phase1_confidence_joined.image_fixed_answer_confidence",
        "L_t": "phase1_confidence_joined.text_fixed_answer_log_odds", "L_i": "phase1_confidence_joined.image_fixed_answer_log_odds", "G_L": "phase1_confidence_joined.G_L",
        "D_t": "unimodal_scores.entropy_difficulty via text_score_unique_key", "D_i": "unimodal_scores.entropy_difficulty via image_score_unique_key",
        "Hard": "condition == conflict_hard", "prior_bin": "probe_train_manifest.prior_bin",
        "answer_origin": "answer_matches_text/image -> follow_text/follow_image/neither_match", "fixed_answer_color": "phase0_normalized_answer, parity checked against joined.fixed_answer",
        "hidden": "hidden reuse/capture index by case_id + P1_LAT__L{layer}",
    }
    audit = {"status": "passed", "inventory": inventory, "field_mapping": mapping, "design": design}
    atomic_json(root / "artifacts/audits/input_audit.json", audit)
    config = _prepare_config(inventory, smoke=smoke); fingerprint = check_fingerprint(root / "progress/prepare_config.json", config, resume=resume)
    if resume:
        complete = _resume_complete(root, fingerprint)
        if complete is not None: return complete

    rows = prepare_rows(train_manifest, test_manifest, joined, scores)
    if smoke:
        train_rows, selected_test, selection = smoke_subset(rows, [*train_manifest, *test_manifest])
        test_ids = {r["case_id"] for r in selected_test}; test_rows = [r for r in test_manifest if r["case_id"] in test_ids]
    else:
        train_rows = [r for r in rows if r["split"] == "train"]; test_rows = list(test_manifest); selection = {"formal": True, "train_family_count": 128, "test_family_count": 50}
    selected_manifest = [r for r in train_manifest if r["case_id"] in {x["case_id"] for x in train_rows}] + test_rows
    position_audit = validate_positions(selected_manifest) if revalidate_positions else {"status": "skipped_for_unit_test", "record_count": len(selected_manifest)}
    atomic_json(root / "artifacts/audits/position_audit.json", position_audit); atomic_json(root / "artifacts/audits/smoke_selection.json", selection)

    oof, models = oof_residualize(train_rows)
    fold_manifest = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in oof: by_family[row["family_id"]].append(row)
    for family, members in sorted(by_family.items()):
        folds = {int(r["outer_fold"]) for r in members}
        if len(folds) != 1: raise ValueError("Family crossed nuisance folds")
        fold_manifest.append({"family_id": family, "fold": next(iter(folds)), "case_ids": sorted(r["case_id"] for r in members), "record_count": len(members),
                              "condition_counts": dict(Counter(r["condition"] for r in members)), "answer_origin_counts": dict(Counter(r["answer_origin"] for r in members)), "fixed_answer_color_counts": dict(Counter(r["fixed_answer_color"] for r in members))})
    atomic_jsonl(root / "artifacts/residualization/fold_manifest.jsonl", fold_manifest)
    atomic_jsonl(root / "artifacts/residualization/oof_predictions.jsonl", oof)
    atomic_jsonl(root / "artifacts/residualization/confidence_training_records.jsonl", oof)
    model_hashes = {}
    for fold, model in models.items():
        path = root / f"artifacts/residualization/fold_models/fold_{fold}.joblib"; atomic_joblib(path, model); model_hashes[str(fold)] = sha256_file(path)
    residual_audit = {"status": "passed", "record_count": len(oof), "family_count": len(by_family), "fold_count": 5,
                      "models": model_hashes, "features": ["D_t", "D_i", "Hard", "prior_bin", "answer_origin", "fixed_answer_color"],
                      "forbidden_fields_read": [], "all_records_have_exactly_one_oof_prediction": True,
                      "fingerprint": canonical_hash([{k: r[k] for k in ("case_id", "outer_fold", "G_L", "predicted_G_L_oof", "R_C")} for r in oof])}
    atomic_json(root / "artifacts/residualization/residualization_audit.json", residual_audit)

    layers = SMOKE_LAYERS if smoke else LAYERS; cells, cell_arrays = family_answer_cells(oof, layers=layers)
    for layer in layers: atomic_npz(root / f"artifacts/family_answer_cells/P1_LAT__L{layer}.npz", cell_arrays[int(layer)])
    atomic_jsonl(root / "artifacts/family_answer_cells/cells.jsonl", cells)
    assignments, eligibility = tail_assignments(cells)
    atomic_jsonl(root / "artifacts/family_answer_cells/eligibility_and_tails.jsonl", eligibility)
    atomic_json(root / "artifacts/family_answer_cells/assignments.json", assignments)
    recipient_answers = [str(r["phase0_normalized_answer"]) for r in test_rows]
    vectors, metadata = build_vectors(cells, cell_arrays, assignments, eligibility, recipient_answers, layers=layers)
    vector_files = {}
    for layer in layers:
        path = root / f"artifacts/vectors/P1_LAT__L{layer}.npz"; atomic_npz(path, vectors[int(layer)]); vector_files[str(layer)] = sha256_file(path)
    vector_fingerprint = canonical_hash([{k: r[k] for k in ("recipient_answer", "layer", "direction", "scaled_hash", "included_answers")} for r in metadata])
    atomic_json(root / "artifacts/vectors/vector_metadata.json", {"status": "complete", "vectors": metadata, "files": vector_files, "fingerprint": vector_fingerprint})
    atomic_jsonl(root / "artifacts/audits/test_manifest.jsonl", [{**r, "answer_origin": answer_origin(r), "fixed_answer_color": r["phase0_normalized_answer"]} for r in test_rows])
    atomic_jsonl(root / "artifacts/audits/train_manifest.jsonl", train_rows)
    result = {"status": "complete", "smoke_only": smoke, "train_records": len(oof), "train_families": len(by_family), "test_records": len(test_rows),
              "cell_count": len(cells), "eligible_answers": [r["fixed_answer_color"] for r in eligibility if r["eligible"]], "vector_count": len(metadata),
              "config_fingerprint": fingerprint, "vector_fingerprint": vector_fingerprint, "resumed_noop": False}
    atomic_json(root / "progress/prepare.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv); root = Path(args.output_root) if args.output_root else RESULTS_ROOT
    if args.smoke and not args.output_root: parser.error("--smoke requires an explicit output root outside formal results")
    print(json.dumps(run_prepare(output_root=root, smoke=args.smoke, resume=args.resume), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
