from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import (
    ANSWER_MATCHED_ROOT, CANDIDATE_MANIFEST_PATH, CONSTRUCTION_CELLS_PATH,
    EXPECTED_CLE_ELIGIBLE, EXPECTED_FORMAL_CASES, EXPECTED_SHA256,
    FOLD_ASSIGNMENTS_PATH, HISTORICAL_CLEAN_PATH, HISTORICAL_LOG_PATH,
    HIDDEN_DEFINITION, LAT_LAYER, MANIFEST_PATH, MODEL_PATH, PREPROCESSOR_CONFIG_PATH,
    PROBE_CONFIG_PATH, PROBE_CONSTRUCTION_PATH, PROBE_INDEX_PATH, PROBE_ROOT,
    RESULTS_ROOT, SMOKE_FAMILIES, VECTOR_FILE_SHA256, VECTOR_METADATA_PATH,
)
from .io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file


def _require_hash(label: str, path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Frozen {label} hash mismatch: expected={expected}, actual={actual}, path={path}")
    return actual


def _resolve_probe() -> tuple[dict[str, Any], Path]:
    matches = [row for row in load_jsonl(PROBE_INDEX_PATH)
               if row.get("target") == "final_soft_sa"
               and row.get("position") == "P1_CLASS_LIST_END"
               and int(row.get("layer", -1)) == 20]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one frozen CLE probe, found {len(matches)}")
    row = matches[0]
    probe_path = PROBE_ROOT / str(row["probe_file"])
    _require_hash("probe", probe_path, EXPECTED_SHA256["probe"])
    if row.get("probe_sha256") != EXPECTED_SHA256["probe"]:
        raise ValueError("Probe index hash disagrees with frozen probe")
    if not row.get("readout_reliable") or float(row["r2"]) <= 0 or float(row["pearson"]) <= 0 or float(row["pearson_ci_low"]) <= 0:
        raise ValueError("Frozen probe reliability gate failed")
    probe_config = json.loads(PROBE_CONFIG_PATH.read_text())
    if probe_config.get("hidden_definition") != HIDDEN_DEFINITION:
        raise ValueError("Probe hidden definition mismatch")
    return row, probe_path


def _vector_selection(test: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata = json.loads(VECTOR_METADATA_PATH.read_text())
    rows = [row for row in metadata.get("vectors", [])
            if row.get("position") == "P1_LAT" and int(row.get("layer", -1)) == LAT_LAYER
            and row.get("direction") == "matched_loao"]
    index: dict[tuple[int, str], dict[str, Any]] = {}
    checked_files: dict[int, str] = {}
    for row in rows:
        key = (int(row["fold"]), str(row["recipient_answer"]))
        if key in index:
            raise ValueError(f"Duplicate vector metadata key: {key}")
        path = ANSWER_MATCHED_ROOT / str(row["vector_file"])
        expected = VECTOR_FILE_SHA256[int(row["fold"])]
        actual = _require_hash(f"fold {row['fold']} L14 vector", path, expected)
        if row.get("vector_file_sha256") != actual:
            raise ValueError(f"Vector metadata file hash mismatch: {key}")
        with np.load(path, allow_pickle=False) as archive:
            if str(row["scaled_key"]) not in archive:
                raise ValueError(f"Scaled vector key absent: {key}")
            vector = np.asarray(archive[str(row["scaled_key"])], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isclose(norm, float(row["scaled_norm"]), rtol=1e-6, atol=1e-6):
            raise ValueError(f"Vector norm mismatch: {key}")
        index[key] = row
        checked_files[int(row["fold"])] = actual
    if set(checked_files) != set(range(15)):
        raise ValueError("Did not verify all 15 L14 vector files")
    output = []
    for case in test:
        key = (int(case["fold"]), str(case["test_answer"]))
        if key not in index:
            raise ValueError(f"Missing fold/recipient vector for {case['case_id']}: {key}")
        row = index[key]
        output.append({
            "case_id": str(case["case_id"]), "family_id": str(case["family_id"]),
            "fold": key[0], "recipient_answer": key[1], "position": "P1_LAT",
            "layer": LAT_LAYER, "direction": "matched_loao", "scaled_key": row["scaled_key"],
            "scaled_norm": float(row["scaled_norm"]), "vector_fingerprint": row["vector_fingerprint"],
            "vector_file": str(row["vector_file"]), "vector_file_sha256": row["vector_file_sha256"],
        })
    return output


def _eligibility(test: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    construction = load_jsonl(PROBE_CONSTRUCTION_PATH)
    train_families = {str(row["family_id"]) for row in construction}
    train_items = {str(row["item_id"]) for row in construction}
    train_images = {str(row["image_sha256"]) for row in construction}
    rows = []
    for case in test:
        reasons = []
        if str(case["family_id"]) in train_families: reasons.append("family_overlap")
        if str(case["item_id"]) in train_items: reasons.append("item_overlap")
        if str(case["image_sha256"]) in train_images: reasons.append("image_hash_overlap")
        rows.append({"case_id": str(case["case_id"]), "cle_probe_eligible": not reasons,
                     "cle_probe_exclusion_reasons": reasons})
    eligible_ids = {row["case_id"] for row in rows if row["cle_probe_eligible"]}
    eligible = [row for row in test if str(row["case_id"]) in eligible_ids]
    answer = Counter(str(row.get("test_answer", row["canonical_answer"])) for row in eligible)
    side = Counter(str(row.get("test_side", row["sa_side"])) for row in eligible)
    condition = Counter(str(row["condition"]) for row in eligible)
    answer_side: dict[str, Counter[str]] = defaultdict(Counter)
    for row in eligible:
        answer_side[str(row.get("test_answer", row["canonical_answer"]))][str(row.get("test_side", row["sa_side"]))] += 1
    audit = {
        "eligible_count": len(eligible), "excluded_count": len(test) - len(eligible),
        "family_count": len({row["family_id"] for row in eligible}),
        "item_count": len({str(row["item_id"]) for row in eligible}),
        "image_hash_count": len({row["image_sha256"] for row in eligible}),
        "answer_counts": dict(sorted(answer.items())), "side_counts": dict(sorted(side.items())),
        "condition_counts": dict(sorted(condition.items())),
        "answer_side_counts": {key: dict(sorted(value.items())) for key, value in sorted(answer_side.items())},
        "figure2_status": "exploratory",
        "figure2_reason": "strict no-overlap subset is imbalanced: brown n=1 and has no high-image case",
    }
    return rows, audit


def prepare(*, output_root: Path = RESULTS_ROOT, smoke: bool = False, resume: bool = False) -> dict[str, Any]:
    import transformers
    from transformers import Qwen2VLImageProcessorFast
    root = Path(output_root)
    frozen = {
        "manifest": _require_hash("formal manifest", MANIFEST_PATH, EXPECTED_SHA256["manifest"]),
        "vector_metadata": _require_hash("vector metadata", VECTOR_METADATA_PATH, EXPECTED_SHA256["vector_metadata"]),
        "probe_construction": _require_hash("probe construction", PROBE_CONSTRUCTION_PATH, EXPECTED_SHA256["probe_construction"]),
        "preprocessor_config": _require_hash("preprocessor config", PREPROCESSOR_CONFIG_PATH, EXPECTED_SHA256["preprocessor_config"]),
        "historical_log": _require_hash("historical Fast-processor log", HISTORICAL_LOG_PATH, EXPECTED_SHA256["historical_log"]),
        "historical_clean": _require_hash("historical clean capture", HISTORICAL_CLEAN_PATH, EXPECTED_SHA256["historical_clean"]),
    }
    if "fast" not in HISTORICAL_LOG_PATH.read_text(errors="replace").casefold():
        raise ValueError("Historical log does not contain Fast processor evidence")
    probe_row, probe_path = _resolve_probe(); frozen["probe"] = sha256_file(probe_path)
    formal = load_jsonl(MANIFEST_PATH)
    if len(formal) != EXPECTED_FORMAL_CASES or len({row["case_id"] for row in formal}) != EXPECTED_FORMAL_CASES:
        raise ValueError("Formal manifest is not the frozen 174-case design")
    if len({row["family_id"] for row in formal}) != 174 or len({str(row["item_id"]) for row in formal}) != 174:
        raise ValueError("Formal manifest family/item identities are not independent")
    assignments = {str(row["family_id"]): int(row["fold"]) for row in load_jsonl(FOLD_ASSIGNMENTS_PATH) if row.get("is_test_family")}
    construction = load_jsonl(CONSTRUCTION_CELLS_PATH)
    leaks = {(int(row["fold"]), str(row["family_id"])) for row in construction}
    for row in formal:
        family, fold = str(row["family_id"]), int(row["fold"])
        if assignments.get(family) != fold: raise ValueError(f"Fold assignment mismatch: {row['case_id']}")
        if (fold, family) in leaks: raise ValueError(f"Own-family vector construction leakage: {row['case_id']}")
    selection = _vector_selection(formal)
    eligibility, cle_audit = _eligibility(formal)
    if cle_audit["eligible_count"] != EXPECTED_CLE_ELIGIBLE:
        raise ValueError(f"CLE eligibility changed: {cle_audit['eligible_count']} != {EXPECTED_CLE_ELIGIBLE}")
    eligibility_by_id = {row["case_id"]: row for row in eligibility}
    augmented = [{**row, **eligibility_by_id[str(row["case_id"])]} for row in formal]
    if smoke:
        smoke_base = [dict(row) for row in load_jsonl(CANDIDATE_MANIFEST_PATH)
                      if str(row["family_id"]) in SMOKE_FAMILIES]
        smoke_eligibility, _ = _eligibility(smoke_base)
        smoke_eligibility_by_id = {row["case_id"]: row for row in smoke_eligibility}
        selected = []
        for row in smoke_base:
            family = str(row["family_id"])
            selected.append({**row, "fold": assignments[family],
                             "test_answer": str(row["canonical_answer"]),
                             "test_side": str(row["sa_side"]), "test_status": "smoke_audit",
                             **smoke_eligibility_by_id[str(row["case_id"])]})
    else:
        selected = augmented
    if smoke and (len(selected) != 24 or {row["family_id"] for row in selected} != set(SMOKE_FAMILIES)):
        raise ValueError("Smoke selection is not the frozen 4-family/24-case audit set")
    selected_vectors = _vector_selection(selected) if smoke else selection
    audit_payload = {"rows": eligibility, "summary": cle_audit}
    eligibility_hash = canonical_hash(audit_payload)
    source_paths = [
        Path(__file__), Path(__file__).with_name("config.py"), Path(__file__).with_name("hooks.py"),
        Path(__file__).with_name("io_utils.py"), Path(__file__).with_name("run.py"),
        Path(__file__).with_name("analyze.py"), Path(__file__).with_name("run_pipeline.py"),
        Path(__file__).resolve().parents[3] / "qwen-2.5-vl" / "inference.py",
        Path(__file__).resolve().parents[2] / "positions.py", Path(__file__).resolve().parents[2] / "prompts.py",
        Path(__file__).resolve().parents[2] / "soft_score.py",
        Path(__file__).resolve().parents[3] / "layer_metacognition" / "model_adapter.py",
        Path(__file__).resolve().parents[3] / "layer_metacognition" / "conversation_builder.py",
        Path(MODEL_PATH) / "config.json", Path(MODEL_PATH) / "model.safetensors.index.json",
    ]
    config = {
        "experiment": "LAT14_to_PANL_SA_mediation", "smoke": smoke,
        "hidden_definition": HIDDEN_DEFINITION, "formal_case_count": 174,
        "selected_case_count": len(selected), "cle_eligibility_audit_sha256": eligibility_hash,
        "frozen_sha256": frozen, "probe_index_row": probe_row,
        "processor": {"image_processor_class": Qwen2VLImageProcessorFast.__module__ + "." + Qwen2VLImageProcessorFast.__name__, "is_fast": True,
                      "transformers_version": transformers.__version__,
                      "min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28,
                      "historical_identity_basis": "preprocessor config + historical log + runtime assertion + C0 parity"},
        "implementation_and_model_sha256": {str(path): sha256_file(path) for path in source_paths},
        "git": {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True).strip(),
            "origin_main": subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=Path(__file__).resolve().parents[3], text=True).strip(),
        },
    }
    config["semantic_fingerprint"] = canonical_hash(config)
    config_path = root / "fingerprint.json"; existed = config_path.exists()
    if existed:
        previous = json.loads(config_path.read_text())
        if previous.get("semantic_fingerprint") != config["semantic_fingerprint"]:
            raise ValueError("Existing output fingerprint mismatch")
        if not resume: raise FileExistsError(f"Prepared output exists; use --resume: {root}")
    atomic_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl", selected)
    atomic_jsonl(root / "artifacts" / "manifests" / "vector_selection.jsonl", selected_vectors)
    atomic_jsonl(root / "artifacts" / "diagnostics" / "cle_probe_eligibility.jsonl", eligibility)
    atomic_json(root / "artifacts" / "diagnostics" / "cle_probe_eligibility_audit.json", {**cle_audit, "audit_sha256": eligibility_hash})
    atomic_json(root / "fingerprint.json", config)
    result = {"status": "complete", "smoke": smoke, "case_count": len(selected),
              "formal_case_count": len(formal), "cle_eligible_formal": cle_audit["eligible_count"],
              "vector_coverage": len(selected_vectors), "fold_errors": 0, "own_family_construction_leaks": 0,
              "semantic_fingerprint": config["semantic_fingerprint"], "resumed": existed}
    atomic_json(root / "progress" / "prepare.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(output_root=args.output_root, smoke=args.smoke, resume=args.resume), ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
