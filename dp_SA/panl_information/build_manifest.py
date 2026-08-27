from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from confidence_test.dataset_utils import EvaluationCase, load_evaluation_cases

from .config import ARTIFACT_NAMES, CONDITIONS, DATASET_PATH, RESULTS_ROOT, SOURCE_OOF, SOURCE_RESULTS, SPLIT_PATH
from .io_utils import atomic_json, atomic_jsonl, ensure_output_layout, load_jsonl, sha256_file
from .metrics import difficulty_factors


def _case_map(dataset: Path) -> dict[tuple[str, int], EvaluationCase]:
    cases, _ = load_evaluation_cases(dataset)
    result: dict[tuple[str, int], EvaluationCase] = {}
    for case in cases:
        key = (str(case.item_id), int(case.prior_index))
        old = result.get(key)
        if old is not None and (old.question, old.text_clue, old.answer_classes) != (case.question, case.text_clue, case.answer_classes):
            raise ValueError(f"Inconsistent dataset case key: {key}")
        result[key] = case
    return result


def build_manifest(
    root: Path,
    *,
    dataset: Path = DATASET_PATH,
    source_results: Path = SOURCE_RESULTS,
    split_path: Path = SPLIT_PATH,
) -> dict[str, Any]:
    artifacts = root / "artifacts"
    cases = _case_map(dataset)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if int(split.get("n_splits", -1)) != 5 or split.get("group_key") != "item_id":
        raise ValueError("Original split must be the frozen five-fold item_id assignment")
    item_to_fold = {str(key): int(value) for key, value in split["item_to_fold"].items()}
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_index, row in enumerate(load_jsonl(source_results), 1):
        reasons: list[str] = []
        item_id = str(row.get("item_id", ""))
        prior_index = int(row.get("prior_index", -1))
        condition = str(row.get("condition", ""))
        case = cases.get((item_id, prior_index))
        if row.get("status") != "completed":
            reasons.append("phase1_clean_not_completed")
        if not row.get("phase0_normalized_answer") or not row.get("phase0_raw_answer"):
            reasons.append("phase0_answer_parse_failed")
        if condition not in CONDITIONS:
            reasons.append("condition_not_in_scope")
        image_path = Path(str(row.get("image_path", "")))
        if not image_path.is_file():
            reasons.append("image_missing")
        if case is None:
            reasons.append("dataset_case_missing")
        elif len(case.answer_classes) != 12 or len(set(case.answer_classes)) != 12:
            reasons.append("candidate_colors_not_unique_12")
        if item_id not in item_to_fold:
            reasons.append("item_missing_from_original_split")
        case_id = str(row.get("case_id", ""))
        if not reasons and case_id in seen:
            reasons.append("duplicate_completed_case_id")
        if reasons:
            excluded.append({"source_index": source_index, "case_id": case_id or None, "item_id": item_id or None, "prior_index": prior_index, "condition": condition or None, "reasons": reasons, "source_status": row.get("status"), "source_error": row.get("error")})
            continue
        assert case is not None
        seen.add(case_id)
        eligible.append({
            **row,
            "item_id": item_id,
            "prior_index": prior_index,
            "condition": condition,
            "outer_fold": item_to_fold[item_id],
            "answer_classes": list(case.answer_classes),
            "text_answer": case.text_answer,
            "image_answer": case.conflict_answer,
            "prior_bin": case.prior_bin,
            "ground_truth_answer": case.ground_truth_answer,
            "historical_hidden_file": row["hidden_file"],
        })
    if len(seen) != len(eligible):
        raise AssertionError("Eligible case IDs are not unique")
    if {str(row["item_id"]) for row in eligible} - set(item_to_fold):
        raise AssertionError("Eligible manifest contains an unassigned item")
    atomic_jsonl(artifacts / ARTIFACT_NAMES["manifest"], eligible)
    atomic_jsonl(artifacts / ARTIFACT_NAMES["excluded"], excluded)
    split_audit = {
        "source_split": str(split_path.resolve()),
        "source_split_sha256": sha256_file(split_path),
        "n_splits": 5,
        "eligible_sample_count": len(eligible),
        "eligible_item_count": len({str(row["item_id"]) for row in eligible}),
        "fold_sample_counts": {str(fold): sum(int(row["outer_fold"]) == fold for row in eligible) for fold in range(5)},
        "item_overlap_by_fold": {str(fold): 0 for fold in range(5)},
        "status": "passed",
    }
    atomic_json(artifacts / "split_audit.json", split_audit)
    return {
        "eligible_count": len(eligible),
        "excluded_count": len(excluded),
        "item_count": len({str(row["item_id"]) for row in eligible}),
        "text_key_count": len({(str(row["item_id"]), int(row["prior_index"])) for row in eligible}),
        "image_key_count": len({(str(row["item_id"]), str(row["condition"])) for row in eligible}),
    }


def join_scores(root: Path, *, source_oof: Path = SOURCE_OOF) -> dict[str, Any]:
    artifacts = root / "artifacts"
    manifest = load_jsonl(artifacts / ARTIFACT_NAMES["manifest"])
    scores = load_jsonl(artifacts / ARTIFACT_NAMES["unimodal"], repair_trailing=True)
    text: dict[tuple[str, int], dict[str, Any]] = {}
    image: dict[tuple[str, str], dict[str, Any]] = {}
    for row in scores:
        if row["modality"] == "text":
            key = (str(row["item_id"]), int(row["prior_index"]))
            if key in text:
                raise ValueError(f"Duplicate text score key: {key}")
            text[key] = row
        else:
            key = (str(row["item_id"]), str(row["condition"]))
            if key in image:
                raise ValueError(f"Duplicate image score key: {key}")
            image[key] = row
    panl_oof: dict[str, float] = {}
    for row in load_jsonl(source_oof):
        if str(row.get("position")) == "P1_PANL" and int(row.get("layer", -1)) == 14:
            case_id = str(row["case_id"])
            if case_id in panl_oof:
                raise ValueError(f"Duplicate historical PANL L14 OOF prediction: {case_id}")
            panl_oof[case_id] = float(row["prediction"])
    joined: list[dict[str, Any]] = []
    for row in manifest:
        text_key = (str(row["item_id"]), int(row["prior_index"]))
        image_key = (str(row["item_id"]), str(row["condition"]))
        if text_key not in text or image_key not in image:
            raise ValueError(f"Missing unimodal score for {row['case_id']}")
        if row["case_id"] not in panl_oof:
            raise ValueError(f"Missing historical PANL L14 OOF prediction: {row['case_id']}")
        text_fields = {key: value for key, value in text[text_key].items() if key.startswith("text_")}
        image_fields = {key: value for key, value in image[image_key].items() if key.startswith("image_")}
        factors = difficulty_factors(text_fields["text_model_perceived_difficulty"], image_fields["image_model_perceived_difficulty"])
        answer = str(row["phase0_normalized_answer"])
        text_match = answer == str(row["text_answer"])
        image_match = answer == str(row["image_answer"])
        decision = "follow_text" if text_match and not image_match else "follow_image" if image_match and not text_match else None
        joined.append({
            **row,
            **text_fields,
            **image_fields,
            **factors,
            "Hard": int(row["condition"] == "conflict_hard"),
            "decision_side": decision,
            "decision_exclusion_reason": None if decision else "matches_both" if text_match and image_match else "matches_neither",
            "final_sa": float(row["soft_sa_image_score"]),
            "panl_l14_oof_sa_prediction": panl_oof[row["case_id"]],
        })
    atomic_jsonl(artifacts / ARTIFACT_NAMES["joined"], joined)
    return {"joined_count": len(joined), "decision_eligible_count": sum(row["decision_side"] is not None for row in joined)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = ensure_output_layout(RESULTS_ROOT, resume=args.resume)
    print(json.dumps(build_manifest(root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
