#!/usr/bin/env python3
"""Build the Probe manifest from existing joint results and behavior labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from confidence_test.answer_metrics import normalize_answer
from layer_metacognition.hidden_state_store import atomic_write_json

from . import EASY_CONDITIONS, HIDDEN_STATE_DEFINITION
from .common import atomic_write_jsonl, iter_jsonl, load_optional_jsonl, probe_output_dir


def _label_maps(
    output_dir: Path,
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    text: dict[tuple[str, int], dict[str, Any]] = {}
    image: dict[tuple[str, str], dict[str, Any]] = {}
    for record in load_optional_jsonl(output_dir / "text_only_labels.jsonl"):
        key = (str(record["item_id"]), int(record["prior_index"]))
        if key in text:
            raise ValueError(f"Duplicate text-only label key: {key}")
        text[key] = record
    for record in load_optional_jsonl(output_dir / "image_only_labels.jsonl"):
        key = (str(record["item_id"]), str(record["condition"]))
        if key in image:
            raise ValueError(f"Duplicate image-only label key: {key}")
        image[key] = record
    return text, image


def _same_list(left: Any, right: Any) -> bool:
    return [str(value) for value in left or []] == [str(value) for value in right or []]


def validate_hidden_reference(
    case_id: str,
    reference: dict[str, Any],
    index_reference: dict[str, Any],
) -> None:
    if str(index_reference.get("case_id")) != case_id:
        raise ValueError(f"Hidden index case ID mismatch for {case_id}")
    for field in ("shard_path", "offset", "hidden_size", "hidden_state_definition"):
        if reference.get(field) != index_reference.get(field):
            raise ValueError(
                f"Hidden reference mismatch for {case_id}: {field} "
                f"result={reference.get(field)!r} index={index_reference.get(field)!r}"
            )
    for field in ("layer_indices", "position_names"):
        if not _same_list(reference.get(field), index_reference.get(field)):
            raise ValueError(f"Hidden reference mismatch for {case_id}: {field}")
    if reference.get("hidden_state_definition") != HIDDEN_STATE_DEFINITION:
        raise ValueError(
            f"Unsupported hidden-state definition for {case_id}: "
            f"{reference.get('hidden_state_definition')!r}"
        )


def build_manifest(experiment_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    experiment = Path(experiment_dir).resolve()
    results_path = experiment / "results.jsonl"
    index_path = experiment / "hidden_states" / "index.json"
    output_dir = probe_output_dir(experiment)
    if not results_path.is_file():
        raise FileNotFoundError(f"Results do not exist: {results_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Hidden-state index does not exist: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index_cases = index.get("cases")
    if not isinstance(index_cases, dict):
        raise ValueError(f"Hidden-state index has no cases object: {index_path}")
    text_labels, image_labels = _label_maps(output_dir)

    manifest: list[dict[str, Any]] = []
    condition_counts: Counter[str] = Counter()
    version_counts: Counter[str] = Counter()
    selected_count = 0
    for source in iter_jsonl(results_path):
        generated = source.get("generated")
        reference = source.get("hidden_state_reference")
        if not (
            source.get("status") == "completed"
            and source.get("attribution_mode") == "joint"
            and isinstance(reference, dict)
            and isinstance(generated, dict)
            and generated.get("current_answer") is not None
        ):
            continue
        selected_count += 1
        case_id = str(source["case_id"])
        index_reference = index_cases.get(case_id)
        if not isinstance(index_reference, dict):
            raise ValueError(f"Completed case is absent from hidden-state index: {case_id}")
        validate_hidden_reference(case_id, reference, index_reference)
        item_id = str(source["item_id"])
        prior_index = int(source["prior_index"])
        condition = str(source["condition"])
        text_record = text_labels.get((item_id, prior_index))
        image_record = image_labels.get((item_id, condition))
        text_answer = (
            normalize_answer(text_record.get("text_only_answer"))
            if text_record and text_record.get("parse_success")
            else None
        )
        image_answer = (
            normalize_answer(image_record.get("image_only_answer"))
            if image_record and image_record.get("parse_success")
            else None
        )
        current_raw = str(generated["current_answer"])
        current_answer = normalize_answer(current_raw)
        if current_answer is None:
            raise ValueError(f"Selected case has unusable current answer: {case_id}")
        answer_classes: list[str] = []
        for candidate in (text_record, image_record):
            if candidate and isinstance(candidate.get("answer_classes"), list):
                answer_classes = [str(value) for value in candidate["answer_classes"]]
                break
        if not answer_classes:
            result = generated.get("current_answer_result") or {}
            probabilities = result.get("answer_class_probabilities")
            if isinstance(probabilities, dict):
                answer_classes = [str(value) for value in probabilities]
        row = {
            "case_id": case_id,
            "item_id": item_id,
            "prior_index": prior_index,
            "condition": condition,
            "version": str(source["version"]),
            "answer_classes": answer_classes,
            "text_only_answer": text_answer,
            "text_only_answer_raw": (
                text_record.get("text_only_answer_raw") if text_record else None
            ),
            "image_only_answer": image_answer,
            "image_only_answer_raw": (
                image_record.get("image_only_answer_raw") if image_record else None
            ),
            "current_answer": current_answer,
            "current_answer_raw": current_raw,
            "eligible_text_probe": text_answer is not None,
            "eligible_image_probe": condition in EASY_CONDITIONS and image_answer is not None,
            "hidden_state_reference": dict(reference),
        }
        manifest.append(row)
        condition_counts[condition] += 1
        version_counts[row["version"]] += 1

    hard_conditions = {"consistent_hard", "conflict_hard"}
    summary = {
        "selected_record_count": selected_count,
        "manifest_record_count": len(manifest),
        "item_count": len({row["item_id"] for row in manifest}),
        "item_prior_count": len(
            {(row["item_id"], row["prior_index"]) for row in manifest}
        ),
        "condition_counts": dict(sorted(condition_counts.items())),
        "version_counts": dict(sorted(version_counts.items())),
        "matched_easy_record_count": sum(
            row["condition"] in EASY_CONDITIONS for row in manifest
        ),
        "eligible_text_probe_count": sum(
            bool(row["eligible_text_probe"]) for row in manifest
        ),
        "eligible_image_probe_count": sum(
            bool(row["eligible_image_probe"]) for row in manifest
        ),
        "missing_text_label_count": sum(
            row["text_only_answer"] is None for row in manifest
        ),
        "missing_image_label_count": sum(
            row["condition"] in EASY_CONDITIONS and row["image_only_answer"] is None
            for row in manifest
        ),
        "image_hard_excluded_count": sum(
            row["condition"] in hard_conditions for row in manifest
        ),
        "image_null_irr_excluded_count": sum(
            row["condition"] in {"null", "irr"} for row in manifest
        ),
        "hidden_state_definition": HIDDEN_STATE_DEFINITION,
    }
    return manifest, summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = probe_output_dir(args.experiment_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, summary = build_manifest(args.experiment_dir)
    atomic_write_jsonl(output_dir / "probe_manifest.jsonl", manifest)
    atomic_write_json(output_dir / "manifest_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
