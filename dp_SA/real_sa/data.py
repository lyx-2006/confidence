from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    HISTORICAL_PHASE0, IMAGE_DONOR_MANIFEST, SEED, SPLIT_AUDIT, TEST_MANIFEST,
    TEXT_DONOR_MANIFEST, TRAIN_MANIFEST,
)
from .io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file


@dataclass(frozen=True)
class FrozenCohort:
    tests: list[dict[str, Any]]
    image_donors: list[dict[str, Any]]
    text_donors: list[dict[str, Any]]
    audit: dict[str, Any]


def answer_side(row: dict[str, Any]) -> str:
    fixed = str(row["phase0_normalized_answer"])
    image = str(row["image_answer"])
    text = str(row["text_answer"])
    if fixed == image and fixed != text:
        return "follow_image"
    if fixed == text and fixed != image:
        return "follow_text"
    raise ValueError(f"Test answer has no unique modality side: {row['case_id']}")


def _rank_text_donors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[str, str]:
        raw = json.dumps([SEED, row["unique_key"]], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest(), raw
    return sorted(rows, key=key)


def load_frozen_cohort() -> FrozenCohort:
    required = (TEST_MANIFEST, TRAIN_MANIFEST, IMAGE_DONOR_MANIFEST, TEXT_DONOR_MANIFEST,
                SPLIT_AUDIT, HISTORICAL_PHASE0)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen experiment inputs: {missing}")
    tests = load_jsonl(TEST_MANIFEST)
    train = load_jsonl(TRAIN_MANIFEST)
    image_specs = load_jsonl(IMAGE_DONOR_MANIFEST)
    text_specs = load_jsonl(TEXT_DONOR_MANIFEST)
    history = {str(row["case_id"]): row for row in load_jsonl(HISTORICAL_PHASE0)}
    if len(history) != 1656:
        raise ValueError(f"Historical Phase 0 count changed: {len(history)}")
    if len(tests) != 100 or len(image_specs) != 200 or len(text_specs) != 200:
        raise ValueError("Frozen cohort must contain 100 tests and 200 donors per modality")

    train_by_image: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    train_by_text: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    family_by_item: dict[str, str] = {}
    for row in train:
        item = str(row["item_id"])
        old = family_by_item.setdefault(item, str(row["family_id"]))
        if old != str(row["family_id"]):
            raise ValueError(f"Item crosses train families: {item}")
        train_by_image[(item, str(row["condition"]), str(row["image_sha256"]))].append(row)
        key = (item, int(row["prior_index"]))
        condition = str(row["condition"])
        if condition in train_by_text[key]:
            raise ValueError(f"Duplicate train text condition: {key} {condition}")
        train_by_text[key][condition] = row

    image_donors: list[dict[str, Any]] = []
    for spec in image_specs:
        key = (str(spec["item_id"]), str(spec["condition"]), str(spec["image_hash"]))
        candidates = sorted(train_by_image.get(key, []), key=lambda row: (int(row["prior_index"]), str(row["case_id"])))
        if not candidates:
            raise ValueError(f"Frozen image donor cannot be joined: {key}")
        context = candidates[0]
        donor_id = "image:" + canonical_hash(spec["unique_key"])
        image_donors.append({**context, "donor_id": donor_id, "donor_unique_key": spec["unique_key"],
                             "donor_modality": "image", "context_policy": "lowest_prior_index"})

    ranked_text = _rank_text_donors(text_specs)
    text_donors: list[dict[str, Any]] = []
    for index, spec in enumerate(ranked_text):
        key = (str(spec["item_id"]), int(spec["prior_index"]))
        contexts = train_by_text.get(key, {})
        if set(contexts) != {"conflict_easy", "conflict_hard"}:
            raise ValueError(f"Text donor lacks both image conditions: {key}")
        condition = "conflict_easy" if index % 2 == 0 else "conflict_hard"
        context = contexts[condition]
        for name in ("question", "text_clue", "answer_classes"):
            if context[name] != spec[name]:
                raise ValueError(f"Text donor join changed {name}: {key}")
        donor_id = "text:" + canonical_hash(spec["unique_key"])
        text_donors.append({**context, "donor_id": donor_id, "donor_unique_key": spec["unique_key"],
                            "donor_modality": "text", "context_policy": "sha256_rank_alternating_100_100"})

    joined_tests: list[dict[str, Any]] = []
    for row in tests:
        case_id = str(row["case_id"])
        phase0 = history.get(case_id)
        if phase0 is None:
            raise ValueError(f"Test has no historical Phase 0 row: {case_id}")
        for name in ("question", "text_clue", "image_path", "image_sha256", "phase0_raw_answer", "phase0_normalized_answer"):
            if row.get(name) != phase0.get(name):
                raise ValueError(f"Frozen test/history mismatch for {case_id}: {name}")
        if str(row["phase0_normalized_answer"]) not in list(row["answer_classes"]):
            raise ValueError(f"Fixed answer is outside candidates: {case_id}")
        joined_tests.append({**row, "historical_phase0": phase0, "answer_side": answer_side(row)})

    test_cases = {str(row["case_id"]) for row in joined_tests}
    test_families = {str(row["family_id"]) for row in joined_tests}
    test_items = {str(row["item_id"]) for row in joined_tests}
    test_hashes = {str(row["image_sha256"]) for row in joined_tests}
    donors = [*image_donors, *text_donors]
    overlaps = {
        "case": len(test_cases & {str(row["case_id"]) for row in donors}),
        "family": len(test_families & {str(row["family_id"]) for row in donors}),
        "item": len(test_items & {str(row["item_id"]) for row in donors}),
        "image_hash": len(test_hashes & {str(row["image_sha256"]) for row in donors}),
    }
    if any(overlaps.values()):
        raise ValueError(f"Test/donor leakage: {overlaps}")
    side_counts = Counter(row["answer_side"] for row in joined_tests)
    cells = Counter((row["condition"], row["answer_side"]) for row in joined_tests)
    if side_counts != {"follow_image": 50, "follow_text": 50}:
        raise ValueError(f"Test answer-side balance changed: {side_counts}")
    expected_cells = {(condition, side): 25 for condition in ("conflict_easy", "conflict_hard")
                      for side in ("follow_image", "follow_text")}
    if dict(cells) != expected_cells:
        raise ValueError(f"Test condition/answer-side balance changed: {cells}")
    text_condition_counts = Counter(row["condition"] for row in text_donors)
    if text_condition_counts != {"conflict_easy": 100, "conflict_hard": 100}:
        raise ValueError(f"Text donor context balance failed: {text_condition_counts}")
    audit = {
        "status": "passed", "test_count": len(joined_tests),
        "test_family_count": len(test_families), "image_donor_count": len(image_donors),
        "text_donor_count": len(text_donors), "test_answer_side_counts": dict(side_counts),
        "test_cells": {f"{condition}__{side}": count for (condition, side), count in sorted(cells.items())},
        "text_donor_context_conditions": dict(text_condition_counts), "overlaps": overlaps,
        "source_sha256": {
            "test": sha256_file(TEST_MANIFEST), "train": sha256_file(TRAIN_MANIFEST),
            "image_donors": sha256_file(IMAGE_DONOR_MANIFEST), "text_donors": sha256_file(TEXT_DONOR_MANIFEST),
            "split_audit": sha256_file(SPLIT_AUDIT), "historical_phase0": sha256_file(HISTORICAL_PHASE0),
        },
    }
    return FrozenCohort(joined_tests, image_donors, text_donors, audit)


def write_manifests(root: Path, cohort: FrozenCohort) -> None:
    destination = root / "artifacts" / "manifests"
    atomic_jsonl(destination / "test_manifest.jsonl", cohort.tests)
    atomic_jsonl(destination / "image_donor_manifest.jsonl", cohort.image_donors)
    atomic_jsonl(destination / "text_donor_manifest.jsonl", cohort.text_donors)
    atomic_json(destination / "donor_audit.json", cohort.audit)


__all__ = ["FrozenCohort", "answer_side", "load_frozen_cohort", "write_manifests"]
