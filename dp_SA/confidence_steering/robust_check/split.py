from __future__ import annotations

import inspect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold

from .config import (
    ALTERNATIVE_SEEDS, AUDIT_FOLD, EXPECTED_ASSIGNMENT_HASHES,
    EXPECTED_FORMAL_SHA256, EXPECTED_TRAIN_SHA256, FORMAL_MANIFEST,
    N_SPLITS, SEEDS, TRAIN_MANIFEST,
)
from .io_utils import atomic_csv, atomic_json, canonical_hash, load_jsonl, sha256_file


def sklearn_audit() -> dict[str, Any]:
    signature = inspect.signature(GroupKFold)
    if "shuffle" not in signature.parameters or "random_state" not in signature.parameters:
        raise RuntimeError(f"GroupKFold lacks shuffle/random_state support: {signature}")
    probe = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=43)
    if not probe.shuffle or probe.random_state != 43:
        raise RuntimeError("GroupKFold shuffle/random_state were not applied")
    return {
        "scikit_learn_version": sklearn.__version__,
        "group_kfold_class": f"{GroupKFold.__module__}.{GroupKFold.__name__}",
        "group_kfold_signature": str(signature),
        "parameters": {"n_splits": N_SPLITS, "shuffle": True},
        "supported": True,
    }


def load_frozen_train() -> list[dict[str, Any]]:
    if sha256_file(TRAIN_MANIFEST) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("Frozen 1112-record training manifest changed")
    rows = load_jsonl(TRAIN_MANIFEST)
    if len(rows) != 1112 or len({str(row["family_id"]) for row in rows}) != 128:
        raise RuntimeError("Frozen training cardinality changed")
    return rows


def load_formal() -> list[dict[str, Any]]:
    if sha256_file(FORMAL_MANIFEST) != EXPECTED_FORMAL_SHA256:
        raise RuntimeError("Frozen formal manifest changed")
    rows = load_jsonl(FORMAL_MANIFEST)
    if len(rows) != 100 or len({str(row["family_id"]) for row in rows}) != 50:
        raise RuntimeError("Frozen formal cardinality changed")
    return rows


def assignment_hash(assignments: dict[str, int]) -> str:
    payload = "".join(
        json.dumps({"family_id": family, "fold": int(assignments[family])}, sort_keys=True, separators=(",", ":")) + "\n"
        for family in sorted(assignments)
    )
    import hashlib
    return hashlib.sha256(payload.encode()).hexdigest()


def family_assignments(rows: Sequence[dict[str, Any]], seed: int) -> dict[str, int]:
    if seed not in SEEDS:
        raise ValueError(f"Unregistered split seed: {seed}")
    assignments: dict[str, int] = {}
    if seed == 42:
        for row in rows:
            family, fold = str(row["family_id"]), int(row["outer_fold"])
            if family in assignments and assignments[family] != fold:
                raise RuntimeError(f"Historical family split across folds: {family}")
            assignments[family] = fold
    else:
        groups = np.asarray([str(row["family_id"]) for row in rows], dtype=object)
        splitter = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (_train, held_out) in enumerate(splitter.split(np.zeros((len(rows), 1)), groups=groups)):
            for index in held_out:
                family = groups[index]
                if family in assignments and assignments[family] != fold:
                    raise RuntimeError(f"Family split across generated folds: {family}")
                assignments[family] = fold
    if len(assignments) != 128 or set(assignments.values()) != set(range(N_SPLITS)):
        raise RuntimeError("Incomplete family assignment")
    digest = assignment_hash(assignments)
    if digest != EXPECTED_ASSIGNMENT_HASHES[seed]:
        raise RuntimeError(f"Unexpected assignment hash for seed {seed}: {digest}")
    return assignments


def build_all_assignments(rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, int]]:
    sklearn_audit()
    first = {seed: family_assignments(rows, seed) for seed in SEEDS}
    second = {seed: family_assignments(rows, seed) for seed in SEEDS}
    if {seed: assignment_hash(value) for seed, value in first.items()} != {seed: assignment_hash(value) for seed, value in second.items()}:
        raise RuntimeError("Split generation is not deterministic")
    alternative_hashes = [assignment_hash(first[seed]) for seed in ALTERNATIVE_SEEDS]
    if len(set(alternative_hashes)) != len(alternative_hashes):
        raise RuntimeError("Seeds 43, 44 and 45 do not have pairwise-distinct family assignments")
    return first


def apply_assignment(rows: Sequence[dict[str, Any]], assignments: dict[str, int]) -> list[dict[str, Any]]:
    return [{**row, "outer_fold": int(assignments[str(row["family_id"])])} for row in rows]


def _identity_sets(rows: Sequence[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        "case": {str(row["case_id"]) for row in rows},
        "family": {str(row["family_id"]) for row in rows},
        "item": {str(row["item_id"]) for row in rows},
        "image_hash": {str(row["image_sha256"]) for row in rows},
    }


def leakage_audit(construction: Sequence[dict[str, Any]], audit: Sequence[dict[str, Any]], formal: Sequence[dict[str, Any]] | None) -> dict[str, Any]:
    groups = {"construction": _identity_sets(construction), "audit": _identity_sets(audit)}
    if formal is not None:
        groups["formal"] = _identity_sets(formal)
    overlaps = {}
    names = sorted(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            for identity in groups[left]:
                overlaps[f"{left}__{right}__{identity}"] = len(groups[left][identity] & groups[right][identity])
    if any(overlaps.values()):
        raise RuntimeError(f"Split leakage detected: {overlaps}")
    return {"status": "passed", "overlaps": overlaps, "formal_manifest_opened": formal is not None}


def _fold_summary(rows: Sequence[dict[str, Any]], fold: int) -> dict[str, Any]:
    selected = [row for row in rows if int(row["outer_fold"]) == fold]
    result: dict[str, Any] = {
        "fold": fold,
        "records": len(selected),
        "families": len({str(row["family_id"]) for row in selected}),
        "items": len({str(row["item_id"]) for row in selected}),
        "images": len({str(row["image_sha256"]) for row in selected}),
    }
    for color, count in sorted(Counter(str(row["phase0_normalized_answer"]) for row in selected).items()):
        result[f"color_{color}"] = count
    for field in ("condition", "prior_bin", "answer_origin", "unimodal_chosen_match"):
        if selected and field not in selected[0]:
            continue
        for value, count in sorted(Counter(str(row[field]) for row in selected).items()):
            result[f"{field}_{value}"] = count / len(selected)
    if selected and "Hard" in selected[0]:
        result["Hard_proportion"] = float(np.mean([float(row["Hard"]) for row in selected]))
    for field in ("G_L", "C_t", "C_i", "D_t", "D_i", "clean_final_sa"):
        if selected and field in selected[0]:
            values = np.asarray([float(row[field]) for row in selected])
            result[f"{field}_mean"] = float(values.mean())
            result[f"{field}_sd"] = float(values.std(ddof=1))
    return result


def _balance_summary(construction: Sequence[dict[str, Any]], audit: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"continuous": {}, "categorical": {}}
    for field in ("G_L", "C_t", "C_i", "D_t", "D_i", "clean_final_sa"):
        if not construction or field not in construction[0]:
            continue
        left = np.asarray([float(row[field]) for row in construction])
        right = np.asarray([float(row[field]) for row in audit])
        pooled = math.sqrt((float(left.var(ddof=1)) + float(right.var(ddof=1))) / 2.0)
        result["continuous"][field] = {
            "construction_mean": float(left.mean()), "construction_sd": float(left.std(ddof=1)),
            "audit_mean": float(right.mean()), "audit_sd": float(right.std(ddof=1)),
            "standardized_mean_difference": (float(left.mean() - right.mean()) / pooled) if pooled else None,
        }
    for field in ("fixed_answer_color", "answer_origin", "Hard", "condition", "prior_bin", "unimodal_chosen_match"):
        if not construction or field not in construction[0]:
            continue
        levels = sorted({str(row[field]) for row in construction} | {str(row[field]) for row in audit})
        result["categorical"][field] = {level: {
            "construction_proportion": sum(str(row[field]) == level for row in construction) / len(construction),
            "audit_proportion": sum(str(row[field]) == level for row in audit) / len(audit),
        } for level in levels}
    return result


def write_seed_split(root: Path, rows: Sequence[dict[str, Any]], seed: int, assignments: dict[str, int], formal: Sequence[dict[str, Any]] | None) -> dict[str, Any]:
    assigned = apply_assignment(rows, assignments)
    construction = [row for row in assigned if int(row["outer_fold"]) != AUDIT_FOLD]
    audit = [row for row in assigned if int(row["outer_fold"]) == AUDIT_FOLD]
    leakage = leakage_audit(construction, audit, formal)
    destination = root / f"artifacts/splits/seed_{seed}"
    assignment_rows = [{"family_id": family, "fold": assignments[family]} for family in sorted(assignments)]
    keep = ("case_id", "family_id", "item_id", "image_sha256", "phase0_normalized_answer", "condition", "prior_bin", "outer_fold")
    atomic_csv(destination / "fold_assignments.csv", assignment_rows)
    atomic_csv(destination / "construction.csv", [{key: row[key] for key in keep} for row in construction])
    atomic_csv(destination / "audit.csv", [{key: row[key] for key in keep} for row in audit])
    split_audit = {
        "seed": seed,
        "policy": "historical outer_fold" if seed == 42 else "GroupKFold(n_splits=5, shuffle=True, random_state=seed)",
        "assignment_sha256": assignment_hash(assignments),
        "construction_records": len(construction),
        "construction_families": len({row["family_id"] for row in construction}),
        "audit_records": len(audit),
        "audit_families": len({row["family_id"] for row in audit}),
        "folds": [_fold_summary(assigned, fold) for fold in range(N_SPLITS)],
        "construction_vs_audit_balance": _balance_summary(construction, audit),
        "leakage": leakage,
        "scikit_learn": sklearn_audit(),
    }
    atomic_json(destination / "split_audit.json", split_audit)
    return {"assigned": assigned, "construction": construction, "audit": audit, "audit_record": split_audit}


def select_smoke_families(audit: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import itertools
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit:
        grouped[str(row["family_id"])].append(row)
    candidates = []
    for families in itertools.combinations(sorted(grouped), 4):
        selected = [row for family in families for row in grouped[family]]
        colors = {str(row["phase0_normalized_answer"]) for row in selected}
        if len(colors) >= 4:
            candidates.append((len(selected), families, selected, colors))
    if not candidates:
        raise RuntimeError("No four-family smoke selection covers at least four colors")
    count, families, selected, colors = min(candidates, key=lambda value: (value[0], value[1]))
    return sorted(selected, key=lambda row: str(row["case_id"])), {
        "source": "seed_45_audit_only",
        "records": count,
        "families": list(families),
        "colors": sorted(colors),
        "formal_manifest_opened": False,
    }
