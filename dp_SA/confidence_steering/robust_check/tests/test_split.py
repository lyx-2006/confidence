from __future__ import annotations

import pytest

from ..config import EXPECTED_ASSIGNMENT_HASHES
from ..split import (
    apply_assignment, assignment_hash, build_all_assignments, family_assignments,
    leakage_audit, load_frozen_train, select_smoke_families, sklearn_audit,
)


@pytest.fixture(scope="module")
def rows():
    return load_frozen_train()


def test_group_kfold_supports_explicit_shuffle():
    audit = sklearn_audit()
    assert audit["supported"] and "shuffle=False" in audit["group_kfold_signature"]


def test_expected_assignment_hashes(rows):
    for seed, expected in EXPECTED_ASSIGNMENT_HASHES.items():
        assert assignment_hash(family_assignments(rows, seed)) == expected


def test_alternative_assignment_hashes_are_pairwise_distinct(rows):
    assignments = build_all_assignments(rows)
    assert len({assignment_hash(assignments[seed]) for seed in (43, 44, 45)}) == 3


def test_assignment_generation_is_deterministic(rows):
    assert family_assignments(rows, 44) == family_assignments(rows, 44)


def test_family_never_crosses_folds(rows):
    assigned = apply_assignment(rows, family_assignments(rows, 45))
    seen = {}
    for row in assigned:
        seen.setdefault(row["family_id"], row["outer_fold"])
        assert seen[row["family_id"]] == row["outer_fold"]


def test_leakage_audit_rejects_overlap(rows):
    with pytest.raises(RuntimeError, match="leakage"):
        leakage_audit(rows[:1], rows[:1], None)


def test_seed45_smoke_is_audit_only_and_color_complete(rows):
    assigned = apply_assignment(rows, family_assignments(rows, 45))
    smoke, audit = select_smoke_families([row for row in assigned if row["outer_fold"] == 0])
    assert len({row["family_id"] for row in smoke}) == 4
    assert len({row["phase0_normalized_answer"] for row in smoke}) >= 4
    assert audit["formal_manifest_opened"] is False


def test_seed42_frozen_cardinality(rows):
    assigned = apply_assignment(rows, family_assignments(rows, 42))
    assert sum(row["outer_fold"] == 0 for row in assigned) == 230
    assert len({row["family_id"] for row in assigned if row["outer_fold"] == 0}) == 25
