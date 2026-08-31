from __future__ import annotations

from collections import Counter

import pytest

from dp_SA.answer_matched_lat_steering.manifests import (
    build_families, build_formal_design, build_smoke_design, leakage_audit,
)


def _row(case, item, image):
    return {"case_id": case, "item_id": item, "image_sha256": image}


def test_family_union_connects_shared_images():
    rows = [_row("a", "1", "x"), _row("b", "1", "y"), _row("c", "2", "x"), _row("d", "3", "z")]
    families, mapping, audit = build_families(rows)
    assert mapping["1"] == mapping["2"] != mapping["3"]
    assert Counter(len(row["item_ids"]) for row in families) == Counter({2: 1, 1: 1})
    assert audit["cross_item_shared_image_hashes"] == {"x": ["1", "2"]}


@pytest.fixture(scope="module")
def formal_design():
    return build_formal_design()


def test_formal_split_distribution_and_gate(formal_design):
    assert len(formal_design["candidates"]) == 1625
    assert len(formal_design["families"]) == 178
    assert len(formal_design["test"]) == 174
    counts = formal_design["test_distribution"]
    assert counts["blue"] == {"high_text": 4, "high_image": 5, "total": 9}
    assert all(value["total"] == 15 for answer, value in counts.items() if answer != "blue")
    gate = leakage_audit(formal_design, smoke=False)
    assert gate["status"] == "passed" and gate["loao_minimum"] >= 7
    assert formal_design["comparison"]["grouped_crossfit_15"]["eligible_total"] > formal_design["comparison"]["grouped_crossfit_12"]["eligible_total"]


def test_test_representatives_are_family_unique_and_condition_balanced(formal_design):
    test = formal_design["test"]
    assert len({row["family_id"] for row in test}) == len(test)
    conditions = Counter(row["condition"] for row in test)
    assert conditions == Counter({"conflict_easy": 87, "conflict_hard": 87})
    assert all(row["test_status"] == ("exploratory_sparse" if row["test_answer"] == "blue" else "confirmatory") for row in test)


def test_every_eligible_construction_cell_has_fifteen_families(formal_design):
    for row in formal_design["construction_distribution"]:
        if row["eligible_for_direction"]:
            assert row["construction_high_text_family_count"] >= 15
            assert row["construction_high_image_family_count"] >= 15
        assert row["eligible_answer_count"] >= 8


def test_smoke_construction_is_not_marked_heldout():
    design = build_smoke_design()
    heldout = {row["family_id"] for row in design["fold_assignments"] if row["fold"] == 0}
    construction = {row["family_id"] for row in design["construction_cells"]}
    assert len(heldout) == 4 and len(construction) == 16
    assert not heldout & construction
