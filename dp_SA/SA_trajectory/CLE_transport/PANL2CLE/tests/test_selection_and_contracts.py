from __future__ import annotations

import json
from pathlib import Path

import pytest

from dp_SA.SA_trajectory.CLE_transport.PANL2CLE.config import AUDIT_MANIFEST, CANDIDATE_MANIFEST, CONSTRUCTION_MANIFEST, OUTPUT_PARENT, require_output_root
from dp_SA.SA_trajectory.CLE_transport.PANL2CLE.io_utils import load_jsonl
from dp_SA.SA_trajectory.CLE_transport.PANL2CLE.selection import select_recipients


def test_real_frozen_recipient_selection_contract():
    candidates = load_jsonl(CANDIDATE_MANIFEST); by_id = {row["case_id"]: row for row in candidates}
    construction = [by_id[row["case_id"]] for row in load_jsonl(CONSTRUCTION_MANIFEST) if row["case_id"] in by_id]
    cells = {(row["phase0_raw_answer"], row["sa_side"]) for row in construction}
    allowed = {answer for answer, _side in cells if (answer, "high_image") in cells and (answer, "high_text") in cells}
    rows, audit = select_recipients(candidates, load_jsonl(AUDIT_MANIFEST), load_jsonl(CONSTRUCTION_MANIFEST), allowed_answers=allowed)
    assert len(rows) == 50 and audit["counts"] == {"high_image": 25, "high_text": 25}
    assert audit["audit_count"] == 25 and audit["fallback_count"] == 25
    for field in ("case_id", "family_id", "item_id"): assert len({str(row[field]) for row in rows}) == 50
    assert all(len(audit["answer_counts"][side]) == 11 for side in ("high_image", "high_text"))


def test_output_root_is_contained(tmp_path):
    assert require_output_root(OUTPUT_PARENT / "smoke/round_1").is_absolute()
    with pytest.raises(ValueError): require_output_root(tmp_path)
