from __future__ import annotations

from SA_trajectory.PANL2CLE.config import CAPTURE_ROOT, STEERING_ROOT
from SA_trajectory.PANL2CLE.contracts import load_jsonl
from SA_trajectory.PANL2CLE.prepare import split_manifests


def test_frozen_500_case_split_contract():
    construction,audit,summary=split_manifests(load_jsonl(CAPTURE_ROOT/"results.jsonl"),load_jsonl(STEERING_ROOT/"test_manifest.jsonl"))
    assert (len(construction),len(audit))==(252,69);assert summary["item_overlap"]==summary["image_overlap"]==0
    assert sorted({row["outer_fold"] for row in construction})==[1,2,3,4]
