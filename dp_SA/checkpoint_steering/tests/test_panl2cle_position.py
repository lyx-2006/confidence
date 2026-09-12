from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from dp_SA.checkpoint_steering.manifests import _smoke_subset
from dp_SA.checkpoint_steering.run_panl2cle_position import (
    ALPHAS,
    EXPECTED_FORMAL_CAPTURES,
    EXPECTED_FORMAL_IMPORTS,
    LAYERS,
    OUTPUT_ROOT,
    POSITIONS,
    SOURCE_CAPTURE,
    _validate_probe_identity,
    expected_hidden_keys,
)
from dp_SA.checkpoint_steering.config import HISTORICAL_CONSTRUCTION, HISTORICAL_TEST
from dp_SA.checkpoint_steering.io_utils import load_jsonl
from dp_SA.checkpoint_steering.vectors import construct_direction
from dp_SA.SA_probe.config import TRAIN_MANIFEST


def test_panl2cle_grid_and_output_boundary():
    assert POSITIONS == (
        "P1_ATTRIBUTION_QUERY_END_NL",
        "P1_INTEGER_INSTRUCTION_END_NL",
        "P1_IMAGE_POLARITY_SENTENCE_END",
        "P1_SCALE_LINE_END_NL",
        "P1_CLASS4_RULE_END_NL",
    )
    assert LAYERS == (16, 18, 20)
    assert ALPHAS == (-10.0, -2.0, 0.0, 2.0, 10.0)
    assert len(expected_hidden_keys()) == 15
    assert OUTPUT_ROOT.name == "PANL2CLE_position"
    assert OUTPUT_ROOT.parent.name == "output"


def test_frozen_protocol_and_probe_overlap_counts():
    construction = load_jsonl(HISTORICAL_CONSTRUCTION)
    test = load_jsonl(HISTORICAL_TEST)
    probe_ids = {str(row["case_id"]) for row in load_jsonl(SOURCE_CAPTURE) if row.get("status") == "completed"}
    all_rows = [*construction, *test]
    assert len(construction) == 50 and Counter(row["construction_side"] for row in construction) == Counter({"high_image": 25, "high_text": 25})
    assert len(test) == 100 and Counter(row["test_side"] for row in test) == Counter({"image_side": 50, "text_side": 50})
    assert not ({str(row["item_id"]) for row in construction} & {str(row["item_id"]) for row in test})
    assert sum(str(row["case_id"]) in probe_ids for row in all_rows) == EXPECTED_FORMAL_IMPORTS
    assert sum(str(row["case_id"]) not in probe_ids for row in all_rows) == EXPECTED_FORMAL_CAPTURES
    smoke_construction, smoke_test = _smoke_subset(construction, test)
    assert sum(str(row["case_id"]) in probe_ids for row in [*smoke_construction, *smoke_test]) == 4


def test_probe_identity_gate_accepts_real_overlap_and_rejects_change():
    construction = {str(row["case_id"]): row for row in load_jsonl(HISTORICAL_CONSTRUCTION)}
    manifests = {str(row["case_id"]): row for row in load_jsonl(TRAIN_MANIFEST)}
    captures = {str(row["case_id"]): row for row in load_jsonl(SOURCE_CAPTURE)}
    case_id = sorted(set(construction) & set(captures))[0]
    _validate_probe_identity(construction[case_id], manifests[case_id], captures[case_id])
    changed = dict(construction[case_id], phase0_raw_answer="definitely-different")
    with pytest.raises(ValueError, match="identity mismatch"):
        _validate_probe_identity(changed, manifests[case_id], captures[case_id])


def test_direction_is_high_image_minus_high_text_and_scaled_three_percent():
    high = np.asarray([[4.0, 2.0], [6.0, 2.0]], dtype=np.float16)
    low = np.asarray([[1.0, 2.0], [1.0, 2.0]], dtype=np.float16)
    arrays, metadata = construct_direction(high, low)
    assert arrays["raw_vector"][0] > 0
    assert arrays["raw_vector"][1] == pytest.approx(0)
    assert metadata["normalization_fraction"] == pytest.approx(0.03)
    assert metadata["scaled_norm"] == pytest.approx(0.03 * metadata["mean_residual_norm"], rel=1e-6)
