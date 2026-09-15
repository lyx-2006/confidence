from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SHORT_ROOT = HERE.parents[1]
for candidate in (SHORT_ROOT, SHORT_ROOT.parent, SHORT_ROOT.parent.parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from sa_trajectory.panl2cle.analyze import effect_values
from sa_trajectory.panl2cle.config import ALPHAS, CLE_LAYERS, PAIRS, PANL_LAYERS
from sa_trajectory.panl2cle.contracts import expected_logical_count, expected_physical_count, load_jsonl, validate_layer_design
from sa_trajectory.panl2cle.prepare import CAPTURE_ROOT, STEERING_ROOT, build_short_panl_vectors, split_probe_manifests


def test_layer_and_grid_contracts() -> None:
    validate_layer_design()
    assert PANL_LAYERS == (14, 16, 18)
    assert CLE_LAYERS == (15, 17, 19)
    assert PAIRS == ((14, 15), (16, 17), (18, 19))
    assert ALPHAS == (-5.0, 5.0)
    assert expected_physical_count(80) == 1520
    assert expected_logical_count(80) == 1920


def test_short_manifests_and_probe_split() -> None:
    capture = [r for r in load_jsonl(CAPTURE_ROOT / "results.jsonl") if r.get("status") == "completed"]
    test = load_jsonl(STEERING_ROOT / "test_manifest.jsonl")
    construction, audit, summary = split_probe_manifests(capture, test)
    assert (len(construction), len(audit), len(test)) == (267, 55, 80)
    assert summary["item_overlap"] == summary["image_overlap"] == 0
    assert Counter(r["test_side"] for r in test) == Counter({"image_side": 74, "text_side": 6})


def test_short_vector_rebuild_and_parity() -> None:
    vectors, metadata, artifacts, construction = build_short_panl_vectors(CAPTURE_ROOT, STEERING_ROOT)
    assert len(vectors) == len(artifacts) == 3
    assert len(construction) == 50
    assert metadata["existing_short_vector_max_abs_error"] == {"14": 0.0, "16": 0.0}
    assert set(vectors) == {("PANL", 14), ("PANL", 16), ("PANL", 18)}


def test_four_cell_effect_formulas() -> None:
    rows = [
        {"condition": "C0", "value": 1.0},
        {"condition": "C1", "value": 4.0},
        {"condition": "C2", "value": 2.0},
        {"condition": "C3", "value": 3.0},
    ]
    assert effect_values(rows, "value") == {
        "total": 3.0, "residual": 1.0, "attenuation": 2.0,
        "transfer": 2.0, "interaction": 0.0,
    }
