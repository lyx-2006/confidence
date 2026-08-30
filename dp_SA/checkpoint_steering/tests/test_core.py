from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dp_SA.checkpoint_steering.analyze import (
    build_dose_metrics,
    build_long_metrics,
    build_position_contrasts,
    build_wide,
    item_dose_metrics,
)
from dp_SA.checkpoint_steering.config import ALPHAS, LAYERS, POSITIONS, VECTOR_NORM_FRACTION
from dp_SA.checkpoint_steering.fingerprint import check_or_write_config
from dp_SA.checkpoint_steering.io_utils import atomic_json, atomic_npz
from dp_SA.checkpoint_steering.run import validate_alpha_zero
from dp_SA.checkpoint_steering.vectors import build_vectors, construct_direction


def _trials(alphas=(-10.0, -2.0, 0.0, 2.0, 10.0), layers=(8, 14)):
    rows = []
    for item in range(4):
        for position_index, position in enumerate(POSITIONS):
            for layer in layers:
                strength = 0.01 * (position_index + 1) + layer * 0.0001
                for alpha in alphas:
                    rows.append({
                        "status": "completed", "case_id": f"c{item}", "item_id": str(item),
                        "test_side": "image_side" if item < 2 else "text_side", "position": position,
                        "layer": layer, "alpha": alpha, "delta_soft_sa": strength * alpha,
                        "hard_class_changed": alpha != 0, "hard_class_delta": 1 if alpha > 0 else -1 if alpha < 0 else 0,
                        "margin_change": strength * alpha, "saturated": False, "finite_values": True, "probability_sum": 1.0,
                    })
    return rows


def test_fixed_layers_alphas_and_direction_scaling():
    assert LAYERS == (8, 10, 14, 18, 20, 24, 26)
    assert ALPHAS == (-10.0, -2.0, 0.0, 2.0, 10.0)
    high = np.asarray([[3.0, 4.0], [5.0, 4.0]], dtype=np.float16)
    low = np.asarray([[1.0, 4.0], [1.0, 4.0]], dtype=np.float16)
    arrays, metadata = construct_direction(high, low)
    assert arrays["raw_vector"][0] > 0 and arrays["raw_vector"][1] == pytest.approx(0)
    assert metadata["scaled_norm"] == pytest.approx(VECTOR_NORM_FRACTION * metadata["mean_residual_norm"], rel=1e-6)


def test_vector_builder_includes_layer8_and_never_reuses_positions(tmp_path: Path):
    construction = []
    clean = []
    for item, side, value in ((0, "high_image", 3.0), (1, "high_image", 4.0), (2, "high_text", 1.0), (3, "high_text", 1.5)):
        case_id = f"c{item}"
        relative = Path("artifacts") / "hidden" / f"{case_id}.npz"
        arrays = {
            f"{position}__L{layer}": np.full(4, value + position_index, dtype=np.float16)
            for position_index, position in enumerate(POSITIONS) for layer in (8, 14)
        }
        atomic_npz(tmp_path / relative, arrays)
        construction.append({"case_id": case_id, "item_id": str(item), "construction_side": side})
        clean.append({"case_id": case_id, "hidden_file": str(relative), "status": "completed"})
    metadata = build_vectors(tmp_path, construction, clean, smoke=True, resume=False)
    assert len(metadata["vectors"]) == len(POSITIONS) * 2
    assert {(row["position"], row["layer"]) for row in metadata["vectors"]} == {(position, layer) for position in POSITIONS for layer in (8, 14)}
    assert len({row["vector_file"] for row in metadata["vectors"]}) == len(metadata["vectors"])


def test_alpha_zero_parity_requires_hook_and_identical_activation():
    logits = np.arange(9, dtype=float)
    probabilities = np.full(9, 1 / 9)
    scored = {"class_logits": logits.tolist(), "class_probabilities": probabilities.tolist(), "soft_sa_image_score": 0.5, "argmax_hard_class": 8}
    activation = np.asarray([1.0, 2.0], dtype=np.float32)
    result = validate_alpha_zero(
        clean_logits=logits, clean_probabilities=probabilities, clean_soft_sa=0.5, clean_hard_class=8,
        scored=scored, before=activation, after=activation.copy(),
        diagnostics={"hook_call_count": 1, "steering_applied_count": 1},
    )
    assert result["passed"]
    with pytest.raises(RuntimeError, match="Alpha-zero parity failed"):
        validate_alpha_zero(
            clean_logits=logits, clean_probabilities=probabilities, clean_soft_sa=0.5, clean_hard_class=8,
            scored=scored, before=activation, after=activation + 1,
            diagnostics={"hook_call_count": 1, "steering_applied_count": 1},
        )


def test_cluster_statistics_dose_symmetry_and_paired_contrasts():
    rows = _trials()
    long_rows = build_long_metrics(rows, repeats=100, seed=42)
    dose = build_dose_metrics(rows, repeats=100, seed=42)
    contrasts = build_position_contrasts(rows, repeats=100, seed=42)
    first = next(row for row in dose if row["position"] == "P1_LAT" and row["layer"] == 8 and row["group"] == "all")
    assert first["slope"] > 0 and first["bidirectional_pass"]
    assert first["symmetric_effect_10"] > 0 and first["asymmetry_10"] == pytest.approx(0)
    transition = next(row for row in contrasts if row["from_position"] == "P1_LAT" and row["to_position"] == "P1_PANL" and row["layer"] == 8 and row["group"] == "all" and row["metric"] == "slope")
    assert transition["contrast"] == pytest.approx(0.01)
    wide, fields = build_wide(long_rows)
    assert fields[0] == "metric" and "P1_LAT__L8__a-10" in fields
    assert {row["metric"] for row in wide} >= {"all__mean_delta_soft_sa", "image_side__sample_count"}


def test_item_dose_rejects_duplicate_alpha():
    rows = _trials(layers=(8,))
    subset = [row for row in rows if row["position"] == "P1_LAT" and row["item_id"] == "0"]
    with pytest.raises(ValueError, match="Duplicate alpha"):
        item_dose_metrics([*subset, dict(subset[0])])


def test_config_resume_refuses_fingerprint_change(tmp_path: Path):
    path = tmp_path / "config.json"
    check_or_write_config(path, {"fingerprint": "a"}, resume=False)
    check_or_write_config(path, {"fingerprint": "a"}, resume=True)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        check_or_write_config(path, {"fingerprint": "b"}, resume=True)
    with pytest.raises(FileExistsError):
        check_or_write_config(path, {"fingerprint": "a"}, resume=False)

