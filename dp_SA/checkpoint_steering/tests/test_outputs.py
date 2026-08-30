from __future__ import annotations

import json
from pathlib import Path

from dp_SA.checkpoint_steering.analyze import analyze
from dp_SA.checkpoint_steering.config import POSITIONS
from dp_SA.checkpoint_steering.io_utils import atomic_jsonl, sha256_file
from dp_SA.checkpoint_steering.manifests import prepare_manifests


def _trial_rows():
    rows = []
    for item in range(4):
        for position_index, position in enumerate(POSITIONS):
            for layer in (8, 14):
                for alpha in (-2.0, 0.0, 2.0):
                    value = alpha * (position_index + 1) * 0.002
                    rows.append({
                        "status": "completed", "case_id": f"c{item}", "item_id": str(item),
                        "test_side": "image_side" if item < 2 else "text_side", "position": position,
                        "layer": layer, "alpha": alpha, "processed_position": 10 + position_index,
                        "delta_soft_sa": value, "hard_class_changed": alpha != 0, "hard_class_delta": 0,
                        "margin_change": value, "saturated": False, "finite_values": True, "probability_sum": 1.0,
                        "hook_diagnostics": {"hook_call_count": 1, "steering_applied_count": 1},
                        "alpha_zero_parity": {"passed": True} if alpha == 0 else None,
                        "activation_before_hash": "a", "activation_after_hash": "a" if alpha == 0 else "b",
                    })
    return rows


def _clean_rows():
    rows = []
    all_positions = (*POSITIONS, "P1_SAC")
    for item in range(4):
        positions = {
            position: {
                "processed_index": 10 + index, "rendered_index": 20 + index,
                "token_id": 100 + index, "token_text": "\n", "anchor_text": position,
                "anchor_occurrence_count": 1, "token_window": [],
            }
            for index, position in enumerate(all_positions)
        }
        positions["causal_order"] = {position: 10 + index for index, position in enumerate(all_positions)}
        rows.append({"status": "completed", "case_id": f"c{item}", "item_id": str(item), "positions": positions})
    return rows


def test_smoke_analysis_writes_full_table_and_figure_schema(tmp_path: Path):
    diagnostics = tmp_path / "artifacts" / "diagnostics"
    atomic_jsonl(diagnostics / "steering_trials.jsonl", _trial_rows())
    atomic_jsonl(diagnostics / "clean_capture.jsonl", _clean_rows())
    result = analyze(output_root=tmp_path, smoke=True, resume=False, repeats=50)
    assert result["trial_count"] == 120 and result["figures"] == 6
    expected_tables = {
        "steering_delta_sa_long.csv", "steering_delta_sa_wide.csv", "dose_response_by_position_layer.csv",
        "position_contrasts.csv", "run_audit.csv", "README.md",
    }
    assert expected_tables == {path.name for path in (tmp_path / "tables").iterdir()}
    assert len(list((tmp_path / "figures").glob("*.png"))) == 6
    resumed = analyze(output_root=tmp_path, smoke=True, resume=True, repeats=50)
    assert resumed["resumed_noop"]


def test_manifest_freeze_uses_history_without_modifying_it(tmp_path: Path):
    from dp_SA.checkpoint_steering.config import HISTORICAL_CONSTRUCTION, HISTORICAL_TEST

    before = {path: sha256_file(path) for path in (HISTORICAL_CONSTRUCTION, HISTORICAL_TEST)}
    construction, test, source = prepare_manifests(tmp_path, smoke=True, resume=False)
    after = {path: sha256_file(path) for path in (HISTORICAL_CONSTRUCTION, HISTORICAL_TEST)}
    assert before == after
    assert len(construction) == 4 and len(test) == 4
    assert source["selection_source"] == "historical_frozen_steering_manifests"
    resumed = prepare_manifests(tmp_path, smoke=True, resume=True)
    assert [row["case_id"] for row in resumed[0]] == [row["case_id"] for row in construction]

