from __future__ import annotations

import csv
from pathlib import Path

import pytest

from dp_SA.answer_matched_lat_steering.analyze import analyze, bh_fdr, build_delta_table, family_dose_metrics, validate_trial_pairing
from dp_SA.answer_matched_lat_steering.config import POSITIONS, SMOKE_DIRECTIONS, SMOKE_LAYERS
from dp_SA.answer_matched_lat_steering.io_utils import atomic_csv, atomic_json, atomic_jsonl


def _test_rows():
    answers = ("brown", "cyan", "orange", "yellow")
    return [{"case_id": f"c{i}", "family_id": f"f{i}", "item_id": str(i), "fold": 0, "condition": "conflict_easy" if i % 2 == 0 else "conflict_hard", "test_answer": answer, "test_side": "high_text" if i % 2 == 0 else "high_image", "test_status": "smoke_only"} for i, answer in enumerate(answers)]


def _trials():
    output = []
    for item, test in enumerate(_test_rows()):
        for position_index, position in enumerate(POSITIONS, 1):
            for direction_index, direction in enumerate(SMOKE_DIRECTIONS, 1):
                for layer in SMOKE_LAYERS:
                    for alpha in (-2.0, 0.0, 2.0):
                        value = alpha * direction_index * position_index * layer * 1e-4
                        output.append({"status": "completed", **test, "answer": test["test_answer"], "position": position, "direction": direction, "layer": layer, "alpha": alpha, "delta_soft_sa": value, "margin_change": value / 2, "hard_class_changed": False, "alpha_zero_parity": {"passed": True} if alpha == 0 else None})
    return output


def test_bh_and_family_dose_metrics():
    assert bh_fdr([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    metrics = family_dose_metrics([row for row in _trials() if row["position"] == "P1_LAT" and row["direction"] == "matched_loao" and row["layer"] == 9])
    assert metrics["f0"]["slope"] > 0 and metrics["f0"]["symmetric_effect_2"] > 0


def test_answer_equal_is_not_family_micro_when_answer_counts_differ():
    test = [{"family_id": "a", "test_answer": "brown", "test_side": "high_text", "test_status": "confirmatory"}, {"family_id": "b", "test_answer": "cyan", "test_side": "high_image", "test_status": "confirmatory"}, {"family_id": "c", "test_answer": "cyan", "test_side": "high_image", "test_status": "confirmatory"}]
    trials = [{"family_id": row["family_id"], "direction": "matched_loao", "layer": 10, "alpha": 2.0, "delta_soft_sa": value} for row, value in zip(test, (0.0, 1.0, 1.0))]
    table = build_delta_table(trials, test, repeats=20, seed=42)
    assert table[0]["total_delta_sa_answer_equal"] == pytest.approx(0.5)
    assert table[0]["family_micro_delta_sa"] == pytest.approx(2 / 3)


def test_smoke_analysis_writes_two_tables_and_four_figures(tmp_path: Path):
    atomic_jsonl(tmp_path / "artifacts/manifests/test_manifest.jsonl", _test_rows())
    atomic_jsonl(tmp_path / "artifacts/manifests/construction_distribution.jsonl", [{"fold": 0, "answer": answer, "construction_high_text_family_count": 2, "construction_high_image_family_count": 2, "eligible_for_direction": True, "eligible_answer_count": 4} for answer in ("brown", "cyan", "orange", "yellow")])
    atomic_json(tmp_path / "progress/split_gate.json", {"status": "passed", "folds": [{"fold": 0, "family_leakage_count": 0, "item_leakage_count": 0, "image_hash_leakage_count": 0, "case_leakage_count": 0}]})
    atomic_jsonl(tmp_path / "artifacts/diagnostics/steering_trials.jsonl", _trials())
    probe_rows = []
    for position in POSITIONS:
        for layer in SMOKE_LAYERS:
            probe_rows.append({"position": position, "layer": layer, "r2": .1, "r2_ci_low": 0, "r2_ci_high": .2, "pearson": .3, "pearson_ci_low": .2, "pearson_ci_high": .4, "spearman": .25, "spearman_ci_low": .1, "spearman_ci_high": .35, "mae": .1, "mae_ci_low": .05, "mae_ci_high": .15, "sample_count": 4, "family_count": 4, "valid_bootstrap_repeats": 30})
    atomic_csv(tmp_path / "tables/table2_lat_panl_probe.csv", probe_rows)
    atomic_json(tmp_path / "progress/probe_progress.json", {"status": "complete", "cell_count": 8, "config_fingerprint": "probe"})
    result = analyze(output_root=tmp_path, smoke=True, repeats=30)
    assert result["trial_count"] == 192 and result["figures"] == 4
    assert {path.name for path in (tmp_path / "tables").iterdir()} == {"table1_lat_panl_steering.csv", "table2_lat_panl_probe.csv", "README.md"}
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
    with (tmp_path / "tables/table1_lat_panl_steering.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and "red_delta_sa" in rows[0] and {row["position"] for row in rows} == set(POSITIONS)
    assert analyze(output_root=tmp_path, smoke=True, resume=True, repeats=30)["resumed_noop"]


def test_position_pairing_gate_rejects_missing_row():
    rows = _trials()
    validate_trial_pairing(rows)
    with pytest.raises(ValueError, match="not paired"):
        validate_trial_pairing(rows[:-1])
