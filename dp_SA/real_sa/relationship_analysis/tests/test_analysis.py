from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from dp_SA.real_sa.relationship_analysis.analysis import (
    CaseRecord, cluster_bootstrap, fit_ols, load_cases, run_analysis,
)


def _write_input(path: Path) -> None:
    fields = ["case_id", "family_id", "verbal_sa", "G_R", "phi_I", "phi_T", "sign_type", "R_I_eligible"]
    rows = [
        ["c1", "f1", "1", "0", "0.2", "0.2", "both_support", "True"],
        ["c2", "f1", "3", "1", "1.2", "0.2", "weak_total_effect", "False"],
        ["c3", "f2", "5", "2", "2.2", "0.2", "image_support_text_suppress", "False"],
        ["c4", "f2", "7", "3", "3.2", "0.2", "text_support_image_suppress", "True"],
        ["bad", "f3", "nan", "4", "4.2", "0.2", "both_support", "True"],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)


def test_load_filters_only_nonfinite_values(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"; _write_input(source)
    total, cases = load_cases(source)
    assert total == 5 and len(cases) == 4
    assert {case.case_id for case in cases} == {"c1", "c2", "c3", "c4"}


def test_bidirectional_ols_metrics() -> None:
    x = np.asarray([0.0, 1.0, 2.0, 3.0])
    y = 1.0 + 2.0 * x
    forward, reverse = fit_ols(x, y), fit_ols(y, x)
    assert np.isclose(forward.intercept, 1.0) and np.isclose(forward.slope, 2.0)
    assert np.isclose(reverse.intercept, -0.5) and np.isclose(reverse.slope, 0.5)
    assert np.isclose(forward.r2, reverse.r2) and np.isclose(forward.pearson, reverse.pearson)
    assert np.isclose(forward.spearman, reverse.spearman)
    assert np.allclose(forward.residual, y - forward.predicted)


def test_signed_verbal_transform_in_formal_models(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"; _write_input(source)
    result = run_analysis(source, tmp_path / "output", repeats=20, seed=42)
    with result.table1_path.open(newline="", encoding="utf-8") as handle:
        model1 = next(csv.DictReader(handle))
    with result.table2_path.open(newline="", encoding="utf-8") as handle:
        model2 = next(csv.DictReader(handle))
    assert model1["outcome"] == "signed_verbal_sa" and model1["predictor"] == "G_R"
    assert np.isclose(float(model1["intercept_a"]), 1.0)
    assert np.isclose(float(model1["slope_b"]), 4.0)
    assert model2["outcome"] == "G_R" and model2["predictor"] == "signed_verbal_sa"


def test_cluster_bootstrap_is_deterministic() -> None:
    cases = [CaseRecord(f"c{i}", f"f{i // 2}", float(i + (i % 2)), float(i)) for i in range(8)]
    first = cluster_bootstrap(cases, repeats=50, seed=42)
    second = cluster_bootstrap(cases, repeats=50, seed=42)
    assert first == second and 0 < first[2] <= 50


def test_formal_outputs_and_schema(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"; _write_input(source)
    result = run_analysis(source, tmp_path / "output", repeats=50, seed=42)
    assert result.input_row_count == 5 and result.case_count == 4 and result.family_count == 2
    for path in (result.table1_path, result.table2_path, result.predictions_path, result.figure_path):
        assert path.is_file() and path.stat().st_size > 0
    with result.table1_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["outcome"] == "signed_verbal_sa" and row["predictor"] == "G_R"
    assert row["bootstrap_repeats"] == "50"
    with result.predictions_path.open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 4
    assert "signed_verbal_sa" in predictions[0]
    assert "predicted_signed_verbal_sa" in predictions[0]
