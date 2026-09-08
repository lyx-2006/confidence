from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


SEED = 42
BOOTSTRAP_REPEATS = 2000
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PACKAGE_ROOT.parent / "output" / "tables" / "real_sa_per_case.csv"
DEFAULT_OUTPUT = PACKAGE_ROOT / "output"


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    family_id: str
    verbal_sa: float
    gr: float


@dataclass(frozen=True)
class RegressionResult:
    slope: float
    intercept: float
    r2: float
    pearson: float
    spearman: float
    mae: float
    predicted: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class AnalysisResult:
    input_row_count: int
    case_count: int
    family_count: int
    valid_bootstrap_repeats: int
    table1_path: Path
    table2_path: Path
    predictions_path: Path
    figure_path: Path


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_cases(path: Path) -> tuple[int, list[CaseRecord]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "family_id", "verbal_sa", "G_R"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input is missing columns: {sorted(missing)}")
        raw_rows = list(reader)

    cases: list[CaseRecord] = []
    for row in raw_rows:
        verbal_sa = _finite_float(row.get("verbal_sa"))
        gr = _finite_float(row.get("G_R"))
        if verbal_sa is None or gr is None:
            continue
        case_id, family_id = str(row["case_id"]), str(row["family_id"])
        if not case_id or not family_id:
            raise ValueError("Finite analysis rows require case_id and family_id")
        phi_i, phi_t = _finite_float(row.get("phi_I")), _finite_float(row.get("phi_T"))
        if phi_i is not None and phi_t is not None and not math.isclose(gr, phi_i - phi_t, abs_tol=1e-12):
            raise ValueError(f"G_R != phi_I - phi_T for {case_id}")
        cases.append(CaseRecord(case_id, family_id, verbal_sa, gr))

    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Valid input contains duplicate case_id values")
    if len(cases) < 2:
        raise ValueError("At least two finite cases are required")
    return len(raw_rows), cases


def fit_ols(x: np.ndarray, y: np.ndarray) -> RegressionResult:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("OLS requires aligned one-dimensional arrays with at least two cases")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OLS input must be finite")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("OLS and correlation metrics require non-constant variables")

    design = np.column_stack((np.ones(x.size, dtype=np.float64), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = intercept + slope * x
    residual = y - predicted
    ss_total = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - np.sum(residual**2) / ss_total
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(spearmanr(x, y).statistic)
    mae = float(np.mean(np.abs(residual)))
    values = (intercept, slope, r2, pearson, spearman, mae)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Regression produced a non-finite metric")
    return RegressionResult(
        slope=float(slope), intercept=float(intercept), r2=float(r2),
        pearson=pearson, spearman=spearman, mae=mae,
        predicted=predicted, residual=residual,
    )


def cluster_bootstrap(cases: list[CaseRecord], repeats: int = BOOTSTRAP_REPEATS,
                      seed: int = SEED) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]], int]:
    families = sorted({case.family_id for case in cases})
    by_family = {family: [case for case in cases if case.family_id == family] for family in families}
    rng = np.random.default_rng(seed)
    metrics1: list[list[float]] = []
    metrics2: list[list[float]] = []
    for _ in range(repeats):
        sampled = rng.choice(families, size=len(families), replace=True)
        replicate = [case for family in sampled for case in by_family[str(family)]]
        signed_verbal = np.asarray([2.0 * case.verbal_sa - 1.0 for case in replicate], dtype=np.float64)
        gr = np.asarray([case.gr for case in replicate], dtype=np.float64)
        try:
            model1, model2 = fit_ols(gr, signed_verbal), fit_ols(signed_verbal, gr)
        except ValueError:
            continue
        metrics1.append([model1.intercept, model1.slope, model1.r2, model1.pearson,
                         model1.spearman, model1.mae])
        metrics2.append([model2.slope, model2.intercept, model2.r2, model2.pearson,
                         model2.spearman, model2.mae])
    if not metrics1:
        raise ValueError("No valid family-cluster bootstrap replicates")

    names1 = ("intercept", "slope", "r2", "pearson", "spearman", "mae")
    names2 = ("slope", "intercept", "r2", "pearson", "spearman", "mae")
    def intervals(values: list[list[float]], names: Iterable[str]) -> dict[str, tuple[float, float]]:
        array = np.asarray(values, dtype=np.float64)
        bounds = np.percentile(array, [2.5, 97.5], axis=0)
        return {name: (float(bounds[0, index]), float(bounds[1, index]))
                for index, name in enumerate(names)}
    return intervals(metrics1, names1), intervals(metrics2, names2), len(metrics1)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _metric_row(result: RegressionResult, ci: dict[str, tuple[float, float]]) -> dict[str, float]:
    row: dict[str, float] = {
        "r2": result.r2, "pearson": result.pearson,
        "spearman": result.spearman, "mae": result.mae,
    }
    for metric in ("r2", "pearson", "spearman", "mae"):
        row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = ci[metric]
    return row


def _make_figure(path: Path, cases: list[CaseRecord], model1: RegressionResult,
                 model2: RegressionResult) -> None:
    signed_verbal = np.asarray([2.0 * case.verbal_sa - 1.0 for case in cases], dtype=np.float64)
    gr = np.asarray([case.gr for case in cases], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    specifications = (
        (axes[0], gr, signed_verbal, model1, "G_R", "signed_verbal_sa", "Signed verbal SA from Real SA"),
        (axes[1], signed_verbal, gr, model2, "signed_verbal_sa", "G_R", "Real SA from signed verbal SA"),
    )
    for axis, x, y, model, xlabel, ylabel, title in specifications:
        axis.scatter(x, y, s=25, alpha=0.7, color="#356A9A", edgecolors="none")
        line_x = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        axis.plot(line_x, model.intercept + model.slope * line_x, color="#C44E52", linewidth=2)
        axis.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--", alpha=0.7)
        axis.axvline(0.0, color="0.35", linewidth=1.0, linestyle="--", alpha=0.7)
        axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
        axis.grid(alpha=0.2)
        annotation = (f"$R^2$ = {model.r2:.3f}\nPearson = {model.pearson:.3f}\n"
                      f"Spearman = {model.spearman:.3f}\nn = {len(cases)}")
        axis.text(0.04, 0.96, annotation, transform=axis.transAxes, va="top", ha="left",
                  bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(temporary, dpi=200, bbox_inches="tight")
    plt.close(fig)
    temporary.replace(path)


def run_analysis(input_path: Path = DEFAULT_INPUT, output_root: Path = DEFAULT_OUTPUT,
                 repeats: int = BOOTSTRAP_REPEATS, seed: int = SEED) -> AnalysisResult:
    input_rows, cases = load_cases(input_path)
    signed_verbal = np.asarray([2.0 * case.verbal_sa - 1.0 for case in cases], dtype=np.float64)
    gr = np.asarray([case.gr for case in cases], dtype=np.float64)
    model1, model2 = fit_ols(gr, signed_verbal), fit_ols(signed_verbal, gr)
    ci1, ci2, valid_repeats = cluster_bootstrap(cases, repeats=repeats, seed=seed)
    family_count = len({case.family_id for case in cases})

    tables = output_root / "tables"
    artifacts = output_root / "artifacts"
    figures = output_root / "figures"
    table1_path = tables / "table1_verbal_sa_from_gr.csv"
    table2_path = tables / "table2_gr_from_verbal_sa.csv"
    predictions_path = artifacts / "predictions.csv"
    figure_path = figures / "fig1_bidirectional_relationship.png"

    common = {"case_count": len(cases), "family_count": family_count,
              "bootstrap_repeats": repeats, "valid_bootstrap_repeats": valid_repeats}
    row1: dict[str, Any] = {
        "model": "signed_verbal_sa = intercept_a + slope_b * G_R", "outcome": "signed_verbal_sa", "predictor": "G_R",
        **common, "intercept_a": model1.intercept, "slope_b": model1.slope,
        **_metric_row(model1, ci1),
        "intercept_ci_low": ci1["intercept"][0], "intercept_ci_high": ci1["intercept"][1],
        "slope_ci_low": ci1["slope"][0], "slope_ci_high": ci1["slope"][1],
    }
    row2: dict[str, Any] = {
        "model": "G_R = slope_a * signed_verbal_sa + intercept_b", "outcome": "G_R", "predictor": "signed_verbal_sa",
        **common, "slope_a": model2.slope, "intercept_b": model2.intercept,
        **_metric_row(model2, ci2),
        "slope_ci_low": ci2["slope"][0], "slope_ci_high": ci2["slope"][1],
        "intercept_ci_low": ci2["intercept"][0], "intercept_ci_high": ci2["intercept"][1],
    }
    fields1 = ["model", "outcome", "predictor", "case_count", "family_count", "intercept_a", "slope_b",
               "r2", "pearson", "spearman", "mae", "intercept_ci_low", "intercept_ci_high",
               "slope_ci_low", "slope_ci_high", "r2_ci_low", "r2_ci_high", "pearson_ci_low",
               "pearson_ci_high", "spearman_ci_low", "spearman_ci_high", "mae_ci_low", "mae_ci_high",
               "bootstrap_repeats", "valid_bootstrap_repeats"]
    fields2 = ["model", "outcome", "predictor", "case_count", "family_count", "slope_a", "intercept_b",
               "r2", "pearson", "spearman", "mae", "slope_ci_low", "slope_ci_high",
               "intercept_ci_low", "intercept_ci_high", "r2_ci_low", "r2_ci_high", "pearson_ci_low",
               "pearson_ci_high", "spearman_ci_low", "spearman_ci_high", "mae_ci_low", "mae_ci_high",
               "bootstrap_repeats", "valid_bootstrap_repeats"]
    _write_csv(table1_path, fields1, [row1])
    _write_csv(table2_path, fields2, [row2])

    prediction_fields = ["case_id", "family_id", "verbal_sa", "signed_verbal_sa", "G_R",
                         "predicted_signed_verbal_sa", "residual_signed_verbal_sa",
                         "predicted_G_R", "residual_G_R"]
    prediction_rows = [{
        "case_id": case.case_id, "family_id": case.family_id,
        "verbal_sa": case.verbal_sa, "signed_verbal_sa": signed_verbal[index], "G_R": case.gr,
        "predicted_signed_verbal_sa": model1.predicted[index],
        "residual_signed_verbal_sa": model1.residual[index],
        "predicted_G_R": model2.predicted[index], "residual_G_R": model2.residual[index],
    } for index, case in enumerate(cases)]
    _write_csv(predictions_path, prediction_fields, prediction_rows)
    _make_figure(figure_path, cases, model1, model2)
    return AnalysisResult(input_rows, len(cases), family_count, valid_repeats,
                          table1_path, table2_path, predictions_path, figure_path)


__all__ = [
    "AnalysisResult", "BOOTSTRAP_REPEATS", "CaseRecord", "DEFAULT_INPUT", "DEFAULT_OUTPUT",
    "RegressionResult", "SEED", "cluster_bootstrap", "fit_ols", "load_cases", "run_analysis",
]
