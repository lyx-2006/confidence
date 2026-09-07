from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean, median, pstdev

import numpy as np


ROOT = Path(__file__).resolve().parent / "output" / "results"
OUT = ROOT / "artifacts" / "diagnostics" / "probe_robustness"
PRED = ROOT / "artifacts" / "probes" / "audit_predictions.jsonl"
CONSTRUCTION = ROOT / "artifacts" / "manifests" / "construction_manifest.jsonl"
AUDIT = ROOT / "artifacts" / "manifests" / "audit_manifest.jsonl"
VECTOR_AUDIT = ROOT / "artifacts" / "diagnostics" / "vector_audit.jsonl"


def percentile(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q, method="linear"))


def logit_probability(p: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def stats(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = predicted - actual
    ae = np.abs(err)
    sd = float(np.std(actual, ddof=1)) if len(actual) > 1 else 0.0
    iqr = percentile(actual, 75) - percentile(actual, 25)
    mean0, median0 = float(np.mean(actual)), float(np.median(actual))
    return {
        "n": int(len(actual)), "actual_mean": mean0, "actual_median": median0,
        "actual_sd": sd, "actual_iqr": float(iqr), "actual_min": float(np.min(actual)),
        "actual_p01": percentile(actual, 1), "actual_p05": percentile(actual, 5),
        "actual_p25": percentile(actual, 25), "actual_p75": percentile(actual, 75),
        "actual_p95": percentile(actual, 95), "actual_p99": percentile(actual, 99),
        "actual_max": float(np.max(actual)), "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err * err))), "bias": float(np.mean(err)),
        "mae_over_sd": float(np.mean(ae) / max(sd, 1e-12)),
        "mae_over_iqr": float(np.mean(ae) / max(iqr, 1e-12)),
        "baseline_mean_mae": float(np.mean(np.abs(actual - mean0))),
        "baseline_median_mae": float(np.mean(np.abs(actual - median0))),
        "mae_over_mean_baseline": float(np.mean(ae) / max(np.mean(np.abs(actual - mean0)), 1e-12)),
        "mae_over_median_baseline": float(np.mean(ae) / max(np.mean(np.abs(actual - median0)), 1e-12)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def main() -> None:
    predictions = load_jsonl(PRED)
    construction = load_jsonl(CONSTRUCTION)
    audit = load_jsonl(AUDIT)
    vector_audit = load_jsonl(VECTOR_AUDIT)
    by_key: dict[tuple[str, str, int, str], dict[str, dict]] = {}
    for row in predictions:
        key = (str(row["case_id"]), str(row["position"]), int(row["layer"]), str(row["target"]))
        by_key.setdefault(key, {})["prediction"] = row
    # The audit prediction file carries all C_i/C_t/G_L targets for the same cell.
    cells: dict[tuple[str, str, int], dict[str, dict]] = {}
    for row in predictions:
        key = (str(row["case_id"]), str(row["position"]), int(row["layer"]))
        cells.setdefault(key, {})[str(row["target"])] = row

    # Verify the construction manifest's definition and probability values independently.
    c_map = {str(row["case_id"]): row for row in construction}
    formula_errors = []
    for row in construction:
        ci, ct = float(row["C_i"]), float(row["C_t"])
        formula_errors.append(abs(float(row["G_L"]) - float(logit_probability(np.asarray([ci]))[0] - logit_probability(np.asarray([ct]))[0])))

    summary_rows = []
    clipped_rows = []
    gc_rows = []
    decile_rows = []
    tail_rows = []
    epsilons = (1e-6, 1e-4)
    for (case_id, position, layer), target_rows in sorted(cells.items()):
        if not {"C_i", "C_t", "G_L"}.issubset(target_rows):
            continue
        ci = float(target_rows["C_i"]["actual"]); ct = float(target_rows["C_t"]["actual"])
        pred_ci = float(target_rows["C_i"]["predicted"]); pred_ct = float(target_rows["C_t"]["predicted"])
        gl = float(target_rows["G_L"]["actual"]); pred_gl = float(target_rows["G_L"]["predicted"])
        # Each cell is one case; aggregate below by position/layer.
        target_rows["_values"] = {"ci": ci, "ct": ct, "pred_ci": pred_ci, "pred_ct": pred_ct, "gl": gl, "pred_gl": pred_gl}

    groups: dict[tuple[str, int], list[dict]] = {}
    for (case_id, position, layer), target_rows in cells.items():
        if "_values" in target_rows:
            groups.setdefault((position, layer), []).append(target_rows["_values"])

    for (position, layer), rows in sorted(groups.items()):
        actual_gl = np.asarray([r["gl"] for r in rows], dtype=float)
        pred_gl = np.asarray([r["pred_gl"] for r in rows], dtype=float)
        base = {"position": position, "layer": layer, "target": "G_L", **stats(actual_gl, pred_gl)}
        summary_rows.append(base)
        actual_gc = np.asarray([r["ci"] - r["ct"] for r in rows], dtype=float)
        pred_gc = np.asarray([r["pred_ci"] - r["pred_ct"] for r in rows], dtype=float)
        gc_rows.append({"position": position, "layer": layer, "target": "G_C=C_i-C_t", **stats(actual_gc, pred_gc)})
        order = np.argsort(actual_gl, kind="mergesort")
        deciles = np.empty(len(order), dtype=int)
        deciles[order] = np.minimum(10, (np.arange(len(order)) * 10 // len(order)) + 1)
        for decile in range(1, 11):
            mask = deciles == decile
            a, p = actual_gl[mask], pred_gl[mask]
            err = p - a
            decile_rows.append({"position": position, "layer": layer, "target": "G_L", "decile": decile, "n": int(mask.sum()), "actual_mean": float(np.mean(a)), "actual_min": float(np.min(a)), "actual_max": float(np.max(a)), "mae": float(np.mean(np.abs(err))), "median_abs_error": float(np.median(np.abs(err))), "bias": float(np.mean(err)), "rmse": float(np.sqrt(np.mean(err * err)))})
        abs_err = np.abs(pred_gl - actual_gl)
        order_abs = np.argsort(actual_gl, kind="mergesort")
        trim_rows = []
        for trim_fraction in (0.0, 0.01, 0.05):
            lo = int(math.floor(len(actual_gl) * trim_fraction))
            hi = len(actual_gl) - lo
            keep = order_abs[lo:hi]
            trim_rows.append((trim_fraction, float(np.mean(abs_err[keep])), int(len(keep))))
        bottom = deciles == 1
        top = deciles == 10
        total_abs = float(np.sum(abs_err))
        tail_rows.append({
            "position": position, "layer": layer, "target": "G_L", "overall_mae": float(np.mean(abs_err)),
            "trim_1pct_mae": trim_rows[1][1], "trim_5pct_mae": trim_rows[2][1],
            "bottom_decile_mae": float(np.mean(abs_err[bottom])), "top_decile_mae": float(np.mean(abs_err[top])),
            "bottom_decile_abs_error_share": float(np.sum(abs_err[bottom]) / max(total_abs, 1e-12)),
            "top_decile_abs_error_share": float(np.sum(abs_err[top]) / max(total_abs, 1e-12)),
            "both_tail_deciles_abs_error_share": float(np.sum(abs_err[bottom | top]) / max(total_abs, 1e-12)),
            "n": int(len(actual_gl)),
        })
        for eps in epsilons:
            clipped = logit_probability(np.asarray([r["ci"] for r in rows]), eps) - logit_probability(np.asarray([r["ct"] for r in rows]), eps)
            err = pred_gl - clipped
            s = stats(clipped, pred_gl)
            clipped_rows.append({"position": position, "layer": layer, "target": "G_L_clipped", "clip_epsilon": eps, **s})

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "gl_robustness_summary.csv", summary_rows)
    write_csv(OUT / "gl_decile_errors.csv", decile_rows)
    write_csv(OUT / "gl_clipped_metrics.csv", clipped_rows)
    write_csv(OUT / "gc_probe_metrics.csv", gc_rows)
    write_csv(OUT / "gl_tail_sensitivity.csv", tail_rows)
    audit_payload = {
        "status": "complete", "prediction_rows": len(predictions), "construction_rows": len(construction),
        "audit_rows": len(audit), "vector_audit_rows": len(vector_audit), "cells": len(cells),
        "positions": sorted({k[1] for k in cells}), "layers": sorted({k[2] for k in cells}),
        "construction_gl_formula_max_abs_error": float(max(formula_errors)),
        "missing_requested_files": [
            "artifacts/construction_records.jsonl",
            "artifacts/direction_audit_records.jsonl",
        ],
        "substitutes_used": {
            "construction_records": str(CONSTRUCTION.relative_to(ROOT)),
            "direction_audit_records": str(VECTOR_AUDIT.relative_to(ROOT)),
        },
        "clipping_definition": "logit(clip(C_i, epsilon, 1-epsilon))-logit(clip(C_t, epsilon, 1-epsilon))",
        "gc_definition": "C_i-C_t; prediction is independently predicted C_i minus independently predicted C_t",
        "read_only_inputs": [str(PRED), str(CONSTRUCTION), str(AUDIT), str(VECTOR_AUDIT)],
    }
    (OUT / "analysis_audit.json").write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n")
    (OUT / "README.md").write_text("""# Probe robustness diagnostics\n\nCPU-only, read-only analysis of the trajectory audit predictions. Existing result files were not modified.\n\n- `gl_robustness_summary.csv`: G_L distribution, MAE/SD, MAE/IQR, mean/median baseline MAE and ratios by position × layer.\n- `gl_decile_errors.csv`: G_L actual-value deciles and error statistics.\n- `gl_clipped_metrics.csv`: the existing G_L probe predictions evaluated against clipped log-odds `logit(clip(C_i, eps))-logit(clip(C_t, eps))` for eps=1e-6 and 1e-4.\n- `gc_probe_metrics.csv`: bounded target `G_C=C_i-C_t`, using independently predicted C_i minus independently predicted C_t.\n- `analysis_audit.json`: input counts, formula checks, and missing-file/substitute audit.\n\nThe requested `construction_records.jsonl` and `direction_audit_records.jsonl` names were absent. Their available counterparts were used: `artifacts/manifests/construction_manifest.jsonl` and `artifacts/diagnostics/vector_audit.jsonl`.\n""")
    print(json.dumps(audit_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
