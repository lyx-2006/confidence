from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from SA_trajectory.PANL2CLE.probes import (
    _atomic_joblib, choose_alpha, item_bootstrap, pipeline, probe_value,
    raw_parameters, regression_metrics,
)

from .config import BOOTSTRAP_REPEATS, CLE_LAYERS, RAW_EXPRESSION_ATOL
from .contracts import atomic_json, atomic_jsonl, load_jsonl, sha256_file


def _load_hidden(capture_root: Path, row: dict[str, Any], layer: int) -> np.ndarray:
    with np.load(capture_root / row["hidden_file"], allow_pickle=False) as archive:
        value = np.asarray(archive[f"CLE__L{layer}"], np.float32)
    if value.shape != (4096,) or not np.isfinite(value).all():
        raise ValueError(f"Bad short CLE L{layer} hidden: {row['case_id']}")
    return value


def train_probes(
    root: Path, capture_root: Path, *, resume: bool = False,
    repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, Any]:
    root = Path(root)
    capture_root = Path(capture_root)
    fingerprint = __import__("json").loads((root / "fingerprint.json").read_text())["fingerprint"]
    construction = load_jsonl(root / "artifacts/manifests/probe_construction.jsonl")
    audit = load_jsonl(root / "artifacts/manifests/probe_audit.jsonl")
    index_path = root / "artifacts/probes/probe_index.jsonl"
    existing = load_jsonl(index_path)
    if resume and len(existing) == len(CLE_LAYERS):
        valid = all(
            (root / row["probe_file"]).is_file()
            and sha256_file(root / row["probe_file"]) == row["probe_sha256"]
            for row in existing
        )
        if valid:
            return {
                "status": "complete", "probe_count": len(CLE_LAYERS),
                "resumed_noop": True, "all_reliable": all(r["readout_reliable"] for r in existing),
            }
    folds = np.asarray([int(row["outer_fold"]) for row in construction])
    y_train = np.asarray([float(row["soft_sa_image_score"]) for row in construction])
    y_audit = np.asarray([float(row["soft_sa_image_score"]) for row in audit])
    index: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for layer in CLE_LAYERS:
        x_train = np.stack([_load_hidden(capture_root, row, layer) for row in construction])
        x_audit = np.stack([_load_hidden(capture_root, row, layer) for row in audit])
        alpha, oof, trace = choose_alpha(x_train, y_train, folds)
        model = pipeline(alpha)
        model.fit(x_train, y_train)
        predicted = np.asarray(model.predict(x_audit), float)
        weight, intercept = raw_parameters(model)
        raw = x_audit.astype(np.float64) @ weight + intercept
        residual = predicted - raw
        intercept += float((residual.max() + residual.min()) / 2)
        raw = x_audit.astype(np.float64) @ weight + intercept
        error = float(np.max(np.abs(predicted - raw)))
        if not math.isfinite(error) or error > RAW_EXPRESSION_ATOL:
            raise ValueError(f"Raw expression mismatch at short CLE L{layer}: {error}")
        rows = [
            {"case_id": r["case_id"], "item_id": r["item_id"], "layer": layer,
             "actual": float(y), "predicted": float(p)}
            for r, y, p in zip(audit, y_audit, predicted)
        ]
        metrics = regression_metrics(y_audit, predicted)
        ci = item_bootstrap(rows, repeats)
        reliable = bool(metrics["r2"] > 0 and metrics["pearson"] > 0 and ci["pearson"][0] > 0)
        path = root / f"artifacts/probes/final_soft_sa__CLE__L{layer}.joblib"
        _atomic_joblib(path, {
            "pipeline": model, "target": "short_final_soft_sa", "position": "CLE", "layer": layer,
            "selected_alpha": alpha, "alpha_trace": trace, "raw_weight": weight,
            "raw_intercept": intercept, "config_fingerprint": fingerprint,
            "construction_case_ids": [r["case_id"] for r in construction],
        })
        record = {
            "target": "short_final_soft_sa", "position": "CLE", "layer": layer,
            "selected_alpha": alpha, **metrics,
            **{f"{k}_ci_low": v[0] for k, v in ci.items()},
            **{f"{k}_ci_high": v[1] for k, v in ci.items()},
            "readout_reliable": reliable, "raw_expression_max_abs_error": error,
            "probe_file": str(path.relative_to(root)), "probe_sha256": sha256_file(path),
        }
        index.append(record)
        audit_rows.extend(rows)
        oof_rows.extend(
            {"case_id": r["case_id"], "item_id": r["item_id"], "outer_fold": r["outer_fold"],
             "layer": layer, "actual": float(y), "predicted": float(p)}
            for r, y, p in zip(construction, y_train, oof)
        )
        atomic_json(root / "progress/train_probes.json", {"status": "running", "probe_count": len(index)})
    atomic_jsonl(index_path, index)
    atomic_jsonl(root / "artifacts/probes/audit_predictions.jsonl", audit_rows)
    atomic_jsonl(root / "artifacts/probes/construction_oof_predictions.jsonl", oof_rows)
    result = {
        "status": "complete", "probe_count": len(index), "resumed_noop": False,
        "all_reliable": all(r["readout_reliable"] for r in index),
    }
    atomic_json(root / "progress/train_probes.json", result)
    return result


def load_probes(root: Path) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    fingerprint = __import__("json").loads((root / "fingerprint.json").read_text())["fingerprint"]
    for row in load_jsonl(root / "artifacts/probes/probe_index.jsonl"):
        path = root / row["probe_file"]
        if sha256_file(path) != row["probe_sha256"]:
            raise ValueError(f"Probe hash mismatch: {path}")
        payload = joblib.load(path)
        if payload.get("config_fingerprint") != fingerprint:
            raise ValueError(f"Probe fingerprint mismatch: {path}")
        output[int(row["layer"])] = payload
    return output


__all__ = ["train_probes", "load_probes", "probe_value"]
