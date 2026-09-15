from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import BOOTSTRAP_REPEATS, CLE_LAYERS, RAW_EXPRESSION_ATOL, RIDGE_ALPHAS, SEED
from .contracts import atomic_json, atomic_jsonl, load_jsonl, sha256_file


def pipeline(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=float(alpha), solver="lsqr"))])


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    return {
        "r2": float(r2_score(y, prediction)),
        "pearson": float(pearsonr(y, prediction).statistic) if len(y) > 1 and np.ptp(y) and np.ptp(prediction) else math.nan,
        "spearman": float(spearmanr(y, prediction).statistic) if len(np.unique(y)) > 1 and len(np.unique(prediction)) > 1 else math.nan,
        "mae": float(mean_absolute_error(y, prediction)),
    }


def choose_alpha(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[float, np.ndarray, list[dict[str, float]]]:
    if sorted(set(map(int, folds))) != [1, 2, 3, 4]:
        raise ValueError("Probe construction folds must be 1..4")
    predictions: dict[float, np.ndarray] = {}; trace = []
    for alpha in RIDGE_ALPHAS:
        predicted = np.full(len(y), np.nan)
        for fold in (1, 2, 3, 4):
            train = folds != fold
            model = pipeline(alpha); model.fit(x[train], y[train])
            predicted[~train] = model.predict(x[~train])
        score = float(r2_score(y, predicted))
        trace.append({"alpha": float(alpha), "construction_oof_r2": score})
        predictions[float(alpha)] = predicted
    selected = max(trace, key=lambda row: (row["construction_oof_r2"], row["alpha"]))["alpha"]
    return float(selected), predictions[float(selected)], trace


def raw_parameters(model: Pipeline) -> tuple[np.ndarray, float]:
    scale: StandardScaler = model.named_steps["scale"]; ridge: Ridge = model.named_steps["ridge"]
    weight = np.asarray(ridge.coef_, np.float64).reshape(-1) / np.asarray(scale.scale_, np.float64)
    intercept = float(np.asarray(ridge.intercept_).reshape(-1)[0] if np.asarray(ridge.intercept_).ndim else ridge.intercept_)
    intercept -= float(weight @ np.asarray(scale.mean_, np.float64))
    return weight, intercept


def item_bootstrap(rows: Sequence[dict[str, Any]], repeats: int = BOOTSTRAP_REPEATS) -> dict[str, tuple[float, float]]:
    items = sorted({str(row["item_id"]) for row in rows}); by_item = {item: [r for r in rows if str(r["item_id"]) == item] for item in items}
    rng = np.random.default_rng(SEED); values = {name: [] for name in ("r2", "pearson", "spearman", "mae")}
    for _ in range(repeats):
        sample = [row for item in rng.choice(items, len(items), replace=True) for row in by_item[str(item)]]
        measured = regression_metrics(np.asarray([r["actual"] for r in sample]), np.asarray([r["predicted"] for r in sample]))
        for name, value in measured.items():
            if np.isfinite(value): values[name].append(value)
    return {name: tuple(map(float, np.quantile(series, [.025, .975]))) for name, series in values.items() if series}


def _load_hidden(capture_root: Path, row: dict[str, Any], layer: int) -> np.ndarray:
    with np.load(capture_root / row["hidden_file"], allow_pickle=False) as archive:
        value = np.asarray(archive[f"CLE__L{layer}"], np.float32)
    if value.shape != (4096,) or not np.isfinite(value).all(): raise ValueError(f"Bad CLE L{layer} hidden: {row['case_id']}")
    return value


def _atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent); os.close(fd)
    try: joblib.dump(payload, temporary); os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def train_probes(root: Path, capture_root: Path, *, resume: bool = False, repeats: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    fingerprint = __import__("json").loads((root / "fingerprint.json").read_text())["fingerprint"]
    construction = load_jsonl(root / "artifacts/manifests/probe_construction.jsonl")
    audit = load_jsonl(root / "artifacts/manifests/probe_audit.jsonl")
    index_path = root / "artifacts/probes/probe_index.jsonl"
    if resume and len(load_jsonl(index_path)) == len(CLE_LAYERS):
        index = load_jsonl(index_path)
        if all((root / row["probe_file"]).is_file() and sha256_file(root / row["probe_file"]) == row["probe_sha256"] for row in index):
            return {"status": "complete", "probe_count": 4, "resumed_noop": True, "all_reliable": all(r["readout_reliable"] for r in index)}
    folds = np.asarray([int(row["outer_fold"]) for row in construction]); y_train = np.asarray([row["soft_sa_image_score"] for row in construction], float)
    y_audit = np.asarray([row["soft_sa_image_score"] for row in audit], float); index=[]; oof_rows=[]; audit_rows=[]
    for layer in CLE_LAYERS:
        x_train = np.stack([_load_hidden(capture_root, row, layer) for row in construction]); x_audit = np.stack([_load_hidden(capture_root, row, layer) for row in audit])
        alpha, oof, trace = choose_alpha(x_train, y_train, folds); model = pipeline(alpha); model.fit(x_train, y_train)
        predicted = np.asarray(model.predict(x_audit), float); weight, intercept = raw_parameters(model)
        raw = x_audit.astype(np.float64) @ weight + intercept; residual = predicted - raw; intercept += float((residual.max()+residual.min())/2); raw=x_audit.astype(np.float64)@weight+intercept
        error=float(np.max(np.abs(predicted-raw)))
        if not np.isfinite(error) or error > RAW_EXPRESSION_ATOL: raise ValueError(f"Raw expression mismatch at CLE L{layer}: {error}")
        rows=[{"case_id":r["case_id"],"item_id":r["item_id"],"layer":layer,"actual":float(y),"predicted":float(p)} for r,y,p in zip(audit,y_audit,predicted)]
        metrics=regression_metrics(y_audit,predicted); ci=item_bootstrap(rows,repeats); reliable=bool(metrics["r2"]>0 and metrics["pearson"]>0 and ci["pearson"][0]>0)
        path=root/f"artifacts/probes/final_soft_sa__CLE__L{layer}.joblib"
        _atomic_joblib(path,{"pipeline":model,"target":"final_soft_sa","position":"CLE","layer":layer,"selected_alpha":alpha,"alpha_trace":trace,"raw_weight":weight,"raw_intercept":intercept,"config_fingerprint":fingerprint,"construction_case_ids":[r["case_id"] for r in construction]})
        record={"target":"final_soft_sa","position":"CLE","layer":layer,"selected_alpha":alpha,**metrics,**{f"{k}_ci_low":v[0] for k,v in ci.items()},**{f"{k}_ci_high":v[1] for k,v in ci.items()},"readout_reliable":reliable,"raw_expression_max_abs_error":error,"probe_file":str(path.relative_to(root)),"probe_sha256":sha256_file(path)}
        index.append(record); audit_rows.extend(rows)
        oof_rows.extend({"case_id":r["case_id"],"item_id":r["item_id"],"outer_fold":r["outer_fold"],"layer":layer,"actual":float(y),"predicted":float(p)} for r,y,p in zip(construction,y_train,oof))
        atomic_json(root/"progress/train_probes.json",{"status":"running","probe_count":len(index)})
    atomic_jsonl(index_path,index); atomic_jsonl(root/"artifacts/probes/audit_predictions.jsonl",audit_rows); atomic_jsonl(root/"artifacts/probes/construction_oof_predictions.jsonl",oof_rows)
    result={"status":"complete","probe_count":4,"resumed_noop":False,"all_reliable":all(r["readout_reliable"] for r in index)}; atomic_json(root/"progress/train_probes.json",result)
    return result


def load_probes(root: Path) -> dict[int, dict[str, Any]]:
    output={}; fingerprint=__import__("json").loads((root/"fingerprint.json").read_text())["fingerprint"]
    for row in load_jsonl(root/"artifacts/probes/probe_index.jsonl"):
        path=root/row["probe_file"]
        if sha256_file(path)!=row["probe_sha256"]: raise ValueError(f"Probe hash mismatch: {path}")
        payload=joblib.load(path)
        if payload.get("config_fingerprint")!=fingerprint: raise ValueError(f"Probe fingerprint mismatch: {path}")
        output[int(row["layer"])]=payload
    return output


def probe_value(hidden: torch.Tensor, payload: dict[str, Any]) -> tuple[float, float]:
    array=hidden.float().cpu().numpy().reshape(1,-1); pipeline_value=float(payload["pipeline"].predict(array)[0]); raw=float(array.reshape(-1)@np.asarray(payload["raw_weight"])+payload["raw_intercept"])
    error=abs(pipeline_value-raw)
    if error>RAW_EXPRESSION_ATOL: raise RuntimeError(f"Probe expression drift: {error}")
    return pipeline_value,error
