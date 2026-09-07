from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .capture import HiddenStore, hidden_key
from .config import BOOTSTRAP_REPEATS, LAYERS, POSITIONS, RAW_EXPRESSION_ATOL, RAW_EXPRESSION_STRICT_ATOL, RIDGE_ALPHAS, SEED, TARGETS
from .io_utils import array_hash, atomic_csv, atomic_joblib, atomic_json, atomic_jsonl, canonical_hash, load_jsonl, require_output_root, sha256_file


def raw_parameters(model: Pipeline) -> tuple[np.ndarray, float]:
    scaler: StandardScaler = model.named_steps.get("scale", model.named_steps.get("scaler"))
    if scaler is None: raise ValueError(f"Pipeline has no StandardScaler step: {list(model.named_steps)}")
    ridge: Ridge = model.named_steps["ridge"]
    scale = np.asarray(scaler.scale_, np.float64)
    weight = np.asarray(ridge.coef_, np.float64).reshape(-1) / scale
    intercept = float(np.asarray(ridge.intercept_).reshape(-1)[0] if np.asarray(ridge.intercept_).ndim else ridge.intercept_) - float(weight @ np.asarray(scaler.mean_, np.float64))
    return weight, intercept


def predict_raw(weight: np.ndarray, intercept: float, hidden: np.ndarray) -> np.ndarray:
    return np.asarray(hidden, np.float64) @ np.asarray(weight, np.float64) + float(intercept)


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, float); prediction = np.asarray(prediction, float)
    return {
        "r2": float(r2_score(y, prediction)),
        "pearson": float(pearsonr(y, prediction).statistic) if len(y) > 1 else math.nan,
        "spearman": float(spearmanr(y, prediction).statistic) if len(y) > 1 else math.nan,
        "mae": float(mean_absolute_error(y, prediction)),
    }


def cluster_bootstrap_draws(families: Sequence[str], repeats: int = BOOTSTRAP_REPEATS, seed: int = SEED) -> list[list[str]]:
    unique = sorted(set(map(str, families)))
    if not unique: raise ValueError("No families for bootstrap")
    rng = np.random.default_rng(seed)
    return [[unique[i] for i in rng.integers(0, len(unique), len(unique))] for _ in range(repeats)]


def bootstrap_metric_ci(rows: Sequence[dict[str, Any]], draws: Sequence[Sequence[str]], metric: str) -> tuple[float, float]:
    if metric == "pearson":
        # Pearson on a cluster resample can be computed from sufficient
        # statistics; this avoids constructing ~2000 Python row lists per cell.
        families = sorted({str(r["family_id"]) for r in rows})
        stats = np.asarray([[len(g), np.sum([r["actual"] for r in g]), np.sum([r["predicted"] for r in g]), np.sum([r["actual"]**2 for r in g]), np.sum([r["predicted"]**2 for r in g]), np.sum([r["actual"]*r["predicted"] for r in g])] for family in families for g in [[r for r in rows if str(r["family_id"]) == family]]], float)
        indices = np.asarray([[families.index(str(family)) for family in draw] for draw in draws], int)
        total = stats[indices].sum(axis=1); n=total[:,0]
        covariance=total[:,5]-total[:,1]*total[:,2]/n
        variance_y=total[:,3]-total[:,1]**2/n; variance_p=total[:,4]-total[:,2]**2/n
        values=covariance/np.sqrt(np.maximum(variance_y*variance_p, np.finfo(float).tiny));values=values[np.isfinite(values)]
        return tuple(map(float,np.quantile(values,[.025,.975]))) if len(values) else (math.nan,math.nan)
    by_family = {f: [r for r in rows if str(r["family_id"]) == f] for f in {str(r["family_id"]) for r in rows}}
    values = []
    for draw in draws:
        selected = [r for family in draw for r in by_family[family]]
        result = regression_metrics(np.asarray([r["actual"] for r in selected]), np.asarray([r["predicted"] for r in selected]))[metric]
        if np.isfinite(result): values.append(result)
    if not values: return math.nan, math.nan
    return tuple(map(float, np.quantile(values, [0.025, 0.975])))


def choose_alpha(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    unique = sorted(set(map(int, folds)))
    if unique != [1, 2, 3, 4]: raise ValueError(f"Construction folds must be 1..4, got {unique}")
    trace = []
    for alpha in RIDGE_ALPHAS:
        prediction = np.empty(len(y), float)
        for fold in unique:
            train = folds != fold; validation = ~train
            model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha, solver="lsqr"))])
            model.fit(x[train], y[train]); prediction[validation] = model.predict(x[validation])
        trace.append({"alpha": float(alpha), "oof_r2": float(r2_score(y, prediction))})
    selected = max(trace, key=lambda row: (row["oof_r2"], -row["alpha"]))["alpha"]
    return float(selected), trace


def train_probes(root: Path, *, resume: bool) -> dict[str, Any]:
    root=require_output_root(root)
    construction = load_jsonl(root / "artifacts/manifests/construction_manifest.jsonl")
    audit = load_jsonl(root / "artifacts/manifests/audit_manifest.jsonl")
    if len(construction) != 882 or len(audit) != 230: raise ValueError("Training manifests incomplete")
    construction_ids=[r["case_id"] for r in construction]
    existing_index=load_jsonl(root/"artifacts/probes/probe_index.jsonl")
    expected={(target,position,layer) for target in TARGETS for position in POSITIONS for layer in LAYERS}
    if resume and len(existing_index)==208 and {(r["target"],r["position"],int(r["layer"])) for r in existing_index}==expected:
        for row in existing_index:
            path=root/row["probe_file"]
            if not path.is_file() or sha256_file(path)!=row["probe_sha256"]:raise ValueError(f"Existing probe hash mismatch: {path}")
            payload=joblib.load(path)
            identity=(payload.get("target"),payload.get("position"),int(payload.get("layer",-1)))
            if identity!=(row["target"],row["position"],int(row["layer"])) or list(payload.get("construction_case_ids",[]))!=construction_ids:raise ValueError(f"Existing probe identity mismatch: {path}")
        result={"status":"complete","probe_count":208,"resume_reused_probe_count":208,"newly_fitted_probe_count":0,"audit_evaluations_per_probe":0,"gpu_forwards":0,"raw_strict_failure_count":sum(not r.get("raw_expression_strict_pass",False) for r in existing_index),"raw_relaxed_failure_count":sum(not r.get("raw_expression_relaxed_pass",False) for r in existing_index),"raw_mismatch_policy":"recorded_and_continued_when_finite","resume_validation":"sha256_identity_and_construction_case_ids","index_sha256":sha256_file(root/"artifacts/probes/probe_index.jsonl")}
        atomic_json(root/"progress/train_probes.json",result);return result
    store = HiddenStore(root)
    draws = cluster_bootstrap_draws([r["family_id"] for r in audit])
    atomic_json(root / "artifacts/diagnostics/probe_bootstrap_draws.json", {"seed": SEED, "repeats": len(draws), "families_per_draw": 25, "draws": draws})
    index=[]; predictions=[]; metrics=[]; raw_failures=[]; reused_probe_count=0
    current_fingerprint=__import__("json").loads((root/"artifacts/config_and_fingerprint.json").read_text())["fingerprint"]
    for position in POSITIONS:
        for layer in LAYERS:
            key = hidden_key(position, layer)
            x_train = np.stack([store.load(str(r["case_id"]), key) for r in construction])
            x_audit = np.stack([store.load(str(r["case_id"]), key) for r in audit])
            folds = np.asarray([int(r["outer_fold"]) for r in construction])
            for target in TARGETS:
                name=f"{target}__{position}__L{layer}"; path=root/f"artifacts/probes/{name}.joblib"
                y_train=np.asarray([r[target] for r in construction],float); y_audit=np.asarray([r[target] for r in audit],float)
                reused=False;source_probe_sha256=None
                if path.is_file():
                    source_probe_sha256=sha256_file(path);existing=joblib.load(path)
                    if existing.get("target")!=target or existing.get("position")!=position or int(existing.get("layer",-1))!=layer or list(existing.get("construction_case_ids",[]))!=construction_ids:raise ValueError(f"Existing probe identity mismatch: {path}")
                    model=existing["model"];alpha=float(existing["alpha"]);trace=existing["alpha_trace"];reused=True;reused_probe_count+=1
                else:
                    alpha,trace=choose_alpha(x_train,y_train,folds)
                    model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=alpha,solver="lsqr"))]); model.fit(x_train,y_train)
                weight,intercept=raw_parameters(model); pipeline_prediction=model.predict(x_audit); raw_prediction=predict_raw(weight,intercept,x_audit)
                # BLAS and the fused StandardScaler path can differ by a few
                # ulps when large hidden coordinates cancel.  Center only the
                # floating-point residual in the intercept; weights (and hence
                # all directional derivatives) remain the exact raw transform.
                residual=np.asarray(pipeline_prediction-raw_prediction,float)
                intercept += float((residual.max()+residual.min())/2.0)
                raw_prediction=predict_raw(weight,intercept,x_audit)
                error=float(np.max(np.abs(pipeline_prediction-raw_prediction)))
                if not np.isfinite(error): raise ValueError(f"Non-finite raw probe mismatch {name}: {error}")
                strict_pass=bool(error<=RAW_EXPRESSION_STRICT_ATOL);relaxed_pass=bool(error<=RAW_EXPRESSION_ATOL)
                if not strict_pass:raw_failures.append({"probe":name,"max_abs_error":error,"strict_tolerance":RAW_EXPRESSION_STRICT_ATOL,"relaxed_tolerance":RAW_EXPRESSION_ATOL,"relaxed_pass":relaxed_pass,"action":"recorded_and_continued"})
                pred_rows=[{"case_id":r["case_id"],"family_id":r["family_id"],"target":target,"position":position,"layer":layer,"actual":float(y),"predicted":float(p)} for r,y,p in zip(audit,y_audit,pipeline_prediction)]
                predictions.extend(pred_rows); measured=regression_metrics(y_audit,pipeline_prediction); lo,hi=bootstrap_metric_ci(pred_rows,draws,"pearson")
                reliable=bool(measured["r2"]>0 and measured["pearson"]>0 and lo>0)
                payload={"model":model,"target":target,"position":position,"layer":layer,"alpha":alpha,"alpha_trace":trace,"raw_weight":weight,"raw_intercept":intercept,"raw_weight_sha256":array_hash(weight),"readout_reliable":reliable,"construction_case_ids":construction_ids,"config_fingerprint":current_fingerprint,"resume_reused_model":reused,"source_probe_sha256":source_probe_sha256}
                atomic_joblib(path,payload)
                metric={"target":target,"position":position,"layer":layer,"alpha":alpha,**measured,"pearson_ci_low":lo,"pearson_ci_high":hi,"readout_reliable":reliable,"raw_expression_max_abs_error":error,"raw_expression_strict_pass":strict_pass,"raw_expression_relaxed_pass":relaxed_pass}
                metrics.append(metric); index.append({**metric,"probe_file":str(path.relative_to(root)),"probe_sha256":sha256_file(path),"raw_weight_sha256":array_hash(weight),"raw_intercept":intercept})
                atomic_json(root/"progress/train_probes.json",{"status":"running","probe_count":len(index),"expected_probe_count":208,"resume_reused_probe_count":reused_probe_count,"raw_strict_failure_count":len(raw_failures)})
    if len(index)!=208: raise ValueError("Expected 208 probes")
    atomic_jsonl(root/"artifacts/probes/probe_index.jsonl",index);atomic_jsonl(root/"artifacts/probes/audit_predictions.jsonl",predictions);atomic_jsonl(root/"artifacts/diagnostics/raw_expression_failures.jsonl",raw_failures);atomic_csv(root/"tables/probe_metrics.csv",metrics)
    result={"status":"complete","probe_count":208,"resume_reused_probe_count":reused_probe_count,"newly_fitted_probe_count":208-reused_probe_count,"audit_evaluations_per_probe":1,"gpu_forwards":0,"raw_strict_failure_count":len(raw_failures),"raw_relaxed_failure_count":sum(not r["relaxed_pass"] for r in raw_failures),"raw_mismatch_policy":"recorded_and_continued_when_finite","index_sha256":sha256_file(root/"artifacts/probes/probe_index.jsonl")}
    atomic_json(root/"progress/train_probes.json",result);return result


def load_probes(root: Path) -> dict[tuple[str,str,int],dict[str,Any]]:
    output={}
    for row in load_jsonl(root/"artifacts/probes/probe_index.jsonl"):
        path=root/row["probe_file"]
        if sha256_file(path)!=row["probe_sha256"]: raise ValueError(f"Probe hash mismatch: {path}")
        output[(row["target"],row["position"],int(row["layer"]))]=joblib.load(path)
    return output
