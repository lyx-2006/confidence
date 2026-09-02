from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import BOOTSTRAP_REPEATS, PROBE_LAYERS, PROBE_POSITIONS, RESULTS_ROOT, RIDGE_ALPHA, SEED, TARGETS
from .io_utils import atomic_csv, atomic_json, atomic_jsonl, canonical_hash, ensure_layout, load_jsonl, sha256_file
from .metrics import family_bootstrap, regression_values


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); fd,temp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent); os.close(fd)
    try: joblib.dump(value,temp); os.replace(temp,path)
    except Exception:
        try: os.unlink(temp)
        except FileNotFoundError: pass
        raise


class HiddenResolver:
    def __init__(self, root: Path):
        self.root=root; self.reuse={r["case_id"]:r for r in load_jsonl(root/"confidence_probe/artifacts/hidden/reuse_manifest.jsonl")}; self.delta={r["case_id"]:r for r in load_jsonl(root/"confidence_probe/artifacts/hidden/capture_results.jsonl")}
    def load(self, case_id: str, key: str) -> np.ndarray:
        delta=self.delta.get(case_id)
        if delta and key in delta["delta_keys"]:
            with np.load(self.root/delta["delta_file"]) as payload: value=np.asarray(payload[key],dtype=np.float32)
        else:
            source=self.reuse[case_id]["cell_sources"].get(key)
            if source is None: raise KeyError(f"Missing hidden cell: {case_id} {key}")
            with np.load(source["path"]) as payload: value=np.asarray(payload[key],dtype=np.float32)
        if value.ndim!=1 or not np.isfinite(value).all(): raise ValueError(f"Invalid hidden: {case_id} {key}")
        return value


def fit_probe(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model=Pipeline([("scaler",StandardScaler()),("ridge",Ridge(alpha=RIDGE_ALPHA,solver="lsqr"))]); model.fit(X_train,y_train); return model


def train_probe(root: Path, *, bootstrap: int = BOOTSTRAP_REPEATS) -> dict[str,Any]:
    joined=load_jsonl(root/"unimodal_confidence/artifacts/predictions/phase1_confidence_joined.jsonl")
    if not joined: raise FileNotFoundError("Run fit_temperature first")
    train=[r for r in joined if r["split"]=="train"]; test=[r for r in joined if r["split"]=="test"]
    if len(test)!=100 or len({r["family_id"] for r in test})!=50: raise ValueError("Probe test is not locked 100/50")
    resolver=HiddenResolver(root); metrics=[]; train_predictions=[]; test_predictions=[]
    for position in PROBE_POSITIONS:
        for layer in PROBE_LAYERS:
            key=f"{position}__L{layer}"; X_train=np.stack([resolver.load(r["case_id"],key) for r in train]); X_test=np.stack([resolver.load(r["case_id"],key) for r in test])
            for target in TARGETS:
                y_train=np.asarray([float(r[target]) for r in train]); y_test=np.asarray([float(r[target]) for r in test]); model=fit_probe(X_train,y_train); pred_train=model.predict(X_train); pred_test=model.predict(X_test)
                path=root/f"confidence_probe/artifacts/models/{target}__{position}__L{layer}.joblib"; payload={"model":model,"target":target,"position":position,"layer":layer,"target_mean":float(y_train.mean()),"target_std":float(y_train.std(ddof=0)),"train_case_fingerprint":canonical_hash([r["case_id"] for r in train])}; _atomic_joblib(path,payload); digest=sha256_file(path); values=regression_values(y_test,pred_test)
                bootstrap_rows=[{"family_id":r["family_id"],"true":float(y),"predicted":float(p)} for r,y,p in zip(test,y_test,pred_test,strict=True)]; ci=family_bootstrap(bootstrap_rows,repeats=bootstrap,seed=int(canonical_hash([SEED,target,position,layer])[:8],16))
                modality="text" if target.startswith("text_") else "image"; definition="chosen" if "chosen" in target else "fixed_answer"
                metrics.append({"target":target,"confidence_definition":definition,"modality":modality,"position":position,"layer":layer,"r2":values["r2"],"r2_ci_low":ci["r2"]["low"],"r2_ci_high":ci["r2"]["high"],"spearman":values["spearman"],"spearman_ci_low":ci["spearman"]["low"],"spearman_ci_high":ci["spearman"]["high"],"pearson":values["pearson"],"pearson_ci_low":ci["pearson"]["low"],"pearson_ci_high":ci["pearson"]["high"],"train_sample_count":len(train),"train_family_count":len({r["family_id"] for r in train}),"test_sample_count":len(test),"test_family_count":len({r["family_id"] for r in test}),"valid_bootstrap_repeats":min(ci[name]["valid"] for name in ci),"model_path":str(path.relative_to(root)),"model_sha256":digest})
                model_fp=canonical_hash({"sha256":digest,"target":target,"position":position,"layer":layer})
                for split_rows,truth,prediction,destination in ((train,y_train,pred_train,train_predictions),(test,y_test,pred_test,test_predictions)):
                    for row,y,p in zip(split_rows,truth,prediction,strict=True): destination.append({"case_id":row["case_id"],"item_id":row["item_id"],"family_id":row["family_id"],"prior_index":row["prior_index"],"condition":row["condition"],"target":target,"position":position,"layer":layer,"true_confidence":float(y),"predicted_confidence":float(p),"split":row["split"],"model_fingerprint":model_fp})
    atomic_csv(root/"confidence_probe/tables/probe_metrics.csv",metrics)
    wide=[]; mapping={(r["target"],r["position"],r["layer"]):r for r in metrics}
    for target in TARGETS:
        for metric,label in (("r2","R2"),("spearman","Spearman"),("pearson","Pearson")):
            row={"target":target,"metric":label}
            for position in PROBE_POSITIONS:
                for layer in PROBE_LAYERS: row[f"{position}__L{layer}"]=mapping[target,position,layer][metric]
            wide.append(row)
    atomic_csv(root/"confidence_probe/tables/probe_metrics_wide.csv",wide)
    atomic_jsonl(root/"confidence_probe/artifacts/predictions/train_predictions.jsonl",train_predictions); atomic_jsonl(root/"confidence_probe/artifacts/predictions/test_predictions.jsonl",test_predictions)
    summary={"status":"complete","cell_count":len(metrics),"train_sample_count":len(train),"test_sample_count":len(test),"prediction_count":len(train_predictions)+len(test_predictions)}; atomic_json(root/"confidence_probe/progress/train_probe.json",summary); return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",default=str(RESULTS_ROOT)); parser.add_argument("--bootstrap",type=int,default=BOOTSTRAP_REPEATS); parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv); root=ensure_layout(args.output_root,resume=True); print(json.dumps(train_probe(root,bootstrap=args.bootstrap),ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
