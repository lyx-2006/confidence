from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .config import (
    DECISION_OOF_PATH, DIFFICULTY_OOF_PATH, HISTORICAL_SA_OOF_PATH, JOINED_PATH,
    PANL_ARTIFACTS, PANL_CAPTURE_PATH, PANL_READOUT_BY_SWAP_LAYER, PROBE_MODEL_ROOT,
    PROBE_PARITY_TOLERANCE, SA_RECONSTRUCTION_LAYERS,
)
from .io_utils import atomic_json, atomic_jsonl, load_jsonl, sha256_file


def quantize_probe_hidden(hidden: Any) -> np.ndarray:
    vector = np.asarray(hidden, dtype=np.float32).reshape(-1)
    vector = vector.astype(np.float16).astype(np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Probe hidden must be one finite vector")
    return vector


def _capture_map() -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in load_jsonl(PANL_CAPTURE_PATH) if row.get("status") == "completed"}


def _features(rows: Sequence[dict[str, Any]], capture: dict[str, dict[str, Any]], layer: int) -> np.ndarray:
    vectors = []
    key = f"P1_PANL__L{int(layer)}"
    for row in rows:
        record = capture[str(row["case_id"])]
        with np.load(PANL_ARTIFACTS.parent / str(record["hidden_file"])) as payload:
            if key not in payload:
                raise ValueError(f"Missing historical hidden: {record['case_id']} {key}")
            vectors.append(quantize_probe_hidden(payload[key]))
    return np.stack(vectors)


def _sa_model_path(root: Path, layer: int, fold: int) -> Path:
    return root / "artifacts" / "probe_models" / f"sa__P1_PANL__L{layer}__fold{fold}.joblib"


def prepare_probe_models(root: Path, *, resume: bool) -> dict[str, Any]:
    joined = load_jsonl(JOINED_PATH); capture = _capture_map()
    if not joined or {str(row["case_id"]) for row in joined} - set(capture):
        raise ValueError("PANL joined/capture artifacts are incomplete")
    sa_predictions: list[dict[str, Any]] = []
    y = np.asarray([float(row["soft_sa_image_score"]) for row in joined], dtype=float)
    historical_sa = {(str(row["case_id"]), str(row["position"]), int(row["layer"])): float(row["prediction"]) for row in load_jsonl(HISTORICAL_SA_OOF_PATH)}
    anchor_differences: list[float] = []
    model_hashes: dict[str, str] = {}
    for layer in SA_RECONSTRUCTION_LAYERS:
        X = _features(joined, capture, layer); prediction = np.empty(len(joined), dtype=float)
        for fold in range(5):
            train = np.asarray([int(row["outer_fold"]) != fold for row in joined]); test = ~train
            if not train.any() or not test.any():
                raise ValueError(f"Empty outer fold {fold}")
            path = _sa_model_path(root, layer, fold)
            if path.exists() and resume:
                model = joblib.load(path)
            else:
                model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0, solver="lsqr"))])
                with threadpool_limits(limits=1): model.fit(X[train], y[train])
                path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, path)
            with threadpool_limits(limits=1): prediction[test] = model.predict(X[test])
            model_hashes[str(path.relative_to(root))] = sha256_file(path)
        for row, truth, predicted in zip(joined, y, prediction, strict=True):
            sa_predictions.append({"case_id": row["case_id"], "item_id": row["item_id"], "outer_fold": row["outer_fold"], "target": float(truth), "prediction": float(predicted), "position": "P1_PANL", "layer": layer})
            if layer == 14:
                key = (str(row["case_id"]), "P1_PANL", 14)
                if key not in historical_sa:
                    raise ValueError(f"Missing historical SA anchor OOF: {key[0]}")
                anchor_differences.append(abs(float(predicted) - historical_sa[key]))
    if not anchor_differences or max(anchor_differences) > PROBE_PARITY_TOLERANCE:
        raise ValueError(f"Reconstructed PANL-SA L14 parity failed: {max(anchor_differences, default=float('inf'))}")
    atomic_jsonl(root / "artifacts" / "probe_predictions" / "panl_sa_oof.jsonl", sa_predictions)

    difficulty_oof = {(str(row["case_id"]), str(row["target_name"]), int(row["layer"])): float(row["prediction"]) for row in load_jsonl(DIFFICULTY_OOF_PATH) if row["position"] == "P1_PANL"}
    decision_oof = {(str(row["case_id"]), int(row["layer"])): float(row["probability_follow_image"]) for row in load_jsonl(DECISION_OOF_PATH) if row["position"] == "P1_PANL"}
    parity = {"sa_l14_max_abs_difference": max(anchor_differences), "difficulty": {}, "decision": {}}
    for layer in sorted(set(PANL_READOUT_BY_SWAP_LAYER.values())):
        X = _features(joined, capture, layer)
        for target in ("text", "image"):
            diffs = []
            for fold in range(5):
                path = PROBE_MODEL_ROOT / f"{target}__P1_PANL__L{layer}__fold{fold}.joblib"
                payload = joblib.load(path); model = payload["model"]
                test_indices = [index for index, row in enumerate(joined) if int(row["outer_fold"]) == fold]
                with threadpool_limits(limits=16): values = model.predict(X[test_indices]) * float(payload["target_scale"]) + float(payload["target_mean"])
                for index, value in zip(test_indices, values, strict=True):
                    key = (str(joined[index]["case_id"]), target, layer)
                    diffs.append(abs(float(value) - difficulty_oof[key]))
                model_hashes[str(path.resolve())] = sha256_file(path)
            maximum = max(diffs, default=float("inf")); parity["difficulty"][f"{target}_L{layer}"] = maximum
            if maximum > PROBE_PARITY_TOLERANCE:
                raise ValueError(f"Difficulty probe parity failed: {target} L{layer} {maximum}")
        diffs = []
        for fold in range(5):
            path = PROBE_MODEL_ROOT / f"decision__P1_PANL__L{layer}__fold{fold}.joblib"; model = joblib.load(path)
            indices = [index for index, row in enumerate(joined) if int(row["outer_fold"]) == fold and (str(row["case_id"]), layer) in decision_oof]
            if indices:
                classes = list(model.named_steps["classifier"].classes_)
                with threadpool_limits(limits=16): values = model.predict_proba(X[indices])[:, classes.index(1)]
                diffs.extend(abs(float(value) - decision_oof[(str(joined[index]["case_id"]), layer)]) for index, value in zip(indices, values, strict=True))
            model_hashes[str(path.resolve())] = sha256_file(path)
        maximum = max(diffs, default=float("inf")); parity["decision"][f"L{layer}"] = maximum
        if maximum > PROBE_PARITY_TOLERANCE:
            raise ValueError(f"Decision probe parity failed: L{layer} {maximum}")
    audit = {"status": "passed", "record_count": len(joined), "model_hashes": model_hashes, "parity": parity, "sa_reconstruction": "historical_clean_item_oof_standard_scaler_ridge_alpha1_lsqr"}
    atomic_json(root / "artifacts" / "probe_parity_audit.json", audit)
    return audit


class ProbeBank:
    def __init__(self, root: Path) -> None:
        self.root = root; self._models: dict[tuple[str, int, int], Any] = {}
        self.rows = load_jsonl(JOINED_PATH); self.capture = _capture_map(); self._reference_features: dict[int, np.ndarray] = {}
        self.difficulty_oof = {(str(row["case_id"]), str(row["target_name"]), int(row["layer"])): float(row["prediction"]) for row in load_jsonl(DIFFICULTY_OOF_PATH) if row["position"] == "P1_PANL"}
        self.decision_oof = {(str(row["case_id"]), int(row["layer"])): float(row["probability_follow_image"]) for row in load_jsonl(DECISION_OOF_PATH) if row["position"] == "P1_PANL"}
        self.sa_oof = {(str(row["case_id"]), int(row["layer"])): float(row["prediction"]) for row in load_jsonl(root / "artifacts" / "probe_predictions" / "panl_sa_oof.jsonl")}

    def _load(self, target: str, layer: int, fold: int) -> Any:
        key = (target, int(layer), int(fold))
        if key not in self._models:
            path = _sa_model_path(self.root, layer, fold) if target == "sa" else PROBE_MODEL_ROOT / f"{target}__P1_PANL__L{layer}__fold{fold}.joblib"
            self._models[key] = joblib.load(path)
        return self._models[key]

    def _reference(self, layer: int) -> np.ndarray:
        if layer not in self._reference_features: self._reference_features[layer] = _features(self.rows, self.capture, layer)
        return self._reference_features[layer]

    def _predict_at_case(self, target: str, hidden: np.ndarray, *, case_id: str, layer: int, fold: int) -> float:
        reference = self._reference(layer)
        if target == "decision":
            indices = [index for index, row in enumerate(self.rows) if int(row["outer_fold"]) == fold and (str(row["case_id"]), layer) in self.decision_oof]
        else:
            indices = [index for index, row in enumerate(self.rows) if int(row["outer_fold"]) == fold]
        local = next((offset for offset, index in enumerate(indices) if str(self.rows[index]["case_id"]) == case_id), None)
        if local is None:
            X = hidden[None, :]; local = 0
        else:
            X = reference[indices].copy(); X[local] = hidden
        loaded = self._load(target, layer, fold)
        if target == "sa":
            with threadpool_limits(limits=1): return float(loaded.predict(X)[local])
        if target in ("text", "image"):
            with threadpool_limits(limits=16): value = loaded["model"].predict(X)[local]
            return float(value * loaded["target_scale"] + loaded["target_mean"])
        classes = list(loaded.named_steps["classifier"].classes_)
        with threadpool_limits(limits=16): return float(loaded.predict_proba(X)[local, classes.index(1)])

    def predict(self, hidden: Any, *, case_id: str, layer: int, fold: int) -> dict[str, float]:
        vector = quantize_probe_hidden(hidden)
        sa = self._predict_at_case("sa", vector, case_id=case_id, layer=layer, fold=fold); output = {"panl_sa_prediction": sa}
        for target in ("text", "image"):
            output[f"panl_{target}_difficulty_prediction"] = self._predict_at_case(target, vector, case_id=case_id, layer=layer, fold=fold)
        output["panl_decision_follow_image_probability"] = self._predict_at_case("decision", vector, case_id=case_id, layer=layer, fold=fold)
        if not np.isfinite(list(output.values())).all():
            raise ValueError("Non-finite probe prediction")
        return output

    def historical(self, case_id: str, layer: int) -> dict[str, Any]:
        result: dict[str, Any] = {"panl_sa_prediction": self.sa_oof[(case_id, layer)]}
        for target in ("text", "image"):
            result[f"panl_{target}_difficulty_prediction"] = self.difficulty_oof[(case_id, target, layer)]
        decision = self.decision_oof.get((case_id, layer))
        result["panl_decision_follow_image_probability"] = decision
        result["decision_historical_status"] = "available" if decision is not None else "historical_not_applicable"
        return result
