from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dp_SA.unimodal_logit_confidence.config import FIXED_EPSILON

from .config import (
    AUDIT_OUTER_FOLD, CANONICAL_COLORS, CONFIDENCE_JOINED, EXPECTED_PRELOCK_INPUTS,
    EXPECTED_SPLIT_AUDIT_SHA256,
    HIDDEN_CAPTURE, HIDDEN_DEFINITION, HIDDEN_REUSE, HIDDEN_SIZE, PANL_LAYER,
    RIDGE_ALPHA_GRID, SEED, SOURCE_ROOT, SOURCE_SPLIT_AUDIT, STEERING_LAYERS, UNIMODAL_SCORES,
    VECTOR_NORM_FRACTION,
)
from .io_utils import array_hash, canonical_hash, load_jsonl, sha256_file


def answer_origin(row: dict[str, Any]) -> str:
    text, image = bool(row["answer_matches_text"]), bool(row["answer_matches_image"])
    if text and not image: return "follow_text"
    if image and not text: return "follow_image"
    if not text and not image: return "neither_match"
    return "both_match"


def chosen_match(text_answer: str, image_answer: str, fixed: str) -> str:
    tm, im = text_answer == fixed, image_answer == fixed
    if tm and im: return "both"
    if tm: return "text_only"
    if im: return "image_only"
    return "neither"


def prelock_inventory() -> dict[str, Any]:
    output = {}
    for name, (path, rows, digest) in EXPECTED_PRELOCK_INPUTS.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Frozen pre-lock input mismatch: {name}")
        # Mixed source files are fingerprinted as immutable sources; only train IDs
        # are materialized below. The sealed test manifest is never opened here.
        count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        if count != rows: raise ValueError(f"Frozen row count mismatch: {name}={count}/{rows}")
        output[name] = {"path": str(path.resolve()), "record_count": count, "sha256": digest}
    if sha256_file(SOURCE_SPLIT_AUDIT) != EXPECTED_SPLIT_AUDIT_SHA256:
        raise ValueError("Frozen upstream split audit mismatch")
    split = json.loads(SOURCE_SPLIT_AUDIT.read_text())
    required_zero = ("sample", "item", "family", "image_hash")
    if split.get("status") != "passed" or any(int(split["overlaps"].get(k, -1)) != 0 for k in required_zero):
        raise ValueError("Upstream train/test isolation audit failed")
    output["upstream_split_audit"] = {"path": str(SOURCE_SPLIT_AUDIT.resolve()), "sha256": EXPECTED_SPLIT_AUDIT_SHA256, "overlaps": {k: 0 for k in required_zero}}
    return output


def split_train(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != 1112 or len({str(r["family_id"]) for r in rows}) != 128:
        raise ValueError("Frozen train cardinality changed")
    construction = [r for r in rows if int(r["outer_fold"]) != AUDIT_OUTER_FOLD]
    audit = [r for r in rows if int(r["outer_fold"]) == AUDIT_OUTER_FOLD]
    cf, af = {str(r["family_id"]) for r in construction}, {str(r["family_id"]) for r in audit}
    if len(construction) != 882 or len(audit) != 230 or len(cf) != 103 or len(af) != 25 or cf & af:
        raise ValueError("Frozen construction/audit split changed")
    if {str(r["phase0_normalized_answer"]) for r in audit} != set(CANONICAL_COLORS):
        raise ValueError("Direction-audit split no longer covers all colors")
    return construction, audit, {
        "status": "passed", "policy": "outer_fold != 0 construction; outer_fold == 0 audit",
        "construction_records": 882, "construction_families": 103,
        "audit_records": 230, "audit_families": 25,
        "family_overlap": 0, "test_manifest_opened": False,
    }


def _selected_rows(path: Path, allowed: set[str]) -> dict[str, dict[str, Any]]:
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            row = json.loads(line); case = str(row.get("case_id", ""))
            if case not in allowed: continue
            if case in output: raise ValueError(f"Duplicate selected case: {case}")
            output[case] = row
    if set(output) != allowed: raise ValueError(f"Missing selected rows from {path}")
    return output


def prepare_train_rows(manifest: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {str(r["case_id"]) for r in manifest}
    joined = _selected_rows(CONFIDENCE_JOINED, allowed)
    # Scores have no case_id, so index only the exact train unique keys requested.
    requested = set()
    for case in allowed:
        requested.add(("text", tuple(joined[case]["text_score_unique_key"])))
        requested.add(("image", tuple(joined[case]["image_score_unique_key"])))
    scores = {}
    with UNIMODAL_SCORES.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); key = (str(row["modality"]), tuple(row["unique_key"]))
            if key in requested: scores[key] = row
    if set(scores) != requested: raise ValueError("Missing train difficulty score")
    output = []
    for m in manifest:
        case = str(m["case_id"]); c = joined[case]
        fixed = str(m["phase0_normalized_answer"])
        if fixed != str(c["fixed_answer"]) or str(m["image_sha256"]) != str(c["image_hash"]):
            raise ValueError(f"Frozen identity mismatch: {case}")
        ct, ci = float(c["text_fixed_answer_confidence"]), float(c["image_fixed_answer_confidence"])
        lt = math.log((ct + FIXED_EPSILON) / (1.0 - ct + FIXED_EPSILON))
        li = math.log((ci + FIXED_EPSILON) / (1.0 - ci + FIXED_EPSILON))
        gl = li - lt
        if not all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in ((lt, float(c["text_fixed_answer_log_odds"])), (li, float(c["image_fixed_answer_log_odds"])), (gl, float(c["G_L"])))):
            raise ValueError(f"G_L recomputation mismatch: {case}")
        tk = ("text", tuple(c["text_score_unique_key"])); ik = ("image", tuple(c["image_score_unique_key"]))
        row = {
            "case_id": case, "item_id": str(m["item_id"]), "family_id": str(m["family_id"]),
            "outer_fold": int(m["outer_fold"]), "condition": str(m["condition"]), "prior_bin": str(m["prior_bin"]),
            "fixed_answer_color": fixed, "answer_origin": answer_origin(m), "Hard": int(m["condition"] == "conflict_hard"),
            "C_t": ct, "C_i": ci, "L_t": lt, "L_i": li, "G_L": gl,
            "D_t": float(scores[tk]["entropy_difficulty"]), "D_i": float(scores[ik]["entropy_difficulty"]),
            "unimodal_chosen_match": chosen_match(str(c["text_chosen_answer"]), str(c["image_chosen_answer"]), fixed),
            "clean_final_sa": float(m["soft_sa_image_score"]),
        }
        if not np.isfinite([row[k] for k in ("C_t", "C_i", "L_t", "L_i", "G_L", "D_t", "D_i", "clean_final_sa")]).all():
            raise ValueError(f"Non-finite train row: {case}")
        output.append(row)
    return output


def inner_fold(row_or_family: dict[str, Any] | str) -> int:
    if isinstance(row_or_family, dict):
        value = int(row_or_family["outer_fold"])
        if value not in (1, 2, 3, 4): raise ValueError("Inner OOF requires construction outer folds 1..4")
        return value
    return int(canonical_hash([SEED, "audit_bootstrap_fold", str(row_or_family)])[:16], 16) % 5


class HiddenResolver:
    def __init__(self):
        self.reuse = {str(r["case_id"]): r for r in load_jsonl(HIDDEN_REUSE)}
        self.delta = {str(r["case_id"]): r for r in load_jsonl(HIDDEN_CAPTURE)}
        self.file_hashes: dict[str, str] = {}

    def load(self, case_id: str, key: str) -> np.ndarray:
        delta = self.delta.get(case_id)
        if delta and key in delta["delta_keys"]:
            path = SOURCE_ROOT / delta["delta_file"]; expected_file = delta["delta_file_sha256"]; expected_tensor = None
        else:
            source = self.reuse[case_id]["cell_sources"].get(key)
            if source is None: raise KeyError(f"Missing hidden: {case_id} {key}")
            path = Path(source["path"]); expected_file = source["file_sha256"]; expected_tensor = source["tensor_sha256"]
        resolved = str(path.resolve())
        if resolved not in self.file_hashes: self.file_hashes[resolved] = sha256_file(path)
        if self.file_hashes[resolved] != expected_file: raise ValueError(f"Hidden file hash mismatch: {path}")
        with np.load(path) as payload: raw = np.asarray(payload[key])
        if raw.dtype != np.float16 or raw.shape != (HIDDEN_SIZE,) or not np.isfinite(raw).all(): raise ValueError(f"Invalid hidden: {case_id} {key}")
        if expected_tensor is not None and array_hash(raw) != expected_tensor: raise ValueError(f"Hidden tensor hash mismatch: {case_id} {key}")
        return raw.astype(np.float32)


def make_cells(rows: Sequence[dict[str, Any]], resolver: HiddenResolver) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[row["family_id"], row["fixed_answer_color"]].append(row)
    arrays = {layer: {} for layer in STEERING_LAYERS}; metadata = []
    for index, ((family, color), members) in enumerate(sorted(grouped.items())):
        key = f"cell_{index:04d}"; hashes = {}
        for layer in STEERING_LAYERS:
            value = np.stack([resolver.load(r["case_id"], f"P1_LAT__L{layer}") for r in members]).mean(axis=0, dtype=np.float32)
            arrays[layer][key] = value; hashes[f"L{layer}"] = array_hash(value)
        metadata.append({
            "array_key": key, "family_id": family, "fixed_answer_color": color,
            "case_ids": sorted(r["case_id"] for r in members), "record_count": len(members),
            "outer_fold": int(members[0]["outer_fold"]),
            **{f"mean_{field}": float(np.mean([r[field] for r in members])) for field in ("G_L", "D_t", "D_i", "clean_final_sa")},
            "hidden_hashes": hashes,
        })
    return metadata, arrays


def continuous_pattern(hidden: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = np.asarray(hidden, dtype=np.float64); y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y): raise ValueError("Pattern input shape mismatch")
    yc = y - y.mean(); denom = float(yc @ yc)
    if not math.isfinite(denom) or denom <= 0: raise ValueError("Pattern target has zero variance")
    value = (yc[:, None] * (x - x.mean(axis=0))).sum(axis=0) / denom
    norm = float(np.linalg.norm(value))
    if value.shape != (HIDDEN_SIZE,) or not np.isfinite(value).all() or norm <= 0: raise ValueError("Invalid continuous pattern")
    return value


def _split_half_stability(selected: Sequence[dict[str, Any]], hidden: dict[str, np.ndarray], target_field: str) -> float | None:
    ordered = sorted(selected, key=lambda c: (canonical_hash([SEED, "split_half", c["family_id"]]), c["family_id"]))
    halves = (ordered[::2], ordered[1::2])
    if min(map(len, halves)) < 2: return None
    try:
        vectors = [continuous_pattern(np.stack([hidden[c["array_key"]] for c in half]), np.asarray([c[target_field] for c in half])) for half in halves]
    except ValueError:
        return None
    return float(vectors[0] @ vectors[1] / np.linalg.norm(vectors[0]) / np.linalg.norm(vectors[1]))


def answer_patterns(cells: Sequence[dict[str, Any]], hidden: dict[str, np.ndarray], target_field: str, *, return_audit: bool = False):
    by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells: by_answer[cell["fixed_answer_color"]].append(cell)
    output = {}; audits = []
    for answer in CANONICAL_COLORS:
        selected = by_answer.get(answer, [])
        target = np.asarray([c[target_field] for c in selected], dtype=np.float64)
        variance = float(np.var(target)) if len(target) else 0.0
        valid = len(selected) >= 10 and variance > 0 and np.isfinite(target).all()
        value = None; raw_norm = None
        if valid:
            try:
                raw = continuous_pattern(np.stack([hidden[c["array_key"]] for c in selected]), target)
                raw_norm = float(np.linalg.norm(raw)); value = raw / raw_norm
                valid = bool(np.isfinite(value).all())
            except ValueError:
                valid = False
        if valid and value is not None: output[answer] = value
        audits.append({"fixed_answer_color": answer, "target": target_field, "cell_count": len(selected), "family_count": len({c["family_id"] for c in selected}), "target_variance": variance, "direction_norm": raw_norm, "split_half_stability": _split_half_stability(selected, hidden, target_field) if valid else None, "valid": valid})
    return (output, audits) if return_audit else output


def loao(patterns: dict[str, np.ndarray], recipient: str) -> tuple[np.ndarray, list[str]]:
    included = [a for a in CANONICAL_COLORS if a != recipient and a in patterns]
    if len(included) < 3: raise ValueError(f"Insufficient valid LOAO donors for {recipient}")
    normalized = [np.asarray(patterns[a], np.float64) / np.linalg.norm(patterns[a]) for a in included]
    value = np.stack(normalized).mean(axis=0, dtype=np.float64)
    if not np.isfinite(value).all() or np.linalg.norm(value) <= 0: raise ValueError("Invalid LOAO direction")
    return value / np.linalg.norm(value), included


def svd_basis(columns: Sequence[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    normalized = []
    for column in columns:
        value = np.asarray(column, dtype=np.float64); norm = float(np.linalg.norm(value))
        if not math.isfinite(norm) or norm <= 0: raise ValueError("Invalid nuisance basis column")
        normalized.append(value / norm)
    matrix = np.stack(normalized, axis=1); u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * float(singular[0]); rank = int(np.sum(singular > tolerance))
    if rank < 1: raise ValueError("Zero-rank nuisance subspace")
    return u[:, :rank], {"rank": rank, "tolerance": tolerance, "singular_values": singular.tolist()}


def project_out(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    original = np.asarray(vector, dtype=np.float64)
    result = original - basis @ (basis.T @ original)
    # A second pass keeps float64 numerical error comfortably below the gate.
    result = result - basis @ (basis.T @ result)
    return result


def natural_sa_decomposition(vector: np.ndarray, basis: np.ndarray, norm: float) -> dict[str, Any]:
    confidence = np.asarray(vector, dtype=np.float64)
    parallel = basis @ (basis.T @ confidence)
    perpendicular = project_out(confidence, basis)
    common_scale = float(norm) / float(np.linalg.norm(confidence))
    raw = scale_vector(confidence, norm)
    parallel_scaled = (common_scale * parallel).astype(np.float32)
    perpendicular_scaled = (common_scale * perpendicular).astype(np.float32)
    reconstruction = raw.astype(np.float64) - parallel_scaled.astype(np.float64) - perpendicular_scaled.astype(np.float64)
    relative_error = float(np.linalg.norm(reconstruction) / np.linalg.norm(raw))
    raw_matches = bool(np.allclose(raw, (common_scale * confidence).astype(np.float32), rtol=1e-6, atol=1e-7))
    return {"raw": raw, "parallel": parallel, "perpendicular": perpendicular,
            "parallel_scaled": parallel_scaled, "perpendicular_scaled": perpendicular_scaled,
            "common_scale": common_scale, "reconstruction_relative_error": relative_error,
            "raw_matches_existing": raw_matches}


def select_ridge_alpha(X: np.ndarray, y: np.ndarray, folds: Sequence[int]) -> tuple[float, list[dict[str, float]]]:
    folds_array = np.asarray(folds); trace = []
    for alpha in RIDGE_ALPHA_GRID:
        prediction = np.full(len(y), np.nan)
        for fold in sorted(set(folds_array.tolist())):
            test = folds_array == fold; train = ~test
            model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, solver="lsqr"))]); model.fit(X[train], y[train]); prediction[test] = model.predict(X[test])
        trace.append({"alpha": float(alpha), "oof_r2": float(r2_score(y, prediction))})
    best = max(trace, key=lambda row: (row["oof_r2"], row["alpha"]))
    return float(best["alpha"]), trace


def weighted_sa_probe(cells: Sequence[dict[str, Any]], hidden: dict[str, np.ndarray], recipient: str) -> tuple[Pipeline, np.ndarray, list[str], float, float, list[dict[str, float]]]:
    selected = [c for c in cells if c["fixed_answer_color"] != recipient]
    X = np.stack([hidden[c["array_key"]] for c in selected]); y = np.asarray([c["mean_clean_final_sa"] for c in selected])
    folds = [int(c["outer_fold"]) for c in selected]; alpha, trace = select_ridge_alpha(X, y, folds)
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, solver="lsqr"))]); model.fit(X, y)
    raw = np.asarray(model.named_steps["ridge"].coef_ / model.named_steps["scaler"].scale_, dtype=np.float64)
    delta = X[0].astype(np.float64) - X[-1].astype(np.float64)
    error = abs(float(model.predict(X[[0]])[0] - model.predict(X[[-1]])[0]) - float(raw @ delta))
    if error > 1e-7: raise ValueError(f"Ridge raw-gradient conversion failed: {error}")
    return model, raw, [c["array_key"] for c in selected], error, alpha, trace


def shuffled_targets(cells: Sequence[dict[str, Any]], replicate: int) -> dict[str, float]:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 731, replicate])); output = {}
    by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells: by_answer[cell["fixed_answer_color"]].append(cell)
    for answer in CANONICAL_COLORS:
        ordered = sorted(by_answer[answer], key=lambda c: (c["family_id"], c["array_key"]))
        values = np.asarray([c["mean_G_L"] for c in ordered]); permuted = rng.permutation(values)
        output.update({c["array_key"]: float(v) for c, v in zip(ordered, permuted, strict=True)})
    return output


def target_norm(cells: Sequence[dict[str, Any]], hidden: dict[str, np.ndarray], included: Sequence[str]) -> float:
    allowed = set(included); values = [hidden[c["array_key"]] for c in cells if c["fixed_answer_color"] in allowed]
    return float(VECTOR_NORM_FRACTION * np.mean([np.linalg.norm(v) for v in values]))


def scale_vector(raw: np.ndarray, norm: float) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64); current = float(np.linalg.norm(value))
    if current <= 0 or not math.isfinite(current): raise ValueError("Cannot scale zero/non-finite vector")
    result = np.asarray(value / current * norm, dtype=np.float32)
    if not np.isfinite(result).all(): raise ValueError("Scaled vector is non-finite")
    return result


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    pearson = math.nan if np.ptp(y) == 0 or np.ptp(prediction) == 0 else float(pearsonr(y, prediction).statistic)
    spearman = math.nan if len(np.unique(y)) < 2 or len(np.unique(prediction)) < 2 else float(spearmanr(y, prediction).statistic)
    return {"r2": float(r2_score(y, prediction)), "pearson": pearson, "spearman": spearman, "mae": float(mean_absolute_error(y, prediction))}


def fit_oof_probe(X: np.ndarray, y: np.ndarray, rows: Sequence[dict[str, Any]], folds: Sequence[int]) -> tuple[np.ndarray, dict[int, Pipeline], dict[str, float], Pipeline, float, list[dict[str, float]]]:
    fold_values = sorted(set(map(int, folds))); prediction = np.full(len(rows), np.nan); models = {}
    families = np.asarray([str(r["family_id"]) for r in rows])
    alpha, trace = select_ridge_alpha(X, y, folds)
    for fold in fold_values:
        test = np.asarray(folds) == fold; train = ~test
        if not train.any() or not test.any() or set(families[train]) & set(families[test]): raise ValueError("Probe family split invalid")
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, solver="lsqr"))]); model.fit(X[train], y[train]); prediction[test] = model.predict(X[test]); models[fold] = model
    if not np.isfinite(prediction).all(): raise ValueError("Incomplete OOF probe predictions")
    full = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, solver="lsqr"))]); full.fit(X, y)
    return prediction, models, regression_metrics(y, prediction), full, alpha, trace


def raw_gradient(model: Pipeline) -> np.ndarray:
    return np.asarray(model.named_steps["ridge"].coef_ / model.named_steps["scaler"].scale_, dtype=np.float64)
