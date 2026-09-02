from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CANONICAL_COLORS, CONFIDENCE_JOINED, LAYERS, RESULTS_ROOT, SEED, TRAIN_MANIFEST
from .core import build_vectors, tail_assignments
from .io_utils import atomic_joblib, atomic_json, atomic_jsonl, atomic_npz, canonical_hash, load_jsonl, sha256_file


AUDIT_DIR = Path("artifacts/followup_audits")
SA_ROOT = Path(__file__).resolve().parents[1] / "answer_matched_lat_steering/output/results"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 0 else math.nan


class NonlinearNuisanceFeatures(BaseEstimator, TransformerMixin):
    """Fold-fitted nuisance features with the prespecified nonlinear terms."""

    def fit(self, x: np.ndarray, y: np.ndarray | None = None):
        rows = np.asarray(x, dtype=object)
        numeric = self._numeric(rows)
        self.numeric_scaler_ = StandardScaler().fit(numeric)
        self.categorical_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(rows[:, 2:6])
        self.origin_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(rows[:, 4:5])
        return self

    @staticmethod
    def _numeric(rows: np.ndarray) -> np.ndarray:
        dt, di = rows[:, 0].astype(float), rows[:, 1].astype(float)
        return np.column_stack((dt, di, dt * dt, di * di, dt * di))

    def transform(self, x: np.ndarray) -> np.ndarray:
        rows = np.asarray(x, dtype=object)
        numeric = self.numeric_scaler_.transform(self._numeric(rows))
        categorical = self.categorical_encoder_.transform(rows[:, 2:6])
        origin = self.origin_encoder_.transform(rows[:, 4:5])
        interactions = np.concatenate((numeric[:, 0:1] * origin, numeric[:, 1:2] * origin), axis=1)
        return np.concatenate((numeric, categorical, interactions), axis=1)


def _matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[r[k] for k in ("D_t", "D_i", "Hard", "prior_bin", "answer_origin", "fixed_answer_color")] for r in rows], dtype=object)


def nonlinear_oof(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    predictions: dict[str, float] = {}
    models: dict[int, dict[str, Any]] = {}
    for fold in range(5):
        fit = [r for r in rows if int(r["outer_fold"]) != fold]
        held = [r for r in rows if int(r["outer_fold"]) == fold]
        features = NonlinearNuisanceFeatures().fit(_matrix(fit))
        ridge = Ridge(alpha=1.0, solver="lsqr").fit(features.transform(_matrix(fit)), np.asarray([r["G_L"] for r in fit]))
        values = ridge.predict(features.transform(_matrix(held)))
        predictions.update({r["case_id"]: float(v) for r, v in zip(held, values, strict=True)})
        models[fold] = {"features": features, "ridge": ridge}
    if len(predictions) != len(rows):
        raise ValueError("Incomplete nonlinear OOF predictions")
    return [{**r, "predicted_G_L_oof_nonlinear": predictions[r["case_id"]], "R_C_nonlinear": float(r["G_L"] - predictions[r["case_id"]])} for r in rows], models


def _smd(high: Sequence[float], low: Sequence[float]) -> float:
    h, l = np.asarray(high, float), np.asarray(low, float)
    pooled = math.sqrt((float(np.var(h, ddof=1)) + float(np.var(l, ddof=1))) / 2)
    return float((h.mean() - l.mean()) / pooled) if pooled > 0 else 0.0


def _categorical_gap(high: Sequence[Any], low: Sequence[Any]) -> dict[str, Any]:
    cats = sorted({str(x) for x in high} | {str(x) for x in low})
    hp, lp = Counter(map(str, high)), Counter(map(str, low))
    gaps = {c: hp[c] / len(high) - lp[c] / len(low) for c in cats}
    key = max(gaps, key=lambda c: abs(gaps[c]))
    return {"max_abs_proportion_difference": abs(gaps[key]), "max_category": key, "signed_difference_high_minus_low": gaps[key],
            "high_proportions": {c: hp[c] / len(high) for c in cats}, "low_proportions": {c: lp[c] / len(low) for c in cats}}


def _balance(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifests = {r["case_id"]: r for r in load_jsonl(TRAIN_MANIFEST)}
    joined = {r["case_id"]: r for r in load_jsonl(CONFIDENCE_JOINED) if r["split"] == "train"}
    enriched = []
    for r in rows:
        m, j = manifests[r["case_id"]], joined[r["case_id"]]
        tm, im = j["text_chosen_answer"] == j["fixed_answer"], j["image_chosen_answer"] == j["fixed_answer"]
        match = "both" if tm and im else "text_only" if tm else "image_only" if im else "neither"
        enriched.append({**r, "unimodal_chosen_match": match, "clean_final_sa": float(m["soft_sa_image_score"])})

    tails = load_jsonl(root / "artifacts/family_answer_cells/eligibility_and_tails.jsonl")
    label = {}
    for t in tails:
        if not t["eligible"]:
            continue
        for side in ("high", "low"):
            for family in t[f"true_{side}_families"]:
                label[(family, t["fixed_answer_color"])] = side
    tail_records = [dict(r, tail=label[(r["family_id"], r["fixed_answer_color"])]) for r in enriched if (r["family_id"], r["fixed_answer_color"]) in label]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in tail_records:
        grouped[(r["family_id"], r["fixed_answer_color"])].append(r)
    cells = []
    for (family, color), members in sorted(grouped.items()):
        row: dict[str, Any] = {"family_id": family, "fixed_answer_color": color, "tail": members[0]["tail"]}
        for key in ("R_C", "D_t", "D_i", "Hard", "clean_final_sa"):
            row[key] = float(np.mean([x[key] for x in members]))
        for key in ("answer_origin", "prior_bin", "unimodal_chosen_match"):
            row[key] = dict(Counter(str(x[key]) for x in members))
        cells.append(row)

    corr = lambda values, a, b: float(np.corrcoef([r[a] for r in values], [r[b] for r in values])[0, 1])
    result: dict[str, Any] = {
        "correlations": {
            "all_training_records": {"record_count": len(enriched), "corr_R_C_D_t": corr(enriched, "R_C", "D_t"), "corr_R_C_D_i": corr(enriched, "R_C", "D_i")},
            "tail_cells": {"cell_count": len(cells), "corr_R_C_D_t": corr(cells, "R_C", "D_t"), "corr_R_C_D_i": corr(cells, "R_C", "D_i")},
        },
        "tail_cell_balance": {}, "tail_record_balance": {},
    }
    high_cells, low_cells = [r for r in cells if r["tail"] == "high"], [r for r in cells if r["tail"] == "low"]
    high_rec, low_rec = [r for r in tail_records if r["tail"] == "high"], [r for r in tail_records if r["tail"] == "low"]
    for key in ("D_t", "D_i", "Hard", "clean_final_sa"):
        result["tail_cell_balance"][key] = {"high_mean": float(np.mean([r[key] for r in high_cells])), "low_mean": float(np.mean([r[key] for r in low_cells])), "smd": _smd([r[key] for r in high_cells], [r[key] for r in low_cells])}
        result["tail_record_balance"][key] = {"high_mean": float(np.mean([r[key] for r in high_rec])), "low_mean": float(np.mean([r[key] for r in low_rec])), "smd": _smd([r[key] for r in high_rec], [r[key] for r in low_rec])}
    # For cell-level categorical balance, each cell contributes its within-cell distribution equally.
    for key in ("answer_origin", "prior_bin", "unimodal_chosen_match"):
        cats = sorted({c for r in cells for c in r[key]})
        def props(group):
            return {c: float(np.mean([r[key].get(c, 0) / sum(r[key].values()) for r in group])) for c in cats}
        hp, lp = props(high_cells), props(low_cells); gaps = {c: hp[c] - lp[c] for c in cats}; cat = max(gaps, key=lambda c: abs(gaps[c]))
        result["tail_cell_balance"][key] = {"max_abs_proportion_difference": abs(gaps[cat]), "max_category": cat, "signed_difference_high_minus_low": gaps[cat], "high_proportions": hp, "low_proportions": lp}
        result["tail_record_balance"][key] = _categorical_gap([r[key] for r in high_rec], [r[key] for r in low_rec])
    for key in ("fixed_answer_color",):
        result["tail_cell_balance"][key] = _categorical_gap([r[key] for r in high_cells], [r[key] for r in low_cells])
        result["tail_record_balance"][key] = _categorical_gap([r[key] for r in high_rec], [r[key] for r in low_rec])
    result["tail_cell_balance"]["Hard"]["interpretation_as_binary_proportion_gap"] = result["tail_cell_balance"]["Hard"]["high_mean"] - result["tail_cell_balance"]["Hard"]["low_mean"]
    result["thresholds"] = {"absolute_correlation_flag": 0.1, "absolute_smd_flag": 0.2, "categorical_proportion_gap_flag": 0.1}
    return result


def _load_cell_arrays(root: Path) -> dict[int, dict[str, np.ndarray]]:
    output = {}
    for layer in LAYERS:
        with np.load(root / f"artifacts/family_answer_cells/P1_LAT__L{layer}.npz") as z:
            output[layer] = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
    return output


def _relabel_cells(base_cells: list[dict[str, Any]], rows: list[dict[str, Any]], residual_key: str) -> list[dict[str, Any]]:
    by_case = {r["case_id"]: r for r in rows}
    return [{**cell, "mean_residual": float(np.mean([by_case[c][residual_key] for c in cell["case_ids"]]))} for cell in base_cells]


def _score_vectors(cells: list[dict[str, Any]], arrays: dict[int, dict[str, np.ndarray]], score_by_cell: dict[str, float], recipients: list[str]) -> dict[tuple[int, str], np.ndarray]:
    scored = [{**c, "mean_residual": score_by_cell[c["array_key"]]} for c in cells]
    assignments, eligibility = tail_assignments(scored)
    vectors, _ = build_vectors(scored, arrays, assignments, eligibility, recipients, layers=LAYERS)
    return {(layer, recipient): values[f"{recipient}__residual_confidence_loao__raw"] for layer, values in vectors.items() for recipient in recipients}


def _load_confidence_vectors(root: Path, direction: str = "residual_confidence_loao") -> dict[tuple[int, str], np.ndarray]:
    out = {}
    for layer in LAYERS:
        with np.load(root / f"artifacts/vectors/P1_LAT__L{layer}.npz") as z:
            for recipient in CANONICAL_COLORS:
                out[layer, recipient] = np.asarray(z[f"{recipient}__{direction}__raw"], dtype=np.float32)
    return out


def _load_sa_vectors() -> tuple[dict[tuple[int, str], np.ndarray], list[dict[str, Any]]]:
    metadata = json.loads((SA_ROOT / "artifacts/vectors/vector_metadata.json").read_text())
    unit_groups: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    detail = []
    for row in metadata["vectors"]:
        if row["direction"] != "matched_loao" or int(row["layer"]) not in LAYERS:
            continue
        with np.load(SA_ROOT / row["vector_file"]) as z:
            value = np.asarray(z[row["raw_key"]], dtype=np.float32)
        unit_groups[int(row["layer"]), str(row["recipient_answer"])].append(value / np.linalg.norm(value))
        detail.append({"layer": int(row["layer"]), "recipient_answer": str(row["recipient_answer"]), "fold": int(row["fold"]), "vector": value})
    consensus = {key: np.stack(values).mean(axis=0, dtype=np.float32) for key, values in unit_groups.items()}
    return consensus, detail


def _shuffle_assignments(cells: list[dict[str, Any]], replicate: int) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        by_answer[c["fixed_answer_color"]].append(c)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, replicate]))
    assignments, eligibility = {}, []
    for answer in CANONICAL_COLORS:
        ordered = sorted(by_answer.get(answer, []), key=lambda r: (float(r["mean_residual"]), str(r["family_id"])))
        n, k = len(ordered), max(2, int(math.floor(.3 * len(ordered)))) if ordered else 0
        eligible = n >= 8 and 2 * k <= n
        if eligible:
            stable = sorted(ordered, key=lambda r: str(r["family_id"]))
            labels = ["low"] * k + ["middle"] * (n - 2 * k) + ["high"] * k
            labels = [str(v) for v in rng.permutation(labels)]
            assignments[answer] = {f"true:{c['array_key']}": label for c, label in zip(stable, labels, strict=True)}
            assignments[answer].update({f"shuffled:{c['array_key']}": label for c, label in zip(stable, labels, strict=True)})
        eligibility.append({"fixed_answer_color": answer, "cell_count": n, "tail_count": k, "eligible": eligible})
    return assignments, eligibility


def _raw_vectors_from_assignments(
    cells: list[dict[str, Any]],
    arrays: dict[int, dict[str, np.ndarray]],
    assignments: dict[str, dict[str, str]],
    eligibility: list[dict[str, Any]],
    recipients: list[str],
) -> dict[tuple[int, str], np.ndarray]:
    """Fast raw-only equivalent of build_vectors for repeated shuffles."""
    eligible = [r["fixed_answer_color"] for r in eligibility if r["eligible"]]
    keys_by_answer: dict[str, list[str]] = defaultdict(list)
    for cell in cells:
        keys_by_answer[cell["fixed_answer_color"]].append(cell["array_key"])
    output: dict[tuple[int, str], np.ndarray] = {}
    for layer in LAYERS:
        answer_directions = {}
        for answer in eligible:
            mapping = assignments[answer]
            high = [arrays[layer][key] for key in keys_by_answer[answer] if mapping[f"true:{key}"] == "high"]
            low = [arrays[layer][key] for key in keys_by_answer[answer] if mapping[f"true:{key}"] == "low"]
            answer_directions[answer] = np.stack(high).mean(axis=0, dtype=np.float32) - np.stack(low).mean(axis=0, dtype=np.float32)
        total = np.stack([answer_directions[a] for a in eligible]).sum(axis=0, dtype=np.float32)
        for recipient in recipients:
            included_count = len(eligible) - int(recipient in answer_directions)
            value = total - answer_directions.get(recipient, 0)
            output[layer, recipient] = np.asarray(value / included_count, dtype=np.float32)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run_audit(output_root: Path = RESULTS_ROOT, shuffle_repeats: int = 1000) -> dict[str, Any]:
    root = output_root.resolve(); out = root / AUDIT_DIR; out.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(root / "artifacts/residualization/confidence_training_records.jsonl")
    cells = load_jsonl(root / "artifacts/family_answer_cells/cells.jsonl")
    arrays = _load_cell_arrays(root); recipients = list(CANONICAL_COLORS)
    balance = _balance(root, rows); atomic_json(out / "residual_balance.json", balance)

    nonlinear, models = nonlinear_oof(rows)
    atomic_jsonl(out / "nonlinear_oof_predictions.jsonl", nonlinear)
    model_hashes = {}
    for fold, model in models.items():
        path = out / f"nonlinear_fold_{fold}.joblib"; atomic_joblib(path, model); model_hashes[str(fold)] = sha256_file(path)
    nonlinear_cells = _relabel_cells(cells, nonlinear, "R_C_nonlinear")
    assignments, eligibility = tail_assignments(nonlinear_cells)
    nonlinear_vectors, nonlinear_meta = build_vectors(nonlinear_cells, arrays, assignments, eligibility, recipients, layers=LAYERS)
    old_vectors = _load_confidence_vectors(root)
    nonlinear_cos = []
    saved_nonlinear = {}
    for layer in LAYERS:
        for recipient in recipients:
            key = f"{recipient}__residual_confidence_loao__raw"
            value = nonlinear_vectors[layer][key]; saved_nonlinear[f"L{layer}__{recipient}"] = value
            nonlinear_cos.append({"layer": layer, "recipient_answer": recipient, "cosine_old_vs_nonlinear": cosine(old_vectors[layer, recipient], value)})
    atomic_npz(out / "nonlinear_vectors.npz", saved_nonlinear); _write_csv(out / "nonlinear_vector_cosines.csv", nonlinear_cos)

    by_case = {r["case_id"]: r for r in rows}
    difficulty_score = {c["array_key"]: float(np.mean([by_case[x]["D_t"] - by_case[x]["D_i"] for x in c["case_ids"]])) for c in cells}
    dt_score = {c["array_key"]: float(np.mean([by_case[x]["D_t"] for x in c["case_ids"]])) for c in cells}
    di_score = {c["array_key"]: float(np.mean([by_case[x]["D_i"] for x in c["case_ids"]])) for c in cells}
    difficulty_vectors = _score_vectors(cells, arrays, difficulty_score, recipients)
    dt_vectors = _score_vectors(cells, arrays, dt_score, recipients); di_vectors = _score_vectors(cells, arrays, di_score, recipients)
    sa_consensus, sa_detail = _load_sa_vectors()

    direction_rows = []
    for layer in LAYERS:
        for recipient in recipients:
            current = old_vectors[layer, recipient]
            matching_sa = [r for r in sa_detail if r["layer"] == layer and r["recipient_answer"] == recipient]
            fold_cos = [cosine(current, r["vector"]) for r in matching_sa]
            direction_rows.append({"layer": layer, "recipient_answer": recipient,
                "cos_confidence_difficulty_advantage": cosine(current, difficulty_vectors[layer, recipient]),
                "cos_confidence_D_t": cosine(current, dt_vectors[layer, recipient]), "cos_confidence_D_i": cosine(current, di_vectors[layer, recipient]),
                "cos_confidence_sa_consensus": cosine(current, sa_consensus[layer, recipient]) if (layer, recipient) in sa_consensus else math.nan,
                "sa_fold_cosine_mean": float(np.mean(fold_cos)) if fold_cos else math.nan, "sa_fold_cosine_min": min(fold_cos) if fold_cos else math.nan,
                "sa_fold_cosine_max": max(fold_cos) if fold_cos else math.nan, "sa_fold_count": len(fold_cos)})
    _write_csv(out / "direction_cosines.csv", direction_rows)

    shuffle_rows, fixed_arrays = [], {}
    for replicate in range(shuffle_repeats):
        shuffled, shuffled_eligibility = _shuffle_assignments(cells, replicate)
        vectors = _raw_vectors_from_assignments(cells, arrays, shuffled, shuffled_eligibility, recipients)
        for layer in LAYERS:
            for recipient in recipients:
                raw = vectors[layer, recipient]
                shuffle_rows.append({"replicate": replicate, "layer": layer, "recipient_answer": recipient,
                    "cos_shuffle_true_confidence": cosine(raw, old_vectors[layer, recipient]),
                    "cos_shuffle_difficulty_advantage": cosine(raw, difficulty_vectors[layer, recipient]),
                    "cos_shuffle_sa_consensus": cosine(raw, sa_consensus[layer, recipient]) if (layer, recipient) in sa_consensus else math.nan})
                if replicate < 20 and layer == 14:
                    target = next(r["target_norm"] for r in json.loads((root / "artifacts/vectors/vector_metadata.json").read_text())["vectors"] if int(r["layer"]) == 14 and r["recipient_answer"] == recipient and r["direction"] == "residual_confidence_loao")
                    fixed_arrays[f"rep_{replicate:03d}__{recipient}__scaled"] = np.asarray(raw / np.linalg.norm(raw) * target, dtype=np.float32)
    _write_csv(out / "shuffle_vector_cosines.csv", shuffle_rows); atomic_npz(out / "shuffle_vectors_l14.npz", fixed_arrays)

    layer_summary = []
    for layer in LAYERS:
        nr = [r["cosine_old_vs_nonlinear"] for r in nonlinear_cos if r["layer"] == layer]
        dr = [r for r in direction_rows if r["layer"] == layer]
        sr = [r for r in shuffle_rows if r["layer"] == layer]
        def stats(values):
            x = np.asarray([v for v in values if math.isfinite(v)]); return {"mean": float(x.mean()) if len(x) else None, "median": float(np.median(x)) if len(x) else None, "min": float(x.min()) if len(x) else None, "max": float(x.max()) if len(x) else None, "n": int(len(x))}
        layer_summary.append({"layer": layer, "old_vs_nonlinear": stats(nr),
            "confidence_vs_difficulty_advantage": stats([r["cos_confidence_difficulty_advantage"] for r in dr]),
            "confidence_vs_answer_matched_sa_consensus": stats([r["cos_confidence_sa_consensus"] for r in dr]),
            "shuffle_vs_true_confidence": stats([r["cos_shuffle_true_confidence"] for r in sr]),
            "shuffle_vs_difficulty_advantage": stats([r["cos_shuffle_difficulty_advantage"] for r in sr]),
            "shuffle_vs_answer_matched_sa_consensus": stats([r["cos_shuffle_sa_consensus"] for r in sr])})
    config = {"seed": SEED, "shuffle_repeats": shuffle_repeats, "fixed_gpu_shuffle_replicates": list(range(20)), "difficulty_direction_definition": "cell mean(D_t-D_i), high minus low, answer-equal LOAO", "sa_source": str(SA_ROOT), "sa_L8_availability": False,
              "inputs": {"training_records": sha256_file(root / "artifacts/residualization/confidence_training_records.jsonl"), "cells": sha256_file(root / "artifacts/family_answer_cells/cells.jsonl")}, "nonlinear_model_hashes": model_hashes}
    config["fingerprint"] = canonical_hash(config); atomic_json(out / "audit_config.json", config)
    result = {"status": "complete", "balance": balance, "layer_summary": layer_summary, "config_fingerprint": config["fingerprint"], "fixed_shuffle_vector_file": str(out / "shuffle_vectors_l14.npz")}
    atomic_json(out / "audit_summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", default=str(RESULTS_ROOT)); parser.add_argument("--shuffle-repeats", type=int, default=1000)
    args = parser.parse_args(argv); result = run_audit(Path(args.output_root), args.shuffle_repeats); print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
