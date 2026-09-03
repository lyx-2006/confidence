from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import PERMUTATION_REPEATS, SEED
from .core import inner_fold


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denom) if denom > 0 else math.nan


def incremental_r2(score: np.ndarray, variable: np.ndarray, colors: Sequence[str]) -> float:
    score, variable = np.asarray(score, float), np.asarray(variable, float)
    categories = np.asarray(colors, object).reshape(-1, 1)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    base = encoder.fit_transform(categories)
    base_r2 = LinearRegression().fit(base, score).score(base, score)
    full = np.column_stack((base, StandardScaler().fit_transform(variable.reshape(-1, 1))))
    return float(LinearRegression().fit(full, score).score(full, score) - base_r2)


def continuous_association(score: np.ndarray, variable: np.ndarray, colors: Sequence[str]) -> dict[str, float]:
    score, variable = np.asarray(score, float), np.asarray(variable, float)
    return {
        "pearson": float(pearsonr(score, variable).statistic) if np.ptp(score) and np.ptp(variable) else math.nan,
        "spearman": float(spearmanr(score, variable).statistic) if len(np.unique(score)) > 1 and len(np.unique(variable)) > 1 else math.nan,
        "incremental_r2_controlling_fixed_answer": incremental_r2(score, variable, colors),
    }


def _family_permutations(labels: np.ndarray, families: Sequence[str], repeats: int) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(families): grouped[str(family)].append(index)
    ordered = sorted(grouped); blocks = [labels[grouped[f]] for f in ordered]
    rng = np.random.default_rng(SEED + 991)
    output = []
    for _ in range(repeats):
        permutation = rng.permutation(len(ordered)); value = labels.copy()
        for target, source in enumerate(permutation):
            target_indices = grouped[ordered[target]]
            value[target_indices] = np.resize(blocks[source], len(target_indices))
        output.append(value)
    return output


def discrete_association(score: np.ndarray, labels: Sequence[Any], rows: Sequence[dict[str, Any]], repeats: int = PERMUTATION_REPEATS) -> dict[str, Any]:
    score = np.asarray(score, float); labels_array = np.asarray([str(x) for x in labels]); unique = sorted(set(labels_array))
    families = [str(r["family_id"]) for r in rows]; folds = np.asarray([inner_fold(f) for f in families]); predicted = np.empty(len(rows), dtype=object)
    if len(unique) == 2:
        binary = (labels_array == unique[1]).astype(int); auc = float(roc_auc_score(binary, score)); observed = max(auc, 1.0 - auc); metric = "oriented_auroc"
        permutations = _family_permutations(binary, families, repeats)
        null = [max(float(roc_auc_score(p, score)), 1.0 - float(roc_auc_score(p, score))) for p in permutations if len(np.unique(p)) == 2]
    else:
        for fold in sorted(set(folds)):
            test = folds == fold; train = ~test
            model = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=2000, class_weight="balanced"))])
            model.fit(score[train, None], labels_array[train]); predicted[test] = model.predict(score[test, None])
        observed = float(balanced_accuracy_score(labels_array, predicted)); metric = "balanced_accuracy"
        null = [float(balanced_accuracy_score(p, predicted)) for p in _family_permutations(labels_array, families, repeats)]
    return {"metric": metric, "value": observed, "permutation_mean": float(np.mean(null)), "permutation_p": float((1 + sum(v >= observed for v in null)) / (len(null) + 1)), "permutation_repeats": len(null)}


def projection_audit_rows(
    audit_rows: Sequence[dict[str, Any]],
    hidden: np.ndarray,
    vectors: dict[str, np.ndarray],
    *, layer: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    table: list[dict[str, Any]] = []; summary: dict[str, dict[str, float]] = {}
    numeric = {
        "G_L": np.asarray([r["G_L"] for r in audit_rows]),
        "D_t": np.asarray([r["D_t"] for r in audit_rows]),
        "D_i": np.asarray([r["D_i"] for r in audit_rows]),
        "D_gap": np.asarray([r["D_t"] - r["D_i"] for r in audit_rows]),
        "clean_final_sa": np.asarray([r["clean_final_sa"] for r in audit_rows]),
        "Hard": np.asarray([r["Hard"] for r in audit_rows]),
    }
    categorical = ("Hard", "answer_origin", "prior_bin", "unimodal_chosen_match", "fixed_answer_color")
    colors = [r["fixed_answer_color"] for r in audit_rows]
    for direction, matrix in vectors.items():
        scores = np.asarray([hidden[i] @ matrix[colors[i]] for i in range(len(audit_rows))], float)
        summary[direction] = {}
        for variable, values in numeric.items():
            result = continuous_association(scores, values, colors)
            table.append({"split": "direction_audit", "layer": layer, "direction": direction, "variable": variable, "association_type": "continuous", **result})
            summary[direction][f"{variable}_pearson"] = result["pearson"]; summary[direction][f"{variable}_spearman"] = result["spearman"]
        for variable in categorical:
            result = discrete_association(scores, [r[variable] for r in audit_rows], audit_rows)
            table.append({"split": "direction_audit", "layer": layer, "direction": direction, "variable": variable, "association_type": "discrete", **result})
    return table, summary
