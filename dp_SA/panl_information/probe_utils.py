from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    log_loss, mean_absolute_error, roc_auc_score,
)

from .config import ARTIFACT_NAMES
from .io_utils import load_jsonl
from .metrics import item_bootstrap, pooled_r2


def stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def load_probe_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = load_jsonl(root / "artifacts" / ARTIFACT_NAMES["joined"])
    capture = {str(row["case_id"]): row for row in load_jsonl(root / "artifacts" / ARTIFACT_NAMES["capture"], repair_trailing=True) if row.get("status") == "completed"}
    selected = [row for row in rows if str(row["case_id"]) in capture]
    if not selected:
        raise ValueError("No joined records have captured hidden states")
    return selected, capture


def feature_matrix(root: Path, rows: Sequence[dict[str, Any]], capture: dict[str, dict[str, Any]], position: str, layer: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    key = f"{position}__L{layer}"
    for row in rows:
        captured = capture[str(row["case_id"])]
        path = root / str(captured["hidden_file"])
        with np.load(path) as payload:
            if key not in payload:
                raise ValueError(f"Hidden key is missing: {row['case_id']} {key}")
            vector = np.asarray(payload[key], dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"Invalid hidden vector: {row['case_id']} {key}")
        vectors.append(vector)
    return np.stack(vectors)


def regression_metrics(rows: Sequence[dict[str, Any]], target: np.ndarray, prediction: np.ndarray, *, repeats: int, seed: int) -> dict[str, Any]:
    base_rows = [{**row, "_target": float(y), "_prediction": float(p)} for row, y, p in zip(rows, target, prediction, strict=True)]
    observed = {
        "r2": pooled_r2(target, prediction),
        "spearman": float(spearmanr(target, prediction).statistic),
        "pearson": float(pearsonr(target, prediction).statistic),
        "mae": float(mean_absolute_error(target, prediction)),
    }
    for offset, metric in enumerate(("r2", "spearman", "pearson", "mae")):
        def statistic(sample: list[dict[str, Any]], name: str = metric) -> float:
            y = [row["_target"] for row in sample]
            p = [row["_prediction"] for row in sample]
            if name == "r2": return pooled_r2(y, p)
            if name == "spearman": return float(spearmanr(y, p).statistic)
            if name == "pearson": return float(pearsonr(y, p).statistic)
            return float(mean_absolute_error(y, p))
        observed[f"{metric}_ci"] = item_bootstrap(base_rows, statistic, repeats=repeats, seed=seed + offset)
    observed.update({"sample_count": len(rows), "item_count": len({str(row["item_id"]) for row in rows})})
    return observed


def classification_metrics(rows: Sequence[dict[str, Any]], y: np.ndarray, probability: np.ndarray, *, repeats: int, seed: int) -> dict[str, Any]:
    predicted = (probability >= 0.5).astype(int)
    observed = {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "auroc": float(roc_auc_score(y, probability)),
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "log_loss": float(log_loss(y, np.column_stack([1.0 - probability, probability]), labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "direction": float(probability[y == 1].mean() - probability[y == 0].mean()),
        "sample_count": len(rows), "item_count": len({str(row["item_id"]) for row in rows}),
    }
    base = [{**row, "_y": int(value), "_p": float(score)} for row, value, score in zip(rows, y, probability, strict=True)]
    for offset, metric in enumerate(("balanced_accuracy", "auroc")):
        def statistic(sample: list[dict[str, Any]], name: str = metric) -> float:
            truth = np.asarray([row["_y"] for row in sample], dtype=int)
            score = np.asarray([row["_p"] for row in sample], dtype=float)
            if name == "auroc": return float(roc_auc_score(truth, score))
            return float(balanced_accuracy_score(truth, score >= 0.5))
        observed[f"{metric}_ci"] = item_bootstrap(base, statistic, repeats=repeats, seed=seed + offset)
    return observed


def _unique_key(row: dict[str, Any], target: str) -> tuple[Any, ...]:
    if target == "text": return str(row["item_id"]), int(row["prior_index"])
    if target == "image": return str(row["item_id"]), str(row["condition"])
    if target == "decision": return str(row["case_id"]),
    raise ValueError(target)


def _layout_suffix(row: dict[str, Any], target: str) -> tuple[Any, ...]:
    if target == "text": return (int(row["prior_index"]),)
    if target == "image": return (str(row["condition"]),)
    return (int(row["prior_index"]), str(row["condition"]))


def fixed_oof_cluster_permutation(
    rows: Sequence[dict[str, Any]], truth: Sequence[float], prediction: Sequence[float], *, target: str,
    metric: Callable[[np.ndarray, np.ndarray], float], repeats: int, seed: int,
) -> dict[str, Any]:
    if len(rows) != len(truth) or len(rows) != len(prediction):
        raise ValueError("Permutation inputs differ in length")
    key_label: dict[tuple[Any, ...], float] = {}
    key_rows: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    key_suffix: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for index, (row, value) in enumerate(zip(rows, truth, strict=True)):
        key = _unique_key(row, target)
        old = key_label.setdefault(key, float(value))
        if old != float(value):
            raise ValueError(f"Conflicting label for unique target key: {key}")
        key_rows[key].append(index)
        key_suffix[key] = _layout_suffix(row, target)
    item_keys: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for key in key_label:
        item = str(key[0]).split("__", 1)[0] if target == "decision" else str(key[0])
        item_keys[item].append(key)
    for item in item_keys:
        item_keys[item].sort(key=lambda key: key_suffix[key])
    strata: dict[tuple[tuple[Any, ...], ...], list[str]] = defaultdict(list)
    for item, keys in item_keys.items():
        strata[tuple(key_suffix[key] for key in keys)].append(item)
    observed = float(metric(np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)))
    rng = np.random.default_rng(seed)
    greater = 0
    for _ in range(repeats):
        permuted_key_label: dict[tuple[Any, ...], float] = {}
        for items in strata.values():
            donors = list(rng.permutation(items))
            for recipient, donor in zip(items, donors, strict=True):
                recipient_keys, donor_keys = item_keys[recipient], item_keys[donor]
                for recipient_key, donor_key in zip(recipient_keys, donor_keys, strict=True):
                    permuted_key_label[recipient_key] = key_label[donor_key]
        permuted = np.empty(len(rows), dtype=float)
        for key, indices in key_rows.items():
            permuted[indices] = permuted_key_label[key]
        value = float(metric(permuted, np.asarray(prediction, dtype=float)))
        greater += int(value >= observed)
    return {
        "observed": observed, "p_value": float((greater + 1) / (repeats + 1)), "repeats": repeats,
        "greater_or_equal_count": greater, "unique_target_key_count": len(key_label),
        "item_count": len(item_keys), "stratum_count": len(strata), "policy": "fixed_oof_item_block_same_layout",
    }


def majority_oof(rows: Sequence[dict[str, Any]], y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    probability = np.empty(len(rows), dtype=float)
    details = {}
    for fold in range(5):
        train = np.asarray([int(row["outer_fold"]) != fold for row in rows])
        test = ~train
        if not test.any():
            continue
        counts = Counter(y[train].tolist())
        label = 0 if counts[0] >= counts[1] else 1
        probability[test] = float(label)
        details[str(fold)] = {"label": int(label), "train_counts": {str(k): int(v) for k, v in counts.items()}}
    return probability, details
