from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
for candidate in (REPOSITORY_ROOT, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
from scipy import stats

from dp_SA.io_utils import atomic_json, load_jsonl
from config import BOOTSTRAP_REPEATS, OUTPUT_ROOT, SEED


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _corr(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float | None, float | None]:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None
    result = stats.pearsonr(x, y) if kind == "pearson" else stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _fit(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    if len(x) < 2 or np.ptp(x) == 0:
        return {key: None for key in ("intercept", "slope", "r2", "mae", "rmse")}
    design = np.column_stack([np.ones(len(x)), x]); beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta; total = np.sum((y - y.mean()) ** 2)
    return {
        "intercept": float(beta[0]), "slope": float(beta[1]),
        "r2": float(1 - np.sum(residual ** 2) / total) if total > 0 else None,
        "mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(np.mean(residual ** 2))),
    }


def _pairs(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(float(row[x_key]), float(row[y_key])) for row in rows if row.get(x_key) is not None and row.get(y_key) is not None]
    return (np.asarray([x for x, _ in pairs]), np.asarray([y for _, y in pairs])) if pairs else (np.array([]), np.array([]))


def _bootstrap(rows: list[dict[str, Any]], statistic: Callable[[list[dict[str, Any]]], float], repeats: int) -> list[float | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[str(row["item_id"])].append(row)
    ids = sorted(grouped)
    if len(ids) < 2: return [None, None]
    rng = np.random.default_rng(SEED); values = []
    for _ in range(repeats):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        value = statistic([row for item in sampled for row in grouped[str(item)]])
        if math.isfinite(value): values.append(value)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else [None, None]


def _metric(rows: list[dict[str, Any]], group: str, x_key: str, y_key: str, label: str, repeats: int) -> dict[str, Any]:
    x, y = _pairs(rows, x_key, y_key)
    p, pp = _corr(x, y, "pearson"); s, sp = _corr(x, y, "spearman"); fit = _fit(x, y)
    def correlation(sample: list[dict[str, Any]], kind: str) -> float:
        bx, by = _pairs(sample, x_key, y_key); value, _ = _corr(bx, by, kind)
        return float("nan") if value is None else value
    def slope(sample: list[dict[str, Any]]) -> float:
        bx, by = _pairs(sample, x_key, y_key); value = _fit(bx, by)["slope"]
        return float("nan") if value is None else float(value)
    pci = _bootstrap(rows, lambda sample: correlation(sample, "pearson"), repeats)
    sci = _bootstrap(rows, lambda sample: correlation(sample, "spearman"), repeats)
    lci = _bootstrap(rows, slope, repeats)
    nonzero = [(np.sign(float(row[x_key])), np.sign(float(row[y_key]))) for row in rows if float(row[x_key]) != 0 and float(row[y_key]) != 0]
    return {
        "group": group, "comparison": label, "n": len(x), "pearson": p, "pearson_p_value": pp,
        "pearson_ci_low": pci[0], "pearson_ci_high": pci[1], "spearman": s, "spearman_p_value": sp,
        "spearman_ci_low": sci[0], "spearman_ci_high": sci[1], **fit,
        "slope_ci_low": lci[0], "slope_ci_high": lci[1],
        "sign_agreement_rate": float(np.mean([a == b for a, b in nonzero])) if nonzero else None,
    }


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list): return [_clean(item) for item in value]
    return value


def analyze(root: Path, reverse_root: Path, repeats: int) -> dict[str, Any]:
    trials = {row["case_id"]: row for row in load_jsonl(root / "trials.jsonl") if row.get("status") == "completed"}
    reverse = {row["case_id"]: row for row in load_jsonl(reverse_root / "reverse_softsa.jsonl") if row.get("status") == "completed"}
    if len(trials) != 110 or len(reverse) != 110:
        raise ValueError(f"Expected 110 source and reverse records, found {len(trials)} and {len(reverse)}")
    rows = []
    for case_id, trial in trials.items():
        rev = reverse[case_id]
        rows.append({
            "case_id": case_id, "item_id": trial["item_id"], "difficulty": trial["difficulty"],
            "cma_signed": trial["cma_logit"]["cma_signed"],
            "cma_log_probability_signed": trial["cma_log_probability"]["cma_signed"],
            "short_sa_signed": trial["short_sa"]["soft_sa_signed"],
            "reverse_sa_signed": rev["soft_sa_signed"],
            "reverse_sa_raw": rev["soft_sa_image_score"],
            "reverse_raw_argmax_class": rev["raw_argmax_class"],
            "short_reverse_delta": rev["soft_sa_signed"] - trial["short_sa"]["soft_sa_signed"],
        })
    table = root / "reverse_softsa" / "tables"; _csv(table / "case_level.csv", rows)
    metrics = []
    for group, subset in (("overall", rows), ("easy", [r for r in rows if r["difficulty"] == "easy"]), ("hard", [r for r in rows if r["difficulty"] == "hard"])):
        metrics.append(_metric(subset, group, "cma_signed", "reverse_sa_signed", "cma_vs_reverse_soft_sa", repeats))
        metrics.append(_metric(subset, group, "short_sa_signed", "reverse_sa_signed", "short_vs_reverse_soft_sa", repeats))
        metrics.append(_metric(subset, group, "cma_log_probability_signed", "reverse_sa_signed", "log_probability_cma_vs_reverse_soft_sa", repeats))
    _csv(table / "correlation_and_fit.csv", metrics)
    x, y = _pairs(rows, "cma_signed", "reverse_sa_signed")
    summary = {
        "status": "complete", "n": len(rows), "bootstrap_repeats": repeats,
        "metrics": metrics, "mean_reverse_sa_signed": float(np.mean([r["reverse_sa_signed"] for r in rows])),
        "mean_short_sa_signed": float(np.mean([r["short_sa_signed"] for r in rows])),
        "mean_reverse_minus_short": float(np.mean([r["short_reverse_delta"] for r in rows])),
        "reverse_vs_short_mae": float(np.mean(np.abs([r["short_reverse_delta"] for r in rows]))),
        "reverse_vs_short_sign_agreement": float(np.mean(np.sign([r["short_sa_signed"] for r in rows]) == np.sign([r["reverse_sa_signed"] for r in rows]))),
    }
    cleaned = _clean(summary); atomic_json(reverse_root / "summary.json", cleaned); return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze reverse Soft-SA results")
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT.parent / "faithful_check_extended" / "balanced_subset")
    parser.add_argument("--reverse-root", type=Path, default=OUTPUT_ROOT.parent / "faithful_check_extended" / "balanced_subset" / "reverse_softsa")
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(); print(json.dumps(analyze(args.root, args.reverse_root, args.bootstrap_repeats), ensure_ascii=False, indent=2))
