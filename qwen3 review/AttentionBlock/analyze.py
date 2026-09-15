from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dp_SA.io_utils import atomic_json, load_jsonl

from .config import (
    BOOTSTRAP_REPEATS, CONDITIONS, GROUPS, METRICS, PRIMARY_GROUP,
    ROW_SUM_TOLERANCE, SEED,
)
from .contracts import load_trials


def bh_fdr(values: Sequence[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def _pvalue(samples: np.ndarray) -> float:
    low = (1 + np.count_nonzero(samples <= 0)) / (len(samples) + 1)
    high = (1 + np.count_nonzero(samples >= 0)) / (len(samples) + 1)
    return float(min(1.0, 2 * min(low, high)))


def _aggregate(values: dict[str, float], manifest: dict[str, dict[str, Any]],
               group: str, repeats: int) -> dict[str, Any]:
    if group == "image_side":
        values = {key: value for key, value in values.items() if manifest[key]["test_side"] == group}
    elif group == "text_side":
        values = {key: value for key, value in values.items() if manifest[key]["test_side"] == group}
    rng = np.random.default_rng(SEED)
    if group == PRIMARY_GROUP:
        by_answer: dict[str, list[str]] = {}
        for family in sorted(values):
            by_answer.setdefault(str(manifest[family]["test_answer"]), []).append(family)
        observed_parts, boot_parts = [], []
        for answer in sorted(by_answer):
            families = by_answer[answer]
            vector = np.asarray([values[family] for family in families], dtype=float)
            observed_parts.append(float(vector.mean()))
            indices = rng.integers(0, len(vector), size=(repeats, len(vector)))
            boot_parts.append(vector[indices].mean(axis=1))
        observed = float(np.mean(observed_parts))
        boot = np.stack(boot_parts).mean(axis=0)
        answer_count = len(by_answer)
    else:
        families = sorted(values)
        vector = np.asarray([values[family] for family in families], dtype=float)
        observed = float(vector.mean())
        indices = rng.integers(0, len(vector), size=(repeats, len(vector)))
        boot = vector[indices].mean(axis=1)
        answer_count = len({manifest[family]["test_answer"] for family in families})
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"mean": observed, "sem": float(np.std(boot, ddof=1)),
            "ci95_low": float(low), "ci95_high": float(high), "boot": boot,
            "family_count": len(values), "answer_count": answer_count}


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _plots(effect_rows: list[dict[str, Any]], output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    labels = {CONDITIONS[0]: "main block", CONDITIONS[1]: "source+1 control"}
    colors = {CONDITIONS[0]: "#d62728", CONDITIONS[1]: "#1f77b4"}
    names = {
        "delta_soft_sa": "Delta soft SA", "token_change_rate": "Token change rate",
        "logit_change_diff": "Logit change diff",
    }
    paths = []
    for index, metric in enumerate(METRICS, 1):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        base = [row for row in effect_rows if row["group"] == PRIMARY_GROUP and row["metric"] == metric]
        windows = sorted({(row["window_start"], row["window_end"]) for row in base})
        for condition in CONDITIONS:
            selected = sorted((row for row in base if row["condition"] == condition),
                              key=lambda row: row["window_start"])
            mean = np.asarray([row["mean"] for row in selected])
            low = np.asarray([row["ci95_low"] for row in selected])
            high = np.asarray([row["ci95_high"] for row in selected])
            x = np.arange(len(selected))
            ax.errorbar(x, mean, yerr=np.vstack((mean-low, high-mean)), marker="o",
                        linewidth=2, capsize=3, color=colors[condition], label=labels[condition])
        if metric != "token_change_rate":
            ax.axhline(0, color="black", linewidth=.8, alpha=.65)
        ax.set_xticks(np.arange(len(windows)), [f"L{a}–{b}" for a, b in windows])
        ax.set_xlabel("Layer window"); ax.set_ylabel(names[metric]); ax.legend(frameon=False)
        ax.grid(axis="y", alpha=.2); fig.tight_layout()
        path = output / f"fig{index}_{metric}.png"
        fig.savefig(path, dpi=220); plt.close(fig); paths.append(str(path))
    return paths


def analyze(*, experiment: str, output_root: Path, repeats: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    root = output_root.resolve()
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    manifest_rows = load_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl")
    manifest = {str(row["family_id"]): row for row in manifest_rows}
    trials = load_trials(root)
    clean = [row for row in trials if row["condition"] == "C0_clean"]
    blocked = [row for row in trials if row["condition"] in CONDITIONS]
    windows = tuple(tuple(value) for value in config["spec"]["windows"])
    expected = len(manifest) * (1 + len(windows) * len(CONDITIONS))
    if len(clean) + len(blocked) != expected:
        raise RuntimeError(f"Incomplete analysis grid: {len(clean)+len(blocked)}/{expected}")

    effects: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for window in windows:
        for condition in CONDITIONS:
            subset = [row for row in blocked if row["condition"] == condition
                      and (row["window_start"], row["window_end"]) == window]
            for metric in METRICS:
                values = {str(row["family_id"]): float(row[metric]) for row in subset}
                for group in GROUPS:
                    stats = _aggregate(values, manifest, group, repeats)
                    effects.append({"window_start": window[0], "window_end": window[1],
                                    "condition": condition, "metric": metric, "group": group,
                                    **{key: stats[key] for key in ("mean", "sem", "ci95_low", "ci95_high", "family_count", "answer_count")},
                                    "bootstrap_repeats": repeats})
        lookup = {(row["case_id"], row["condition"]): row for row in blocked
                  if (row["window_start"], row["window_end"]) == window}
        for metric in METRICS:
            values = {str(row["family_id"]): float(row[metric]) - float(lookup[(row["case_id"], CONDITIONS[1])][metric])
                      for row in blocked if row["condition"] == CONDITIONS[0]
                      and (row["window_start"], row["window_end"]) == window}
            for group in GROUPS:
                stats = _aggregate(values, manifest, group, repeats)
                paired.append({"window_start": window[0], "window_end": window[1],
                               "comparison": "main_block_minus_source_plus_1_control",
                               "metric": metric, "group": group, "specific_effect": stats["mean"],
                               **{key: stats[key] for key in ("sem", "ci95_low", "ci95_high", "family_count", "answer_count")},
                               "p_raw": _pvalue(stats["boot"]) if group == PRIMARY_GROUP else "",
                               "q_bh": "", "bootstrap_repeats": repeats})
    for metric in METRICS:
        family = [row for row in paired if row["metric"] == metric and row["group"] == PRIMARY_GROUP]
        for row, q in zip(family, bh_fdr([float(row["p_raw"]) for row in family])):
            row["q_bh"] = q

    audit: list[dict[str, Any]] = []
    for row in blocked:
        expected_layers = list(range(int(row["window_start"]), int(row["window_end"]) + 1))
        diagnostics = row["attention_diagnostics"]
        if diagnostics["layers"] != expected_layers:
            raise RuntimeError("Attention audit layer mismatch")
        for layer in expected_layers:
            detail = diagnostics["by_layer"][str(layer)]
            passed = (detail["max_blocked_weight"] == 0.0 and detail["max_row_sum_error"] <= ROW_SUM_TOLERANCE
                      and detail["finite"] and detail["hook_call_count"] == 1)
            if not passed:
                raise RuntimeError(f"Attention audit failed: {row['case_id']} L{layer}")
            audit.append({"case_id": row["case_id"], "condition": row["condition"],
                          "window_start": row["window_start"], "window_end": row["window_end"],
                          "layer": layer, "query_name": row["query_name"], "source_name": row["source_name"],
                          **detail, "passed": True})
    tables = root / "tables"
    _csv(tables / "case_level_trials.csv", clean + blocked)
    _csv(tables / "window_effects.csv", effects)
    _csv(tables / "paired_main_vs_control.csv", paired)
    _csv(tables / "attention_audit.csv", audit)
    atomic_json(tables / "summary.json", {
        "experiment": experiment, "case_count": len(manifest),
        "side_counts": dict(Counter(row["test_side"] for row in manifest_rows)),
        "trial_count": len(trials), "attention_audit_rows": len(audit),
    })
    figures = _plots(effects, root / "figures")
    result = {"status": "complete", "experiment": experiment, "case_count": len(manifest),
              "trial_count": len(trials), "attention_audit_rows": len(audit),
              "bootstrap_repeats": repeats, "figures": figures}
    atomic_json(root / "progress" / "analysis.json", result)
    return result

