from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dp_SA.io_utils import atomic_json, load_jsonl

from .config import BOOTSTRAP_REPEATS, SEED
from .steered_block import ALPHAS, WINDOWS


def summarize(values: Sequence[float], repeats: int, seed: int) -> dict[str, float]:
    vector = np.asarray(values, dtype=float)
    if not len(vector) or not np.isfinite(vector).all(): raise ValueError("Cannot summarize empty/non-finite values")
    rng = np.random.default_rng(seed)
    boot = vector[rng.integers(0, len(vector), (repeats, len(vector)))].mean(1)
    low, high = np.quantile(boot, [.025, .975])
    return {"mean": float(vector.mean()), "ci95_low": float(low), "ci95_high": float(high), "count": len(vector)}


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def analyze(*, output_root: Path, repeats: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    root = Path(output_root).resolve(); config = json.loads((root / "run_config.json").read_text())
    if config.get("mode") != "steered_block": raise ValueError("Output root is not steered-block mode")
    manifest = load_jsonl(root / "artifacts/manifests/test.jsonl")
    trials = load_jsonl(root / "artifacts/trials.jsonl")
    clean = {r["case_id"]: r for r in trials if r["condition"] == "C0"}
    steered = {(r["case_id"], float(r["alpha"])): r for r in trials if r["condition"] == "S"}
    blocked = [r for r in trials if r["condition"] == "SB"]
    expected = len(manifest) * (1 + len(ALPHAS) + len(ALPHAS)*len(WINDOWS))
    if len(trials) != expected or len(clean) != len(manifest) or len(steered) != len(manifest)*len(ALPHAS):
        raise RuntimeError("Incomplete steered-block analysis grid")
    rows: list[dict[str, Any]] = []; counter = 0
    for alpha in ALPHAS:
        s_rows = [r for r in steered.values() if float(r["alpha"]) == alpha]
        for comparison, metric_fields, selected in (
            ("S_vs_C0", (("token_change_rate", "vs_clean_token_change_rate"),
                         ("logit_change_diff", "vs_clean_logit_change_diff")), s_rows),
        ):
            for metric, field in metric_fields:
                for group in ("overall", "text_side", "image_side"):
                    chosen = selected if group == "overall" else [r for r in selected if r["test_side"] == group]
                    rows.append({"comparison": comparison, "alpha": alpha, "window_start": "", "window_end": "",
                                 "metric": metric, "group": group,
                                 **summarize([float(r[field]) for r in chosen], repeats, SEED+counter)})
                    counter += 1
        for window in WINDOWS:
            cell = [r for r in blocked if float(r["alpha"]) == alpha and
                    (int(r["window_start"]), int(r["window_end"])) == window]
            for comparison, metric_fields in (
                ("SB_vs_S", (("token_change_rate", "vs_steered_token_change_rate"),
                             ("logit_change_diff", "vs_steered_logit_change_diff"))),
                ("SB_vs_C0", (("token_change_rate", "vs_clean_token_change_rate"),
                              ("logit_change_diff", "vs_clean_logit_change_diff"))),
            ):
                for metric, field in metric_fields:
                    for group in ("overall", "text_side", "image_side"):
                        chosen = cell if group == "overall" else [r for r in cell if r["test_side"] == group]
                        rows.append({"comparison": comparison, "alpha": alpha,
                                     "window_start": window[0], "window_end": window[1],
                                     "metric": metric, "group": group,
                                     **summarize([float(r[field]) for r in chosen], repeats, SEED+counter)})
                        counter += 1
    recovery = []
    for alpha in ALPHAS:
        for window in WINDOWS:
            cell = [r for r in blocked if float(r["alpha"]) == alpha and
                    (int(r["window_start"]), int(r["window_end"])) == window]
            for group in ("overall", "text_side", "image_side"):
                chosen = cell if group == "overall" else [r for r in cell if r["test_side"] == group]
                flipped = sum(bool(r["steering_flipped"]) for r in chosen)
                restored = sum(bool(r["restored_clean_label"]) for r in chosen)
                recovery.append({"alpha": alpha, "window_start": window[0], "window_end": window[1],
                    "group": group, "case_count": len(chosen), "steering_flip_count": flipped,
                    "restored_clean_label_count": restored,
                    "restoration_rate_given_flip": restored/flipped if flipped else ""})
    audit_bad = []
    for row in blocked:
        expected_layers = list(range(int(row["window_start"]), int(row["window_end"])+1))
        diagnostics = row["attention_diagnostics"]
        ok = diagnostics["layers"] == expected_layers and all(
            detail["max_blocked_weight"] == 0.0 and detail["max_row_sum_error"] <= .01 and
            detail["finite"] and detail["hook_call_count"] == 1
            for detail in diagnostics["by_layer"].values())
        ok = ok and int(row["steering_diagnostics"]["steering_applied_count"]) == 1
        if not ok: audit_bad.append((row["case_id"], row["alpha"], row["window_start"]))
    if audit_bad: raise RuntimeError(f"Steered-block hook audit failed: {audit_bad[:5]}")
    _csv(root / "tables/effects.csv", rows); _csv(root / "tables/restoration_counts.csv", recovery)
    figures = _plots(root, rows)
    result = {"status": "complete", "case_count": len(manifest), "trial_count": len(trials),
              "side_counts": dict(Counter(r["test_side"] for r in manifest)),
              "attention_audit_rows": len(blocked), "bootstrap_repeats": repeats, "figures": figures}
    atomic_json(root / "tables/summary.json", result); return result


def _plots(root: Path, rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = root / "figures"; output.mkdir(parents=True, exist_ok=True); paths = []
    for metric in ("token_change_rate", "logit_change_diff"):
        selected = [r for r in rows if r["comparison"] == "SB_vs_S" and r["metric"] == metric and r["group"] == "overall"]
        fig, ax = plt.subplots(figsize=(8, 4.6)); x = np.arange(len(WINDOWS))
        for alpha in ALPHAS:
            cell = sorted((r for r in selected if float(r["alpha"]) == alpha), key=lambda r: int(r["window_start"]))
            mean=np.asarray([r["mean"] for r in cell]); low=np.asarray([r["ci95_low"] for r in cell]); high=np.asarray([r["ci95_high"] for r in cell])
            ax.errorbar(x, mean, yerr=np.vstack((mean-low, high-mean)), marker="o", capsize=3, label=f"alpha={alpha:g}")
        if metric != "token_change_rate": ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(x, [f"L{a}–{b}" for a,b in WINDOWS]); ax.set_xlabel("SAC→PANL blocked window")
        ax.set_ylabel(f"SB vs S {metric.replace('_',' ')}"); ax.grid(axis="y", alpha=.2); ax.legend(frameon=False)
        fig.tight_layout(); path=output/f"sb_vs_s_{metric}.png"; fig.savefig(path,dpi=220); plt.close(fig); paths.append(str(path))
    return paths
