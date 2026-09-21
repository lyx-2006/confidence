from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dp_SA.io_utils import atomic_json, load_jsonl
from .config import BOOTSTRAP_REPEATS, CONDITIONS, PAIRS, SEED, default_output


def summarize_values(values: Sequence[float], repeats: int, seed: int = SEED) -> dict[str, float]:
    vector = np.asarray(values, dtype=float)
    if not len(vector) or not np.isfinite(vector).all(): raise ValueError("Cannot summarize empty/non-finite values")
    rng = np.random.default_rng(seed)
    boot = vector[rng.integers(0, len(vector), size=(repeats, len(vector)))].mean(1)
    low, high = np.quantile(boot, [.025, .975])
    return {"mean": float(vector.mean()), "ci95_low": float(low), "ci95_high": float(high), "count": len(vector)}


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _plots(root: Path, effects: list[dict[str, Any]], paired: list[dict[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = root / "figures"; output.mkdir(parents=True, exist_ok=True); paths = []
    for metric in ("token_change_rate", "logit_change_diff"):
        rows = [r for r in effects if r["metric"] == metric and r["group"] == "overall"]
        windows = sorted({(r["window_start"], r["window_end"]) for r in rows}); x = np.arange(len(windows))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for condition in CONDITIONS:
            chosen = sorted((r for r in rows if r["condition"] == condition), key=lambda r: r["window_start"])
            mean = np.asarray([r["mean"] for r in chosen]); low = np.asarray([r["ci95_low"] for r in chosen]); high = np.asarray([r["ci95_high"] for r in chosen])
            ax.errorbar(x, mean, yerr=np.vstack((mean-low, high-mean)), marker="o", capsize=3, label=condition)
        if metric != "token_change_rate": ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(x, [f"L{a}–{b}" for a, b in windows]); ax.set_xlabel("Inclusive decoder-layer window")
        ax.set_ylabel(metric.replace("_", " ")); ax.grid(axis="y", alpha=.2); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); path = output / f"effects_{metric}.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(str(path))
        rows = [r for r in paired if r["metric"] == metric and r["group"] == "overall"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for source in PAIRS:
            chosen = sorted((r for r in rows if r["source"] == source), key=lambda r: r["window_start"])
            mean = np.asarray([r["mean"] for r in chosen]); low = np.asarray([r["ci95_low"] for r in chosen]); high = np.asarray([r["ci95_high"] for r in chosen])
            ax.errorbar(x, mean, yerr=np.vstack((mean-low, high-mean)), marker="o", capsize=3, label=f"{source} − {source}+1")
        ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(x, [f"L{a}–{b}" for a, b in windows])
        ax.set_xlabel("Inclusive decoder-layer window"); ax.set_ylabel(f"Paired {metric.replace('_', ' ')}")
        ax.grid(axis="y", alpha=.2); ax.legend(frameon=False); fig.tight_layout()
        path = output / f"paired_{metric}.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(str(path))
    return paths


def analyze(*, output_root: Path, repeats: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    root = Path(output_root).resolve(); config = json.loads((root / "run_config.json").read_text())
    manifest = load_jsonl(root / "artifacts/manifests/test_manifest.jsonl")
    trials = load_jsonl(root / "artifacts/trials.jsonl")
    blocked = [r for r in trials if r["condition"] in CONDITIONS]
    windows = [tuple(x) for x in config["windows"]]
    if len(blocked) != len(manifest) * len(windows) * len(CONDITIONS): raise RuntimeError("Incomplete attention grid")
    metrics = ("token_change_rate", "logit_change_diff")
    effects, paired = [], []
    for w_i, window in enumerate(windows):
        cell = [r for r in blocked if (r["window_start"], r["window_end"]) == window]
        for condition in CONDITIONS:
            rows = [r for r in cell if r["condition"] == condition]
            for metric in metrics:
                for group in ("overall", "text_side", "image_side"):
                    chosen = rows if group == "overall" else [r for r in rows if r["test_side"] == group]
                    effects.append({"window_start": window[0], "window_end": window[1],
                                    "condition": condition, "metric": metric, "group": group,
                                    **summarize_values([float(r[metric]) for r in chosen], repeats, SEED+w_i)})
        lookup = {(r["case_id"], r["condition"]): r for r in cell}
        for source, (main, control) in PAIRS.items():
            for metric in metrics:
                for group in ("overall", "text_side", "image_side"):
                    rows = [r for r in cell if r["condition"] == main and (group == "overall" or r["test_side"] == group)]
                    values = [float(r[metric]) - float(lookup[(r["case_id"], control)][metric]) for r in rows]
                    paired.append({"window_start": window[0], "window_end": window[1], "source": source,
                                   "comparison": f"{main}_minus_{control}", "metric": metric, "group": group,
                                   **summarize_values(values, repeats, SEED+100+w_i)})
    _csv(root / "tables/effects.csv", effects); _csv(root / "tables/paired_main_minus_control.csv", paired)
    figures = _plots(root, effects, paired)
    summary = {"status": "complete", "case_count": len(manifest), "trial_count": len(trials),
               "side_counts": dict(Counter(r["test_side"] for r in manifest)), "bootstrap_repeats": repeats}
    summary["figures"] = figures
    atomic_json(root / "tables/summary.json", summary); return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze five-class attention blocking")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--steered-block", action="store_true")
    args = parser.parse_args(argv)
    if args.steered_block:
        from .steered_block import default_output as enhanced_output
        from .steered_block_analysis import analyze as enhanced_analyze
        root = args.output_root or enhanced_output(False)
        result = enhanced_analyze(output_root=root, repeats=args.bootstrap_repeats)
    else:
        result = analyze(output_root=args.output_root or default_output(False), repeats=args.bootstrap_repeats)
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
