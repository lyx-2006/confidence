from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from dp_SA.io_utils import atomic_json, load_jsonl


SEED = 42
REPEATS = 2000
GROUPS = ("overall", "image_side", "text_side")


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def family_mean_ci(
    rows: Sequence[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *, repeats: int = REPEATS,
    seed: int = SEED,
) -> tuple[float, float, float, int]:
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_family[str(row["item_id"])].append(float(value(row)))
    families = sorted(by_family)
    if not families:
        return math.nan, math.nan, math.nan, 0
    family_values = np.asarray([np.mean(by_family[name]) for name in families], dtype=float)
    if not np.isfinite(family_values).all():
        raise ValueError("Non-finite family statistic")
    draws = np.random.default_rng(seed).integers(0, len(families), size=(repeats, len(families)))
    boot = family_values[draws].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(family_values.mean()), float(low), float(high), len(families)


def _group(rows: Sequence[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    return list(rows) if group == "overall" else [row for row in rows if row["test_side"] == group]


def analyze(predictions: Path, output_root: Path, *, repeats: int = REPEATS) -> dict[str, Any]:
    rows = [row for row in load_jsonl(predictions) if row.get("status") == "completed"]
    if not rows:
        raise ValueError(f"No completed steering predictions: {predictions}")
    effect_rows: list[dict[str, Any]] = []
    dose_rows: list[dict[str, Any]] = []
    modes = sorted({row["direction_mode"] for row in rows})
    positions = sorted({row["position"] for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})
    alphas = sorted({float(row["alpha"]) for row in rows})
    counter = 0
    for mode in modes:
        for position in positions:
            for layer in layers:
                base = [row for row in rows if row["direction_mode"] == mode and row["position"] == position and int(row["layer"]) == layer]
                for alpha in alphas:
                    cell = [row for row in base if float(row["alpha"]) == alpha]
                    for group in GROUPS:
                        selected = _group(cell, group)
                        for metric in ("delta_soft_sa", "delta_hard_midpoint", "hard_class_changed"):
                            mean, low, high, families = family_mean_ci(
                                selected, lambda row, field=metric: float(row[field]),
                                repeats=repeats, seed=SEED + counter,
                            )
                            effect_rows.append({
                                "direction_mode": mode, "position": position, "layer": layer,
                                "alpha": alpha, "group": group, "metric": metric,
                                "mean": mean, "ci95_low": low, "ci95_high": high,
                                "family_count": families, "bootstrap_repeats": repeats,
                            })
                            counter += 1
                by_family: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
                for row in base:
                    by_family[str(row["item_id"])][float(row["alpha"])] = row
                for magnitude in (2.0, 5.0):
                    synthesized: list[dict[str, Any]] = []
                    for family, values in by_family.items():
                        if magnitude not in values or -magnitude not in values:
                            raise ValueError(f"Incomplete +/-{magnitude:g} steering pair for {family}")
                        plus, minus = values[magnitude], values[-magnitude]
                        synthesized.append({
                            "item_id": family, "test_side": plus["test_side"],
                            "symmetric_soft_sa": (float(plus["delta_soft_sa"]) - float(minus["delta_soft_sa"])) / 2,
                            "asymmetry_soft_sa": float(plus["delta_soft_sa"]) + float(minus["delta_soft_sa"]),
                            "symmetric_hard_midpoint": (float(plus["delta_hard_midpoint"]) - float(minus["delta_hard_midpoint"])) / 2,
                        })
                    for group in GROUPS:
                        selected = _group(synthesized, group)
                        for metric in ("symmetric_soft_sa", "asymmetry_soft_sa", "symmetric_hard_midpoint"):
                            mean, low, high, families = family_mean_ci(
                                selected, lambda row, field=metric: float(row[field]),
                                repeats=repeats, seed=SEED + 100000 + counter,
                            )
                            dose_rows.append({
                                "direction_mode": mode, "position": position, "layer": layer,
                                "magnitude": magnitude, "group": group, "metric": metric,
                                "mean": mean, "ci95_low": low, "ci95_high": high,
                                "family_count": families, "bootstrap_repeats": repeats,
                            })
                            counter += 1
    tables = output_root / "tables"
    _atomic_csv(tables / "alpha_effects.csv", effect_rows)
    _atomic_csv(tables / "symmetric_effects.csv", dose_rows)
    parity = [row for row in rows if float(row["alpha"]) == 0.0]
    parity_summary = {
        "cell_count": len(parity),
        "max_abs_delta_soft_sa": max(abs(float(row["delta_soft_sa"])) for row in parity),
        "max_logit_abs_error": max(float(row["alpha_zero_logit_max_abs_error"]) for row in parity),
        "max_probability_abs_error": max(float(row["alpha_zero_probability_max_abs_error"]) for row in parity),
        "hook_applied_counts": sorted({int(row["hook_diagnostics"]["steering_applied_count"]) for row in parity}),
    }
    atomic_json(tables / "parity_summary.json", parity_summary)
    _plots(effect_rows, dose_rows, output_root / "figures")
    summary = {
        "status": "complete", "prediction_count": len(rows), "modes": modes,
        "positions": positions, "layers": layers, "alphas": alphas,
        "effect_rows": len(effect_rows), "symmetric_rows": len(dose_rows),
        "bootstrap_repeats": repeats, "parity": parity_summary,
    }
    atomic_json(output_root / "analysis_summary.json", summary)
    return summary


def _plots(effect_rows: list[dict[str, Any]], dose_rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    for mode in sorted({row["direction_mode"] for row in effect_rows}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for axis, position in zip(axes, ("LAT", "PANL", "CLE")):
            for alpha in (-5.0, -2.0, 2.0, 5.0):
                selected = sorted(
                    [row for row in effect_rows if row["direction_mode"] == mode and row["position"] == position
                     and row["group"] == "overall" and row["metric"] == "delta_soft_sa" and row["alpha"] == alpha],
                    key=lambda row: row["layer"],
                )
                axis.plot([row["layer"] for row in selected], [row["mean"] for row in selected], marker="o", label=f"α={alpha:g}")
            axis.axhline(0, color="black", lw=.8); axis.set_title(position); axis.set_xlabel("layer"); axis.grid(axis="y", alpha=.2)
        axes[0].set_ylabel("mean Δsoft-SA"); axes[-1].legend(fontsize=8)
        fig.suptitle(mode); fig.tight_layout()
        fig.savefig(output / f"{mode}_dose_response.png", dpi=220); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, position in zip(axes, ("LAT", "PANL", "CLE")):
        for mode in sorted({row["direction_mode"] for row in dose_rows}):
            selected = sorted(
                [row for row in dose_rows if row["direction_mode"] == mode and row["position"] == position
                 and row["group"] == "overall" and row["metric"] == "symmetric_soft_sa" and row["magnitude"] == 5.0],
                key=lambda row: row["layer"],
            )
            axis.plot([row["layer"] for row in selected], [row["mean"] for row in selected], marker="o", label=mode)
        axis.axhline(0, color="black", lw=.8); axis.set_title(position); axis.set_xlabel("layer"); axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("S5 symmetric Δsoft-SA"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "symmetric_effect_comparison.png", dpi=220); plt.close(fig)

