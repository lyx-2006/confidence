from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import ALPHAS, BOOTSTRAP_REPEATS, CANONICAL_ANSWERS, DIRECTIONS, LAYERS, POSITIONS, RESULTS_ROOT, SEED, SMOKE_ALPHAS, SMOKE_BOOTSTRAP_REPEATS, SMOKE_DIRECTIONS, SMOKE_LAYERS
from .io_utils import atomic_csv, atomic_json, atomic_text, canonical_hash, load_jsonl, sha256_file
from .run import paired_trial_key


def bh_fdr(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float); order = np.argsort(values); output = np.empty(len(values), dtype=float); running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = order[reverse_rank]; rank = reverse_rank + 1; running = min(running, float(values[index]) * len(values) / rank); output[index] = min(1.0, running)
    return output.tolist()


def family_dose_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_family: dict[str, dict[float, float]] = defaultdict(dict)
    for row in rows:
        family = str(row["family_id"]); alpha = float(row["alpha"])
        if alpha in by_family[family]: raise ValueError(f"Duplicate alpha for family {family}")
        by_family[family][alpha] = float(row["delta_soft_sa"])
    output = {}
    for family, values in by_family.items():
        if set(values) not in ({-10.0, -2.0, 0.0, 2.0, 10.0}, {-2.0, 0.0, 2.0}): raise ValueError(f"Incomplete dose grid for {family}: {sorted(values)}")
        x = np.asarray(sorted(values)); y = np.asarray([values[value] for value in x])
        output[family] = {"slope": float(np.polyfit(x, y, 1)[0]), "symmetric_effect_2": float((values[2.0] - values[-2.0]) / 2), "symmetric_effect_10": float((values[10.0] - values[-10.0]) / 2) if 10.0 in values else np.nan}
    return output


class BootstrapDesign:
    def __init__(self, test_rows: Sequence[dict[str, Any]], *, repeats: int, seed: int):
        self.repeats = repeats; self.rng = np.random.default_rng(seed); self.rows = {str(row["family_id"]): row for row in test_rows}
        self.by_answer = {answer: sorted(str(row["family_id"]) for row in test_rows if row["test_answer"] == answer) for answer in CANONICAL_ANSWERS}
        self.group_families: dict[tuple[str, str | None], list[str]] = {}; self.draws: dict[tuple[str, str | None], np.ndarray] = {}
        for answer, families in self.by_answer.items():
            for side in (None, "high_text", "high_image"):
                selected = [family for family in families if side is None or self.rows[family]["test_side"] == side]
                if selected:
                    key = (answer, side); self.group_families[key] = selected; self.draws[key] = self.rng.integers(0, len(selected), size=(repeats, len(selected)))
        all_families = sorted(self.rows)
        for side in (None, "high_text", "high_image"):
            selected = [family for family in all_families if side is None or self.rows[family]["test_side"] == side]
            self.group_families[("__all__", side)] = selected; self.draws[("__all__", side)] = self.rng.integers(0, len(selected), size=(repeats, len(selected)))

    def aggregate(self, values: dict[str, float], *, mode: str, side: str | None = None) -> tuple[float, float, float, float, np.ndarray, int, int]:
        confirmatory = [answer for answer in CANONICAL_ANSWERS if self.by_answer.get(answer) and answer != "blue"]
        answer_observed = []; answer_boot = []; used = []
        for answer in confirmatory:
            base = self.group_families.get((answer, side), []); families = [family for family in base if family in values]
            if not families: continue
            if families != base: raise ValueError("Bootstrap cell is missing frozen families")
            vector = np.asarray([values[family] for family in families]); indices = self.draws[answer, side]
            answer_observed.append(float(vector.mean())); answer_boot.append(vector[indices].mean(axis=1)); used.extend(families)
        if mode == "answer_equal":
            if not answer_observed: return (np.nan, np.nan, np.nan, np.nan, np.full(self.repeats, np.nan), 0, 0)
            observed = float(np.mean(answer_observed)); boot = np.stack(answer_boot).mean(axis=0); answer_count = len(answer_observed)
        elif mode == "family_micro":
            base = self.group_families[("__all__", side)]; families = [family for family in base if family in values]
            if families != base: raise ValueError("Bootstrap cell is missing frozen families")
            vector = np.asarray([values[family] for family in families]); boot = vector[self.draws["__all__", side]].mean(axis=1)
            observed = float(vector.mean()); used = families; answer_count = len({self.rows[family]["test_answer"] for family in families})
        else: raise ValueError(mode)
        low, high = np.percentile(boot, [2.5, 97.5])
        return observed, float(np.std(boot, ddof=1)), float(low), float(high), boot, answer_count, len(set(used))


def _value_map(rows: Sequence[dict[str, Any]], field: str = "delta_soft_sa") -> dict[str, float]:
    output = {}
    for row in rows:
        family = str(row["family_id"])
        if family in output: raise ValueError(f"Duplicate family value: {family}")
        output[family] = float(row[field])
    return output


def build_delta_table(trials: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> list[dict[str, Any]]:
    bootstrap = BootstrapDesign(test, repeats=repeats, seed=seed); output = []
    positions = [position for position in POSITIONS if any(row.get("position", "P1_LAT") == position for row in trials)]
    for position in positions:
        for direction in DIRECTIONS:
            if not any(row.get("position", "P1_LAT") == position and row["direction"] == direction for row in trials): continue
            for layer in sorted({int(row["layer"]) for row in trials}):
                dose_rows = [row for row in trials if row.get("position", "P1_LAT") == position and row["direction"] == direction and int(row["layer"]) == layer]
                complete_dose = len({float(row["alpha"]) for row in dose_rows}) in (3, 5)
                dose = family_dose_metrics(dose_rows) if complete_dose else {}
                slope = bootstrap.aggregate({family: value["slope"] for family, value in dose.items()}, mode="answer_equal") if dose else (np.nan,) * 7
                symmetric2 = bootstrap.aggregate({family: value["symmetric_effect_2"] for family, value in dose.items()}, mode="answer_equal") if dose else (np.nan,) * 7
                finite10 = dose and all(math.isfinite(value["symmetric_effect_10"]) for value in dose.values())
                symmetric10 = bootstrap.aggregate({family: value["symmetric_effect_10"] for family, value in dose.items()}, mode="answer_equal") if finite10 else (np.nan,) * 7
                for alpha in sorted({float(row["alpha"]) for row in dose_rows}):
                    selected = [row for row in dose_rows if float(row["alpha"]) == alpha]; values = _value_map(selected); total = bootstrap.aggregate(values, mode="answer_equal"); micro = bootstrap.aggregate(values, mode="family_micro")
                    row: dict[str, Any] = {"position": position, "direction": direction, "layer": layer, "alpha": alpha, "total_delta_sa_answer_equal": total[0], "total_sem": total[1], "total_ci_low": total[2], "total_ci_high": total[3], "family_micro_delta_sa": micro[0], "symmetric_effect_10": symmetric10[0], "symmetric_effect_2": symmetric2[0], "slope": slope[0], "confirmatory_answer_count": total[5], "family_count": total[6]}
                    for answer in CANONICAL_ANSWERS:
                        family_ids = bootstrap.by_answer.get(answer, []); answer_values = [values[family] for family in family_ids if family in values]
                        row[f"{answer}_delta_sa"] = float(np.mean(answer_values)) if answer_values else np.nan; row[f"{answer}_family_count"] = len(answer_values)
                    output.append(row)
    return output


def validate_trial_pairing(trials: Sequence[dict[str, Any]]) -> None:
    keys = {position: {paired_trial_key(row) for row in trials if row["position"] == position} for position in POSITIONS}
    if keys[POSITIONS[0]] != keys[POSITIONS[1]]: raise ValueError("LAT/PANL trial keys are not paired")


def _two_sided_p(boot: np.ndarray) -> float:
    return float(min(1.0, 2 * min((1 + np.sum(boot <= 0)) / (len(boot) + 1), (1 + np.sum(boot >= 0)) / (len(boot) + 1))))


def peak_and_contrast_tables(trials: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]], *, repeats: int, seed: int, smoke: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bootstrap = BootstrapDesign(test, repeats=repeats, seed=seed); layers = sorted({int(row["layer"]) for row in trials}); field = "symmetric_effect_2" if smoke else "symmetric_effect_10"
    observed: dict[tuple[str, int], float] = {}; boots: dict[tuple[str, int], np.ndarray] = {}
    for position in POSITIONS:
        for layer in layers:
            selected = [row for row in trials if row["position"] == position and row["direction"] == "matched_loao" and int(row["layer"]) == layer]; metric = family_dose_metrics(selected); aggregate = bootstrap.aggregate({family: value[field] for family, value in metric.items()}, mode="answer_equal"); observed[position, layer] = aggregate[0]; boots[position, layer] = aggregate[4]
    peaks = []
    for position in POSITIONS:
        matrix = np.stack([boots[position, layer] for layer in layers], axis=1); winners = np.argmax(matrix, axis=1); counts = Counter(winners.tolist()); point_index = int(np.argmax([observed[position, layer] for layer in layers])); point_layer = layers[point_index]
        frequencies = np.asarray([counts.get(index, 0) / repeats for index in range(len(layers))]); weighted = float(np.dot(np.asarray(layers), frequencies)); window = {point_layer}; neighbor_rows = {}
        for index in (point_index - 1, point_index + 1):
            if 0 <= index < len(layers):
                neighbor = layers[index]; difference = boots[position, point_layer] - boots[position, neighbor]; low, high = np.percentile(difference, [2.5, 97.5]); neighbor_rows[neighbor] = (float(observed[position, point_layer] - observed[position, neighbor]), float(low), float(high))
                if low <= 0 <= high: window.add(neighbor)
        for index, layer in enumerate(layers):
            contrast = neighbor_rows.get(layer, (np.nan, np.nan, np.nan)); peaks.append({"position": position, "layer": layer, "effect_endpoint_alpha": 2 if smoke else 10, "observed_effect": observed[position, layer], "peak_bootstrap_frequency": frequencies[index], "point_peak_layer": point_layer, "most_common_peak_layer": layers[int(np.argmax(frequencies))], "weighted_peak_center": weighted, "sensitive_window": f"L{min(window)}–L{max(window)}" if len(window) > 1 else f"L{point_layer}", "point_peak_minus_this_layer": contrast[0], "contrast_ci_low": contrast[1], "contrast_ci_high": contrast[2]})
    contrasts = []
    for layer in layers:
        difference = boots["P1_PANL", layer] - boots["P1_LAT", layer]; low, high = np.percentile(difference, [2.5, 97.5]); contrasts.append({"layer": layer, "metric": f"symmetric_effect_{2 if smoke else 10}", "panl_minus_lat": observed["P1_PANL", layer] - observed["P1_LAT", layer], "ci_low": float(low), "ci_high": float(high), "p_value": _two_sided_p(difference), "q_value": np.nan, "valid_bootstrap_repeats": repeats})
    for row, value in zip(contrasts, bh_fdr([row["p_value"] for row in contrasts])): row["q_value"] = value
    return peaks, contrasts


def _diagnostic_aggregates(trials: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> list[dict[str, Any]]:
    bootstrap = BootstrapDesign(test, repeats=repeats, seed=seed); output = []
    for position in POSITIONS:
        for direction in sorted({row["direction"] for row in trials}):
            for layer in sorted({int(row["layer"]) for row in trials}):
                for alpha in sorted({float(row["alpha"]) for row in trials}):
                    selected = [row for row in trials if row["position"] == position and row["direction"] == direction and int(row["layer"]) == layer and float(row["alpha"]) == alpha]
                    for field in ("margin_change", "hard_class_changed"):
                        for side in ("high_text", "high_image"):
                            aggregate = bootstrap.aggregate(_value_map(selected, field), mode="answer_equal", side=side); output.append({"position": position, "direction": direction, "layer": layer, "alpha": alpha, "metric": field, "side": side, "answer_equal_value": aggregate[0], "ci_low": aggregate[2], "ci_high": aggregate[3], "family_count": aggregate[6]})
    return output


def _split_audit(root: Path) -> list[dict[str, Any]]:
    gate = json.loads((root / "progress" / "split_gate.json").read_text()); distribution = load_jsonl(root / "artifacts" / "manifests" / "construction_distribution.jsonl"); leaks = {int(row["fold"]): row for row in gate["folds"]}
    return [{**row, **{key: leaks[int(row["fold"])][key] for key in ("family_leakage_count", "item_leakage_count", "image_hash_leakage_count", "case_leakage_count")}} for row in distribution]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def _plot_steering(root: Path, table_path: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    rows = [row for row in _read_csv(table_path) if row["direction"] == "matched_loao"]; alphas = sorted({float(row["alpha"]) for row in rows if float(row["alpha"]) != 0}); palette = dict(zip(alphas, ("#2166ac", "#67a9cf", "#ef8a62", "#b2182b")[-len(alphas):])); bounds = [abs(float(row[key])) for row in rows for key in ("total_ci_low", "total_ci_high") if row[key] and math.isfinite(float(row[key]))]; limit = max(bounds, default=1e-6) * 1.12; paths = []
    for position, filename in (("P1_LAT", "figure1_lat_steering.png"), ("P1_PANL", "figure2_panl_steering.png")):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for alpha in alphas:
            data = sorted([row for row in rows if row["position"] == position and float(row["alpha"]) == alpha], key=lambda row: int(row["layer"])); mean = np.asarray([float(row["total_delta_sa_answer_equal"]) for row in data]); low = np.asarray([float(row["total_ci_low"]) for row in data]); high = np.asarray([float(row["total_ci_high"]) for row in data]); ax.errorbar([int(row["layer"]) for row in data], mean, yerr=np.vstack([mean-low, high-mean]), marker="o", capsize=3, color=palette[alpha], label=f"alpha={alpha:+g}")
        zero = sorted([row for row in rows if row["position"] == position and float(row["alpha"]) == 0], key=lambda row: int(row["layer"])); ax.plot([int(row["layer"]) for row in zero], [0] * len(zero), "o", color="#888", label="alpha=0 parity"); ax.set_ylim(-limit, limit); ax.axhline(0, color="black", lw=.8); ax.set_xticks(sorted({int(row["layer"]) for row in rows})); ax.set_xlabel("Zero-based decoder layer"); ax.set_ylabel("Answer-equal mean delta final soft SA"); ax.set_title(f"{position} answer-matched steering"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2); fig.tight_layout(); path = root / "figures" / filename; path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=300); plt.close(fig); paths.append(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for position, style in (("P1_LAT", "-"), ("P1_PANL", "--")):
        for alpha in alphas:
            data = sorted([row for row in rows if row["position"] == position and float(row["alpha"]) == alpha], key=lambda row: int(row["layer"])); ax.plot([int(row["layer"]) for row in data], [float(row["total_delta_sa_answer_equal"]) for row in data], linestyle=style, marker="o", color=palette[alpha], label=f"{position} alpha={alpha:+g}")
    ax.axhline(0, color="#777", lw=.8, label="alpha=0 parity"); ax.set_xlabel("Zero-based decoder layer"); ax.set_ylabel("Answer-equal mean delta final soft SA"); ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=.2); fig.tight_layout(); path = root / "figures" / "figure3_lat_panl_steering_overlay.png"; fig.savefig(path, dpi=300); plt.close(fig); paths.append(path); return paths


def _plot_probe(root: Path, table_path: Path) -> Path:
    import matplotlib.pyplot as plt
    rows = _read_csv(table_path); fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, metric in zip(axes, ("r2", "pearson", "spearman")):
        for position, color, marker in (("P1_LAT", "#2166ac", "o"), ("P1_PANL", "#b2182b", "s")):
            data = sorted([row for row in rows if row["position"] == position], key=lambda row: int(row["layer"])); mean = np.asarray([float(row[metric]) for row in data]); low = np.asarray([float(row[f"{metric}_ci_low"]) for row in data]); high = np.asarray([float(row[f"{metric}_ci_high"]) for row in data]); ax.errorbar([int(row["layer"]) for row in data], mean, yerr=np.vstack([mean-low, high-mean]), color=color, marker=marker, capsize=3, label=position)
        if metric == "r2": ax.axhline(0, color="black", lw=.8)
        ax.set_title(metric.upper() if metric == "r2" else metric.capitalize()); ax.set_xlabel("Zero-based decoder layer"); ax.grid(axis="y", alpha=.2)
    axes[0].legend(); fig.tight_layout(); path = root / "figures" / "figure4_lat_panl_probe.png"; path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=300); plt.close(fig); return path


def _readme() -> str:
    return """# LAT–PANL comparison tables

- `table1_lat_panl_steering.csv` is the sole source for Figures 1–3. Blue is exploratory and excluded from confirmatory answer-equal totals.
- `table2_lat_panl_probe.csv` is the sole source for Figure 4.
- Control, side, margin, hard-change, peak and leakage audits are under `artifacts/diagnostics/`.
"""


def analyze(*, output_root: Path = RESULTS_ROOT, smoke: bool = False, resume: bool = False, repeats: int | None = None) -> dict[str, Any]:
    root = Path(output_root); trial_path = root / "artifacts" / "diagnostics" / "steering_trials.jsonl"; trials = [row for row in load_jsonl(trial_path) if row.get("status") == "completed"]; test = load_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl")
    layers = SMOKE_LAYERS if smoke else LAYERS; alphas = SMOKE_ALPHAS if smoke else ALPHAS; directions = SMOKE_DIRECTIONS if smoke else DIRECTIONS; expected = len(test) * len(POSITIONS) * len(directions) * len(layers) * len(alphas)
    if len(trials) != expected or len({(row["case_id"], row["position"], row["direction"], int(row["layer"]), float(row["alpha"])) for row in trials}) != expected: raise ValueError(f"Analysis trial grid incomplete: {len(trials)}/{expected}")
    validate_trial_pairing(trials); zero = [row for row in trials if float(row["alpha"]) == 0]
    if not zero or any(not row.get("alpha_zero_parity", {}).get("passed") for row in zero): raise ValueError("Alpha-zero parity gate failed")
    probe_progress = json.loads((root / "progress" / "probe_progress.json").read_text())
    if probe_progress.get("status") != "complete": raise ValueError("Analysis requires a complete probe")
    repeats = int(repeats or (SMOKE_BOOTSTRAP_REPEATS if smoke else BOOTSTRAP_REPEATS)); config = {"smoke_only": smoke, "trial_sha256": sha256_file(trial_path), "probe_fingerprint": probe_progress["config_fingerprint"], "bootstrap_repeats": repeats, "seed": SEED}; config["fingerprint"] = canonical_hash(config)
    analysis_path = root / "progress" / "analysis_progress.json"; required = [root / "tables" / "table1_lat_panl_steering.csv", root / "tables" / "table2_lat_panl_probe.csv"] + [root / "figures" / name for name in ("figure1_lat_steering.png", "figure2_panl_steering.png", "figure3_lat_panl_steering_overlay.png", "figure4_lat_panl_probe.png")]
    if analysis_path.exists():
        previous = json.loads(analysis_path.read_text())
        if previous.get("config_fingerprint") != config["fingerprint"]: raise ValueError("Analysis resume fingerprint mismatch")
        if resume and previous.get("status") == "complete" and all(path.is_file() and path.stat().st_size for path in required): return {**previous, "resumed_noop": True}
        if not resume: raise FileExistsError("Analysis exists; use --resume")
    table1 = build_delta_table(trials, test, repeats=repeats, seed=SEED); table1_path = root / "tables" / "table1_lat_panl_steering.csv"; atomic_csv(table1_path, table1); peaks, contrasts = peak_and_contrast_tables(trials, test, repeats=repeats, seed=SEED, smoke=smoke)
    atomic_csv(root / "artifacts" / "diagnostics" / "peak_layer_bootstrap.csv", peaks); atomic_csv(root / "artifacts" / "diagnostics" / "position_layer_contrasts.csv", contrasts); atomic_csv(root / "artifacts" / "diagnostics" / "steering_side_margin_hard_change.csv", _diagnostic_aggregates(trials, test, repeats=repeats, seed=SEED)); atomic_csv(root / "artifacts" / "diagnostics" / "split_and_selection_audit.csv", _split_audit(root)); atomic_text(root / "tables" / "README.md", _readme())
    figures = _plot_steering(root, table1_path); figures.append(_plot_probe(root, root / "tables" / "table2_lat_panl_probe.csv"))
    if len(figures) != 4 or any(not path.is_file() or not path.stat().st_size for path in figures): raise RuntimeError("Figure completion gate failed")
    windows = {row["position"]: row["sensitive_window"] for row in peaks if int(row["layer"]) == int(row["point_peak_layer"])}; atomic_text(root / "summary.md", f"# LAT–PANL Answer-matched Comparison\n\n- smoke_only: `{str(smoke).lower()}`\n- LAT sensitive window: {windows.get('P1_LAT')}\n- PANL sensitive window: {windows.get('P1_PANL')}\n\nThe experiment compares causal steering and linear readability. It does not identify an attention-head pathway, prove mediation, or establish that the decoded state is difficulty or confidence.\n")
    if not smoke:
        required_fields = ["total_delta_sa_answer_equal", "total_sem", "total_ci_low", "total_ci_high", "family_micro_delta_sa", "symmetric_effect_10", "symmetric_effect_2", "slope"] + [f"{answer}_delta_sa" for answer in CANONICAL_ANSWERS]
        if any(not math.isfinite(float(row[field])) for row in table1 for field in required_fields): raise ValueError("Formal steering table contains non-finite required values")
    result = {"status": "complete", "smoke_only": smoke, "trial_count": len(trials), "expected_trial_count": expected, "alpha_zero_count": len(zero), "alpha_zero_parity": "passed", "paired_positions": "passed", "bootstrap_repeats": repeats, "probe_cells": probe_progress["cell_count"], "tables": 2, "figures": 4, "config_fingerprint": config["fingerprint"], "resumed_noop": False}; atomic_json(analysis_path, result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--bootstrap", type=int); args = parser.parse_args(argv); root = Path(args.output_root) if args.output_root else RESULTS_ROOT
    if args.smoke and not args.output_root: parser.error("--smoke requires an explicit output root")
    print(json.dumps(analyze(output_root=root, smoke=args.smoke, resume=args.resume, repeats=args.bootstrap), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
