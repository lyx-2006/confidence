from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import ALPHAS, BOOTSTRAP_REPEATS, CANONICAL_ANSWERS, DIRECTIONS, LAYERS, RESULTS_ROOT, SEED, SMOKE_ALPHAS, SMOKE_BOOTSTRAP_REPEATS, SMOKE_LAYERS
from .io_utils import atomic_csv, atomic_json, atomic_text, canonical_hash, load_jsonl, sha256_file


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
        if set(values) != {-10.0, -2.0, 0.0, 2.0, 10.0} and len(values) != 3: raise ValueError(f"Incomplete dose grid for {family}")
        x = np.asarray(sorted(values)); y = np.asarray([values[value] for value in x])
        output[family] = {"slope": float(np.polyfit(x, y, 1)[0]), "minus10": float(values.get(-10.0, np.nan)), "minus2": float(values.get(-2.0, np.nan)), "plus2": float(values.get(2.0, np.nan)), "plus10": float(values.get(10.0, np.nan)), "symmetric_effect_2": float((values.get(2.0, np.nan) - values.get(-2.0, np.nan)) / 2), "symmetric_effect_10": float((values.get(10.0, np.nan) - values.get(-10.0, np.nan)) / 2)}
    return output


class BootstrapDesign:
    def __init__(self, test_rows: Sequence[dict[str, Any]], *, repeats: int, seed: int):
        self.repeats = repeats; self.rng = np.random.default_rng(seed); self.rows = {str(row["family_id"]): row for row in test_rows}
        self.by_answer = {answer: sorted(str(row["family_id"]) for row in test_rows if row["test_answer"] == answer) for answer in CANONICAL_ANSWERS}
        self.group_families = {}
        self.draws = {}
        for answer, families in self.by_answer.items():
            for side in (None, "high_text", "high_image"):
                selected = [family for family in families if side is None or self.rows[family]["test_side"] == side]
                if selected:
                    key = (answer, side); self.group_families[key] = selected; self.draws[key] = self.rng.integers(0, len(selected), size=(repeats, len(selected)))
        all_families = sorted(self.rows); self.group_families[("__all__", None)] = all_families; self.draws[("__all__", None)] = self.rng.integers(0, len(all_families), size=(repeats, len(all_families)))
        for side in ("high_text", "high_image"):
            selected = [family for family in all_families if self.rows[family]["test_side"] == side]
            self.group_families[("__all__", side)] = selected; self.draws[("__all__", side)] = self.rng.integers(0, len(selected), size=(repeats, len(selected)))

    def aggregate(self, values: dict[str, float], *, mode: str, side: str | None = None) -> tuple[float, float, float, float, np.ndarray, int, int]:
        confirmatory = [answer for answer in CANONICAL_ANSWERS if self.by_answer.get(answer) and any(self.rows[family].get("test_status") != "exploratory_sparse" for family in self.by_answer[answer])]
        answer_observed = []; answer_boot = []; used_families = []
        for answer in confirmatory:
            base_families = self.group_families.get((answer, side), [])
            families = [family for family in base_families if family in values]
            if not families: continue
            vector = np.asarray([values[family] for family in families], dtype=float)
            if families != base_families: raise ValueError("Bootstrap cell is missing frozen families")
            indices = self.draws[answer, side]
            answer_observed.append(float(vector.mean())); answer_boot.append(vector[indices].mean(axis=1)); used_families.extend(families)
        if mode == "answer_equal":
            if not answer_observed: return (np.nan,) * 4 + (np.full(self.repeats, np.nan), 0, 0)
            observed = float(np.mean(answer_observed)); boot = np.stack(answer_boot).mean(axis=0); answer_count = len(answer_observed)
        elif mode == "family_micro":
            base_families = self.group_families["__all__", side]; families = [family for family in base_families if family in values]
            if families != base_families: raise ValueError("Bootstrap cell is missing frozen families")
            vector = np.asarray([values[family] for family in families], dtype=float); indices = self.draws["__all__", side]
            observed = float(vector.mean()); boot = vector[indices].mean(axis=1); used_families = families; answer_count = len({self.rows[family]["test_answer"] for family in families})
        else: raise ValueError(mode)
        low, high = np.percentile(boot, [2.5, 97.5]); sem = float(np.std(boot, ddof=1))
        return observed, sem, float(low), float(high), boot, answer_count, len(set(used_families))


def _value_map(rows: Sequence[dict[str, Any]], field: str = "delta_soft_sa") -> dict[str, float]:
    output = {}
    for row in rows:
        family = str(row["family_id"])
        if family in output: raise ValueError(f"Duplicate family value: {family}")
        output[family] = float(row[field])
    return output


def build_delta_table(trials: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> list[dict[str, Any]]:
    bootstrap = BootstrapDesign(test, repeats=repeats, seed=seed); output = []
    present_directions = [direction for direction in DIRECTIONS if any(row["direction"] == direction for row in trials)]
    for direction in present_directions:
        for layer in sorted({int(row["layer"]) for row in trials}):
            for alpha in sorted({float(row["alpha"]) for row in trials}):
                selected = [row for row in trials if row["direction"] == direction and int(row["layer"]) == layer and float(row["alpha"]) == alpha]
                values = _value_map(selected); total = bootstrap.aggregate(values, mode="answer_equal"); micro = bootstrap.aggregate(values, mode="family_micro")
                row: dict[str, Any] = {"direction": direction, "layer": layer, "alpha": alpha, "total_delta_sa_answer_equal": total[0], "total_sem": total[1], "total_ci_low": total[2], "total_ci_high": total[3], "total_answer_count": total[5], "total_family_count": total[6], "family_micro_delta_sa": micro[0]}
                for answer in CANONICAL_ANSWERS:
                    family_ids = bootstrap.by_answer.get(answer, []); answer_values = [values[family] for family in family_ids if family in values]
                    row[f"{answer}_delta_sa"] = float(np.mean(answer_values)) if answer_values else np.nan; row[f"{answer}_family_count"] = len(answer_values)
                output.append(row)
    return output


def _dose_aggregation(metrics: dict[str, dict[str, float]], metric: str, bootstrap: BootstrapDesign, *, mode: str, side: str | None = None) -> tuple[float, float, float, float, np.ndarray, int, int]:
    finite = {family: values[metric] for family, values in metrics.items() if math.isfinite(values[metric])}
    if not finite:
        return np.nan, np.nan, np.nan, np.nan, np.full(bootstrap.repeats, np.nan), 0, 0
    return bootstrap.aggregate(finite, mode=mode, side=side)


def build_dose_table(trials: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> list[dict[str, Any]]:
    bootstrap = BootstrapDesign(test, repeats=repeats, seed=seed + 1000); output = []; metric_cache = {}
    aggregations = (("answer_equal_confirmatory", "answer_equal", None), ("family_micro_all", "family_micro", None), ("text_side_answer_equal", "answer_equal", "high_text"), ("image_side_answer_equal", "answer_equal", "high_image"))
    for direction in DIRECTIONS:
        for layer in sorted({int(row["layer"]) for row in trials}):
            selected = [row for row in trials if row["direction"] == direction and int(row["layer"]) == layer]
            metrics = family_dose_metrics(selected); metric_cache[direction, layer] = metrics
            for aggregation, mode, side in aggregations:
                slope = _dose_aggregation(metrics, "slope", bootstrap, mode=mode, side=side)
                sym2 = _dose_aggregation(metrics, "symmetric_effect_2", bootstrap, mode=mode, side=side); sym10 = _dose_aggregation(metrics, "symmetric_effect_10", bootstrap, mode=mode, side=side)
                endpoint = {name: _dose_aggregation(metrics, name, bootstrap, mode=mode, side=side)[0] for name in ("minus10", "minus2", "plus2", "plus10")}
                p_value = float((1 + np.sum(slope[4] <= 0)) / (len(slope[4]) + 1))
                row = {"direction": direction, "layer": layer, "aggregation": aggregation, "slope": slope[0], "slope_sem": slope[1], "slope_ci_low": slope[2], "slope_ci_high": slope[3], "minus10_mean": endpoint["minus10"], "minus2_mean": endpoint["minus2"], "plus2_mean": endpoint["plus2"], "plus10_mean": endpoint["plus10"], "symmetric_effect_2": sym2[0], "symmetric_effect_2_ci_low": sym2[2], "symmetric_effect_2_ci_high": sym2[3], "symmetric_effect_10": sym10[0], "symmetric_effect_10_ci_low": sym10[2], "symmetric_effect_10_ci_high": sym10[3], "matched_minus_shuffled_slope": np.nan, "matched_minus_shuffled_ci_low": np.nan, "matched_minus_shuffled_ci_high": np.nan, "matched_minus_unmatched_slope": np.nan, "matched_minus_unmatched_ci_low": np.nan, "matched_minus_unmatched_ci_high": np.nan, "p_value": p_value, "q_value": np.nan, "family_count": slope[6], "valid_bootstrap_repeats": repeats}
                if direction == "matched_loao":
                    for control, prefix in (("within_answer_shuffled", "matched_minus_shuffled"), ("unmatched_global", "matched_minus_unmatched")):
                        control_metrics = metric_cache.get((control, layer))
                        if control_metrics is None:
                            control_rows = [candidate for candidate in trials if candidate["direction"] == control and int(candidate["layer"]) == layer]
                            control_metrics = family_dose_metrics(control_rows); metric_cache[control, layer] = control_metrics
                        common = set(metrics) & set(control_metrics); differences = {family: metrics[family]["slope"] - control_metrics[family]["slope"] for family in common}
                        contrast = bootstrap.aggregate(differences, mode=mode, side=side)
                        row[prefix + "_slope"] = contrast[0]; row[prefix + "_ci_low"] = contrast[2]; row[prefix + "_ci_high"] = contrast[3]
                output.append(row)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(output): groups[row["direction"], row["aggregation"]].append(index)
    for indices in groups.values():
        q_values = bh_fdr([output[index]["p_value"] for index in indices])
        for index, q_value in zip(indices, q_values): output[index]["q_value"] = q_value
    return output


def build_split_audit(root: Path, test: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    distribution = load_jsonl(root / "artifacts" / "manifests" / "construction_distribution.jsonl"); gate = json.loads((root / "progress" / "split_gate.json").read_text())
    leak = {int(row["fold"]): row for row in gate["folds"]}; output = []
    for row in distribution:
        fold = int(row["fold"]); answer = row["answer"]; tests = [value for value in test if int(value["fold"]) == fold and value["test_answer"] == answer]
        count = len(tests); audit = leak[fold]; eligible_after = int(row["eligible_answer_count"]) - int(bool(row["eligible_for_direction"]))
        output.append({"fold": fold, "answer": answer, "construction_high_text_family_count": row["construction_high_text_family_count"], "construction_high_image_family_count": row["construction_high_image_family_count"], "eligible_for_direction": row["eligible_for_direction"], "test_family_count": count, "test_text_side_count": sum(value["test_side"] == "high_text" for value in tests), "test_image_side_count": sum(value["test_side"] == "high_image" for value in tests), "test_target_reached": sum(value["test_answer"] == answer for value in test) >= 10, "eligible_answer_count_after_loao": eligible_after, "family_leakage_count": audit["family_leakage_count"], "item_leakage_count": audit["item_leakage_count"], "image_hash_leakage_count": audit["image_hash_leakage_count"], "case_leakage_count": audit["case_leakage_count"], "exclusion_reason": "insufficient_global_answer_families" if not row["eligible_for_direction"] else ("blue_has_only_9_test_families" if answer == "blue" else "")})
    return output


def _plot(root: Path, delta_rows: Sequence[dict[str, Any]]) -> Path:
    import matplotlib.pyplot as plt
    rows = [row for row in delta_rows if row["direction"] == "matched_loao"]
    colors = {-10.0: "#2166ac", -2.0: "#67a9cf", 2.0: "#ef8a62", 10.0: "#b2182b"}; styles = {-10.0: ("-", "o"), -2.0: ("--", "s"), 2.0: ("--", "^"), 10.0: ("-", "D")}
    fig, ax = plt.subplots(figsize=(7.2, 4.8)); bounds = []
    for alpha in (-10.0, -2.0, 2.0, 10.0):
        data = sorted([row for row in rows if float(row["alpha"]) == alpha], key=lambda row: int(row["layer"])); means = np.asarray([row["total_delta_sa_answer_equal"] for row in data]); low = np.asarray([row["total_ci_low"] for row in data]); high = np.asarray([row["total_ci_high"] for row in data]); bounds.extend(abs(value) for value in np.concatenate([low, high]) if math.isfinite(value))
        ax.errorbar([row["layer"] for row in data], means, yerr=np.vstack([means - low, high - means]), color=colors[alpha], linestyle=styles[alpha][0], marker=styles[alpha][1], capsize=3, label=f"alpha={alpha:+g}")
    zero = sorted([row for row in rows if float(row["alpha"]) == 0], key=lambda row: int(row["layer"])); ax.plot([row["layer"] for row in zero], [0.0] * len(zero), "o", color="#888888", alpha=0.7, label="alpha=0 parity baseline")
    limit = max(bounds, default=1e-6) * 1.12; ax.set_ylim(-limit, limit); ax.axhline(0, color="black", linewidth=0.9); ax.set_xticks(sorted({int(row["layer"]) for row in rows})); ax.set_xlabel("Zero-based decoder layer"); ax.set_ylabel("Answer-equal mean delta final soft SA"); ax.set_title("P1_LAT answer-matched steering"); ax.grid(axis="y", alpha=0.2); ax.legend(fontsize=8, ncol=2, loc="best"); fig.tight_layout()
    path = root / "figures" / "P1_LAT_delta_sa_by_layer.png"; path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=300); plt.close(fig); return path


def _readme() -> str:
    return """# Answer-matched LAT steering 表格说明

- `delta_sa_by_layer_alpha_and_answer.csv`：逐方向、层、alpha 的逐颜色和颜色等权结果；Blue 保留但不进入确认性 total。
- `dose_response_and_controls.csv`：family-level slope、symmetric effects、matched-control paired contrasts及BH-FDR。
- `split_and_selection_audit.csv`：每折×颜色的construction门禁、test数量和四级泄漏审计。
- `NaN` 表示该统计量无定义或无数据，不表示零效应。
"""


def _summary(dose: Sequence[dict[str, Any]], *, smoke: bool) -> str:
    primary = [row for row in dose if row["direction"] == "matched_loao" and row["aggregation"] == "answer_equal_confirmatory"]
    passed = [row for row in primary if row["slope_ci_low"] > 0 and row["symmetric_effect_10_ci_low"] > 0 and row["matched_minus_shuffled_ci_low"] > 0 and row["q_value"] < 0.05]
    return f"""# Delayed-SA Answer-matched LAT Steering Summary

- smoke_only: `{str(smoke).lower()}`
- 通过主要统计门禁的 layer 数：{len(passed)}/{len(primary)}
- 主结果为达到冻结测试门槛颜色的 answer-equal macro aggregation；逐颜色结果为探索性。

Answer matching和leave-one-answer-out降低了方向由answer类别组成差异驱动的可能性，但由于本实验不探测steering后的answer representation，因此不能证明干预完全不影响潜在answer/commitment状态。

本实验也不能证明LAT已经形成完整verbal SA，不能建立LAT→PANL→SAC完整中介，也不支持声称所有12种颜色均成立。
"""


def analyze(*, output_root: Path = RESULTS_ROOT, smoke: bool = False, resume: bool = False, repeats: int | None = None) -> dict[str, Any]:
    root = Path(output_root); trial_path = root / "artifacts" / "diagnostics" / "steering_trials.jsonl"; trials = [row for row in load_jsonl(trial_path) if row.get("status") == "completed"]; test = load_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl")
    layers = SMOKE_LAYERS if smoke else LAYERS; alphas = SMOKE_ALPHAS if smoke else ALPHAS; expected = len(test) * len(DIRECTIONS) * len(layers) * len(alphas)
    if len(trials) != expected or len({(row["case_id"], row["direction"], int(row["layer"]), float(row["alpha"])) for row in trials}) != expected: raise ValueError(f"Analysis trial grid incomplete: {len(trials)}/{expected}")
    alpha_zero = [row for row in trials if float(row["alpha"]) == 0]
    if not alpha_zero or any(not row.get("alpha_zero_parity", {}).get("passed") for row in alpha_zero): raise ValueError("Alpha-zero parity gate failed")
    repeats = int(repeats or (SMOKE_BOOTSTRAP_REPEATS if smoke else BOOTSTRAP_REPEATS)); config = {"smoke_only": smoke, "trial_sha256": sha256_file(trial_path), "test_sha256": sha256_file(root / "artifacts" / "manifests" / "test_manifest.jsonl"), "bootstrap_repeats": repeats, "seed": SEED}; config["fingerprint"] = canonical_hash(config)
    progress = root / "progress" / "analysis_progress.json"
    if progress.exists():
        previous = json.loads(progress.read_text())
        if previous.get("config_fingerprint") != config["fingerprint"]: raise ValueError("Analysis resume fingerprint mismatch")
        required = [root / "tables" / name for name in ("delta_sa_by_layer_alpha_and_answer.csv", "dose_response_and_controls.csv", "split_and_selection_audit.csv", "README.md")] + [root / "figures" / "P1_LAT_delta_sa_by_layer.png", root / "summary.md"]
        if resume and previous.get("status") == "complete" and all(path.is_file() and path.stat().st_size for path in required): return {**previous, "resumed_noop": True}
        if not resume: raise FileExistsError("Analysis exists; use --resume")
    atomic_json(progress, {"status": "running", "config_fingerprint": config["fingerprint"], **config})
    delta = build_delta_table(trials, test, repeats=repeats, seed=SEED); dose = build_dose_table(trials, test, repeats=repeats, seed=SEED); split = build_split_audit(root, test)
    tables = root / "tables"; atomic_csv(tables / "delta_sa_by_layer_alpha_and_answer.csv", delta); atomic_csv(tables / "dose_response_and_controls.csv", dose); atomic_csv(tables / "split_and_selection_audit.csv", split); atomic_text(tables / "README.md", _readme()); figure = _plot(root, delta); atomic_text(root / "summary.md", _summary(dose, smoke=smoke))
    # Traceability gate: plotted matched values are exactly the table values used above.
    if not figure.is_file() or figure.stat().st_size == 0: raise RuntimeError("Main figure was not produced")
    completion = {"status": "complete", "smoke_only": smoke, "trial_count": len(trials), "expected_trial_count": expected, "alpha_zero_count": len(alpha_zero), "alpha_zero_parity": "passed", "bootstrap_repeats": repeats, "tables": 4, "figures": 1, "config_fingerprint": config["fingerprint"], "resumed_noop": False}
    atomic_json(progress, completion); atomic_json(root / "progress" / "completion.json", completion); return completion


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--bootstrap", type=int)
    args = parser.parse_args(argv); root = Path(args.output_root) if args.output_root else RESULTS_ROOT
    if args.smoke and not args.output_root: parser.error("--smoke requires an explicit output root outside formal results")
    print(json.dumps(analyze(output_root=root, smoke=args.smoke, resume=args.resume, repeats=args.bootstrap), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
