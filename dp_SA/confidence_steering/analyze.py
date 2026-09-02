from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ALPHAS, BOOTSTRAP_REPEATS, DIRECTIONS, LAYERS, RESULTS_ROOT, SEED, SMOKE_ALPHAS, SMOKE_BOOTSTRAP_REPEATS, SMOKE_LAYERS
from .io_utils import atomic_csv, atomic_json, atomic_text, canonical_hash, check_fingerprint, ensure_layout, load_jsonl, sha256_file

GROUPS = ("all", "follow_text", "follow_image", "conflict_easy", "conflict_hard", "answer_equal_macro")


def _selected(rows: Sequence[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group in ("all", "answer_equal_macro"): return list(rows)
    if group in ("follow_text", "follow_image"): return [r for r in rows if r["answer_origin"] == group]
    return [r for r in rows if r["condition"] == group]


def point_summary(rows: Sequence[dict[str, Any]], field: str, *, answer_macro: bool) -> tuple[float, float, float, int, int, int]:
    if not rows: return (math.nan, math.nan, math.nan, 0, 0, 0)
    if answer_macro:
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows: groups[str(row["fixed_answer"])].append(float(row[field]))
        values = np.asarray([np.mean(groups[color]) for color in sorted(groups)], dtype=float)
    else: values = np.asarray([float(row[field]) for row in rows], dtype=float)
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0; sem = float(sd / math.sqrt(len(values)))
    return float(values.mean()), sd, sem, len(rows), len({r["family_id"] for r in rows}), len({r["fixed_answer"] for r in rows})


def family_draws(test: Sequence[dict[str, Any]], repeats: int, seed: int) -> tuple[list[list[str]], str]:
    families = sorted({str(r["family_id"]) for r in test}); rng = np.random.default_rng(seed)
    draws = [[str(x) for x in rng.choice(families, size=len(families), replace=True)] for _ in range(repeats)]
    return draws, canonical_hash(draws)


def bootstrap_values(rows: Sequence[dict[str, Any]], field: str, draws: Sequence[Sequence[str]], *, answer_macro: bool) -> np.ndarray:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: by_family[str(row["family_id"])].append(row)
    output = []
    for draw in draws:
        sampled = [row for family in draw for row in by_family.get(str(family), [])]
        if not sampled: continue
        value = point_summary(sampled, field, answer_macro=answer_macro)[0]
        if math.isfinite(value): output.append(value)
    return np.asarray(output, dtype=float)


def ci(values: np.ndarray) -> tuple[float, float, int]:
    if not len(values): return math.nan, math.nan, 0
    low, high = np.percentile(values, [2.5, 97.5]); return float(low), float(high), len(values)


def build_delta_table(trials: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> tuple[list[dict[str, Any]], list[list[str]]]:
    draws, draw_fp = family_draws(test, repeats, seed); output = []
    for direction in DIRECTIONS:
        for group in GROUPS:
            for layer in sorted({int(r["layer"]) for r in trials}):
                for alpha in sorted({float(r["alpha"]) for r in trials}):
                    rows = _selected([r for r in trials if r["direction"] == direction and int(r["layer"]) == layer and float(r["alpha"]) == alpha], group)
                    mean, sd, sem, n, nf, na = point_summary(rows, "delta_soft_sa", answer_macro=group == "answer_equal_macro")
                    boot = bootstrap_values(rows, "delta_soft_sa", draws, answer_macro=group == "answer_equal_macro"); low, high, valid = ci(boot)
                    hard = point_summary(rows, "hard_class_changed", answer_macro=group == "answer_equal_macro")[0]
                    margin = point_summary(rows, "margin_change", answer_macro=group == "answer_equal_macro")[0]
                    output.append({"direction": direction, "group": group, "layer": layer, "alpha": alpha, "mean_delta_sa": mean, "sample_sd": sd, "sem": sem,
                                   "ci95_low": low, "ci95_high": high, "hard_change_rate": hard, "mean_margin_change": margin,
                                   "record_count": n, "family_count": nf, "answer_color_count": na, "valid_bootstrap_repeats": valid, "bootstrap_draw_fingerprint": draw_fp})
    return output, draws


def _case_metrics(trials: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in trials: grouped[str(row["case_id"]), str(row["direction"]), int(row["layer"])].append(row)
    output = []
    for (_case, direction, layer), rows in sorted(grouped.items()):
        values = {float(r["alpha"]): float(r["delta_soft_sa"]) for r in rows}
        if len(values) != len(rows): raise ValueError("Duplicate case dose")
        x = np.asarray(sorted(values)); y = np.asarray([values[a] for a in x]); source = rows[0]
        output.append({"case_id": source["case_id"], "item_id": source["item_id"], "family_id": source["family_id"], "fixed_answer": source["fixed_answer"],
                       "condition": source["condition"], "answer_origin": source["answer_origin"], "direction": direction, "layer": layer,
                       "S2": (values[2.0]-values[-2.0])/2.0, "S10": (values[10.0]-values[-10.0])/2.0 if 10.0 in values else math.nan,
                       "slope": float(np.polyfit(x, y, 1)[0])})
    return output


def _metric_rows(case_metrics: Sequence[dict[str, Any]], draws: Sequence[Sequence[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output, family_slopes = [], []
    layers = sorted({int(r["layer"]) for r in case_metrics})
    for direction in DIRECTIONS:
        for group in GROUPS:
            for layer in layers:
                rows = _selected([r for r in case_metrics if r["direction"] == direction and int(r["layer"]) == layer], group)
                for metric in ("S2", "S10", "slope"):
                    finite = [r for r in rows if math.isfinite(float(r[metric]))]
                    mean, sd, sem, n, nf, na = point_summary(finite, metric, answer_macro=group == "answer_equal_macro")
                    boot = bootstrap_values(finite, metric, draws, answer_macro=group == "answer_equal_macro"); low, high, valid = ci(boot)
                    output.append({"direction": direction, "group": group, "layer": layer, "metric": metric, "effect": mean, "sample_sd": sd, "sem": sem,
                                   "ci95_low": low, "ci95_high": high, "record_count": n, "family_count": nf, "answer_color_count": na, "valid_bootstrap_repeats": valid})
                grouped = defaultdict(list)
                for row in rows: grouped[str(row["family_id"])].append(float(row["slope"]))
                family_slopes.extend({"direction": direction, "group": group, "layer": layer, "family_id": family, "slope": float(np.mean(values))} for family, values in sorted(grouped.items()))
    # Paired true-minus-shuffled S10 contrast at case level.
    index = {(r["case_id"], int(r["layer"]), r["direction"]): r for r in case_metrics}
    for group in GROUPS:
        for layer in layers:
            rows = []
            for key, true in index.items():
                if key[1] != layer or key[2] != DIRECTIONS[0]: continue
                shuffled = index.get((key[0], layer, DIRECTIONS[1]))
                if shuffled is None or not all(math.isfinite(float(x["S10"])) for x in (true, shuffled)): continue
                rows.append({**true, "contrast": float(true["S10"])-float(shuffled["S10"])})
            rows = _selected(rows, group); mean, sd, sem, n, nf, na = point_summary(rows, "contrast", answer_macro=group == "answer_equal_macro")
            boot = bootstrap_values(rows, "contrast", draws, answer_macro=group == "answer_equal_macro"); low, high, valid = ci(boot)
            output.append({"direction": "true_minus_shuffled", "group": group, "layer": layer, "metric": "S10", "effect": mean, "sample_sd": sd, "sem": sem,
                           "ci95_low": low, "ci95_high": high, "record_count": n, "family_count": nf, "answer_color_count": na, "valid_bootstrap_repeats": valid})
    return output, family_slopes


def make_wide(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for direction in DIRECTIONS:
        for group in GROUPS:
            selected = [r for r in rows if r["direction"] == direction and r["group"] == group]
            for metric, field in (("mean_delta_sa", "mean_delta_sa"), ("ci_low", "ci95_low"), ("ci_high", "ci95_high"), ("n", "record_count")):
                row = {"direction": direction, "group": group, "metric": metric}
                for value in selected:
                    alpha = f"+{float(value['alpha']):g}" if float(value["alpha"]) > 0 else f"{float(value['alpha']):g}"
                    row[f"L{int(value['layer'])}_a{alpha}"] = value[field]
                output.append(row)
    return output


def _plot_delta(rows: Sequence[dict[str, Any]], path: Path) -> None:
    all_rows = [r for r in rows if r["group"] == "all"]; layers = sorted({int(r["layer"]) for r in all_rows}); alphas = sorted({float(r["alpha"]) for r in all_rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True); palette = plt.cm.coolwarm(np.linspace(0, 1, len(alphas)))
    for ax, direction in zip(axes, DIRECTIONS, strict=True):
        for alpha, color in zip(alphas, palette, strict=True):
            data = sorted([r for r in all_rows if r["direction"] == direction and float(r["alpha"]) == alpha], key=lambda r: int(r["layer"]))
            x = [int(r["layer"]) for r in data]; mean = np.asarray([r["mean_delta_sa"] for r in data]); low = np.asarray([r["ci95_low"] for r in data]); high = np.asarray([r["ci95_high"] for r in data])
            ax.errorbar(x, mean, yerr=np.vstack([mean-low, high-mean]), marker="o", capsize=3, color="#888888" if alpha == 0 else color, label=f"alpha={alpha:+g}")
        ax.axhline(0, color="black", lw=.8); ax.set_xticks(layers); ax.set_title(direction); ax.set_xlabel("Zero-based decoder layer"); ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Mean delta soft SA"); axes[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def _plot_symmetric(rows: Sequence[dict[str, Any]], path: Path, *, smoke: bool) -> None:
    metric = "S2" if smoke else "S10"; selected = [r for r in rows if r["group"] == "all" and r["metric"] == metric and r["direction"] in DIRECTIONS]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for direction, color in zip(DIRECTIONS, ("#2166ac", "#b2182b"), strict=True):
        data = sorted([r for r in selected if r["direction"] == direction], key=lambda r: int(r["layer"])); mean=np.asarray([r["effect"] for r in data]); low=np.asarray([r["ci95_low"] for r in data]); high=np.asarray([r["ci95_high"] for r in data])
        ax.errorbar([r["layer"] for r in data], mean, yerr=np.vstack([mean-low,high-mean]), marker="o", capsize=3, label=direction, color=color)
    ax.axhline(0,color="black",lw=.8); ax.set_xlabel("Zero-based decoder layer"); ax.set_ylabel(metric); ax.set_title(f"Symmetric effect ({metric}{' smoke surrogate' if smoke else ''})"); ax.legend(fontsize=8); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)


def analyze(*, output_root: Path = RESULTS_ROOT, smoke: bool = False, resume: bool = False, repeats: int | None = None) -> dict[str, Any]:
    root = ensure_layout(output_root); trial_path = root / "artifacts/trials/trials.jsonl"; test_path = root / "artifacts/audits/test_manifest.jsonl"
    trials, test = load_jsonl(trial_path), load_jsonl(test_path); layers = SMOKE_LAYERS if smoke else LAYERS; alphas = SMOKE_ALPHAS if smoke else ALPHAS
    expected = len(test)*len(DIRECTIONS)*len(layers)*len(alphas)
    if len(trials) != expected or len({(r["case_id"],r["direction"],r["layer"],r["alpha"]) for r in trials}) != expected: raise ValueError("Analysis trial grid incomplete")
    zero = [r for r in trials if float(r["alpha"]) == 0]
    if any(not r.get("alpha_zero_parity",{}).get("passed") for r in zero): raise ValueError("Alpha-zero parity gate failed")
    repeats = int(repeats or (SMOKE_BOOTSTRAP_REPEATS if smoke else BOOTSTRAP_REPEATS)); config={"smoke_only":smoke,"trials_sha256":sha256_file(trial_path),"test_sha256":sha256_file(test_path),"repeats":repeats,"seed":SEED,"analysis_code_sha256":sha256_file(Path(__file__))}
    fingerprint=check_fingerprint(root/"progress/analyze_config.json",config,resume=resume); progress=root/"progress/analyze.json"
    if resume and progress.is_file():
        old=json.loads(progress.read_text()); required=[root/"tables/delta_sa_by_layer_alpha.csv",root/"figures/delta_sa_by_layer.png",root/"figures/symmetric_effect_s10.png"]
        if old.get("config_fingerprint")==fingerprint and old.get("status")=="complete" and all(p.is_file() and p.stat().st_size for p in required): return {**old,"resumed_noop":True}
    delta, draws=build_delta_table(trials,test,repeats=repeats,seed=SEED); case_metrics=_case_metrics(trials); effects,family_slopes=_metric_rows(case_metrics,draws)
    atomic_csv(root/"tables/delta_sa_by_layer_alpha.csv",delta); atomic_csv(root/"tables/delta_sa_by_layer_alpha_wide.csv",make_wide(delta)); atomic_csv(root/"tables/symmetric_effect_and_contrast.csv",effects)
    atomic_csv(root/"artifacts/audits/case_symmetric_effects_and_slopes.csv",case_metrics); atomic_csv(root/"artifacts/audits/family_slopes.csv",family_slopes)
    _plot_delta(delta,root/"figures/delta_sa_by_layer.png"); _plot_symmetric(effects,root/"figures/symmetric_effect_s10.png",smoke=smoke)
    boundary="本实验干预的是由residual fixed-answer confidence标签定义的LAT方向。成功结果说明该confidence-defined方向具有改变final SA的因果能力，但不能证明它是纯粹、唯一的confidence子空间，也不能单独证明完整LAT→PANL→SA中介链。"
    atomic_text(root/"summary.md",f"# Residual Confidence LAT Steering\n\n- smoke_only: `{str(smoke).lower()}`\n- formal_experiment_executed: `{str(not smoke).lower()}`\n- bootstrap_repeats: `{repeats}`\n\n> {boundary}\n")
    result={"status":"complete","smoke_only":smoke,"trial_count":len(trials),"bootstrap_repeats":repeats,"alpha_zero_parity":"passed","tables":3,"figures":2,"config_fingerprint":fingerprint,"resumed_noop":False}; atomic_json(progress,result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root"); parser.add_argument("--resume",action="store_true"); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--bootstrap",type=int)
    args=parser.parse_args(argv); root=Path(args.output_root) if args.output_root else RESULTS_ROOT
    print(json.dumps(analyze(output_root=root,smoke=args.smoke,resume=args.resume,repeats=args.bootstrap),ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
