from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dp_SA.attention_block.analyze import _atomic_csv, _mean_ci, bh_fdr
from dp_SA.io_utils import atomic_json, atomic_jsonl, load_jsonl

from .core import CONDITIONS, WINDOWS, effects, one_sided_sign_flip

EFFECTS = ("interaction", "bridge_gain", "matched_gain")
FACTORIAL_CONDITIONS = ("C00", "C10", "C01", "C11")
CONDITION_COLORS = {"C00":"#d62728", "C10":"#1f77b4", "C01":"#2ca02c", "C11":"#e377c2"}


def _item_effects(blocked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["case_id"], int(r["window_start"]), r["condition"]): r for r in blocked}
    cases = sorted({r["case_id"] for r in blocked})
    output: list[dict[str, Any]] = []
    for case_id in cases:
        exemplar = next(r for r in blocked if r["case_id"] == case_id)
        for start, end in WINDOWS:
            rows = {condition: lookup[(case_id, start, condition)] for condition in CONDITIONS}
            margin = effects(*(float(rows[c]["margin"]) for c in CONDITIONS))
            soft = effects(*(float(rows[c]["soft_sa"]) for c in CONDITIONS))
            output.append({
                "case_id": case_id, "item_id": exemplar["item_id"], "test_side": exemplar["test_side"],
                "window_start": start, "window_end": end, "window_center": (start + end) / 2,
                **dict(zip(EFFECTS, margin)),
                **{f"soft_{name}": value for name, value in zip(EFFECTS, soft)},
                **{f"token_change_{c.lower()}": int(rows[c]["token_changed"]) for c in CONDITIONS},
                **{f"margin_{c.lower()}": float(rows[c]["margin"]) for c in CONDITIONS},
                **{f"soft_sa_{c.lower()}": float(rows[c]["soft_sa"]) for c in CONDITIONS},
                **{f"delta_soft_{c.lower()}": float(rows[c]["delta_soft_sa"]) for c in CONDITIONS},
            })
    return output


def _technical_checks(output: Path, blocked: list[dict[str, Any]], expected_n: int) -> dict[str, Any]:
    parity = load_jsonl(output / "clean_parity.jsonl")
    failures = load_jsonl(output / "failures.jsonl")
    expected_blocked = expected_n * len(WINDOWS) * len(CONDITIONS)
    diagnostics_ok = True
    diagnostic_error = None
    try:
        if len(blocked) != expected_blocked or len(parity) != expected_n or failures:
            raise ValueError(f"counts blocked={len(blocked)}/{expected_blocked}, parity={len(parity)}/{expected_n}, failures={len(failures)}")
        keys = {(r["case_id"], r["condition"], int(r["window_start"])) for r in blocked}
        if len(keys) != expected_blocked:
            raise ValueError("duplicate or missing case/condition/window cells")
        for row in blocked:
            diag = row["attention_diagnostics"]
            if diag.get("empty") or len(diag["layers"]) != 6:
                raise ValueError("empty or non-six-layer diagnostic")
            for layer in diag["by_layer"].values():
                if (layer["max_blocked_weight"] != 0.0 or not layer["finite"] or
                        layer["head_count"] != 28 or layer.get("hook_call_count") != 1 or
                        layer["max_row_sum_error"] > 0.01):
                    raise ValueError(f"invalid attention diagnostic: {layer}")
        if not all(r["hard_equal"] and r["max_abs_logit_difference"] <= 0.125 and r["abs_soft_sa_difference"] <= 1e-6 for r in parity):
            raise ValueError("clean parity outside tolerance")
    except Exception as exc:
        diagnostics_ok = False; diagnostic_error = str(exc)
    return {"passed": diagnostics_ok, "error": diagnostic_error, "clean_parity_count": len(parity),
            "blocked_count": len(blocked), "failure_count": len(failures)}


def _summaries(items: list[dict[str, Any]], *, repeats: int, sign_repeats: int, seed: int):
    primary: list[dict[str, Any]] = []
    descriptive: list[dict[str, Any]] = []
    for wi, (start, end) in enumerate(WINDOWS):
        window_rows = [r for r in items if int(r["window_start"]) == start]
        for ei, effect in enumerate(EFFECTS):
            values = [float(r[effect]) for r in window_rows]
            primary.append({"window_start": start, "window_end": end, "window_center": (start + end) / 2,
                            "group": "all", "metric": effect, "n": len(values),
                            **_mean_ci(values, repeats=repeats, seed=seed + wi * 101 + ei),
                            "p_sign_flip": one_sided_sign_flip(values, seed=seed + 1000 + wi * 101 + ei, repeats=sign_repeats)})
        for group in ("all", "image_side", "text_side"):
            subset = window_rows if group == "all" else [r for r in window_rows if r["test_side"] == group]
            for mi, metric in enumerate(EFFECTS + tuple(f"soft_{x}" for x in EFFECTS)):
                descriptive.append({"window_start": start, "window_end": end, "window_center": (start + end) / 2,
                                    "group": group, "metric": metric, "n": len(subset),
                                    **_mean_ci([float(r[metric]) for r in subset], repeats=repeats,
                                               seed=seed + 3000 + wi * 211 + mi * 7 + len(group))})
            for ci, condition in enumerate(CONDITIONS):
                metric = f"token_change_{condition.lower()}"
                descriptive.append({"window_start": start, "window_end": end, "window_center": (start + end) / 2,
                                    "group": group, "metric": metric, "n": len(subset),
                                    **_mean_ci([float(r[metric]) for r in subset], repeats=repeats,
                                               seed=seed + 6000 + wi * 211 + ci * 7 + len(group))})
                metric = f"delta_soft_{condition.lower()}"
                descriptive.append({"window_start": start, "window_end": end, "window_center": (start + end) / 2,
                                    "group": group, "metric": metric, "n": len(subset),
                                    **_mean_ci([float(r[metric]) for r in subset], repeats=repeats,
                                               seed=seed + 7000 + wi * 211 + ci * 7 + len(group))})
    descriptive_lookup={(r["window_start"],r["metric"],r["group"]):r for r in descriptive}
    for row in descriptive:
        if row["group"] not in {"image_side","text_side"}: continue
        all_mean=descriptive_lookup[(row["window_start"],row["metric"],"all")]["mean"]
        other="text_side" if row["group"]=="image_side" else "image_side"
        other_mean=descriptive_lookup[(row["window_start"],row["metric"],other)]["mean"]
        row["direction_consistent_with_all"] = bool(np.sign(row["mean"]) == np.sign(all_mean))
        row["direction_consistent_between_sides"] = bool(np.sign(row["mean"]) == np.sign(other_mean))
    for effect in EFFECTS:
        family = [r for r in primary if r["metric"] == effect]
        for row, q in zip(family, bh_fdr([r["p_sign_flip"] for r in family])):
            row["q_bh"] = q
    by_key = {(r["window_start"], r["metric"]): r for r in primary}
    windows = []
    for start, end in WINDOWS:
        rows = [by_key[(start, effect)] for effect in EFFECTS]
        supported = all(r["mean"] > 0 and r["ci_low"] > 0 and r["q_bh"] < 0.05 for r in rows)
        windows.append({"window_start": start, "window_end": end, "supported": supported,
                        "criteria": {r["metric"]: {k: r[k] for k in ("mean", "ci_low", "ci_high", "p_sign_flip", "q_bh")} for r in rows}})
    return primary, descriptive, windows


def _plots(output: Path, primary: list[dict[str, Any]]) -> list[str]:
    names = {"interaction": "PANL interaction", "bridge_gain": "PANL bridge gain", "matched_gain": "Matched-control gain"}
    files = []
    for metric in EFFECTS:
        rows = sorted((r for r in primary if r["metric"] == metric), key=lambda r: r["window_center"])
        fig, ax = plt.subplots(figsize=(6.4, 4.1))
        ax.plot([r["window_center"] for r in rows], [r["mean"] for r in rows], color="#d62728", marker="o", linewidth=2)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xticks([r["window_center"] for r in rows], [f"L{r['window_start']}–{r['window_end']}" for r in rows])
        ax.set_xlabel("Center of selectively blocked consecutive layers")
        ax.set_ylabel("Fixed-clean-class logit margin effect")
        ax.set_title(names[metric]); ax.grid(axis="y", alpha=0.2); fig.tight_layout()
        stem = {"interaction": "interaction", "bridge_gain": "bridge", "matched_gain": "matched_control"}[metric]
        for suffix in ("png", "pdf"):
            path = output / f"{stem}.{suffix}"; fig.savefig(path, dpi=220); files.append(path.name)
        plt.close(fig)
    return files


def _condition_summaries(blocked: list[dict[str, Any]], *, repeats: int, seed: int) -> list[dict[str, Any]]:
    output=[]
    for wi,(start,end) in enumerate(WINDOWS):
        for ci,condition in enumerate(FACTORIAL_CONDITIONS):
            rows=[r for r in blocked if int(r["window_start"])==start and r["condition"]==condition]
            for group in ("all","image_side","text_side"):
                subset=rows if group=="all" else [r for r in rows if r["test_side"]==group]
                for mi,(metric,source) in enumerate((("logit_diff_change","logit_margin_disruption"),("token_change_rate","token_changed"))):
                    output.append({"window_start":start,"window_end":end,"window_center":(start+end)/2,
                                   "condition":condition,"group":group,"metric":metric,"n":len(subset),
                                   **_mean_ci([float(r[source]) for r in subset],repeats=repeats,
                                              seed=seed+9000+wi*307+ci*31+mi*7+len(group))})
    lookup={(r["window_start"],r["condition"],r["metric"],r["group"]):r for r in output}
    for row in output:
        if row["group"] not in {"image_side","text_side"}:continue
        all_mean=lookup[(row["window_start"],row["condition"],row["metric"],"all")]["mean"]
        other="text_side" if row["group"]=="image_side" else "image_side"
        other_mean=lookup[(row["window_start"],row["condition"],row["metric"],other)]["mean"]
        row["direction_consistent_with_all"]=bool(np.sign(row["mean"])==np.sign(all_mean))
        row["direction_consistent_between_sides"]=bool(np.sign(row["mean"])==np.sign(other_mean))
    return output


def _condition_plots(output:Path,rows:list[dict[str,Any]])->list[str]:
    files=[]
    labels={"logit_diff_change":"Logit diff change (clean margin − blocked margin)",
            "token_change_rate":"Token change rate"}
    for metric in labels:
        fig,ax=plt.subplots(figsize=(7.0,4.4))
        for condition in FACTORIAL_CONDITIONS:
            values=sorted((r for r in rows if r["metric"]==metric and r["group"]=="all" and r["condition"]==condition),key=lambda r:r["window_start"])
            ax.plot([r["window_center"] for r in values],[r["mean"] for r in values],marker="o",linewidth=2,
                    color=CONDITION_COLORS[condition],label=condition)
        if metric=="logit_diff_change":ax.axhline(0,color="black",linewidth=.8,alpha=.6)
        centers=[(a+b)/2 for a,b in WINDOWS]
        ax.set_xticks(centers,[f"L{a}–{b}" for a,b in WINDOWS]);ax.set_xlabel("Center of selectively blocked consecutive layers")
        ax.set_ylabel(labels[metric]);ax.set_title(f"Four factorial conditions: {labels[metric]}")
        ax.legend(frameon=False,ncol=4);ax.grid(axis="y",alpha=.2);fig.tight_layout()
        for suffix in ("png","pdf"):
            path=output/f"four_conditions_{metric}.{suffix}";fig.savefig(path,dpi=220);files.append(path.name)
        plt.close(fig)
    return files


def analyze(output: Path) -> dict[str, Any]:
    output = output.resolve(); config = json.loads((output / "run_config.json").read_text())
    blocked = load_jsonl(output / "blocked_results.jsonl")
    expected_n = 4 if config["smoke"] else 100
    technical = _technical_checks(output, blocked, expected_n)
    if not technical["passed"]:
        raise RuntimeError(f"Technical validation failed: {technical['error']}")
    items = _item_effects(blocked); atomic_jsonl(output / "effects_item_level.jsonl", items)
    primary, descriptive, windows = _summaries(items, repeats=int(config["bootstrap_repeats"]),
                                               sign_repeats=int(config["sign_flip_repeats"]), seed=int(config["seed"]))
    _atomic_csv(output / "summary.csv", [{"table": "primary", **r} for r in primary] + [{"table": "descriptive", **r} for r in descriptive])
    condition_rows=_condition_summaries(blocked,repeats=int(config["bootstrap_repeats"]),seed=int(config["seed"]))
    _atomic_csv(output/"four_conditions_metrics.csv",condition_rows)
    atomic_json(output/"four_conditions_metrics.json",condition_rows)
    any_supported = any(r["supported"] for r in windows)
    interpretation = ("direct PANL bridge supported" if any_supported else
                      "不支持直接 PANL→SAC 两跳充分路径；不能否定 PANL 经其他 downstream tokens 中继。")
    result = {"experiment": "direct_panl_bridge", "n": expected_n, "technical_validation": technical,
              "support_rule": "All I_PANL, G_PANL, and G_matched must have mean>0, bootstrap CI low>0, and within-metric BH q<0.05 in the same window.",
              "windows": windows, "direct_panl_bridge_supported": any_supported,
              "interpretation": interpretation, "primary": primary, "descriptive": descriptive,
              "four_conditions_metrics":condition_rows}
    atomic_json(output / "summary.json", result)
    plot_files = _plots(output, primary)+_condition_plots(output,condition_rows)
    lines = ["# Direct PANL bridge", "", f"- n: {expected_n}", f"- Technical validation: PASS", f"- Conclusion: {interpretation}", "",
             "| Window | I mean [95% CI], q | G mean [95% CI], q | Matched mean [95% CI], q | Supported |",
             "|---|---:|---:|---:|:---:|"]
    for w in windows:
        cells=[]
        for metric in EFFECTS:
            r=w["criteria"][metric]; cells.append(f"{r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}], {r['q_bh']:.4g}")
        lines.append(f"| L{w['window_start']}–{w['window_end']} | " + " | ".join(cells) + f" | {'yes' if w['supported'] else 'no'} |")
    lines += ["", "## Four conditions analyzed separately", "",
             "`logit diff change = clean fixed-class margin − blocked fixed-class margin`; positive means blocking reduced the clean-class margin.", "",
             "| Window | Metric | C00 | C10 | C01 | C11 |", "|---|---|---:|---:|---:|---:|"]
    condition_lookup={(r["window_start"],r["metric"],r["condition"]):r for r in condition_rows if r["group"]=="all"}
    for start,end in WINDOWS:
        for metric in ("logit_diff_change","token_change_rate"):
            values=[condition_lookup[(start,metric,c)]["mean"] for c in FACTORIAL_CONDITIONS]
            lines.append(f"| L{start}–{end} | {metric} | "+" | ".join(f"{x:.4f}" for x in values)+" |")
    lines += ["", "All/image-side/text-side means, SEMs, and bootstrap CIs are in `four_conditions_metrics.csv`. Side results are descriptive only.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    marker = {"status": "complete", "technical_validation_passed": True, "direct_panl_bridge_supported": any_supported,
              "outputs": ["effects_item_level.jsonl", "summary.csv", "summary.json", "summary.md",
                          "four_conditions_metrics.csv","four_conditions_metrics.json",*plot_files]}
    atomic_json(output / "analysis_completion.json", marker)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True); args=parser.parse_args(argv)
    analyze(args.output_dir); return 0


if __name__ == "__main__": raise SystemExit(main())
