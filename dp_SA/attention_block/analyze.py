from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from dp_SA.io_utils import atomic_json, load_jsonl

from .config import BOOTSTRAP_REPEATS, MATCHED_PAIRS, REFINE_Q_THRESHOLD, SEED


def _mean_ci(values: Sequence[float], *, repeats: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"mean": math.nan, "sem": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(repeats, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "sem": float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
        "ci_low": float(np.percentile(sampled, 2.5)), "ci_high": float(np.percentile(sampled, 97.5)),
    }


def _paired_test(values: Sequence[float], *, repeats: int, seed: int) -> dict[str, float]:
    result = _mean_ci(values, repeats=repeats, seed=seed)
    array = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(repeats, len(array)))].mean(axis=1)
    result["p_raw"] = float((1 + np.count_nonzero(sampled <= 0)) / (repeats + 1))
    return result


def bh_fdr(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values); ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted); output[order] = np.minimum(adjusted, 1.0)
    return [float(x) for x in output]


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent); os.close(fd)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _group(rows: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], Any]):
    output: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: output[key(row)].append(row)
    return output


def _metrics(blocked: list[dict[str, Any]], repeats: int, seed: int):
    metric_rows: list[dict[str, Any]] = []; side_rows: list[dict[str, Any]] = []
    keyed = _group(blocked, lambda r: (r["arm"], r["phase"], r["condition"], int(r["window_start"]), int(r["window_end"]), r.get("refine_pair")))
    measures = ("first_token_changed", "logit_margin_disruption", "delta_soft_sa", "abs_delta_soft_sa")
    for index, (key, rows) in enumerate(sorted(keyed.items())):
        arm, phase, condition, start, end, refine_pair = key
        for measure in measures:
            stats = _mean_ci([float(r[measure]) for r in rows], repeats=repeats, seed=seed + index * 11 + len(measure))
            metric_rows.append({"arm": arm, "phase": phase, "condition": condition, "window_start": start,
                                "window_end": end, "window_center": (start + end) / 2, "blocked_layer_count": end-start+1,
                                "refine_pair": refine_pair, "group": "all", "metric": measure, "n": len(rows), **stats})
            directions = {}
            for side in ("image_side", "text_side"):
                subset = [r for r in rows if r["test_side"] == side]
                side_stats = _mean_ci([float(r[measure]) for r in subset], repeats=repeats, seed=seed + index * 17 + len(measure) + len(side))
                directions[side] = side_stats["mean"]
                side_rows.append({"arm": arm, "phase": phase, "condition": condition, "window_start": start,
                                  "window_end": end, "window_center": (start+end)/2, "group": side, "metric": measure,
                                  "refine_pair": refine_pair, "n": len(subset), "mean": side_stats["mean"], "ci_low": side_stats["ci_low"], "ci_high": side_stats["ci_high"]})
            all_mean = stats["mean"]
            consistent = all(np.sign(value) == np.sign(all_mean) for value in directions.values()) and np.sign(directions["image_side"]) == np.sign(directions["text_side"])
            for row in side_rows[-2:]: row["direction_consistent"] = bool(consistent)
    return metric_rows, side_rows


def _paired(rows: list[dict[str, Any]], repeats: int, seed: int):
    look = {(r["arm"], r["phase"], r["condition"], int(r["window_start"]), r.get("refine_pair"), r["case_id"]): r for r in rows}
    tests: list[dict[str, Any]] = []
    windows = sorted({(r["arm"], r["phase"], int(r["window_start"]), int(r["window_end"])) for r in rows if r["condition"] in {x for pair in MATCHED_PAIRS.values() for x in pair}})
    for index, (arm, phase, start, end) in enumerate(windows):
        for pair_name, (main, control) in MATCHED_PAIRS.items():
            pair_key = pair_name if phase == "refine" else None
            main_rows = [r for r in rows if r["arm"] == arm and r["phase"] == phase and r["condition"] == main and int(r["window_start"]) == start and r.get("refine_pair") == pair_key]
            differences=[]
            for row in main_rows:
                other = look.get((arm, phase, control, start, pair_key, row["case_id"]))
                if other is not None: differences.append(float(row["logit_margin_disruption"]) - float(other["logit_margin_disruption"]))
            if differences:
                tests.append({"arm": arm, "phase": phase, "comparison": pair_name, "main_condition": main,
                              "control_condition": control, "window_start": start, "window_end": end,
                              "n": len(differences), **_paired_test(differences, repeats=repeats, seed=seed+index*7+len(pair_name))})
    for arm in sorted({r["arm"] for r in rows}):
        globals_by_condition = {condition: {r["case_id"]: r for r in rows if r["arm"] == arm and r["phase"] == "coarse" and r["condition"] == condition} for condition in (
            "all_downstream_to_panl", "all_downstream_to_panl_plus_1", "all_later_to_evidence", "all_later_to_evidence_keep_panl",
            "all_later_to_answer", "all_later_to_answer_keep_panl", "all_later_to_evidence_answer", "all_later_to_evidence_answer_keep_panl")}
        global_pairs = {
            "global_panl_cache": ("all_downstream_to_panl", "all_downstream_to_panl_plus_1"),
            "evidence_rescue": ("all_later_to_evidence", "all_later_to_evidence_keep_panl"),
            "answer_rescue": ("all_later_to_answer", "all_later_to_answer_keep_panl"),
            "evidence_answer_rescue": ("all_later_to_evidence_answer", "all_later_to_evidence_answer_keep_panl"),
        }
        for index, (name, (main, control)) in enumerate(global_pairs.items()):
            shared = sorted(set(globals_by_condition[main]) & set(globals_by_condition[control]))
            differences = [float(globals_by_condition[main][case]["logit_margin_disruption"]) - float(globals_by_condition[control][case]["logit_margin_disruption"]) for case in shared]
            if differences:
                tests.append({"arm": arm, "phase": "global", "comparison": name, "main_condition": main,
                              "control_condition": control, "window_start": 0, "window_end": 27, "n": len(differences),
                              **_paired_test(differences, repeats=repeats, seed=seed+500+index)})
    for arm in sorted({r["arm"] for r in tests}):
        for phase in ("coarse", "refine", "global"):
            family = [r for r in tests if r["arm"] == arm and r["phase"] == phase]
            if family:
                for row, q in zip(family, bh_fdr([r["p_raw"] for r in family])): row["q_bh"] = q
    return tests


def _selection(tests: list[dict[str, Any]], threshold: float):
    selected = {"joint": [], "delayed": []}; evidence = []
    for arm in selected:
        for name in MATCHED_PAIRS:
            qualifying = [r for r in tests if r["arm"] == arm and r["phase"] == "coarse" and r["comparison"] == name and r["ci_low"] > 0 and r.get("q_bh", 1) < threshold]
            if qualifying: selected[arm].append(name)
            evidence.extend({**r, "qualifies": r in qualifying} for r in tests if r["arm"] == arm and r["phase"] == "coarse" and r["comparison"] == name)
    return {"criterion": {"metric": "paired_logit_margin_disruption_difference", "ci_low_gt": 0, "q_bh_lt": threshold,
                           "fdr_family": "15 coarse matched-pair-by-window tests per arm"},
            "selected_pairs": selected, "any_selected": any(selected.values()), "evidence": evidence}


def _cross_arm_paired(output: Path, blocked: list[dict[str, Any]], repeats: int, seed: int) -> list[dict[str, Any]]:
    manifests = {}
    for arm in ("joint", "delayed"):
        path = output / f"{arm}_case_manifest.json"
        manifests[arm] = json.loads(path.read_text()) if path.exists() else []
    def composite(row: dict[str, Any]):
        return (str(row["item_id"]), int(row["prior_index"]), row["condition"], row["version"])
    case_to_key = {arm: {row["case_id"]: composite(row) for row in rows} for arm, rows in manifests.items()}
    shared = set(case_to_key["joint"].values()) & set(case_to_key["delayed"].values())
    values: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in blocked:
        key = case_to_key.get(row["arm"], {}).get(row["case_id"])
        if key not in shared:
            continue
        cell = (row["phase"], row["condition"], int(row["window_start"]), int(row["window_end"]), row.get("refine_pair"), key)
        values[cell][row["arm"]] = float(row["logit_margin_disruption"])
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for cell, by_arm in values.items():
        if set(by_arm) == {"joint", "delayed"}:
            phase, condition, start, end, refine_pair, _key = cell
            grouped[(phase, condition, start, end, refine_pair)].append(by_arm["joint"] - by_arm["delayed"])
    rows = []
    for index, (cell, differences) in enumerate(sorted(grouped.items(), key=str)):
        phase, condition, start, end, refine_pair = cell
        rows.append({"phase": phase, "condition": condition, "window_start": start,
                     "window_end": end, "refine_pair": refine_pair, "contrast": "joint_minus_delayed",
                     "n_exact_overlap": len(differences), **_mean_ci(differences, repeats=repeats, seed=seed+9000+index)})
    _atomic_csv(output / "cross_arm_paired_comparison.csv", rows)
    return rows


def _path_summary(arm: str, tests: list[dict[str, Any]], metrics: list[dict[str, Any]], selection: dict[str, Any]) -> dict[str, Any]:
    selected = set(selection["selected_pairs"][arm])
    globals_by_name = {row["comparison"]: row for row in tests if row["arm"] == arm and row["phase"] == "global"}
    coarse_margin = [row for row in metrics if row["arm"] == arm and row["phase"] == "coarse" and row["group"] == "all" and row["metric"] == "logit_margin_disruption"]
    def any_positive(condition: str) -> bool:
        return any(row["condition"] == condition and row["ci_low"] > 0 for row in coarse_margin)
    def significant_global(name: str) -> bool:
        row = globals_by_name.get(name)
        return bool(row and row["ci_low"] > 0 and row.get("q_bh", 1) < 0.05)
    components = {
        "panl_cache_selected": "panl_cache" in selected,
        "panl_gather_e_plus_a_selected": "panl_gather" in selected,
        "sac_all_content_selected": "jit_all_content" in selected,
        "evidence_rescue_positive": significant_global("evidence_rescue"),
        "answer_rescue_positive": significant_global("answer_rescue"),
        "evidence_answer_rescue_positive": significant_global("evidence_answer_rescue"),
        "panl_to_evidence_descriptive_ci_above_zero": any_positive("panl_to_evidence"),
        "panl_to_answer_descriptive_ci_above_zero": any_positive("panl_to_answer"),
        "panl_to_evidence_answer_descriptive_ci_above_zero": any_positive("panl_to_evidence_answer"),
    }
    path_a = components["panl_cache_selected"] and components["panl_gather_e_plus_a_selected"] and components["evidence_rescue_positive"]
    path_b = components["panl_cache_selected"] and components["panl_gather_e_plus_a_selected"] and components["answer_rescue_positive"]
    redundancy = (not components["panl_to_evidence_descriptive_ci_above_zero"] and
                  not components["panl_to_answer_descriptive_ci_above_zero"] and
                  components["panl_to_evidence_answer_descriptive_ci_above_zero"])
    path_c = components["sac_all_content_selected"]
    return {
        "arm": arm, "components": components,
        "path_A_evidence_to_panl_to_sac": "supported" if path_a else "not_established",
        "path_B_answer_to_panl_to_sac": "supported" if path_b else "not_established",
        "A_B_redundancy_pattern": "present" if redundancy else "not_established",
        "path_C_sac_direct_content_read": "supported_signature" if path_c else "not_established",
        "path_C_dominance": "not_claimed" if path_c and components["panl_cache_selected"] else "undetermined",
        "selected_refinement_pairs": selection["selected_pairs"][arm],
        "coarse_matched_tests": [row for row in tests if row["arm"] == arm and row["phase"] == "coarse"],
        "global_tests": list(globals_by_name.values()),
        "refine_tests": [row for row in tests if row["arm"] == arm and row["phase"] == "refine"],
        "interpretation_limit": "Negative attention-blocking results cannot prove a pathway is absent because information may have been copied earlier.",
    }


def _figures(output: Path, metrics: list[dict[str, Any]], tests: list[dict[str, Any]], cross_rows: list[dict[str, Any]]) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    names = {
        "joint_change_rate.png": ("joint", "first_token_changed"), "joint_logit_disruption.png": ("joint", "logit_margin_disruption"),
        "joint_soft_sa_delta.png": ("joint", "delta_soft_sa"), "delayed_change_rate.png": ("delayed", "first_token_changed"),
        "delayed_logit_disruption.png": ("delayed", "logit_margin_disruption"), "delayed_soft_sa_delta.png": ("delayed", "delta_soft_sa"),
    }
    for filename, (arm, measure) in names.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        rows = [r for r in metrics if r["arm"] == arm and r["metric"] == measure and r["group"] == "all" and r["phase"] in {"coarse", "refine"}]
        for condition in sorted({r["condition"] for r in rows}):
            data = sorted([r for r in rows if r["condition"] == condition], key=lambda r: (r["phase"], r["window_center"]))
            ax.plot([r["window_center"] for r in data], [r["mean"] for r in data], marker="o", label=condition)
        ax.axhline(0, color="black", lw=.8); ax.set_title(f"{arm}: {measure}"); ax.set_xlabel("window center")
        if rows: ax.legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(figures / filename, dpi=160); plt.close(fig)
    global_rows = [r for r in metrics if r["group"] == "all" and r["metric"] == "logit_margin_disruption" and r["condition"].startswith("all_downstream_to_panl")]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r['arm']}\n{r['condition'].replace('all_downstream_to_', '')}" for r in global_rows]
    means = [r["mean"] for r in global_rows]
    errors = [[r["mean"]-r["ci_low"] for r in global_rows], [r["ci_high"]-r["mean"] for r in global_rows]]
    if global_rows: ax.bar(range(len(global_rows)), means, yerr=errors, capsize=3); ax.set_xticks(range(len(labels)), labels)
    ax.set_title("Global PANL blocking (95% CI)"); ax.axhline(0, color="black", lw=.8); fig.tight_layout(); fig.savefig(figures/"global_panl_blocking.png", dpi=160); plt.close(fig)
    rescue = [r for r in tests if r["phase"] == "global" and r["comparison"].endswith("rescue")]
    fig, ax = plt.subplots(figsize=(8, 4)); labels=[f"{r['arm']}\n{r['comparison']}" for r in rescue]
    if rescue:
        ax.bar(range(len(rescue)), [r["mean"] for r in rescue], yerr=[[r["mean"]-r["ci_low"] for r in rescue],[r["ci_high"]-r["mean"] for r in rescue]], capsize=3); ax.set_xticks(range(len(labels)),labels,rotation=20,ha="right")
    ax.set_title("PANL relay rescue: full block minus keep PANL (95% CI)"); ax.axhline(0,color="black",lw=.8); fig.tight_layout(); fig.savefig(figures/"panl_rescue.png",dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4)); plot_rows=[r for r in cross_rows if r["phase"]=="coarse"]
    for condition in sorted({r["condition"] for r in plot_rows}):
        data=sorted([r for r in plot_rows if r["condition"]==condition],key=lambda r:r["window_start"])
        ax.plot([(r["window_start"]+r["window_end"])/2 for r in data],[r["mean"] for r in data],marker="o",label=condition)
    if plot_rows: ax.legend(fontsize=6,ncol=2)
    ax.set_title("Exact-overlap paired contrast: joint minus delayed"); ax.set_xlabel("window center"); ax.axhline(0,color="black",lw=.8); fig.tight_layout(); fig.savefig(figures/"joint_vs_delayed.png",dpi=160); plt.close(fig)


def analyze(output_dir: Path, *, repeats: int = BOOTSTRAP_REPEATS, seed: int = SEED, final: bool = False) -> dict[str, Any]:
    blocked = load_jsonl(output_dir / "blocked_results.jsonl")
    if not blocked: raise ValueError("No blocked results to analyze")
    config = json.loads((output_dir / "run_config.json").read_text())
    clean = load_jsonl(output_dir / "clean_baselines.jsonl")
    failures = load_jsonl(output_dir / "failures.jsonl")
    if not config.get("smoke"):
        for arm in config["arms"]:
            arm_clean = [row for row in clean if row["arm"] == arm]
            arm_coarse = [row for row in blocked if row["arm"] == arm and row["phase"] == "coarse"]
            if len(arm_clean) != 100 or len({row["item_id"] for row in arm_clean}) != 100:
                raise RuntimeError(f"{arm} main population is incomplete: {len(arm_clean)}/100 clean unique items")
            expected_coarse = 100 * (9 * len(config["coarse_windows"]) + 8)
            if len(arm_coarse) != expected_coarse:
                raise RuntimeError(f"{arm} coarse grid is incomplete: {len(arm_coarse)}/{expected_coarse} blocked forwards")
        if failures:
            raise RuntimeError(f"All-100 analysis cannot proceed with {len(failures)} failed forwards")
    metrics, sides = _metrics(blocked, repeats, seed); tests = _paired(blocked, repeats, seed)
    selection = _selection(tests, float(config.get("refine_q_threshold", REFINE_Q_THRESHOLD)))
    if final and not config.get("smoke"):
        for arm in config["arms"]:
            expected = 200 * len(config["refine_windows"]) * len(selection["selected_pairs"][arm])
            if not config.get("auto_refine", True):
                expected = 0
            actual = sum(row["arm"] == arm and row["phase"] == "refine" for row in blocked)
            if actual != expected:
                raise RuntimeError(f"{arm} refine grid is incomplete: {actual}/{expected} blocked forwards")
    cross_rows = _cross_arm_paired(output_dir, blocked, repeats, seed)
    _atomic_csv(output_dir / "window_metrics.csv", metrics)
    _atomic_csv(output_dir / "window_metrics_by_arm_and_side.csv", sides)
    _atomic_csv(output_dir / "paired_path_tests.csv", tests)
    atomic_json(output_dir / "refine_selection.json", selection)
    summaries = {}
    for arm in ("joint", "delayed"):
        summary = _path_summary(arm, tests, metrics, selection)
        atomic_json(output_dir / f"{arm}_path_hypothesis_summary.json", summary); summaries[arm] = summary
    cross = {"joint_selected_pairs": selection["selected_pairs"]["joint"], "delayed_selected_pairs": selection["selected_pairs"]["delayed"],
             "shared_selected_pairs": sorted(set(selection["selected_pairs"]["joint"]) & set(selection["selected_pairs"]["delayed"])),
             "exact_overlap_paired_cells": cross_rows,
             "claim_limit": "No cross-prompt transfer was performed; shared effects do not establish identical vectors or mechanisms."}
    atomic_json(output_dir / "cross_arm_comparison.json", cross)
    diagnostics = {
        "blocked_forward_count": len(blocked), "failure_count": len(failures),
        "all_blocked_weights_zero": all(r["attention_diagnostics"]["max_blocked_weight"] == 0 for r in blocked),
        "all_attention_finite": all(r["attention_diagnostics"]["finite"] for r in blocked),
        "max_row_sum_error": max(r["attention_diagnostics"]["max_row_sum_error"] for r in blocked),
        "head_counts": sorted({h for r in blocked for h in r["attention_diagnostics"]["head_counts"]}),
    }
    atomic_json(output_dir / "technical_diagnostics.json", diagnostics); _figures(output_dir, metrics, tests, cross_rows)
    report = ["# Source Attribution attention-blocking report", "", f"Status: {'complete' if final else 'coarse analysis / provisional'}", "",
              f"Clean forwards: {len(clean)}; blocked forwards: {len(blocked)}; failures: {diagnostics['failure_count']}.", "",
              "## Refinement gate", "", f"Joint: {selection['selected_pairs']['joint'] or 'none'}", f"Delayed: {selection['selected_pairs']['delayed'] or 'none'}", "",
              "Qualifying coarse cells:", ""]
    qualifying = [row for row in selection["evidence"] if row["qualifies"]]
    report.extend(
        f"- {row['arm']} {row['comparison']} W{row['window_start']}–{row['window_end']}: "
        f"mean={row['mean']:.6g}, 95% CI [{row['ci_low']:.6g}, {row['ci_high']:.6g}], "
        f"p={row['p_raw']:.6g}, q={row['q_bh']:.6g}, n={row['n']}"
        for row in qualifying
    )
    if not qualifying: report.append("- None")
    report.extend(["", "## Technical diagnostics", "", f"All blocked weights zero: {diagnostics['all_blocked_weights_zero']}",
              f"All attention finite: {diagnostics['all_attention_finite']}", f"Maximum row-sum error: {diagnostics['max_row_sum_error']:.6g}", "",
              "Side analyses in window_metrics_by_arm_and_side.csv are descriptive only (mean, 95% CI, direction consistency); all hypothesis tests use all 100 items."])
    (output_dir / "summary.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    result = {"status": "complete" if final else "provisional", "selection": selection, "diagnostics": diagnostics}
    if final:
        atomic_json(output_dir / "completion.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--bootstrap-repeats",type=int,default=BOOTSTRAP_REPEATS); parser.add_argument("--seed",type=int,default=SEED); parser.add_argument("--final",action="store_true")
    args=parser.parse_args(argv); analyze(args.output_dir,repeats=args.bootstrap_repeats,seed=args.seed,final=args.final); return 0


if __name__ == "__main__": raise SystemExit(main())
