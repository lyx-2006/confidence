from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dp_SA.attention_block.analyze import bh_fdr
from dp_SA.io_utils import atomic_json, load_jsonl

from .core import WINDOWS

LABELS = {"sac_to_panl": "SAC→PANL", "sac_to_panl_plus_1": "SAC→PANL+1", "empty_block_parity": "empty-block parity"}


def _stats(values: Sequence[float], seed: int, repeats: int = 2000) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(repeats, len(array)))].mean(axis=1)
    return {"mean": float(array.mean()), "sem": float(array.std(ddof=1) / math.sqrt(len(array))),
            "ci_low": float(np.percentile(sampled, 2.5)), "ci_high": float(np.percentile(sampled, 97.5)),
            "p_raw": float((1 + np.count_nonzero(sampled <= 0)) / (repeats + 1))}


def _ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float); order = np.argsort(values, kind="mergesort"); ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]: end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1; start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return float(np.corrcoef(_ranks(left), _ranks(right))[0, 1])


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}); fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent); os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def analyze(output: Path, full_output: Path, repeats: int = 2000, seed: int = 42) -> dict[str, Any]:
    clean = load_jsonl(output / "clean_results.jsonl"); blocked = load_jsonl(output / "block_results.jsonl")
    spans = load_jsonl(output / "token_spans.jsonl"); failures = load_jsonl(output / "failures.jsonl")
    if len(clean) != 30 or len(blocked) != 450 or len(spans) != 30 or failures:
        raise ValueError(f"Minimal formal counts invalid: clean={len(clean)}, blocked={len(blocked)}, spans={len(spans)}, failures={len(failures)}")
    empty = [row for row in blocked if row["condition"] == "empty_block_parity"]
    empty_ok = all(row["empty_parity"]["hard_equal"] and row["empty_parity"]["max_abs_logit_difference"] <= 1e-6 and row["empty_parity"]["abs_soft_sa_difference"] <= 1e-9 for row in empty)
    if len(empty) != 150 or not empty_ok: raise ValueError("Formal empty-block parity gate failed")
    by = {(row["case_id"], row["condition"], int(row["window_start"])): row for row in blocked}
    main_tests=[]; contrast_tests=[]; metric_rows=[]
    metrics = ("logit_margin_disruption", "first_token_changed", "delta_soft_sa", "original_token_logit_diff_change")
    for wi, (start, end) in enumerate(WINDOWS):
        main = [row for row in blocked if row["condition"] == "sac_to_panl" and row["window_start"] == start]
        control = [row for row in blocked if row["condition"] == "sac_to_panl_plus_1" and row["window_start"] == start]
        for condition_rows in (main, control):
            for metric in metrics:
                result = _stats([float(row[metric]) for row in condition_rows], seed + wi * 31 + len(metric))
                metric_rows.append({"condition": LABELS[condition_rows[0]["condition"]], "window_start": start, "window_end": end,
                                    "metric": metric, "n": len(condition_rows), **{key: result[key] for key in ("mean", "sem", "ci_low", "ci_high")}})
                for side in ("image_side", "text_side"):
                    subset = [row for row in condition_rows if row["test_side"] == side]
                    side_result = _stats([float(row[metric]) for row in subset], seed + wi * 37 + len(metric) + len(side))
                    metric_rows.append({"condition": LABELS[condition_rows[0]["condition"]], "window_start": start, "window_end": end,
                                        "metric": metric, "group": side, "n": len(subset), **{key: side_result[key] for key in ("mean", "ci_low", "ci_high")}})
        main_result = _stats([float(row["logit_margin_disruption"]) for row in main], seed + 100 + wi, repeats)
        differences = [float(by[(row["case_id"], "sac_to_panl", start)]["logit_margin_disruption"]) - float(by[(row["case_id"], "sac_to_panl_plus_1", start)]["logit_margin_disruption"]) for row in main]
        contrast_result = _stats(differences, seed + 200 + wi, repeats)
        main_tests.append({"test": "SAC→PANL > 0", "window_start": start, "window_end": end, "n": len(main), **main_result})
        contrast_tests.append({"test": "SAC→PANL − SAC→PANL+1 > 0", "window_start": start, "window_end": end, "n": len(differences), **contrast_result})
    for rows in (main_tests, contrast_tests):
        for row, q in zip(rows, bh_fdr([row["p_raw"] for row in rows])): row["q_bh"] = q
    decisions=[]
    for main, contrast in zip(main_tests, contrast_tests):
        main_positive = main["ci_low"] > 0 and main["q_bh"] < 0.05
        contrast_positive = contrast["ci_low"] > 0 and contrast["q_bh"] < 0.05
        decisions.append({"window_start": main["window_start"], "window_end": main["window_end"],
                          "sac_to_panl_itself_positive": main_positive, "matched_contrast_positive": contrast_positive,
                          "supports_direct_path": main_positive and contrast_positive})

    full_clean = {row["case_id"]: row for row in load_jsonl(full_output / "clean_baselines.jsonl") if row["arm"] == "delayed"}
    minimal_clean = {row["case_id"]: row for row in clean}; shared = sorted(minimal_clean)
    if not all(case in full_clean for case in shared): raise ValueError("Full delayed clean baseline missing selected minimal cases")
    hard_agreement = float(np.mean([minimal_clean[c]["clean_class"] == full_clean[c]["clean_class"] for c in shared]))
    spearman = _spearman([minimal_clean[c]["soft_sa_image_score"] for c in shared], [full_clean[c]["soft_sa_image_score"] for c in shared])
    distance_rows=[]; span_map={row["case_id"]: row for row in spans}
    for case in shared:
        row=span_map[case]; distance_rows.append({"case_id":case,"item_id":minimal_clean[case]["item_id"],"test_side":minimal_clean[case]["test_side"],
            "minimal_panl_to_sac":row["PANL_TO_SAC_DISTANCE"],"full_panl_to_sac":row["FULL_PANL_TO_SAC_DISTANCE"],"distance_reduction":row["distance_reduction"]})
    full_block = [row for row in load_jsonl(full_output / "blocked_results.jsonl") if row["arm"] == "delayed" and row["phase"] == "coarse" and row["case_id"] in minimal_clean and row["condition"] in {"sac_to_panl","sac_to_panl_plus_1"}]
    full_by={(r["case_id"],r["condition"],r["window_start"]):r for r in full_block}; comparison=[]
    for condition in ("sac_to_panl","sac_to_panl_plus_1"):
        for start,end in WINDOWS:
            mini=[by[(case,condition,start)] for case in shared]; full=[full_by[(case,condition,start)] for case in shared]
            for metric in ("logit_margin_disruption","first_token_changed","delta_soft_sa"):
                delta=[float(m[metric])-float(f[metric]) for m,f in zip(mini,full)]
                result=_stats(delta,seed+700+start+len(metric),repeats)
                comparison.append({"condition":LABELS[condition],"window_start":start,"window_end":end,"metric":metric,
                    "minimal_mean":float(np.mean([m[metric] for m in mini])),"full_mean":float(np.mean([f[metric] for f in full])),
                    "minimal_minus_full":result["mean"],"ci_low":result["ci_low"],"ci_high":result["ci_high"]})
    clean_comparison={"n":len(shared),"hard_agreement_rate":hard_agreement,"soft_sa_spearman":spearman,
        "minimal_class_distribution":{str(i):sum(minimal_clean[c]["clean_class"]==i for c in shared) for i in range(9)},
        "full_class_distribution":{str(i):sum(full_clean[c]["clean_class"]==i for c in shared) for i in range(9)},
        "minimal_panl_to_sac":{"mean":float(np.mean([r["minimal_panl_to_sac"] for r in distance_rows])),"min":min(r["minimal_panl_to_sac"] for r in distance_rows),"max":max(r["minimal_panl_to_sac"] for r in distance_rows)},
        "full_panl_to_sac":{"mean":float(np.mean([r["full_panl_to_sac"] for r in distance_rows])),"min":min(r["full_panl_to_sac"] for r in distance_rows),"max":max(r["full_panl_to_sac"] for r in distance_rows)}}
    result={"criterion":"support only if SAC→PANL itself and SAC→PANL − SAC→PANL+1 both have CI_low>0 and BH q<0.05",
            "supports_direct_path_any_window":any(row["supports_direct_path"] for row in decisions),"decisions":decisions,
            "main_tests":main_tests,"contrast_tests":contrast_tests,"clean_comparison":clean_comparison,
            "technical":{"empty_block_rows":len(empty),"empty_block_parity":empty_ok,"failures":len(failures)}}
    atomic_json(output/"analysis.json",result);atomic_json(output/"clean_comparison.json",clean_comparison)
    _csv(output/"window_metrics.csv",metric_rows);_csv(output/"minimal_vs_full_block.csv",comparison);_csv(output/"distance_comparison.csv",distance_rows)
    lines=["# Minimal-prompt direct SAC→PANL diagnostic","",f"- Direct path supported: **{result['supports_direct_path_any_window']}**",f"- Minimal/full clean hard agreement: {hard_agreement:.3f}",f"- Minimal/full soft-SA Spearman: {spearman:.3f}",f"- PANL→SAC distance: minimal {clean_comparison['minimal_panl_to_sac']['mean']:.1f}, full {clean_comparison['full_panl_to_sac']['mean']:.1f}","","|Window|SAC→PANL mean [CI]|q|Contrast mean [CI]|q|Direct support|","|---|---:|---:|---:|---:|---|"]
    for main,contrast,decision in zip(main_tests,contrast_tests,decisions):
        lines.append(f"|{main['window_start']}–{main['window_end']}|{main['mean']:+.4f} [{main['ci_low']:+.4f},{main['ci_high']:+.4f}]|{main['q_bh']:.4f}|{contrast['mean']:+.4f} [{contrast['ci_low']:+.4f},{contrast['ci_high']:+.4f}]|{contrast['q_bh']:.4f}|{decision['supports_direct_path']}|")
    (output/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--full-output",type=Path,required=True);parser.add_argument("--bootstrap-repeats",type=int,default=2000);parser.add_argument("--seed",type=int,default=42);args=parser.parse_args(argv)
    analyze(args.output_dir.resolve(),args.full_output.resolve(),args.bootstrap_repeats,args.seed);return 0


if __name__ == "__main__": raise SystemExit(main())

