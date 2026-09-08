from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dp_SA.config import MIDPOINTS
from dp_SA.io_utils import atomic_json, load_jsonl, sha256_file

from .analyze import FamilyBootstrap, atomic_csv, bh_fdr
from .config import (
    CONDITIONS, GROUPS, PRIMARY_GROUP, SEED, WINDOWS, WINDOW_NAMES, default_output,
)

SA_STABLE_EPSILON = 0.001
SIGN_FLIP_REPEATS = 2000


def probabilities(logits: Sequence[float]) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("Expected nine finite SA logits")
    shifted = values - values.max()
    result = np.exp(shifted)
    result /= result.sum()
    return result


def distribution_distances(clean: Sequence[float], blocked: Sequence[float]) -> tuple[float, float]:
    p, q = probabilities(clean), probabilities(blocked)
    midpoint = (p + q) / 2.0
    js = 0.5 * float(np.sum(p * np.log(p / midpoint))) + 0.5 * float(np.sum(q * np.log(q / midpoint)))
    tv = 0.5 * float(np.abs(p - q).sum())
    return js, tv


def sa_state(delta: float, epsilon: float = SA_STABLE_EPSILON) -> str:
    if delta > epsilon:
        return "SA_up"
    if delta < -epsilon:
        return "SA_down"
    return "SA_stable"


def enrich_trial(row: dict[str, Any], epsilon: float = SA_STABLE_EPSILON) -> dict[str, Any]:
    clean_class = int(row["clean_hard_sa_class"])
    blocked_class = int(row["blocked_hard_sa_class"])
    ordinal = blocked_class - clean_class
    midpoint_shift = float(MIDPOINTS[blocked_class] - MIDPOINTS[clean_class])
    js, tv = distribution_distances(row["clean_class_logits"], row["blocked_class_logits"])
    delta = float(row["delta_soft_sa"])
    return {
        "experiment": row["experiment"], "case_id": row["case_id"],
        "family_id": row["family_id"], "item_id": row["item_id"],
        "answer": row["answer"], "test_side": row["test_side"],
        "condition": row["condition"], "window_name": row["window_name"],
        "window_start": int(row["window_start"]), "window_end": int(row["window_end"]),
        "delta_soft_sa": delta, "abs_delta_soft_sa": abs(delta),
        "sa_direction": sa_state(delta, epsilon), "sa_stable_epsilon": epsilon,
        "js_divergence_nats": js, "total_variation": tv,
        "clean_hard_sa_class": clean_class, "blocked_hard_sa_class": blocked_class,
        "ordinal_shift": ordinal, "abs_ordinal_shift": abs(ordinal),
        "hard_shift_up": float(ordinal > 0), "hard_shift_down": float(ordinal < 0),
        "hard_shift_stable": float(ordinal == 0),
        "large_ordinal_shift_rate": float(abs(ordinal) >= 2),
        "hard_midpoint_shift": midpoint_shift,
        "abs_hard_midpoint_shift": abs(midpoint_shift),
    }


def _group_point(values: dict[str, float], group: str,
                 manifest: dict[str, dict[str, Any]]) -> float:
    if not values:
        return math.nan
    if group in ("family_micro", "all"):
        return float(np.mean(list(values.values())))
    side = {"image_side": "high_image", "text_side": "high_text"}.get(group)
    by_answer: dict[str, list[float]] = {}
    for family, value in values.items():
        row = manifest[family]
        if row["test_answer"] == "blue" or (side is not None and row["test_side"] != side):
            continue
        by_answer.setdefault(str(row["test_answer"]), []).append(float(value))
    return float(np.mean([np.mean(cells) for cells in by_answer.values()])) if by_answer else math.nan


def sign_flip_p(values: dict[str, float], group: str,
                manifest: dict[str, dict[str, Any]], *, repeats: int = SIGN_FLIP_REPEATS,
                seed: int = SEED) -> float:
    families = sorted(values)
    observed = abs(_group_point(values, group, manifest))
    rng = np.random.default_rng(seed)
    extreme = 0
    vector = np.asarray([values[family] for family in families], dtype=float)
    for _ in range(repeats):
        signs = rng.choice((-1.0, 1.0), size=len(families))
        permuted = {family: float(value) for family, value in zip(families, vector * signs)}
        extreme += int(abs(_group_point(permuted, group, manifest)) >= observed)
    return float((extreme + 1) / (repeats + 1))


def _lookup(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    result = {(str(row["case_id"]), int(row["window_start"]), str(row["condition"])): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate PANL2CLE trial cells")
    return result


def paired_metric_table(rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]],
                        metrics: Sequence[str], *, repeats: int = SIGN_FLIP_REPEATS,
                        seed: int = SEED) -> list[dict[str, Any]]:
    bootstrap = FamilyBootstrap(manifest_rows, repeats, seed)
    manifest = {str(row["family_id"]): row for row in manifest_rows}
    lookup = _lookup(rows)
    output = []
    for window in WINDOWS:
        main = [row for row in rows if row["condition"] == CONDITIONS[0]
                and int(row["window_start"]) == window[0]]
        for metric_index, metric in enumerate(metrics):
            main_values = {str(row["family_id"]): float(row[metric]) for row in main}
            control_values = {
                str(row["family_id"]): float(lookup[row["case_id"], window[0], CONDITIONS[1]][metric])
                for row in main
            }
            differences = {family: main_values[family] - control_values[family] for family in main_values}
            for group_index, group in enumerate(GROUPS):
                main_stats = bootstrap.aggregate(main_values, group)
                control_stats = bootstrap.aggregate(control_values, group)
                paired_stats = bootstrap.aggregate(differences, group)
                output.append({
                    "window_name": WINDOW_NAMES[window], "window_start": window[0], "window_end": window[1],
                    "metric": metric, "group": group,
                    "main_mean": main_stats["mean"], "main_ci95_low": main_stats["ci_low"],
                    "main_ci95_high": main_stats["ci_high"],
                    "control_mean": control_stats["mean"], "control_ci95_low": control_stats["ci_low"],
                    "control_ci95_high": control_stats["ci_high"],
                    "paired_difference": paired_stats["mean"],
                    "paired_ci95_low": paired_stats["ci_low"], "paired_ci95_high": paired_stats["ci_high"],
                    "paired_family_count": paired_stats["family_count"],
                    "paired_answer_count": paired_stats["answer_count"],
                    "sign_flip_p": sign_flip_p(
                        differences, group, manifest, repeats=repeats,
                        seed=seed + window[0] * 101 + metric_index * 17 + group_index,
                    ) if group == PRIMARY_GROUP else None,
                    "bh_fdr_q": None, "sign_flip_repeats": repeats,
                })
    for metric in metrics:
        primary = sorted((row for row in output if row["metric"] == metric
                          and row["group"] == PRIMARY_GROUP), key=lambda row: row["window_start"])
        for row, q_value in zip(primary, bh_fdr([float(row["sign_flip_p"]) for row in primary])):
            row["bh_fdr_q"] = q_value
    return output


def _direction_values(rows: list[dict[str, Any]], lookup: dict[tuple[str, int, str], dict[str, Any]],
                      window: tuple[int, int], condition: str, statistic: str) -> dict[str, float]:
    selected = [row for row in rows if row["condition"] == CONDITIONS[0]
                and int(row["window_start"]) == window[0]]
    output = {}
    for main in selected:
        row = main if condition == CONDITIONS[0] else lookup[main["case_id"], window[0], condition]
        state = row["sa_direction"]
        if statistic.endswith("_rate"):
            expected = {"SA_up_rate": "SA_up", "SA_down_rate": "SA_down", "SA_stable_rate": "SA_stable"}[statistic]
            output[str(row["family_id"])] = float(state == expected)
        elif statistic == "SA_up_mean_delta" and state == "SA_up":
            output[str(row["family_id"])] = float(row["delta_soft_sa"])
        elif statistic == "SA_down_mean_delta" and state == "SA_down":
            output[str(row["family_id"])] = float(row["delta_soft_sa"])
    return output


def direction_table(rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]],
                    *, repeats: int = SIGN_FLIP_REPEATS, seed: int = SEED) -> list[dict[str, Any]]:
    bootstrap = FamilyBootstrap(manifest_rows, repeats, seed)
    lookup = _lookup(rows); output = []
    statistics = ("SA_up_rate", "SA_down_rate", "SA_stable_rate",
                  "SA_up_mean_delta", "SA_down_mean_delta")
    for window in WINDOWS:
        for statistic in statistics:
            main = _direction_values(rows, lookup, window, CONDITIONS[0], statistic)
            control = _direction_values(rows, lookup, window, CONDITIONS[1], statistic)
            paired_families = sorted(set(main) & set(control))
            paired = {family: main[family] - control[family] for family in paired_families}
            for group in GROUPS:
                main_stats = bootstrap.aggregate(main, group); control_stats = bootstrap.aggregate(control, group)
                paired_stats = bootstrap.aggregate(paired, group)
                output.append({
                    "window_name": WINDOW_NAMES[window], "window_start": window[0], "window_end": window[1],
                    "statistic": statistic, "group": group, "epsilon": SA_STABLE_EPSILON,
                    "main_value": main_stats["mean"], "main_ci95_low": main_stats["ci_low"],
                    "main_ci95_high": main_stats["ci_high"], "main_family_count": main_stats["family_count"],
                    "control_value": control_stats["mean"], "control_ci95_low": control_stats["ci_low"],
                    "control_ci95_high": control_stats["ci_high"], "control_family_count": control_stats["family_count"],
                    "paired_difference": paired_stats["mean"],
                    "paired_ci95_low": paired_stats["ci_low"], "paired_ci95_high": paired_stats["ci_high"],
                    "paired_family_count": paired_stats["family_count"],
                    "pairing_note": ("within-case indicator difference" if statistic.endswith("_rate")
                                     else "conditional means; paired subset contains cases in the same direction under both conditions"),
                })
    return output


def analyze_statistics(*, output_root: Path | None = None,
                       epsilon: float = SA_STABLE_EPSILON,
                       repeats: int = SIGN_FLIP_REPEATS) -> dict[str, Any]:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    root = Path(output_root or default_output("PANL2CLE")).resolve()
    trials_path = root / "artifacts" / "trials.jsonl"
    manifest_path = root / "artifacts" / "manifests" / "test_manifest.jsonl"
    manifest = load_jsonl(manifest_path)
    physical = load_jsonl(trials_path)
    blocked = [row for row in physical if row["condition"] in CONDITIONS]
    expected = len(manifest) * len(WINDOWS) * len(CONDITIONS)
    if len(manifest) != 174 or len(blocked) != expected:
        raise RuntimeError(f"PANL2CLE formal grid is incomplete: manifest={len(manifest)}, blocked={len(blocked)}/{expected}")
    enriched = [enrich_trial(row, epsilon) for row in blocked]
    tables = root / "tables"
    atomic_csv(tables / "panl2cle_supplement_case_level.csv", enriched)
    absolute = paired_metric_table(enriched, manifest, ("abs_delta_soft_sa",), repeats=repeats)
    distribution = paired_metric_table(
        enriched, manifest, ("js_divergence_nats", "total_variation"), repeats=repeats
    )
    hard = paired_metric_table(
        enriched, manifest,
        ("abs_ordinal_shift", "hard_shift_up", "hard_shift_down",
         "large_ordinal_shift_rate", "abs_hard_midpoint_shift"), repeats=repeats,
    )
    directions = direction_table(enriched, manifest, repeats=repeats)
    atomic_csv(tables / "paired_absolute_effect.csv", absolute)
    atomic_csv(tables / "sa_direction_summary.csv", directions)
    atomic_csv(tables / "distribution_distance_summary.csv", distribution)
    atomic_csv(tables / "hard_shift_summary.csv", hard)
    readme = f"""# PANL2CLE 补充统计（前四项）

- `panl2cle_supplement_case_level.csv`：逐 case/window/condition 的方向分类、JS/TV 和 hard ordinal shift。
- `paired_absolute_effect.csv`：主要量 `|ΔSA_PANL| - |ΔSA_PANL+1|`；95% paired family-bootstrap CI、双侧 sign-flip p、四窗口 BH-FDR q。
- `sa_direction_summary.csv`：以 epsilon={epsilon:g} 定义 SA_up/down/stable，报告比例、条件内 up/down 平均变化及主对照差异。条件均值的 paired subset 只含两条件方向一致的 case。
- `distribution_distance_summary.csv`：九类概率的 Jensen-Shannon divergence（自然对数，单位 nats）和 total variation，以及主阻断减控制的配对检验。
- `hard_shift_summary.csv`：hard class 的绝对 ordinal shift、上/下移率、至少跨两级的变化率和 midpoint shift。

所有 CI 使用 seed=42 的 {repeats} 次 family bootstrap。双侧 sign-flip p 和 BH-FDR q 只用于主要 answer_equal_macro；该口径排除 blue 并对其余 11 个答案等权。
"""
    (tables / "README_panl2cle_statistics_zh.md").write_text(readme, encoding="utf-8")
    result = {
        "status": "complete", "experiment": "PANL2CLE", "scope": "statistics_1_to_4",
        "gpu_forwards": 0, "case_count": len(manifest), "blocked_trial_count": len(enriched),
        "epsilon": epsilon, "bootstrap_repeats": repeats, "sign_flip_repeats": repeats,
        "source_trials_sha256": sha256_file(trials_path),
        "outputs": [
            "tables/panl2cle_supplement_case_level.csv", "tables/paired_absolute_effect.csv",
            "tables/sa_direction_summary.csv", "tables/distribution_distance_summary.csv",
            "tables/hard_shift_summary.csv", "tables/README_panl2cle_statistics_zh.md",
        ],
    }
    atomic_json(root / "statistics_1_to_4_completion.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--epsilon", type=float, default=SA_STABLE_EPSILON)
    parser.add_argument("--repeats", type=int, default=SIGN_FLIP_REPEATS)
    args = parser.parse_args(argv)
    print(json.dumps(analyze_statistics(output_root=args.output_root, epsilon=args.epsilon,
                                        repeats=args.repeats), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
