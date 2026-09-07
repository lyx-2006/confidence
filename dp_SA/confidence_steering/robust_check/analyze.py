from __future__ import annotations

import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import joblib
from scipy.linalg import subspace_angles
from scipy.stats import pearsonr, spearmanr

from .config import BOOTSTRAP_REPEATS, DIRECTIONS
from .io_utils import atomic_csv, atomic_json, atomic_text, canonical_hash, load_jsonl
from .rebuild import signed_cosine

ENDPOINTS = {
    "final_soft_sa": "direct_final_endpoint",
    "anchor_panl_sa": "anchor_frozen_probe",
    "anchor_lat_confidence": "anchor_frozen_probe",
    "seed_panl_sa": "seed_specific_probe",
    "seed_lat_confidence": "seed_specific_probe",
}


def shared_family_draws(families: Sequence[str], repeats: int = BOOTSTRAP_REPEATS) -> tuple[list[list[str]], str]:
    ordered = sorted(set(map(str, families)))
    if not ordered:
        raise ValueError("Cannot bootstrap zero families")
    rng = np.random.default_rng(np.random.SeedSequence([42, 2000, len(ordered)]))
    draws = [[ordered[index] for index in rng.integers(0, len(ordered), len(ordered))] for _ in range(repeats)]
    return draws, canonical_hash(draws)


def _aggregate(rows: Sequence[dict[str, Any]], group: str) -> float:
    if not rows:
        return math.nan
    if group == "all":
        return float(np.mean([row["effect"] for row in rows]))
    key = "fixed_answer" if group == "answer_equal_macro" else "family_id"
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[str(row[key])].append(float(row["effect"]))
    return float(np.mean([np.mean(values) for values in buckets.values()]))


def summarize(rows: Sequence[dict[str, Any]], group: str, draws: Sequence[Sequence[str]]) -> dict[str, Any]:
    mean = _aggregate(rows, group)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family_id"])].append(row)
    samples = []
    for draw in draws:
        sampled = [dict(row) for family in draw for row in by_family.get(str(family), [])]
        value = _aggregate(sampled, group)
        if math.isfinite(value):
            samples.append(value)
    low, high = (np.percentile(samples, [2.5, 97.5]) if samples else (math.nan, math.nan))
    return {"mean_effect": mean, "ci95_low": float(low), "ci95_high": float(high), "bootstrap_valid": len(samples), "n_cases": len(rows), "n_families": len(by_family)}


def casewise_effects(trials: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in trials:
        if row["direction"] == "shared_alpha_zero":
            continue
        grouped[(int(row["seed"]), str(row["case_id"]), str(row["direction"]))][float(row["alpha"])] = row
    output = []
    for (seed, case, direction), values in sorted(grouped.items()):
        if set(values) != {-0.5, 0.5}:
            raise RuntimeError(f"Missing symmetric pair: {(seed, case, direction)}")
        minus, plus = values[-0.5], values[0.5]
        for endpoint, scale in ENDPOINTS.items():
            output.append({
                "seed": seed, "case_id": case, "family_id": str(plus["family_id"]),
                "item_id": str(plus["item_id"]), "fixed_answer": str(plus["fixed_answer"]),
                "direction": direction, "endpoint": endpoint, "measurement_scale": scale,
                "effect": (float(plus[endpoint]) - float(minus[endpoint])) / 2.0,
            })
    return output


def _random_cosines(repeats: int = 2000, dimension: int = 3584) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([42, 14, 731]))
    output = []
    for count in range(0, repeats, 100):
        size = min(100, repeats - count)
        left = rng.standard_normal((size, dimension)); right = rng.standard_normal((size, dimension))
        output.extend(np.sum(left * right, axis=1) / np.linalg.norm(left, axis=1) / np.linalg.norm(right, axis=1))
    return np.asarray(output, np.float64)


def _model_gradient(path: Path) -> np.ndarray:
    payload = joblib.load(path)
    if "raw_gradient" in payload:
        return np.asarray(payload["raw_gradient"], np.float64)
    model = payload["model"]
    return np.asarray(model.named_steps["ridge"].coef_ / model.named_steps["scaler"].scale_, np.float64)


def vector_stability(root: Path, seeds: Sequence[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    vectors, metadata, subspaces = {}, {}, {}
    for seed in seeds:
        direction_root = root / f"artifacts/directions/seed_{seed}"
        payload = np.load(direction_root / "P1_LAT__L14.npz")
        vectors[seed] = {key: np.asarray(payload[key]) for key in payload.files}
        payload.close()
        current = __import__("json").loads((direction_root / "vector_metadata.json").read_text())
        metadata[seed] = {(row["recipient_answer"], row["direction"]): row for row in current["vectors"]}
        current_subspaces = __import__("json").loads((direction_root / "subspaces.json").read_text())
        subspaces[seed] = {row["recipient_answer"]: row for row in current_subspaces}
    vector_rows = []
    random_values = _random_cosines()
    random_reference = {
        "seed": 42, "repeats": len(random_values), "dimension": 3584,
        "mean": float(random_values.mean()), "sd": float(random_values.std(ddof=1)),
        "q025": float(np.percentile(random_values, 2.5)), "q975": float(np.percentile(random_values, 97.5)),
        "sha256": __import__("hashlib").sha256(random_values.tobytes()).hexdigest(),
        "role": "geometric_reference_only_not_a_steering_result",
    }
    for seed in seeds:
        for (recipient, direction), row in sorted(metadata[seed].items()):
            vector_rows.append({
                "record_type": "direction_audit", "seed_a": seed, "seed_b": seed,
                "recipient_answer": recipient, "direction": direction, "signed_cosine": 1.0,
                "confidence_probe_dot_raw": row["confidence_probe_dot_raw"],
                "confidence_probe_dot_parallel": row["confidence_probe_dot_parallel"],
                "confidence_probe_dot_perpendicular": row["confidence_probe_dot_perpendicular"],
                "direction_sign_inconsistent": row["direction_sign_inconsistent"],
                "direction_sign_degenerate": row["direction_sign_degenerate"],
                "relative_norm": row["relative_norm"],
                "natural_projection_std": row["natural_projection_std"],
            })
    for left, right in itertools.combinations(seeds, 2):
        for recipient, direction in sorted(metadata[left]):
            key = f"{recipient}__{direction}__scaled"
            a, b = vectors[left][key], vectors[right][key]
            vector_rows.append({
                "record_type": "seed_pair", "seed_a": left, "seed_b": right,
                "recipient_answer": recipient, "direction": direction,
                "signed_cosine": signed_cosine(a, b),
                "relative_norm_difference": metadata[right][recipient, direction]["relative_norm"] - metadata[left][recipient, direction]["relative_norm"],
                "natural_projection_sd_difference": metadata[right][recipient, direction]["natural_projection_std"] - metadata[left][recipient, direction]["natural_projection_std"],
                "sign_alignment_applied": False,
                "random_cosine_mean": random_reference["mean"], "random_cosine_q025": random_reference["q025"],
                "random_cosine_q975": random_reference["q975"],
                "percentile_vs_random": 100.0 * float(np.mean(random_values < signed_cosine(a, b))),
            })
        probe_specs = (
            ("confidence_probe_gradient", root / f"artifacts/probes/seed_{left}/confidence_gap__P1_LAT__L14__full.joblib", root / f"artifacts/probes/seed_{right}/confidence_gap__P1_LAT__L14__full.joblib", None),
            ("panl_final_sa_probe_gradient", root / f"artifacts/probes/seed_{left}/final_sa__P1_PANL__L18__full.joblib", root / f"artifacts/probes/seed_{right}/final_sa__P1_PANL__L18__full.joblib", None),
        )
        for label, left_path, right_path, recipient in probe_specs:
            vector_rows.append({"record_type": "probe_gradient_seed_pair", "seed_a": left, "seed_b": right, "recipient_answer": recipient, "direction": label, "signed_cosine": signed_cosine(_model_gradient(left_path), _model_gradient(right_path)), "sign_alignment_applied": False})
        for recipient in sorted(subspaces[left]):
            left_path = root / f"artifacts/probes/seed_{left}/construction_lat_sa__{recipient}__L14.joblib"
            right_path = root / f"artifacts/probes/seed_{right}/construction_lat_sa__{recipient}__L14.joblib"
            vector_rows.append({"record_type": "probe_gradient_seed_pair", "seed_a": left, "seed_b": right, "recipient_answer": recipient, "direction": "recipient_excluded_final_sa_gradient", "signed_cosine": signed_cosine(_model_gradient(left_path), _model_gradient(right_path)), "sign_alignment_applied": False})
    angle_rows = []
    for left, right in itertools.combinations(seeds, 2):
        for recipient in sorted(subspaces[left]):
            qa = vectors[left][subspaces[left][recipient]["basis_key"]]
            qb = vectors[right][subspaces[right][recipient]["basis_key"]]
            angles = np.degrees(subspace_angles(qa.astype(np.float64), qb.astype(np.float64)))
            angle_rows.append({
                "seed_a": left, "seed_b": right, "recipient_answer": recipient,
                "rank_a": qa.shape[1], "rank_b": qb.shape[1],
                "principal_angles_degrees": ";".join(f"{value:.12g}" for value in angles),
                "max_principal_angle_degrees": float(np.max(angles)),
                "mean_principal_angle_degrees": float(np.mean(angles)),
            })
    return vector_rows, angle_rows, random_reference


def _effect_tables(effects: Sequence[dict[str, Any]], draws: Sequence[Sequence[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    seeds = sorted({int(row["seed"]) for row in effects})
    for seed in seeds:
        for direction in DIRECTIONS:
            for endpoint, scale in ENDPOINTS.items():
                selected = [row for row in effects if row["seed"] == seed and row["direction"] == direction and row["endpoint"] == endpoint]
                for group in ("answer_equal_macro", "family_micro", "all"):
                    summaries.append({"record_type": "seed_effect", "seed": seed, "direction": direction, "endpoint": endpoint, "measurement_scale": scale, "group": group, **summarize(selected, group, draws)})
    for left, right in itertools.combinations(seeds, 2):
        for direction in DIRECTIONS:
            for endpoint, scale in ENDPOINTS.items():
                left_rows = {row["case_id"]: row for row in effects if row["seed"] == left and row["direction"] == direction and row["endpoint"] == endpoint}
                right_rows = {row["case_id"]: row for row in effects if row["seed"] == right and row["direction"] == direction and row["endpoint"] == endpoint}
                contrast = [{**right_rows[case], "effect": right_rows[case]["effect"] - left_rows[case]["effect"]} for case in sorted(set(left_rows) & set(right_rows))]
                summaries.append({"record_type": "paired_seed_contrast", "seed": f"{right}-{left}", "direction": direction, "endpoint": endpoint, "measurement_scale": scale, "group": "answer_equal_macro", **summarize(contrast, "answer_equal_macro", draws)})
    if len(seeds) > 1:
        seed_rows = [row for row in summaries if row["record_type"] == "seed_effect" and row["group"] == "answer_equal_macro"]
        for direction in DIRECTIONS:
            for endpoint, scale in ENDPOINTS.items():
                selected = [row for row in seed_rows if row["direction"] == direction and row["endpoint"] == endpoint]
                values = np.asarray([row["mean_effect"] for row in selected], np.float64)
                reference_sign = np.sign(next(row["mean_effect"] for row in selected if int(row["seed"]) == 42))
                summaries.append({
                    "record_type": "across_seed_stability", "seed": "42-45", "direction": direction,
                    "endpoint": endpoint, "measurement_scale": scale, "group": "answer_equal_macro",
                    "across_seed_mean": float(values.mean()), "across_seed_sd": float(values.std(ddof=1)),
                    "same_sign_as_seed42_fraction": float(np.mean(np.sign(values) == reference_sign)),
                    "all_seeds_same_sign": bool(len(set(np.sign(values))) == 1), "seed_count": len(values),
                })
    reproducibility = []
    for left, right in itertools.combinations(seeds, 2):
        for direction in DIRECTIONS:
            for endpoint in ENDPOINTS:
                a = {row["case_id"]: row["effect"] for row in effects if row["seed"] == left and row["direction"] == direction and row["endpoint"] == endpoint}
                b = {row["case_id"]: row["effect"] for row in effects if row["seed"] == right and row["direction"] == direction and row["endpoint"] == endpoint}
                common = sorted(set(a) & set(b)); av = np.asarray([a[key] for key in common]); bv = np.asarray([b[key] for key in common])
                reproducibility.append({
                    "seed_a": left, "seed_b": right, "direction": direction, "endpoint": endpoint,
                    "n_cases": len(common),
                    "pearson": float(pearsonr(av, bv).statistic) if len(common) > 1 and np.ptp(av) and np.ptp(bv) else None,
                    "spearman": float(spearmanr(av, bv).statistic) if len(common) > 1 and np.ptp(av) and np.ptp(bv) else None,
                })
    additivity = []
    indexed = {(row["seed"], row["case_id"], row["endpoint"], row["direction"]): row for row in effects}
    for seed in seeds:
        cases = sorted({row["case_id"] for row in effects if row["seed"] == seed})
        for case in cases:
            for endpoint in ENDPOINTS:
                raw = indexed[seed, case, endpoint, "confidence_raw"]
                parallel = indexed[seed, case, endpoint, "confidence_parallel_sa"]
                perpendicular = indexed[seed, case, endpoint, "confidence_perp_sa_natural_scale"]
                additivity.append({
                    "seed": seed, "case_id": case, "family_id": raw["family_id"], "endpoint": endpoint,
                    "raw_effect": raw["effect"], "parallel_effect": parallel["effect"],
                    "perpendicular_effect": perpendicular["effect"],
                    "additivity_error": raw["effect"] - parallel["effect"] - perpendicular["effect"],
                })
    return summaries, reproducibility, additivity


def _plot(root: Path, vector_rows: Sequence[dict[str, Any]], effects: Sequence[dict[str, Any]], summaries: Sequence[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [row for row in vector_rows if row["record_type"] == "seed_pair"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for index, direction in enumerate(DIRECTIONS):
        values = [row["signed_cosine"] for row in pairs if row["direction"] == direction]
        if values:
            axes[0].boxplot(values, positions=[index], widths=.6)
    axes[0].set_xticks(range(3), ["raw", "parallel", "perp"]); axes[0].set_ylabel("signed cosine")
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[1].axis("off"); axes[1].text(.02, .95, "No post-hoc sign alignment\nNegative cosine is retained as instability.", va="top")
    fig.tight_layout(); fig.savefig(root / "figures/vector_stability.png", dpi=180); plt.close(fig)

    for endpoint, filename, ylabel in (
        ("final_soft_sa", "final_sa_stability.png", "symmetric effect: final soft SA"),
        ("anchor_panl_sa", "panl_sa_stability.png", "symmetric effect: frozen PANL-SA probe"),
    ):
        selected = [row for row in summaries if row["record_type"] == "seed_effect" and row["endpoint"] == endpoint and row["group"] == "answer_equal_macro"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for direction in DIRECTIONS:
            rows = sorted((row for row in selected if row["direction"] == direction), key=lambda row: int(row["seed"]))
            if rows:
                x = [int(row["seed"]) for row in rows]; y = [row["mean_effect"] for row in rows]
                low = [value - row["ci95_low"] for value, row in zip(y, rows)]; high = [row["ci95_high"] - value for value, row in zip(y, rows)]
                ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=3, label=direction)
        ax.axhline(0, color="black", linewidth=.7); ax.set_xlabel("split seed"); ax.set_ylabel(ylabel); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(root / f"figures/{filename}", dpi=180); plt.close(fig)


def analyze(root: Path, seeds: Sequence[int], *, bootstrap_repeats: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    trials = load_jsonl(root / "artifacts/trials/trials.jsonl")
    effects = casewise_effects(trials)
    families = sorted({str(row["family_id"]) for row in effects})
    draws, draw_hash = shared_family_draws(families, bootstrap_repeats)
    atomic_json(root / "artifacts/diagnostics/shared_family_bootstrap.json", {"seed": 42, "repeats": bootstrap_repeats, "families": families, "draw_sha256": draw_hash, "draws": draws})
    vector_rows, angle_rows, random_reference = vector_stability(root, seeds)
    summaries, reproducibility, additivity = _effect_tables(effects, draws)
    probe_metrics = []
    split_rows = []
    for seed in seeds:
        probe_metrics.extend(__import__("json").loads((root / f"artifacts/probes/seed_{seed}/metrics.json").read_text()))
        audit = __import__("json").loads((root / f"artifacts/splits/seed_{seed}/split_audit.json").read_text())
        split_rows.extend({"seed": seed, **fold, "assignment_sha256": audit["assignment_sha256"]} for fold in audit["folds"])
    atomic_csv(root / "tables/split_audit.csv", split_rows)
    atomic_csv(root / "tables/probe_metrics_by_seed.csv", probe_metrics)
    atomic_csv(root / "tables/vector_stability.csv", vector_rows)
    atomic_csv(root / "tables/subspace_principal_angles.csv", angle_rows)
    atomic_csv(root / "tables/steering_effects_by_seed.csv", summaries)
    atomic_csv(root / "tables/casewise_effect_reproducibility.csv", reproducibility)
    atomic_csv(root / "tables/component_additivity.csv", additivity)
    atomic_csv(root / "artifacts/diagnostics/casewise_symmetric_effects.csv", effects)
    atomic_json(root / "artifacts/diagnostics/random_direction_cosine_reference.json", random_reference)
    geometry_function = []
    for left, right in itertools.combinations(seeds, 2):
        for direction in DIRECTIONS:
            geometry = [row["signed_cosine"] for row in vector_rows if row["record_type"] == "seed_pair" and row["seed_a"] == left and row["seed_b"] == right and row["direction"] == direction]
            for endpoint in ENDPOINTS:
                functional = next((row for row in reproducibility if row["seed_a"] == left and row["seed_b"] == right and row["direction"] == direction and row["endpoint"] == endpoint), None)
                geometry_function.append({"seed_a": left, "seed_b": right, "direction": direction, "endpoint": endpoint, "mean_recipient_vector_cosine": float(np.mean(geometry)), "casewise_effect_pearson": functional["pearson"] if functional else None, "casewise_effect_spearman": functional["spearman"] if functional else None})
    atomic_csv(root / "artifacts/diagnostics/geometry_function_stability.csv", geometry_function)
    _plot(root, vector_rows, effects, summaries)
    inconsistent = [row for row in vector_rows if row["record_type"] == "direction_audit" and row["direction"] == "confidence_raw" and row["direction_sign_inconsistent"]]
    null_root = Path(__file__).resolve().parent.parent / "output/natural_decomposition/tables/random_sa_subspace_null"
    null_context = {
        "role": "historical_context_only",
        "reason": "historical random-SA null used dose ±2 and an older runtime processor path; it is not a matched low-dose formal control",
        "tables": {path.name: {"path": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()} for path in sorted(null_root.glob("*.csv"))},
    }
    atomic_json(root / "artifacts/diagnostics/historical_random_sa_null_context.json", null_context)
    primary = [row for row in summaries if row["record_type"] == "seed_effect" and row["group"] == "answer_equal_macro"]
    def values(direction: str, endpoint: str) -> list[float]:
        return [float(row["mean_effect"]) for row in primary if row["direction"] == direction and row["endpoint"] == endpoint]
    verdict = {
        "status": "descriptive_only" if len(seeds) < 4 else "evaluated",
        "direction_sign_consistency_pass": not inconsistent,
        "raw_increases_lat_confidence_all_seeds": bool(values("confidence_raw", "anchor_lat_confidence")) and all(value > 0 for value in values("confidence_raw", "anchor_lat_confidence")),
        "parallel_positive_final_sa_majority": sum(value > 0 for value in values("confidence_parallel_sa", "final_soft_sa")) > len(values("confidence_parallel_sa", "final_soft_sa")) / 2,
        "parallel_positive_panl_anchor_majority": sum(value > 0 for value in values("confidence_parallel_sa", "anchor_panl_sa")) > len(values("confidence_parallel_sa", "anchor_panl_sa")) / 2,
        "perpendicular_negative_panl_anchor_all_seeds": bool(values("confidence_perp_sa_natural_scale", "anchor_panl_sa")) and all(value < 0 for value in values("confidence_perp_sa_natural_scale", "anchor_panl_sa")),
        "formal_seed_count": len(seeds), "adverse_results_never_block_engineering_completion": True,
    }
    atomic_json(root / "artifacts/diagnostics/stability_verdict.json", verdict)
    readme = f"""# Confidence方向与steering训练划分稳定性结果

本报告是 **fixed-evaluation-set split-seed stability analysis**。更换family划分是为了检验训练数据划分变化后方向、SA子空间及功能效应是否保持；seed 42复建衡量工程复现，seed 43–45衡量划分稳定性。100-case formal集合曾被历史实验使用，因此不是全新独立测试集。

方向符号只由 `G_L=L_i-L_t` 固定。分析没有按probe响应或seed 42事后翻转方向，pairwise cosine也是有符号值。发现 {len(inconsistent)} 个raw方向的 `confidence_probe_dot_raw < 0`；这些记录保留原向量，并作为科学稳定性失败，而非被代码纠正。

主要跨seed标尺是direct final soft SA与冻结PANL L18 SA probe。seed-specific probe仅作敏感性分析，因为不同seed训练的probe不是完全相同的测量尺。probe可解码某个量不等于模型必然在决策中使用该表征；final soft SA是直接行为endpoint。历史random-SA-subspace null仅作背景参照：其剂量为±2且来自旧runtime processor路径，不能冒充本实验±0.5的匹配对照。

稳定结果应表现为：有符号vector cosine较高、SA principal angles较小、四seed主要效应同向且casewise相关较高。若效应只在seed 42出现、替代seed反向，或raw方向出现负probe-dot，则说明结论依赖原始划分。

原始trial位于 `artifacts/trials/`，probe位于 `artifacts/probes/seed_*`，方向和basis位于 `artifacts/directions/seed_*`；汇总表和三张主图分别位于 `tables/` 与 `figures/`。
"""
    atomic_text(root / "README_RESULTS_zh.md", readme)
    result = {"status": "complete", "seeds": list(seeds), "trial_rows": len(trials), "casewise_effect_rows": len(effects), "bootstrap_draw_sha256": draw_hash, "direction_sign_inconsistent_count": len(inconsistent), "stability_verdict": verdict}
    atomic_json(root / "progress/analyze.json", result)
    return result
