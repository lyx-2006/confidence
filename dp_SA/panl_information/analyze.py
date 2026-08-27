from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from .config import ARTIFACT_NAMES, BOOTSTRAP_REPEATS, LAYERS, PARAMETERIZATION_TOLERANCE, POSITIONS, RESULTS_ROOT, SEED
from .io_utils import atomic_json, atomic_text, ensure_output_layout, load_jsonl, stage_update
from .metrics import calibration_metrics, clustered_ols, item_bootstrap, safe_correlation
from .probe_utils import stable_seed


def _ci(value: Any) -> str:
    if not value or value.get("lower") is None: return ""
    return f"[{float(value['lower']):.10g}, {float(value['upper']):.10g}]"


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows); atomic_text(path, buffer.getvalue())


def _bootstrap_coefficients(rows: Sequence[dict[str, Any]], outcome: str, parameterization: str, repeats: int, seed: int) -> list[dict[str, Any]]:
    items = sorted({str(row["item_id"]) for row in rows}); buckets = {item: [row for row in rows if str(row["item_id"]) == item] for item in items}
    rng = np.random.default_rng(seed); values: list[np.ndarray] = []
    for _ in range(repeats):
        sample = [row for item in rng.choice(items, len(items), replace=True) for row in buckets[str(item)]]
        X, _ = _design(sample, parameterization); y = np.asarray([float(row[outcome]) for row in sample])
        try: beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError: continue
        if np.isfinite(beta).all(): values.append(beta)
    array = np.asarray(values)
    if not len(array): return [{"lower": None, "upper": None, "valid_repeats": 0} for _ in range(_design(rows, parameterization)[0].shape[1])]
    bounds = np.percentile(array, [2.5, 97.5], axis=0)
    return [{"lower": float(bounds[0, index]), "upper": float(bounds[1, index]), "valid_repeats": len(array)} for index in range(array.shape[1])]


def _design(rows: Sequence[dict[str, Any]], parameterization: str) -> tuple[np.ndarray, list[str]]:
    include_hard = len({int(row["Hard"]) for row in rows}) > 1
    if parameterization == "gap_overall":
        names = ["intercept", "G", "U"] + (["Hard"] if include_hard else [])
    else:
        names = ["intercept", "d_text", "d_image"] + (["Hard"] if include_hard else [])
    return np.asarray([[1.0, *[float(row[name]) for name in names[1:]]] for row in rows], dtype=float), names


def fit_parameterizations(rows: Sequence[dict[str, Any]], outcome: str, *, repeats: int, seed: int) -> dict[str, Any]:
    results = {}
    y = np.asarray([float(row[outcome]) for row in rows]); groups = [str(row["item_id"]) for row in rows]
    for offset, parameterization in enumerate(("gap_overall", "direct")):
        X, names = _design(rows, parameterization); fit = clustered_ols(X, y, groups)
        intervals = _bootstrap_coefficients(rows, outcome, parameterization, repeats, seed + offset)
        coefficients = []
        for index, name in enumerate(names):
            standardized = None if name == "intercept" or float(y.std(ddof=0)) == 0 else float(fit["coefficient"][index] * X[:, index].std(ddof=0) / y.std(ddof=0))
            coefficients.append({"name": name, "estimate": float(fit["coefficient"][index]), "standard_error": float(fit["standard_error"][index]), "ci": intervals[index], "p_value": float(fit["p_value"][index]), "standardized_coefficient": standardized})
        results[parameterization] = {"names": names, "coefficients": coefficients, "r2": fit["r2"], "adjusted_r2": fit["adjusted_r2"], "sample_count": len(rows), "item_count": len(set(groups)), "fitted": fit["fitted"], "residual": fit["residual"]}
    gap = results["gap_overall"]; direct = results["direct"]
    gap_map = {row["name"]: row["estimate"] for row in gap["coefficients"]}; direct_map = {row["name"]: row["estimate"] for row in direct["coefficients"]}
    mapping = {
        "gamma_t_expected": gap_map["G"] + gap_map["U"] / 2.0,
        "gamma_i_expected": -gap_map["G"] + gap_map["U"] / 2.0,
        "gamma_t_observed": direct_map["d_text"], "gamma_i_observed": direct_map["d_image"],
        "max_fitted_difference": float(np.max(np.abs(gap["fitted"] - direct["fitted"]))),
        "max_residual_difference": float(np.max(np.abs(gap["residual"] - direct["residual"]))),
        "r2_difference": abs(float(gap["r2"]) - float(direct["r2"])),
    }
    mapping["status"] = "passed" if max(mapping["max_fitted_difference"], mapping["max_residual_difference"], mapping["r2_difference"], abs(mapping["gamma_t_expected"] - mapping["gamma_t_observed"]), abs(mapping["gamma_i_expected"] - mapping["gamma_i_observed"])) <= PARAMETERIZATION_TOLERANCE else "failed"
    if mapping["status"] != "passed": raise ValueError(f"Regression parameterization equivalence failed: {mapping}")
    for value in results.values(): value.pop("fitted"); value.pop("residual")
    return {**results, "mapping_audit": mapping}


def _correlations(rows: Sequence[dict[str, Any]], *, repeats: int) -> dict[str, Any]:
    groups = {
        "all": list(rows), "easy": [row for row in rows if not row["Hard"]], "hard": [row for row in rows if row["Hard"]],
        "follow_text": [row for row in rows if row["decision_side"] == "follow_text"], "follow_image": [row for row in rows if row["decision_side"] == "follow_image"],
    }
    factors = {"D_text": "text_model_perceived_difficulty", "D_image": "image_model_perceived_difficulty", "G": "G", "U": "U"}
    outcomes = {"final_sa": "final_sa", "panl_sa": "panl_l14_oof_sa_prediction"}; result = {}
    for group, selected in groups.items():
        result[group] = {}
        for outcome_name, outcome in outcomes.items():
            result[group][outcome_name] = {}
            for factor_name, factor in factors.items():
                entry = {}
                for kind in ("pearson", "spearman"):
                    value = safe_correlation(kind, [row[factor] for row in selected], [row[outcome] for row in selected])
                    def statistic(sample: list[dict[str, Any]], k: str = kind, f: str = factor, o: str = outcome) -> float:
                        return safe_correlation(k, [row[f] for row in sample], [row[o] for row in sample])
                    entry[kind] = value; entry[f"{kind}_ci"] = item_bootstrap(selected, statistic, repeats=repeats, seed=stable_seed(SEED, group, outcome_name, factor_name, kind))
                entry.update({"sample_count": len(selected), "item_count": len({str(row['item_id']) for row in selected})}); result[group][outcome_name][factor_name] = entry
    return result


def _wide_probe_tables(root: Path, difficulty: dict[str, Any], decision: dict[str, Any]) -> None:
    columns = [f"{position}_L{layer}" for position in POSITIONS for layer in LAYERS]
    dmap = {(row["target"], row["position"], int(row["layer"])): row for row in difficulty["metrics"]}
    drows = []
    for target in ("text", "image"):
        for metric in ("r2", "r2_ci", "spearman", "pearson", "mae", "sample_count"):
            output = {"metric": f"{target}_{metric}"}
            for position in POSITIONS:
                for layer in LAYERS:
                    cell = dmap[(target, position, layer)]; value = _ci(cell[metric]) if metric.endswith("_ci") else cell[metric]
                    output[f"{position}_L{layer}"] = value
            drows.append(output)
    _atomic_csv(root / "tables" / "difficulty_probe.csv", ["metric", *columns], drows)
    pmap = {(row["position"], int(row["layer"])): row for row in decision["metrics"]}; prows = []
    for metric in ("balanced_accuracy", "balanced_accuracy_ci", "auroc", "auroc_ci", "accuracy", "macro_f1", "log_loss", "majority_baseline", "answer_identity_baseline", "difficulty_only_baseline", "sample_count", "item_count"):
        output = {"metric": metric}
        for position in POSITIONS:
            for layer in LAYERS:
                cell = pmap[(position, layer)]; output[f"{position}_L{layer}"] = _ci(cell[metric]) if metric.endswith("_ci") else cell[metric]
        prows.append(output)
    _atomic_csv(root / "tables" / "decision_probe.csv", ["metric", *columns], prows)


def _correlation_table(root: Path, correlations: dict[str, Any], calibration: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    factors = ("D_text", "D_image", "G", "U"); output = []
    for outcome in ("final_sa", "panl_sa"):
        for group in ("all", "easy", "hard", "follow_text", "follow_image"):
            for kind in ("pearson", "spearman"):
                output.append({"metric": f"{outcome}_{kind}_{group}", **{factor: correlations[group][outcome][factor][kind] for factor in factors}})
                output.append({"metric": f"{outcome}_{kind}_ci_{group}", **{factor: _ci(correlations[group][outcome][factor][f'{kind}_ci']) for factor in factors}})
    output.append({"metric": "sample_count", **{factor: len(rows) for factor in factors}}); output.append({"metric": "item_count", **{factor: len({str(row['item_id']) for row in rows}) for factor in factors}})
    mapping = {"top1_accuracy": "restricted_top1_accuracy", "nll": "multiclass_nll", "brier": "brier_score", "ece": "max_probability_ece", "entropy_error_auroc": "difficulty_error_auroc", "entropy_correctness_spearman": "difficulty_correctness_spearman"}
    for label, key in mapping.items(): output.append({"metric": label, "D_text": calibration["text"][key], "D_image": calibration["image"][key], "G": "", "U": ""})
    _atomic_csv(root / "tables" / "sa_factor_correlations.csv", ["metric", *factors], output)


def _regression_markdown(regressions: dict[str, Any]) -> str:
    lines = ["# SA–难度回归参数", "", "模型A：`SA = β₀ + β₁G + β₂U + β₃Hard + ε`。模型B：`SA = γ₀ + γₜd_text + γᵢd_image + γ₃Hard + ε`；分层分析删除常量 Hard。", "", "变量中，G 为 text 相对 image 的难度差，U 为两侧总体难度；正 SA 表示更偏 image。标准误为 item-cluster robust，区间为 2,000 次 item bootstrap percentile CI。", ""]
    for outcome, groups in regressions.items():
        lines += [f"## Outcome：{outcome}", ""]
        for group, result in groups.items():
            lines += [f"### Group：{group}", ""]
            for name, label in (("gap_overall", "模型A"), ("direct", "模型B")):
                model = result[name]; lines += [f"#### {label}", "", "| 参数 | 估计 | 标准误 | 95% CI | p-value | 标准化系数 |", "|---|---:|---:|---:|---:|---:|"]
                for coefficient in model["coefficients"]:
                    lines.append(f"| {coefficient['name']} | {coefficient['estimate']:.8g} | {coefficient['standard_error']:.8g} | {_ci(coefficient['ci'])} | {coefficient['p_value']:.8g} | {'' if coefficient['standardized_coefficient'] is None else f'{coefficient['standardized_coefficient']:.8g}'} |")
                lines += ["", f"R²={model['r2']:.8g}；调整 R²={model['adjusted_r2']:.8g}；n={model['sample_count']}；item={model['item_count']}。", ""]
            audit = result["mapping_audit"]
            lines += [f"参数映射审计：γₜ={audit['gamma_t_observed']:.10g}（映射值 {audit['gamma_t_expected']:.10g}），γᵢ={audit['gamma_i_observed']:.10g}（映射值 {audit['gamma_i_expected']:.10g}）；fitted 最大差={audit['max_fitted_difference']:.3g}，状态={audit['status']}。", "", "科学解释：系数描述难度与 SA 的统计关系，不构成单模态难度被模型因果使用的证据。", ""]
    return "\n".join(lines)


def _plots(root: Path, difficulty: dict[str, Any], decision: dict[str, Any]) -> None:
    styles = {position: (color, marker) for position, color, marker in zip(POSITIONS, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"), ("o", "s", "^", "D", "v"), strict=True)}
    for metric, filename, ylabel in (("r2", "difficulty_probe_R2.png", "Pooled OOF R²"), ("spearman", "difficulty_probe_spearman.png", "Spearman correlation")):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for axis, target, title in zip(axes, ("text", "image"), ("Text difficulty", "Image difficulty"), strict=True):
            for position in POSITIONS:
                cells = sorted((row for row in difficulty["metrics"] if row["target"] == target and row["position"] == position), key=lambda row: row["layer"])
                color, marker = styles[position]; axis.plot([row["layer"] for row in cells], [row[metric] for row in cells], color=color, marker=marker, label=position)
            axis.axhline(0, color="black", linestyle="--", linewidth=1); axis.set_title(title); axis.set_xlabel("Zero-based decoder layer"); axis.set_ylabel(ylabel); axis.set_xticks(LAYERS); axis.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(root / "figures" / filename, dpi=300); plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for position in POSITIONS:
        cells = sorted((row for row in decision["metrics"] if row["position"] == position), key=lambda row: row["layer"]); color, marker = styles[position]
        label = f"{position} (answer-token positive control)" if position == "P1_LAT" else position
        axis.plot([row["layer"] for row in cells], [row["balanced_accuracy"] for row in cells], color=color, marker=marker, label=label)
        onset = decision["onsets"].get(position)
        if onset: axis.annotate(f"onset L{onset['layer']}", (onset["layer"], next(row["balanced_accuracy"] for row in cells if row["layer"] == onset["layer"])), fontsize=7)
    axis.axhline(float(decision["baselines"]["majority"]["balanced_accuracy"]), color="black", linestyle="--", label="outer-train majority baseline")
    axis.set_xlabel("Zero-based decoder layer"); axis.set_ylabel("Pooled OOF balanced accuracy"); axis.set_xticks(LAYERS); axis.legend(fontsize=7); fig.tight_layout(); fig.savefig(root / "figures" / "decision_probe_accuracy.png", dpi=300); plt.close(fig)


def analyze(root: Path, *, bootstrap: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    rows = load_jsonl(root / "artifacts" / ARTIFACT_NAMES["joined"]); unimodal = load_jsonl(root / "artifacts" / ARTIFACT_NAMES["unimodal"])
    difficulty = json.loads((root / "artifacts" / "difficulty_probe_metrics.json").read_text()); decision = json.loads((root / "artifacts" / "decision_probe_metrics.json").read_text())
    text_rows = [row for row in unimodal if row["modality"] == "text"]; image_rows = [row for row in unimodal if row["modality"] == "image"]
    calibration = {"text": calibration_metrics(text_rows, "text"), "image": calibration_metrics(image_rows, "image")}
    by_item = {}; paired = []
    for row in image_rows: by_item.setdefault(str(row["item_id"]), {})[row["condition"]] = row
    for item, values in by_item.items():
        if set(values) == {"conflict_easy", "conflict_hard"}: paired.append({"item_id": item, "difference_hard_minus_easy": float(values["conflict_hard"]["image_model_perceived_difficulty"] - values["conflict_easy"]["image_model_perceived_difficulty"])})
    paired_ci = item_bootstrap(paired, lambda sample: float(np.mean([row["difference_hard_minus_easy"] for row in sample])), repeats=bootstrap, seed=SEED)
    bins = sorted({str(row["prior_bin"]) for row in rows}, key=lambda value: float(value.split("-", 1)[0])); bin_summary = []
    for value in bins:
        selected = [row for row in text_rows if str(next(source["prior_bin"] for source in rows if str(source["item_id"]) == str(row["item_id"]) and int(source["prior_index"]) == int(row["prior_index"]))) == value]
        bin_summary.append({"prior_bin": value, "count": len(selected), "difficulty_mean": float(np.mean([row["text_model_perceived_difficulty"] for row in selected]))})
    trend = float(spearmanr([float(row["prior_bin"].split("-", 1)[0]) for row in bin_summary], [row["difficulty_mean"] for row in bin_summary]).statistic)
    calibration["image_easy_hard_paired"] = {"mean_hard_minus_easy": float(np.mean([row["difference_hard_minus_easy"] for row in paired])), "ci": paired_ci}
    calibration["text_prior_bin"] = {"bins": bin_summary, "mean_difficulty_spearman": trend}
    atomic_json(root / "artifacts" / "difficulty_calibration.json", calibration)
    regressions = {}
    for outcome in ("final_sa", "panl_l14_oof_sa_prediction"):
        regressions[outcome] = {}
        for group, selected in (("all", rows), ("conflict_easy", [row for row in rows if not row["Hard"]]), ("conflict_hard", [row for row in rows if row["Hard"]])):
            regressions[outcome][group] = fit_parameterizations(selected, outcome, repeats=bootstrap, seed=stable_seed(SEED, outcome, group))
    correlations = _correlations(rows, repeats=bootstrap)
    atomic_json(root / "artifacts" / "regression_results.json", regressions); atomic_json(root / "artifacts" / "correlation_results.json", correlations)
    _wide_probe_tables(root, difficulty, decision); _correlation_table(root, correlations, calibration, rows)
    atomic_text(root / "tables" / "regression_parameters.md", _regression_markdown(regressions)); _plots(root, difficulty, decision)
    calibrated_text, calibrated_image = calibration["text"]["calibrated_difficulty_supported"], calibration["image"]["calibrated_difficulty_supported"]
    panl_coencoded = any(row["position"] == "P1_PANL" and row["permutation"]["q_value"] < .05 for row in difficulty["metrics"] if row["target"] == "text") and any(row["position"] == "P1_PANL" and row["permutation"]["q_value"] < .05 for row in difficulty["metrics"] if row["target"] == "image") and any(row["position"] == "P1_PANL" and row["permutation"]["q_value"] < .05 for row in decision["metrics"])
    summary_lines = ["# delayed-SA PANL information 实验总结", "", f"1. 单模态难度校准：text={'通过' if calibrated_text else '未通过'}，image={'通过' if calibrated_image else '未通过'}。未通过时 model_perceived_difficulty 只能解释为候选分布熵。", f"2. D_text、D_image、G、U 与 final SA 的 Pearson/Spearman 及 item-bootstrap CI 已写入主相关表；G–SA 为统计关系，不作因果解释。", "3. 模型A和模型B的完整参数、cluster robust 标准误、bootstrap CI 与等价性审计见 tables/regression_parameters.md。", f"4. difficulty-only decision baseline 的 balanced accuracy={decision['baselines']['difficulty_only']['balanced_accuracy']:.4f}，仅表示预测能力。", f"5. Text/image difficulty onset：{json.dumps(difficulty['onsets'], ensure_ascii=False)}。", f"6. Decision-side onset：{json.dumps(decision['onsets'], ensure_ascii=False)}；P1_LAT 仅为 answer-token positive control。", f"7. PANL 汇合：{'结果与 difficulty、choice、SA 在 PANL 共处的假设一致' if panl_coencoded else '未观察到满足预注册标准的三类信息 PANL 共处'}。", "8. 结果支持：hidden 中相应信息的 OOF 可解码性，以及难度与 SA/选择之间的统计关联。", "9. 结果不能证明：这些信息被模型因果使用，也不能证明 PANL 是唯一或必要的机制位置。", "10. 下一步最有价值的因果实验：按 probe onset 设计 Evidence → Answer → PANL 定向 attention block，同时观察 PANL difficulty、decision probe 与 final SA 是否下降。", ""]
    atomic_text(root / "summary.md", "\n".join(summary_lines))
    summary = {"status": "complete", "calibration": calibration, "panl_coencoded": panl_coencoded, "formal_files": ["tables/difficulty_probe.csv", "tables/decision_probe.csv", "tables/sa_factor_correlations.csv", "tables/regression_parameters.md", "figures/difficulty_probe_R2.png", "figures/difficulty_probe_spearman.png", "figures/decision_probe_accuracy.png", "summary.md"]}
    stage_update(root, "analyze", "complete", formal_file_count=len(summary["formal_files"])); return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--resume", action="store_true"); parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPEATS)
    args = parser.parse_args(argv); root = ensure_output_layout(RESULTS_ROOT, resume=args.resume); print(json.dumps(analyze(root, bootstrap=args.bootstrap), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
