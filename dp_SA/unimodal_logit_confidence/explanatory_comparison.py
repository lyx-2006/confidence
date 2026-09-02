from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .config import PACKAGE_ROOT, SEED
from .io_utils import atomic_csv, atomic_json, atomic_text, canonical_hash, load_jsonl, sha256_file


RESULTS_ROOT = PACKAGE_ROOT / "output" / "results"
OUTPUT_ROOT = PACKAGE_ROOT / "output" / "explanatory_comparison"
CONFIDENCE_PATH = RESULTS_ROOT / "unimodal_confidence/artifacts/predictions/phase1_confidence_joined.jsonl"
SCORE_PATH = RESULTS_ROOT / "unimodal_confidence/artifacts/calibrated_scores/unimodal_scores.jsonl"
TRAIN_MANIFEST_PATH = RESULTS_ROOT / "shared/manifests/probe_train_manifest.jsonl"
TEST_MANIFEST_PATH = RESULTS_ROOT / "shared/manifests/test_manifest.jsonl"
SA_PATH = PACKAGE_ROOT.parent / "panl_information/output/results/artifacts/joined_records.jsonl"

EPSILON = 1e-6
BOOTSTRAP_REPEATS = 2000
OUTCOMES = {
    "final_sa": "final soft SA",
    "panl_probe_sa": "PANL probe SA (P1_PANL L14 OOF)",
}
MODELS: dict[str, tuple[str, ...]] = {
    "M0": ("Hard",),
    "MD": ("D_t", "D_i", "Hard"),
    "MC": ("L_t", "L_i", "Hard"),
    "MDC": ("D_t", "D_i", "L_t", "L_i", "Hard"),
    "MG": ("G_L", "M_L", "Hard"),
}
CONTRASTS = (
    ("MD_minus_M0", "MD", "M0"),
    ("MC_minus_M0", "MC", "M0"),
    ("MDC_minus_M0", "MDC", "M0"),
    ("MG_minus_M0", "MG", "M0"),
    ("MC_minus_MD", "MC", "MD"),
    ("MDC_minus_MC", "MDC", "MC"),
    ("MDC_minus_MD", "MDC", "MD"),
)


def clipped_logit(probability: float, epsilon: float = EPSILON) -> float:
    value = float(np.clip(float(probability), epsilon, 1.0 - epsilon))
    return float(math.log(value / (1.0 - value)))


def _unique_map(rows: Sequence[dict[str, Any]], key: Any, label: str) -> dict[Any, dict[str, Any]]:
    output: dict[Any, dict[str, Any]] = {}
    for row in rows:
        value = key(row)
        if value in output:
            raise ValueError(f"Duplicate {label}: {value}")
        output[value] = row
    return output


def prepare_analysis_rows(
    confidence_rows: Sequence[dict[str, Any]],
    score_rows: Sequence[dict[str, Any]],
    sa_rows: Sequence[dict[str, Any]],
    train_manifest: Sequence[dict[str, Any]],
    test_manifest: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    confidence = _unique_map(confidence_rows, lambda row: str(row["case_id"]), "confidence case_id")
    scores = _unique_map(score_rows, lambda row: tuple(row["stable_key"]), "score stable_key")
    sa = _unique_map(sa_rows, lambda row: str(row["case_id"]), "SA case_id")
    frozen_split: dict[str, str] = {}
    for split, manifest in (("train", train_manifest), ("test", test_manifest)):
        for row in manifest:
            case_id = str(row["case_id"])
            if case_id in frozen_split:
                raise ValueError(f"Case occurs in both frozen manifests: {case_id}")
            frozen_split[case_id] = split
    if set(confidence) != set(frozen_split):
        raise ValueError("Confidence cases do not exactly match the frozen train/test manifests")

    output: list[dict[str, Any]] = []
    for case_id, row in confidence.items():
        if case_id not in sa:
            raise ValueError(f"Missing SA row: {case_id}")
        split = str(row["split"])
        if split != frozen_split[case_id]:
            raise ValueError(f"Frozen split mismatch: {case_id}")
        text_key = ("text", *row["text_score_unique_key"])
        image_key = ("image", *row["image_score_unique_key"])
        if text_key not in scores or image_key not in scores:
            raise ValueError(f"Missing difficulty score: {case_id}")
        sa_row = sa[case_id]
        fixed_answer = str(row["fixed_answer"])
        if fixed_answer != str(sa_row["phase0_normalized_answer"]):
            raise ValueError(f"Fixed-answer mismatch: {case_id}")
        hard = int(str(row["condition"]) == "conflict_hard")
        if hard != int(sa_row["Hard"]):
            raise ValueError(f"Hard mismatch: {case_id}")
        values = {
            "case_id": case_id,
            "item_id": str(row["item_id"]),
            "family_id": str(row["family_id"]),
            "split": split,
            "condition": str(row["condition"]),
            "fixed_answer": fixed_answer,
            "Hard": hard,
            "D_t": float(scores[text_key]["entropy_difficulty"]),
            "D_i": float(scores[image_key]["entropy_difficulty"]),
            "L_t": clipped_logit(float(row["text_fixed_answer_confidence"])),
            "L_i": clipped_logit(float(row["image_fixed_answer_confidence"])),
            "final_sa": float(sa_row["final_sa"]),
            "panl_probe_sa": float(sa_row["panl_l14_oof_sa_prediction"]),
        }
        values["G_L"] = values["L_i"] - values["L_t"]
        values["M_L"] = (values["L_i"] + values["L_t"]) / 2.0
        if not all(math.isfinite(float(values[name])) for name in ("D_t", "D_i", "L_t", "L_i", "G_L", "M_L", *OUTCOMES)):
            raise ValueError(f"Non-finite analysis value: {case_id}")
        output.append(values)

    output.sort(key=lambda row: (row["split"], row["family_id"], row["case_id"]))
    train = [row for row in output if row["split"] == "train"]
    test = [row for row in output if row["split"] == "test"]
    train_items = {row["item_id"] for row in train}; test_items = {row["item_id"] for row in test}
    train_families = {row["family_id"] for row in train}; test_families = {row["family_id"] for row in test}
    if train_items & test_items:
        raise ValueError("Train/test item leakage")
    if train_families & test_families:
        raise ValueError("Train/test family leakage")
    if len(test) != 100 or len(test_families) != 50:
        raise ValueError("Frozen test set is not 100 records / 50 families")
    family_sizes = {family: sum(row["family_id"] == family for row in test) for family in test_families}
    if set(family_sizes.values()) != {2}:
        raise ValueError("Every frozen test family must contribute exactly two records")
    audit = {
        "status": "passed",
        "train_record_count": len(train), "test_record_count": len(test),
        "train_item_count": len(train_items), "test_item_count": len(test_items), "item_overlap_count": 0,
        "train_family_count": len(train_families), "test_family_count": len(test_families), "family_overlap_count": 0,
        "test_records_per_family": 2,
        "join": {"confidence_case_count": len(confidence), "matched_sa_count": len(output), "matched_text_score_count": len({tuple(row["text_score_unique_key"]) for row in confidence_rows}), "matched_image_score_count": len({tuple(row["image_score_unique_key"]) for row in confidence_rows})},
    }
    return output, audit


def design_matrices(
    train: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]], terms: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float | bool]]]:
    continuous = [term for term in terms if term != "Hard"]
    parameters: dict[str, dict[str, float | bool]] = {}
    transformed_train: dict[str, np.ndarray] = {}
    transformed_test: dict[str, np.ndarray] = {}
    if continuous:
        scaler = StandardScaler().fit(np.asarray([[float(row[t]) for t in continuous] for row in train]))
        train_scaled = scaler.transform(np.asarray([[float(row[t]) for t in continuous] for row in train]))
        test_scaled = scaler.transform(np.asarray([[float(row[t]) for t in continuous] for row in test]))
        for index, term in enumerate(continuous):
            transformed_train[term] = train_scaled[:, index]; transformed_test[term] = test_scaled[:, index]
            parameters[term] = {"standardized": True, "train_mean": float(scaler.mean_[index]), "train_scale": float(scaler.scale_[index])}
    transformed_train["Hard"] = np.asarray([float(row["Hard"]) for row in train])
    transformed_test["Hard"] = np.asarray([float(row["Hard"]) for row in test])
    parameters["Hard"] = {"standardized": False, "train_mean": 0.0, "train_scale": 1.0}
    return np.column_stack([transformed_train[t] for t in terms]), np.column_stack([transformed_test[t] for t in terms]), parameters


def metric_values(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    values = {
        "r2": float(r2_score(y, prediction)),
        "mae": float(mean_absolute_error(y, prediction)),
        "pearson": float(pearsonr(y, prediction).statistic),
        "spearman": float(spearmanr(y, prediction).statistic),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite held-out metric")
    return values


def cluster_bootstrap_indices(test: Sequence[dict[str, Any]], repeats: int, seed: int) -> tuple[list[np.ndarray], str]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(test): groups[str(row["family_id"])].append(index)
    families = sorted(groups)
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    family_draws: list[list[str]] = []
    for _ in range(repeats):
        chosen = [str(value) for value in rng.choice(families, size=len(families), replace=True)]
        family_draws.append(chosen)
        draws.append(np.asarray([index for family in chosen for index in groups[family]], dtype=int))
    return draws, canonical_hash(family_draws)


def percentile_ci(values: Sequence[float]) -> tuple[float, float, int]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if not len(finite): raise ValueError("No valid bootstrap values")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high), int(len(finite))


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def _interpret(outcome: str, contrasts: dict[str, dict[str, float]]) -> str:
    confidence_vs_difficulty = contrasts["MC_minus_MD"]
    difficulty_after_confidence = contrasts["MDC_minus_MC"]
    confidence_after_difficulty = contrasts["MDC_minus_MD"]
    parts = []
    if confidence_vs_difficulty["delta_r2"] > 0:
        stable = confidence_vs_difficulty["ci_low"] > 0
        parts.append(f"MC−MD={confidence_vs_difficulty['delta_r2']:.4f}，点估计上 confidence 比 difficulty 有更强的样本外解释力；" + ("其95% CI排除0。" if stable else "但其95% CI包含0，不能认为这一差异已稳定确立。"))
    else:
        parts.append(f"MC−MD={confidence_vs_difficulty['delta_r2']:.4f}，没有观察到 confidence 强于 difficulty 的样本外解释力。")
    near_zero = abs(difficulty_after_confidence["delta_r2"]) < 0.01 and difficulty_after_confidence["ci_low"] <= 0 <= difficulty_after_confidence["ci_high"]
    if near_zero and confidence_after_difficulty["delta_r2"] > 0:
        parts.append(f"MDC−MC={difficulty_after_confidence['delta_r2']:.4f} 接近零，而 MDC−MD={confidence_after_difficulty['delta_r2']:.4f}>0；这支持 difficulty 的信息主要被 confidence 吸收。")
    elif difficulty_after_confidence["delta_r2"] > 0 and confidence_after_difficulty["delta_r2"] > 0:
        stable_d = difficulty_after_confidence["ci_low"] > 0
        stable_c = confidence_after_difficulty["ci_low"] > 0
        evidence = "两项增量的95% CI都排除0" if stable_d and stable_c else "仅 confidence 在 difficulty 之外的增量95% CI排除0" if stable_c else "仅 difficulty 在 confidence 之外的增量95% CI排除0" if stable_d else "两项增量的95% CI都包含0"
        parts.append(f"MDC−MC={difficulty_after_confidence['delta_r2']:.4f}、MDC−MD={confidence_after_difficulty['delta_r2']:.4f} 均为正，点估计表现为混合解释；{evidence}。")
    else:
        parts.append(f"MDC−MC={difficulty_after_confidence['delta_r2']:.4f}，MDC−MD={confidence_after_difficulty['delta_r2']:.4f}；结果不满足“二者加入后均提升”的混合解释模式。")
    return " ".join(parts)


def run_analysis(
    output_root: Path = OUTPUT_ROOT,
    *,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
    seed: int = SEED,
) -> dict[str, Any]:
    input_paths = [CONFIDENCE_PATH, SCORE_PATH, TRAIN_MANIFEST_PATH, TEST_MANIFEST_PATH, SA_PATH]
    for path in input_paths:
        if not path.is_file(): raise FileNotFoundError(path)
    rows, audit = prepare_analysis_rows(load_jsonl(CONFIDENCE_PATH), load_jsonl(SCORE_PATH), load_jsonl(SA_PATH), load_jsonl(TRAIN_MANIFEST_PATH), load_jsonl(TEST_MANIFEST_PATH))
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    draws, draw_fingerprint = cluster_bootstrap_indices(test, bootstrap_repeats, seed)
    audit.update({
        "bootstrap": {"repeats": bootstrap_repeats, "seed": seed, "cluster": "family_id", "paired_draw_fingerprint": draw_fingerprint},
        "confidence_clip": [EPSILON, 1.0 - EPSILON],
        "inputs": {str(path): sha256_file(path) for path in input_paths},
        "field_mapping": {"D_t/D_i": "calibrated_scores.unimodal_scores.jsonl:entropy_difficulty via score unique keys", "L_t/L_i": "logit(clipped joined fixed-answer confidence)", "Hard": "condition == conflict_hard", "final_sa": "panl_information joined_records.final_sa", "panl_probe_sa": "panl_information joined_records.panl_l14_oof_sa_prediction"},
    })

    model_performance: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    truths: dict[str, np.ndarray] = {}
    bootstrap_r2: dict[tuple[str, str], list[float]] = {}

    for outcome, outcome_label in OUTCOMES.items():
        y_train = np.asarray([float(row[outcome]) for row in train])
        y_test = np.asarray([float(row[outcome]) for row in test])
        truths[outcome] = y_test
        for model_name, terms in MODELS.items():
            X_train, X_test, scaling = design_matrices(train, test, terms)
            model = Ridge(alpha=1.0, solver="lsqr").fit(X_train, y_train)
            predicted = np.asarray(model.predict(X_test), dtype=float)
            predictions[outcome, model_name] = predicted
            observed = metric_values(y_test, predicted)
            boot_metrics = {name: [] for name in observed}
            for indices in draws:
                values = metric_values(y_test[indices], predicted[indices])
                for name, value in values.items(): boot_metrics[name].append(value)
            bootstrap_r2[outcome, model_name] = boot_metrics["r2"]
            performance = {"outcome": outcome, "outcome_label": outcome_label, "model": model_name, "features": "+".join(terms), **observed}
            for name, values in boot_metrics.items():
                low, high, valid = percentile_ci(values)
                performance[f"{name}_ci_low"] = low; performance[f"{name}_ci_high"] = high; performance[f"{name}_bootstrap_valid"] = valid
            model_performance.append(performance)
            coefficient_rows.append({"outcome": outcome, "model": model_name, "term": "intercept", "standardized_coefficient": float(model.intercept_), "absolute_coefficient": abs(float(model.intercept_)), "predictor_standardized": False, "train_mean": 0.0, "train_scale": 1.0})
            for term, coefficient in zip(terms, model.coef_, strict=True):
                coefficient_rows.append({"outcome": outcome, "model": model_name, "term": term, "standardized_coefficient": float(coefficient), "absolute_coefficient": abs(float(coefficient)), "predictor_standardized": scaling[term]["standardized"], "train_mean": scaling[term]["train_mean"], "train_scale": scaling[term]["train_scale"]})
            for row, truth, prediction in zip(test, y_test, predicted, strict=True):
                prediction_rows.append({"case_id": row["case_id"], "item_id": row["item_id"], "family_id": row["family_id"], "condition": row["condition"], "outcome": outcome, "model": model_name, "observed_sa": float(truth), "predicted_sa": float(prediction), "residual": float(truth - prediction), "D_t": row["D_t"], "D_i": row["D_i"], "L_t": row["L_t"], "L_i": row["L_i"], "Hard": row["Hard"]})

        baseline = next(row for row in model_performance if row["outcome"] == outcome and row["model"] == "M0")
        for performance in [row for row in model_performance if row["outcome"] == outcome]:
            model_name = str(performance["model"])
            deltas = [value - base for value, base in zip(bootstrap_r2[outcome, model_name], bootstrap_r2[outcome, "M0"], strict=True)]
            low, high, valid = percentile_ci(deltas)
            performance["delta_r2_vs_M0"] = float(performance["r2"] - baseline["r2"])
            performance["delta_r2_vs_M0_ci_low"] = low; performance["delta_r2_vs_M0_ci_high"] = high; performance["delta_r2_vs_M0_bootstrap_valid"] = valid
        for name, left, right in CONTRASTS:
            observed_delta = float(prediction_metric_r2(y_test, predictions[outcome, left]) - prediction_metric_r2(y_test, predictions[outcome, right]))
            deltas = [a - b for a, b in zip(bootstrap_r2[outcome, left], bootstrap_r2[outcome, right], strict=True)]
            low, high, valid = percentile_ci(deltas)
            contrast_rows.append({"outcome": outcome, "contrast": name, "left_model": left, "right_model": right, "delta_r2": observed_delta, "ci_low": low, "ci_high": high, "bootstrap_repeats": bootstrap_repeats, "valid_bootstrap_repeats": valid, "paired_draw_fingerprint": draw_fingerprint})

    output_root = Path(output_root)
    for directory in ("tables", "figures", "artifacts"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    atomic_csv(output_root / "tables/model_performance.csv", model_performance)
    atomic_csv(output_root / "tables/paired_model_contrasts.csv", contrast_rows)
    atomic_csv(output_root / "tables/standardized_coefficients.csv", coefficient_rows)
    atomic_csv(output_root / "artifacts/predictions.csv", prediction_rows)
    atomic_json(output_root / "artifacts/split_audit.json", audit)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    model_order = list(MODELS)
    for axis, (outcome, label) in zip(axes, OUTCOMES.items(), strict=True):
        cells = {row["model"]: row for row in model_performance if row["outcome"] == outcome}
        values = np.asarray([cells[model]["r2"] for model in model_order])
        low = np.asarray([cells[model]["r2_ci_low"] for model in model_order]); high = np.asarray([cells[model]["r2_ci_high"] for model in model_order])
        axis.bar(model_order, values, color=["#777777", "#4C78A8", "#F58518", "#54A24B", "#B279A2"])
        axis.errorbar(np.arange(len(model_order)), values, yerr=np.vstack([values-low, high-values]), fmt="none", ecolor="black", capsize=3, lw=1)
        axis.axhline(0, color="black", lw=.8, ls="--"); axis.set_title(label); axis.set_xlabel("Model"); axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Held-out $R^2$ (95% family-bootstrap CI)")
    fig.tight_layout(); fig.savefig(output_root / "figures/heldout_model_comparison.png", dpi=300); plt.close(fig)

    contrast_lookup = {(row["outcome"], row["contrast"]): row for row in contrast_rows}
    perf_table = []
    for outcome in OUTCOMES:
        for model in MODELS:
            row = next(value for value in model_performance if value["outcome"] == outcome and value["model"] == model)
            perf_table.append([outcome, model, f"{row['r2']:.4f}", f"{row['mae']:.4f}", f"{row['pearson']:.4f}", f"{row['spearman']:.4f}", f"{row['delta_r2_vs_M0']:.4f} [{row['delta_r2_vs_M0_ci_low']:.4f}, {row['delta_r2_vs_M0_ci_high']:.4f}]"])
    contrast_table = []
    for outcome in OUTCOMES:
        for contrast in ("MC_minus_MD", "MDC_minus_MC", "MDC_minus_MD"):
            row = contrast_lookup[outcome, contrast]
            contrast_table.append([outcome, contrast, f"{row['delta_r2']:.4f}", f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"])
    coefficient_table = [[row["outcome"], row["model"], row["term"], f"{row['standardized_coefficient']:.6f}", "是" if row["predictor_standardized"] else "否"] for row in coefficient_rows]
    interpretations = []
    for outcome in OUTCOMES:
        selected = {name: contrast_lookup[outcome, name] for name in ("MC_minus_MD", "MDC_minus_MC", "MDC_minus_MD")}
        interpretations.append(f"- **{OUTCOMES[outcome]}**：{_interpret(outcome, selected)}")
    summary = f"""# 难度与单模态 confidence 对 SA 的样本外解释力比较

## 数据与方法

- 使用冻结的 1,112 条训练记录和 100 条测试记录；训练/测试分别覆盖 128/50 个 item 与 family，item/family 交集均为 0。
- `final_sa` 为 final soft SA；`panl_probe_sa` 为历史 P1_PANL × L14 的 item-level OOF probe prediction。
- $D_t,D_i$ 是冻结 unique-spec score 的 τ=1 entropy difficulty。
- $L_t,L_i$ 由 Phase-0 fixed answer 的校准概率裁剪至 $[10^{{-6}},1-10^{{-6}}]$ 后取 logit。
- 连续预测变量只使用训练集均值和标准差做 `StandardScaler`；`Hard` 保持 0/1。所有模型使用 `Ridge(alpha=1, solver=lsqr)`，仅在训练集拟合。
- 95% CI 来自 2,000 次 family-cluster paired bootstrap；所有 outcome 和模型共享完全相同的抽样 draws。

## Held-out 预测结果

{_markdown_table(["SA outcome", "模型", "R²", "MAE", "Pearson", "Spearman", "ΔR² vs M0（95% CI）"], perf_table)}

## 关键 paired contrasts

{_markdown_table(["SA outcome", "对比", "ΔR²", "95% CI"], contrast_table)}

## 拟合参数

连续变量系数表示预测变量增加一个训练集标准差时 SA 的变化；`Hard` 系数表示 hard 相对 easy 的变化。截距也完整列出。

{_markdown_table(["SA outcome", "模型", "参数", "系数", "预测变量已标准化"], coefficient_table)}

## 严格解释

{chr(10).join(interpretations)}

这些结果比较的是冻结测试集上的统计解释力，不能说明 confidence 因果驱动 SA。逐 case 的 observed/predicted SA 位于 `artifacts/predictions.csv`。
"""
    atomic_text(output_root / "summary.md", summary)
    return {"status": "complete", "output_root": str(output_root.resolve()), "train_count": len(train), "test_count": len(test), "model_count": len(model_performance), "contrast_count": len(contrast_rows), "prediction_count": len(prediction_rows), "bootstrap_repeats": bootstrap_repeats}


def prediction_metric_r2(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(r2_score(y, prediction))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.bootstrap <= 0: raise ValueError("--bootstrap must be positive")
    print(json.dumps(run_analysis(Path(args.output_root), bootstrap_repeats=args.bootstrap, seed=args.seed), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
