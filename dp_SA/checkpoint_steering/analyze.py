from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import (
    ALPHAS,
    BOOTSTRAP_REPEATS,
    LAYERS,
    POSITIONS,
    RESULTS_ROOT,
    SEED,
    SMOKE_ALPHAS,
    SMOKE_BOOTSTRAP_REPEATS,
    SMOKE_LAYERS,
)
from .io_utils import atomic_csv, atomic_json, atomic_text, canonical_hash, load_jsonl, sha256_file


GROUPS = ("all", "image_side", "text_side")
POSITION_PAIRS = tuple(zip(POSITIONS, POSITIONS[1:]))


def _group_rows(rows: Sequence[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    return list(rows) if group == "all" else [row for row in rows if str(row.get("test_side")) == group]


def _percentile(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def clustered_mean_ci(rows: Sequence[dict[str, Any]], value: Callable[[dict[str, Any]], float], *, repeats: int, seed: int) -> tuple[float, float, float]:
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_item[str(row["item_id"])].append(row)
    items = sorted(by_item)
    if not items:
        return float("nan"), float("nan"), float("nan")
    item_values = {item: float(np.mean([value(row) for row in by_item[item]])) for item in items}
    observed = float(np.mean(list(item_values.values())))
    rng = np.random.default_rng(seed)
    boot = [float(np.mean([item_values[str(item)] for item in rng.choice(items, size=len(items), replace=True)])) for _ in range(repeats)]
    low, high = _percentile(boot)
    return observed, low, high


def build_long_metrics(rows: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    counter = 0
    for position in POSITIONS:
        for layer in sorted({int(row["layer"]) for row in rows if row["position"] == position}):
            for alpha in sorted({float(row["alpha"]) for row in rows if row["position"] == position and int(row["layer"]) == layer}):
                cell = [row for row in rows if row["position"] == position and int(row["layer"]) == layer and float(row["alpha"]) == alpha]
                for group in GROUPS:
                    selected = _group_rows(cell, group)
                    if not selected:
                        continue
                    mean, low, high = clustered_mean_ci(selected, lambda row: float(row["delta_soft_sa"]), repeats=repeats, seed=seed + counter)
                    values = np.asarray([float(row["delta_soft_sa"]) for row in selected], dtype=float)
                    sem = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
                    output.append({
                        "position": position,
                        "layer": layer,
                        "alpha": alpha,
                        "group": group,
                        "mean_delta_soft_sa": mean,
                        "sem": sem,
                        "ci_low": low,
                        "ci_high": high,
                        "hard_change_rate": float(np.mean([bool(row["hard_class_changed"]) for row in selected])),
                        "hard_class_mean_delta": float(np.mean([float(row["hard_class_delta"]) for row in selected])),
                        "margin_change": float(np.mean([float(row["margin_change"]) for row in selected])),
                        "sample_count": len(selected),
                        "item_count": len({str(row["item_id"]) for row in selected}),
                        "saturation_rate": float(np.mean([bool(row["saturated"]) for row in selected])),
                        "invalid_count": sum(not bool(row.get("finite_values")) or abs(float(row.get("probability_sum", float("nan"))) - 1.0) > 1e-9 for row in selected),
                    })
                    counter += 1
    return output


def item_dose_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_item: dict[str, dict[float, float]] = defaultdict(dict)
    for row in rows:
        item = str(row["item_id"])
        alpha = float(row["alpha"])
        if alpha in by_item[item]:
            raise ValueError(f"Duplicate alpha {alpha:g} for item {item}")
        by_item[item][alpha] = float(row["delta_soft_sa"])
    output: dict[str, dict[str, float]] = {}
    for item, values in by_item.items():
        if len(values) < 2:
            raise ValueError(f"Dose response needs at least two alphas for item {item}")
        x = np.asarray(sorted(values), dtype=float)
        y = np.asarray([values[alpha] for alpha in x], dtype=float)
        metrics = {"slope": float(np.polyfit(x, y, 1)[0])}
        for alpha, label in ((-10.0, "minus10"), (-2.0, "minus2"), (2.0, "plus2"), (10.0, "plus10")):
            metrics[f"{label}_mean_delta"] = float(values.get(alpha, float("nan")))
        for magnitude in (2, 10):
            plus = values.get(float(magnitude), float("nan"))
            minus = values.get(float(-magnitude), float("nan"))
            metrics[f"symmetric_effect_{magnitude}"] = float((plus - minus) / 2)
            metrics[f"asymmetry_{magnitude}"] = float(plus + minus)
        output[item] = metrics
    return output


def _aggregate_item_metric(item_metrics: dict[str, dict[str, float]], metric: str, *, repeats: int, seed: int) -> tuple[float, float, float]:
    items = sorted(item for item, values in item_metrics.items() if math.isfinite(float(values.get(metric, float("nan")))))
    if not items:
        return float("nan"), float("nan"), float("nan")
    observed = float(np.mean([item_metrics[item][metric] for item in items]))
    rng = np.random.default_rng(seed)
    boot = [float(np.mean([item_metrics[str(item)][metric] for item in rng.choice(items, size=len(items), replace=True)])) for _ in range(repeats)]
    low, high = _percentile(boot)
    return observed, low, high


def build_dose_metrics(rows: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    counter = 0
    for position in POSITIONS:
        for layer in sorted({int(row["layer"]) for row in rows if row["position"] == position}):
            base = [row for row in rows if row["position"] == position and int(row["layer"]) == layer]
            for group in GROUPS:
                selected = _group_rows(base, group)
                if not selected:
                    continue
                per_item = item_dose_metrics(selected)
                result: dict[str, Any] = {"position": position, "layer": layer, "group": group, "item_count": len(per_item)}
                metrics = (
                    "slope", "minus10_mean_delta", "minus2_mean_delta", "plus2_mean_delta", "plus10_mean_delta",
                    "symmetric_effect_2", "symmetric_effect_10", "asymmetry_2", "asymmetry_10",
                )
                for offset, metric in enumerate(metrics):
                    mean, low, high = _aggregate_item_metric(per_item, metric, repeats=repeats, seed=seed + 10000 + counter * 20 + offset)
                    result[metric] = mean
                    result[f"{metric}_ci_low"] = low
                    result[f"{metric}_ci_high"] = high
                result["bidirectional_pass"] = bool(
                    math.isfinite(result["plus10_mean_delta"])
                    and result["plus10_mean_delta"] > 0
                    and result["minus10_mean_delta"] < 0
                    and result["slope"] > 0
                )
                output.append(result)
                counter += 1
    return output


def build_position_contrasts(rows: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    counter = 0
    layers = sorted({int(row["layer"]) for row in rows})
    for left, right in POSITION_PAIRS:
        for layer in layers:
            for group in GROUPS:
                left_rows = _group_rows([row for row in rows if row["position"] == left and int(row["layer"]) == layer], group)
                right_rows = _group_rows([row for row in rows if row["position"] == right and int(row["layer"]) == layer], group)
                if not left_rows or not right_rows:
                    continue
                left_items = item_dose_metrics(left_rows)
                right_items = item_dose_metrics(right_rows)
                items = sorted(set(left_items) & set(right_items))
                if not items:
                    continue
                for metric in ("slope", "symmetric_effect_2", "symmetric_effect_10"):
                    valid = [item for item in items if math.isfinite(left_items[item][metric]) and math.isfinite(right_items[item][metric])]
                    if not valid:
                        estimate = low = high = float("nan")
                    else:
                        differences = {item: right_items[item][metric] - left_items[item][metric] for item in valid}
                        estimate = float(np.mean(list(differences.values())))
                        rng = np.random.default_rng(seed + 20000 + counter)
                        boot = [float(np.mean([differences[str(item)] for item in rng.choice(valid, size=len(valid), replace=True)])) for _ in range(repeats)]
                        low, high = _percentile(boot)
                    output.append({
                        "from_position": left,
                        "to_position": right,
                        "layer": layer,
                        "group": group,
                        "metric": metric,
                        "contrast": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "item_count": len(valid),
                    })
                    counter += 1
    return output


def build_wide(long_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    metrics = ("mean_delta_soft_sa", "ci_low", "ci_high", "hard_change_rate", "sample_count")
    columns = [
        f'{position}__L{layer}__a{float(alpha):g}'
        for position in POSITIONS
        for layer in sorted({int(row["layer"]) for row in long_rows if row["position"] == position})
        for alpha in sorted({float(row["alpha"]) for row in long_rows if row["position"] == position and int(row["layer"]) == layer})
    ]
    lookup = {(row["position"], int(row["layer"]), float(row["alpha"]), row["group"]): row for row in long_rows}
    output = []
    for group in GROUPS:
        for metric in metrics:
            row: dict[str, Any] = {"metric": f"{group}__{metric}"}
            for column in columns:
                position, layer_text, alpha_text = column.split("__")
                source = lookup.get((position, int(layer_text[1:]), float(alpha_text[1:]), group))
                row[column] = source.get(metric, "") if source else ""
            output.append(row)
    return output, ["metric", *columns]


def _plots(root: Path, long_rows: Sequence[dict[str, Any]], dose_rows: Sequence[dict[str, Any]]) -> list[str]:
    import matplotlib.pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    all_rows = [row for row in long_rows if row["group"] == "all"]
    no_ci_positions = {
        "P1_ATTRIBUTION_DEFINITION_END",
        "P1_FORMAT_DESCRIPTION_END",
    }
    styles = {
        -10.0: ("#174a7e", "-"), -2.0: ("#5b9bd5", "--"),
        2.0: ("#ef8a82", "--"), 10.0: ("#b2182b", "-"),
    }
    written: list[str] = []
    for position in POSITIONS:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        position_rows = [row for row in all_rows if row["position"] == position]
        bound_keys = ("mean_delta_soft_sa",) if position in no_ci_positions else ("ci_low", "ci_high")
        finite_bounds = [
            abs(float(row[key]))
            for row in position_rows
            for key in bound_keys
            if math.isfinite(float(row[key]))
        ]
        position_extent = max(finite_bounds, default=1e-6)
        position_y_limit = max(1e-6, position_extent * 1.12)
        zero = sorted([row for row in position_rows if float(row["alpha"]) == 0], key=lambda row: int(row["layer"]))
        if zero:
            if position in no_ci_positions:
                ax.plot(
                    [row["layer"] for row in zero], [row["mean_delta_soft_sa"] for row in zero],
                    linestyle="none", marker="o", color="#999999", alpha=0.65,
                    markersize=3, label="alpha=0 parity baseline",
                )
            else:
                ax.errorbar(
                    [row["layer"] for row in zero], [row["mean_delta_soft_sa"] for row in zero],
                    yerr=[[row["mean_delta_soft_sa"] - row["ci_low"] for row in zero], [row["ci_high"] - row["mean_delta_soft_sa"] for row in zero]],
                    fmt="o", color="#999999", alpha=0.65, markersize=3, label="alpha=0 parity baseline",
                )
        for alpha, (color, line_style) in styles.items():
            data = sorted([row for row in position_rows if float(row["alpha"]) == alpha], key=lambda row: int(row["layer"]))
            if not data:
                continue
            means = np.asarray([row["mean_delta_soft_sa"] for row in data], dtype=float)
            if position in no_ci_positions:
                ax.plot(
                    [row["layer"] for row in data], means, marker="o", color=color,
                    linestyle=line_style, label=f"alpha={alpha:+g}",
                )
            else:
                errors = np.asarray([[means[index] - row["ci_low"] for index, row in enumerate(data)], [row["ci_high"] - means[index] for index, row in enumerate(data)]])
                ax.errorbar([row["layer"] for row in data], means, yerr=errors, marker="o", color=color, linestyle=line_style, capsize=3, label=f"alpha={alpha:+g}")
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_xticks(sorted({int(row["layer"]) for row in position_rows}))
        ax.set_ylim(-position_y_limit, position_y_limit)
        ax.set_xlabel("Zero-based decoder layer")
        ax.set_ylabel("Mean delta soft SA")
        ax.set_title(position)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.18)
        fig.tight_layout()
        name = f"{position}_delta_sa_by_layer.png"
        fig.savefig(root / name, dpi=300)
        plt.close(fig)
        written.append(name)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for position in POSITIONS:
        data = sorted([row for row in dose_rows if row["position"] == position and row["group"] == "all"], key=lambda row: int(row["layer"]))
        if data:
            means = np.asarray([row["slope"] for row in data], dtype=float)
            errors = np.asarray([[means[index] - row["slope_ci_low"] for index, row in enumerate(data)], [row["slope_ci_high"] - means[index] for index, row in enumerate(data)]])
            ax.errorbar([row["layer"] for row in data], means, yerr=errors, marker="o", capsize=2, label=position)
    slope_bounds = [
        abs(float(row[key]))
        for row in dose_rows
        if row["group"] == "all"
        for key in ("slope_ci_low", "slope_ci_high")
        if math.isfinite(float(row[key]))
    ]
    slope_extent = max(slope_bounds, default=1e-6)
    slope_y_limit = max(1e-6, slope_extent * 1.12)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_ylim(-slope_y_limit, slope_y_limit)
    ax.set_xlabel("Zero-based decoder layer")
    ax.set_ylabel("Dose-response slope")
    ax.set_title("Checkpoint steering slope comparison")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    name = "position_slope_comparison.png"
    fig.savefig(root / name, dpi=300)
    plt.close(fig)
    written.append(name)
    return written


def _audit_rows(clean_rows: Sequence[dict[str, Any]], trials: Sequence[dict[str, Any]], failures: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for clean in clean_rows:
        for position in (*POSITIONS, "P1_SAC"):
            record = clean["positions"][position]
            output.append({
                "audit_type": "position", "status": "passed", "case_id": clean["case_id"], "item_id": clean["item_id"],
                "position": position, "layer": "", "alpha": "", "rendered_index": record["rendered_index"],
                "processed_index": record["processed_index"], "token_id": record["token_id"], "token_text": record["token_text"],
                "anchor_occurrence_count": record["anchor_occurrence_count"], "hook_call_count": "", "hook_applied_count": "",
                "alpha_zero_parity": "", "finite_values": True, "failure": "",
                "details_json": json.dumps({"anchor_text": record["anchor_text"], "token_window": record["token_window"], "causal_order": clean["positions"]["causal_order"]}, ensure_ascii=False, separators=(",", ":")),
            })
    for trial in trials:
        diag = trial["hook_diagnostics"]
        parity = trial.get("alpha_zero_parity")
        output.append({
            "audit_type": "trial", "status": "passed", "case_id": trial["case_id"], "item_id": trial["item_id"],
            "position": trial["position"], "layer": trial["layer"], "alpha": trial["alpha"], "rendered_index": "",
            "processed_index": trial["processed_position"], "token_id": "", "token_text": "", "anchor_occurrence_count": "",
            "hook_call_count": diag["hook_call_count"], "hook_applied_count": diag["steering_applied_count"],
            "alpha_zero_parity": parity["passed"] if parity else "not_applicable", "finite_values": trial["finite_values"], "failure": "",
            "details_json": json.dumps({"activation_before_hash": trial["activation_before_hash"], "activation_after_hash": trial["activation_after_hash"], "probability_sum": trial["probability_sum"]}, separators=(",", ":")),
        })
    for failure in failures:
        output.append({
            "audit_type": "failure", "status": "failed", "case_id": failure.get("case_id", ""), "item_id": failure.get("item_id", ""),
            "position": failure.get("position", ""), "layer": failure.get("layer", ""), "alpha": failure.get("alpha", ""),
            "rendered_index": "", "processed_index": "", "token_id": "", "token_text": "", "anchor_occurrence_count": "",
            "hook_call_count": "", "hook_applied_count": "", "alpha_zero_parity": "", "finite_values": "", "failure": failure.get("message", ""),
            "details_json": json.dumps(failure, ensure_ascii=False, separators=(",", ":")),
        })
    return output


def _readme() -> str:
    return """# Checkpoint steering 表格说明

- `mean_delta_soft_sa`：干预后 soft SA 减去同一样本 clean soft SA；正值表示更偏图像。
- `sem`：该 cell 样本 delta 的标准误；`ci_low/ci_high` 为按 item 聚类重采样的 95% 区间。
- `hard_change_rate` / `hard_class_mean_delta`：硬类别改变比例与平均类别差。
- `margin_change`：干预后 clean 类别相对其余类别最大 logit 的 margin 变化。
- `saturation_rate`：soft SA 到达 `[0.05, 0.95]` 边界的比例；`invalid_count` 为非有限值或概率和异常数。
- `slope`：同一 item 完整 alpha 剂量响应的线性斜率，再跨 item 聚合。
- `symmetric_effect_2/10`：`(delta(+a)-delta(-a))/2`；`asymmetry_2/10`：`delta(+a)+delta(-a)`。
- `position_contrasts.csv` 的 contrast 始终为后一个检查点减前一个检查点，并在同 item、同 layer 内配对。
- `run_audit.csv` 汇总字符/token 定位、hook 命中、alpha=0 parity、finite-value 与失败事件。
"""


def _summary(dose_rows: Sequence[dict[str, Any]]) -> str:
    passed = [row for row in dose_rows if row["group"] == "all" and row["bidirectional_pass"]]
    lines = [
        "# Delayed-SA checkpoint steering summary",
        "",
        f"全样本组中通过双向门禁的位置×层单元：{len(passed)}。主要推断应联合查看 slope、双向 symmetric effect 及相邻位置 paired contrasts。",
        "",
        "解释边界：",
        "",
        "- LAT 与 PANL 均稳定且强度相近，只支持答案末端已有可操纵的 SA-compatible/precursor state，不能据此断言完整 verbal SA 已形成。",
        "- LAT 弱而 PANL 增强，支持答案之后、PANL 附近开始整合。",
        "- LAT/PANL 弱而 attribution definition 后增强，支持显式来源归因指令触发 SA 构造。",
        "- class list 后才增强，更像类别语义或数值映射形成；format description 后才增强，更像最终输出格式准备。",
        "- LAT steering 也可能操纵 answer commitment、模态选择、难度或置信状态，随后才被转换成 verbal SA。",
    ]
    return "\n".join(lines) + "\n"


def _required_files(root: Path) -> list[Path]:
    tables = [
        "steering_delta_sa_long.csv", "steering_delta_sa_wide.csv", "dose_response_by_position_layer.csv",
        "position_contrasts.csv", "run_audit.csv", "README.md",
    ]
    figures = [f"{position}_delta_sa_by_layer.png" for position in POSITIONS] + ["position_slope_comparison.png"]
    return [*(root / "tables" / name for name in tables), *(root / "figures" / name for name in figures), root / "artifacts" / "diagnostics" / "summary.md"]


def analyze(
    *,
    output_root: Path = RESULTS_ROOT,
    smoke: bool = False,
    resume: bool = False,
    repeats: int | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    diagnostics = root / "artifacts" / "diagnostics"
    trial_paths = [diagnostics / "steering_trials.jsonl", *sorted(diagnostics.glob("steering_trials_layer_*.jsonl"))]
    trial_paths = [path for path in trial_paths if path.is_file()]
    trials = [row for path in trial_paths for row in load_jsonl(path) if row.get("status") == "completed"]
    if not trials:
        raise ValueError("No completed checkpoint-steering trials")
    repeats = int(repeats or (SMOKE_BOOTSTRAP_REPEATS if smoke else BOOTSTRAP_REPEATS))
    cases = {str(row["case_id"]) for row in trials}
    unique_trials = {(str(row["case_id"]), row["position"], int(row["layer"]), float(row["alpha"])) for row in trials}
    if len(unique_trials) != len(trials):
        raise ValueError("Analysis found duplicate case/position/layer/alpha trials")
    cells: dict[tuple[str, int, float], set[str]] = defaultdict(set)
    position_layers: dict[str, set[int]] = defaultdict(set)
    for row in trials:
        position = str(row["position"])
        if position not in POSITIONS:
            raise ValueError(f"Unknown checkpoint position in trial artifacts: {position}")
        layer = int(row["layer"])
        alpha = float(row["alpha"])
        cells[(position, layer, alpha)].add(str(row["case_id"]))
        position_layers[position].add(layer)
    expected_alphas = set(SMOKE_ALPHAS if smoke else ALPHAS)
    for position, layers_for_position in position_layers.items():
        for layer in layers_for_position:
            actual_alphas = {alpha for candidate_position, candidate_layer, alpha in cells if candidate_position == position and candidate_layer == layer}
            if actual_alphas != expected_alphas:
                raise ValueError(f"Incomplete alpha grid at {position} L{layer}: {sorted(actual_alphas)}")
            for alpha in expected_alphas:
                if cells[(position, layer, alpha)] != cases:
                    raise ValueError(f"Incomplete case grid at {position} L{layer} alpha={alpha:g}")
    expected = len(cases) * len(expected_alphas) * sum(len(values) for values in position_layers.values())
    if len(trials) != expected:
        raise ValueError(f"Analysis requires a complete configured trial grid: {len(trials)}/{expected}")
    alpha_zero = [row for row in trials if float(row["alpha"]) == 0.0]
    if not alpha_zero or any(not row.get("alpha_zero_parity", {}).get("passed") for row in alpha_zero):
        raise ValueError("Analysis alpha-zero parity gate failed")
    config = {
        "format_version": 2,
        "smoke": smoke,
        "bootstrap_repeats": repeats,
        "seed": SEED,
        "trial_files": {str(path.relative_to(root)): sha256_file(path) for path in trial_paths},
        "trial_count": len(trials),
        "position_layers": {position: sorted(values) for position, values in position_layers.items()},
    }
    config["fingerprint"] = canonical_hash(config)
    progress_path = root / "progress" / "analysis_progress.json"
    if progress_path.exists():
        previous = json.loads(progress_path.read_text())
        if previous.get("config_fingerprint") != config["fingerprint"] and not refresh:
            raise ValueError("Analysis resume fingerprint mismatch")
        if previous.get("status") == "complete" and resume and not refresh and all(path.is_file() and path.stat().st_size > 0 for path in _required_files(root)):
            return {**previous, "resumed_noop": True}
        if not resume and not refresh:
            raise FileExistsError("Analysis output exists; use --resume")
    atomic_json(progress_path, {"status": "running", "config_fingerprint": config["fingerprint"], **config})

    long_rows = build_long_metrics(trials, repeats=repeats, seed=SEED)
    dose_rows = build_dose_metrics(trials, repeats=repeats, seed=SEED)
    contrasts = build_position_contrasts(trials, repeats=repeats, seed=SEED)
    wide_rows, wide_fields = build_wide(long_rows)
    tables = root / "tables"
    atomic_csv(tables / "steering_delta_sa_long.csv", long_rows)
    atomic_csv(tables / "steering_delta_sa_wide.csv", wide_rows, wide_fields)
    atomic_csv(tables / "dose_response_by_position_layer.csv", dose_rows)
    atomic_csv(tables / "position_contrasts.csv", contrasts)
    clean_rows = [row for row in load_jsonl(root / "artifacts" / "diagnostics" / "clean_capture.jsonl") if row.get("status") == "completed"]
    failures = load_jsonl(root / "progress" / "failures.jsonl")
    atomic_csv(tables / "run_audit.csv", _audit_rows(clean_rows, trials, failures))
    atomic_text(tables / "README.md", _readme())
    figure_names = _plots(root / "figures", long_rows, dose_rows)
    atomic_text(root / "artifacts" / "diagnostics" / "summary.md", _summary(dose_rows))
    missing = [str(path) for path in _required_files(root) if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Completion artifact audit failed: {missing}")
    completion = {
        "status": "complete", "smoke": smoke, "formal_run_started": not smoke, "trial_count": len(trials),
        "expected_trial_count": expected, "alpha_zero_count": len(alpha_zero), "alpha_zero_parity": "passed",
        "bootstrap_repeats": repeats, "tables": 6, "figures": len(figure_names), "config_fingerprint": config["fingerprint"],
    }
    atomic_json(progress_path, completion)
    atomic_json(root / "progress" / "completion.json", completion)
    return completion


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke and not args.output_root:
        parser.error("--smoke requires an explicit --output-root outside the formal results directory; prefer run_pipeline --smoke")
    root = Path(args.output_root) if args.output_root else RESULTS_ROOT
    if args.smoke:
        try:
            root.resolve().relative_to(RESULTS_ROOT.resolve())
        except ValueError:
            pass
        else:
            parser.error("smoke output cannot be inside the formal results directory")
    print(json.dumps(analyze(output_root=root, smoke=args.smoke, resume=args.resume, repeats=args.bootstrap, refresh=args.refresh), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
