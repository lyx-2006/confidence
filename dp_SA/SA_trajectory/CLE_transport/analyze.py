from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dp_SA.io_utils import atomic_json, load_jsonl

from .config import (
    ALL_METRICS, BOOTSTRAP_REPEATS, CONDITIONS, CONFIRMATORY_ANSWERS,
    EXPERIMENTS, GROUPS, PRIMARY_GROUP, PRIMARY_METRICS, SEED,
    SMOKE_BOOTSTRAP_REPEATS, WINDOWS, WINDOW_NAMES, default_output,
)
from .run import load_trials


def bh_fdr(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return []
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return [float(value) for value in output]


class FamilyBootstrap:
    """Shared family draws for conditions and their within-case contrasts."""

    def __init__(self, manifest: Sequence[dict[str, Any]], repeats: int, seed: int = SEED):
        self.rows = {str(row["family_id"]): row for row in manifest}
        self.repeats = int(repeats)
        rng = np.random.default_rng(seed)
        self.uniforms = {
            key: rng.random((self.repeats, max(1, len(manifest))))
            for key in (*CONFIRMATORY_ANSWERS, "__all__")
        }

    def _resample(self, families: list[str], key: str, values: dict[str, float]) -> tuple[float, np.ndarray]:
        vector = np.asarray([values[family] for family in families], dtype=float)
        indices = np.floor(self.uniforms[key][:, :len(families)] * len(families)).astype(int)
        return float(vector.mean()), vector[indices].mean(axis=1)

    def aggregate(self, values: dict[str, float], group: str) -> dict[str, Any]:
        if not values:
            return {"mean": math.nan, "sem": math.nan, "ci_low": math.nan,
                    "ci_high": math.nan, "boot": np.full(self.repeats, np.nan),
                    "family_count": 0, "answer_count": 0}
        if group in ("family_micro", "all"):
            families = sorted(values)
            observed, boot = self._resample(families, "__all__", values)
            answer_count = len({str(self.rows[family]["test_answer"]) for family in families})
        else:
            side = {"image_side": "high_image", "text_side": "high_text"}.get(group)
            observed_by_answer: list[float] = []
            boot_by_answer: list[np.ndarray] = []
            families = []
            for answer in CONFIRMATORY_ANSWERS:
                selected = sorted(
                    family for family in values
                    if str(self.rows[family]["test_answer"]) == answer
                    and (side is None or self.rows[family]["test_side"] == side)
                )
                if selected:
                    observed, sampled = self._resample(selected, answer, values)
                    observed_by_answer.append(observed)
                    boot_by_answer.append(sampled)
                    families.extend(selected)
            if not observed_by_answer:
                return self.aggregate({}, group)
            observed = float(np.mean(observed_by_answer))
            boot = np.stack(boot_by_answer).mean(axis=0)
            answer_count = len(observed_by_answer)
        low, high = np.percentile(boot, [2.5, 97.5])
        return {"mean": observed, "sem": float(np.std(boot, ddof=1)),
                "ci_low": float(low), "ci_high": float(high), "boot": boot,
                "family_count": len(families), "answer_count": answer_count}


def two_sided_bootstrap_p(samples: np.ndarray) -> float:
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return math.nan
    lower = (1 + np.count_nonzero(finite <= 0)) / (len(finite) + 1)
    upper = (1 + np.count_nonzero(finite >= 0)) / (len(finite) + 1)
    return float(min(1.0, 2 * min(lower, upper)))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: _csv_value(row.get(key)) for key in fields} for row in rows)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _complete_windows(blocked: list[dict[str, Any]], case_ids: set[str]) -> tuple[tuple[int, int], ...]:
    complete = []
    for window in WINDOWS:
        rows = [row for row in blocked if (int(row["window_start"]), int(row["window_end"])) == window]
        keys = {(row["case_id"], row["condition"]) for row in rows}
        expected = {(case_id, condition) for case_id in case_ids for condition in CONDITIONS}
        if keys and keys != expected:
            raise RuntimeError(f"Partially completed window {window}: {len(keys)}/{len(expected)}")
        if keys == expected:
            complete.append(window)
    if not complete:
        raise RuntimeError("No complete blocked window is available for analysis")
    return tuple(complete)


def _window_effects(blocked: list[dict[str, Any]], windows: Sequence[tuple[int, int]],
                    bootstrap: FamilyBootstrap) -> list[dict[str, Any]]:
    output = []
    for window in windows:
        for condition in CONDITIONS:
            selected = [row for row in blocked if row["condition"] == condition
                        and (int(row["window_start"]), int(row["window_end"])) == window]
            for metric in ALL_METRICS:
                values = {str(row["family_id"]): float(row[metric]) for row in selected}
                for group in GROUPS:
                    stats = bootstrap.aggregate(values, group)
                    output.append({
                        "window_name": WINDOW_NAMES[window], "window_start": window[0], "window_end": window[1],
                        "condition": condition, "group": group, "metric": metric,
                        "mean": stats["mean"], "sem": stats["sem"],
                        "ci95_low": stats["ci_low"], "ci95_high": stats["ci_high"],
                        "family_count": stats["family_count"], "answer_count": stats["answer_count"],
                        "bootstrap_repeats": bootstrap.repeats,
                    })
    return output


def _paired(blocked: list[dict[str, Any]], windows: Sequence[tuple[int, int]],
            bootstrap: FamilyBootstrap) -> list[dict[str, Any]]:
    lookup = {(row["case_id"], row["condition"], int(row["window_start"])): row for row in blocked}
    output = []
    for window in windows:
        main = [row for row in blocked if row["condition"] == CONDITIONS[0]
                and int(row["window_start"]) == window[0]]
        for metric in PRIMARY_METRICS:
            values = {
                str(row["family_id"]): float(row[metric])
                - float(lookup[row["case_id"], CONDITIONS[1], window[0]][metric])
                for row in main
            }
            for group in GROUPS:
                stats = bootstrap.aggregate(values, group)
                output.append({
                    "window_name": WINDOW_NAMES[window], "window_start": window[0], "window_end": window[1],
                    "comparison": "C1_main_block_minus_C2_source_plus_1_control",
                    "metric": metric, "group": group, "specific_effect": stats["mean"],
                    "sem": stats["sem"], "ci95_low": stats["ci_low"], "ci95_high": stats["ci_high"],
                    "family_count": stats["family_count"], "answer_count": stats["answer_count"],
                    "p_raw": two_sided_bootstrap_p(stats["boot"]) if group == PRIMARY_GROUP else None,
                    "q_bh": None, "bootstrap_repeats": bootstrap.repeats,
                })
    if set(windows) == set(WINDOWS):
        for metric in PRIMARY_METRICS:
            family = [row for row in output if row["metric"] == metric and row["group"] == PRIMARY_GROUP]
            family.sort(key=lambda row: row["window_start"])
            for row, q_value in zip(family, bh_fdr([float(row["p_raw"]) for row in family])):
                row["q_bh"] = q_value
    return output


def _attention_audit(blocked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in blocked:
        diagnostics = row["attention_diagnostics"]
        expected_layers = list(range(int(row["window_start"]), int(row["window_end"]) + 1))
        if diagnostics["layers"] != expected_layers:
            raise RuntimeError(f"Blocked layers mismatch: {row['case_id']} {row['condition']}")
        for layer in expected_layers:
            detail = diagnostics["by_layer"][str(layer)]
            passed = (float(detail["max_blocked_weight"]) == 0.0
                      and float(detail["max_row_sum_error"]) <= 0.01
                      and bool(detail["finite"]) and int(detail["hook_call_count"]) == 1)
            if not passed:
                raise RuntimeError(f"Attention audit failed: {row['case_id']} layer {layer}")
            output.append({
                "case_id": row["case_id"], "condition": row["condition"],
                "window_name": row["window_name"], "blocked_layers": json.dumps(expected_layers),
                "layer": layer, "query_name": row["query_name"], "source_name": row["source_name"],
                "query_index": row["query_index"], "source_index": row["source_index"],
                "blocked_edge_weight": detail["max_blocked_weight"],
                "attention_row_sum_error": detail["max_row_sum_error"],
                "hook_count": detail["hook_call_count"], "head_count": detail["head_count"],
                "passed": passed,
            })
    return output


def _plots_from_csv(path: Path, figures: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["group"] == PRIMARY_GROUP]
    figures.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("delta_soft_sa", "Delta soft SA", "fig1_delta_sa_by_window.png", True),
        ("token_change_rate", "Token change rate", "fig2_token_change_rate_by_window.png", False),
        ("logit_change_diff", "Logit change diff", "fig3_logit_change_diff_by_window.png", True),
    )
    colors = {CONDITIONS[0]: "#d62728", CONDITIONS[1]: "#1f77b4"}
    labels = {CONDITIONS[0]: "main block", CONDITIONS[1]: "source+1 control"}
    output = []
    for metric, ylabel, filename, zero_line in definitions:
        fig, axis = plt.subplots(figsize=(8, 4.8))
        for condition in CONDITIONS:
            selected = sorted((row for row in rows if row["metric"] == metric
                               and row["condition"] == condition), key=lambda row: int(row["window_start"]))
            x = np.arange(len(selected)); means = np.asarray([float(row["mean"]) for row in selected])
            low = np.asarray([float(row["ci95_low"]) for row in selected])
            high = np.asarray([float(row["ci95_high"]) for row in selected])
            axis.errorbar(x, means, yerr=np.vstack((means - low, high - means)), marker="o",
                          linewidth=2, capsize=3, color=colors[condition], label=labels[condition])
        selected_windows = sorted({(int(row["window_start"]), int(row["window_end"]))
                                   for row in rows if row["metric"] == metric})
        axis.set_xticks(np.arange(len(selected_windows)), [f"L{a}\u2013{b}" for a, b in selected_windows])
        if zero_line:
            axis.axhline(0, color="black", linewidth=.8, alpha=.65)
        axis.set_xlabel("Layer window"); axis.set_ylabel(ylabel); axis.legend(frameon=False)
        axis.grid(axis="y", alpha=.2); fig.tight_layout()
        destination = figures / filename
        fig.savefig(destination, dpi=220); plt.close(fig); output.append(str(destination))
    return output


def analyze(*, experiment: str, output_root: Path | None = None, smoke: bool = False,
            repeats: int | None = None) -> dict[str, Any]:
    root = Path(output_root or default_output(experiment)).resolve()
    manifest = load_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl")
    trials = load_trials(root); case_ids = {str(row["case_id"]) for row in manifest}
    clean = [row for row in trials if row["condition"] == "C0_clean"]
    blocked = [row for row in trials if row["condition"] in CONDITIONS]
    if len(clean) != len(case_ids) or {row["case_id"] for row in clean} != case_ids:
        raise RuntimeError(f"Clean grid incomplete: {len(clean)}/{len(case_ids)}")
    windows = _complete_windows(blocked, case_ids)
    expected = len(case_ids) * (1 + 2 * len(windows))
    if len(clean) + len(blocked) != expected:
        raise RuntimeError(f"Trial grid has unexpected rows: {len(clean) + len(blocked)}/{expected}")
    repeats = int(repeats or (SMOKE_BOOTSTRAP_REPEATS if smoke else BOOTSTRAP_REPEATS))
    bootstrap = FamilyBootstrap(manifest, repeats, SEED)
    effects = _window_effects(blocked, windows, bootstrap)
    paired = _paired(blocked, windows, bootstrap)
    audit = _attention_audit(blocked)
    tables = root / "tables"
    case_path = tables / "case_level_trials.csv"
    atomic_csv(case_path, sorted(clean + blocked, key=lambda row: (
        row["case_id"], row["condition"], row.get("window_start") or -1
    )))
    effect_path = tables / "window_effects.csv"; atomic_csv(effect_path, effects)
    atomic_csv(tables / "paired_main_vs_control.csv", paired)
    atomic_csv(tables / "attention_audit.csv", audit)
    table_readme = (
        f"# {experiment} attention transport blocking 表格\n\n"
        f"- query/source：`{EXPERIMENTS[experiment].query}` 分别读取 "
        f"`{EXPERIMENTS[experiment].main_source}` 与相邻控制 "
        f"`{EXPERIMENTS[experiment].control_source}`。\n"
        "- 窗口：W1=L8–12，W2=L13–17，W3=L18–22，W4=L23–26；结果只能解释为窗口级效应。\n"
        "- `delta_soft_sa` 正值偏图像侧；`token_change_rate` 是相对 clean 的 hard class 改变率；"
        "`logit_change_diff` 正值表示 clean 决策 margin 被削弱。\n"
        "- `paired_main_vs_control.csv` 报告主阻断减 source+1 控制；主要推断口径是 answer-equal macro。\n"
        "- 阻断一个 pre-softmax edge 后，softmax 会把质量重新分配给同 row 的其他 source，故其他 attention weights 可以变化。\n"
        "- 正结果只说明该直接 edge 在指定窗口具有功能作用；负结果不能证明信息路径不存在。\n"
    )
    (tables / "README_zh.md").write_text(table_readme, encoding="utf-8")
    figures = _plots_from_csv(effect_path, root / "figures")
    all_windows = set(windows) == set(WINDOWS)
    result = {"status": "complete" if all_windows else "partial", "experiment": experiment,
              "smoke": smoke, "case_count": len(case_ids), "trial_count": expected,
              "blocked_trial_count": len(blocked), "attention_audit_rows": len(audit),
              "complete_windows": [list(window) for window in windows],
              "all_windows_complete": all_windows, "bootstrap_repeats": repeats,
              "tables": 5, "figures": figures}
    atomic_json(root / "progress" / "analysis.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bootstrap", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(experiment=args.experiment, output_root=args.output_root,
                             smoke=args.smoke, repeats=args.bootstrap), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
