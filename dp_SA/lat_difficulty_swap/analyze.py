from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import t as student_t

from .config import BOOTSTRAP_REPEATS, RESULTS_ROOT, SEED
from .io_utils import atomic_csv, atomic_json, atomic_jsonl, atomic_text, load_jsonl, stage_update
from .metrics import bh_fdr, clustered_ols, item_bootstrap, sign_flip_p, stable_seed

CROSS = ("A_E2H", "A_H2E", "B_TE2TH", "B_TH2TE")


def _stats(rows: Sequence[dict[str, Any]], field: str, repeats: int, seed: int) -> dict[str, Any]:
    return item_bootstrap(rows, lambda row: float(row[field]), repeats=repeats, seed=stable_seed(seed, field, len(rows), rows[0].get("condition", "pooled")))


def _ci(stats: dict[str, Any]) -> str:
    return json.dumps([stats["ci_low"], stats["ci_high"]], separators=(",", ":"))


def _groups(rows: Sequence[dict[str, Any]], layer: int | None = None) -> list[tuple[str, list[dict[str, Any]]]]:
    selected = list(rows) if layer is None else [row for row in rows if int(row["layer"]) == layer]
    suffix = "ALL_LAYERS" if layer is None else f"L{layer}"
    output = []
    for condition in CROSS:
        part = [row for row in selected if row["condition"] == condition]
        if part: output.append((f"{condition}_{suffix}", part))
    for arm, label in (("A", "A_POOLED"), ("B", "B_POOLED")):
        part = [row for row in selected if row["arm"] == arm]
        if part: output.append((f"{label}_{suffix}", part))
    if selected: output.append((f"AB_POOLED_{suffix}", selected))
    return output


def _wide(path: Path, values: dict[str, dict[str, Any]], metrics: Sequence[str]) -> None:
    conditions = list(values)
    rows = [{"metric": metric, **{condition: values[condition].get(metric) for condition in conditions}} for metric in metrics]
    atomic_csv(path, rows, ["metric", *conditions])


def _delta_table(rows: Sequence[dict[str, Any]], layers: Sequence[int], repeats: int, seed: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    fields = {
        "signed_delta_sa": "delta_sa", "raw_absolute_delta_sa": "raw_absolute_delta_sa",
        "oriented_delta_sa": "oriented_delta_sa", "toward_target_absolute_delta_sa": "toward_target_absolute_delta_sa",
        "wrong_direction_absolute_delta_sa": "wrong_direction_absolute_delta_sa", "toward_target_rate": "toward_target",
        "wrong_way_rate": "wrong_way", "no_change_rate": "no_change", "hard_class_change_rate": "hard_class_changed",
        "hard_class_toward_target_rate": "hard_class_toward_target", "hard_class_wrong_way_rate": "hard_class_wrong_way",
    }
    for layer in (*layers, None):
        for name, part in _groups(rows, layer):
            cell: dict[str, Any] = {}
            for metric, field in fields.items():
                stats = _stats(part, field, repeats, stable_seed(seed, name, metric)); cell[f"mean_{metric}" if not metric.endswith("rate") else metric] = stats["mean"]; cell[f"{metric}_ci"] = _ci(stats)
            oriented = _stats(part, "oriented_delta_sa", repeats, stable_seed(seed, name, "confirmatory"))
            cell.update({"pair_count": oriented["pair_count"], "item_count": oriented["item_count"], "sem_oriented_delta_sa": oriented["sem"], "valid_bootstrap_repeats": oriented["valid_bootstrap_repeats"], "toward_minus_wrong_delta_sa": cell["mean_toward_target_absolute_delta_sa"] - cell["mean_wrong_direction_absolute_delta_sa"], "confirmatory_ci_low_gt_zero": bool(layer is None and oriented["ci_low"] > 0)})
            if abs(cell["toward_minus_wrong_delta_sa"] - cell["mean_oriented_delta_sa"]) > 1e-12: raise AssertionError("Toward/wrong/oriented aggregate identity failed")
            result[name] = cell
    exploratory = []
    for name, cell in result.items():
        if "_L" in name and not name.endswith("ALL_LAYERS"):
            layer = int(name.rsplit("L", 1)[1]); condition = name.rsplit("_L", 1)[0]
            part = [row for row in rows if int(row["layer"]) == layer and (row["condition"] == condition or condition == "A_POOLED" and row["arm"] == "A" or condition == "B_POOLED" and row["arm"] == "B" or condition == "AB_POOLED")]
            p = sign_flip_p(part, "oriented_delta_sa", repeats=repeats, seed=stable_seed(seed, name, "p")); exploratory.append((name, p))
    adjusted = bh_fdr([value for _, value in exploratory])
    for (name, p), q in zip(exploratory, adjusted, strict=True): result[name]["oriented_delta_sa_p_value"] = p; result[name]["oriented_delta_sa_bh_q_value"] = q
    return result


def _probe_table(rows: Sequence[dict[str, Any]], layers: Sequence[int], repeats: int, seed: int) -> dict[str, dict[str, Any]]:
    result = {}; probes = ("panl_sa_prediction", "panl_text_difficulty_prediction", "panl_image_difficulty_prediction", "panl_decision_follow_image_probability")
    for layer in layers:
        for name, part in _groups(rows, layer):
            if "POOLED" not in name and not any(name.startswith(condition) for condition in CROSS): continue
            cell: dict[str, Any] = {"panl_readout_layer": part[0]["panl_readout_layer"], "pair_count": len({row["pair_id"] for row in part}), "item_count": len({row["item_id"] for row in part})}
            for probe in probes:
                clean_field, swap_field, delta_field = probe, probe, f"{probe}_delta"
                clean = [float(row["clean_probes"][clean_field]) for row in part]; swap = [float(row["swap_probes"][swap_field]) for row in part]
                stats = _stats(part, delta_field, repeats, stable_seed(seed, name, probe))
                prefix = {"panl_sa_prediction": "panl_sa", "panl_text_difficulty_prediction": "text_difficulty", "panl_image_difficulty_prediction": "image_difficulty", "panl_decision_follow_image_probability": "decision_follow_image"}[probe]
                cell.update({f"{prefix}_clean": float(np.mean(clean)), f"{prefix}_swap": float(np.mean(swap)), f"{prefix}_delta": stats["mean"], f"{prefix}_delta_ci": _ci(stats)})
                if probe in ("panl_sa_prediction", "panl_decision_follow_image_probability"):
                    oriented_values = [{**row, "_oriented_probe": float(row[delta_field]) * int(row["target_sign"])} for row in part]
                    oriented = item_bootstrap(oriented_values, lambda row: row["_oriented_probe"], repeats=repeats, seed=stable_seed(seed, name, probe, "oriented")); cell[f"{prefix}_oriented_delta"] = oriented["mean"]; cell[f"{prefix}_oriented_delta_ci"] = _ci(oriented)
            text_abs = np.mean([abs(float(row["panl_text_difficulty_prediction_delta"])) for row in part]); image_abs = np.mean([abs(float(row["panl_image_difficulty_prediction_delta"])) for row in part])
            cell["modality_specificity"] = float(image_abs - text_abs if part[0]["arm"] == "A" else text_abs - image_abs) if len({row["arm"] for row in part}) == 1 else None
            result[name] = cell
    return result


def _logit_table(rows: Sequence[dict[str, Any]], layers: Sequence[int], repeats: int, seed: int) -> dict[str, dict[str, Any]]:
    result = {}
    fields = ("logit_diff_change", "logit_disruption", "hard_class_changed", "hard_class_toward_target", "hard_class_wrong_way")
    for layer in layers:
        for name, part in _groups(rows, layer):
            cell = {}
            for field in fields:
                stats = _stats(part, field, repeats, stable_seed(seed, name, field)); label = {"hard_class_changed": "token_change_rate", "hard_class_toward_target": "hard_class_toward_target_rate", "hard_class_wrong_way": "hard_class_wrong_way_rate"}.get(field, f"mean_{field}")
                cell[label] = stats["mean"]; cell[f"{label}_ci"] = _ci(stats)
            result[name] = cell
    return result


def _regression(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for group in ("A", "B", "AB"):
        part = [row for row in rows if group == "AB" or row["arm"] == group]
        conditions = sorted({row["condition"] for row in part}); layers = sorted({int(row["layer"]) for row in part})
        X = []
        for row in part:
            X.append([1.0, float(row["donor_recipient_difficulty_gap"]), *[float(row["condition"] == value) for value in conditions[1:]], *[float(int(row["layer"]) == value) for value in layers[1:]]])
        fit = clustered_ols(np.asarray(X), np.asarray([row["delta_sa"] for row in part]), [str(row["item_id"]) for row in part]); coefficient = float(fit["coefficient"][1]); se = float(fit["standard_error"][1]); critical = float(student_t.ppf(.975, max(fit["cluster_count"] - 1, 1)))
        output.append({"group": group, "coefficient": coefficient, "cluster_robust_se": se, "ci_low": coefficient - critical * se, "ci_high": coefficient + critical * se, "p_value": float(fit["p_value"][1]), "r2": fit["r2"], "pair_count": len({row["pair_id"] for row in part}), "item_count": len({row["item_id"] for row in part})})
    return output


def _plots(root: Path, rows: Sequence[dict[str, Any]], delta: dict[str, dict[str, Any]], probe: dict[str, dict[str, Any]], logit: dict[str, dict[str, Any]], regressions: Sequence[dict[str, Any]], layers: Sequence[int]) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    styles = {"A_E2H": ("#0072B2", "o", "-"), "A_H2E": ("#D55E00", "s", "-"), "B_TE2TH": ("#009E73", "^", "-"), "B_TH2TE": ("#CC79A7", "D", "-")}
    def series(table: dict[str, dict[str, Any]], prefix: str, metric: str) -> list[float]: return [float(table[f"{prefix}_L{layer}"][metric]) for layer in layers]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    panels = (("A", ("A_E2H", "A_H2E", "A_POOLED")), ("B", ("B_TE2TH", "B_TH2TE", "B_POOLED")), ("A+B", ("A_POOLED", "B_POOLED", "AB_POOLED")))
    for ax, (title, names) in zip(axes, panels):
        for index, name in enumerate(names):
            color, marker, _ = styles.get(name, (("#0072B2", "#009E73", "#333333")[index], "o", "-"))
            ax.plot(layers, series(delta, name, "mean_toward_target_absolute_delta_sa"), color=color, marker=marker, label=f"{name} toward")
            ax.plot(layers, series(delta, name, "mean_wrong_direction_absolute_delta_sa"), color=color, marker=marker, linestyle="--", alpha=.75, label=f"{name} wrong")
        ax.axhline(0, color="black", lw=.7); ax.set_title(title); ax.set_xlabel("LAT swap layer"); ax.set_xticks(layers)
    axes[0].set_ylabel("Mean directional magnitude"); axes[-1].legend(fontsize=7); fig.suptitle("Descriptive toward-target and wrong-direction magnitudes")
    fig.savefig(root / "figures" / "target_directed_absolute_delta_sa.png", dpi=320); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True); metrics = (("mean_oriented_delta_sa", "Oriented ΔSA"), ("mean_raw_absolute_delta_sa", "Raw |ΔSA|"), ("mean_toward_target_absolute_delta_sa", "Toward magnitude"), ("mean_wrong_direction_absolute_delta_sa", "Wrong magnitude"), ("toward_target_rate", "Toward rate"), ("wrong_way_rate", "Wrong-way rate"))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for name, (color, marker, line) in styles.items(): ax.plot(layers, series(delta, name, metric), color=color, marker=marker, linestyle=line, label=name)
        ax.axhline(0, color="black", lw=.7); ax.set_title(title); ax.set_xticks(layers)
    axes[0, 0].legend(fontsize=7); fig.savefig(root / "figures" / "oriented_and_raw_absolute_delta_sa.png", dpi=320); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric, title in zip(axes, ("mean_logit_diff_change", "token_change_rate"), ("Fixed-clean-class margin change", "Token change rate")):
        for name, (color, marker, line) in styles.items(): ax.plot(layers, series(logit, name, metric), color=color, marker=marker, linestyle=line, label=name)
        ax.axhline(0, color="black", lw=.7); ax.set_title(title); ax.set_xlabel("LAT swap layer"); ax.set_xticks(layers)
    axes[0].legend(fontsize=8); fig.savefig(root / "figures" / "logit_and_token_change.png", dpi=320); plt.close(fig)

    probe_metrics = (("panl_sa_delta", "PANL SA probe Δ"), ("text_difficulty_delta", "Text difficulty probe Δ"), ("image_difficulty_delta", "Image difficulty probe Δ"), ("decision_follow_image_delta", "Follow-image probability Δ"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, probe_metrics):
        for name, (color, marker, line) in styles.items(): ax.plot(layers, series(probe, name, metric), color=color, marker=marker, linestyle=line, label=name)
        ax.axhline(0, color="black", lw=.7); ax.set_title(title); ax.set_xticks(layers)
    axes[0, 0].legend(fontsize=8); fig.suptitle("LAT→PANL mapping: " + ", ".join(f"L{k}→L{v}" for k, v in zip(layers, [probe[f'A_E2H_L{k}']['panl_readout_layer'] for k in layers])))
    fig.savefig(root / "figures" / "panl_probe_changes.png", dpi=320); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, group in zip(axes, ("A", "B", "AB")):
        part = [row for row in rows if group == "AB" or row["arm"] == group]; x = np.asarray([row["donor_recipient_difficulty_gap"] for row in part]); y = np.asarray([row["delta_sa"] for row in part]); ax.scatter(x, y, s=8, alpha=.18)
        if len(set(x)) > 1:
            design = np.column_stack([np.ones(len(x)), x]); fit = clustered_ols(design, y, [row["item_id"] for row in part]); grid = np.linspace(x.min(), x.max(), 100); beta = fit["coefficient"][:2]; ax.plot(grid, beta[0] + beta[1] * grid, color="#D55E00", lw=2)
            covariance = np.asarray(fit["covariance"])[:2, :2]; grid_design = np.column_stack([np.ones(len(grid)), grid]); prediction_se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", grid_design, covariance, grid_design), 0)); critical = float(student_t.ppf(.975, max(fit["cluster_count"] - 1, 1))); band = critical * prediction_se; ax.fill_between(grid, beta[0] + beta[1] * grid - band, beta[0] + beta[1] * grid + band, color="#D55E00", alpha=.15)
        ax.axhline(0, color="black", lw=.7); ax.set_title(group); ax.set_xlabel("Donor − recipient difficulty gap")
    axes[0].set_ylabel("Swap − clean ΔSA"); fig.savefig(root / "figures" / "difficulty_gap_dose_response.png", dpi=320); plt.close(fig)


def analyze(root: Path, *, repeats: int = BOOTSTRAP_REPEATS, seed: int = SEED, resume: bool = False) -> dict[str, Any]:
    all_rows = load_jsonl(root / "artifacts" / "swap_results.jsonl", repair_trailing=True); rows = [row for row in all_rows if row.get("swap_kind") == "cross"]
    if not rows: raise ValueError("No completed cross-swap rows")
    execution = json.loads((root / "progress" / "execution_config.json").read_text())
    from .io_utils import canonical_hash, sha256_file
    analysis_config = {"execution_fingerprint": execution["fingerprint"], "swap_results_sha256": sha256_file(root / "artifacts" / "swap_results.jsonl"), "bootstrap": int(repeats), "seed": int(seed), "analysis_schema_version": 1}; analysis_config["fingerprint"] = canonical_hash(analysis_config)
    analysis_path = root / "progress" / "analysis_config.json"
    if analysis_path.exists():
        previous = json.loads(analysis_path.read_text())
        if previous.get("fingerprint") != analysis_config["fingerprint"]: raise ValueError("Analysis fingerprint mismatch; refusing resume")
        if not resume: raise FileExistsError("Analysis already exists; pass --resume")
    else: atomic_json(analysis_path, analysis_config)
    layers = sorted({int(row["layer"]) for row in rows}); expected = len(load_jsonl(root / "artifacts" / "image_pair_manifest.jsonl") + load_jsonl(root / "artifacts" / "text_pair_manifest.jsonl"))
    if not all(np.isfinite([row["delta_sa"], row["oriented_delta_sa"], row["logit_diff_change"]]).all() for row in rows): raise ValueError("Non-finite analysis result")
    atomic_jsonl(root / "artifacts" / "item_level_metrics.jsonl", rows)
    delta = _delta_table(rows, layers, repeats, seed); probe = _probe_table(rows, layers, repeats, seed); logit = _logit_table(rows, layers, repeats, seed); regression = _regression(rows)
    delta_metrics = ["pair_count", "item_count", "mean_signed_delta_sa", "signed_delta_sa_ci", "mean_raw_absolute_delta_sa", "raw_absolute_delta_sa_ci", "mean_oriented_delta_sa", "oriented_delta_sa_ci", "sem_oriented_delta_sa", "mean_toward_target_absolute_delta_sa", "toward_target_absolute_delta_sa_ci", "mean_wrong_direction_absolute_delta_sa", "wrong_direction_absolute_delta_sa_ci", "toward_minus_wrong_delta_sa", "toward_target_rate", "toward_target_rate_ci", "wrong_way_rate", "wrong_way_rate_ci", "no_change_rate", "no_change_rate_ci", "hard_class_change_rate", "hard_class_toward_target_rate", "hard_class_wrong_way_rate", "valid_bootstrap_repeats", "confirmatory_ci_low_gt_zero", "oriented_delta_sa_p_value", "oriented_delta_sa_bh_q_value"]
    _wide(root / "tables" / "delta_sa.csv", delta, delta_metrics)
    _wide(root / "tables" / "panl_probe_results.csv", probe, sorted({key for cell in probe.values() for key in cell}))
    _wide(root / "tables" / "logit_token_metrics.csv", logit, sorted({key for cell in logit.values() for key in cell}))
    atomic_csv(root / "tables" / "difficulty_gap_regression.csv", regression)
    figure_data = {"layers": layers, "delta": delta, "probe": probe, "logit": logit, "regression": regression}; atomic_json(root / "artifacts" / "figure_data.json", figure_data)
    _plots(root, rows, delta, probe, logit, regression, layers)
    confirmatory = {name: delta[name] for name in ("A_POOLED_ALL_LAYERS", "B_POOLED_ALL_LAYERS", "AB_POOLED_ALL_LAYERS")}
    lines = ["# LAT difficulty swap summary", "", "## Confirmatory oriented ΔSA", "", "The confirmatory endpoint is the item-bootstrap mean oriented ΔSA; truncated directional magnitudes are descriptive only.", ""]
    for name, cell in confirmatory.items(): lines.append(f"- {name}: mean={cell['mean_oriented_delta_sa']:.8g}, 95% CI={cell['oriented_delta_sa_ci']}, CI lower > 0={cell['confirmatory_ci_low_gt_zero']}")
    lines += ["", "## Interpretation guardrails", "", "Claims require aligned movement in the corresponding difficulty probe, PANL-SA readout, and final SA. PANL movement without final movement indicates downstream correction; final movement without PANL movement does not establish PANL mediation. Large raw absolute change with near-zero oriented change is consistent with generic replacement disruption. Self-swap parity failure invalidates the experiment. Null results do not prove that LAT encodes only the answer string.", ""]
    atomic_text(root / "summary.md", "\n".join(lines)); summary = {"status": "complete", "cross_row_count": len(rows), "pair_count": len({row["pair_id"] for row in rows}), "item_count": len({row["item_id"] for row in rows}), "layers": layers, "confirmatory": confirmatory}
    atomic_json(root / "artifacts" / "analysis_summary.json", summary); stage_update(root, "analysis", "complete", cross_row_count=len(rows), expected_formal_pairs=expected); return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze LAT difficulty swaps"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT); parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPEATS); parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv); print(json.dumps(analyze(args.output_root.resolve(), repeats=args.bootstrap, seed=args.seed, resume=args.resume), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
