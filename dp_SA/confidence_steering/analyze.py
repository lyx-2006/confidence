from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    BOOTSTRAP_REPEATS, DIRECTIONS, FORMAL_ROOT, NULL_EXPAND_P_THRESHOLD,
    PRIMARY_DIRECTION, SEED, SMOKE_BOOTSTRAP_REPEATS, STEERING_POSITION, TRUE_DIRECTIONS,
)
from .io_utils import atomic_csv, atomic_json, atomic_text, canonical_hash, ensure_layout, load_jsonl, semantic_fingerprint, sha256_file
from .run_spec import SHUFFLE_DIRECTION, add_run_spec_arguments, normalize_run_spec, run_spec_from_args

GROUPS = ("all", "follow_text", "follow_image", "conflict_easy", "conflict_hard", "answer_equal_macro", "family_micro")
ENDPOINTS = ("delta_confidence_LAT_immediate", "delta_confidence_PANL_L18", "delta_panl_probe_sa", "delta_final_soft_sa")


def family_draws(rows: Sequence[dict[str, Any]], repeats: int, seed: int = SEED) -> tuple[list[list[str]], str]:
    families = sorted({str(r["family_id"]) for r in rows}); rng = np.random.default_rng(seed)
    draws = [[str(x) for x in rng.choice(families, len(families), replace=True)] for _ in range(repeats)]
    return draws, canonical_hash(draws)


def _selected(rows: Sequence[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group in ("all", "answer_equal_macro", "family_micro"): return list(rows)
    if group in ("follow_text", "follow_image"): return [r for r in rows if r["answer_origin"] == group]
    return [r for r in rows if r["condition"] == group]


def _mean(rows: Sequence[dict[str, Any]], field: str, group: str) -> float:
    if not rows: return math.nan
    if group == "answer_equal_macro":
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows: grouped[row["fixed_answer"]].append(float(row[field]))
        return float(np.mean([np.mean(v) for v in grouped.values()]))
    if group == "family_micro":
        grouped = defaultdict(list)
        for row in rows: grouped[row["family_id"]].append(float(row[field]))
        return float(np.mean([np.mean(v) for v in grouped.values()]))
    return float(np.mean([float(r[field]) for r in rows]))


def summarize(rows: Sequence[dict[str, Any]], field: str, group: str, draws: Sequence[Sequence[str]]) -> dict[str, Any]:
    selected = _selected(rows, group); observed = _mean(selected, field, group)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected: by_family[row["family_id"]].append(row)
    boot = []
    for draw in draws:
        sample = [r for family in draw for r in by_family.get(family, [])]
        value = _mean(sample, field, group)
        if math.isfinite(value): boot.append(value)
    sem = float(np.std(boot, ddof=1)) if len(boot) > 1 else 0.0
    low, high = np.percentile(boot, [2.5, 97.5]) if boot else (math.nan, math.nan)
    hard_change = _mean(selected, "hard_sa_changed", group) if selected and "hard_sa_changed" in selected[0] else math.nan
    return {"mean_delta": observed, "sem": sem, "ci95_low": float(low), "ci95_high": float(high), "hard_change_rate": hard_change, "family_count": len({r["family_id"] for r in selected}), "item_count": len({r["item_id"] for r in selected}), "color_count": len({r["fixed_answer"] for r in selected}), "trial_count": len(selected), "valid_bootstrap_repeats": len(boot)}


def symmetric_effect(rows: Sequence[dict[str, Any]], field: str, dose: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in rows: grouped[row["case_id"], row["direction"], int(row["layer"])][float(row["alpha"])] = row
    output = []
    for (_case, direction, layer), values in grouped.items():
        if dose not in values or -dose not in values: continue
        source = values[dose]; output.append({**{k: source[k] for k in ("case_id", "item_id", "family_id", "condition", "answer_origin", "fixed_answer")}, "direction": direction, "layer": layer, "effect": (float(values[dose][field]) - float(values[-dose][field])) / 2.0})
    return output


def build_endpoint_table(trials: Sequence[dict[str, Any]], endpoint: str, draws: Sequence[Sequence[str]], cleanliness: dict[int, str], run_spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    run_spec = normalize_run_spec() if run_spec is None else run_spec
    output = []; layers = list(run_spec["layers"]); alphas = list(run_spec["alphas"])
    symmetric = {dose: symmetric_effect(trials, endpoint, dose) for dose in run_spec["paired_doses"]}
    for direction in run_spec["directions"]:
        for group in GROUPS:
            for layer in layers:
                effects = {}
                for dose in run_spec["paired_doses"]:
                    selected = [r for r in symmetric[dose] if r["direction"] == direction and r["layer"] == layer]
                    effects[dose] = summarize(selected, "effect", group, draws) if selected else None
                for alpha in alphas:
                    selected = [r for r in trials if r["direction"] == direction and int(r["layer"]) == layer and float(r["alpha"]) == alpha]
                    summary = summarize(selected, endpoint, group, draws)
                    effect_columns = {}
                    for dose, effect in effects.items():
                        label = f"S{dose:g}"
                        effect_columns.update({label: effect["mean_delta"] if effect else None, f"{label}_ci95_low": effect["ci95_low"] if effect else None, f"{label}_ci95_high": effect["ci95_high"] if effect else None})
                    output.append({"group": group, "direction": direction, "layer": layer, "alpha": alpha, **summary, **effect_columns, "cleanliness_status": cleanliness.get(layer, "not_evaluated")})
    return output


def empirical_null(main: Sequence[dict[str, Any]], null: Sequence[dict[str, Any]], endpoint: str, run_spec: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], bool]:
    run_spec = normalize_run_spec() if run_spec is None else run_spec
    output = []; expand = False
    directions = [d for d in run_spec["directions"] if d != SHUFFLE_DIRECTION]
    for layer in run_spec["layers"]:
        if not any(int(r["layer"]) == layer for r in null):
            continue
        for direction in directions:
            for dose in run_spec["paired_doses"]:
                true_rows = [r for r in symmetric_effect(main, endpoint, dose) if r["direction"] == direction and r["layer"] == layer]
                if not true_rows:
                    continue
                observed = _mean(true_rows, "effect", "answer_equal_macro")
                null_values = []
                for replicate in sorted({int(r["null_replicate"]) for r in null}):
                    rows = [r for r in null if int(r["null_replicate"]) == replicate and int(r["layer"]) == layer]
                    grouped: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
                    for row in rows:
                        grouped[row["case_id"]][float(row["alpha"])] = row
                    effect_rows = []
                    for values in grouped.values():
                        if dose in values and -dose in values:
                            source = values[dose]
                            effect_rows.append({**source, "effect": (float(values[dose][endpoint]) - float(values[-dose][endpoint])) / 2.0})
                    if effect_rows:
                        null_values.append(_mean(effect_rows, "effect", "answer_equal_macro"))
                b = len(null_values)
                one = (1 + sum(v >= observed for v in null_values)) / (b + 1)
                two = (1 + sum(abs(v) >= abs(observed) for v in null_values)) / (b + 1)
                percentile = 100.0 * sum(v < observed for v in null_values) / b if b else math.nan
                output.append({"endpoint": endpoint, "direction": direction, "layer": layer, "dose": dose, "true_symmetric_effect": observed, "null_repeats": b, "null_mean": float(np.mean(null_values)) if b else math.nan, "null_sd": float(np.std(null_values, ddof=1)) if b > 1 else math.nan, "true_percentile": percentile, "empirical_p_one_sided": one, "empirical_p_two_sided": two})
                if run_spec["analysis_kind"] == "legacy_confirmatory" and endpoint == "delta_final_soft_sa" and direction == PRIMARY_DIRECTION and layer == 14 and dose == 2.0:
                    expand |= b == 20 and one <= NULL_EXPAND_P_THRESHOLD
    return output, expand


def _plot_direction(table: Sequence[dict[str, Any]], path: Path, ylabel: str, direction: str) -> None:
    rows = [r for r in table if r["group"] == "answer_equal_macro" and r["direction"] == direction]
    layers = sorted({r["layer"] for r in rows}); alphas = sorted({r["alpha"] for r in rows})
    styles = ("-", "--", ":", "-.", (0, (3, 1, 1, 1)))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(alphas)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for index, (alpha, color) in enumerate(zip(alphas, colors, strict=True)):
        style = styles[index % len(styles)]
        data = sorted([r for r in rows if r["alpha"] == alpha], key=lambda r: r["layer"])
        mean = np.asarray([r["mean_delta"] for r in data]); low = np.asarray([r["ci95_low"] for r in data]); high = np.asarray([r["ci95_high"] for r in data])
        ax.errorbar([r["layer"] for r in data], mean, yerr=np.vstack((mean-low, high-mean)), color=color, ls=style, marker="o", capsize=3, label=f"α={alpha:+g}")
    ax.axhline(0, color="black", lw=.7); ax.set_xticks(layers); ax.set_xlabel("LAT layer"); ax.set_ylabel(ylabel); ax.set_title(direction); ax.legend(fontsize=8, ncol=min(3, len(alphas))); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=300); plt.close(fig)


def _plot_by_direction(table: Sequence[dict[str, Any]], directory: Path, ylabel: str, directions: Sequence[str] | None = None) -> list[Path]:
    paths = []
    for direction in (TRUE_DIRECTIONS if directions is None else directions):
        path = directory / f"{direction}.png"
        _plot_direction(table, path, ylabel, direction)
        paths.append(path)
    return paths


def _slope(rows: Sequence[dict[str, Any]], field: str) -> float:
    if not rows: return math.nan
    by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: by_answer[row["fixed_answer"]].append(row)
    values = []
    for selected in by_answer.values():
        x = np.asarray([float(r["alpha"]) for r in selected]); y = np.asarray([float(r[field]) for r in selected]); xc = x - x.mean(); denom = float(xc @ xc)
        if denom > 0: values.append(float(xc @ (y - y.mean()) / denom))
    return float(np.mean(values)) if values else math.nan


def local_slope_rows(trials: Sequence[dict[str, Any]], run_spec: dict[str, Any], draws: Sequence[Sequence[str]], endpoints: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    for direction in run_spec["directions"]:
        for layer in run_spec["layers"]:
            selected = [r for r in trials if r["direction"] == direction and int(r["layer"]) == layer]
            by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in selected: by_family[row["family_id"]].append(row)
            for endpoint in endpoints:
                observed = _slope(selected, endpoint); boot = []
                for draw in draws:
                    value = _slope([r for family in draw for r in by_family.get(family, [])], endpoint)
                    if math.isfinite(value): boot.append(value)
                low, high = np.percentile(boot, [2.5, 97.5]) if boot else (math.nan, math.nan)
                output.append({"position": STEERING_POSITION, "direction": direction, "layer": layer, "endpoint": endpoint, "slope": observed, "sem": float(np.std(boot, ddof=1)) if len(boot) > 1 else 0.0, "ci95_low": float(low), "ci95_high": float(high), "case_count": len({r["case_id"] for r in selected}), "family_count": len(by_family), "valid_bootstrap_repeats": len(boot)})
    return output


def symmetric_rows(trials: Sequence[dict[str, Any]], run_spec: dict[str, Any], draws: Sequence[Sequence[str]], endpoints: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    for endpoint in endpoints:
        for dose in run_spec["paired_doses"]:
            effects = symmetric_effect(trials, endpoint, dose)
            for direction in run_spec["directions"]:
                for layer in run_spec["layers"]:
                    selected = [r for r in effects if r["direction"] == direction and r["layer"] == layer]
                    if selected: output.append({"position": STEERING_POSITION, "direction": direction, "layer": layer, "dose": dose, "endpoint": endpoint, **summarize(selected, "effect", "answer_equal_macro", draws)})
    return output


def component_additivity_rows(trials: Sequence[dict[str, Any]], run_spec: dict[str, Any], draws: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    required = ("confidence_raw", "confidence_parallel_sa", "confidence_perp_sa_natural_scale")
    if not set(required) <= set(run_spec["directions"]): return []
    indexed = {(r["case_id"], int(r["layer"]), float(r["alpha"]), r["direction"]): r for r in trials}
    output = []
    for layer in run_spec["layers"]:
        for alpha in run_spec["alphas"]:
            for endpoint in ("delta_panl_probe_sa", "delta_final_soft_sa"):
                errors = []
                for case in sorted({r["case_id"] for r in trials}):
                    keys = [(case, layer, float(alpha), direction) for direction in required]
                    if not all(key in indexed for key in keys): continue
                    raw, parallel, perpendicular = (indexed[key] for key in keys)
                    errors.append({**{k: raw[k] for k in ("case_id", "item_id", "family_id", "condition", "answer_origin", "fixed_answer")}, "error": float(raw[endpoint]) - float(parallel[endpoint]) - float(perpendicular[endpoint])})
                if errors:
                    summary = summarize(errors, "error", "answer_equal_macro", draws)
                    output.append({"position": STEERING_POSITION, "layer": layer, "alpha": alpha, "endpoint": endpoint, "raw_direction": required[0], "parallel_direction": required[1], "perpendicular_direction": required[2], "additivity_error": summary.pop("mean_delta"), **summary})
    return output


def steering_effect_rows(tables: dict[str, list[dict[str, Any]]], run_spec: dict[str, Any]) -> list[dict[str, Any]]:
    endpoints = ("delta_confidence_LAT_immediate", "delta_panl_probe_sa", "delta_final_soft_sa", "clean_class_margin_change", "hard_sa_changed")
    prefixes = {"delta_confidence_LAT_immediate": "confidence_lat_delta", "delta_panl_probe_sa": "panl_sa_delta", "delta_final_soft_sa": "final_sa_delta", "clean_class_margin_change": "clean_class_margin_delta", "hard_sa_changed": "hard_change_rate"}
    indexes = {endpoint: {(r["direction"], int(r["layer"]), float(r["alpha"])): r for r in tables[endpoint] if r["group"] == "answer_equal_macro"} for endpoint in endpoints}
    output = []
    for direction in run_spec["directions"]:
        for layer in run_spec["layers"]:
            for alpha in run_spec["alphas"]:
                key = (direction, layer, float(alpha)); final = indexes["delta_final_soft_sa"][key]
                row = {"position": STEERING_POSITION, "direction": direction, "layer": layer, "alpha": alpha, "case_count": final["trial_count"], "family_count": final["family_count"], "item_count": final["item_count"]}
                for endpoint in endpoints:
                    value = indexes[endpoint][key]; prefix = prefixes[endpoint]
                    row.update({f"{prefix}_mean": value["mean_delta"], f"{prefix}_sem": value["sem"], f"{prefix}_ci95_low": value["ci95_low"], f"{prefix}_ci95_high": value["ci95_high"]})
                output.append(row)
    return output


def _audit_figures(root: Path) -> None:
    rows = list(csv.DictReader((root / "tables/orthogonalization_audit.csv").open()))
    directions = list(DIRECTIONS); layers = sorted({int(r["layer"]) for r in rows}); matrix = np.zeros((len(directions), len(directions))); counts = np.zeros_like(matrix)
    metadata = json.loads((root / "artifacts/directions/vector_metadata.json").read_text())["vectors"]
    index = {(int(r["layer"]), r["recipient_answer"], r["direction"]): r["scaled_key"] for r in metadata}
    recipients = sorted({r["recipient_answer"] for r in metadata})
    for layer in layers:
        with np.load(root / f"artifacts/directions/P1_LAT__L{layer}.npz") as payload:
            for recipient in recipients:
                values = [np.asarray(payload[index[layer, recipient, d]], float) for d in directions]
                for i, left in enumerate(values):
                    for j, right in enumerate(values):
                        matrix[i, j] += float(left @ right / np.linalg.norm(left) / np.linalg.norm(right)); counts[i, j] += 1
    matrix /= counts
    fig, ax = plt.subplots(figsize=(7, 6)); image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm"); ax.set_xticks(range(len(directions)), directions, rotation=45, ha="right", fontsize=7); ax.set_yticks(range(len(directions)), directions, fontsize=7); fig.colorbar(image, ax=ax, label="mean cosine across recipient × layer"); fig.tight_layout(); fig.savefig(root / "figures/direction_cosine_heatmap.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4));
    for direction in directions[1:]:
        values = [np.mean([float(r["retained_norm_ratio"]) for r in rows if r["direction"] == direction and int(r["layer"]) == layer]) for layer in layers]; ax.plot(layers, values, marker="o", label=direction)
    ax.axhline(.2, color="black", ls="--"); ax.set_xticks(layers); ax.set_ylabel("Mean retained norm ratio"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(root / "figures/retained_norm_ratio.png", dpi=300); plt.close(fig)


def analyze(*, output_root: Path = FORMAL_ROOT, smoke: bool = False, resume: bool = False, repeats: int | None = None, run_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    run_spec = normalize_run_spec() if run_spec is None else run_spec
    root = ensure_layout(output_root); main_path = root / "artifacts/trials/main_trials.jsonl"; null_path = root / "artifacts/trials/null_trials.jsonl"; trials = load_jsonl(main_path); null = load_jsonl(null_path) if null_path.is_file() else []
    material = json.loads((root / "artifacts/diagnostics/prelock_material.json").read_text())
    if material.get("run_spec") != run_spec: raise ValueError("Analysis run spec does not match prepared artifacts")
    if bool(null) != bool(run_spec["shuffle_requested"] and not smoke): raise ValueError("Null trials do not match run spec")
    repeats = int(repeats or (SMOKE_BOOTSTRAP_REPEATS if smoke else BOOTSTRAP_REPEATS)); config = {"main_sha256": sha256_file(main_path), "null_sha256": sha256_file(null_path) if null_path.is_file() else None, "smoke": smoke, "repeats": repeats, "seed": SEED, "run_spec": run_spec, "plot_layout": "endpoint_directory_per_direction_v2"}; fingerprint = semantic_fingerprint(root / "progress/analyze_config.json", config, resume=resume)
    progress = root / "progress/analyze.json"
    if resume and progress.is_file():
        old = json.loads(progress.read_text());
        if old.get("config_fingerprint") == fingerprint and old.get("status") == "complete": return {**old, "resumed_noop": True}
    draws, draw_fp = family_draws(trials, repeats); verdict = json.loads((root / "artifacts/diagnostics/prelock_verdict.json").read_text()); cleanliness = {int(k): ("reliable" if v else "unreliable_confidence_readout") for k, v in verdict["probe_layer_reliable"].items()}
    analysis_endpoints = ENDPOINTS + ("clean_class_margin_change", "hard_sa_changed")
    tables = {endpoint: build_endpoint_table(trials, endpoint, draws, cleanliness, run_spec) for endpoint in analysis_endpoints}
    if run_spec["analysis_kind"] == "legacy_confirmatory":
        atomic_csv(root / "tables/table2_panl_sa.csv", tables["delta_panl_probe_sa"]); atomic_csv(root / "tables/table1_final_sa.csv", tables["delta_final_soft_sa"])
    else:
        atomic_csv(root / "tables/steering_effects.csv", steering_effect_rows(tables, run_spec))
        slope_endpoints = ENDPOINTS + ("clean_class_margin_change",)
        atomic_csv(root / "tables/local_slopes.csv", local_slope_rows(trials, run_spec, draws, slope_endpoints))
        atomic_csv(root / "tables/symmetric_effects.csv", symmetric_rows(trials, run_spec, draws, slope_endpoints))
        additivity = component_additivity_rows(trials, run_spec, draws); atomic_csv(root / "tables/component_additivity.csv", additivity)
    plotted = [direction for direction in run_spec["directions"] if direction != SHUFFLE_DIRECTION]
    final_figures = _plot_by_direction(tables["delta_final_soft_sa"], root / "figures/final", "Mean final Δ soft SA", plotted)
    panl_figures = _plot_by_direction(tables["delta_panl_probe_sa"], root / "figures/panl", "Mean Δ SA (PANL L18 frozen probe)", plotted)
    manipulation = tables["delta_confidence_LAT_immediate"] + tables["delta_confidence_PANL_L18"]
    atomic_csv(root / "tables/confidence_manipulation_checks.csv", manipulation)
    null_rows = []; expand = False
    if run_spec["shuffle_requested"] and null:
        for endpoint in ENDPOINTS:
            rows, decision = empirical_null(trials, null, endpoint, run_spec); null_rows.extend(rows); expand |= decision
        atomic_csv(root / "tables/l14_shuffle_null.csv", null_rows)
    if run_spec["analysis_kind"] == "legacy_confirmatory":
        primary = next((r for r in tables["delta_final_soft_sa"] if r["direction"] == PRIMARY_DIRECTION and r["layer"] == 14 and r["group"] == "answer_equal_macro"), None)
        if primary is not None and primary.get("S2_ci95_low") is not None and primary["S2_ci95_low"] > 0 and len({r.get("null_replicate") for r in null}) == 20: expand = True
    atomic_text(root / "tables/README.md", "# Tables\n\n`heldout_projection_audit.csv`仅使用outer_fold=0的230条方向审计数据，不包含正式100条test。confidence manipulation probes与独立SA probe均未参与方向或正交基构造。\n")
    boundary = "本实验只评估confidence-related direction orthogonal to measured difficulty and LAT-SA directions。数值正交是实现检查；独立SA probe仅衡量对一个额外已测SA readout的去除程度，不代表所有SA已删除。confidence变化仅由独立probe提供manipulation-check支持。"
    if run_spec["analysis_kind"] == "mechanism_diagnostic":
        atomic_text(root / "summary.md", f"# LAT confidence自然尺度SA分解\n\n- analysis_kind: `mechanism_diagnostic`\n- smoke_only: `{str(smoke).lower()}`\n- formal_test_opened: `{str(not smoke).lower()}`\n- directions: `{', '.join(run_spec['directions'])}`\n- layers: `{run_spec['layers']}`\n- alphas: `{run_spec['alphas']}`\n- paired_doses: `{run_spec['paired_doses']}`\n- bootstrap_repeats: `{repeats}`\n\n本配置比较原始confidence方向、SA平行分量和保持自然相对长度的SA正交分量。`confidence_perp_sa_natural_scale`仅表示对当前线性SA子空间正交的confidence相关分量，不称为纯confidence。\n\n本配置是机制诊断，不继承旧的L14正向SA确认性假设，也未请求shuffle null。PANL/final变化是功能结果；局部加和仅是小剂量近似。\n\n> {boundary}\n")
    else:
        atomic_text(root / "summary.md", f"# LAT confidence正交steering\n\n- analysis_kind: `{run_spec['analysis_kind']}`\n- smoke_only: `{str(smoke).lower()}`\n- formal_test_opened: `{str(not smoke).lower()}`\n- paired_doses: `{run_spec['paired_doses']}`\n- bootstrap_repeats: `{repeats}`\n\n> {boundary}\n")
    result = {"status": "complete", "smoke_only": smoke, "analysis_kind": run_spec["analysis_kind"], "run_spec_fingerprint": run_spec["fingerprint"], "main_trials": len(trials), "null_trials": len(null), "bootstrap_repeats": repeats, "bootstrap_draw_fingerprint": draw_fp, "expand_null_to_99": bool(expand and run_spec["analysis_kind"] == "legacy_confirmatory" and not smoke and len({r["null_replicate"] for r in null}) == 20), "main_tables": 2 if run_spec["analysis_kind"] == "legacy_confirmatory" else 4, "main_figures": len(final_figures) + len(panl_figures), "figure_directories": {"final": str((root / "figures/final").resolve()), "panl": str((root / "figures/panl").resolve())}, "config_fingerprint": fingerprint, "resumed_noop": False}; atomic_json(progress, result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--bootstrap", type=int); add_run_spec_arguments(parser); args = parser.parse_args(argv); print(json.dumps(analyze(output_root=Path(args.output_root) if args.output_root else FORMAL_ROOT, smoke=args.smoke, resume=args.resume, repeats=args.bootstrap, run_spec=run_spec_from_args(args)), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
