from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .config import (
    ALPHAS, BOOTSTRAP_REPEATS, CANONICAL_ANSWERS, CONFIRMATORY_ANSWERS,
    DOSES, PANL_MEDIATOR_LAYERS, RESULTS_ROOT, SEED, SMOKE_ALPHAS,
    SMOKE_BOOTSTRAP_REPEATS, SMOKE_LAYERS,
)
from .io_utils import atomic_csv, atomic_json, atomic_jsonl, canonical_hash, load_bf16_npz, load_jsonl, sha256_file


class SharedBootstrap:
    """Classical clustered bootstrap driven by one shared uniform stream."""
    def __init__(self, rows: Sequence[dict[str, Any]], repeats: int, seed: int = SEED) -> None:
        self.repeats = repeats; self.answer = {}
        for row in rows: self.answer.setdefault(str(row["family_id"]), str(row["answer"]))
        rng = np.random.default_rng(seed); maximum = max(1, len(rows))
        self.uniforms = {key: rng.random((repeats, maximum)) for key in (*CANONICAL_ANSWERS, "__all__")}
        self.design_hash = canonical_hash({key: value.tolist() for key, value in self.uniforms.items()})

    def aggregate(self, values: dict[str, float], group: str) -> tuple[float, np.ndarray]:
        if not values: return float("nan"), np.full(self.repeats, np.nan)
        if group == "family_micro":
            families = sorted(values); vector = np.asarray([values[key] for key in families], dtype=float)
            indexes = np.floor(self.uniforms["__all__"][:, :len(families)] * len(families)).astype(int)
            return float(vector.mean()), vector[indexes].mean(axis=1)
        answers = CONFIRMATORY_ANSWERS if group == "answer_equal_macro" else CANONICAL_ANSWERS
        observed = []; boot = []
        for answer in answers:
            families = sorted(key for key in values if self.answer[key] == answer)
            if not families: continue
            vector = np.asarray([values[key] for key in families], dtype=float)
            indexes = np.floor(self.uniforms[answer][:, :len(families)] * len(families)).astype(int)
            observed.append(float(vector.mean())); boot.append(vector[indexes].mean(axis=1))
        if not observed: return float("nan"), np.full(self.repeats, np.nan)
        return float(np.mean(observed)), np.stack(boot).mean(axis=0)


def ratio_summary(total: float, numerator: float, total_boot: np.ndarray,
                  numerator_boot: np.ndarray) -> dict[str, Any]:
    low, high = np.percentile(total_boot, [2.5, 97.5])
    point_sign = np.sign(total)
    same_sign = float(np.mean(np.sign(total_boot) == point_sign)) if point_sign else 0.0
    reportable = bool((low > 0 or high < 0) and same_sign >= .975 and total != 0)
    if not reportable:
        return {"ratio": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "denominator_same_sign_rate": same_sign, "ratio_reportable": False}
    ratios = np.divide(numerator_boot, total_boot, out=np.full_like(total_boot, np.nan), where=total_boot != 0)
    finite = ratios[np.isfinite(ratios)]; ratio_low, ratio_high = np.percentile(finite, [2.5, 97.5])
    return {"ratio": numerator / total, "ci_low": float(ratio_low), "ci_high": float(ratio_high),
            "denominator_same_sign_rate": same_sign, "ratio_reportable": True}


def _logical_rows(trials: list[dict[str, Any]], *, smoke: bool) -> list[dict[str, Any]]:
    alphas = SMOKE_ALPHAS if smoke else ALPHAS; layers = SMOKE_LAYERS if smoke else PANL_MEDIATOR_LAYERS
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trials: by_case[str(row["case_id"])].append(row)
    output = []
    for case_id, rows in by_case.items():
        c0 = next(row for row in rows if row["condition"] == "C0")
        lookup = {(row["condition"], row["panl_mediator_layer"], float(row["alpha"])): row for row in rows}
        cells = []
        for alpha in alphas:
            cells.append(("C0", None, alpha, c0))
            cells.append(("C1", None, alpha, c0 if alpha == 0 else lookup["C1", None, float(alpha)]))
            for layer in layers:
                for condition in ("C2", "C3"):
                    cells.append((condition, layer, alpha, c0 if alpha == 0 else lookup[condition, layer, float(alpha)]))
        for condition, layer, alpha, row in cells:
            base = {key: row[key] for key in ("case_id", "family_id", "item_id", "image_sha256", "answer", "test_side",
                                                "cle_probe_eligible", "cle_probe_exclusion_reasons")}
            output.append({**base, "condition": condition, "panl_mediator_layer": layer, "alpha": float(alpha),
                           "final_soft_sa": row["final_soft_sa"], "clean_final_soft_sa": c0["final_soft_sa"],
                           "delta_final_sa": float(row["final_soft_sa"]) - float(c0["final_soft_sa"]),
                           "hard_sa_class": row["hard_sa_class"], "clean_hard_sa_class": c0["hard_sa_class"],
                           "hard_change": int(row["hard_sa_class"] != c0["hard_sa_class"]),
                           "fixed_clean_class_margin": _margin(row["class_logits"], int(c0["hard_sa_class"])),
                           "cle_probe_sa": row["cle_probe_sa"], "clean_cle_probe_sa": c0["cle_probe_sa"],
                           "delta_cle_probe_sa": None if row["cle_probe_sa"] is None else float(row["cle_probe_sa"]) - float(c0["cle_probe_sa"])})
    return output


def _margin(logits: Sequence[float], selected: int) -> float:
    vector = np.asarray(logits, dtype=float); return float(vector[selected] - np.max(np.delete(vector, selected)))


def _counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {"case_count": len({row["case_id"] for row in rows}), "family_count": len({row["family_id"] for row in rows}),
            "item_count": len({row["item_id"] for row in rows})}


def _value_map(rows: Sequence[dict[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        family = str(row["family_id"])
        value = row[field]
        if value is not None: grouped[family].append(float(value))
    return {family: float(np.mean(values)) for family, values in grouped.items()}


def condition_table(logical: list[dict[str, Any]], bootstrap: SharedBootstrap) -> list[dict[str, Any]]:
    output = []; groups = ("answer_equal_macro", "family_micro", "all")
    endpoints = (("final_soft_sa", "delta_final_sa"), ("cle_probe_sa", "delta_cle_probe_sa"))
    cells = sorted({(row["condition"], row["panl_mediator_layer"], float(row["alpha"])) for row in logical},
                   key=lambda key: (key[0], key[1] or -1, key[2]))
    for endpoint, field in endpoints:
        for condition, layer, alpha in cells:
            selected = [row for row in logical if row["condition"] == condition and row["panl_mediator_layer"] == layer
                        and float(row["alpha"]) == alpha and row[field] is not None]
            values = _value_map(selected, field)
            hard_values = _value_map(selected, "hard_change") if endpoint == "final_soft_sa" else {}
            counts = _counts(selected)
            for group in groups:
                observed, boot = bootstrap.aggregate(values, group); low, high = np.percentile(boot, [2.5, 97.5])
                hard = bootstrap.aggregate(hard_values, group)[0] if hard_values else np.nan
                output.append({"endpoint": endpoint, "condition": condition, "panl_mediator_layer": layer,
                               "alpha": alpha, "group": group, "mean_delta": observed,
                               "sem": float(np.std(boot, ddof=1)), "ci95_low": float(low), "ci95_high": float(high),
                               **counts, "hard_change_rate": hard})
    return output


def mediation_table(logical: list[dict[str, Any]], bootstrap: SharedBootstrap, *, smoke: bool) -> list[dict[str, Any]]:
    output = []; layers = SMOKE_LAYERS if smoke else PANL_MEDIATOR_LAYERS
    doses = (2.0,) if smoke else DOSES
    for endpoint, field in (("final_soft_sa", "delta_final_sa"), ("cle_probe_sa", "delta_cle_probe_sa")):
        endpoint_rows = [row for row in logical if row[field] is not None]
        for layer in layers:
            for dose in doses:
                def symmetric(condition: str, selected_layer: int | None) -> dict[str, float]:
                    positive = _value_map([row for row in endpoint_rows if row["condition"] == condition and row["panl_mediator_layer"] == selected_layer and row["alpha"] == dose], field)
                    negative = _value_map([row for row in endpoint_rows if row["condition"] == condition and row["panl_mediator_layer"] == selected_layer and row["alpha"] == -dose], field)
                    if set(positive) != set(negative): raise ValueError("Unpaired symmetric-effect families")
                    return {key: (positive[key] - negative[key]) / 2 for key in positive}
                total = symmetric("C1", None); residual = symmetric("C2", layer); transfer = symmetric("C3", layer)
                attenuation = {key: total[key] - residual[key] for key in total}
                for group in ("answer_equal_macro", "family_micro", "all"):
                    t, tb = bootstrap.aggregate(total, group); r, rb = bootstrap.aggregate(residual, group)
                    a, ab = bootstrap.aggregate(attenuation, group); p, pb = bootstrap.aggregate(transfer, group)
                    al, ah = np.percentile(ab, [2.5, 97.5]); pl, ph = np.percentile(pb, [2.5, 97.5])
                    recovery = ratio_summary(t, a, tb, ab); transfer_ratio = ratio_summary(t, p, tb, pb)
                    selected = [row for row in endpoint_rows if row["condition"] == "C1" and row["alpha"] == dose]
                    output.append({"endpoint": endpoint, "panl_mediator_layer": layer, "dose": dose, "group": group,
                                   "S_total": t, "S_total_ci_low": float(np.percentile(tb, 2.5)), "S_total_ci_high": float(np.percentile(tb, 97.5)),
                                   "S_residual_after_restore": r, "S_attenuation": a,
                                   "S_attenuation_ci_low": float(al), "S_attenuation_ci_high": float(ah),
                                   "S_panl_transfer": p, "S_panl_transfer_ci_low": float(pl), "S_panl_transfer_ci_high": float(ph),
                                   "recovery_ratio": recovery["ratio"], "recovery_ratio_ci_low": recovery["ci_low"],
                                   "recovery_ratio_ci_high": recovery["ci_high"],
                                   "transfer_ratio": transfer_ratio["ratio"], "transfer_ratio_ci_low": transfer_ratio["ci_low"],
                                   "transfer_ratio_ci_high": transfer_ratio["ci_high"],
                                   "denominator_same_sign_rate": recovery["denominator_same_sign_rate"],
                                   "ratio_reportable": recovery["ratio_reportable"] and transfer_ratio["ratio_reportable"],
                                   **_counts(selected)})
    return output


def manipulation_table(root: Path, trials: list[dict[str, Any]], *, smoke: bool) -> list[dict[str, Any]]:
    output = []
    c0_files = {row["case_id"]: root / row["captured_hidden_file"] for row in trials if row["condition"] == "C0"}
    for row in trials:
        hook = row["hook"]
        base = {key: row.get(key) for key in ("case_id", "condition", "alpha", "panl_mediator_layer")}
        output.append({**base, "check": "hook_integrity", "passed": hook["non_target_unchanged"] and all(v == 1 for v in hook["prefill_hits"].values()),
                       "lat_hook_hits": hook["prefill_hits"].get("14", hook["prefill_hits"].get(14, 0)) if row["condition"] in ("C1", "C2") else 0,
                       "panl_patch_hits": 1 if row["condition"] in ("C2", "C3") else 0,
                       "non_target_tokens_unchanged": hook["non_target_unchanged"],
                       "replacement_bitwise_equal": hook["replacement_bitwise_equal"],
                       "source_hidden_bits_sha256": row.get("source_hidden_bits_sha256"),
                       "vector_fingerprint": (row.get("vector") or {}).get("vector_fingerprint")})
        if row["condition"] == "C0":
            output.append({**base, "check": "alpha_zero_parity", "passed": row["parity"]["passed"],
                           "alpha_zero_parity": True, "vector_fingerprint": None})
        if row["condition"] in ("C2", "C3"):
            output.append({**base, "check": "exact_panl_replacement", "passed": bool(row["panl_manipulation"]["replacement_bitwise_equal"]),
                           **row["panl_manipulation"], "source_hidden_bits_sha256": row["source_hidden_bits_sha256"]})
    c1 = [row for row in trials if row["condition"] == "C1"]
    for row in c1:
        clean, _ = load_bf16_npz(c0_files[row["case_id"]], "PANL_L14")
        source, _ = load_bf16_npz(root / row["captured_hidden_file"], "PANL_L14")
        equal = bool(np.array_equal(clean.view(torch.uint16).numpy(), source.view(torch.uint16).numpy()))
        output.append({"case_id": row["case_id"], "condition": "C1", "alpha": row["alpha"], "panl_mediator_layer": 14,
                       "check": "L14_causal_order_negative_control", "passed": equal,
                       "l14_panl_bitwise_equal_clean": equal, "vector_fingerprint": row["vector"]["vector_fingerprint"]})
    return output


CONDITION_LABELS = {
    "C0": "clean LAT + clean PANL",
    "C1": "Steering LAT + PANL",
    "C2": "Steering LAT + clean PANL",
    "C3": "clean LAT + Steering PANL",
}


def _plot(condition_path: Path, destination: Path, endpoint: str, title: str,
          *, attenuation: bool = False) -> None:
    import matplotlib.pyplot as plt
    with condition_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["endpoint"] == endpoint and row["group"] == "answer_equal_macro"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    conditions = ("C0", "C1", "C2", "C3")
    alphas = sorted({float(row["alpha"]) for row in rows})
    layers = sorted({int(float(row["panl_mediator_layer"])) for row in rows if row["panl_mediator_layer"] not in ("", "None")})
    alpha_colors = {-10.0: "#762a83", -2.0: "#2c7fb8", 0.0: "#666666", 2.0: "#d95f02", 10.0: "#b2182b"}
    by_key = {(row["condition"], row["panl_mediator_layer"], float(row["alpha"])): row for row in rows}
    total_by_alpha = {(row["panl_mediator_layer"], float(row["alpha"])): float(row["mean_delta"])
                      for row in rows if row["condition"] == "C1"}

    def point(condition: str, layer: int, alpha: float) -> float:
        layer_key = "" if condition in ("C0", "C1") else str(layer)
        row = by_key.get((condition, layer_key, alpha))
        if row is None:
            return float("nan")
        value = float(row["mean_delta"])
        if not attenuation:
            return value
        if alpha == 0:
            return float("nan")
        total = total_by_alpha.get(("", alpha), float("nan"))
        if not math.isfinite(total) or total == 0:
            return float("nan")
        return 1.0 - value / total

    values = [point(condition, layer, alpha) for condition in conditions for layer in layers for alpha in alphas]
    finite = [abs(value) for value in values if math.isfinite(value)]
    if attenuation:
        lower, upper = -0.05, 1.05
    else:
        limit = max(finite, default=1e-3) * 1.12
        lower, upper = -limit, limit
    for ax, condition in zip(axes.flat, conditions):
        for alpha in alphas:
            y = np.asarray([point(condition, layer, alpha) for layer in layers], dtype=float)
            ax.plot(layers, y, marker="o", lw=1.8, color=alpha_colors.get(alpha, None), label=f"alpha={alpha:g}")
        ax.axhline(0, color="black", lw=.8)
        ax.set_title(CONDITION_LABELS[condition])
        ax.set_ylim(lower, upper)
        ax.set_xticks(layers)
        ax.grid(axis="y", alpha=.2)
    for ax in axes[1]: ax.set_xlabel("PANL layer")
    for ax in axes[:, 0]: ax.set_ylabel("attenuation rate" if attenuation else ("mean delta CLE probe SA" if endpoint == "cle_probe_sa" else "mean delta final soft SA"))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(alphas), frameon=False, bbox_to_anchor=(.5, .98))
    fig.suptitle(title, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, .93)); destination.parent.mkdir(parents=True, exist_ok=True); fig.savefig(destination, dpi=250, bbox_inches="tight"); plt.close(fig)


def analyze(*, output_root: Path = RESULTS_ROOT, smoke: bool = False, resume: bool = False,
            repeats: int | None = None) -> dict[str, Any]:
    root = Path(output_root); trials_path = root / "artifacts" / "trials.jsonl"
    trials = load_jsonl(trials_path); manifest = load_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl")
    expected = len(manifest) * (15 if smoke else 45)
    if len(trials) != expected: raise ValueError(f"Analysis trial grid incomplete: {len(trials)}/{expected}")
    previous_path = root / "progress" / "analysis.json"
    required = [root / "tables" / name for name in ("condition_effects.csv", "mediation_contrasts.csv", "manipulation_checks.csv", "case_level_effects.csv", "README_zh.md")] + [root / "figures" / name for name in ("fig1_final_sa_delta.png", "fig2_cle_probe_sa_delta.png", "fig3_final_sa_attenuation.png")]
    if resume and previous_path.is_file() and all(path.is_file() and path.stat().st_size for path in required):
        previous = json.loads(previous_path.read_text())
        if previous.get("trial_sha256") == sha256_file(trials_path): return {**previous, "resumed_noop": True}
    logical = _logical_rows(trials, smoke=smoke)
    repeats = int(repeats or (SMOKE_BOOTSTRAP_REPEATS if smoke else BOOTSTRAP_REPEATS))
    bootstrap_rows = [{"family_id": row["family_id"], "answer": row["test_answer"]} for row in manifest]
    bootstrap = SharedBootstrap(bootstrap_rows, repeats)
    case_path = root / "tables" / "case_level_effects.csv"; atomic_csv(case_path, logical)
    condition = condition_table(logical, bootstrap); condition_path = root / "tables" / "condition_effects.csv"; atomic_csv(condition_path, condition)
    mediation = mediation_table(logical, bootstrap, smoke=smoke); atomic_csv(root / "tables" / "mediation_contrasts.csv", mediation)
    manipulation = manipulation_table(root, trials, smoke=smoke); atomic_csv(root / "tables" / "manipulation_checks.csv", manipulation)
    audit = json.loads((root / "artifacts" / "diagnostics" / "cle_probe_eligibility_audit.json").read_text())
    readme = f"""# LAT→PANL SA中介实验表格\n\n- `condition_effects.csv`：四种条件相对 clean LAT + clean PANL 的逐 alpha 效应；曲线图横轴为 PANL layer，alpha 用不同曲线表示。final endpoint使用全部case；CLE只使用严格无probe训练重叠case。\n- `mediation_contrasts.csv`：对称总效应、恢复后残余、绝对衰减和PANL移植效应。`S_attenuation`是主要量；ratio仅在分母bootstrap稳定时报告。\n- `case_level_effects.csv`：逐case逻辑单元，含`cle_probe_eligible`与排除原因。\n- `manipulation_checks.csv`：parity、hook、L14负对照和bf16逐bit替换门禁。\n\n条件：clean LAT + clean PANL；Steering LAT + PANL；Steering LAT + clean PANL；clean LAT + Steering PANL。\n\nFigure 2为探索性结果，只使用严格无训练重叠子集（正式n={audit['eligible_count']}）；brown仅1例且没有high-image样本。Figure 3 是相对 Steering LAT + PANL 的逐 alpha 衰减率（alpha=0 无完全 steering 分母，留空）。这里的恢复比例不是严格自然间接效应。\n"""
    (root / "tables").mkdir(parents=True, exist_ok=True); (root / "tables" / "README_zh.md").write_text(readme, encoding="utf-8")
    _plot(condition_path, root / "figures" / "fig1_final_sa_delta.png", "final_soft_sa", "Final soft SA by PANL layer and LAT dose")
    _plot(condition_path, root / "figures" / "fig2_cle_probe_sa_delta.png", "cle_probe_sa", f"Exploratory CLE probe SA by PANL layer and LAT dose (strict no-overlap n={audit['eligible_count']})")
    _plot(condition_path, root / "figures" / "fig3_final_sa_attenuation.png", "final_soft_sa", "Final soft SA attenuation relative to complete steering", attenuation=True)
    result = {"status": "complete", "smoke": smoke, "physical_trial_count": len(trials), "logical_case_rows": len(logical),
              "bootstrap_repeats": repeats, "bootstrap_seed": SEED, "bootstrap_design_hash": bootstrap.design_hash,
              "tables": 5, "figures": 3, "cle_figure_status": "exploratory", "trial_sha256": sha256_file(trials_path),
              "resumed_noop": False}
    atomic_json(root / "progress" / "analysis.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bootstrap", type=int); args = parser.parse_args(argv)
    print(json.dumps(analyze(output_root=args.output_root, smoke=args.smoke, resume=args.resume, repeats=args.bootstrap), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
