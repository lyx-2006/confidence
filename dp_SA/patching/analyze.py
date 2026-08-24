from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import BOOTSTRAP_REPEATS, DEFAULT_CAPTURE_DIR, DEFAULT_EVAL_CASES, DEFAULT_LAYERS, DEFAULT_POSITIONS, SEED
from .io import atomic_csv, atomic_json, load_jsonl_strict, stable_seed
from .metrics import bh_fdr, bootstrap_ratio, oriented, sign_flip_p


PRIMARY = ("fixed_clean_class_margin", "oriented_soft", "oriented_hard")


def _values(record: dict[str, Any], endpoint: str, group: str) -> tuple[float, float, float]:
    source = "fixed_clean_class_margin" if endpoint == "fixed_clean_class_margin" else "soft_sa" if endpoint in {"soft_sa", "oriented_soft"} else "hard_midpoint" if endpoint in {"hard_midpoint", "oriented_hard"} else endpoint
    values = tuple(float(record[state][source]) for state in ("clean", "corrupt", "patched"))
    if endpoint.startswith("oriented_"):
        return tuple(oriented(value, record["test_side"]) for value in values)  # type: ignore[return-value]
    return values  # type: ignore[return-value]


def _flatten_stats(base: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    row = dict(base)
    for name in ("clean", "corrupt", "patched", "disruption", "patch_gain", "recovery"):
        for field, value in stats[name].items():
            row[f"{name}_{field}"] = value
    for name in ("sample_count", "item_count", "undefined_item_recovery_count"):
        row[name] = stats[name]
    return row


def _rate_summary(rows: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> dict[str, Any]:
    corrupt_changed = [bool(row["first_token"]["corrupt_changed"]) for row in rows]
    patched_changed = [bool(row["first_token"]["patched_changed_from_clean"]) for row in rows]
    recovered = [bool(row["first_token"]["clean_class_recovered"]) for row in rows]
    affected = [index for index, value in enumerate(corrupt_changed) if value]
    values = {
        "corrupt_first_token_change_rate": float(np.mean(corrupt_changed)),
        "patched_vs_clean_first_token_change_rate": float(np.mean(patched_changed)),
        "hard_clean_class_recovered_rate": float(np.mean(recovered)),
        "conditional_recovery_rate": (float(np.mean([recovered[index] for index in affected])) if affected else None),
        "conditional_count": len(affected),
    }
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in (
        "corrupt_first_token_change_rate", "patched_vs_clean_first_token_change_rate",
        "hard_clean_class_recovered_rate", "conditional_recovery_rate")}
    for _ in range(repeats):
        indices = rng.integers(0, len(rows), len(rows))
        cc = np.asarray(corrupt_changed, dtype=np.float64)[indices]
        pc = np.asarray(patched_changed, dtype=np.float64)[indices]
        rc = np.asarray(recovered, dtype=np.float64)[indices]
        samples["corrupt_first_token_change_rate"].append(float(cc.mean()))
        samples["patched_vs_clean_first_token_change_rate"].append(float(pc.mean()))
        samples["hard_clean_class_recovered_rate"].append(float(rc.mean()))
        if bool(cc.any()):
            samples["conditional_recovery_rate"].append(float(rc[cc.astype(bool)].mean()))
    for name, sample in samples.items():
        if sample:
            values[f"{name}_ci_low"] = float(np.percentile(sample, 2.5))
            values[f"{name}_ci_high"] = float(np.percentile(sample, 97.5))
            values[f"{name}_valid_bootstrap_repeats"] = len(sample)
        else:
            values[f"{name}_ci_low"] = None
            values[f"{name}_ci_high"] = None
            values[f"{name}_valid_bootstrap_repeats"] = 0
    return values


def _contrast_bootstrap(panl: Sequence[dict[str, Any]], control: Sequence[dict[str, Any]], endpoint: str,
                        group: str, *, repeats: int, seed: int) -> dict[str, Any]:
    a = {row["case_id"]: row for row in panl}
    b = {row["case_id"]: row for row in control}
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("PANL/control contrast has no paired cases")
    clean = np.asarray([_values(a[key], endpoint, group)[0] for key in keys])
    corrupt = np.asarray([_values(a[key], endpoint, group)[1] for key in keys])
    pa = np.asarray([_values(a[key], endpoint, group)[2] for key in keys])
    pb = np.asarray([_values(b[key], endpoint, group)[2] for key in keys])
    disruption = clean - corrupt
    gain_a, gain_b = pa - corrupt, pb - corrupt
    denom = float(disruption.mean())
    observed_recovery = (float(gain_a.mean() / denom) if abs(denom) > 1e-8 else None)
    observed_control = (float(gain_b.mean() / denom) if abs(denom) > 1e-8 else None)
    observed_difference = (None if observed_recovery is None else observed_recovery - observed_control)
    rng = np.random.default_rng(seed)
    gain_values, recovery_values = [], []
    for _ in range(repeats):
        indices = rng.integers(0, len(keys), len(keys))
        gain_values.append(float((gain_a[indices] - gain_b[indices]).mean()))
        d = float(disruption[indices].mean())
        if abs(d) > 1e-8:
            recovery_values.append(float((gain_a[indices].mean() - gain_b[indices].mean()) / d))
    def summary(values: Sequence[float], observed: float | None):
        array = np.asarray(values, dtype=np.float64)
        return {"value": observed, "sem": float(array.std(ddof=1)) if len(array) > 1 else None,
                "ci_low": float(np.percentile(array, 2.5)) if len(array) else None,
                "ci_high": float(np.percentile(array, 97.5)) if len(array) else None,
                "valid_bootstrap_repeats": len(array)}
    return {
        "item_count": len(keys), "panl_patch_gain": float(gain_a.mean()),
        "control_patch_gain": float(gain_b.mean()),
        "patch_gain_difference": summary(gain_values, float((gain_a - gain_b).mean())),
        "panl_recovery": observed_recovery, "control_recovery": observed_control,
        "recovery_difference": summary(recovery_values, observed_difference),
    }


def analyze(output_dir: Path, *, repeats: int, seed: int, final: bool = True) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    config = json.loads((output_dir / "run_config.json").read_text())
    results = load_jsonl_strict(output_dir / "results.jsonl")
    baselines = load_jsonl_strict(output_dir / "baselines.jsonl")
    expected = int(config["expected_patch_cells"])
    if len(results) != expected or len({row["cell_key"] for row in results}) != expected:
        raise RuntimeError(f"Incomplete/duplicate patch grid: {len(results)}/{expected}")
    if len(baselines) != int(config["expected_baselines"]):
        raise RuntimeError("Incomplete baseline grid")
    if any(row.get("status") != "completed" for row in [*baselines, *results]):
        raise RuntimeError("Failed records prevent final analysis")
    bootstrap_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in results:
        grouped[(str(record["position"]), int(record["layer"]))].append(record)
    endpoints_by_group = {
        "all": ("fixed_clean_class_margin", "oriented_soft", "oriented_hard", "soft_sa", "hard_midpoint", "entropy"),
        "image_side": ("fixed_clean_class_margin", "soft_sa", "hard_midpoint", "entropy"),
        "text_side": ("fixed_clean_class_margin", "soft_sa", "hard_midpoint", "entropy"),
    }
    for (position, layer), cell_rows in sorted(grouped.items()):
        for group, endpoints in endpoints_by_group.items():
            subset = cell_rows if group == "all" else [row for row in cell_rows if row["test_side"] == group]
            for endpoint in endpoints:
                vectors = [_values(row, endpoint, group) for row in subset]
                clean, corrupt, patched = map(list, zip(*vectors))
                cell_seed = stable_seed(seed, "bootstrap", group, endpoint, position, str(layer))
                stats, samples = bootstrap_ratio(clean, corrupt, patched, repeats=repeats, seed=cell_seed)
                base = {"record_type": "cell", "position": position, "layer": layer,
                        "group": group, "endpoint": endpoint,
                        **_rate_summary(subset, repeats=repeats,
                                       seed=stable_seed(seed, "rates", group, position, str(layer)))}
                metric_rows.append(_flatten_stats(base, stats))
                for sample in samples:
                    bootstrap_rows.append({**base, **sample})
    for endpoint in PRIMARY:
        family = [row for row in metric_rows if row["group"] == "all" and row["endpoint"] == endpoint]
        p_values = []
        for row in family:
            source_rows = grouped[(row["position"], int(row["layer"]))]
            gains = []
            for record in source_rows:
                _clean, corrupt, patched = _values(record, endpoint, "all")
                gains.append(patched - corrupt)
            p = sign_flip_p(gains, repeats=repeats,
                            seed=stable_seed(seed, "signflip", endpoint, row["position"], str(row["layer"])))
            row["p_raw"] = p
            row["fdr_family"] = f"{endpoint}:2_positions_x_5_layers"
            p_values.append(p)
        for row, q in zip(family, bh_fdr(p_values)):
            row["q_bh"] = q
    contrasts: list[dict[str, Any]] = []
    for layer in sorted({int(row["layer"]) for row in results}):
        panl = grouped.get(("P1_PANL", layer), [])
        control = grouped.get(("P1_PANL_PLUS_1", layer), [])
        if not panl or not control:
            continue
        for endpoint in PRIMARY:
            value = _contrast_bootstrap(panl, control, endpoint, "all", repeats=repeats,
                                        seed=stable_seed(seed, "contrast", endpoint, str(layer)))
            contrasts.append({"layer": layer, "endpoint": endpoint, **value})
    lookup = {(row["position"], int(row["layer"]), row["group"], row["endpoint"]): row for row in metric_rows}
    contrast_lookup = {(int(row["layer"]), row["endpoint"]): row for row in contrasts}
    claims: list[dict[str, Any]] = []
    for layer in sorted({int(row["layer"]) for row in results}):
        endpoint_checks = {}
        for endpoint in ("fixed_clean_class_margin", "oriented_soft"):
            row = lookup[("P1_PANL", layer, "all", endpoint)]
            contrast = contrast_lookup[(layer, endpoint)]
            endpoint_checks[endpoint] = {
                "corruption_damage_ci_above_zero": row["disruption_ci_low"] is not None and row["disruption_ci_low"] > 0,
                "panl_recovery_ci_above_zero": row["recovery_ci_low"] is not None and row["recovery_ci_low"] > 0,
                "panl_patch_gain_ci_above_zero": row["patch_gain_ci_low"] is not None and row["patch_gain_ci_low"] > 0,
                "panl_better_than_control_ci_above_zero": contrast["recovery_difference"]["ci_low"] is not None and contrast["recovery_difference"]["ci_low"] > 0,
            }
        hard = lookup[("P1_PANL", layer, "all", "oriented_hard")]
        hard_contrast = contrast_lookup[(layer, "oriented_hard")]
        hard_noncontradictory = (
            float(hard["patch_gain_value"]) >= 0 and
            (hard["recovery_value"] is None or float(hard["recovery_value"]) >= 0) and
            float(hard_contrast["patch_gain_difference"]["value"]) >= 0
        )
        pass_gate = all(all(checks.values()) for checks in endpoint_checks.values()) and hard_noncontradictory
        claims.append({"layer": layer, "continuous_endpoint_checks": endpoint_checks,
                       "hard_midpoint_noncontradictory": hard_noncontradictory,
                       "functional_information_claim_supported": pass_gate})
    metric_rows.extend({"record_type": "position_contrast", **row} for row in contrasts)
    atomic_csv(output_dir / "metrics.csv", metric_rows)
    atomic_csv(output_dir / "bootstrap.csv", bootstrap_rows)
    summary = {
        "status": "complete" if final else "provisional", "run_fingerprint": config["fingerprint"],
        "baseline_count": len(baselines), "patch_cell_count": len(results),
        "bootstrap_repeats": repeats, "sampling_unit": "item_id",
        "primary_fdr_families": {endpoint: 10 for endpoint in PRIMARY},
        "metrics": metric_rows, "position_contrasts": contrasts,
        "claim_gate": "margin_and_oriented_soft_CI_with_noncontradictory_hard",
        "layer_claims": claims,
        "any_layer_supports_functional_information_claim": any(row["functional_information_claim_supported"] for row in claims),
    }
    atomic_json(output_dir / "summary.json", summary)
    _plots(output_dir, metric_rows)
    _write_summary(output_dir, summary)
    if final:
        required_plots = {
            "delayed_patching_logit_recovery.png", "delayed_patching_soft_recovery.png",
            "delayed_patching_hard_midpoint_recovery.png", "delayed_patching_clean_corrupt_patched.png",
            "delayed_patching_change_rate.png", "delayed_patching_side_comparison.png",
        }
        missing = sorted(name for name in required_plots if not (output_dir / "plots" / name).is_file())
        if missing:
            raise RuntimeError(f"Missing required plots: {missing}")
        atomic_json(output_dir / "completion.json", {
            "status": "complete", "run_fingerprint": config["fingerprint"],
            "baseline_count": len(baselines), "patch_cell_count": len(results),
            "artifacts_complete": True,
        })
    return summary


def _cell_rows(rows: Sequence[dict[str, Any]], endpoint: str, group: str = "all"):
    return [row for row in rows if row.get("record_type") == "cell" and row.get("endpoint") == endpoint and row.get("group") == group]


def _plots(output: Path, rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    styles = {"P1_PANL": ("#E07B39", "o"), "P1_PANL_PLUS_1": ("#4C78A8", "s")}
    for endpoint, filename, title in (
        ("fixed_clean_class_margin", "delayed_patching_logit_recovery.png", "Fixed-clean-class logit-margin recovery"),
        ("oriented_soft", "delayed_patching_soft_recovery.png", "Oriented soft-SA recovery"),
        ("oriented_hard", "delayed_patching_hard_midpoint_recovery.png", "Oriented hard-midpoint recovery"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        data = _cell_rows(rows, endpoint)
        for position, (color, marker) in styles.items():
            part = sorted([row for row in data if row["position"] == position], key=lambda row: int(row["layer"]))
            x = [int(row["layer"]) for row in part]
            y = [row["recovery_value"] for row in part]
            low = [value - row["recovery_ci_low"] for value, row in zip(y, part)]
            high = [row["recovery_ci_high"] - value for value, row in zip(y, part)]
            ax.errorbar(x, y, yerr=[low, high], label=position, color=color, marker=marker, capsize=3)
        ax.axhline(0, color="black", lw=.8); ax.axhline(1, color="black", lw=.8, ls="--")
        ax.set_xticks(sorted({int(row["layer"]) for row in data})); ax.set_xlabel("Zero-based decoder layer")
        ax.set_ylabel("Ratio-of-means recovery"); ax.set_title(f"{title} (n=50; 25+25)"); ax.legend(); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(plot_dir / filename, dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, endpoint, label in zip(axes, ("oriented_soft", "fixed_clean_class_margin"), ("Oriented soft SA", "Logit margin")):
        data = _cell_rows(rows, endpoint)
        for position, (color, marker) in styles.items():
            part = sorted([row for row in data if row["position"] == position], key=lambda row: int(row["layer"]))
            x = [int(row["layer"]) for row in part]
            for state, ls in (("clean", ":"), ("corrupt", "--"), ("patched", "-")):
                y = [row[f"{state}_value"] for row in part]
                low = [value - row[f"{state}_ci_low"] for value, row in zip(y, part)]
                high = [row[f"{state}_ci_high"] - value for value, row in zip(y, part)]
                ax.errorbar(x, y, yerr=[low, high], color=color,
                            marker=marker if state == "patched" else None,
                            ls=ls, capsize=2, label=f"{position} {state}")
        ax.set_title(label); ax.set_xlabel("Layer"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Mean"); axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Clean / corrupt / patched means (n=50; 25+25)"); fig.tight_layout()
    fig.savefig(plot_dir / "delayed_patching_clean_corrupt_patched.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); data = _cell_rows(rows, "fixed_clean_class_margin")
    for position, (color, marker) in styles.items():
        part = sorted([row for row in data if row["position"] == position], key=lambda row: int(row["layer"]))
        y = [row["patched_vs_clean_first_token_change_rate"] for row in part]
        low = [value - row["patched_vs_clean_first_token_change_rate_ci_low"] for value, row in zip(y, part)]
        high = [row["patched_vs_clean_first_token_change_rate_ci_high"] - value for value, row in zip(y, part)]
        ax.errorbar([row["layer"] for row in part], y, yerr=[low, high],
                    color=color, marker=marker, capsize=3, label=f"{position}: patched vs clean")
    if data:
        ax.axhline(data[0]["corrupt_first_token_change_rate"], color="black", ls="--", label="corrupt vs clean")
        ax.axhspan(data[0]["corrupt_first_token_change_rate_ci_low"],
                   data[0]["corrupt_first_token_change_rate_ci_high"], color="black", alpha=.1)
    ax.set_title("First-token change rate (n=50; 25+25)"); ax.set_xlabel("Layer"); ax.set_ylabel("Rate"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(plot_dir / "delayed_patching_change_rate.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for group, color in (("image_side", "#D95F02"), ("text_side", "#1B9E77")):
        for position, (_base, marker) in styles.items():
            part = sorted([row for row in _cell_rows(rows, "soft_sa", group) if row["position"] == position], key=lambda row: int(row["layer"]))
            y = [row["recovery_value"] for row in part]
            low = [value - row["recovery_ci_low"] for value, row in zip(y, part)]
            high = [row["recovery_ci_high"] - value for value, row in zip(y, part)]
            ax.errorbar([row["layer"] for row in part], y, yerr=[low, high],
                        color=color, marker=marker, capsize=2,
                        ls="-" if position == "P1_PANL" else "--", label=f"{group} {position}")
    ax.axhline(0, color="black", lw=.8); ax.axhline(1, color="black", lw=.8, ls=":")
    ax.set_title("Side-specific soft-SA recovery (n=25 each)"); ax.set_xlabel("Layer"); ax.set_ylabel("Recovery"); ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(plot_dir / "delayed_patching_side_comparison.png", dpi=180); plt.close(fig)


def _write_summary(output: Path, summary: dict[str, Any]) -> None:
    rows = summary["metrics"]
    lines = ["# Delayed-SA activation patching", "", f"Status: {summary['status']}",
             f"Baselines: {summary['baseline_count']}; patch cells: {summary['patch_cell_count']}.", "",
             "## Primary recovery (all; ratio of means)", ""]
    for endpoint in PRIMARY:
        lines.append(f"### {endpoint}")
        lines.append("")
        for row in sorted(_cell_rows(rows, endpoint), key=lambda value: (value["position"], int(value["layer"]))):
            lines.append(f"- {row['position']} L{row['layer']}: R={row['recovery_value']:.6g}, "
                         f"95% CI [{row['recovery_ci_low']:.6g}, {row['recovery_ci_high']:.6g}], "
                         f"p={row.get('p_raw', float('nan')):.6g}, q={row.get('q_bh', float('nan')):.6g}, n={row['item_count']}")
        lines.append("")
    lines.extend(["## Functional-information claim gate", ""])
    for row in summary["layer_claims"]:
        lines.append(f"- L{row['layer']}: {'SUPPORTED' if row['functional_information_claim_supported'] else 'not established'}; "
                     f"hard midpoint non-contradictory={row['hard_midpoint_noncontradictory']}.")
    lines.extend(["", "Scientific null effects are valid outcomes and do not indicate execution failure.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--corruption", choices=("all", "answer_only"), default="all")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--positions", nargs="+", default=list(DEFAULT_POSITIONS))
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--eval-cases", type=int, default=DEFAULT_EVAL_CASES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyze(args.output_dir, repeats=(100 if args.smoke else args.bootstrap), seed=args.seed, final=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
