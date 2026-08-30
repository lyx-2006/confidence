"""Raw final-SA reanalysis for the LAT difficulty swap experiment.

This module deliberately reports only ``delta_sa = swap_sa - clean_sa``.
It does not orient, truncate, take absolute values, or pool across layers.
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import JOINED_PATH, RESULTS_ROOT, SWAP_LAYERS
from .io_utils import atomic_csv, atomic_json, atomic_text, load_jsonl


REANALYSIS_ROOT = RESULTS_ROOT / "reanalysis"
DIRECTIONS = ("E→H", "H→E")
ORIGINS = ("follow_text", "follow_image")
ARMS = ("A", "B")
ARM_LABELS = {"A": "image difficulty", "B": "text difficulty"}


def classify_answer_origin(row: dict[str, Any]) -> str:
    text = bool(row.get("answer_matches_text"))
    image = bool(row.get("answer_matches_image"))
    if text and not image:
        return "follow_text"
    if image and not text:
        return "follow_image"
    if text and image:
        return "both_match"
    return "neither_match"


def explicit_direction(recipient_level: str, donor_level: str) -> str:
    levels = (str(recipient_level), str(donor_level))
    if levels == ("hard", "easy"):
        return "E→H"
    if levels == ("easy", "hard"):
        return "H→E"
    raise ValueError(f"Invalid recipient/donor difficulty levels: {levels}")


def canonicalize_raw_delta(
    swap_rows: Sequence[dict[str, Any]], joined_rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    joined: dict[str, dict[str, Any]] = {}
    for row in joined_rows:
        if row.get("status") != "completed":
            continue
        case_id = str(row["case_id"])
        if case_id in joined:
            raise AssertionError(f"Duplicate joined case_id: {case_id}")
        joined[case_id] = row

    canonical: list[dict[str, Any]] = []
    cross_count = 0
    for source in swap_rows:
        if source.get("status") != "completed" or source.get("swap_kind") != "cross":
            continue
        cross_count += 1
        recipient_id = str(source["recipient_case_id"])
        if recipient_id not in joined:
            raise AssertionError(f"Missing recipient in joined records: {recipient_id}")
        recipient = joined[recipient_id]
        arm = str(source["arm"])
        if arm not in ARMS:
            raise AssertionError(f"Unexpected arm: {arm}")
        direction = explicit_direction(source["recipient_level"], source["donor_level"])
        delta = float(source["swap_score"]["soft_sa"]) - float(source["clean_score"]["soft_sa"])
        stored = float(source["delta_sa"])
        if not math.isclose(delta, stored, rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(f"delta_sa is not swap-clean for {source['trial_key']}")
        canonical.append(
            {
                "item_id": str(source["item_id"]),
                "pair_id": str(source["pair_id"]),
                "arm": arm,
                "arm_label": ARM_LABELS[arm],
                "direction": direction,
                "layer": int(source["layer"]),
                "answer_origin": classify_answer_origin(recipient),
                "recipient_case_id": recipient_id,
                "donor_case_id": str(source["donor_case_id"]),
                "clean_sa": float(source["clean_score"]["soft_sa"]),
                "swap_sa": float(source["swap_score"]["soft_sa"]),
                "delta_sa": stored,
            }
        )

    if len(canonical) != cross_count:
        raise AssertionError("Cross-row canonicalization lost rows")
    observed_layers = sorted({row["layer"] for row in canonical})
    if observed_layers != list(SWAP_LAYERS):
        raise AssertionError(f"Unexpected layer grid: {observed_layers}")
    audit = {
        "definition": "delta_sa = swap_sa - recipient_clean_sa",
        "orientation_or_absolute_value_applied": False,
        "cross_trial_count": len(canonical),
        "pair_count": len({row["pair_id"] for row in canonical}),
        "item_count": len({row["item_id"] for row in canonical}),
        "self_rows_excluded": sum(row.get("swap_kind") == "self" for row in swap_rows),
        "answer_origin_pair_counts": {
            arm: {
                origin: len({row["pair_id"] for row in canonical if row["arm"] == arm and row["answer_origin"] == origin})
                for origin in ("follow_text", "follow_image", "both_match", "neither_match")
            }
            for arm in ARMS
        },
    }
    return canonical, audit


def summarize_groups(rows: Iterable[dict[str, Any]], group_fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: tuple(str(part) for part in value)):
        group = groups[key]
        values = np.asarray([float(row["delta_sa"]) for row in group], dtype=float)
        sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        record = dict(zip(group_fields, key, strict=True))
        record.update(
            {
                "mean_delta_sa": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "sem": sem,
                "ci95_low": float(values.mean() - 1.96 * sem),
                "ci95_high": float(values.mean() + 1.96 * sem),
                "trial_count": len(group),
                "pair_count": len({row["pair_id"] for row in group}),
                "item_count": len({row["item_id"] for row in group}),
            }
        )
        output.append(record)
    return output


def _lookup(rows: Sequence[dict[str, Any]], **filters: Any) -> dict[str, Any]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in filters.items())]
    if len(matches) != 1:
        raise AssertionError(f"Expected one row for {filters}, got {len(matches)}")
    return matches[0]


def make_wide(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[tuple(row[key] for key in keys)][int(row["layer"])] = row
    output: list[dict[str, Any]] = []
    for group_key in sorted(groups, key=lambda value: tuple(str(part) for part in value)):
        by_layer = groups[group_key]
        if sorted(by_layer) != list(SWAP_LAYERS):
            raise AssertionError(f"Incomplete layer grid for {group_key}: {sorted(by_layer)}")
        record = dict(zip(keys, group_key, strict=True))
        record["pair_count"] = by_layer[SWAP_LAYERS[0]]["pair_count"]
        for layer in SWAP_LAYERS:
            record[f"L{layer}_mean_delta_sa"] = by_layer[layer]["mean_delta_sa"]
            record[f"L{layer}_ci95_low"] = by_layer[layer]["ci95_low"]
            record[f"L{layer}_ci95_high"] = by_layer[layer]["ci95_high"]
        output.append(record)
    return output


def _line(ax: Any, layers: Sequence[int], means: Sequence[float], lows: Sequence[float], highs: Sequence[float], label: str, **style: Any) -> None:
    # Keep confidence intervals in the CSV tables; line charts show means only.
    del lows, highs
    ax.plot(layers, means, linewidth=1.8, markersize=5, label=label, **style)


def plot_old_lines(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    colors = {"A": "#1976d2", "B": "#d97706"}
    markers = {"E→H": "o", "H→E": "s"}
    for arm in ARMS:
        for direction in DIRECTIONS:
            selected = [_lookup(rows, arm=arm, direction=direction, layer=layer) for layer in SWAP_LAYERS]
            _line(ax, SWAP_LAYERS, [r["mean_delta_sa"] for r in selected], [r["ci95_low"] for r in selected],
                  [r["ci95_high"] for r in selected], f"{ARM_LABELS[arm]} {direction}",
                  color=colors[arm], marker=markers[direction], linestyle="-" if direction == "E→H" else "--")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Old grouping: raw final soft-SA delta by LAT swap layer",
           xlabel="LAT swap layer", ylabel="Mean raw ΔSA (swap − recipient clean)")
    ax.set_xticks(SWAP_LAYERS); ax.grid(alpha=.25); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=320); plt.close(fig)


def plot_new_arm_lines(rows: Sequence[dict[str, Any]], arm: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    colors = {"follow_text": "#009e73", "follow_image": "#cc79a7"}
    markers = {"E→H": "o", "H→E": "s"}
    modality = "image" if arm == "A" else "text"
    for origin in ORIGINS:
        for direction in DIRECTIONS:
            selected = [_lookup(rows, arm=arm, answer_origin=origin, direction=direction, layer=layer) for layer in SWAP_LAYERS]
            _line(ax, SWAP_LAYERS, [r["mean_delta_sa"] for r in selected], [r["ci95_low"] for r in selected],
                  [r["ci95_high"] for r in selected], f"{origin} + {modality} {direction}",
                  color=colors[origin], marker=markers[direction], linestyle="-" if direction == "E→H" else "--")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title=f"New grouping — Arm {arm} ({ARM_LABELS[arm]}): raw final soft-SA delta",
           xlabel="LAT swap layer", ylabel="Mean raw ΔSA (swap − recipient clean)")
    ax.set_xticks(SWAP_LAYERS); ax.grid(alpha=.25); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=320); plt.close(fig)


def plot_forest(rows: Sequence[dict[str, Any]], path: Path, *, new: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 8.0), sharex=True, sharey=True)
    origins = ORIGINS if new else (None,)
    colors = {None: "#1976d2", "follow_text": "#009e73", "follow_image": "#cc79a7"}
    offsets = [-.22, .22] if not new else [-.30, -.10, .10, .30]
    for ax, arm in zip(axes, ARMS, strict=True):
        index = 0
        modality = "image" if arm == "A" else "text"
        for origin in origins:
            for direction in DIRECTIONS:
                selected = [
                    _lookup(rows, arm=arm, direction=direction, layer=layer, **({"answer_origin": origin} if new else {}))
                    for layer in SWAP_LAYERS
                ]
                means = np.asarray([r["mean_delta_sa"] for r in selected])
                lows = np.asarray([r["ci95_low"] for r in selected]); highs = np.asarray([r["ci95_high"] for r in selected])
                label = f"{origin} + {modality} {direction}" if new else f"{modality} {direction}"
                ax.errorbar(means, np.asarray(SWAP_LAYERS) + offsets[index], xerr=np.vstack((means-lows, highs-means)),
                            fmt="o" if direction == "E→H" else "s", color=colors[origin], capsize=3, label=label)
                index += 1
        ax.axvline(0, color="black", linewidth=1); ax.grid(alpha=.25); ax.set_title(f"Arm {arm}: {ARM_LABELS[arm]}")
        ax.set_yticks(SWAP_LAYERS)
        ax.set_xlabel("Mean raw ΔSA (swap − recipient clean)"); ax.legend(fontsize=8)
    axes[0].set_ylabel("LAT swap layer")
    fig.suptitle(("New grouping" if new else "Old grouping") + ": raw final soft-SA delta by arm")
    fig.tight_layout(); fig.savefig(path, dpi=320); plt.close(fig)


def _summary_text(audit: dict[str, Any]) -> str:
    counts = audit["answer_origin_pair_counts"]
    return f"""# Raw ΔSA reanalysis

本目录只报告最终 soft-SA 的原始差值：`delta_sa = swap_sa - recipient clean_sa`。
没有使用 oriented、绝对值、toward/wrong，也没有跨 layer pooling。

- Cross trials: {audit['cross_trial_count']}
- Pairs: {audit['pair_count']}
- Items: {audit['item_count']}
- 旧版：每个 layer 分别统计 A/image difficulty 与 B/text difficulty 的 E→H、H→E。
- 新版：每个 layer 分别统计 follow_text/follow_image × E→H/H→E；both/neither 排除。
- A 新版 pair 数：follow_text={counts['A']['follow_text']}，follow_image={counts['A']['follow_image']}；排除 both={counts['A']['both_match']}，neither={counts['A']['neither_match']}。
- B 新版 pair 数：follow_text={counts['B']['follow_text']}，follow_image={counts['B']['follow_image']}；排除 both={counts['B']['both_match']}，neither={counts['B']['neither_match']}。
- 三张折线图只画逐层均值，不画 CI；CI 保留在表格中。
- 两张 A/B 分面图的误差棒是 `mean ± 1.96×SEM`。

具体数值以 `tables/old_delta_sa_by_layer_long.csv` 和 `tables/new_delta_sa_by_layer_long.csv` 为准。
"""


def run_reanalysis(root: Path = REANALYSIS_ROOT) -> dict[str, Any]:
    resolved = root.resolve()
    expected = REANALYSIS_ROOT.resolve()
    if resolved != expected:
        raise ValueError(f"Refusing to replace unexpected directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    figures = resolved / "figures"; tables = resolved / "tables"; artifacts = resolved / "artifacts"; progress = resolved / "progress"
    for directory in (figures, tables, artifacts, progress):
        directory.mkdir(parents=True, exist_ok=True)

    swap_rows = load_jsonl(RESULTS_ROOT / "artifacts" / "swap_results.jsonl")
    joined_rows = load_jsonl(JOINED_PATH)
    canonical, audit = canonicalize_raw_delta(swap_rows, joined_rows)
    old_rows = summarize_groups(canonical, ("arm", "arm_label", "direction", "layer"))
    eligible = [row for row in canonical if row["answer_origin"] in ORIGINS]
    new_rows = summarize_groups(eligible, ("arm", "arm_label", "answer_origin", "direction", "layer"))

    if len(old_rows) != 4 * len(SWAP_LAYERS):
        raise AssertionError(f"Old table does not have 4×{len(SWAP_LAYERS)} rows")
    if len(new_rows) != 8 * len(SWAP_LAYERS):
        raise AssertionError(f"New table does not have 8×{len(SWAP_LAYERS)} rows")
    if any("oriented" in key or "absolute" in key for row in old_rows + new_rows for key in row):
        raise AssertionError("Forbidden transformed metric in output")

    canonical_fields = ("item_id", "pair_id", "arm", "arm_label", "answer_origin", "direction", "layer",
                        "recipient_case_id", "donor_case_id", "clean_sa", "swap_sa", "delta_sa")
    stat_fields = ("arm", "arm_label", "answer_origin", "direction", "layer", "mean_delta_sa", "sample_sd", "sem",
                   "ci95_low", "ci95_high", "trial_count", "pair_count", "item_count")
    atomic_csv(artifacts / "canonical_raw_delta_sa.csv", canonical, canonical_fields)
    atomic_json(artifacts / "audit.json", audit)
    atomic_csv(tables / "old_delta_sa_by_layer_long.csv", old_rows, [field for field in stat_fields if field != "answer_origin"])
    atomic_csv(tables / "new_delta_sa_by_layer_long.csv", new_rows, stat_fields)
    layer_fields = [field for layer in SWAP_LAYERS for field in
                    (f"L{layer}_mean_delta_sa", f"L{layer}_ci95_low", f"L{layer}_ci95_high")]
    atomic_csv(tables / "old_delta_sa_by_layer_wide.csv", make_wide(old_rows, ("arm", "arm_label", "direction")),
               ("arm", "arm_label", "direction", "pair_count", *layer_fields))
    atomic_csv(tables / "new_delta_sa_by_layer_wide.csv", make_wide(new_rows, ("arm", "arm_label", "answer_origin", "direction")),
               ("arm", "arm_label", "answer_origin", "direction", "pair_count", *layer_fields))
    exclusions = [
        {"arm": arm, "answer_origin": origin, "pair_count": audit["answer_origin_pair_counts"][arm][origin],
         "included_in_new_version": origin in ORIGINS}
        for arm in ARMS for origin in ("follow_text", "follow_image", "both_match", "neither_match")
    ]
    atomic_csv(tables / "answer_origin_counts.csv", exclusions)

    plot_old_lines(old_rows, figures / "old_delta_sa_by_layer.png")
    plot_new_arm_lines(new_rows, "A", figures / "new_delta_sa_by_layer_A.png")
    plot_new_arm_lines(new_rows, "B", figures / "new_delta_sa_by_layer_B.png")
    plot_forest(old_rows, figures / "old_final_raw_delta_by_arm.png", new=False)
    plot_forest(new_rows, figures / "new_final_raw_delta_by_answer_origin.png", new=True)
    atomic_text(resolved / "summary.md", _summary_text(audit))

    completion = {
        "status": "complete", "metric": "raw final soft-SA delta = swap - recipient clean",
        "orientation_applied": False, "absolute_value_applied": False, "layer_pooling_applied": False,
        "cross_trial_count": len(canonical), "old_table_rows": len(old_rows), "new_table_rows": len(new_rows),
        "figure_count": len(list(figures.glob("*.png"))), "gpu_forward_count": 0,
    }
    atomic_json(progress / "completion.json", completion)
    return completion


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(run_reanalysis())


if __name__ == "__main__":
    main()
