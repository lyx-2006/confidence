#!/usr/bin/env python3
"""Build Phase-1 arbitration trajectories from existing OOF/readout artifacts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from layer_metacognition.hidden_state_store import atomic_write_json, atomic_write_text

from . import DECISION_SIDE_LOCATIONS
from .common import iter_jsonl


SUBGROUPS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "pooled": lambda _row: True,
    "conflict_easy": lambda row: row["condition"] == "conflict_easy",
    "conflict_hard": lambda row: row["condition"] == "conflict_hard",
    "follows_text": lambda row: row["decision_side"] == "follows_text",
    "follows_image": lambda row: row["decision_side"] == "follows_image",
}
TRAJECTORY_FIELDS = ("R_T", "R_I_preliminary", "C", "K", "SA_sac_semantic")


def _require_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage-1 input does not exist: {path}")


def _load_oof(
    path: Path,
    *,
    case_ids: set[str],
    layers: set[int],
    positions: set[str],
    task_suffixes: set[str],
    version_setting: str,
) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    output: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for record in iter_jsonl(path):
        if (
            record.get("model_type") != "hidden_state_probe"
            or record.get("version_setting") != version_setting
            or str(record.get("case_id")) not in case_ids
            or record.get("layer") is None
        ):
            continue
        layer = int(record["layer"])
        position = str(record.get("position"))
        task = str(record.get("task"))
        suffix = next(
            (
                candidate
                for candidate in task_suffixes
                if task == f"{position}_{candidate}"
            ),
            None,
        )
        if layer not in layers or position not in positions or suffix is None:
            continue
        key = (str(record["case_id"]), layer, position, suffix)
        if key in output:
            raise ValueError(f"Duplicate OOF prediction key in {path}: {key}")
        output[key] = record
    return output


def _probability(record: dict[str, Any], label: str, key: tuple[Any, ...]) -> float:
    probabilities = record.get("class_probabilities")
    if not isinstance(probabilities, dict) or label not in probabilities:
        raise ValueError(f"OOF probability for {label!r} is missing at {key}")
    value = float(probabilities[label])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid OOF probability at {key}: {value}")
    return value


def _load_semantic_sa(
    path: Path,
    *,
    case_ids: set[str],
    layers: set[int],
) -> dict[tuple[str, int], float]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a compact readout list in {path}")
    output: dict[tuple[str, int], float] = {}
    for record in records:
        case_id = str(record.get("case_id"))
        if case_id not in case_ids:
            continue
        values = record.get("layers") or {}
        for layer in layers:
            compact = values.get(str(layer))
            if not isinstance(compact, list) or len(compact) <= 5:
                raise ValueError(f"SA_sac_semantic is missing for {(case_id, layer)}")
            value = float(compact[5])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid SA_sac_semantic for {(case_id, layer)}")
            key = (case_id, layer)
            if key in output:
                raise ValueError(f"Duplicate SA key: {key}")
            output[key] = value
    return output


def build_case_trajectories(
    manifest: Sequence[dict[str, Any]],
    answer_oof: dict[tuple[str, int, str, str], dict[str, Any]],
    decision_oof: dict[tuple[str, int, str, str], dict[str, Any]],
    semantic_sa: dict[tuple[str, int], float],
    *,
    layers: Sequence[int],
    positions: Sequence[str],
) -> list[dict[str, Any]]:
    """Join by stable keys only; input ordering has no effect."""

    output: list[dict[str, Any]] = []
    for source in sorted(manifest, key=lambda row: str(row["case_id"])):
        case_id = str(source["case_id"])
        for layer in layers:
            sa_key = (case_id, int(layer))
            if sa_key not in semantic_sa:
                raise ValueError(f"Missing SA key: {sa_key}")
            for position in positions:
                base = (case_id, int(layer), str(position))
                required = {
                    suffix: answer_oof.get((*base, suffix))
                    for suffix in ("text_answer", "image_answer", "conflict")
                }
                decision = decision_oof.get((*base, "decision_side"))
                missing = [name for name, value in required.items() if value is None]
                if decision is None:
                    missing.append("decision_side")
                if missing:
                    raise ValueError(f"Missing OOF keys for {base}: {missing}")
                assert decision is not None
                row = {
                    "case_id": case_id,
                    "item_id": str(source["item_id"]),
                    "condition": str(source["condition"]),
                    "decision_side": str(source["decision_side"]),
                    "layer": int(layer),
                    "position": str(position),
                    "R_T": _probability(
                        required["text_answer"],
                        str(source["text_only_answer"]),
                        (*base, "text_answer"),
                    ),
                    "R_I_preliminary": _probability(
                        required["image_answer"],
                        str(source["image_only_answer"]),
                        (*base, "image_answer"),
                    ),
                    "C": _probability(
                        required["conflict"], "conflict", (*base, "conflict")
                    ),
                    "K": _probability(
                        decision, "follows_image", (*base, "decision_side")
                    ),
                    "SA_sac_semantic": float(semantic_sa[sa_key]),
                }
                output.append(row)
    return output


def _spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    n = int(len(left))
    if n < 2:
        return {"rho": None, "p_value": None, "n": n, "reason": "insufficient_n"}
    if len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
        return {"rho": None, "p_value": None, "n": n, "reason": "constant_input"}
    result = spearmanr(left, right)
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": n,
        "reason": None,
    }


def _headline_metrics(
    metrics_path: Path,
    *,
    layers: Sequence[int],
    positions: Sequence[str],
    version_setting: str,
) -> dict[tuple[int, str], dict[str, Any]]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in payload.get("fold_results", []):
        if (
            result.get("status") != "valid"
            or result.get("target_field") != "decision_side"
            or result.get("model_type") != "hidden_state_probe"
            or result.get("version_setting") != version_setting
            or int(result.get("layer", -1)) not in set(layers)
            or str(result.get("position")) not in set(positions)
        ):
            continue
        subset = (result.get("subset_metrics") or {}).get("pooled_overall") or {}
        key = (int(result["layer"]), str(result["position"]))
        for name in ("balanced_accuracy", "roc_auc"):
            value = subset.get(name)
            if value is not None:
                grouped[key][name].append(float(value))
    output: dict[tuple[int, str], dict[str, Any]] = {}
    for layer in layers:
        for position in positions:
            key = (int(layer), str(position))
            values = grouped.get(key, {})
            if not values.get("balanced_accuracy") or not values.get("roc_auc"):
                raise ValueError(f"Decision headline metrics are incomplete for {key}")
            output[key] = {
                "balanced_accuracy_mean": float(
                    np.mean(values["balanced_accuracy"])
                ),
                "balanced_accuracy_std": float(np.std(values["balanced_accuracy"])),
                "roc_auc_mean": float(np.mean(values["roc_auc"])),
                "roc_auc_std": float(np.std(values["roc_auc"])),
                "valid_fold_count": len(values["balanced_accuracy"]),
            }
    return output


def _aggregate(
    case_rows: Sequence[dict[str, Any]],
    *,
    layers: Sequence[int],
    positions: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    case_correlations: list[dict[str, Any]] = []
    for layer in layers:
        for position in positions:
            layer_position = [
                row
                for row in case_rows
                if row["layer"] == layer and row["position"] == position
            ]
            for subgroup, predicate in SUBGROUPS.items():
                selected = [row for row in layer_position if predicate(row)]
                if not selected:
                    raise ValueError(f"Empty trajectory subgroup: {(layer, position, subgroup)}")
                correlation = _spearman(
                    [row["K"] for row in selected],
                    [row["SA_sac_semantic"] for row in selected],
                )
                aggregate = {
                    "layer": int(layer),
                    "position": position,
                    "subgroup": subgroup,
                    "sample_count": len(selected),
                    **{
                        field: float(np.mean([row[field] for row in selected]))
                        for field in TRAJECTORY_FIELDS
                    },
                    "K_SA_case_spearman_rho": correlation["rho"],
                    "K_SA_case_spearman_p": correlation["p_value"],
                    "K_SA_case_spearman_n": correlation["n"],
                    "K_SA_case_spearman_reason": correlation["reason"],
                }
                long_rows.append(aggregate)
                case_correlations.append(
                    {
                        "layer": int(layer),
                        "position": position,
                        "subgroup": subgroup,
                        **correlation,
                    }
                )
    return long_rows, case_correlations


def _layer_correlations(
    long_rows: Sequence[dict[str, Any]], positions: Sequence[str]
) -> list[dict[str, Any]]:
    output = []
    for position in positions:
        for subgroup in SUBGROUPS:
            selected = sorted(
                (
                    row
                    for row in long_rows
                    if row["position"] == position and row["subgroup"] == subgroup
                ),
                key=lambda row: int(row["layer"]),
            )
            correlation = _spearman(
                [row["K"] for row in selected],
                [row["SA_sac_semantic"] for row in selected],
            )
            output.append(
                {"position": position, "subgroup": subgroup, **correlation}
            )
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fieldnames = list(rows[0])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _wide_rows(
    long_rows: Sequence[dict[str, Any]],
    *,
    layers: Sequence[int],
    positions: Sequence[str],
    item_metrics: dict[tuple[int, str], dict[str, Any]],
    pair_metrics: dict[tuple[int, str], dict[str, Any]],
    layer_correlations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (int(row["layer"]), str(row["position"]), str(row["subgroup"])): row
        for row in long_rows
    }
    layer_corr = {
        (str(row["position"]), str(row["subgroup"])): row
        for row in layer_correlations
    }
    output = []
    for layer in layers:
        for subgroup in SUBGROUPS:
            row: dict[str, Any] = {"layer": int(layer), "subgroup": subgroup}
            first = lookup[(int(layer), str(positions[0]), subgroup)]
            row["sample_count"] = first["sample_count"]
            row["SA_sac_semantic"] = first["SA_sac_semantic"]
            for position in positions:
                prefix = position.upper()
                values = lookup[(int(layer), position, subgroup)]
                for field in ("R_T", "R_I_preliminary", "C", "K"):
                    row[f"{prefix}_{field}"] = values[field]
                row[f"{prefix}_K_SA_case_spearman_rho"] = values[
                    "K_SA_case_spearman_rho"
                ]
                row[f"{prefix}_K_SA_layer_spearman_rho"] = layer_corr[
                    (position, subgroup)
                ]["rho"]
                row[f"{prefix}_K_balanced_acc"] = item_metrics[
                    (int(layer), position)
                ]["balanced_accuracy_mean"]
                row[f"{prefix}_K_auc"] = item_metrics[(int(layer), position)][
                    "roc_auc_mean"
                ]
                row[f"{prefix}_K_answer_pair_balanced_acc"] = pair_metrics[
                    (int(layer), position)
                ]["balanced_accuracy_mean"]
                row[f"{prefix}_K_answer_pair_auc"] = pair_metrics[
                    (int(layer), position)
                ]["roc_auc_mean"]
            output.append(row)
    return output


def _plot(
    output_dir: Path,
    long_rows: Sequence[dict[str, Any]],
    *,
    layers: Sequence[int],
    positions: Sequence[str],
) -> None:
    for subgroup in SUBGROUPS:
        figure, axes = plt.subplots(
            1, len(positions), figsize=(5.2 * len(positions), 4.5), sharey=True
        )
        axes_values = np.atleast_1d(axes)
        for axis, position in zip(axes_values, positions):
            selected = sorted(
                (
                    row
                    for row in long_rows
                    if row["position"] == position and row["subgroup"] == subgroup
                ),
                key=lambda row: int(row["layer"]),
            )
            for field in TRAJECTORY_FIELDS:
                axis.plot(
                    layers,
                    [row[field] for row in selected],
                    marker="o",
                    label=field,
                )
            suffix = " (post-answer)" if position == "panl" else ""
            axis.set_title(f"{position.upper()}{suffix}")
            axis.set_xlabel("Layer")
            axis.set_xticks(list(layers))
            axis.grid(alpha=0.25)
        axes_values[0].set_ylabel("Mean probability / score")
        axes_values[-1].legend(fontsize=8, loc="best")
        figure.suptitle(
            f"Stage 1 {subgroup}: OOF readouts (R_I preliminary; no post-image state)"
        )
        figure.tight_layout()
        destination = output_dir / f"trajectory_{subgroup}.png"
        figure.savefig(destination, dpi=160)
        plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--probe-results-dir", required=True)
    parser.add_argument("--decision-item-results-dir", required=True)
    parser.add_argument("--decision-answer-pair-results-dir", required=True)
    parser.add_argument("--source-readout-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument(
        "--decision-side-probe-location",
        nargs="+",
        choices=list(DECISION_SIDE_LOCATIONS),
        required=True,
    )
    parser.add_argument("--version-setting", default="v4_to_v4")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    layers = [int(value) for value in args.layers]
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("--layers must contain distinct values")
    positions = [
        value
        for value in DECISION_SIDE_LOCATIONS
        if value in set(args.decision_side_probe_location)
    ]
    experiment_dir = Path(args.experiment_dir).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    probe_dir = Path(args.probe_results_dir).resolve()
    item_dir = Path(args.decision_item_results_dir).resolve()
    pair_dir = Path(args.decision_answer_pair_results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    source_path = (
        Path(args.source_readout_path).resolve()
        if args.source_readout_path
        else experiment_dir / "analysis_layer_readout_minimal_v4.json"
    )
    required = (
        manifest_path,
        probe_dir / "layer_probe_predictions.jsonl",
        item_dir / "layer_probe_predictions.jsonl",
        item_dir / "layer_probe_metrics.json",
        pair_dir / "layer_probe_metrics.json",
        source_path,
    )
    _require_files(required)
    output_dir.mkdir(parents=True, exist_ok=True)
    protected = (
        output_dir / "trajectory_summary.json",
        output_dir / "stage1_core_table.csv",
        output_dir / "trajectory_long.csv",
    )
    if any(path.exists() for path in protected):
        raise FileExistsError("Stage-1 trajectory outputs already exist")

    manifest = [
        row
        for row in iter_jsonl(manifest_path)
        if row.get("eligible_decision_side_probe") is True
    ]
    if not manifest:
        raise ValueError("No eligible Decision-Side cases in manifest")
    case_ids = {str(row["case_id"]) for row in manifest}
    if len(case_ids) != len(manifest):
        raise ValueError("Eligible manifest contains duplicate case_id values")
    layer_set = set(layers)
    position_set = set(positions)
    for row in manifest:
        reference = row.get("hidden_state_reference") or {}
        available_layers = {int(value) for value in reference.get("layer_indices", [])}
        available_positions = {
            str(value) for value in reference.get("position_names", [])
        }
        if not layer_set.issubset(available_layers):
            raise ValueError(f"Requested layer unavailable for {row['case_id']}")
        if not position_set.issubset(available_positions):
            raise ValueError(f"Requested position unavailable for {row['case_id']}")

    answer_oof = _load_oof(
        probe_dir / "layer_probe_predictions.jsonl",
        case_ids=case_ids,
        layers=layer_set,
        positions=position_set,
        task_suffixes={"text_answer", "image_answer", "conflict"},
        version_setting=args.version_setting,
    )
    decision_oof = _load_oof(
        item_dir / "layer_probe_predictions.jsonl",
        case_ids=case_ids,
        layers=layer_set,
        positions=position_set,
        task_suffixes={"decision_side"},
        version_setting=args.version_setting,
    )
    semantic_sa = _load_semantic_sa(
        source_path, case_ids=case_ids, layers=layer_set
    )
    case_rows = build_case_trajectories(
        manifest,
        answer_oof,
        decision_oof,
        semantic_sa,
        layers=layers,
        positions=positions,
    )
    long_rows, case_correlations = _aggregate(
        case_rows, layers=layers, positions=positions
    )
    layer_correlations = _layer_correlations(long_rows, positions)
    item_metrics = _headline_metrics(
        item_dir / "layer_probe_metrics.json",
        layers=layers,
        positions=positions,
        version_setting=args.version_setting,
    )
    pair_metrics = _headline_metrics(
        pair_dir / "layer_probe_metrics.json",
        layers=layers,
        positions=positions,
        version_setting=args.version_setting,
    )
    wide_rows = _wide_rows(
        long_rows,
        layers=layers,
        positions=positions,
        item_metrics=item_metrics,
        pair_metrics=pair_metrics,
        layer_correlations=layer_correlations,
    )
    manifest_summary = {
        "eligible_case_count": len(manifest),
        "item_count": len({str(row["item_id"]) for row in manifest}),
        "decision_side_counts": dict(Counter(row["decision_side"] for row in manifest)),
        "condition_counts": dict(Counter(row["condition"] for row in manifest)),
        "answer_pair_count": len(
            {str(row["unordered_answer_pair_key"]) for row in manifest}
        ),
    }
    atomic_write_json(
        output_dir / "decision_side_manifest_summary.json", manifest_summary
    )
    _write_csv(output_dir / "trajectory_long.csv", long_rows)
    _write_csv(output_dir / "stage1_core_table.csv", wide_rows)
    summary = {
        "format_version": 1,
        "version_setting": args.version_setting,
        "layers": layers,
        "positions": positions,
        "case_universe": "eligible_decision_side_probe",
        "manifest_summary": manifest_summary,
        "trajectory_source": {
            "R_T_R_I_C": str(probe_dir / "layer_probe_predictions.jsonl"),
            "K": str(item_dir / "layer_probe_predictions.jsonl"),
            "SA_sac_semantic": str(source_path),
        },
        "item_split_headline_metrics": [
            {"layer": key[0], "position": key[1], **value}
            for key, value in sorted(item_metrics.items())
        ],
        "answer_pair_split_headline_metrics": [
            {"layer": key[0], "position": key[1], **value}
            for key, value in sorted(pair_metrics.items())
        ],
        "casewise_K_SA_spearman": case_correlations,
        "layerwise_mean_K_SA_spearman": layer_correlations,
        "trajectory": long_rows,
        "interpretation_limits": {
            "R_I_preliminary": (
                "No post-image-token hidden state was saved; this readout cannot "
                "establish complete image-candidate formation timing."
            ),
            "PANL": "PANL is post-answer and is not evidence of pre-answer arbitration.",
            "causality": "All Stage-1 results are decodability/correlation, not causality.",
        },
    }
    atomic_write_json(output_dir / "trajectory_summary.json", summary)
    _plot(output_dir, long_rows, layers=layers, positions=positions)
    print(
        json.dumps(
            {
                "status": "complete",
                "eligible_cases": len(manifest),
                "trajectory_rows": len(long_rows),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
