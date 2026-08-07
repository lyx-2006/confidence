#!/usr/bin/env python3
"""Aggregate fold-level Probe metrics into a compact experiment summary."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from layer_metacognition.hidden_state_store import atomic_write_json

from . import (
    DEFAULT_PROBE_CONDITIONS,
    DEFAULT_PROBE_LOCATIONS,
    PROBE_CONDITIONS,
    PROBE_TASKS,
    build_probe_tasks,
)
from .common import iter_jsonl, probe_output_dir
from .train_layer_probes import filter_task_records

AGGREGATE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "cross_entropy",
    "majority_baseline_accuracy",
    "permuted_label_accuracy_mean",
)


def _aggregate(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "valid_fold_count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "valid_fold_count": len(values),
    }


def build_summary(
    manifest: list[dict[str, Any]],
    manifest_summary: dict[str, Any],
    metric_payload: dict[str, Any],
    *,
    probe_conditions: tuple[str, ...] = DEFAULT_PROBE_CONDITIONS,
    probe_tasks: dict[str, tuple[str, str]] = PROBE_TASKS,
) -> dict[str, Any]:
    task_counts: dict[str, Any] = {}
    for task, (_position, target) in probe_tasks.items():
        rows = filter_task_records(
            manifest,
            target,
            probe_conditions=probe_conditions,
        )
        task_counts[task] = {
            "record_count": len(rows),
            "item_count": len({str(row["item_id"]) for row in rows}),
            "version_counts": dict(Counter(str(row["version"]) for row in rows)),
            "condition_counts": dict(Counter(str(row["condition"]) for row in rows)),
        }
    selected_with_both = [
        row
        for row in manifest
        if row.get("condition") in probe_conditions
        and row.get("text_only_answer") is not None
        and row.get("image_only_answer") is not None
    ]
    answer_relationships = {
        "both_available_count": len(selected_with_both),
        "text_equals_image_count": sum(
            row["text_only_answer"] == row["image_only_answer"]
            for row in selected_with_both
        ),
        "text_differs_image_count": sum(
            row["text_only_answer"] != row["image_only_answer"]
            for row in selected_with_both
        ),
    }

    hidden_groups: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    baseline_groups: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    invalid: list[dict[str, Any]] = []
    for result in metric_payload.get("fold_results", []):
        if result.get("status") != "valid":
            invalid.append(
                {
                    key: result.get(key)
                    for key in (
                        "task",
                        "position",
                        "layer",
                        "fold",
                        "version_setting",
                        "model_type",
                        "invalid_reason",
                    )
                }
            )
            continue
        for subset, metrics in result.get("subset_metrics", {}).items():
            if result["model_type"] == "hidden_state_probe":
                key = (
                    result["task"],
                    int(result["layer"]),
                    result["version_setting"],
                    subset,
                )
                destination = hidden_groups[key]
            else:
                key = (result["task"], result["version_setting"], subset)
                destination = baseline_groups[key]
            if metrics.get("status") != "valid":
                continue
            for metric_name in AGGREGATE_METRICS:
                value = metrics.get(metric_name)
                if value is not None:
                    destination[metric_name].append(float(value))
            destination["_sample_count"].append(
                float(metrics.get("sample_count", 0))
            )
            destination["_item_count"].append(
                float(metrics.get("item_count", 0))
            )

    hidden_aggregates = []
    for key in sorted(
        hidden_groups,
        key=lambda value: (value[0], value[1], value[2], value[3]),
    ):
        task, layer, setting, subset = key
        hidden_aggregates.append(
            {
                "task": task,
                "position": probe_tasks[task][0],
                "layer": layer,
                "version_setting": setting,
                "subset": subset,
                "metrics": {
                    name: _aggregate(hidden_groups[key].get(name, []))
                    for name in AGGREGATE_METRICS
                },
                "sample_count": int(sum(hidden_groups[key].get("_sample_count", []))),
                "item_count": int(sum(hidden_groups[key].get("_item_count", []))),
            }
        )
    baseline_aggregates = []
    for key in sorted(
        baseline_groups,
        key=lambda value: (value[0], value[1], value[2]),
    ):
        task, setting, subset = key
        baseline_aggregates.append(
            {
                "task": task,
                "position": "panl",
                "version_setting": setting,
                "subset": subset,
                "metrics": {
                    name: _aggregate(baseline_groups[key].get(name, []))
                    for name in AGGREGATE_METRICS
                },
                "sample_count": int(sum(baseline_groups[key].get("_sample_count", []))),
                "item_count": int(sum(baseline_groups[key].get("_item_count", []))),
            }
        )
    return {
        "format_version": 1,
        "probe_conditions": list(probe_conditions),
        "probe_tasks": {
            key: {"position": value[0], "target_field": value[1]}
            for key, value in probe_tasks.items()
        },
        "answer_probe_locations": list(
            dict.fromkeys(
                position
                for position, target in probe_tasks.values()
                if target in {"text_only_answer", "image_only_answer"}
            )
        ),
        "conflict_probe_locations": list(
            dict.fromkeys(
                position
                for position, target in probe_tasks.values()
                if target == "conflict_label"
            )
        ),
        "manifest_record_count": len(manifest),
        "task_counts": task_counts,
        "overall_version_counts": dict(
            Counter(str(row["version"]) for row in manifest)
        ),
        "overall_condition_counts": dict(
            Counter(str(row["condition"]) for row in manifest)
        ),
        "selected_condition_counts": dict(
            Counter(
                str(row["condition"])
                for row in manifest
                if row.get("condition") in probe_conditions
            )
        ),
        "selected_missing_text_label_count": sum(
            row.get("condition") in probe_conditions
            and row.get("text_only_answer") is None
            for row in manifest
        ),
        "selected_missing_image_label_count": sum(
            row.get("condition") in probe_conditions
            and row.get("image_only_answer") is None
            for row in manifest
        ),
        "answer_relationships": answer_relationships,
        "hidden_state_probe": hidden_aggregates,
        "current_answer_only_baseline": baseline_aggregates,
        "invalid_folds": invalid,
        "invalid_fold_result_count": len(invalid),
        "image_hard_excluded_count": manifest_summary.get(
            "image_hard_excluded_count", 0
        ),
        "image_null_irr_excluded_count": manifest_summary.get(
            "image_null_irr_excluded_count", 0
        ),
        "aggregation": {
            "fold_weighting": "unweighted",
            "std_ddof": 0,
            "invalid_folds_excluded": True,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = probe_output_dir(args.experiment_dir, args.output_dir)
    metrics_path = output_dir / "layer_probe_metrics.json"
    run_config_path = output_dir / "run_config.json"
    for path in (metrics_path, run_config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Probe output does not exist: {path}")
    metric_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    manifest_path = Path(
        run_config.get("manifest_path") or output_dir / "probe_manifest.jsonl"
    ).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Required Probe manifest does not exist: {manifest_path}")
    manifest_summary_path = manifest_path.parent / "manifest_summary.json"
    manifest = list(iter_jsonl(manifest_path))
    manifest_summary = (
        json.loads(manifest_summary_path.read_text(encoding="utf-8"))
        if manifest_summary_path.is_file()
        else {}
    )
    configured_conditions = run_config.get("probe_conditions")
    if configured_conditions is None:
        configured_conditions = (
            list(DEFAULT_PROBE_CONDITIONS)
            if run_config.get("text_scope", "matched_easy") == "matched_easy"
            else list(PROBE_CONDITIONS)
        )
    probe_conditions = tuple(str(value) for value in configured_conditions)
    configured_tasks = run_config.get("probe_tasks") or {}
    probe_tasks = {
        str(key): (str(value["position"]), str(value["target_field"]))
        for key, value in configured_tasks.items()
    } or build_probe_tasks(DEFAULT_PROBE_LOCATIONS, ())
    summary = build_summary(
        manifest,
        manifest_summary,
        metric_payload,
        probe_conditions=probe_conditions,
        probe_tasks=probe_tasks,
    )
    atomic_write_json(output_dir / "probe_summary.json", summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "hidden_aggregate_count": len(summary["hidden_state_probe"]),
                "baseline_aggregate_count": len(
                    summary["current_answer_only_baseline"]
                ),
                "invalid_fold_result_count": summary["invalid_fold_result_count"],
                "output": str(output_dir / "probe_summary.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
