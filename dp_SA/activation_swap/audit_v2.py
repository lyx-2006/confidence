from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .metrics import (
    bh_fdr,
    bootstrap_values,
    condition_side,
    score_logits,
    sign_flip_p,
    stratified_effect_summary,
)
from .utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file, stable_seed


POSITIONS = ("P1_PANL", "P1_PANL_PLUS_1", "P1_SAC")
POSITION_LABELS = {"P1_PANL": "PANL", "P1_PANL_PLUS_1": "PANL + 1", "P1_SAC": "SAC"}
CONDITIONS = ("I_from_I", "I_from_T", "T_from_I", "T_from_T")
CONDITION_LABELS = {"I_from_I": "I2I", "I_from_T": "I2T", "T_from_I": "T2I", "T_from_T": "T2T"}
TARGET_LAYERS = (14, 16)
METRIC_FIELDS = {
    "soft_sa": "swap_soft_sa",
    "hard_midpoint": "swap_hard_midpoint",
    "fixed_clean_class_margin": "swap_fixed_clean_class_margin",
    "first_token_change": "first_token_changed",
}
INPUT_FILES = (
    "swap_predictions.jsonl",
    "clean_predictions.jsonl",
    "swap_pair_manifest.jsonl",
    "activation_diagnostics.csv",
    "bootstrap_results.csv",
    "position_contrasts.csv",
    "summary.json",
    "summary.md",
    "run_config.json",
    "input_fingerprints.json",
)


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    if not fields:
        raise ValueError(f"cannot write empty CSV: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def oriented_effect(same: float, cross: float, side: str, metric: str) -> float:
    """Prespecified recipient-paired directions used by audit v2."""
    if metric == "first_token_change":
        return float(cross - same)
    if metric == "fixed_clean_class_margin":
        return float(same - cross)
    if metric in {"soft_sa", "hard_midpoint"}:
        if side == "image_side":
            return float(same - cross)
        if side == "text_side":
            return float(cross - same)
    raise ValueError(f"invalid side/metric: {side}/{metric}")


def margin_regression_example() -> dict[str, float]:
    image = oriented_effect(-0.0069, -0.0759, "image_side", "fixed_clean_class_margin")
    text = oriented_effect(-0.0088, -0.0272, "text_side", "fixed_clean_class_margin")
    return {"image_margin_effect": image, "text_margin_effect": text,
            "combined_margin_effect": 0.5 * (image + text)}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _source_hashes(source: Path) -> dict[str, str]:
    missing = [name for name in INPUT_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing audit inputs: {missing}")
    return {name: sha256_file(source / name) for name in INPUT_FILES}


def _prepare_output(source: Path, output: Path, config: dict[str, Any], hashes: dict[str, str], *, resume: bool) -> None:
    source = source.resolve()
    output = output.resolve()
    if output == source or source not in output.parents:
        raise ValueError("audit output must be a dedicated child of the formal result directory")
    config_path = output / "process" / "audit_config.json"
    fingerprint_path = output / "process" / "audit_fingerprints.json"
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(f"audit output already exists: {output}; pass --resume")
        if not config_path.is_file() or not fingerprint_path.is_file():
            raise ValueError("audit resume metadata is incomplete")
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        previous_hashes = json.loads(fingerprint_path.read_text(encoding="utf-8"))["source_file_sha256"]
        if previous_config.get("fingerprint") != config.get("fingerprint"):
            raise ValueError("audit config fingerprint changed; refusing resume")
        if previous_hashes != hashes:
            raise ValueError("audit input fingerprint changed; refusing resume")
    for name in ("figures", "tables", "experiment_data", "process", "report"):
        (output / name).mkdir(parents=True, exist_ok=True)
    atomic_json(config_path, config)
    atomic_json(fingerprint_path, {"source_file_sha256": hashes, "source_root": str(source)})


def _recompute_trials(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(source / "swap_predictions.jsonl")
    clean_rows = load_jsonl(source / "clean_predictions.jsonl")
    config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    expected = int(config["expected_swap_forwards"])
    if len(rows) != expected or len({str(row["trial_key"]) for row in rows}) != expected:
        raise ValueError(f"incomplete or duplicate trial grid: {len(rows)}/{expected}")
    clean_by_case = {str(row["case_id"]): row for row in clean_rows}
    if len(clean_by_case) != int(config["expected_clean_forwards"]):
        raise ValueError("clean prediction count mismatch")
    recomputed: list[dict[str, Any]] = []
    maximum_metric_delta = 0.0
    for raw in rows:
        case_id = str(raw["recipient_case_id"])
        if case_id not in clean_by_case:
            raise ValueError(f"recipient clean row missing: {case_id}")
        clean_logits = np.asarray(raw["clean_class_logits"], dtype=np.float64)
        swap_logits = np.asarray(raw["swap_class_logits"], dtype=np.float64)
        if clean_logits.shape != (9,) or swap_logits.shape != (9,) or not np.isfinite(clean_logits).all() or not np.isfinite(swap_logits).all():
            raise ValueError(f"invalid logits: {raw['trial_key']}")
        clean_class = int(np.argmax(clean_logits))
        if clean_class != int(raw["clean_class"]):
            raise ValueError(f"clean class mismatch: {raw['trial_key']}")
        clean_score = score_logits(clean_logits, clean_class=clean_class)
        swap_score = score_logits(swap_logits, clean_class=clean_class)
        changed = bool(int(swap_score["hard_class"]) != clean_class)
        expected_condition = condition_side(str(raw["recipient_side"]), str(raw["donor_side"]))
        if expected_condition != str(raw["condition"]):
            raise ValueError(f"condition mismatch: {raw['trial_key']}")
        checks = {
            "clean_soft_sa": clean_score["soft_sa"],
            "clean_hard_midpoint": clean_score["hard_midpoint"],
            "clean_fixed_clean_class_margin": clean_score["fixed_clean_class_margin"],
            "swap_soft_sa": swap_score["soft_sa"],
            "swap_hard_midpoint": swap_score["hard_midpoint"],
            "swap_fixed_clean_class_margin": swap_score["fixed_clean_class_margin"],
            "first_token_changed": changed,
        }
        for field, value in checks.items():
            old = raw[field]
            delta = 0.0 if isinstance(value, bool) and bool(old) == value else abs(float(old) - float(value))
            maximum_metric_delta = max(maximum_metric_delta, delta)
            if delta > 1e-12:
                raise ValueError(f"stored trial metric differs from raw logits: {raw['trial_key']} {field} {delta}")
        diag = raw.get("activation_diagnostics", {})
        required = ("cosine_distance", "norm_ratio", "abs_log_norm_ratio", "recipient_norm", "donor_norm")
        if any(name not in diag or not math.isfinite(float(diag[name])) for name in required):
            raise ValueError(f"activation diagnostics insufficient: {raw['trial_key']}")
        row = dict(raw)
        row.update(checks)
        row["margin_change"] = float(swap_score["fixed_clean_class_margin"] - clean_score["fixed_clean_class_margin"])
        row["cosine_distance"] = float(diag["cosine_distance"])
        row["norm_ratio"] = float(diag["norm_ratio"])
        row["abs_log_norm_ratio"] = abs(math.log(float(diag["norm_ratio"])))
        recomputed.append(row)
    return recomputed, {"trial_count": len(recomputed), "clean_count": len(clean_rows),
                        "maximum_stored_vs_recomputed_metric_delta": maximum_metric_delta}


def _paired_rows(rows: Sequence[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    field = METRIC_FIELDS[metric]
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["recipient_case_id"])][str(row["swap_kind"])] = row
    output = []
    for case_id, pair in sorted(grouped.items()):
        if set(pair) != {"same", "cross"}:
            raise ValueError(f"recipient pair incomplete: {case_id}")
        same, cross = pair["same"], pair["cross"]
        side = str(same["recipient_side"])
        same_value, cross_value = float(same[field]), float(cross[field])
        output.append({
            "recipient_case_id": case_id,
            "recipient_side": side,
            "same": same_value,
            "cross": cross_value,
            "effect": oriented_effect(same_value, cross_value, side, metric),
            "same_donor_case_id": str(same.get("donor_case_id", "")),
            "cross_donor_case_id": str(cross.get("donor_case_id", "")),
        })
    return output


def _summarize_pairs(pairs: Sequence[dict[str, Any]], group: str, *, repeats: int, seed: int) -> dict[str, Any]:
    subset = list(pairs) if group == "all" else [row for row in pairs if row["recipient_side"] == group]
    if not subset:
        raise ValueError(f"empty pair group: {group}")
    if group == "all":
        return stratified_effect_summary(subset, repeats=repeats, seed=seed)
    return bootstrap_values([float(row["effect"]) for row in subset], repeats=repeats, seed=seed)


def _position_effect_rows(
    grouped: dict[tuple[str, int], list[dict[str, Any]]], position: str, control: str, layer: int, metric: str,
) -> list[dict[str, Any]]:
    left = {str(row["recipient_case_id"]): row for row in _paired_rows(grouped[(position, layer)], metric)}
    right = {str(row["recipient_case_id"]): row for row in _paired_rows(grouped[(control, layer)], metric)}
    output = []
    for case_id in sorted(set(left) & set(right)):
        if left[case_id]["recipient_side"] != right[case_id]["recipient_side"]:
            raise ValueError("position contrast side mismatch")
        output.append({"recipient_case_id": case_id, "recipient_side": left[case_id]["recipient_side"],
                       "effect": float(left[case_id]["effect"] - right[case_id]["effect"]),
                       "left_effect": left[case_id]["effect"], "right_effect": right[case_id]["effect"]})
    return output


def _statistical_tables(rows: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["position"]), int(row["layer"]))].append(row)
    bootstrap: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    margin_conditions: list[dict[str, Any]] = []
    paired_data: list[dict[str, Any]] = []
    p_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (position, layer), cell in sorted(grouped.items()):
        for condition in CONDITIONS:
            subset = [row for row in cell if row["condition"] == condition]
            for metric, field, outer_metric in (("swap_soft_sa", "swap_soft_sa", "soft_sa"),
                                  ("swap_hard_midpoint", "swap_hard_midpoint", "hard_midpoint"),
                                  ("swap_fixed_clean_class_margin", "swap_fixed_clean_class_margin", "fixed_clean_class_margin"),
                                  ("first_token_changed", "first_token_changed", "first_token_change")):
                stats = bootstrap_values([float(row[field]) for row in subset], repeats=repeats,
                                         seed=stable_seed(stable_seed(seed, "condition", position, str(layer), outer_metric),
                                                          "condition", condition, metric))
                bootstrap.append({"analysis": "condition", "position": position, "layer": layer,
                                  "condition": condition, "metric": metric, **stats})
            stats = bootstrap_values([float(row["margin_change"]) for row in subset], repeats=repeats,
                                     seed=stable_seed(seed, "audit-margin-condition", position, layer, condition))
            margin_conditions.append({"analysis": "condition_margin_change", "position": position, "layer": layer,
                                      "condition": condition, "recipient_side": subset[0]["recipient_side"],
                                      "swap_kind": subset[0]["swap_kind"], "metric": "fixed_clean_class_margin_change", **stats})
        for arm in ("same", "cross"):
            arm_rows = [row for row in cell if row["swap_kind"] == arm]
            for group in ("all", "image_side", "text_side"):
                subset = arm_rows if group == "all" else [row for row in arm_rows if row["recipient_side"] == group]
                values = [float(row["margin_change"]) for row in subset]
                stats = (stratified_effect_summary(
                    [{"recipient_side": row["recipient_side"], "effect": float(row["margin_change"])} for row in subset],
                    repeats=repeats, seed=stable_seed(seed, "audit-margin-arm", position, layer, arm, group))
                    if group == "all" else bootstrap_values(values, repeats=repeats,
                    seed=stable_seed(seed, "audit-margin-arm", position, layer, arm, group)))
                margin_conditions.append({"analysis": "margin_arm_change", "position": position, "layer": layer,
                                          "swap_kind": arm, "group": group,
                                          "metric": "fixed_clean_class_margin_change", **stats})
        for metric in METRIC_FIELDS:
            pairs = _paired_rows(cell, metric)
            for pair in pairs:
                paired_data.append({"position": position, "layer": layer, "metric": metric, **pair})
            for group in ("all", "image_side", "text_side"):
                subset = pairs if group == "all" else [row for row in pairs if row["recipient_side"] == group]
                stats = _summarize_pairs(pairs, group, repeats=repeats,
                                         seed=stable_seed(seed, "effect", position, str(layer), metric, group))
                p_raw = (sign_flip_p([float(row["effect"]) for row in subset], repeats=repeats,
                                     seed=stable_seed(seed, "signflip", position, str(layer), metric, group))
                         if group == "all" else None)
                result = {"analysis": "paired_effect", "position": position, "layer": layer,
                          "metric": metric, "group": group, "p_raw": p_raw,
                          "targeted_confirmatory": bool(position == "P1_PANL" and layer in TARGET_LAYERS), **stats}
                bootstrap.append(result)
                if group == "all":
                    p_candidates[metric].append(result)
            if metric == "first_token_change":
                for arm in ("same", "cross"):
                    arm_rows = [row for row in cell if row["swap_kind"] == arm]
                    for group in ("all", "image_side", "text_side"):
                        subset = arm_rows if group == "all" else [row for row in arm_rows if row["recipient_side"] == group]
                        stats = (stratified_effect_summary(
                            [{"recipient_side": row["recipient_side"], "effect": float(row["first_token_changed"])} for row in subset],
                            repeats=repeats, seed=stable_seed(seed, "arm", position, str(layer), metric, arm, group))
                            if group == "all" else bootstrap_values([float(row["first_token_changed"]) for row in subset], repeats=repeats,
                            seed=stable_seed(seed, "arm", position, str(layer), metric, arm, group)))
                        bootstrap.append({"analysis": "swap_arm_rate", "position": position, "layer": layer,
                                          "metric": metric, "arm": arm, "group": group, **stats})
    for metric, candidates in p_candidates.items():
        exploratory = [row for row in candidates if not row["targeted_confirmatory"]]
        for row, q in zip(exploratory, bh_fdr([float(row["p_raw"]) for row in exploratory])):
            row["q_bh"] = q
            row["fdr_family"] = f"{metric}:non_target_position_layer_cells"
        for row in candidates:
            row.setdefault("q_bh", None)
            row.setdefault("fdr_family", None)
    for layer in sorted({layer for _position, layer in grouped}):
        for metric in METRIC_FIELDS:
            for position, name in (("P1_PANL", "PANL_minus_PANL_PLUS_1"), ("P1_SAC", "SAC_minus_PANL_PLUS_1")):
                pairs = _position_effect_rows(grouped, position, "P1_PANL_PLUS_1", layer, metric)
                for group in ("all", "image_side", "text_side"):
                    subset = pairs if group == "all" else [row for row in pairs if row["recipient_side"] == group]
                    stats = _summarize_pairs(pairs, group, repeats=repeats,
                                             seed=stable_seed(seed, "contrast", name, str(layer), metric, group))
                    contrasts.append({"analysis": "position_contrast", "contrast": name, "layer": layer,
                                      "metric": metric, "group": group,
                                      "p_raw": sign_flip_p([float(row["effect"]) for row in subset], repeats=repeats,
                                      seed=stable_seed(seed, "contrast-p", name, str(layer), metric)) if group == "all" else None,
                                      "targeted_confirmatory": bool(layer in TARGET_LAYERS and name.startswith("PANL")), **stats})
    return bootstrap, contrasts, margin_conditions, paired_data


def _numeric_equal(new: Any, old: Any, tolerance: float = 1e-12) -> bool:
    if new in (None, "") and old in (None, ""):
        return True
    try:
        return abs(float(new) - float(old)) <= tolerance
    except (TypeError, ValueError):
        return str(new) == str(old)


def _parity_gate(source: Path, corrected_bootstrap: Sequence[dict[str, Any]], corrected_contrasts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    old_bootstrap = _read_csv(source / "bootstrap_results.csv")
    old_contrasts = _read_csv(source / "position_contrasts.csv")
    keys = ("analysis", "position", "layer", "metric", "group", "condition", "arm")
    old_lookup = {tuple(str(row.get(k, "")) for k in keys): row for row in old_bootstrap}
    checked = 0
    for row in corrected_bootstrap:
        if row.get("metric") in {"fixed_clean_class_margin", "swap_fixed_clean_class_margin"}:
            continue
        key = tuple(str(row.get(k, "")) for k in keys)
        old = old_lookup.get(key)
        if old is None:
            raise ValueError(f"non-margin parity row missing: {key}")
        for field in ("mean", "sem", "ci_low", "ci_high", "p_raw", "q_bh", "sample_count"):
            if not _numeric_equal(row.get(field), old.get(field)):
                raise ValueError(f"non-margin bootstrap parity failed: {key} {field}: {row.get(field)} != {old.get(field)}")
        checked += 1
    contrast_keys = ("analysis", "contrast", "layer", "metric", "group")
    old_contrast_lookup = {tuple(str(row.get(k, "")) for k in contrast_keys): row for row in old_contrasts}
    contrast_checked = 0
    for row in corrected_contrasts:
        if row["metric"] == "fixed_clean_class_margin":
            continue
        key = tuple(str(row.get(k, "")) for k in contrast_keys)
        old = old_contrast_lookup.get(key)
        if old is None:
            raise ValueError(f"non-margin contrast parity row missing: {key}")
        for field in ("mean", "sem", "ci_low", "ci_high", "p_raw", "sample_count"):
            if not _numeric_equal(row.get(field), old.get(field)):
                raise ValueError(f"non-margin contrast parity failed: {key} {field}")
        contrast_checked += 1
    return {"non_margin_bootstrap_rows_checked": checked, "non_margin_contrast_rows_checked": contrast_checked,
            "tolerance": 1e-12, "passed": True}


def _percentiles(values: Sequence[float], points: Sequence[float]) -> list[float]:
    return [float(value) for value in np.percentile(np.asarray(values, dtype=np.float64), points)]


def _activation_summary(rows: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["position"]), int(row["layer"]), str(row["condition"]))].append(row)
    output = []
    for (position, layer, condition), cell in sorted(grouped.items()):
        cos = [float(row["cosine_distance"]) for row in cell]
        ratio = [float(row["norm_ratio"]) for row in cell]
        cos_median, cos_p90, cos_p95 = _percentiles(cos, (50, 90, 95))
        norm_p5, norm_median, norm_p95 = _percentiles(ratio, (5, 50, 95))
        output.append({
            "analysis": "condition",
            "position": position, "layer": layer, "condition": condition,
            "recipient_side": cell[0]["recipient_side"], "swap_kind": cell[0]["swap_kind"], "n": len(cell),
            "cosine_mean": float(np.mean(cos)), "cosine_median": cos_median, "cosine_p90": cos_p90,
            "cosine_p95": cos_p95, "cosine_max": max(cos),
            "cosine_gt_0_1_count": sum(value > 0.1 for value in cos),
            "cosine_gt_0_1_rate": float(np.mean(np.asarray(cos) > 0.1)),
            "norm_ratio_mean": float(np.mean(ratio)), "norm_ratio_median": norm_median,
            "norm_ratio_p5": norm_p5, "norm_ratio_p95": norm_p95,
            "norm_ratio_min": min(ratio), "norm_ratio_max": max(ratio),
            "norm_ratio_outside_0_5_2_count": sum(value < 0.5 or value > 2.0 for value in ratio),
            "norm_ratio_outside_0_5_2_rate": float(np.mean([(value < 0.5 or value > 2.0) for value in ratio])),
        })
    by_cell: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(str(row["position"]), int(row["layer"]), str(row["swap_kind"]))].append(row)
    for (position, layer, arm), cell in sorted(by_cell.items()):
        for diagnostic, field in (("cosine_distance", "cosine_distance"),
                                  ("abs_log_norm_ratio", "abs_log_norm_ratio")):
            side_rows = [{"recipient_side": row["recipient_side"], "effect": float(row[field])} for row in cell]
            stats = stratified_effect_summary(
                side_rows, repeats=repeats,
                seed=stable_seed(seed, "activation-arm", position, layer, arm, diagnostic),
            )
            output.append({"analysis": "arm", "position": position, "layer": layer,
                           "swap_kind": arm, "diagnostic_metric": diagnostic, **stats})
    return output


def matched_drift_pairs(rows: Sequence[dict[str, Any]], position: str, layer: int, diagnostic: str) -> list[dict[str, Any]]:
    cell = [row for row in rows if row["position"] == position and int(row["layer"]) == int(layer)]
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in cell:
        grouped[str(row["recipient_case_id"])][str(row["swap_kind"])] = row
    output = []
    for case_id, pair in sorted(grouped.items()):
        if set(pair) != {"same", "cross"}:
            raise ValueError(f"diagnostic pair incomplete: {case_id}")
        field = "cosine_distance" if diagnostic == "cosine_distance" else "abs_log_norm_ratio"
        same, cross = float(pair["same"][field]), float(pair["cross"][field])
        output.append({"recipient_case_id": case_id, "recipient_side": pair["same"]["recipient_side"],
                       "same": same, "cross": cross, "effect": float(cross - same)})
    return output


def panl_drift_contrast(rows: Sequence[dict[str, Any]], layer: int, diagnostic: str) -> list[dict[str, Any]]:
    left = {row["recipient_case_id"]: row for row in matched_drift_pairs(rows, "P1_PANL", layer, diagnostic)}
    right = {row["recipient_case_id"]: row for row in matched_drift_pairs(rows, "P1_PANL_PLUS_1", layer, diagnostic)}
    return [{"recipient_case_id": case_id, "recipient_side": left[case_id]["recipient_side"],
             "effect": float(left[case_id]["effect"] - right[case_id]["effect"])}
            for case_id in sorted(set(left) & set(right))]


def _activation_paired_contrasts(rows: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> list[dict[str, Any]]:
    layers = sorted({int(row["layer"]) for row in rows})
    output = []
    for diagnostic in ("cosine_distance", "abs_log_norm_ratio"):
        for layer in layers:
            for position in POSITIONS:
                pairs = matched_drift_pairs(rows, position, layer, diagnostic)
                for group in ("all", "image_side", "text_side"):
                    subset = pairs if group == "all" else [row for row in pairs if row["recipient_side"] == group]
                    stats = _summarize_pairs(pairs, group, repeats=repeats,
                                             seed=stable_seed(seed, "audit-drift", position, layer, diagnostic, group))
                    output.append({"analysis": "cross_minus_same", "position": position, "layer": layer,
                                   "diagnostic_metric": diagnostic, "group": group,
                                   "p_raw": sign_flip_p([float(row["effect"]) for row in subset], repeats=repeats,
                                   seed=stable_seed(seed, "audit-drift-p", position, layer, diagnostic, group)), **stats})
            pairs = panl_drift_contrast(rows, layer, diagnostic)
            for group in ("all", "image_side", "text_side"):
                subset = pairs if group == "all" else [row for row in pairs if row["recipient_side"] == group]
                stats = _summarize_pairs(pairs, group, repeats=repeats,
                                         seed=stable_seed(seed, "audit-drift-position", layer, diagnostic, group))
                output.append({"analysis": "PANL_minus_PANL_PLUS_1_cross_specific_drift", "position": "",
                               "layer": layer, "diagnostic_metric": diagnostic, "group": group,
                               "p_raw": sign_flip_p([float(row["effect"]) for row in subset], repeats=repeats,
                               seed=stable_seed(seed, "audit-drift-position-p", layer, diagnostic, group)), **stats})
    return output


def _side(value: Any) -> str:
    text = str(value or "").lower()
    if "image" in text:
        return "image"
    if "text" in text:
        return "text"
    raise ValueError(f"cannot normalize side: {value}")


def _load_clean_hidden(source: Path, clean_rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    import torch
    output: dict[str, dict[str, Any]] = {}
    run_config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    for row in clean_rows:
        case_id = str(row["case_id"])
        # The formal runner passed this path into _clean_record but the
        # historical record constructor did not persist the hidden_file
        # argument.  The immutable runner rule is clean_cache/{case_id}.pt.
        relative = row.get("hidden_file") or f"clean_cache/{case_id}.pt"
        path = source / str(relative)
        if not path.is_file() or sha256_file(path) != str(row["cache_sha256"]):
            raise FileNotFoundError(f"clean hidden cache unavailable or changed: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if str(payload.get("case_id")) != case_id or payload.get("run_fingerprint") != run_config.get("fingerprint"):
            raise ValueError(f"clean hidden cache identity mismatch: {path}")
        side = _side(row.get("test_side") or row.get("construction_side"))
        output[case_id] = {"side": side, "hidden": payload["hidden"]}
    return output


def _hidden_vector(record: dict[str, Any], position: str, layer: int) -> np.ndarray:
    value = record["hidden"][position][layer]
    return np.asarray(value.detach().cpu().numpy(), dtype=np.float64).reshape(-1)


def _natural_reference(
    source: Path, rows: Sequence[dict[str, Any]], *, seed: int, pairs_per_condition: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    clean_rows = load_jsonl(source / "clean_predictions.jsonl")
    try:
        hidden = _load_clean_hidden(source, clean_rows)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return ([{"status": "natural_pairwise_reference_unavailable", "reason": str(exc)}], [], "unavailable")
    recipient_ids = sorted({str(row["recipient_case_id"]) for row in rows})
    donor_ids = sorted({str(row["donor_case_id"]) for row in rows})
    pair_rows: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    layers = sorted({int(row["layer"]) for row in rows})
    swap_grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        swap_grouped[(str(row["position"]), int(row["layer"]), str(row["condition"]))].append(float(row["cosine_distance"]))
    for position in POSITIONS:
        for layer in layers:
            vectors = {case_id: _hidden_vector(hidden[case_id], position, layer) for case_id in set(recipient_ids + donor_ids)}
            norms = {case_id: float(np.linalg.norm(vector)) for case_id, vector in vectors.items()}
            for condition in CONDITIONS:
                recipient_side = "image" if condition.startswith("I_") else "text"
                donor_side = "image" if condition.endswith("_I") else "text"
                candidates = [(r, d) for r in recipient_ids if hidden[r]["side"] == recipient_side
                              for d in donor_ids if hidden[d]["side"] == donor_side]
                rng = np.random.default_rng(stable_seed(seed, "natural-reference", position, layer, condition))
                order = rng.permutation(len(candidates))[:min(pairs_per_condition, len(candidates))]
                natural_cos: list[float] = []
                natural_ratio: list[float] = []
                for index in order:
                    recipient_id, donor_id = candidates[int(index)]
                    cosine = float(1.0 - np.dot(vectors[recipient_id], vectors[donor_id]) /
                                   (norms[recipient_id] * norms[donor_id]))
                    ratio = float(norms[donor_id] / norms[recipient_id])
                    natural_cos.append(cosine)
                    natural_ratio.append(ratio)
                    pair_rows.append({"position": position, "layer": layer, "condition": condition,
                                      "recipient_case_id": recipient_id, "donor_case_id": donor_id,
                                      "cosine_distance": cosine, "norm_ratio": ratio})
                cp5, cp50, cp95, cp99 = _percentiles(natural_cos, (5, 50, 95, 99))
                rp5, rp50, rp95, rp99 = _percentiles(natural_ratio, (5, 50, 95, 99))
                swap_values = swap_grouped[(position, layer, condition)]
                percentiles = [100.0 * float(np.mean(np.asarray(natural_cos) <= value)) for value in swap_values]
                reference.append({"status": "available", "position": position, "layer": layer,
                                  "condition": condition, "swap_kind": "same" if condition in {"I_from_I", "T_from_T"} else "cross",
                                  "natural_pair_count": len(natural_cos), "cosine_p5": cp5, "cosine_p50": cp50,
                                  "cosine_p95": cp95, "cosine_p99": cp99, "norm_ratio_p5": rp5,
                                  "norm_ratio_p50": rp50, "norm_ratio_p95": rp95, "norm_ratio_p99": rp99,
                                  "swap_count": len(swap_values), "swap_cosine_mean": float(np.mean(swap_values)),
                                  "swap_natural_percentile_mean": float(np.mean(percentiles)),
                                  "swap_natural_percentile_median": float(np.median(percentiles))})
    return reference, pair_rows, "available"


def _distance_matched_sensitivity(
    rows: Sequence[dict[str, Any]], *, repeats: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["position"]), int(row["layer"]))].append(row)
    output: list[dict[str, Any]] = []
    membership: list[dict[str, Any]] = []
    for layer in TARGET_LAYERS:
        drift = {row["recipient_case_id"]: row for row in matched_drift_pairs(rows, "P1_PANL", layer, "cosine_distance")}
        absolute = {case_id: abs(float(row["effect"])) for case_id, row in drift.items()}
        thresholds = {"full": math.inf, "abs_delta_median": float(np.median(list(absolute.values()))),
                      "abs_delta_p75": float(np.percentile(list(absolute.values()), 75))}
        panl_pairs = {row["recipient_case_id"]: row for row in _paired_rows(grouped[("P1_PANL", layer)], "soft_sa")}
        control_pairs = {row["recipient_case_id"]: row for row in _paired_rows(grouped[("P1_PANL_PLUS_1", layer)], "soft_sa")}
        for rule, tau in thresholds.items():
            selected = sorted(case_id for case_id, value in absolute.items() if value <= tau)
            for case_id in sorted(absolute):
                membership.append({"layer": layer, "rule": rule, "tau": None if math.isinf(tau) else tau,
                                   "recipient_case_id": case_id, "recipient_side": drift[case_id]["recipient_side"],
                                   "abs_cross_minus_same_cosine": absolute[case_id], "selected": case_id in selected})
            panl = [panl_pairs[case_id] for case_id in selected]
            contrast = [{"recipient_case_id": case_id, "recipient_side": panl_pairs[case_id]["recipient_side"],
                         "effect": float(panl_pairs[case_id]["effect"] - control_pairs[case_id]["effect"])} for case_id in selected]
            for analysis, pairs in (("PANL_soft_effect", panl), ("PANL_minus_PANL_PLUS_1_soft_contrast", contrast)):
                for group in ("all", "image_side", "text_side"):
                    subset = pairs if group == "all" else [row for row in pairs if row["recipient_side"] == group]
                    if not subset:
                        continue
                    stats = _summarize_pairs(pairs, group, repeats=repeats,
                                             seed=stable_seed(seed, "distance-sensitivity", layer, rule, analysis, group))
                    output.append({"layer": layer, "rule": rule, "tau": None if math.isinf(tau) else tau,
                                   "analysis": analysis, "group": group,
                                   "image_count": sum(row["recipient_side"] == "image_side" for row in subset),
                                   "text_count": sum(row["recipient_side"] == "text_side" for row in subset), **stats})
    return output, membership


def _combined_mean(rows: Sequence[dict[str, Any]]) -> float:
    image = [float(row["effect"]) for row in rows if row["recipient_side"] == "image_side"]
    text = [float(row["effect"]) for row in rows if row["recipient_side"] == "text_side"]
    if not image or not text:
        raise ValueError("combined effect requires both recipient sides")
    return float(0.5 * (np.mean(image) + np.mean(text)))


def _donor_sensitivity(rows: Sequence[dict[str, Any]], *, contribution_threshold: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["position"]), int(row["layer"]))].append(row)
    donor_ids = sorted({str(row["donor_case_id"]) for row in rows})
    output = []
    for layer in TARGET_LAYERS:
        for metric in ("soft_sa", "fixed_clean_class_margin"):
            pairs = _paired_rows(grouped[("P1_PANL", layer)], metric)
            for group in ("all", "image_side", "text_side"):
                base = pairs if group == "all" else [row for row in pairs if row["recipient_side"] == group]
                full = _combined_mean(base) if group == "all" else float(np.mean([row["effect"] for row in base]))
                for donor_id in donor_ids:
                    remaining = [row for row in base if donor_id not in {row["same_donor_case_id"], row["cross_donor_case_id"]}]
                    if not remaining:
                        continue
                    leave = _combined_mean(remaining) if group == "all" else float(np.mean([row["effect"] for row in remaining]))
                    relative = abs(leave - full) / abs(full) if full != 0 else math.inf
                    output.append({"position": "P1_PANL", "layer": layer, "metric": metric, "group": group,
                                   "donor_case_id": donor_id, "full_effect": full, "leave_one_donor_effect": leave,
                                   "remaining_recipient_count": len(remaining),
                                   "direction_reversed": bool(full * leave < 0),
                                   "relative_effect_change": relative,
                                   "contribution_threshold": contribution_threshold,
                                   "exceeds_contribution_threshold": bool(relative > contribution_threshold)})
    return output


def classify_ood_status(
    activation_summary: Sequence[dict[str, Any]], paired: Sequence[dict[str, Any]], *, natural_status: str,
) -> tuple[str, dict[str, Any]]:
    if not paired:
        return "unavailable", {"reason": "matched drift analysis unavailable"}
    heuristic_count = sum(int(row["cosine_gt_0_1_count"]) for row in activation_summary if "cosine_gt_0_1_count" in row)
    target_cross = [row for row in paired if row["analysis"] == "cross_minus_same" and
                    row["position"] == "P1_PANL" and int(row["layer"]) in TARGET_LAYERS and
                    row["group"] == "all" and row["diagnostic_metric"] == "cosine_distance"]
    target_interaction = [row for row in paired if row["analysis"] == "PANL_minus_PANL_PLUS_1_cross_specific_drift" and
                          int(row["layer"]) in TARGET_LAYERS and row["group"] == "all" and
                          row["diagnostic_metric"] == "cosine_distance"]
    systematic_cross = any(float(row["ci_low"]) > 0 for row in target_cross)
    panl_specific = any(float(row["ci_low"]) > 0 for row in target_interaction)
    if systematic_cross and panl_specific:
        status = "failed"
    elif heuristic_count == 0 and all(float(row["ci_high"]) <= 0 for row in [*target_cross, *target_interaction]):
        status = "passed"
    else:
        status = "caveat"
    return status, {"heuristic_cosine_gt_0_1_count": heuristic_count,
                    "systematic_target_panl_cross_drift": systematic_cross,
                    "target_panl_specific_cross_drift": panl_specific,
                    "natural_activation_reference": natural_status,
                    "decision_rule": "failed only if target PANL cross>same and PANL cross-specific drift>PANL+1 both have CI_low>0"}


def original_pipeline_fields(original: dict[str, Any]) -> dict[str, Any]:
    """Keep the historical implementation decision separate and unchanged."""
    return {"original_pipeline_success": bool(original.get("panl_transfer_supported", False)),
            "original_pipeline_interpretation": original.get("interpretation")}


def _lookup(rows: Sequence[dict[str, Any]], **query: Any) -> dict[str, Any]:
    matches = [row for row in rows if all(str(row.get(key, "")) == str(value) for key, value in query.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one result for {query}, found {len(matches)}")
    return matches[0]


def _plots(
    output: Path, bootstrap: Sequence[dict[str, Any]], contrasts: Sequence[dict[str, Any]],
    margin_conditions: Sequence[dict[str, Any]], activation_summary: Sequence[dict[str, Any]],
    paired_drift: Sequence[dict[str, Any]], sensitivity: Sequence[dict[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"I_from_I": "#0072B2", "I_from_T": "#D55E00", "T_from_I": "#CC79A7", "T_from_T": "#009E73"}
    markers = {"I_from_I": "o", "I_from_T": "s", "T_from_I": "D", "T_from_T": "^"}
    layers = sorted({int(row["layer"]) for row in bootstrap if row.get("layer") not in (None, "")})
    figures = output / "figures"

    def errorbar(ax: Any, rows: Sequence[dict[str, Any]], label: str, color: str, marker: str = "o") -> None:
        part = sorted(rows, key=lambda row: int(row["layer"]))
        x = np.asarray([int(row["layer"]) for row in part])
        y = np.asarray([float(row["mean"]) for row in part])
        lo = np.asarray([float(row["ci_low"]) for row in part])
        hi = np.asarray([float(row["ci_high"]) for row in part])
        ax.errorbar(x, y, yerr=np.vstack([y-lo, hi-y]), color=color, marker=marker, label=label,
                    linewidth=1.5, capsize=3, markeredgecolor="white")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, position in zip(axes, POSITIONS):
        for arm, color in (("same", "#0072B2"), ("cross", "#D55E00")):
            part = [row for row in margin_conditions if row.get("analysis") == "margin_arm_change" and
                    row.get("position") == position and row.get("group") == "all" and row.get("swap_kind") == arm]
            errorbar(ax, part, arm, color)
        ax.axhline(0, color="#333", linewidth=.8); ax.set_title(POSITION_LABELS[position]); ax.set_xticks(layers)
        ax.set_xlabel("Decoder layer")
    axes[0].set_ylabel("Fixed-clean margin change (swap − clean)")
    axes[0].legend(frameon=False)
    fig.suptitle("Corrected fixed-clean margin by swap arm")
    fig.savefig(figures / "corrected_margin_by_layer.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.5), constrained_layout=True)
    panl = [row for row in bootstrap if row.get("analysis") == "paired_effect" and row.get("position") == "P1_PANL" and
            row.get("metric") == "soft_sa" and row.get("group") == "all"]
    contrast = [row for row in contrasts if row.get("contrast") == "PANL_minus_PANL_PLUS_1" and
                row.get("metric") == "soft_sa" and row.get("group") == "all"]
    errorbar(ax, panl, "PANL effect", "#D55E00")
    errorbar(ax, contrast, "PANL − PANL+1", "#0072B2", "s")
    ax.axhline(0, color="#333", linewidth=.8); ax.set_xticks(layers); ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Oriented soft-SA effect"); ax.legend(frameon=False)
    for layer in TARGET_LAYERS: ax.axvspan(layer-.35, layer+.35, color="#F0E442", alpha=.12)
    fig.savefig(figures / "soft_effect_and_position_contrast.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    for diagnostic, filename, ylabel in (("cosine_distance", "cosine_drift_same_vs_cross.png", "Cosine distance"),
                                          ("abs_log_norm_ratio", "norm_drift_same_vs_cross.png", "|log norm ratio|")):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
        for ax, position in zip(axes, POSITIONS):
            for arm, color in (("same", "#0072B2"), ("cross", "#D55E00")):
                part = [row for row in activation_summary if row.get("analysis") == "arm" and
                        row.get("position") == position and row.get("swap_kind") == arm and
                        row.get("diagnostic_metric") == diagnostic]
                errorbar(ax, part, arm, color)
            ax.set_title(POSITION_LABELS[position]); ax.set_xticks(layers); ax.set_xlabel("Decoder layer")
        axes[0].set_ylabel(ylabel); axes[0].legend(frameon=False)
        fig.savefig(figures / filename, dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, layer in zip(axes, TARGET_LAYERS):
        for analysis, color, marker in (("PANL_soft_effect", "#D55E00", "o"),
                                        ("PANL_minus_PANL_PLUS_1_soft_contrast", "#0072B2", "s")):
            part = [row for row in sensitivity if int(row["layer"]) == layer and row["analysis"] == analysis and row["group"] == "all"]
            order = {"full": 0, "abs_delta_median": 1, "abs_delta_p75": 2}; part=sorted(part,key=lambda r:order[r["rule"]])
            x=np.arange(len(part)); y=np.asarray([float(r["mean"]) for r in part]); lo=np.asarray([float(r["ci_low"]) for r in part]); hi=np.asarray([float(r["ci_high"]) for r in part])
            ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),color=color,marker=marker,label=analysis.replace("_"," "),capsize=3)
        ax.axhline(0,color="#333",linewidth=.8); ax.set_xticks(range(3),["full","median","p75"]); ax.set_title(f"L{layer}")
    axes[0].set_ylabel("Soft-SA effect"); axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(figures / "distance_matched_sensitivity.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # Supplemental condition plots: one position per file, four lines, and no CI bands/error bars.
    token_conditions = [row for row in bootstrap if row.get("analysis") == "condition" and row.get("metric") == "first_token_changed"]
    for position in POSITIONS:
        for kind, table, metric, ylabel, stem in (
            ("margin", margin_conditions, "fixed_clean_class_margin_change", "Fixed-clean margin change (swap − clean)", "logit_change_diff"),
            ("token", token_conditions, "first_token_changed", "First-token change rate", "token_change_rate"),
        ):
            fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
            for condition in CONDITIONS:
                part=[r for r in table if r.get("position")==position and r.get("condition")==condition and r.get("metric")==metric]
                part=sorted(part,key=lambda r:int(r["layer"]))
                ax.plot([int(r["layer"]) for r in part],[float(r["mean"]) for r in part],
                        color=colors[condition],marker=markers[condition],linewidth=1.8,label=CONDITION_LABELS[condition])
            if kind == "margin": ax.axhline(0,color="#333",linewidth=.8)
            else: ax.set_ylim(-.02,1.02)
            ax.set_xticks(layers); ax.set_xlabel("Decoder layer"); ax.set_ylabel(ylabel); ax.legend(frameon=False,ncol=4)
            ax.set_title(f"{POSITION_LABELS[position]} — {ylabel}")
            fig.savefig(figures / f"{stem}_{position}.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def _formula_report(source: Path, old_margin_error: bool, fields_sufficient: bool) -> str:
    config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    threshold_registered = any("cosine" in str(key).lower() or "warning" in str(key).lower() for key in config)
    return "\n".join([
        "# Metric formula audit", "",
        "- soft SA: image = same − cross; text = cross − same.",
        "- hard midpoint: image = same − cross; text = cross − same.",
        "- fixed-clean-class margin: image = same − cross; text = same − cross.",
        "- first-token change: image = cross − same; text = cross − same.",
        f"- Historical formal margin side-orientation error confirmed: {old_margin_error}.",
        f"- Trial-level fields sufficient for offline recomputation: {fields_sufficient}.",
        f"- run_config contains a registered cosine/warning criterion: {threshold_registered}.",
        "- `cosine_distance > 0.1` occurs only in analysis code as a descriptive heuristic; the old summary nevertheless promoted any such PANL row into the implementation success gate.",
        "- Audit v2 retains the original pipeline decision verbatim and reports a separate scientific endpoint and OOD robustness status.", "",
    ])


def audit(source_root: Path, output_root: Path, *, repeats: int = 2000, seed: int = 42,
          natural_pairs_per_condition: int = 200, donor_contribution_threshold: float = 0.25,
          resume: bool = False) -> dict[str, Any]:
    source = source_root.resolve(); output = output_root.resolve()
    hashes_before = _source_hashes(source)
    immutable = {"format_version": 2, "analysis": "delayed_sa_activation_swap_audit_v2",
                 "source_root": str(source), "output_root": str(output), "bootstrap": int(repeats), "seed": int(seed),
                 "positions": list(POSITIONS), "target_layers": list(TARGET_LAYERS),
                 "natural_pairs_per_condition": int(natural_pairs_per_condition),
                 "distance_subset_rules": ["full", "abs_delta_median", "abs_delta_p75"],
                 "donor_contribution_threshold": float(donor_contribution_threshold),
                 "cosine_heuristic_threshold": 0.1, "norm_ratio_heuristic_bounds": [0.5, 2.0]}
    immutable["fingerprint"] = canonical_hash(immutable)
    _prepare_output(source, output, immutable, hashes_before, resume=resume)
    trials, validation = _recompute_trials(source)
    bootstrap, contrasts, margin_conditions, paired_data = _statistical_tables(trials, repeats=repeats, seed=seed)
    parity = _parity_gate(source, bootstrap, contrasts)
    activation_summary = _activation_summary(trials, repeats=repeats, seed=seed)
    paired_drift = _activation_paired_contrasts(trials, repeats=repeats, seed=seed)
    natural_reference, natural_pairs, natural_status = _natural_reference(
        source, trials, seed=seed, pairs_per_condition=natural_pairs_per_condition)
    sensitivity, membership = _distance_matched_sensitivity(trials, repeats=repeats, seed=seed)
    donor = _donor_sensitivity(trials, contribution_threshold=donor_contribution_threshold)

    tables = output / "tables"; data = output / "experiment_data"; process = output / "process"; report = output / "report"
    _atomic_csv(tables / "corrected_bootstrap_results.csv", bootstrap)
    _atomic_csv(tables / "corrected_position_contrasts.csv", contrasts)
    _atomic_csv(tables / "corrected_margin_conditions.csv", margin_conditions)
    _atomic_csv(tables / "activation_distance_summary.csv", activation_summary)
    _atomic_csv(tables / "activation_distance_paired_contrasts.csv", paired_drift)
    _atomic_csv(tables / "natural_activation_reference.csv", natural_reference)
    _atomic_csv(tables / "distance_matched_sensitivity.csv", sensitivity)
    _atomic_csv(tables / "donor_sensitivity_corrected.csv", donor)
    _atomic_csv(tables / "logit_change_diff_by_position.csv", [r for r in margin_conditions if r["analysis"] == "condition_margin_change"])
    _atomic_csv(tables / "token_change_rate_by_position.csv", [r for r in bootstrap if r.get("analysis") == "condition" and r.get("metric") == "first_token_changed"])
    atomic_jsonl(data / "recomputed_trial_metrics.jsonl", ({k: v for k, v in row.items() if k != "activation_diagnostics"} for row in trials))
    _atomic_csv(data / "paired_recipient_metrics.csv", paired_data)
    _atomic_csv(data / "distance_matched_membership.csv", membership)
    if natural_pairs:
        atomic_jsonl(data / "natural_activation_pairs.jsonl", natural_pairs)
    else:
        atomic_jsonl(data / "natural_activation_pairs.jsonl", [{"status": "natural_pairwise_reference_unavailable"}])

    original = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    old_margin = _lookup(_read_csv(source / "bootstrap_results.csv"), analysis="paired_effect", position="P1_PANL",
                         layer="14", metric="fixed_clean_class_margin", group="text_side")
    corrected_margin = _lookup(bootstrap, analysis="paired_effect", position="P1_PANL", layer=14,
                               metric="fixed_clean_class_margin", group="text_side")
    old_margin_error = float(old_margin["mean"]) < 0 < float(corrected_margin["mean"])
    (process / "metric_formula_audit.md").write_text(_formula_report(source, old_margin_error, True), encoding="utf-8")

    ood_status, ood_details = classify_ood_status(activation_summary, paired_drift, natural_status=natural_status)
    target_checks=[]
    for layer in TARGET_LAYERS:
        soft_all=_lookup(bootstrap,analysis="paired_effect",position="P1_PANL",layer=layer,metric="soft_sa",group="all")
        soft_i=_lookup(bootstrap,analysis="paired_effect",position="P1_PANL",layer=layer,metric="soft_sa",group="image_side")
        soft_t=_lookup(bootstrap,analysis="paired_effect",position="P1_PANL",layer=layer,metric="soft_sa",group="text_side")
        contrast=_lookup(contrasts,analysis="position_contrast",contrast="PANL_minus_PANL_PLUS_1",layer=layer,metric="soft_sa",group="all")
        reversals=any(bool(row["direction_reversed"]) for row in donor if int(row["layer"])==layer and row["metric"]=="soft_sa")
        met=bool(float(soft_all["ci_low"])>0 and float(soft_i["mean"])>0 and float(soft_t["mean"])>0 and
                 float(contrast["ci_low"])>0 and not reversals)
        target_checks.append({"layer":layer,"soft_effect":soft_all,"image_mean_positive":float(soft_i["mean"])>0,
                              "text_mean_positive":float(soft_t["mean"])>0,"position_contrast":contrast,
                              "lodo_direction_reversal":reversals,"primary_endpoint_met_at_layer":met})
    primary=any(row["primary_endpoint_met_at_layer"] for row in target_checks)
    if not primary:
        interpretation="未确立PANL transferable SA。"
    elif ood_status=="passed":
        interpretation="PANL存在弱但可靠的可转移SA信号。"
    elif ood_status=="caveat":
        interpretation="PANL存在弱但可靠的可转移SA信号，同时保留activation-distance稳健性限制。"
    else:
        interpretation="统计效应成立，但无法排除cross-specific OOD artifact。"
    summary={"status":"complete","audit_fingerprint":immutable["fingerprint"],"source_run_fingerprint":original.get("run_fingerprint"),
             **original_pipeline_fields(original),
             "margin_side_orientation_error_confirmed":old_margin_error,
             "primary_statistical_endpoint_met":primary,"target_checks":target_checks,
             "ood_robustness_status":ood_status,"ood_robustness_details":ood_details,
             "revised_scientific_interpretation":interpretation,"trial_validation":validation,
             "non_margin_parity_gate":parity,"natural_activation_reference_status":natural_status,
             "bootstrap_repeats":repeats,"seed":seed}
    atomic_json(report / "audit_summary.json", summary)
    lines=["# Delayed-SA activation-swap analysis audit v2","",
           f"- Margin side-orientation error confirmed: {old_margin_error}.",
           f"- Primary statistical endpoint met: {primary}.",
           f"- OOD robustness status: {ood_status}.",
           f"- Original pipeline success: {summary['original_pipeline_success']}.",
           f"- Revised interpretation: {interpretation}","","## Target layers",""]
    for check in target_checks:
        margin=_lookup(bootstrap,analysis="paired_effect",position="P1_PANL",layer=check["layer"],metric="fixed_clean_class_margin",group="all")
        lines.append(f"- L{check['layer']}: soft={float(check['soft_effect']['mean']):.6f} "
                     f"CI [{float(check['soft_effect']['ci_low']):.6f}, {float(check['soft_effect']['ci_high']):.6f}]; "
                     f"corrected margin={float(margin['mean']):.6f} CI [{float(margin['ci_low']):.6f}, {float(margin['ci_high']):.6f}]; "
                     f"PANL−PANL+1={float(check['position_contrast']['mean']):.6f} "
                     f"CI [{float(check['position_contrast']['ci_low']):.6f}, {float(check['position_contrast']['ci_high']):.6f}].")
    (report / "audit_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    _plots(output, bootstrap, contrasts, margin_conditions, activation_summary, paired_drift, sensitivity)
    hashes_after=_source_hashes(source)
    if hashes_after != hashes_before:
        raise RuntimeError("original formal result files changed during audit")
    atomic_json(process / "original_immutability_check.json", {"passed":True,"before":hashes_before,"after":hashes_after})
    artifact_files=sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())
    atomic_json(process / "artifact_manifest.json", {"file_count":len(artifact_files),"files":artifact_files})
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description="Offline audit of delayed-SA activation-swap formal results")
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,default=None)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--natural-pairs-per-condition",type=int,default=200)
    parser.add_argument("--donor-contribution-threshold",type=float,default=.25)
    parser.add_argument("--resume",action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args=build_parser().parse_args(argv)
    output=args.output_root or args.source_root/"analysis_audit_v2"
    audit(args.source_root,output,repeats=args.bootstrap,seed=args.seed,
          natural_pairs_per_condition=args.natural_pairs_per_condition,
          donor_contribution_threshold=args.donor_contribution_threshold,resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
