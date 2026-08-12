"""Development-only descriptive comparison of behavior, attribution, and report.

The screen joins existing artifacts strictly by ``case_id``.  It never falls
back to ``item_id`` because different priors/conditions for the same item are
different experimental cases.  The primary cohort additionally requires the
verbal-attribution endpoint to equal the answer selected by the behavioral
measurement.  Endpoint mismatches are retained in a separate, non-primary
cohort.

This is deliberately a descriptive analysis.  In particular,
``A_rank = shared_prediction_oof`` was trained to predict
``V = shared_target_oof`` on other items.  Their OOF association is useful as
a construction check but is not an independent three-layer validation.  No
confirmatory case, causal intervention, or mediation claim enters this module.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from layer_metacognition.hidden_state_store import atomic_write_json, atomic_write_text

from .core import (
    SEED,
    item_cluster_bootstrap,
    read_json,
    sha256_file,
    stable_hash,
    write_jsonl_atomic,
)


FORMAT_VERSION = 1
TRUTH_AUDIT_DIR = "stage3_sa_truth_audit"
ATTRIBUTION_DIR = "10_protocol_shared_attribution_component"
ATTRIBUTION_SOURCE_DIR = "05_protocol_granularity_bridge"
BRIDGE_DIR = "stage3_sa_computational_bridge"
MEASUREMENT_DIR = "01_actual_source_reliance"
DONOR_EXTENSION_DIR = "02_donor_replication_extension"
REPRESENTATION_DIR = "03_reliance_representation_devfit_confirm"
SCREEN_DIR = "05_three_layer_descriptive_screen"

CORE_FIELDS = {
    "B": "b_raw_shared_d_m12",
    "A": "a_rank_shared_prediction_oof",
    "V": "v_shared_target_oof",
}
PAIR_FIELDS = {
    "B_vs_V": (CORE_FIELDS["B"], CORE_FIELDS["V"]),
    "B_vs_A": (CORE_FIELDS["B"], CORE_FIELDS["A"]),
    "A_vs_V": (CORE_FIELDS["A"], CORE_FIELDS["V"]),
}


@dataclass(frozen=True)
class ThreeLayerPaths:
    """Authoritative read-only inputs for the screen."""

    experiment_dir: Path
    attribution_results: Path
    attribution_manifest: Path
    attribution_source_results: Path
    behavior_development: Path
    donor_development: Path
    raw_representation_development: Path
    graded_representation_development: Path
    frozen_measurement_rule: Path
    behavior_confirmatory: Path
    donor_confirmatory: Path
    raw_representation_confirmatory: Path
    graded_representation_confirmatory: Path

    def files(self) -> dict[str, Path]:
        return {
            "attribution_results": self.attribution_results,
            "attribution_manifest": self.attribution_manifest,
            "attribution_source_results": self.attribution_source_results,
            "behavior_development": self.behavior_development,
            "donor_development": self.donor_development,
            "raw_representation_development": self.raw_representation_development,
            "graded_representation_development": self.graded_representation_development,
            "frozen_measurement_rule": self.frozen_measurement_rule,
            "behavior_confirmatory_audit": self.behavior_confirmatory,
            "donor_confirmatory_audit": self.donor_confirmatory,
            "raw_representation_confirmatory_audit": self.raw_representation_confirmatory,
            "graded_representation_confirmatory_audit": self.graded_representation_confirmatory,
        }


@dataclass(frozen=True)
class ThreeLayerPanel:
    rows: tuple[dict[str, Any], ...]
    primary_rows: tuple[dict[str, Any], ...]
    endpoint_mismatch_rows: tuple[dict[str, Any], ...]
    join_audit: dict[str, Any]
    input_provenance: dict[str, Any]
    answer_vocabulary: tuple[str, ...]


def discover_three_layer_paths(experiment_dir: str | Path) -> ThreeLayerPaths:
    root = Path(experiment_dir).resolve()
    attribution = root / TRUTH_AUDIT_DIR / ATTRIBUTION_DIR
    attribution_source = root / TRUTH_AUDIT_DIR / ATTRIBUTION_SOURCE_DIR
    bridge = root / BRIDGE_DIR
    measurement = bridge / MEASUREMENT_DIR
    donor = bridge / DONOR_EXTENSION_DIR
    representation = bridge / REPRESENTATION_DIR
    paths = ThreeLayerPaths(
        experiment_dir=root,
        attribution_results=attribution / "results.jsonl",
        attribution_manifest=attribution / "cohort_manifest.json",
        attribution_source_results=attribution_source / "results.jsonl",
        behavior_development=measurement / "development_analysis.jsonl",
        donor_development=donor / "development_analysis.jsonl",
        raw_representation_development=(
            representation / "raw_choice_coupled" / "development_oof_predictions.jsonl"
        ),
        graded_representation_development=(
            representation / "graded_preregistered" / "development_oof_predictions.jsonl"
        ),
        frozen_measurement_rule=measurement / "frozen_measurement_rule.json",
        behavior_confirmatory=measurement / "confirmatory_analysis.jsonl",
        donor_confirmatory=donor / "confirmatory_analysis.jsonl",
        raw_representation_confirmatory=(
            representation / "raw_choice_coupled" / "confirmatory_frozen_predictions.jsonl"
        ),
        graded_representation_confirmatory=(
            representation / "graded_preregistered" / "confirmatory_frozen_predictions.jsonl"
        ),
    )
    missing = [str(path) for path in paths.files().values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Three-layer inputs are missing: " + ", ".join(missing))
    return paths


def build_input_provenance(paths: ThreeLayerPaths) -> dict[str, Any]:
    files = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.files().items()
    }
    aggregate = stable_hash({name: value["sha256"] for name, value in files.items()})
    return {
        "format_version": FORMAT_VERSION,
        "files": files,
        "input_aggregate_sha256": aggregate,
        "aggregate_definition": "SHA256 of the sorted logical-name to file-SHA256 map",
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"Input JSONL is empty: {path}")
    return rows


def _unique_index(
    rows: Sequence[Mapping[str, Any]],
    name: str,
    *,
    required_split: str | None = None,
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    item_to_case: dict[str, str] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        item_id = str(row.get("item_id", ""))
        if not case_id or not item_id:
            raise ValueError(f"{name} contains a row without case_id/item_id")
        if case_id in output:
            raise ValueError(f"{name} duplicates case_id {case_id}")
        if item_id in item_to_case:
            raise ValueError(
                f"{name} has multiple cases for item {item_id}: "
                f"{item_to_case[item_id]} and {case_id}"
            )
        if required_split is not None and row.get("split") != required_split:
            raise ValueError(
                f"{name} is not strictly {required_split}: {case_id} has split={row.get('split')!r}"
            )
        output[case_id] = row
        item_to_case[item_id] = case_id
    return output


def _latest_completed_index(
    rows: Sequence[Mapping[str, Any]], name: str
) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if not case_id:
            raise ValueError(f"{name} contains a row without case_id")
        latest[case_id] = row
    completed = {
        case_id: row for case_id, row in latest.items() if row.get("status") == "completed"
    }
    if not completed:
        raise ValueError(f"{name} has no latest completed cases")
    return completed


def _finite(row: Mapping[str, Any], key: str, source: str) -> float:
    if row.get(key) is None:
        raise ValueError(f"{source} lacks {key}")
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{source}.{key} is non-finite")
    return value


def load_frozen_raw_calibration(
    path: str | Path,
) -> tuple[dict[str, dict[str, float]], tuple[str, ...], dict[str, str]]:
    payload = read_json(path)
    rule_fingerprint = str(payload.get("rule_fingerprint", ""))
    without_rule = dict(payload)
    without_rule.pop("rule_fingerprint", None)
    if not rule_fingerprint or stable_hash(without_rule) != rule_fingerprint:
        raise ValueError("Frozen measurement rule fingerprint mismatch")
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Frozen measurement rule lacks calibration")
    calibration_fingerprint = str(calibration.get("calibration_fingerprint", ""))
    without_calibration = dict(calibration)
    without_calibration.pop("calibration_fingerprint", None)
    if (
        not calibration_fingerprint
        or stable_hash(without_calibration) != calibration_fingerprint
    ):
        raise ValueError("Frozen raw calibration fingerprint mismatch")
    nuisance = calibration.get("nuisance")
    vocabulary = nuisance.get("answer_vocabulary") if isinstance(nuisance, dict) else None
    if not isinstance(vocabulary, list) or len(set(map(str, vocabulary))) < 2:
        raise ValueError("Frozen calibration lacks answer vocabulary")
    folds: dict[str, dict[str, float]] = {}
    for fold, entry in calibration.get("folds", {}).items():
        replacement = entry.get("methods", {}).get("replacement", {})
        mean = float(replacement.get("raw_mean"))
        sd = float(replacement.get("raw_sd"))
        if not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
            raise ValueError(f"Invalid frozen replacement raw scale for fold {fold}")
        folds[str(int(fold))] = {"mean": mean, "sd": sd}
    if len(folds) < 3:
        raise ValueError("Frozen calibration has fewer than three folds")
    return (
        folds,
        tuple(sorted(set(map(str, vocabulary)))),
        {
            "rule_fingerprint": rule_fingerprint,
            "calibration_fingerprint": calibration_fingerprint,
        },
    )


def _item_case_map(index: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {str(row["item_id"]): case_id for case_id, row in index.items()}


def join_three_layer_rows(
    attribution_rows: Sequence[Mapping[str, Any]],
    attribution_source_rows: Sequence[Mapping[str, Any]],
    behavior_rows: Sequence[Mapping[str, Any]],
    donor_rows: Sequence[Mapping[str, Any]],
    raw_prediction_rows: Sequence[Mapping[str, Any]],
    graded_prediction_rows: Sequence[Mapping[str, Any]],
    *,
    replacement_raw_calibration: Mapping[str, Mapping[str, float]],
    confirmatory_case_ids: Mapping[str, Sequence[str]] | None = None,
    expected_primary_n: int | None = None,
    expected_mismatch_n: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Strictly join the six development sources and audit item-only near matches."""

    attribution = _unique_index(attribution_rows, "attribution")
    source = _latest_completed_index(attribution_source_rows, "attribution source")
    behavior = _unique_index(behavior_rows, "behavior", required_split="development")
    donor = _unique_index(donor_rows, "donor extension", required_split="development")
    raw = _unique_index(
        raw_prediction_rows, "raw representation", required_split="development"
    )
    graded = _unique_index(
        graded_prediction_rows, "graded representation", required_split="development"
    )
    if any(row.get("estimand") != "raw_choice_coupled" for row in raw.values()):
        raise ValueError("Raw representation input contains a different estimand")
    if any(row.get("estimand") != "graded_preregistered" for row in graded.values()):
        raise ValueError("Graded representation input contains a different estimand")
    if any(row.get("status") != "completed" for row in behavior.values()):
        raise ValueError("Behavior development contains a non-completed row")
    if any(row.get("status") != "completed" for row in donor.values()):
        raise ValueError("Donor development contains a non-completed row")
    if any(int(row.get("measurement_method_version", -1)) != 2 for row in behavior.values()):
        raise ValueError("Behavior development is not method v2")
    if any(int(row.get("extension_method_version", -1)) != 3 for row in donor.values()):
        raise ValueError("Donor extension is not method v3")

    indices = {
        "attribution": attribution,
        "attribution_source": source,
        "behavior": behavior,
        "donor_extension": donor,
        "raw_representation": raw,
        "graded_representation": graded,
    }
    common = set.intersection(*(set(index) for index in indices.values()))
    if not common:
        raise ValueError("No strict case_id overlap across three-layer inputs")
    confirmatory = {
        name: {str(case_id) for case_id in values}
        for name, values in (confirmatory_case_ids or {}).items()
    }
    confirmatory_union = set().union(*confirmatory.values()) if confirmatory else set()
    confirmatory_overlap = sorted(common.intersection(confirmatory_union))
    if confirmatory_overlap:
        raise ValueError(
            "Development screen overlaps confirmatory cases: "
            + ", ".join(confirmatory_overlap)
        )

    item_maps = {name: _item_case_map(index) for name, index in indices.items()}
    item_overlap = set.intersection(*(set(values) for values in item_maps.values()))
    common_items = {str(attribution[case_id]["item_id"]) for case_id in common}
    item_only = sorted(item_overlap - common_items, key=lambda value: (int(value) if value.isdigit() else 10**20, value))
    item_only_details = [
        {"item_id": item_id, "case_ids_by_input": {name: values[item_id] for name, values in item_maps.items()}}
        for item_id in item_only
    ]

    output: list[dict[str, Any]] = []
    for case_id in sorted(common):
        parts = {name: index[case_id] for name, index in indices.items()}
        item_ids = {str(row["item_id"]) for row in parts.values()}
        if len(item_ids) != 1:
            raise ValueError(f"case_id {case_id} maps to inconsistent item_ids: {item_ids}")
        folds = {
            int(parts[name]["fold"])
            for name in (
                "attribution",
                "behavior",
                "donor_extension",
                "raw_representation",
                "graded_representation",
            )
        }
        if len(folds) != 1:
            raise ValueError(f"case_id {case_id} maps to inconsistent folds: {folds}")
        fold = folds.pop()
        b = parts["behavior"]
        d = parts["donor_extension"]
        r = parts["raw_representation"]
        g = parts["graded_representation"]
        a = parts["attribution"]
        s = parts["attribution_source"]
        for key in ("answer_star", "answer_star_side"):
            if any(str(row.get(key)) != str(b.get(key)) for row in (d, r, g)):
                raise ValueError(f"case_id {case_id} disagrees on {key}")
        if str(d.get("difficulty")) != str(b.get("difficulty")):
            raise ValueError(f"case_id {case_id} disagrees on difficulty")
        for key in ("prior_strength", "full_margin"):
            if not math.isclose(_finite(b, key, case_id), _finite(d, key, case_id), abs_tol=1e-12):
                raise ValueError(f"case_id {case_id} disagrees on {key}")
        if not math.isclose(
            _finite(b, "reliance_raw_shared", case_id),
            0.5 * (_finite(b, "raw_z_delete", case_id) + _finite(b, "raw_z_replace", case_id)),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(f"case_id {case_id} has an invalid formal raw shared score")
        if not math.isclose(
            _finite(r, "target_shared", case_id),
            _finite(b, "reliance_raw_shared", case_id),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(f"case_id {case_id} raw representation target differs from B")
        calibration = replacement_raw_calibration.get(str(fold))
        if calibration is None:
            raise ValueError(f"Frozen replacement calibration omits fold {fold}")
        replacement_mean = float(calibration["mean"])
        replacement_sd = float(calibration["sd"])
        if not math.isfinite(replacement_mean) or replacement_sd <= 0:
            raise ValueError(f"Invalid replacement calibration for fold {fold}")
        m34_effect = _finite(d, "behavior_replace_imageward_d34_mean", case_id)
        final_answer = str(s.get("final_answer", ""))
        answer_star = str(b.get("answer_star", ""))
        if not final_answer or not answer_star:
            raise ValueError(f"case_id {case_id} lacks an endpoint answer")
        endpoint_matched = final_answer == answer_star
        if endpoint_matched and s.get("final_image") is not None:
            source_side = "image" if bool(s["final_image"]) else "text"
            if source_side != str(b["answer_star_side"]):
                raise ValueError(f"case_id {case_id} matches answer but disagrees on answer side")
        output.append(
            {
                "case_id": case_id,
                "item_id": next(iter(item_ids)),
                "fold": fold,
                "analysis_split": "development",
                "cohort": (
                    "primary_endpoint_matched" if endpoint_matched else "endpoint_mismatch"
                ),
                "endpoint_matched": endpoint_matched,
                "final_answer": final_answer,
                "answer_star": answer_star,
                "answer_star_side": str(b["answer_star_side"]),
                "difficulty": str(b["difficulty"]),
                "prior_strength": _finite(b, "prior_strength", case_id),
                "full_margin": _finite(b, "full_margin", case_id),
                "b_raw_shared_d_m12": _finite(b, "reliance_raw_shared", case_id),
                "b_delete_formal_raw_z": _finite(b, "raw_z_delete", case_id),
                "b_fresh_m34_frozen_raw_z": (m34_effect - replacement_mean) / replacement_sd,
                "b_delete_raw_effect": _finite(b, "behavior_delete_imageward", case_id),
                "b_fresh_m34_raw_effect": m34_effect,
                "a_rank_shared_prediction_oof": _finite(a, "shared_prediction_oof", case_id),
                "v_shared_target_oof": _finite(a, "shared_target_oof", case_id),
                "raw_representation_prediction_shared_diagnostic": _finite(
                    r, "prediction_shared", case_id
                ),
                "graded_representation_prediction_shared_diagnostic": _finite(
                    g, "prediction_shared", case_id
                ),
                "raw_representation_target_shared_audit": _finite(
                    r, "target_shared", case_id
                ),
                "graded_representation_target_shared_diagnostic": _finite(
                    g, "target_shared", case_id
                ),
            }
        )
    primary_n = sum(bool(row["endpoint_matched"]) for row in output)
    mismatch_n = len(output) - primary_n
    if expected_primary_n is not None and primary_n != expected_primary_n:
        raise ValueError(f"Expected {expected_primary_n} endpoint matches, found {primary_n}")
    if expected_mismatch_n is not None and mismatch_n != expected_mismatch_n:
        raise ValueError(f"Expected {expected_mismatch_n} endpoint mismatches, found {mismatch_n}")
    audit = {
        "join_key": "case_id",
        "item_id_fallback_used": False,
        "input_case_counts": {name: len(index) for name, index in indices.items()},
        "strict_case_overlap_n": len(output),
        "item_candidate_overlap_n": len(item_overlap),
        "item_only_nonjoin_n": len(item_only_details),
        "item_only_nonjoins": item_only_details,
        "primary_endpoint_matched_n": primary_n,
        "endpoint_mismatch_n": mismatch_n,
        "confirmatory_case_sources": {name: len(values) for name, values in confirmatory.items()},
        "confirmatory_case_overlap": confirmatory_overlap,
        "development_only": True,
    }
    return sorted(output, key=lambda row: str(row["case_id"])), audit


def load_three_layer_panel(
    experiment_dir: str | Path,
    *,
    expected_primary_n: int | None = 67,
    expected_mismatch_n: int | None = 7,
) -> ThreeLayerPanel:
    paths = discover_three_layer_paths(experiment_dir)
    provenance = build_input_provenance(paths)
    manifest = read_json(paths.attribution_manifest)
    if Path(str(manifest.get("source_results", ""))).resolve() != paths.attribution_source_results.resolve():
        raise ValueError("Attribution manifest points to an unexpected source cohort")
    if str(manifest.get("source_results_sha256", "")) != sha256_file(
        paths.attribution_source_results
    ):
        raise ValueError("Attribution source cohort SHA256 mismatch")
    attribution_rows = _jsonl(paths.attribution_results)
    manifest_cases = {str(value) for value in manifest.get("case_ids", [])}
    result_cases = {str(row.get("case_id", "")) for row in attribution_rows}
    if not manifest_cases or manifest_cases != result_cases:
        raise ValueError("Attribution results do not exactly match their cohort manifest")
    calibration, vocabulary, fingerprints = load_frozen_raw_calibration(
        paths.frozen_measurement_rule
    )
    confirmatory_rows = {
        "behavior": _jsonl(paths.behavior_confirmatory),
        "donor_extension": _jsonl(paths.donor_confirmatory),
        "raw_representation": _jsonl(paths.raw_representation_confirmatory),
        "graded_representation": _jsonl(paths.graded_representation_confirmatory),
    }
    confirmatory_case_ids = {
        name: [str(row["case_id"]) for row in values]
        for name, values in confirmatory_rows.items()
    }
    rows, audit = join_three_layer_rows(
        attribution_rows,
        _jsonl(paths.attribution_source_results),
        _jsonl(paths.behavior_development),
        _jsonl(paths.donor_development),
        _jsonl(paths.raw_representation_development),
        _jsonl(paths.graded_representation_development),
        replacement_raw_calibration=calibration,
        confirmatory_case_ids=confirmatory_case_ids,
        expected_primary_n=expected_primary_n,
        expected_mismatch_n=expected_mismatch_n,
    )
    audit["frozen_calibration_fingerprints"] = fingerprints
    primary = tuple(row for row in rows if row["endpoint_matched"])
    mismatch = tuple(row for row in rows if not row["endpoint_matched"])
    return ThreeLayerPanel(
        rows=tuple(rows),
        primary_rows=primary,
        endpoint_mismatch_rows=mismatch,
        join_audit=audit,
        input_provenance=provenance,
        answer_vocabulary=vocabulary,
    )


def _safe_spearman(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> float:
    if len(rows) < 3:
        return float("nan")
    x = np.asarray([float(row[left]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[right]) for row in rows], dtype=np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _within_side_spearman(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> float:
    if len(rows) < 3:
        return float("nan")
    groups = sorted({str(row["answer_star_side"]) for row in rows})
    left_means = {
        group: float(np.mean([float(row[left]) for row in rows if str(row["answer_star_side"]) == group]))
        for group in groups
    }
    right_means = {
        group: float(np.mean([float(row[right]) for row in rows if str(row["answer_star_side"]) == group]))
        for group in groups
    }
    synthetic = [
        {
            "left": float(row[left]) - left_means[str(row["answer_star_side"])],
            "right": float(row[right]) - right_means[str(row["answer_star_side"])],
        }
        for row in rows
    ]
    return _safe_spearman(synthetic, "left", "right")


def _association(
    rows: Sequence[dict[str, Any]],
    left: str,
    right: str,
    *,
    iterations: int,
    label: str,
    within_side: bool = False,
) -> dict[str, Any]:
    statistic: Callable[[Sequence[Mapping[str, Any]]], float]
    statistic = (
        (lambda sample: _within_side_spearman(sample, left, right))
        if within_side
        else (lambda sample: _safe_spearman(sample, left, right))
    )
    seed = int(stable_hash({"seed": SEED, "label": label})[:8], 16)
    bootstrap = item_cluster_bootstrap(
        rows, statistic, iterations=iterations, seed=seed
    )
    fold_estimates: list[dict[str, Any]] = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        selected = [row for row in rows if int(row["fold"]) == fold]
        value = statistic(selected)
        fold_estimates.append(
            {"fold": fold, "n": len(selected), "spearman": value if math.isfinite(value) else None}
        )
    valid = [row["spearman"] for row in fold_estimates if row["spearman"] is not None]
    return {
        "left": left,
        "right": right,
        "n": len(rows),
        "unique_items": len({str(row["item_id"]) for row in rows}),
        "spearman": bootstrap["estimate"],
        "spearman_item_bootstrap": bootstrap,
        "fold_estimates": fold_estimates,
        "fold_positive_count": sum(float(value) > 0 for value in valid),
        "fold_valid_count": len(valid),
        "within_answer_side_centered": within_side,
    }


def _delta_rho(
    rows: Sequence[dict[str, Any]],
    *,
    b_key: str,
    a_key: str,
    v_key: str,
    iterations: int,
    label: str,
    within_side: bool = False,
) -> dict[str, Any]:
    def statistic(sample: Sequence[Mapping[str, Any]]) -> float:
        correlation = _within_side_spearman if within_side else _safe_spearman
        return correlation(sample, b_key, a_key) - correlation(sample, b_key, v_key)

    seed = int(stable_hash({"seed": SEED, "label": label})[:8], 16)
    bootstrap = item_cluster_bootstrap(
        rows, statistic, iterations=iterations, seed=seed
    )
    return {
        "definition": "rho(B,A) - rho(B,V)",
        "estimate": bootstrap["estimate"],
        "ci95": bootstrap["ci95"],
        "iterations": bootstrap["iterations"],
        "valid": bootstrap["valid"],
        "paired_same_item_resamples": True,
        "within_answer_side_centered": within_side,
    }


def _cross_fitted_residuals(
    rows: Sequence[dict[str, Any]],
    keys: Sequence[str],
    *,
    answer_vocabulary: Sequence[str],
    include_answer_identity: bool,
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = [dict(row) for row in rows]
    folds = sorted({int(row["fold"]) for row in rows})
    vocabulary = tuple(sorted(set(map(str, answer_vocabulary))))
    if len(folds) < 3:
        raise ValueError("Cross-fitted nuisance residualization requires at least three folds")
    if include_answer_identity and len(vocabulary) < 2:
        raise ValueError("Answer-identity sensitivity requires a fixed vocabulary")
    reference = vocabulary[0] if vocabulary else ""
    columns = [
        "intercept",
        "answer_side_image",
        "difficulty_hard",
        "prior_strength",
        "full_margin",
    ]
    if include_answer_identity:
        columns.extend(f"answer_identity[{answer}]" for answer in vocabulary if answer != reference)
    audits: list[dict[str, Any]] = []
    for fold in folds:
        training = [index for index, row in enumerate(rows) if int(row["fold"]) != fold]
        testing = [index for index, row in enumerate(rows) if int(row["fold"]) == fold]
        train_items = {str(rows[index]["item_id"]) for index in training}
        test_items = {str(rows[index]["item_id"]) for index in testing}
        if train_items.intersection(test_items):
            raise RuntimeError(f"Nuisance residualization leaked items in fold {fold}")
        prior = np.asarray([float(rows[index]["prior_strength"]) for index in training])
        margin = np.asarray([float(rows[index]["full_margin"]) for index in training])
        prior_mean, margin_mean = float(prior.mean()), float(margin.mean())
        prior_sd, margin_sd = float(prior.std(ddof=1)), float(margin.std(ddof=1))
        prior_sd = prior_sd if prior_sd > 1e-12 else 1.0
        margin_sd = margin_sd if margin_sd > 1e-12 else 1.0

        def design(indices: Sequence[int]) -> np.ndarray:
            values: list[list[float]] = []
            for index in indices:
                row = rows[index]
                answer = str(row["answer_star"])
                if include_answer_identity and answer not in vocabulary:
                    raise ValueError(f"Answer {answer!r} is outside the frozen vocabulary")
                values.append(
                    [
                        1.0,
                        float(str(row["answer_star_side"]) == "image"),
                        float(str(row["difficulty"]) == "hard"),
                        (float(row["prior_strength"]) - prior_mean) / prior_sd,
                        (float(row["full_margin"]) - margin_mean) / margin_sd,
                        *(
                            [float(answer == value) for value in vocabulary if value != reference]
                            if include_answer_identity
                            else []
                        ),
                    ]
                )
            return np.asarray(values, dtype=np.float64)

        x_train, x_test = design(training), design(testing)
        y_train = np.asarray(
            [[float(rows[index][key]) for key in keys] for index in training], dtype=np.float64
        )
        y_test = np.asarray(
            [[float(rows[index][key]) for key in keys] for index in testing], dtype=np.float64
        )
        beta = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
        design_rank = int(np.linalg.matrix_rank(x_train))
        residuals = y_test - x_test @ beta
        for local, index in enumerate(testing):
            for column, key in enumerate(keys):
                result[index][f"{prefix}{key}"] = float(residuals[local, column])
        audits.append(
            {
                "fold": fold,
                "train_n": len(training),
                "test_n": len(testing),
                "train_item_sha256": stable_hash(sorted(train_items)),
                "test_item_sha256": stable_hash(sorted(test_items)),
                "item_overlap": [],
                "test_used_for_fit": False,
                "columns": columns,
                "design_column_count": int(x_train.shape[1]),
                "design_rank": design_rank,
                "full_column_rank": design_rank == int(x_train.shape[1]),
                "training_answer_levels": sorted(
                    {str(rows[index]["answer_star"]) for index in training}
                ),
                "missing_frozen_answer_levels": sorted(
                    set(vocabulary).difference(
                        {str(rows[index]["answer_star"]) for index in training}
                    )
                )
                if include_answer_identity
                else [],
                "answer_identity_included": include_answer_identity,
                "answer_reference": reference if include_answer_identity else None,
                "training_prior_mean": prior_mean,
                "training_prior_sd": prior_sd,
                "training_full_margin_mean": margin_mean,
                "training_full_margin_sd": margin_sd,
            }
        )
    for row in result:
        for key in keys:
            if f"{prefix}{key}" not in row:
                raise RuntimeError(f"No cross-fitted residual for {row['case_id']} / {key}")
    return result, {
        "cross_fitted": True,
        "fixed_preexisting_folds": folds,
        "columns": columns,
        "answer_identity_included": include_answer_identity,
        "folds": audits,
    }


def _pair_associations(
    rows: Sequence[dict[str, Any]],
    fields: Mapping[str, tuple[str, str]],
    *,
    iterations: int,
    label_prefix: str,
    within_side: bool = False,
) -> dict[str, Any]:
    return {
        name: _association(
            rows,
            left,
            right,
            iterations=iterations,
            label=f"{label_prefix}|{name}",
            within_side=within_side,
        )
        for name, (left, right) in fields.items()
    }


def analyze_three_layer_panel(
    panel: ThreeLayerPanel,
    output_dir: str | Path,
    *,
    bootstrap_iterations: int = 1000,
    config_fingerprint: str | None = None,
) -> dict[str, Any]:
    if bootstrap_iterations < 20:
        raise ValueError("bootstrap_iterations must be at least 20")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    primary = [dict(row) for row in panel.primary_rows]
    if not primary:
        raise ValueError("Primary endpoint-matched cohort is empty")
    fixed, fixed_audit = _cross_fitted_residuals(
        primary,
        tuple(CORE_FIELDS.values()),
        answer_vocabulary=panel.answer_vocabulary,
        include_answer_identity=False,
        prefix="residual_fixed_",
    )
    identity, identity_audit = _cross_fitted_residuals(
        primary,
        tuple(CORE_FIELDS.values()),
        answer_vocabulary=panel.answer_vocabulary,
        include_answer_identity=True,
        prefix="residual_answer_identity_",
    )
    fixed_by_case = {row["case_id"]: row for row in fixed}
    identity_by_case = {row["case_id"]: row for row in identity}
    residual_fixed_fields = {
        name: (f"residual_fixed_{left}", f"residual_fixed_{right}")
        for name, (left, right) in PAIR_FIELDS.items()
    }
    residual_identity_fields = {
        name: (
            f"residual_answer_identity_{left}",
            f"residual_answer_identity_{right}",
        )
        for name, (left, right) in PAIR_FIELDS.items()
    }
    primary_associations = _pair_associations(
        primary,
        PAIR_FIELDS,
        iterations=bootstrap_iterations,
        label_prefix="primary",
    )
    within_side = _pair_associations(
        primary,
        PAIR_FIELDS,
        iterations=bootstrap_iterations,
        label_prefix="within_side",
        within_side=True,
    )
    fixed_associations = _pair_associations(
        fixed,
        residual_fixed_fields,
        iterations=bootstrap_iterations,
        label_prefix="fixed_nuisance",
    )
    identity_associations = _pair_associations(
        identity,
        residual_identity_fields,
        iterations=bootstrap_iterations,
        label_prefix="answer_identity",
    )
    b, a, v = CORE_FIELDS["B"], CORE_FIELDS["A"], CORE_FIELDS["V"]
    fixed_b, fixed_a, fixed_v = (f"residual_fixed_{key}" for key in (b, a, v))
    identity_b, identity_a, identity_v = (
        f"residual_answer_identity_{key}" for key in (b, a, v)
    )
    delta = {
        "primary": _delta_rho(
            primary,
            b_key=b,
            a_key=a,
            v_key=v,
            iterations=bootstrap_iterations,
            label="delta_primary",
        ),
        "within_answer_side_centered": _delta_rho(
            primary,
            b_key=b,
            a_key=a,
            v_key=v,
            iterations=bootstrap_iterations,
            label="delta_within_side",
            within_side=True,
        ),
        "fixed_nuisance_cross_fitted": _delta_rho(
            fixed,
            b_key=fixed_b,
            a_key=fixed_a,
            v_key=fixed_v,
            iterations=bootstrap_iterations,
            label="delta_fixed_nuisance",
        ),
        "answer_identity_sensitivity_cross_fitted": _delta_rho(
            identity,
            b_key=identity_b,
            a_key=identity_a,
            v_key=identity_v,
            iterations=bootstrap_iterations,
            label="delta_answer_identity",
        ),
    }
    robustness_fields = {
        "B_D_vs_A": ("b_delete_formal_raw_z", a),
        "B_D_vs_V": ("b_delete_formal_raw_z", v),
        "B_fresh_M34_vs_A": ("b_fresh_m34_frozen_raw_z", a),
        "B_fresh_M34_vs_V": ("b_fresh_m34_frozen_raw_z", v),
        "B_D_vs_B_fresh_M34": (
            "b_delete_formal_raw_z",
            "b_fresh_m34_frozen_raw_z",
        ),
    }
    robustness = _pair_associations(
        primary,
        robustness_fields,
        iterations=bootstrap_iterations,
        label_prefix="behavior_robustness",
    )
    diagnostic_fields = {
        "raw_prediction_vs_B": (
            "raw_representation_prediction_shared_diagnostic",
            b,
        ),
        "raw_prediction_vs_A": (
            "raw_representation_prediction_shared_diagnostic",
            a,
        ),
        "raw_prediction_vs_V": (
            "raw_representation_prediction_shared_diagnostic",
            v,
        ),
        "graded_prediction_vs_B": (
            "graded_representation_prediction_shared_diagnostic",
            b,
        ),
        "graded_prediction_vs_A": (
            "graded_representation_prediction_shared_diagnostic",
            a,
        ),
        "graded_prediction_vs_V": (
            "graded_representation_prediction_shared_diagnostic",
            v,
        ),
    }
    representation_diagnostics = _pair_associations(
        primary,
        diagnostic_fields,
        iterations=bootstrap_iterations,
        label_prefix="representation_diagnostic",
    )

    output_rows: list[dict[str, Any]] = []
    for row in panel.rows:
        value = dict(row)
        if row["endpoint_matched"]:
            for key, item in fixed_by_case[row["case_id"]].items():
                if key.startswith("residual_fixed_"):
                    value[key] = item
            for key, item in identity_by_case[row["case_id"]].items():
                if key.startswith("residual_answer_identity_"):
                    value[key] = item
        output_rows.append(value)
    manifest = {
        "format_version": FORMAT_VERSION,
        "experiment": "three_layer_development_descriptive_screen",
        "analysis_split": "development_only",
        "join_key": "case_id",
        "item_id_fallback_used": False,
        "primary_definition": "final_answer == answer_star",
        "primary_case_ids": [row["case_id"] for row in panel.primary_rows],
        "endpoint_mismatch_case_ids": [
            row["case_id"] for row in panel.endpoint_mismatch_rows
        ],
        "primary_n": len(panel.primary_rows),
        "endpoint_mismatch_n": len(panel.endpoint_mismatch_rows),
        "field_definitions": {
            "B": "formal method-v2 raw shared score: mean of fold-standardized deletion D and old replacement M12",
            "B_D": "formal method-v2 fold-standardized raw deletion score",
            "B_fresh_M34": "fresh donor3/4 replacement mean transformed with the original frozen fold-specific replacement raw mean/SD; no refit",
            "A_rank": "protocol-shared attribution component OOF Ridge prediction",
            "V": "protocol-shared verbal-attribution target",
            "raw_and_graded_representation_predictions": "diagnostic only; not used to define B or any claim gate",
        },
        "join_audit": panel.join_audit,
        "input_aggregate_sha256": panel.input_provenance["input_aggregate_sha256"],
        "config_fingerprint": config_fingerprint,
    }
    summary: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "title": "Three-layer development-only descriptive screen",
        "status": "complete",
        "classification": "development-only descriptive, noncausal, nonconfirmatory screen",
        "n": len(primary),
        "endpoint_mismatch_n": len(panel.endpoint_mismatch_rows),
        "bootstrap_iterations": bootstrap_iterations,
        "analysis_scope": {
            "development_only": True,
            "confirmatory_overlap_n": len(panel.join_audit["confirmatory_case_overlap"]),
            "confirmatory_overlap": panel.join_audit["confirmatory_case_overlap"],
            "has_confirmatory_three_layer_overlap": False,
            "causal_intervention": False,
            "causal_mediation_authorized": False,
            "behavior_facing": True,
            "internal_attribution_facing": True,
            "report_facing": True,
            "report_facing_only": False,
        },
        "construction_dependence": {
            "A_rank_vs_V_independent": False,
            "reason": (
                "A_rank is an item-OOF prediction trained on the same kind of shared verbal-SA target V; "
                "OOF prevents held-out item fitting leakage but does not make A and V independent constructs"
            ),
            "interpret_A_vs_V_as": "construction/transport check, not independent convergence evidence",
        },
        "primary_estimands": {
            "B": CORE_FIELDS["B"],
            "A_rank": CORE_FIELDS["A"],
            "V": CORE_FIELDS["V"],
        },
        "primary_associations": primary_associations,
        "within_answer_side_centered": within_side,
        "fixed_nuisance_cross_fitted": {
            "covariates": ["answer_side", "difficulty", "prior_strength", "full_margin"],
            "residual_models_fit_on_other_folds_only": True,
            "bootstrap_refits_residual_models": False,
            "fold_audit": fixed_audit,
            "associations": fixed_associations,
        },
        "answer_identity_sensitivity_cross_fitted": {
            "covariates": [
                "answer_side",
                "difficulty",
                "prior_strength",
                "full_margin",
                "answer_identity",
            ],
            "frozen_answer_vocabulary": list(panel.answer_vocabulary),
            "residual_models_fit_on_other_folds_only": True,
            "bootstrap_refits_residual_models": False,
            "primary_inference": False,
            "support_warning": (
                "Sparse frozen answer levels make every training-fold design rank deficient; "
                "this panel is a sensitivity only and its apparent negative B-V association "
                "is not primary evidence."
                if any(not row["full_column_rank"] for row in identity_audit["folds"])
                else "All training-fold designs have full column rank."
            ),
            "fold_audit": identity_audit,
            "associations": identity_associations,
        },
        "paired_delta_rho": delta,
        "behavior_measurement_robustness": {
            "claim_role": "robustness only",
            "fresh_M34_transformation_refit": False,
            "associations": robustness,
        },
        "representation_prediction_diagnostics": {
            "claim_role": "diagnostic only",
            "used_in_primary_B_definition": False,
            "used_for_gate": False,
            "associations": representation_diagnostics,
        },
        "endpoint_mismatch": {
            "role": "separate excluded endpoint cohort; no primary inference",
            "n": len(panel.endpoint_mismatch_rows),
            "cases": [
                {
                    "case_id": row["case_id"],
                    "item_id": row["item_id"],
                    "final_answer": row["final_answer"],
                    "answer_star": row["answer_star"],
                }
                for row in panel.endpoint_mismatch_rows
            ],
        },
        "join_audit": panel.join_audit,
        "input_aggregate_sha256": panel.input_provenance["input_aggregate_sha256"],
        "claim_limit": (
            "These estimates describe development cases only. They cannot establish that actual reliance, "
            "internal attribution, and verbal report are the same computation, and they cannot support a causal or confirmatory claim."
        ),
    }
    write_jsonl_atomic(output / "results.jsonl", output_rows)
    atomic_write_json(output / "cohort_manifest.json", manifest)
    atomic_write_json(output / "summary.json", summary)
    atomic_write_text(output / "summary.md", _summary_markdown(summary))
    return summary


def _format_correlation(value: Mapping[str, Any]) -> str:
    estimate = value.get("spearman")
    ci = value.get("spearman_item_bootstrap", {}).get("ci95", [None, None])
    if estimate is None or not math.isfinite(float(estimate)):
        return "NA"
    return f"{float(estimate):.3f} [{float(ci[0]):.3f}, {float(ci[1]):.3f}]"


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    associations = summary["primary_associations"]
    delta = summary["paired_delta_rho"]["primary"]
    lines = [
        "# Three-layer development-only descriptive screen",
        "",
        f"- Primary endpoint-matched n: `{summary['n']}`",
        f"- Endpoint mismatches (separate): `{summary['endpoint_mismatch_n']}`",
        "- Scope: development-only, descriptive, noncausal, and nonconfirmatory.",
        "- Confirmatory overlap: `0`.",
        "",
        "| Association | Spearman, item-bootstrap 95% CI | Positive folds |",
        "|---|---:|---:|",
    ]
    for name in ("B_vs_V", "B_vs_A", "A_vs_V"):
        value = associations[name]
        lines.append(
            f"| {name} | {_format_correlation(value)} | "
            f"{value['fold_positive_count']}/{value['fold_valid_count']} |"
        )
    lines.extend(
        [
            "",
            f"Paired delta `rho(B,A)-rho(B,V)`: {delta['estimate']:.3f} "
            f"[{delta['ci95'][0]:.3f}, {delta['ci95'][1]:.3f}].",
            "",
            "## Interpretation limits",
            "",
            "`A_rank` is an item-OOF readout trained to predict the same type of verbal-SA target `V`. "
            "Therefore A↔V is constructively non-independent: it is a construction/transport check, not independent convergence evidence.",
            "",
            "The raw/graded reliance-representation predictions are diagnostics only. The screen contains no confirmatory overlap, intervention, mediation, or causal test.",
            "",
            str(summary["claim_limit"]),
            "",
        ]
    )
    return "\n".join(lines)
