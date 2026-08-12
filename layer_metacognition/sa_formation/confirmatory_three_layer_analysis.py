"""CPU-only confirmatory comparison of behavior, attribution, and report.

This module joins the completed Stage-06 attribution panel to the already
frozen Stage-03 confirmatory reliance targets/readouts and the Stage-04 fresh
M34 sensitivity.  Every join is exact on ``case_id`` and is independently
audited on ``item_id``; item-only fallback joins are forbidden.

The resulting associations are descriptive.  In particular, ``A`` is a
frozen readout trained on Stage-10 verbal-attribution targets and ``V`` is a
transformed verbal-attribution target.  Neither quantity is an intervention,
and their association is not evidence of causal mediation.
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

from .core import SEED, sha256_file, stable_hash, write_jsonl_atomic


FORMAT_VERSION = 1
EXPECTED_CONFIRMATORY_N = 76
BRIDGE_DIR = "stage3_sa_computational_bridge"
PANEL_DIR = "06_confirmatory_attribution_panel"
ANALYSIS_DIR = "three_layer_analysis"
REPRESENTATION_DIR = "03_reliance_representation_devfit_confirm"
SENSITIVITY_DIR = "04_reliance_representation_sensitivities"
TRUTH_AUDIT_DIR = "stage3_sa_truth_audit"
ATTRIBUTION_COMPONENT_DIR = "10_protocol_shared_attribution_component"

A_FIELD = "a_core_consensus_prediction"
V_FIELD = "v_frozen_shared_target"
ORIGINAL_B_FIELDS = {
    "raw_choice_coupled": "b_raw_target_shared",
    "graded_preregistered": "b_graded_target_shared",
}
FRESH_B_FIELDS = {
    "raw_choice_coupled": "b_raw_fresh_m34_target_shared",
    "graded_preregistered": "b_graded_fresh_m34_target_shared",
}
PREDICTION_FIELDS = {
    "raw_choice_coupled": "p_raw_frozen_prediction_shared",
    "graded_preregistered": "p_graded_frozen_prediction_shared",
}
ALL_NUMERIC_FIELDS = (
    A_FIELD,
    V_FIELD,
    *ORIGINAL_B_FIELDS.values(),
    *FRESH_B_FIELDS.values(),
    *PREDICTION_FIELDS.values(),
)


@dataclass(frozen=True)
class ConfirmatoryThreeLayerPaths:
    """Authoritative, read-only inputs to the confirmatory join."""

    experiment_dir: Path
    panel_analysis: Path
    panel_summary: Path
    panel_manifest: Path
    panel_artifact_manifest: Path
    panel_frozen_rule: Path
    stage10_manifest: Path
    representation_summary: Path
    representation_authorization: Path
    raw_representation: Path
    graded_representation: Path
    sensitivity_summary: Path
    raw_fresh_m34: Path
    graded_fresh_m34: Path

    def files(self) -> dict[str, Path]:
        return {
            "panel_analysis": self.panel_analysis,
            "panel_summary": self.panel_summary,
            "panel_manifest": self.panel_manifest,
            "panel_artifact_manifest": self.panel_artifact_manifest,
            "panel_frozen_rule": self.panel_frozen_rule,
            "stage10_manifest": self.stage10_manifest,
            "representation_summary": self.representation_summary,
            "representation_authorization": self.representation_authorization,
            "raw_representation": self.raw_representation,
            "graded_representation": self.graded_representation,
            "sensitivity_summary": self.sensitivity_summary,
            "raw_fresh_m34": self.raw_fresh_m34,
            "graded_fresh_m34": self.graded_fresh_m34,
        }


@dataclass(frozen=True)
class ConfirmatoryThreeLayerPanel:
    rows: tuple[dict[str, Any], ...]
    join_audit: dict[str, Any]
    input_provenance: dict[str, Any]
    source_scope: dict[str, Any]


def analysis_root(experiment_dir: str | Path) -> Path:
    return (
        Path(experiment_dir).resolve()
        / BRIDGE_DIR
        / PANEL_DIR
        / ANALYSIS_DIR
    )


def discover_confirmatory_three_layer_paths(
    experiment_dir: str | Path,
) -> ConfirmatoryThreeLayerPaths:
    root = Path(experiment_dir).resolve()
    bridge = root / BRIDGE_DIR
    panel = bridge / PANEL_DIR
    representation = bridge / REPRESENTATION_DIR
    sensitivity = bridge / SENSITIVITY_DIR
    paths = ConfirmatoryThreeLayerPaths(
        experiment_dir=root,
        panel_analysis=panel / "analysis.jsonl",
        panel_summary=panel / "summary.json",
        panel_manifest=panel / "cohort_manifest.json",
        panel_artifact_manifest=panel / "artifact_manifest.json",
        panel_frozen_rule=panel / "frozen_rule.json",
        stage10_manifest=(
            root / TRUTH_AUDIT_DIR / ATTRIBUTION_COMPONENT_DIR / "cohort_manifest.json"
        ),
        representation_summary=representation / "summary.json",
        representation_authorization=representation / "measurement_authorization.json",
        raw_representation=(
            representation
            / "raw_choice_coupled"
            / "confirmatory_frozen_predictions.jsonl"
        ),
        graded_representation=(
            representation
            / "graded_preregistered"
            / "confirmatory_frozen_predictions.jsonl"
        ),
        sensitivity_summary=sensitivity / "summary.json",
        raw_fresh_m34=(
            sensitivity
            / "raw_choice_coupled"
            / "confirmatory_fresh_donor_predictions.jsonl"
        ),
        graded_fresh_m34=(
            sensitivity
            / "graded_preregistered"
            / "confirmatory_fresh_donor_predictions.jsonl"
        ),
    )
    missing = [str(path) for path in paths.files().values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Confirmatory three-layer inputs are not complete. Finish Stage 06 and "
            "run its --analyze-only step first. Missing: " + ", ".join(missing)
        )
    return paths


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def build_confirmatory_input_provenance(
    paths: ConfirmatoryThreeLayerPaths,
) -> dict[str, Any]:
    files = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.files().items()
    }
    digest_map = {name: value["sha256"] for name, value in sorted(files.items())}
    return {
        "format_version": FORMAT_VERSION,
        "files": files,
        "input_aggregate_sha256": stable_hash(digest_map),
        "aggregate_definition": (
            "SHA256 of canonical sorted logical-input-name to file-SHA256 mapping"
        ),
    }


def _unique_index(
    rows: Sequence[Mapping[str, Any]],
    name: str,
    *,
    estimand: str | None = None,
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
                f"{name} maps item_id {item_id} to multiple cases: "
                f"{item_to_case[item_id]} and {case_id}"
            )
        if row.get("fold") is None:
            raise ValueError(f"{name} lacks fold for {case_id}")
        if estimand is not None and row.get("estimand") != estimand:
            raise ValueError(
                f"{name} contains estimand={row.get('estimand')!r}; expected {estimand!r}"
            )
        if estimand is not None and row.get("split") != "confirmatory":
            raise ValueError(f"{name} is not strictly confirmatory for {case_id}")
        output[case_id] = row
        item_to_case[item_id] = case_id
    return output


def _finite(row: Mapping[str, Any], key: str, source: str) -> float:
    if row.get(key) is None:
        raise ValueError(f"{source} lacks {key}")
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{source}.{key} is non-finite")
    return value


def _case_set_differences(
    indices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    reference_name: str,
) -> dict[str, dict[str, list[str]]]:
    reference = set(indices[reference_name])
    return {
        name: {
            "missing_from_source": sorted(reference.difference(index)),
            "extra_in_source": sorted(set(index).difference(reference)),
        }
        for name, index in indices.items()
    }


def join_confirmatory_three_layer_rows(
    panel_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    graded_rows: Sequence[Mapping[str, Any]],
    raw_fresh_rows: Sequence[Mapping[str, Any]],
    graded_fresh_rows: Sequence[Mapping[str, Any]],
    *,
    stage10_item_ids: Sequence[str] = (),
    expected_n: int | None = EXPECTED_CONFIRMATORY_N,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join all sources exactly; never use an item-only fallback."""

    indices: dict[str, dict[str, Mapping[str, Any]]] = {
        "stage06_analysis": _unique_index(panel_rows, "Stage-06 analysis"),
        "stage06_manifest": _unique_index(manifest_rows, "Stage-06 manifest"),
        "stage03_raw": _unique_index(
            raw_rows, "Stage-03 raw", estimand="raw_choice_coupled"
        ),
        "stage03_graded": _unique_index(
            graded_rows, "Stage-03 graded", estimand="graded_preregistered"
        ),
        "stage04_raw_fresh_m34": _unique_index(
            raw_fresh_rows,
            "Stage-04 raw fresh M34",
            estimand="raw_choice_coupled",
        ),
        "stage04_graded_fresh_m34": _unique_index(
            graded_fresh_rows,
            "Stage-04 graded fresh M34",
            estimand="graded_preregistered",
        ),
    }
    differences = _case_set_differences(indices, "stage06_manifest")
    nonexact = {
        name: value
        for name, value in differences.items()
        if value["missing_from_source"] or value["extra_in_source"]
    }
    if nonexact:
        raise ValueError(
            "Confirmatory sources do not have identical case_id sets: "
            + json.dumps(nonexact, ensure_ascii=False, sort_keys=True)
        )
    case_ids = set(indices["stage06_manifest"])
    if expected_n is not None and len(case_ids) != int(expected_n):
        raise ValueError(
            f"Expected {expected_n} exact confirmatory cases, found {len(case_ids)}"
        )

    item_sets = {
        name: {str(row["item_id"]) for row in index.values()}
        for name, index in indices.items()
    }
    reference_items = item_sets["stage06_manifest"]
    item_differences = {
        name: {
            "missing_from_source": sorted(reference_items.difference(values)),
            "extra_in_source": sorted(values.difference(reference_items)),
        }
        for name, values in item_sets.items()
    }
    nonexact_items = {
        name: value
        for name, value in item_differences.items()
        if value["missing_from_source"] or value["extra_in_source"]
    }
    if nonexact_items:
        raise ValueError(
            "Confirmatory sources do not have identical item_id sets: "
            + json.dumps(nonexact_items, ensure_ascii=False, sort_keys=True)
        )

    stage10_overlap = sorted(
        reference_items.intersection(map(str, stage10_item_ids)),
        key=lambda value: (int(value) if value.isdigit() else 10**20, value),
    )
    if stage10_overlap:
        raise ValueError(
            "Stage-06 confirmatory items overlap Stage-10 development items: "
            + ", ".join(stage10_overlap)
        )

    output: list[dict[str, Any]] = []
    for case_id in sorted(case_ids):
        parts = {name: index[case_id] for name, index in indices.items()}
        item_ids = {str(row["item_id"]) for row in parts.values()}
        if len(item_ids) != 1:
            raise ValueError(f"case_id {case_id} maps to inconsistent item_ids: {item_ids}")
        folds = {int(row["fold"]) for row in parts.values()}
        if len(folds) != 1:
            raise ValueError(f"case_id {case_id} maps to inconsistent folds: {folds}")
        manifest = parts["stage06_manifest"]
        answer_star = str(manifest.get("answer_star", ""))
        answer_side = str(manifest.get("answer_star_side", ""))
        if not answer_star or answer_side not in {"text", "image", "other"}:
            raise ValueError(f"Stage-06 manifest has an invalid endpoint for {case_id}")
        for name, row in parts.items():
            if row.get("answer_star") is not None and str(row["answer_star"]) != answer_star:
                raise ValueError(f"{name} disagrees on answer_star for {case_id}")
            if row.get("answer_star_side") is not None and str(row["answer_star_side"]) != answer_side:
                raise ValueError(f"{name} disagrees on answer_star_side for {case_id}")

        raw = parts["stage03_raw"]
        graded = parts["stage03_graded"]
        raw_fresh = parts["stage04_raw_fresh_m34"]
        graded_fresh = parts["stage04_graded_fresh_m34"]
        for name, original, fresh in (
            ("raw", raw, raw_fresh),
            ("graded", graded, graded_fresh),
        ):
            original_prediction = _finite(original, "prediction_shared", case_id)
            fresh_prediction = _finite(fresh, "frozen_prediction_shared", case_id)
            if not math.isclose(
                original_prediction, fresh_prediction, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Stage-04 {name} frozen prediction does not replay Stage-03 for {case_id}"
                )
            if bool(fresh.get("hidden_or_readout_refit")):
                raise ValueError(f"Stage-04 {name} unexpectedly refit hidden/readout for {case_id}")
            if bool(fresh.get("gate_bearing")):
                raise ValueError(f"Stage-04 {name} unexpectedly declares gate-bearing evidence")
            replay_error = _finite(fresh, "original_target_replay_max_abs_error", case_id)
            if replay_error > 1e-10:
                raise ValueError(
                    f"Stage-04 {name} target replay failed for {case_id}: {replay_error}"
                )

        panel = parts["stage06_analysis"]
        output.append(
            {
                "case_id": case_id,
                "item_id": next(iter(item_ids)),
                "fold": next(iter(folds)),
                "analysis_split": "confirmatory",
                "cohort": "exact_confirmatory_join",
                "answer_star": answer_star,
                "answer_star_side": answer_side,
                A_FIELD: _finite(panel, "core_consensus_prediction", case_id),
                V_FIELD: _finite(panel, "frozen_shared_target", case_id),
                ORIGINAL_B_FIELDS["raw_choice_coupled"]: _finite(
                    raw, "target_shared", case_id
                ),
                ORIGINAL_B_FIELDS["graded_preregistered"]: _finite(
                    graded, "target_shared", case_id
                ),
                FRESH_B_FIELDS["raw_choice_coupled"]: _finite(
                    raw_fresh, "fresh_target_shared_d_m34", case_id
                ),
                FRESH_B_FIELDS["graded_preregistered"]: _finite(
                    graded_fresh, "fresh_target_shared_d_m34", case_id
                ),
                PREDICTION_FIELDS["raw_choice_coupled"]: _finite(
                    raw, "prediction_shared", case_id
                ),
                PREDICTION_FIELDS["graded_preregistered"]: _finite(
                    graded, "prediction_shared", case_id
                ),
                "stage04_raw_target_replay_max_abs_error": _finite(
                    raw_fresh, "original_target_replay_max_abs_error", case_id
                ),
                "stage04_graded_target_replay_max_abs_error": _finite(
                    graded_fresh, "original_target_replay_max_abs_error", case_id
                ),
            }
        )

    folds = sorted({int(row["fold"]) for row in output})
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected fixed folds 0-4, found {folds}")
    audit = {
        "join_key": "case_id",
        "item_id_fallback_used": False,
        "strict_case_set_equality": True,
        "strict_item_set_equality": True,
        "per_case_item_id_equality": True,
        "per_case_fold_equality": True,
        "per_case_endpoint_equality": True,
        "input_case_counts": {name: len(index) for name, index in indices.items()},
        "input_unique_item_counts": {
            name: len(values) for name, values in item_sets.items()
        },
        "case_set_differences": differences,
        "item_set_differences": item_differences,
        "joined_n": len(output),
        "unique_items": len(reference_items),
        "fold_counts": {
            str(fold): sum(int(row["fold"]) == fold for row in output)
            for fold in folds
        },
        "stage10_development_item_n": len(set(map(str, stage10_item_ids))),
        "stage10_development_item_overlap": stage10_overlap,
        "stage10_item_isolation_passed": not stage10_overlap,
        "stage03_stage04_prediction_replay_passed": True,
        "stage04_hidden_or_readout_refit": False,
        "stage04_gate_bearing": False,
    }
    return output, audit


def load_confirmatory_three_layer_panel(
    experiment_dir: str | Path,
    *,
    expected_n: int = EXPECTED_CONFIRMATORY_N,
) -> ConfirmatoryThreeLayerPanel:
    paths = discover_confirmatory_three_layer_paths(experiment_dir)
    provenance_before = build_confirmatory_input_provenance(paths)
    panel_summary = _read_json(paths.panel_summary)
    if panel_summary.get("status") != "completed":
        raise ValueError(
            f"Stage-06 panel is not technically complete: {panel_summary.get('status')!r}"
        )
    if not bool(panel_summary.get("technical_gate", {}).get("passed")):
        raise ValueError("Stage-06 technical gate did not pass")
    if bool(panel_summary.get("causal_intervention")):
        raise ValueError("Stage-06 unexpectedly declares a causal intervention")
    if bool(panel_summary.get("causal_mediator_authorized")):
        raise ValueError("Stage-06 unexpectedly authorizes a causal mediator")

    panel_manifest = _read_json(paths.panel_manifest)
    manifest_rows = panel_manifest.get("rows")
    if not isinstance(manifest_rows, list) or not manifest_rows:
        raise ValueError("Stage-06 cohort manifest lacks rows")
    if int(panel_manifest.get("n", -1)) != len(manifest_rows):
        raise ValueError("Stage-06 cohort manifest n does not match its rows")
    if int(panel_summary.get("n", -1)) != len(manifest_rows):
        raise ValueError("Stage-06 summary n does not match its cohort manifest")

    artifact_manifest = _read_json(paths.panel_artifact_manifest)
    artifact_entries = {
        str(entry.get("path")): entry
        for entry in artifact_manifest.get("files", [])
        if isinstance(entry, dict)
    }
    analysis_entry = artifact_entries.get("analysis.jsonl")
    if analysis_entry is None:
        raise ValueError("Stage-06 artifact manifest omits analysis.jsonl")
    if str(analysis_entry.get("sha256")) != sha256_file(paths.panel_analysis):
        raise ValueError("Stage-06 analysis.jsonl checksum disagrees with artifact manifest")
    if str(panel_summary.get("artifact_aggregate_sha256")) != str(
        artifact_manifest.get("aggregate_sha256")
    ):
        raise ValueError("Stage-06 summary disagrees with its artifact aggregate SHA256")

    stage10_manifest = _read_json(paths.stage10_manifest)
    stage10_items = stage10_manifest.get("item_ids")
    if not isinstance(stage10_items, list) or not stage10_items:
        raise ValueError("Stage-10 cohort manifest lacks item_ids")
    representation_summary = _read_json(paths.representation_summary)
    authorization = _read_json(paths.representation_authorization)
    sensitivity_summary = _read_json(paths.sensitivity_summary)
    if representation_summary.get("status") != "completed":
        raise ValueError("Stage-03 representation artifact is not complete")
    if sensitivity_summary.get("status") != "completed":
        raise ValueError("Stage-04 sensitivity artifact is not complete")
    if bool(sensitivity_summary.get("gate_bearing")):
        raise ValueError("Stage-04 sensitivity unexpectedly became gate-bearing")
    if not bool(sensitivity_summary.get("post_hoc")):
        raise ValueError("Stage-04 sensitivity is not marked post-hoc")
    if bool(sensitivity_summary.get("original_03_gate_modified")):
        raise ValueError("Stage-04 sensitivity claims to modify the original Stage-03 gate")

    rows, audit = join_confirmatory_three_layer_rows(
        _read_jsonl(paths.panel_analysis),
        manifest_rows,
        _read_jsonl(paths.raw_representation),
        _read_jsonl(paths.graded_representation),
        _read_jsonl(paths.raw_fresh_m34),
        _read_jsonl(paths.graded_fresh_m34),
        stage10_item_ids=[str(value) for value in stage10_items],
        expected_n=expected_n,
    )
    provenance_after = build_confirmatory_input_provenance(paths)
    if provenance_before["input_aggregate_sha256"] != provenance_after[
        "input_aggregate_sha256"
    ]:
        raise RuntimeError("A confirmatory input changed while the join was being read")
    source_scope = {
        "stage06_status": panel_summary["status"],
        "stage06_technical_gate_passed": True,
        "stage06_rank_gate_passed": bool(
            panel_summary.get("frozen_rank_gate", {}).get("passed")
        ),
        "stage06_coordinate_gate_passed": bool(
            panel_summary.get("frozen_common_coordinate_gate", {}).get("passed")
        ),
        "stage03_causal_mediator_authorized": bool(
            authorization.get("causal_mediator_authorized")
        ),
        "stage04_post_hoc": True,
        "stage04_gate_bearing": False,
        "stage04_original_03_gate_modified": False,
    }
    return ConfirmatoryThreeLayerPanel(
        rows=tuple(rows),
        join_audit=audit,
        input_provenance=provenance_after,
        source_scope=source_scope,
    )


def _safe_spearman(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    *,
    center_by_answer_side: bool = False,
) -> float | None:
    if len(rows) < 3:
        return None
    x = np.asarray([float(row[left]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[right]) for row in rows], dtype=np.float64)
    if center_by_answer_side:
        sides = np.asarray([str(row["answer_star_side"]) for row in rows])
        for side in sorted(set(sides.tolist())):
            selected = sides == side
            x[selected] -= float(np.mean(x[selected]))
            y[selected] -= float(np.mean(y[selected]))
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else None


def _fold_item_cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float | None],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Resample item clusters within each fixed fold."""

    observed = statistic(rows)
    by_fold_item: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        fold = int(row["fold"])
        item = str(row["item_id"])
        by_fold_item.setdefault(fold, {}).setdefault(item, []).append(row)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sample: list[dict[str, Any]] = []
        for fold in sorted(by_fold_item):
            clusters = by_fold_item[fold]
            items = sorted(clusters)
            chosen = rng.choice(items, size=len(items), replace=True)
            for item in chosen:
                sample.extend(clusters[str(item)])
        value = statistic(sample)
        if value is not None and math.isfinite(float(value)):
            samples.append(float(value))
    ci: list[float | None] = [None, None]
    if samples:
        ci = [float(value) for value in np.quantile(samples, [0.025, 0.975])]
    return {
        "estimate": observed,
        "ci95": ci,
        "iterations": int(iterations),
        "valid": len(samples),
        "cluster_unit": "item_id",
        "stratified_by": "fixed_fold",
        "same_item_rows_kept_together": True,
    }


def _association(
    rows: Sequence[dict[str, Any]],
    left: str,
    right: str,
    *,
    iterations: int,
    label: str,
    center_by_answer_side: bool,
) -> dict[str, Any]:
    statistic = lambda sample: _safe_spearman(
        sample,
        left,
        right,
        center_by_answer_side=center_by_answer_side,
    )
    seed = int(stable_hash({"seed": SEED, "label": label})[:8], 16)
    bootstrap = _fold_item_cluster_bootstrap(
        list(rows), statistic, iterations=iterations, seed=seed
    )
    fold_metrics: list[dict[str, Any]] = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        selected = [row for row in rows if int(row["fold"]) == fold]
        fold_metrics.append(
            {
                "fold": fold,
                "n": len(selected),
                "unique_items": len({str(row["item_id"]) for row in selected}),
                "spearman": statistic(selected),
            }
        )
    return {
        "left": left,
        "right": right,
        "n": len(rows),
        "unique_items": len({str(row["item_id"]) for row in rows}),
        "spearman": bootstrap["estimate"],
        "fold_item_cluster_bootstrap": bootstrap,
        "fold_metrics": fold_metrics,
        "positive_fold_count": sum(
            value["spearman"] is not None and float(value["spearman"]) > 0
            for value in fold_metrics
        ),
        "valid_fold_count": sum(value["spearman"] is not None for value in fold_metrics),
        "answer_side_centered": center_by_answer_side,
        "centering_refit_in_each_bootstrap_sample": center_by_answer_side,
    }


def _paired_rho_difference(
    rows: Sequence[dict[str, Any]],
    *,
    b_field: str,
    iterations: int,
    label: str,
    center_by_answer_side: bool,
) -> dict[str, Any]:
    def statistic(sample: Sequence[Mapping[str, Any]]) -> float | None:
        b_a = _safe_spearman(
            sample,
            b_field,
            A_FIELD,
            center_by_answer_side=center_by_answer_side,
        )
        b_v = _safe_spearman(
            sample,
            b_field,
            V_FIELD,
            center_by_answer_side=center_by_answer_side,
        )
        if b_a is None or b_v is None:
            return None
        return float(b_a - b_v)

    seed = int(stable_hash({"seed": SEED, "label": label})[:8], 16)
    bootstrap = _fold_item_cluster_bootstrap(
        list(rows), statistic, iterations=iterations, seed=seed
    )
    return {
        "definition": "rho(B,A) - rho(B,V)",
        "estimate": bootstrap["estimate"],
        "ci95": bootstrap["ci95"],
        "iterations": bootstrap["iterations"],
        "valid": bootstrap["valid"],
        "paired_same_item_resamples": True,
        "cluster_unit": "item_id",
        "stratified_by": "fixed_fold",
        "answer_side_centered": center_by_answer_side,
        "centering_refit_in_each_bootstrap_sample": center_by_answer_side,
    }


def _three_layer_block(
    rows: Sequence[dict[str, Any]],
    *,
    b_field: str,
    iterations: int,
    label: str,
) -> dict[str, Any]:
    def mode(centered: bool) -> dict[str, Any]:
        suffix = "centered" if centered else "uncentered"
        return {
            "associations": {
                "B_vs_A": _association(
                    rows,
                    b_field,
                    A_FIELD,
                    iterations=iterations,
                    label=f"{label}|{suffix}|B_vs_A",
                    center_by_answer_side=centered,
                ),
                "B_vs_V": _association(
                    rows,
                    b_field,
                    V_FIELD,
                    iterations=iterations,
                    label=f"{label}|{suffix}|B_vs_V",
                    center_by_answer_side=centered,
                ),
                "A_vs_V": _association(
                    rows,
                    A_FIELD,
                    V_FIELD,
                    iterations=iterations,
                    label=f"{label}|{suffix}|A_vs_V",
                    center_by_answer_side=centered,
                ),
            },
            "paired_rho_difference": _paired_rho_difference(
                rows,
                b_field=b_field,
                iterations=iterations,
                label=f"{label}|{suffix}|paired_delta",
                center_by_answer_side=centered,
            ),
        }

    return {
        "B_field": b_field,
        "A_field": A_FIELD,
        "V_field": V_FIELD,
        "uncentered": mode(False),
        "answer_side_centered": mode(True),
    }


def _prediction_diagnostics(
    rows: Sequence[dict[str, Any]],
    *,
    estimand: str,
    iterations: int,
) -> dict[str, Any]:
    prediction = PREDICTION_FIELDS[estimand]
    original = ORIGINAL_B_FIELDS[estimand]
    fresh = FRESH_B_FIELDS[estimand]
    block = _three_layer_block(
        rows,
        b_field=prediction,
        iterations=iterations,
        label=f"prediction|{estimand}",
    )
    block.update(
        {
            "claim_role": "diagnostic only; not a validated causal mediator",
            "prediction_vs_original_B": _association(
                rows,
                prediction,
                original,
                iterations=iterations,
                label=f"prediction|{estimand}|vs_original_B",
                center_by_answer_side=False,
            ),
            "prediction_vs_fresh_M34_B": _association(
                rows,
                prediction,
                fresh,
                iterations=iterations,
                label=f"prediction|{estimand}|vs_fresh_B",
                center_by_answer_side=False,
            ),
        }
    )
    return block


def _answer_side_centered_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    means: dict[str, dict[str, float]] = {}
    for field in ALL_NUMERIC_FIELDS:
        means[field] = {}
        for side in sorted({str(row["answer_star_side"]) for row in rows}):
            values = [float(row[field]) for row in rows if row["answer_star_side"] == side]
            means[field][side] = float(np.mean(values))
    output: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        side = str(row["answer_star_side"])
        for field in ALL_NUMERIC_FIELDS:
            value[f"{field}_answer_side_centered"] = float(row[field]) - means[field][side]
        output.append(value)
    return output, means


def analyze_confirmatory_three_layer_panel(
    panel: ConfirmatoryThreeLayerPanel,
    output_dir: str | Path,
    *,
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    if bootstrap_iterations < 20:
        raise ValueError("bootstrap_iterations must be at least 20")
    rows = [dict(row) for row in panel.rows]
    if not rows:
        raise ValueError("Confirmatory three-layer cohort is empty")
    output_rows, side_means = _answer_side_centered_rows(rows)

    primary = {
        estimand: _three_layer_block(
            rows,
            b_field=b_field,
            iterations=bootstrap_iterations,
            label=f"original|{estimand}",
        )
        for estimand, b_field in ORIGINAL_B_FIELDS.items()
    }
    fresh = {
        estimand: _three_layer_block(
            rows,
            b_field=b_field,
            iterations=bootstrap_iterations,
            label=f"fresh_m34|{estimand}",
        )
        for estimand, b_field in FRESH_B_FIELDS.items()
    }
    predictions = {
        estimand: _prediction_diagnostics(
            rows,
            estimand=estimand,
            iterations=bootstrap_iterations,
        )
        for estimand in PREDICTION_FIELDS
    }

    summary: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "title": "Confirmatory three-layer attribution/reliance analysis",
        "status": "complete",
        "classification": "confirmatory exact-join descriptive association; noncausal",
        "n": len(rows),
        "unique_items": len({str(row["item_id"]) for row in rows}),
        "bootstrap_iterations": bootstrap_iterations,
        "analysis_scope": {
            "confirmatory_exact_join": True,
            "causal_intervention": False,
            "causal_mediator_authorized": False,
            "association_is_causal": False,
            "stage04_fresh_m34_gate_bearing": False,
            "stage04_fresh_m34_post_hoc_sensitivity": True,
        },
        "field_definitions": {
            "B_raw": "Stage-03 confirmatory raw_choice_coupled target_shared",
            "B_graded": "Stage-03 confirmatory graded_preregistered target_shared",
            "B_fresh_M34": (
                "Stage-04 post-hoc fresh donor sensitivity target_shared using deletion D "
                "and replacement donors M34; no target/readout refit"
            ),
            "P_B": "Stage-03 frozen reliance representation prediction_shared; diagnostic only",
            "A": "Stage-06 core_consensus_prediction from the frozen Stage-10 attribution readout",
            "V": "Stage-06 frozen_shared_target transformed from verbal-SA protocol scores",
        },
        "primary_original_B": primary,
        "fresh_M34_sensitivity": {
            "post_hoc": True,
            "gate_bearing": False,
            "original_stage03_gate_modified": False,
            "hidden_or_readout_refit": False,
            "estimands": fresh,
        },
        "reliance_representation_prediction_diagnostics": {
            "used_as_primary_B": False,
            "causal_mediator_authorized": False,
            "estimands": predictions,
        },
        "answer_side_centering": {
            "group": "answer_star_side",
            "levels": sorted({str(row["answer_star_side"]) for row in rows}),
            "full_cohort_means_for_results_jsonl": side_means,
            "bootstrap_recomputes_group_means_within_each_resample": True,
        },
        "bootstrap": {
            "resampling_unit": "item_id cluster",
            "stratified_by": "fixed fold",
            "same_resamples_for_each_paired_rho_difference": True,
            "fold_metrics_reported_for_each_association": True,
        },
        "construction_and_causal_limits": {
            "A_and_V_are_observational_readouts": True,
            "A_was_trained_on_V_kind_target": True,
            "A_vs_V_constructively_independent": False,
            "A_vs_V_interpretation": (
                "frozen out-of-sample transport/construction check, not an independent "
                "causal or convergent-measure validation"
            ),
            "B_associations_are_causal": False,
            "reason": (
                "Although B is constructed from behavioral source perturbations, the across-item "
                "B-A/B-V correlations do not intervene on A or V and do not identify mediation."
            ),
        },
        "join_audit": panel.join_audit,
        "source_scope": panel.source_scope,
        "input_provenance": panel.input_provenance,
        "input_aggregate_sha256": panel.input_provenance[
            "input_aggregate_sha256"
        ],
        "claim_limit": (
            "This exact confirmatory join can test whether item-level behavioral reliance "
            "co-varies with a frozen internal-attribution readout and verbal attribution. "
            "It cannot show that A or V causes reliance, that A causally produces V, or that "
            "any of the three is a validated causal mediator."
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output / "results.jsonl", output_rows)
    atomic_write_json(output / "summary.json", summary)
    atomic_write_text(output / "summary.md", _summary_markdown(summary))
    return summary


def _format_metric(metric: Mapping[str, Any]) -> str:
    estimate = metric.get("spearman")
    ci = metric.get("fold_item_cluster_bootstrap", {}).get("ci95", [None, None])
    if estimate is None or ci[0] is None or ci[1] is None:
        return "NA"
    return f"{float(estimate):.3f} [{float(ci[0]):.3f}, {float(ci[1]):.3f}]"


def _format_delta(metric: Mapping[str, Any]) -> str:
    estimate = metric.get("estimate")
    ci = metric.get("ci95", [None, None])
    if estimate is None or ci[0] is None or ci[1] is None:
        return "NA"
    return f"{float(estimate):.3f} [{float(ci[0]):.3f}, {float(ci[1]):.3f}]"


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Confirmatory three-layer attribution/reliance analysis",
        "",
        f"- Exact joined cases/items: `{summary['n']}` / `{summary['unique_items']}`.",
        "- Join key: `case_id`; item-only fallback: `false`.",
        "- Inference: fixed-fold-stratified item-cluster bootstrap.",
        "- Scope: confirmatory descriptive association, not causal mediation.",
        "",
        "## Original Stage-03 behavioral targets",
        "",
        "| Estimand | B↔A rho [95% CI] | B↔V rho [95% CI] | A↔V rho [95% CI] | rho(B,A)-rho(B,V) [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for estimand in ("raw_choice_coupled", "graded_preregistered"):
        block = summary["primary_original_B"][estimand]["uncentered"]
        associations = block["associations"]
        lines.append(
            f"| {estimand} | {_format_metric(associations['B_vs_A'])} | "
            f"{_format_metric(associations['B_vs_V'])} | "
            f"{_format_metric(associations['A_vs_V'])} | "
            f"{_format_delta(block['paired_rho_difference'])} |"
        )
    lines.extend(
        [
            "",
            "## Answer-side-centered analysis",
            "",
            "Group means are recomputed inside every bootstrap resample.",
            "",
            "| Estimand | B↔A rho [95% CI] | B↔V rho [95% CI] | A↔V rho [95% CI] | rho(B,A)-rho(B,V) [95% CI] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for estimand in ("raw_choice_coupled", "graded_preregistered"):
        block = summary["primary_original_B"][estimand]["answer_side_centered"]
        associations = block["associations"]
        lines.append(
            f"| {estimand} | {_format_metric(associations['B_vs_A'])} | "
            f"{_format_metric(associations['B_vs_V'])} | "
            f"{_format_metric(associations['A_vs_V'])} | "
            f"{_format_delta(block['paired_rho_difference'])} |"
        )
    lines.extend(
        [
            "",
            "## Fresh M34 sensitivity",
            "",
            "Stage-04 fresh M34 is post-hoc, non-gate-bearing, and does not alter the original Stage-03 gate.",
            "",
            "| Estimand | B_fresh↔A rho [95% CI] | B_fresh↔V rho [95% CI] | rho(B_fresh,A)-rho(B_fresh,V) [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for estimand in ("raw_choice_coupled", "graded_preregistered"):
        block = summary["fresh_M34_sensitivity"]["estimands"][estimand]["uncentered"]
        associations = block["associations"]
        lines.append(
            f"| {estimand} | {_format_metric(associations['B_vs_A'])} | "
            f"{_format_metric(associations['B_vs_V'])} | "
            f"{_format_delta(block['paired_rho_difference'])} |"
        )
    lines.extend(
        [
            "",
            "## Frozen reliance-representation predictions",
            "",
            "These readout results are diagnostic only; Stage-03 did not authorize either prediction as a causal mediator.",
            "",
            "| Estimand | P_B↔A rho [95% CI] | P_B↔V rho [95% CI] | P_B↔B_original rho [95% CI] | P_B↔B_fresh rho [95% CI] | rho(P_B,A)-rho(P_B,V) [95% CI] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for estimand in ("raw_choice_coupled", "graded_preregistered"):
        block = summary["reliance_representation_prediction_diagnostics"]["estimands"][estimand]
        uncentered = block["uncentered"]
        associations = uncentered["associations"]
        lines.append(
            f"| {estimand} | {_format_metric(associations['B_vs_A'])} | "
            f"{_format_metric(associations['B_vs_V'])} | "
            f"{_format_metric(block['prediction_vs_original_B'])} | "
            f"{_format_metric(block['prediction_vs_fresh_M34_B'])} | "
            f"{_format_delta(uncentered['paired_rho_difference'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limit",
            "",
            "`A` is a frozen readout trained on the same kind of verbal-attribution target used to construct `V`; A↔V is therefore constructively non-independent. Neither A nor V is an intervention.",
            "",
            str(summary["claim_limit"]),
            "",
        ]
    )
    return "\n".join(lines)
