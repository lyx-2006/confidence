"""CPU-only fixed-L18 source-use/attribution representation audit.

This Stage-08 audit deliberately keeps two readouts separate:

* ``U`` is fit only on the 97 method-v2 development items, at the answer-only
  ``post_answer`` position of layer 18.  Alpha selection is nested inside each
  development outer fold and no confirmatory value is used for fitting.
* ``A`` is the byte-frozen Stage-10 protocol-shared attribution direction.

Both directions are then projected over the exact 76-item confirmatory panel.
The cross-context projections are descriptive/OOD diagnostics.  They contain
no activation intervention, are never gate-bearing, and cannot change the
authorization status of Stages 01, 03, 10, or 07.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from layer_metacognition.hidden_state_store import atomic_write_json, atomic_write_text

from .confirmatory_attribution_panel import (
    CORE_PROTOCOL_NAMES,
    POSTQUERY_PROTOCOL_NAME,
)
from .core import (
    RIDGE_ALPHAS,
    SEED,
    atomic_save_npz,
    sha256_file,
    stable_hash,
    write_jsonl_atomic,
)
from .reliance_external_representation import (
    ExplicitNuisanceEncoder,
    fit_hidden_transform,
    fit_target_transform,
    transform_explicit_nuisance,
)
from .reliance_representation import (
    DELETE_KEY,
    REPLACE_KEY,
    fit_ridge,
    load_hidden_panel,
    normalize_measurement_row,
)


FORMAT_VERSION = 1
EXPECTED_DEVELOPMENT_N = 97
EXPECTED_CONFIRMATORY_N = 76
PRIMARY_LAYER = 18
PRIMARY_POSITION = "post_answer"
BRIDGE_DIR = "stage3_sa_computational_bridge"
OUTPUT_DIR = "08_fixed_l18_representation_divergence"
METHOD01_DIR = "01_actual_source_reliance"
REPRESENTATION03_DIR = "03_reliance_representation_devfit_confirm"
PANEL06_DIR = "06_confirmatory_attribution_panel"
TRACING07_DIR = "07_causal_divergence_tracing"
STAGE10_RELATIVE = Path(
    "stage3_sa_truth_audit/10_protocol_shared_attribution_component"
)

CONTEXTS = (
    "answer_only_pre_answer",
    "answer_only_post_answer",
    "postquery_prefix",
    "joint_common9",
    "joint_core_consensus",
)
TARGETS = ("b_raw", "b_graded", "v")
AXES = ("u", "a")


@dataclass(frozen=True)
class Stage08Paths:
    experiment_dir: Path
    development_analysis: Path
    development_manifest: Path
    development_summary: Path
    confirmatory_analysis: Path
    confirmatory_manifest: Path
    confirmatory_summary: Path
    method01_summary: Path
    representation03_summary: Path
    representation03_authorization: Path
    representation03_raw_development: Path
    representation03_raw_confirmatory: Path
    representation03_graded_confirmatory: Path
    panel06_manifest: Path
    panel06_results: Path
    panel06_analysis: Path
    panel06_summary: Path
    panel06_frozen_rule: Path
    panel06_artifact_manifest: Path
    stage10_manifest: Path
    stage10_summary: Path
    stage10_direction_index: Path
    tracing07_summary: Path

    def fixed_files(self) -> dict[str, Path]:
        return {
            "method01_development_analysis": self.development_analysis,
            "method01_development_manifest": self.development_manifest,
            "method01_development_summary": self.development_summary,
            "method01_confirmatory_analysis": self.confirmatory_analysis,
            "method01_confirmatory_manifest": self.confirmatory_manifest,
            "method01_confirmatory_summary": self.confirmatory_summary,
            "method01_summary": self.method01_summary,
            "representation03_summary": self.representation03_summary,
            "representation03_authorization": self.representation03_authorization,
            "representation03_raw_development": self.representation03_raw_development,
            "representation03_raw_confirmatory": self.representation03_raw_confirmatory,
            "representation03_graded_confirmatory": self.representation03_graded_confirmatory,
            "panel06_manifest": self.panel06_manifest,
            "panel06_results": self.panel06_results,
            "panel06_analysis": self.panel06_analysis,
            "panel06_summary": self.panel06_summary,
            "panel06_frozen_rule": self.panel06_frozen_rule,
            "panel06_artifact_manifest": self.panel06_artifact_manifest,
            "stage10_manifest": self.stage10_manifest,
            "stage10_summary": self.stage10_summary,
            "stage10_direction_index": self.stage10_direction_index,
            "tracing07_summary": self.tracing07_summary,
        }


@dataclass
class SourceUseFoldModel:
    fold: int
    alpha: float
    ridge_coefficient: np.ndarray
    ridge_intercept: float
    raw_direction: np.ndarray
    unit_direction: np.ndarray
    hidden_nuisance_beta: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_nuisance_beta: np.ndarray
    target_residual_mean: np.ndarray
    target_residual_scale: np.ndarray
    nuisance_encoder: ExplicitNuisanceEncoder
    train_z_mean: float
    train_z_sd: float
    audit: dict[str, Any]

    def _residual_hidden(
        self, hidden: np.ndarray, row: Mapping[str, Any]
    ) -> np.ndarray:
        value = np.asarray(hidden, dtype=np.float64)
        design = transform_explicit_nuisance(
            [row], [0], self.nuisance_encoder
        )[0]
        residual = value - design @ self.hidden_nuisance_beta
        if residual.shape != self.raw_direction.shape or not np.isfinite(residual).all():
            raise ValueError(f"Fold {self.fold} received an invalid hidden vector")
        return residual

    def project(
        self, hidden: np.ndarray, row: Mapping[str, Any]
    ) -> tuple[float, float]:
        residual = self._residual_hidden(hidden, row)
        standardized = (residual - self.feature_mean) / self.feature_scale
        prediction = float(
            standardized @ self.ridge_coefficient + self.ridge_intercept
        )
        coordinate = float(
            (residual @ self.unit_direction - self.train_z_mean) / self.train_z_sd
        )
        if not math.isfinite(prediction) or not math.isfinite(coordinate):
            raise ValueError(f"Fold {self.fold} produced a non-finite U projection")
        return coordinate, prediction

    def transform_raw_target(self, row: Mapping[str, Any]) -> float:
        design = transform_explicit_nuisance(
            [row], [0], self.nuisance_encoder
        )[0]
        observed = np.asarray(
            [float(row[DELETE_KEY]), float(row[REPLACE_KEY])], dtype=np.float64
        )
        residual = observed - design @ self.target_nuisance_beta
        standardized = (
            residual - self.target_residual_mean
        ) / self.target_residual_scale
        return float(np.mean(standardized))


@dataclass(frozen=True)
class AttributionFoldDirection:
    fold: int
    d_raw: np.ndarray
    d_unit: np.ndarray
    raw_intercept: float
    train_z_mean: float
    train_z_sd: float
    source_file: Path
    source_sha256: str

    def project(self, hidden: np.ndarray) -> tuple[float, float]:
        value = np.asarray(hidden, dtype=np.float64)
        prediction = float(value @ self.d_raw + self.raw_intercept)
        coordinate = float(
            (value @ self.d_unit - self.train_z_mean) / self.train_z_sd
        )
        if not math.isfinite(prediction) or not math.isfinite(coordinate):
            raise ValueError(f"Fold {self.fold} produced a non-finite A projection")
        return coordinate, prediction


@dataclass(frozen=True)
class LoadedStage08Panel:
    paths: Stage08Paths
    development_rows: tuple[dict[str, Any], ...]
    confirmatory_rows: tuple[dict[str, Any], ...]
    confirmatory_sources: dict[str, dict[str, dict[str, Any]]]
    development_hidden_paths: tuple[Path, ...]
    confirmatory_hidden01_paths: tuple[Path, ...]
    confirmatory_hidden06_paths: tuple[Path, ...]
    direction_paths: tuple[Path, ...]
    join_audit: dict[str, Any]
    gate_snapshot: dict[str, Any]

    def all_input_files(self) -> dict[str, Path]:
        values = dict(self.paths.fixed_files())
        for index, path in enumerate(self.direction_paths):
            values[f"stage10_direction_{index}"] = path
        for index, path in enumerate(self.development_hidden_paths):
            values[f"development_hidden_{index:03d}"] = path
        for index, path in enumerate(self.confirmatory_hidden01_paths):
            values[f"confirmatory_hidden01_{index:03d}"] = path
        for index, path in enumerate(self.confirmatory_hidden06_paths):
            values[f"confirmatory_hidden06_{index:03d}"] = path
        return values


def output_root(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir).resolve() / BRIDGE_DIR / OUTPUT_DIR


def discover_stage08_paths(experiment_dir: str | Path) -> Stage08Paths:
    root = Path(experiment_dir).resolve()
    bridge = root / BRIDGE_DIR
    method01 = bridge / METHOD01_DIR
    representation03 = bridge / REPRESENTATION03_DIR
    panel06 = bridge / PANEL06_DIR
    stage10 = root / STAGE10_RELATIVE
    paths = Stage08Paths(
        experiment_dir=root,
        development_analysis=method01 / "development_analysis.jsonl",
        development_manifest=method01 / "development_cohort_manifest.json",
        development_summary=method01 / "development_summary.json",
        confirmatory_analysis=method01 / "confirmatory_analysis.jsonl",
        confirmatory_manifest=method01 / "confirmatory_cohort_manifest.json",
        confirmatory_summary=method01 / "confirmatory_summary.json",
        method01_summary=method01 / "summary.json",
        representation03_summary=representation03 / "summary.json",
        representation03_authorization=(
            representation03 / "measurement_authorization.json"
        ),
        representation03_raw_development=(
            representation03
            / "raw_choice_coupled"
            / "development_oof_predictions.jsonl"
        ),
        representation03_raw_confirmatory=(
            representation03
            / "raw_choice_coupled"
            / "confirmatory_frozen_predictions.jsonl"
        ),
        representation03_graded_confirmatory=(
            representation03
            / "graded_preregistered"
            / "confirmatory_frozen_predictions.jsonl"
        ),
        panel06_manifest=panel06 / "cohort_manifest.json",
        panel06_results=panel06 / "results.jsonl",
        panel06_analysis=panel06 / "analysis.jsonl",
        panel06_summary=panel06 / "summary.json",
        panel06_frozen_rule=panel06 / "frozen_rule.json",
        panel06_artifact_manifest=panel06 / "artifact_manifest.json",
        stage10_manifest=stage10 / "cohort_manifest.json",
        stage10_summary=stage10 / "summary.json",
        stage10_direction_index=stage10 / "directions" / "index.json",
        tracing07_summary=bridge / TRACING07_DIR / "summary.json",
    )
    missing = [str(path) for path in paths.fixed_files().values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Stage08 inputs are incomplete: " + ", ".join(missing))
    return paths


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
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


def _unique_index(
    rows: Sequence[Mapping[str, Any]],
    source: str,
    *,
    expected_split: str | None = None,
    expected_estimand: str | None = None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    item_to_case: dict[str, str] = {}
    for raw in rows:
        row = dict(raw)
        case_id = str(row.get("case_id", ""))
        item_id = str(row.get("item_id", ""))
        if not case_id or not item_id:
            raise ValueError(f"{source} contains a row without case_id/item_id")
        if case_id in output:
            raise ValueError(f"{source} duplicates case_id {case_id}")
        if item_id in item_to_case:
            raise ValueError(
                f"{source} maps item {item_id} to both {item_to_case[item_id]} and {case_id}"
            )
        if row.get("fold") is None:
            raise ValueError(f"{source} lacks fold for {case_id}")
        if expected_split is not None and row.get("split") != expected_split:
            raise ValueError(
                f"{source} contains split={row.get('split')!r}; expected {expected_split!r}"
            )
        if expected_estimand is not None and row.get("estimand") != expected_estimand:
            raise ValueError(
                f"{source} contains estimand={row.get('estimand')!r}; "
                f"expected {expected_estimand!r}"
            )
        output[case_id] = row
        item_to_case[item_id] = case_id
    return output


def _resolve_relative_file(root: Path, raw: Any, source: str) -> Path:
    if raw is None:
        raise ValueError(f"{source} omits a hidden file")
    path = Path(str(raw))
    if not path.is_absolute():
        direct = (root / path).resolve()
        hidden_subdirectory = (root / "hidden" / path).resolve()
        path = direct if direct.is_file() else hidden_subdirectory
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{source} hidden file escapes its read-only stage root: {path}")
    return path


def _completed_analysis_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    completed = [row for row in rows if row.get("status", "completed") == "completed"]
    if len(completed) != len(rows):
        raise ValueError(f"{path} contains non-completed rows")
    return completed


def load_stage08_panel(
    experiment_dir: str | Path,
    *,
    expected_development_n: int = EXPECTED_DEVELOPMENT_N,
    expected_confirmatory_n: int = EXPECTED_CONFIRMATORY_N,
) -> LoadedStage08Panel:
    paths = discover_stage08_paths(experiment_dir)
    method01_root = paths.development_analysis.parent.resolve()
    panel06_root = paths.panel06_results.parent.resolve()

    development_rows = [
        normalize_measurement_row(row)
        for row in _completed_analysis_rows(paths.development_analysis)
    ]
    confirmatory_rows = [
        normalize_measurement_row(row)
        for row in _completed_analysis_rows(paths.confirmatory_analysis)
    ]
    development = _unique_index(
        development_rows, "method01_development", expected_split="development"
    )
    confirmatory = _unique_index(
        confirmatory_rows, "method01_confirmatory", expected_split="confirmatory"
    )
    if len(development) != expected_development_n:
        raise ValueError(
            f"Expected {expected_development_n} development items, found {len(development)}"
        )
    if len(confirmatory) != expected_confirmatory_n:
        raise ValueError(
            f"Expected {expected_confirmatory_n} confirmatory items, found {len(confirmatory)}"
        )

    stage03_dev = _unique_index(
        _read_jsonl(paths.representation03_raw_development),
        "stage03_raw_development",
        expected_split="development",
        expected_estimand="raw_choice_coupled",
    )
    stage03_raw = _unique_index(
        _read_jsonl(paths.representation03_raw_confirmatory),
        "stage03_raw_confirmatory",
        expected_split="confirmatory",
        expected_estimand="raw_choice_coupled",
    )
    stage03_graded = _unique_index(
        _read_jsonl(paths.representation03_graded_confirmatory),
        "stage03_graded_confirmatory",
        expected_split="confirmatory",
        expected_estimand="graded_preregistered",
    )
    panel_manifest_value = _read_json(paths.panel06_manifest)
    panel_manifest_rows = panel_manifest_value.get("rows")
    if not isinstance(panel_manifest_rows, list):
        raise ValueError("Stage06 cohort manifest lacks rows")
    panel_manifest = _unique_index(panel_manifest_rows, "stage06_manifest")
    panel_results = _unique_index(
        _completed_analysis_rows(paths.panel06_results), "stage06_results"
    )
    panel_analysis = _unique_index(
        _read_jsonl(paths.panel06_analysis), "stage06_analysis"
    )

    development_sources = {
        "method01_development": development,
        "stage03_raw_development": stage03_dev,
    }
    development_case_sets = {name: set(index) for name, index in development_sources.items()}
    if len({frozenset(values) for values in development_case_sets.values()}) != 1:
        raise ValueError("Development inputs do not have identical case_id sets")

    confirmatory_sources = {
        "method01_confirmatory": confirmatory,
        "stage03_raw_confirmatory": stage03_raw,
        "stage03_graded_confirmatory": stage03_graded,
        "stage06_manifest": panel_manifest,
        "stage06_results": panel_results,
        "stage06_analysis": panel_analysis,
    }
    case_sets = {name: set(index) for name, index in confirmatory_sources.items()}
    reference_cases = set(confirmatory)
    differences = {
        name: {
            "missing": sorted(reference_cases.difference(values)),
            "extra": sorted(values.difference(reference_cases)),
        }
        for name, values in case_sets.items()
    }
    nonexact = {
        name: value
        for name, value in differences.items()
        if value["missing"] or value["extra"]
    }
    if nonexact:
        raise ValueError(
            "Confirmatory inputs do not have identical case_id sets: "
            + json.dumps(nonexact, sort_keys=True)
        )

    for case_id in sorted(reference_cases):
        rows = [index[case_id] for index in confirmatory_sources.values()]
        item_ids = {str(row["item_id"]) for row in rows}
        folds = {int(row["fold"]) for row in rows}
        answers = {
            str(row["answer_star"])
            for row in rows
            if row.get("answer_star") is not None
        }
        sides = {
            str(row["answer_star_side"])
            for row in rows
            if row.get("answer_star_side") is not None
        }
        if len(item_ids) != 1 or len(folds) != 1 or len(answers) != 1 or len(sides) != 1:
            raise ValueError(f"Confirmatory exact join disagreement for {case_id}")
        if next(iter(sides)) not in {"text", "image", "other"}:
            raise ValueError(f"Invalid answer side for {case_id}: {sides}")

    dev_items = {str(row["item_id"]) for row in development.values()}
    confirm_items = {str(row["item_id"]) for row in confirmatory.values()}
    overlap = sorted(dev_items.intersection(confirm_items))
    if overlap:
        raise ValueError(f"Development/confirmatory item leakage: {overlap}")
    for name, index in {
        **development_sources,
        **confirmatory_sources,
    }.items():
        folds = sorted({int(row["fold"]) for row in index.values()})
        if folds != [0, 1, 2, 3, 4]:
            raise ValueError(f"{name} does not contain fixed folds 0-4: {folds}")

    ordered_development = tuple(
        development[case_id] for case_id in sorted(development)
    )
    ordered_confirmatory = tuple(
        confirmatory[case_id] for case_id in sorted(confirmatory)
    )
    development_hidden = tuple(
        _resolve_relative_file(
            method01_root, row.get("hidden_file"), f"development {row['case_id']}"
        )
        for row in ordered_development
    )
    confirmatory_hidden01 = tuple(
        _resolve_relative_file(
            method01_root, row.get("hidden_file"), f"confirmatory {row['case_id']}"
        )
        for row in ordered_confirmatory
    )
    confirmatory_hidden06 = tuple(
        _resolve_relative_file(
            panel06_root,
            panel_results[str(row["case_id"])].get("hidden_file"),
            f"stage06 {row['case_id']}",
        )
        for row in ordered_confirmatory
    )
    for path, row in zip(confirmatory_hidden06, ordered_confirmatory):
        expected_sha = panel_results[str(row["case_id"])].get("hidden_sha256")
        if expected_sha and sha256_file(path) != str(expected_sha):
            raise ValueError(f"Stage06 hidden checksum mismatch: {path}")

    direction_index = _read_json(paths.stage10_direction_index)
    direction_entries = direction_index.get("folds")
    if not isinstance(direction_entries, list) or len(direction_entries) != 5:
        raise ValueError("Stage10 direction index must contain exactly five folds")
    by_direction_fold = {int(entry["fold"]): entry for entry in direction_entries}
    if sorted(by_direction_fold) != [0, 1, 2, 3, 4]:
        raise ValueError("Stage10 direction index does not contain folds 0-4")
    direction_paths = tuple(
        (paths.stage10_direction_index.parent / str(by_direction_fold[fold]["file"])).resolve()
        for fold in range(5)
    )
    if any(not path.is_file() for path in direction_paths):
        raise FileNotFoundError("One or more Stage10 frozen directions are missing")
    frozen_rule = _read_json(paths.panel06_frozen_rule)
    frozen_by_fold = {
        int(entry["fold"]): entry for entry in frozen_rule.get("folds", [])
    }
    for fold, path in enumerate(direction_paths):
        expected_sha = frozen_by_fold.get(fold, {}).get("source_sha256")
        if expected_sha is None or sha256_file(path) != str(expected_sha):
            raise ValueError(f"Stage10/Stage06 frozen direction disagreement for fold {fold}")

    stage10_manifest = _read_json(paths.stage10_manifest)
    stage10_items = {str(value) for value in stage10_manifest.get("item_ids", [])}
    stage10_confirm_overlap = sorted(stage10_items.intersection(confirm_items))
    if stage10_confirm_overlap:
        raise ValueError(
            f"Stage10 development items overlap Stage08 confirmatory items: {stage10_confirm_overlap}"
        )

    method01_summary = _read_json(paths.method01_summary)
    representation03_summary = _read_json(paths.representation03_summary)
    authorization = _read_json(paths.representation03_authorization)
    panel06_summary = _read_json(paths.panel06_summary)
    stage10_summary = _read_json(paths.stage10_summary)
    tracing07_summary = _read_json(paths.tracing07_summary)
    if bool(authorization.get("causal_mediator_authorized")):
        raise ValueError("Stage03 unexpectedly authorizes a causal mediator")
    if bool(panel06_summary.get("causal_intervention")) or bool(
        panel06_summary.get("causal_mediator_authorized")
    ):
        raise ValueError("Stage06 unexpectedly declares causal evidence")
    if tracing07_summary.get("status") != "skipped_by_gate":
        raise ValueError("Stage07 is not frozen in its expected skipped-by-gate state")
    answer_vocabulary = (
        representation03_summary.get("measurement_authorization", {}).get(
            "answer_vocabulary"
        )
    )
    if not isinstance(answer_vocabulary, list) or len(answer_vocabulary) < 2:
        raise ValueError("Stage03 summary lacks its frozen answer vocabulary")
    observed_answers = {
        str(row["answer_star"]) for row in [*development.values(), *confirmatory.values()]
    }
    if not observed_answers.issubset({str(value) for value in answer_vocabulary}):
        raise ValueError("Observed method-v2 answers escape the frozen Stage03 vocabulary")

    join_audit = {
        "development_n": len(ordered_development),
        "confirmatory_n": len(ordered_confirmatory),
        "development_unique_items": len(dev_items),
        "confirmatory_unique_items": len(confirm_items),
        "development_confirmatory_item_overlap": overlap,
        "strict_case_id_join": True,
        "item_id_fallback_used": False,
        "per_case_item_fold_endpoint_equality": True,
        "case_set_differences": differences,
        "stage10_confirmatory_item_overlap": stage10_confirm_overlap,
        "fold_counts": {
            str(fold): sum(int(row["fold"]) == fold for row in ordered_confirmatory)
            for fold in range(5)
        },
    }
    gate_snapshot = {
        "stage01_summary_sha256": sha256_file(paths.method01_summary),
        "stage03_summary_sha256": sha256_file(paths.representation03_summary),
        "stage03_authorization_sha256": sha256_file(paths.representation03_authorization),
        "stage03_causal_mediator_authorized": False,
        "stage03_raw_readout_allowed": bool(
            representation03_summary.get("measurement_authorization", {}).get(
                "raw_readout_allowed"
            )
        ),
        "stage03_frozen_answer_vocabulary": [
            str(value) for value in answer_vocabulary
        ],
        "stage10_summary_sha256": sha256_file(paths.stage10_summary),
        "stage10_rank_gate_passed": bool(
            stage10_summary.get("rank_gate", {}).get("passed")
        ),
        "stage10_original_coordinate_gate_passed": bool(
            stage10_summary.get("coordinate_gate", {}).get("passed")
        ),
        "stage06_summary_sha256": sha256_file(paths.panel06_summary),
        "stage06_rank_gate_passed": bool(
            panel06_summary.get("frozen_rank_gate", {}).get("passed")
        ),
        "stage06_common_coordinate_gate_passed": bool(
            panel06_summary.get("frozen_common_coordinate_gate", {}).get("passed")
        ),
        "stage06_postquery_report_transfer_passed": bool(
            panel06_summary.get("postquery_report_transfer", {}).get("passed")
        ),
        "stage07_summary_sha256": sha256_file(paths.tracing07_summary),
        "stage07_status": tracing07_summary.get("status"),
        "stage08_changes_any_prior_gate": False,
    }
    return LoadedStage08Panel(
        paths=paths,
        development_rows=ordered_development,
        confirmatory_rows=ordered_confirmatory,
        confirmatory_sources=confirmatory_sources,
        development_hidden_paths=development_hidden,
        confirmatory_hidden01_paths=confirmatory_hidden01,
        confirmatory_hidden06_paths=confirmatory_hidden06,
        direction_paths=direction_paths,
        join_audit=join_audit,
        gate_snapshot=gate_snapshot,
    )


def build_input_provenance(panel: LoadedStage08Panel) -> dict[str, Any]:
    files = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for name, path in panel.all_input_files().items()
    }
    digest_map = {name: value["sha256"] for name, value in sorted(files.items())}
    return {
        "format_version": FORMAT_VERSION,
        "read_only_inputs": True,
        "files": files,
        "input_aggregate_sha256": stable_hash(digest_map),
        "aggregate_definition": (
            "SHA256 of canonical logical-input-name to file-SHA256 mapping"
        ),
    }


def _load_l18_answer_hidden(row: Mapping[str, Any], root: Path) -> dict[str, np.ndarray]:
    panel = load_hidden_panel(
        row,
        hidden_root=root,
        layers=(PRIMARY_LAYER,),
        positions=("pre_answer", "post_answer"),
    )
    by_name = {cell.position: value for cell, value in panel.items()}
    return {
        "answer_only_pre_answer": by_name["pre_answer"],
        "answer_only_post_answer": by_name["post_answer"],
    }


def _load_stage06_hidden(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"protocols", "hidden", "layer"}
        if not required.issubset(payload.files):
            raise ValueError(f"Stage06 hidden file lacks {sorted(required)}: {path}")
        protocols = [str(value) for value in payload["protocols"].tolist()]
        hidden = np.asarray(payload["hidden"], dtype=np.float64)
        layer = int(payload["layer"])
    if layer != PRIMARY_LAYER or hidden.ndim != 2 or hidden.shape[0] != len(protocols):
        raise ValueError(f"Invalid Stage06 hidden panel: {path}")
    if not np.isfinite(hidden).all() or len(set(protocols)) != len(protocols):
        raise ValueError(f"Non-finite or duplicate Stage06 hidden panel: {path}")
    missing = sorted(
        set(CORE_PROTOCOL_NAMES + (POSTQUERY_PROTOCOL_NAME,)).difference(protocols)
    )
    if missing:
        raise ValueError(f"Stage06 hidden panel misses protocols {missing}: {path}")
    core = np.stack([hidden[protocols.index(name)] for name in CORE_PROTOCOL_NAMES])
    return {
        "postquery_prefix": hidden[protocols.index(POSTQUERY_PROTOCOL_NAME)].copy(),
        "joint_common9": hidden[protocols.index(CORE_PROTOCOL_NAMES[0])].copy(),
        "joint_core_consensus": core.mean(axis=0),
    }


def _inner_select_alpha_fixed_l18(
    rows: Sequence[Mapping[str, Any]],
    hidden: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
    alphas: Sequence[float],
    answer_vocabulary: Sequence[str],
) -> tuple[float, dict[str, Any]]:
    outer_train = folds != int(outer_fold)
    losses = {float(alpha): [0.0, 0] for alpha in alphas}
    split_audit: list[dict[str, Any]] = []
    for validation_fold in sorted(set(folds[outer_train].tolist())):
        inner_train = np.flatnonzero(outer_train & (folds != validation_fold))
        inner_valid = np.flatnonzero(outer_train & (folds == validation_fold))
        target_fit = fit_target_transform(
            rows,
            inner_train,
            estimand="raw_choice_coupled",
            answer_vocabulary=answer_vocabulary,
        )
        train_targets, x_train = target_fit.apply(rows, inner_train)
        valid_targets, x_valid = target_fit.apply(rows, inner_valid)
        hidden_fit = fit_hidden_transform(
            hidden,
            inner_train,
            x_train,
            estimand="raw_choice_coupled",
        )
        h_train = hidden_fit.apply(hidden, inner_train, x_train)
        h_valid = hidden_fit.apply(hidden, inner_valid, x_valid)
        for alpha in alphas:
            fitted = fit_ridge(
                h_train, train_targets["shared"], float(alpha)
            )
            error = fitted.predict(h_valid) - valid_targets["shared"]
            losses[float(alpha)][0] += float(error @ error)
            losses[float(alpha)][1] += len(error)
        train_items = {str(rows[index]["item_id"]) for index in inner_train}
        valid_items = {str(rows[index]["item_id"]) for index in inner_valid}
        overlap = sorted(train_items.intersection(valid_items))
        if overlap:
            raise RuntimeError(f"Inner-fold item leakage: {overlap}")
        split_audit.append(
            {
                "validation_fold": int(validation_fold),
                "train_folds": sorted(set(folds[inner_train].tolist())),
                "validation_folds": sorted(set(folds[inner_valid].tolist())),
                "train_n": len(inner_train),
                "validation_n": len(inner_valid),
                "item_overlap": overlap,
            }
        )
    mean_mse = {
        str(alpha): float(total / count)
        for alpha, (total, count) in losses.items()
        if count
    }
    if len(mean_mse) != len(alphas):
        raise RuntimeError("Nested alpha selection did not evaluate every candidate")
    selected = min(
        (float(alpha) for alpha in alphas),
        key=lambda alpha: (mean_mse[str(alpha)], alpha),
    )
    return selected, {"splits": split_audit, "pooled_mse": mean_mse}


def fit_source_use_directions(
    rows: Sequence[dict[str, Any]],
    hidden: np.ndarray,
    output_dir: Path,
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    answer_vocabulary: Sequence[str] | None = None,
) -> tuple[dict[int, SourceUseFoldModel], list[dict[str, Any]], dict[str, Any]]:
    values = np.asarray(hidden, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(rows) or not np.isfinite(values).all():
        raise ValueError("Development hidden must be finite [items, hidden]")
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    if sorted(set(folds.tolist())) != [0, 1, 2, 3, 4]:
        raise ValueError("Source-use fitting requires fixed folds 0-4")
    vocabulary = tuple(
        sorted(
            {str(row["answer_star"]) for row in rows}
            if answer_vocabulary is None
            else {str(value) for value in answer_vocabulary}
        )
    )
    models: dict[int, SourceUseFoldModel] = {}
    oof_rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        train_items = {str(rows[index]["item_id"]) for index in train}
        test_items = {str(rows[index]["item_id"]) for index in test}
        overlap = sorted(train_items.intersection(test_items))
        if overlap:
            raise RuntimeError(f"Outer-fold item leakage in fold {fold}: {overlap}")
        alpha, inner = _inner_select_alpha_fixed_l18(
            rows, values, folds, fold, alphas, vocabulary
        )
        target_fit = fit_target_transform(
            rows,
            train,
            estimand="raw_choice_coupled",
            answer_vocabulary=vocabulary,
        )
        train_targets, x_train = target_fit.apply(rows, train)
        test_targets, x_test = target_fit.apply(rows, test)
        hidden_fit = fit_hidden_transform(
            values, train, x_train, estimand="raw_choice_coupled"
        )
        h_train = hidden_fit.apply(values, train, x_train)
        fitted = fit_ridge(h_train, train_targets["shared"], alpha)
        raw_direction = fitted.coefficient / hidden_fit.feature_scale
        norm = float(np.linalg.norm(raw_direction))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError(f"Degenerate U direction in fold {fold}")
        unit_direction = raw_direction / norm
        train_residual = h_train * hidden_fit.feature_scale + hidden_fit.feature_mean
        correlation = float(
            pearsonr(train_residual @ unit_direction, train_targets["shared"]).statistic
        )
        if correlation < 0:
            unit_direction = -unit_direction
        train_z = train_residual @ unit_direction
        train_z_mean = float(np.mean(train_z))
        train_z_sd = float(np.std(train_z, ddof=1))
        if not math.isfinite(train_z_sd) or train_z_sd <= 1e-12:
            raise RuntimeError(f"Invalid U coordinate scale in fold {fold}")
        audit = {
            "fold": fold,
            "train_items": sorted(train_items),
            "heldout_items": sorted(test_items),
            "train_n": len(train),
            "heldout_n": len(test),
            "item_overlap": overlap,
            "heldout_fold_excluded_from_fit": True,
            "selected_alpha": alpha,
            "alpha_candidates": [float(value) for value in alphas],
            "alpha_selection_scope": "nested development folds only",
            "inner_cv": inner,
            "train_projection_target_pearson": correlation,
            "train_z_mean": train_z_mean,
            "train_z_sd": train_z_sd,
        }
        model = SourceUseFoldModel(
            fold=fold,
            alpha=alpha,
            ridge_coefficient=fitted.coefficient,
            ridge_intercept=float(fitted.intercept),
            raw_direction=raw_direction,
            unit_direction=unit_direction,
            hidden_nuisance_beta=hidden_fit.nuisance_beta,
            feature_mean=hidden_fit.feature_mean,
            feature_scale=hidden_fit.feature_scale,
            target_nuisance_beta=target_fit.nuisance_beta,
            target_residual_mean=target_fit.target_mean,
            target_residual_scale=target_fit.target_scale,
            nuisance_encoder=target_fit.encoder,
            train_z_mean=train_z_mean,
            train_z_sd=train_z_sd,
            audit=audit,
        )
        models[fold] = model
        filename = f"fold_{fold}_layer_18_post_answer_raw_choice_coupled.npz"
        atomic_save_npz(
            output_dir / filename,
            alpha=np.asarray(alpha),
            ridge_coefficient=fitted.coefficient,
            ridge_intercept=np.asarray(float(fitted.intercept)),
            raw_direction=raw_direction,
            unit_direction=unit_direction,
            hidden_nuisance_beta=hidden_fit.nuisance_beta,
            feature_mean=hidden_fit.feature_mean,
            feature_scale=hidden_fit.feature_scale,
            target_nuisance_beta=target_fit.nuisance_beta,
            target_residual_mean=target_fit.target_mean,
            target_residual_scale=target_fit.target_scale,
            train_z_mean=np.asarray(train_z_mean),
            train_z_sd=np.asarray(train_z_sd),
        )
        entries.append(
            {
                "file": filename,
                "sha256": sha256_file(output_dir / filename),
                "explicit_nuisance": target_fit.encoder.to_dict(),
                "hidden_size": int(values.shape[1]),
                "raw_direction_norm": norm,
                **audit,
            }
        )
        for local, index in enumerate(test):
            coordinate, prediction = model.project(values[index], rows[index])
            target = float(test_targets["shared"][local])
            oof_rows.append(
                {
                    "case_id": str(rows[index]["case_id"]),
                    "item_id": str(rows[index]["item_id"]),
                    "fold": fold,
                    "split": "development_oof",
                    "layer": PRIMARY_LAYER,
                    "position": PRIMARY_POSITION,
                    "selected_alpha": alpha,
                    "target_shared": target,
                    "prediction_shared": prediction,
                    "u_coordinate": coordinate,
                }
            )
    if len(oof_rows) != len(rows) or sorted(models) != [0, 1, 2, 3, 4]:
        raise RuntimeError("U OOF fit did not cover every development item")
    index = {
        "format_version": FORMAT_VERSION,
        "definition": (
            "fixed L18 answer-only post_answer; raw choice-coupled shared target; "
            "training-fold target/feature scaling without nuisance removal; "
            "nested existing-fold alpha"
        ),
        "development_only": True,
        "confirmatory_fit_or_selection": False,
        "folds": entries,
    }
    atomic_write_json(output_dir / "index.json", index)
    audit = {
        "format_version": FORMAT_VERSION,
        "development_n": len(rows),
        "unique_items": len({str(row["item_id"]) for row in rows}),
        "outer_folds": entries,
        "all_outer_item_overlaps_empty": all(not entry["item_overlap"] for entry in entries),
        "all_alpha_selection_development_only": True,
        "confirmatory_values_used_for_fit_or_selection": False,
    }
    return models, sorted(oof_rows, key=lambda row: row["case_id"]), audit


def load_attribution_directions(
    panel: LoadedStage08Panel,
) -> dict[int, AttributionFoldDirection]:
    frozen_rule = _read_json(panel.paths.panel06_frozen_rule)
    expected = {int(entry["fold"]): entry for entry in frozen_rule["folds"]}
    output: dict[int, AttributionFoldDirection] = {}
    for fold, path in enumerate(panel.direction_paths):
        checksum = sha256_file(path)
        if checksum != str(expected[fold]["source_sha256"]):
            raise ValueError(f"Frozen A checksum changed for fold {fold}")
        with np.load(path, allow_pickle=False) as payload:
            direction = AttributionFoldDirection(
                fold=fold,
                d_raw=np.asarray(payload["d_raw"], dtype=np.float64),
                d_unit=np.asarray(payload["d_unit"], dtype=np.float64),
                raw_intercept=float(payload["raw_intercept"]),
                train_z_mean=float(payload["train_z_mean"]),
                train_z_sd=float(payload["train_z_sd"]),
                source_file=path,
                source_sha256=checksum,
            )
        if not math.isclose(
            float(np.linalg.norm(direction.d_unit)), 1.0, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(f"Stage10 fold {fold} A direction is not unit norm")
        if direction.train_z_sd <= 0:
            raise ValueError(f"Stage10 fold {fold} A scale is invalid")
        output[fold] = direction
    return output


def direction_geometry(
    source_use: Mapping[int, SourceUseFoldModel],
    attribution: Mapping[int, AttributionFoldDirection],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        u = source_use[fold].unit_direction
        a = attribution[fold].d_unit
        cosine = float(u @ a)
        retention = float(math.sqrt(max(0.0, 1.0 - cosine * cosine)))
        u_perp = u - cosine * a
        a_perp = a - cosine * u
        rows.append(
            {
                "fold": fold,
                "cosine_u_a": cosine,
                "absolute_cosine_u_a": abs(cosine),
                "u_orthogonalized_to_a_norm_fraction": retention,
                "a_orthogonalized_to_u_norm_fraction": retention,
                "u_perp_dot_a": float(u_perp @ a),
                "a_perp_dot_u": float(a_perp @ u),
            }
        )
    return {
        "folds": rows,
        "cosine_mean": float(np.mean([row["cosine_u_a"] for row in rows])),
        "absolute_cosine_mean": float(
            np.mean([row["absolute_cosine_u_a"] for row in rows])
        ),
        "absolute_cosine_maximum": float(
            np.max([row["absolute_cosine_u_a"] for row in rows])
        ),
        "orthogonalized_direction_retention_mean": float(
            np.mean([row["u_orthogonalized_to_a_norm_fraction"] for row in rows])
        ),
        "orthogonalized_direction_retention_minimum": float(
            np.min([row["u_orthogonalized_to_a_norm_fraction"] for row in rows])
        ),
        "interpretation_limit": (
            "Direction cosine and orthogonal norm retention are geometric descriptions, "
            "not evidence of causal or functional independence."
        ),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        raise ValueError("Cannot compute hidden cosine for a zero vector")
    return float(left @ right / denominator)


def build_confirmatory_results(
    panel: LoadedStage08Panel,
    source_use: Mapping[int, SourceUseFoldModel],
    attribution: Mapping[int, AttributionFoldDirection],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method01_root = panel.paths.development_analysis.parent
    raw03 = panel.confirmatory_sources["stage03_raw_confirmatory"]
    graded03 = panel.confirmatory_sources["stage03_graded_confirmatory"]
    analysis06 = panel.confirmatory_sources["stage06_analysis"]
    rows: list[dict[str, Any]] = []
    target_replay_errors: list[float] = []
    for row, hidden01_path, hidden06_path in zip(
        panel.confirmatory_rows,
        panel.confirmatory_hidden01_paths,
        panel.confirmatory_hidden06_paths,
    ):
        case_id = str(row["case_id"])
        fold = int(row["fold"])
        contexts = {
            **_load_l18_answer_hidden(row, method01_root),
            **_load_stage06_hidden(hidden06_path),
        }
        if tuple(contexts) != CONTEXTS:
            raise RuntimeError(f"Unexpected context order for {case_id}: {tuple(contexts)}")
        context_values: dict[str, dict[str, float]] = {}
        for name, hidden in contexts.items():
            u_coordinate, u_prediction = source_use[fold].project(hidden, row)
            a_coordinate, a_prediction = attribution[fold].project(hidden)
            context_values[name] = {
                "u_coordinate": u_coordinate,
                "u_prediction": u_prediction,
                "a_coordinate": a_coordinate,
                "a_prediction": a_prediction,
            }
        post = contexts["answer_only_post_answer"]
        postquery = contexts["postquery_prefix"]
        difference = postquery - post
        relative_l2 = float(np.linalg.norm(difference) / np.linalg.norm(post))
        replayed_target = source_use[fold].transform_raw_target(row)
        b_raw = float(raw03[case_id]["target_shared"])
        error = abs(replayed_target - b_raw)
        target_replay_errors.append(error)
        rows.append(
            {
                "case_id": case_id,
                "item_id": str(row["item_id"]),
                "fold": fold,
                "split": "confirmatory",
                "answer_star": str(row["answer_star"]),
                "answer_star_side": str(row["answer_star_side"]),
                "condition": str(row.get("condition", "")),
                "difficulty": str(row.get("difficulty", "")),
                "prior_index": int(row.get("prior_index", -1)),
                "b_raw": b_raw,
                "b_graded": float(graded03[case_id]["target_shared"]),
                "v": float(analysis06[case_id]["frozen_shared_target"]),
                "raw_target_replayed_from_stage01": replayed_target,
                "raw_target_replay_abs_error": error,
                "contexts": context_values,
                "prefix_replication": {
                    "hidden_cosine": _cosine(post, postquery),
                    "hidden_relative_l2": relative_l2,
                    "hidden_max_abs_difference": float(np.max(np.abs(difference))),
                    "u_coordinate_delta_postquery_minus_post": (
                        context_values["postquery_prefix"]["u_coordinate"]
                        - context_values["answer_only_post_answer"]["u_coordinate"]
                    ),
                    "u_prediction_delta_postquery_minus_post": (
                        context_values["postquery_prefix"]["u_prediction"]
                        - context_values["answer_only_post_answer"]["u_prediction"]
                    ),
                    "a_coordinate_delta_postquery_minus_post": (
                        context_values["postquery_prefix"]["a_coordinate"]
                        - context_values["answer_only_post_answer"]["a_coordinate"]
                    ),
                    "a_prediction_delta_postquery_minus_post": (
                        context_values["postquery_prefix"]["a_prediction"]
                        - context_values["answer_only_post_answer"]["a_prediction"]
                    ),
                },
                "hidden_inputs": {
                    "method01": str(hidden01_path),
                    "panel06": str(hidden06_path),
                },
            }
        )
    maximum = float(max(target_replay_errors))
    if maximum > 1e-10:
        raise RuntimeError(
            f"Stage08 raw target transform does not replay Stage03: max error {maximum}"
        )
    return rows, {
        "raw_target_replay_max_abs_error": maximum,
        "raw_target_replay_passed": True,
        "definition": (
            "Stage08 fold target preprocessing exactly replays frozen Stage03 raw "
            "choice-coupled confirmatory target_shared"
        ),
    }


def _center(values: np.ndarray, sides: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    for side in sorted(set(sides.tolist())):
        selected = sides == side
        output[selected] -= float(np.mean(output[selected]))
    return output


def _safe_corr(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    if rank:
        x = rankdata(x, method="average")
        y = rankdata(y, method="average")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = float(np.sqrt((x @ x) * (y @ y)))
    if denominator <= 1e-12:
        return None
    value = float(x @ y / denominator)
    return value if math.isfinite(value) else None


def fixed_fold_item_bootstrap_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int = SEED,
) -> list[np.ndarray]:
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    by_fold_item: dict[int, dict[str, list[int]]] = {}
    for index, row in enumerate(rows):
        by_fold_item.setdefault(int(row["fold"]), {}).setdefault(
            str(row["item_id"]), []
        ).append(index)
    if sorted(by_fold_item) != [0, 1, 2, 3, 4]:
        raise ValueError("Fixed-fold bootstrap requires folds 0-4")
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for _ in range(iterations):
        indices: list[int] = []
        for fold in range(5):
            clusters = by_fold_item[fold]
            items = sorted(clusters)
            chosen = rng.choice(items, size=len(items), replace=True)
            for item in chosen:
                indices.extend(clusters[str(item)])
        samples.append(np.asarray(indices, dtype=np.int64))
    return samples


def _ci(values: Sequence[float]) -> list[float | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return [None, None]
    return [float(value) for value in np.quantile(finite, [0.025, 0.975])]


def association_with_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    sides: np.ndarray,
    samples: Sequence[np.ndarray],
    *,
    centered: bool,
) -> dict[str, Any]:
    x = _center(left, sides) if centered else np.asarray(left, dtype=np.float64)
    y = _center(right, sides) if centered else np.asarray(right, dtype=np.float64)
    observed = {
        "pearson": _safe_corr(x, y, rank=False),
        "spearman": _safe_corr(x, y, rank=True),
    }
    boot = {"pearson": [], "spearman": []}
    for indices in samples:
        sample_x = np.asarray(left, dtype=np.float64)[indices]
        sample_y = np.asarray(right, dtype=np.float64)[indices]
        sample_sides = sides[indices]
        if centered:
            sample_x = _center(sample_x, sample_sides)
            sample_y = _center(sample_y, sample_sides)
        for name, rank in (("pearson", False), ("spearman", True)):
            value = _safe_corr(sample_x, sample_y, rank=rank)
            if value is not None:
                boot[name].append(value)
    return {
        "n": len(left),
        "answer_side_centered": centered,
        "centering_refit_in_each_bootstrap_sample": centered,
        "cluster_unit": "item_id",
        "stratified_by": "fixed_fold",
        "pearson": {
            "estimate": observed["pearson"],
            "ci95": _ci(boot["pearson"]),
            "valid_bootstrap": len(boot["pearson"]),
        },
        "spearman": {
            "estimate": observed["spearman"],
            "ci95": _ci(boot["spearman"]),
            "valid_bootstrap": len(boot["spearman"]),
        },
        "bootstrap_iterations": len(samples),
    }


def _paired_statistic_bootstrap(
    arrays: Mapping[str, np.ndarray],
    sides: np.ndarray,
    samples: Sequence[np.ndarray],
    statistic: Callable[[Mapping[str, np.ndarray]], float | None],
    *,
    centered: bool,
) -> dict[str, Any]:
    def prepared(indices: np.ndarray | None) -> dict[str, np.ndarray]:
        selected_sides = sides if indices is None else sides[indices]
        output = {
            name: value if indices is None else value[indices]
            for name, value in arrays.items()
        }
        if centered:
            output = {
                name: _center(value, selected_sides) for name, value in output.items()
            }
        return output

    observed = statistic(prepared(None))
    values: list[float] = []
    for indices in samples:
        value = statistic(prepared(indices))
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return {
        "estimate": observed,
        "ci95": _ci(values),
        "valid_bootstrap": len(values),
        "bootstrap_iterations": len(samples),
        "paired_same_item_resamples": True,
        "answer_side_centered": centered,
        "centering_refit_in_each_bootstrap_sample": centered,
        "cluster_unit": "item_id",
        "stratified_by": "fixed_fold",
    }


def _rho(values: Mapping[str, np.ndarray], left: str, right: str) -> float | None:
    return _safe_corr(values[left], values[right], rank=True)


def specialization_contrasts(
    u: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    v: np.ndarray,
    sides: np.ndarray,
    samples: Sequence[np.ndarray],
    *,
    centered: bool,
) -> dict[str, Any]:
    arrays = {"u": u, "a": a, "b": b, "v": v}

    def difference(
        first: tuple[str, str], second: tuple[str, str]
    ) -> Callable[[Mapping[str, np.ndarray]], float | None]:
        def statistic(values: Mapping[str, np.ndarray]) -> float | None:
            left = _rho(values, *first)
            right = _rho(values, *second)
            return None if left is None or right is None else float(left - right)

        return statistic

    def double(values: Mapping[str, np.ndarray]) -> float | None:
        ub = _rho(values, "u", "b")
        ab = _rho(values, "a", "b")
        av = _rho(values, "a", "v")
        uv = _rho(values, "u", "v")
        if any(value is None for value in (ub, ab, av, uv)):
            return None
        return float(((ub - ab) + (av - uv)) / 2.0)

    return {
        "b_axis_advantage_rho_u_minus_a": _paired_statistic_bootstrap(
            arrays, sides, samples, difference(("u", "b"), ("a", "b")), centered=centered
        ),
        "v_axis_advantage_rho_a_minus_u": _paired_statistic_bootstrap(
            arrays, sides, samples, difference(("a", "v"), ("u", "v")), centered=centered
        ),
        "u_target_specificity_rho_b_minus_v": _paired_statistic_bootstrap(
            arrays, sides, samples, difference(("u", "b"), ("u", "v")), centered=centered
        ),
        "a_target_specificity_rho_v_minus_b": _paired_statistic_bootstrap(
            arrays, sides, samples, difference(("a", "v"), ("a", "b")), centered=centered
        ),
        "symmetric_double_specialization": _paired_statistic_bootstrap(
            arrays, sides, samples, double, centered=centered
        ),
    }


def _extract_arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {
        target: np.asarray([float(row[target]) for row in rows], dtype=np.float64)
        for target in TARGETS
    }
    for context in CONTEXTS:
        for axis in AXES:
            for kind in ("coordinate", "prediction"):
                key = f"{context}|{axis}_{kind}"
                values[key] = np.asarray(
                    [float(row["contexts"][context][f"{axis}_{kind}"]) for row in rows],
                    dtype=np.float64,
                )
    return values


def _mean_bootstrap(
    values: np.ndarray, samples: Sequence[np.ndarray]
) -> dict[str, Any]:
    observed = float(np.mean(values))
    boot = [float(np.mean(values[indices])) for indices in samples]
    return {
        "estimate": observed,
        "ci95": _ci(boot),
        "bootstrap_iterations": len(samples),
        "cluster_unit": "item_id",
        "stratified_by": "fixed_fold",
    }


def analyze_results(
    rows: list[dict[str, Any]],
    geometry: dict[str, Any],
    *,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    arrays = _extract_arrays(rows)
    sides = np.asarray([str(row["answer_star_side"]) for row in rows])
    samples = fixed_fold_item_bootstrap_indices(
        rows, iterations=bootstrap_iterations, seed=SEED
    )
    associations: dict[str, Any] = {}
    cross_axis: dict[str, Any] = {}
    specialization: dict[str, Any] = {}
    prediction_sensitivity: dict[str, Any] = {}
    for context in CONTEXTS:
        associations[context] = {}
        cross_axis[context] = {}
        specialization[context] = {}
        prediction_sensitivity[context] = {}
        for centered, mode in ((False, "raw"), (True, "answer_side_centered")):
            associations[context][mode] = {}
            prediction_sensitivity[context][mode] = {}
            for axis in AXES:
                coordinate = arrays[f"{context}|{axis}_coordinate"]
                prediction = arrays[f"{context}|{axis}_prediction"]
                associations[context][mode][axis] = {
                    target: association_with_bootstrap(
                        coordinate,
                        arrays[target],
                        sides,
                        samples,
                        centered=centered,
                    )
                    for target in TARGETS
                }
                prediction_sensitivity[context][mode][axis] = {
                    target: {
                        "pearson": _safe_corr(
                            _center(prediction, sides) if centered else prediction,
                            _center(arrays[target], sides) if centered else arrays[target],
                            rank=False,
                        ),
                        "spearman": _safe_corr(
                            _center(prediction, sides) if centered else prediction,
                            _center(arrays[target], sides) if centered else arrays[target],
                            rank=True,
                        ),
                    }
                    for target in TARGETS
                }
            cross_axis[context][mode] = association_with_bootstrap(
                arrays[f"{context}|u_coordinate"],
                arrays[f"{context}|a_coordinate"],
                sides,
                samples,
                centered=centered,
            )
            specialization[context][mode] = {
                target: specialization_contrasts(
                    arrays[f"{context}|u_coordinate"],
                    arrays[f"{context}|a_coordinate"],
                    arrays[target],
                    arrays["v"],
                    sides,
                    samples,
                    centered=centered,
                )
                for target in ("b_raw", "b_graded")
            }

    prefix_cosine = np.asarray(
        [float(row["prefix_replication"]["hidden_cosine"]) for row in rows]
    )
    prefix_relative = np.asarray(
        [float(row["prefix_replication"]["hidden_relative_l2"]) for row in rows]
    )
    prefix_max_abs = np.asarray(
        [float(row["prefix_replication"]["hidden_max_abs_difference"]) for row in rows]
    )
    prefix: dict[str, Any] = {
        "definition": (
            "01 answer-only post_answer versus 06 postquery answer-prefix; the causal "
            "message prefix through the fixed answer is identical, while BF16 sequence-shape "
            "drift need not be bit-exact"
        ),
        "hidden_cosine_mean": _mean_bootstrap(prefix_cosine, samples),
        "hidden_cosine_median": float(np.median(prefix_cosine)),
        "hidden_cosine_minimum": float(np.min(prefix_cosine)),
        "hidden_relative_l2_mean": _mean_bootstrap(prefix_relative, samples),
        "hidden_relative_l2_maximum": float(np.max(prefix_relative)),
        "hidden_max_abs_difference_mean": _mean_bootstrap(prefix_max_abs, samples),
    }
    for axis in AXES:
        left = arrays[f"answer_only_post_answer|{axis}_coordinate"]
        right = arrays[f"postquery_prefix|{axis}_coordinate"]
        delta = right - left
        prefix[f"{axis}_coordinate_reproduction"] = {
            "association": association_with_bootstrap(
                left, right, sides, samples, centered=False
            ),
            "mean_delta": _mean_bootstrap(delta, samples),
            "mean_absolute_delta": _mean_bootstrap(np.abs(delta), samples),
            "maximum_absolute_delta": float(np.max(np.abs(delta))),
        }

    primary = {
        "u_answer_post_vs_b_raw": associations["answer_only_post_answer"]["raw"]["u"]["b_raw"],
        "a_joint_core_vs_v": associations["joint_core_consensus"]["raw"]["a"]["v"],
        "u_a_answer_post": cross_axis["answer_only_post_answer"]["raw"],
        "u_a_joint_core": cross_axis["joint_core_consensus"]["raw"],
        "double_specialization_answer_post_b_raw": specialization[
            "answer_only_post_answer"
        ]["raw"]["b_raw"]["symmetric_double_specialization"],
        "double_specialization_joint_core_b_raw": specialization[
            "joint_core_consensus"
        ]["raw"]["b_raw"]["symmetric_double_specialization"],
    }
    return {
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": SEED,
            "cluster_unit": "item_id",
            "stratified_by": "fixed_fold",
            "paired_resamples_reused_across_metrics": True,
        },
        "associations": associations,
        "cross_axis_associations": cross_axis,
        "paired_specialization_contrasts": specialization,
        "decoder_prediction_point_sensitivity": prediction_sensitivity,
        "prefix_replication": prefix,
        "direction_geometry": geometry,
        "primary_descriptive_panel": primary,
    }


def _center_result_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sides = sorted({str(row["answer_star_side"]) for row in rows})
    fields: dict[str, Callable[[dict[str, Any]], float]] = {
        target: lambda row, key=target: float(row[key]) for target in TARGETS
    }
    for context in CONTEXTS:
        for axis in AXES:
            fields[f"{context}|{axis}_coordinate"] = (
                lambda row, c=context, a=axis: float(
                    row["contexts"][c][f"{a}_coordinate"]
                )
            )
    means: dict[str, dict[str, float]] = {}
    for side in sides:
        selected = [row for row in rows if str(row["answer_star_side"]) == side]
        means[side] = {
            field: float(np.mean([getter(row) for row in selected]))
            for field, getter in fields.items()
        }
    for row in rows:
        side = str(row["answer_star_side"])
        row["answer_side_centered"] = {
            field: getter(row) - means[side][field]
            for field, getter in fields.items()
        }
    return {
        "group_means": means,
        "saved_values_definition": "full-confirmatory-sample descriptive centering",
        "bootstrap_definition": "group means are refit within every resample",
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    def metric(block: Mapping[str, Any]) -> str:
        rho = block["spearman"]["estimate"]
        ci = block["spearman"]["ci95"]
        return f"{rho:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if rho is not None else "NA"

    lines = [
        "# Stage08 Fixed-L18 Representation Divergence Audit",
        "",
        "## Scope",
        "",
        (
            "This is a CPU-only, post-hoc descriptive/OOD cross-context audit. It contains "
            "no intervention, is non-gate-bearing, and does not modify Stages 01, 03, 10, "
            "or the skipped Stage 07."
        ),
        "",
        f"- Development fit: {summary['development_n']} unique items",
        f"- Exact confirmatory panel: {summary['n']} unique items",
        "- U: development-only nested-fold raw choice-coupled Ridge, L18 answer-only post-answer",
        "- A: byte-frozen Stage10 protocol-shared attribution directions",
        "",
        "## Primary descriptive results",
        "",
        "| Contrast | Spearman rho [fixed-fold item bootstrap 95% CI] |",
        "|---|---:|",
    ]
    primary = summary["analysis"]["primary_descriptive_panel"]
    lines.extend(
        [
            f"| U(answer-only post) vs B_raw | {metric(primary['u_answer_post_vs_b_raw'])} |",
            f"| A(joint core mean) vs V | {metric(primary['a_joint_core_vs_v'])} |",
            f"| U vs A at answer-only post | {metric(primary['u_a_answer_post'])} |",
            f"| U vs A at joint core mean | {metric(primary['u_a_joint_core'])} |",
        ]
    )
    lines.extend(
        [
            "",
            "## Context matrix (raw coordinates)",
            "",
            "| Context | U-B_raw | U-V | A-B_raw | A-V | U-A |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    associations = summary["analysis"]["associations"]
    cross = summary["analysis"]["cross_axis_associations"]
    for context in CONTEXTS:
        block = associations[context]["raw"]
        lines.append(
            "| "
            + context
            + " | "
            + " | ".join(
                (
                    metric(block["u"]["b_raw"]),
                    metric(block["u"]["v"]),
                    metric(block["a"]["b_raw"]),
                    metric(block["a"]["v"]),
                    metric(cross[context]["raw"]),
                )
            )
            + " |"
        )
    geometry = summary["analysis"]["direction_geometry"]
    prefix = summary["analysis"]["prefix_replication"]
    lines.extend(
        [
            "",
            "## Geometry and exact-prefix reconstruction",
            "",
            f"- Mean foldwise cos(U,A): {geometry['cosine_mean']:.4f}",
            f"- Maximum absolute foldwise cosine: {geometry['absolute_cosine_maximum']:.4f}",
            (
                "- Minimum orthogonalized direction norm retained: "
                f"{geometry['orthogonalized_direction_retention_minimum']:.4f}"
            ),
            (
                "- Mean hidden cosine, 01 post-answer vs 06 postquery prefix: "
                f"{prefix['hidden_cosine_mean']['estimate']:.6f}"
            ),
            (
                "- Mean relative L2 drift for that reconstruction: "
                f"{prefix['hidden_relative_l2_mean']['estimate']:.6f}"
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "Low direction cosine, cross-decoding specialization, or prefix replication "
                "does not establish functional independence or causal mediation. U is a "
                "post-hoc fixed-L18 readout, and projecting it into joint-report contexts is OOD. "
                "A is constructively tied to Stage10 verbal-attribution targets, while Stage06 "
                "did not confirm postquery report transfer. Existing gate decisions remain frozen."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def analyze_fixed_l18_representation_divergence(
    panel: LoadedStage08Panel,
    output_dir: str | Path,
    *,
    bootstrap_iterations: int = 1000,
    config_fingerprint: str,
    input_provenance: dict[str, Any],
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    method01_root = panel.paths.development_analysis.parent
    development_hidden = np.stack(
        [
            _load_l18_answer_hidden(row, method01_root)["answer_only_post_answer"]
            for row in panel.development_rows
        ]
    )
    directions_dir = output / "directions"
    source_use, oof_rows, fold_audit = fit_source_use_directions(
        panel.development_rows,
        development_hidden,
        directions_dir,
        answer_vocabulary=panel.gate_snapshot["stage03_frozen_answer_vocabulary"],
    )
    stage03_dev = panel.confirmatory_sources.get("stage03_raw_development")
    # The development Stage03 index is not part of the confirmatory source mapping;
    # replay it directly to audit that target preprocessing, not its selected readout.
    stage03_dev_index = _unique_index(
        _read_jsonl(panel.paths.representation03_raw_development),
        "stage03_raw_development_replay",
        expected_split="development",
        expected_estimand="raw_choice_coupled",
    )
    replay_errors = [
        abs(float(row["target_shared"]) - float(stage03_dev_index[row["case_id"]]["target_shared"]))
        for row in oof_rows
    ]
    fold_audit["development_target_replay_max_abs_error"] = float(max(replay_errors))
    fold_audit["development_target_replay_passed"] = bool(max(replay_errors) <= 1e-10)
    if not fold_audit["development_target_replay_passed"]:
        raise RuntimeError("Stage08 development target preprocessing does not replay Stage03")

    attribution = load_attribution_directions(panel)
    geometry = direction_geometry(source_use, attribution)
    result_rows, confirm_replay = build_confirmatory_results(
        panel, source_use, attribution
    )
    centering = _center_result_rows(result_rows)
    analysis = analyze_results(
        result_rows, geometry, bootstrap_iterations=bootstrap_iterations
    )

    manifest = {
        "format_version": FORMAT_VERSION,
        "development_n": len(panel.development_rows),
        "development_unique_items": len(
            {str(row["item_id"]) for row in panel.development_rows}
        ),
        "confirmatory_n": len(result_rows),
        "confirmatory_unique_items": len({row["item_id"] for row in result_rows}),
        "exact_confirmatory_case_ids": [row["case_id"] for row in result_rows],
        "development_case_ids": [str(row["case_id"]) for row in panel.development_rows],
        "join_audit": panel.join_audit,
        "contexts": list(CONTEXTS),
        "u_fit_site": {"layer": PRIMARY_LAYER, "position": PRIMARY_POSITION},
        "a_source": "byte-frozen Stage10 fold directions",
        "confirmatory_fit_or_selection": False,
    }
    summary: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "experiment": OUTPUT_DIR,
        "cpu_only": True,
        "new_model_forwards": 0,
        "development_n": len(panel.development_rows),
        "n": len(result_rows),
        "unique_items": len({row["item_id"] for row in result_rows}),
        "config_fingerprint": config_fingerprint,
        "input_aggregate_sha256": input_provenance["input_aggregate_sha256"],
        "fit_definition": {
            "u": (
                "raw choice-coupled shared source-use Ridge at answer-only post_answer L18; "
                "five outer item folds; alpha selected by inner development folds only"
            ),
            "a": (
                "Stage10 fold-specific d_raw/d_unit/intercept/training scale, byte-frozen; "
                "no refit, recalibration, or sign change"
            ),
            "alpha_candidates": [float(value) for value in RIDGE_ALPHAS],
            "confirmatory_fit_or_selection": False,
        },
        "target_definition": {
            "b_raw": "Stage03 raw_choice_coupled frozen target_shared",
            "b_graded": "Stage03 graded_preregistered frozen target_shared",
            "v": "Stage06 frozen_shared_target from the frozen Stage10 target transform",
        },
        "contexts": {
            "answer_only_pre_answer": "01 SA-free answer-only state immediately before answer",
            "answer_only_post_answer": "01 SA-free teacher-forced answer-token state",
            "postquery_prefix": (
                "06 later-query branch answer-token state with identical causal prefix through answer"
            ),
            "joint_common9": "06 common_9_ordered joint-report PANL state",
            "joint_core_consensus": "mean L18 PANL hidden over seven 06 core common protocols",
        },
        "fold_audit": fold_audit,
        "target_replay": confirm_replay,
        "answer_side_centering": centering,
        "analysis": analysis,
        "gate_snapshot": panel.gate_snapshot,
        "claim_scope": {
            "post_hoc": True,
            "descriptive": True,
            "ood_cross_context": True,
            "gate_bearing": False,
            "causal_intervention": False,
            "causal_mediator_authorized": False,
            "changes_stage01": False,
            "changes_stage03": False,
            "changes_stage10": False,
            "changes_stage07": False,
            "allowed_claim": (
                "descriptive fixed-L18 linear-axis geometry, cross-decoding specialization, "
                "and exact-prefix reconstruction only"
            ),
            "forbidden_claims": [
                "causal source-use mediator",
                "functional independence of U and A",
                "new gate authorization",
                "confirmation of Stage06 postquery report transfer",
            ],
        },
    }

    run_config = {
        "format_version": FORMAT_VERSION,
        "experiment": OUTPUT_DIR,
        "config_fingerprint": config_fingerprint,
        "bootstrap_iterations": bootstrap_iterations,
        "seed": SEED,
        "ridge_alphas": [float(value) for value in RIDGE_ALPHAS],
        "layer": PRIMARY_LAYER,
        "u_fit_position": PRIMARY_POSITION,
        "contexts": list(CONTEXTS),
        "expected_development_n": EXPECTED_DEVELOPMENT_N,
        "expected_confirmatory_n": EXPECTED_CONFIRMATORY_N,
        "cpu_only": True,
        "analyze_only": True,
    }
    provenance = {
        **input_provenance,
        "config_fingerprint": config_fingerprint,
        "output_scope": str(output_root(panel.paths.experiment_dir)),
        "prior_stage_inputs_read_only": True,
        "prior_stage_writes": [],
    }
    write_jsonl_atomic(output / "development_oof_predictions.jsonl", oof_rows)
    atomic_write_json(output / "fold_audit.json", fold_audit)
    atomic_write_json(output / "cohort_manifest.json", manifest)
    write_jsonl_atomic(output / "results.jsonl", result_rows)
    atomic_write_json(output / "run_config.json", run_config)
    atomic_write_json(output / "provenance.json", provenance)
    atomic_write_json(output / "summary.json", summary)
    atomic_write_text(output / "summary.md", _markdown(summary))
    return summary


def configuration_payload(*, bootstrap_iterations: int) -> dict[str, Any]:
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")
    return {
        "format_version": FORMAT_VERSION,
        "experiment": OUTPUT_DIR,
        "bootstrap_iterations": int(bootstrap_iterations),
        "seed": SEED,
        "ridge_alphas": [float(value) for value in RIDGE_ALPHAS],
        "layer": PRIMARY_LAYER,
        "u_fit_position": PRIMARY_POSITION,
        "contexts": list(CONTEXTS),
        "expected_development_n": EXPECTED_DEVELOPMENT_N,
        "expected_confirmatory_n": EXPECTED_CONFIRMATORY_N,
        "post_hoc": True,
        "gate_bearing": False,
        "causal": False,
    }
