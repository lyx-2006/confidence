"""Join conflict-only joint/no-SA records and train grouped OOF probes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import sklearn
from confidence_test.answer_metrics import normalize_answer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from layer_metacognition.hidden_state_store import (
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
)
from layer_metacognition.probe.hidden_state_loader import HiddenStateLoader
from layer_metacognition.probe.provenance import canonical_fingerprint, sha256_file
from layer_metacognition.probe.torch_logistic_probe import (
    fit_torch_logistic_probe,
    resolve_torch_device,
)
from layer_metacognition.probe_sa_prediction.train_sa_probes import _fit_hard
from layer_metacognition.probe_sa_no_prompt import (
    CONFLICT_CONDITIONS,
    DEFAULT_COHORTS,
    DEFAULT_JOINT_EXPERIMENT_DIR,
    DEFAULT_LAYERS,
    DEFAULT_NO_SA_EXPERIMENT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSITIONS,
    DEFAULT_SPLIT_ASSIGNMENTS,
    FORBIDDEN_SA_TEXT,
    HIDDEN_STATE_DEFINITION,
    SA_CLASSES,
    TASKS,
    join_key,
    prediction_key,
)
from layer_metacognition.probe_sa_prediction.train_sa_probes import _fit_soft


_FORBIDDEN_PATTERN = re.compile(r"source\s+attribution|sa\s+class", re.IGNORECASE)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _append_failure(path: Path, row: dict[str, Any]) -> None:
    existing = []
    if path.is_file():
        existing = load_jsonl(path, repair_trailing=False)
    key = tuple(str(row.get(field)) for field in ("side", "case_id", "reason", "condition"))
    if any(tuple(str(item.get(field)) for field in ("side", "case_id", "reason", "condition")) == key for item in existing):
        return
    _atomic_jsonl(path, [*existing, row])


def _record_key(record: dict[str, Any], *, side: str) -> tuple[str, int, str, str] | None:
    try:
        item_id = str(record["item_id"])
        prior_index = int(record["prior_index"])
        condition = str(record["condition"])
        version = str(record["version"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not item_id or not condition or version != "v4" or condition not in CONFLICT_CONDITIONS:
        return None
    del side
    return item_id, prior_index, condition, version


def _map_records(
    path: Path,
    *,
    side: str,
    index: dict[str, Any],
    layers: Sequence[int],
    positions: Sequence[str],
) -> tuple[dict[tuple[str, int, str, str], dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int, str, str], dict[str, Any]]]:
    records = load_jsonl(path)
    structural: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    valid: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for record in records:
        key = _record_key(record, side=side)
        if key is None:
            failures.append({"side": side, "case_id": record.get("case_id"), "stage": "input_validation", "reason": "invalid_join_key_or_condition"})
            continue
        if key in structural:
            raise ValueError(f"Duplicate {side} join key: {key}")
        structural[key] = record
        if record.get("status") != "completed":
            failures.append({"side": side, "case_id": record.get("case_id"), "stage": "input_validation", "reason": "record_not_completed", "join_key": list(key)})
            continue
        generated = record.get("generated")
        if not isinstance(generated, dict):
            failures.append({"side": side, "case_id": record.get("case_id"), "stage": "input_validation", "reason": "missing_generated", "join_key": list(key)})
            continue
        answer = generated.get("current_answer")
        if not isinstance(answer, str) or not answer.strip():
            failures.append({"side": side, "case_id": record.get("case_id"), "stage": "input_validation", "reason": "missing_current_answer", "join_key": list(key)})
            continue
        if side == "no_sa":
            answer_result = generated.get("current_answer_result")
            raw_output = answer_result.get("raw_output", "") if isinstance(answer_result, dict) else ""
            if _FORBIDDEN_PATTERN.search(str(raw_output)):
                failures.append({"side": side, "case_id": record.get("case_id"), "stage": "input_validation", "reason": "source_attribution_text_in_no_sa_wire", "join_key": list(key)})
                continue
            reference = record.get("hidden_state_reference")
            case_id = str(record.get("case_id") or "")
            index_reference = index.get("cases", {}).get(case_id)
            if not case_id or not isinstance(reference, dict) or not isinstance(index_reference, dict):
                failures.append({"side": side, "case_id": case_id, "stage": "input_validation", "reason": "missing_hidden_state_reference", "join_key": list(key)})
                continue
            reference_fields = ("shard_path", "offset", "hidden_size", "hidden_state_definition", "layer_indices", "position_names")
            if any(reference.get(field) != index_reference.get(field) for field in reference_fields):
                raise ValueError(f"No-SA hidden reference/index schema mismatch for {case_id}")
            available_layers = [int(value) for value in index.get("layer_indices", [])]
            available_positions = [str(value) for value in index.get("position_names", [])]
            if set(layers) - set(available_layers) or set(positions) - set(available_positions):
                raise ValueError("No-SA hidden-state index is missing requested layers or positions")
            if reference.get("hidden_state_definition") != HIDDEN_STATE_DEFINITION:
                raise ValueError("No-SA hidden-state definition does not match the Probe contract")
            valid[key] = {
                "case_id": case_id,
                "item_id": str(record["item_id"]),
                "prior_index": int(record["prior_index"]),
                "condition": str(record["condition"]),
                "version": "v4",
                "answer": answer,
                "normalized_answer": normalize_answer(answer),
                "hidden_state_reference": dict(reference),
            }
        else:
            attribution = generated.get("source_attribution")
            if not isinstance(attribution, dict):
                failures.append({"side": side, "case_id": record.get("case_id"), "stage": "input_validation", "reason": "missing_source_attribution", "join_key": list(key)})
                continue
            label = str(attribution.get("parsed_label"))
            soft = attribution.get("soft_image_score")
            if label not in SA_CLASSES:
                failures.append({"side": side, "case_id": record.get("case_id"), "stage": "input_validation", "reason": "invalid_hard_target", "join_key": list(key)})
                continue
            if not isinstance(soft, (int, float)) or not math.isfinite(float(soft)) or not 0.0 <= float(soft) <= 1.0:
                failures.append({"side": side, "case_id": record.get("case_id"), "stage": "input_validation", "reason": "invalid_soft_target", "join_key": list(key)})
                continue
            valid[key] = {
                "case_id": str(record.get("case_id") or ""),
                "item_id": str(record["item_id"]),
                "prior_index": int(record["prior_index"]),
                "condition": str(record["condition"]),
                "version": "v4",
                "answer": answer,
                "normalized_answer": normalize_answer(answer),
                "hard_label": label,
                "soft_score": float(soft),
            }
    return valid, failures, structural


def _validate_experiment(
    experiment_dir: Path,
    *,
    side: str,
    layers: Sequence[int],
    positions: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path]:
    config_path = experiment_dir / "config.json"
    results_path = experiment_dir / "results.jsonl"
    index_path = experiment_dir / "hidden_states" / "index.json"
    required_paths = (config_path, results_path, index_path) if side == "no_sa" else (config_path, results_path)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required {side} artifact does not exist: {path}")
    config = _json(config_path)
    index = _json(index_path) if side == "no_sa" else {}
    if config.get("versions") != ["v4"] or config.get("attribution_mode") != ("joint" if side == "joint" else "none"):
        raise ValueError(f"{side} experiment must be V4 with the required attribution_mode")
    if side == "joint" and config.get("source_prompt_variant") != "answer_basis_9":
        raise ValueError("Joint target experiment must use source_prompt_variant=answer_basis_9")
    if side == "no_sa":
        if config.get("source_prompt_variant") != "baseline":
            raise ValueError("No-SA experiment must use source_prompt_variant=baseline")
        if set(config.get("conditions", [])) != set(CONFLICT_CONDITIONS):
            raise ValueError("No-SA experiment must contain only conflict_easy and conflict_hard")
    index_layers = [int(value) for value in index.get("layer_indices", [])]
    index_positions = [str(value) for value in index.get("position_names", [])]
    if side == "no_sa" and (set(layers) - set(index_layers) or set(positions) - set(index_positions)):
        raise ValueError(f"No-SA hidden schema is missing requested layers/positions: layers={index_layers}, positions={index_positions}")
    if side == "no_sa" and not isinstance(index.get("cases"), dict):
        raise ValueError(f"Hidden-state index has no cases object: {index_path}")
    return config, index, config_path, results_path, index_path


def _join_inputs(
    joint_dir: Path,
    no_sa_dir: Path,
    split_path: Path,
    *,
    layers: Sequence[int],
    positions: Sequence[str],
    n_splits: int,
    max_items: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    joint_config, _joint_index, joint_config_path, joint_results_path, _ = _validate_experiment(joint_dir, side="joint", layers=layers, positions=positions)
    no_config, no_index, no_config_path, no_results_path, no_index_path = _validate_experiment(no_sa_dir, side="no_sa", layers=layers, positions=positions)
    joint, joint_failures, joint_structural = _map_records(joint_results_path, side="joint", index={}, layers=layers, positions=positions)
    no_sa, no_failures, no_structural = _map_records(no_results_path, side="no_sa", index=no_index, layers=layers, positions=positions)
    failures = [*joint_failures, *no_failures]
    keys = sorted(set(joint) | set(no_sa), key=lambda value: tuple(str(part) for part in value))
    unmatched: list[dict[str, Any]] = []
    for key in keys:
        if key not in joint or key not in no_sa:
            unmatched.append({"join_key": list(key), "missing_side": "joint" if key not in joint else "no_sa", "joint_case_id": joint.get(key, {}).get("case_id"), "no_sa_case_id": no_sa.get(key, {}).get("case_id")})
    joined: list[dict[str, Any]] = []
    for key in sorted(set(joint).intersection(no_sa), key=lambda value: tuple(str(part) for part in value)):
        left = joint[key]
        right = no_sa[key]
        joined.append({
            "join_key": {"item_id": key[0], "prior_index": key[1], "condition": key[2], "version": key[3]},
            "item_id": key[0],
            "prior_index": key[1],
            "condition": key[2],
            "version": key[3],
            "joint_case_id": left["case_id"],
            "no_sa_case_id": right["case_id"],
            "case_id": right["case_id"],
            "joint_answer": left["answer"],
            "no_sa_answer": right["answer"],
            "joint_normalized_answer": left["normalized_answer"],
            "no_sa_normalized_answer": right["normalized_answer"],
            "answer_match": left["normalized_answer"] == right["normalized_answer"],
            "hard_label": left["hard_label"],
            "soft_score": left["soft_score"],
            "hidden_state_reference": right["hidden_state_reference"],
        })
    if not joined:
        raise ValueError("No valid conflict records joined between joint and no-SA experiments")
    item_ids = sorted({row["item_id"] for row in joined}, key=lambda value: (int(value) if str(value).isdigit() else str(value)))
    if max_items is not None:
        if max_items < n_splits:
            raise ValueError("--max-samples must be at least --n-splits")
        selected = set(item_ids[: int(max_items)])
        joined = [row for row in joined if row["item_id"] in selected]
        item_ids = item_ids[: int(max_items)]
    assignment = _json(split_path)
    if int(assignment.get("n_splits", -1)) != int(n_splits):
        raise ValueError("External split assignment n_splits differs from the request")
    if assignment.get("group_key") != "item_id" or not isinstance(assignment.get("item_to_fold"), dict):
        raise ValueError("External split assignment must use group_key=item_id")
    item_to_fold = {str(key): int(value) for key, value in assignment["item_to_fold"].items()}
    missing_items = sorted(set(item_ids) - set(item_to_fold))
    if missing_items:
        raise ValueError(f"Selected joined items missing split assignments: {missing_items}")
    invalid = {key: value for key, value in item_to_fold.items() if value < 0 or value >= n_splits}
    if invalid:
        raise ValueError(f"Invalid external split assignment folds: {invalid}")
    selected_folds = {item_to_fold[item] for item in item_ids}
    if selected_folds != set(range(n_splits)):
        raise ValueError(f"Selected joined items do not cover all folds: {selected_folds}")
    cohort_items = {
        "all_joined": {row["item_id"] for row in joined},
        "answer_matched": {row["item_id"] for row in joined if row["answer_match"]},
    }
    if len(cohort_items["answer_matched"]) < n_splits:
        raise ValueError("answer_matched cohort has fewer items than n_splits")
    if {item_to_fold[item] for item in cohort_items["answer_matched"]} != set(range(n_splits)):
        raise ValueError("answer_matched cohort does not cover every external split fold")
    provenance = {
        "joint_config_path": str(joint_config_path),
        "joint_results_path": str(joint_results_path),
        "no_sa_config_path": str(no_config_path),
        "no_sa_results_path": str(no_results_path),
        "no_sa_hidden_index_path": str(no_index_path),
        "joint_config_sha256": sha256_file(joint_config_path),
        "joint_results_sha256": sha256_file(joint_results_path),
        "no_sa_config_sha256": sha256_file(no_config_path),
        "no_sa_results_sha256": sha256_file(no_results_path),
        "no_sa_hidden_index_sha256": sha256_file(no_index_path),
        "split_assignments_sha256": sha256_file(split_path),
        "joint_attribution_mode": joint_config.get("attribution_mode"),
        "no_sa_attribution_mode": no_config.get("attribution_mode"),
        "conditions": list(CONFLICT_CONDITIONS),
        "joined_case_count": len(joined),
        "joined_item_count": len(item_ids),
        "answer_matched_case_count": sum(bool(row["answer_match"]) for row in joined),
        "answer_matched_item_count": len(cohort_items["answer_matched"]),
        "answer_match_rate": float(sum(bool(row["answer_match"]) for row in joined) / len(joined)),
        "answer_match_rate_by_condition": {
            condition: float(sum(row["answer_match"] for row in joined if row["condition"] == condition) / max(1, sum(row["condition"] == condition for row in joined)))
            for condition in CONFLICT_CONDITIONS
        },
        "answer_match_rate_by_prior_index": {
            str(prior): float(sum(row["answer_match"] for row in joined if row["prior_index"] == prior) / max(1, sum(row["prior_index"] == prior for row in joined)))
            for prior in sorted({int(row["prior_index"]) for row in joined})
        },
        "hard_label_distribution": dict(Counter(row["hard_label"] for row in joined)),
        "item_to_fold": item_to_fold,
        "joint_case_structural_count": len(joint_structural),
        "no_sa_case_structural_count": len(no_structural),
    }
    del joint_config, no_config
    return joined, failures, unmatched, assignment, provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-experiment-dir", default=str(DEFAULT_JOINT_EXPERIMENT_DIR))
    parser.add_argument("--no-sa-experiment-dir", default=str(DEFAULT_NO_SA_EXPERIMENT_DIR))
    parser.add_argument("--split-assignments", default=str(DEFAULT_SPLIT_ASSIGNMENTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--positions", nargs="+", choices=list(DEFAULT_POSITIONS), default=list(DEFAULT_POSITIONS))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cohorts", nargs="+", choices=list(DEFAULT_COHORTS), default=list(DEFAULT_COHORTS))
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    return parser


def _prepare_output(output_dir: Path, immutable: dict[str, Any], *, resume: bool) -> tuple[dict[str, Any], bool]:
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        if not resume:
            raise FileExistsError(f"Output already exists; pass --resume: {output_dir}")
        saved = _json(config_path)
        if saved.get("config_fingerprint") != immutable["config_fingerprint"]:
            raise ValueError("Cannot resume because immutable no-SA Probe configuration differs")
        return saved, saved.get("status") == "complete"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty protected output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return {**immutable, "status": "running"}, False


def _feature_matrix(loader: HiddenStateLoader, records: Sequence[dict[str, Any]], *, layer: int, position: str) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    usable: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        try:
            vector = loader.load_vector(record, layer=layer, position_name=position)
            if vector.ndim != 1 or not np.isfinite(vector).all():
                raise ValueError(f"invalid vector {vector.shape}")
            usable.append(record)
            vectors.append(vector)
        except Exception as exc:
            failures.append({"side": "no_sa", "case_id": record["no_sa_case_id"], "stage": "feature_loading", "position": position, "layer": int(layer), "reason": type(exc).__name__, "message": str(exc)})
    if not vectors:
        return usable, np.empty((0, 0), dtype=np.float32), failures
    return usable, np.stack(vectors).astype(np.float32, copy=False), failures


def _outer_rows(records: Sequence[dict[str, Any]], item_to_fold: dict[str, int], *, fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row for row in records if item_to_fold[str(row["item_id"])] != fold]
    test = [row for row in records if item_to_fold[str(row["item_id"])] == fold]
    if {row["item_id"] for row in train}.intersection(row["item_id"] for row in test):
        raise AssertionError(f"Item leakage in fold {fold}")
    return train, test


def _majority_label(records: Sequence[dict[str, Any]]) -> str:
    counts = Counter(str(row["hard_label"]) for row in records)
    if not counts:
        raise ValueError("Cannot select majority label from empty training fold")
    return min(SA_CLASSES, key=lambda label: (-counts[label], label))


def _job_id(cohort: str, task: str, position: str, layer: int, fold: int) -> str:
    return f"{cohort}|{task}|{position}|{int(layer)}|{int(fold)}"


def _existing_predictions(path: Path, *, repair: bool) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    rows = load_jsonl(path, repair_trailing=repair) if path.is_file() else []
    keys: set[tuple[Any, ...]] = set()
    for row in rows:
        key = prediction_key(row)
        if key in keys:
            raise ValueError(f"Duplicate existing prediction key: {key}")
        keys.add(key)
    return rows, keys


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    layers = sorted(set(int(value) for value in args.layers))
    positions = [value for value in DEFAULT_POSITIONS if value in set(args.positions)]
    cohorts = [value for value in DEFAULT_COHORTS if value in set(args.cohorts)]
    if not layers or any(value < 0 for value in layers):
        raise ValueError("--layers must contain non-negative indices")
    if not positions or not cohorts:
        raise ValueError("--positions and --cohorts cannot be empty")
    if args.n_splits < 2 or args.bootstrap_repeats < 1:
        raise ValueError("--n-splits must be >=2 and --bootstrap-repeats must be positive")
    joint_dir = Path(args.joint_experiment_dir).resolve()
    no_sa_dir = Path(args.no_sa_experiment_dir).resolve()
    split_path = Path(args.split_assignments).resolve()
    output_dir = Path(args.output_dir).resolve()
    joined, failures, unmatched, assignment, provenance = _join_inputs(
        joint_dir, no_sa_dir, split_path,
        layers=layers, positions=positions, n_splits=args.n_splits, max_items=args.max_samples,
    )
    immutable = {
        "format_version": 1,
        "joint_experiment_dir": str(joint_dir),
        "no_sa_experiment_dir": str(no_sa_dir),
        "split_assignments": str(split_path),
        "output_dir": str(output_dir),
        "conditions": list(CONFLICT_CONDITIONS),
        "layers": layers,
        "positions": positions,
        "cohorts": cohorts,
        "tasks": list(TASKS),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "max_samples_item_groups": args.max_samples,
        "requested_device": args.device,
        "resolved_hard_probe_device": resolve_torch_device(args.device),
        "bootstrap_repeats": int(args.bootstrap_repeats),
        "hard_model": {"type": "balanced_l2_multinomial_logistic_regression", "C": 1.0},
        "soft_model": {"type": "standard_scaler_plus_ridge", "alpha": 1.0, "solver": "lsqr", "prediction_clipping": False},
        "hidden_state_definition": HIDDEN_STATE_DEFINITION,
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
        **provenance,
    }
    immutable["config_fingerprint"] = canonical_fingerprint(immutable)
    run_config, already_complete = _prepare_output(output_dir, immutable, resume=args.resume)
    if already_complete:
        return {"status": "complete", "resumed": True, "output_dir": str(output_dir)}
    atomic_write_json(output_dir / "run_config.json", run_config)
    _atomic_jsonl(output_dir / "join_records.jsonl", joined)
    _atomic_jsonl(output_dir / "unmatched_answers.jsonl", unmatched)
    _atomic_jsonl(output_dir / "input_failures.jsonl", failures)
    atomic_write_json(output_dir / "join_manifest.json", {"format_version": 1, **provenance, "unmatched_count": len(unmatched), "input_failure_count": len(failures)})
    split_audit = {"format_version": 1, "group_key": assignment["group_key"], "n_splits": int(assignment["n_splits"]), "seed": assignment.get("seed"), "split_assignments_sha256": sha256_file(split_path), "item_to_fold": assignment["item_to_fold"], "cohort_items": {cohort: sorted({row["item_id"] for row in joined if cohort == "all_joined" or row["answer_match"]}) for cohort in cohorts}}
    atomic_write_json(output_dir / "split_audit.json", split_audit)
    atomic_write_json(output_dir / "split_assignments.json", assignment)
    item_to_fold = {str(key): int(value) for key, value in assignment["item_to_fold"].items()}
    cohort_records = {"all_joined": joined, "answer_matched": [row for row in joined if row["answer_match"]]}
    predictions_path = output_dir / "predictions" / "oof_predictions.jsonl"
    _old_rows, existing_keys = _existing_predictions(predictions_path, repair=args.resume)
    progress_path = output_dir / "progress.json"
    progress = _json(progress_path) if progress_path.is_file() else {}
    completed_jobs = set(str(value) for value in progress.get("completed_jobs", []))
    invalid_jobs = dict(progress.get("invalid_jobs", {}))
    job_audits = dict(progress.get("job_audits", {}))
    total_jobs = len(cohorts) * len(TASKS) * len(positions) * len(layers) * args.n_splits
    loader = HiddenStateLoader(no_sa_dir, cache_size=2)

    def checkpoint(status: str) -> None:
        atomic_write_json(progress_path, {"status": status, "total_job_count": total_jobs, "completed_job_count": len(completed_jobs), "invalid_job_count": len(invalid_jobs), "prediction_count": len(existing_keys), "completed_jobs": sorted(completed_jobs), "invalid_jobs": invalid_jobs, "job_audits": job_audits, "elapsed_seconds": float(time.perf_counter() - started)})

    checkpoint("running")
    for cohort in cohorts:
        records = cohort_records[cohort]
        for position in positions:
            for layer in layers:
                usable, matrix, feature_failures = _feature_matrix(loader, records, layer=layer, position=position)
                for failure in feature_failures:
                    _append_failure(output_dir / "input_failures.jsonl", failure)
                ordinal = {row["no_sa_case_id"]: index for index, row in enumerate(usable)}
                for task in TASKS:
                    for fold in range(args.n_splits):
                        identifier = _job_id(cohort, task, position, layer, fold)
                        if identifier in invalid_jobs:
                            continue
                        train_records, test_records = _outer_rows(usable, item_to_fold, fold=fold)
                        expected = {row["no_sa_case_id"] for row in test_records}
                        present = {str(key[-1]) for key in existing_keys if key[:5] == (cohort, task, position, int(layer), int(fold))}
                        if present - expected:
                            raise ValueError(f"OOF predictions contain unexpected cases for {identifier}")
                        if present == expected and expected:
                            completed_jobs.add(identifier)
                            continue
                        train_items = {row["item_id"] for row in train_records}
                        test_items = {row["item_id"] for row in test_records}
                        job_audits[identifier] = {"train_sample_count": len(train_records), "test_sample_count": len(test_records), "train_item_count": len(train_items), "test_item_count": len(test_items), "item_overlap_count": len(train_items.intersection(test_items))}
                        try:
                            if not train_records or not test_records:
                                raise ValueError("Outer fold has an empty train or test partition")
                            train_X = matrix[[ordinal[row["no_sa_case_id"]] for row in train_records]]
                            test_X = matrix[[ordinal[row["no_sa_case_id"]] for row in test_records]]
                            if task == "hard_label":
                                task_rows, diagnostics = _fit_hard(train_X, test_X, train_records, test_records, device=immutable["resolved_hard_probe_device"], seed=args.seed)
                                majority = _majority_label(train_records)
                            else:
                                task_rows, diagnostics = _fit_soft(train_X, test_X, train_records, test_records)
                                majority = None
                            rows_to_write: list[dict[str, Any]] = []
                            for row in task_rows:
                                payload = {"cohort": cohort, "task": task, "position": position, "layer": int(layer), "fold": int(fold), "no_sa_case_id": row["case_id"], "joint_case_id": row["joint_case_id"], "item_id": row["item_id"], "prior_index": int(row["prior_index"]), "condition": row["condition"]}
                                if task == "hard_label":
                                    payload.update({"true_label": row["true_label"], "predicted_label": row["predicted_label"], "class_probabilities": row["class_probabilities"], "majority_label": majority, "majority_correct": bool(majority == row["true_label"])})
                                else:
                                    payload.update({"true_score": row["true_score"], "predicted_score": row["predicted_score"]})
                                rows_to_write.append(payload)
                            predictions_path.parent.mkdir(parents=True, exist_ok=True)
                            with predictions_path.open("a", encoding="utf-8") as handle:
                                for payload in rows_to_write:
                                    key = prediction_key(payload)
                                    if key not in existing_keys:
                                        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                                        existing_keys.add(key)
                                handle.flush()
                            completed_jobs.add(identifier)
                            job_audits[identifier]["model_diagnostics"] = diagnostics
                        except Exception as exc:
                            invalid_jobs[identifier] = {"type": type(exc).__name__, "message": str(exc)}
                        checkpoint("running")
                del matrix
                gc.collect()
    checkpoint("training_complete")
    run_config.update({"status": "training_complete", "completed_job_count": len(completed_jobs), "invalid_job_count": len(invalid_jobs), "prediction_count": len(existing_keys), "hidden_loader_shard_load_count": loader.shard_load_count, "training_seconds": float(time.perf_counter() - started)})
    atomic_write_json(output_dir / "run_config.json", run_config)
    return {"status": "training_complete", "completed_jobs": len(completed_jobs), "invalid_jobs": len(invalid_jobs), "predictions": len(existing_keys), "output_dir": str(output_dir)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_training(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
