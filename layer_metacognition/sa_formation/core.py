"""Artifact discovery, OOF Ridge directions, intervention algebra, and statistics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from layer_metacognition.hidden_state_store import atomic_write_json, atomic_write_text, load_jsonl
from layer_metacognition.steering.decision_side_steering import BaselineHiddenStateRepository


FORMAT_VERSION = 1
PRIMARY_LAYER = 18
PRIMARY_POSITION = "panl"
SEED = 42
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
EXPERIMENT_DIR_NAMES = {
    0: "00_natural_state",
    1: "01_evidence_baseline",
    2: "02_history",
    3: "03_answer_mismatch",
    4: "04_mediation",
    5: "05_future_policy",
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_save_npz(path: str | Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class SAFormationArtifacts:
    experiment_dir: Path
    results: Path
    hidden_index: Path
    manifest: Path
    text_labels: Path
    image_labels: Path
    item_split: Path
    answer_force_manifest: Path
    answer_force_results: Path
    decision_direction_dir: Path
    dataset: Path
    model_path: Path
    inference_path: Path

    @classmethod
    def discover(cls, experiment_dir: str | Path) -> "SAFormationArtifacts":
        root = Path(experiment_dir).resolve()
        config_path = root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Base experiment config is missing: {config_path}")
        config = read_json(config_path)
        if config.get("versions") != ["v4"] or config.get("attribution_mode") != "joint":
            raise ValueError("Stage 3 primary requires v4 / joint base artifacts")
        if config.get("source_prompt_variant") != "answer_basis_9":
            raise ValueError("Stage 3 primary requires answer_basis_9")
        paths = cls(
            experiment_dir=root,
            results=root / "results.jsonl",
            hidden_index=root / "hidden_states" / "index.json",
            manifest=root / "extended_probe" / "probe_manifest.jsonl",
            text_labels=root / "extended_probe" / "text_only_labels.jsonl",
            image_labels=root / "extended_probe" / "image_only_labels.jsonl",
            item_split=root / "stage1_metacognition" / "item_split" / "split_assignments.json",
            answer_force_manifest=root / "stage2_teacher_forced_source_origin" / "cohort_manifest.json",
            answer_force_results=root / "stage2_teacher_forced_source_origin" / "results.jsonl",
            decision_direction_dir=root / "stage1_metacognition" / "item_split",
            dataset=Path(config["dataset"]).resolve(),
            model_path=Path(config["model_path"]).resolve(),
            inference_path=Path(config["inference_path"]).resolve(),
        )
        required = [
            paths.results,
            paths.hidden_index,
            paths.manifest,
            paths.text_labels,
            paths.image_labels,
            paths.item_split,
            paths.answer_force_manifest,
            paths.answer_force_results,
            paths.dataset,
            paths.inference_path,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if not paths.model_path.is_dir():
            missing.append(str(paths.model_path))
        if missing:
            raise FileNotFoundError("Required Stage 3 inputs are missing: " + ", ".join(missing))
        return paths

    def provenance(self) -> dict[str, Any]:
        files = {
            "base_config": self.experiment_dir / "config.json",
            "base_results": self.results,
            "hidden_index": self.hidden_index,
            "probe_manifest": self.manifest,
            "text_labels": self.text_labels,
            "image_labels": self.image_labels,
            "item_split": self.item_split,
            "answer_force_manifest": self.answer_force_manifest,
            "answer_force_results": self.answer_force_results,
        }
        return {
            name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        }


def validate_output_dir(experiment_dir: str | Path, output_dir: str | Path) -> Path:
    root = Path(experiment_dir).resolve()
    output = Path(output_dir).resolve()
    expected = root / "stage3_sa_formation"
    if output != expected:
        raise ValueError(f"Stage 3 output is fixed to {expected}; got {output}")
    protected = [
        root / "results.jsonl",
        root / "hidden_states",
        root / "stage1_metacognition",
    ] + list(root.glob("stage2_*"))
    if any(output == path.resolve() or path.resolve().is_relative_to(output) for path in protected):
        raise ValueError("Stage 3 output would contain a protected Stage 1/2 artifact")
    return output


def initialize_run(
    output_dir: str | Path,
    configuration: dict[str, Any],
    *,
    resume: bool,
) -> str:
    output = Path(output_dir)
    fingerprint = stable_hash(configuration)
    config_path = output / "run_config.json"
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Stage 3 output exists; pass --resume: {output}")
    if config_path.exists():
        old = read_json(config_path)
        if old.get("config_fingerprint") != fingerprint:
            raise ValueError("Stage 3 resume configuration fingerprint mismatch")
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(configuration)
    payload.update({"format_version": FORMAT_VERSION, "config_fingerprint": fingerprint})
    atomic_write_json(config_path, payload)
    return fingerprint


def load_baseline_rows(artifacts: SAFormationArtifacts) -> list[dict[str, Any]]:
    results = {
        str(row["case_id"]): row
        for row in load_jsonl(artifacts.results)
        if row.get("status") == "completed"
    }
    manifests = {
        str(row["case_id"]): row
        for row in load_jsonl(artifacts.manifest)
        if row.get("version") == "v4" and str(row.get("case_id", "")).endswith("__v4__joint")
    }
    split = read_json(artifacts.item_split)
    item_to_fold = {str(k): int(v) for k, v in split["item_to_fold"].items()}
    rows: list[dict[str, Any]] = []
    for case_id in sorted(set(results).intersection(manifests)):
        result = results[case_id]
        source = result.get("generated", {}).get("source_attribution")
        if not isinstance(source, dict) or source.get("soft_image_score") is None:
            continue
        manifest = manifests[case_id]
        item_id = str(manifest["item_id"])
        if item_id not in item_to_fold:
            raise ValueError(f"No item fold for baseline case {case_id}")
        rows.append(
            {
                "case_id": case_id,
                "item_id": item_id,
                "prior_index": int(manifest["prior_index"]),
                "condition": str(manifest["condition"]),
                "difficulty": str(manifest["condition"]).rsplit("_", 1)[-1],
                "fold": item_to_fold[item_id],
                "answer_classes": list(manifest["answer_classes"]),
                "text_answer": manifest.get("text_only_answer"),
                "image_answer": manifest.get("image_only_answer"),
                "final_answer": manifest.get("current_answer"),
                "decision_side": manifest.get("decision_side"),
                "source_label": source.get("parsed_label"),
                "sa": float(source["soft_image_score"]),
                "source_probabilities": source.get("class_probabilities"),
                "manifest": manifest,
                "baseline": result,
            }
        )
    if not rows:
        raise ValueError("No completed V4 joint baseline rows with SA")
    return rows


@dataclass(frozen=True)
class FoldDirection:
    fold: int
    alpha: float
    d_raw: np.ndarray
    d_unit: np.ndarray
    raw_intercept: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    sigma_z: float
    sign_flipped: bool

    def z(self, hidden: np.ndarray) -> float:
        return float(np.asarray(hidden, dtype=np.float64) @ self.d_unit)

    def predict(self, hidden: np.ndarray) -> float:
        return float(np.asarray(hidden, dtype=np.float64) @ self.d_raw + self.raw_intercept)


class SAOOFDirectionRepository:
    def __init__(self, direction_dir: str | Path) -> None:
        self.direction_dir = Path(direction_dir)
        index_path = self.direction_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"SA direction index missing: {index_path}")
        self.index = read_json(index_path)
        self._cache: dict[int, FoldDirection] = {}

    def get(self, fold: int) -> FoldDirection:
        fold = int(fold)
        if fold in self._cache:
            return self._cache[fold]
        entry = next((x for x in self.index["folds"] if int(x["fold"]) == fold), None)
        if entry is None:
            raise KeyError(f"No SA direction for fold {fold}")
        payload = np.load(self.direction_dir / entry["file"])
        direction = FoldDirection(
            fold=fold,
            alpha=float(payload["alpha"]),
            d_raw=np.asarray(payload["d_raw"], dtype=np.float64),
            d_unit=np.asarray(payload["d_unit"], dtype=np.float64),
            raw_intercept=float(payload["raw_intercept"]),
            scaler_mean=np.asarray(payload["scaler_mean"], dtype=np.float64),
            scaler_scale=np.asarray(payload["scaler_scale"], dtype=np.float64),
            sigma_z=float(payload["sigma_z"]),
            sign_flipped=bool(payload["sign_flipped"]),
        )
        if not math.isclose(float(np.linalg.norm(direction.d_unit)), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Fold {fold} SA direction is not unit norm")
        if direction.sigma_z <= 0:
            raise ValueError(f"Fold {fold} sigma_z must be positive")
        self._cache[fold] = direction
        return direction


def ridge_raw_space(
    scaler: StandardScaler,
    ridge: RidgeCV,
) -> tuple[np.ndarray, float]:
    coef = np.asarray(ridge.coef_, dtype=np.float64).reshape(-1)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    raw = coef / scale
    intercept = float(ridge.intercept_ - np.dot(mean, raw))
    return raw, intercept


def fit_oof_directions(
    artifacts: SAFormationArtifacts,
    output_dir: str | Path,
    *,
    layer: int = PRIMARY_LAYER,
    position: str = PRIMARY_POSITION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_baseline_rows(artifacts)
    hidden_repo = BaselineHiddenStateRepository(artifacts.experiment_dir)
    hidden = np.stack(
        [hidden_repo.get(row["manifest"], layer, position) for row in rows], axis=0
    )
    targets = np.asarray([row["sa"] for row in rows], dtype=np.float64)
    folds = np.asarray([row["fold"] for row in rows], dtype=np.int64)
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    coordinates = np.full(len(rows), np.nan, dtype=np.float64)
    output = Path(output_dir)
    direction_dir = output / "directions"
    audits: list[dict[str, Any]] = []
    index_entries: list[dict[str, Any]] = []
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = folds == fold
        train_items = {rows[i]["item_id"] for i in np.flatnonzero(train)}
        test_items = {rows[i]["item_id"] for i in np.flatnonzero(test)}
        overlap = sorted(train_items.intersection(test_items))
        if overlap:
            raise RuntimeError(f"OOF item leakage in fold {fold}: {overlap[:5]}")
        scaler = StandardScaler().fit(hidden[train])
        ridge = RidgeCV(alphas=np.asarray(RIDGE_ALPHAS), scoring="neg_mean_squared_error")
        ridge.fit(scaler.transform(hidden[train]), targets[train])
        d_raw, raw_intercept = ridge_raw_space(scaler, ridge)
        norm = float(np.linalg.norm(d_raw))
        if norm <= 0 or not np.isfinite(norm):
            raise RuntimeError(f"Degenerate Ridge direction in fold {fold}")
        d_unit = d_raw / norm
        sign_flipped = bool(np.corrcoef(hidden[train] @ d_unit, targets[train])[0, 1] < 0)
        if sign_flipped:
            d_unit = -d_unit
            d_raw = -d_raw
            raw_intercept = float(np.mean(targets[train] - hidden[train] @ d_raw))
        train_z = hidden[train] @ d_unit
        sigma_z = float(np.std(train_z, ddof=1))
        if sigma_z <= 0 or not np.isfinite(sigma_z):
            raise RuntimeError(f"Invalid training-only sigma_z in fold {fold}")
        predictions[test] = hidden[test] @ d_raw + raw_intercept
        coordinates[test] = hidden[test] @ d_unit
        file_name = f"fold_{fold}_layer_{layer}_{position}.npz"
        atomic_save_npz(
            direction_dir / file_name,
            alpha=np.asarray(float(ridge.alpha_)),
            d_raw=d_raw,
            d_unit=d_unit,
            raw_intercept=np.asarray(raw_intercept),
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            sigma_z=np.asarray(sigma_z),
            sign_flipped=np.asarray(sign_flipped),
        )
        audit = {
            "fold": fold,
            "train_n": int(train.sum()),
            "test_n": int(test.sum()),
            "train_item_count": len(train_items),
            "test_item_count": len(test_items),
            "item_overlap": overlap,
            "selected_alpha": float(ridge.alpha_),
            "sigma_z": sigma_z,
            "sigma_source": "fold_training_items_only",
            "sign_flipped": sign_flipped,
            "test_r2": float(r2_score(targets[test], predictions[test])),
            "test_mae": float(mean_absolute_error(targets[test], predictions[test])),
        }
        audits.append(audit)
        index_entries.append({"fold": fold, "file": file_name, **audit})
    if not (np.isfinite(predictions).all() and np.isfinite(coordinates).all()):
        raise RuntimeError("OOF directions did not cover every row")
    oof: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        record = {key: row[key] for key in row if key not in {"manifest", "baseline"}}
        record.update(
            {
                "hidden_state_reference": row["manifest"]["hidden_state_reference"],
                "z_sa": float(coordinates[index]),
                "ridge_prediction": float(predictions[index]),
                "residual": float(row["sa"] - predictions[index]),
            }
        )
        oof.append(record)
    atomic_write_json(
        direction_dir / "index.json",
        {
            "format_version": FORMAT_VERSION,
            "definition": "StandardScaler + RidgeCV OOF continuous SA direction",
            "layer": layer,
            "position": position,
            "alphas": list(RIDGE_ALPHAS),
            "target": "baseline soft_image_score",
            "folds": index_entries,
        },
    )
    return oof, {"fold_audits": audits}


def coordinate_delta(hidden: np.ndarray, direction_unit: np.ndarray, target_z: float) -> np.ndarray:
    h = np.asarray(hidden, dtype=np.float64)
    unit = np.asarray(direction_unit, dtype=np.float64)
    if not math.isclose(float(np.linalg.norm(unit)), 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("direction_unit must have unit L2 norm")
    return (float(target_z) - float(h @ unit)) * unit


def transplant_delta(recipient: np.ndarray, donor: np.ndarray, direction_unit: np.ndarray) -> np.ndarray:
    return coordinate_delta(recipient, direction_unit, float(np.asarray(donor) @ direction_unit))


def orthogonal_equal_norm_control(
    direction_unit: np.ndarray,
    target_l2: float,
    *,
    seed_material: str,
) -> np.ndarray:
    unit = np.asarray(direction_unit, dtype=np.float64)
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(unit.shape)
    vector = vector - float(vector @ unit) * unit
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise RuntimeError("Could not construct orthogonal control")
    return vector / norm * float(target_l2)


def item_cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    statistic,
    *,
    iterations: int = 1000,
    seed: int = SEED,
) -> dict[str, Any]:
    items = sorted({str(row["item_id"]) for row in rows})
    by_item = {item: [row for row in rows if str(row["item_id"]) == item] for item in items}
    observed = float(statistic(rows))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        chosen = rng.choice(items, size=len(items), replace=True)
        sample = [row for item in chosen for row in by_item[str(item)]]
        value = float(statistic(sample))
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        return {"estimate": observed, "ci95": [None, None], "iterations": iterations, "valid": 0}
    return {
        "estimate": observed,
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "iterations": iterations,
        "valid": len(samples),
    }


def paired_effect_summary(
    rows: Sequence[dict[str, Any]],
    value_key: str,
    *,
    iterations: int = 1000,
) -> dict[str, Any]:
    valid = [row for row in rows if row.get(value_key) is not None and np.isfinite(row[value_key])]
    if not valid:
        return {"n": 0, "mean": None, "sd": None, "ci95": [None, None], "direction_rate": None}
    values = [float(row[value_key]) for row in valid]
    boot = item_cluster_bootstrap(valid, lambda sample: statistics.fmean(float(r[value_key]) for r in sample), iterations=iterations)
    return {
        "n": len(valid),
        "unique_items": len({str(row["item_id"]) for row in valid}),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "ci95": boot["ci95"],
        "direction_rate": sum(value > 0 for value in values) / len(values),
    }


def natural_projection_summary(oof: Sequence[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([row["sa"] for row in oof], dtype=np.float64)
    z = np.asarray([row["z_sa"] for row in oof], dtype=np.float64)
    pred = np.asarray([row["ridge_prediction"] for row in oof], dtype=np.float64)
    spearman = float(spearmanr(z, y).statistic)
    spearman_boot = item_cluster_bootstrap(
        oof,
        lambda rows: spearmanr(
            [row["z_sa"] for row in rows], [row["sa"] for row in rows]
        ).statistic,
    )
    quintile_edges = np.quantile(z, np.linspace(0, 1, 6))
    quintiles: list[dict[str, Any]] = []
    for index in range(5):
        mask = (z >= quintile_edges[index]) & (
            z <= quintile_edges[index + 1] if index == 4 else z < quintile_edges[index + 1]
        )
        quintiles.append(
            {"quintile": index + 1, "n": int(mask.sum()), "mean_z": float(z[mask].mean()), "mean_sa": float(y[mask].mean())}
        )
    fold_metrics = []
    for fold in sorted({int(row["fold"]) for row in oof}):
        selected = [row for row in oof if int(row["fold"]) == fold]
        fy = np.asarray([row["sa"] for row in selected])
        fp = np.asarray([row["ridge_prediction"] for row in selected])
        fz = np.asarray([row["z_sa"] for row in selected])
        fold_metrics.append(
            {
                "fold": fold,
                "n": len(selected),
                "pearson": float(pearsonr(fz, fy).statistic),
                "spearman": float(spearmanr(fz, fy).statistic),
                "r2": float(r2_score(fy, fp)),
                "mae": float(mean_absolute_error(fy, fp)),
            }
        )
    r2 = float(r2_score(y, pred))
    effective = bool(r2 > 0 and spearman_boot["ci95"][0] is not None and spearman_boot["ci95"][0] > 0)
    return {
        "n": len(oof),
        "unique_items": len({str(row["item_id"]) for row in oof}),
        "pearson": float(pearsonr(z, y).statistic),
        "spearman": spearman,
        "spearman_item_bootstrap": spearman_boot,
        "r2": r2,
        "mae": float(mean_absolute_error(y, pred)),
        "z_quintiles": quintiles,
        "quintile_monotonic": all(quintiles[i]["mean_sa"] <= quintiles[i + 1]["mean_sa"] for i in range(4)),
        "fold_metrics": fold_metrics,
        "natural_effective": effective,
        "criterion": "OOF R2 > 0 and item-cluster bootstrap Spearman CI lower > 0",
    }


@dataclass(frozen=True)
class GateDecision:
    level: int
    natural_effective: bool
    transplant_effective: bool
    run_natural_formation: bool
    allow_causal_mediator: bool
    allow_policy_steering: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_gate(natural_effective: bool, transplant_summary: dict[str, Any] | None) -> GateDecision:
    if not natural_effective:
        return GateDecision(3, False, False, False, False, False, "Natural OOF projection failed")
    transplant_effective = bool(transplant_summary and transplant_summary.get("coordinate_effective"))
    if transplant_effective:
        return GateDecision(1, True, True, True, True, True, "Natural projection and coordinate transplant passed")
    return GateDecision(2, True, False, True, False, False, "Natural projection passed but coordinate transplant failed")


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def write_csv_atomic(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    destination = Path(path)
    keys = sorted({key for row in rows for key in row}) if rows else []
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_experiment_summary(directory: str | Path, summary: dict[str, Any]) -> None:
    target = Path(directory)
    atomic_write_json(target / "summary.json", summary)
    lines = [f"# {summary.get('title', target.name)}", ""]
    lines.append(f"- Status: `{summary.get('status', 'unknown')}`")
    if summary.get("n") is not None:
        lines.append(f"- Effective n: `{summary['n']}`")
    if summary.get("reason"):
        lines.append(f"- Reason: {summary['reason']}")
    lines.extend(["", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    atomic_write_text(target / "summary.md", "\n".join(lines))


def canonical_message_hash(messages: Sequence[dict[str, Any]]) -> str:
    return stable_hash(list(messages))


def assert_endpoint_evidence_equal(left_messages: Sequence[dict[str, Any]], right_messages: Sequence[dict[str, Any]]) -> None:
    left = [message for message in left_messages if message.get("role") == "user"][-1]
    right = [message for message in right_messages if message.get("role") == "user"][-1]
    if canonical_message_hash([left]) != canonical_message_hash([right]):
        raise ValueError("TF/IF final evidence user turn differs")


def assert_policy_no_verbal_sa(messages: Sequence[dict[str, Any]]) -> None:
    for message in messages[:-1]:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict)) if isinstance(content, list) else str(content)
        if "Source Attribution" in text:
            raise ValueError("Policy branch assistant history leaks verbal SA")
