from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from dp_SA.config import MODEL_PATH, ROOT, SEED

from .io_utils import array_hash, atomic_csv, atomic_json, atomic_jsonl, atomic_npz, canonical_hash, inventory, load_jsonl, sha256_file, verify_inventory
from .train_template_sa_probes import DEFAULT_ROOT as PROBE_RUN_ROOT, TRAJECTORY_ROOT
from .config import TEST_MANIFEST


TEMPLATES = ("T0", "T1", "T2", "T3")
DEFAULT_NODES = (("P1_LAT", 14), ("P1_LAT", 15), ("P1_PANL", 15), ("P1_PANL", 16), ("P1_PANL", 17))
PILOT_NODE = ("P1_LAT", 14)
BOOTSTRAPS = 2000
RANDOM_SUBSPACE_RANKS = (64, 128, 256, 512)
RANDOM_SD_TOLERANCE = .05
T0_HIDDEN_ROOT = TRAJECTORY_ROOT


def node_name(position: str, layer: int) -> str:
    return f"{position}__L{int(layer)}"


def parse_nodes(values: Sequence[str]) -> tuple[tuple[str, int], ...]:
    output = []
    for value in values:
        try:
            position, layer = value.rsplit(":", 1); item = (position, int(layer))
        except Exception as exc:
            raise ValueError(f"Invalid node {value!r}; expected POSITION:LAYER") from exc
        if item not in DEFAULT_NODES: raise ValueError(f"Node is outside preregistered grid: {value}")
        output.append(item)
    if tuple(output) != DEFAULT_NODES: raise ValueError(f"Nodes must be exactly {DEFAULT_NODES}")
    return tuple(output)


def uncentered_shared_axis(units: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.stack([np.asarray(value, np.float64) for value in units])
    if matrix.ndim != 2 or not np.isfinite(matrix).all(): raise ValueError("Invalid direction matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-10): raise ValueError("SVD inputs must be unit directions")
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False); axis = vt[0]
    if float(np.mean(matrix @ axis)) < 0: axis = -axis
    return axis.astype(np.float64), singular.astype(np.float64)


def equal_dose_vector(unit: np.ndarray, sigma_loto: float, alpha: float) -> np.ndarray:
    value = np.asarray(unit, np.float64)
    norm = float(np.linalg.norm(value))
    # Persisted axes are float32, so their round-trip norm error is naturally
    # around 1e-9.  Reject genuinely non-unit inputs, then renormalize in
    # float64 before constructing the dose so the comparison is exact.
    if not math.isclose(norm, 1.0, abs_tol=1e-6): raise ValueError("Direction is not unit norm")
    value = value / norm
    if not sigma_loto > 0: raise ValueError("Natural projection SD must be positive")
    return (float(alpha) * float(sigma_loto) * value).astype(np.float32)


def retention_eligible(point: float, boot: np.ndarray) -> dict[str, Any]:
    values = np.asarray(boot, float); low, high = np.quantile(values, [.025, .975]); same = float(np.mean(np.sign(values) == np.sign(point)))
    return {"eligible": bool(point != 0 and not (low <= 0 <= high) and same >= .975), "ci_low": float(low), "ci_high": float(high), "same_sign_fraction": same}


def _safe_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, float); prediction = np.asarray(prediction, float)
    return {
        "r2": float(r2_score(y, prediction)),
        "pearson": float(pearsonr(y, prediction).statistic) if np.ptp(y) and np.ptp(prediction) else math.nan,
        "spearman": float(spearmanr(y, prediction).statistic) if len(set(y)) > 1 and len(set(prediction)) > 1 else math.nan,
        "mae": float(mean_absolute_error(y, prediction)),
    }


def _family_draws(rows: Sequence[dict[str, Any]], repeats: int = BOOTSTRAPS) -> tuple[list[str], np.ndarray]:
    families = sorted({str(row["family_id"]) for row in rows}); rng = np.random.default_rng(SEED + 731)
    return families, rng.integers(0, len(families), size=(repeats, len(families)))


def _bootstrap_metrics(y: np.ndarray, prediction: np.ndarray, row_families: Sequence[str], families: Sequence[str], draws: np.ndarray) -> dict[str, float]:
    fa = np.asarray(row_families); by = {family: np.flatnonzero(fa == family) for family in families}; values = {key: [] for key in ("r2", "pearson", "spearman", "mae")}
    for draw in draws:
        indices = np.concatenate([by[families[int(index)]] for index in draw]); measured = _safe_metrics(y[indices], prediction[indices])
        for key, value in measured.items():
            if np.isfinite(value): values[key].append(value)
    output = {}
    for key, observed in values.items():
        low, high = np.quantile(observed, [.025, .975]); output[f"{key}_ci_low"] = float(low); output[f"{key}_ci_high"] = float(high)
    return output


class HiddenRepository:
    def __init__(self) -> None:
        self.prompt_rows = {template: {str(row["case_id"]): row for row in load_jsonl(PROBE_RUN_ROOT / f"artifacts/diagnostics/capture.{template}.jsonl")} for template in TEMPLATES}
        self.t0_reuse = {str(row["case_id"]): row for row in load_jsonl(T0_HIDDEN_ROOT / "artifacts/clean_hidden/reuse_manifest.jsonl")}
        self.t0_capture = {str(row["case_id"]): row for row in load_jsonl(T0_HIDDEN_ROOT / "artifacts/clean_hidden/capture_manifest.jsonl")}
        self.verified_files: set[str] = set()

    def label(self, template: str, case: str) -> float:
        return float(self.prompt_rows[template][str(case)]["canonical_soft_sa"])

    def _verified_npz(self, path: Path, digest: str | None, key: str, tensor_digest: str | None) -> np.ndarray:
        resolved = str(path.resolve())
        if digest and resolved not in self.verified_files:
            if sha256_file(path) != digest: raise ValueError(f"Hidden file hash changed: {path}")
            self.verified_files.add(resolved)
        with np.load(path) as payload:
            if key not in payload.files: raise KeyError(f"{key} missing from {path}")
            value = np.asarray(payload[key])
        if value.shape != (3584,) or value.dtype != np.float16 or not np.isfinite(value).all(): raise ValueError(f"Invalid hidden {path}:{key}")
        if tensor_digest and array_hash(value) != tensor_digest: raise ValueError(f"Hidden tensor hash changed: {path}:{key}")
        return value.astype(np.float32)

    def load(self, template: str, case: str, position: str, layer: int) -> np.ndarray:
        case = str(case); key = node_name(position, layer)
        if template != "T0":
            row = self.prompt_rows[template][case]; path = PROBE_RUN_ROOT / row["hidden_file"]
            return self._verified_npz(path, row["hidden_sha256"], key, row["hidden_tensor_sha256"][key])
        reuse = self.t0_reuse[case]; source = reuse.get("cell_sources", {}).get(key)
        if source:
            return self._verified_npz(Path(source["path"]), source.get("file_sha256"), key, source.get("tensor_sha256"))
        row = self.t0_capture[case]; path = T0_HIDDEN_ROOT / row["delta_file"]
        return self._verified_npz(path, row.get("delta_file_sha256"), key, None)

    def matrix(self, template: str, rows: Sequence[dict[str, Any]], position: str, layer: int) -> np.ndarray:
        return np.stack([self.load(template, str(row["case_id"]), position, layer) for row in rows]).astype(np.float32)


def _probe_index() -> list[dict[str, Any]]:
    rows = load_jsonl(PROBE_RUN_ROOT / "artifacts/probes/probe_index.jsonl")
    if len(rows) != 48: raise ValueError(f"Expected 48 probe cells, got {len(rows)}")
    return rows


def load_unit_probes(nodes: Sequence[tuple[str, int]]) -> tuple[dict[tuple[str, str, int], np.ndarray], list[dict[str, Any]], list[Path]]:
    index = _probe_index(); units = {}; audits = []; paths = [PROBE_RUN_ROOT / "artifacts/probes/probe_index.jsonl"]
    expected_construction = [str(row["case_id"]) for row in load_jsonl(PROBE_RUN_ROOT / "artifacts/manifests/construction_manifest.jsonl")]
    for template in TEMPLATES:
        for position, layer in nodes:
            matches = [row for row in index if row["template"] == template and row["position"] == position and int(row["layer"]) == layer]
            if len(matches) != 1: raise ValueError(f"Probe identity ambiguous: {template} {position} L{layer}")
            row = matches[0]; path = Path(row["probe_file"]); path = path if path.is_absolute() else PROBE_RUN_ROOT / path
            if sha256_file(path) != row["probe_sha256"]: raise ValueError(f"Probe hash changed: {path}")
            payload = joblib.load(path); model = payload["model"]
            if payload.get("target") not in {"canonical_soft_sa", "final_soft_sa"}: raise ValueError("Probe target is not canonical SA")
            if list(map(str, payload.get("construction_case_ids", []))) != expected_construction: raise ValueError("Probe construction IDs changed")
            calculated = np.asarray(model.named_steps["ridge"].coef_, np.float64).reshape(-1) / np.asarray(model.named_steps["scale"].scale_, np.float64)
            stored = np.asarray(payload["raw_weight"], np.float64)
            error = float(np.max(np.abs(calculated - stored)))
            if error > 1e-12: raise ValueError(f"Raw weight parity failed: {error}")
            unit = stored / np.linalg.norm(stored); units[template, position, layer] = unit
            audits.append({"template": template, "position": position, "layer": layer, "target": payload["target"], "canonical_image_positive": True, "extra_sign_flip": False, "raw_weight_parity_max_abs_error": error, "raw_weight_norm": float(np.linalg.norm(stored)), "unit_norm": float(np.linalg.norm(unit)), "probe_file": str(path.resolve()), "probe_sha256": row["probe_sha256"], "raw_weight_sha256": array_hash(stored)})
            paths.append(path)
    return units, audits, paths


def matched_random_directions(hidden: np.ndarray, target_sd: float, *, seed: int, count: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Construct random unit directions with exactly matched empirical projection SD.

    A seeded random orthogonal subspace is independent of SA.  Within that
    subspace, covariance eigenvectors diagonalize projection variance, so an
    analytic mixture of one eigenvector below and one above the target variance
    simultaneously preserves unit norm and matches the target SD.
    """
    centered = np.asarray(hidden, np.float64); centered = centered - centered.mean(axis=0, keepdims=True)
    if centered.ndim != 2 or centered.shape[0] < 2 or not target_sd > 0: raise ValueError("Invalid hidden matrix or target SD")
    target_variance = float(target_sd) ** 2; chosen = None
    for rank in RANDOM_SUBSPACE_RANKS:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xffffffff, int(seed) >> 32, rank])); gaussian = rng.standard_normal((centered.shape[1], rank)); basis, _ = np.linalg.qr(gaussian, mode="reduced"); projected = centered @ basis; covariance = projected.T @ projected / (centered.shape[0] - 1); eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        below = np.flatnonzero(eigenvalues < target_variance); above = np.flatnonzero(eigenvalues > target_variance)
        if len(below) and len(above): chosen = (rank, rng, basis, eigenvalues, eigenvectors, below, above); break
    if chosen is None: raise RuntimeError(f"Target variance {target_variance} is outside all seeded random-subspace spectra")
    rank, rng, basis, eigenvalues, eigenvectors, below, above = chosen; pairs = [(int(low), int(high)) for low in below for high in above]; order = rng.permutation(len(pairs)); vectors = []; metadata = []
    for replicate, pair_index in enumerate(order[:count], start=1):
        low, high = pairs[int(pair_index)]; low_value = float(eigenvalues[low]); high_value = float(eigenvalues[high]); low_weight = (high_value - target_variance) / (high_value - low_value); sign = -1.0 if int(rng.integers(0, 2)) else 1.0
        coordinates = math.sqrt(low_weight) * eigenvectors[:, low] + sign * math.sqrt(1.0 - low_weight) * eigenvectors[:, high]; vector = basis @ coordinates; vector /= np.linalg.norm(vector); projection_sd = float(np.std(centered @ vector, ddof=1)); error = abs(projection_sd / float(target_sd) - 1.0)
        if error > 1e-8: raise RuntimeError(f"Analytic random SD match failed: {error}")
        vectors.append(vector); metadata.append({"replicate": replicate, "candidate_id": int(pair_index), "random_subspace_rank": rank, "low_eigenvalue": low_value, "high_eigenvalue": high_value, "projection_sd": projection_sd, "target_sd": float(target_sd), "relative_sd_error": error, "unit_norm": float(np.linalg.norm(vector)), "vector_sha256": array_hash(vector)})
    if len(vectors) != count: raise RuntimeError(f"Random subspace provided {len(vectors)}/{count} distinct matched directions")
    return np.stack(vectors), metadata


def source_files() -> list[Path]:
    paths = [PROBE_RUN_ROOT / "artifacts/probes/probe_index.jsonl", PROBE_RUN_ROOT / "artifacts/manifests/construction_manifest.jsonl", PROBE_RUN_ROOT / "artifacts/manifests/audit_manifest.jsonl", PROBE_RUN_ROOT / "artifacts/config_and_fingerprint.json", T0_HIDDEN_ROOT / "artifacts/clean_hidden/reuse_manifest.jsonl", T0_HIDDEN_ROOT / "artifacts/clean_hidden/capture_manifest.jsonl", TEST_MANIFEST, MODEL_PATH / "config.json", MODEL_PATH / "tokenizer.json", MODEL_PATH / "preprocessor_config.json"]
    paths.extend(PROBE_RUN_ROOT / f"artifacts/diagnostics/capture.{template}.jsonl" for template in TEMPLATES)
    return paths


def prepare_cpu(root: Path, nodes: Sequence[tuple[str, int]], random_count: int, *, validation_cases: int = 100, resume: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True); construction = load_jsonl(PROBE_RUN_ROOT / "artifacts/manifests/construction_manifest.jsonl"); audit = load_jsonl(PROBE_RUN_ROOT / "artifacts/manifests/audit_manifest.jsonl")
    if (len(construction), len(audit), len({row['family_id'] for row in construction}), len({row['family_id'] for row in audit})) != (882, 230, 103, 25): raise ValueError("Frozen 882/230 split changed")
    units, probe_audit, probe_paths = load_unit_probes(nodes); sources = inventory([*source_files(), *probe_paths]); code = inventory([Path(__file__), Path(__file__).with_name("shared_axis_steering.py"), Path(__file__).with_name("run_shared_axis_loto.py"), Path(__file__).with_name("runtime.py"), Path(__file__).with_name("scoring.py"), Path(__file__).with_name("templates.py")])
    config = {"format_version": 2, "experiment": "t0_t3_loto_shared_sa_axis_equal_dose", "nodes": [node_name(*node) for node in nodes], "pilot_node": node_name(*PILOT_NODE), "templates": list(TEMPLATES), "construction_cases": 882, "audit_cases": 230, "validation_cases": validation_cases, "alphas": [-2.0, 0.0, 2.0], "random_directions": random_count, "random_method": "seeded_random_subspace_analytic_variance_match", "random_subspace_ranks": list(RANDOM_SUBSPACE_RANKS), "random_sd_tolerance": RANDOM_SD_TOLERANCE, "bootstrap_repeats": BOOTSTRAPS, "seed": SEED, "source_hashes": sources, "implementation_hashes": code}
    fingerprint = canonical_hash(config); config_path = root / "artifacts/config_and_fingerprint.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old.get("fingerprint") != fingerprint: raise ValueError("Resume configuration fingerprint mismatch")
        if not resume: raise FileExistsError(config_path)
    atomic_json(config_path, {**config, "fingerprint": fingerprint}); atomic_json(root / "artifacts/source_hashes_before.json", sources); atomic_jsonl(root / "tables/probe_direction_audit.jsonl", probe_audit)
    progress_path = root / "progress/cpu_prepare.json"
    if resume and progress_path.exists():
        prior = json.loads(progress_path.read_text())
        if prior.get("status") == "complete" and prior.get("fingerprint") == fingerprint and all((root / f"artifacts/axes/{node_name(*node)}.npz").is_file() for node in nodes): return {**prior, "resumed_noop": True}
    repository = HiddenRepository(); families, draws = _family_draws(audit); atomic_json(root / "artifacts/bootstrap/audit_family_draws.json", {"seed": SEED + 731, "repeats": len(draws), "ordered_families": families, "draws": draws.tolist(), "fingerprint": canonical_hash(draws.tolist()), "shared": True})
    geometry = []; pairwise = []; prediction_rows = []; random_audit = []; axes_hashes = {}
    matrices: dict[tuple[str, str, int, str], np.ndarray] = {}
    for position, layer in nodes:
        for template in TEMPLATES:
            matrices[template, position, layer, "construction"] = repository.matrix(template, construction, position, layer)
            matrices[template, position, layer, "audit"] = repository.matrix(template, audit, position, layer)
        all_units = [units[template, position, layer] for template in TEMPLATES]; shared, singular = uncentered_shared_axis(all_units); cosines = np.stack(all_units) @ shared
        energy = singular ** 2 / np.sum(singular ** 2)
        geometry.append({"position": position, "layer": layer, **{f"cosine_{template}": float(cosines[index]) for index, template in enumerate(TEMPLATES)}, "minimum_shared_cosine": float(np.min(cosines)), "sigma1": float(singular[0]), "sigma2": float(singular[1]), "sigma3": float(singular[2]), "sigma4": float(singular[3]), "first_axis_energy": float(energy[0]), "first_two_axes_energy": float(energy[:2].sum()), "sigma1_over_sigma2": float(singular[0] / singular[1]), "subspace_flag": bool(energy[0] < .60 and energy[:2].sum() > energy[0])})
        for left_index, left in enumerate(TEMPLATES):
            for right in TEMPLATES[left_index + 1:]: pairwise.append({"position": position, "layer": layer, "template_a": left, "template_b": right, "signed_cosine": float(units[left, position, layer] @ units[right, position, layer]), "absolute_cosine": abs(float(units[left, position, layer] @ units[right, position, layer]))})
        arrays = {f"unit_{template}": units[template, position, layer].astype(np.float32) for template in TEMPLATES}; arrays["shared_all"] = shared.astype(np.float32); arrays["singular_values"] = singular.astype(np.float32)
        for held_out in TEMPLATES:
            donors = [units[template, position, layer] for template in TEMPLATES if template != held_out]; loto, _ = uncentered_shared_axis(donors); arrays[f"loto_{held_out}"] = loto.astype(np.float32)
            x_train = matrices[held_out, position, layer, "construction"]; x_audit = matrices[held_out, position, layer, "audit"]; y_train = np.asarray([repository.label(held_out, str(row["case_id"])) for row in construction]); y_audit = np.asarray([repository.label(held_out, str(row["case_id"])) for row in audit]); projection_train = x_train @ loto; projection_audit = x_audit @ loto
            slope, intercept = np.polyfit(projection_train, y_train, 1); calibrated = slope * projection_audit + intercept; measured = _safe_metrics(y_audit, calibrated); raw_pearson = float(pearsonr(projection_audit, y_audit).statistic); raw_spearman = float(spearmanr(projection_audit, y_audit).statistic)
            prediction_rows.append({"position": position, "layer": layer, "held_out_template": held_out, "donor_templates": ",".join(template for template in TEMPLATES if template != held_out), "held_out_probe_used_for_axis_or_sign": False, "loto_probe_cosine": float(loto @ units[held_out, position, layer]), "raw_audit_pearson": raw_pearson, "raw_audit_spearman": raw_spearman, "calibration_slope": float(slope), "calibration_intercept": float(intercept), **{f"audit_{key}": value for key, value in measured.items()}, **{f"audit_{key}": value for key, value in _bootstrap_metrics(y_audit, calibrated, [str(row["family_id"]) for row in audit], families, draws).items()}})
            sigma_loto = float(np.std(projection_train, ddof=1)); sigma_self = float(np.std(x_train @ units[held_out, position, layer], ddof=1)); seed = int(hashlib.sha256(f"{SEED}|{held_out}|{position}|{layer}".encode()).hexdigest()[:16], 16); randoms, metadata = matched_random_directions(x_train, sigma_loto, seed=seed, count=random_count)
            arrays[f"random_{held_out}"] = randoms.astype(np.float32)
            for row, random_vector in zip(metadata, randoms): random_audit.append({"template": held_out, "position": position, "layer": layer, "sigma_loto": sigma_loto, "sigma_self": sigma_self, "candidate_seed": seed, **row, "cosine_shared_loto": float(random_vector @ loto), "cosine_self": float(random_vector @ units[held_out, position, layer])})
        destination = root / f"artifacts/axes/{node_name(position, layer)}.npz"; atomic_npz(destination, arrays); axes_hashes[node_name(position, layer)] = sha256_file(destination)
    atomic_csv(root / "tables/geometry_summary.csv", geometry); atomic_csv(root / "tables/pairwise_probe_cosines.csv", pairwise); atomic_csv(root / "tables/loto_prediction_metrics.csv", prediction_rows); atomic_csv(root / "tables/random_direction_audit.csv", random_audit); atomic_json(root / "artifacts/axes/vector_metadata.json", {"nodes": [node_name(*node) for node in nodes], "axis_files_sha256": axes_hashes, "sign_rule": "mean donor cosine positive; held-out probe never used", "equal_dose_scale": "sigma_loto", "fingerprint": fingerprint})
    pilot = [row for row in prediction_rows if (row["position"], int(row["layer"])) == PILOT_NODE]; gate = {"all_loto_cosines_gt_0_20": all(row["loto_probe_cosine"] > .20 for row in pilot), "all_raw_audit_pearson_positive": all(row["raw_audit_pearson"] > 0 for row in pilot), "all_raw_audit_spearman_positive": all(row["raw_audit_spearman"] > 0 for row in pilot)}; gate["passed"] = all(gate.values()); atomic_json(root / "progress/cpu_geometry_gate.json", gate)
    result = {"status": "complete", "fingerprint": fingerprint, "node_count": len(nodes), "probe_count": len(probe_audit), "prediction_rows": len(prediction_rows), "random_direction_rows": len(random_audit), "gpu_gate": gate}; atomic_json(root / "progress/cpu_prepare.json", result); return result


def verify_sources(root: Path) -> None:
    expected = json.loads((root / "artifacts/source_hashes_before.json").read_text()); verify_inventory(expected); atomic_json(root / "artifacts/source_hashes_after.json", inventory(map(Path, expected)))
