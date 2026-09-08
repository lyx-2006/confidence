from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from layer_metacognition.model_adapter import model_input_device

from .data import FrozenCohort
from .io_utils import atomic_json, atomic_torch_save, canonical_hash, sha256_file
from .protocol import prepare_phase0


class GroupedFloat64Mean:
    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}

    def add(self, key: str, vector: torch.Tensor) -> None:
        if vector.ndim != 2 or not bool(torch.isfinite(vector).all()):
            raise ValueError("Grouped mean requires a finite rank-2 tensor")
        value = vector.detach().double().cpu()
        if key not in self.sums:
            self.sums[key] = torch.zeros_like(value, dtype=torch.float64)
            self.counts[key] = 0
        if tuple(self.sums[key].shape) != tuple(value.shape):
            raise ValueError(f"Shape group {key} contains incompatible tensors")
        self.sums[key] += value
        self.counts[key] += 1

    def means(self) -> dict[str, torch.Tensor]:
        if not self.sums:
            raise ValueError("Grouped mean has no donors")
        return {key: (value / self.counts[key]).float().contiguous() for key, value in self.sums.items()}


class RaggedFloat64Mean:
    def __init__(self) -> None:
        self.total: torch.Tensor | None = None
        self.counts: torch.Tensor | None = None

    def add(self, vector: torch.Tensor) -> None:
        if vector.ndim != 2 or not bool(torch.isfinite(vector).all()):
            raise ValueError("Ragged mean requires a finite rank-2 tensor")
        value = vector.detach().double().cpu()
        length, hidden = map(int, value.shape)
        if self.total is None:
            self.total = torch.zeros(length, hidden, dtype=torch.float64)
            self.counts = torch.zeros(length, dtype=torch.int64)
        elif int(self.total.shape[1]) != hidden:
            raise ValueError("Ragged mean hidden size changed")
        elif length > int(self.total.shape[0]):
            extension = length - int(self.total.shape[0])
            self.total = torch.cat((self.total, torch.zeros(extension, hidden, dtype=torch.float64)))
            assert self.counts is not None
            self.counts = torch.cat((self.counts, torch.zeros(extension, dtype=torch.int64)))
        assert self.total is not None and self.counts is not None
        self.total[:length] += value
        self.counts[:length] += 1

    def mean(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.total is None or self.counts is None or bool((self.counts <= 0).any()):
            raise ValueError("Ragged mean has an uncovered position")
        return (self.total / self.counts.unsqueeze(1)).float().contiguous(), self.counts.clone()


@dataclass(frozen=True)
class MeanArtifacts:
    image_means: dict[str, torch.Tensor]
    text_mean: torch.Tensor
    text_counts: torch.Tensor
    manifest: dict[str, Any]

    def image_for(self, shape_key: str, length: int) -> torch.Tensor:
        if shape_key not in self.image_means:
            raise ValueError(f"No exact image mean for shape group {shape_key}")
        value = self.image_means[shape_key]
        if int(value.shape[0]) != int(length):
            raise ValueError("Image mean/token span shape mismatch")
        return value

    def text_for(self, length: int) -> torch.Tensor:
        if length <= 0:
            raise ValueError("Text span must be non-empty")
        covered = min(int(length), int(self.text_mean.shape[0]))
        if bool((self.text_counts[:covered] <= 0).any()):
            raise ValueError("Text donor mean has an uncovered position")
        if length <= covered:
            return self.text_mean[:length]
        # Explicit relaxed-coverage policy: replicate the last donor-supported
        # position for an uncovered tail. The fallback is audited by callers.
        tail = self.text_mean[covered - 1:covered].expand(length - covered, -1)
        return torch.cat((self.text_mean[:covered], tail), dim=0).contiguous()


def image_shape_key(inputs: Any, feature_shape: tuple[int, int]) -> tuple[str, dict[str, Any]]:
    grid = inputs.get("image_grid_thw")
    if grid is None:
        raise ValueError("Input has no image_grid_thw")
    metadata = {
        "image_grid_thw": [int(value) for value in grid.detach().cpu().reshape(-1).tolist()],
        "feature_shape": list(feature_shape), "hidden_size": int(feature_shape[1]),
    }
    return canonical_hash(metadata), metadata


def _load_checked(directory: Path, cohort_hash: str, runtime_hash: str) -> MeanArtifacts | None:
    manifest_path = directory / "manifest.json"
    image_path = directory / "image_means.pt"
    text_path = directory / "text_mean.pt"
    existing = [path.exists() for path in (manifest_path, image_path, text_path)]
    if not any(existing):
        return None
    if not all(existing):
        raise ValueError("Mean artifact set is partially present")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("cohort_fingerprint") != cohort_hash or manifest.get("runtime_fingerprint") != runtime_hash:
        raise ValueError("Existing mean artifacts use another cohort/runtime")
    for name, path in (("image", image_path), ("text", text_path)):
        if sha256_file(path) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Mean artifact hash changed: {name}")
    image = torch.load(image_path, map_location="cpu", weights_only=False)
    text = torch.load(text_path, map_location="cpu", weights_only=False)
    return MeanArtifacts(
        {str(key): value.float().contiguous() for key, value in image["means"].items()},
        text["mean"].float().contiguous(), text["position_counts"].long().contiguous(), manifest,
    )


def build_or_load_means(root: Path, cohort: FrozenCohort, inference: Any,
                        runtime_identity: dict[str, Any], hidden_size: int) -> MeanArtifacts:
    directory = root / "artifacts" / "mean_embeddings"
    directory.mkdir(parents=True, exist_ok=True)
    cohort_hash = canonical_hash(cohort.audit)
    cached = _load_checked(directory, cohort_hash, runtime_identity["fingerprint"])
    if cached is not None:
        return cached
    device = model_input_device(inference)
    image_accumulator = GroupedFloat64Mean()
    image_groups: dict[str, dict[str, Any]] = {}
    image_sources: list[dict[str, Any]] = []
    for row in cohort.image_donors:
        _rendered, inputs, details = prepare_phase0(inference.processor, row, device=device)
        with torch.inference_mode():
            chunks = inference.model.get_image_features(inputs.pixel_values, inputs.image_grid_thw)
        vector = torch.cat(list(chunks), dim=0).detach().cpu()
        expected = len(details["image_positions"])
        if tuple(vector.shape) != (expected, hidden_size):
            raise ValueError(f"Image feature/span mismatch for {row['donor_id']}: {tuple(vector.shape)}")
        key, metadata = image_shape_key(inputs, (expected, hidden_size))
        if key not in image_groups:
            image_groups[key] = metadata
        image_accumulator.add(key, vector)
        image_sources.append({
            "donor_id": row["donor_id"], "unique_key": row["donor_unique_key"],
            "case_id": row["case_id"], "family_id": row["family_id"], "item_id": str(row["item_id"]),
            "condition": row["condition"], "image_sha256": row["image_sha256"], "shape_key": key,
            **metadata,
        })
        del inputs, chunks, vector
    image_means = image_accumulator.means()

    text_accumulator = RaggedFloat64Mean()
    text_sources: list[dict[str, Any]] = []
    for row in cohort.text_donors:
        _rendered, inputs, details = prepare_phase0(inference.processor, row, device=device)
        positions = details["text_positions"]
        with torch.inference_mode():
            embeddings = inference.model.get_input_embeddings()(inputs.input_ids)
        vector = embeddings[0, positions].detach().cpu()
        if vector.ndim != 2 or int(vector.shape[1]) != hidden_size:
            raise ValueError(f"Text embedding shape mismatch for {row['donor_id']}")
        length = int(vector.shape[0])
        text_accumulator.add(vector)
        text_sources.append({
            "donor_id": row["donor_id"], "unique_key": row["donor_unique_key"],
            "case_id": row["case_id"], "family_id": row["family_id"], "item_id": str(row["item_id"]),
            "condition": row["condition"], "image_sha256": row["image_sha256"],
            "token_length": length, "token_ids": details["text_token_ids"],
        })
        del inputs, embeddings, vector
    text_mean, text_counts = text_accumulator.mean()
    image_path = directory / "image_means.pt"
    text_path = directory / "text_mean.pt"
    atomic_torch_save(image_path, {"format_version": 1, "means": image_means})
    atomic_torch_save(text_path, {"format_version": 1, "mean": text_mean, "position_counts": text_counts})
    manifest = {
        "format_version": 1, "experiment": "perturbation_based_answer_reliance",
        "cohort_fingerprint": cohort_hash, "runtime_fingerprint": runtime_identity["fingerprint"],
        "accumulation_dtype": "float64", "storage_dtype": "float32", "hidden_size": hidden_size,
        "image_groups": image_groups, "image_group_counts": image_accumulator.counts,
        "image_sources": image_sources, "text_shape": list(text_mean.shape),
        "text_position_counts": text_counts.tolist(), "text_sources": text_sources,
        "artifact_sha256": {"image": sha256_file(image_path), "text": sha256_file(text_path)},
    }
    manifest["fingerprint"] = canonical_hash(manifest)
    atomic_json(directory / "manifest.json", manifest)
    return MeanArtifacts(image_means, text_mean, text_counts, manifest)


def evaluation_shape_key(inputs: Any, details: dict[str, Any], hidden_size: int) -> str:
    key, _metadata = image_shape_key(inputs, (len(details["image_positions"]), hidden_size))
    return key


def audit_coverage(rows: list[dict[str, Any]], inference: Any, artifacts: MeanArtifacts,
                   hidden_size: int) -> dict[str, Any]:
    device = model_input_device(inference)
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        try:
            _rendered, inputs, details = prepare_phase0(inference.processor, row, device=device)
            key = evaluation_shape_key(inputs, details, hidden_size)
            image = artifacts.image_for(key, len(details["image_positions"]))
            text = artifacts.text_for(len(details["text_positions"]))
        except Exception as exc:
            failures.append({"case_id": row["case_id"], "error": str(exc)})
            continue
        checks.append({
            "case_id": row["case_id"], "image_shape_key": key,
            "image_tokens": int(image.shape[0]), "text_tokens": int(text.shape[0]),
            "text_last_position_donor_count": int(artifacts.text_counts[min(
                len(details["text_positions"]), int(artifacts.text_mean.shape[0])) - 1]),
            "text_fallback_positions": max(0, len(details["text_positions"]) - int(artifacts.text_mean.shape[0])),
        })
        del inputs
    return {
        "status": "passed" if not failures else "failed", "case_count": len(checks),
        "failed_count": len(failures), "failures": failures, "checks": checks,
        "image_shape_groups": sorted({row["image_shape_key"] for row in checks}),
        "max_text_tokens": max((row["text_tokens"] for row in checks), default=0),
    }


__all__ = [
    "GroupedFloat64Mean", "MeanArtifacts", "RaggedFloat64Mean", "audit_coverage",
    "build_or_load_means", "evaluation_shape_key", "image_shape_key",
]
