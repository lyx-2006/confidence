from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from layer_metacognition.sa_patching.artifacts import ragged_position_mean

from .io import atomic_json, atomic_torch_save, canonical_hash, sha256_file
from .protocol import prepare_delayed_case, span_positions


@dataclass(frozen=True)
class CorruptionArtifacts:
    images: dict[str, torch.Tensor]
    image_metadata: dict[str, dict[str, Any]]
    text: torch.Tensor
    text_counts: torch.Tensor
    answers: dict[int, torch.Tensor]
    metadata: dict[str, Any]

    def replacements_for(self, *, image_key: str, spans: dict[str, Any]):
        image_positions = span_positions(spans, "IMAGE")
        text_positions = span_positions(spans, "TEXT_CLUE")
        answer_positions = span_positions(spans, "ANSWER")
        if image_key not in self.images:
            raise ValueError(f"No exact image mean artifact for shape key {image_key}")
        image = self.images[image_key]
        if tuple(image.shape[:1]) != (len(image_positions),):
            raise ValueError("Image mean embedding shape mismatch")
        if len(text_positions) > len(self.text):
            raise ValueError("Text mean artifact does not cover the evaluation clue")
        if bool((self.text_counts[:len(text_positions)] <= 0).any()):
            raise ValueError("Text mean has an uncovered evaluation position")
        answer = self.answers.get(len(answer_positions))
        if answer is None:
            raise ValueError(f"No answer mean donors for token length {len(answer_positions)}")
        if len(answer) != len(answer_positions):
            raise ValueError("Answer mean embedding shape mismatch")
        return image, self.text[:len(text_positions)], answer


def grouped_shape_means(entries: Sequence[tuple[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for key, vector in entries:
        if vector.ndim != 2 or not bool(torch.isfinite(vector).all()):
            raise ValueError("Grouped mean requires finite rank-2 tensors")
        if groups[key] and tuple(groups[key][0].shape) != tuple(vector.shape):
            raise ValueError(f"Shape key {key} contains incompatible tensors")
        groups[key].append(vector.detach().double().cpu())
    if not groups:
        raise ValueError("Grouped mean requires at least one tensor")
    return {key: torch.stack(values).mean(dim=0).float() for key, values in groups.items()}


def length_conditioned_means(vectors: Sequence[torch.Tensor]) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    entries = [(str(int(vector.shape[0])), vector) for vector in vectors]
    means = grouped_shape_means(entries)
    counts = {int(key): sum(int(vector.shape[0]) == int(key) for vector in vectors) for key in means}
    return {int(key): value for key, value in means.items()}, counts


def image_shape_key(inputs: Any, vector: torch.Tensor, hidden_size: int) -> tuple[str, dict[str, Any]]:
    grid = inputs.get("image_grid_thw")
    if grid is None:
        raise ValueError("Prepared multimodal input has no image_grid_thw")
    metadata = {
        "grid_thw": [int(value) for value in grid.detach().cpu().reshape(-1).tolist()],
        "feature_shape": list(vector.shape), "hidden_size": int(hidden_size),
    }
    return canonical_hash(metadata), metadata


def _extract_image(inference: Any, inputs: Any, *, expected_tokens: int, hidden_size: int):
    pixel_values = inputs.get("pixel_values")
    image_grid = inputs.get("image_grid_thw")
    if pixel_values is None or image_grid is None:
        raise ValueError("Image donor is missing vision tensors")
    with torch.inference_mode():
        features = inference.model.get_image_features(pixel_values, image_grid)
    vector = torch.cat(list(features), dim=0).detach().float().cpu()
    if tuple(vector.shape) != (expected_tokens, hidden_size):
        raise ValueError(f"Vision feature/span mismatch: {tuple(vector.shape)} != {(expected_tokens, hidden_size)}")
    key, metadata = image_shape_key(inputs, vector, hidden_size)
    return key, metadata, vector


def build_or_load_artifacts(
    *, artifact_dir: Path, calibration: dict[str, Any], inference: Any,
    hidden_size: int, image_root: Path | None,
) -> CorruptionArtifacts:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "artifact_manifest.json"
    image_path = artifact_dir / "image_means.pt"
    text_path = artifact_dir / "text_mean.pt"
    answer_path = artifact_dir / "answer_means.pt"
    calibration_hash = canonical_hash(calibration)
    if all(path.is_file() for path in (manifest_path, image_path, text_path, answer_path)):
        manifest = __import__("json").loads(manifest_path.read_text())
        if manifest.get("calibration_fingerprint") != calibration_hash:
            raise ValueError("Existing corruption artifacts use a different calibration cohort")
        expected_hashes = manifest.get("artifact_sha256", {})
        for name, path in (("image", image_path), ("text", text_path), ("answer", answer_path)):
            if sha256_file(path) != expected_hashes.get(name):
                raise ValueError(f"Corruption artifact fingerprint changed: {name}")
        image_payload = torch.load(image_path, map_location="cpu", weights_only=False)
        text_payload = torch.load(text_path, map_location="cpu", weights_only=False)
        answer_payload = torch.load(answer_path, map_location="cpu", weights_only=False)
    else:
        if any(path.exists() for path in (manifest_path, image_path, text_path, answer_path)):
            raise ValueError("Corruption artifact set is partially present")
        image_groups: dict[str, list[torch.Tensor]] = defaultdict(list)
        image_metadata: dict[str, dict[str, Any]] = {}
        image_sources: dict[str, list[str]] = defaultdict(list)
        image_source_metadata: list[dict[str, Any]] = []
        for row in calibration["pools"]["image"]:
            _rendered, inputs, details = prepare_delayed_case(inference, row, image_root=image_root)
            positions = span_positions(details["spans"], "IMAGE")
            key, metadata, vector = _extract_image(inference, inputs, expected_tokens=len(positions), hidden_size=hidden_size)
            image_groups[key].append(vector)
            image_metadata[key] = metadata
            image_sources[key].append(str(row["case_id"]))
            image_source_metadata.append({"case_id": str(row["case_id"]), "item_id": str(row["item_id"]),
                                          "shape_key": key, **metadata})
            del inputs, vector
        image_means = {key: value.contiguous() for key, value in
                       grouped_shape_means([(key, vector) for key, vectors in image_groups.items() for vector in vectors]).items()}
        image_payload = {"format_version": 1, "calibration_fingerprint": calibration_hash,
                         "means": image_means, "shape_metadata": image_metadata,
                         "source_case_ids": image_sources,
                         "source_metadata": image_source_metadata,
                         "counts": {key: len(value) for key, value in image_groups.items()}}

        text_vectors: list[torch.Tensor] = []
        text_source_metadata: list[dict[str, Any]] = []
        for row in calibration["pools"]["text"]:
            _rendered, inputs, details = prepare_delayed_case(inference, row, image_root=image_root)
            positions = span_positions(details["spans"], "TEXT_CLUE")
            with torch.inference_mode():
                embeddings = inference.model.get_input_embeddings()(inputs.input_ids)
            vector = embeddings[0, list(positions)].detach().float().cpu()
            if vector.ndim != 2 or int(vector.shape[1]) != hidden_size:
                raise ValueError("Text donor embedding shape mismatch")
            text_vectors.append(vector)
            text_source_metadata.append({"case_id": str(row["case_id"]), "item_id": str(row["item_id"]),
                                         "token_length": len(positions), "embedding_shape": list(vector.shape)})
            del inputs, embeddings
        text_mean, text_counts = ragged_position_mean(text_vectors)
        text_payload = {"format_version": 1, "calibration_fingerprint": calibration_hash,
                        "mean": text_mean.contiguous(), "position_counts": text_counts.contiguous(),
                        "source_case_ids": [row["case_id"] for row in calibration["pools"]["text"]],
                        "source_metadata": text_source_metadata}

        answer_groups: dict[int, list[torch.Tensor]] = defaultdict(list)
        answer_sources: dict[int, list[str]] = defaultdict(list)
        answer_source_metadata: list[dict[str, Any]] = []
        for row in calibration["pools"]["answer"]:
            _rendered, inputs, details = prepare_delayed_case(inference, row, image_root=image_root)
            positions = span_positions(details["spans"], "ANSWER")
            with torch.inference_mode():
                embeddings = inference.model.get_input_embeddings()(inputs.input_ids)
            vector = embeddings[0, list(positions)].detach().float().cpu()
            if vector.ndim != 2 or int(vector.shape[1]) != hidden_size:
                raise ValueError("Answer donor embedding shape mismatch")
            answer_groups[len(positions)].append(vector)
            answer_sources[len(positions)].append(str(row["case_id"]))
            answer_source_metadata.append({"case_id": str(row["case_id"]), "item_id": str(row["item_id"]),
                                           "token_length": len(positions), "embedding_shape": list(vector.shape)})
            del inputs, embeddings
        answer_means, answer_counts = length_conditioned_means(
            [vector for values in answer_groups.values() for vector in values]
        )
        answer_means = {length: value.contiguous() for length, value in answer_means.items()}
        answer_payload = {"format_version": 1, "calibration_fingerprint": calibration_hash,
                          "means": answer_means, "counts": answer_counts,
                          "source_case_ids": answer_sources,
                          "source_metadata": answer_source_metadata}
        atomic_torch_save(image_path, image_payload)
        atomic_torch_save(text_path, text_payload)
        atomic_torch_save(answer_path, answer_payload)
        manifest = {
            "format_version": 1, "calibration_fingerprint": calibration_hash,
            "hidden_size": hidden_size, "dtype": "float32",
            "image_groups": image_metadata,
            "image_group_counts": image_payload["counts"],
            "text_shape": list(text_mean.shape), "text_position_counts": text_counts.tolist(),
            "answer_shapes": {str(key): list(value.shape) for key, value in answer_means.items()},
            "answer_counts": {str(key): value for key, value in answer_payload["counts"].items()},
            "source_metadata": {
                "image": image_payload["source_metadata"],
                "text": text_payload["source_metadata"],
                "answer": answer_payload["source_metadata"],
            },
            "artifact_sha256": {"image": sha256_file(image_path), "text": sha256_file(text_path), "answer": sha256_file(answer_path)},
        }
        manifest["fingerprint"] = canonical_hash(manifest)
        atomic_json(manifest_path, manifest)
    images = {str(key): value.float().contiguous() for key, value in image_payload["means"].items()}
    answers = {int(key): value.float().contiguous() for key, value in answer_payload["means"].items()}
    text = text_payload["mean"].float().contiguous()
    counts = text_payload["position_counts"].long().contiguous()
    tensors = [*images.values(), text, *answers.values()]
    if not all(bool(value.isfinite().all()) for value in tensors):
        raise ValueError("A mean embedding artifact contains NaN/Inf")
    manifest = __import__("json").loads(manifest_path.read_text())
    return CorruptionArtifacts(images, image_payload["shape_metadata"], text, counts, answers, manifest)


def evaluation_image_key(inference: Any, inputs: Any, spans: dict[str, Any], *, hidden_size: int) -> str:
    positions = span_positions(spans, "IMAGE")
    grid = inputs.get("image_grid_thw")
    if grid is None:
        raise ValueError("Evaluation input has no image_grid_thw")
    metadata = {
        "grid_thw": [int(value) for value in grid.detach().cpu().reshape(-1).tolist()],
        "feature_shape": [len(positions), int(hidden_size)],
        "hidden_size": int(hidden_size),
    }
    return canonical_hash(metadata)
