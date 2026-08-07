"""Fingerprints and strict source validation for Probe manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import HIDDEN_STATE_DEFINITION
from .common import iter_jsonl


PROVENANCE_FIELDS = (
    "source_experiment_dir",
    "hidden_state_index_fingerprint",
    "dataset_fingerprint",
    "manifest_fingerprint",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_file_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def manifest_records_fingerprint(records: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(manifest_file_bytes(records)).hexdigest()


def _identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(record["case_id"]),
        "item_id": str(record["item_id"]),
        "prior_index": int(record["prior_index"]),
        "condition": str(record["condition"]),
        "version": str(record["version"]),
    }


def selected_result_identities(results_path: str | Path) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for source in iter_jsonl(results_path):
        generated = source.get("generated")
        reference = source.get("hidden_state_reference")
        if (
            source.get("status") == "completed"
            and source.get("attribution_mode") == "joint"
            and isinstance(reference, dict)
            and isinstance(generated, dict)
            and generated.get("current_answer") is not None
        ):
            identities.append(_identity(source))
    return identities


def dataset_fingerprint_from_results(results_path: str | Path) -> str:
    return canonical_fingerprint(selected_result_identities(results_path))


def dataset_fingerprint_from_manifest(
    manifest: Sequence[dict[str, Any]],
) -> str:
    return canonical_fingerprint([_identity(record) for record in manifest])


def _same_list(left: Any, right: Any) -> bool:
    return [str(value) for value in left or []] == [str(value) for value in right or []]


def _validate_reference(
    case_id: str,
    manifest_reference: dict[str, Any],
    index_reference: dict[str, Any],
) -> None:
    if str(index_reference.get("case_id")) != case_id:
        raise ValueError(f"Hidden index case ID mismatch for {case_id}")
    for field in ("shard_path", "offset", "hidden_size", "hidden_state_definition"):
        if manifest_reference.get(field) != index_reference.get(field):
            raise ValueError(
                f"Manifest/index mismatch for {case_id}: {field} "
                f"{manifest_reference.get(field)!r} != {index_reference.get(field)!r}"
            )
    for field in ("layer_indices", "position_names"):
        if not _same_list(manifest_reference.get(field), index_reference.get(field)):
            raise ValueError(f"Manifest/index mismatch for {case_id}: {field}")
    if manifest_reference.get("hidden_state_definition") != HIDDEN_STATE_DEFINITION:
        raise ValueError(
            f"Unsupported hidden-state definition for {case_id}: "
            f"{manifest_reference.get('hidden_state_definition')!r}"
        )


def validate_manifest_provenance(
    experiment_dir: str | Path,
    manifest_path: str | Path,
    manifest: Sequence[dict[str, Any]],
    *,
    selected_conditions: Sequence[str],
    requested_layers: Sequence[int],
    requested_positions: Sequence[str],
    requested_versions: Sequence[str],
) -> dict[str, Any]:
    """Validate an external or local manifest against the current experiment."""

    experiment = Path(experiment_dir).resolve()
    manifest_file = Path(manifest_path).resolve()
    results_path = experiment / "results.jsonl"
    index_path = experiment / "hidden_states" / "index.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Results do not exist: {results_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Hidden-state index does not exist: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    cases = index.get("cases")
    if not isinstance(cases, dict):
        raise ValueError(f"Hidden-state index has no cases object: {index_path}")

    versions = {str(record.get("version")) for record in manifest}
    if "v4" not in versions:
        raise ValueError("Probe manifest must contain V4 records")
    missing_versions = sorted(set(requested_versions) - versions)
    if missing_versions:
        raise ValueError(
            f"Requested versions are absent from the manifest: {missing_versions}; "
            f"available={sorted(versions)}"
        )
    conditions = {str(record.get("condition")) for record in manifest}
    missing_conditions = sorted(set(selected_conditions) - conditions)
    if missing_conditions:
        raise ValueError(
            f"Requested conditions are absent from the manifest: {missing_conditions}; "
            f"available={sorted(conditions)}"
        )

    requested_layer_set = {int(value) for value in requested_layers}
    requested_position_set = {str(value) for value in requested_positions}
    for record in manifest:
        case_id = str(record.get("case_id"))
        reference = record.get("hidden_state_reference")
        if not isinstance(reference, dict):
            raise ValueError(f"Manifest case has no hidden_state_reference: {case_id}")
        index_reference = cases.get(case_id)
        if not isinstance(index_reference, dict):
            raise ValueError(f"Completed case is absent from hidden-state index: {case_id}")
        _validate_reference(case_id, reference, index_reference)
        missing_layers = requested_layer_set.difference(
            int(value) for value in reference.get("layer_indices", [])
        )
        if missing_layers:
            raise ValueError(
                f"Requested layers are absent for {case_id}: {sorted(missing_layers)}"
            )
        missing_positions = requested_position_set.difference(
            str(value) for value in reference.get("position_names", [])
        )
        if missing_positions:
            raise ValueError(
                f"Requested positions are absent for {case_id}: {sorted(missing_positions)}"
            )

    index_fingerprint = sha256_file(index_path)
    manifest_fingerprint = sha256_file(manifest_file)
    source_dataset_fingerprint = dataset_fingerprint_from_results(results_path)
    manifest_dataset_fingerprint = dataset_fingerprint_from_manifest(manifest)
    if source_dataset_fingerprint != manifest_dataset_fingerprint:
        raise ValueError(
            "Manifest dataset fingerprint does not match completed joint records in "
            f"the current experiment: {manifest_file}"
        )

    summary_path = manifest_file.parent / "manifest_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    complete_provenance = all(summary.get(field) is not None for field in PROVENANCE_FIELDS)
    if complete_provenance:
        expected = {
            "source_experiment_dir": str(experiment),
            "hidden_state_index_fingerprint": index_fingerprint,
            "dataset_fingerprint": source_dataset_fingerprint,
            "manifest_fingerprint": manifest_fingerprint,
        }
        mismatches = {
            field: {"summary": summary.get(field), "current": value}
            for field, value in expected.items()
            if summary.get(field) != value
        }
        if mismatches:
            raise ValueError(f"Manifest provenance mismatch: {mismatches}")
        validation_mode = "fingerprint_and_exhaustive"
    else:
        validation_mode = "legacy_exhaustive"

    return {
        "provenance_validation": validation_mode,
        "source_experiment_dir": str(experiment),
        "hidden_state_index_fingerprint": index_fingerprint,
        "dataset_fingerprint": source_dataset_fingerprint,
        "manifest_fingerprint": manifest_fingerprint,
        "manifest_summary_path": str(summary_path) if summary_path.is_file() else None,
    }
