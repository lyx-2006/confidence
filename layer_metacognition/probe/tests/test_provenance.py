from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer_metacognition.probe import HIDDEN_STATE_DEFINITION
from layer_metacognition.probe.provenance import validate_manifest_provenance


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def _experiment(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    experiment = tmp_path / "experiment"
    manifest_path = tmp_path / "external-probe" / "probe_manifest.jsonl"
    case_id = "1__prior_0__consistent_easy__v4__joint"
    reference = {
        "shard_path": "hidden_states/shard.pt",
        "offset": 0,
        "layer_indices": [14],
        "position_names": ["ac"],
        "hidden_size": 4,
        "hidden_state_definition": HIDDEN_STATE_DEFINITION,
    }
    result = {
        "case_id": case_id,
        "item_id": "1",
        "prior_index": 0,
        "condition": "consistent_easy",
        "version": "v4",
        "status": "completed",
        "attribution_mode": "joint",
        "generated": {"current_answer": "blue"},
        "hidden_state_reference": reference,
    }
    manifest = [
        {
            "case_id": case_id,
            "item_id": "1",
            "prior_index": 0,
            "condition": "consistent_easy",
            "version": "v4",
            "hidden_state_reference": reference,
        }
    ]
    index_reference = {"case_id": case_id, **reference}
    index_path = experiment / "hidden_states" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps({"cases": {case_id: index_reference}}), encoding="utf-8"
    )
    _write_jsonl(experiment / "results.jsonl", [result])
    _write_jsonl(manifest_path, manifest)
    return experiment, manifest_path, manifest


def _validate(experiment: Path, manifest_path: Path, manifest: list[dict]) -> dict:
    return validate_manifest_provenance(
        experiment,
        manifest_path,
        manifest,
        selected_conditions=["consistent_easy"],
        requested_layers=[14],
        requested_positions=["ac"],
        requested_versions=["v4"],
    )


def test_legacy_manifest_uses_exhaustive_validation(tmp_path: Path) -> None:
    experiment, manifest_path, manifest = _experiment(tmp_path)
    result = _validate(experiment, manifest_path, manifest)
    assert result["provenance_validation"] == "legacy_exhaustive"
    summary = {
        key: result[key]
        for key in (
            "source_experiment_dir",
            "hidden_state_index_fingerprint",
            "dataset_fingerprint",
            "manifest_fingerprint",
        )
    }
    (manifest_path.parent / "manifest_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    assert _validate(experiment, manifest_path, manifest)[
        "provenance_validation"
    ] == "fingerprint_and_exhaustive"


def test_summary_fingerprint_mismatch_fails(tmp_path: Path) -> None:
    experiment, manifest_path, manifest = _experiment(tmp_path)
    result = _validate(experiment, manifest_path, manifest)
    summary = {
        key: result[key]
        for key in (
            "source_experiment_dir",
            "hidden_state_index_fingerprint",
            "dataset_fingerprint",
            "manifest_fingerprint",
        )
    }
    summary["hidden_state_index_fingerprint"] = "0" * 64
    (manifest_path.parent / "manifest_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        _validate(experiment, manifest_path, manifest)


def test_manifest_reference_mismatch_fails(tmp_path: Path) -> None:
    experiment, manifest_path, manifest = _experiment(tmp_path)
    manifest[0]["hidden_state_reference"] = {
        **manifest[0]["hidden_state_reference"],
        "offset": 9,
    }
    _write_jsonl(manifest_path, manifest)
    with pytest.raises(ValueError, match="offset"):
        _validate(experiment, manifest_path, manifest)
