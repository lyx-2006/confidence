from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from layer_metacognition.probe import HIDDEN_STATE_DEFINITION
from layer_metacognition.probe.analyze_probe_results import main as analyze_main
from layer_metacognition.probe.train_layer_probes import main


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_train_pools_conditions_and_writes_to_custom_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment = tmp_path / "experiment"
    output = tmp_path / "custom-probe"
    conditions = ("consistent_easy", "conflict_easy")
    positions = ["ac", "panl"]
    records: list[dict] = []
    results: list[dict] = []
    case_ids: list[str] = []
    index_cases: dict[str, dict] = {}
    for item in range(8):
        for condition in conditions:
            for version in ("v3", "v4"):
                case_id = f"{item}-{condition}-{version}"
                offset = len(case_ids)
                case_ids.append(case_id)
                reference = {
                    "case_id": case_id,
                    "shard_path": "hidden_states/probe.pt",
                    "offset": offset,
                    "layer_indices": [3],
                    "position_names": positions,
                    "hidden_size": 2,
                    "hidden_state_definition": HIDDEN_STATE_DEFINITION,
                }
                index_cases[case_id] = reference
                manifest_reference = dict(reference)
                manifest_reference.pop("case_id")
                records.append(
                    {
                        "case_id": case_id,
                        "item_id": str(item),
                        "prior_index": 0,
                        "condition": condition,
                        "version": version,
                        "text_only_answer": "blue" if item % 2 == 0 else "yellow",
                        "image_only_answer": "yellow" if item % 2 == 0 else "blue",
                        "conflict_label": (
                            "consistent"
                            if condition.startswith("consistent_")
                            else "conflict"
                        ),
                        "current_answer": "blue" if item % 2 == 0 else "yellow",
                        "eligible_text_probe": True,
                        "eligible_image_probe": True,
                        "eligible_conflict_probe": True,
                        "hidden_state_reference": manifest_reference,
                    }
                )
                results.append(
                    {
                        "case_id": case_id,
                        "item_id": str(item),
                        "prior_index": 0,
                        "condition": condition,
                        "version": version,
                        "status": "completed",
                        "attribution_mode": "joint",
                        "generated": {
                            "current_answer": "blue" if item % 2 == 0 else "yellow"
                        },
                        "hidden_state_reference": manifest_reference,
                    }
                )

    shard = experiment / "hidden_states" / "probe.pt"
    shard.parent.mkdir(parents=True)
    torch.save(
        {
            "format_version": 2,
            "case_ids": case_ids,
            "layer_indices": [3],
            "position_names": positions,
            "hidden_states": torch.arange(
                len(case_ids) * len(positions) * 2,
                dtype=torch.float16,
            ).reshape(len(case_ids), len(positions), 2),
            "hidden_size": 2,
            "hidden_state_definition": HIDDEN_STATE_DEFINITION,
        },
        shard,
    )
    (experiment / "hidden_states" / "index.json").write_text(
        json.dumps({"format_version": 2, "cases": index_cases}),
        encoding="utf-8",
    )
    _write_jsonl(experiment / "results.jsonl", results)
    _write_jsonl(output / "probe_manifest.jsonl", records)
    monkeypatch.setattr(
        "layer_metacognition.probe.train_layer_probes.choose_regularization_C",
        lambda *args, **kwargs: (1.0, {"status": "test"}),
    )

    assert main(
        [
            "--experiment-dir",
            str(experiment),
            "--output-dir",
            str(output),
            "--layers",
            "3",
            "--n-splits",
            "2",
            "--permutations",
            "0",
        ]
    ) == 0
    with pytest.raises(FileExistsError, match="already exists"):
        main(
            [
                "--experiment-dir",
                str(experiment),
                "--output-dir",
                str(output),
                "--layers",
                "3",
                "--n-splits",
                "2",
                "--permutations",
                "0",
            ]
        )
    with pytest.raises(ValueError, match="immutable configuration differs"):
        main(
            [
                "--experiment-dir",
                str(experiment),
                "--output-dir",
                str(output),
                "--layers",
                "3",
                "--n-splits",
                "2",
                "--permutations",
                "0",
                "--fixed-c",
                "2.0",
                "--resume",
            ]
        )
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert config["output_dir"] == str(output.resolve())
    assert config["probe_conditions"] == list(conditions)
    assert config["provenance_validation"] == "legacy_exhaustive"
    assert config["performance"]["feature_matrix_load_count"] == 2
    assert config["performance"]["fit_count_by_model_type"] == {
        "current_answer_only_baseline": 8,
        "hidden_state_probe": 24,
    }
    assert {value["target_field"] for value in config["probe_tasks"].values()} == {
        "text_only_answer",
        "image_only_answer",
        "conflict_label",
    }
    assert analyze_main(
        ["--experiment-dir", str(experiment), "--output-dir", str(output)]
    ) == 0
    summary = json.loads((output / "probe_summary.json").read_text(encoding="utf-8"))
    assert summary["probe_conditions"] == list(conditions)
    condition_rows = [
        row
        for row in summary["hidden_state_probe"]
        if row["subset"] in conditions
    ]
    assert condition_rows
    assert all(row["sample_count"] > 0 for row in condition_rows)
    assert not (experiment / "probe").exists()

    # Identical completed runs are idempotent only with explicit --resume.
    stale_temporary = output / ".layer_probe_predictions.old-run.jsonl.tmp"
    stale_temporary.write_text("old attempt\n", encoding="utf-8")
    assert main(
        [
            "--experiment-dir",
            str(experiment),
            "--output-dir",
            str(output),
            "--layers",
            "3",
            "--n-splits",
            "2",
            "--permutations",
            "0",
            "--resume",
        ]
    ) == 0
    assert stale_temporary.read_text(encoding="utf-8") == "old attempt\n"

    torch_output = tmp_path / "torch-probe"
    assert main(
        [
            "--experiment-dir",
            str(experiment),
            "--output-dir",
            str(torch_output),
            "--manifest-path",
            str(output / "probe_manifest.jsonl"),
            "--layers",
            "3",
            "--n-splits",
            "2",
            "--version-settings",
            "v4_to_v4",
            "--backend",
            "torch",
            "--fixed-c",
            "1.0",
            "--device",
            "cpu",
            "--permutations",
            "0",
        ]
    ) == 0
    torch_metrics = json.loads(
        (torch_output / "layer_probe_metrics.json").read_text(encoding="utf-8")
    )
    assert torch_metrics["performance"]["fit_count_by_model_type"] == {
        "current_answer_only_baseline": 4,
        "hidden_state_probe": 12,
    }
    hidden_results = [
        result
        for result in torch_metrics["fold_results"]
        if result["model_type"] == "hidden_state_probe"
        and result["status"] == "valid"
    ]
    assert hidden_results
    assert all(result["fit_diagnostics"] for result in hidden_results)
    assert all(
        result["fit_diagnostics"]["binary_single_logit"]
        is (result["target_field"] == "conflict_label")
        for result in hidden_results
    )
    assert analyze_main(
        ["--experiment-dir", str(experiment), "--output-dir", str(torch_output)]
    ) == 0
