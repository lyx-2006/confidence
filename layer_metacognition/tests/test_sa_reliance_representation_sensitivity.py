from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from layer_metacognition.run_sa_reliance_representation_sensitivity import (
    validate_output,
)
from layer_metacognition.sa_formation.core import stable_hash
from layer_metacognition.sa_formation.reliance_representation_sensitivity import (
    SENSITIVITY_DIR,
    NestedCalibrators,
    SplitLineage,
    apply_nested_calibrators,
    build_fresh_donor_records,
    fit_nested_calibrators,
    input_provenance,
    load_frozen_target_transforms,
    paired_squared_error_bootstrap,
    strict_join_split,
    summarize_nested_predictions,
)


def _join_fixture() -> tuple[list[dict], list[dict], list[dict], dict, SplitLineage]:
    result = {
        "case_id": "case-1",
        "item_id": "item-1",
        "fold": 2,
        "split": "development",
        "status": "completed",
        "measurement_method_version": 2,
        "answer_star": "blue",
        "answer_star_side": "image",
        "difficulty": "hard",
        "prior_strength": 0.4,
        "full_margin": 1.2,
        "behavior_delete_imageward": 0.7,
        "behavior_replace_imageward": 0.3,
        "selection_rendered_hash": "render-hash",
        "selection": {"messages_hash": "message-hash"},
        "manifest_fingerprint": "measurement-manifest",
    }
    analysis = {
        **result,
        "calibration_fingerprint": "measurement-calibration",
        "reliance_raw_shared": 0.1,
    }
    analysis_sha = "analysis-file-sha"
    donor = {
        "case_id": "case-1",
        "item_id": "item-1",
        "fold": 2,
        "split": "development",
        "status": "completed",
        "extension_method_version": 3,
        "answer_star": "blue",
        "answer_star_side": "image",
        "answer_star_reused": True,
        "full_messages_hash_equal": True,
        "selection_reused_without_forward": True,
        "verbal_sa_leakage": False,
        "hidden_captured": False,
        "method_v2_selection_rendered_hash": "render-hash",
        "method_v2_full_messages_hash": "message-hash",
        "reconstructed_full_messages_hash": "message-hash",
        "method_v2_analysis_sha256": analysis_sha,
        "method_v2_row_fingerprint": stable_hash(analysis),
        "manifest_fingerprint": "extension-manifest",
        "margin_repair_calibration_fingerprint": "margin-repair",
        "behavior_replace_imageward_d34_mean": 0.5,
    }
    predictions = {}
    for estimand in ("raw_choice_coupled", "graded_preregistered"):
        predictions[estimand] = [
            {
                "case_id": "case-1",
                "item_id": "item-1",
                "fold": 2,
                "split": "development",
                "estimand": estimand,
                "answer_star": "blue",
                "answer_star_side": "image",
                "target_deletion": 0.1,
                "target_replacement": 0.2,
                "target_shared": 0.15,
                "prediction_replacement": 0.25,
                "prediction_shared": 0.2,
                "prediction_nuisance": 0.0,
            }
        ]
    lineage = SplitLineage(
        measurement_analysis_sha256=analysis_sha,
        measurement_manifest_fingerprint="measurement-manifest",
        extension_manifest_fingerprint="extension-manifest",
        measurement_calibration_fingerprint="measurement-calibration",
        margin_repair_calibration_fingerprint="margin-repair",
    )
    return [result], [analysis], [donor], predictions, lineage


def test_strict_join_checks_identifiers_hashes_and_row_fingerprint() -> None:
    results, analysis, donor, predictions, lineage = _join_fixture()
    joined = strict_join_split(
        "development",
        results,
        analysis,
        donor,
        predictions,
        lineage=lineage,
    )
    assert len(joined) == 1
    assert joined[0]["item_id"] == "item-1"

    corrupted = copy.deepcopy(donor)
    corrupted[0]["method_v2_full_messages_hash"] = "another-message"
    with pytest.raises(ValueError, match="full_messages_hash"):
        strict_join_split(
            "development",
            results,
            analysis,
            corrupted,
            predictions,
            lineage=lineage,
        )

    corrupted = copy.deepcopy(donor)
    corrupted[0]["method_v2_row_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="row fingerprint"):
        strict_join_split(
            "development",
            results,
            analysis,
            corrupted,
            predictions,
            lineage=lineage,
        )

    corrupted_predictions = copy.deepcopy(predictions)
    corrupted_predictions["graded_preregistered"][0]["fold"] = 4
    with pytest.raises(ValueError, match="fold"):
        strict_join_split(
            "development",
            results,
            analysis,
            donor,
            corrupted_predictions,
            lineage=lineage,
        )


def _write_frozen_transforms(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = root / "graded_preregistered" / "directions"
    direction.mkdir(parents=True)
    columns = [
        "intercept",
        "choice_image",
        "choice_other",
        "difficulty_hard",
        "prior_strength",
        "full_margin",
        "answer=red",
    ]
    encoder = {
        "answer_vocabulary": ["blue", "red"],
        "answer_reference": "blue",
        "prior_mean": 0.0,
        "prior_scale": 1.0,
        "margin_mean": 0.0,
        "margin_scale": 1.0,
        "columns": columns,
    }
    beta = np.asarray(
        [
            [0.2, -0.1],
            [0.3, 0.4],
            [0.0, 0.0],
            [-0.2, 0.1],
            [0.1, -0.3],
            [0.05, 0.2],
            [0.4, -0.2],
        ],
        dtype=np.float64,
    )
    mean = np.asarray([0.25, -0.5], dtype=np.float64)
    scale = np.asarray([2.0, 4.0], dtype=np.float64)
    entries = []
    for fold in range(5):
        for objective in ("shared", "deletion", "replacement"):
            filename = f"fold_{fold}_{objective}.npz"
            np.savez(
                direction / filename,
                target_nuisance_beta=beta,
                target_mean=mean,
                target_scale=scale,
            )
            entries.append(
                {
                    "fold": fold,
                    "objective": objective,
                    "estimand": "graded_preregistered",
                    "file": filename,
                    "explicit_nuisance": encoder,
                }
            )
    (direction / "index.json").write_text(
        __import__("json").dumps(
            {
                "estimand": "graded_preregistered",
                "confirmatory_used_for_selection_or_fit": False,
                "entries": entries,
            }
        )
    )
    return beta, mean, scale


def test_fresh_donor_uses_frozen_target_transform_and_replays_original(
    tmp_path: Path,
) -> None:
    beta, mean, scale = _write_frozen_transforms(tmp_path)
    transforms = load_frozen_target_transforms(tmp_path, "graded_preregistered")
    joined = []
    for fold in range(5):
        measurement = {
            "case_id": f"case-{fold}",
            "item_id": f"item-{fold}",
            "fold": fold,
            "answer_star": "red" if fold % 2 else "blue",
            "answer_star_side": "image" if fold % 2 else "text",
            "difficulty": "hard" if fold % 3 == 0 else "easy",
            "prior_strength": 0.1 * fold,
            "full_margin": 1.0 + fold,
            "behavior_delete_imageward": 1.0 + fold,
            "behavior_replace_imageward": -0.5 + 0.2 * fold,
        }
        x = np.asarray(
            [
                1.0,
                float(measurement["answer_star_side"] == "image"),
                0.0,
                float(measurement["difficulty"] == "hard"),
                measurement["prior_strength"],
                measurement["full_margin"],
                float(measurement["answer_star"] == "red"),
            ]
        )
        old_y = np.asarray(
            [
                measurement["behavior_delete_imageward"],
                measurement["behavior_replace_imageward"],
            ]
        )
        expected = (old_y - x @ beta - mean) / scale
        joined.append(
            {
                "split": "development",
                "case_id": measurement["case_id"],
                "item_id": measurement["item_id"],
                "fold": fold,
                "measurement": measurement,
                "donor": {"behavior_replace_imageward_d34_mean": 2.0 + fold},
                "predictions": {
                    "graded_preregistered": {
                        "target_deletion": expected[0],
                        "target_replacement": expected[1],
                        "target_shared": expected.mean(),
                        "prediction_replacement": 0.2,
                        "prediction_shared": 0.3,
                    }
                },
            }
        )
    records = build_fresh_donor_records(
        joined, transforms, estimand="graded_preregistered"
    )
    assert len(records) == 5
    assert max(row["original_target_replay_max_abs_error"] for row in records) < 1e-12
    first = records[0]
    x0 = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    fresh_y0 = np.asarray([1.0, 2.0])
    expected_fresh = (fresh_y0 - x0 @ beta - mean) / scale
    assert first["fresh_target_replacement_m34"] == pytest.approx(expected_fresh[1])
    assert first["fresh_target_shared_d_m34"] == pytest.approx(expected_fresh.mean())
    assert first["hidden_or_readout_refit"] is False


def _prediction_rows(split: str, n: int, offset: int = 0) -> list[dict]:
    rows = []
    for index in range(n):
        nuisance = -1.0 + 2.0 * index / max(n - 1, 1)
        hidden = np.sin(index * 0.71) + 0.05 * index
        target = 0.4 + 1.7 * nuisance + 2.3 * hidden
        rows.append(
            {
                "split": split,
                "case_id": f"{split}-{offset + index}",
                "item_id": f"item-{offset + index}",
                "fold": index % 5,
                "estimand": "raw_choice_coupled",
                "target_shared": target,
                "prediction_nuisance": nuisance,
                "prediction_shared": hidden,
            }
        )
    return rows


def test_nested_calibrators_fit_development_only_and_never_refit_confirm() -> None:
    development = _prediction_rows("development", 30)
    confirmatory = _prediction_rows("confirmatory", 15, 100)
    calibrators = fit_nested_calibrators(
        development, estimand="raw_choice_coupled"
    )
    assert isinstance(calibrators, NestedCalibrators)
    assert calibrators.to_dict()["confirmatory_used_for_fit"] is False
    original = apply_nested_calibrators(confirmatory, calibrators)

    altered = copy.deepcopy(confirmatory)
    for index, row in enumerate(altered):
        row["target_shared"] += 10_000.0 * (-1) ** index
    unchanged_calibrators = fit_nested_calibrators(
        development, estimand="raw_choice_coupled"
    )
    changed_targets = apply_nested_calibrators(altered, unchanged_calibrators)
    assert np.array_equal(
        calibrators.nuisance_coefficient,
        unchanged_calibrators.nuisance_coefficient,
    )
    assert np.array_equal(
        calibrators.augmented_coefficient,
        unchanged_calibrators.augmented_coefficient,
    )
    assert [row["nuisance_calibrated_prediction"] for row in original] == [
        row["nuisance_calibrated_prediction"] for row in changed_targets
    ]
    assert [row["nuisance_plus_hidden_calibrated_prediction"] for row in original] == [
        row["nuisance_plus_hidden_calibrated_prediction"]
        for row in changed_targets
    ]
    summary = summarize_nested_predictions(original, bootstrap_iterations=20)
    assert summary["delta_r2"] > 0
    assert summary["nuisance_plus_hidden"]["mae"] < 1e-10


def test_paired_squared_error_bootstrap_is_item_clustered_and_paired() -> None:
    rows = [
        {
            "item_id": f"item-{item}",
            "paired_squared_error_improvement": 1.0,
        }
        for item in range(8)
        for _ in range(2)
    ]
    result = paired_squared_error_bootstrap(rows, iterations=50, seed=7)
    assert result["estimate"] == pytest.approx(1.0)
    assert result["ci95"] == pytest.approx([1.0, 1.0])
    assert result["valid"] == 50


def test_sensitivity_output_is_fixed_and_does_not_overlap_03(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    expected = experiment / "stage3_sa_computational_bridge" / SENSITIVITY_DIR
    assert validate_output(experiment) == expected.resolve()
    with pytest.raises(ValueError, match="fixed"):
        validate_output(experiment, str(tmp_path / "elsewhere"))


def test_aggregate_input_sha_covers_01_02_and_all_03_files(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    bridge = experiment / "stage3_sa_computational_bridge"
    measurement = bridge / "01_actual_source_reliance"
    extension = bridge / "02_donor_replication_extension"
    representation = bridge / "03_reliance_representation_devfit_confirm"
    measurement.mkdir(parents=True)
    extension.mkdir(parents=True)
    directions = representation / "raw_choice_coupled" / "directions"
    directions.mkdir(parents=True)
    for split in ("development", "confirmatory"):
        for kind in ("results", "analysis"):
            (measurement / f"{split}_{kind}.jsonl").write_text("{}\n")
        (measurement / f"{split}_cohort_manifest.json").write_text("{}")
        (extension / f"{split}_analysis.jsonl").write_text("{}\n")
        (extension / f"{split}_cohort_manifest.json").write_text("{}")
    (measurement / "frozen_measurement_rule.json").write_text("{}")
    (extension / "full_margin_protocol_repair.json").write_text("{}")
    (extension / "frozen_extension_rule.json").write_text("{}")
    prediction = representation / "raw_choice_coupled" / "development_oof_predictions.jsonl"
    prediction.write_text("{}\n")
    index = directions / "index.json"
    index.write_text("{}")
    direction = directions / "fold_0_shared.npz"
    direction.write_bytes(b"direction-v1")

    first = input_provenance(experiment)
    keys = set(first["files"])
    assert "01_actual_source_reliance/development_results.jsonl" in keys
    assert "02_donor_replication_extension/confirmatory_analysis.jsonl" in keys
    assert (
        "03_reliance_representation_devfit_confirm/raw_choice_coupled/"
        "development_oof_predictions.jsonl"
    ) in keys
    assert (
        "03_reliance_representation_devfit_confirm/raw_choice_coupled/"
        "directions/fold_0_shared.npz"
    ) in keys

    direction.write_bytes(b"direction-v2")
    second = input_provenance(experiment)
    assert second["aggregate_sha256"] != first["aggregate_sha256"]
