from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from layer_metacognition.run_sa_reliance_external_representation import (
    build_input_manifest,
    validate_output,
)
from layer_metacognition.sa_formation.core import stable_hash
from layer_metacognition.sa_formation.reliance_external_representation import (
    BRIDGE_DIR,
    EXTERNAL_REPRESENTATION_DIR,
    fit_explicit_nuisance_encoder,
    fit_external_estimand,
    fit_external_reliance_representation,
    load_measurement_rows,
    transform_explicit_nuisance,
)


def _save_hidden(path: Path, latent: float, side: float, rng: np.random.Generator) -> None:
    hidden = rng.normal(0.0, 0.02, size=(2, 28, 8))
    hidden[0, 12, 0] = latent
    hidden[0, 12, 1] = side
    hidden[0, 12, 2] = 0.6 * latent + 0.2 * side
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        hidden=hidden.astype(np.float32),
        positions=np.asarray(["pre_answer", "post_answer"]),
        layers=np.arange(28, dtype=np.int64),
    )


def _rows(root: Path, split: str, n: int, offset: int) -> list[dict]:
    rng = np.random.default_rng(11 + offset)
    answers = ("blue", "green", "red")
    rows = []
    for local in range(n):
        index = offset + local
        fold = local % 5
        latent = 1.2 * np.sin(index * 0.71) + 0.025 * local
        answer = answers[index % len(answers)]
        side_name = "image" if index % 2 else "text"
        side = 1.0 if side_name == "image" else -1.0
        difficulty = "hard" if index % 3 == 0 else "easy"
        prior = (index % 7) / 6.0
        margin = 1.0 + 0.15 * (index % 5) + 0.12 * abs(latent)
        answer_effect = {"blue": -0.25, "green": 0.1, "red": 0.35}[answer]
        nuisance_delete = 2.0 * side + 0.5 * prior + 0.2 * margin + answer_effect
        nuisance_replace = 1.5 * side - 0.3 * prior + 0.15 * margin - 0.5 * answer_effect
        deletion = latent + nuisance_delete + rng.normal(0.0, 0.015)
        replacement = 0.9 * latent + nuisance_replace + rng.normal(0.0, 0.015)
        hidden_path = root / "hidden" / split / f"case_{index}.npz"
        _save_hidden(hidden_path, latent, side, rng)
        rows.append(
            {
                "intervention_key": f"actual_reliance|{split}|case_{index}",
                "experiment": "clean_actual_source_reliance",
                "split": split,
                "status": "completed",
                "measurement_method_version": 2,
                "case_id": f"case_{index}",
                "item_id": str(index),
                "fold": fold,
                "answer_star": answer,
                "answer_star_side": side_name,
                "difficulty": difficulty,
                "condition": f"conflict_{difficulty}",
                "prior_strength": prior,
                "full_margin": margin,
                "behavior_delete_imageward": deletion,
                "behavior_replace_imageward": replacement,
                "hidden_file": str(hidden_path),
                "verbal_sa_leakage": False,
                "teacher_forced_causal_prefix_equal": True,
                "selection_measurement_same_forward": True,
            }
        )
    return rows


def _authorization(*, graded: bool) -> dict:
    payload = {
        "format_version": 1,
        "measurement_method_version": 2,
        "source": "synthetic frozen summaries",
        "summaries": {},
        "source_files": {},
        "frozen_rule": {},
        "answer_vocabulary": ["blue", "green", "red"],
        "raw_readout_allowed": True,
        "graded_candidate_allowed": graded,
        "causal_mediator_authorized": False,
        "reason": "synthetic test",
    }
    payload["authorization_fingerprint"] = stable_hash(payload)
    return payload


def test_explicit_nuisance_is_preregistered_and_training_only(tmp_path: Path) -> None:
    development = _rows(tmp_path, "development", 40, 0)
    confirmatory = _rows(tmp_path, "confirmatory", 20, 100)
    encoder = fit_explicit_nuisance_encoder(
        development,
        np.arange(30),
        ["blue", "green", "red"],
    )
    assert "full_margin" in encoder.columns
    assert not any("condition" in value for value in encoder.columns)
    assert encoder.columns[:6] == (
        "intercept",
        "choice_image",
        "choice_other",
        "difficulty_hard",
        "prior_strength",
        "full_margin",
    )
    before = encoder.to_dict()
    for row in confirmatory:
        row["full_margin"] += 10_000.0
        row["condition"] = "confirm-only-level"
    transformed = transform_explicit_nuisance(
        confirmatory, np.arange(len(confirmatory)), encoder
    )
    assert transformed.shape == (20, len(encoder.columns))
    assert encoder.to_dict() == before


def test_dev_selection_is_frozen_for_confirm_and_failed_auth_blocks_claim(
    tmp_path: Path,
) -> None:
    development = _rows(tmp_path, "development", 50, 0)
    confirmatory = _rows(tmp_path, "confirmatory", 30, 100)
    output = tmp_path / "external"
    summary = fit_external_reliance_representation(
        development,
        confirmatory,
        output,
        measurement_authorization=_authorization(graded=False),
        hidden_root=None,
        layers=[12],
        positions=["pre_answer"],
        alphas=[1.0],
        bootstrap_iterations=50,
    )
    raw = summary["estimands"]["raw_choice_coupled"]
    graded = summary["estimands"]["graded_preregistered"]
    assert raw["classification"].startswith("endpoint-coupled")
    assert raw["candidate_source_use_representation"] is False
    assert "hidden_minus_nuisance_r2" in raw["confirmatory"]
    assert "incremental_r2" not in raw["confirmatory"]
    assert raw["confirmatory"]["r2_comparison_is_nested"] is False
    assert graded["measurement_authorized"] is False
    assert graded["candidate_source_use_representation"] is False
    assert graded["readout_gate_passed"] is False
    assert isinstance(graded["statistical_pattern_passed"], bool)
    assert graded["classification"].startswith("exploratory graded")
    assert summary["causal_mediator_authorized"] is False
    assert all(
        result["causal_mediator_authorized"] is False
        for result in summary["estimands"].values()
    )
    audit = json.loads(
        (output / "graded_preregistered" / "fold_audit.json").read_text()
    )
    assert len(audit["folds"]) == 5
    assert all(
        row["confirmatory_used_for_selection_or_fit"] is False
        and row["heldout_item_overlap"] == []
        for row in audit["folds"]
    )
    assert all(
        "full_margin" in row["explicit_nuisance_columns"]
        and not any("condition" in key for key in row["explicit_nuisance_columns"])
        for row in audit["folds"]
    )
    assert sum(1 for _ in (output / "raw_choice_coupled" / "development_oof_predictions.jsonl").open()) == 50
    assert sum(1 for _ in (output / "raw_choice_coupled" / "confirmatory_frozen_predictions.jsonl").open()) == 30


def test_input_manifest_hashes_every_referenced_hidden_file(tmp_path: Path) -> None:
    development = _rows(tmp_path, "development", 10, 0)
    confirmatory = _rows(tmp_path, "confirmatory", 5, 100)
    for name in (
        "development_results.jsonl",
        "confirmatory_results.jsonl",
        "development_summary.json",
        "confirmatory_summary.json",
        "frozen_measurement_rule.json",
        "development_cohort_manifest.json",
        "confirmatory_cohort_manifest.json",
    ):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    manifest = build_input_manifest(tmp_path, development, confirmatory)
    assert manifest["hidden_file_count"] == 15
    assert manifest["file_count"] == 22
    assert len(manifest["aggregate_sha256"]) == 64
    hidden_entries = [row for row in manifest["files"] if row["kind"] == "hidden"]
    assert len(hidden_entries) == 15
    before = manifest["aggregate_sha256"]
    path = Path(development[0]["hidden_file"])
    with np.load(path) as payload:
        hidden = np.asarray(payload["hidden"])
        positions = np.asarray(payload["positions"])
        layers = np.asarray(payload["layers"])
    hidden[0, 0, 0] += 1.0
    np.savez(path, hidden=hidden, positions=positions, layers=layers)
    assert build_input_manifest(tmp_path, development, confirmatory)[
        "aggregate_sha256"
    ] != before


def test_changing_confirmatory_data_cannot_change_dev_selected_models(
    tmp_path: Path,
) -> None:
    development = _rows(tmp_path, "development", 40, 0)
    confirmatory = _rows(tmp_path, "confirmatory", 20, 100)
    altered = [dict(row) for row in confirmatory]
    rng = np.random.default_rng(123)
    for index, row in enumerate(altered):
        row["behavior_delete_imageward"] += 1000.0 * (-1) ** index
        row["behavior_replace_imageward"] -= 700.0 * (-1) ** index
        path = tmp_path / "altered" / f"case_{index}.npz"
        _save_hidden(path, float(rng.normal() * 1000), float(rng.normal() * 1000), rng)
        row["hidden_file"] = str(path)
    left = tmp_path / "left"
    right = tmp_path / "right"
    fit_external_estimand(
        development,
        confirmatory,
        left,
        estimand="graded_preregistered",
        hidden_root=None,
        layers=[12],
        positions=["pre_answer"],
        alphas=[1.0],
        answer_vocabulary=["blue", "green", "red"],
        bootstrap_iterations=20,
    )
    fit_external_estimand(
        development,
        altered,
        right,
        estimand="graded_preregistered",
        hidden_root=None,
        layers=[12],
        positions=["pre_answer"],
        alphas=[1.0],
        answer_vocabulary=["blue", "green", "red"],
        bootstrap_iterations=20,
    )
    left_audit = json.loads((left / "fold_audit.json").read_text())
    right_audit = json.loads((right / "fold_audit.json").read_text())
    assert left_audit == right_audit
    for left_entry, right_entry in zip(
        json.loads((left / "directions" / "index.json").read_text())["entries"],
        json.loads((right / "directions" / "index.json").read_text())["entries"],
        strict=True,
    ):
        assert left_entry == right_entry
        with np.load(left / "directions" / left_entry["file"]) as first, np.load(
            right / "directions" / right_entry["file"]
        ) as second:
            for key in first.files:
                assert np.array_equal(first[key], second[key])


def test_loader_rejects_mixed_measurement_versions_and_output_is_fixed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "measurement"
    root.mkdir()
    rows = _rows(tmp_path, "development", 2, 0)
    rows[1]["measurement_method_version"] = 1
    (root / "development_results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    with pytest.raises(ValueError, match="Mixed/legacy"):
        load_measurement_rows(root, "development")

    experiment = tmp_path / "experiment"
    expected = experiment / BRIDGE_DIR / EXTERNAL_REPRESENTATION_DIR
    assert validate_output(experiment) == expected.resolve()
    with pytest.raises(ValueError, match="fixed"):
        validate_output(experiment, str(tmp_path / "elsewhere"))
