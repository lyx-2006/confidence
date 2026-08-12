from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from layer_metacognition.sa_formation.reliance_representation import (
    PanelCell,
    analyze_history_zero_shot,
    derive_reliance_indicators,
    fit_reliance_representation,
    load_hidden_panel,
)


def _save_panel(
    path: Path,
    *,
    latent: float,
    rng: np.random.Generator,
    transpose: bool = False,
) -> None:
    layers = np.arange(28, dtype=np.int64)
    positions = np.asarray(["pre_answer", "post_answer"])
    hidden = rng.normal(0.0, 0.08, size=(2, 28, 8))
    # The formal synthetic representation is localized to pre-answer L12.
    hidden[0, 12, 0] = latent + rng.normal(0.0, 0.015)
    hidden[0, 12, 1] = 0.5 * latent + rng.normal(0.0, 0.015)
    values = hidden.transpose(1, 0, 2) if transpose else hidden
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        hidden=values.astype(np.float32),
        layers=layers,
        positions=positions,
    )


def _synthetic_measurements(root: Path, n: int = 60) -> list[dict]:
    rng = np.random.default_rng(42)
    answers = ("cyan", "brown", "orange")
    rows = []
    for index in range(n):
        latent = float(rng.normal())
        answer = answers[index % len(answers)]
        final_side = "image" if (index // 2) % 2 else "text"
        answer_offset = {"cyan": -0.5, "brown": 0.2, "orange": 0.7}[answer]
        side_offset = 0.35 if final_side == "image" else -0.35
        deletion = latent + answer_offset + side_offset + rng.normal(0.0, 0.04)
        replacement = (
            0.85 * latent - 0.4 * answer_offset + 0.6 * side_offset
            + rng.normal(0.0, 0.04)
        )
        hidden_file = root / "hidden" / f"case_{index}.npz"
        _save_panel(
            hidden_file,
            latent=latent,
            rng=rng,
            transpose=index == 0,
        )
        rows.append(
            {
                "metadata": {
                    "case_id": f"case_{index}",
                    "item_id": str(index),
                    "fold": index % 5,
                    "fixed_answer": answer,
                    "final_side": final_side,
                    "difficulty": "hard" if index % 3 == 0 else "easy",
                    "prior_strength": float((index % 7) / 7),
                    "answer_margin": 1.0 + 0.1 * (index % 5),
                    "hidden_file": str(hidden_file),
                },
                "deletion": deletion,
                # Exercise the measurement runner's two replacement donors.
                "replacement_d1": replacement - 0.02,
                "replacement_d2": replacement + 0.02,
            }
        )
    return rows


def test_indicator_aliases_and_raw_measurement_reconstruction() -> None:
    deletion, replacement = derive_reliance_indicators(
        {"deletion": 1.0, "replacement_d1": 2.0, "replacement_d2": 4.0}
    )
    assert deletion == 1.0
    assert replacement == 3.0

    deletion, replacement = derive_reliance_indicators(
        {
            "measurements": {
                "no_text": {"fixed_answer_log_probability": -1.0},
                "no_image": {"fixed_answer_log_probability": -3.0},
                "replace_text_d1": {"fixed_answer_log_probability": -2.0},
                "replace_image_d1": {"fixed_answer_log_probability": -4.0},
                "replace_text_d2": {"fixed_answer_log_probability": -1.5},
                "replace_image_d2": {"fixed_answer_log_probability": -2.5},
            }
        }
    )
    assert deletion == 2.0
    assert replacement == 1.5


def test_panel_loader_accepts_layers_first_layout(tmp_path: Path) -> None:
    path = tmp_path / "transposed.npz"
    _save_panel(path, latent=2.0, rng=np.random.default_rng(3), transpose=True)
    panel = load_hidden_panel(
        {"hidden_file": str(path)},
        layers=[12],
        positions=["pre_answer", "post_answer"],
    )
    assert set(panel) == {
        PanelCell(12, "pre_answer"),
        PanelCell(12, "post_answer"),
    }
    assert panel[PanelCell(12, "pre_answer")].shape == (8,)
    assert panel[PanelCell(12, "pre_answer")][0] > 1.8


def test_nested_oof_and_zero_shot_history(tmp_path: Path) -> None:
    rows = _synthetic_measurements(tmp_path)
    output = tmp_path / "representation"
    summary = fit_reliance_representation(
        rows,
        output,
        layers=[8, 12],
        positions=["pre_answer", "post_answer"],
        alphas=[0.1, 1.0],
        min_reliable_n=40,
        bootstrap_iterations=100,
    )
    assert summary["target_reliability"]["gate_passed"] is True
    assert summary["nested_panel_oof"]["r2"] > 0.5
    assert summary["incremental_oof_r2"] > 0
    assert summary["representation_gate_passed"] is True
    assert (output / "oof_predictions.jsonl").is_file()
    direction_index = json.loads(
        (output / "directions" / "index.json").read_text(encoding="utf-8")
    )
    assert len(direction_index["entries"]) == 5 * 3
    assert all(entry["item_overlap"] == [] for entry in json.loads(
        (output / "fold_audit.json").read_text(encoding="utf-8")
    )["folds"])

    rng = np.random.default_rng(19)
    history_rows = []
    for index in range(20):
        answer = ("cyan", "brown", "orange")[index % 3]
        final_side = "image" if index % 2 else "text"
        shift = 0.6 + 0.035 * index
        nuisance = {
            "answer_identity": answer,
            "final_side": final_side,
            "difficulty": "easy",
            "prior_strength": 0.2,
            "answer_margin": 1.2,
        }
        text_path = tmp_path / "history" / f"tf_{index}.npz"
        image_path = tmp_path / "history" / f"if_{index}.npz"
        _save_panel(text_path, latent=-0.2, rng=rng)
        _save_panel(image_path, latent=-0.2 + shift, rng=rng)
        history_rows.append(
            {
                "case_id": f"history_{index}",
                "item_id": f"h{index}",
                "fold": index % 5,
                "fixed_answer": answer,
                "nuisance": nuisance,
                "history_hidden": {
                    "text_first": {"hidden_file": str(text_path)},
                    "image_first": {"hidden_file": str(image_path)},
                },
                "delta_history_delete": shift,
                "delta_history_replace": 0.9 * shift,
                "old_delta_sa_if_minus_tf": 0.5 * shift,
                "answer_only_natural_endpoints": {
                    "text_first": answer,
                    "image_first": answer,
                },
            }
        )
    history = analyze_history_zero_shot(
        history_rows,
        output / "directions",
        output_dir=tmp_path / "history_analysis",
        min_primary_n=15,
        bootstrap_iterations=100,
    )
    assert history["primary_endpoint_matched_n"] == 20
    assert history["primary"]["delta_z_reliance"]["mean"] > 0
    assert history["population_shift_gate_passed"] is True
    assert (tmp_path / "history_analysis" / "results.jsonl").is_file()


def test_outer_item_fold_leakage_is_rejected(tmp_path: Path) -> None:
    rows = [
        {
            "case_id": f"case_{index}",
            "item_id": "shared" if index < 2 else str(index),
            "fold": index,
            "fixed_answer": "cyan",
            "final_side": "image",
            "deletion": 1.0 + index,
            "replacement": 1.0 + index,
            "hidden_file": str(tmp_path / "missing.npz"),
        }
        for index in range(3)
    ]
    with pytest.raises(RuntimeError, match="Outer item-fold leakage"):
        fit_reliance_representation(
            rows,
            tmp_path / "out",
            layers=[12],
            positions=["pre_answer"],
            alphas=[1.0],
            min_reliable_n=1,
            bootstrap_iterations=10,
        )

