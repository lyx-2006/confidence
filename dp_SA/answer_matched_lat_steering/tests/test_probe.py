from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from dp_SA.answer_matched_lat_steering.config import POSITIONS, SMOKE_LAYERS
from dp_SA.answer_matched_lat_steering.io_utils import atomic_jsonl, atomic_npz
from dp_SA.answer_matched_lat_steering.probe import _split_indices, run_probe, shared_family_draws


def _write_probe_fixture(root: Path) -> list[dict]:
    candidates = []; clean = []; folds = []
    for index in range(20):
        case = f"c{index}"; family = f"f{index}"; hidden_file = Path("artifacts/hidden") / f"{case}.npz"
        arrays = {}
        for position_index, position in enumerate(POSITIONS):
            for layer in SMOKE_LAYERS:
                arrays[f"{position}__L{layer}"] = np.asarray([index, (index * index) % 11, np.sin(index) + position_index + layer / 100], dtype=np.float16)
        atomic_npz(root / hidden_file, arrays)
        candidates.append({"case_id": case, "family_id": family, "item_id": str(index), "image_sha256": f"hash-{index}"})
        clean.append({"status": "completed", "case_id": case, "hidden_file": str(hidden_file), "soft_sa_image_score": float(index / 25 + .03 * np.sin(index))})
        folds.append({"family_id": family, "fold": -1 if index < 16 else 0, "is_test_family": index >= 16})
    atomic_jsonl(root / "artifacts/manifests/candidate_manifest.jsonl", candidates); atomic_jsonl(root / "artifacts/manifests/fold_assignments.jsonl", folds); atomic_jsonl(root / "artifacts/diagnostics/clean_capture.jsonl", clean)
    return candidates


def test_smoke_probe_uses_train_only_scaler_and_writes_eight_cells(tmp_path: Path):
    candidates = _write_probe_fixture(tmp_path); result = run_probe(output_root=tmp_path, smoke=True, repeats=30)
    assert result["cell_count"] == 8 and result["prediction_count"] == 32
    model = joblib.load(tmp_path / "artifacts/probe/fold_models/P1_LAT__L9__fold_00.joblib")
    train_x = np.stack([np.asarray([i, (i * i) % 11, np.sin(i) + .09], dtype=np.float16).astype(np.float32) for i in range(16)])
    assert model.named_steps["scale"].mean_ == pytest.approx(train_x.mean(axis=0))
    assert run_probe(output_root=tmp_path, smoke=True, resume=True, repeats=30)["resumed_noop"]


def test_probe_split_rejects_item_or_image_leakage():
    records = [{"family_id": "a", "item_id": "same", "image_sha256": "x"}, {"family_id": "b", "item_id": "same", "image_sha256": "y"}]
    with pytest.raises(ValueError, match="leakage"):
        _split_indices(records, {"a": -1, "b": 0}, 0)


def test_shared_bootstrap_draws_are_reusable_across_positions():
    families, first = shared_family_draws(["b", "a", "c"], 20, 42); other_families, second = shared_family_draws(["c", "b", "a"], 20, 42)
    assert families == other_families and np.array_equal(first, second)
