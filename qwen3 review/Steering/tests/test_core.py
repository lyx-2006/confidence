from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from layer_metacognition.model_adapter import LanguageModules
from Steering.contracts import (
    all_capture_keys,
    ensure_fingerprinted_config,
    hidden_key,
    parse_alphas,
    parse_layers,
    parse_positions,
)
from Steering.hooks import SelectedHiddenCapture, decoder_output_tensor
from Steering.steering import prediction_key, scaled_direction


class TensorLayer(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


class TupleLayer(torch.nn.Module):
    def forward(self, value: torch.Tensor):
        return (value + 1, "cache")


def _modules(hidden_size: int = 4) -> LanguageModules:
    layers = [TensorLayer() for _ in range(36)]
    return LanguageModules(
        language_layers=layers,
        final_norm=torch.nn.Identity(),
        lm_head=torch.nn.Linear(hidden_size, 3, bias=False),
        hidden_size=hidden_size,
        num_hidden_layers=36,
    )


def test_parameter_contracts():
    assert parse_positions(["LAT", "PANL+1", "SAC"]) == ("LAT", "PANL+1", "SAC")
    assert parse_layers([8, 20, 35]) == (8, 20, 35)
    assert parse_alphas([-2, 0, 2.5]) == (-2.0, 0.0, 2.5)
    for values in (["LAT", "LAT"], ["UNKNOWN"]):
        with pytest.raises(ValueError):
            parse_positions(values)
    for values in ([7], [36], [8, 8]):
        with pytest.raises(ValueError):
            parse_layers(values)
    for values in ([float("nan")], [1.0, 1.0]):
        with pytest.raises(ValueError):
            parse_alphas(values)


def test_capture_key_contract_is_five_positions_times_28_layers():
    keys = all_capture_keys()
    assert len(keys) == 140
    assert len(set(keys)) == 140
    assert hidden_key("LAT", 8) == "LAT__L8"
    assert hidden_key("SAC", 35) == "SAC__L35"


def test_decoder_output_accepts_qwen3_tensor_and_qwen25_tuple():
    value = torch.zeros(1, 3, 4)
    assert decoder_output_tensor(value) is value
    assert decoder_output_tensor((value, "cache")) is value
    with pytest.raises(TypeError):
        decoder_output_tensor("bad")


def test_selected_capture_records_only_requested_layers_and_positions():
    modules = _modules()
    value = torch.zeros(1, 3, 4)
    capture = SelectedHiddenCapture(
        modules,
        positions={"LAT": 0, "SAC": 2},
        layers=(8, 35),
        prefill_sequence_length=3,
    )
    with capture:
        current = value
        for layer in modules.language_layers:
            current = layer(current)
    hidden = capture.validate()
    assert set(hidden) == {"LAT", "SAC"}
    assert set(hidden["LAT"]) == {8, 35}
    assert torch.equal(hidden["LAT"][8], torch.full((4,), 9.0))
    assert torch.equal(hidden["SAC"][35], torch.full((4,), 36.0))
    assert capture.diagnostics()["capture_counts"] == {8: 1, 35: 1}


def test_tuple_layer_hook_capture():
    modules = _modules()
    modules.language_layers[8] = TupleLayer()
    capture = SelectedHiddenCapture(
        modules, positions={"LAT": 1}, layers=(8,), prefill_sequence_length=3
    )
    with capture:
        output = modules.language_layers[8](torch.zeros(1, 3, 4))
    assert isinstance(output, tuple)
    assert torch.equal(capture.validate()["LAT"][8], torch.ones(4))


def test_scaled_direction_has_three_percent_residual_norm():
    high = np.asarray([[3.0, 4.0], [4.0, 3.0]], dtype=np.float32)
    low = np.asarray([[1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    vector, metadata = scaled_direction(high, low)
    expected = 0.03 * np.linalg.norm(np.concatenate([high, low]), axis=1).mean()
    assert np.linalg.norm(vector) == pytest.approx(expected)
    assert metadata["target_vector_norm"] == pytest.approx(expected)


def test_config_resume_requires_same_fingerprint(tmp_path: Path):
    path = tmp_path / "config.json"
    first = ensure_fingerprinted_config(path, {"grid": [1]}, resume=False, label="test")
    second = ensure_fingerprinted_config(path, {"grid": [1]}, resume=True, label="test")
    assert first == second == json.loads(path.read_text())
    with pytest.raises(FileExistsError):
        ensure_fingerprinted_config(path, {"grid": [1]}, resume=False, label="test")
    with pytest.raises(ValueError, match="fingerprint"):
        ensure_fingerprinted_config(path, {"grid": [2]}, resume=True, label="test")


def test_prediction_key_is_unique_by_requested_cell():
    row = {"case_id": "c1", "position": "CLE", "layer": 12, "alpha": -2.0}
    assert prediction_key(row) == "c1|CLE|L12|a-2"

