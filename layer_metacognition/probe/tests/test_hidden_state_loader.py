from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from layer_metacognition.probe import HIDDEN_STATE_DEFINITION
from layer_metacognition.probe.hidden_state_loader import HiddenStateLoader


def _write_case(
    root: Path,
    *,
    case_id: str,
    shard_name: str,
    tensor: torch.Tensor,
    layers: list[int],
    positions: list[str],
    payload_case_ids: list[str] | None = None,
    offset: int = 0,
) -> dict:
    relative = f"hidden_states/{shard_name}"
    shard_path = root / relative
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    case_ids = payload_case_ids or [case_id]
    torch.save(
        {
            "format_version": 1,
            "case_ids": case_ids,
            "layer_indices": layers,
            "position_names": positions,
            "hidden_states": tensor,
            "hidden_size": int(tensor.shape[-1]),
            "hidden_state_definition": HIDDEN_STATE_DEFINITION,
        },
        shard_path,
    )
    reference = {
        "case_id": case_id,
        "shard_path": relative,
        "offset": offset,
        "layer_indices": layers,
        "position_names": positions,
        "hidden_size": int(tensor.shape[-1]),
        "hidden_state_definition": HIDDEN_STATE_DEFINITION,
    }
    return reference


def _write_index(root: Path, references: list[dict]) -> None:
    path = root / "hidden_states" / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "cases": {
                    reference["case_id"]: reference for reference in references
                },
            }
        ),
        encoding="utf-8",
    )


def _record(reference: dict) -> dict:
    value = dict(reference)
    value.pop("case_id")
    return {
        "case_id": reference["case_id"],
        "hidden_state_reference": value,
    }


def test_single_layer_uses_payload_position_names_and_converts_float32(
    tmp_path: Path,
) -> None:
    tensor = torch.tensor([[[10, 11, 12], [20, 21, 22]]], dtype=torch.float16)
    reference = _write_case(
        tmp_path,
        case_id="single",
        shard_name="single.pt",
        tensor=tensor,
        layers=[7],
        positions=["panl", "ac"],
    )
    _write_index(tmp_path, [reference])
    loader = HiddenStateLoader(tmp_path)
    ac = loader.load_vector(_record(reference), layer=7, position_name="ac")
    panl = loader.load_vector(_record(reference), layer=7, position_name="panl")
    assert ac.dtype == np.float32
    np.testing.assert_array_equal(ac, np.asarray([20, 21, 22], dtype=np.float32))
    np.testing.assert_array_equal(panl, np.asarray([10, 11, 12], dtype=np.float32))
    assert loader.shard_load_count == 1


def test_multi_layer_lookup_and_bounded_cache(tmp_path: Path) -> None:
    tensor = torch.arange(1 * 2 * 2 * 4, dtype=torch.float16).reshape(1, 2, 2, 4)
    first = _write_case(
        tmp_path,
        case_id="multi",
        shard_name="multi.pt",
        tensor=tensor,
        layers=[19, 23],
        positions=["ac", "panl"],
    )
    second = _write_case(
        tmp_path,
        case_id="other",
        shard_name="other.pt",
        tensor=torch.ones((1, 2, 2, 4), dtype=torch.float16),
        layers=[19, 23],
        positions=["ac", "panl"],
    )
    _write_index(tmp_path, [first, second])
    loader = HiddenStateLoader(tmp_path, cache_size=1)
    actual = loader.load_vector(_record(first), layer=23, position_name="panl")
    np.testing.assert_array_equal(actual, tensor[0, 1, 1].float().numpy())
    loader.load_vector(_record(second), layer=19, position_name="ac")
    assert loader.cached_shard_count == 1
    assert loader.shard_load_count == 2


def test_loader_reads_new_named_position(tmp_path: Path) -> None:
    positions = ["ac", "panl", "ltt", "ptnl", "sac"]
    tensor = torch.arange(1 * 5 * 3, dtype=torch.float16).reshape(1, 5, 3)
    reference = _write_case(
        tmp_path,
        case_id="five",
        shard_name="five.pt",
        tensor=tensor,
        layers=[9],
        positions=positions,
    )
    _write_index(tmp_path, [reference])
    loader = HiddenStateLoader(tmp_path)
    actual = loader.load_vector(_record(reference), layer=9, position_name="sac")
    np.testing.assert_array_equal(actual, tensor[0, 4].float().numpy())


def test_bad_offset_and_case_id_mismatch_are_explicit(tmp_path: Path) -> None:
    tensor = torch.zeros((1, 2, 3), dtype=torch.float16)
    bad_offset = _write_case(
        tmp_path,
        case_id="offset",
        shard_name="offset.pt",
        tensor=tensor,
        layers=[5],
        positions=["ac", "panl"],
        offset=4,
    )
    mismatch = _write_case(
        tmp_path,
        case_id="expected",
        shard_name="mismatch.pt",
        tensor=tensor,
        layers=[5],
        positions=["ac", "panl"],
        payload_case_ids=["different"],
    )
    _write_index(tmp_path, [bad_offset, mismatch])
    loader = HiddenStateLoader(tmp_path)
    with pytest.raises(IndexError, match="Offset 4"):
        loader.load_vector(_record(bad_offset), layer=5, position_name="ac")
    with pytest.raises(ValueError, match="Case ID mismatch"):
        loader.load_vector(_record(mismatch), layer=5, position_name="ac")
