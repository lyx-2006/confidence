from __future__ import annotations

import json

import pytest
import torch

from dp_SA.SA_trajectory.CLE_transport.config import WINDOWS, parse_windows
from dp_SA.SA_trajectory.CLE_transport.prepare import prepare, select_smoke_cases, validate_frozen_inputs
from dp_SA.SA_trajectory.CLE_transport.run import add_class_list_end_plus_1, edge_for_condition


class Tokenizer:
    def decode(self, values, **_kwargs):
        return f"token-{values[0]}"


def test_window_parser_only_accepts_frozen_numeric_ranges():
    assert parse_windows(None) == WINDOWS
    assert parse_windows("8-12,18-22") == ((8, 12), (18, 22))
    with pytest.raises(ValueError):
        parse_windows("W1")
    with pytest.raises(ValueError):
        parse_windows("8-13")
    with pytest.raises(ValueError):
        parse_windows("8-12,8-12")


def test_frozen_manifest_and_smoke_selection():
    rows, hashes = validate_frozen_inputs()
    assert len(rows) == 174
    assert len({row["family_id"] for row in rows}) == 174
    selected = select_smoke_cases(rows)
    assert [row["test_side"] for row in selected] == ["high_image", "high_text"]
    assert set(hashes) == {"manifest", "historical_clean", "preprocessor_config"}


def test_prepare_resume_is_nonduplicating(tmp_path):
    first = prepare(experiment="PANL2CLE", output_root=tmp_path, smoke=True)
    second = prepare(experiment="PANL2CLE", output_root=tmp_path, smoke=True, resume=True)
    assert first["case_count"] == 2
    assert second["resumed"]
    rows = [json.loads(line) for line in (tmp_path / "artifacts/manifests/test_manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 2


def test_plus_one_order_and_experiment_edge_direction():
    located = {name: {"processed_index": value} for name, value in {
        "P1_LAT": 1, "P1_PANL": 2, "P1_PANL_PLUS_1": 3,
        "P1_CLASS_LIST_END": 6, "P1_SAC": 9,
    }.items()}
    enriched = add_class_list_end_plus_1(located, Tokenizer(), torch.arange(12).reshape(1, -1))
    assert enriched["P1_CLASS_LIST_END_PLUS_1"]["processed_index"] == 7
    positions = {key: value["processed_index"] for key, value in enriched.items()
                 if isinstance(value, dict) and "processed_index" in value}
    main, query, source = edge_for_condition("PANL2CLE", "C1_main_block", positions)
    assert (query, source, main.pairs) == ("P1_CLASS_LIST_END", "P1_PANL", ((6, 2),))
    control, query, source = edge_for_condition("CLE2SAC", "C2_source_plus_1_control", positions)
    assert (query, source, control.pairs) == ("P1_SAC", "P1_CLASS_LIST_END_PLUS_1", ((9, 7),))
