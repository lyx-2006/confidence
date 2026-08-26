from __future__ import annotations

from types import SimpleNamespace

import pytest

from dp_SA.answer_force.selection import (
    EligibleCase,
    answer_type,
    build_unrelated_manifest,
    classify_origin,
    delta_e76,
    select_recipients,
)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [sum(ord(value) for value in str(text))]


def _case(item: int, origin: str, difficulty: str) -> EligibleCase:
    answer = "red" if origin == "text" else "blue"
    return EligibleCase(
        {"case_id": f"{item}|{origin}|{difficulty}", "item_id": str(item), "prior_index": 0,
         "condition": f"conflict_{difficulty}", "phase0_raw_answer": answer, "question": "Which color? Choose from: red, blue, green, yellow.",
         "text_clue": "A neutral clue.", "image_path": "/tmp/image.png", "image_sha256": "x"},
        origin, difficulty, "red", "blue", answer,
    )


def test_origin_mapping_and_exclusions():
    assert classify_origin({"phase0_raw_answer": "red"}, "red", "blue") == "text"
    assert classify_origin({"phase0_raw_answer": "blue"}, "red", "blue") == "image"
    assert classify_origin({"phase0_raw_answer": "green"}, "red", "blue") is None
    assert classify_origin({"phase0_raw_answer": "red"}, "red", "red") is None


def test_seeded_four_cell_item_disjoint_selection_and_smoke():
    eligible = [_case(item, origin, difficulty) for item in range(100) for origin in ("text", "image") for difficulty in ("easy", "hard")]
    selected, summary = select_recipients(eligible, seed=42, per_cell=25)
    assert len(selected) == 100
    assert len({row["item_id"] for row in selected}) == 100
    assert set(summary["cell_counts"].values()) == {25}
    smoke, smoke_summary = select_recipients(eligible, seed=42, per_cell=25, smoke=True)
    assert len(smoke) == 4 and len({row["item_id"] for row in smoke}) == 4
    assert set(smoke_summary["cell_counts"].values()) == {1}


def test_unrelated_donor_leakage_distance_and_token_match():
    recipient = {
        "case_id": "r", "item_id": "1", "phase0_raw_answer": "red", "text_answer": "red", "image_answer": "blue",
        "forced_opposite_answer": "blue", "question": "Which color? Choose from: red, blue, green, yellow.", "text_clue": "The sky is not mentioned.",
    }
    pool = [
        {"entry_id": "1|text|red", "item_id": "1", "role": "text", "answer": "red"},
        {"entry_id": "2|text|blue", "item_id": "2", "role": "text", "answer": "blue"},
        {"entry_id": "3|text|green", "item_id": "3", "role": "text", "answer": "green"},
        {"entry_id": "4|text|yellow", "item_id": "4", "role": "text", "answer": "yellow"},
        {"entry_id": "5|text|purple", "item_id": "5", "role": "text", "answer": "purple"},
    ]
    manifest, diagnostics = build_unrelated_manifest([recipient], pool, TinyTokenizer(), seed=42)
    assert manifest[0]["donor_item_id"] not in {"1"}
    assert manifest[0]["normalized_answer"] not in {"red", "blue"}
    assert manifest[0]["answer_token_length_equal"]
    assert manifest[0]["delta_e76_to_text"] >= 40
    assert diagnostics["exact_token_length_match_rate"] == 1.0


def test_answer_types_and_color_distance():
    assert answer_type("-12.5%") == "numeric"
    assert answer_type("blue") == "single_word"
    assert answer_type("light blue") == "phrase"
    assert delta_e76("red", "blue") >= 40
    assert delta_e76("red", "pink") >= 40
    assert delta_e76("red", "not-a-color") is None
