from __future__ import annotations

import pytest

from dp_SA.attention_block.minimal_prompt_check.core import CONDITIONS, MINIMAL_PROMPT_TEMPLATE, WINDOWS, minimal_prompt, select_frozen_items
from dp_SA.attention_block.minimal_prompt_check.clean_screen import select_clean_candidates, select_remaining, summarize
from dp_SA.attention_block.minimal_prompt_check.gather_addon import CONDITIONS as GATHER_CONDITIONS, _edges as gather_edges
from dp_SA.attention_block.minimal_prompt_check.run import _edges


def test_prompt_is_exact_and_answer_is_raw():
    expected = """You will be shown a question, a text clue, an image, and an answer you previously provided.

Question:
Q?

Text clue:
T.

**Answer**: raw answer

State source attribution from 0 to 8: 0=text only, 4=both equally, 8=image only; intermediate integers indicate degree.
Do not choose class 4 merely because both sources were shown. Choose class 4 only if you believe the text clue and the image contributed to the fixed answer to a similar extent.
"""
    assert minimal_prompt("Q?", "T.", "raw answer") == expected
    assert MINIMAL_PROMPT_TEMPLATE.format(question="Q?", text_clue="T.", answer="raw answer") == expected
    with pytest.raises(ValueError): minimal_prompt("Q", "T", "two\nlines")


def test_registered_design_is_only_five_windows_and_three_conditions():
    assert WINDOWS == ((0,11),(4,15),(8,19),(12,23),(16,27))
    assert CONDITIONS == ("sac_to_panl", "sac_to_panl_plus_1", "empty_block_parity")


def test_frozen_selection_balanced_unique_and_ranked():
    rows=[]
    for side in ("image_side","text_side"):
        for rank in range(1,21):
            rows.append({"case_id":f"{side}-{rank}","item_id":f"{side}-{rank}","test_side":side,"selection_rank":rank,
                         "random_permutation_index":20-rank,"phase0_raw_answer":"x","phase1_inserted_raw_answer":"x"})
    selected=select_frozen_items(rows,15)
    assert len(selected)==30 and len({r["item_id"] for r in selected})==30
    assert {r["selection_rank"] for r in selected if r["test_side"]=="image_side"}==set(range(1,16))


def test_edges_are_exact_single_edges_or_empty():
    positions={"SAC":20,"PANL":7,"PANL_PLUS_1":8}
    assert _edges("sac_to_panl",positions).pairs==((20,7),)
    assert _edges("sac_to_panl_plus_1",positions).pairs==((20,8),)
    assert _edges("empty_block_parity",positions).pairs==()


def test_clean_screen_uses_only_untouched_unique_items():
    rows = [
        {"case_id": f"c{i}", "item_id": f"i{i}", "test_side": "image_side" if i < 3 else "text_side", "selection_rank": i}
        for i in range(6)
    ]
    selected = select_remaining(rows, {"c0", "c3"})
    assert [row["case_id"] for row in selected] == ["c1", "c2", "c4", "c5"]


def test_clean_candidate_selection_is_hard_matched_soft_ranked_and_balanced():
    rows = [
        {"case_id": "i1", "test_side": "image_side", "clean_class": 5, "full_clean_class": 5, "abs_soft_sa_difference": .03,
         "soft_sa_image_score": .53, "full_soft_sa_image_score": .50},
        {"case_id": "i2", "test_side": "image_side", "clean_class": 4, "full_clean_class": 5, "abs_soft_sa_difference": .01,
         "soft_sa_image_score": .51, "full_soft_sa_image_score": .52},
        {"case_id": "i3", "test_side": "image_side", "clean_class": 5, "full_clean_class": 5, "abs_soft_sa_difference": .02,
         "soft_sa_image_score": .54, "full_soft_sa_image_score": .52},
        {"case_id": "t1", "test_side": "text_side", "clean_class": 2, "full_clean_class": 2, "abs_soft_sa_difference": .04,
         "soft_sa_image_score": .40, "full_soft_sa_image_score": .36},
        {"case_id": "t2", "test_side": "text_side", "clean_class": 4, "full_clean_class": 2, "abs_soft_sa_difference": .01,
         "soft_sa_image_score": .45, "full_soft_sa_image_score": .44},
    ]
    eligible, balanced = select_clean_candidates(rows)
    assert [row["case_id"] for row in eligible] == ["i3", "i1", "t1"]
    assert [row["case_id"] for row in balanced] == ["i3", "t1"]
    report = summarize(rows)
    assert report["hard_agreement_count"] == 3
    assert report["balanced_candidate_count"] == 2


def test_minimal_gather_addon_edges_are_exact_and_matched():
    positions = {"PANL": 20, "PANL_PLUS_1": 21, "EVIDENCE_ANSWER": [2, 3, 8, 9]}
    assert GATHER_CONDITIONS == ("panl_to_evidence_answer", "panl_plus_1_to_evidence_answer")
    assert set(gather_edges(GATHER_CONDITIONS[0], positions).pairs) == {(20, 2), (20, 3), (20, 8), (20, 9)}
    assert set(gather_edges(GATHER_CONDITIONS[1], positions).pairs) == {(21, 2), (21, 3), (21, 8), (21, 9)}
