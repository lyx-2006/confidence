from __future__ import annotations

from layer_metacognition.probe.analyze_stage1_trajectory import build_case_trajectories


def test_trajectory_join_is_independent_of_input_order() -> None:
    manifest = [
        {
            "case_id": "case-b",
            "item_id": "2",
            "condition": "conflict_easy",
            "decision_side": "follows_image",
            "text_only_answer": "yellow",
            "image_only_answer": "blue",
        },
        {
            "case_id": "case-a",
            "item_id": "1",
            "condition": "conflict_hard",
            "decision_side": "follows_text",
            "text_only_answer": "red",
            "image_only_answer": "green",
        },
    ]
    answer_oof = {}
    decision_oof = {}
    sa = {}
    for index, row in enumerate(reversed(manifest)):
        base = (row["case_id"], 3, "ac")
        answer_oof[(*base, "text_answer")] = {
            "class_probabilities": {row["text_only_answer"]: 0.6}
        }
        answer_oof[(*base, "image_answer")] = {
            "class_probabilities": {row["image_only_answer"]: 0.7}
        }
        answer_oof[(*base, "conflict")] = {
            "class_probabilities": {"conflict": 0.8}
        }
        decision_oof[(*base, "decision_side")] = {
            "class_probabilities": {"follows_image": 0.4 + index * 0.1}
        }
        sa[(row["case_id"], 3)] = 0.3 + index * 0.1
    result = build_case_trajectories(
        list(reversed(manifest)),
        dict(reversed(list(answer_oof.items()))),
        dict(reversed(list(decision_oof.items()))),
        dict(reversed(list(sa.items()))),
        layers=[3],
        positions=["ac"],
    )
    assert [row["case_id"] for row in result] == ["case-a", "case-b"]
    assert all(row["R_T"] == 0.6 for row in result)
    assert all(row["R_I_preliminary"] == 0.7 for row in result)
    assert all(row["C"] == 0.8 for row in result)
