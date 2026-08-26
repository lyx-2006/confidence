from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from dp_SA.answer_force.analyze import bootstrap_aggregates, hard_class_directional_proportions, item_metrics, regression_results, specificity_results


def _rows():
    rows = []
    for item, origin, difficulty, direction in (("1", "text", "easy", -1), ("2", "image", "hard", 1), ("3", "text", "hard", -1), ("4", "image", "easy", 1)):
        for condition, panl, final, hard, pseudo, length in (("clean", .4, .5, 4, 4, 1), ("force_opposite", .6, .7, 5, 5, 1), ("force_unrelated", .3, .45, 3, 3, 2)):
            rows.append({"status": "completed", "trial_key": f"{item}|{condition}", "case_id": f"case-{item}", "item_id": item, "condition": condition, "origin": origin, "difficulty": difficulty, "forced_direction": direction, "phase0_raw_answer": "red", "fixed_answer": "red", "text_answer": "red", "image_answer": "blue", "question": "q", "text_clue": "c", "image_path": "i", "phase1_answer_span": [1, 2], "phase1_answer_token_ids": [1], "answer_token_length": length, "panl_sa": panl, "final_soft_sa": final, "final_hard_class": hard, "panl_pseudo_hard_class": pseudo, "final_class_logits": [0.0] * 9, "final_class_probabilities": [1 / 9] * 9, "hidden_file": "h"})
    return rows


def test_item_metrics_deltas_direction_and_label_changes():
    rows = item_metrics(_rows())
    assert len(rows) == 4
    text = next(row for row in rows if row["origin"] == "text")
    assert text["opposite_panl_delta"] == pytest.approx(.2)
    assert text["opposite_panl_oriented_delta"] == pytest.approx(-.2)
    assert text["opposite_final_hard_change"] == 1
    assert text["paired_abs_contrast_panl"] == pytest.approx(0.1)


def test_joint_bootstrap_and_regression_outputs():
    rows = item_metrics(_rows())
    aggregate, bootstrap = bootstrap_aggregates(rows, repeats=20, seed=42)
    assert len(bootstrap) == 5 * 20
    all_rows = [row for row in aggregate if row["group"] == "all"]
    assert any(row["metric"] == "paired_abs_contrast_panl" and row["valid_bootstrap_repeats"] == 20 for row in all_rows)
    regressions = regression_results(rows)
    assert any(row["condition"] == "opposite" and row["outcome"] == "oriented" for row in regressions)
    specificity = specificity_results(aggregate, bootstrap)
    assert {row["metric"] for row in specificity if row["group"] == "all"} == {"D_panl_opp", "D_panl_unrel", "C_panl", "D_final_opp", "D_final_unrel", "C_final"}
    hard = hard_class_directional_proportions(rows)
    final_all = [row for row in hard if row["group"] == "all" and row["endpoint"] == "final"]
    assert sum(row["proportion"] for row in final_all) == pytest.approx(1.0)
