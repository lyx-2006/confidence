from __future__ import annotations

import unittest

from dp_SA.lat_difficulty_swap.reanalyze import classify_answer_origin, explicit_direction, summarize_groups


class RawDeltaReanalysisTests(unittest.TestCase):
    def test_answer_origin_four_way_classification(self) -> None:
        self.assertEqual(classify_answer_origin({"answer_matches_text": True, "answer_matches_image": False}), "follow_text")
        self.assertEqual(classify_answer_origin({"answer_matches_text": False, "answer_matches_image": True}), "follow_image")
        self.assertEqual(classify_answer_origin({"answer_matches_text": True, "answer_matches_image": True}), "both_match")
        self.assertEqual(classify_answer_origin({"answer_matches_text": False, "answer_matches_image": False}), "neither_match")

    def test_direction_is_donor_to_recipient(self) -> None:
        self.assertEqual(explicit_direction("hard", "easy"), "E→H")
        self.assertEqual(explicit_direction("easy", "hard"), "H→E")
        with self.assertRaises(ValueError):
            explicit_direction("easy", "easy")

    def test_summary_is_untransformed_arithmetic_mean(self) -> None:
        rows = [
            {"arm": "A", "layer": 10, "delta_sa": -0.2, "pair_id": "p1", "item_id": "i1"},
            {"arm": "A", "layer": 10, "delta_sa": 0.4, "pair_id": "p2", "item_id": "i2"},
        ]
        result = summarize_groups(rows, ("arm", "layer"))[0]
        self.assertAlmostEqual(result["mean_delta_sa"], 0.1)
        self.assertNotIn("oriented", result)
        self.assertNotIn("absolute", result)


if __name__ == "__main__":
    unittest.main()
