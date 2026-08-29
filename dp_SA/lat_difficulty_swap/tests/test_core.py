from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dp_SA.lat_difficulty_swap.build_pairs import build_pairs
from dp_SA.lat_difficulty_swap.config import PANL_READOUT_BY_SWAP_LAYER
from dp_SA.lat_difficulty_swap.hooks import LATSwapHook
from dp_SA.lat_difficulty_swap.io_utils import atomic_json, ensure_layout
from dp_SA.lat_difficulty_swap.metrics import directional_metrics, hard_direction, item_bootstrap
from dp_SA.lat_difficulty_swap.run import self_parity


def record(item: str, prior: int, condition: str, *, answer: str = "red", text_difficulty: float = 10, image_difficulty: float = 10, text: str | None = None) -> dict:
    return {"status": "completed", "case_id": f"{item}_{prior}_{condition}", "item_id": item, "prior_index": prior, "condition": condition, "question": "Q", "text_clue": text or f"clue-{prior}", "image_path": f"/{item}_{condition}.png", "image_sha256": f"hash-{item}-{condition}", "phase0_raw_answer": answer, "phase0_normalized_answer": answer, "phase1_answer_token_ids": [1], "outer_fold": int(item) % 5, "text_model_perceived_difficulty": text_difficulty, "image_model_perceived_difficulty": image_difficulty}


def p0(row: dict) -> dict:
    return {"status": "completed", "case_id": row["case_id"], "phase0_answer_token_ids": [1]}


class CoreTests(unittest.TestCase):
    def test_no_change_tolerance_and_decomposition(self) -> None:
        for value in (1e-9, -1e-8, 1e-7, 1e-6):
            result = directional_metrics(value, 1); self.assertTrue(result["no_change"]); self.assertEqual(result["toward_target_absolute_delta_sa"], 0)
        result = directional_metrics(2e-6, -1); self.assertTrue(result["wrong_way"]); self.assertAlmostEqual(result["toward_target_absolute_delta_sa"] - result["wrong_direction_absolute_delta_sa"], result["oriented_delta_sa"])

    def test_truncated_metric_is_not_confirmatory(self) -> None:
        values = [directional_metrics(value, 1) for value in (.01, -.01)]
        self.assertGreater(np.mean([row["toward_target_absolute_delta_sa"] for row in values]), 0)
        self.assertEqual(np.mean([row["oriented_delta_sa"] for row in values]), 0)

    def test_target_sign_and_hard_direction(self) -> None:
        self.assertTrue(hard_direction(3, 4, 1)["hard_class_toward_target"])
        self.assertTrue(hard_direction(3, 4, -1)["hard_class_wrong_way"])

    def test_image_pair_and_minimum_prior_tie(self) -> None:
        rows = []
        for prior in (0, 1):
            rows += [record("1", prior, "conflict_easy", image_difficulty=5, text=f"same-{prior}"), record("1", prior, "conflict_hard", image_difficulty=35, text=f"same-{prior}")]
        phase = [p0(row) for row in rows]; _, image, _, _ = build_pairs(rows, phase)
        self.assertEqual(len(image), 1); self.assertEqual(image[0]["easy_prior_index"], 0); self.assertEqual(image[0]["difficulty_gap"], 30)

    def test_text_pair_max_gap_and_easy_condition_tie(self) -> None:
        rows = []
        for condition in ("conflict_easy", "conflict_hard"):
            image_hash = f"hash-2-{condition}"; image_path = f"/2_{condition}.png"
            for prior, difficulty in ((0, 5), (1, 30), (2, 60)):
                row = record("2", prior, condition, text_difficulty=difficulty); row["image_sha256"] = image_hash; row["image_path"] = image_path; rows.append(row)
        # Supply an independent A-eligible item because the builder enforces smoke feasibility per arm.
        for item in ("3", "4"):
            rows += [record(item, 0, "conflict_easy", image_difficulty=5, text="same"), record(item, 0, "conflict_hard", image_difficulty=35, text="same")]
        # A second B-eligible item.
        for prior, difficulty in ((0, 5), (1, 30)):
            row = record("5", prior, "conflict_easy", text_difficulty=difficulty); row["image_sha256"] = "fixed"; row["image_path"] = "/fixed.png"; rows.append(row)
        _, _, text, _ = build_pairs(rows, [p0(row) for row in rows])
        chosen = next(row for row in text if row["item_id"] == "2"); self.assertEqual(chosen["difficulty_gap"], 55); self.assertEqual(chosen["easy_condition"], "conflict_easy")

    def test_layer_mapping(self) -> None:
        self.assertEqual(PANL_READOUT_BY_SWAP_LAYER, {10: 12, 14: 16, 16: 18, 18: 20, 22: 24, 24: 26, 26: 27})

    def test_item_bootstrap_keeps_item_rows(self) -> None:
        rows = [{"item_id": "a", "pair_id": "a", "x": 1}, {"item_id": "a", "pair_id": "a", "x": 3}, {"item_id": "b", "pair_id": "b", "x": 5}]
        result = item_bootstrap(rows, lambda row: row["x"], repeats=20, seed=1); self.assertEqual(result["item_count"], 2); self.assertEqual(result["observation_count"], 3)

    def test_resume_layout_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ensure_layout(root, resume=False); atomic_json(root / "progress" / "x.json", {"x": 1})
            with self.assertRaises(FileExistsError): ensure_layout(root, resume=False)
            ensure_layout(root, resume=True)

    def test_hook_only_changes_lat(self) -> None:
        class Modules:
            language_layers = [torch.nn.Identity()]
            num_hidden_layers = 1
            hidden_size = 3
        module = Modules(); original = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3); donor = torch.tensor([9., 8., 7.])
        hook = LATSwapHook(module, layer=0, recipient_position=2, donor_hidden=donor, prefill_length=4)
        with hook: output = module.language_layers[0](original)
        self.assertTrue(torch.equal(output[0, 2], donor)); self.assertTrue(torch.equal(output[0, :2], original[0, :2])); self.assertTrue(hook.diagnostics()["other_tokens_equal"])

    def test_self_parity_uses_registered_tolerances(self) -> None:
        clean = {"soft_sa": .5, "class_logits": list(range(9)), "hard_class": 8}; swap = {"soft_sa": .5 + 1e-7, "class_logits": [value + .01 for value in range(9)], "hard_class": 8}
        self.assertTrue(self_parity(clean, swap, {"p": .2}, {"p": .2 + 1e-11})["passed"])
        swap["hard_class"] = 7; self.assertFalse(self_parity(clean, swap, {"p": .2}, {"p": .2})["passed"])


if __name__ == "__main__": unittest.main()
