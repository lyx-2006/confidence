from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from confidence_test.prompt_utils import V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT
from layer_metacognition.hidden_state_store import TargetLayerHiddenStateStore
from layer_metacognition.probe.common import iter_jsonl
from layer_metacognition.probe_sa_no_prompt.analyze_no_sa_probe_results import (
    _detect_onset,
    clustered_bootstrap_hard,
    run_analysis,
)
from layer_metacognition.probe_sa_no_prompt.r2_analysis import (
    clustered_bootstrap_r2,
    load_hard_midpoint_map,
    pooled_r2,
    run_r2_analysis,
)
from layer_metacognition.probe_sa_no_prompt.train_no_sa_probes import (
    _join_inputs,
    run_training,
)
from layer_metacognition.probe_sa_no_prompt import SA_CLASSES
from layer_metacognition.token_positions import locate_answer_panl_position, locate_last_answer_token


class _FusedTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        result: list[int] = []
        index = 0
        while index < len(text):
            if text[index:index + 2] == "e\n":
                result.append(1000)
                index += 2
            else:
                result.append(ord(text[index]))
                index += 1
        return result

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False) -> dict[str, object]:
        del add_special_tokens
        value: dict[str, object] = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            offsets = []
            index = 0
            while index < len(text):
                width = 2 if text[index:index + 2] == "e\n" else 1
                offsets.append((index, index + width))
                index += width
            value["offset_mapping"] = offsets
        return value

    def decode(self, ids: list[int], **_: object) -> str:
        return "".join("e\n" if int(value) == 1000 else chr(int(value)) for value in ids)


class NoSAPredictionProbeTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        joint = root / "joint"
        no_sa = root / "no_sa"
        joint.mkdir()
        no_sa.mkdir()
        split = root / "split_assignments.json"
        (joint / "config.json").write_text(
            json.dumps({
                "versions": ["v4"],
                "attribution_mode": "joint",
                "source_prompt_variant": "answer_basis_9",
                "source_attribution_classes": [str(index) for index in range(9)],
                "source_attribution_midpoints": [
                    0.05,
                    0.175,
                    0.325,
                    0.4375,
                    0.5,
                    0.5625,
                    0.675,
                    0.825,
                    0.95,
                ],
            }),
            encoding="utf-8",
        )
        (no_sa / "config.json").write_text(
            json.dumps({
                "versions": ["v4"],
                "attribution_mode": "none",
                "source_prompt_variant": "baseline",
                "conditions": ["conflict_easy", "conflict_hard"],
            }),
            encoding="utf-8",
        )
        store = TargetLayerHiddenStateStore(
            no_sa,
            layer_index=[0],
            position_names=["ptnl", "pit", "ac", "lat", "panl"],
            shard_size=8,
        )
        joint_rows = []
        no_rows = []
        for item in range(6):
            for prior, condition in enumerate(("conflict_easy", "conflict_hard")):
                key = f"item-{item}-{condition}"
                label = str(item % 3)
                answer = f"answer-{item}"
                soft = 0.1 + item / 10.0
                joint_rows.append({
                    "case_id": f"joint-{key}",
                    "item_id": str(item),
                    "prior_index": prior,
                    "condition": condition,
                    "version": "v4",
                    "attribution_mode": "joint",
                    "status": "completed",
                    "generated": {
                        "current_answer": answer,
                        "source_attribution": {"parsed_label": label, "soft_image_score": soft},
                    },
                })
                result = {
                    "case_id": f"none-{key}",
                    "item_id": str(item),
                    "prior_index": prior,
                    "condition": condition,
                    "version": "v4",
                    "attribution_mode": "none",
                    "status": "completed",
                    "generated": {
                        "current_answer": answer,
                        "current_answer_result": {"raw_output": f"**Answer**: {answer}"},
                    },
                    "token_positions": {name: index for index, name in enumerate(("ptnl", "pit", "ac", "lat", "panl"))},
                    "token_position_stages": {name: "answer" for name in ("ptnl", "pit", "ac", "lat", "panl")},
                    "token_position_records": {"panl": {"token_text": "\n", "stage": "answer"}},
                }
                vector = torch.tensor(
                    [[float(item % 3), float(item), float(prior), 0.5], [0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [float(item % 3), float(item), 1.0, 2.0]],
                    dtype=torch.float32,
                )
                store.add(
                    result["case_id"],
                    vector,
                    positions=result["token_positions"],
                    stages=result["token_position_stages"],
                    result=result,
                )
                no_rows.append(result)
        store.flush(no_sa / "results.jsonl")
        (joint / "results.jsonl").write_text("\n".join(json.dumps(row) for row in joint_rows) + "\n", encoding="utf-8")
        (split).write_text(json.dumps({
            "format_version": 1,
            "n_splits": 2,
            "seed": 42,
            "group_key": "item_id",
            "item_to_fold": {str(item): item % 2 for item in range(6)},
        }), encoding="utf-8")
        return joint, no_sa, split

    def _args(self, joint: Path, no_sa: Path, split: Path, output: Path, **kwargs: object) -> SimpleNamespace:
        values = {
            "joint_experiment_dir": str(joint),
            "no_sa_experiment_dir": str(no_sa),
            "split_assignments": str(split),
            "output_dir": str(output),
            "layers": [0],
            "positions": ["ptnl", "pit", "ac", "lat", "panl"],
            "n_splits": 2,
            "seed": 42,
            "max_samples": None,
            "device": "cpu",
            "cohorts": ["answer_matched", "all_joined"],
            "bootstrap_repeats": 20,
            "resume": False,
        }
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_join_uses_composite_key_and_answer_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            joint, no_sa, split = self._fixture(Path(directory))
            joined, failures, unmatched, _assignment, provenance = _join_inputs(
                joint, no_sa, split, layers=[0], positions=["ac"], n_splits=2, max_items=None
            )
            self.assertEqual(len(joined), 12)
            self.assertFalse(failures)
            self.assertFalse(unmatched)
            self.assertEqual(provenance["answer_matched_case_count"], 12)
            self.assertNotEqual(joined[0]["joint_case_id"], joined[0]["no_sa_case_id"])

    def test_only_no_sa_hidden_store_is_used_and_training_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            joint, no_sa, split = self._fixture(root)
            output = root / "probe"
            result = run_training(self._args(joint, no_sa, split, output))
            self.assertEqual(result["status"], "training_complete")
            predictions = list(iter_jsonl(output / "predictions" / "oof_predictions.jsonl"))
            self.assertTrue(predictions)
            self.assertTrue(all(row["no_sa_case_id"].startswith("none-") for row in predictions))
            self.assertTrue(all("joint_case_id" in row for row in predictions))
            resumed = run_training(self._args(joint, no_sa, split, output, resume=True))
            self.assertEqual(resumed["status"], "training_complete")
            analysis = run_analysis(output)
            self.assertEqual(analysis["status"], "complete")
            self.assertTrue((output / "plots" / "answer_matched_hard_accuracy.png").is_file())
            self.assertTrue((output / "plots" / "answer_matched_hard_midpoint_r2.png").is_file())
            self.assertTrue((output / "results" / "answer_matched" / "hard_midpoint_r2.csv").is_file())
            self.assertTrue((output / "results" / "answer_matched" / "soft_score_r2.json").is_file())
            self.assertTrue((output / "onset.json").is_file())
            resumed_r2 = run_r2_analysis(output)
            self.assertEqual(resumed_r2["status"], "complete")

    def test_oof_r2_midpoint_mapping_and_negative_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            joint, _no_sa, _split = self._fixture(Path(directory))
            mapping = load_hard_midpoint_map(joint / "config.json")
            self.assertEqual(mapping["0"], 0.05)
            self.assertEqual(mapping["8"], 0.95)
            missing = Path(directory) / "missing-midpoints.json"
            missing.write_text(json.dumps({"source_attribution_classes": ["0"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_attribution_midpoints"):
                load_hard_midpoint_map(missing)
        self.assertEqual(pooled_r2([0.0, 1.0], [0.0, 1.0]), 1.0)
        negative = pooled_r2([0.0, 1.0], [1.0, 0.0])
        self.assertIsNotNone(negative)
        self.assertLess(float(negative), 0.0)
        rows = [
            {"item_id": "a", "true": 0.0, "predicted": 1.0},
            {"item_id": "a", "true": 0.2, "predicted": 0.8},
            {"item_id": "b", "true": 0.8, "predicted": 0.2},
            {"item_id": "b", "true": 1.0, "predicted": 0.0},
        ]
        interval = clustered_bootstrap_r2(
            rows,
            true_field="true",
            predicted_field="predicted",
            repeats=25,
            seed=42,
        )
        self.assertEqual(interval["sampling_unit"], "item_id")
        self.assertEqual(interval["item_count"], 2)
        self.assertLess(interval["upper"], 0.0)

    def test_invalid_target_and_duplicate_join_key_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            joint, no_sa, split = self._fixture(root)
            rows = list(iter_jsonl(joint / "results.jsonl"))
            rows[0]["generated"]["source_attribution"]["parsed_label"] = "invalid"
            rows.append(rows[1])
            (joint / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate joint join key"):
                _join_inputs(joint, no_sa, split, layers=[0], positions=["ac"], n_splits=2, max_items=None)

    def test_accuracy_onset_requires_two_consecutive_layers_and_bootstrap_is_item_clustered(self) -> None:
        rows = [
            {"item_id": "a", "true_label": "0", "predicted_label": "0", "majority_correct": True},
            {"item_id": "a", "true_label": "1", "predicted_label": "0", "majority_correct": False},
            {"item_id": "b", "true_label": "0", "predicted_label": "0", "majority_correct": True},
            {"item_id": "b", "true_label": "1", "predicted_label": "1", "majority_correct": False},
        ]
        bootstrap = clustered_bootstrap_hard(rows, repeats=25, seed=42)
        self.assertEqual(bootstrap["sampling_unit"], "item_id")
        self.assertEqual(bootstrap["item_count"], 2)
        self.assertIsNone(_detect_onset({10: -0.1, 12: 0.1, 14: None})["layer"])
        self.assertEqual(_detect_onset({10: 0.01, 12: 0.02, 14: 0.0})["layer"], 10)

    def test_panl_locator_accepts_answer_newline_fusion_and_keeps_lat_before_it(self) -> None:
        tokenizer = _FusedTokenizer()
        token_ids = tokenizer.encode("**Answer**: blue\n")
        panl = locate_answer_panl_position(tokenizer, token_ids, "**Answer**:", "blue")
        self.assertIn("\n", panl["token_text"])
        lat = locate_last_answer_token(tokenizer, token_ids, "**Answer**:", "blue", panl_position=panl["position"])
        self.assertLess(lat["position"], panl["position"])
        self.assertEqual(panl["validation_status"], "passed")

    def test_no_sa_prompt_and_teacher_wire_have_answer_only_contract(self) -> None:
        prompt = V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT.format(question="Q", text_clue="T")
        wire = "**Answer**: blue\n"
        self.assertNotIn("Source Attribution", prompt)
        self.assertNotIn("Source Attribution", wire)
        self.assertNotIn("SA class", prompt)
        self.assertNotIn("SA class", wire)


if __name__ == "__main__":
    unittest.main()
