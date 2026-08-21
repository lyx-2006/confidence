from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from layer_metacognition.hidden_state_store import TargetLayerHiddenStateStore, append_jsonl
from layer_metacognition.probe.common import iter_jsonl
from layer_metacognition.probe_sa_prediction import SA_CLASSES, prediction_key
from layer_metacognition.probe_sa_prediction.analyze_sa_probe_results import (
    hard_metrics,
    run_analysis,
    soft_metrics,
)
from layer_metacognition.probe_sa_prediction.train_sa_probes import (
    _validated_records,
    run_training,
)


class SAPredictionProbeTests(unittest.TestCase):
    def _source_experiment(self, root: Path) -> Path:
        experiment = root / "source" / "answer_basis_9"
        experiment.mkdir(parents=True)
        (experiment / "config.json").write_text(
            json.dumps(
                {
                    "source_attribution_classes": list(SA_CLASSES),
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
                }
            ),
            encoding="utf-8",
        )
        store = TargetLayerHiddenStateStore(
            experiment,
            layer_index=0,
            position_names=["ac"],
            shard_size=12,
        )
        results_path = experiment / "results.jsonl"
        for item in range(6):
            label = str(item % 3)
            for replicate in range(2):
                case_id = f"item-{item}-case-{replicate}"
                soft = (item + replicate / 2.0) / 6.0
                vector = torch.tensor(
                    [[float(item % 3), soft, float(replicate), 1.0]],
                    dtype=torch.float32,
                )
                result = {
                    "case_id": case_id,
                    "item_id": str(item),
                    "prior_index": replicate,
                    "condition": "conflict_easy" if replicate else "consistent_easy",
                    "version": "v4",
                    "attribution_mode": "joint",
                    "status": "completed",
                    "generated": {
                        "source_attribution": {
                            "parsed_label": label,
                            "soft_image_score": soft,
                        }
                    },
                }
                store.add(
                    case_id,
                    vector,
                    positions={"ac": 3},
                    stages={"ac": "joint_answer_source"},
                    result=result,
                )
        store.flush(results_path)
        return experiment

    def _args(self, experiment: Path, output: Path, *, resume: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            experiment_dir=str(experiment),
            output_dir=str(output),
            layers=[0],
            positions=["ac"],
            n_splits=2,
            seed=42,
            max_samples=6,
            device="cpu",
            resume=resume,
        )

    def test_item_cap_retains_complete_item_groups_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = self._source_experiment(Path(directory))
            records, failures, metadata = _validated_records(
                experiment,
                layers=[0],
                positions=["ac"],
                max_items=4,
            )
            self.assertFalse(failures)
            self.assertEqual(metadata["selected_item_count"], 4)
            self.assertEqual(len(records), 8)
            self.assertEqual({row["hard_label"] for row in records}, {"0", "1", "2"})
            self.assertTrue(all(0.0 <= row["soft_score"] <= 1.0 for row in records))

    def test_invalid_case_metadata_is_recorded_without_aborting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = self._source_experiment(Path(directory))
            append_jsonl(
                experiment / "results.jsonl",
                {
                    "case_id": "damaged-case",
                    "version": "v4",
                    "attribution_mode": "joint",
                    "status": "completed",
                    "prior_index": 0,
                    "condition": "conflict_easy",
                    "generated": {
                        "source_attribution": {
                            "parsed_label": "4",
                            "soft_image_score": 0.5,
                        }
                    },
                },
            )
            records, failures, _metadata = _validated_records(
                experiment,
                layers=[0],
                positions=["ac"],
                max_items=None,
            )
            self.assertEqual(len(records), 12)
            self.assertEqual(
                [row["reason"] for row in failures if row["case_id"] == "damaged-case"],
                ["missing_item_id"],
            )

    def test_hard_and_soft_metric_edge_cases(self) -> None:
        hard_rows = [
            {
                "item_id": str(index),
                "true_label": str(index % 2),
                "predicted_label": str(index % 2),
                "class_probabilities": {
                    label: (1.0 if label == str(index % 2) else 0.0)
                    for label in SA_CLASSES
                },
            }
            for index in range(6)
        ]
        hard = hard_metrics(hard_rows)
        self.assertEqual(hard["configured_class_count"], 9)
        self.assertEqual(hard["observed_class_count"], 2)
        self.assertIsNone(hard["per_class_auroc"]["8"])
        self.assertEqual(hard["balanced_accuracy"], 1.0)

        soft = soft_metrics(
            [
                {"item_id": str(index), "true_score": value, "predicted_score": value + 0.01}
                for index, value in enumerate((0.1, 0.4, 0.8))
            ]
        )
        self.assertAlmostEqual(soft["mae"], 0.01)
        self.assertFalse(soft["prediction_clipping_applied"])

    def test_end_to_end_oof_no_item_leakage_resume_and_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = self._source_experiment(root)
            output = root / "probe"
            result = run_training(self._args(experiment, output))
            self.assertEqual(result["status"], "training_complete")
            self.assertEqual(result["invalid_jobs"], 0)
            predictions = list(iter_jsonl(output / "predictions" / "oof_predictions.jsonl"))
            self.assertEqual(len(predictions), 24)
            keys = [prediction_key(row) for row in predictions]
            self.assertEqual(len(keys), len(set(keys)))
            for task in ("hard_label", "soft_score"):
                task_rows = [row for row in predictions if row["task"] == task]
                self.assertEqual(len(task_rows), 12)
                self.assertEqual(Counter(row["case_id"] for row in task_rows).most_common(1)[0][1], 1)

            split = json.loads((output / "split_assignments.json").read_text())
            for fold in range(2):
                train = {item for item, assigned in split["item_to_fold"].items() if assigned != fold}
                test = {item for item, assigned in split["item_to_fold"].items() if assigned == fold}
                self.assertFalse(train.intersection(test))

            analysis = run_analysis(output)
            self.assertEqual(analysis["hard_combination_count"], 1)
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "tables" / "layer_position_summary.csv").is_file())
            self.assertTrue((output / "plots" / "hard_label_layer_trajectory.png").is_file())
            count_before = len(predictions)
            resumed = run_training(self._args(experiment, output, resume=True))
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(
                len(list(iter_jsonl(output / "predictions" / "oof_predictions.jsonl"))),
                count_before,
            )


if __name__ == "__main__":
    unittest.main()
