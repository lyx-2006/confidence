from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

from confidence_test.prompt_utils import V3_STAGE4_META_CONFIDENCE_PROMPT
from confidence_test.source_attribution_analyzer import (
    parse_joint_answer_source_output,
    parse_parallel_source_output,
)
from confidence_test.source_attribution_prompt_utils import (
    V3_STAGE4_META_SOURCE_ATTRIBUTION_PROMPT,
)
from confidence_test.source_attribution_schema import (
    SOURCE_ATTRIBUTION_CLASSES,
    SOURCE_ATTRIBUTION_CLASS_TEXT,
    SOURCE_ATTRIBUTION_MIDPOINTS,
    build_source_token_specification,
    gather_source_class_logits,
    source_distribution,
)
from layer_metacognition.analyze_source_sink_results import (
    build_source_sink_minimal,
    group_records_by_version,
    split_analysis_by_version,
    write_layer_readout_minimal,
    write_source_sink_minimal,
)
from layer_metacognition.attention_sinks import compute_attention_sink
from layer_metacognition.layer_stage_analyzer import (
    validate_restricted_reconstruction,
)
from layer_metacognition.run_v3_v4_source_experiment import (
    expand_attribution_modes,
    run_case_groups,
    share_initial_cache,
)
from layer_metacognition.token_positions import (
    locate_field_value_span,
    locate_marker_in_assistant,
)
from layer_metacognition.v3_v4_source_runner import (
    V3V4SourceRunner,
    reconstruction_tolerance,
)


class DigitTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text.startswith(" "):
            return [99, 10 + int(text.strip())]
        return [10 + int(text)]


class SimpleTokenizer:
    """Character tokenizer sufficient to exercise exact subsequence rules."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(value) for value in text]

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict:
        del add_special_tokens
        value = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            value["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return value

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(value) for value in ids)


class SourceSchemaTests(unittest.TestCase):
    def test_nine_class_boundaries_and_midpoints(self) -> None:
        self.assertEqual(
            SOURCE_ATTRIBUTION_CLASSES,
            ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
        )
        self.assertEqual(
            SOURCE_ATTRIBUTION_MIDPOINTS,
            [0.05, 0.175, 0.325, 0.4375, 0.5, 0.5625, 0.675, 0.825, 0.95],
        )
        self.assertIn(
            "8: Strongly image dominant — Image 0.900-1.000, Text 0.000-0.100",
            SOURCE_ATTRIBUTION_CLASS_TEXT,
        )

    def test_digit_token_sets_are_disjoint_and_space_is_structural(self) -> None:
        specification = build_source_token_specification(DigitTokenizer())
        self.assertEqual(specification.shared_leading_token_ids, [99])
        self.assertEqual(specification.class_token_ids["0"], [10])
        self.assertEqual(specification.class_token_ids["8"], [18])
        sets = [set(value) for value in specification.class_token_ids.values()]
        for index, left in enumerate(sets):
            for right in sets[index + 1 :]:
                self.assertTrue(left.isdisjoint(right))

    def test_restricted_distribution_and_complement_scores(self) -> None:
        ids = {label: [index] for index, label in enumerate(SOURCE_ATTRIBUTION_CLASSES)}
        logits = torch.arange(9, dtype=torch.float32)
        gathered = gather_source_class_logits(logits, ids)
        result = source_distribution(
            gathered,
            class_token_ids=ids,
            raw_output="**Source Attribution**:8",
            parsed_label="8",
        )
        self.assertAlmostEqual(sum(result.class_probabilities), 1.0, places=6)
        self.assertAlmostEqual(
            result.hard_image_score + result.hard_text_score, 1.0, places=7
        )
        self.assertAlmostEqual(
            result.soft_image_score + result.soft_text_score, 1.0, places=7
        )
        self.assertEqual(result.hard_label, "8")
        self.assertTrue(0.0 <= result.normalized_source_entropy <= 1.0)

    def test_strict_parallel_and_joint_parsing(self) -> None:
        for valid in (
            "**Source Attribution**:8",
            "**Source Attribution**: 8",
            "**Source Attribution**: 8**",
            "**Source Attribution**:8\n",
        ):
            self.assertEqual(parse_parallel_source_output(valid), "8")
        for invalid in (
            "**Source Attribution**:9",
            "**Source Attribution**:6 because image",
            "Reason\n**Source Attribution**:6",
        ):
            self.assertIsNone(parse_parallel_source_output(invalid))
        for valid in (
            "**Answer**: yellow\n**Source Attribution**:8",
            "**Answer**: yellow  \n**Source Attribution**: 8",
            "**Answer**: yellow\n\n**Source Attribution**: 8**",
            "**Answer**: yellow\n addCriterion\n8",
        ):
            self.assertEqual(
                parse_joint_answer_source_output(valid),
                ("yellow", "8", True),
            )
        for invalid in (
            "**Answer**: \n**Source Attribution**:6",
            "**Answer**: yellow\nReason\n**Source Attribution**:6",
            "**Answer**: yellow\n**Source Attribution**:9",
        ):
            self.assertEqual(
                parse_joint_answer_source_output(invalid),
                (None, None, False),
            )


class RealQwenTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model_path = (
            Path(__file__).resolve().parents[2]
            / "qwen-2.5-vl"
            / "models"
            / "Qwen2.5-VL-7B-Instruct"
        )
        if not model_path.is_dir():
            raise unittest.SkipTest("Local Qwen tokenizer is unavailable")
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )

    def test_real_qwen_digit_protocol_and_marker(self) -> None:
        specification = build_source_token_specification(self.tokenizer)
        self.assertEqual(
            [specification.class_token_ids[label][0] for label in SOURCE_ATTRIBUTION_CLASSES],
            list(range(15, 24)),
        )
        self.assertEqual(specification.shared_leading_token_ids, [220])
        assistant = "**Answer**: yellow\n**Source Attribution**:6"
        ids = self.tokenizer.encode(assistant, add_special_tokens=False)
        sac = locate_marker_in_assistant(
            self.tokenizer,
            ids,
            assistant,
            "**Source Attribution**:",
            name="sac",
        )
        self.assertEqual(ids[sac["position"] + 1], 21)


class TokenAndAttentionTests(unittest.TestCase):
    def test_marker_and_field_are_unique_token_subsequences(self) -> None:
        tokenizer = SimpleTokenizer()
        assistant = "**Answer**: yellow\n**Source Attribution**:6"
        rendered = (
            "Question\nText clue:\nblue-ish\n"
            "**Initial Answer**: red\n"
            + assistant
        )
        ids = tokenizer.encode(rendered)
        sac = locate_marker_in_assistant(
            tokenizer,
            ids,
            assistant,
            "**Source Attribution**:",
            name="sac",
        )
        self.assertEqual(ids[sac["position"]], ord(":"))
        clue = locate_field_value_span(
            tokenizer,
            ids,
            "Text clue:",
            "blue-ish",
            separator="\n",
            name="text_clue",
        )
        start, end = clue["span"]
        self.assertEqual(tokenizer.decode(ids[start:end]), "blue-ish")

    def test_non_unique_marker_fails_with_debug_information(self) -> None:
        tokenizer = SimpleTokenizer()
        assistant = "**Source Attribution**:6"
        ids = tokenizer.encode(assistant + "\n" + assistant)
        with self.assertRaisesRegex(ValueError, "found 2"):
            locate_marker_in_assistant(
                tokenizer,
                ids,
                assistant,
                "**Source Attribution**:",
                name="sac",
            )

    def test_sink_and_mass_are_distinct_per_head(self) -> None:
        attention = torch.tensor(
            [
                [
                    [
                        [0.1, 0.2, 0.3, 0.4],
                        [0.2, 0.4, 0.1, 0.3],
                        [0.3, 0.1, 0.2, 0.4],
                    ],
                    [
                        [0.4, 0.3, 0.2, 0.1],
                        [0.1, 0.1, 0.4, 0.4],
                        [0.2, 0.2, 0.3, 0.3],
                    ],
                ]
            ],
            dtype=torch.float32,
        )
        sink, mass = compute_attention_sink(
            attention,
            query_span=[1, 3],
            source_span=[0, 2],
            expected_heads=2,
        )
        expected = attention[0, :, 1:3, 0:2]
        torch.testing.assert_close(sink, expected.mean(dim=(-1, -2)))
        torch.testing.assert_close(mass, expected.sum(dim=-1).mean(dim=-1))
        torch.testing.assert_close(mass, sink * 2)
        self.assertEqual(tuple(sink.shape), (2,))


class ReadoutAndPromptTests(unittest.TestCase):
    def test_bfloat16_reconstruction_tolerance_accepts_observed_quantization(self) -> None:
        self.assertEqual(reconstruction_tolerance("bfloat16"), 0.1)
        self.assertEqual(reconstruction_tolerance("torch.bfloat16"), 0.1)
        self.assertEqual(reconstruction_tolerance("float16"), 1e-3)

    def test_reconstruction_checks_labels_probabilities_and_soft_score(self) -> None:
        reference = torch.tensor([0.0, 1.0, 2.0, 3.0])
        token_ids = {"low": [1], "high": [3]}
        passed = validate_restricted_reconstruction(
            reference.clone(),
            reference,
            labels=["low", "high"],
            class_token_ids=token_ids,
            midpoints=[0.25, 0.75],
            tolerance=1e-3,
        )
        self.assertTrue(passed["passed"])
        failed = validate_restricted_reconstruction(
            reference + torch.tensor([0.0, 0.0, 0.0, 1.0]),
            reference,
            labels=["low", "high"],
            class_token_ids=token_ids,
            midpoints=[0.25, 0.75],
            tolerance=1e-3,
        )
        self.assertFalse(failed["passed"])

    def test_parallel_prompts_do_not_leak_current_branch_outputs(self) -> None:
        confidence = V3_STAGE4_META_CONFIDENCE_PROMPT.format(
            question="Q",
            text_clue="T",
            initial_answer="red",
            initial_confidence="Likely",
            stage3_answer="blue",
            classes="classes",
        )
        source = V3_STAGE4_META_SOURCE_ATTRIBUTION_PROMPT.format(
            question="Q",
            text_clue="T",
            initial_answer="red",
            initial_confidence="Likely",
            stage3_answer="blue",
            source_classes="sources",
        )
        self.assertNotIn("Source Attribution**:6", confidence)
        self.assertNotIn("Current Confidence", source)
        self.assertIn("**Initial Confidence**: Likely", source)


class PersistenceAndResumeTests(unittest.TestCase):
    def test_layer_analysis_splits_v3_and_v4(self) -> None:
        analysis = [
            {"case_id": "a__v3__none", "version": "v3", "layers": {}},
            {"case_id": "a__v4__none", "version": "v4", "layers": {}},
            {"case_id": "b__v3__joint", "version": "v3", "layers": {}},
        ]
        split = split_analysis_by_version(analysis)
        self.assertEqual(
            [record["case_id"] for record in split["v3"]],
            ["a__v3__none", "b__v3__joint"],
        )
        self.assertEqual(
            [record["case_id"] for record in split["v4"]],
            ["a__v4__none"],
        )

    def test_analysis_records_group_v3_before_v4_stably(self) -> None:
        records = [
            {"case_id": "a__v4__none", "version": "v4"},
            {"case_id": "a__v3__none", "version": "v3"},
            {"case_id": "a__v3__parallel", "version": "v3"},
            {"case_id": "b__v4__none", "version": "v4"},
            {"case_id": "a__v3__joint", "version": "v3"},
        ]
        self.assertEqual(
            [record["case_id"] for record in group_records_by_version(records)],
            [
                "a__v3__none",
                "a__v3__parallel",
                "a__v3__joint",
                "a__v4__none",
                "b__v4__none",
            ],
        )

    def test_all_modes_finish_before_the_next_case_and_refresh_analysis(self) -> None:
        events: list[str] = []

        class FakeRunner:
            def __init__(self, mode: str) -> None:
                self.mode = mode
                self.call_counts = {"processed": 0}

            def _log(self, event: str, **fields: object) -> None:
                events.append(f"{event}:{fields['case_id']}")

            def process_case(
                self,
                *,
                case: object,
                condition: str,
                version: str,
            ) -> dict:
                self.call_counts["processed"] += 1
                case_id = (
                    f"{case.item_id}__prior_{case.prior_index}__"
                    f"{condition}__{version}__{self.mode}"
                )
                events.append(f"run:{case_id}")
                return {"case_id": case_id}

        runners = [
            FakeRunner("none"),
            FakeRunner("parallel"),
            FakeRunner("joint"),
        ]
        cases = [
            SimpleNamespace(item_id="a", prior_index=0),
            SimpleNamespace(item_id="b", prior_index=0),
        ]
        committed: list[str] = []
        snapshots: list[list[str]] = []
        counts = run_case_groups(
            runners,
            cases,
            versions=["v3"],
            conditions=["consistent_easy"],
            existing_ids=set(),
            commit=lambda record: committed.append(record["case_id"]),
            after_group=lambda: snapshots.append(list(committed)),
        )
        self.assertEqual(
            [event for event in events if event.startswith("run:")],
            [
                "run:a__prior_0__consistent_easy__v3__none",
                "run:a__prior_0__consistent_easy__v3__parallel",
                "run:a__prior_0__consistent_easy__v3__joint",
                "run:b__prior_0__consistent_easy__v3__none",
                "run:b__prior_0__consistent_easy__v3__parallel",
                "run:b__prior_0__consistent_easy__v3__joint",
            ],
        )
        self.assertEqual([len(snapshot) for snapshot in snapshots], [3, 6])
        self.assertEqual(
            counts,
            {
                "none": {"processed": 2},
                "parallel": {"processed": 2},
                "joint": {"processed": 2},
            },
        )

    def test_all_expands_and_all_runners_share_one_initial_cache(self) -> None:
        self.assertEqual(
            expand_attribution_modes("all"),
            ("none", "parallel", "joint"),
        )

        class FakeRunner:
            def __init__(self) -> None:
                self.shared_initial = {"private": {}}

            def seed_shared_initial(self, records: list[dict]) -> None:
                self.shared_initial["seeded"] = {"records": len(records)}

        runners = [FakeRunner(), FakeRunner(), FakeRunner()]
        shared = share_initial_cache(runners, [{"case_id": "existing"}])
        self.assertTrue(all(runner.shared_initial is shared for runner in runners))
        self.assertEqual(shared["seeded"], {"records": 1})
        runners[2].shared_initial["during_run"] = {"answer": "blue"}
        self.assertEqual(runners[0].shared_initial["during_run"]["answer"], "blue")

    def test_minimal_keeps_nulls_and_head_order(self) -> None:
        records = [
            {
                "case_id": "1",
                "version": "v3",
                "ground_truths": {"answer": "orange", "conflict_answer": "purple"},
                "text_answer": "orange",
                "direct_readout": {
                    "ac_layers": [
                        {
                            "layer_index": 0,
                            "predicted_answer": "orange",
                            "predicted_answer_probability": 0.21,
                            "answer_entropy": 2.209,
                        }
                    ],
                    "cc_layers": [],
                    "sac_layers": [
                        {"layer_index": 0, "soft_image_score": 0.312}
                    ],
                },
                "attention_sinks": {
                    "sac": {
                        "image": {
                            "layers": {
                                "0": {
                                    "sink_score_by_head": [0.3, 0.1, 0.2],
                                    "attention_mass_by_head": [0.6, 0.2, 0.4],
                                }
                            }
                        }
                    }
                },
            }
        ]
        analysis, stats = build_source_sink_minimal(records)
        self.assertEqual(
            analysis[0]["layers"]["0"],
            ["orange", 0.21, 2.209, None, 0.312],
        )
        self.assertEqual(
            analysis[0]["sinks"]["sac"]["image"]["0"],
            {
                "sink_score_by_head": [0.3, 0.1, 0.2],
                "attention_mass_by_head": [0.6, 0.2, 0.4],
            },
        )
        self.assertEqual(stats["invalid_layer_values"], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimal.json"
            write_source_sink_minimal(path, analysis)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded[0]["layers"]["0"][3], None)
            self.assertEqual(
                loaded[0]["sinks"]["sac"]["image"]["0"],
                {
                    "sink_score_by_head": [0.3, 0.1, 0.2],
                    "attention_mass_by_head": [0.6, 0.2, 0.4],
                },
            )
            layer_path = Path(directory) / "layers.json"
            write_layer_readout_minimal(layer_path, analysis)
            layer_loaded = json.loads(layer_path.read_text(encoding="utf-8"))
            self.assertEqual(
                layer_loaded,
                [
                    {
                        "case_id": "1",
                        "ground_truths": {
                            "answer": "orange",
                            "conflict_answer": "purple",
                        },
                        "text_answer": "orange",
                        "layers": {
                            "0": ["orange", 0.21, 2.209, None, 0.312]
                        },
                    }
                ],
            )

    def test_runner_skips_existing_ids_without_duplicate_commit(self) -> None:
        runner = object.__new__(V3V4SourceRunner)
        runner.conditions = ["consistent_easy"]
        runner.versions = ["v3", "v4"]
        runner.mode = "none"
        runner.call_counts = {}
        runner._log = MethodType(lambda self, *args, **kwargs: None, runner)

        def process(self: object, *, case: object, condition: str, version: str) -> dict:
            return {
                "case_id": (
                    f"{case.item_id}__prior_{case.prior_index}__"
                    f"{condition}__{version}__none"
                )
            }

        runner.process_case = MethodType(process, runner)
        case = SimpleNamespace(item_id="1", prior_index=0)
        existing = {"1__prior_0__consistent_easy__v3__none"}
        committed: list[str] = []
        runner.run(
            [case],
            existing_ids=existing,
            commit=lambda record: committed.append(record["case_id"]),
        )
        self.assertEqual(
            committed,
            ["1__prior_0__consistent_easy__v4__none"],
        )


if __name__ == "__main__":
    unittest.main()
