from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from confidence_test.source_attribution_schema import (
    ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
    SOURCE_ATTRIBUTION_CLASSES,
    SOURCE_ATTRIBUTION_CLASS_TEXT,
    build_source_token_specification,
)
from layer_metacognition.hidden_state_store import append_jsonl, load_jsonl
from layer_metacognition.model_adapter import HookedForwardResult
from layer_metacognition.source_patchscope import (
    PreparedSourceTarget,
    SourcePatchscopeDecoder,
)
from layer_metacognition.source_patchscope_prompts import (
    SEMANTIC_PATCHSCOPE_USER_PROMPT,
)
from layer_metacognition.valid_experiment.analyze_validation_results import (
    analyze_result_set,
    average_ranks,
    build_validation_summary,
    spearman_correlation,
)
from layer_metacognition.valid_experiment.run_semantic_patchscope_validation import (
    PreparedValidationTarget,
    SemanticValidationRunner,
    SkippableCaseError,
    ValidationSemanticPatchscopeDecoder,
    _parse_joint_candidate_output,
    _progress,
    _validate_existing_records,
    _validate_skipped_records,
    build_main_results,
    build_parser,
    build_resume_signature,
    main,
    result_columns,
    write_rebuilt_outputs,
)
from layer_metacognition.valid_experiment.semantic_variants import (
    CANONICAL_CLASS_ORDER,
    CANONICAL_CLASS_ROWS,
    CANONICAL_IMAGE_MIDPOINTS,
    CLASS_ORDERS,
    RESULT_COLUMNS,
    REVERSED_CLASS_ROWS,
    REVERSED_IMAGE_MIDPOINTS,
    SEMANTIC_VARIANT_BY_ID,
    SEMANTIC_VARIANTS,
    build_semantic_prompt,
    format_source_classes,
    select_semantic_variants,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "qwen-2.5-vl" / "models" / "Qwen2.5-VL-7B-Instruct"


def sample_detail(
    *,
    case_id: str = "1__prior_0__conflict_easy__v3__joint",
    variant_ids: tuple[str, ...] = ("base", "reverse_direction"),
    layer_count: int = 3,
) -> dict:
    layers = {}
    for layer in range(layer_count):
        semantic = {}
        for offset, variant_id in enumerate(variant_ids):
            raw = min(1.0, 0.1 * layer + 0.02 * offset)
            corrected = min(1.0, raw + 0.01)
            semantic[variant_id] = {
                "soft_image_score": raw,
                "baseline_corrected_soft_image_score": corrected,
            }
        layers[str(layer)] = {
            "answer": {
                "predicted_answer": "yellow",
                "predicted_answer_probability": 0.2 + 0.1 * layer,
            },
            "semantic_variants": semantic,
        }
    return {
        "case_id": case_id,
        "ground_truths": {"answer": "yellow", "conflict_answer": "gray"},
        "text_answer": "yellow",
        "generated_answer": "gray",
        "generated_source_class": "6",
        "layers": layers,
        "status": "completed",
    }


class SemanticVariantTests(unittest.TestCase):
    def test_base_prompt_is_byte_for_byte_current_semantic_prompt(self) -> None:
        expected = SEMANTIC_PATCHSCOPE_USER_PROMPT.format(
            source_classes=SOURCE_ATTRIBUTION_CLASS_TEXT
        )
        self.assertEqual(
            build_semantic_prompt(SEMANTIC_VARIANT_BY_ID["base"]),
            expected,
        )

    def test_reverse_semantics_and_midpoints_are_genuinely_reversed(self) -> None:
        reverse = SEMANTIC_VARIANT_BY_ID["reverse_direction"]
        self.assertEqual(reverse.class_rows, REVERSED_CLASS_ROWS)
        self.assertEqual(reverse.image_midpoints, REVERSED_IMAGE_MIDPOINTS)
        for index in CANONICAL_CLASS_ORDER:
            self.assertEqual(
                reverse.class_rows[index],
                CANONICAL_CLASS_ROWS[8 - index],
            )
            self.assertEqual(
                reverse.image_midpoints[index],
                CANONICAL_IMAGE_MIDPOINTS[8 - index],
            )

    def test_order_shuffle_only_changes_display_order(self) -> None:
        base = SEMANTIC_VARIANT_BY_ID["base"]
        for variant_id, expected_order in CLASS_ORDERS.items():
            variant = SEMANTIC_VARIANT_BY_ID[variant_id]
            with self.subTest(variant=variant_id):
                self.assertEqual(variant.class_rows, base.class_rows)
                self.assertEqual(variant.image_midpoints, base.image_midpoints)
                self.assertEqual(variant.instruction, base.instruction)
                self.assertEqual(variant.class_order, expected_order)
                rendered_rows = format_source_classes(variant).splitlines()[1:]
                self.assertEqual(
                    [int(row.split(":", 1)[0]) for row in rendered_rows],
                    list(expected_order),
                )

    def test_variant_selection_is_canonical_and_requires_base(self) -> None:
        selected = select_semantic_variants(
            ["order_shuffle_3", "base", "synonym_reliance"]
        )
        self.assertEqual(
            [variant.variant_id for variant in selected],
            ["base", "synonym_reliance", "order_shuffle_3"],
        )
        with self.assertRaisesRegex(ValueError, "include base"):
            select_semantic_variants(["reverse_direction"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_semantic_variants(["base", "base"])


class RealTokenizerTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not MODEL_PATH.is_dir():
            raise unittest.SkipTest("Local Qwen processor is unavailable")
        from transformers import AutoProcessor

        cls.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
        cls.tokenizer = getattr(cls.processor, "tokenizer", cls.processor)
        cls.token_specification = build_source_token_specification(cls.tokenizer)

        class FakeInference:
            def __init__(self, processor: object):
                self.processor = processor
                self.model = object()

            @staticmethod
            def _get_inputs_device() -> torch.device:
                return torch.device("cpu")

        def baseline_logits(
            _model: object,
            _inputs: object,
            positions: list[int],
            _modules: object,
        ) -> dict[int, torch.Tensor]:
            return {
                position: torch.zeros(32, dtype=torch.float32)
                for position in positions
            }

        with patch(
            "layer_metacognition.valid_experiment."
            "run_semantic_patchscope_validation.run_logits_forward",
            side_effect=baseline_logits,
        ):
            cls.decoder = ValidationSemanticPatchscopeDecoder(
                inference=FakeInference(cls.processor),
                modules=SimpleNamespace(),
                class_token_ids=cls.token_specification.class_token_ids,
                variants=SEMANTIC_VARIANTS,
            )

    def test_all_targets_patch_final_valid_input_token_and_baseline_once(self) -> None:
        self.assertEqual(
            self.decoder.call_counts["baseline"],
            len(SEMANTIC_VARIANTS),
        )
        for target in self.decoder.targets.values():
            with self.subTest(variant=target.variant.variant_id):
                length = int(target.inputs.input_ids.shape[1])
                self.assertEqual(target.target_position, length - 1)

    def test_prefill_is_followed_by_disjoint_raw_digit_tokens(self) -> None:
        specification = self.token_specification
        token_sets = []
        for label in SOURCE_ATTRIBUTION_CLASSES:
            raw = specification.raw_encodings[label]
            self.assertEqual(len(raw), 1)
            self.assertEqual(specification.class_token_ids[label], raw)
            combined = self.tokenizer.encode(
                ASSISTANT_SOURCE_ATTRIBUTION_PREFILL + label,
                add_special_tokens=False,
            )
            self.assertEqual(int(combined[-1]), raw[0])
            token_sets.append(set(raw))
        for index, left in enumerate(token_sets):
            for right in token_sets[index + 1 :]:
                self.assertTrue(left.isdisjoint(right))

    def test_base_decoder_matches_existing_semantic_decoder(self) -> None:
        logits = torch.linspace(-1.0, 1.0, 32)
        class FakeInference:
            def __init__(self, processor: object):
                self.processor = processor
                self.model = object()

            @staticmethod
            def _get_inputs_device() -> torch.device:
                return torch.device("cpu")

        def baseline_logits(
            _model: object,
            _inputs: object,
            positions: list[int],
            _modules: object,
        ) -> dict[int, torch.Tensor]:
            return {position: torch.zeros(32) for position in positions}

        with patch(
            "layer_metacognition.source_patchscope.run_logits_forward",
            side_effect=baseline_logits,
        ):
            existing = SourcePatchscopeDecoder(
                inference=FakeInference(self.processor),
                modules=SimpleNamespace(),
                class_token_ids=self.token_specification.class_token_ids,
                analysis_modes=["Semantic"],
            )
        new = object.__new__(ValidationSemanticPatchscopeDecoder)
        new.model = object()
        new.modules = SimpleNamespace()
        new.class_token_ids = self.token_specification.class_token_ids
        new.call_counts = {"baseline": 0, "patched": 0}
        base_variant = SEMANTIC_VARIANT_BY_ID["base"]
        new.targets = {
            "base": PreparedValidationTarget(
                variant=base_variant,
                inputs=object(),
                target_position=3,
                user_prompt=build_semantic_prompt(base_variant),
                rendered_prompt="rendered",
                baseline={
                    "class_logits": [0.0] * 9,
                    "class_probabilities": [1.0 / 9.0] * 9,
                    "hard_source_class": "0",
                    "soft_image_score": 0.5,
                },
            )
        }
        existing.targets["Semantic"] = PreparedSourceTarget(
            name="semantic",
            inputs=object(),
            target_position=3,
            rendered_prompt="rendered",
            baseline={},
        )
        with (
            patch(
                "layer_metacognition.source_patchscope."
                "run_patched_logits_forward",
                return_value=logits,
            ),
            patch(
                "layer_metacognition.valid_experiment."
                "run_semantic_patchscope_validation.run_patched_logits_forward",
                return_value=logits,
            ),
        ):
            existing_result, _ = existing.run_patched_source_readout(
                analysis_mode="Semantic",
                layer_index=0,
                source_hidden=torch.tensor([0.2, 0.8]),
            )
            new_result = new.run_patched_source_readout(
                variant_id="base",
                layer_index=0,
                source_hidden=torch.tensor([0.2, 0.8]),
            )
        self.assertEqual(
            new_result["class_logits"],
            existing_result["class_logits"],
        )
        self.assertEqual(
            new_result["class_probabilities"],
            existing_result["class_probabilities"],
        )
        self.assertEqual(
            new_result["soft_image_score"],
            existing_result["soft_image_score"],
        )


class RunnerAndResultTests(unittest.TestCase):
    class AnswerTokenizer:
        _ids = {"red": 100, "blue": 200}

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            stripped = text.strip()
            if stripped in self._ids:
                return [self._ids[stripped] + (1 if text.startswith(" ") else 0)]
            return [ord(value) for value in text]

        def decode(
            self,
            ids: list[int],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = False,
        ) -> str:
            del skip_special_tokens, clean_up_tokenization_spaces
            return "".join(chr(value) for value in ids if value < 100)

    def test_case_runs_one_source_hooked_forward_for_all_variants(self) -> None:
        variants = (SEMANTIC_VARIANT_BY_ID["base"],)

        class FakeDecoder:
            class_token_ids = {
                label: [index]
                for index, label in enumerate(SOURCE_ATTRIBUTION_CLASSES)
            }
            call_counts = {"baseline": 1, "patched": 0}

            def run_patched_source_readout(
                self,
                *,
                variant_id: str,
                layer_index: int,
                source_hidden: torch.Tensor,
            ) -> dict:
                del variant_id, source_hidden
                self.call_counts["patched"] += 1
                return {
                    "class_logits": [0.0] * 9,
                    "class_probabilities": [1.0 / 9.0] * 9,
                    "hard_source_class": "0",
                    "soft_image_score": 0.5 + 0.01 * layer_index,
                    "baseline_corrected_class_logits": [0.0] * 9,
                    "baseline_corrected_class_probabilities": [1.0 / 9.0] * 9,
                    "baseline_corrected_hard_source_class": "0",
                    "baseline_corrected_soft_image_score": 0.5,
                }

            @staticmethod
            def target_metadata() -> dict:
                return {"base": {"target_position": 9}}

        tokenizer = self.AnswerTokenizer()
        inference = SimpleNamespace(
            model=object(),
            processor=SimpleNamespace(tokenizer=tokenizer),
            _get_inputs_device=lambda: torch.device("cpu"),
        )
        modules = SimpleNamespace(
            num_hidden_layers=2,
            final_norm=torch.nn.Identity(),
            lm_head=torch.nn.Identity(),
        )
        runner = SemanticValidationRunner(
            inference=inference,
            modules=modules,
            confidence_analyzer=object(),
            decoder=FakeDecoder(),
            variants=variants,
            max_answer_tokens=24,
            max_source_tokens=4,
        )
        image_path = ROOT / "datasets" / "images" / "1_conflict_easy.png"
        case = SimpleNamespace(
            item_id="1",
            prior_index=0,
            question="Choose from: red, blue.",
            text_clue="red",
            answer_classes=["red", "blue"],
            answer_class_error=None,
            ground_truth_answer="red",
            conflict_answer="blue",
            text_answer="red",
            conditions={
                "conflict_easy": SimpleNamespace(
                    error=None,
                    resolved_image_path=str(image_path),
                )
            },
        )
        fake_inputs = SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )
        alignment = SimpleNamespace(
            rendered_ids=[1, 2, 3],
            processed_ids=[1, 2, 3],
            rendered_to_processed={0: 0, 1: 1, 2: 2},
        )
        hidden = {
            "ac": {0: torch.zeros(2), 1: torch.ones(2)},
            "sac": {0: torch.zeros(2), 1: torch.ones(2)},
        }
        hooked_result = HookedForwardResult(
            hidden_by_name=hidden,
            logits_by_position={
                0: torch.zeros(512),
                1: torch.zeros(512),
            },
        )
        locations = [
            {"position": 0, "token_id": 1, "token_text": ":"},
            {"position": 1, "token_id": 2, "token_text": ":"},
        ]
        with (
            patch(
                "layer_metacognition.valid_experiment."
                "run_semantic_patchscope_validation._generate_continuation",
                return_value=(
                    "**Answer**: blue\n**Source Attribution**:6"
                ),
            ),
            patch(
                "layer_metacognition.valid_experiment."
                "run_semantic_patchscope_validation.render_continued_assistant",
                return_value="rendered",
            ),
            patch(
                "layer_metacognition.valid_experiment."
                "run_semantic_patchscope_validation.prepare_multimodal_inputs",
                return_value=fake_inputs,
            ),
            patch(
                "layer_metacognition.valid_experiment."
                "run_semantic_patchscope_validation.build_rendered_alignment",
                return_value=alignment,
            ),
            patch(
                "layer_metacognition.valid_experiment."
                "run_semantic_patchscope_validation.locate_marker_in_assistant",
                side_effect=locations,
            ),
            patch(
                "layer_metacognition.valid_experiment."
                "run_semantic_patchscope_validation.run_hooked_forward",
                return_value=hooked_result,
            ) as hooked,
            patch(
                "layer_metacognition.valid_experiment."
                "run_semantic_patchscope_validation._answer_readout",
                return_value={
                    "predicted_answer": "blue",
                    "predicted_answer_probability": 0.75,
                    "answer_class_logits": [0.0, 1.0],
                    "answer_class_probabilities": [0.25, 0.75],
                },
            ),
        ):
            record = runner.process_case(
                case=case,
                condition="conflict_easy",
                version="v4",
            )
        hooked.assert_called_once()
        self.assertEqual(runner.call_counts["source_hooked_forward"], 1)
        self.assertEqual(runner.decoder.call_counts["patched"], 2)
        self.assertNotIn("confidence", json.dumps(record).casefold())
        self.assertNotIn("entropy", json.dumps(record).casefold())

    def test_result_rows_have_fixed_prefix_and_canonical_variant_order(self) -> None:
        variants = select_semantic_variants(
            ["order_shuffle_1", "base", "reverse_direction"]
        )
        columns = result_columns(variants)
        self.assertEqual(
            columns,
            [
                "answer",
                "answer_probability",
                "base",
                "reverse_direction",
                "order_shuffle_1",
            ],
        )
        detail = sample_detail(
            variant_ids=("base", "reverse_direction", "order_shuffle_1")
        )
        raw = build_main_results([detail], columns, corrected=False)
        corrected = build_main_results([detail], columns, corrected=True)
        for payload in (raw, corrected):
            self.assertEqual(payload["columns"][:2], ["answer", "answer_probability"])
            for row in payload["records"][0]["layers"].values():
                self.assertEqual(len(row), len(columns))
                self.assertEqual(row[0], "yellow")
                self.assertIsInstance(row[1], float)
        forbidden = {"confidence", "answer_entropy", "source_entropy"}
        self.assertTrue(forbidden.isdisjoint(columns))
        self.assertEqual(RESULT_COLUMNS[:2], ["answer", "answer_probability"])


class PersistenceTests(unittest.TestCase):
    def test_malformed_joint_output_is_a_skippable_case_error(self) -> None:
        raw = "**Answer**: white\n addCriterionation**:8"
        with self.assertRaises(SkippableCaseError) as caught:
            _parse_joint_candidate_output(raw, ["white", "black"])
        self.assertEqual(caught.exception.stage, "joint_output_parse")
        self.assertIn("addCriterionation", str(caught.exception))

    def test_jsonl_trailing_recovery_rebuilds_both_main_results(self) -> None:
        columns = ["answer", "answer_probability", "base", "reverse_direction"]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            path = output_dir / "validation_details.jsonl"
            record = sample_detail()
            append_jsonl(path, record)
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"case_id":')
            recovered = load_jsonl(path, repair_trailing=True)
            self.assertEqual(len(recovered), 1)
            write_rebuilt_outputs(output_dir, recovered, columns)
            raw_path = output_dir / "validation_results.json"
            raw_text = raw_path.read_text(encoding="utf-8")
            raw = json.loads(raw_text)
            corrected = json.loads(
                (
                    output_dir / "validation_results_corrected.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(len(raw["records"]), 1)
            self.assertEqual(len(corrected["records"]), 1)
            self.assertEqual(
                corrected["source_value_definition"],
                "baseline_corrected_soft_image_score",
            )
            layer_rows = [
                line.strip()
                for line in raw_text.splitlines()
                if line.strip().startswith('"0": [')
            ]
            self.assertEqual(len(layer_rows), 1)
            self.assertEqual(
                layer_rows[0],
                '"0": ["yellow",0.200,0.000,0.020],',
            )

    def test_existing_record_validation_rejects_duplicates_and_extras(self) -> None:
        record = sample_detail()
        expected = {record["case_id"]}
        self.assertEqual(
            _validate_existing_records([record], expected),
            expected,
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _validate_existing_records([record, record], expected)
        with self.assertRaisesRegex(ValueError, "outside"):
            _validate_existing_records(
                [sample_detail(case_id="unexpected")],
                expected,
            )

    def test_skipped_records_are_resumable_and_reduce_pending_count(self) -> None:
        completed = sample_detail(case_id="completed")
        skipped = {
            "case_id": "skipped",
            "status": "skipped",
            "error": {
                "stage": "joint_output_parse",
                "type": "SkippableCaseError",
                "message": "malformed output",
            },
        }
        self.assertEqual(
            _validate_skipped_records([skipped], {"completed", "skipped"}),
            {"skipped"},
        )
        progress = _progress(
            [completed],
            expected_count=3,
            status="running_with_skips",
            skipped_records=[skipped],
        )
        self.assertEqual(progress["completed_count"], 1)
        self.assertEqual(progress["skipped_count"], 1)
        self.assertEqual(progress["attempted_count"], 2)
        self.assertEqual(progress["pending_count"], 1)
        self.assertEqual(progress["skipped_case_ids"], ["skipped"])

    def test_resume_skips_complete_case_without_loading_model_and_config_changes_fail(
        self,
    ) -> None:
        dataset = ROOT / "datasets" / "datasets.json"
        inference_path = ROOT / "qwen-2.5-vl" / "inference.py"
        variants = select_semantic_variants(["base"])
        columns = result_columns(variants)
        fake_case = SimpleNamespace(
            item_id="1",
            prior_index=0,
            conditions={},
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            signature = build_resume_signature(
                dataset=dataset.resolve(),
                image_root=None,
                model_path=MODEL_PATH.resolve(),
                inference_path=inference_path.resolve(),
                versions=["v3"],
                conditions=["conflict_easy"],
                item_ids=None,
                prior_indices=None,
                max_items=None,
                variants=variants,
                columns=columns,
                max_answer_tokens=24,
                max_source_tokens=4,
            )
            (output_dir / "config.json").write_text(
                json.dumps({"resume_signature": signature}),
                encoding="utf-8",
            )
            append_jsonl(
                output_dir / "validation_details.jsonl",
                sample_detail(
                    variant_ids=("base",),
                    case_id="1__prior_0__conflict_easy__v3__joint",
                ),
            )
            command = [
                "--dataset",
                str(dataset),
                "--model-path",
                str(MODEL_PATH),
                "--inference-path",
                str(inference_path),
                "--output-dir",
                str(output_dir),
                "--versions",
                "v3",
                "--conditions",
                "conflict_easy",
                "--semantic-variants",
                "base",
                "--resume",
            ]
            with (
                patch(
                    "layer_metacognition.valid_experiment."
                    "run_semantic_patchscope_validation.load_evaluation_cases",
                    return_value=([fake_case], {}),
                ),
                patch(
                    "layer_metacognition.valid_experiment."
                    "run_semantic_patchscope_validation.load_runtime"
                ) as runtime_loader,
            ):
                self.assertEqual(main(command), 0)
                runtime_loader.assert_not_called()
                self.assertEqual(
                    main([*command, "--max-answer-tokens", "25"]),
                    1,
                )

    def test_parser_has_only_requested_surface(self) -> None:
        parser = build_parser()
        parsed = parser.parse_args([])
        self.assertEqual(parsed.versions, ["v3"])
        self.assertEqual(parsed.attribution_mode, "joint")
        self.assertEqual(parsed.semantic_variants, ["all"])
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertNotIn("--analysis-mode", option_strings)
        self.assertNotIn("--skip-attention", option_strings)
        self.assertNotIn("--max-confidence-tokens", option_strings)


class AnalysisTests(unittest.TestCase):
    def _payload(self, *, corrected: bool = False) -> dict:
        columns = [
            "answer",
            "answer_probability",
            "base",
            "synonym_reliance",
            "reverse_direction",
            "order_shuffle_1",
        ]
        records = []
        for case_index in range(3):
            layers = {}
            for layer in range(3):
                base = 0.1 * case_index + 0.05 * layer
                adjustment = 0.01 if corrected else 0.0
                layers[str(layer)] = [
                    "red",
                    0.8,
                    base + adjustment,
                    base + 0.01 * (case_index + layer) + adjustment,
                    1.0 - base - adjustment,
                    base + 0.02 * layer + adjustment,
                ]
            records.append({"case_id": str(case_index), "layers": layers})
        return {
            "columns": columns,
            "source_value_definition": (
                "baseline_corrected_soft_image_score"
                if corrected
                else "soft_image_score"
            ),
            "records": records,
        }

    def _config(self) -> dict:
        return {
            "comparison_layers": {"start": 0, "end": 2},
            "semantic_variants": [
                {"variant_id": "base", "group": "base"},
                {"variant_id": "synonym_reliance", "group": "synonym"},
                {"variant_id": "reverse_direction", "group": "reverse"},
                {"variant_id": "order_shuffle_1", "group": "order"},
            ],
        }

    def test_spearman_uses_average_tie_ranks_and_constant_is_null(self) -> None:
        self.assertEqual(average_ranks([2, 1, 2]), [2.5, 1.0, 2.5])
        self.assertAlmostEqual(
            spearman_correlation([1, 2, 3], [2, 4, 6]),
            1.0,
        )
        self.assertIsNone(spearman_correlation([1, 1, 1], [1, 2, 3]))

    def test_analysis_has_layer_transition_cumulative_and_base_in_group_variance(
        self,
    ) -> None:
        summary = analyze_result_set(
            self._payload(),
            self._config(),
            analysis_start_layer=0,
            analysis_end_layer=2,
        )
        synonym = summary["variants"]["synonym_reliance"]
        self.assertEqual(set(synonym["layers"]), {"0", "1", "2"})
        self.assertEqual(
            set(synonym["layer_transitions"]),
            {"0->1", "1->2"},
        )
        self.assertEqual(synonym["cumulative_change"]["start_layer"], 0)
        self.assertEqual(synonym["cumulative_change"]["end_layer"], 2)
        self.assertEqual(
            summary["group_variance"]["reverse"]["members"],
            ["base", "reverse_direction"],
        )
        self.assertIsNotNone(
            summary["group_variance"]["reverse"]["mean_population_variance"]
        )
        reverse = summary["variants"]["reverse_direction"]
        self.assertIn(
            "one_minus_variant_alignment",
            reverse["layers"]["2"],
        )
        self.assertIn(
            "one_minus_variant_alignment",
            reverse["layer_transitions"]["1->2"],
        )
        self.assertIn(
            "one_minus_variant_alignment",
            reverse["cumulative_change"],
        )

    def test_summary_analyzes_raw_and_corrected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            (input_dir / "config.json").write_text(
                json.dumps(self._config()),
                encoding="utf-8",
            )
            (input_dir / "validation_results.json").write_text(
                json.dumps(self._payload()),
                encoding="utf-8",
            )
            (
                input_dir / "validation_results_corrected.json"
            ).write_text(
                json.dumps(self._payload(corrected=True)),
                encoding="utf-8",
            )
            summary = build_validation_summary(
                input_dir,
                analysis_start_layer=0,
                analysis_end_layer=2,
            )
            self.assertEqual(set(summary), {
                "analysis_start_layer",
                "analysis_end_layer",
                "raw",
                "corrected",
            })
            self.assertEqual(
                summary["raw"]["source_value_definition"],
                "soft_image_score",
            )
            self.assertEqual(
                summary["corrected"]["source_value_definition"],
                "baseline_corrected_soft_image_score",
            )


if __name__ == "__main__":
    unittest.main()
