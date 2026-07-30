from __future__ import annotations

import argparse
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
    ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
    SOURCE_ATTRIBUTION_CLASSES,
    SOURCE_ATTRIBUTION_CLASS_TEXT,
    SOURCE_ATTRIBUTION_MIDPOINTS,
    build_source_token_specification,
    gather_source_class_logits,
    source_distribution,
)
from layer_metacognition.conversation_builder import (
    prepare_multimodal_inputs,
    render_continued_assistant,
)
from layer_metacognition.analyze_source_sink_results import (
    COMPACT_LAYER_COLUMNS,
    COMPACT_LAYER_COLUMNS_WITH_ANSWER_VAL,
    build_answer_validation_summary,
    build_source_sink_minimal,
    build_source_sink_summary,
    group_records_by_version,
    split_analysis_by_version,
    write_layer_readout_minimal,
    write_source_sink_minimal,
)
from layer_metacognition.attention_sinks import compute_attention_sink
from layer_metacognition.layer_stage_analyzer import (
    source_layer_readout,
    validate_restricted_reconstruction,
)
from layer_metacognition.hidden_state_store import (
    TargetLayerHiddenStateStore,
    load_jsonl,
)
from layer_metacognition.direct_readout import (
    build_first_token_collision_report,
    project_hidden_to_vocab,
)
from layer_metacognition.model_adapter import (
    LanguageModules,
    run_patched_logits_forward,
)
from layer_metacognition.run_v3_v4_source_experiment import (
    FORMAT_VERSION,
    build_parser,
    expand_attribution_modes,
    normalize_analysis_modes,
    normalize_save_hidden_states,
    parse_save_hidden_state,
    run_case_groups,
    saved_configuration_for_comparison,
    share_initial_cache,
    validate_save_hidden_state,
    validate_resume_format,
)
from layer_metacognition.source_patchscope import (
    ANSWER_PATCHSCOPE_VARIANTS,
    AnswerPatchscopeDecoder,
    PreparedSourceTarget,
    SourcePatchscopeDecoder,
    answer_candidate_orders,
)
from layer_metacognition.source_patchscope_prompts import (
    IDENTITY_PATCHSCOPE_ASSISTANT_PREFILL,
    IDENTITY_PATCHSCOPE_USER_PROMPT,
    SEMANTIC_ANSWER_PATCHSCOPE_USER_PROMPT,
    SEMANTIC_PATCHSCOPE_USER_PROMPT,
)
from layer_metacognition.token_positions import (
    encode_without_special_tokens,
    locate_field_value_span,
    locate_marker_in_assistant,
    unique_subsequence,
)
from layer_metacognition.token_spans import build_rendered_alignment
from layer_metacognition.v3_v4_source_runner import (
    V3V4SourceRunner,
    capture_target_layer_hidden_states,
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
        from transformers import AutoProcessor

        cls.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        cls.tokenizer = getattr(cls.processor, "tokenizer", cls.processor)

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

    def test_real_qwen_patchscope_targets_end_at_patch_positions(self) -> None:
        semantic_user = SEMANTIC_PATCHSCOPE_USER_PROMPT.format(
            source_classes=SOURCE_ATTRIBUTION_CLASS_TEXT
        )
        targets = (
            (
                "Identity",
                IDENTITY_PATCHSCOPE_USER_PROMPT,
                IDENTITY_PATCHSCOPE_ASSISTANT_PREFILL,
            ),
            (
                "Semantic",
                semantic_user,
                ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
            ),
        )
        for mode, user_prompt, assistant_prefill in targets:
            with self.subTest(mode=mode):
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_prefill}],
                    },
                ]
                rendered = render_continued_assistant(
                    self.processor,
                    messages,
                    assistant_prefill,
                )
                inputs = prepare_multimodal_inputs(
                    self.processor,
                    messages,
                    rendered,
                    device="cpu",
                )
                alignment = build_rendered_alignment(
                    self.tokenizer,
                    rendered,
                    inputs.input_ids,
                    inputs.attention_mask,
                )
                if mode == "Identity":
                    assistant_ids = encode_without_special_tokens(
                        self.tokenizer,
                        assistant_prefill,
                    )
                    assistant_start, _ = unique_subsequence(
                        alignment.rendered_ids,
                        assistant_ids,
                        name="identity assistant",
                    )
                    _placeholder_start, placeholder_end = unique_subsequence(
                        assistant_ids,
                        encode_without_special_tokens(self.tokenizer, "?"),
                        name="identity placeholder",
                    )
                    position = alignment.rendered_to_processed[
                        assistant_start + placeholder_end - 1
                    ]
                else:
                    position = locate_marker_in_assistant(
                        self.tokenizer,
                        alignment.rendered_ids,
                        assistant_prefill,
                        ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
                        name="semantic sac",
                        position_map=alignment.rendered_to_processed,
                        processed_ids=alignment.processed_ids,
                    )["position"]
                self.assertEqual(position, len(alignment.processed_ids) - 1)

    def test_patchscope_decoder_prepares_and_reuses_real_qwen_targets(self) -> None:
        layers = [
            AddDecoderLayer(tuple_output=True),
            AddDecoderLayer(tuple_output=True),
        ]
        model = FakePatchModel(layers, vocab_size=32)
        modules = LanguageModules(
            language_layers=list(model.layers),
            final_norm=torch.nn.Identity(),
            lm_head=model.lm_head,
            hidden_size=2,
            num_hidden_layers=2,
        )

        class FakeInference:
            def __init__(self, processor: object, fake_model: torch.nn.Module):
                self.processor = processor
                self.model = fake_model

            @staticmethod
            def _get_inputs_device() -> torch.device:
                return torch.device("cpu")

        token_ids = {
            label: [15 + index]
            for index, label in enumerate(SOURCE_ATTRIBUTION_CLASSES)
        }
        decoder = SourcePatchscopeDecoder(
            inference=FakeInference(self.processor, model),
            modules=modules,
            class_token_ids=token_ids,
            analysis_modes=["Semantic", "Identity"],
        )
        self.assertEqual(decoder.call_counts["baseline"], 2)
        self.assertEqual(set(decoder.baselines()), {"Identity", "Semantic"})
        for mode in ("Identity", "Semantic"):
            result, logits = decoder.run_patched_source_readout(
                analysis_mode=mode,
                layer_index=0,
                source_hidden=torch.tensor([0.25, 0.75]),
            )
            self.assertEqual(result["analysis_mode"], mode)
            self.assertEqual(result["layer_index"], 0)
            self.assertAlmostEqual(sum(result["class_probabilities"]), 1.0)
            self.assertEqual(tuple(logits.shape), (32,))
        self.assertEqual(decoder.call_counts, {"baseline": 2, "patched": 2})

    def test_answer_patchscope_caches_target_and_reconstructs_final_layer(
        self,
    ) -> None:
        candidates = ["red", "blue"]
        collision_report = build_first_token_collision_report(
            self.tokenizer,
            candidates,
        )
        self.assertFalse(collision_report["collisions"])
        token_ids = {
            label: collision_report["labels"][label]["first_token_variants"]
            for label in candidates
        }
        vocab_size = max(
            token_id
            for values in token_ids.values()
            for token_id in values
        ) + 1
        layers = [
            AddDecoderLayer(tuple_output=True),
            AddDecoderLayer(tuple_output=True),
        ]
        model = FakePatchModel(layers, vocab_size=vocab_size)
        with torch.no_grad():
            for token_id in token_ids["red"]:
                model.lm_head.weight[token_id] = torch.tensor([1.0, 0.0])
            for token_id in token_ids["blue"]:
                model.lm_head.weight[token_id] = torch.tensor([0.0, 1.0])
        modules = LanguageModules(
            language_layers=list(model.layers),
            final_norm=torch.nn.Identity(),
            lm_head=model.lm_head,
            hidden_size=2,
            num_hidden_layers=2,
        )

        class FakeInference:
            def __init__(self, processor: object, fake_model: torch.nn.Module):
                self.processor = processor
                self.model = fake_model

            @staticmethod
            def _get_inputs_device() -> torch.device:
                return torch.device("cpu")

        decoder = AnswerPatchscopeDecoder(
            inference=FakeInference(self.processor, model),
            modules=modules,
        )
        first_target = decoder.target_for(candidates)
        second_target = decoder.target_for(candidates)
        shuffled_target = decoder.target_for(
            candidates,
            list(reversed(candidates)),
        )
        self.assertIs(first_target, second_target)
        self.assertIsNot(first_target, shuffled_target)
        self.assertEqual(
            shuffled_target.display_candidates,
            ("blue", "red"),
        )
        self.assertEqual(decoder.call_counts["target_prepare"], 2)
        self.assertEqual(
            first_target.target_position,
            int(first_target.inputs.input_ids.shape[1]) - 1,
        )
        self.assertNotIn("sample question", first_target.rendered_prompt)
        self.assertNotIn("sample clue", first_target.rendered_prompt)

        source_hidden = torch.tensor([0.25, 0.75])
        result, logits = decoder.run_patched_answer_readout(
            layer_index=1,
            source_hidden=source_hidden,
            candidates=candidates,
            collision_report=collision_report,
        )
        self.assertEqual(result["predicted_answer"], "blue")
        self.assertAlmostEqual(
            sum(result["answer_class_probabilities"].values()),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            result["predicted_answer_probability"],
            result["answer_class_probabilities"]["blue"],
            places=6,
        )
        reference = project_hidden_to_vocab(
            source_hidden,
            modules.final_norm,
            modules.lm_head,
        )
        check = validate_restricted_reconstruction(
            logits,
            reference,
            labels=candidates,
            class_token_ids=token_ids,
            midpoints=None,
            tolerance=1e-3,
        )
        self.assertTrue(check["passed"])
        self.assertEqual(decoder.call_counts, {"target_prepare": 2, "patched": 1})


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
    def test_lmhead_source_readout_keeps_direct_projection_math(self) -> None:
        hidden = torch.tensor([0.25, -0.5, 1.0])
        final_norm = torch.nn.LayerNorm(3)
        lm_head = torch.nn.Linear(3, 9, bias=False)
        with torch.no_grad():
            lm_head.weight.copy_(torch.arange(27).reshape(9, 3) / 10)
        token_ids = {label: [index] for index, label in enumerate(SOURCE_ATTRIBUTION_CLASSES)}
        direct_logits = project_hidden_to_vocab(hidden, final_norm, lm_head)
        expected = source_distribution(
            gather_source_class_logits(direct_logits, token_ids),
            class_token_ids=token_ids,
            raw_output="",
            parsed_label=None,
        ).to_dict()
        actual = source_layer_readout(
            3,
            hidden,
            final_norm,
            lm_head,
            token_ids,
        )
        self.assertEqual(actual["analysis_mode"], "LMhead")
        self.assertEqual(actual["layer_index"], 3)
        self.assertEqual(actual["class_logits"], expected["class_logits"])
        self.assertEqual(
            actual["class_probabilities"],
            expected["class_probabilities"],
        )
        self.assertEqual(
            actual["soft_image_score"],
            expected["soft_image_score"],
        )

    def test_patchscope_prompts_are_content_free_and_exact(self) -> None:
        self.assertEqual(
            IDENTITY_PATCHSCOPE_ASSISTANT_PREFILL,
            "0 -> 0\n1 -> 1\n2 -> 2\n3 -> 3\n4 -> 4\n"
            "5 -> 5\n6 -> 6\n7 -> 7\n8 -> 8\n?",
        )
        self.assertEqual(IDENTITY_PATCHSCOPE_ASSISTANT_PREFILL.count("?"), 1)
        self.assertTrue(IDENTITY_PATCHSCOPE_ASSISTANT_PREFILL.endswith("?"))
        for sample_value in ("sample question", "sample clue", "orange", "purple"):
            self.assertNotIn(sample_value, IDENTITY_PATCHSCOPE_USER_PROMPT)
            self.assertNotIn(sample_value, SEMANTIC_PATCHSCOPE_USER_PROMPT)
            self.assertNotIn(
                sample_value,
                SEMANTIC_ANSWER_PATCHSCOPE_USER_PROMPT,
            )
        self.assertIn(
            "{answer_classes}",
            SEMANTIC_ANSWER_PATCHSCOPE_USER_PROMPT,
        )
        self.assertIn("{source_classes}", SEMANTIC_PATCHSCOPE_USER_PROMPT)

    def test_answer_candidate_orders_are_distinct_deterministic_permutations(
        self,
    ) -> None:
        candidates = ["red", "blue", "green", "yellow"]
        first = answer_candidate_orders(candidates)
        second = answer_candidate_orders(candidates)
        self.assertEqual(first, second)
        self.assertEqual(tuple(first), ANSWER_PATCHSCOPE_VARIANTS)
        self.assertEqual(first["original"], tuple(candidates))
        self.assertEqual(len(set(first.values())), 4)
        for order in first.values():
            self.assertEqual(set(order), set(candidates))

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
    def test_analysis_mode_cli_defaults_orders_and_rejects_duplicates(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).analysis_mode, ["LMhead"])
        self.assertFalse(parser.parse_args([]).answer_val)
        self.assertEqual(parser.parse_args([]).save_hidden_state, [None])
        self.assertTrue(parser.parse_args(["--answer_val"]).answer_val)
        self.assertEqual(
            parser.parse_args(["--save_hidden_state", "23"]).save_hidden_state,
            [23],
        )
        self.assertEqual(
            parser.parse_args(
                ["--save_hidden_state", "26", "20", "23"]
            ).save_hidden_state,
            [26, 20, 23],
        )
        self.assertEqual(
            normalize_save_hidden_states([26, 20, 23]),
            (20, 23, 26),
        )
        self.assertEqual(
            normalize_save_hidden_states(
                parser.parse_args(["--save_hidden_state", "none"]).save_hidden_state
            ),
            (),
        )
        parsed = parser.parse_args(
            ["--analysis_mode", "Semantic", "LMhead"]
        )
        self.assertEqual(
            normalize_analysis_modes(parsed.analysis_mode),
            ("LMhead", "Semantic"),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_analysis_modes(["LMhead", "LMhead"])
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "non-negative integer",
        ):
            parse_save_hidden_state("-1")
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "non-negative integer",
        ):
            parse_save_hidden_state("last")
        with self.assertRaisesRegex(ValueError, "skip-layer-readout"):
            validate_save_hidden_state(
                (23,),
                skip_layer_readout=True,
            )
        with self.assertRaisesRegex(ValueError, r"outside \[0, 27\]"):
            validate_save_hidden_state(
                (23, 28),
                skip_layer_readout=False,
                num_hidden_layers=28,
            )
        validate_save_hidden_state(
            (23, 27),
            skip_layer_readout=False,
            num_hidden_layers=28,
        )
        with self.assertRaisesRegex(ValueError, "cannot combine"):
            normalize_save_hidden_states([None, 23])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_save_hidden_states([23, 23])

    def test_legacy_config_defaults_to_lmhead(self) -> None:
        normalized, missing = saved_configuration_for_comparison(
            {"format_version": 1}
        )
        self.assertTrue(missing)
        self.assertEqual(normalized["analysis_modes"], ["LMhead"])
        self.assertEqual(normalized["save_hidden_state"], "none")
        explicit, missing = saved_configuration_for_comparison(
            {
                "analysis_modes": ["Identity", "Semantic"],
                "save_hidden_state": 23,
            }
        )
        self.assertFalse(missing)
        self.assertEqual(explicit["analysis_modes"], ["Identity", "Semantic"])
        self.assertEqual(explicit["save_hidden_state"], 23)

    def test_target_layer_hidden_state_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            results_path = output_dir / "results.jsonl"
            store = TargetLayerHiddenStateStore(
                output_dir,
                layer_index=23,
                shard_size=2,
            )
            expected: dict[str, torch.Tensor] = {}
            records: list[dict] = []
            for index in range(2):
                case_id = f"case-{index}"
                tensor = torch.arange(
                    index * 8,
                    index * 8 + 8,
                    dtype=torch.float32,
                ).reshape(2, 4)
                expected[case_id] = tensor.half()
                result = {"case_id": case_id, "status": "completed"}
                records.append(result)
                should_flush = store.add(
                    case_id=case_id,
                    hidden_states=tensor,
                    positions={"ac": 11 + index, "panl": 21 + index},
                    stages={"ac": "answer", "panl": "confidence"},
                    result=result,
                )
            self.assertTrue(should_flush)
            self.assertEqual(store.flush(results_path), ["case-0", "case-1"])

            stored_records = load_jsonl(results_path)
            self.assertEqual(len(stored_records), 2)
            for record in stored_records:
                case_id = record["case_id"]
                restored, reference = store.read_case(case_id)
                self.assertEqual(restored.dtype, torch.float16)
                self.assertEqual(tuple(restored.shape), (2, 4))
                self.assertTrue(torch.equal(restored, expected[case_id]))
                self.assertEqual(reference["layer_index"], 23)
                self.assertEqual(reference["position_names"], ["ac", "panl"])
                self.assertEqual(
                    record["hidden_state_reference"]["stages"],
                    {"ac": "answer", "panl": "confidence"},
                )
                self.assertEqual(
                    record["hidden_state_reference"]["positions"],
                    {
                        "ac": 11 + int(case_id[-1]),
                        "panl": 21 + int(case_id[-1]),
                    },
                )

            payload_path = (
                output_dir
                / stored_records[0]["hidden_state_reference"]["shard_path"]
            )
            try:
                payload = torch.load(
                    payload_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                payload = torch.load(payload_path, map_location="cpu")
            self.assertEqual(
                tuple(payload["hidden_states"].shape),
                (2, 2, 4),
            )
            self.assertEqual(payload["dtype"], "float16")

    def test_multiple_target_layers_use_layer_token_hidden_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            results_path = output_dir / "results.jsonl"
            store = TargetLayerHiddenStateStore(
                output_dir,
                layer_index=[20, 23, 26],
                shard_size=1,
            )
            tensor = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
            result = {"case_id": "multi", "status": "completed"}
            self.assertTrue(
                store.add(
                    case_id="multi",
                    hidden_states=tensor,
                    positions={"ac": 11, "panl": 21},
                    stages={"ac": "answer", "panl": "confidence"},
                    result=result,
                )
            )
            store.flush(results_path)
            restored, reference = store.read_case("multi")
            self.assertEqual(tuple(restored.shape), (3, 2, 4))
            self.assertTrue(torch.equal(restored, tensor.half()))
            self.assertEqual(reference["layer_indices"], [20, 23, 26])
            self.assertNotIn("layer_index", reference)
            self.assertIn(
                "hidden_states/target_layers_20_23_26/",
                reference["shard_path"],
            )

    def test_target_layer_capture_selects_ac_and_panl_only_once(self) -> None:
        hidden_by_name = {
            "ac": {
                22: torch.tensor([1.0, 2.0]),
                23: torch.tensor([3.0, 4.0]),
            },
            "panl": {
                22: torch.tensor([5.0, 6.0]),
                23: torch.tensor([7.0, 8.0]),
            },
            "cc": {23: torch.tensor([9.0, 10.0])},
        }
        captured: dict[str, torch.Tensor] = {}
        capture_target_layer_hidden_states(hidden_by_name, 23, captured)
        self.assertEqual(set(captured), {"ac", "panl"})
        self.assertTrue(
            torch.equal(captured["ac"], torch.tensor([3.0, 4.0]).half())
        )
        self.assertTrue(
            torch.equal(captured["panl"], torch.tensor([7.0, 8.0]).half())
        )
        self.assertEqual(captured["ac"].device.type, "cpu")
        self.assertEqual(captured["panl"].dtype, torch.float16)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            capture_target_layer_hidden_states(hidden_by_name, 23, captured)

    def test_resume_rejects_legacy_format_and_accepts_current_format(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "legacy layer-readout format",
        ):
            validate_resume_format({"format_version": FORMAT_VERSION - 1})
        validate_resume_format({"format_version": FORMAT_VERSION})

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
                    "answer_patchscope_layers": [
                        {
                            "layer_index": 0,
                            "predicted_answer": "purple",
                            "predicted_answer_probability": 0.34,
                        }
                    ],
                    "cc_layers": [],
                    "sac_layers_by_mode": {
                        "Semantic": [
                            {"layer_index": 0, "soft_image_score": 0.312}
                        ]
                    },
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
            ["orange", 0.21, "purple", 0.34, None, 0.312],
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
            self.assertEqual(loaded[0]["layers"]["0"][4], None)
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
                            "0": [
                                "orange",
                                0.21,
                                "purple",
                                0.34,
                                None,
                                0.312,
                            ]
                        },
                    }
                ],
            )

    def test_six_column_semantic_value_is_null_when_not_collected(self) -> None:
        analysis, statistics = build_source_sink_minimal(
            [
                {
                    "case_id": "without-semantic",
                    "direct_readout": {
                        "ac_layers": [
                            {
                                "layer_index": 0,
                                "predicted_answer": "red",
                                "predicted_answer_probability": 0.6,
                            }
                        ],
                        "answer_patchscope_layers": [
                            {
                                "layer_index": 0,
                                "predicted_answer": "blue",
                                "predicted_answer_probability": 0.4,
                            }
                        ],
                        "cc_layers": [
                            {"layer_index": 0, "soft_confidence": 0.5}
                        ],
                        "sac_layers_by_mode": {
                            "LMhead": [
                                {"layer_index": 0, "soft_image_score": 0.2}
                            ],
                            "Identity": [
                                {"layer_index": 0, "soft_image_score": 0.3}
                            ],
                        },
                    },
                }
            ]
        )
        self.assertEqual(
            analysis[0]["layers"]["0"],
            ["red", 0.6, "blue", 0.4, 0.5, None],
        )
        self.assertEqual(statistics["invalid_layer_values"], 1)

    def test_answer_validation_summary_reports_pairwise_mae_and_pearson(
        self,
    ) -> None:
        def layer(probabilities: dict[str, float]) -> dict:
            answer = max(probabilities, key=probabilities.get)
            return {
                "layer_index": 0,
                "predicted_answer": answer,
                "predicted_answer_probability": probabilities[answer],
                "answer_class_probabilities": probabilities,
            }

        records = [
            {
                "direct_readout": {
                    "answer_patchscope_layers_by_variant": {
                        "original": [layer({"red": 0.8, "blue": 0.2})],
                        "shuffle_1": [layer({"red": 0.8, "blue": 0.2})],
                        "shuffle_2": [layer({"red": 0.2, "blue": 0.8})],
                        "shuffle_3": [layer({"red": 0.6, "blue": 0.4})],
                    }
                },
                "validation": {
                    "answer_patchscope_by_variant": {
                        variant: {"passed": True}
                        for variant in ANSWER_PATCHSCOPE_VARIANTS
                    }
                },
            }
        ]
        summary = build_answer_validation_summary(records)
        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["records_with_all_variants"], 1)
        self.assertEqual(
            summary["overall"]["original__vs__shuffle_1"],
            {"value_count": 2, "mae": 0.0, "pearson_r": 1.0},
        )
        self.assertEqual(
            summary["layers"]["0"]["original__vs__shuffle_2"],
            {"value_count": 2, "mae": 0.6, "pearson_r": -1.0},
        )
        self.assertEqual(
            summary["overall"]["original__vs__shuffle_3"]["mae"],
            0.2,
        )

    def test_answer_validation_appends_six_columns_to_compact_rows(self) -> None:
        def answer(name: str, probability: float) -> dict:
            return {
                "layer_index": 0,
                "predicted_answer": name,
                "predicted_answer_probability": probability,
                "answer_class_probabilities": {
                    "red": probability if name == "red" else 1.0 - probability,
                    "blue": probability if name == "blue" else 1.0 - probability,
                },
            }

        analysis, _statistics = build_source_sink_minimal(
            [
                {
                    "direct_readout": {
                        "ac_layers": [
                            {
                                "layer_index": 0,
                                "predicted_answer": "red",
                                "predicted_answer_probability": 0.7,
                            }
                        ],
                        "answer_patchscope_layers": [answer("blue", 0.6)],
                        "answer_patchscope_layers_by_variant": {
                            "original": [answer("blue", 0.6)],
                            "shuffle_1": [answer("red", 0.55)],
                            "shuffle_2": [answer("blue", 0.65)],
                            "shuffle_3": [answer("red", 0.58)],
                        },
                        "cc_layers": [
                            {"layer_index": 0, "soft_confidence": 0.8}
                        ],
                        "sac_layers_by_mode": {
                            "Semantic": [
                                {"layer_index": 0, "soft_image_score": 0.4}
                            ]
                        },
                    }
                }
            ]
        )
        row = analysis[0]["layers"]["0"]
        self.assertEqual(
            row,
            [
                "red",
                0.7,
                "blue",
                0.6,
                0.8,
                0.4,
                "red",
                0.55,
                "blue",
                0.65,
                "red",
                0.58,
            ],
        )
        self.assertEqual(
            len(row),
            len(COMPACT_LAYER_COLUMNS_WITH_ANSWER_VAL),
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

    def test_new_readouts_feed_six_columns_and_legacy_sac_stays_in_summary(
        self,
    ) -> None:
        records = [
            {
                "case_id": "new",
                "status": "completed",
                "direct_readout": {
                    "ac_layers": [
                        {
                            "layer_index": 0,
                            "predicted_answer": "blue",
                            "predicted_answer_probability": 0.55,
                        }
                    ],
                    "answer_patchscope_layers": [
                        {
                            "layer_index": 0,
                            "predicted_answer": "green",
                            "predicted_answer_probability": 0.45,
                        }
                    ],
                    "cc_layers": [{"layer_index": 0, "soft_confidence": 0.4}],
                    "sac_layers": [{"layer_index": 0, "soft_image_score": 0.5}],
                    "sac_layers_by_mode": {
                        "LMhead": [{"layer_index": 0, "soft_image_score": 0.5}],
                        "Identity": [{"layer_index": 0, "soft_image_score": 0.6}],
                        "Semantic": [{"layer_index": 0, "soft_image_score": 0.7}],
                    },
                },
                "validation": {
                    "ac_last_layer": {"passed": True},
                    "answer_patchscope_last_layer": {"passed": True},
                    "cc_last_layer": {"passed": True},
                    "sac_last_layer": {"passed": True},
                    "sac_by_mode": {
                        "LMhead": {"passed": True},
                        "Identity": {"passed": False},
                        "Semantic": None,
                    },
                },
            },
            {
                "case_id": "legacy",
                "status": "completed",
                "direct_readout": {
                    "ac_layers": [],
                    "cc_layers": [],
                    "sac_layers": [{"layer_index": 0, "soft_image_score": 0.2}],
                },
                "validation": {
                    "ac_last_layer": None,
                    "cc_last_layer": None,
                    "sac_last_layer": {"passed": True},
                },
            },
        ]
        analysis, statistics = build_source_sink_minimal(records)
        self.assertEqual(
            analysis[0]["layers"]["0"],
            ["blue", 0.55, "green", 0.45, 0.4, 0.7],
        )
        self.assertEqual(len(analysis[0]["layers"]["0"]), len(COMPACT_LAYER_COLUMNS))
        self.assertEqual(analysis[1]["layers"], {})
        summary = build_source_sink_summary(records, analysis, statistics)
        self.assertEqual(
            summary["sac_readout_coverage_by_mode"],
            {"LMhead": 2, "Identity": 1, "Semantic": 1},
        )
        self.assertEqual(
            summary["sac_validation_by_mode"]["LMhead"],
            {"passed": 2, "failed": 0, "not_run": 0},
        )
        self.assertEqual(
            summary["sac_validation_by_mode"]["Identity"],
            {"passed": 0, "failed": 1, "not_run": 1},
        )
        self.assertEqual(
            summary["sac_validation_by_mode"]["Semantic"],
            {"passed": 0, "failed": 0, "not_run": 2},
        )


class AddDecoderLayer(torch.nn.Module):
    def __init__(self, *, tuple_output: bool) -> None:
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, hidden: torch.Tensor) -> object:
        value = hidden + 1
        return (value, "preserved") if self.tuple_output else value


class FakePatchModel(torch.nn.Module):
    def __init__(
        self,
        layers: list[AddDecoderLayer],
        *,
        fail_after_layers: bool = False,
        vocab_size: int = 3,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.lm_head = torch.nn.Linear(2, vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.weight[:3].copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            )
        self.fail_after_layers = fail_after_layers
        self.last_hidden: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        logits_to_keep: torch.Tensor | int = 0,
        **_kwargs: object,
    ) -> object:
        hidden = torch.stack(
            [input_ids.float(), input_ids.float() * 2],
            dim=-1,
        )
        for layer in self.layers:
            output = layer(hidden)
            if isinstance(output, tuple):
                self.assert_tuple_metadata(output)
                hidden = output[0]
            else:
                hidden = output
        self.last_hidden = hidden.detach().clone()
        if self.fail_after_layers:
            raise RuntimeError("synthetic forward failure")
        indices = (
            logits_to_keep
            if isinstance(logits_to_keep, torch.Tensor)
            else slice(-logits_to_keep, None)
        )
        return SimpleNamespace(logits=self.lm_head(hidden[:, indices, :]))

    @staticmethod
    def assert_tuple_metadata(output: tuple[object, ...]) -> None:
        if output[1] != "preserved":
            raise AssertionError("Tuple metadata was not preserved")


class PatchedForwardTests(unittest.TestCase):
    def _fixture(
        self,
        *,
        tuple_output: bool,
        fail_after_layers: bool = False,
    ) -> tuple[FakePatchModel, LanguageModules]:
        layers = [
            AddDecoderLayer(tuple_output=tuple_output),
            AddDecoderLayer(tuple_output=tuple_output),
        ]
        model = FakePatchModel(layers, fail_after_layers=fail_after_layers)
        modules = LanguageModules(
            language_layers=list(model.layers),
            final_norm=torch.nn.Identity(),
            lm_head=model.lm_head,
            hidden_size=2,
            num_hidden_layers=2,
        )
        return model, modules

    def test_patch_supports_tensor_and_tuple_and_only_changes_batch_zero(self) -> None:
        inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}
        for tuple_output in (False, True):
            with self.subTest(tuple_output=tuple_output):
                model, modules = self._fixture(tuple_output=tuple_output)
                logits = run_patched_logits_forward(
                    model,
                    inputs,
                    modules,
                    layer_index=0,
                    target_position=1,
                    source_hidden=torch.tensor([10.0, 20.0], dtype=torch.float64),
                )
                torch.testing.assert_close(
                    logits,
                    torch.tensor([11.0, 21.0, 32.0]),
                )
                assert model.last_hidden is not None
                torch.testing.assert_close(
                    model.last_hidden[1, 1],
                    torch.tensor([7.0, 12.0]),
                )
                self.assertEqual(len(model.layers[0]._forward_hooks), 0)

    def test_final_layer_patch_reconstructs_source_logits(self) -> None:
        model, modules = self._fixture(tuple_output=True)
        logits = run_patched_logits_forward(
            model,
            {"input_ids": torch.tensor([[1, 2, 3]])},
            modules,
            layer_index=1,
            target_position=2,
            source_hidden=torch.tensor([4.0, 9.0]),
        )
        torch.testing.assert_close(logits, torch.tensor([4.0, 9.0, 13.0]))

    def test_hook_is_removed_when_forward_raises(self) -> None:
        model, modules = self._fixture(
            tuple_output=True,
            fail_after_layers=True,
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            run_patched_logits_forward(
                model,
                {"input_ids": torch.tensor([[1, 2, 3]])},
                modules,
                layer_index=0,
                target_position=1,
                source_hidden=torch.tensor([4.0, 9.0]),
            )
        self.assertEqual(len(model.layers[0]._forward_hooks), 0)

    def test_cached_baseline_is_returned_by_copy(self) -> None:
        decoder = object.__new__(SourcePatchscopeDecoder)
        decoder.targets = {
            "Identity": PreparedSourceTarget(
                name="identity",
                inputs=None,
                target_position=7,
                rendered_prompt="target",
                baseline={"soft_image_score": 0.5},
            )
        }
        first = decoder.run_target_baseline("Identity")
        first["soft_image_score"] = 0.9
        second = decoder.run_target_baseline("Identity")
        self.assertEqual(second["soft_image_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
