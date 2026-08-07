#!/usr/bin/env python3
"""CPU unit tests and one-case GPU smoke test for the main experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_metacognition.analyze_main_results import build_minimal_analysis  # noqa: E402
from layer_metacognition.confidence_schema import (  # noqa: E402
    CONFIDENCE_CLASSES,
    CONFIDENCE_CLASS_TEXT,
)
from layer_metacognition.conversation_builder import (  # noqa: E402
    build_stage1_messages,
    build_stage2_messages,
    render_continued_assistant,
)
from layer_metacognition.dataset_loader import (  # noqa: E402
    load_experiment_cases,
    parse_choice_colors,
    parse_stage1_answer,
)
from layer_metacognition.direct_readout import build_first_token_collision_report  # noqa: E402
from layer_metacognition.hidden_state_store import (  # noqa: E402
    HiddenStateStore,
    atomic_write_json,
    load_jsonl,
)
from layer_metacognition.model_adapter import (  # noqa: E402
    load_qwen_inference,
    resolve_language_modules,
)
from layer_metacognition.prompts import (  # noqa: E402
    ASSISTANT_ANSWER_PREFILL,
    ASSISTANT_CONFIDENCE_PREFILL,
    STAGE1_MULTIMODAL_ANSWER_PROMPT,
    STAGE2_CONFIDENCE_PROMPT,
)
from layer_metacognition.run_main_experiment import process_case  # noqa: E402
from layer_metacognition.token_positions import (  # noqa: E402
    locate_cc,
    locate_panl,
    locate_suffix_colon,
    locate_text_clue,
)
from layer_metacognition.token_spans import build_rendered_alignment, unique_text_span  # noqa: E402


CPU_CONTEXT: dict[str, Any] = {}


class CpuSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoProcessor

        args = CPU_CONTEXT
        cls.processor = AutoProcessor.from_pretrained(args["model_path"], local_files_only=True)
        cls.tokenizer = getattr(cls.processor, "tokenizer", cls.processor)
        cls.cases, _ = load_experiment_cases(
            args["dataset"], image_dir=args["image_dir"], max_items=args["max_items"]
        )
        cls.case = cls.cases[0]
        cls.answer = cls.case.dataset_answer or parse_choice_colors(cls.case.question)[0]
        cls.stage1_generation_messages, cls.stage1_prefill = build_stage1_messages(
            cls.case.question, cls.case.text_clue, cls.case.image_path
        )
        cls.stage1_generation = render_continued_assistant(
            cls.processor, cls.stage1_generation_messages, cls.stage1_prefill
        )
        cls.stage1_messages, cls.stage1_assistant = build_stage1_messages(
            cls.case.question, cls.case.text_clue, cls.case.image_path, cls.answer
        )
        cls.stage1 = render_continued_assistant(
            cls.processor, cls.stage1_messages, cls.stage1_assistant
        )
        cls.stage2_messages, cls.stage2_prefill = build_stage2_messages(
            cls.case.question,
            cls.case.text_clue,
            cls.answer,
            cls.case.image_path,
            CONFIDENCE_CLASS_TEXT,
        )
        cls.stage2 = render_continued_assistant(
            cls.processor, cls.stage2_messages, cls.stage2_prefill
        )

    def alignment(self, rendered: str) -> Any:
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        return build_rendered_alignment(
            self.tokenizer, rendered, encoded.input_ids, encoded.attention_mask
        )

    def test_01_prompt_format(self) -> None:
        rendered = STAGE1_MULTIMODAL_ANSWER_PROMPT.format(question="Q", text_clue="C")
        self.assertIn("Question:\nQ\n\nText clue:\nC", rendered)
        self.assertTrue(rendered.endswith("Do not include any additional text."))
        confidence = STAGE2_CONFIDENCE_PROMPT.format(question="Q", text_clue="C", answer="red", classes="X")
        self.assertIn("given the question, text clue, and image.", confidence)

    def test_02_assistant_prefill_rendering(self) -> None:
        self.assertTrue(self.stage1_generation.endswith(ASSISTANT_ANSWER_PREFILL))
        self.assertTrue(self.stage2.endswith(ASSISTANT_CONFIDENCE_PREFILL))
        self.assertNotIn("<|im_end|>", self.stage2[-len(ASSISTANT_CONFIDENCE_PREFILL) :])

    def test_03_stage_sequences_are_independent(self) -> None:
        self.assertNotEqual(self.stage1, self.stage2)
        self.assertIn(f"**Answer**: {self.answer}\n\nClassify", self.stage2)

    def test_04_choose_from_and_answer_parser(self) -> None:
        choices = parse_choice_colors(self.case.question)
        self.assertIn(self.answer, choices)
        self.assertEqual(parse_stage1_answer(f" **Answer**: {self.answer}", choices), self.answer)

    def test_05_first_token_variants(self) -> None:
        report = build_first_token_collision_report(self.tokenizer, ["red", "blue"])
        for details in report["labels"].values():
            self.assertTrue(details["raw_token_ids"])
            self.assertTrue(details["space_token_ids"])
            self.assertTrue(details["first_token_variants"])

    def test_06_confidence_collision_report(self) -> None:
        self.assertEqual(
            CONFIDENCE_CLASSES,
            [
                "No chance",
                "Really unlikely",
                "Chances are slight",
                "Unlikely",
                "Less than even",
                "Better than even",
                "Likely",
                "Very good chance",
                "Highly likely",
                "Almost certain",
            ],
        )
        report = build_first_token_collision_report(self.tokenizer, CONFIDENCE_CLASSES)
        self.assertEqual(report["collisions"], [])
        self.assertEqual(set(report["labels"]), set(CONFIDENCE_CLASSES))

    def test_07_ac_cc_positions(self) -> None:
        ac_alignment = self.alignment(self.stage1)
        ac = locate_suffix_colon(self.tokenizer, ac_alignment, self.stage1_assistant)
        self.assertIn(":", ac["token_text"])
        cc_alignment = self.alignment(self.stage2)
        cc = locate_cc(self.tokenizer, cc_alignment, self.stage2_prefill)
        self.assertEqual(cc["position"], len(cc_alignment.processed_ids) - 1)

    def test_08_panl_unique_span(self) -> None:
        alignment = self.alignment(self.stage2)
        panl = locate_panl(self.tokenizer, alignment, self.answer)
        self.assertEqual(panl["decoded_span"], "\n\n")
        with self.assertRaises(ValueError):
            unique_text_span("x x", "x")

    def test_09_text_clue_span(self) -> None:
        alignment = self.alignment(self.stage2)
        clue = locate_text_clue(self.tokenizer, alignment, self.case.text_clue, "**Answer**:")
        self.assertEqual(" ".join(clue["token_text"].split()), " ".join(self.case.text_clue.split()))

    def test_10_jsonl_resume_repairs_only_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text('{"case_id":"a"}\n{"case_id":', encoding="utf-8")
            records = load_jsonl(path, repair_trailing=True)
            self.assertEqual(records, [{"case_id": "a"}])
            self.assertEqual(load_jsonl(path), records)

    def test_11_hidden_state_shard_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = HiddenStateStore(directory, shard_size=2)
            result = {
                "case_id": "case-a",
                "positions": {"panl": {"position": 7}},
                "model_structure": {"num_hidden_layers": 3},
            }
            tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            self.assertFalse(store.add("case-a", tensor, [0, 1, 2], result))
            store.flush(Path(directory) / "results.jsonl")
            restored, layers = store.read_case("case-a")
            self.assertEqual(layers, [0, 1, 2])
            self.assertTrue(torch.equal(restored, tensor.half()))
            index = json.loads((Path(directory) / "hidden_states" / "index.json").read_text())
            self.assertEqual(index["cases"]["case-a"]["position"], 7)
            self.assertEqual(index["cases"]["case-a"]["num_layers"], 3)

    def test_12_minimal_analysis_shape(self) -> None:
        records = [{
            "case_id": "x",
            "direct_readout": {
                "ac_layers": [{
                    "layer_index": 0,
                    "predicted_answer": "yellow",
                    "predicted_answer_probability": 0.31,
                    "answer_entropy": 1.84,
                }],
                "cc_layers": [{"layer_index": 0, "soft_confidence": 0.42}],
            },
        }]
        output, skipped = build_minimal_analysis(records)
        self.assertEqual(output[0]["layers"]["0"], ["yellow", 0.31, 1.84, 0.42])
        self.assertEqual(skipped, 0)
        self.assertTrue(all(math.isfinite(value) for value in output[0]["layers"]["0"][1:]))


def run_cpu_tests(args: argparse.Namespace) -> int:
    CPU_CONTEXT.update(vars(args))
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CpuSmokeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "mode": "cpu_only",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
        "gpu_smoke_test": "not_run",
    }
    output = Path(args.output_dir) / "cpu_smoke_report.json"
    atomic_write_json(output, report)
    return 0 if result.wasSuccessful() else 1


def run_gpu_smoke(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is unavailable; use --cpu-only for CPU tests.", file=sys.stderr)
        return 1
    cases, dataset_metadata = load_experiment_cases(
        args.dataset, image_dir=args.image_dir, max_items=args.max_items
    )
    case = cases[0]
    inference = load_qwen_inference(args.model_path)
    modules = resolve_language_modules(inference.model)
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    confidence_report = build_first_token_collision_report(tokenizer, CONFIDENCE_CLASSES)
    if confidence_report["collisions"]:
        raise RuntimeError(f"Confidence token collisions: {confidence_report['collisions']}")
    choices = parse_choice_colors(case.question)
    answer_report = build_first_token_collision_report(tokenizer, choices)
    if answer_report["collisions"]:
        raise RuntimeError(f"Answer token collisions: {answer_report['collisions']}")
    selected_layers = list(range(modules.num_hidden_layers))
    result, panl = process_case(
        case,
        inference,
        modules,
        selected_layers,
        confidence_report,
        answer_report,
    )
    output_dir = Path(args.output_dir).resolve()
    store = HiddenStateStore(output_dir, shard_size=1)
    store.add(case.case_id, panl, selected_layers, result)
    store.flush(output_dir / "results.jsonl")
    restored, restored_layers = store.read_case(case.case_id)
    if restored_layers != selected_layers or not torch.equal(restored, panl.half()):
        raise RuntimeError("GPU smoke PANL shard round-trip failed")
    report = {
        "mode": "gpu",
        "successful": True,
        "model_load_count": 1,
        "case_id": case.case_id,
        "dataset": dataset_metadata,
        "positions": result["positions"],
        "model_structure": result["model_structure"],
        "validation": result["validation"],
        "panl_readout_performed": False,
        "panl_pre_final_norm": True,
    }
    atomic_write_json(output_dir / "gpu_smoke_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--dataset", default="datasets/datasets.json")
    parser.add_argument("--image-dir")
    parser.add_argument("--max-items", type=int, default=1)
    parser.add_argument("--output-dir", default="layer_metacognition/output/smoke")
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_cpu_tests(args) if args.cpu_only else run_gpu_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
