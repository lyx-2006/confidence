#!/usr/bin/env python3
"""Run the Qwen2.5-VL layer-wise answer/confidence main experiment."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_metacognition.confidence_schema import (  # noqa: E402
    CONFIDENCE_CLASSES,
    CONFIDENCE_CLASS_TEXT,
)
from layer_metacognition.conversation_builder import (  # noqa: E402
    build_stage1_messages,
    build_stage2_messages,
    prepare_multimodal_inputs,
    render_continued_assistant,
)
from layer_metacognition.dataset_loader import (  # noqa: E402
    ExperimentCase,
    load_experiment_cases,
    parse_choice_colors,
    parse_stage1_answer,
)
from layer_metacognition.direct_readout import (  # noqa: E402
    answer_layer_readout,
    build_first_token_collision_report,
    confidence_layer_readout,
    project_hidden_to_vocab,
    reconstruction_metrics,
)
from layer_metacognition.hidden_state_store import (  # noqa: E402
    HiddenStateStore,
    append_jsonl,
    atomic_write_json,
    load_jsonl,
)
from layer_metacognition.metrics import panl_layer_statistics  # noqa: E402
from layer_metacognition.model_adapter import (  # noqa: E402
    LanguageModules,
    generate_from_prefill,
    load_qwen_inference,
    model_input_device,
    parse_layer_selection,
    resolve_language_modules,
    run_hooked_forward,
)
from layer_metacognition.token_positions import (  # noqa: E402
    locate_cc,
    locate_image_span,
    locate_panl,
    locate_suffix_colon,
    locate_text_clue,
)
from layer_metacognition.token_spans import build_rendered_alignment  # noqa: E402


class CaseProcessingError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _inputs_and_alignment(
    inference: Any,
    messages: list[dict[str, Any]],
    rendered: str,
) -> tuple[Any, Any]:
    processor = inference.processor
    inputs = prepare_multimodal_inputs(
        processor,
        messages,
        rendered,
        device=model_input_device(inference),
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    alignment = build_rendered_alignment(
        tokenizer,
        rendered,
        inputs.input_ids,
        inputs.attention_mask,
    )
    return inputs, alignment


def _layer_readouts(
    hidden_by_layer: dict[int, torch.Tensor],
    selected_layers: list[int],
    function: Any,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return [
        function(layer_index=layer, hidden=hidden_by_layer[layer], **kwargs)
        for layer in selected_layers
    ]


def process_case(
    case: ExperimentCase,
    inference: Any,
    modules: LanguageModules,
    selected_layers: list[int],
    confidence_collision_report: dict[str, Any],
    answer_collision_report: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor]:
    processor = inference.processor
    tokenizer = getattr(processor, "tokenizer", processor)
    candidates = parse_choice_colors(case.question)
    final_layer = modules.num_hidden_layers - 1
    stage = "stage1_generation"
    try:
        generation_messages, generation_prefill = build_stage1_messages(
            case.question, case.text_clue, case.image_path
        )
        generation_rendered = render_continued_assistant(
            processor, generation_messages, generation_prefill
        )
        generation_inputs = prepare_multimodal_inputs(
            processor,
            generation_messages,
            generation_rendered,
            device=model_input_device(inference),
        )
        raw_stage1_output = generate_from_prefill(inference, generation_inputs, max_new_tokens=8)
        stage1_answer = parse_stage1_answer(raw_stage1_output, candidates)
        if stage1_answer is None:
            raise ValueError(f"Could not parse a legal colour from Stage1 output {raw_stage1_output!r}")
        del generation_inputs

        stage = "stage1_teacher_forced"
        stage1_messages, stage1_assistant = build_stage1_messages(
            case.question, case.text_clue, case.image_path, answer=stage1_answer
        )
        stage1_rendered = render_continued_assistant(processor, stage1_messages, stage1_assistant)
        stage1_inputs, stage1_alignment = _inputs_and_alignment(
            inference, stage1_messages, stage1_rendered
        )
        ac = locate_suffix_colon(tokenizer, stage1_alignment, stage1_assistant)
        stage1_clue = locate_text_clue(
            tokenizer,
            stage1_alignment,
            case.text_clue,
            "Answer the question using",
        )
        stage1_image = locate_image_span(
            tokenizer,
            processor,
            stage1_alignment,
            stage1_inputs.image_grid_thw,
        )
        stage1_forward = run_hooked_forward(
            inference.model,
            stage1_inputs,
            modules,
            {"ac": ac["position"]},
        )
        ac_hidden = stage1_forward.hidden_by_name["ac"]
        ac_layers = _layer_readouts(
            ac_hidden,
            selected_layers,
            answer_layer_readout,
            final_norm=modules.final_norm,
            lm_head=modules.lm_head,
            candidates=candidates,
            collision_report=answer_collision_report,
            dataset_answer=case.dataset_answer,
            stage1_answer=stage1_answer,
            image_target=case.image_target,
            text_target=case.text_target,
        )
        ac_final = next((record for record in ac_layers if record["layer_index"] == final_layer), None)
        if ac_final is None:
            ac_final = answer_layer_readout(
                final_layer,
                ac_hidden[final_layer],
                modules.final_norm,
                modules.lm_head,
                candidates,
                answer_collision_report,
                case.dataset_answer,
                stage1_answer,
                case.image_target,
                case.text_target,
            )
        ac_reconstructed = project_hidden_to_vocab(
            ac_hidden[final_layer], modules.final_norm, modules.lm_head
        )
        ac_reconstruction = reconstruction_metrics(
            ac_reconstructed,
            stage1_forward.logits_by_position[ac["position"]],
        )
        if not ac_reconstruction["allclose"]:
            raise RuntimeError(f"AC FinalNorm+LM Head reconstruction failed: {ac_reconstruction}")
        del stage1_inputs, stage1_forward, ac_reconstructed

        stage = "stage2_confidence"
        stage2_messages, stage2_prefill = build_stage2_messages(
            case.question,
            case.text_clue,
            stage1_answer,
            case.image_path,
            CONFIDENCE_CLASS_TEXT,
        )
        stage2_rendered = render_continued_assistant(processor, stage2_messages, stage2_prefill)
        stage2_inputs, stage2_alignment = _inputs_and_alignment(
            inference, stage2_messages, stage2_rendered
        )
        panl = locate_panl(tokenizer, stage2_alignment, stage1_answer)
        cc = locate_cc(tokenizer, stage2_alignment, stage2_prefill)
        stage2_clue = locate_text_clue(
            tokenizer, stage2_alignment, case.text_clue, "**Answer**:"
        )
        stage2_image = locate_image_span(
            tokenizer,
            processor,
            stage2_alignment,
            stage2_inputs.image_grid_thw,
        )
        stage2_forward = run_hooked_forward(
            inference.model,
            stage2_inputs,
            modules,
            {"panl": panl["position"], "cc": cc["position"]},
            logits_positions=[cc["position"]],
        )
        panl_hidden = stage2_forward.hidden_by_name["panl"]
        cc_hidden = stage2_forward.hidden_by_name["cc"]
        cc_layers = _layer_readouts(
            cc_hidden,
            selected_layers,
            confidence_layer_readout,
            final_norm=modules.final_norm,
            lm_head=modules.lm_head,
            collision_report=confidence_collision_report,
        )
        cc_final = next((record for record in cc_layers if record["layer_index"] == final_layer), None)
        if cc_final is None:
            cc_final = confidence_layer_readout(
                final_layer,
                cc_hidden[final_layer],
                modules.final_norm,
                modules.lm_head,
                confidence_collision_report,
            )
        cc_reconstructed = project_hidden_to_vocab(
            cc_hidden[final_layer], modules.final_norm, modules.lm_head
        )
        cc_reconstruction = reconstruction_metrics(
            cc_reconstructed,
            stage2_forward.logits_by_position[cc["position"]],
        )
        if not cc_reconstruction["allclose"]:
            raise RuntimeError(f"CC FinalNorm+LM Head reconstruction failed: {cc_reconstruction}")
        panl_statistics = panl_layer_statistics(panl_hidden, selected_layers)
        stored_panl = torch.stack(
            [panl_hidden[layer].detach().to(device="cpu", dtype=torch.float16) for layer in selected_layers]
        )

        result = {
            **case.to_dict(),
            "stage1": {
                "raw_output": raw_stage1_output,
                "raw_stage1_output": raw_stage1_output,
                "answer": stage1_answer,
                "stage1_answer": stage1_answer,
                "answer_parsed": True,
                "stage1_answer_parsed": True,
            },
            "positions": {
                "ac": ac,
                "panl": panl,
                "cc": cc,
                "text_clue_stage1": stage1_clue,
                "text_clue_stage2": stage2_clue,
                "image_stage1": stage1_image,
                "image_stage2": stage2_image,
            },
            "direct_readout": {
                "ac_layers": ac_layers,
                "cc_layers": cc_layers,
                "panl_statistics": panl_statistics,
            },
            "final_outputs": {
                "stage1_final_predicted_answer": ac_final["predicted_answer"],
                "stage1_final_predicted_answer_probability": ac_final["predicted_answer_probability"],
                "stage1_final_answer_entropy": ac_final["answer_entropy"],
                "stage2_final_hard_confidence_label": cc_final["hard_confidence_label"],
                "stage2_final_soft_confidence": cc_final["soft_confidence"],
            },
            "validation": {
                "ac_final_norm_lm_head": ac_reconstruction,
                "cc_final_norm_lm_head": cc_reconstruction,
                "stage1_stage2_sequences_independent": stage1_rendered != stage2_rendered,
                "same_image_path": stage1_messages[0]["content"][0]["image"]
                == stage2_messages[0]["content"][0]["image"],
            },
            "model_structure": {
                "num_hidden_layers": modules.num_hidden_layers,
                "hidden_size": modules.hidden_size,
                "selected_layers": selected_layers,
                "layer_definition": "decoder_block_output_pre_final_norm",
            },
            "hidden_state_reference": None,
        }
        del stage2_inputs, stage2_forward, cc_reconstructed
        return result, stored_panl
    except Exception as exc:
        if isinstance(exc, CaseProcessingError):
            raise
        raise CaseProcessingError(stage, str(exc)) from exc


def _hidden_roundtrip(
    tensor: torch.Tensor,
    layer_indices: list[int],
    num_hidden_layers: int,
    panl_position: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="layer-metacognition-preflight-") as directory:
        preflight_store = HiddenStateStore(directory, shard_size=1)
        preflight_result = {
            "case_id": "__preflight__",
            "positions": {"panl": {"position": panl_position}},
            "model_structure": {"num_hidden_layers": num_hidden_layers},
        }
        preflight_store.add("__preflight__", tensor, layer_indices, preflight_result)
        preflight_store.flush(Path(directory) / "results.jsonl")
        restored, restored_layers = preflight_store.read_case("__preflight__")
        if restored_layers != layer_indices or not torch.equal(restored, tensor.cpu().half()):
            raise RuntimeError("PANL hidden-state shard round-trip failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--dataset", default="datasets/datasets.json")
    parser.add_argument("--image-dir")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--save-hidden-states", choices=["panl"], default="panl")
    parser.add_argument("--output-dir", default="layer_metacognition/output/main")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--case-id")
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    results_path = output_dir / "results.jsonl"
    failures_path = output_dir / "failures.jsonl"
    metadata_path = output_dir / "metadata.json"

    cases, dataset_metadata = load_experiment_cases(
        args.dataset,
        image_dir=args.image_dir,
        max_items=args.max_items,
        case_id=args.case_id,
    )
    existing_results = load_jsonl(results_path, repair_trailing=args.resume)
    existing_failures = load_jsonl(failures_path, repair_trailing=args.resume)
    if (existing_results or existing_failures) and not args.resume:
        print("[ERROR] Output already exists; pass --resume or use a new --output-dir", file=sys.stderr)
        return 2
    completed_ids = {record["case_id"] for record in existing_results}
    failed_ids = {record["case_id"] for record in existing_failures}
    pending_cases = [
        case
        for case in cases
        if case.case_id not in completed_ids
        and (args.retry_failures or case.case_id not in failed_ids)
    ]
    metadata: dict[str, Any] = {
        "status": "initializing",
        "started_at": utc_timestamp(),
        "model_path": str(Path(args.model_path).resolve()),
        "dataset": dataset_metadata,
        "configuration": vars(args),
        "resume": {
            "existing_results": len(existing_results),
            "existing_failures": len(existing_failures),
            "pending_cases": len(pending_cases),
        },
        "gpu_smoke_test": {"status": "not_run_by_implementation"},
    }
    atomic_write_json(metadata_path, metadata)
    store = HiddenStateStore(output_dir, shard_size=args.shard_size)
    if not pending_cases:
        store.rebuild_index(completed_ids)
        metadata.update({"status": "complete", "finished_at": utc_timestamp()})
        atomic_write_json(metadata_path, metadata)
        print("[INFO] No pending cases.")
        return 0

    try:
        inference = load_qwen_inference(str(Path(args.model_path).resolve()))
        modules = resolve_language_modules(inference.model)
        selected_layers = parse_layer_selection(args.layers, modules.num_hidden_layers)
        tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
        confidence_report = build_first_token_collision_report(tokenizer, CONFIDENCE_CLASSES)
        if confidence_report["collisions"]:
            raise RuntimeError(f"Confidence first-token collisions: {confidence_report['collisions']}")
        answer_reports: dict[tuple[str, ...], dict[str, Any]] = {}
        for case in pending_cases:
            candidates = tuple(parse_choice_colors(case.question))
            if candidates not in answer_reports:
                answer_reports[candidates] = build_first_token_collision_report(tokenizer, candidates)
        metadata.update(
            {
                "status": "running_preflight",
                "model": {
                    "dtype": inference.dtype_name,
                    "num_hidden_layers": modules.num_hidden_layers,
                    "hidden_size": modules.hidden_size,
                    "selected_layers": selected_layers,
                },
                "collision_reports": {
                    "confidence": confidence_report,
                    "answer_candidate_sets": {
                        "|".join(labels): report for labels, report in answer_reports.items()
                    },
                },
            }
        )
        atomic_write_json(metadata_path, metadata)
    except Exception as exc:
        metadata.update({"status": "startup_failed", "error": str(exc), "finished_at": utc_timestamp()})
        atomic_write_json(metadata_path, metadata)
        print(f"[ERROR] Startup failed: {exc}", file=sys.stderr)
        return 1

    interrupted = False
    try:
        for pending_index, case in enumerate(pending_cases):
            try:
                candidates = tuple(parse_choice_colors(case.question))
                answer_report = answer_reports[candidates]
                if answer_report["collisions"]:
                    raise CaseProcessingError(
                        "answer_token_collision",
                        f"Answer candidate first-token collisions: {answer_report['collisions']}",
                    )
                result, panl_hidden = process_case(
                    case,
                    inference,
                    modules,
                    selected_layers,
                    confidence_report,
                    answer_report,
                )
                if pending_index == 0:
                    _hidden_roundtrip(
                        panl_hidden,
                        selected_layers,
                        modules.num_hidden_layers,
                        result["positions"]["panl"]["position"],
                    )
                    metadata["gpu_preflight"] = {
                        "status": "passed",
                        "case_id": case.case_id,
                        "completed_at": utc_timestamp(),
                        "checks": result["validation"],
                    }
                    metadata["status"] = "running"
                    atomic_write_json(metadata_path, metadata)
                if store.add(case.case_id, panl_hidden, selected_layers, result):
                    committed = store.flush(results_path)
                    completed_ids.update(committed)
                    print(f"[INFO] Committed {len(committed)} cases through {committed[-1]}")
            except Exception as exc:
                cause = exc.__cause__ if isinstance(exc, CaseProcessingError) and exc.__cause__ else exc
                failure = {
                    "case_id": case.case_id,
                    "failure_stage": getattr(exc, "stage", "case_processing"),
                    "exception_type": type(cause).__name__,
                    "message": str(exc),
                    "traceback": "".join(traceback.format_exception(exc)),
                    "recoverable": True,
                    "timestamp": utc_timestamp(),
                }
                append_jsonl(failures_path, failure)
                print(f"[WARN] Case failed: {case.case_id}: {exc}", file=sys.stderr)
                if pending_index == 0:
                    metadata.update(
                        {
                            "status": "gpu_preflight_failed",
                            "gpu_preflight": {"status": "failed", "case_id": case.case_id, "error": str(exc)},
                            "finished_at": utc_timestamp(),
                        }
                    )
                    atomic_write_json(metadata_path, metadata)
                    return 1
    except KeyboardInterrupt:
        interrupted = True
        print("[WARN] Interrupted; committing completed in-memory cases before exit.", file=sys.stderr)
    finally:
        committed = store.flush(results_path)
        completed_ids.update(committed)

    final_results = load_jsonl(results_path, repair_trailing=True)
    final_failures = load_jsonl(failures_path, repair_trailing=True)
    metadata.update(
        {
            "status": "interrupted" if interrupted else "complete",
            "finished_at": utc_timestamp(),
            "counts": {"results": len(final_results), "failures": len(final_failures)},
        }
    )
    atomic_write_json(metadata_path, metadata)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
