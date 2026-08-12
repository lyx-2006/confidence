"""CLI for the Stage 3 Source Attribution Formation Pilot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from confidence_test.source_attribution_schema import ASSISTANT_SOURCE_ATTRIBUTION_PREFILL
from layer_metacognition.hidden_state_store import atomic_write_json

from .sa_formation.analysis import build_final_analysis
from .sa_formation.core import (
    EXPERIMENT_DIR_NAMES,
    SAFormationArtifacts,
    SAOOFDirectionRepository,
    canonical_message_hash,
    fit_oof_directions,
    initialize_run,
    load_baseline_rows,
    read_json,
    validate_output_dir,
    write_jsonl_atomic,
)
from .sa_formation.experiments import (
    run_experiment_0,
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    run_experiment_4,
    run_experiment_5,
    select_history_cohort,
    write_skipped_experiments,
)
from .sa_formation.runtime import (
    SOURCE_CHOICE_PROMPT,
    Stage3Runtime,
    assistant_message,
    build_history_messages,
    prepare_measurement,
    prepare_policy_measurement,
    right_pad_measurement_inputs,
    source_prefix_from_generation,
    text_content,
)


DEFAULT_EXPERIMENT = Path(__file__).resolve().parent / "output" / "Final_v4_run" / "answer_basis_9"


class TimeBudgetExceeded(RuntimeError):
    pass


class Deadline:
    def __init__(self, minutes: float) -> None:
        if minutes <= 0:
            raise ValueError("--max-minutes must be positive")
        self.started = time.monotonic()
        self.seconds = minutes * 60

    def check(self) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed >= self.seconds:
            raise TimeBudgetExceeded(f"Stage 3 wall-clock budget exhausted after {elapsed / 60:.1f} minutes")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    value.add_argument("--output-dir")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--smoke-only", action="store_true")
    value.add_argument("--max-minutes", type=float, default=60.0)
    return value


def _configuration(artifacts: SAFormationArtifacts, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment_dir": str(artifacts.experiment_dir),
        "output_dir": str(output),
        "model_path": str(artifacts.model_path),
        "inference_path": str(artifacts.inference_path),
        "dataset": str(artifacts.dataset),
        "primary": {"version": "v4", "attribution_mode": "joint", "source_prompt_variant": "answer_basis_9", "position": "panl", "layer": 18, "seed": 42},
        "ridge_alphas": [0.1, 1, 10, 100, 1000],
        "alpha_unit": "fold-training SD(h dot d_SA_unit)",
        "max_minutes": args.max_minutes,
    }


def _implementation_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    relative_paths = [
        "confidence_test/joint_answer_source_extension.py",
        "confidence_test/source_attribution_analyzer.py",
        "layer_metacognition/conversation_builder.py",
        "layer_metacognition/run_sa_formation.py",
        "layer_metacognition/sa_formation/core.py",
        "layer_metacognition/sa_formation/runtime.py",
        "layer_metacognition/sa_formation/experiments.py",
        "layer_metacognition/sa_formation/analysis.py",
    ]
    from .sa_formation.core import sha256_file

    return {
        relative: {"path": str(root / relative), "sha256": sha256_file(root / relative)}
        for relative in relative_paths
    }


def _dry_run(artifacts: SAFormationArtifacts, output: Path) -> dict[str, Any]:
    rows = load_baseline_rows(artifacts)
    payload = {
        "status": "dry_run",
        "output_dir": str(output),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "baseline_completed_with_sa": len(rows),
        "unique_items": len({row["item_id"] for row in rows}),
        "fold_counts": {str(fold): sum(row["fold"] == fold for row in rows) for fold in range(5)},
        "provenance": artifacts.provenance(),
        "would_mutate_existing_stage1_stage2": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _gpu_smoke(
    runtime: Stage3Runtime,
    oof: list[dict[str, Any]],
    directions: SAOOFDirectionRepository,
) -> dict[str, Any]:
    row = select_history_cohort(oof, 1)[0]
    case = runtime.case(row["item_id"], row["prior_index"])
    direction = directions.get(row["fold"])
    messages = build_history_messages(case, row["condition"], "text_first", str(row["text_answer"]))
    generated = runtime.generator.generate_messages(messages, case.answer_classes, max_new_tokens=48)
    if not generated.parse_success or not generated.source_label or generated.source_attribution is None:
        raise RuntimeError(f"History smoke Pass 1 failed: {generated.error}")
    assistant_text = source_prefix_from_generation(generated.raw_output, generated.source_label)
    pass2_messages = messages[:-1] + [assistant_message(assistant_text)]
    prepared = prepare_measurement(runtime.generator, pass2_messages, assistant_text=assistant_text, answer=str(generated.normalized_answer))
    measured = runtime.measure(prepared, direction)
    logit_error = max(abs(float(a) - float(b)) for a, b in zip(generated.source_attribution["class_logits"], measured.source["class_logits"]))
    if logit_error > 0.25:
        raise RuntimeError(f"History Pass 1/Pass 2 BF16 reconstruction failed: {logit_error}")

    prefix = messages[:-1]
    answer = str(generated.normalized_answer)
    branch_a_text = f"**Answer**: {answer}\n\n{ASSISTANT_SOURCE_ATTRIBUTION_PREFILL}"
    branch_a_messages = prefix + [assistant_message(branch_a_text)]
    branch_b_messages = prefix + [assistant_message(f"**Answer**: {answer}\n\n"), {"role": "user", "content": text_content(SOURCE_CHOICE_PROMPT)}, assistant_message("**Source Choice**:")]
    branch_a = prepare_measurement(runtime.generator, branch_a_messages, assistant_text=branch_a_text, answer=answer)
    branch_b = prepare_policy_measurement(runtime.generator, branch_b_messages, assistant_text="**Source Choice**:", fixed_answer=answer)
    common_length = max(int(branch_a.inputs.input_ids.shape[1]), int(branch_b.inputs.input_ids.shape[1]))
    pad_token_id = int(runtime.generator.tokenizer.pad_token_id)
    right_pad_measurement_inputs(branch_a, common_length, pad_token_id=pad_token_id)
    right_pad_measurement_inputs(branch_b, common_length, pad_token_id=pad_token_id)
    a_state = runtime.measure(branch_a, direction)
    b_state = runtime.measure(branch_b, direction, policy=True)
    difference = a_state.hidden - b_state.hidden
    hidden_error = float(np.max(np.abs(difference)))
    relative_l2 = float(np.linalg.norm(difference) / max(np.linalg.norm(a_state.hidden), 1e-12))
    cosine = float(a_state.hidden @ b_state.hidden / max(np.linalg.norm(a_state.hidden) * np.linalg.norm(b_state.hidden), 1e-12))
    z_error_sigma = abs(a_state.z_sa - b_state.z_sa) / direction.sigma_z
    if not (cosine >= 0.999 and relative_l2 <= 0.03 and z_error_sigma <= 0.10):
        raise RuntimeError(
            "Branched continuation prefix smoke failed: "
            f"max={hidden_error}, rel_l2={relative_l2}, cosine={cosine}, z_error_sigma={z_error_sigma}"
        )
    tokenizer = runtime.generator.tokenizer
    source_tokens = runtime.source_analyzer.token_specification.to_dict()
    policy_tokens = runtime.policy_analyzer.token_specification.to_dict()
    runtime.release_inputs(prepared, branch_a, branch_b)
    return {
        "status": "passed",
        "case_id": row["case_id"],
        "history_pass1_pass2_max_logit_error": logit_error,
        "history_bf16_tolerance": 0.25,
        "branched_prefix_hidden_max_abs_error": hidden_error,
        "branched_prefix_hidden_relative_l2": relative_l2,
        "branched_prefix_hidden_cosine": cosine,
        "branched_prefix_z_error_sigma": z_error_sigma,
        "hook_exactly_once": measured.applied_count == 1,
        "alpha_zero_injection_l2": measured.injection_l2,
        "source_token_specification": source_tokens,
        "policy_token_specification": policy_tokens,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    deadline = Deadline(args.max_minutes)
    artifacts = SAFormationArtifacts.discover(args.experiment_dir)
    output = validate_output_dir(
        artifacts.experiment_dir,
        args.output_dir or (artifacts.experiment_dir / "stage3_sa_formation"),
    )
    if args.dry_run:
        _dry_run(artifacts, output)
        return 0
    configuration = _configuration(artifacts, output, args)
    initialize_run(output, configuration, resume=args.resume)
    atomic_write_json(
        output / "provenance.json",
        {"inputs": artifacts.provenance(), "implementation": _implementation_provenance()},
    )
    natural_dir = output / EXPERIMENT_DIR_NAMES[0]
    oof_path = natural_dir / "oof_predictions.jsonl"
    if (output / "directions" / "index.json").is_file() and oof_path.is_file():
        from layer_metacognition.hidden_state_store import load_jsonl
        oof = load_jsonl(oof_path)
    else:
        natural_dir.mkdir(parents=True, exist_ok=True)
        oof, audits = fit_oof_directions(artifacts, output)
        write_jsonl_atomic(oof_path, oof)
        atomic_write_json(natural_dir / "fold_audit.json", audits)
    directions = SAOOFDirectionRepository(output / "directions")
    if not torch.cuda.is_available():
        atomic_write_json(
            output / "gpu_skipped.json",
            {
                "status": "gpu_skipped",
                "reason": "torch.cuda.is_available() is false",
                "formal_experiments_run": False,
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
        )
        return 0
    runtime = Stage3Runtime(artifacts)
    smoke = _gpu_smoke(runtime, oof, directions)
    atomic_write_json(output / "gpu_smoke.json", smoke)
    if args.smoke_only:
        return 0
    deadline.check()
    exp0, gate = run_experiment_0(runtime, artifacts, output, oof, directions, deadline=deadline.check)
    if not gate.run_natural_formation:
        write_skipped_experiments(output, 1, gate.reason)
    else:
        run_experiment_1(runtime, artifacts, output, oof)
        run_experiment_2(runtime, output, oof, directions, deadline=deadline.check)
        run_experiment_3(runtime, artifacts, output, directions, deadline=deadline.check)
        run_experiment_4(runtime, output, directions, gate, deadline=deadline.check)
        run_experiment_5(runtime, output, directions, gate, deadline=deadline.check)
    final = build_final_analysis(output, artifacts.decision_direction_dir)
    atomic_write_json(output / "progress.json", {"status": "complete", "gate": gate.to_dict(), "final_analysis": str(output / "analysis" / "FINAL_ANALYSIS.md")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
