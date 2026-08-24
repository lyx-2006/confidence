from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from confidence_test.dataset_utils import load_evaluation_cases
from confidence_test.joint_answer_source_extension import JointAnswerSourceGenerator
from confidence_test.source_attribution_variants import get_source_prompt_variant
from dp_SA.io_utils import canonical_hash, load_jsonl, sha256_file
from dp_SA.prompts import SA_PREFILL
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant

from .config import DATASET_PATH

JOINT_OUTPUT = re.compile(r"^\*\*Answer\*\*: (?P<answer>[^\r\n]+)\n\*\*Source Attribution\*\*:(?P<space> ?)(?P<label>[0-8])$")


def record_key(row: dict[str, Any]) -> tuple[Any, ...]:
    raw = str(row["item_id"])
    item = (0, int(raw)) if raw.isdigit() else (1, raw)
    return (*item, int(row["prior_index"]), str(row["condition"]), str(row.get("version", "v4")))


def _dataset_index() -> dict[tuple[str, int], Any]:
    cases, _ = load_evaluation_cases(DATASET_PATH)
    return {(str(case.item_id), int(case.prior_index)): case for case in cases}


def joint_candidates(source_dir: Path) -> list[dict[str, Any]]:
    index = _dataset_index()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with (source_dir / "results.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            if not (source.get("status") == "completed" and source.get("version") == "v4" and
                    source.get("attribution_mode") == "joint" and source.get("condition") in {"conflict_easy", "conflict_hard"}):
                continue
            generated = source.get("generated") or {}
            attribution = generated.get("source_attribution") or {}
            answer_result = generated.get("current_answer_result") or {}
            raw = answer_result.get("raw_output")
            match = JOINT_OUTPUT.fullmatch(raw or "")
            hard = attribution.get("hard_label")
            if match is None or hard is None:
                continue
            case_id = str(source["case_id"])
            if case_id in seen:
                raise ValueError(f"Duplicate joint source case: {case_id}")
            case = index[(str(source["item_id"]), int(source["prior_index"]))]
            answer = str(generated.get("current_answer"))
            if match.group("answer") != answer:
                raise ValueError(f"Joint raw answer differs bytewise for {case_id}")
            image = case.conditions[str(source["condition"])].resolved_image_path
            rows.append({
                "status": "completed", "arm": "joint", "case_id": case_id,
                "item_id": str(source["item_id"]), "prior_index": int(source["prior_index"]),
                "condition": str(source["condition"]), "version": "v4", "question": case.question,
                "text_clue": case.text_clue, "image_path": str(image), "raw_answer": answer,
                # Match V3V4SourceRunner's historical teacher stage exactly.  It
                # preserves the natural answer bytes, appends the parsed class
                # directly to the colon, and reads the logits at that colon.
                "raw_output": raw,
                "assistant_prefix": (
                    f"**Answer**: {answer}\n**Source Attribution**:"
                    f"{match.group('label')}"
                ),
                "free_generation_pre_digit_whitespace": match.group("space"),
                "parsed_class": int(match.group("label")),
                "argmax_hard_class": int(hard), "soft_sa_image_score": float(attribution["soft_image_score"]),
                "class_logits": [float(x) for x in attribution["class_logits"]],
                "class_probabilities": [float(x) for x in attribution["class_probabilities"]],
            })
            seen.add(case_id)
    return sorted(rows, key=record_key)


def select_joint_manifest(candidates: Iterable[dict[str, Any]], *, per_side: int, seed: int) -> list[dict[str, Any]]:
    eligible = []
    for row in candidates:
        hard = int(row["argmax_hard_class"])
        side = "image_side" if hard in {5, 6, 7, 8} else "text_side" if hard in {0, 1, 2, 3} else None
        if side:
            eligible.append({**row, "test_side": side})
    ordered = sorted(eligible, key=record_key)
    random.Random(seed).shuffle(ordered)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for side in ("image_side", "text_side"):
        rank = 0
        for permutation, row in enumerate(ordered):
            item = str(row["item_id"])
            if row["test_side"] != side or item in used:
                continue
            rank += 1
            selected.append({**row, "selection_rank": rank, "random_permutation_index": permutation})
            used.add(item)
            if rank == per_side:
                break
        if rank != per_side:
            raise ValueError(f"Insufficient unique joint {side} candidates: {rank}/{per_side}")
    if len(selected) != 2 * per_side or len(used) != len(selected):
        raise AssertionError("Joint manifest is not item-unique and balanced")
    return sorted(selected, key=record_key)


def delayed_manifest(source_dir: Path, *, per_side: int) -> list[dict[str, Any]]:
    path = source_dir / "steering" / "test_manifest.jsonl"
    rows = load_jsonl(path)
    selected = []
    for row in rows:
        if row["phase0_raw_answer"] != row["phase1_inserted_raw_answer"]:
            raise ValueError(f"Delayed raw-answer byte parity failed for {row['case_id']}")
        selected.append({
            **row, "arm": "delayed", "raw_answer": row["phase0_raw_answer"],
            "raw_output": f"{SA_PREFILL}{row['raw_generated_class']}", "assistant_prefix": SA_PREFILL,
        })
    counts = {side: sum(r.get("test_side") == side for r in selected) for side in ("image_side", "text_side")}
    if counts != {"image_side": per_side, "text_side": per_side} or len({str(r["item_id"]) for r in selected}) != 2 * per_side:
        raise ValueError(f"Delayed frozen manifest does not satisfy requested balance: {counts}")
    return sorted(selected, key=record_key)


def prepare_case(inference: Any, row: dict[str, Any]) -> tuple[str, Any]:
    if row["arm"] == "joint":
        variant = get_source_prompt_variant("answer_basis_9")
        prompt = variant.v4_joint_prompt.format(
            question=row["question"], text_clue=row["text_clue"], source_classes=variant.class_text
        )
        generator = JointAnswerSourceGenerator(inference)
        _messages, rendered, inputs = generator.prepare_inputs(
            prompt, row["image_path"], assistant_text=row["assistant_prefix"]
        )
        return rendered, inputs
    messages = [
        {"role": "user", "content": [{"type": "image", "image": str(Path(row["image_path"]).resolve())}, {"type": "text", "text": row["phase1_prompt"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]},
    ]
    rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
    inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=inference._get_inputs_device())
    return rendered, inputs


def prepare_joint_replay(inference: Any, row: dict[str, Any]) -> tuple[Any, list[int]]:
    """Prepare the historical joint generation prefix and forced tokens up to SAC."""
    if row["arm"] != "joint":
        raise ValueError("Joint replay requested for a non-joint row")
    variant = get_source_prompt_variant("answer_basis_9")
    prompt = variant.v4_joint_prompt.format(
        question=row["question"], text_clue=row["text_clue"], source_classes=variant.class_text
    )
    generator = JointAnswerSourceGenerator(inference)
    _messages, _rendered, base_inputs = generator.prepare_inputs(
        prompt, row["image_path"], assistant_text="**Answer**:"
    )
    suffix = row["assistant_prefix"][len("**Answer**:"):]
    forced = generator.tokenizer.encode(suffix, add_special_tokens=False)
    if not forced:
        raise ValueError(f"Joint replay suffix is empty for {row['case_id']}")
    return base_inputs, [int(x) for x in forced]


def input_fingerprints(joint_dir: Path, delayed_dir: Path, manifests: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    variant = get_source_prompt_variant("answer_basis_9")
    model_dir = Path(__file__).resolve().parents[2] / "qwen-2.5-vl" / "models" / "Qwen2.5-VL-7B-Instruct"
    implementation_paths = [
        Path(__file__), Path(__file__).with_name("config.py"), Path(__file__).with_name("masking.py"),
        Path(__file__).with_name("spans.py"), Path(__file__).with_name("run.py"),
        Path(__file__).with_name("analyze.py"), Path(__file__).with_name("run_pipeline.py"),
        Path(__file__).resolve().parents[2] / "layer_metacognition" / "conversation_builder.py",
        Path(__file__).resolve().parents[2] / "layer_metacognition" / "token_spans.py",
        Path(__file__).resolve().parents[2] / "qwen-2.5-vl" / "inference.py",
    ]
    image_paths = sorted({str(Path(row["image_path"]).resolve()) for rows in manifests.values() for row in rows})
    values = {
        "joint_config_sha256": sha256_file(joint_dir / "config.json"),
        "joint_results_sha256": sha256_file(joint_dir / "results.jsonl"),
        "delayed_capture_config_sha256": sha256_file(delayed_dir / "capture" / "config.json"),
        "delayed_manifest_sha256": sha256_file(delayed_dir / "steering" / "test_manifest.jsonl"),
        "manifest_hashes": {arm: canonical_hash(rows) for arm, rows in manifests.items()},
        "prompt_hashes": {
            "joint_v4_prompt": canonical_hash(variant.v4_joint_prompt),
            "joint_class_text": canonical_hash(variant.class_text),
            "delayed_phase1_prompts": canonical_hash([row["phase1_prompt"] for row in manifests["delayed"]]),
        },
        "model_processor_files": {
            name: sha256_file(model_dir / name)
            for name in ("config.json", "preprocessor_config.json", "processor_config.json", "tokenizer_config.json")
            if (model_dir / name).exists()
        },
        "image_sha256": {path: sha256_file(path) for path in image_paths},
        "implementation_sha256": {str(path.resolve()): sha256_file(path) for path in implementation_paths},
    }
    values["fingerprint"] = canonical_hash(values)
    return values
