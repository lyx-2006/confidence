from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
for candidate in (REPOSITORY_ROOT, HERE.parent.parent, HERE.parent, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dp_SA.io_utils import append_jsonl, atomic_json, canonical_hash, load_jsonl, sha256_file

from config import (
    ATTRIBUTION_EPSILON, COLORS, EASY_QUOTA, HARD_QUOTA, MODEL_PATH, OUTPUT_ROOT,
    SEED, SOURCE_DATASET, TEXT_ENTROPY_TOLERANCE, TEXT_POOL,
    TEXT_PROBABILITY_TOLERANCE, TEXT_TARGET_PER_COLOR,
)
from core import (
    cma_scores, eligible_third_colors, load_image_candidates, load_text_pool,
    matched_text_pairs, order_image_candidates, recolor_target, stable_key,
)
from prompts import (
    ANSWER_PREFILL, PHASE0_IMAGE_ONLY_TEMPLATE, PHASE0_TEXT_ONLY_TEMPLATE,
)


FILES = {
    "text": "text_single_modal.jsonl",
    "image": "image_single_modal.jsonl",
    "excluded": "excluded.jsonl",
    "manifest": "manifest.jsonl",
    "trials": "trials.jsonl",
}


def _path(root: Path, name: str) -> Path:
    return root / FILES[name]


def _record_index(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in load_jsonl(path) if key in row}


def _fingerprint(model: Path, dataset: Path, pool: Path, quotas: dict[str, int]) -> dict[str, Any]:
    model_files = (
        "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.json",
        "preprocessor_config.json", "model.safetensors.index.json",
    )
    missing = [name for name in model_files if not (model / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Qwen3 model directory is incomplete: {missing}")
    payload = {
        "schema_version": 1,
        "model_path": str(model.resolve()),
        "model_files": {name: sha256_file(model / name) for name in model_files},
        "source_dataset": {"path": str(dataset.resolve()), "sha256": sha256_file(dataset)},
        "text_pool": {"path": str(pool.resolve()), "sha256": sha256_file(pool)},
        "prompts": {
            "text_only": PHASE0_TEXT_ONLY_TEMPLATE,
            "image_only": PHASE0_IMAGE_ONLY_TEMPLATE,
            "answer_prefill": ANSWER_PREFILL,
        },
        "colors": list(COLORS),
        "quotas": quotas,
        "seed": SEED,
        "text_target_per_color": TEXT_TARGET_PER_COLOR,
        "entropy_tolerance": TEXT_ENTROPY_TOLERANCE,
        "probability_tolerance": TEXT_PROBABILITY_TOLERANCE,
        "attribution_epsilon": ATTRIBUTION_EPSILON,
    }
    return {**payload, "fingerprint": canonical_hash(payload)}


def preflight(
    *, model_path: Path, dataset_path: Path, pool_path: Path, output_root: Path,
    quotas: dict[str, int], write: bool = True,
) -> dict[str, Any]:
    rows = load_image_candidates(dataset_path)
    text_pool = load_text_pool(pool_path)
    eligible_counts = Counter()
    for row in rows:
        layout = json.loads(Path(row["source_layout"]).read_text(encoding="utf-8"))
        if eligible_third_colors(layout, row["text_color"], row["image_color"]):
            eligible_counts[row["difficulty"]] += 1
    summary = {
        "status": "passed",
        "image_candidates": dict(Counter(row["difficulty"] for row in rows)),
        "eligible_third_color_candidates": dict(eligible_counts),
        "accepted_unique_text_clues": {color: len(text_pool[color]) for color in COLORS},
        "requested_quotas": quotas,
        "quota_feasible_before_model_gates": {
            difficulty: eligible_counts[difficulty] >= quota
            for difficulty, quota in quotas.items()
        },
        "fingerprint": _fingerprint(model_path, dataset_path, pool_path, quotas),
    }
    if write:
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(output_root / "preflight.json", summary)
    return summary


class Pipeline:
    def __init__(
        self, *, model_path: Path, dataset_path: Path, pool_path: Path,
        output_root: Path, quotas: dict[str, int], resume: bool,
    ) -> None:
        self.model_path = model_path.resolve()
        self.dataset_path = dataset_path.resolve()
        self.pool_path = pool_path.resolve()
        self.root = output_root.resolve()
        self.quotas = quotas
        self.resume = resume
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = _fingerprint(self.model_path, self.dataset_path, self.pool_path, quotas)
        config_path = self.root / "config.json"
        existing = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
        state_exists = any(_path(self.root, name).exists() for name in FILES)
        if existing and existing.get("fingerprint") != self.config["fingerprint"]:
            raise RuntimeError("Existing faithful-check output has a different model/data/config fingerprint")
        if state_exists and not resume:
            raise RuntimeError("Output already contains run state; pass --resume or use a new --output-root")
        atomic_json(config_path, self.config)

        self.images = load_image_candidates(self.dataset_path)
        self.text_pool = load_text_pool(self.pool_path)
        self.text_records = _record_index(_path(self.root, "text"), "gate_id")
        self.image_records = _record_index(_path(self.root, "image"), "gate_id")
        self.manifest = _record_index(_path(self.root, "manifest"), "case_id")
        self.trials = _record_index(_path(self.root, "trials"), "case_id")
        self.excluded_ids = {
            (str(row.get("case_id")), str(row.get("stage")))
            for row in load_jsonl(_path(self.root, "excluded"))
        }
        self.text_usage = Counter()
        self.third_usage = Counter()
        for row in self.manifest.values():
            self.text_usage[row["original_text"]["text_clue"]] += 1
            self.text_usage[row["counterfactual_text"]["text_clue"]] += 1
            self.third_usage[row["third_color"]] += 1
        self.runner = None

    def _runner(self):
        if self.runner is None:
            from runtime import FaithfulQwenRunner
            self.runner = FaithfulQwenRunner(self.model_path)
        return self.runner

    def _exclude(self, row: dict[str, Any], stage: str, reason: str, **details: Any) -> None:
        key = (str(row["case_id"]), stage)
        if key in self.excluded_ids:
            return
        record = {
            "case_id": row["case_id"], "item_id": row["item_id"],
            "difficulty": row["difficulty"], "stage": stage, "reason": reason,
            "details": details, "config_fingerprint": self.config["fingerprint"],
        }
        append_jsonl(_path(self.root, "excluded"), record)
        self.excluded_ids.add(key)

    def _text_gate(self, question: str, candidate: dict[str, Any]) -> dict[str, Any]:
        gate_id = canonical_hash({
            "kind": "text", "question": question, "text_clue": candidate["text_clue"],
            "expected": candidate["color"], "config": self.config["fingerprint"],
        })
        if gate_id in self.text_records:
            return self.text_records[gate_id]
        # A wrong model answer is a completed gate with gate_passed=False.
        # Infrastructure/model exceptions must stop the run rather than make
        # every remaining clue look like a bad scientific candidate.
        result = self._runner().text_only(question, candidate["text_clue"], candidate["color"])
        record = {
            "gate_id": gate_id, "status": "completed", "error": None,
            "source": candidate,
            "text_pool_path": str(self.pool_path),
            "text_pool_sha256": self.config["text_pool"]["sha256"],
            "config_fingerprint": self.config["fingerprint"], **result,
        }
        append_jsonl(_path(self.root, "text"), record)
        self.text_records[gate_id] = record
        return record

    def _image_gate(self, row: dict[str, Any], image_path: Path, expected: str) -> dict[str, Any]:
        image_hash = sha256_file(image_path)
        gate_id = canonical_hash({
            "kind": "image", "question": row["question"], "image_sha256": image_hash,
            "expected": expected, "config": self.config["fingerprint"],
        })
        if gate_id in self.image_records:
            return self.image_records[gate_id]
        result = self._runner().image_only(row["question"], str(image_path), expected)
        record = {
            "gate_id": gate_id, "case_id": row["case_id"], "status": "completed",
            "error": None, "expected_color": expected, "image_sha256": image_hash,
            "config_fingerprint": self.config["fingerprint"], **result,
        }
        append_jsonl(_path(self.root, "image"), record)
        self.image_records[gate_id] = record
        return record

    def _passing_texts(
        self, question: str, color: str, limit: int | None = TEXT_TARGET_PER_COLOR,
    ) -> list[dict[str, Any]]:
        passing: list[dict[str, Any]] = []
        candidates = self.text_pool[color]
        # First seek the requested ten. If model failures leave fewer, exhaust this color.
        for candidate in candidates:
            record = self._text_gate(question, candidate)
            if record.get("gate_passed") is True:
                passing.append(record)
                if limit is not None and len(passing) >= limit:
                    break
        if passing:
            return passing
        # The loop already exhausted the pool when no passing record was found.
        return []

    def _counterfactual_paths(self, row: dict[str, Any], color: str) -> dict[str, Path]:
        directory = self.root / "counterfactual_images"
        stem = f"{row['case_id']}__target_{color}"
        return {
            "image": directory / f"{stem}.png",
            "layout": directory / f"{stem}.layout.json",
            "target_mask": directory / f"{stem}.target_mask.png",
            "occluder_mask": directory / f"{stem}.occluder_mask.png",
        }

    def _try_select(self, row: dict[str, Any]) -> dict[str, Any] | None:
        original_image_gate = self._image_gate(row, Path(row["source_image"]), row["image_color"])
        if original_image_gate.get("gate_passed") is not True:
            self._exclude(row, "selection", "original_image_single_modal_gate_failed",
                          gate_id=original_image_gate["gate_id"])
            return None

        original_texts = self._passing_texts(row["question"], row["text_color"])
        if not original_texts:
            self._exclude(row, "selection", "no_passing_original_text_clue")
            return None

        layout = json.loads(Path(row["source_layout"]).read_text(encoding="utf-8"))
        colors = eligible_third_colors(layout, row["text_color"], row["image_color"])
        colors.sort(key=lambda color: (self.third_usage[color], stable_key(row["case_id"], color)))
        attempted: list[dict[str, Any]] = []
        for third_color in colors:
            counterfactual_texts = self._passing_texts(row["question"], third_color)
            matches = matched_text_pairs(
                original_texts, counterfactual_texts,
                TEXT_ENTROPY_TOLERANCE, TEXT_PROBABILITY_TOLERANCE,
            )
            # The first pass uses about ten clues per color. Only if that set
            # cannot produce a matched pair do we consume the remaining clues.
            if not matches:
                original_texts = self._passing_texts(row["question"], row["text_color"], None)
                counterfactual_texts = self._passing_texts(row["question"], third_color, None)
                matches = matched_text_pairs(
                    original_texts, counterfactual_texts,
                    TEXT_ENTROPY_TOLERANCE, TEXT_PROBABILITY_TOLERANCE,
                )
            matches.sort(key=lambda match: (
                self.text_usage[match[0]["text_clue"]] + self.text_usage[match[1]["text_clue"]],
                match[2]["entropy_delta"], match[2]["target_probability_delta"],
                match[2]["target_margin_delta"], match[2]["length_delta"],
                stable_key(match[0]["gate_id"], match[1]["gate_id"]),
            ))
            if not matches:
                attempted.append({"third_color": third_color, "reason": "no_difficulty_matched_text_pair"})
                continue
            paths = self._counterfactual_paths(row, third_color)
            audit = recolor_target(
                source_image=Path(row["source_image"]), source_layout=Path(row["source_layout"]),
                target_mask=Path(row["source_target_mask"]),
                occluder_mask=Path(row["source_occluder_mask"]),
                destination_image=paths["image"], destination_layout=paths["layout"],
                destination_target_mask=paths["target_mask"],
                destination_occluder_mask=paths["occluder_mask"], new_color=third_color,
            )
            counterfactual_image_gate = self._image_gate(row, paths["image"], third_color)
            if counterfactual_image_gate.get("gate_passed") is not True:
                attempted.append({
                    "third_color": third_color, "reason": "counterfactual_image_single_modal_gate_failed",
                    "gate_id": counterfactual_image_gate["gate_id"],
                })
                continue
            original_text, counterfactual_text, deltas = matches[0]
            return {
                **row,
                "third_color": third_color,
                "counterfactual_image": str(paths["image"]),
                "counterfactual_layout": str(paths["layout"]),
                "counterfactual_target_mask": str(paths["target_mask"]),
                "counterfactual_occluder_mask": str(paths["occluder_mask"]),
                "original_text": original_text,
                "counterfactual_text": counterfactual_text,
                "original_image_gate": original_image_gate,
                "counterfactual_image_gate": counterfactual_image_gate,
                "text_match_deltas": deltas,
                "pixel_audit": audit,
                "selection_attempts_before_acceptance": attempted,
                "config_fingerprint": self.config["fingerprint"],
            }
        self._exclude(row, "selection", "all_clue_and_third_color_candidates_exhausted",
                      attempts=attempted)
        return None

    def select(self) -> list[dict[str, Any]]:
        used_items = {str(row["item_id"]) for row in self.manifest.values()}
        for difficulty in ("easy", "hard"):
            have = sum(row["difficulty"] == difficulty for row in self.manifest.values())
            target = self.quotas[difficulty]
            if have >= target:
                continue
            ordered = order_image_candidates(self.images, difficulty, used_items)
            for row in ordered:
                if row["case_id"] in self.manifest:
                    continue
                if sum(value["difficulty"] == difficulty for value in self.manifest.values()) >= target:
                    break
                selected = self._try_select(row)
                if selected is None:
                    continue
                append_jsonl(_path(self.root, "manifest"), selected)
                self.manifest[selected["case_id"]] = selected
                used_items.add(str(selected["item_id"]))
                self.text_usage[selected["original_text"]["text_clue"]] += 1
                self.text_usage[selected["counterfactual_text"]["text_clue"]] += 1
                self.third_usage[selected["third_color"]] += 1
        return list(self.manifest.values())

    def _trial(self, case: dict[str, Any]) -> dict[str, Any]:
        question = case["question"]
        original_clue = case["original_text"]["text_clue"]
        counterfactual_clue = case["counterfactual_text"]["text_clue"]
        original_image = case["source_image"]
        counterfactual_image = case["counterfactual_image"]
        original_answer = self._runner().original_answer(question, original_clue, original_image)
        if not original_answer.get("legal_color_answer"):
            raise ValueError(f"Original multimodal answer is not a legal color: {original_answer.get('actual_output')!r}")
        fixed = original_answer["normalized_answer"]
        cell_inputs = {
            "original": (original_image, original_clue),
            "image_counterfactual": (counterfactual_image, original_clue),
            "text_counterfactual": (original_image, counterfactual_clue),
            "joint_counterfactual": (counterfactual_image, counterfactual_clue),
        }
        cells = {
            name: self._runner().multimodal_score(question, clue, image, fixed)
            for name, (image, clue) in cell_inputs.items()
        }
        logits = [cells[name]["target_logit"] for name in cell_inputs]
        log_probabilities = [cells[name]["fixed_answer_log_probability"] for name in cell_inputs]
        if any(value is None for value in logits + log_probabilities):
            raise ValueError("A four-cell fixed-answer score is missing")
        cma_logit = cma_scores(*logits, epsilon=ATTRIBUTION_EPSILON)
        cma_log_probability = cma_scores(*log_probabilities, epsilon=ATTRIBUTION_EPSILON)
        short_sa = self._runner().short_sa(question, original_clue, original_image, fixed)
        if short_sa["fixed_answer"] != fixed:
            raise RuntimeError("Short-SA fixed answer differs from original multimodal generation")
        return {
            "case_id": case["case_id"], "item_id": case["item_id"],
            "difficulty": case["difficulty"], "question": question,
            "text_color": case["text_color"], "image_color": case["image_color"],
            "third_color": case["third_color"], "fixed_answer": fixed,
            "status": "completed", "original_answer": original_answer,
            "cells": cells, "cma_logit": cma_logit,
            "cma_log_probability": cma_log_probability, "short_sa": short_sa,
            "original_text_gate_id": case["original_text"]["gate_id"],
            "counterfactual_text_gate_id": case["counterfactual_text"]["gate_id"],
            "original_image_gate_id": case["original_image_gate"]["gate_id"],
            "counterfactual_image_gate_id": case["counterfactual_image_gate"]["gate_id"],
            "text_match_deltas": case["text_match_deltas"],
            "config_fingerprint": self.config["fingerprint"],
        }

    def run_trials(self) -> list[dict[str, Any]]:
        for case in sorted(self.manifest.values(), key=lambda row: (row["difficulty"], row["case_id"])):
            if case["case_id"] in self.trials:
                continue
            try:
                record = self._trial(case)
            except Exception as exc:
                self._exclude(case, "trial", "trial_failed", error={
                    "type": type(exc).__name__, "message": str(exc),
                })
                continue
            append_jsonl(_path(self.root, "trials"), record)
            self.trials[case["case_id"]] = record
        return list(self.trials.values())

    def run(self) -> dict[str, Any]:
        started = time.time()
        selected = self.select()
        trials = self.run_trials()
        summary = {
            "status": "complete", "config_fingerprint": self.config["fingerprint"],
            "requested": self.quotas,
            "selected": dict(Counter(row["difficulty"] for row in selected)),
            "completed_trials": dict(Counter(row["difficulty"] for row in trials)),
            "selected_total": len(selected), "completed_trial_total": len(trials),
            "excluded_total": len(load_jsonl(_path(self.root, "excluded"))),
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(self.root / "run_summary.json", summary)
        return summary


def _pid_lock(path: Path):
    class Lock:
        def __enter__(self):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                try:
                    pid = int(path.read_text(encoding="utf-8"))
                    os.kill(pid, 0)
                    raise RuntimeError(f"Faithful-check is already active as PID {pid}")
                except ProcessLookupError:
                    path.unlink()
                except ValueError:
                    raise RuntimeError(f"Invalid PID lock: {path}") from None
            path.write_text(str(os.getpid()), encoding="utf-8")
            return self

        def __exit__(self, *_args):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run faithful-check counterfactual modality dependence")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one easy and one hard case")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=SOURCE_DATASET)
    parser.add_argument("--text-pool", type=Path, default=TEXT_POOL)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quotas = {"easy": 1 if args.smoke else EASY_QUOTA, "hard": 1 if args.smoke else HARD_QUOTA}
    result = preflight(
        model_path=args.model_path, dataset_path=args.dataset, pool_path=args.text_pool,
        output_root=args.output_root, quotas=quotas,
    )
    if args.preflight_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    with _pid_lock(args.output_root / ".run.pid"):
        pipeline = Pipeline(
            model_path=args.model_path, dataset_path=args.dataset, pool_path=args.text_pool,
            output_root=args.output_root, quotas=quotas, resume=args.resume,
        )
        print(json.dumps(pipeline.run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
