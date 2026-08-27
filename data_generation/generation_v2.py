"""V2 text-entropy and conflict-image producers.

This module deliberately reuses the repository's trusted layout/renderer
functions, while keeping V2 state and output formats separate from the legacy
confidence-based generators.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import string
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from generation_runtime import PersistentQwenQueue, atomic_write_json, ensure_isolated_root, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]


def _legacy_output_paths() -> tuple[Path, Path, Path]:
    """Legacy output roots that V2 must never overlap (kept for audit)."""
    return (
        REPO_ROOT / "datasets",
        REPO_ROOT / "data_generation" / "legacy" / "generate_color_pool" / "output",
        REPO_ROOT / "data_generation" / "legacy" / "generate_dataset" / "datasets",
    )


TEXT_POOL_SCHEMA = "text_entropy_pool.v2"
DATASET_SCHEMA = "shape_color_dataset.v2"
ENTROPY_BIN_RANGES: tuple[tuple[float, float], ...] = (
    (0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0), (80.0, 100.0)
)
# Candidates requested per DeepSeek planner call.  A batch of 20 candidates
# (~2048 tokens) made single requests time out on a slow server; count=1 made
# the request per-candidate but exploded the thread pool.  Two candidates per
# request (~300 output tokens) keeps each request fast while cutting request
# count in half: 20 candidates -> 10 requests per bin, 5 bins concurrently
# -> 50 planner calls in flight.
PER_REQUEST_CANDIDATES = 2
COLOR_SET_A = (
    "red", "orange", "yellow", "green", "blue", "cyan",
    "purple", "pink", "brown", "white", "black", "gray",
)
COLOR_SET_B = (
    "maroon", "lime", "navy", "teal", "olive", "magenta",
    "silver", "gold", "beige", "coral", "violet", "turquoise",
)
COMMON_COLOR_TERMS = (
    "scarlet", "crimson", "vermilion", "burgundy", "wine", "amber", "ochre",
    "azure", "indigo", "chartreuse", "emerald", "jade", "khaki", "lavender",
    "lilac", "tan", "auburn", "bronze", "copper", "rose", "ruby", "saffron",
    "mustard", "peach", "salmon", "fuchsia", "plum", "ivory", "charcoal",
    "reddish", "orangish", "yellowish", "greenish", "bluish", "cyanish",
    "purplish", "pinkish", "brownish", "grayish", "greyish", "whitish",
    "blackish", "golden", "silvery", "coppery", "bronzy",
)
ABSTRACT_TONE_TERMS = {
    "warm", "warm-toned", "cool", "cool-toned", "light", "dark", "pale",
    "deep", "muted", "vivid", "tone", "toned", "saturated", "desaturated",
}
FORBIDDEN_COLOR_TERMS = frozenset(
    {value.casefold() for value in (*COLOR_SET_A, *COLOR_SET_B, *COMMON_COLOR_TERMS)}
)
SHAPE_WORDS = frozenset({
    "rectangle", "heart", "octagon", "square", "circle", "triangle", "hexagon",
    "star", "diamond", "pentagon", "oval", "trapezoid", "crescent", "cross",
    "parallelogram", "semicircle", "arrow", "shape",
})

BIN_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts" / "text_entropy_bin_prompts.json"


def _load_text_bin_prompts() -> dict[str, dict[str, str]]:
    """Load the per-bin DeepSeek prompt templates from the JSON prompt file.

    Each entropy bin has its own specialized generator and analyzer template.
    The JSON file is the single source of truth for prompt wording; loading
    fails loudly so a missing, malformed, or placeholder-incomplete file can
    never silently change generation behaviour.
    """
    if not BIN_PROMPTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing per-bin prompt file: {BIN_PROMPTS_PATH}. "
            "Create it with generator/analyzer templates for bins 0-4."
        )
    with BIN_PROMPTS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema") != "text_entropy_bin_prompts.v1":
        raise ValueError(f"Unexpected prompt-file schema in {BIN_PROMPTS_PATH}")
    generator = data.get("generator") or {}
    analyzer = data.get("analyzer") or {}
    expected_bins = {str(bin_id) for bin_id in range(5)}
    if set(generator) != expected_bins or set(analyzer) != expected_bins:
        raise ValueError(
            f"Prompt file {BIN_PROMPTS_PATH} must define generator and analyzer "
            f"templates for bins 0-4"
        )
    required_fields = {
        "generator": {"color", "colors", "count", "accepted_json"},
        "analyzer": {"color", "bin_id", "candidate_json"},
    }
    formatter = string.Formatter()
    for kind, templates in (("generator", generator), ("analyzer", analyzer)):
        for bin_id, template in templates.items():
            fields = {field for _literal, field, _spec, _conv in formatter.parse(template) if field}
            missing = required_fields[kind] - fields
            if missing:
                raise ValueError(
                    f"{kind} prompt for bin {bin_id} is missing placeholder(s): "
                    f"{sorted(missing)}"
                )
    return {"generator": generator, "analyzer": analyzer}


_TEXT_BIN_PROMPTS = _load_text_bin_prompts()


def entropy_bin_for(score: float) -> int | None:
    value = float(score)
    if not 0.0 <= value <= 100.0:
        return None
    for index, (low, high) in enumerate(ENTROPY_BIN_RANGES):
        if low <= value <= high if index == 4 else low <= value < high:
            return index
    return None


def entropy_bin_contains(score: float, bin_id: int) -> bool:
    return entropy_bin_for(score) == int(bin_id)


def entropy_bin_label(bin_id: int) -> str:
    low, high = ENTROPY_BIN_RANGES[int(bin_id)]
    return f"[{low:g}, {high:g}{']' if int(bin_id) == 4 else ')'}"


def range_tolerance(bin_id: int) -> float:
    """Max allowed entropy-score spread across the three shape questions.

    The spread is question-induced (each test uses a different shape, which
    shifts the colour competition at the answer position) and grows with the
    clue's ambiguity: strong clues vary by ~0.5, mid-ambiguity by ~8-10 and
    weak clues by 15-30.  A flat 5.0 makes bins 2-4 practically unreachable,
    so the tolerance scales with the bin: 5 + 2.5 * bin_id.
    """
    return 5.0 + 2.5 * int(bin_id)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def forbidden_color_terms(text: str) -> list[str]:
    """Return deterministic whole-token color matches, including hyphens."""
    normalized = normalize_text(text).replace("–", "-").replace("—", "-")
    tokens = [token for token in re.split(r"[^\w-]+", normalized) if token]
    matches: list[str] = []
    for token in tokens:
        parts = [part for part in token.split("-") if part]
        for part in parts:
            candidate = part[:-1] if part.endswith("s") and part[:-1] in FORBIDDEN_COLOR_TERMS else part
            if candidate in FORBIDDEN_COLOR_TERMS:
                matches.append(candidate)
    return sorted(set(matches))


def validate_color_lexical_contract(clue: str, target: str, bin_id: int) -> tuple[bool, str | None]:
    terms = forbidden_color_terms(clue)
    if bin_id == 0:
        if normalize_text(target) not in normalize_text(clue).split():
            # Whole-word matching also handles punctuation without accepting a
            # color embedded in another word.
            if not re.search(rf"(?<!\w){re.escape(normalize_text(target))}(?!\w)", normalize_text(clue)):
                return False, "bin0_missing_target_color"
        other = [term for term in terms if term != normalize_text(target)]
        return (not other, None if not other else "bin0_other_color_term")
    return (not terms, None if not terms else "forbidden_color_term")


def _shape_independent(clue: str) -> bool:
    normalized = normalize_text(clue)
    return not any(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", normalized) for word in SHAPE_WORDS)


def _minimum_reasoning_steps(bin_id: int) -> int:
    return (0, 1, 1, 2, 3)[int(bin_id)]


def validate_candidate_contract(
    candidate: dict[str, Any],
    target: str,
    requested_bin: int,
    accepted_clues: Iterable[str] = (),
    near_duplicate_threshold: float = 0.88,
) -> tuple[bool, str | None]:
    clue = str(candidate.get("text_clue", "")).strip()
    if not clue:
        return False, "empty_clue"
    if not isinstance(candidate.get("candidate_id"), (str, int)):
        return False, "missing_candidate_id"
    clue_type = candidate.get("clue_type")
    if not isinstance(clue_type, str) or not clue_type.strip():
        return False, "missing_clue_type"
    rarity = candidate.get("rarity_category")
    if not isinstance(rarity, str) or not rarity.strip():
        return False, "invalid_rarity_category"
    if "entropy_bin_id" in candidate:
        try:
            if int(candidate["entropy_bin_id"]) != int(requested_bin):
                return False, "candidate_bin_mismatch"
        except (TypeError, ValueError):
            return False, "invalid_entropy_bin_id"
    if not isinstance(candidate.get("reasoning_steps", []), list):
        return False, "reasoning_steps_not_array"
    if any(not isinstance(step, str) or not step.strip() for step in candidate.get("reasoning_steps", [])):
        return False, "invalid_reasoning_step"
    if not isinstance(candidate.get("shape_independent"), bool):
        return False, "missing_shape_independence_flag"
    if candidate.get("shape_independent") is False:
        return False, "shape_dependent"
    if not _shape_independent(clue):
        return False, "shape_dependent"
    lexical_ok, lexical_reason = validate_color_lexical_contract(clue, target, requested_bin)
    if not lexical_ok:
        return False, lexical_reason
    if len(candidate.get("reasoning_steps", [])) < _minimum_reasoning_steps(requested_bin):
        return False, "insufficient_reasoning_steps"
    normalized = normalize_text(clue)
    for other in accepted_clues:
        other_normalized = normalize_text(other)
        if normalized == other_normalized:
            return False, "duplicate_or_near_duplicate"
        # Avoid importing SequenceMatcher in every caller and keep the lexical
        # threshold deterministic.
        from difflib import SequenceMatcher
        if SequenceMatcher(None, normalized, other_normalized).ratio() >= near_duplicate_threshold:
            return False, "duplicate_or_near_duplicate"
    return True, None


def _load_dataset_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "data_generation" / "legacy" / "generate_dataset" / "generate_shape_color_dataset.py"
    name = "_shape_color_legacy_runtime"
    existing = __import__("sys").modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dataset generator from {path}")
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[name] = module
    spec.loader.exec_module(module)
    return module


def _question_bank(dataset: Any, colors: Iterable[str] = COLOR_SET_A) -> dict[str, list[dict[str, Any]]]:
    module = _load_dataset_module()
    from confidence_test.answer_metrics import parse_answer_classes
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in module.iter_dataset_items(dataset):
        question = module.question_text(item)
        try:
            choices = parse_answer_classes(question)
        except Exception:
            continue
        target = normalize_text(item.get("answer", item.get("text_ans")))
        shape = module.parse_shape(question)
        if target in colors:
            records[target].append({"question": question, "shape": shape, "choices": choices, "id": str(item.get("id"))})
    for color in colors:
        values = records[color]
        seen_shapes: set[str] = set()
        selected = [value for value in values if not (value["shape"] in seen_shapes or seen_shapes.add(value["shape"]))][:3]
        if len(selected) < 3:
            source = values[0] if values else {
                "question": f"What is the color of the square? Choose from: {', '.join(COLOR_SET_A)}.",
                "shape": "square", "choices": list(COLOR_SET_A), "id": "synthetic",
            }
            for shape in ("triangle", "hexagon", "star", "diamond", "oval"):
                if len(selected) >= 3:
                    break
                if shape in seen_shapes:
                    continue
                question = re.sub(r"color of the .+?\?", f"color of the {shape}?", source["question"], count=1, flags=re.IGNORECASE)
                selected.append({**source, "question": question, "shape": shape, "id": f"synthetic-{color}-{shape}"})
                seen_shapes.add(shape)
        if len(selected) < 3:
            raise ValueError(f"Cannot build three shape-independent questions for {color}")
        records[color] = selected[:3]
    return dict(records)


def _qwen_job(job_id: str, kind: str, prompt: str, classes: list[str], image_path: Path | None, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "kind": kind,
        "prompt": prompt,
        "answer_classes": classes,
        "image_path": str(image_path.resolve()) if image_path is not None else None,
        "max_new_tokens": 24,
        "metadata": metadata,
        "run_index": metadata.get("run_index"),
        "input_sha256": metadata.get("input_sha256"),
        "input_hash": metadata.get("input_sha256"),
    }


def _aggregate_qwen(runs: list[dict[str, Any]], target: str) -> dict[str, Any]:
    answers = [normalize_text(run.get("normalized_answer") or run.get("answer")) or None for run in runs]
    restricted = [normalize_text(run.get("restricted_top1")) or None for run in runs]
    scores = [float(run["entropy_score"]) for run in runs if run.get("entropy_score") is not None]
    raw = [
        float(run.get("raw_entropy", run.get("raw_answer_entropy")))
        for run in runs
        if run.get("raw_entropy", run.get("raw_answer_entropy")) is not None
    ]
    correct = sum(bool(run.get("parse_success")) and run.get("answer_metric_status") == "completed" and normalize_text(run.get("normalized_answer") or run.get("answer")) == normalize_text(target) for run in runs)
    top1_correct = sum(value == normalize_text(target) for value in restricted)
    return {
        "top1_answers": answers,
        "top1_answer": answers[0] if answers and len(set(answers)) == 1 else None,
        "restricted_top1": restricted,
        "restricted_top1_all_target": len(runs) == 3 and top1_correct == 3,
        "ground_truth_answer": target,
        "correct_count": correct,
        "all_correct": len(runs) == 3 and correct == 3,
        "parse_success": len(runs) == 3 and all(bool(run.get("parse_success")) for run in runs),
        "metrics_success": len(runs) == 3 and all(run.get("answer_metric_status") == "completed" for run in runs),
        "raw_entropy": sum(raw) / len(raw) if len(raw) == len(runs) and raw else None,
        "entropy_score": sum(scores) / len(scores) if len(scores) == len(runs) and scores else None,
        "entropy_score_range": max(scores) - min(scores) if len(scores) == len(runs) and scores else None,
        "runs": runs,
    }


class TextEntropyProducer:
    """DeepSeek candidate producer that submits all local tests to Qwen V2."""

    def __init__(self, *, input_path: Path, output_path: Path, queue: PersistentQwenQueue,
                 api_config_path: Path, selected_colors: list[str], selected_bins: list[int],
                 target_per_bin: int = 5, rounds: int = 5, batch_sizes: dict[int, int] | None = None,
                 near_duplicate_threshold: float = 0.88, qwen_timeout: float = 86400.0):
        self.input_path = input_path.resolve()
        self.output_path = output_path.resolve()
        ensure_isolated_root(
            self.output_path.parent.parent,
            _legacy_output_paths(),
        )
        self.queue = queue
        self.api_config_path = api_config_path.resolve()
        self.selected_colors = selected_colors
        self.selected_bins = selected_bins
        if not self.selected_colors or any(color not in COLOR_SET_A for color in self.selected_colors):
            raise ValueError("selected_colors must be a non-empty subset of the basic color set")
        if not self.selected_bins or any(int(bin_id) not in range(5) for bin_id in self.selected_bins):
            raise ValueError("selected_bins must contain entropy bin IDs 0-4")
        self.target_per_bin = int(target_per_bin)
        self.rounds = int(rounds)
        self.batch_sizes = batch_sizes or {0: 20, 1: 20, 2: 20, 3: 20, 4: 20}
        self.near_duplicate_threshold = near_duplicate_threshold
        self.qwen_timeout = qwen_timeout
        self.dataset = json.loads(self.input_path.read_text(encoding="utf-8"))
        if (
            isinstance(self.dataset, list)
            and self.dataset
            and all(isinstance(item, dict) and "prior_levels" in item for item in self.dataset)
        ) or (
            isinstance(self.dataset, dict)
            and self.dataset.get("schema_version") in {"confidence_pool", "color_prior_pool.v1"}
        ):
            raise ValueError(
                "Detected a legacy confidence pool as text input; V2 requires a question dataset, "
                "and does not migrate or resume confidence-pool artifacts"
            )
        self.questions = _question_bank(self.dataset, selected_colors)
        self.output = self._load_or_initialize()
        self.deepseek = self._build_deepseek()

    def _build_deepseek(self) -> Any:
        module = _load_dataset_module()
        return module.DeepSeekAgents(self.api_config_path)

    def _config(self) -> dict[str, Any]:
        return {
            "schema_version": TEXT_POOL_SCHEMA,
            "input_path": str(self.input_path),
            "input_sha256": sha256_file(self.input_path),
            "api_config_path": str(self.api_config_path),
            "api_config_sha256": sha256_file(self.api_config_path) if self.api_config_path.is_file() else None,
            "colors": self.selected_colors,
            "bins": self.selected_bins,
            "target_per_bin": self.target_per_bin,
            "rounds": self.rounds,
            "batch_sizes": [self.batch_sizes.get(index) for index in range(5)],
        }

    def _load_or_initialize(self) -> dict[str, Any]:
        if self.output_path.exists():
            value = json.loads(self.output_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("schema_version") != TEXT_POOL_SCHEMA:
                raise ValueError("Existing output is not a text_entropy_pool.v2; refusing legacy confidence pool")
            if not isinstance(value.get("colors"), list):
                raise ValueError("text_entropy_pool.v2 output must contain a colors array")
            saved_config = value.get("config")
            if saved_config is not None:
                # batch_sizes only controls how many DeepSeek candidates are
                # generated per round; it does not change the pool semantics
                # of already-accepted priors, so it is exempt from the resume
                # equality check.
                comparable = {key: self._config()[key] for key in self._config() if key != "batch_sizes"}
                saved_comparable = {key: saved_config[key] for key in saved_config if key != "batch_sizes"}
                if comparable != saved_comparable:
                    raise ValueError("V2 text pool resume configuration mismatch")
            return value
        colors = []
        for color in self.selected_colors:
            colors.append({
                "color": color,
                "choice_colors": list(COLOR_SET_A),
                "entropy_bins": [
                    {"entropy_bin_id": index, "range": entropy_bin_label(index), "lower": ENTROPY_BIN_RANGES[index][0], "upper": ENTROPY_BIN_RANGES[index][1], "upper_inclusive": index == 4, "priors": [], "complete": False}
                    for index in range(5)
                ],
                "complete": False,
            })
        value = {"schema_version": TEXT_POOL_SCHEMA, "entropy_definition": "restricted_12_class_natural_log: H=-sum(p*ln(p)); score=100*H/ln(12)", "config": self._config(), "colors": colors}
        atomic_write_json(self.output_path, value)
        return value

    def _entry(self, color: str) -> dict[str, Any]:
        return next(item for item in self.output["colors"] if item["color"] == color)

    def _level(self, color: str, bin_id: int) -> dict[str, Any]:
        return next(level for level in self._entry(color)["entropy_bins"] if int(level["entropy_bin_id"]) == int(bin_id))

    def _persist(self) -> None:
        for entry in self.output["colors"]:
            for level in entry["entropy_bins"]:
                level["complete"] = len(level["priors"]) >= self.target_per_bin
            entry["complete"] = all(level["complete"] for level in entry["entropy_bins"] if int(level["entropy_bin_id"]) in self.selected_bins)
        self.output["updated_at"] = time.time()
        atomic_write_json(self.output_path, self.output)

    def _generator_prompt(self, color: str, bin_id: int, count: int, accepted: list[str]) -> str:
        return _TEXT_BIN_PROMPTS["generator"][str(bin_id)].format(
            color=color,
            colors=", ".join(COLOR_SET_A),
            count=count,
            accepted_json=json.dumps(accepted, ensure_ascii=False),
        )

    def _analyzer_verdict(self, candidate: dict[str, Any], color: str, bin_id: int) -> dict[str, Any]:
        prompt = _TEXT_BIN_PROMPTS["analyzer"][str(bin_id)].format(
            color=color,
            bin_id=bin_id,
            candidate_json=json.dumps(candidate, ensure_ascii=False),
        )
        try:
            verdict = self.deepseek.call(prompt, "analyzer", retries=3)
        except Exception as exc:
            return {"accepted": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
        required = ("accepted", "clue_type", "rarity_category", "reasoning_steps_valid", "shape_independent")
        if any(key not in verdict for key in required):
            return {"accepted": False, "error": {"type": "AnalyzerSchemaError", "message": "missing verdict field"}}
        return verdict

    def _test_candidate(self, candidate: dict[str, Any], color: str, requested_bin: int, round_index: int) -> dict[str, Any]:
        clue = str(candidate["text_clue"])
        jobs: list[dict[str, Any]] = []
        for run_index, case in enumerate(self.questions[color]):
            job_id = hashlib.sha256(f"text:{color}:{requested_bin}:{round_index}:{candidate['candidate_id']}:{run_index}:{clue}".encode()).hexdigest()
            job = _qwen_job(job_id, "text", f"Question:\n{case['question']}\n\nText clue:\n{clue}\n\nOutput exactly: **Answer**: <your answer>", list(COLOR_SET_A), None, {"color": color, "entropy_bin_id": requested_bin, "candidate_id": str(candidate["candidate_id"]), "run_index": run_index, "input_sha256": hashlib.sha256(f"{case['question']}\n{clue}".encode()).hexdigest()})
            jobs.append(job)
        for job in jobs:
            self.queue.enqueue(job)
        runs = [self.queue.wait(str(job["job_id"]), self.qwen_timeout) for job in jobs]
        result = _aggregate_qwen(runs, color)
        scores = [float(run["entropy_score"]) for run in runs if run.get("entropy_score") is not None]
        measured = [entropy_bin_for(score) for score in scores]
        result.update({
            "requested_entropy_bin": requested_bin,
            "measured_entropy_bins": measured,
            "measured_entropy_bin": entropy_bin_for(result["entropy_score"]) if result.get("entropy_score") is not None else None,
            "answers_unchanged": result["all_correct"],
            "accepted": bool(result["all_correct"] and result["restricted_top1_all_target"] and result["metrics_success"] and len(scores) == 3 and result["entropy_score_range"] is not None and result["entropy_score_range"] < range_tolerance(requested_bin) and len(set(measured)) == 1),
        })
        return result

    def _accepted_clues(self, color: str) -> list[str]:
        """All currently accepted clues across the color's bins (thread-safe read)."""
        return [
            str(item["text_clue"])
            for existing_level in self._entry(color)["entropy_bins"]
            for item in existing_level.get("priors", [])
            if isinstance(item, dict) and isinstance(item.get("text_clue"), str)
        ]

    def _produce_request(self, color: str, bin_id: int, round_index: int, request_index: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """One DeepSeek request generating PER_REQUEST_CANDIDATES candidates.

        Each request carries count=PER_REQUEST_CANDIDATES (~300 output tokens),
        so it completes in tens of seconds even on a slow server.  The chain
        (generation -> lexical contract -> analyzer verdict) runs in the caller
        thread; the caller fans out one thread per request.  Returns the
        (candidate, verdict) pairs that passed both checks.
        """
        accepted_clues = self._accepted_clues(color)
        # No retry on the planner: a failed request is skipped and the round
        # moves straight to testing whatever succeeded (one attempt each).
        payload = self.deepseek.call(
            self._generator_prompt(color, bin_id, PER_REQUEST_CANDIDATES, accepted_clues), "planner", retries=1
        )
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if not isinstance(candidates, list):
            return []
        kept: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for offset, candidate in enumerate(candidates[:PER_REQUEST_CANDIDATES]):
            if not isinstance(candidate, dict):
                continue
            candidate.setdefault("candidate_id", f"{color}-{bin_id}-{round_index}-{request_index * PER_REQUEST_CANDIDATES + offset}")
            candidate.setdefault("entropy_bin_id", bin_id)
            ok, _reason = validate_candidate_contract(candidate, color, bin_id, accepted_clues, self.near_duplicate_threshold)
            if not ok:
                continue
            verdict = self._analyzer_verdict(candidate, color, bin_id)
            if not verdict.get("accepted") or not verdict.get("shape_independent") or not verdict.get("reasoning_steps_valid"):
                continue
            kept.append((candidate, verdict))
        return kept

    def run(self) -> dict[str, Any]:
        incomplete: list[str] = []
        for color in self.selected_colors:
            pending_bins = [
                bin_id for bin_id in self.selected_bins
                if len(self._level(color, bin_id)["priors"]) < self.target_per_bin
            ]
            for round_index in range(1, self.rounds + 1):
                if not pending_bins:
                    break
                # Full-parallelism generation: one DeepSeek request per
                # PER_REQUEST_CANDIDATES candidates, all fired concurrently
                # across every active bin.  Each request carries count=2
                # (~300 output tokens) and finishes in tens of seconds even on
                # a slow server; the per-candidate contract check and analyzer
                # verdict run in the same thread.  Qwen testing stays serial
                # on this thread.
                tasks = [
                    (bin_id, request_index)
                    for bin_id in pending_bins
                    for request_index in range(int(math.ceil(self.batch_sizes.get(bin_id, 1) / PER_REQUEST_CANDIDATES)))
                ]
                produced: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {
                    bin_id: [] for bin_id in pending_bins
                }
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(tasks))) as executor:
                    futures = {
                        executor.submit(self._produce_request, color, bin_id, round_index, request_index): (bin_id, request_index)
                        for bin_id, request_index in tasks
                    }
                    for future in concurrent.futures.as_completed(futures):
                        bin_id, _request_index = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            print(f"[WARN] bin {bin_id} candidate generation failed: {exc}", flush=True)
                            continue
                        if result:
                            produced[bin_id].extend(result)
                # Qwen testing and persistence stay serial on the caller thread.
                for bin_id in pending_bins:
                    level = self._level(color, bin_id)
                    accepted_this_round = 0
                    for candidate, verdict in produced.get(bin_id, []):
                        if len(level["priors"]) >= self.target_per_bin:
                            break
                        test = self._test_candidate(candidate, color, bin_id, round_index)
                        measured = test.get("measured_entropy_bin")
                        if not test["accepted"] or measured is None:
                            continue
                        # Cross-bin routing re-checks the destination bin's own
                        # tolerance: the first filter used the requested bin, but
                        # the prior is archived under the measured bin.
                        if test.get("entropy_score_range") is not None and test["entropy_score_range"] >= range_tolerance(int(measured)):
                            continue
                        # Cross-bin routing is allowed only after validating the
                        # destination's textual contract as well.
                        accepted_clues = self._accepted_clues(color)
                        destination_candidate = {**candidate, "entropy_bin_id": int(measured)}
                        destination_ok, _ = validate_candidate_contract(destination_candidate, color, int(measured), accepted_clues, self.near_duplicate_threshold)
                        if not destination_ok:
                            continue
                        destination_verdict = verdict
                        if int(measured) != int(bin_id):
                            destination_verdict = self._analyzer_verdict(candidate, color, int(measured))
                            if not destination_verdict.get("accepted") or not destination_verdict.get("shape_independent") or not destination_verdict.get("reasoning_steps_valid"):
                                continue
                        record = {
                            **candidate,
                            "requested_entropy_bin": bin_id,
                            "measured_entropy_bin": measured,
                            "raw_entropy": test.get("raw_entropy"),
                            "entropy_score": test.get("entropy_score"),
                            "entropy_score_range": test.get("entropy_score_range"),
                            "answers_unchanged": test.get("answers_unchanged"),
                            "contains_forbidden_color_term": bool(forbidden_color_terms(str(candidate["text_clue"])) and measured != 0),
                            "analyzer_verdict": verdict,
                            "measured_bin_analyzer_verdict": destination_verdict,
                            "test_results": test,
                            "accepted": True,
                        }
                        destination = self._level(color, int(measured))
                        destination_clues = [str(item["text_clue"]) for item in destination["priors"]]
                        if len(destination["priors"]) >= self.target_per_bin or normalize_text(record["text_clue"]) in {normalize_text(item) for item in destination_clues}:
                            continue
                        destination["priors"].append(record)
                        accepted_this_round += 1
                        self._persist()
                    print(
                        f"[TextEntropy] color={color} bin={bin_id} round={round_index}/{self.rounds} "
                        f"accepted={len(level['priors'])}/{self.target_per_bin} "
                        f"(+{accepted_this_round} this round)",
                        flush=True,
                    )
                    if len(level["priors"]) < self.target_per_bin:
                        # Never silently publish a partial pool.  State is still
                        # durable for an explicit resume, but the caller must see
                        # a failed generation run.
                        incomplete.append(f"{color}:{bin_id}")
                pending_bins = [
                    bin_id for bin_id in pending_bins
                    if len(self._level(color, bin_id)["priors"]) < self.target_per_bin
                ]
        self._persist()
        if incomplete:
            raise RuntimeError(
                "Text entropy generation incomplete after configured rounds: "
                + ", ".join(incomplete)
            )
        return self.output


class DinoSimilarityIndex:
    """DINOv2 CLS embeddings and metadata-aware cosine comparisons."""

    enabled = True

    def __init__(self, model_path: Path, old_dataset: Path, old_image_root: Path | None,
                 cache_path: Path, device: str = "cpu"):
        if not model_path.is_dir():
            raise FileNotFoundError(f"DINOv2 model path does not exist: {model_path}")
        self.model_path = model_path.resolve()
        self.old_dataset = old_dataset.resolve()
        self.old_image_root = old_image_root.resolve() if old_image_root else None
        self.cache_path = cache_path.resolve()
        self.device = device
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("DINOv2 similarity requires torch and transformers") from exc
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True).to(device).eval()
        # Keep cache identity stable across runs, while invalidating it when a
        # local revision/configuration changes.  Hashing file contents for a
        # multi-gigabyte checkpoint would make every startup prohibitively
        # expensive, so include the revision marker and complete file
        # metadata (path, size, mtime) instead.
        identity_parts: list[str] = [str(self.model_path)]
        for marker in ("refs/main", "revision", "v2_revision.txt", "config.json", "preprocessor_config.json"):
            marker_path = self.model_path / marker
            if marker_path.is_file():
                try:
                    identity_parts.append(marker + ":" + marker_path.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    identity_parts.append(marker + ":unreadable")
        for child in sorted(self.model_path.rglob("*")):
            if child.is_file():
                stat = child.stat()
                identity_parts.append(f"{child.relative_to(self.model_path)}:{stat.st_size}:{stat.st_mtime_ns}")
        self.model_identity = hashlib.sha256("\n".join(identity_parts).encode()).hexdigest()
        revision_markers = (self.model_path / "v2_revision.txt", self.model_path / "refs" / "main", self.model_path / "revision")
        revision_marker = next((path for path in revision_markers if path.is_file()), None)
        self.model_revision = revision_marker.read_text(encoding="utf-8", errors="ignore").strip() if revision_marker else None
        self.entries = self._load_old_entries()
        self.embeddings: dict[str, Any] = {}
        self._load_cache()
        self._precompute_old()

    def _load_old_entries(self) -> list[dict[str, Any]]:
        payload = json.loads(self.old_dataset.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            items = [item for group in payload if isinstance(group, dict) for item in group.get("items", []) if isinstance(item, dict)]
        elif isinstance(payload, dict):
            items = [item for item in payload.get("items", []) if isinstance(item, dict)]
        else:
            raise ValueError("Old image dataset must be a list or object")
        entries: list[dict[str, Any]] = []
        for item in items:
            question = item.get("question", "")
            if isinstance(question, dict):
                question = question.get("text", "")
            try:
                shape = normalize_text(_load_dataset_module().extract_shape(str(question)))
                if not shape:
                    raise ValueError("shape parser returned no shape")
            except Exception:
                shape_match = re.search(r"color\s+of\s+(?:the\s+)?(.+?)\?", str(question), re.IGNORECASE)
                shape = normalize_text(shape_match.group(1)) if shape_match else None
                known_shapes = getattr(_load_dataset_module(), "SHAPES", ())
                matches = [value for value in known_shapes if re.search(rf"(?<!\w){re.escape(value)}(?!\w)", shape or "")]
                if matches:
                    shape = max(matches, key=len)
            image_clue = item.get("image_clue")
            if not isinstance(image_clue, dict):
                # Summary datasets use groups instead.
                image_clue = {"summary": item.get("groups", {})}
            for branch in ("consistent", "conflict"):
                target = normalize_text(item.get("answer")) if branch == "consistent" else normalize_text(item.get("conflict_ans", item.get("conflict_answer")))
                value = image_clue.get(branch)
                for difficulty in ("easy", "hard"):
                    raw = None
                    if isinstance(value, dict):
                        raw = value.get(difficulty)
                        if isinstance(raw, dict):
                            raw = raw.get("image")
                    if raw is None and isinstance(image_clue.get("summary"), dict):
                        group = image_clue["summary"].get(f"{branch}_{difficulty}")
                        raw = group.get("image") if isinstance(group, dict) else None
                    raw_values = raw if isinstance(raw, list) else [raw]
                    if not raw_values:
                        raw_values = [None]
                    for raw_value in raw_values:
                        raw = raw_value.get("image") if isinstance(raw_value, dict) else raw_value
                        if not isinstance(raw, str) or not raw.strip():
                            # A declared branch/difficulty with missing metadata is
                            # an input error, whereas an entirely absent optional
                            # branch is simply not part of the old comparison set.
                            if isinstance(value, dict) or isinstance(image_clue.get("summary"), dict):
                                raise ValueError(f"Old image metadata missing {branch}/{difficulty} for item {item.get('id')}")
                            continue
                        if not shape or not target:
                            raise ValueError(f"Old image metadata missing target shape/color for item {item.get('id')} {branch}/{difficulty}")
                        path = Path(raw)
                        resolved = path if path.is_absolute() else (self.old_image_root / path.name if self.old_image_root else self.old_dataset.parent / path)
                        if not resolved.is_file():
                            raise FileNotFoundError(f"Old image metadata points to missing file: {resolved}")
                        entries.append({"path": str(resolved.resolve()), "branch": branch, "difficulty": difficulty, "target_shape": shape, "target_color": target})
        if not entries:
            raise ValueError("Old image dataset contains no readable branch image metadata")
        return entries

    def _load_cache(self) -> None:
        if not self.cache_path.is_file():
            return
        try:
            value = self.torch.load(self.cache_path, map_location="cpu", weights_only=False)
            if value.get("schema_version") == "dinov2_embedding_cache.v1" and value.get("model_identity") == self.model_identity:
                self.embeddings = value.get("embeddings", {})
        except Exception:
            self.embeddings = {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.cache_path.name}.", suffix=".tmp", dir=self.cache_path.parent)
        os.close(fd)
        try:
            self.torch.save({"schema_version": "dinov2_embedding_cache.v1", "model_identity": self.model_identity, "embeddings": self.embeddings}, temporary)
            os.replace(temporary, self.cache_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _embed(self, path: Path) -> list[float]:
        from PIL import Image
        image_hash = sha256_file(path)
        key = str(path.resolve())
        cached = self.embeddings.get(key)
        if isinstance(cached, dict) and cached.get("sha256") == image_hash and isinstance(cached.get("vector"), list):
            return [float(value) for value in cached["vector"]]
        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model(**inputs)
            vector = output.last_hidden_state[:, 0, :]
            vector = self.torch.nn.functional.normalize(vector.float(), p=2, dim=-1)[0].cpu()
        values = [float(value) for value in vector.tolist()]
        self.embeddings[key] = {"sha256": image_hash, "vector": values}
        return values

    def _precompute_old(self) -> None:
        for entry in self.entries:
            self._embed(Path(entry["path"]))
        self._save_cache()

    @staticmethod
    def cosine(first: Iterable[float], second: Iterable[float]) -> float:
        import math
        left, right = list(map(float, first)), list(map(float, second))
        if len(left) != len(right) or not left:
            raise ValueError("Embedding dimensions do not match")
        dot = sum(a * b for a, b in zip(left, right))
        norm_left = math.sqrt(sum(a * a for a in left))
        norm_right = math.sqrt(sum(b * b for b in right))
        if norm_left == 0.0 or norm_right == 0.0:
            raise ValueError("Cannot compare zero embedding")
        return dot / (norm_left * norm_right)

    def compare(self, image_path: Path, group: tuple[str, str, str, str], siblings: Iterable[Path] = ()) -> dict[str, Any]:
        branch, shape, color, difficulty = group
        vector = self._embed(image_path)
        old = [entry for entry in self.entries if (entry["branch"], entry["target_shape"], entry["target_color"], entry["difficulty"]) == group]
        old_scores = [(self.cosine(vector, self._embed(Path(entry["path"]))), entry["path"]) for entry in old]
        sibling_scores = [(self.cosine(vector, self._embed(path)), str(path)) for path in siblings]
        candidates = old_scores + sibling_scores
        best = max(candidates, default=(None, None), key=lambda value: value[0] if value[0] is not None else -1.0)
        maximum = best[0]
        return {
            "similarity_model": "dinov2",
            "similarity_model_name": "facebook/dinov2-base",
            "comparison_group": {"branch": branch, "target_shape": shape, "target_color": color, "difficulty": difficulty},
            "compared_old_image_count": len(old),
            "max_old_cosine_similarity": max(old_scores, default=(None, None))[0],
            "most_similar_old_image": max(old_scores, default=(None, None))[1],
            "compared_sibling_image_count": len(sibling_scores),
            "max_sibling_cosine_similarity": max(sibling_scores, default=(None, None))[0],
            "most_similar_sibling_image": max(sibling_scores, default=(None, None))[1],
            "hard_threshold": 0.90,
            "target_threshold": 0.85,
            "passed": maximum is None or maximum < 0.90,
            "target_met": maximum is None or maximum <= 0.85,
        }


class DisabledSimilarityIndex:
    """Explicit no-op used when similarity validation is disabled by CLI."""

    enabled = False
    model_identity = None
    model_revision = None
    old_dataset = None
    old_image_root = None

    def compare(
        self,
        image_path: Path,
        group: tuple[str, str, str, str],
        siblings: Iterable[Path] = (),
    ) -> dict[str, Any]:
        branch, shape, color, difficulty = group
        return {
            "status": "skipped",
            "performed": False,
            "skip_reason": "explicit_cli_flag",
            "similarity_model": None,
            "similarity_model_name": None,
            "comparison_group": {
                "branch": branch,
                "target_shape": shape,
                "target_color": color,
                "difficulty": difficulty,
            },
            "compared_old_image_count": 0,
            "max_old_cosine_similarity": None,
            "most_similar_old_image": None,
            "compared_sibling_image_count": 0,
            "max_sibling_cosine_similarity": None,
            "most_similar_sibling_image": None,
            "hard_threshold": None,
            "target_threshold": None,
            # The producer consumes this gate.  ``status`` and ``performed``
            # make clear that this is an explicit bypass, not a measured pass.
            "passed": True,
            "target_met": None,
        }


def _atomic_publish(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(source) == sha256_file(destination):
            return str(destination.resolve())
        raise FileExistsError(f"Refusing to overwrite a different published artifact: {destination}")
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(destination.resolve())


class ImageDatasetV2Producer:
    """Conflict-first image producer using trusted local geometry and Qwen V2."""

    def __init__(self, *, input_path: Path, output_path: Path, image_dir: Path,
                 queue: PersistentQwenQueue, api_config_path: Path, old_image_dataset: Path,
                 old_image_root: Path | None, similarity_model_path: Path | None, similarity_cache: Path,
                 branches: list[str], seed: int = 42, images_per_difficulty: int = 1,
                 conflict_easy_count: int | None = None, conflict_hard_count: int | None = None,
                 target_rotation_mode: str = "safe", target_rotation_min: float = -30.0,
                 target_rotation_max: float = 30.0, distractor_rotation_min: float = 0.0,
                 distractor_rotation_max: float = 360.0, qwen_timeout: float = 86400.0,
                 combination_limit: int | None = None,
                 skip_similarity_validation: bool = False,
                 combination_ids: Iterable[str] | None = None,
                 state_path: Path | None = None):
        self.module = _load_dataset_module()
        self.input_path = input_path.resolve()
        self.output_path = output_path.resolve()
        self.image_dir = image_dir.resolve()
        legacy_paths = _legacy_output_paths()
        ensure_isolated_root(self.output_path.parent.parent, legacy_paths)
        ensure_isolated_root(self.image_dir, legacy_paths)
        # Parallel workers do not persist per-item state files: the parent
        # coordinator output is the single resume source of truth, so a worker
        # keeps its state in memory only (state_path=None).
        self.state_path = state_path.resolve() if state_path is not None else None
        if self.state_path is not None:
            ensure_isolated_root(self.state_path.parent, legacy_paths)
        self.queue = queue
        self.api_config_path = api_config_path.resolve()
        self.seed = seed
        if branches not in (["conflict"], ["conflict", "consistent"]):
            raise ValueError("branches must be conflict or conflict,consistent")
        self.branches = branches
        self.images_per_difficulty = images_per_difficulty
        if not 1 <= int(images_per_difficulty) <= 32:
            raise ValueError("images_per_difficulty must be between 1 and 32")
        self.easy_count = images_per_difficulty if conflict_easy_count is None else int(conflict_easy_count)
        self.hard_count = images_per_difficulty if conflict_hard_count is None else int(conflict_hard_count)
        if not 1 <= int(self.easy_count) <= 32 or not 1 <= int(self.hard_count) <= 32:
            raise ValueError("conflict variant counts must be between 1 and 32")
        if self.hard_count > self.easy_count:
            raise ValueError("conflict hard variant count cannot exceed easy variant count")
        self.rotation_mode = target_rotation_mode
        self.target_rotation_min = target_rotation_min
        self.target_rotation_max = target_rotation_max
        self.distractor_rotation_min = distractor_rotation_min
        self.distractor_rotation_max = distractor_rotation_max
        self.qwen_timeout = qwen_timeout
        self.combination_limit = int(combination_limit) if combination_limit is not None else None
        self.combination_ids = (
            sorted({str(value) for value in combination_ids})
            if combination_ids is not None
            else None
        )
        ensure_isolated_root(Path(similarity_cache).resolve().parent, legacy_paths)
        self.input_payload = json.loads(self.input_path.read_text(encoding="utf-8"))
        self.manifest = self.module.build_manifest(self.input_payload, seed)
        if self.combination_limit is not None:
            if self.combination_limit < 1:
                raise ValueError("combination_limit must be positive")
            self.manifest["combinations"] = self.manifest.get("combinations", [])[: self.combination_limit]
            self.manifest["planned_count"] = len(self.manifest["combinations"])
        if self.combination_ids is not None:
            requested_ids = set(self.combination_ids)
            selected = [
                combo
                for combo in self.manifest.get("combinations", [])
                if str(combo.get("id")) in requested_ids
            ]
            found_ids = {str(combo.get("id")) for combo in selected}
            missing_ids = sorted(requested_ids - found_ids)
            if missing_ids:
                raise ValueError(f"Unknown image combination IDs: {missing_ids}")
            self.manifest["combinations"] = selected
            self.manifest["planned_count"] = len(selected)
        # Persist every deterministic variant/attempt seed in the V2 manifest;
        # this makes resume and audit independent of process ordering.
        for combo in self.manifest.get("combinations", []):
            combo["variant_seeds"] = {
                branch: {
                    difficulty: {
                        str(index): {
                            str(attempt): self.module.derive_seed(
                                self.seed, combo["id"], branch, difficulty, index, attempt
                            )
                            for attempt in range(1, (self.module.EASY_MAX_ATTEMPTS if difficulty == "easy" else self.module.HARD_MAX_ATTEMPTS) + 1)
                        }
                        for index in range(
                            1,
                            ((self.easy_count if branch == "conflict" else self.images_per_difficulty)
                             if difficulty == "easy" else
                             (self.hard_count if branch == "conflict" else self.images_per_difficulty)) + 1,
                        )
                    }
                    for difficulty in ("easy", "hard")
                }
                for branch in self.branches
            }
        self.irr_path, self.null_path = self.module.find_shared_clue_paths(self.input_path, self.input_payload, self.output_path)
        self.skip_similarity_validation = bool(skip_similarity_validation)
        if self.skip_similarity_validation:
            self.similarity = DisabledSimilarityIndex()
        else:
            if similarity_model_path is None:
                raise ValueError(
                    "similarity_model_path is required unless similarity validation is explicitly skipped"
                )
            self.similarity = DinoSimilarityIndex(
                similarity_model_path, old_image_dataset, old_image_root, similarity_cache
            )
        self.agents = self.module.DeepSeekAgents(self.api_config_path)
        self.state = self._load_state()
        self.output = self._load_output()

    def _config(self) -> dict[str, Any]:
        old_dataset_value = getattr(self.similarity, "old_dataset", None)
        old_dataset = Path(old_dataset_value) if old_dataset_value else None
        similarity_enabled = bool(getattr(self.similarity, "enabled", True))
        return {"schema_version": DATASET_SCHEMA, "input_path": str(self.input_path), "input_sha256": sha256_file(self.input_path), "output_path": str(self.output_path), "image_dir": str(self.image_dir), "seed": self.seed, "branches": self.branches, "easy_count": self.easy_count, "hard_count": self.hard_count, "rotation_mode": self.rotation_mode, "target_rotation_min": self.target_rotation_min, "target_rotation_max": self.target_rotation_max, "distractor_rotation_min": self.distractor_rotation_min, "distractor_rotation_max": self.distractor_rotation_max, "combination_limit": self.combination_limit, "combination_ids": self.combination_ids, "api_config_path": str(self.api_config_path), "api_config_sha256": sha256_file(self.api_config_path) if self.api_config_path.is_file() else None, "similarity_validation_enabled": similarity_enabled, "old_image_dataset": str(old_dataset) if old_dataset else None, "old_image_dataset_sha256": sha256_file(old_dataset) if old_dataset and old_dataset.is_file() else None, "old_image_root": str(getattr(self.similarity, "old_image_root", None)) if getattr(self.similarity, "old_image_root", None) else None, "similarity_model": "facebook/dinov2-base" if similarity_enabled else None, "similarity_model_hash": getattr(self.similarity, "model_identity", None), "similarity_model_revision": getattr(self.similarity, "model_revision", None)}

    def _load_state(self) -> dict[str, Any]:
        if self.state_path is not None and self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") != DATASET_SCHEMA or state.get("config") != self._config():
                raise ValueError("V2 dataset resume configuration/schema mismatch")
            return state
        state = {"schema_version": DATASET_SCHEMA, "created_at": time.time(), "config": self._config(), "manifest": self.manifest, "items": {}, "failures": []}
        if self.state_path is not None:
            atomic_write_json(self.state_path, state)
        return state

    def _load_output(self) -> dict[str, Any]:
        if self.output_path.is_file():
            value = json.loads(self.output_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("schema_version") != DATASET_SCHEMA:
                raise ValueError("Existing dataset output is not shape_color_dataset.v2")
            if not isinstance(value.get("items"), list):
                raise ValueError("shape_color_dataset.v2 output must contain an items array")
            return value
        value = {"schema_version": DATASET_SCHEMA, "category": "colour", "branches": list(self.branches), "images_per_difficulty": self.images_per_difficulty, "conflict_easy_count": self.easy_count, "conflict_hard_count": self.hard_count, "items": []}
        atomic_write_json(self.output_path, value)
        return value

    def _persist(self) -> None:
        if self.state_path is not None:
            self.state["updated_at"] = time.time()
            atomic_write_json(self.state_path, self.state)
        atomic_write_json(self.output_path, self.output)

    def _rotation(self, shape: str, seed: int) -> tuple[float, bool]:
        if self.rotation_mode == "none" or shape == "circle":
            return 0.0, False
        low, high = self.target_rotation_min, self.target_rotation_max
        if self.rotation_mode == "safe":
            cap = 15.0 if shape in {"square", "diamond"} else 20.0 if shape in {"rectangle", "parallelogram"} else 30.0
            negative = (max(low, -cap), min(high, -5.0))
            positive = (max(low, 5.0), min(high, cap))
            options = [value for value in (negative, positive) if value[0] <= value[1]]
            if not options:
                raise ValueError(f"No usable target rotation range for {shape}")
            import random
            rng = random.Random(seed)
            chosen_low, chosen_high = rng.choice(options)
            return round(rng.uniform(chosen_low, chosen_high), 3), True
        if low > high:
            raise ValueError(f"No usable target rotation range for {shape}")
        import random
        rng = random.Random(seed)
        value = rng.uniform(low, high)
        return round(value, 3), True

    def _apply_target_rotation(self, layout: dict[str, Any], rotation: float) -> dict[str, Any]:
        for obj in layout.get("objects", []):
            if obj.get("role") == "target":
                obj["rotation"] = rotation
                layout.setdefault("target_geometry", {})["rotation"] = rotation
        return layout

    def _apply_distractor_rotations(self, layout: dict[str, Any], seed: int) -> None:
        import random
        rng = random.Random(seed)
        for obj in layout.get("objects", []):
            if obj.get("role") == "target":
                continue
            if obj.get("role") == "occluder":
                continue
            obj["rotation"] = round(rng.uniform(self.distractor_rotation_min, self.distractor_rotation_max), 3)

    def _test_image(self, path: Path, question: str, target: str, metadata: dict[str, Any]) -> dict[str, Any]:
        jobs = []
        for run_index in range(3):
            identity = f"image:{metadata['item_id']}:{metadata['branch']}:{metadata['difficulty']}:{metadata['variant_index']}:{metadata['attempt']}:{run_index}:{sha256_file(path)}"
            job_id = hashlib.sha256(identity.encode()).hexdigest()
            jobs.append(_qwen_job(job_id, "image", f"Question:\n{question}\n\nAnswer using only the image.\n\nOutput exactly: **Answer**: <your answer>", list(COLOR_SET_A), path, {**metadata, "run_index": run_index, "input_sha256": sha256_file(path)}))
        for job in jobs:
            self.queue.enqueue(job)
        runs = [self.queue.wait(str(job["job_id"]), self.qwen_timeout) for job in jobs]
        return _aggregate_qwen(runs, target)

    def _candidate(self, layout: dict[str, Any], work: Path, question: str, target: str, metadata: dict[str, Any], easy_entropy: float | None = None) -> dict[str, Any]:
        work.mkdir(parents=True, exist_ok=True)
        planner = self.agents.plan(layout)
        rendered = self.module.render_scene_locally(layout, work, planner["render_style"])
        geometry = self.module.validate_rendered_scene(rendered, layout)
        image = rendered / "image.png"
        group = (metadata["branch"], str(layout["target_shape"]), target, str(layout["difficulty"]))
        similarity = self.similarity.compare(image, group, metadata.get("siblings", []))
        candidate = {"attempt": metadata["attempt"], "variant_index": metadata["variant_index"], "seed": layout["case_seed"], "rotation": next(obj["rotation"] for obj in layout["objects"] if obj.get("role") == "target"), "layout": layout, "geometry": geometry, "similarity_check": similarity, "artifact_dir": str(rendered), "accepted": False}
        if not similarity["passed"]:
            candidate["failure_reason"] = "old_or_sibling_cosine_threshold"
            return candidate
        calibration = self._test_image(image, question, target, metadata)
        candidate["calibration"] = calibration
        candidate["entropy_score"] = calibration.get("entropy_score")
        candidate["all_correct"] = calibration.get("all_correct")
        if not calibration.get("all_correct"):
            candidate["failure_reason"] = "image_not_correct_in_all_three_runs"
        elif easy_entropy is not None and (calibration.get("entropy_score") is None or float(calibration["entropy_score"]) - float(easy_entropy) <= 25.0):
            candidate["failure_reason"] = "entropy_gap_not_strictly_greater_than_25"
        else:
            candidate["accepted"] = True
            candidate["failure_reason"] = "pass"
        return candidate

    def _publish(self, candidate: dict[str, Any], stem: str) -> dict[str, str]:
        source = Path(candidate["artifact_dir"])
        return {source_name: _atomic_publish(source / source_name, self.image_dir / f"{stem}.{suffix}") for source_name, suffix in (("image.png", "png"), ("layout.json", "layout.json"), ("target_mask.png", "target_mask.png"), ("occluder_mask.png", "occluder_mask.png"))}

    def _public_artifacts_exist(self, public: dict[str, Any]) -> bool:
        for key in ("image", "layout", "target_mask", "occluder_mask"):
            raw = public.get(key)
            if not isinstance(raw, str) or not (self.output_path.parent / raw).is_file():
                return False
        return True

    def _branch(self, combo: dict[str, Any], branch: str, target: str) -> dict[str, Any] | None:
        count_easy = self.easy_count if branch == "conflict" else self.images_per_difficulty
        count_hard = self.hard_count if branch == "conflict" else self.images_per_difficulty
        branch_state = self.state["items"].setdefault(str(combo["id"]), {}).setdefault(branch, {"easy": {}, "hard": {}})
        accepted_easy: list[dict[str, Any]] = []
        accepted_hard: list[dict[str, Any]] = []
        easy_layouts: dict[int, dict[str, Any]] = {}
        easy_entropy: dict[int, float] = {}
        question = self.module.QUESTION_TEMPLATE.format(shape=combo["shape"])
        for variant_index in range(1, count_easy + 1):
            saved_easy = branch_state.get("easy", {}).get(str(variant_index))
            if isinstance(saved_easy, dict) and saved_easy.get("accepted") and self._public_artifacts_exist(saved_easy.get("public", {})):
                accepted_easy.append(saved_easy["public"])
                layout_path = self.output_path.parent / saved_easy["public"]["layout"]
                easy_layouts[variant_index] = json.loads(layout_path.read_text(encoding="utf-8")) if layout_path.is_file() else saved_easy["layout"]
                easy_entropy[variant_index] = float(saved_easy["public"]["calibration"]["entropy_score"])
                continue
            accepted = None
            for attempt in range(1, self.module.EASY_MAX_ATTEMPTS + 1):
                seed = self.module.derive_seed(self.seed, combo["id"], branch, "easy", variant_index, attempt)
                layout = self.module.build_easy_layout(seed, branch, str(combo["shape"]), target)
                rotation, effective = self._rotation(combo["shape"], seed)
                self._apply_target_rotation(layout, rotation)
                self._apply_distractor_rotations(layout, seed)
                work = Path(tempfile.mkdtemp(prefix=f"v2-{combo['id']}-{branch}-easy-{variant_index}-"))
                candidate = self._candidate(layout, work, question, target, {"item_id": str(combo["id"]), "branch": branch, "difficulty": "easy", "variant_index": variant_index, "attempt": attempt, "siblings": [self.output_path.parent / item["image"] for item in accepted_easy]}, None)
                self.state["items"][str(combo["id"])].setdefault(branch, {}).setdefault("easy_attempts", []).append({key: value for key, value in candidate.items() if key != "artifact_dir"})
                self._persist()
                if candidate.get("accepted"):
                    stem = f"{combo['id']}_{branch}_easy({variant_index})"
                    published = self._publish(candidate, stem)
                    public = {"variant_index": variant_index, "seed": seed, "rotation": rotation, "rotation_is_diversity": bool(effective), "image": os.path.relpath(published["image.png"], self.output_path.parent), "layout": os.path.relpath(published["layout.json"], self.output_path.parent), "target_mask": os.path.relpath(published["target_mask.png"], self.output_path.parent), "occluder_mask": os.path.relpath(published["occluder_mask.png"], self.output_path.parent), "calibration": candidate["calibration"], "entropy_check": {"entropy_score": candidate["calibration"].get("entropy_score"), "passed": bool(candidate["calibration"].get("all_correct"))}, "similarity_check": candidate["similarity_check"], "artifact_sha256": {key: sha256_file(Path(value)) for key, value in published.items()}}
                    accepted = public
                    accepted_easy.append(public)
                    easy_layouts[variant_index] = layout
                    easy_entropy[variant_index] = float(candidate["calibration"]["entropy_score"])
                    branch_state.setdefault("easy", {})[str(variant_index)] = {"accepted": True, "public": public, "layout": layout}
                    self._persist()
                    break
            if accepted is None:
                branch_state["status"] = "partial"
                self._persist()
                return None
        for variant_index in range(1, count_hard + 1):
            saved_hard = branch_state.get("hard", {}).get(str(variant_index))
            if isinstance(saved_hard, dict) and saved_hard.get("accepted") and self._public_artifacts_exist(saved_hard.get("public", {})):
                accepted_hard.append(saved_hard["public"])
                continue
            easy_layout = easy_layouts[variant_index]
            accepted = None
            for attempt in range(1, self.module.HARD_MAX_ATTEMPTS + 1):
                print(
                    f"[{combo['id']}] {branch} hard attempt {attempt}/{self.module.HARD_MAX_ATTEMPTS}",
                    flush=True,
                )
                seed = self.module.derive_seed(self.seed, combo["id"], branch, "hard", variant_index, attempt)
                layout = self.module.build_hard_layout(seed, easy_layout)
                # Hard target rotation is inherited from its corresponding easy.
                target_obj = next(obj for obj in layout["objects"] if obj.get("role") == "target")
                target_obj["rotation"] = next(obj["rotation"] for obj in easy_layout["objects"] if obj.get("role") == "target")
                layout["target_geometry"]["rotation"] = target_obj["rotation"]
                self._apply_distractor_rotations(layout, seed)
                work = Path(tempfile.mkdtemp(prefix=f"v2-{combo['id']}-{branch}-hard-{variant_index}-"))
                candidate = self._candidate(layout, work, question, target, {"item_id": str(combo["id"]), "branch": branch, "difficulty": "hard", "variant_index": variant_index, "attempt": attempt, "siblings": [self.output_path.parent / item["image"] for item in accepted_hard]}, easy_entropy[variant_index])
                self.state["items"][str(combo["id"])].setdefault(branch, {}).setdefault("hard_attempts", []).append({key: value for key, value in candidate.items() if key != "artifact_dir"})
                self._persist()
                if candidate.get("accepted"):
                    stem = f"{combo['id']}_{branch}_hard({variant_index})"
                    published = self._publish(candidate, stem)
                    gap = float(candidate["calibration"]["entropy_score"]) - easy_entropy[variant_index]
                    print(
                        f"[{combo['id']}] {branch} hard accepted on attempt {attempt} "
                        f"(gap={gap:.2f})",
                        flush=True,
                    )
                    public = {"variant_index": variant_index, "base_easy_variant_index": variant_index, "seed": seed, "rotation": target_obj["rotation"], "rotation_is_diversity": bool(target_obj["rotation"]), "image": os.path.relpath(published["image.png"], self.output_path.parent), "layout": os.path.relpath(published["layout.json"], self.output_path.parent), "target_mask": os.path.relpath(published["target_mask.png"], self.output_path.parent), "occluder_mask": os.path.relpath(published["occluder_mask.png"], self.output_path.parent), "calibration": candidate["calibration"], "entropy_check": {"easy_entropy_score": easy_entropy[variant_index], "hard_entropy_score": candidate["calibration"]["entropy_score"], "entropy_gap": gap, "passed": gap > 25.0}, "similarity_check": candidate["similarity_check"], "artifact_sha256": {key: sha256_file(Path(value)) for key, value in published.items()}}
                    accepted = public
                    accepted_hard.append(public)
                    branch_state.setdefault("hard", {})[str(variant_index)] = {"accepted": True, "public": public}
                    self._persist()
                    break
                gap_note = ""
                score = candidate.get("entropy_score")
                if score is not None:
                    gap_note = f" gap={float(score) - easy_entropy[variant_index]:.2f}"
                print(
                    f"[{combo['id']}] {branch} hard attempt {attempt} failed: "
                    f"{candidate.get('failure_reason')}{gap_note}",
                    flush=True,
                )
            if accepted is None:
                print(
                    f"[{combo['id']}] {branch} hard failed after {self.module.HARD_MAX_ATTEMPTS} attempts",
                    flush=True,
                )
                branch_state["status"] = "partial"
                self._persist()
                return None
        branch_state["status"] = "completed"
        branch_state["easy"] = {str(item["variant_index"]): {"accepted": True, "public": item} for item in accepted_easy}
        branch_state["hard"] = {str(item["variant_index"]): {"accepted": True, "public": item} for item in accepted_hard}
        return {"easy": accepted_easy, "hard": accepted_hard}

    def run(self) -> dict[str, Any]:
        existing = {str(item.get("id")) for item in self.output.get("items", [])}
        for combo in self.manifest["combinations"]:
            item_id = str(combo["id"])
            if item_id in existing:
                continue
            branches: dict[str, Any] = {}
            failed = False
            for branch in self.branches:
                target = combo["conflict_color"] if branch == "conflict" else combo["text_color"]
                result = self._branch(combo, branch, target)
                if result is None:
                    failed = True
                    break
                branches[branch] = result
            if failed:
                self.state["failures"].append({"item_id": item_id, "reason": "variant_target_not_reached"})
                self._persist()
                continue
            image_clue = {branch: value for branch, value in branches.items()}
            image_clue.update({"irr": self.irr_path, "null": self.null_path})
            item = {"id": item_id, "question": {"text": self.module.QUESTION_TEMPLATE.format(shape=combo["shape"])}, "answer": combo["text_color"], "text_ans": combo["text_color"], "candidate_colors": list(COLOR_SET_A), "conflict_ans": combo["conflict_color"], "conflict_answer": combo["conflict_color"], "image_clue": image_clue, "irr": self.irr_path, "null": self.null_path}
            self.output["items"].append(item)
            self._persist()
        if self.state.get("failures"):
            failures = ", ".join(str(entry.get("item_id")) for entry in self.state["failures"][-10:])
            raise RuntimeError(
                "Image dataset generation incomplete; no best-effort publication is allowed "
                f"(failed items: {failures})"
            )
        return self.output
