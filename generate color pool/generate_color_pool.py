#!/usr/bin/env python3
"""Build stable five-bin text-prior pools for two colour vocabularies.

The local Qwen model is intentionally serial.  Only remote DeepSeek generator
and analyzer requests are concurrent.  To protect the disk, all artifacts are
buffered and atomically checkpointed once after each colour is fully handled.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from confidence_analysis import ConfidenceAnalyzer  # noqa: E402


COLOR_SET_A = [
    "red", "orange", "yellow", "green", "blue", "cyan",
    "purple", "pink", "brown", "white", "black", "gray",
]
COLOR_SET_B = [
    "maroon", "lime", "navy", "teal", "olive", "magenta",
    "silver", "gold", "beige", "coral", "violet", "turquoise",
]
ALL_COLORS = COLOR_SET_A + COLOR_SET_B

BIN_RANGES = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
DEFAULT_BATCH_SIZE_BY_BIN = {0: 30, 1: 30, 2: 20, 3: 10, 4: 20}
LOW_BIN_DIFFICULTIES = ("multi_step_reasoning", "not_exclusion", "pure_hard")
PRIOR_FIELD_NAMES = {"text_clue", "selected_text_priors", "text_prior", "prior", "clue"}
SYNTHETIC_SHAPES = [
    "triangle", "hexagon", "star", "diamond", "pentagon", "oval",
    "trapezoid", "crescent", "cross", "parallelogram", "semicircle",
]
SHAPE_WORDS = {
    "rectangle", "heart", "octagon", "square", "circle", "triangle",
    "hexagon", "star", "diamond", "pentagon", "oval", "trapezoid",
    "crescent", "cross", "parallelogram", "semicircle", "shape",
}

STAGE1_TEXT_ANSWER_PROMPT = """Question:
{question}

Text clue:
{text_clue}

Answer the question using only the question and text clue above.

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, or any additional text.

Output exactly:

**Answer**: <your answer>

Do not include any additional text."""

BIN_STRATEGIES = {
    0: "The target must be only a very weak best guess. Produce exactly 10 multi_step_reasoning, 10 not_exclusion, and 10 pure_hard candidates.",
    1: "Give the target a slight advantage while retaining several credible alternatives. Produce exactly 10 multi_step_reasoning, 10 not_exclusion, and 10 pure_hard candidates.",
    2: "Make the target the strongest candidate while retaining one or two reasonable alternatives.",
    3: "Clearly support the target while preserving a small amount of reasonable uncertainty.",
    4: "Use strongly diagnostic common-knowledge clues that point to the target colour.",
}

GENERATOR_PROMPT = """You are the generation agent responsible only for confidence bin
"{target_bin}" for target color "{target_color}".

Candidate color set:
{choice_colors}

Generate exactly {batch_size} shape-independent text clues.

The clues will be tested on three different shape questions. Never mention
a specific shape.

Every clue must make "{target_color}" the final answer while keeping the
model's soft confidence inside {target_bin}.

Do not mention images, AI, models, prompts, confidence classes, probability
values, tests, datasets, or output evaluation.

Do not create synonym-only variations.

Previously accepted clues:
{accepted_priors}

Previously rejected clues and measured results:
{rejected_results}

Use the following bin-specific strategy:
{bin_strategy}

Return JSON only:

{{
  "color": "{target_color}",
  "bin_id": {bin_id},
  "target_bin": "{target_bin}",
  "candidates": [
    {{
      "candidate_id": "unique id",
      "strategy_family": "strategy name",
      "difficulty_type": "difficulty type",
      "text_clue": "shape-independent clue"
    }}
  ]
}}"""

ANALYZER_PROMPT = """You are the analysis agent paired exclusively with the generator for
target color "{target_color}" and confidence bin "{target_bin}".

Do not analyze or modify any other confidence bin.

Current generator prompt:
{current_generator_prompt}

Accepted candidates:
{accepted_results}

Rejected candidates and measured results:
{rejected_results}

Summary statistics:
{statistics}

Analyze:

1. Which strategies preserve the target answer?
2. Which strategies place soft confidence inside the target bin?
3. Which strategies make confidence too high?
4. Which strategies make confidence too low?
5. Which strategies cause the answer to change across the three questions?
6. Which strategies produce soft confidence range greater than or equal to 0.1?
7. Which strategies create duplicates or shape-dependent clues?

The next prompt must continue to request exactly {batch_size} candidates.

If answers are wrong, add a subtle target-specific cue.
If confidence is too high, weaken evidence and preserve more alternatives.
If confidence is too low, add one useful property or remove one major competitor.
If confidence is unstable, prefer evidence independent of wording and shape.

Return JSON only:

{{
  "target_color": "{target_color}",
  "bin_id": {bin_id},
  "target_bin": "{target_bin}",
  "analysis": {{
    "successful_strategies": [],
    "strategies_to_reduce": [],
    "new_strategies_to_try": [],
    "wrong_answer_causes": [],
    "confidence_instability_causes": [],
    "duplicate_causes": []
  }},
  "revised_generator_prompt": "complete prompt for the paired generator's next round"
}}"""


@dataclass
class QuestionCase:
    question_id: str
    question: str
    target_color: str
    shape: str
    choice_colors: list[str]
    source_question_id: str
    question_source: str
    template_rewritten: bool


@dataclass
class Candidate:
    candidate_id: str
    strategy_family: str
    difficulty_type: str
    text_clue: str


@dataclass
class AgentState:
    color: str
    bin_id: int
    current_generator_prompt: str
    accepted_priors: list[dict[str, Any]] = field(default_factory=list)
    rejected_results: list[dict[str, Any]] = field(default_factory=list)
    round_index: int = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = re.sub(r"\s+", " ", value)
    return re.sub(r"[^\w\s]", "", value)


def normalize_answer(value: str) -> str:
    return normalize_text(value).strip()


def parse_choice_colors(question: str) -> list[str]:
    match = re.search(r"Choose\s+from\s*:\s*(.+?)(?:\.|$)", question, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Question has no 'Choose from:' colour set: {question!r}")
    colors = [part.strip().casefold() for part in match.group(1).split(",") if part.strip()]
    if not colors:
        raise ValueError(f"Question has an empty colour set: {question!r}")
    return colors


def extract_shape(question: str) -> str:
    match = re.search(r"color\s+of\s+(?:the\s+)?(.+?)\?", question, flags=re.IGNORECASE)
    return normalize_text(match.group(1)) if match else "unknown"


def replace_shape(question: str, new_shape: str) -> str:
    pattern = re.compile(
        r"(color\s+of\s+(?:the\s+)?)(.+?)(\?\s*Choose\s+from\s*:)",
        flags=re.IGNORECASE,
    )
    rewritten, count = pattern.subn(lambda m: f"{m.group(1)}{new_shape}{m.group(3)}", question, count=1)
    if count != 1:
        raise ValueError(f"Cannot replace shape in question: {question!r}")
    return rewritten


def in_bin(value: float, bin_id: int) -> bool:
    low, high = BIN_RANGES[bin_id]
    return low <= value <= high if bin_id == 4 else low <= value < high


def bin_for_value(value: float) -> int | None:
    for bin_id in range(5):
        if in_bin(value, bin_id):
            return bin_id
    return None


def bin_label(bin_id: int) -> str:
    low, high = BIN_RANGES[bin_id]
    close = "]" if bin_id == 4 else ")"
    return f"[{low:.1f}, {high:.1f}{close}"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_inference_class(path: Path) -> type[Any]:
    specification = importlib.util.spec_from_file_location("qwen_vl_inference", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load inference module from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module.QwenVLInference


def iter_dataset_items(dataset: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(dataset, list):
        raise ValueError("Dataset root must be an array")
    for group in dataset:
        if not isinstance(group, dict):
            continue
        items = group.get("items")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    yield item


def question_text(item: dict[str, Any]) -> str:
    question = item.get("question", "")
    if isinstance(question, dict):
        question = question.get("text", "")
    return str(question)


def build_question_bank(dataset: Any, seed: int) -> dict[str, list[QuestionCase]]:
    records: list[dict[str, Any]] = []
    for item in iter_dataset_items(dataset):
        question = question_text(item)
        try:
            choices = parse_choice_colors(question)
        except ValueError:
            continue
        records.append(
            {
                "id": str(item.get("id", "unknown")),
                "question": question,
                "answer": normalize_answer(str(item.get("answer", item.get("text_ans", "")))),
                "shape": extract_shape(question),
                "choices": choices,
            }
        )

    bank: dict[str, list[QuestionCase]] = {}
    for color in ALL_COLORS:
        expected_choices = COLOR_SET_A if color in COLOR_SET_A else COLOR_SET_B
        same_set = [record for record in records if record["choices"] == expected_choices]
        real = [record for record in same_set if record["answer"] == color]
        rng = random.Random(f"{seed}:{color}:questions")
        rng.shuffle(real)
        selected: list[QuestionCase] = []
        seen_questions: set[str] = set()
        seen_shapes: set[str] = set()
        for record in real:
            if record["question"] in seen_questions or record["shape"] in seen_shapes:
                continue
            selected.append(
                QuestionCase(
                    question_id=record["id"],
                    question=record["question"],
                    target_color=color,
                    shape=record["shape"],
                    choice_colors=list(record["choices"]),
                    source_question_id=record["id"],
                    question_source="existing_dataset",
                    template_rewritten=False,
                )
            )
            seen_questions.add(record["question"])
            seen_shapes.add(record["shape"])
            if len(selected) == 3:
                break

        if len(selected) < 3 and not same_set:
            raise ValueError(f"No real question template exists for colour set containing {color}")
        template_index = 0
        for shape in SYNTHETIC_SHAPES:
            if len(selected) == 3:
                break
            if shape in seen_shapes:
                continue
            template = same_set[template_index % len(same_set)]
            template_index += 1
            rewritten = replace_shape(template["question"], shape)
            if rewritten in seen_questions:
                continue
            selected.append(
                QuestionCase(
                    question_id=f"synthetic-{color}-{len(selected) + 1}",
                    question=rewritten,
                    target_color=color,
                    shape=shape,
                    choice_colors=list(template["choices"]),
                    source_question_id=template["id"],
                    question_source="template_rewrite",
                    template_rewritten=True,
                )
            )
            seen_questions.add(rewritten)
            seen_shapes.add(shape)
        if len(selected) != 3:
            raise ValueError(f"Could not construct three distinct questions for {color}")
        bank[color] = selected
    return bank


def _recursive_prior_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"text_clue", "text_prior", "prior", "clue"} and isinstance(nested, str):
                if nested.strip():
                    yield nested.strip()
            else:
                yield from _recursive_prior_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _recursive_prior_strings(nested)


def extract_existing_priors(dataset: Any) -> dict[str, list[str]]:
    priors: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for item in iter_dataset_items(dataset):
        color = normalize_answer(str(item.get("answer", item.get("text_ans", ""))))
        if color not in ALL_COLORS:
            continue
        for clue in _recursive_prior_strings(item):
            key = normalize_text(clue)
            if key and key not in seen[color]:
                seen[color].add(key)
                priors[color].append(clue)
    return dict(priors)


def parse_stage1_answer(raw_output: str) -> str:
    match = re.search(r"\*\*Answer\*\*\s*:\s*([^\r\n]+)", raw_output, flags=re.IGNORECASE)
    return normalize_answer(match.group(1) if match else raw_output)


class LocalTester:
    def __init__(self, inference: Any, stability_threshold: float):
        self.inference = inference
        self.confidence = ConfidenceAnalyzer(inference)
        self.stability_threshold = stability_threshold
        self.call_count = 0

    def _single_test(self, case: QuestionCase, text_clue: str, target_color: str) -> dict[str, Any]:
        stage1_prompt = STAGE1_TEXT_ANSWER_PROMPT.format(question=case.question, text_clue=text_clue)
        self.call_count += 1
        stage1 = self.inference.generate(prompt=stage1_prompt, image_path=None, max_new_tokens=24)
        model_answer = parse_stage1_answer(stage1.response)
        result: dict[str, Any] = {
            "question": asdict(case),
            "stage1_raw_output": stage1.response,
            "normalized_answer": model_answer,
            "answer_correct": model_answer == target_color,
        }
        if model_answer != target_color:
            result["stage2_skipped"] = True
            return result
        confidence = self.confidence.analyze(case.question, text_clue, model_answer)
        confidence_data = asdict(confidence)
        # Avoid repeating large/constant diagnostics in every pool entry.
        confidence_data.pop("rendered_prompt", None)
        confidence_data.pop("class_token_variants", None)
        result.update(confidence_data)
        result["stage2_skipped"] = False
        return result

    def test_prior(
        self,
        text_clue: str,
        target_color: str,
        cases: list[QuestionCase],
        source: str,
        generation_round: int,
        target_bin_id: int | None,
        candidate: Candidate | None = None,
    ) -> dict[str, Any]:
        tests: list[dict[str, Any]] = []
        first = self._single_test(cases[0], text_clue, target_color)
        tests.append(first)
        if not first["answer_correct"] or "soft_confidence" not in first:
            return self._final_result(
                text_clue, source, generation_round, target_bin_id, tests,
                candidate, False, "first_answer_incorrect_or_no_soft_confidence",
            )
        first_bin = bin_for_value(float(first["soft_confidence"]))
        effective_bin = first_bin if source == "existing_dataset" else target_bin_id
        if effective_bin is None or (source != "existing_dataset" and not in_bin(float(first["soft_confidence"]), effective_bin)):
            return self._final_result(
                text_clue, source, generation_round, effective_bin, tests,
                candidate, False, "first_soft_confidence_outside_target_bin",
            )
        for case in cases[1:]:
            tests.append(self._single_test(case, text_clue, target_color))

        answers = [test.get("normalized_answer", "") for test in tests]
        soft_values = [float(test["soft_confidence"]) for test in tests if "soft_confidence" in test]
        answers_unchanged = (
            len(tests) == 3
            and len(set(answers)) == 1
            and answers[0] == target_color
        )
        if len(soft_values) != 3:
            return self._final_result(
                text_clue, source, generation_round, effective_bin, tests,
                candidate, False, "answer_changed_or_stage2_failed",
            )
        soft_range = max(soft_values) - min(soft_values)
        soft_mean = sum(soft_values) / 3.0
        stable = answers_unchanged and soft_range < self.stability_threshold
        if not answers_unchanged:
            reason = "answers_changed"
        elif not soft_range < self.stability_threshold:
            reason = "soft_confidence_unstable"
        elif not in_bin(soft_mean, int(effective_bin)):
            reason = "soft_mean_outside_target_bin"
        else:
            reason = "accepted"
        accepted = stable and in_bin(soft_mean, int(effective_bin))
        final = self._final_result(
            text_clue, source, generation_round, effective_bin, tests,
            candidate, accepted, reason,
        )
        final["answers_unchanged"] = answers_unchanged
        final["soft_range"] = soft_range
        final["soft_mean"] = soft_mean
        final["stable"] = stable
        return final

    @staticmethod
    def _final_result(
        text_clue: str,
        source: str,
        generation_round: int,
        bin_id: int | None,
        tests: list[dict[str, Any]],
        candidate: Candidate | None,
        accepted: bool,
        reason: str,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "text_clue": text_clue,
            "source": source,
            "generation_round": generation_round,
            "hidden_state_collected": False,
            "bin_id": bin_id,
            "accepted": accepted,
            "rejection_reason": None if accepted else reason,
            "test_results": tests,
        }
        if candidate is not None:
            value.update(
                {
                    "candidate_id": candidate.candidate_id,
                    "strategy_family": candidate.strategy_family,
                    "difficulty_type": candidate.difficulty_type,
                }
            )
        return value


class DeepSeekClient:
    def __init__(self, workers: int):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("DeepSeek generation requires the 'openai' Python package") from exc
        api_key = "70753601968e6440544540e4cc55bd0f596bd4cbf8655df21091803a9b32b28f"
        if not api_key:
            raise RuntimeError("Set CSTCLOUD_API_KEY or OPENAI_API_KEY for DeepSeek generation")
        self.client = OpenAI(api_key=api_key, base_url="https://uni-api.cstcloud.cn/v1")
        self.workers = workers
        self.generator_calls = 0
        self.analyzer_calls = 0
        self._counter_lock = threading.Lock()

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("DeepSeek response JSON must be an object")
        return parsed

    def call(self, prompt: str, kind: str, retries: int = 3) -> dict[str, Any]:
        with self._counter_lock:
            if kind == "generator":
                self.generator_calls += 1
            else:
                self.analyzer_calls += 1
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model="DeepSeek-V4-Flash",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.8 if kind == "generator" else 0.2,
                )
                content = response.choices[0].message.content or ""
                return self._parse_json(content)
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"DeepSeek {kind} failed after {retries} attempts: {last_error}")


def empty_color_entry(color: str, choice_colors: list[str]) -> dict[str, Any]:
    return {
        "color": color,
        "choice_colors": choice_colors,
        "prior_levels": [
            {
                "bin_id": bin_id,
                "target_bin": bin_label(bin_id),
                "priors": [],
                "complete": False,
            }
            for bin_id in range(5)
        ],
        "tested_prior_keys": [],
        "complete": False,
    }


def ensure_pool(pool: Any, question_bank: dict[str, list[QuestionCase]]) -> list[dict[str, Any]]:
    if pool is None:
        pool = []
    if not isinstance(pool, list):
        raise ValueError("Output pool root must be an array")
    by_color = {entry.get("color"): entry for entry in pool if isinstance(entry, dict)}
    result: list[dict[str, Any]] = []
    for color in ALL_COLORS:
        entry = by_color.get(color)
        if entry is None:
            entry = empty_color_entry(color, question_bank[color][0].choice_colors)
        if not isinstance(entry.get("prior_levels"), list):
            raise ValueError(f"prior_levels for {color} must be an array")
        entry.setdefault("tested_prior_keys", [])
        entry.setdefault("choice_colors", question_bank[color][0].choice_colors)
        result.append(entry)
    for color, entry in by_color.items():
        if color not in ALL_COLORS:
            result.append(entry)
    return result


def level_for(entry: dict[str, Any], bin_id: int) -> dict[str, Any]:
    for level in entry["prior_levels"]:
        if int(level.get("bin_id", -1)) == bin_id:
            level.setdefault("priors", [])
            return level
    level = {"bin_id": bin_id, "target_bin": bin_label(bin_id), "priors": [], "complete": False}
    entry["prior_levels"].append(level)
    entry["prior_levels"].sort(key=lambda item: int(item["bin_id"]))
    return level


def accepted_texts(entry: dict[str, Any]) -> list[str]:
    return [
        str(prior.get("text_clue", ""))
        for level in entry["prior_levels"]
        for prior in level.get("priors", [])
        if prior.get("text_clue")
    ]


def is_near_duplicate(text: str, others: Iterable[str], threshold: float) -> bool:
    normalized = normalize_text(text)
    for other in others:
        candidate = normalize_text(other)
        if normalized == candidate or SequenceMatcher(None, normalized, candidate).ratio() >= threshold:
            return True
    return False


def validate_generator_response(
    payload: dict[str, Any],
    state: AgentState,
    batch_size: int,
    near_duplicate_threshold: float,
    tested_keys: set[str],
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    if normalize_answer(str(payload.get("color", ""))) != state.color or int(payload.get("bin_id", -1)) != state.bin_id:
        return [], [{"rejection_reason": "generator_identity_mismatch", "payload": payload}]
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != batch_size:
        return [], [{"rejection_reason": "wrong_candidate_count", "actual": len(raw_candidates) if isinstance(raw_candidates, list) else None}]
    if state.bin_id in (0, 1):
        counts = Counter(str(item.get("difficulty_type", "")) for item in raw_candidates if isinstance(item, dict))
        if any(counts[name] != 10 for name in LOW_BIN_DIFFICULTIES):
            return [], [{"rejection_reason": "wrong_low_bin_difficulty_quota", "counts": dict(counts)}]

    candidates: list[Candidate] = []
    existing = [prior["text_clue"] for prior in state.accepted_priors if prior.get("text_clue")]
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            rejected.append({"rejection_reason": "candidate_not_object", "candidate_index": index})
            continue
        clue = str(item.get("text_clue", "")).strip()
        lowered = normalize_text(clue)
        if not clue:
            rejected.append({"rejection_reason": "empty_clue", "candidate_index": index})
            continue
        if lowered in tested_keys:
            rejected.append({"text_clue": clue, "rejection_reason": "already_tested"})
            continue
        if any(re.search(rf"\b{re.escape(shape)}\b", lowered) for shape in SHAPE_WORDS):
            rejected.append({"text_clue": clue, "rejection_reason": "shape_dependent"})
            continue
        if is_near_duplicate(clue, existing + [candidate.text_clue for candidate in candidates], near_duplicate_threshold):
            rejected.append({"text_clue": clue, "rejection_reason": "duplicate_or_near_duplicate"})
            continue
        candidates.append(
            Candidate(
                candidate_id=str(item.get("candidate_id", f"{state.color}-{state.bin_id}-{state.round_index}-{index}")),
                strategy_family=str(item.get("strategy_family", "unspecified")),
                difficulty_type=str(item.get("difficulty_type", "unspecified")),
                text_clue=clue,
            )
        )
    return candidates, rejected


def compact_json(value: Any, limit: int = 12000) -> str:
    rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return rendered if len(rendered) <= limit else rendered[-limit:]


def initial_generator_prompt(state: AgentState, batch_size: int, choices: list[str]) -> str:
    return GENERATOR_PROMPT.format(
        target_bin=bin_label(state.bin_id),
        target_color=state.color,
        choice_colors=", ".join(choices),
        batch_size=batch_size,
        accepted_priors=compact_json([p.get("text_clue") for p in state.accepted_priors]),
        rejected_results=compact_json(state.rejected_results),
        bin_strategy=BIN_STRATEGIES[state.bin_id],
        bin_id=state.bin_id,
    )


def summarize_state(state: AgentState) -> dict[str, Any]:
    reasons = Counter(str(result.get("rejection_reason", "accepted")) for result in state.rejected_results)
    strategy_total = Counter()
    strategy_pass = Counter()
    difficulty_total = Counter()
    difficulty_pass = Counter()
    soft_values: list[float] = []
    wrong_answers = Counter()
    for result in state.rejected_results + state.accepted_priors:
        strategy = str(result.get("strategy_family", "unspecified"))
        difficulty = str(result.get("difficulty_type", "unspecified"))
        strategy_total[strategy] += 1
        difficulty_total[difficulty] += 1
        if result.get("accepted"):
            strategy_pass[strategy] += 1
            difficulty_pass[difficulty] += 1
        for test in result.get("test_results", []):
            if "soft_confidence" in test:
                soft_values.append(float(test["soft_confidence"]))
            if test.get("normalized_answer") and not test.get("answer_correct"):
                wrong_answers[str(test["normalized_answer"])] += 1
    return {
        "generated_count": len(state.rejected_results) + len(state.accepted_priors),
        "valid_candidate_count": sum(1 for r in state.rejected_results + state.accepted_priors if r.get("text_clue")),
        "duplicate_count": reasons["duplicate_or_near_duplicate"],
        "first_answer_wrong_count": reasons["first_answer_incorrect_or_no_soft_confidence"],
        "confidence_too_low_count": reasons["confidence_too_low"],
        "confidence_too_high_count": reasons["confidence_too_high"],
        "answer_changed_count": reasons["answers_changed"] + reasons["answer_changed_or_stage2_failed"],
        "soft_unstable_count": reasons["soft_confidence_unstable"],
        "accepted_count": len(state.accepted_priors),
        "strategy_family_pass_rate": {
            name: strategy_pass[name] / total for name, total in strategy_total.items()
        },
        "difficulty_type_pass_rate": {
            name: difficulty_pass[name] / total for name, total in difficulty_total.items()
        },
        "soft_confidence_mean": sum(soft_values) / len(soft_values) if soft_values else None,
        "soft_confidence_range": max(soft_values) - min(soft_values) if soft_values else None,
        "common_wrong_colors": dict(wrong_answers.most_common(10)),
        "rejection_reasons": dict(reasons),
    }


class PoolBuilder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.input_path = Path(args.input).resolve()
        self.output_path = Path(args.output).resolve()
        self.artifact_dir = ROOT_DIR / "generate color pool" / "output"
        self.dataset = load_json(self.input_path)
        self.question_bank = build_question_bank(self.dataset, args.seed)
        existing_pool = load_json(self.output_path) if self.output_path.exists() else None
        self.pool = ensure_pool(existing_pool, self.question_bank)
        self.pool_by_color = {entry["color"]: entry for entry in self.pool}
        self.events = self._load_events()
        self.prompt_history = self._load_optional_json("color_prior_prompt_history.json", [])
        self.log_lines = self._load_log()
        self.report = self._load_optional_json("color_prior_generation_report.json", {})
        self.report.update(
            {
                "started_at": utc_now(),
                "input": str(self.input_path),
                "output": str(self.output_path),
                "selected_colors": args.selected_colors,
                "checkpoint_policy": "once_per_completed_color",
                "find_enabled": args.find,
            }
        )
        self.inference: Any = None
        self.tester: LocalTester | None = None
        self.deepseek: DeepSeekClient | None = None

    def _load_optional_json(self, name: str, default: Any) -> Any:
        path = self.artifact_dir / name
        return load_json(path) if path.exists() else default

    def _load_events(self) -> list[dict[str, Any]]:
        path = self.artifact_dir / "color_prior_generation_events.jsonl"
        if not path.exists():
            return []
        values = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    values.append(json.loads(line))
        return values

    def _load_log(self) -> list[str]:
        path = self.artifact_dir / "color_prior_generation.log"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def event(self, event_type: str, **details: Any) -> None:
        value = {"timestamp": utc_now(), "event": event_type, **details}
        self.events.append(value)
        self.log_lines.append(f"{value['timestamp']} {event_type} {compact_json(details, 2000)}")

    def load_local_model(self) -> None:
        inference_class = load_inference_class(ROOT_DIR / "qwen-2.5-vl" / "inference.py")
        self.inference = inference_class(model_path=str((ROOT_DIR / "qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct").resolve()))
        self.tester = LocalTester(self.inference, self.args.stability_threshold)

    def checkpoint_color(self, color: str) -> None:
        entry = self.pool_by_color[color]
        for level in entry["prior_levels"]:
            level["complete"] = len(level.get("priors", [])) >= self.args.target_per_bin
        entry["complete"] = all(level["complete"] for level in entry["prior_levels"])
        entry["checkpointed_at"] = utc_now()
        self.report["updated_at"] = utc_now()
        self.report["colors"] = {
            item["color"]: {
                "complete": bool(item.get("complete")),
                "bin_counts": {
                    str(level["bin_id"]): len(level.get("priors", []))
                    for level in item["prior_levels"]
                },
            }
            for item in self.pool
            if item.get("color") in ALL_COLORS
        }
        self.report["local_stage1_calls"] = self.tester.call_count if self.tester else 0
        self.report["deepseek_generator_calls"] = self.deepseek.generator_calls if self.deepseek else 0
        self.report["deepseek_analyzer_calls"] = self.deepseek.analyzer_calls if self.deepseek else 0
        self.report["incomplete"] = [
            {"color": item["color"], "bins": [level["bin_id"] for level in item["prior_levels"] if not level["complete"]]}
            for item in self.pool
            if item.get("color") in self.args.selected_colors and not item.get("complete")
        ]
        atomic_write_json(self.output_path, self.pool)
        atomic_write_json(self.artifact_dir / "color_prior_generation_report.json", self.report)
        atomic_write_text(
            self.artifact_dir / "color_prior_generation_events.jsonl",
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in self.events),
        )
        atomic_write_text(
            self.artifact_dir / "color_prior_generation.log",
            "\n".join(self.log_lines) + ("\n" if self.log_lines else ""),
        )
        atomic_write_json(
            self.artifact_dir / "color_prior_prompt_history.json",
            self.prompt_history,
        )

    def run_find(self) -> None:
        assert self.tester is not None
        existing = extract_existing_priors(self.dataset)
        self.event("find_started", generator_calls=0, analyzer_calls=0)
        for color in self.args.selected_colors:
            entry = self.pool_by_color[color]
            tested = set(entry.get("tested_prior_keys", []))
            already_accepted = accepted_texts(entry)
            for clue in existing.get(color, []):
                key = normalize_text(clue)
                if key in tested or is_near_duplicate(clue, already_accepted, 1.0):
                    continue
                result = self.tester.test_prior(
                    clue,
                    color,
                    self.question_bank[color],
                    source="existing_dataset",
                    generation_round=0,
                    target_bin_id=None,
                )
                tested.add(key)
                if result["accepted"]:
                    level_for(entry, int(result["bin_id"]))["priors"].append(result)
                    already_accepted.append(clue)
                self.event("find_prior_tested", color=color, accepted=result["accepted"], reason=result["rejection_reason"])
            entry["tested_prior_keys"] = sorted(tested)
        self.event("find_completed", generator_calls=0, analyzer_calls=0)

    def _make_states(self, color: str) -> dict[int, AgentState]:
        entry = self.pool_by_color[color]
        histories = [item for item in self.prompt_history if item.get("color") == color]
        states: dict[int, AgentState] = {}
        for bin_id in range(5):
            level = level_for(entry, bin_id)
            latest = max(
                (item for item in histories if int(item.get("bin_id", -1)) == bin_id),
                key=lambda item: int(item.get("round", 0)),
                default=None,
            )
            state = AgentState(
                color=color,
                bin_id=bin_id,
                current_generator_prompt="",
                accepted_priors=level["priors"],
                round_index=(int(latest.get("round", 0)) + 1) if latest else 1,
            )
            batch_size = self.args.batch_sizes[bin_id]
            state.current_generator_prompt = (
                str(latest.get("next_generator_prompt"))
                if latest and latest.get("next_generator_prompt")
                else initial_generator_prompt(state, batch_size, entry["choice_colors"])
            )
            states[bin_id] = state
        return states

    def _generate_one(self, state: AgentState) -> tuple[int, dict[str, Any] | Exception]:
        assert self.deepseek is not None
        try:
            return state.bin_id, self.deepseek.call(state.current_generator_prompt, "generator")
        except Exception as exc:
            return state.bin_id, exc

    def _analyze_one(self, state: AgentState) -> tuple[int, dict[str, Any] | Exception]:
        assert self.deepseek is not None
        statistics = summarize_state(state)
        prompt = ANALYZER_PROMPT.format(
            target_color=state.color,
            target_bin=bin_label(state.bin_id),
            current_generator_prompt=state.current_generator_prompt,
            accepted_results=compact_json(state.accepted_priors),
            rejected_results=compact_json(state.rejected_results),
            statistics=compact_json(statistics),
            batch_size=self.args.batch_sizes[state.bin_id],
            bin_id=state.bin_id,
        )
        try:
            result = self.deepseek.call(prompt, "analyzer", retries=3)
            if normalize_answer(str(result.get("target_color", ""))) != state.color or int(result.get("bin_id", -1)) != state.bin_id:
                raise ValueError("Analyzer returned a different color or bin")
            return state.bin_id, result
        except Exception as exc:
            return state.bin_id, exc

    def generate_color(self, color: str) -> None:
        assert self.tester is not None and self.deepseek is not None
        entry = self.pool_by_color[color]
        states = self._make_states(color)
        for round_index in range(1, self.args.rounds + 1):
            active = [
                state for state in states.values()
                if len(state.accepted_priors) < self.args.target_per_bin
            ]
            if not active:
                break
            for state in active:
                state.round_index = round_index
                state.current_generator_prompt = initial_generator_prompt(
                    state, self.args.batch_sizes[state.bin_id], entry["choice_colors"]
                ) if round_index == 1 and not state.current_generator_prompt else state.current_generator_prompt

            generated: dict[int, dict[str, Any] | Exception] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.args.deepseek_workers, len(active))) as executor:
                for bin_id, result in executor.map(self._generate_one, active):
                    generated[bin_id] = result

            tested_keys = set(entry.get("tested_prior_keys", []))
            for state in active:  # Deliberately serial: one local Qwen call chain at a time.
                payload = generated[state.bin_id]
                if isinstance(payload, Exception):
                    state.rejected_results.append({"rejection_reason": "generator_api_failure", "error": str(payload)})
                    continue
                candidates, validation_rejections = validate_generator_response(
                    payload,
                    state,
                    self.args.batch_sizes[state.bin_id],
                    self.args.near_duplicate_threshold,
                    tested_keys,
                )
                state.rejected_results.extend(validation_rejections)
                for candidate in candidates:
                    if len(state.accepted_priors) >= self.args.target_per_bin:
                        break
                    result = self.tester.test_prior(
                        candidate.text_clue,
                        color,
                        self.question_bank[color],
                        source="deepseek_generated",
                        generation_round=round_index,
                        target_bin_id=state.bin_id,
                        candidate=candidate,
                    )
                    entry["tested_prior_keys"].append(normalize_text(candidate.text_clue))
                    tested_keys.add(normalize_text(candidate.text_clue))
                    if result["accepted"]:
                        state.accepted_priors.append(result)
                    else:
                        soft_values = [
                            float(test["soft_confidence"])
                            for test in result.get("test_results", [])
                            if "soft_confidence" in test
                        ]
                        if soft_values and result["rejection_reason"] in {
                            "first_soft_confidence_outside_target_bin", "soft_mean_outside_target_bin"
                        }:
                            measured = soft_values[0] if len(soft_values) == 1 else sum(soft_values) / len(soft_values)
                            result["rejection_reason"] = (
                                "confidence_too_low" if measured < BIN_RANGES[state.bin_id][0] else "confidence_too_high"
                            )
                        state.rejected_results.append(result)

            analyzable = [state for state in active if len(state.accepted_priors) < self.args.target_per_bin]
            analyzed: dict[int, dict[str, Any] | Exception] = {}
            if analyzable:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.args.deepseek_workers, len(analyzable))) as executor:
                    for bin_id, result in executor.map(self._analyze_one, analyzable):
                        analyzed[bin_id] = result
            for state in active:
                analyzer_output = analyzed.get(state.bin_id)
                next_prompt = state.current_generator_prompt
                if isinstance(analyzer_output, dict):
                    revised = analyzer_output.get("revised_generator_prompt")
                    if isinstance(revised, str) and revised.strip():
                        next_prompt = revised.strip()
                self.prompt_history.append(
                    {
                        "color": color,
                        "bin_id": state.bin_id,
                        "round": round_index,
                        "generator_prompt": state.current_generator_prompt,
                        "analyzer_output": analyzer_output if isinstance(analyzer_output, dict) else {"error": str(analyzer_output)} if analyzer_output else {},
                        "next_generator_prompt": next_prompt,
                    }
                )
                state.current_generator_prompt = next_prompt
            self.event(
                "generation_round_completed",
                color=color,
                round=round_index,
                bin_counts={str(bin_id): len(state.accepted_priors) for bin_id, state in states.items()},
            )

        for bin_id, state in states.items():
            level = level_for(entry, bin_id)
            level["priors"] = state.accepted_priors
            level["statistics"] = summarize_state(state)

    def run(self) -> None:
        self.load_local_model()
        if self.args.find:
            self.run_find()
            if self.deepseek is not None or any(
                event.get("event") in {"generator_called", "analyzer_called"}
                for event in self.events[self.report.get("initial_event_count", len(self.events)):]
            ):
                raise AssertionError("DeepSeek was initialized or called before find completed")

        needs_generation = any(
            len(level_for(self.pool_by_color[color], bin_id)["priors"]) < self.args.target_per_bin
            for color in self.args.selected_colors
            for bin_id in range(5)
        )
        if needs_generation:
            self.deepseek = DeepSeekClient(self.args.deepseek_workers)

        for color in self.args.selected_colors:
            entry = self.pool_by_color[color]
            missing = [
                bin_id for bin_id in range(5)
                if len(level_for(entry, bin_id)["priors"]) < self.args.target_per_bin
            ]
            if missing:
                self.event("color_generation_started", color=color, missing_bins=missing)
                self.generate_color(color)
            self.event("color_completed", color=color)
            self.checkpoint_color(color)


def parse_csv_ints(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if len(values) != 5 or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Exactly five positive batch sizes are required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stable five-bin colour prior pools")
    parser.add_argument("--find", action="store_true", help="Test existing dataset priors before generation")
    parser.add_argument("--after", choices=ALL_COLORS, help="Start after this colour in the global order")
    parser.add_argument("--round", dest="rounds", type=int, default=5, help="Maximum generator/analyzer rounds per colour")
    parser.add_argument("--input", default="/root/autodl-tmp/datasets/dataset.json")
    parser.add_argument("--output", default="datasets/color_prior_pool.json")
    parser.add_argument("--target-per-bin", type=int, default=5)
    parser.add_argument("--bin-batch-sizes", type=parse_csv_ints, default=parse_csv_ints("30,30,20,10,20"))
    parser.add_argument("--deepseek-workers", type=int, default=5)
    parser.add_argument("--colors", help="Comma-separated subset of the 24 supported colours")
    parser.add_argument("--resume", action="store_true", help="Explicit alias for the default incremental-create behavior")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.88)
    parser.add_argument("--stability-threshold", type=float, default=0.1)
    args = parser.parse_args()

    if args.rounds <= 0 or args.target_per_bin <= 0 or args.deepseek_workers <= 0:
        parser.error("--round, --target-per-bin, and --deepseek-workers must be positive")
    if not 0.0 <= args.near_duplicate_threshold <= 1.0:
        parser.error("--near-duplicate-threshold must be between 0 and 1")
    if args.stability_threshold <= 0.0:
        parser.error("--stability-threshold must be positive")
    selected = list(ALL_COLORS)
    if args.after:
        selected = selected[ALL_COLORS.index(args.after) + 1:]
    if args.colors:
        requested = [normalize_answer(part) for part in args.colors.split(",") if part.strip()]
        invalid = [color for color in requested if color not in ALL_COLORS]
        if invalid:
            parser.error(f"Unsupported --colors values: {', '.join(invalid)}")
        requested_set = set(requested)
        selected = [color for color in selected if color in requested_set]
    if not selected:
        parser.error("No colours remain after applying --after and --colors")
    args.selected_colors = selected
    args.batch_sizes = dict(enumerate(args.bin_batch_sizes))
    return args


def main() -> int:
    args = parse_args()
    try:
        builder = PoolBuilder(args)
        builder.run()
    except KeyboardInterrupt:
        print("[WARN] Interrupted; the current colour was not checkpointed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"[INFO] Colour prior pool updated at {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
