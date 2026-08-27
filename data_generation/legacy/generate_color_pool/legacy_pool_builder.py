#!/usr/bin/env python3
"""Legacy confidence PoolBuilder (extracted from generate_color_pool.py).

This module retains the pre-entropy confidence implementation for old
experiments: ``QuestionCase``/``PoolBuilder``, the confidence-era
``GENERATOR_PROMPT``/``ANALYZER_PROMPT``, ``LocalTester`` and
``DeepSeekClient``.  It consumes normalized 0--1 confidence values in
``LEGACY_BIN_RANGES``.

The V2 CLI in generate_color_pool.py never calls this code; it imports
``PoolBuilder`` from here only so old ``from generate_color_pool import
PoolBuilder`` imports keep working.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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


ROOT_DIR = Path(__file__).resolve().parents[3]  # data_generation/legacy/generate_color_pool -> repo root
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# The V2 CLI section below imports generation_runtime/generation_v2, which now
# live one level above the legacy folder.
GENERATION_DIR = ROOT_DIR / "data_generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))


from confidence_analysis import ConfidenceAnalyzer  # noqa: E402


def _load_api_config() -> dict:
    """Load API configuration from the project root api_config.json."""
    config_path = ROOT_DIR / "api_config.json"
    if not config_path.exists():
        raise RuntimeError(
            f"Missing config file: {config_path}. "
            "Create api_config.json with 'api_key' and 'base_url' fields."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


COLOR_SET_A = [
    "red", "orange", "yellow", "green", "blue", "cyan",
    "purple", "pink", "brown", "white", "black", "gray",
]
COLOR_SET_B = [
    "maroon", "lime", "navy", "teal", "olive", "magenta",
    "silver", "gold", "beige", "coral", "violet", "turquoise",
]
# Only the first vocabulary is active for colour-pool generation.  Keep the
# second vocabulary definition for compatibility with existing datasets and
# output files, but never schedule its colours for testing or generation.
ALL_COLORS = list(COLOR_SET_A)


# Legacy PoolBuilder consumes normalized 0--1 confidence values.  Keeping its
# private ranges intact preserves compatibility while V2 uses BIN_RANGES.
LEGACY_BIN_RANGES = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
DEFAULT_BATCH_SIZE_BY_BIN = {0: 40, 1: 40, 2: 20, 3: 10, 4: 40}
LOW_BIN_DIFFICULTY_WEIGHTS = (
    ("prior_knowledge_multistep", 15),
    ("not_exclusion", 10),
    ("high_difficulty", 10),
    ("free_form", 5),
)
LOW_BIN_DIFFICULTIES = tuple(name for name, _weight in LOW_BIN_DIFFICULTY_WEIGHTS)
LOW_BIN_GENERATOR_AGENTS = (
    "prior_knowledge_agent",
    "not_exclusion_agent",
    "high_difficulty_agent",
)
LOW_BIN_AGENT_DIFFICULTIES = {
    "prior_knowledge_agent": ("prior_knowledge_multistep", "free_form"),
    "not_exclusion_agent": ("not_exclusion",),
    "high_difficulty_agent": ("high_difficulty",),
}
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
    2: "Make the target the strongest candidate while retaining one or two reasonable alternatives.",
    3: "Clearly support the target while preserving a small amount of reasonable uncertainty.",
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

Previously generated clues that were stable but measured into another bin:
{routed_priors}

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

ANALYZER_PROMPT = """You are the analysis agent paired exclusively with generator agent
"{generator_agent}" for target color "{target_color}" and confidence bin "{target_bin}".

Do not analyze or modify any other generator agent or confidence bin.

Current generator prompt:
{current_generator_prompt}

Accepted candidates:
{accepted_results}

Rejected candidates and measured results:
{rejected_results}

Stable candidates routed to other measured bins:
{routed_results}

Mandatory bin-specific generation contract:
{bin_strategy}

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

The next prompt must continue to request exactly {batch_size} candidates and
must preserve the mandatory bin-specific generation contract verbatim in
substance, including every exact difficulty_type quota.

If answers are wrong, add a subtle target-specific cue.
If confidence is too high, weaken evidence and preserve more alternatives.
If confidence is too low, add one useful property or remove one major competitor.
If confidence is unstable, prefer evidence independent of wording and shape.

Return JSON only:

{{
  "target_color": "{target_color}",
  "bin_id": {bin_id},
  "target_bin": "{target_bin}",
  "generator_agent": "{generator_agent}",
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
    generator_agent: str = "bin_general"


@dataclass
class AgentState:
    color: str
    bin_id: int
    current_generator_prompt: str
    accepted_priors: list[dict[str, Any]] = field(default_factory=list)
    rejected_results: list[dict[str, Any]] = field(default_factory=list)
    routed_priors: list[dict[str, Any]] = field(default_factory=list)
    agent_prompts: dict[str, str] = field(default_factory=dict)
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
    low, high = LEGACY_BIN_RANGES[bin_id]
    return low <= value <= high if bin_id == 4 else low <= value < high


def bin_for_value(value: float) -> int | None:
    for bin_id in range(5):
        if in_bin(value, bin_id):
            return bin_id
    return None


def bin_label(bin_id: int) -> str:
    low, high = LEGACY_BIN_RANGES[bin_id]
    close = "]" if bin_id == 4 else ")"
    return f"[{low:.1f}, {high:.1f}{close}"


def low_bin_difficulty_quotas(batch_size: int) -> dict[str, int]:
    """Scale the required 15/10/10/5 mix when a non-default batch is used."""
    total_weight = sum(weight for _name, weight in LOW_BIN_DIFFICULTY_WEIGHTS)
    exact = [batch_size * weight / total_weight for _name, weight in LOW_BIN_DIFFICULTY_WEIGHTS]
    counts = [int(value) for value in exact]
    remainder = batch_size - sum(counts)
    order = sorted(
        range(len(exact)),
        key=lambda index: (exact[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return {
        name: counts[index]
        for index, (name, _weight) in enumerate(LOW_BIN_DIFFICULTY_WEIGHTS)
    }


def low_bin_generation_contract(batch_size: int) -> str:
    quotas = low_bin_difficulty_quotas(batch_size)
    return (
        "NON-NEGOTIABLE LOW-BIN MIX (the four groups are mutually exclusive; use the exact difficulty_type strings):\n"
        f"1. Exactly {quotas['prior_knowledge_multistep']} candidates with difficulty_type=\"prior_knowledge_multistep\". "
        "Each must require at least two reasoning hops: identify an indirect or obscure referent, recall a color-related fact about it, "
        "then map that fact to the best candidate color. They must depend strongly on prior/world knowledge. Structural example only: "
        "\"The color has the same color as a morpho butterfly's wings\" requires knowing what a morpho is and recalling its wing color. "
        "Do not copy that example repeatedly; diversify across biology, minerals, astronomy, chemistry, artifacts, history, geography, and culture.\n"
        f"2. Exactly {quotas['not_exclusion']} candidates with difficulty_type=\"not_exclusion\". "
        "Every clue must contain the explicit word \"not\" and negate one or more OTHER candidate color names. Never negate or state the target color. "
        "Leave several candidates viable so exclusion does not make the target obvious, and vary which competing colors are negated.\n"
        f"3. Exactly {quotas['high_difficulty']} candidates with difficulty_type=\"high_difficulty\". "
        "Use genuinely difficult, obscure, technical, culturally specific, or deliberately indirect evidence. The clue must still be truthful and make the target "
        "the unique best answer, but it must not reuse the explicit two-hop referent template from group 1 or the explicit-not construction from group 2.\n"
        f"4. Exactly {quotas['free_form']} candidates with difficulty_type=\"free_form\". "
        "Explore distinct approaches not used above while preserving the target answer and requested confidence range.\n"
        "Across all groups, do not create paraphrase families, do not mention a shape, and keep every clue semantically distinct."
    )


def low_bin_agent_batch_size(agent_name: str, batch_size: int) -> int:
    quotas = low_bin_difficulty_quotas(batch_size)
    return sum(quotas[name] for name in LOW_BIN_AGENT_DIFFICULTIES[agent_name])


def low_bin_agent_contract(agent_name: str, batch_size: int) -> str:
    quotas = low_bin_difficulty_quotas(batch_size)
    marker = f"NON-NEGOTIABLE SPECIALIST CONTRACT: {agent_name}. "
    if agent_name == "prior_knowledge_agent":
        return (
            marker + "You are the dedicated prior-knowledge/multi-step generator. Do not generate not_exclusion or high_difficulty items. "
            f"Generate exactly {quotas['prior_knowledge_multistep']} items with difficulty_type=\"prior_knowledge_multistep\" and "
            f"exactly {quotas['free_form']} items with difficulty_type=\"free_form\". "
            "The prior_knowledge_multistep items must require at least two hops: identify an indirect or obscure referent, recall its color-related fact, "
            "then map that fact to the best candidate. Structural example only: \"The color has the same color as a morpho butterfly's wings\". "
            "Diversify across biology, minerals, astronomy, chemistry, artifacts, history, geography, and culture. The free_form items must use distinct approaches. "
            "Never state any candidate color name and do not use the explicit word \"not\"."
        )
    if agent_name == "not_exclusion_agent":
        return (
            marker + "You are the dedicated explicit-negation generator. Do not generate any other difficulty_type. "
            f"Generate exactly {quotas['not_exclusion']} items with difficulty_type=\"not_exclusion\". "
            "Every item must contain the explicit word \"not\" and negate one or more OTHER colors from the candidate set. "
            "Never state or negate the target color. Leave several candidates viable, vary the competitors negated, and include a subtle indirect reason "
            "that keeps the target as the unique best answer without making it obvious."
        )
    if agent_name == "high_difficulty_agent":
        return (
            marker + "You are the dedicated high-difficulty generator. Do not generate any other difficulty_type. "
            f"Generate exactly {quotas['high_difficulty']} items with difficulty_type=\"high_difficulty\". "
            "Use genuinely obscure, technical, culturally specific, or deliberately indirect evidence while keeping the target the unique best answer. "
            "Do not use the explicit two-hop referent template assigned to the prior-knowledge agent, do not use the word \"not\", and never state any candidate color name."
        )
    raise ValueError(f"Unknown low-bin generator agent: {agent_name}")


def low_bin_base_strategy(bin_id: int) -> str:
    strength = (
        "extremely weak: several alternatives must remain plausible, but the target must still be the unique best answer"
        if bin_id == 0
        else "weak but noticeable: the target must be the unique best answer while two or three alternatives remain credible"
    )
    return (
        f"The evidence must be {strength}. "
        "Correct-answer preservation is mandatory: never make the clue so generic that another candidate becomes equally good or better. "
        "Use subtle, truthful, target-specific anchors and keep every clue shape-independent and semantically distinct. "
        "Avoid direct canonical objects that make the answer obvious."
    )


def build_low_bin_agent_strategy(bin_id: int, agent_name: str, batch_size: int) -> str:
    return f"{low_bin_base_strategy(bin_id)}\n\n{low_bin_agent_contract(agent_name, batch_size)}"


def build_bin_strategy(bin_id: int, batch_size: int) -> str:
    if bin_id in (0, 1):
        return f"{low_bin_base_strategy(bin_id)}\n\n{low_bin_generation_contract(batch_size)}"
    if bin_id == 4:
        return (
            "Make the target the only reasonable answer using direct, universally recognizable, common-knowledge evidence. "
            "Use canonical objects, materials, symbols, natural phenomena, or conventional associations strongly tied to the target. "
            "Avoid hedging, negation, obscure trivia, unreliable sources, competing alternatives, and indirect multi-step reasoning. "
            "Do not state any candidate color name in the clue. Diversify domains and wording while keeping every clue precise, truthful, and unambiguous."
        )
    return BIN_STRATEGIES[bin_id]


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
        allow_cross_bin = source == "deepseek_generated" and target_bin_id in (0, 1)
        effective_bin = first_bin if source == "existing_dataset" else target_bin_id
        if effective_bin is None or (
            source != "existing_dataset"
            and not allow_cross_bin
            and not in_bin(float(first["soft_confidence"]), effective_bin)
        ):
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
        if allow_cross_bin:
            effective_bin = bin_for_value(soft_mean)
        if not answers_unchanged:
            reason = "answers_changed"
        elif not soft_range < self.stability_threshold:
            reason = "soft_confidence_unstable"
        elif effective_bin is None or not in_bin(soft_mean, int(effective_bin)):
            reason = "soft_mean_outside_target_bin"
        else:
            reason = "accepted"
        accepted = stable and effective_bin is not None and in_bin(soft_mean, int(effective_bin))
        final = self._final_result(
            text_clue, source, generation_round, effective_bin, tests,
            candidate, accepted, reason,
        )
        final["answers_unchanged"] = answers_unchanged
        final["soft_range"] = soft_range
        final["soft_mean"] = soft_mean
        final["stable"] = stable
        if source == "deepseek_generated":
            final["generated_for_bin_id"] = target_bin_id
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
                    "generator_agent": candidate.generator_agent,
                }
            )
        return value


class DeepSeekClient:
    def __init__(self, workers: int):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("DeepSeek generation requires the 'openai' Python package") from exc

        config = _load_api_config()
        api_key = config.get("api_key", "")
        base_url = config.get("base_url", "")

        if not api_key or not base_url:
            raise RuntimeError("Missing api_key or base_url in api_config.json")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=150.0,
            max_retries=0,
        )
        self.workers = workers
        self.generator_calls = 0
        self.analyzer_calls = 0
        self._counter_lock = threading.Lock()
        self._print_lock = threading.Lock()
        self._request_semaphore = threading.BoundedSemaphore(workers)

    def _log(self, message: str) -> None:
        """Print immediate API health logs without writing a checkpoint."""
        with self._print_lock:
            print(f"[DeepSeek][{utc_now()}] {message}", flush=True)

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("DeepSeek response JSON must be an object")
        return parsed

    def call(
        self,
        prompt: str,
        kind: str,
        retries: int = 3,
        context: str = "",
    ) -> dict[str, Any]:
        log_name = f"{kind} {context}".strip()
        with self._counter_lock:
            if kind == "generator":
                self.generator_calls += 1
            else:
                self.analyzer_calls += 1
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                acquired = self._request_semaphore.acquire(blocking=False)
                if not acquired:
                    self._log(f"{log_name} request queued (attempt {attempt}/{retries})")
                    self._request_semaphore.acquire()
                try:
                    self._log(f"{log_name} request started (attempt {attempt}/{retries})")
                    response = self.client.chat.completions.create(
                        model="deepseek-v4-flash-aistar",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=8192,
                    )
                finally:
                    self._request_semaphore.release()
                content = response.choices[0].message.content or ""
                parsed = self._parse_json(content)
                self._log(
                    f"{log_name} request succeeded (attempt {attempt}, response_chars={len(content)})"
                )
                return parsed
            except Exception as exc:
                last_error = exc
                self._log(f"{log_name} request failed (attempt {attempt}): {exc}")
                if attempt < retries:
                    self._log(f"{log_name} request will retry")
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
    choice_colors: list[str],
    expected_low_counts: dict[str, int] | None = None,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    if normalize_answer(str(payload.get("color", ""))) != state.color or int(payload.get("bin_id", -1)) != state.bin_id:
        return [], [{"rejection_reason": "generator_identity_mismatch", "payload": payload}]
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != batch_size:
        return [], [{"rejection_reason": "wrong_candidate_count", "actual": len(raw_candidates) if isinstance(raw_candidates, list) else None}]
    if state.bin_id in (0, 1):
        counts = Counter(str(item.get("difficulty_type", "")) for item in raw_candidates if isinstance(item, dict))
        expected = expected_low_counts or low_bin_difficulty_quotas(batch_size)
        if set(counts) != set(expected) or any(counts[name] != expected[name] for name in expected):
            return [], [{"rejection_reason": "wrong_low_bin_difficulty_quota", "counts": dict(counts), "expected": expected}]

    candidates: list[Candidate] = []
    existing = [prior["text_clue"] for prior in state.accepted_priors if prior.get("text_clue")]
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            rejected.append({"rejection_reason": "candidate_not_object", "candidate_index": index})
            continue
        clue = str(item.get("text_clue", "")).strip()
        lowered = normalize_text(clue)
        difficulty_type = str(item.get("difficulty_type", "unspecified"))
        if not clue:
            rejected.append({"rejection_reason": "empty_clue", "candidate_index": index})
            continue
        if lowered in tested_keys:
            rejected.append({"text_clue": clue, "rejection_reason": "already_tested"})
            continue
        if any(re.search(rf"\b{re.escape(shape)}\b", lowered) for shape in SHAPE_WORDS):
            rejected.append({"text_clue": clue, "rejection_reason": "shape_dependent"})
            continue
        if state.bin_id in (0, 1):
            target_pattern = rf"\b{re.escape(normalize_text(state.color))}\b"
            other_colors = [color for color in choice_colors if normalize_answer(color) != state.color]
            mentioned_others = [
                color for color in other_colors
                if re.search(rf"\b{re.escape(normalize_text(color))}\b", lowered)
            ]
            if re.search(target_pattern, lowered):
                rejected.append({"text_clue": clue, "rejection_reason": "low_bin_states_target_color"})
                continue
            if difficulty_type == "not_exclusion":
                if not re.search(r"\bnot\b", lowered) or not mentioned_others:
                    rejected.append({
                        "text_clue": clue,
                        "rejection_reason": "invalid_not_exclusion_structure",
                        "mentioned_other_colors": mentioned_others,
                    })
                    continue
            elif re.search(r"\bnot\b", lowered):
                rejected.append({
                    "text_clue": clue,
                    "rejection_reason": "not_used_outside_not_exclusion",
                })
                continue
            elif mentioned_others:
                rejected.append({
                    "text_clue": clue,
                    "rejection_reason": "other_color_named_outside_not_exclusion",
                    "mentioned_other_colors": mentioned_others,
                })
                continue
        if is_near_duplicate(clue, existing + [candidate.text_clue for candidate in candidates], near_duplicate_threshold):
            rejected.append({"text_clue": clue, "rejection_reason": "duplicate_or_near_duplicate"})
            continue
        candidates.append(
            Candidate(
                candidate_id=str(item.get("candidate_id", f"{state.color}-{state.bin_id}-{state.round_index}-{index}")),
                strategy_family=str(item.get("strategy_family", "unspecified")),
                difficulty_type=difficulty_type,
                text_clue=clue,
                generator_agent=str(item.get("generator_agent", "bin_general")),
            )
        )
    return candidates, rejected


def validate_low_bin_agent_response(
    payload: dict[str, Any],
    state: AgentState,
    agent_name: str,
    batch_size: int,
) -> dict[str, Any] | None:
    if normalize_answer(str(payload.get("color", ""))) != state.color or int(payload.get("bin_id", -1)) != state.bin_id:
        return {
            "rejection_reason": "generator_agent_identity_mismatch",
            "generator_agent": agent_name,
            "payload": payload,
        }
    raw_candidates = payload.get("candidates")
    expected_total = low_bin_agent_batch_size(agent_name, batch_size)
    if not isinstance(raw_candidates, list) or len(raw_candidates) != expected_total:
        return {
            "rejection_reason": "wrong_generator_agent_candidate_count",
            "generator_agent": agent_name,
            "actual": len(raw_candidates) if isinstance(raw_candidates, list) else None,
            "expected": expected_total,
        }
    quotas = low_bin_difficulty_quotas(batch_size)
    allowed = LOW_BIN_AGENT_DIFFICULTIES[agent_name]
    counts = Counter(
        str(item.get("difficulty_type", ""))
        for item in raw_candidates
        if isinstance(item, dict)
    )
    expected_counts = {name: quotas[name] for name in allowed}
    if set(counts) != set(allowed) or any(counts[name] != expected_counts[name] for name in allowed):
        return {
            "rejection_reason": "wrong_generator_agent_difficulty_quota",
            "generator_agent": agent_name,
            "counts": dict(counts),
            "expected": expected_counts,
        }
    for item in raw_candidates:
        item["generator_agent"] = agent_name
        item["candidate_id"] = f"{agent_name}:{item.get('candidate_id', 'candidate')}"
    return None


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
        routed_priors=compact_json(state.routed_priors),
        bin_strategy=build_bin_strategy(state.bin_id, batch_size),
        bin_id=state.bin_id,
    )


def result_belongs_to_agent(
    result: dict[str, Any],
    agent_name: str,
    generated_for_bin_id: int,
) -> bool:
    if result.get("source") == "existing_dataset":
        return False
    source_bin = result.get("generated_for_bin_id")
    if source_bin is None and result.get("source") == "deepseek_generated":
        source_bin = result.get("bin_id")
    if source_bin is not None and int(source_bin) != generated_for_bin_id:
        return False
    recorded_agent = str(result.get("generator_agent", ""))
    if recorded_agent:
        return recorded_agent in {agent_name, "bin_combined"}
    difficulty_type = str(result.get("difficulty_type", ""))
    return difficulty_type in LOW_BIN_AGENT_DIFFICULTIES[agent_name]


def results_for_agent(
    results: list[dict[str, Any]],
    agent_name: str,
    generated_for_bin_id: int,
) -> list[dict[str, Any]]:
    return [
        result for result in results
        if result_belongs_to_agent(result, agent_name, generated_for_bin_id)
    ]


def initial_low_bin_agent_prompt(
    state: AgentState,
    agent_name: str,
    batch_size: int,
    choices: list[str],
) -> str:
    agent_batch_size = low_bin_agent_batch_size(agent_name, batch_size)
    return GENERATOR_PROMPT.format(
        target_bin=bin_label(state.bin_id),
        target_color=state.color,
        choice_colors=", ".join(choices),
        batch_size=agent_batch_size,
        accepted_priors=compact_json(
            [
                p.get("text_clue")
                for p in results_for_agent(state.accepted_priors, agent_name, state.bin_id)
            ]
        ),
        rejected_results=compact_json(
            results_for_agent(state.rejected_results, agent_name, state.bin_id)
        ),
        routed_priors=compact_json(
            results_for_agent(state.routed_priors, agent_name, state.bin_id)
        ),
        bin_strategy=build_low_bin_agent_strategy(state.bin_id, agent_name, batch_size),
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
    for result in state.rejected_results + state.accepted_priors + state.routed_priors:
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
        "generated_count": len(state.rejected_results) + len(state.accepted_priors) + len(state.routed_priors),
        "valid_candidate_count": sum(
            1 for r in state.rejected_results + state.accepted_priors + state.routed_priors
            if r.get("text_clue")
        ),
        "duplicate_count": reasons["duplicate_or_near_duplicate"],
        "first_answer_wrong_count": reasons["first_answer_incorrect_or_no_soft_confidence"],
        "confidence_too_low_count": reasons["confidence_too_low"],
        "confidence_too_high_count": reasons["confidence_too_high"],
        "answer_changed_count": reasons["answers_changed"] + reasons["answer_changed_or_stage2_failed"],
        "soft_unstable_count": reasons["soft_confidence_unstable"],
        "accepted_count": len(state.accepted_priors),
        "routed_to_other_bin_count": len(state.routed_priors),
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
        self.artifact_dir = ROOT_DIR / "data_generation" / "legacy" / "generate_color_pool" / "output"
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
                "selected_bins": args.selected_bins,
                "color_workers": args.color_workers,
                "persistence_policy": "each_accepted_prior",
                "find_enabled": args.find,
            }
        )
        self.inference: Any = None
        self.tester: LocalTester | None = None
        self.deepseek: DeepSeekClient | None = None
        self._local_evaluation_lock = threading.Lock()
        self._state_lock = threading.Lock()
        # Admit at most three colors to a DeepSeek phase at once. Combined with
        # the validated worker count, every active bin of those colors can start
        # immediately instead of waiting for a global API slot.
        self._deepseek_color_phase_slots = threading.BoundedSemaphore(3)

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
        with self._state_lock:
            self.events.append(value)
            log_line = f"{value['timestamp']} {event_type} {compact_json(details, 2000)}"
            self.log_lines.append(log_line)
            self._append_log_line(
                self.artifact_dir / "color_prior_generation_events.jsonl",
                json.dumps(value, ensure_ascii=False),
            )
            self._append_log_line(
                self.artifact_dir / "color_prior_generation.log",
                log_line,
            )

    @staticmethod
    def _append_log_line(path: Path, line: str) -> None:
        """Durably append one event instead of waiting for a checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def load_local_model(self) -> None:
        inference_class = load_inference_class(ROOT_DIR / "qwen-2.5-vl" / "inference.py")
        self.inference = inference_class(model_path=str((ROOT_DIR / "qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct").resolve()))
        self.tester = LocalTester(self.inference, self.args.stability_threshold)

    def _persist_pool(self, color: str) -> None:
        """Persist the pool immediately after an accepted prior is added.

        Callers must hold ``self._state_lock`` so concurrent colour workers
        cannot mutate the shared pool while it is being serialized.
        """
        entry = self.pool_by_color[color]
        for level in entry["prior_levels"]:
            level["complete"] = len(level.get("priors", [])) >= self.args.target_per_bin
        entry["complete"] = all(level["complete"] for level in entry["prior_levels"])
        entry["selected_bins_complete"] = all(
            level_for(entry, bin_id)["complete"] for bin_id in self.args.selected_bins
        )
        entry["updated_at"] = utc_now()
        atomic_write_json(self.output_path, self.pool)

    def run_find(self) -> None:
        assert self.tester is not None
        existing = extract_existing_priors(self.dataset)
        self.event("find_started", generator_calls=0, analyzer_calls=0)
        print(f"[Find] started colors={self.args.selected_colors} deepseek_calls=0", flush=True)
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
                entry["tested_prior_keys"] = sorted(tested)
                if result["accepted"]:
                    with self._state_lock:
                        level_for(entry, int(result["bin_id"]))["priors"].append(result)
                        already_accepted.append(clue)
                        self._persist_pool(color)
                self.event("find_prior_tested", color=color, accepted=result["accepted"], reason=result["rejection_reason"])
                print(
                    f"[Find] color={color} prior_tested accepted={result["accepted"]} "
                    f"bin={result.get("bin_id")} reason={result["rejection_reason"]}",
                    flush=True,
                )
            entry["tested_prior_keys"] = sorted(tested)
            print(
                f"[Find] color={color} completed tested={len(tested)} "
                f"bin_counts={ {level["bin_id"]: len(level.get("priors", [])) for level in entry["prior_levels"]} }",
                flush=True,
            )
        self.event("find_completed", generator_calls=0, analyzer_calls=0)
        print("[Find] completed all colors; deepseek_generator_calls=0 deepseek_analyzer_calls=0", flush=True)

    def _make_states(self, color: str) -> dict[int, AgentState]:
        entry = self.pool_by_color[color]
        histories = [item for item in self.prompt_history if item.get("color") == color]
        states: dict[int, AgentState] = {}
        for bin_id in self.args.selected_bins:
            level = level_for(entry, bin_id)
            state = AgentState(
                color=color,
                bin_id=bin_id,
                current_generator_prompt="",
                accepted_priors=level["priors"],
                routed_priors=[
                    prior
                    for other_level in entry["prior_levels"]
                    for prior in other_level.get("priors", [])
                    if prior.get("generated_for_bin_id") == bin_id
                    and int(prior.get("bin_id", -1)) != bin_id
                ],
            )
            batch_size = self.args.batch_sizes[bin_id]
            if bin_id in (0, 1):
                latest_round = 0
                for agent_name in LOW_BIN_GENERATOR_AGENTS:
                    latest = max(
                        (
                            item for item in histories
                            if int(item.get("bin_id", -1)) == bin_id
                            and item.get("generator_agent") == agent_name
                        ),
                        key=lambda item: int(item.get("round", 0)),
                        default=None,
                    )
                    if latest:
                        latest_round = max(latest_round, int(latest.get("round", 0)))
                    state.agent_prompts[agent_name] = (
                        str(latest.get("next_generator_prompt"))
                        if latest and latest.get("next_generator_prompt")
                        else initial_low_bin_agent_prompt(
                            state, agent_name, batch_size, entry["choice_colors"]
                        )
                    )
                state.round_index = latest_round + 1
            else:
                latest = max(
                    (item for item in histories if int(item.get("bin_id", -1)) == bin_id),
                    key=lambda item: int(item.get("round", 0)),
                    default=None,
                )
                state.current_generator_prompt = (
                    str(latest.get("next_generator_prompt"))
                    if latest and latest.get("next_generator_prompt")
                    else initial_generator_prompt(state, batch_size, entry["choice_colors"])
                )
                state.round_index = (int(latest.get("round", 0)) + 1) if latest else 1
            states[bin_id] = state
        return states

    def _generate_one(
        self,
        state: AgentState,
        agent_name: str,
    ) -> tuple[int, str, dict[str, Any] | Exception]:
        assert self.deepseek is not None
        try:
            if state.bin_id in (0, 1):
                prompt = state.agent_prompts[agent_name]
                marker = f"NON-NEGOTIABLE SPECIALIST CONTRACT: {agent_name}"
                if marker not in prompt:
                    prompt = (
                        f"{prompt}\n\n"
                        f"{build_low_bin_agent_strategy(state.bin_id, agent_name, self.args.batch_sizes[state.bin_id])}"
                    )
                state.agent_prompts[agent_name] = prompt
            else:
                prompt = state.current_generator_prompt
            result = self.deepseek.call(
                prompt,
                "generator",
                context=f"color={state.color} bin={state.bin_id} agent={agent_name}",
            )
            self.event(
                "generator_completed",
                color=state.color,
                bin_id=state.bin_id,
                round=state.round_index,
                generator_agent=agent_name,
                succeeded=True,
                candidate_count=len(result.get("candidates", []))
                if isinstance(result.get("candidates"), list) else None,
            )
            return state.bin_id, agent_name, result
        except Exception as exc:
            self.event(
                "generator_completed",
                color=state.color,
                bin_id=state.bin_id,
                round=state.round_index,
                generator_agent=agent_name,
                succeeded=False,
                error=str(exc),
            )
            return state.bin_id, agent_name, exc

    def _analyze_one(
        self,
        state: AgentState,
        agent_name: str,
    ) -> tuple[int, str, dict[str, Any] | Exception]:
        assert self.deepseek is not None
        if state.bin_id in (0, 1):
            accepted_results = results_for_agent(
                state.accepted_priors, agent_name, state.bin_id
            )
            rejected_results = results_for_agent(
                state.rejected_results, agent_name, state.bin_id
            )
            routed_results = results_for_agent(
                state.routed_priors, agent_name, state.bin_id
            )
            current_prompt = state.agent_prompts[agent_name]
            marker = f"NON-NEGOTIABLE SPECIALIST CONTRACT: {agent_name}"
            if marker not in current_prompt:
                current_prompt = (
                    f"{current_prompt}\n\n"
                    f"{build_low_bin_agent_strategy(state.bin_id, agent_name, self.args.batch_sizes[state.bin_id])}"
                )
                state.agent_prompts[agent_name] = current_prompt
            analyzer_batch_size = low_bin_agent_batch_size(
                agent_name, self.args.batch_sizes[state.bin_id]
            )
            bin_strategy = build_low_bin_agent_strategy(
                state.bin_id, agent_name, self.args.batch_sizes[state.bin_id]
            )
            statistics = summarize_state(AgentState(
                color=state.color,
                bin_id=state.bin_id,
                current_generator_prompt=current_prompt,
                accepted_priors=accepted_results,
                rejected_results=rejected_results,
                routed_priors=routed_results,
            ))
        else:
            accepted_results = state.accepted_priors
            rejected_results = state.rejected_results
            routed_results = state.routed_priors
            current_prompt = state.current_generator_prompt
            analyzer_batch_size = self.args.batch_sizes[state.bin_id]
            bin_strategy = build_bin_strategy(state.bin_id, analyzer_batch_size)
            statistics = summarize_state(state)
        prompt = ANALYZER_PROMPT.format(
            target_color=state.color,
            target_bin=bin_label(state.bin_id),
            generator_agent=agent_name,
            current_generator_prompt=current_prompt,
            accepted_results=compact_json(accepted_results),
            rejected_results=compact_json(rejected_results),
            routed_results=compact_json(routed_results),
            statistics=compact_json(statistics),
            batch_size=analyzer_batch_size,
            bin_strategy=bin_strategy,
            bin_id=state.bin_id,
        )
        try:
            result = self.deepseek.call(
                prompt,
                "analyzer",
                retries=3,
                context=f"color={state.color} bin={state.bin_id} agent={agent_name}",
            )
            if (
                normalize_answer(str(result.get("target_color", ""))) != state.color
                or int(result.get("bin_id", -1)) != state.bin_id
                or str(result.get("generator_agent", "")) != agent_name
            ):
                raise ValueError("Analyzer returned a different color, bin, or generator agent")
            return state.bin_id, agent_name, result
        except Exception as exc:
            return state.bin_id, agent_name, exc

    def generate_color(
        self,
        color: str,
        phase_barrier: threading.Barrier | None = None,
    ) -> None:
        assert self.tester is not None and self.deepseek is not None
        entry = self.pool_by_color[color]
        states = self._make_states(color)
        starting_rounds = {
            bin_id: state.round_index for bin_id, state in states.items()
        }
        for round_index in range(1, self.args.rounds + 1):
            active = [
                state for state in states.values()
                if len(state.accepted_priors) < self.args.target_per_bin
            ]
            if not active and phase_barrier is None:
                break
            if phase_barrier is not None:
                # Cohort starts every Generator phase together.
                phase_barrier.wait()
            for state in active:
                state.round_index = starting_rounds[state.bin_id] + round_index - 1
                if state.bin_id not in (0, 1):
                    state.current_generator_prompt = initial_generator_prompt(
                        state, self.args.batch_sizes[state.bin_id], entry["choice_colors"]
                    ) if round_index == 1 and not state.current_generator_prompt else state.current_generator_prompt

            generation_jobs = [
                (state, agent_name)
                for state in active
                for agent_name in (
                    LOW_BIN_GENERATOR_AGENTS if state.bin_id in (0, 1) else ("bin_general",)
                )
            ]
            generated: dict[tuple[int, str], dict[str, Any] | Exception] = {}
            if generation_jobs:
                with self._deepseek_color_phase_slots:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(generation_jobs)) as executor:
                        futures = [
                            executor.submit(self._generate_one, state, agent_name)
                            for state, agent_name in generation_jobs
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            bin_id, agent_name, result = future.result()
                            generated[(bin_id, agent_name)] = result

            tested_keys = set(entry.get("tested_prior_keys", []))
            for state in active:  # Deliberately serial: one local Qwen call chain at a time.
                candidates: list[Candidate] = []
                validation_rejections: list[dict[str, Any]] = []
                if state.bin_id in (0, 1):
                    for agent_name in LOW_BIN_GENERATOR_AGENTS:
                        agent_payload = generated[(state.bin_id, agent_name)]
                        if isinstance(agent_payload, Exception):
                            state.rejected_results.append({
                                "rejection_reason": "generator_api_failure",
                                "error": str(agent_payload),
                                "generator_agent": agent_name,
                                "generated_for_bin_id": state.bin_id,
                            })
                            continue
                        validation_error = validate_low_bin_agent_response(
                            agent_payload,
                            state,
                            agent_name,
                            self.args.batch_sizes[state.bin_id],
                        )
                        if validation_error:
                            validation_error["generated_for_bin_id"] = state.bin_id
                            state.rejected_results.append(validation_error)
                            continue
                        quotas = low_bin_difficulty_quotas(self.args.batch_sizes[state.bin_id])
                        expected_counts = {
                            name: quotas[name]
                            for name in LOW_BIN_AGENT_DIFFICULTIES[agent_name]
                        }
                        agent_candidates, agent_rejections = validate_generator_response(
                            agent_payload,
                            state,
                            low_bin_agent_batch_size(
                                agent_name, self.args.batch_sizes[state.bin_id]
                            ),
                            self.args.near_duplicate_threshold,
                            tested_keys,
                            entry["choice_colors"],
                            expected_low_counts=expected_counts,
                        )
                        for rejection in agent_rejections:
                            rejection.setdefault("generator_agent", agent_name)
                            rejection.setdefault("generated_for_bin_id", state.bin_id)
                        candidates.extend(agent_candidates)
                        validation_rejections.extend(agent_rejections)
                else:
                    payload = generated[(state.bin_id, "bin_general")]
                    if isinstance(payload, Exception):
                        state.rejected_results.append({
                            "rejection_reason": "generator_api_failure",
                            "error": str(payload),
                            "generator_agent": "bin_general",
                            "generated_for_bin_id": state.bin_id,
                        })
                        continue
                    candidates, validation_rejections = validate_generator_response(
                        payload,
                        state,
                        self.args.batch_sizes[state.bin_id],
                        self.args.near_duplicate_threshold,
                        tested_keys,
                        entry["choice_colors"],
                    )
                    for rejection in validation_rejections:
                        rejection.setdefault("generator_agent", "bin_general")
                        rejection.setdefault("generated_for_bin_id", state.bin_id)
                state.rejected_results.extend(validation_rejections)
                for candidate in candidates:
                    if state.bin_id in (0, 1):
                        if all(
                            len(level_for(entry, bin_id)["priors"]) >= self.args.target_per_bin
                            for bin_id in range(5)
                        ):
                            break
                    elif len(state.accepted_priors) >= self.args.target_per_bin:
                        break
                    print(
                        f"[LocalEval] waiting color={color} bin={state.bin_id} "
                        f"round={state.round_index} candidate={candidate.candidate_id}",
                        flush=True,
                    )
                    with self._local_evaluation_lock:
                        print(
                            f"[LocalEval] started color={color} bin={state.bin_id} "
                            f"round={state.round_index} candidate={candidate.candidate_id}",
                            flush=True,
                        )
                        result = self.tester.test_prior(
                            candidate.text_clue,
                            color,
                            self.question_bank[color],
                            source="deepseek_generated",
                            generation_round=state.round_index,
                            target_bin_id=state.bin_id,
                            candidate=candidate,
                        )
                        print(
                            f"[LocalEval] completed color={color} bin={state.bin_id} "
                            f"round={state.round_index} candidate={candidate.candidate_id} "
                            f"accepted={result['accepted']} reason={result['rejection_reason']}",
                            flush=True,
                        )
                    candidate_key = normalize_text(candidate.text_clue)
                    tested_keys.add(candidate_key)
                    routed_to_bin: int | None = None
                    if result["accepted"]:
                        measured_bin = int(result["bin_id"])
                        with self._state_lock:
                            entry["tested_prior_keys"].append(candidate_key)
                            destination = level_for(entry, measured_bin)["priors"]
                            if is_near_duplicate(
                                candidate.text_clue,
                                accepted_texts(entry),
                                self.args.near_duplicate_threshold,
                            ):
                                result["accepted"] = False
                                result["rejection_reason"] = "duplicate_or_near_duplicate_in_measured_bin"
                            elif len(destination) >= self.args.target_per_bin:
                                result["accepted"] = False
                                result["rejection_reason"] = "measured_bin_already_full"
                            else:
                                result["routed_from_bin_id"] = state.bin_id
                                destination.append(result)
                                if measured_bin != state.bin_id:
                                    state.routed_priors.append(result)
                                    routed_to_bin = measured_bin
                                self._persist_pool(color)
                        if not result["accepted"]:
                            state.rejected_results.append(result)
                        elif routed_to_bin is not None:
                            self.event(
                                "generated_prior_routed",
                                color=color,
                                generated_for_bin=state.bin_id,
                                measured_bin=routed_to_bin,
                                soft_mean=result.get("soft_mean"),
                                candidate_id=candidate.candidate_id,
                            )
                            print(
                                f"[ColorPool] routed color={color} candidate={candidate.candidate_id} "
                                f"from_bin={state.bin_id} to_bin={routed_to_bin} "
                                f"soft_mean={result.get('soft_mean')}",
                                flush=True,
                            )
                    else:
                        with self._state_lock:
                            entry["tested_prior_keys"].append(candidate_key)
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
                                "confidence_too_low" if measured < LEGACY_BIN_RANGES[state.bin_id][0] else "confidence_too_high"
                            )
                        state.rejected_results.append(result)
                    self.event(
                        "generated_prior_tested",
                        color=color,
                        generated_for_bin=state.bin_id,
                        measured_bin=result.get("bin_id"),
                        round=state.round_index,
                        candidate_id=candidate.candidate_id,
                        difficulty_type=candidate.difficulty_type,
                        text_clue=candidate.text_clue,
                        soft_mean=result.get("soft_mean"),
                        accepted=result["accepted"],
                        reason=result["rejection_reason"],
                    )

            if phase_barrier is not None:
                # Qwen remains serial globally; the cohort waits until every
                # color has finished evaluation, then releases all Analyzers.
                phase_barrier.wait()

            # Every Generator invoked this round gets its dedicated Analyzer,
            # even if local evaluation filled the destination bin meanwhile.
            analyzable = active
            analyzer_jobs = [
                (state, agent_name)
                for state in analyzable
                for agent_name in (
                    LOW_BIN_GENERATOR_AGENTS if state.bin_id in (0, 1) else ("bin_general",)
                )
            ]
            analyzed: dict[tuple[int, str], dict[str, Any] | Exception] = {}
            if analyzer_jobs:
                with self._deepseek_color_phase_slots:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(analyzer_jobs)) as executor:
                        futures = [
                            executor.submit(self._analyze_one, state, agent_name)
                            for state, agent_name in analyzer_jobs
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            bin_id, agent_name, result = future.result()
                            analyzed[(bin_id, agent_name)] = result
            for state, agent_name in analyzer_jobs:
                analyzer_output = analyzed.get((state.bin_id, agent_name))
                current_prompt = (
                    state.agent_prompts[agent_name]
                    if state.bin_id in (0, 1)
                    else state.current_generator_prompt
                )
                next_prompt = current_prompt
                if isinstance(analyzer_output, dict):
                    revised = analyzer_output.get("revised_generator_prompt")
                    if isinstance(revised, str) and revised.strip():
                        next_prompt = revised.strip()
                with self._state_lock:
                    self.prompt_history.append(
                        {
                            "color": color,
                            "bin_id": state.bin_id,
                            "generator_agent": agent_name,
                            "round": state.round_index,
                            "generator_prompt": current_prompt,
                            "analyzer_output": analyzer_output if isinstance(analyzer_output, dict) else {"error": str(analyzer_output)} if analyzer_output else {},
                            "next_generator_prompt": next_prompt,
                        }
                    )
                if state.bin_id in (0, 1):
                    state.agent_prompts[agent_name] = next_prompt
                else:
                    state.current_generator_prompt = next_prompt
            if phase_barrier is not None:
                # Do not let the next Generator round overlap this Analyzer phase.
                phase_barrier.wait()
            if not active:
                continue
            self.event(
                "generation_round_completed",
                color=color,
                round=round_index,
                bin_counts={str(bin_id): len(state.accepted_priors) for bin_id, state in states.items()},
            )
            bin_counts = {bin_id: len(state.accepted_priors) for bin_id, state in states.items()}
            missing_bins = [bin_id for bin_id, count in bin_counts.items() if count < self.args.target_per_bin]
            print(
                f"[ColorPool] color={color} round={round_index} completed "
                f"accepted={bin_counts} missing_bins={missing_bins} "
                f"color_complete={not missing_bins}",
                flush=True,
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
            for bin_id in self.args.selected_bins
        )
        if needs_generation:
            self.deepseek = DeepSeekClient(self.args.deepseek_workers)

        generation_jobs: list[tuple[str, list[int]]] = []
        for color in self.args.selected_colors:
            entry = self.pool_by_color[color]
            missing = [
                bin_id for bin_id in self.args.selected_bins
                if len(level_for(entry, bin_id)["priors"]) < self.args.target_per_bin
            ]
            if missing:
                self.event("color_generation_started", color=color, missing_bins=missing)
                generation_jobs.append((color, missing))
            else:
                self.event("color_completed", color=color, generated=False)

        cohort_size = min(3, self.args.color_workers)
        for cohort_start in range(0, len(generation_jobs), cohort_size):
            cohort = generation_jobs[cohort_start:cohort_start + cohort_size]
            cohort_colors = [color for color, _missing in cohort]
            print(
                f"[ColorPool] synchronized color cohort started colors={cohort_colors} "
                f"workers={len(cohort_colors)}",
                flush=True,
            )
            errors: dict[str, Exception] = {}
            phase_barrier = threading.Barrier(len(cohort))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(cohort)) as executor:
                futures = {
                    executor.submit(self.generate_color, color, phase_barrier): color
                    for color, _missing in cohort
                }
                for future in concurrent.futures.as_completed(futures):
                    color = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        errors[color] = exc
                        phase_barrier.abort()
                        print(f"[ColorPool] color={color} generation failed: {exc}", flush=True)

            for color, _missing in cohort:
                if color in errors:
                    self.event("color_generation_failed", color=color, error=str(errors[color]))
                    continue
                self.event("color_completed", color=color, generated=True)
            print(f"[ColorPool] synchronized color cohort completed colors={cohort_colors}", flush=True)
            if errors:
                failed = ", ".join(f"{color}: {error}" for color, error in errors.items())
                raise RuntimeError(f"Color generation batch failed: {failed}")

