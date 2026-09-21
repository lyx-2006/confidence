#!/usr/bin/env python3
"""Calibrate legacy colour clues with Qwen3 text-only normalized entropy."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
ROOT = PROJECT_DIR.parent
DEFAULT_INPUT = ROOT / "generate color pool" / "datasets" / "new_color_prior_pool.json"
DEFAULT_OUTPUT = PROJECT_DIR / "datasets" / "current" / "text_entropy_calibration.json"
DEFAULT_MODEL = ROOT / "qwen-3-vl" / "model"
DEFAULT_API_CONFIG = ROOT / "api_config.json"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash-aistar"
COLORS = (
    "red", "orange", "yellow", "green", "blue", "cyan",
    "purple", "pink", "brown", "white", "black", "gray",
)
QUESTION = (
    "What is the color of the {shape}? Choose from: "
    + ", ".join(COLORS)
    + "."
)
HIGH_INTERVALS = (
    ("0.3-0.4", 0.3, 0.4),
    ("0.5-0.6", 0.5, 0.6),
    (">=0.6", 0.6, 1.0000000001),
)
SHAPE_WORDS = {
    "arrow", "circle", "crescent", "cross", "diamond", "heart", "hexagon",
    "octagon", "oval", "parallelogram", "pentagon", "rectangle", "semicircle",
    "square", "star", "trapezoid", "triangle", "shape",
}
SUBJECTS = {
    "red": ("fruit skin", "flower petal", "bird feather", "gemstone", "painted sign", "autumn leaf", "ceramic glaze", "woven fabric", "berry cluster", "mineral crystal"),
    "orange": ("fruit skin", "gourd rind", "autumn leaf", "clay surface", "sunset cloud", "flower petal", "spice powder", "bird feather", "ceramic glaze", "weathered fabric"),
    "yellow": ("fruit peel", "flower petal", "grain field", "mineral crystal", "aged paper", "bird plumage", "wax surface", "spice powder", "autumn leaf", "ceramic glaze"),
    "green": ("tree leaf", "moss patch", "algae film", "gemstone", "glass bottle", "insect wing", "woven fabric", "mineral crust", "fruit skin", "painted surface"),
    "blue": ("bird feather", "butterfly wing", "gemstone", "water surface", "glass bottle", "woven fabric", "painted surface", "fish scales", "mineral crystal", "ceramic glaze"),
    "cyan": ("shallow water", "glass surface", "mineral crystal", "fish scales", "insect wing", "painted tile", "translucent fabric", "ice surface", "bird feather", "ceramic glaze"),
    "purple": ("flower petal", "fruit skin", "gemstone", "woven fabric", "sunset cloud", "bird feather", "mineral crystal", "ceramic glaze", "butterfly wing", "berry cluster"),
    "pink": ("flower petal", "seashell", "fruit flesh", "bird plumage", "gemstone", "woven fabric", "sunset cloud", "mineral crystal", "ceramic glaze", "insect wing"),
    "brown": ("wooden surface", "river sediment", "animal coat", "mushroom cap", "weathered leather", "tree bark", "soil sample", "ceramic glaze", "mineral crystal", "dried leaf"),
    "white": ("cloud bank", "woven fabric", "paper surface", "flower petal", "seashell", "mineral crystal", "ceramic glaze", "bird feather", "wax surface", "translucent glass"),
    "black": ("volcanic stone", "bird feather", "woven fabric", "glass surface", "mineral crystal", "painted object", "charred wood", "animal coat", "ceramic glaze", "insect shell"),
    "gray": ("river stone", "cloud bank", "mineral surface", "animal coat", "painted wall", "weathered wood", "paper surface", "ceramic glaze", "glass bottle", "woven fabric"),
}
UNCERTAINTY_FRAMES = (
    "The apparent hue of a {subject} changes strongly with viewing angle, surrounding reflections, and mixed illumination.",
    "The observed appearance of a {subject} varies with age, condition, weathering, and the available light.",
    "A {subject} is seen briefly through haze while reflecting several unrelated surroundings.",
    "A weathered {subject} sits between cool daylight and warm artificial illumination.",
    "The surface of a {subject} reflects sky, vegetation, earth, and indoor light at the same time.",
    "A partly translucent {subject} is viewed in shadow with bright reflections crossing its surface.",
    "The remembered hue of a {subject} is uncertain because it was seen at dusk through tinted glass.",
    "A {subject} appears different across its surface because of moisture, age, shadow, and reflected light.",
    "The dominant visual impression of a {subject} is weakened by glare, haze, and nearby materials.",
    "A distant {subject} is partly obscured and alternates between several plausible appearances as the light changes.",
    "The natural variation of a {subject} is compounded by uneven illumination and strong environmental reflections.",
    "A {subject} with no fixed variety is photographed under mixed lighting and an uncertain camera balance.",
)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def extract_clues(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input text pool must be a JSON array")
    result = {color: [] for color in COLORS}
    seen = {color: set() for color in COLORS}
    for color_entry in raw:
        if not isinstance(color_entry, dict):
            continue
        color = str(color_entry.get("color", "")).strip().lower()
        if color not in result:
            continue
        for level in color_entry.get("prior_levels", []):
            if not isinstance(level, dict):
                continue
            for prior in level.get("priors", []):
                if not isinstance(prior, dict) or prior.get("accepted") is not True:
                    continue
                clue = str(prior.get("text_clue", "")).strip()
                key = " ".join(clue.casefold().split())
                if clue and key not in seen[color]:
                    seen[color].add(key)
                    result[color].append(clue)
    return result


def normalized_key(value: str) -> str:
    return " ".join(value.casefold().split())


def entropy_interval(value: float) -> str | None:
    for name, lower, upper in HIGH_INTERVALS:
        if lower <= value < upper:
            return name
    return None


def load_calibration(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return [{"color": color, "clues": []} for color in COLORS]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Calibration output must be a JSON array")
    by_color: dict[str, list[dict[str, Any]]] = {}
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"color", "clues"}:
            raise ValueError("Calibration entries may contain only color and clues")
        color = str(entry["color"])
        if color not in COLORS or color in by_color or not isinstance(entry["clues"], list):
            raise ValueError(f"Invalid calibration color entry: {color!r}")
        checked = []
        for clue in entry["clues"]:
            if not isinstance(clue, dict) or set(clue) != {"clue", "Entropy"}:
                raise ValueError("Clue entries may contain only clue and Entropy")
            text = str(clue["clue"]).strip()
            entropy = float(clue["Entropy"])
            if not text or not 0.0 <= entropy <= 1.0:
                raise ValueError("Invalid clue or Entropy")
            checked.append({"clue": text, "Entropy": entropy})
        by_color[color] = checked
    return [{"color": color, "clues": by_color.get(color, [])} for color in COLORS]


def valid_generated_clue(clue: str) -> bool:
    words = re.findall(r"[a-z]+", clue.casefold())
    if not 8 <= len(words) <= 90:
        return False
    forbidden = set(COLORS) | SHAPE_WORDS
    return not any(word in forbidden for word in words)


class DeepSeekClueGenerator:
    def __init__(self, config_path: Path, model: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("High-entropy clue generation requires the openai package") from exc
        config = json.loads(config_path.read_text(encoding="utf-8"))
        api_key = str(config.get("api_key", ""))
        base_url = str(config.get("base_url", ""))
        if not api_key or not base_url:
            raise RuntimeError("api_config.json must contain api_key and base_url")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=0)
        self.model = model

    def generate(
        self,
        color: str,
        requested: dict[str, int],
        accepted: list[dict[str, Any]],
        feedback: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, str]]:
        interval_guidance = {
            "0.3-0.4": "moderately ambiguous: two or three plausible associations; the target is still the best",
            "0.5-0.6": "strongly ambiguous: four or more plausible associations; the target has only a weak edge",
            ">=0.6": "maximally ambiguous: many everyday colors remain plausible; only a subtle cue favors the target",
        }
        request_lines = [
            f"- {name}: {count} clues; {interval_guidance[name]}"
            for name, count in requested.items() if count > 0
        ]
        recent_feedback = {
            name: values[-12:] for name, values in feedback.items() if name in requested
        }
        accepted_sample = [
            {"clue": item["clue"], "Entropy": round(float(item["Entropy"]), 4)}
            for item in accepted if entropy_interval(float(item["Entropy"])) is not None
        ][-12:]
        prompt = f"""Generate shape-independent text clues whose intended answer is the color {color!r}.
They will be tested by Qwen3 against exactly these 12 answers: {', '.join(COLORS)}.
The score is normalized entropy over those 12 answer probabilities. A clue is accepted only when both
the generated answer and restricted top-1 equal {color!r}; therefore ambiguity must not make another
color win.

Requested candidates:
{chr(10).join(request_lines)}

Rules:
- Return strict JSON only: {{"candidates": [{{"target_interval": "...", "clue": "..."}}, ...]}}.
- Return exactly the requested number for every interval.
- Never write any of the 12 color words in a clue, including the target.
- Never name a geometric shape. Keep each clue between 8 and 90 words.
- Use genuine semantic ambiguity: uncertain lighting, weather, aging, material, landscape, culture,
  biological variation, or a scene with several implicit associations.
- Prefer one concrete object class whose appearance naturally varies across many colors, then add viewing-angle,
  reflection, ripeness, weathering, or mixed-light uncertainty. This has measured better than merely saying a
  clue is ambiguous. Example structure only: "The apparent hue of a fruit skin changes strongly with viewing
  angle, surrounding reflections, and mixed illumination." Choose a different target-appropriate subject.
- Do not say "answer", "candidate", "choose", "probability", "confidence", "entropy", or discuss this task.
- Avoid a single diagnostic object that makes the target obvious. Do not enumerate same-colored objects.
- Every clue must remain defensibly answerable as {color!r}, even though alternatives are plausible.

Previously accepted high-entropy clues for this color:
{json.dumps(accepted_sample, ensure_ascii=False)}

Recent Qwen3 feedback. gate=false means Qwen selected another color; very small Entropy means the clue
was too obvious. Adjust the balance instead of paraphrasing failed clues:
{json.dumps(recent_feedback, ensure_ascii=False)}
"""
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=6144,
                )
                content = (response.choices[0].message.content or "").strip()
                content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
                content = re.sub(r"\s*```$", "", content)
                payload = json.loads(content)
                candidates = payload.get("candidates") if isinstance(payload, dict) else None
                if not isinstance(candidates, list):
                    raise ValueError("response does not contain a candidates array")
                result = []
                for item in candidates:
                    if not isinstance(item, dict):
                        continue
                    target_interval = str(item.get("target_interval", ""))
                    clue = str(item.get("clue", "")).strip()
                    if target_interval in requested and valid_generated_clue(clue):
                        result.append({"target_interval": target_interval, "clue": clue})
                return result
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError(f"DeepSeek clue generation failed: {type(last_error).__name__}: {last_error}")


def load_runner(model_path: Path) -> Any:
    faithful = ROOT / "qwen3 review" / "short prompt check" / "faithful check"
    for path in (ROOT, faithful.parent.parent, faithful.parent, faithful):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    existing = sys.modules.get("runtime")
    runtime_path = (faithful / "runtime.py").resolve()
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() != runtime_path:
        raise RuntimeError(f"Top-level runtime module collision: {getattr(existing, '__file__', None)}")
    from runtime import FaithfulQwenRunner

    return FaithfulQwenRunner(model_path.resolve())


def calibrate(
    clues_by_color: dict[str, list[str]],
    runner: Any,
    shape: str,
    output_path: Path,
    checkpoint_every: int,
) -> list[dict[str, Any]]:
    output = [{"color": color, "clues": []} for color in COLORS]
    output_by_color = {entry["color"]: entry["clues"] for entry in output}
    question = QUESTION.format(shape=shape)
    total = sum(len(values) for values in clues_by_color.values())
    tested = accepted = 0
    for color in COLORS:
        for clue in clues_by_color[color]:
            result = runner.text_only(question, clue, color)
            tested += 1
            if result.get("gate_passed") is True:
                entropy = result.get("normalized_entropy")
                if isinstance(entropy, (int, float)) and 0.0 <= float(entropy) <= 1.0:
                    output_by_color[color].append({"clue": clue, "Entropy": float(entropy)})
                    accepted += 1
            if tested % checkpoint_every == 0 or tested == total:
                atomic_write_json(output_path, output)
                print(
                    f"[calibrate] tested={tested}/{total} accepted={accepted} color={color}",
                    flush=True,
                )
    return output


def high_bin_counts(clues: list[dict[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name, _lower, _upper in HIGH_INTERVALS}
    for item in clues:
        name = entropy_interval(float(item["Entropy"]))
        if name is not None:
            counts[name] += 1
    return counts


def template_candidates(color: str) -> list[str]:
    return [
        frame.format(subject=subject)
        for subject in SUBJECTS[color]
        for frame in UNCERTAINTY_FRAMES
    ]


def run_template_search(
    output: list[dict[str, Any]],
    output_path: Path,
    runner: Any,
    colors: list[str],
    shape: str,
    quota: int,
    attempt_limit: int,
) -> None:
    output_by_color = {entry["color"]: entry["clues"] for entry in output}
    question = QUESTION.format(shape=shape)
    for color in colors:
        seen = {normalized_key(item["clue"]) for item in output_by_color[color]}
        tested = added = 0
        max_entropy = 0.0
        answers: Counter[str] = Counter()
        for clue in template_candidates(color):
            if tested >= attempt_limit or all(
                count >= quota for count in high_bin_counts(output_by_color[color]).values()
            ):
                break
            if normalized_key(clue) in seen:
                continue
            seen.add(normalized_key(clue))
            result = runner.text_only(question, clue, color)
            tested += 1
            answers[str(result.get("normalized_answer"))] += 1
            entropy_value = result.get("normalized_entropy")
            if not isinstance(entropy_value, (int, float)):
                continue
            entropy = float(entropy_value)
            max_entropy = max(max_entropy, entropy)
            actual_interval = entropy_interval(entropy)
            if result.get("gate_passed") is not True or actual_interval is None:
                continue
            if high_bin_counts(output_by_color[color])[actual_interval] >= quota:
                continue
            output_by_color[color].append({"clue": clue, "Entropy": entropy})
            added += 1
            atomic_write_json(output_path, output)
        print(
            f"[template-search] color={color} tested={tested} added={added} max_entropy={max_entropy:.4f} "
            f"status={json.dumps(high_bin_counts(output_by_color[color]), separators=(',', ':'))} "
            f"answers={json.dumps(dict(answers), ensure_ascii=False, separators=(',', ':'))}",
            flush=True,
        )


def augment_high_entropy(
    output_path: Path,
    runner: Any,
    generator: DeepSeekClueGenerator,
    colors: list[str],
    shape: str,
    quota: int,
    max_attempts: int,
    candidates_per_bin: int,
    rounds: int,
    api_workers: int,
    template_attempts: int,
) -> list[dict[str, Any]]:
    output = load_calibration(output_path)
    run_template_search(
        output, output_path, runner, colors, shape, quota, template_attempts
    )
    output_by_color = {entry["color"]: entry["clues"] for entry in output}
    seen = {
        color: {normalized_key(item["clue"]) for item in output_by_color[color]}
        for color in COLORS
    }
    attempts = {
        color: {name: 0 for name, _lower, _upper in HIGH_INTERVALS}
        for color in colors
    }
    feedback: dict[str, dict[str, list[dict[str, Any]]]] = {
        color: {name: [] for name, _lower, _upper in HIGH_INTERVALS}
        for color in colors
    }
    question = QUESTION.format(shape=shape)

    for round_index in range(1, rounds + 1):
        requests: dict[str, dict[str, int]] = {}
        for color in colors:
            counts = high_bin_counts(output_by_color[color])
            needed = {
                name: min(candidates_per_bin, max_attempts - attempts[color][name])
                for name, _lower, _upper in HIGH_INTERVALS
                if counts[name] < quota and attempts[color][name] < max_attempts
            }
            needed = {name: count for name, count in needed.items() if count > 0}
            if needed:
                requests[color] = needed
        if not requests:
            break

        generated: dict[str, list[dict[str, str]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(api_workers, len(requests))) as executor:
            futures = {
                executor.submit(
                    generator.generate,
                    color,
                    needed,
                    output_by_color[color],
                    feedback[color],
                ): color
                for color, needed in requests.items()
            }
            for future in concurrent.futures.as_completed(futures):
                color = futures[future]
                try:
                    generated[color] = future.result()
                except Exception as exc:
                    print(f"[generate-hard] color={color} api_error={type(exc).__name__}: {exc}", flush=True)
                    generated[color] = []

        evaluated = added = 0
        round_entropies: dict[str, list[float]] = defaultdict(list)
        round_passes: Counter[str] = Counter()
        round_answers: dict[str, Counter[str]] = defaultdict(Counter)
        for color in colors:
            for candidate in generated.get(color, []):
                requested_interval = candidate["target_interval"]
                if attempts[color][requested_interval] >= max_attempts:
                    continue
                clue = candidate["clue"]
                key = normalized_key(clue)
                if key in seen[color]:
                    continue
                seen[color].add(key)
                attempts[color][requested_interval] += 1
                result = runner.text_only(question, clue, color)
                evaluated += 1
                entropy_value = result.get("normalized_entropy")
                entropy = float(entropy_value) if isinstance(entropy_value, (int, float)) else None
                actual_interval = entropy_interval(entropy) if entropy is not None else None
                gate = result.get("gate_passed") is True
                if entropy is not None:
                    round_entropies[requested_interval].append(entropy)
                if gate:
                    round_passes[requested_interval] += 1
                round_answers[requested_interval][str(result.get("normalized_answer"))] += 1
                feedback[color][requested_interval].append({
                    "clue": clue,
                    "gate": gate,
                    "answer": result.get("normalized_answer"),
                    "Entropy": round(entropy, 4) if entropy is not None else None,
                })
                if gate and actual_interval is not None:
                    counts = high_bin_counts(output_by_color[color])
                    if counts[actual_interval] < quota:
                        output_by_color[color].append({"clue": clue, "Entropy": entropy})
                        added += 1
                        atomic_write_json(output_path, output)

        status = {color: high_bin_counts(output_by_color[color]) for color in colors}
        print(
            f"[generate-hard] round={round_index}/{rounds} evaluated={evaluated} added={added} "
            f"status={json.dumps(status, ensure_ascii=False, separators=(',', ':'))} "
            f"diagnostics={json.dumps({name: {'max_entropy': max(values) if values else None, 'gate_passed': round_passes[name], 'answers': dict(round_answers[name])} for name, values in round_entropies.items()}, ensure_ascii=False, separators=(',', ':'))}",
            flush=True,
        )
    atomic_write_json(output_path, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--shape", default="circle")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--generate-hard", action="store_true")
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--colors", default=",".join(COLORS))
    parser.add_argument("--quota-per-high-bin", type=int, default=3)
    parser.add_argument("--max-attempts-per-color-bin", type=int, default=200)
    parser.add_argument("--candidates-per-bin", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--api-workers", type=int, default=6)
    parser.add_argument("--template-attempts-per-color", type=int, default=120)
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    selected = [value.strip().lower() for value in args.colors.split(",") if value.strip()]
    if not selected or len(set(selected)) != len(selected) or any(color not in COLORS for color in selected):
        parser.error("--colors must be a unique comma-separated subset of the 12 colors")
    for name in (
        "quota_per_high_bin", "max_attempts_per_color_bin", "candidates_per_bin",
        "rounds", "api_workers",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.template_attempts_per_color < 0:
        parser.error("--template-attempts-per-color must be non-negative")
    args.selected_colors = selected
    return args


def main() -> None:
    args = parse_args()
    if args.generate_hard:
        runner = load_runner(args.model)
        generator = DeepSeekClueGenerator(args.api_config.resolve(), args.deepseek_model)
        output = augment_high_entropy(
            args.output.resolve(), runner, generator, args.selected_colors, args.shape,
            args.quota_per_high_bin, args.max_attempts_per_color_bin,
            args.candidates_per_bin, args.rounds, args.api_workers,
            args.template_attempts_per_color,
        )
        print(
            "[generate-hard] final="
            + json.dumps(
                {color: high_bin_counts(next(item["clues"] for item in output if item["color"] == color))
                 for color in args.selected_colors},
                ensure_ascii=False,
            ),
            flush=True,
        )
        print(f"[generate-hard] output={args.output.resolve()}", flush=True)
        return
    clues = extract_clues(args.input.resolve())
    print(
        "[calibrate] candidates="
        + str(sum(len(values) for values in clues.values()))
        + " per_color="
        + json.dumps({color: len(clues[color]) for color in COLORS}),
        flush=True,
    )
    runner = load_runner(args.model)
    calibrate(clues, runner, args.shape, args.output.resolve(), args.checkpoint_every)
    print(f"[calibrate] output={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
