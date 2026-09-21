#!/usr/bin/env python3
"""Build a small, balanced conflict-only test split from calibrated pools."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable


COLORS = (
    "red", "orange", "yellow", "green", "blue", "cyan",
    "purple", "pink", "brown", "white", "black", "gray",
)
PAIR_TYPES = (
    "hard_text_easy_image",
    "hard_image_easy_text",
    "balanced",
)
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS = PROJECT_DIR / "datasets" / "current"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def image_band(entropy: float, pair_type: str) -> bool:
    if pair_type == "hard_text_easy_image":
        return 0.0 <= entropy < 0.1
    if pair_type == "hard_image_easy_text":
        return entropy >= 0.5
    return 0.3 <= entropy < 0.4


def text_band(entropy: float, pair_type: str) -> bool:
    if pair_type == "hard_text_easy_image":
        return entropy >= 0.5
    if pair_type == "hard_image_easy_text":
        return 0.0 <= entropy < 0.1
    return 0.3 <= entropy < 0.4


def target_entropy(pair_type: str, modality: str) -> float:
    if pair_type == "balanced":
        return 0.35
    is_hard = (pair_type == "hard_text_easy_image") == (modality == "text")
    return 0.60 if is_hard else 0.05


def load_images(pool: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for color in COLORS:
        color_dir = pool / color
        if not color_dir.is_dir():
            raise FileNotFoundError(f"Missing image color directory: {color_dir}")
        for path in sorted(color_dir.glob("*.json")):
            rows = load_json(path)
            if not isinstance(rows, list):
                raise ValueError(f"Expected an array in {path}")
            result[(color, path.stem)] = rows
    return result


def load_clues(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected an array in {path}")
    result = {str(row["color"]): list(row["clues"]) for row in rows}
    if set(result) != set(COLORS):
        raise ValueError("Text calibration must contain exactly the 12 configured colors")
    return result


class Dinic:
    def __init__(self, size: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(size)]

    def add(self, source: int, target: int, capacity: int) -> None:
        forward = [target, capacity, len(self.graph[target])]
        backward = [source, 0, len(self.graph[source])]
        self.graph[source].append(forward)
        self.graph[target].append(backward)

    def flow(self, source: int, sink: int) -> int:
        total = 0
        while True:
            levels = [-1] * len(self.graph)
            levels[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for target, capacity, _ in self.graph[node]:
                    if capacity and levels[target] < 0:
                        levels[target] = levels[node] + 1
                        queue.append(target)
            if levels[sink] < 0:
                return total
            positions = [0] * len(self.graph)

            def send(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                while positions[node] < len(self.graph[node]):
                    edge = self.graph[node][positions[node]]
                    target, capacity, reverse = edge
                    if capacity and levels[target] == levels[node] + 1:
                        pushed = send(target, min(amount, capacity))
                        if pushed:
                            edge[1] -= pushed
                            self.graph[target][reverse][1] += pushed
                            return pushed
                    positions[node] += 1
                return 0

            while pushed := send(source, 10**9):
                total += pushed


def allocate_conflict_answers(
    rows_per_image_color: int,
    quotas: dict[str, int],
) -> dict[str, list[str]]:
    """Meet exact answer quotas while forbidding text_answer == image_answer."""
    count = len(COLORS)
    expected = rows_per_image_color * count
    if sum(quotas.values()) != expected:
        raise ValueError(f"Infeasible text-answer quotas: {quotas}")
    source, image_start, answer_start, sink = 0, 1, 1 + count, 1 + 2 * count
    # Find the smallest per-cell capacity that permits the exact margins. This
    # avoids a valid but poor flow that repeats one text answer for every shape
    # of the same image color.
    for cell_capacity in range(1, rows_per_image_color + 1):
        network = Dinic(sink + 1)
        for image_index, image_color in enumerate(COLORS):
            network.add(source, image_start + image_index, rows_per_image_color)
            for answer_index, answer_color in enumerate(COLORS):
                if answer_color != image_color and quotas.get(answer_color, 0):
                    network.add(
                        image_start + image_index,
                        answer_start + answer_index,
                        cell_capacity,
                    )
        for answer_index, answer_color in enumerate(COLORS):
            network.add(answer_start + answer_index, sink, quotas.get(answer_color, 0))
        if network.flow(source, sink) != expected:
            continue
        result: dict[str, list[str]] = {color: [] for color in COLORS}
        for image_index, image_color in enumerate(COLORS):
            node = image_start + image_index
            for edge in network.graph[node]:
                target, capacity, _ = edge
                if answer_start <= target < answer_start + count:
                    used = cell_capacity - capacity
                    result[image_color].extend([COLORS[target - answer_start]] * used)
        return result
    raise ValueError(f"Infeasible text-answer quotas: {quotas}")


def choose_complete_shapes(
    images: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[str]:
    shapes = sorted({shape for _, shape in images})
    return [
        shape for shape in shapes
        if all(
            any(image_band(float(row["entropy"]), pair_type) for row in images[(color, shape)])
            for color in COLORS for pair_type in PAIR_TYPES
        )
    ]


def per_regime_quotas(clues: dict[str, list[dict[str, Any]]], total: int) -> dict[str, dict[str, int]]:
    """Balance text answers globally, compensating for unavailable hard/balanced colors."""
    final_each, remainder = divmod(total * len(PAIR_TYPES), len(COLORS))
    if remainder:
        raise ValueError("Total test rows cannot be balanced evenly across text answers")

    eligible = {
        pair_type: [
            color for color in COLORS
            if any(text_band(float(row["Entropy"]), pair_type) for row in clues[color])
        ]
        for pair_type in PAIR_TYPES
    }
    quotas: dict[str, dict[str, int]] = {}
    remaining = {color: final_each for color in COLORS}
    for pair_type in ("hard_text_easy_image", "balanced"):
        colors = eligible[pair_type]
        if not colors:
            raise ValueError(f"No calibrated clues for {pair_type}")
        base, extra = divmod(total, len(colors))
        allocation = {
            color: (base + (index < extra) if color in colors else 0)
            for index, color in enumerate(COLORS)
        }
        quotas[pair_type] = allocation
        for color, count in allocation.items():
            remaining[color] -= count
            if remaining[color] < 0:
                raise ValueError("Cannot make text answers globally balanced with available clue bands")
    if any(not any(text_band(float(row["Entropy"]), "hard_image_easy_text") for row in clues[color])
           for color, count in remaining.items() if count):
        raise ValueError("An easy-text quota was assigned to a color without an easy clue")
    if sum(remaining.values()) != total:
        raise AssertionError("Internal quota error")
    quotas["hard_image_easy_text"] = remaining
    return quotas


def sorted_candidates(rows: Iterable[dict[str, Any]], pair_type: str, modality: str) -> list[dict[str, Any]]:
    key = "Entropy" if modality == "text" else "entropy"
    predicate = text_band if modality == "text" else image_band
    selected = [row for row in rows if predicate(float(row[key]), pair_type)]
    center = target_entropy(pair_type, modality)
    return sorted(selected, key=lambda row: (abs(float(row[key]) - center), str(row)))


def build_split(image_pool: Path, text_pool: Path, output: Path, seed: int) -> list[dict[str, Any]]:
    images = load_images(image_pool)
    clues = load_clues(text_pool)
    shapes = choose_complete_shapes(images)
    if not shapes:
        raise ValueError("No shape is complete across every color and all three pairing regimes")
    total_per_type = len(COLORS) * len(shapes)
    quotas = per_regime_quotas(clues, total_per_type)
    rng = random.Random(seed)

    assignments: dict[str, dict[str, list[str]]] = {}
    for pair_type in PAIR_TYPES:
        by_color = allocate_conflict_answers(len(shapes), quotas[pair_type])
        for values in by_color.values():
            rng.shuffle(values)
        assignments[pair_type] = by_color

    clue_offsets: Counter[tuple[str, str]] = Counter()
    result: list[dict[str, Any]] = []
    for image_color in COLORS:
        for pair_type in PAIR_TYPES:
            answers = assignments[pair_type][image_color]
            if len(answers) != len(shapes):
                raise AssertionError("Answer allocation size mismatch")
            for shape, text_answer in zip(shapes, answers):
                image_rows = sorted_candidates(images[(image_color, shape)], pair_type, "image")
                text_rows = sorted_candidates(clues[text_answer], pair_type, "text")
                if not image_rows or not text_rows:
                    raise AssertionError("Completeness checks did not match final selection")
                image_row = image_rows[0]
                clue_key = (pair_type, text_answer)
                clue_row = text_rows[clue_offsets[clue_key] % len(text_rows)]
                clue_offsets[clue_key] += 1
                image_path = image_pool / image_color / str(image_row["image"])
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                relative_image = os.path.relpath(image_path, output.parent)
                result.append({
                    "text": str(clue_row["clue"]),
                    "image": relative_image,
                    "shape": shape,
                    "pair_type": pair_type,
                    "text_entropy": float(clue_row["Entropy"]),
                    "image_entropy": float(image_row["entropy"]),
                    "text_answer": text_answer,
                    "image_answer": image_color,
                })
    atomic_write_json(output, result)
    return result


def validate(rows: list[dict[str, Any]], output: Path) -> None:
    required = {
        "text", "image", "shape", "pair_type", "text_entropy", "image_entropy",
        "text_answer", "image_answer",
    }
    if any(set(row) != required for row in rows):
        raise AssertionError("Output schema is not minimal and uniform")
    if any(row["text_answer"] == row["image_answer"] for row in rows):
        raise AssertionError("A non-conflict pair was emitted")
    if any(not (output.parent / row["image"]).is_file() for row in rows):
        raise AssertionError("An output image path is invalid")
    if any(not text_band(row["text_entropy"], row["pair_type"]) for row in rows):
        raise AssertionError("A text item is outside its declared difficulty band")
    if any(not image_band(row["image_entropy"], row["pair_type"]) for row in rows):
        raise AssertionError("An image item is outside its declared difficulty band")
    image_counts = Counter(row["image_answer"] for row in rows)
    text_counts = Counter(row["text_answer"] for row in rows)
    shape_counts = Counter(row["shape"] for row in rows)
    type_counts = Counter(row["pair_type"] for row in rows)
    if len(set(image_counts.values())) != 1 or len(set(text_counts.values())) != 1:
        raise AssertionError("Answer labels are not exactly balanced")
    if len(set(shape_counts.values())) != 1 or len(set(type_counts.values())) != 1:
        raise AssertionError("Shapes or pair types are not exactly balanced")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-pool", type=Path, default=DEFAULT_DATASETS / "interval_pool_full")
    parser.add_argument("--text-pool", type=Path, default=DEFAULT_DATASETS / "text_entropy_calibration.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASETS / "conflict_test.json")
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    rows = build_split(args.image_pool.resolve(), args.text_pool.resolve(), args.output.resolve(), args.seed)
    validate(rows, args.output.resolve())
    print(f"Wrote {len(rows)} rows to {args.output.resolve()}")
    print("pair types:", dict(sorted(Counter(row["pair_type"] for row in rows).items())))
    print("image answers:", dict(sorted(Counter(row["image_answer"] for row in rows).items())))
    print("text answers:", dict(sorted(Counter(row["text_answer"] for row in rows).items())))
    print("shapes:", sorted({row["shape"] for row in rows}))


if __name__ == "__main__":
    main()
