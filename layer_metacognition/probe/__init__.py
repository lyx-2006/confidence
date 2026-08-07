"""Layer-wise linear probes for existing V3/V4 joint hidden states."""

from __future__ import annotations

from typing import Iterable

C_GRID = [0.01, 0.1, 1.0, 10.0]
POSITION_NAMES = ("ac", "panl", "ltt", "ptnl", "sac")
PROBE_CONDITIONS = (
    "consistent_easy",
    "consistent_hard",
    "conflict_easy",
    "conflict_hard",
)
DEFAULT_PROBE_CONDITIONS = ("consistent_easy", "conflict_easy")
DEFAULT_PROBE_LOCATIONS = ("ac", "panl")
EASY_CONDITIONS = frozenset({"consistent_easy", "conflict_easy"})
HIDDEN_STATE_DEFINITION = "decoder_block_output_pre_final_norm"
VERSION_SETTINGS = {
    "v3_to_v3": ("v3", "v3"),
    "v4_to_v4": ("v4", "v4"),
    "v3_to_v4": ("v3", "v4"),
    "v4_to_v3": ("v4", "v3"),
}


def normalize_ordered_choices(
    values: Iterable[str],
    allowed: tuple[str, ...],
    argument: str,
) -> tuple[str, ...]:
    raw = [str(value) for value in values]
    invalid = [value for value in raw if value not in allowed]
    if invalid:
        raise ValueError(f"Unknown {argument} value(s): {', '.join(invalid)}")
    selected = set(raw)
    return tuple(value for value in allowed if value in selected)


def build_probe_tasks(
    answer_locations: Iterable[str],
    conflict_locations: Iterable[str],
) -> dict[str, tuple[str, str]]:
    tasks: dict[str, tuple[str, str]] = {}
    for position in normalize_ordered_choices(
        answer_locations, POSITION_NAMES, "answer Probe location"
    ):
        tasks[f"{position}_text_answer"] = (position, "text_only_answer")
        tasks[f"{position}_image_answer"] = (position, "image_only_answer")
    for position in normalize_ordered_choices(
        conflict_locations, POSITION_NAMES, "conflict Probe location"
    ):
        tasks[f"{position}_conflict"] = (position, "conflict_label")
    return tasks


PROBE_TASKS = build_probe_tasks(
    DEFAULT_PROBE_LOCATIONS,
    DEFAULT_PROBE_LOCATIONS,
)

__all__ = [
    "C_GRID",
    "DEFAULT_PROBE_CONDITIONS",
    "DEFAULT_PROBE_LOCATIONS",
    "EASY_CONDITIONS",
    "HIDDEN_STATE_DEFINITION",
    "POSITION_NAMES",
    "PROBE_CONDITIONS",
    "PROBE_TASKS",
    "VERSION_SETTINGS",
    "build_probe_tasks",
    "normalize_ordered_choices",
]
