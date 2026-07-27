"""Stage-specific source sets for V3/V4 cognition targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetStageSpec:
    stage_name: str
    target: str
    compared_sources: tuple[str, ...]


TARGET_STAGE_SPECS: dict[tuple[str, str, str], TargetStageSpec] = {
    ("v3", "joint", "ac"): TargetStageSpec(
        "joint_answer_source",
        "ac",
        ("image", "text_clue", "previous_answer", "previous_confidence"),
    ),
    ("v3", "joint", "sac"): TargetStageSpec(
        "joint_answer_source",
        "sac",
        ("image", "text_clue", "previous_answer", "previous_confidence", "current_answer"),
    ),
    ("v3", "parallel", "ac"): TargetStageSpec(
        "answer",
        "ac",
        ("image", "text_clue", "previous_answer", "previous_confidence"),
    ),
    ("v3", "parallel", "sac"): TargetStageSpec(
        "source_attribution",
        "sac",
        ("image", "text_clue", "initial_answer", "initial_confidence", "current_answer"),
    ),
    ("v3", "none", "ac"): TargetStageSpec(
        "answer",
        "ac",
        ("image", "text_clue", "previous_answer", "previous_confidence"),
    ),
    ("v3", "none", "cc"): TargetStageSpec(
        "confidence",
        "cc",
        ("image", "text_clue", "initial_answer", "initial_confidence", "current_answer"),
    ),
    ("v3", "parallel", "cc"): TargetStageSpec(
        "confidence",
        "cc",
        ("image", "text_clue", "initial_answer", "initial_confidence", "current_answer"),
    ),
    ("v3", "joint", "cc"): TargetStageSpec(
        "confidence",
        "cc",
        ("image", "text_clue", "initial_answer", "initial_confidence", "current_answer"),
    ),
    ("v4", "joint", "ac"): TargetStageSpec(
        "joint_answer_source", "ac", ("image", "text_clue")
    ),
    ("v4", "joint", "sac"): TargetStageSpec(
        "joint_answer_source", "sac", ("image", "text_clue", "current_answer")
    ),
    ("v4", "parallel", "ac"): TargetStageSpec(
        "answer", "ac", ("image", "text_clue")
    ),
    ("v4", "none", "ac"): TargetStageSpec(
        "answer", "ac", ("image", "text_clue")
    ),
    ("v4", "parallel", "sac"): TargetStageSpec(
        "source_attribution", "sac", ("image", "text_clue", "current_answer")
    ),
    ("v4", "none", "cc"): TargetStageSpec(
        "confidence", "cc", ("image", "text_clue", "current_answer")
    ),
    ("v4", "parallel", "cc"): TargetStageSpec(
        "confidence", "cc", ("image", "text_clue", "current_answer")
    ),
    ("v4", "joint", "cc"): TargetStageSpec(
        "confidence", "cc", ("image", "text_clue", "current_answer")
    ),
}


def stage_spec(version: str, mode: str, target: str) -> TargetStageSpec:
    try:
        return TARGET_STAGE_SPECS[(version, mode, target)]
    except KeyError as exc:
        raise ValueError(
            f"No stage specification for version={version}, mode={mode}, target={target}"
        ) from exc

