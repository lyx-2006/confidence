"""Layer-wise linear probes for existing V3/V4 joint hidden states."""

from __future__ import annotations

PROBE_TASKS = {
    "ac_text_answer": ("ac", "text_only_answer"),
    "ac_image_answer": ("ac", "image_only_answer"),
    "panl_text_answer": ("panl", "text_only_answer"),
    "panl_image_answer": ("panl", "image_only_answer"),
}

C_GRID = [0.01, 0.1, 1.0, 10.0]
POSITION_NAMES = ("ac", "panl")
EASY_CONDITIONS = frozenset({"consistent_easy", "conflict_easy"})
HIDDEN_STATE_DEFINITION = "decoder_block_output_pre_final_norm"
VERSION_SETTINGS = {
    "v3_to_v3": ("v3", "v3"),
    "v4_to_v4": ("v4", "v4"),
    "v3_to_v4": ("v3", "v4"),
    "v4_to_v3": ("v4", "v3"),
}

__all__ = [
    "C_GRID",
    "EASY_CONDITIONS",
    "HIDDEN_STATE_DEFINITION",
    "POSITION_NAMES",
    "PROBE_TASKS",
    "VERSION_SETTINGS",
]
