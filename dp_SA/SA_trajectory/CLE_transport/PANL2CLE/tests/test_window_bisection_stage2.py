from __future__ import annotations

from types import SimpleNamespace

import torch

from dp_SA.SA_trajectory.CLE_transport.PANL2CLE.window_bisection_stage2 import (
    EXPECTED_NEW_PATCHED,
    SEGMENTS,
    SELECTED_CELLS,
    SegmentSwapHook,
    _classify,
    split_window,
)


def test_exact_stage2_contract():
    assert SELECTED_CELLS == (("W2", 17), ("W5", 16))
    assert SEGMENTS == {"left4": (0, 4), "right4": (4, 8)}
    assert EXPECTED_NEW_PATCHED == 2 * 2 * 50 * 2 == 400


def test_split_window_is_exact_and_rejects_non_contiguous_input():
    positions = list(range(101, 109))
    assert split_window(positions, "left4") == [101, 102, 103, 104]
    assert split_window(positions, "right4") == [105, 106, 107, 108]
    try:
        split_window([1, 2, 4, 5, 6, 7, 8, 9], "left4")
    except ValueError:
        pass
    else:
        raise AssertionError("non-contiguous window accepted")


def test_segment_hook_changes_only_four_rows():
    layer = torch.nn.Identity()
    modules = SimpleNamespace(language_layers=[layer], hidden_size=3)
    source = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    original = torch.full((1, 10, 3), 7, dtype=torch.bfloat16)
    hook = SegmentSwapHook(modules, layer=0, positions=[3, 4, 5, 6], source=source, prefill_length=10)
    with hook:
        output = layer(original)
    assert torch.equal(output[0, 3:7], source)
    assert torch.equal(output[0, :3], original[0, :3])
    assert torch.equal(output[0, 7:], original[0, 7:])
    diagnostics = hook.diagnostics()
    assert diagnostics["target_exact"] and diagnostics["outside_exact"]
    assert diagnostics["segment_length"] == 4


def _effect(value, low, high):
    return {"estimate": value, "ci_low": low, "ci_high": high}


def test_classification_covers_dominant_distributed_interaction_and_opposite():
    full = _effect(1.0, .5, 1.5)
    interaction = _effect(.1, -.2, .4)
    dominant = _classify(full, _effect(.8, .2, 1.2), _effect(.1, -.2, .3), interaction)
    assert dominant["category"] == "one_half_dominant" and dominant["suggested_segments"] == ["left4"]
    distributed = _classify(full, _effect(.5, .1, .9), _effect(.5, .1, .9), interaction)
    assert distributed["category"] == "both_halves_distributed"
    weak = _classify(full, _effect(.1, -.2, .3), _effect(.1, -.2, .3), _effect(.8, .4, 1.2))
    assert weak["category"] == "weak_halves_full_effect_interaction" and not weak["continue_bisection"]
    opposite = _classify(full, _effect(.6, .1, 1.0), _effect(-.3, -.6, -.1), interaction)
    assert opposite["category"] == "opposite_directions"


def test_retention_omitted_when_full_effect_not_clear():
    result = _classify(_effect(.1, -.2, .4), _effect(.1, -.2, .3), _effect(.0, -.1, .1), _effect(.0, -.2, .2))
    assert result["category"] == "full8_not_clearly_nonzero"
    assert result["retention_left"] is None and result["retention_right"] is None
