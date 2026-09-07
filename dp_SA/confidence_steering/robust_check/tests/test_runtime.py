from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ..config import ALPHAS, DIRECTIONS, HIDDEN_SIZE
from ..io_utils import assert_output_path
from ..run import StrictTargetHook, canonical_merge, trial_key


def _modules():
    return SimpleNamespace(language_layers=[torch.nn.Identity() for _ in range(15)], hidden_size=HIDDEN_SIZE)


def test_strict_hook_changes_only_target_token():
    hook = StrictTargetHook(_modules(), 1, torch.ones(HIDDEN_SIZE), 3)
    tensor = torch.zeros((1, 3, HIDDEN_SIZE))
    output = hook._hook(None, None, tensor)
    assert torch.equal(output[:, 0], tensor[:, 0]) and torch.equal(output[:, 2], tensor[:, 2])
    assert torch.equal(output[0, 1], torch.ones(HIDDEN_SIZE)) and hook.off_target_bitwise


def test_zero_hook_is_bitwise_identity():
    hook = StrictTargetHook(_modules(), 1, torch.zeros(HIDDEN_SIZE), 3)
    tensor = torch.randn((1, 3, HIDDEN_SIZE))
    output = hook._hook(None, None, tensor)
    assert torch.equal(output, tensor)


def test_hook_applies_only_once():
    hook = StrictTargetHook(_modules(), 1, torch.ones(HIDDEN_SIZE), 3)
    tensor = torch.zeros((1, 3, HIDDEN_SIZE))
    first = hook._hook(None, None, tensor)
    second = hook._hook(None, None, first)
    assert hook.applied_count == 1 and torch.equal(first, second)


def test_canonical_merge_is_order_independent():
    a = {"seed": 45, "case_id": "b", "direction": "shared_alpha_zero", "alpha": 0.0}
    b = {"seed": 45, "case_id": "a", "direction": "shared_alpha_zero", "alpha": 0.0}
    assert canonical_merge([a, b]) == canonical_merge([b, a])


def test_one_and_two_worker_partitions_have_identical_canonical_merge():
    rows = [{"seed": 45, "case_id": str(index), "direction": "shared_alpha_zero", "alpha": 0.0} for index in range(8)]
    one_worker = canonical_merge(rows)
    two_worker = canonical_merge(rows[::2] + rows[1::2])
    assert one_worker == two_worker


def test_canonical_merge_rejects_conflicting_duplicate():
    a = {"seed": 45, "case_id": "a", "direction": "shared_alpha_zero", "alpha": 0.0, "x": 1}
    b = {**a, "x": 2}
    with pytest.raises(RuntimeError, match="Conflicting"):
        canonical_merge([a, b])


def test_trial_key_includes_seed():
    row = {"seed": 43, "case_id": "x", "direction": "confidence_raw", "alpha": -0.5}
    assert trial_key(row).startswith("43|")


def test_formal_forward_formula_is_700_per_seed():
    assert 100 * (1 + len(DIRECTIONS) * (len(ALPHAS) - 1)) == 700


def test_write_jail_rejects_parent_directory(tmp_path):
    with pytest.raises(ValueError, match="forbidden"):
        assert_output_path(tmp_path / "escape.json")
