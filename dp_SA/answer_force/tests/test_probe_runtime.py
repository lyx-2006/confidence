from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dp_SA.answer_force.probe_runtime import prompt_only_answer_audit


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(value) for value in text]


def _context(answer: str):
    rendered = f"Question: q\n\n**Answer**: {answer}\n\nState what you believe the fixed answer above was mainly based on.\n**Source Attribution**:"
    ids = torch.tensor([[ord(value) for value in rendered]])
    start = rendered.index(answer)
    span = [start, start + len(answer)]
    details = {"located": {"phase1_answer_span": span}, "image_sha256": "same"}
    return {"answer": answer, "rendered": rendered, "inputs": SimpleNamespace(input_ids=ids), "details": details, "image_sha256": "same"}


def test_prompt_audit_only_allows_answer_span_change():
    assert prompt_only_answer_audit(_context("red"), _context("blue"))["passed"]
    with pytest.raises(ValueError, match="prompt-only-answer-span"):
        changed = _context("blue")
        changed["rendered"] = changed["rendered"].replace("State", "Explain")
        prompt_only_answer_audit(_context("red"), changed)
