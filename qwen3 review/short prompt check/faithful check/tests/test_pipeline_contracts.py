from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[2]
for candidate in (REPOSITORY_ROOT, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from run_pipeline import Pipeline
from runtime import _messages
from prompts import ANSWER_PREFILL


def test_text_selection_starts_at_ten_then_can_exhaust_pool() -> None:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.text_pool = {
        "red": [{"color": "red", "text_clue": f"clue-{index}"} for index in range(15)]
    }

    def gate(_question, candidate):
        return {
            **candidate, "gate_id": candidate["text_clue"], "gate_passed": True,
            "normalized_entropy": 0.2, "target_probability": 0.7, "target_margin": 1.0,
        }

    pipeline._text_gate = gate
    assert len(pipeline._passing_texts("question", "red")) == 10
    assert len(pipeline._passing_texts("question", "red", None)) == 15


def test_text_only_messages_have_no_image_item_and_image_only_has_one(tmp_path: Path) -> None:
    text_messages = _messages("prompt", None, ANSWER_PREFILL)
    assert [item["type"] for item in text_messages[0]["content"]] == ["text"]
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    image_messages = _messages("prompt", str(image), ANSWER_PREFILL)
    assert [item["type"] for item in image_messages[0]["content"]] == ["image", "text"]
