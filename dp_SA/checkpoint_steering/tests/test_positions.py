from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dp_SA.prompts import SA_PREFILL, phase1_prompt
from dp_SA.positions import locate_phase1_positions
from dp_SA.checkpoint_steering.config import ANCHORS, POSITION_ORDER
from dp_SA.checkpoint_steering.positions import locate_checkpoint_positions


class CharTokenizer:
    image = "<|image_pad|>"

    def _encode(self, text: str):
        ids = []
        offsets = []
        index = 0
        while index < len(text):
            if text.startswith(self.image, index):
                ids.append(999999)
                offsets.append((index, index + len(self.image)))
                index += len(self.image)
            else:
                ids.append(ord(text[index]))
                offsets.append((index, index + 1))
                index += 1
        return ids, offsets

    def encode(self, text, add_special_tokens=False):
        return self._encode(text)[0]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets = self._encode(text)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(self, ids, **kwargs):
        return "".join(self.image if int(value) == 999999 else chr(int(value)) for value in ids)

    def convert_tokens_to_ids(self, token):
        return 999999 if token == self.image else -1


def _located(*, clue: str = "clue", prefix: str = ""):
    tokenizer = CharTokenizer()
    prompt = phase1_prompt("q", clue, "blue")
    rendered = prefix + prompt + "\nassistant\n" + SA_PREFILL
    rendered_ids = tokenizer.encode(rendered)
    processed_ids = rendered_ids[:]
    if prefix:
        processed_ids = [999999, 999999, 999999, *rendered_ids[1:]]
    inputs = SimpleNamespace(
        input_ids=torch.tensor([processed_ids]),
        attention_mask=torch.ones(1, len(processed_ids), dtype=torch.long),
    )
    return locate_checkpoint_positions(tokenizer, rendered, inputs, "blue")


def test_five_positions_are_uniquely_located_with_full_audit():
    located = _located()
    indices = [located[name]["processed_index"] for name in POSITION_ORDER]
    assert indices == sorted(indices) and len(indices) == len(set(indices))
    for name in POSITION_ORDER[:-1]:
        row = located[name]
        assert row["anchor_occurrence_count"] == 1
        assert len(row["token_window"]) >= 5
    assert located["P1_SAC"]["anchor_occurrence_count"] == 2
    assert len(located["P1_SAC"]["token_window"]) >= 5
    for name in ANCHORS:
        assert "\n" in located[name]["token_text"]


def test_class_list_position_reuses_public_locator_exact_fields():
    tokenizer=CharTokenizer(); rendered=phase1_prompt("q","clue","blue")+"\nassistant\n"+SA_PREFILL
    ids=tokenizer.encode(rendered)
    inputs=SimpleNamespace(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long))
    public=locate_phase1_positions(tokenizer,rendered,inputs,"blue")["P1_CLASS_LIST_END"]
    checkpoint=locate_checkpoint_positions(tokenizer,rendered,inputs,"blue")["P1_CLASS_LIST_END"]
    for field in ("processed_index","rendered_index","token_id","token_text","anchor_text","anchor_occurrence_count","anchor_start_index"):
        assert checkpoint[field]==public[field]


def test_alignment_survives_processed_image_placeholder_expansion():
    plain = _located()
    expanded = _located(prefix=CharTokenizer.image)
    assert expanded["P1_LAT"]["processed_index"] == plain["P1_LAT"]["processed_index"] + 3
    assert expanded["P1_SAC"]["processed_index"] == plain["P1_SAC"]["processed_index"] + 3


def test_duplicate_new_anchor_fails_without_fallback():
    with pytest.raises(ValueError, match="expected one anchor occurrence, found 2"):
        _located(clue=ANCHORS["P1_CLASS_LIST_END"])


def test_missing_or_non_newline_anchor_fails_without_fallback():
    tokenizer = CharTokenizer()
    rendered = phase1_prompt("q", "clue", "blue").replace(ANCHORS["P1_FORMAT_DESCRIPTION_END"], "missing") + "\nassistant\n" + SA_PREFILL
    ids = tokenizer.encode(rendered)
    inputs = SimpleNamespace(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long))
    with pytest.raises(ValueError, match="found 0"):
        locate_checkpoint_positions(tokenizer, rendered, inputs, "blue")
