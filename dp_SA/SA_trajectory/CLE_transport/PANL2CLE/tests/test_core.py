from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dp_SA.SA_trajectory.CLE_transport.PANL2CLE.hooks import WindowSwapHook
from dp_SA.SA_trajectory.CLE_transport.PANL2CLE.io_utils import atomic_bf16_npz, load_bf16_npz
from dp_SA.SA_trajectory.CLE_transport.PANL2CLE.matching import eligible_edge, match_donors
from dp_SA.SA_trajectory.CLE_transport.PANL2CLE.selection import side_from_class
from dp_SA.SA_trajectory.CLE_transport.PANL2CLE.statistics import bh_fdr, donor_contrast
from dp_SA.SA_trajectory.CLE_transport.PANL2CLE.windows import WINDOW_SPECS, locate_swap_windows


class CharTokenizer:
    image_id = 1_000_000
    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}
    def convert_tokens_to_ids(self, token): return self.image_id
    def decode(self, ids, **_kwargs): return "".join(chr(int(x)) for x in ids)


def rendered_prompt():
    return "prefix sufficiently long\n" + "\n\n".join(spec[0] for spec in WINDOW_SPECS.values()) + "\nend"


def test_six_windows_have_exact_boundaries_and_length():
    rendered = rendered_prompt(); tokenizer = CharTokenizer(); inputs = {"input_ids": torch.tensor([[ord(c) for c in rendered]])}
    windows = locate_swap_windows(tokenizer, rendered, inputs)
    assert list(windows) == list(WINDOW_SPECS)
    for name, row in windows.items():
        assert row["window_length"] == 8
        assert row["processed_indices"] == list(range(row["processed_start"], row["processed_end"] + 1))
        if WINDOW_SPECS[name][1] == "newline": assert row["decoded_tokens"][-1] == "\n"
        else: assert row["decoded_tokens"][-1] == "."


def test_duplicate_anchor_rejected():
    rendered = rendered_prompt() + WINDOW_SPECS["W1"][0]
    with pytest.raises(ValueError, match="expected one anchor"):
        locate_swap_windows(CharTokenizer(), rendered, {"input_ids": torch.tensor([[ord(c) for c in rendered]])})


def test_bf16_round_trip(tmp_path):
    value = torch.randn(8, 17).bfloat16(); path = tmp_path / "bits.npz"
    metadata = atomic_bf16_npz(path, {"W1__L12": value}); restored, row = load_bf16_npz(path, "W1__L12")
    assert restored.dtype == torch.bfloat16 and torch.equal(value, restored)
    assert row["bits_sha256"] == metadata["W1__L12"]["bits_sha256"]


def test_window_hook_changes_only_eight_rows():
    layer = torch.nn.Identity(); modules = SimpleNamespace(language_layers=[layer], hidden_size=5, num_hidden_layers=1)
    source = torch.randn(8, 5).bfloat16(); original = torch.randn(1, 12, 5).bfloat16()
    hook = WindowSwapHook(modules, layer=0, positions=list(range(2, 10)), source=source, prefill_length=12)
    with hook: result = layer(original.clone())
    assert torch.equal(result[0, 2:10], source)
    assert torch.equal(result[0, :2], original[0, :2]) and torch.equal(result[0, 10:], original[0, 10:])
    assert hook.diagnostics()["outside_exact"]


def _row(case, side, answer="red", shift=0):
    return {"case_id": case, "family_id": "f" + case, "item_id": "i" + case, "image_sha256": "h" + case,
            "phase0_raw_answer": answer, "template_sha256": "t", "sa_side": side, "soft_sa_image_score": .7 if side == "high_image" else .3,
            "image_token_count": 100 + shift, "sequence_length": 300 + shift,
            "windows": {name: {"processed_start": 200 + shift, "token_ids": list(range(8))} for name in WINDOW_SPECS}}


def test_matching_is_global_and_token_strict():
    recipients = [_row("r1", "high_image"), _row("r2", "high_text", shift=2)]
    donors = [_row("dh1", "high_image", shift=1), _row("dh2", "high_image", shift=3),
              _row("dl1", "high_text", shift=1), _row("dl2", "high_text", shift=3)]
    pairs, audit = match_donors(recipients, donors)
    assert len(pairs) == 4 and audit["max_donor_reuse"] == 1
    broken = _row("bad", "high_image"); broken["windows"]["W3"]["token_ids"][-1] = 99
    assert not eligible_edge(recipients[0], broken)


def test_side_and_statistics():
    assert [side_from_class(x) for x in (0, 3, 4, 5, 8)] == ["high_text", "high_text", "balanced", "high_image", "high_image"]
    assert np.allclose(bh_fdr([.01, .04, .03]), [.03, .04, .04])
    rows = []
    for side, base in (("high_image", .6), ("high_text", .4)):
        case = side
        for donor, value in (("high_image", base + .1), ("high_text", base - .1)):
            rows.append({"window": "W1", "layer": 12, "case_id": case, "family_id": case, "recipient_side": side,
                         "donor_side": donor, "patched_soft_sa": value})
    result = donor_contrast(rows, repeats=100, seed=42)
    combined = next(row for row in result if row["stratum"] == "combined_equal_side")
    assert combined["estimate"] == pytest.approx(.2)

