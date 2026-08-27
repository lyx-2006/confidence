from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dp_SA.panl_information.build_manifest import build_manifest
from dp_SA.panl_information.io_utils import assert_fingerprint, load_jsonl, safe_remove_temp_tree
from dp_SA.panl_information.metrics import (
    bh_fdr, calibration_metrics, candidate_metrics, difficulty_factors,
    expected_calibration_error, model_perceived_difficulty,
    restricted_distribution,
)
from dp_SA.panl_information.score_unimodal import (
    _score_sequences, _score_single, candidate_suffix_ids, unique_specs,
)


def test_restricted_softmax_and_entropy_endpoints() -> None:
    probabilities = restricted_distribution(np.zeros(12))
    assert probabilities.sum() == pytest.approx(1.0)
    assert model_perceived_difficulty(probabilities) == pytest.approx(100.0)
    degenerate = restricted_distribution([0.0] + [-1000.0] * 11)
    assert model_perceived_difficulty(degenerate) < 1e-6
    result = candidate_metrics([f"c{i}" for i in range(12)], list(range(12)), "c11")
    assert result["predicted_answer"] == "c11"
    assert result["correct"] is True
    assert result["brier_score"] >= 0


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(value) for value in text]


def test_candidate_suffix_single_and_multi_token() -> None:
    tokenizer = _Tokenizer()
    assert candidate_suffix_ids(tokenizer, "prefill:", "r") == [ord("r")]
    assert candidate_suffix_ids(tokenizer, "prefill:", "red") == [ord("r"), ord("e"), ord("d")]


def test_single_and_sequence_teacher_forced_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    import dp_SA.panl_information.score_unimodal as module

    inference = SimpleNamespace(processor=object(), model=object())
    monkeypatch.setattr(module, "model_input_device", lambda _inference: torch.device("cpu"))
    monkeypatch.setattr(module, "prepare_multimodal_inputs", lambda *_args, **_kwargs: SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]])))
    logits = torch.zeros(10); logits[4] = 2.0; logits[5] = 1.0
    monkeypatch.setattr(module, "run_logits_forward", lambda *_args, **_kwargs: {2: logits})
    values = _score_single(inference, object(), {"answer_classes": ["a", "b"]}, "base", [], {"a": [4], "b": [5]})
    assert values[0] > values[1]

    def inputs(_processor: object, _messages: object, rendered: str, **_kwargs: object) -> SimpleNamespace:
        length = {"base": 3, "basea": 4, "basebc": 5}[rendered]
        suffix = {"base": [], "basea": [4], "basebc": [5, 6]}[rendered]
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, *suffix]])[:, :length])
    monkeypatch.setattr(module, "prepare_multimodal_inputs", inputs)
    def forward(_model: object, _inputs: object, positions: list[int], _modules: object) -> dict[int, torch.Tensor]:
        result = {}
        for position in positions:
            value = torch.zeros(10); value[4 if position == 2 else 5 if position == 2 else 6] = 2
            result[position] = value
        return result
    monkeypatch.setattr(module, "run_logits_forward", forward)
    values = _score_sequences(inference, object(), {"answer_classes": ["a", "bc"]}, "base", [], {"a": [4], "bc": [5, 6]})
    assert len(values) == 2 and np.isfinite(values).all()


def test_unique_keys_and_factors() -> None:
    base = {"item_id": "1", "prior_index": 0, "question": "q", "text_clue": "t", "answer_classes": list("abcdefghijkl"), "text_answer": "a", "image_answer": "b", "image_path": __file__}
    rows = [{**base, "condition": "conflict_easy"}, {**base, "condition": "conflict_hard"}]
    specs = unique_specs(rows)
    assert sum(row["modality"] == "text" for row in specs) == 1
    assert sum(row["modality"] == "image" for row in specs) == 2
    assert difficulty_factors(80, 20) == pytest.approx({"d_text": .8, "d_image": .2, "G": .6, "U": .5})


def test_calibration_metrics_and_ece() -> None:
    rows = []
    for index in range(20):
        rows.append({"text_model_perceived_difficulty": float(index), "text_correct": index < 10, "text_max_probability": .9 if index < 10 else .2, "text_multiclass_nll": .1 + index / 10, "text_brier_score": .05 + index / 100})
    result = calibration_metrics(rows, "text")
    assert result["difficulty_error_auroc"] == pytest.approx(1.0)
    assert result["difficulty_correctness_spearman"] < 0
    assert len(result["difficulty_deciles"]) == 10
    assert expected_calibration_error([.9, .1], [True, False]) == pytest.approx(.1)


def test_bh_fdr_monotonic() -> None:
    q = bh_fdr([.001, .02, .03, .9])
    assert q[0] <= q[1] <= q[2] <= q[3]


def test_manifest_actual_counts(tmp_path: Path) -> None:
    root = tmp_path / "run"; (root / "artifacts").mkdir(parents=True)
    result = build_manifest(root)
    assert result == {"eligible_count": 1656, "excluded_count": 2, "item_count": 178, "text_key_count": 828, "image_key_count": 356}
    excluded = load_jsonl(root / "artifacts" / "excluded_records.jsonl")
    assert all(row["reasons"] for row in excluded)


def test_resume_tail_repair_and_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "values.jsonl"; path.write_text('{"a":1}\n{"b":', encoding="utf-8")
    assert load_jsonl(path, repair_trailing=True) == [{"a": 1}]
    assert path.read_text() == '{"a":1}\n'
    config = tmp_path / "config.json"; first = assert_fingerprint(config, {"x": 1}, resume=False)
    assert assert_fingerprint(config, {"x": 1}, resume=True) == first
    with pytest.raises(ValueError): assert_fingerprint(config, {"x": 2}, resume=True)


def test_safe_cleanup_is_narrow(tmp_path: Path) -> None:
    allowed = tmp_path / "output" / "smoke_tmp"; allowed.mkdir(parents=True); (allowed / "x").write_text("x")
    safe_remove_temp_tree(allowed, [allowed]); assert not allowed.exists()
    with pytest.raises(ValueError): safe_remove_temp_tree(tmp_path, [allowed])
