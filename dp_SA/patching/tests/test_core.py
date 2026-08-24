from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from layer_metacognition.model_adapter import LanguageModules
from layer_metacognition.sa_patching.artifacts import ragged_position_mean

from dp_SA.patching.artifacts import grouped_shape_means, image_shape_key, length_conditioned_means
from dp_SA.patching.hooks import (
    ActivationReplacementHook, EmbeddingReplacement, EmbeddingReplacementHook,
    EmptyActivationHook, PatchingInvariantError, ResidualActivationCacheHook,
)
from dp_SA.patching.io import atomic_jsonl, atomic_torch_save, load_jsonl_strict
from dp_SA.patching.lock import FormalRunLock
from dp_SA.patching.metrics import (
    bh_fdr, bootstrap_ratio, oriented, probabilities, recovery, score_logits, sign_flip_p,
)
from dp_SA.patching.run import _build_replacements
from dp_SA.patching.selection import select_calibration_manifest, select_evaluation_manifest, stable_case_key


def _row(item: int, side: str, hard: int, soft: float, image: Path, *, condition: str = "conflict_easy", prior: int = 0):
    return {
        "status": "completed", "valid_class": True,
        "case_id": f"{item}-{prior}-{condition}", "item_id": str(item), "prior_index": prior,
        "condition": condition, "version": "v4", "question": "q", "text_clue": "t",
        "image_path": str(image), "phase0_raw_answer": "red", "phase1_inserted_raw_answer": "red",
        "phase1_prompt": "p", "phase1_answer_span": [2, 3],
        "phase1_answer_token_ids": [7],
        "positions": {"P1_PANL": {"processed_index": 3}, "P1_PANL_PLUS_1": {"processed_index": 4},
                      "P1_SAC": {"processed_index": 9}},
        "class_logits": [float(x) for x in range(9)], "argmax_hard_class": hard,
        "soft_sa_image_score": soft, "raw_generated_class": str(hard), "test_side": side,
    }


def test_extreme_selection_is_balanced_unique_and_stable(tmp_path: Path):
    image = tmp_path / "x.png"; image.write_bytes(b"x")
    rows = []
    for item in range(1, 9):
        rows.append(_row(item, "image_side", 5 + (item % 4), .5 + item / 100, image))
        rows.append(_row(item + 20, "text_side", item % 4, .5 - item / 100, image))
    # Duplicate item has a more central row and must not displace the extreme one.
    rows.append(_row(1, "image_side", 5, .99, image, prior=1))
    manifest = select_evaluation_manifest(rows, eval_cases=8)
    selected = manifest["selected"]
    assert len(selected) == 8 and len({row["item_id"] for row in selected}) == 8
    assert sum(row["test_side"] == "image_side" for row in selected) == 4
    assert manifest["candidate_count"] == len(rows)
    assert stable_case_key({"item_id": "2", "prior_index": 0, "condition": "a", "case_id": "x"}) < stable_case_key({"item_id": "10", "prior_index": 0, "condition": "a", "case_id": "x"})


def test_calibration_pools_are_100_unique_and_zero_leakage(tmp_path: Path):
    image = tmp_path / "x.png"; image.write_bytes(b"x")
    rows = []
    for item in range(1, 181):
        side, hard = ("image_side", 5) if item % 2 else ("text_side", 2)
        rows.append(_row(item, side, hard, .6 if hard == 5 else .4, image, condition="conflict_easy"))
        rows.append(_row(item, side, hard, .6 if hard == 5 else .4, image, condition="conflict_hard", prior=1))
    manifest = select_calibration_manifest(rows, evaluation_items={"1", "2"}, seed=42)
    for pool in manifest["pools"].values():
        assert len(pool) == 100 and len({row["item_id"] for row in pool}) == 100
        assert not {row["item_id"] for row in pool} & {"1", "2"}


def test_grouped_image_shape_and_answer_length_means():
    groups = grouped_shape_means([("a", torch.ones(2, 3)), ("a", torch.full((2, 3), 3.0)),
                                  ("b", torch.full((1, 3), 5.0))])
    assert torch.equal(groups["a"], torch.full((2, 3), 2.0))
    assert groups["b"].shape == (1, 3)
    with pytest.raises(ValueError, match="incompatible"):
        grouped_shape_means([("x", torch.ones(1, 2)), ("x", torch.ones(2, 2))])
    means, counts = length_conditioned_means([torch.ones(1, 2), torch.full((1, 2), 3.0), torch.ones(2, 2)])
    assert counts == {1: 2, 2: 1} and torch.equal(means[1], torch.full((1, 2), 2.0))


def test_image_shape_key_includes_grid_feature_and_hidden():
    key1, meta = image_shape_key({"image_grid_thw": torch.tensor([[1, 35, 35]])}, torch.zeros(1225, 8), 8)
    key2, _ = image_shape_key({"image_grid_thw": torch.tensor([[1, 35, 35]])}, torch.zeros(1225, 9), 9)
    assert key1 != key2 and meta["feature_shape"] == [1225, 8]


def test_ragged_text_mean_uses_only_covering_donors():
    mean, counts = ragged_position_mean([
        torch.tensor([[1.0], [3.0], [5.0]]), torch.tensor([[9.0]]), torch.tensor([[2.0], [7.0]])])
    assert counts.tolist() == [3, 2, 1]
    assert mean[:, 0].tolist() == pytest.approx([4.0, 5.0, 5.0])


class KeywordLanguage(torch.nn.Module):
    def forward(self, *, inputs_embeds): return inputs_embeds


class TensorLayer(torch.nn.Module):
    def forward(self, hidden): return hidden


def _modules():
    return LanguageModules([TensorLayer(), TensorLayer()], torch.nn.Identity(), torch.nn.Linear(3, 9, bias=False), 3, 2)


def test_simultaneous_all_embedding_replacement_only_touches_three_spans():
    language = KeywordLanguage(); clean = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
    replacements = [EmbeddingReplacement("image", (0, 1), torch.full((2, 3), -1.0)),
                    EmbeddingReplacement("text", (3,), torch.full((1, 3), -2.0)),
                    EmbeddingReplacement("answer", (6,), torch.full((1, 3), -3.0))]
    hook = EmbeddingReplacementHook(language, replacements=replacements, prefill_sequence_length=8, hidden_size=3)
    with hook: output = language(inputs_embeds=clean)
    assert torch.equal(output[0, [2, 4, 5, 7]], clean[0, [2, 4, 5, 7]])
    assert set(hook.diagnostics()["replacement_l2"]) == {"image", "text", "answer"}
    assert not language._forward_pre_hooks


def test_answer_only_builds_exactly_one_embedding_replacement():
    class Artifacts:
        def replacements_for(self, **_kwargs):
            return torch.ones(2, 3), torch.ones(1, 3), torch.full((1, 3), 7.0)
    spans = {"IMAGE": [0, 2], "TEXT_CLUE": [3, 4], "ANSWER": [6, 7]}
    replacements = _build_replacements(Artifacts(), "shape", spans, corruption="answer_only")
    assert len(replacements) == 1
    assert replacements[0].name == "answer"
    assert replacements[0].positions == (6,)
    assert torch.equal(replacements[0].source, torch.full((1, 3), 7.0))


def test_clean_cache_single_position_layer_patch_and_hook_cleanup(tmp_path: Path):
    modules = _modules(); hidden = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
    cache = ResidualActivationCacheHook(modules, targets={1: {"P1_PANL": 2, "P1_PANL_PLUS_1": 3}}, prefill_sequence_length=6)
    with cache: modules.language_layers[1](hidden)
    cache.validate()
    path = tmp_path / "cache.pt"; atomic_torch_save(path, cache.cache)
    loaded = torch.load(path, weights_only=False)
    source = torch.tensor([-1.0, -2.0, -3.0])
    patch = ActivationReplacementHook(modules, layer=1, position=2, source_hidden=source, prefill_sequence_length=6)
    with patch: output = modules.language_layers[1](hidden)
    assert torch.equal(output[0, 2], source)
    assert torch.equal(output[0, [0, 1, 3, 4, 5]], hidden[0, [0, 1, 3, 4, 5]])
    assert loaded[1]["P1_PANL"].shape == (3,)
    assert not modules.language_layers[1]._forward_hooks


def test_empty_hook_is_exact_and_cleans_up():
    modules = _modules(); hidden = torch.randn(1, 4, 3)
    empty = EmptyActivationHook(modules, layer=0, prefill_sequence_length=4)
    with empty: output = modules.language_layers[0](hidden)
    empty.validate(); assert output is hidden and not modules.language_layers[0]._forward_hooks


def test_soft_hard_midpoint_oriented_margin_entropy_and_recovery():
    logits = [0.0] * 9; logits[8] = 4.0
    score = score_logits(logits)
    assert score["hard_class"] == 8 and score["hard_midpoint"] == .95
    assert 0 < score["entropy"] < np.log(9)
    assert score["fixed_clean_class_margin"] == 4
    assert oriented(.7, "image_side") == pytest.approx(.2)
    assert oriented(.3, "text_side") == pytest.approx(.2)
    assert recovery(1, 0, 2) == 2 and recovery(1, 1, 2) is None


def test_ratio_of_means_bootstrap_zero_denominator_signflip_and_fdr():
    summary, rows = bootstrap_ratio([2, 4], [0, 0], [1, 3], repeats=100, seed=42)
    assert summary["recovery"]["value"] == pytest.approx(2 / 3)
    assert len(rows) == 100 and summary["recovery"]["valid_bootstrap_repeats"] == 100
    zero, _ = bootstrap_ratio([1, 1], [1, 1], [2, 2], repeats=10, seed=1)
    assert zero["recovery"]["value"] is None and zero["recovery"]["valid_bootstrap_repeats"] == 0
    assert sign_flip_p([1, 1, 1, 1], repeats=100, seed=1) < .2
    q = bh_fdr([.01, .04, .2]); assert q[0] <= q[1] <= q[2]


def test_atomic_jsonl_trailing_repair_and_lock(tmp_path: Path):
    path = tmp_path / "rows.jsonl"; atomic_jsonl(path, [{"a": 1}])
    with path.open("a") as handle: handle.write("{bad")
    assert load_jsonl_strict(path, repair_trailing=True) == [{"a": 1}]
    lock_path = tmp_path / "global.lock"
    with FormalRunLock(tmp_path / "one", lock_path=lock_path):
        with pytest.raises(RuntimeError, match="active"):
            with FormalRunLock(tmp_path / "two", lock_path=lock_path): pass
    assert not (tmp_path / "one" / "active.pid").exists()
