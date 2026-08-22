from __future__ import annotations

import pytest
import torch

from layer_metacognition.model_adapter import LanguageModules
from layer_metacognition.sa_patching import POSITION_LAYERS
from layer_metacognition.sa_patching.artifacts import (
    filter_evaluation_records,
    ragged_position_mean,
)
from layer_metacognition.sa_patching.runner import (
    SAPatchingRunner,
    build_summary,
    logit_difference,
    metric_bundle,
    position_layer_grid,
    recovery_value,
)
from layer_metacognition.sa_patching.sa_patching_hook import (
    ActivationReplacementHook,
    EmbeddingReplacement,
    EmbeddingReplacementHook,
    PatchingInvariantError,
    ResidualActivationCacheHook,
)


class KeywordLanguageModel(torch.nn.Module):
    def forward(self, *, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return inputs_embeds


class TensorLayer(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


def _modules(layer_count: int = 3, hidden_size: int = 4) -> LanguageModules:
    return LanguageModules(
        language_layers=[TensorLayer() for _ in range(layer_count)],
        final_norm=torch.nn.Identity(),
        lm_head=torch.nn.Linear(hidden_size, 9, bias=False),
        hidden_size=hidden_size,
        num_hidden_layers=layer_count,
    )


def test_embedding_hook_replaces_only_requested_span_and_applies_once() -> None:
    language = KeywordLanguageModel()
    clean = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    replacement = torch.full((2, 4), -3.0)
    hook = EmbeddingReplacementHook(
        language,
        replacements=[EmbeddingReplacement("text", (1, 3), replacement)],
        prefill_sequence_length=5,
        hidden_size=4,
        capture_clean=True,
    )
    with hook:
        output = language(inputs_embeds=clean)
        cached = language(inputs_embeds=torch.zeros(1, 1, 4))
    assert torch.equal(output[0, 1], replacement[0])
    assert torch.equal(output[0, 3], replacement[1])
    assert torch.equal(output[0, [0, 2, 4]], clean[0, [0, 2, 4]])
    assert torch.equal(cached, torch.zeros(1, 1, 4))
    diagnostics = hook.diagnostics()
    assert diagnostics["applied_count"] == 1
    assert diagnostics["hook_count"] == 2
    assert diagnostics["replacement_l2"]["text"] > 0
    assert torch.equal(hook.clean_embeddings, clean[0])
    assert not language._forward_pre_hooks


def test_embedding_hook_rejects_shape_overlap_and_identical_corruption() -> None:
    language = KeywordLanguageModel()
    with pytest.raises(PatchingInvariantError, match="hidden shape mismatch"):
        EmbeddingReplacementHook(
            language,
            replacements=[EmbeddingReplacement("bad", (0, 1), torch.ones(1, 4))],
            prefill_sequence_length=2,
            hidden_size=4,
        )
    with pytest.raises(PatchingInvariantError, match="overlap"):
        EmbeddingReplacementHook(
            language,
            replacements=[
                EmbeddingReplacement("a", (0,), torch.ones(1, 4)),
                EmbeddingReplacement("b", (0,), torch.ones(1, 4)),
            ],
            prefill_sequence_length=2,
            hidden_size=4,
        )
    hook = EmbeddingReplacementHook(
        language,
        replacements=[EmbeddingReplacement("same", (0,), torch.zeros(1, 4))],
        prefill_sequence_length=2,
        hidden_size=4,
    )
    with pytest.raises(PatchingInvariantError, match="identical"):
        with hook:
            language(inputs_embeds=torch.zeros(1, 2, 4))
    assert not language._forward_pre_hooks


def test_residual_cache_and_patch_touch_one_post_block_token() -> None:
    modules = _modules()
    hidden = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    cache = ResidualActivationCacheHook(
        modules,
        targets={1: {"ac": 1, "sac": 4}},
        prefill_sequence_length=5,
    )
    with cache:
        modules.language_layers[1](hidden)
        modules.language_layers[1](torch.zeros(1, 1, 4))
    cache.validate()
    assert torch.equal(cache.cache[1]["ac"], hidden[0, 1])
    assert torch.equal(cache.cache[1]["sac"], hidden[0, 4])

    source = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    patch = ActivationReplacementHook(
        modules,
        layer=1,
        position=2,
        source_hidden=source,
        prefill_sequence_length=5,
    )
    with patch:
        output = modules.language_layers[1](hidden)
        modules.language_layers[1](torch.zeros(1, 1, 4))
    assert torch.equal(output[0, 2], source)
    assert torch.equal(output[0, [0, 1, 3, 4]], hidden[0, [0, 1, 3, 4]])
    diagnostics = patch.diagnostics()
    assert diagnostics["applied_count"] == 1
    assert diagnostics["hook_count"] == 2
    assert diagnostics["site"] == "decoder_block_output_post_mlp_residual"
    assert not modules.language_layers[1]._forward_hooks


def test_ragged_position_mean_uses_only_covering_sources() -> None:
    first = torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    second = torch.tensor([[9.0, 90.0]])
    third = torch.tensor([[2.0, 20.0], [7.0, 70.0]])
    mean, counts = ragged_position_mean([first, second, third])
    assert counts.tolist() == [3, 2, 1]
    assert torch.allclose(mean[0], torch.tensor([4.0, 40.0]))
    assert torch.allclose(mean[1], torch.tensor([5.0, 50.0]))
    assert torch.allclose(mean[2], torch.tensor([5.0, 50.0]))


def test_evaluation_filter_keeps_only_conflict_conditions() -> None:
    records = [
        {"case_id": "a", "condition": "consistent_easy"},
        {"case_id": "b", "condition": "conflict_easy"},
        {"case_id": "c", "condition": "conflict_hard"},
    ]
    selected = filter_evaluation_records(
        records,
        ("conflict_easy", "conflict_hard"),
    )
    assert [record["case_id"] for record in selected] == ["b", "c"]
    with pytest.raises(ValueError, match="must be non-empty"):
        filter_evaluation_records(records, ())


def _source(hard: str, soft: float, winner: int) -> dict:
    logits = [0.0] * 9
    logits[winner] = 4.0
    return {
        "hard_label": hard,
        "soft_score": soft,
        "logits": logits,
    }


def test_grid_and_recovery_metrics_preserve_overshoot_and_null_denominator() -> None:
    grid = position_layer_grid(
        ["ac", "panl", "sac"],
        [12, 18],
        POSITION_LAYERS,
    )
    assert grid == [("ac", 12), ("panl", 18), ("sac", 18)]
    assert recovery_value(1.0, 0.0, 2.0) == 2.0
    assert recovery_value(1.0, 1.0, 1.0) is None
    assert logit_difference([4.0, 0.0, 0.0], "0") == 4.0
    bundle = metric_bundle(
        _source("8", 0.9, 8),
        _source("2", 0.2, 2),
        _source("8", 1.0, 8),
    )
    assert bundle["recovery"]["hard_match_recovered"] is True
    assert bundle["recovery"]["corrupt_hard_differs"] is True
    assert bundle["recovery"]["soft"] == pytest.approx(8 / 7)


def test_summary_groups_cells_and_excludes_undefined_recovery() -> None:
    records = [
        {
            "corruption_type": "image_only",
            "position": "ac",
            "layer": 12,
            "status": "completed",
            "soft_delta": 0.2,
            "recovery": {
                "soft": 0.5,
                "logit": 0.25,
                "hard_formula": None,
                "hard_match_recovered": True,
                "corrupt_hard_differs": True,
            },
        },
        {
            "corruption_type": "image_only",
            "position": "ac",
            "layer": 12,
            "status": "failed",
        },
    ]
    summary = build_summary(records, expected_count=2)
    assert summary["completed_count"] == 1
    assert summary["failed_count"] == 1
    cell = summary["cells"][0]
    assert cell["soft_SA_recovery"]["mean"] == 0.5
    assert cell["hard_formula_recovery"]["valid_count"] == 0
    assert cell["hard_SA_recovery"]["rate"] == 1.0


def test_clean_parity_gates_on_labels_and_retains_numeric_diagnostics() -> None:
    runner = object.__new__(SAPatchingRunner)
    runner.parity_tolerance = 0.125
    original = {
        "parsed_label": "5",
        "hard_label": "6",
        "soft_score": 0.7,
        "logits": [float(index) for index in range(9)],
    }
    clean = {
        "parsed_label": "5",
        "hard_label": "6",
        "soft_score": 0.68,
        "logits": [float(index) + 0.75 for index in range(9)],
    }
    diagnostics = runner._validate_clean_parity(original, clean)
    assert diagnostics["passed"] is True
    assert diagnostics["numeric_close_at_reconstruction_tolerance"] is False
    changed = dict(clean, hard_label="5")
    with pytest.raises(RuntimeError, match="Clean teacher-forced parity failed"):
        runner._validate_clean_parity(original, changed)
