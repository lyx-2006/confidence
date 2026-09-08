from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dp_SA.real_sa.analyze import combine_cases, corruption_diagnostics
from dp_SA.real_sa.artifacts import GroupedFloat64Mean, MeanArtifacts, RaggedFloat64Mean, image_shape_key
from dp_SA.real_sa.data import FrozenCohort, load_frozen_cohort
from dp_SA.real_sa.hooks import EmbeddingReplacement, EmbeddingReplacementHook, ReplacementInvariantError
from dp_SA.real_sa.metrics import case_metrics, family_cluster_bootstrap, restricted_probabilities, sign_type
from dp_SA.real_sa.run import smoke_records
from dp_SA.real_sa.scoring import (
    condition_components, run_scores, scores_from_vocab, teacher_forced_log_probability,
)


class KeywordLanguage(torch.nn.Module):
    def forward(self, *, inputs_embeds):
        return inputs_embeds


def test_frozen_cohort_balance_and_zero_leakage():
    cohort = load_frozen_cohort()
    assert cohort.audit["test_answer_side_counts"] == {"follow_image": 50, "follow_text": 50}
    assert set(cohort.audit["test_cells"].values()) == {25}
    assert cohort.audit["text_donor_context_conditions"] == {"conflict_easy": 100, "conflict_hard": 100}
    assert cohort.audit["overlaps"] == {"case": 0, "family": 0, "item": 0, "image_hash": 0}
    assert len({row["donor_id"] for row in cohort.image_donors}) == 200
    assert len({row["donor_id"] for row in cohort.text_donors}) == 200


def test_smoke_extremes_cover_conditions_and_answer_sides():
    selected = smoke_records(load_frozen_cohort().tests)
    assert len(selected) == 4
    assert {row["condition"] for row in selected} == {"conflict_easy", "conflict_hard"}
    assert {row["answer_side"] for row in selected} == {"follow_image", "follow_text"}


def test_condition_mapping_is_not_easy_hard_mapping():
    assert condition_components("clean") == ()
    assert condition_components("10_text_corrupt") == ("text",)
    assert condition_components("01_image_corrupt") == ("image",)
    assert condition_components("00_both_corrupt") == ("image", "text")


def test_joint_replacement_lambda_and_non_target_exactness():
    language = KeywordLanguage()
    clean = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3)
    replacements = [
        EmbeddingReplacement("image", (1, 2), torch.full((2, 3), -1.0)),
        EmbeddingReplacement("text", (5, 6), torch.full((2, 3), -2.0)),
    ]
    hook = EmbeddingReplacementHook(language, replacements=replacements, prefill_sequence_length=10, hidden_size=3)
    with hook:
        output = language(inputs_embeds=clean)
    diagnostics = hook.diagnostics()
    assert diagnostics["applied_count"] == 1 and diagnostics["replacement_count"] == 2
    assert diagnostics["non_target_equal"] is True
    assert torch.equal(output[0, [0, 3, 4, 7, 8, 9]], clean[0, [0, 3, 4, 7, 8, 9]])
    zero = EmbeddingReplacementHook(language, replacements=replacements, prefill_sequence_length=10,
                                    hidden_size=3, interpolation=0.0)
    with zero:
        unchanged = language(inputs_embeds=clean)
    assert torch.equal(unchanged, clean)
    assert set(zero.diagnostics()["replacement_l2"].values()) == {0.0}


def test_hook_rejects_overlap_and_shape_mismatch():
    language = KeywordLanguage()
    with pytest.raises(ReplacementInvariantError, match="overlap"):
        EmbeddingReplacementHook(language, replacements=[
            EmbeddingReplacement("image", (1, 2), torch.ones(2, 3)),
            EmbeddingReplacement("text", (2,), torch.ones(1, 3)),
        ], prefill_sequence_length=4, hidden_size=3)
    with pytest.raises(ReplacementInvariantError, match="shape"):
        EmbeddingReplacementHook(language, replacements=[
            EmbeddingReplacement("image", (1, 2), torch.ones(1, 3)),
        ], prefill_sequence_length=4, hidden_size=3)


def test_image_shape_key_is_strict():
    inputs = {"image_grid_thw": torch.tensor([[1, 72, 72]])}
    first, meta = image_shape_key(inputs, (1296, 3584))
    second, _ = image_shape_key(inputs, (1296, 4096))
    assert first != second and meta["feature_shape"] == [1296, 3584]


def test_mean_accumulators_use_float64_and_ragged_position_counts():
    grouped = GroupedFloat64Mean()
    grouped.add("shape", torch.tensor([[1.0], [3.0]], dtype=torch.bfloat16))
    grouped.add("shape", torch.tensor([[3.0], [5.0]], dtype=torch.float32))
    assert grouped.sums["shape"].dtype == torch.float64
    assert grouped.means()["shape"].squeeze().tolist() == [2.0, 4.0]
    ragged = RaggedFloat64Mean()
    ragged.add(torch.tensor([[1.0], [5.0]]))
    ragged.add(torch.tensor([[3.0]]))
    mean, counts = ragged.mean()
    assert mean.squeeze().tolist() == [2.0, 5.0]
    assert counts.tolist() == [2, 1]


def test_single_and_multitoken_candidate_scoring_helpers():
    logits = torch.tensor([1.0, 2.0, 3.0])
    assert scores_from_vocab(logits, ["a", "b"], {"a": [0], "b": [2]}) == [1.0, 3.0]
    with pytest.raises(ValueError, match="multi-token"):
        scores_from_vocab(logits, ["a"], {"a": [0, 1]})
    rows = {4: torch.tensor([0.0, 2.0]), 5: torch.tensor([3.0, 0.0])}
    expected = float(torch.log_softmax(rows[4].double(), -1)[1] + torch.log_softmax(rows[5].double(), -1)[0])
    assert teacher_forced_log_probability(rows, [4, 5], [1, 0]) == pytest.approx(expected)


def test_probabilities_shapley_efficiency_sign_and_ratio_rules():
    p = restricted_probabilities(list(range(12)))
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    values = case_metrics(.8, .6, .5, .2)
    assert values["phi_sum"] == pytest.approx(.6)
    assert values["efficiency_error"] <= 1e-10
    assert values["R_I_eligible"] and 0 <= values["R_I"] <= 1
    suppress = case_metrics(.4, .5, .2, .3)
    assert suppress["R_I"] is None
    assert sign_type(.2, -.19) == "weak_total_effect"
    assert sign_type(.2, -.1) == "image_support_text_suppress"


def test_family_bootstrap_resamples_whole_clusters():
    rows = [{"family_id": "a", "x": 1.0}, {"family_id": "a", "x": 3.0},
            {"family_id": "b", "x": 9.0}]
    metric = lambda sample: float(np.mean([row["x"] for row in sample]))
    first = family_cluster_bootstrap(rows, metric, repeats=200, seed=42)
    second = family_cluster_bootstrap(rows, metric, repeats=200, seed=42)
    assert first == second and first["valid"] == 200


def _source_row(case_id: str = "case"):
    return {
        "case_id": case_id, "family_id": "family", "item_id": "1", "condition": "conflict_easy",
        "phase0_normalized_answer": "red", "soft_sa_image_score": .5, "answer_side": "follow_image",
    }


def test_case_join_and_corruption_diagnostics():
    source = _source_row()
    cohort = FrozenCohort([source], [], [], {})
    probabilities = {"clean": .8, "10_text_corrupt": .6, "01_image_corrupt": .5, "00_both_corrupt": .2}
    scores = []
    for condition, value in probabilities.items():
        scores.append({
            "case_id": "case", "corruption_condition": condition, "fixed_answer_probability": value,
            "dataset_condition": "conflict_easy", "condition_argmax_answer": "red" if condition == "clean" else "blue",
            "probability_sum": 1.0, "replacement_diagnostics": {"replacement_l2": {"image": 1.0, "text": 2.0}},
        })
    joined = combine_cases(scores, cohort)
    assert len(joined) == 1 and joined[0]["v10"] == .6
    diagnostics = corruption_diagnostics(scores)
    assert len(diagnostics) == 12


def test_resume_skips_all_model_work(monkeypatch, tmp_path):
    import dp_SA.real_sa.scoring as scoring

    row = {**_source_row(), "answer_classes": [f"c{i}" for i in range(12)]}
    row["phase0_normalized_answer"] = "c0"
    ids = {name: [index] for index, name in enumerate(row["answer_classes"])}
    preflight = {"policy": "single_token_next_token_logits", "candidate_token_ids": {"case": ids}}
    artifacts = MeanArtifacts({}, torch.zeros(1, 3), torch.ones(1, dtype=torch.long), {
        "artifact_sha256": {"image": "i", "text": "t"}, "fingerprint": "artifact",
    })
    calls = {"parity": 0, "score": 0}
    def fake_parity(_inference, _row, _ids, fingerprint):
        calls["parity"] += 1
        return {"parity_key": "case", "case_id": "case", "run_fingerprint": fingerprint,
                "status": "passed", "generated_token_count": 2, "candidate_scores": list(range(12)),
                "candidate_token_ids": ids, "input_details": {"image_positions": [0], "text_positions": [1]}}
    def fake_score(_inference, _row, condition, _ids, _artifacts, _hidden, interpolation=1.0):
        calls["score"] += 1
        return list(range(12)), {"image_positions": [0], "text_positions": [1]}, {
            "replacement_l2": {name: interpolation for name in condition_components(condition)},
        }
    monkeypatch.setattr(scoring, "_generate_parity", fake_parity)
    monkeypatch.setattr(scoring, "_single_score", fake_score)
    monkeypatch.setattr(scoring, "resolve_language_modules", lambda _model: SimpleNamespace(hidden_size=3))
    inference = SimpleNamespace(model=object())
    first = run_scores(inference, [row], artifacts, preflight, tmp_path, "run", smoke=True)
    before = dict(calls)
    second = run_scores(inference, [row], artifacts, preflight, tmp_path, "run", smoke=True)
    assert first["new_logical_evaluations"] == 5
    assert second["resumed_noop"] and second["new_internal_model_forwards"] == 0
    assert calls == before
