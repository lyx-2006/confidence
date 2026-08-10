from __future__ import annotations

from collections import Counter
import json

import pytest
import torch

from layer_metacognition.model_adapter import (
    HiddenStateReplacement,
    HiddenStateReplacementHook,
    LanguageModules,
)
from layer_metacognition.run_teacher_forced_source_origin import select_smoke_cohort
from layer_metacognition.teacher_forced_source_origin import (
    StreamingStateStore,
    TeacherForcedSourceOriginRunner,
    aligned_answer_force_delta,
    build_summary,
    configuration_fingerprint,
    donor_metrics,
    forced_assistant_text,
    select_balanced_cohort,
    select_state_pairs,
    self_swap_validation,
    state_replacements,
)


class TensorLayer(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class TupleLayer(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, str]:
        return hidden, "cache"


def _modules(layers: list[torch.nn.Module], hidden_size: int = 3) -> LanguageModules:
    return LanguageModules(
        language_layers=layers,
        final_norm=torch.nn.Identity(),
        lm_head=torch.nn.Linear(hidden_size, 5, bias=False),
        hidden_size=hidden_size,
        num_hidden_layers=len(layers),
    )


def test_replacement_hook_supports_multiple_spans_and_cached_decode() -> None:
    layer = TensorLayer()
    modules = _modules([layer])
    hidden = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
    hook = HiddenStateReplacementHook(
        modules,
        prefill_sequence_length=6,
        replacements=[
            HiddenStateReplacement("image", 0, (1, 2), torch.full((2, 3), 9.0)),
            HiddenStateReplacement("text", 0, (4,), torch.tensor([1.0, 2.0, 3.0])),
        ],
    )
    with hook:
        patched = layer(hidden)
        cached = layer(torch.zeros(1, 1, 3))
        second_prefill = layer(hidden)
    assert torch.equal(patched[0, 1:3], torch.full((2, 3), 9.0))
    assert torch.equal(patched[0, 4], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(patched[0, 0], hidden[0, 0])
    assert torch.equal(cached, torch.zeros(1, 1, 3))
    assert torch.equal(second_prefill, hidden)
    diagnostics = hook.diagnostics()
    assert diagnostics["applied_count"] == {0: 1}
    assert diagnostics["hook_call_count"] == {0: 3}
    assert not layer._forward_hooks


def test_replacement_hook_preserves_tuple_output() -> None:
    layer = TupleLayer()
    hook = HiddenStateReplacementHook(
        _modules([layer]),
        prefill_sequence_length=2,
        replacements=[
            HiddenStateReplacement("token", 0, (0,), torch.ones(3)),
        ],
    )
    with hook:
        output, cache = layer(torch.zeros(1, 2, 3))
    assert cache == "cache"
    assert torch.equal(output[0, 0], torch.ones(3))


def test_replacement_rejects_length_mismatch_and_overlap() -> None:
    modules = _modules([TensorLayer()])
    with pytest.raises(ValueError, match="span length mismatch"):
        HiddenStateReplacementHook(
            modules,
            prefill_sequence_length=3,
            replacements=[
                HiddenStateReplacement("bad", 0, (0, 1), torch.ones(1, 3)),
            ],
        )
    with pytest.raises(ValueError, match="overlapping"):
        HiddenStateReplacementHook(
            modules,
            prefill_sequence_length=3,
            replacements=[
                HiddenStateReplacement("left", 0, (0, 1), torch.ones(2, 3)),
                HiddenStateReplacement("right", 0, (1,), torch.ones(1, 3)),
            ],
        )


def test_panl_clamp_builds_every_downstream_layer() -> None:
    positions = {"ac": {"position": 0}, "panl": {"position": 2}}
    recipient = {
        "ac": torch.zeros(4, 1, 3),
        "panl": torch.zeros(4, 1, 3),
    }
    donor = {
        "ac": torch.ones(4, 1, 3),
        "panl": torch.ones(4, 1, 3),
    }
    replacements = state_replacements(
        layer=1,
        intervention="ac_panl_clamp_clean",
        recipient_positions=positions,
        source_core=donor,
        recipient_core=recipient,
        num_hidden_layers=4,
    )
    assert [(value.layer_index, value.target_positions) for value in replacements] == [
        (1, (0,)),
        (1, (2,)),
        (2, (2,)),
        (3, (2,)),
    ]


def _candidate(side: str, condition: str, index: int) -> dict:
    return {
        "case_id": f"{side}|{condition}|{index}",
        "item_id": f"{side}-{condition}-{index}",
        "item_order": index,
        "prior_index": 0,
        "condition": condition,
        "decision_side": side,
        "normalized_answer": f"answer-{index % 3}",
        # A deliberately unique SA-like value proves selection does not need it.
        "free_generation": {"soft_image_score": index / 100.0},
    }


def test_balanced_cohort_has_25_per_cell_and_maximal_diversity() -> None:
    candidates = [
        _candidate(side, condition, index)
        for side in ("follows_text", "follows_image")
        for condition in ("conflict_easy", "conflict_hard")
        for index in range(30)
    ]
    selected = select_balanced_cohort(candidates, cases_per_cell=25)
    counts = Counter((row["decision_side"], row["condition"]) for row in selected)
    assert set(counts.values()) == {25}
    assert len(selected) == 100
    assert len({row["case_id"] for row in selected}) == 100


def _clean_case(
    case_id: str,
    *,
    item: str,
    condition: str,
    answer: str,
    side: str,
    sa: float,
) -> dict:
    return {
        "case_id": case_id,
        "item_id": item,
        "item_order": int(case_id[-1]),
        "prior_index": int(case_id[-1]),
        "condition": condition,
        "normalized_answer": answer,
        "decision_side": side,
        "clean_teacher_forced": {"soft_image_score": sa},
    }


def test_state_pairs_enforce_threshold_tiers_and_no_case_reuse() -> None:
    cases = [
        _clean_case("c0", item="i", condition="conflict_easy", answer="red", side="follows_text", sa=0.1),
        _clean_case("c1", item="i", condition="conflict_easy", answer="red", side="follows_text", sa=0.4),
        _clean_case("c2", item="j", condition="conflict_easy", answer="red", side="follows_text", sa=0.8),
        _clean_case("c3", item="k", condition="conflict_hard", answer="red", side="follows_text", sa=0.0),
    ]
    pairs = select_state_pairs(cases, min_sa_gap=0.15, max_pairs=2)
    assert pairs[0]["match_tier"] == 0
    used = [value for pair in pairs for value in (pair["low_case_id"], pair["high_case_id"])]
    assert len(used) == len(set(used))
    assert all(pair["sa_gap"] >= 0.15 for pair in pairs)


def test_smoke_cohort_contains_pair_and_all_cells() -> None:
    candidates = [
        _candidate(side, condition, index)
        for side in ("follows_text", "follows_image")
        for condition in ("conflict_easy", "conflict_hard")
        for index in range(3)
    ]
    # Ensure a same-answer/same-side pair exists across two rows.
    candidates[0]["normalized_answer"] = "shared"
    candidates[1]["normalized_answer"] = "shared"
    selected = select_smoke_cohort(candidates)
    cells = {(row["decision_side"], row["condition"]) for row in selected}
    assert len(cells) == 4
    assert any(
        left["normalized_answer"] == right["normalized_answer"]
        and left["decision_side"] == right["decision_side"]
        for index, left in enumerate(selected)
        for right in selected[index + 1 :]
    )


def test_metrics_and_forced_wire() -> None:
    assert forced_assistant_text("blue") == (
        "**Answer**: blue\n**Source Attribution**:"
    )
    assert aligned_answer_force_delta(0.2, "image") == pytest.approx(0.2)
    assert aligned_answer_force_delta(-0.2, "text") == pytest.approx(0.2)
    donor = donor_metrics(recipient_sa=0.2, patched_sa=0.5, donor_sa=0.8)
    assert donor["directional"] is True
    assert donor["donor_pull"] is True
    clean = {
        "hard_label": "4",
        "soft_image_score": 0.5,
        "class_probabilities": [0.2, 0.8],
    }
    assert self_swap_validation(clean, dict(clean))["passed"] is True


def test_streaming_state_store_deletes_tensors_but_keeps_index(tmp_path) -> None:
    store = StreamingStateStore(tmp_path)
    core = {"ac": torch.ones(2, 1, 3)}
    spans = {"image": torch.ones(2, 4, 3)}
    store.save(context_id="case", metadata={"layer_indices": [0, 1]}, core=core, spans=spans)
    assert store.exists("case", "core")
    assert store.load("case", "spans")["image"].dtype == torch.float16
    store.delete("case", "spans", consumed_by="test")
    store.delete("case", "core", consumed_by="test")
    assert store.live_tensor_files() == []
    assert store.index["contexts"]["case"]["core"]["deleted"] is True
    assert store.index_path.is_file()


def test_summary_separates_controls_and_causal_rows() -> None:
    records = [
        {
            "experiment": "answer_force",
            "decision_side": "follows_text",
            "difficulty": "easy",
            "delta_sa": 0.2,
            "aligned_delta_sa": 0.2,
            "directional": True,
            "hard_flip": True,
            "donor_pull": None,
            "teacher_forcing_calibration_delta": 0.1,
        },
        {
            "experiment": "evidence_swap",
            "is_control": True,
            "status": "completed",
            "self_swap_validation": {
                "passed": True,
                "soft_abs_error": 0.0,
                "class_probability_max_abs_error": 0.0,
            },
        },
    ]
    summary = build_summary(records)
    assert summary["answer_force"]["groups"][0]["count"] == 1
    assert summary["controls"]["pass_fraction"]["fraction"] == 1.0


def test_completed_resume_returns_summary_without_recapture(tmp_path) -> None:
    configuration = {"format_version": 1, "output_dir": str(tmp_path)}
    fingerprint = configuration_fingerprint(configuration)
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            configuration
            | {"config_fingerprint": fingerprint, "status": "complete"}
        ),
        encoding="utf-8",
    )
    cohort = {"format_version": 1, "cases": []}
    (tmp_path / "cohort_manifest.json").write_text(
        json.dumps(cohort), encoding="utf-8"
    )
    (tmp_path / "pair_manifest.json").write_text(
        json.dumps({"format_version": 1, "evidence_swap": [], "state_swap": []}),
        encoding="utf-8",
    )
    expected = {"format_version": 1, "record_count": 0}
    (tmp_path / "summary.json").write_text(json.dumps(expected), encoding="utf-8")
    runner = TeacherForcedSourceOriginRunner(
        inference=None,
        modules=None,
        joint_generator=None,
        source_analyzer=None,
        source_variant=None,
        output_dir=tmp_path,
        configuration=configuration,
        cohort_manifest=cohort,
        layers=[0],
        resume=True,
    )
    assert runner.execute() == expected
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["stage"] == "complete"
