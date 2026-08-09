from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import confidence_test.joint_answer_source_extension as joint_module
import layer_metacognition.steering.decision_side_steering as steering_module
from confidence_test.answer_metrics import AnswerMetricResult
from confidence_test.joint_answer_source_extension import JointAnswerSourceGenerator
from layer_metacognition.model_adapter import (
    AdditiveActivationHook,
    LanguageModules,
    ReinjectingActivationHook,
)
from layer_metacognition.token_positions import locate_marker_in_assistant
from layer_metacognition.steering.decision_side_steering import (
    DECISION_MAPPING,
    BaselineHiddenStateRepository,
    DecisionDirection,
    DirectionRepository,
    SteeringCase,
    build_paired_summary,
    build_baseline_validation,
    build_reused_baseline_record,
    build_steering_vector,
    initialize_output,
    intervention_key,
    manipulation_diagnostics,
    run_intervention,
    select_balanced_decision_side_cases,
    execute_run,
    teacher_forced_assistant_text,
    validate_grid,
)


class TensorLayer(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class TupleLayer(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, str]:
        return hidden, "cache"


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(value) for value in ids)


def _modules(layer: torch.nn.Module, hidden_size: int = 3) -> LanguageModules:
    return LanguageModules(
        language_layers=[layer],
        final_norm=torch.nn.Identity(),
        lm_head=torch.nn.Linear(hidden_size, 5, bias=False),
        hidden_size=hidden_size,
        num_hidden_layers=1,
    )


def test_additive_hook_changes_only_target_and_skips_cached_steps() -> None:
    layer = TensorLayer()
    modules = _modules(layer)
    prompt = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    cached = torch.full((1, 1, 3), 20.0)
    hook = AdditiveActivationHook(
        modules,
        layer_index=0,
        target_position=2,
        steering_vector=torch.tensor([1.0, -2.0, 3.0]),
        prefill_sequence_length=4,
    )
    with hook:
        changed = layer(prompt)
        unchanged_cached = layer(cached)
        unchanged_second_full = layer(prompt)
    expected = prompt.clone()
    expected[0, 2] += torch.tensor([1.0, -2.0, 3.0])
    assert torch.equal(changed, expected)
    assert torch.equal(unchanged_cached, cached)
    assert torch.equal(unchanged_second_full, prompt)
    assert hook.hook_call_count == 3
    assert hook.applied_count == 1


def test_additive_hook_captures_same_token_at_downstream_layers() -> None:
    layers = [TensorLayer(), TensorLayer(), TensorLayer()]
    modules = LanguageModules(
        language_layers=layers,
        final_norm=torch.nn.Identity(),
        lm_head=torch.nn.Linear(3, 5, bias=False),
        hidden_size=3,
        num_hidden_layers=3,
    )
    hidden = torch.zeros(1, 4, 3)
    hook = AdditiveActivationHook(
        modules,
        layer_index=0,
        target_position=2,
        steering_vector=torch.tensor([1.0, 2.0, 3.0]),
        prefill_sequence_length=4,
        capture_layer_indices=[0, 1, 2],
    )
    with hook:
        output = hidden
        for layer in layers:
            output = layer(output)
    captured = hook.trajectory_hidden()
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert set(captured) == {0, 1, 2}
    assert torch.equal(captured[0], expected)
    assert torch.equal(captured[1], expected)
    assert torch.equal(captured[2], expected)
    assert all(len(layer._forward_hooks) == 0 for layer in layers)


def test_block_input_hook_is_visible_to_target_block_and_captures_its_output() -> None:
    class AddTenLayer(torch.nn.Module):
        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return hidden + 10.0

    layer = AddTenLayer()
    modules = _modules(layer)
    hidden = torch.zeros(1, 3, 3)
    hook = AdditiveActivationHook(
        modules,
        layer_index=0,
        target_position=1,
        steering_vector=torch.tensor([1.0, 2.0, 3.0]),
        prefill_sequence_length=3,
        capture_layer_indices=[0],
        injection_site="block_input",
    )
    with hook:
        output = layer(hidden)
    assert torch.equal(output[0, 1], torch.tensor([11.0, 12.0, 13.0]))
    assert torch.equal(hook.h_before, torch.zeros(3))
    assert torch.equal(hook.h_after, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(hook.trajectory_hidden()[0], torch.tensor([11.0, 12.0, 13.0]))
    assert hook.diagnostics()["injection_site"] == "block_input"
    assert not layer._forward_pre_hooks
    assert not layer._forward_hooks


def test_block_input_hook_supports_hidden_states_keyword() -> None:
    class KeywordLayer(torch.nn.Module):
        def forward(self, *, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

    layer = KeywordLayer()
    modules = _modules(layer)
    hook = AdditiveActivationHook(
        modules,
        layer_index=0,
        target_position=0,
        steering_vector=torch.ones(3),
        prefill_sequence_length=2,
        injection_site="block_input",
    )
    with hook:
        output = layer(hidden_states=torch.zeros(1, 2, 3))
    assert torch.equal(output[0, 0], torch.ones(3))


def test_reinjecting_hook_adds_at_each_layer_and_removes_all_hooks() -> None:
    layers = [TensorLayer(), TensorLayer(), TensorLayer()]
    modules = LanguageModules(
        language_layers=layers,
        final_norm=torch.nn.Identity(),
        lm_head=torch.nn.Linear(3, 5, bias=False),
        hidden_size=3,
        num_hidden_layers=3,
    )
    hook = ReinjectingActivationHook(
        modules,
        primary_layer_index=0,
        target_position=1,
        steering_vectors={
            0: torch.tensor([1.0, 0.0, 0.0]),
            1: torch.tensor([1.0, 0.0, 0.0]),
            2: torch.tensor([1.0, 0.0, 0.0]),
        },
        prefill_sequence_length=2,
    )
    with hook:
        hidden = torch.zeros(1, 2, 3)
        for layer in layers:
            hidden = layer(hidden)
        cached = torch.zeros(1, 1, 3)
        for layer in layers:
            cached = layer(cached)
    captured = hook.trajectory_hidden()
    assert torch.equal(captured[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(captured[1], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.equal(captured[2], torch.tensor([3.0, 0.0, 0.0]))
    assert hook.diagnostics()["reinjected_layers"] == [0, 1, 2]
    assert hook.layer_view(2).diagnostics()["steering_applied_count"] == 1
    assert all(not layer._forward_hooks for layer in layers)


def test_generation_ac_locator_anchors_final_assistant_suffix() -> None:
    tokenizer = CharacterTokenizer()
    marker = "**Answer**:"
    token_ids = tokenizer.encode(
        f"User requested output format {marker} <answer>\nAssistant {marker}"
    )
    with pytest.raises(ValueError, match="found 2"):
        locate_marker_in_assistant(
            tokenizer,
            token_ids,
            marker,
            marker,
            name="ac",
        )
    located = locate_marker_in_assistant(
        tokenizer,
        token_ids,
        marker,
        marker,
        name="ac",
        assistant_occurrence="final_suffix",
    )
    assert located["position"] == len(token_ids) - 1
    assert located["token_text"] == ":"


def test_additive_hook_handles_tuple_and_alpha_zero() -> None:
    layer = TupleLayer()
    modules = _modules(layer)
    hidden = torch.randn(1, 2, 3)
    hook = AdditiveActivationHook(
        modules,
        layer_index=0,
        target_position=1,
        steering_vector=torch.zeros(3),
        prefill_sequence_length=2,
    )
    with hook:
        output, cache = layer(hidden)
    assert cache == "cache"
    assert torch.equal(output, hidden)
    assert hook.diagnostics()["injection_l2"] == 0.0


def test_additive_hook_is_removed_after_exception_and_checks_hidden_size() -> None:
    layer = TensorLayer()
    modules = _modules(layer)
    with pytest.raises(ValueError, match="vector size"):
        AdditiveActivationHook(
            modules,
            layer_index=0,
            target_position=0,
            steering_vector=torch.zeros(4),
            prefill_sequence_length=2,
        )
    hook = AdditiveActivationHook(
        modules,
        layer_index=0,
        target_position=0,
        steering_vector=torch.ones(3),
        prefill_sequence_length=2,
    )
    with pytest.raises(RuntimeError, match="boom"):
        with hook:
            raise RuntimeError("boom")
    assert len(layer._forward_hooks) == 0
    original = torch.zeros(1, 2, 3)
    assert torch.equal(layer(original), original)


def test_joint_generator_optional_context_wraps_existing_generation(monkeypatch) -> None:
    state = {"entered": 0, "exited": 0}

    class Tokenizer:
        def decode(self, *_args, **_kwargs) -> str:
            return " blue\n**Source Attribution**:7"

    class Context:
        def __enter__(self):
            state["entered"] += 1
            return self

        def __exit__(self, *_args):
            state["exited"] += 1

    class Model:
        def generate(self, **_kwargs):
            assert state["entered"] == 1
            return SimpleNamespace(
                sequences=torch.tensor([[1, 2, 3]]),
                scores=[torch.zeros(8).reshape(1, 8)],
            )

    class Batch(dict):
        def __getattr__(self, name):
            return self[name]

    tokenizer = Tokenizer()
    inference = SimpleNamespace(
        model=Model(),
        processor=SimpleNamespace(tokenizer=tokenizer),
        _get_inputs_device=lambda: torch.device("cpu"),
    )
    inputs = Batch(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
    )
    monkeypatch.setattr(joint_module, "render_continued_assistant", lambda *_args: "rendered")
    monkeypatch.setattr(
        joint_module,
        "prepare_multimodal_inputs",
        lambda *_args, **_kwargs: inputs,
    )
    monkeypatch.setattr(
        joint_module,
        "parse_joint_answer_source_output",
        lambda *_args, **_kwargs: ("blue", "7", True),
    )
    monkeypatch.setattr(
        joint_module,
        "compute_answer_metrics",
        lambda *_args, **_kwargs: AnswerMetricResult(
            answer_prob=0.8,
            raw_answer_entropy=0.2,
            answer_entropy=0.2,
            answer_class_logits={"blue": 1.0},
            answer_class_probabilities={"blue": 1.0},
            answer_metric_status="completed",
            candidate_count=1,
        ),
    )
    generator = JointAnswerSourceGenerator(inference)
    result = generator.generate(
        "prompt",
        ["blue"],
        None,
        generation_context_factory=lambda actual_inputs, rendered: (
            Context()
            if actual_inputs is inputs and rendered == "rendered"
            else pytest.fail("context factory received the wrong prepared inputs")
        ),
    )
    assert result.parse_success
    assert state == {"entered": 1, "exited": 1}


def _direction() -> DecisionDirection:
    return DecisionDirection(
        file="direction.npz",
        fold=0,
        layer=0,
        position="ac",
        d_raw=np.asarray([3.0, 4.0]),
        d_K=np.asarray([0.6, 0.8]),
        raw_intercept=-0.25,
    )


def test_probe_logit_and_unit_scaling_have_fixed_positive_direction() -> None:
    direction = _direction()
    probe_vector = build_steering_vector(direction, 2.0, "probe_logit")
    unit_vector = build_steering_vector(direction, 2.0, "unit")
    assert np.dot(direction.d_raw, probe_vector) == pytest.approx(2.0)
    assert np.dot(direction.d_raw, unit_vector) == pytest.approx(10.0)
    assert np.dot(direction.d_raw, build_steering_vector(direction, -1, "probe_logit")) < 0


def test_hidden_delta_diagnostics_separate_directional_and_orthogonal_change() -> None:
    direction = DecisionDirection(
        file="direction.npz",
        fold=0,
        layer=0,
        position="ac",
        d_raw=np.asarray([2.0, 0.0]),
        d_K=np.asarray([1.0, 0.0]),
        raw_intercept=0.0,
    )
    diagnostics = steering_module._hidden_delta_diagnostics(
        baseline_hidden=np.asarray([1.0, 2.0]),
        steered_hidden=torch.tensor([4.0, 6.0]),
        direction=direction,
    )
    assert diagnostics["delta_hidden_l2"] == pytest.approx(5.0)
    assert diagnostics["delta_hidden_projection_on_d_K"] == pytest.approx(3.0)
    assert diagnostics["delta_hidden_cosine_with_d_K"] == pytest.approx(0.6)
    assert diagnostics["delta_hidden_orthogonal_l2"] == pytest.approx(4.0)
    assert diagnostics["directional_energy_fraction"] == pytest.approx(0.36)

    zero = steering_module._hidden_delta_diagnostics(
        baseline_hidden=np.asarray([1.0, 2.0]),
        steered_hidden=np.asarray([1.0, 2.0]),
        direction=direction,
    )
    assert zero["delta_hidden_l2"] == 0.0
    assert zero["delta_hidden_cosine_with_d_K"] is None


def test_realized_manipulation_diagnostics_and_zero_activation() -> None:
    direction = DecisionDirection(
        file="direction.npz",
        fold=0,
        layer=0,
        position="ac",
        d_raw=np.asarray([1.0, 0.0]),
        d_K=np.asarray([1.0, 0.0]),
        raw_intercept=0.0,
    )
    layer = TensorLayer()
    modules = _modules(layer, hidden_size=2)
    hook = AdditiveActivationHook(
        modules,
        layer_index=0,
        target_position=0,
        steering_vector=torch.tensor([2.0, 0.0]),
        prefill_sequence_length=1,
    )
    with hook:
        layer(torch.zeros(1, 1, 2))
    diagnostics = manipulation_diagnostics(
        hook, direction, alpha=2.0, steering_scale="probe_logit"
    )
    assert diagnostics["passed"]
    assert diagnostics["decision_logit_delta"] == pytest.approx(2.0)
    assert diagnostics["K_after"] > diagnostics["K_before"]
    assert diagnostics["requested_delta_passed"]


def test_bfloat16_realized_delta_uses_explicit_quantization_tolerance() -> None:
    direction = DecisionDirection(
        file="direction.npz",
        fold=0,
        layer=0,
        position="ac",
        d_raw=np.asarray([1.0]),
        d_K=np.asarray([1.0]),
        raw_intercept=0.0,
    )

    class FakeHook:
        h_before = torch.tensor([0.0])
        h_after = torch.tensor([-0.766])
        steering_vector = torch.tensor([-1.0], dtype=torch.float64)
        activation_dtype = "bfloat16"

        @staticmethod
        def diagnostics() -> dict:
            return {
                "hook_call_count": 2,
                "steering_applied_count": 1,
                "activation_dtype": "bfloat16",
                "injection_l2": 0.766,
            }

    diagnostics = manipulation_diagnostics(
        FakeHook(), direction, alpha=-1.0, steering_scale="probe_logit"
    )
    assert diagnostics["passed"]
    assert diagnostics["requested_decision_logit_delta"] == pytest.approx(-1.0)
    assert diagnostics["realized_quantization_error"] == pytest.approx(0.234)
    assert diagnostics["decision_logit_delta_tolerance"] == pytest.approx(0.25)


def test_baseline_soft_gap_above_point_zero_five_is_marked_not_rejected() -> None:
    zero = {"injection_l2": 0.0, "decision_logit_delta": 0.0}
    at_threshold = build_baseline_validation(
        answer_match=True,
        hard_source_match=True,
        generated_source_match=True,
        soft_source_abs_error=0.05,
        generation_diagnostics=zero,
        teacher_diagnostics=zero,
    )
    assert at_threshold["passed"]
    assert not at_threshold["soft_source_gap_exceeded"]
    above_threshold = build_baseline_validation(
        answer_match=True,
        hard_source_match=True,
        generated_source_match=True,
        soft_source_abs_error=0.050001,
        generation_diagnostics=zero,
        teacher_diagnostics=zero,
    )
    assert not above_threshold["passed"]
    assert above_threshold["marked"]
    assert above_threshold["soft_source_gap_exceeded"]
    assert above_threshold["used_as_paired_baseline"]


def test_balanced_selection_takes_stable_equal_counts_and_rejects_short_side() -> None:
    records = [
        {
            "case_id": f"{item}__prior_0__conflict_easy__v4__joint",
            "item_id": str(item),
            "prior_index": 0,
            "condition": "conflict_easy",
            "decision_side": side,
        }
        for item, side in [
            (3, "follows_image"),
            (1, "follows_text"),
            (4, "follows_image"),
            (2, "follows_text"),
            (5, "follows_image"),
        ]
    ]
    selected = select_balanced_decision_side_cases(records, 2)
    assert [row["item_id"] for row in selected] == ["1", "2", "3", "4"]
    assert sum(row["decision_side"] == "follows_text" for row in selected) == 2
    assert sum(row["decision_side"] == "follows_image" for row in selected) == 2
    with pytest.raises(ValueError, match="only 2 remain"):
        select_balanced_decision_side_cases(records, 3)


def test_baseline_hidden_state_repository_reads_exact_case_layer_position(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    shard = experiment / "hidden_states" / "target" / "shard.pt"
    shard.parent.mkdir(parents=True)
    tensor = torch.arange(2 * 2 * 2 * 3, dtype=torch.float16).reshape(2, 2, 2, 3)
    torch.save(
        {
            "case_ids": ["case-a", "case-b"],
            "layer_indices": [22, 24],
            "position_names": ["ac", "panl"],
            "hidden_states": tensor,
        },
        shard,
    )
    reference = {
        "shard_path": "hidden_states/target/shard.pt",
        "offset": 1,
    }
    (experiment / "hidden_states" / "index.json").write_text(
        json.dumps({"format_version": 1, "cases": {"case-b": reference}}),
        encoding="utf-8",
    )
    repository = BaselineHiddenStateRepository(experiment)
    manifest = {
        "case_id": "case-b",
        "hidden_state_reference": reference,
    }
    vector = repository.get(manifest, 24, "panl")
    assert np.array_equal(vector, tensor[1, 1, 1].float().numpy())
    with pytest.raises(ValueError, match="omits layer"):
        repository.get(manifest, 23, "panl")


def test_alpha_zero_record_reuses_pre_steering_result_without_forward() -> None:
    manifest = {
        "case_id": "1__prior_0__conflict_easy__v4__joint",
        "item_id": "1",
        "prior_index": 0,
        "condition": "conflict_easy",
        "decision_side": "follows_image",
        "text_only_answer": "red",
        "image_only_answer": "blue",
    }
    baseline = {
        "generated": {
            "current_answer": "Blue",
            "current_answer_result": {
                "raw_output": "**Answer**: Blue\n**Source Attribution**: 7",
                "source_label": "7",
                "answer_class_probabilities": {"red": 0.2, "blue": 0.6},
            },
            "source_attribution": {
                "hard_label": "7",
                "soft_image_score": 0.75,
                "class_probabilities": [0.1, 0.2, 0.7],
                "source_entropy": 0.8,
            },
        }
    }
    direction = DecisionDirection(
        file="direction.npz",
        fold=0,
        layer=22,
        position="ac",
        d_raw=np.asarray([1.0]),
        d_K=np.asarray([1.0]),
        raw_intercept=0.0,
    )
    record = build_reused_baseline_record(
        steering_case=SteeringCase(
            manifest=manifest,
            evaluation=None,
            baseline=baseline,
            fold=0,
        ),
        direction=direction,
        layer=22,
        position="ac",
        steering_scale="probe_logit",
    )
    assert record["status"] == "completed"
    assert record["alpha"] == 0.0
    assert record["steered_decision_side"] == "follows_image"
    assert record["P_text_answer"] == pytest.approx(0.2)
    assert record["P_image_answer"] == pytest.approx(0.6)
    assert record["pair_image_prob"] == pytest.approx(0.75)
    assert record["SA_soft_image_score"] == pytest.approx(0.75)
    assert record["generation_diagnostics"]["executed"] is False
    assert record["teacher_forced_diagnostics"]["executed"] is False
    assert record["baseline_validation"]["source"] == "pre_steering_results"

    repository = SimpleNamespace(get=lambda *_args: direction)
    via_entrypoint, invariant_failure = run_intervention(
        steering_case=SteeringCase(
            manifest=manifest,
            evaluation=None,
            baseline=baseline,
            fold=0,
        ),
        repository=repository,
        joint_generator=None,
        source_analyzer=None,
        modules=None,
        layer=22,
        position="ac",
        alpha=0.0,
        steering_scale="probe_logit",
        max_answer_tokens=24,
    )
    assert via_entrypoint["status"] == "completed"
    assert via_entrypoint["generation_diagnostics"]["executed"] is False
    assert not invariant_failure


def test_execute_run_records_failure_and_continues(monkeypatch, tmp_path: Path) -> None:
    cases = [
        SimpleNamespace(manifest={"case_id": "failed-case"}),
        SimpleNamespace(manifest={"case_id": "completed-case"}),
    ]

    def fake_run_intervention(*, steering_case, layer, position, alpha, steering_scale, **_kwargs):
        case_id = steering_case.manifest["case_id"]
        failed = case_id == "failed-case"
        record = {
            "intervention_key": intervention_key(
                case_id, layer, position, alpha, steering_scale
            ),
            "case_id": case_id,
            "layer": layer,
            "position": position,
            "alpha": alpha,
            "condition": "conflict_easy",
            "baseline_decision_side": "follows_image",
            "steered_decision_side": "follows_image",
            "answer_margin": 1.0,
            "SA_soft_image_score": 0.7,
            "status": "failed" if failed else "completed",
            "error": {"type": "SyntheticFailure"} if failed else None,
            "baseline_validation": {
                "passed": not failed,
                "marked": failed,
                "used_as_paired_baseline": True,
            },
        }
        return record, failed

    monkeypatch.setattr(steering_module, "run_intervention", fake_run_intervention)
    results = execute_run(
        cases=cases,
        repository=None,
        joint_generator=None,
        source_analyzer=None,
        modules=None,
        baseline_hidden_states=None,
        layers=[22],
        positions=["ac"],
        alphas=[0.0],
        steering_scale="probe_logit",
        max_answer_tokens=24,
        existing=[],
        results_path=tmp_path / "results.jsonl",
        progress_path=tmp_path / "progress.json",
        summary_path=tmp_path / "summary.json",
    )
    assert len(results) == 2
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["status"] == "complete_with_failures"
    assert progress["failed_count"] == 1
    assert progress["baseline_marked_count"] == 1


def _write_direction_run(root: Path, *, reversed_semantics: bool = False) -> Path:
    direction_dir = root / "decision_directions"
    direction_dir.mkdir(parents=True)
    (root / "run_config.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "split_mode": "item",
                "decision_side_label_mapping": DECISION_MAPPING,
                "manifest_path": str(root / "manifest.jsonl"),
            }
        )
    )
    (root / "manifest.jsonl").write_text("{}\n")
    (root / "split_assignments.json").write_text(
        json.dumps(
            {
                "n_splits": 2,
                "group_key": "item_id",
                "item_to_fold": {"item-a": 0, "item-b": 1},
            }
        )
    )
    entries = []
    for fold in range(2):
        name = f"v4_to_v4__fold_{fold}__ac__layer_0.npz"
        d_raw = np.asarray([3.0, 4.0])
        np.savez_compressed(
            direction_dir / name,
            scaler_mean=np.zeros(2),
            scaler_scale=np.ones(2),
            weight=d_raw,
            intercept=np.asarray([0.0]),
            d_raw=d_raw,
            d_K=d_raw / np.linalg.norm(d_raw),
            raw_intercept=np.asarray([0.5]),
        )
        entries.append(
            {
                "file": name,
                "fold": fold,
                "layer": 0,
                "position": "ac",
                "version_setting": "v4_to_v4",
                "class0": "follows_image" if reversed_semantics else "follows_text",
                "class1": "follows_text" if reversed_semantics else "follows_image",
                "positive_direction": "+d_K -> follows_image",
            }
        )
    (direction_dir / "index.json").write_text(
        json.dumps(
            {
                "split_mode": "item",
                "class_mapping": DECISION_MAPPING,
                "direction_count": 2,
                "directions": entries,
            }
        )
    )
    return root


def test_direction_repository_maps_case_fold_and_rejects_reversed_semantics(
    tmp_path: Path,
) -> None:
    repository = DirectionRepository(_write_direction_run(tmp_path / "valid"))
    assert repository.fold_for_item("item-b") == 1
    repository.validate_requested_grid([0], ["ac"])
    assert repository.get(1, 0, "ac").raw_intercept == pytest.approx(0.5)

    reversed_repository = DirectionRepository(
        _write_direction_run(tmp_path / "reversed", reversed_semantics=True)
    )
    with pytest.raises(ValueError, match="reversed class semantics"):
        reversed_repository.get(0, 0, "ac")


def test_direction_repository_rejects_invalid_dk(tmp_path: Path) -> None:
    root = _write_direction_run(tmp_path / "invalid")
    path = root / "decision_directions" / "v4_to_v4__fold_0__ac__layer_0.npz"
    with np.load(path) as payload:
        values = {name: payload[name] for name in payload.files}
    values["d_K"] = np.asarray([-0.6, -0.8])
    np.savez_compressed(path, **values)
    repository = DirectionRepository(root)
    with pytest.raises(ValueError, match="invalid d_K"):
        repository.get(0, 0, "ac")


def test_grid_and_intervention_key_are_strict() -> None:
    assert validate_grid([22], ["ac"], [-1, 0, 1]) == (
        [22],
        ["ac"],
        [-1.0, 0.0, 1.0],
    )
    with pytest.raises(ValueError, match="alpha=0"):
        validate_grid([22], ["ac"], [-1, 1])
    assert validate_grid([22], ["panl"], [0])[1] == ["panl"]
    with pytest.raises(ValueError, match="ptnl/ac/panl"):
        validate_grid([22], ["sac"], [0])
    with pytest.raises(ValueError, match="distinct"):
        validate_grid([22, 22], ["ac"], [0])
    assert intervention_key("case", 22, "ac", -0.0, "probe_logit") == intervention_key(
        "case", 22, "ac", 0.0, "probe_logit"
    )
    assert intervention_key(
        "case",
        22,
        "ac",
        0.0,
        "probe_logit",
        intervention_mode="reinject",
    ) != intervention_key("case", 22, "ac", 0.0, "probe_logit")


def _result(case_id: str, alpha: float, margin: float, source: float, side: str) -> dict:
    return {
        "intervention_key": intervention_key(case_id, 22, "ac", alpha, "probe_logit"),
        "case_id": case_id,
        "layer": 22,
        "position": "ac",
        "alpha": alpha,
        "condition": "conflict_easy",
        "baseline_decision_side": "follows_text",
        "steered_decision_side": side,
        "answer_margin": margin,
        "SA_soft_image_score": source,
        "layer_trajectory": [
            {
                "readout_layer": 24,
                "delta_logit": float(alpha) / 2.0,
                "delta_K": float(alpha) / 20.0,
                "retention_fraction": 0.5 if alpha else None,
            }
        ],
        "status": "completed",
    }


def test_paired_summary_uses_within_case_alpha_zero_delta() -> None:
    records = [
        _result("a", 0, 1.0, 0.4, "follows_text"),
        _result("a", 2, 1.75, 0.55, "follows_image"),
        _result("b", 0, -1.0, 0.2, "follows_text"),
        _result("b", 2, -0.5, 0.3, "follows_text"),
    ]
    summary = build_paired_summary(records)
    pooled = next(
        row
        for row in summary["paired_cells"]
        if row["alpha"] == 2 and row["subgroup"] == "pooled"
    )
    assert pooled["delta_answer_margin"]["mean"] == pytest.approx(0.625)
    assert pooled["delta_SA_soft_image_score"]["mean"] == pytest.approx(0.125)
    assert pooled["text_to_image_flip_rate"] == pytest.approx(0.5)
    trajectory = next(
        row
        for row in summary["trajectory_cells"]
        if row["alpha"] == 2 and row["subgroup"] == "pooled"
    )
    assert trajectory["injection_layer"] == 22
    assert trajectory["readout_layer"] == 24
    assert trajectory["delta_logit"]["mean"] == pytest.approx(1.0)
    assert trajectory["retention_fraction"]["mean"] == pytest.approx(0.5)
    near_boundary = next(
        row
        for row in summary["paired_cells"]
        if row["alpha"] == 2
        and row["subgroup"] == "baseline_abs_answer_margin_lt_1"
    )
    assert near_boundary["delta_answer_margin"]["n"] == 0
    far_from_boundary = next(
        row
        for row in summary["paired_cells"]
        if row["alpha"] == 2
        and row["subgroup"] == "baseline_abs_answer_margin_ge_1"
    )
    assert far_from_boundary["delta_answer_margin"]["n"] == 2


def test_teacher_forced_wire_uses_steered_answer() -> None:
    text = teacher_forced_assistant_text("steered blue", "7")
    assert text == "**Answer**: steered blue\n**Source Attribution**:7"
    assert "baseline" not in text


def test_resume_rejects_configuration_change(tmp_path: Path) -> None:
    configuration = {"layers": [22], "positions": ["ac"], "alphas": [0.0]}
    initialize_output(tmp_path, configuration, resume=False)
    initialize_output(tmp_path, configuration, resume=True)
    with pytest.raises(ValueError, match="differs"):
        initialize_output(
            tmp_path,
            {**configuration, "layers": [24]},
            resume=True,
        )
