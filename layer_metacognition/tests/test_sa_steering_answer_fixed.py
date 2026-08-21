from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from layer_metacognition.model_adapter import LanguageModules
from layer_metacognition.sa_steering.artifacts import (
    SteeringVector,
    _normalize_direction,
    select_evaluation_cases,
    select_extreme_sources,
)
from layer_metacognition.sa_steering.runner import (
    build_summary,
    execute_alpha_zero_baselines,
    intervention_key,
    migrate_results_to_corrected_delta,
)
from layer_metacognition.sa_steering.run_sa_steering import _validate_args, build_parser
from layer_metacognition.sa_steering.sa_steering_hook import (
    SAActivationAdditionHook,
    fixed_answer_assistant_prefix,
    locate_sa_steering_position,
)


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict:
        del add_special_tokens
        output = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(value) for value in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|image_pad|>"
        return -999


class TensorLayer(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


def _modules(hidden_size: int = 3) -> LanguageModules:
    return LanguageModules(
        language_layers=[TensorLayer()],
        final_norm=torch.nn.Identity(),
        lm_head=torch.nn.Linear(hidden_size, 5, bias=False),
        hidden_size=hidden_size,
        num_hidden_layers=1,
    )


def _record(item: int, label: str, score: float, fold: int) -> tuple[dict, tuple[str, int]]:
    record = {
        "case_id": f"{item}__prior_0__conflict_easy__v4__joint",
        "item_id": str(item),
        "prior_index": 0,
        "condition": "conflict_easy",
        "baseline": {"generated_label": label, "soft_score": score},
    }
    return record, (str(item), fold)


def test_fixed_answer_prefix_and_all_position_locators_share_one_prefill() -> None:
    tokenizer = CharacterTokenizer()
    assistant = fixed_answer_assistant_prefix("blue")
    assert assistant == "**Answer**: blue\n**Source Attribution**:"
    rendered = "user prompt\nassistant\n" + assistant
    ids = tokenizer.encode(rendered)
    inputs = SimpleNamespace(
        input_ids=torch.tensor([ids]),
        attention_mask=torch.ones(1, len(ids), dtype=torch.long),
    )
    located = {
        position: locate_sa_steering_position(
            tokenizer=tokenizer,
            rendered=rendered,
            inputs=inputs,
            assistant_text=assistant,
            answer="blue",
            position=position,
        )[0]
        for position in ("ac", "lat", "panl", "sac")
    }
    assert located["ac"] < located["lat"] < located["panl"] < located["sac"]
    assert tokenizer.decode([ids[located["lat"]]]) == "e"
    assert tokenizer.decode([ids[located["panl"]]]) == "\n"
    assert tokenizer.decode([ids[located["sac"]]]) == ":"


def test_sa_hook_changes_only_one_token_and_rejects_batch_greater_than_one() -> None:
    modules = _modules()
    layer = modules.language_layers[0]
    hidden = torch.zeros(1, 4, 3)
    hook = SAActivationAdditionHook(
        modules,
        layer_index=0,
        target_position=2,
        steering_vector=torch.tensor([1.0, 2.0, 3.0]),
        prefill_sequence_length=4,
    )
    with hook:
        output = layer(hidden)
    assert torch.equal(output[0, 2], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.count_nonzero(output[0, :2]) == 0
    assert torch.count_nonzero(output[0, 3:]) == 0
    assert hook.diagnostics()["steering_applied_count"] == 1

    invalid = SAActivationAdditionHook(
        modules,
        layer_index=0,
        target_position=0,
        steering_vector=torch.ones(3),
        prefill_sequence_length=2,
    )
    with pytest.raises(ValueError, match="batch size 1"):
        with invalid:
            layer(torch.zeros(2, 2, 3))
    assert not layer._forward_hooks


def test_direction_normalization_is_three_percent_of_mean_hidden_norm() -> None:
    vector, raw_norm, hidden_mean, target = _normalize_direction(
        np.asarray([3.0, 4.0]),
        np.asarray([10.0, 20.0]),
    )
    assert raw_norm == 5.0
    assert hidden_mean == 15.0
    assert target == pytest.approx(0.45)
    assert np.linalg.norm(vector) == pytest.approx(0.45, rel=1e-6)


def test_item_fold_sampling_and_extreme_sources_are_disjoint() -> None:
    rows: list[dict] = []
    mapping: dict[str, int] = {}
    labels = ["0", "1", "2", "5", "6", "8"]
    for item in range(180):
        fold = 0 if item < 120 else 1
        label = labels[item % len(labels)]
        row, assignment = _record(item, label, item / 200.0, fold)
        rows.append(row)
        mapping[assignment[0]] = assignment[1]
    selected = select_evaluation_cases(
        rows,
        mapping,
        test_fold=0,
        eval_cases=20,
        seed=42,
    )
    assert len(selected) == 20
    assert sum(row["baseline_sa_group"] == "low" for row in selected) == 10
    assert sum(row["baseline_sa_group"] == "high" for row in selected) == 10
    assert all(mapping[row["item_id"]] == 0 for row in selected)

    sources = select_extreme_sources(
        rows,
        mapping,
        test_fold=0,
        cases_per_side=10,
    )
    source_items = [row["item_id"] for group in sources.values() for row in group]
    assert len(source_items) == len(set(source_items)) == 20
    assert all(mapping[item] != 0 for item in source_items)
    assert max(row["baseline"]["soft_score"] for row in sources["low"]) < min(
        row["baseline"]["soft_score"] for row in sources["high"]
    )


def test_intervention_key_and_summary_direction_metrics() -> None:
    assert intervention_key("c", "ac", 12, "mean_difference", "high", 2) == (
        "c|position=ac|layer=12|method=mean_difference|direction=high|alpha=2"
    )

    class Repository:
        index = {"positions": ["ac"], "layers": [12]}

        def get(self, method, position, layer):
            del position, layer
            raw = np.asarray([1.0, 0.0]) if method == "mean_difference" else np.asarray([0.0, 1.0])
            return SteeringVector(method, "ac", 12, raw.astype(np.float32), raw, 1, 1, 10, 0.3, {})

    records = [
        {
            "status": "completed",
            "position": "ac",
            "layer": 12,
            "steering_type": "mean_difference",
            "direction": "high",
            "alpha": 2.0,
            "corrected_delta_SA": 0.1,
            "changed": True,
            "scored_hard_label_changed": False,
            "baseline_sa_group": "high",
        },
        {
            "status": "completed",
            "position": "ac",
            "layer": 12,
            "steering_type": "mean_difference",
            "direction": "high",
            "alpha": 2.0,
            "corrected_delta_SA": -0.1,
            "changed": False,
            "scored_hard_label_changed": False,
            "baseline_sa_group": "low",
        },
    ]
    summary = build_summary(records, Repository(), expected_count=2)
    cell = summary["steering_effectiveness"][0]
    assert cell["corrected_direction_consistency"] == 0.5
    assert cell["generated_label_flip_rate"] == 0.5
    assert cell["self_direction"]["corrected_direction_consistency"] == 1.0
    assert summary["method_direction_cosines"][0]["cosine"] == 0.0


def test_alpha_zero_requires_explicit_parity_mode() -> None:
    parser = build_parser()
    with pytest.raises(ValueError, match="explicit --alpha-zero-parity"):
        _validate_args(parser.parse_args(["--alphas", "0"]))
    args = parser.parse_args(["--alphas", "0", "--alpha-zero-parity"])
    assert _validate_args(args)[-1] == [0.0]
    with pytest.raises(ValueError, match="requires exactly --alphas 0"):
        _validate_args(
            parser.parse_args(["--alphas", "0", "2", "--alpha-zero-parity"])
        )


def test_alpha_zero_summary_reports_parity_metrics() -> None:
    class Repository:
        index = {"positions": ["ac", "sac"], "layers": [18]}

        def get(self, method, position, layer):
            del method, position, layer
            raw = np.asarray([1.0, 0.0])
            return SteeringVector(
                "mean_difference", "ac", 18, raw.astype(np.float32), raw, 1, 1, 10, 0.3, {}
            )

    records = []
    for position, delta in (("ac", -0.04), ("sac", -0.03)):
        records.append(
            {
                "status": "completed",
                "case_id": "case",
                "position": position,
                "layer": 18,
                "steering_type": "mean_difference",
                "direction": "high",
                "alpha": 0.0,
                "SA_before": 0.5,
                "SA_after": 0.5 + delta,
                "changed": False,
                "scored_hard_label_changed": False,
                "answer_changed": False,
                "injection_norm": 0.0,
                "baseline_sa_group": "high",
            }
        )
    parity = build_summary(records, Repository(), expected_count=2)["alpha_zero_parity"]
    assert parity["n"] == 2
    assert parity["cached_before_offset_SA"]["mean"] == pytest.approx(-0.035)
    assert parity["absolute_cached_before_offset_SA"]["mean"] == pytest.approx(0.035)
    assert parity["generated_label_match_rate"] == 1.0
    assert parity["answer_unchanged_rate"] == 1.0
    assert parity["max_injection_norm"] == 0.0
    assert parity["max_within_case_SA_after_span"] == pytest.approx(0.01)


def test_migrate_results_replaces_legacy_delta_atomically(tmp_path) -> None:
    baseline = {"status": "completed", "case_id": "case", "SA_after": 0.4}
    source = {
        "status": "completed",
        "case_id": "case",
        "SA_after": 0.55,
        "delta_SA": -0.2,
    }
    path = tmp_path / "results.jsonl"
    migrated = migrate_results_to_corrected_delta(
        [source], [baseline], results_path=path
    )
    assert "delta_SA" not in migrated[0]
    assert migrated[0]["alpha_zero_SA"] == 0.4
    assert migrated[0]["corrected_delta_SA"] == pytest.approx(0.15)
    persisted = __import__("json").loads(path.read_text().strip())
    assert persisted == migrated[0]


def test_alpha_zero_backfill_skips_completed_cases(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "alpha0.jsonl"
    path.write_text(
        '{"status":"completed","case_id":"a","SA_after":0.4}\n',
        encoding="utf-8",
    )
    cases = [
        SimpleNamespace(record={"case_id": "a"}),
        SimpleNamespace(record={"case_id": "b"}),
    ]
    called = []

    def fake_run_intervention(**kwargs):
        case_id = kwargs["runtime_case"].record["case_id"]
        called.append(case_id)
        return {
            "status": "completed",
            "case_id": case_id,
            "SA_after": 0.5,
            "alpha": 0.0,
        }

    monkeypatch.setattr(
        "layer_metacognition.sa_steering.runner.run_intervention",
        fake_run_intervention,
    )
    records = execute_alpha_zero_baselines(
        cases=cases,
        repository=object(),
        joint_generator=object(),
        modules=object(),
        source_variant=object(),
        position="sac",
        layer=18,
        method="mean_difference",
        max_source_tokens=4,
        results_path=path,
        progress_path=tmp_path / "progress.json",
    )
    assert called == ["b"]
    assert {record["case_id"] for record in records if record["status"] == "completed"} == {
        "a",
        "b",
    }
