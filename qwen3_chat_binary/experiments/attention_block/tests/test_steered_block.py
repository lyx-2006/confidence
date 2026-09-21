from __future__ import annotations

import torch

from dp_SA.attention_block.masking import AttentionBlockContext, AttentionEdges
from dp_SA.io_utils import atomic_json, atomic_jsonl
from layer_metacognition.model_adapter import AdditiveActivationHook, LanguageModules
from qwen3_chat_binary.config import MODEL_PATH
from qwen3_chat_binary.experiments.attention_block.steered_block import ALPHAS, WINDOWS, comparison_metrics, run


class ToyAttention(torch.nn.Module):
    def forward(self, hidden, *, attention_mask, cache_position):
        weights = torch.softmax(attention_mask.expand(1, 2, -1, -1).float(), dim=-1)
        return hidden, weights


class ToyLayer(torch.nn.Module):
    def __init__(self): super().__init__(); self.self_attn = ToyAttention()
    def forward(self, hidden, mask):
        return self.self_attn(hidden, attention_mask=mask, cache_position=torch.arange(hidden.shape[1]))[0]


def test_windows_and_comparison_metrics() -> None:
    assert WINDOWS == ((17, 24), (22, 30), (24, 32), (26, 34))
    assert [b-a+1 for a,b in WINDOWS] == [8, 9, 9, 9]
    result = comparison_metrics([5, 1, 1, 1, 1], [2, 4, 1, 1, 1])
    assert result["token_changed"] and result["token_change_rate"] == 1.0
    assert result["logit_change_diff"] == 3.75
    tie = comparison_metrics([5, 5, 1, 1, 1], [5, 5, 1, 1, 1])
    assert tie["reference_class"] == 0 and tie["reference_argmax_tie"]


def test_l16_steering_and_downstream_block_apply_once() -> None:
    layers = [ToyLayer() for _ in range(36)]
    modules = LanguageModules(layers, torch.nn.Identity(), torch.nn.Identity(), 2, 36)
    hidden = torch.zeros(1, 4, 2); mask = torch.zeros(1, 1, 4, 4)
    steering = AdditiveActivationHook(modules, layer_index=16, target_position=1,
        steering_vector=torch.ones(2), prefill_sequence_length=4)
    block = AttentionBlockContext(layers, layer_indices=range(17,25),
        edges=AttentionEdges(((3,1),)), sequence_length=4, row_sum_tolerance=1e-6)
    with steering, block:
        for layer in layers: hidden = layer(hidden, mask)
    assert steering.diagnostics()["steering_applied_count"] == 1
    audit = block.diagnostics()
    assert audit["layers"] == list(range(17,25))
    assert all(v["max_blocked_weight"] == 0.0 for v in audit["by_layer"].values())
    assert all(not layer._forward_hooks and not layer.self_attn._forward_pre_hooks for layer in layers)


def test_complete_resume_returns_before_model_load(tmp_path) -> None:
    fingerprint = "synthetic"
    atomic_json(tmp_path/"run_config.json", {"mode":"steered_block", "model":str(MODEL_PATH.resolve()),
                                             "fingerprint":fingerprint})
    manifest=[{"case_id":"t"},{"case_id":"i"}]
    atomic_jsonl(tmp_path/"artifacts/manifests/test.jsonl",manifest)
    rows=[]
    for item in manifest:
        case=item["case_id"]
        rows.append({"case_id":case,"condition":"C0","alpha":None,"window_start":None,"window_end":None,
                     "fingerprint":fingerprint})
        for alpha in ALPHAS:
            rows.append({"case_id":case,"condition":"S","alpha":alpha,"window_start":None,"window_end":None,
                         "fingerprint":fingerprint})
            for start,end in WINDOWS:
                rows.append({"case_id":case,"condition":"SB","alpha":alpha,"window_start":start,"window_end":end,
                             "fingerprint":fingerprint})
    for index,row in enumerate(rows): atomic_json(tmp_path/f"artifacts/trials/{index}.json",row)
    result=run(output_root=tmp_path,resume=True)
    assert result["new_gpu_forwards"]==0 and result["resumed_noop"]
