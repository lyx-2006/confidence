from __future__ import annotations

from dp_SA.io_utils import atomic_json, atomic_jsonl
from qwen3_chat_binary.experiments.attention_block.steered_block import ALPHAS, WINDOWS
from qwen3_chat_binary.experiments.attention_block.steered_block_analysis import analyze


def test_synthetic_steered_block_analysis(tmp_path) -> None:
    atomic_json(tmp_path/"run_config.json", {"mode":"steered_block"})
    manifest=[{"case_id":"t","test_side":"text_side"},{"case_id":"i","test_side":"image_side"}]
    atomic_jsonl(tmp_path/"artifacts/manifests/test.jsonl",manifest); trials=[]
    for row in manifest:
        trials.append({**row,"condition":"C0","alpha":None,"final":0})
        for alpha in ALPHAS:
            trials.append({**row,"condition":"S","alpha":alpha,"vs_clean_token_change_rate":1.0,
                           "vs_clean_logit_change_diff":2.0})
            for start,end in WINDOWS:
                by_layer={str(x):{"max_blocked_weight":0.0,"max_row_sum_error":0.0,"finite":True,"hook_call_count":1}
                          for x in range(start,end+1)}
                trials.append({**row,"condition":"SB","alpha":alpha,"window_start":start,"window_end":end,
                    "vs_steered_token_change_rate":0.0,"vs_steered_logit_change_diff":0.5,
                    "vs_clean_token_change_rate":1.0,"vs_clean_logit_change_diff":2.5,
                    "steering_flipped":True,"restored_clean_label":False,
                    "steering_diagnostics":{"steering_applied_count":1},
                    "attention_diagnostics":{"layers":list(range(start,end+1)),"by_layer":by_layer}})
    atomic_jsonl(tmp_path/"artifacts/trials.jsonl",trials)
    result=analyze(output_root=tmp_path,repeats=20)
    assert result["trial_count"]==22 and result["attention_audit_rows"]==16
