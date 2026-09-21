from __future__ import annotations

from dp_SA.io_utils import atomic_json, atomic_jsonl
from qwen3_chat_binary.experiments.attention_block.analyze import analyze
from qwen3_chat_binary.experiments.attention_block.config import CONDITIONS


def test_synthetic_analysis_runs_end_to_end(tmp_path) -> None:
    atomic_json(tmp_path / "run_config.json", {"windows": [[12, 16]]})
    manifest = [{"case_id": "t", "test_side": "text_side"}, {"case_id": "i", "test_side": "image_side"}]
    atomic_jsonl(tmp_path / "artifacts/manifests/test_manifest.jsonl", manifest)
    trials = []
    for row in manifest:
        trials.append({**row, "condition": "C0_clean"})
        for index, condition in enumerate(CONDITIONS):
            trials.append({**row, "condition": condition, "window_start": 12, "window_end": 16,
                           "token_change_rate": float(index % 2), "logit_change_diff": float(index)})
    atomic_jsonl(tmp_path / "artifacts/trials.jsonl", trials)
    result = analyze(output_root=tmp_path, repeats=20)
    assert result["case_count"] == 2 and len(result["figures"]) == 4

