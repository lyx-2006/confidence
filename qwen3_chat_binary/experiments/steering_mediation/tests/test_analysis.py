from __future__ import annotations

from dp_SA.io_utils import atomic_jsonl
from qwen3_chat_binary.experiments.steering_mediation.analyze import analyze
from qwen3_chat_binary.experiments.steering_mediation.config import ALPHAS, CHAINS, PAIRS


def test_synthetic_analysis_runs_end_to_end(tmp_path) -> None:
    manifest = [{"case_id": "t", "test_side": "text_side"}, {"case_id": "i", "test_side": "image_side"}]
    atomic_jsonl(tmp_path / "artifacts/manifests/test.jsonl", manifest)
    trials = []
    for row in manifest:
        trials.append({**row, "condition": "C0", "final_sa": .5})
        for chain in CHAINS:
            for upstream, downstream in PAIRS:
                for alpha in ALPHAS:
                    for condition, value in (("C1", .7), ("C2", .55), ("C3", .65)):
                        trials.append({**row, "condition": condition, "final_sa": value, "chain": chain,
                                       "upstream_layer": upstream, "downstream_layer": downstream, "alpha": alpha})
    atomic_jsonl(tmp_path / "artifacts/trials.jsonl", trials)
    result = analyze(output_root=tmp_path, repeats=20)
    assert result["case_count"] == 2 and result["logical_row_count"] == 96
