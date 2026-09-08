from __future__ import annotations

import csv
import json

import torch

from dp_SA.SA_trajectory.LAT2PANL.analyze import analyze
from dp_SA.SA_trajectory.LAT2PANL.io_utils import atomic_bf16_npz, atomic_json, atomic_jsonl


def _trial(case, condition, alpha, layer, value, hidden_file=None):
    return {"status": "completed", "case_id": case, "family_id": case, "item_id": case,
            "image_sha256": case, "answer": "red", "test_side": "high_image",
            "condition": condition, "alpha": alpha, "panl_mediator_layer": layer,
            "final_soft_sa": value, "hard_sa_class": 5, "class_logits": list(range(9)),
            "class_probabilities": [1/9] * 9, "cle_probe_sa": value,
            "cle_probe_eligible": True, "cle_probe_exclusion_reasons": [],
            "hook": {"non_target_unchanged": True, "prefill_hits": {"14": 1, "20": 1},
                     "replacement_bitwise_equal": True}, "vector": {"vector_fingerprint": "v"} if condition in ("C1", "C2") else None,
            "source_hidden_bits_sha256": "h" if condition in ("C2", "C3") else None,
            "captured_hidden_file": hidden_file, "captured_hidden_sha256": "x",
            "parity": {"passed": True} if condition == "C0" else None,
            "panl_manipulation": {"delta_norm": 0., "relative_norm": 0., "cosine_before_source": 1., "replacement_bitwise_equal": True} if condition in ("C2", "C3") else None}


def test_all_table_and_figure_schemas(tmp_path):
    root = tmp_path; case = "family_a"
    manifest = [{"case_id": case, "family_id": case, "item_id": case, "test_answer": "red"}]
    atomic_jsonl(root / "artifacts/manifests/test_manifest.jsonl", manifest)
    hidden = torch.arange(4, dtype=torch.float32).to(torch.bfloat16)
    c0_path = root / "artifacts/hidden/c0.npz"; atomic_bf16_npz(c0_path, {"PANL_L14": hidden})
    rows = [_trial(case, "C0", 0., None, .5, "artifacts/hidden/c0.npz")]
    for alpha in (-2., 2.):
        c1_path = root / f"artifacts/hidden/c1_{alpha}.npz"; atomic_bf16_npz(c1_path, {"PANL_L14": hidden})
        rows.append(_trial(case, "C1", alpha, None, .5 + .01 * alpha, str(c1_path.relative_to(root))))
        for layer in (14, 15, 18):
            rows.append(_trial(case, "C2", alpha, layer, .5 + .005 * alpha))
            rows.append(_trial(case, "C3", alpha, layer, .5 + .004 * alpha))
    atomic_jsonl(root / "artifacts/trials.jsonl", rows)
    atomic_json(root / "artifacts/diagnostics/cle_probe_eligibility_audit.json", {"eligible_count": 73})
    result = analyze(output_root=root, smoke=True, repeats=20)
    assert result["tables"] == 5 and result["figures"] == 3
    required = ["condition_effects.csv", "mediation_contrasts.csv", "manipulation_checks.csv", "case_level_effects.csv", "README_zh.md"]
    assert all((root / "tables" / name).stat().st_size for name in required)
    assert (root / "figures/fig1_final_sa_delta.png").stat().st_size
    assert (root / "figures/fig3_final_sa_attenuation.png").stat().st_size
    with (root / "tables/case_level_effects.csv").open(newline="") as handle:
        fields = csv.DictReader(handle).fieldnames
    assert "cle_probe_eligible" in fields and "cle_probe_exclusion_reasons" in fields
    resumed = analyze(output_root=root, smoke=True, repeats=20, resume=True)
    assert resumed["resumed_noop"]
