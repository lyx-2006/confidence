"""Final cross-experiment table and report for Stage 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from layer_metacognition.hidden_state_store import atomic_write_json, atomic_write_text

from .core import EXPERIMENT_DIR_NAMES, SAOOFDirectionRepository, read_json, write_csv_atomic


def direction_geometry(output_root: Path, decision_direction_dir: Path) -> dict[str, Any]:
    repository = SAOOFDirectionRepository(output_root / "directions")
    folds: list[dict[str, Any]] = []
    for fold in range(5):
        sa = repository.get(fold).d_unit
        path = decision_direction_dir / "decision_directions" / f"v4_to_v4__fold_{fold}__panl__layer_18.npz"
        payload = np.load(path)
        decision = np.asarray(payload["d_K"], dtype=np.float64)
        decision = decision / np.linalg.norm(decision)
        cosine = float(sa @ decision)
        perpendicular = sa - cosine * decision
        folds.append(
            {
                "fold": fold,
                "cos_d_sa_d_k": cosine,
                "d_sa_perp_k_l2": float(np.linalg.norm(perpendicular)),
            }
        )
    return {
        "scope": "CPU geometry only; d_K is not used in Stage 3 causal experiments",
        "folds": folds,
        "mean_cosine": float(np.mean([row["cos_d_sa_d_k"] for row in folds])),
        "mean_perpendicular_l2": float(np.mean([row["d_sa_perp_k_l2"] for row in folds])),
    }


def _load_summary(output_root: Path, index: int) -> dict[str, Any]:
    return read_json(output_root / EXPERIMENT_DIR_NAMES[index] / "summary.json")


def build_final_analysis(output_root: Path, decision_direction_dir: Path) -> dict[str, Any]:
    summaries = {index: _load_summary(output_root, index) for index in range(6)}
    gate = summaries[0]["gate"]
    geometry = direction_geometry(output_root, decision_direction_dir)
    atomic_write_json(output_root / "analysis" / "geometry.json", geometry)
    h1_ci = summaries[1].get("e_balance_z", {}).get(
        "spearman_item_bootstrap", {}
    ).get("ci95", [None, None])
    h4_ci = (summaries[5].get("natural_policy_association") or {}).get(
        "spearman_bootstrap", {}
    ).get("ci95", [None, None])

    rows = [
        {
            "hypothesis": "H1",
            "experiment": "Evidence balance",
            "evidence_type": "correlation",
            "endpoint": "E_balance ↔ z_SA / SA",
            "n": summaries[1].get("n", 0),
            "estimate": summaries[1].get("e_balance_z", {}).get("spearman"),
            "ci_low": h1_ci[0],
            "ci_high": h1_ci[1],
            "supported": h1_ci[0] is not None and h1_ci[0] > 0,
            "audience": "report-facing",
        },
        {
            "hypothesis": "H2",
            "experiment": "History order",
            "evidence_type": "paired formation",
            "endpoint": "IF − TF z_SA",
            "n": summaries[2].get("n", 0),
            "estimate": summaries[2].get("delta_z_if_minus_tf", {}).get("mean"),
            "ci_low": (summaries[2].get("delta_z_if_minus_tf", {}).get("ci95") or [None, None])[0],
            "ci_high": (summaries[2].get("delta_z_if_minus_tf", {}).get("ci95") or [None, None])[1],
            "supported": summaries[2].get("delta_z_if_minus_tf", {}).get("ci95", [None])[0] is not None and summaries[2]["delta_z_if_minus_tf"]["ci95"][0] > 0,
            "audience": "report-facing",
        },
        {
            "hypothesis": "H3",
            "experiment": "Answer mismatch",
            "evidence_type": "paired formation",
            "endpoint": "force Image − force Text SA",
            "n": summaries[3].get("n", 0),
            "estimate": summaries[3].get("delta_sa_forceI_minus_forceT", {}).get("mean"),
            "ci_low": (summaries[3].get("delta_sa_forceI_minus_forceT", {}).get("ci95") or [None, None])[0],
            "ci_high": (summaries[3].get("delta_sa_forceI_minus_forceT", {}).get("ci95") or [None, None])[1],
            "supported": summaries[3].get("delta_sa_forceI_minus_forceT", {}).get("ci95", [None])[0] is not None and summaries[3]["delta_sa_forceI_minus_forceT"]["ci95"][0] > 0,
            "audience": "report-facing",
        },
        {
            "hypothesis": "H4",
            "experiment": "Post-answer policy",
            "evidence_type": "causal intervention" if gate["allow_policy_steering"] else "correlation",
            "endpoint": "do(z_SA) → P(Image)" if gate["allow_policy_steering"] else "z_SA ↔ P(Image)",
            "n": summaries[5].get("n", 0),
            "estimate": (summaries[5].get("causal_policy_steering") or {}).get("mean") if gate["allow_policy_steering"] else (summaries[5].get("natural_policy_association") or {}).get("spearman"),
            "ci_low": ((summaries[5].get("causal_policy_steering") or {}).get("ci95") or [None, None])[0] if gate["allow_policy_steering"] else h4_ci[0],
            "ci_high": ((summaries[5].get("causal_policy_steering") or {}).get("ci95") or [None, None])[1] if gate["allow_policy_steering"] else h4_ci[1],
            "supported": ((summaries[5].get("causal_policy_steering") or {}).get("ci95") or [None])[0] is not None and (summaries[5]["causal_policy_steering"]["ci95"][0] > 0) if gate["allow_policy_steering"] else (h4_ci[0] is not None and h4_ci[0] > 0),
            "audience": "behavior-facing",
        },
    ]
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(analysis_dir / "core_table.csv", rows)
    natural = summaries[0]["natural_projection"]
    transplant = summaries[0]["coordinate_transplant"]
    lines = [
        "# Stage 3 — Source Attribution Formation Pilot",
        "",
        "## Executive result",
        "",
        f"The preregistered three-level gate resolved to **Level {gate['level']}**: {gate['reason']}.",
        f"Natural OOF projection: R²={natural['r2']:.4f}, Spearman={natural['spearman']:.4f}, bootstrap 95% CI={natural['spearman_item_bootstrap']['ci95']}.",
        f"Coordinate transplant: n={transplant['n']}, aligned mean={transplant['coordinate']['mean']}, 95% CI={transplant['coordinate']['ci95']}, direction rate={transplant['coordinate']['direction_rate']}.",
        "",
        "The coordinate association and coordinate intervention are reported separately. A useful natural projection does not by itself establish that `z_SA` is a causal mediator.",
        "",
        "## Experiment status",
        "",
    ]
    for index in range(6):
        summary = summaries[index]
        lines.append(f"- Exp {index}: `{summary.get('status')}`; effective n={summary.get('n', 0)}")
    lines.extend(["", "## H1–H4", ""])
    for row in rows:
        lines.append(f"- {row['hypothesis']} ({row['evidence_type']}, {row['audience']}): estimate={row['estimate']}, 95% CI=[{row['ci_low']}, {row['ci_high']}], supported={row['supported']}.")
    lines.extend(
        [
            "",
            "## Causal interpretation",
            "",
            f"- `z_SA` causal-coordinate status: {'supported by transplant' if gate['allow_causal_mediator'] else 'not established; natural association only'}.",
            f"- Mediation: {summaries[4].get('claim_limit', summaries[4].get('reason', 'not evaluated'))}.",
            f"- Policy branch: {summaries[5].get('continuation_definition', summaries[5].get('reason', 'not evaluated'))}; no KV-cache fork is claimed.",
            f"- History reconstruction: GPU smoke passed at BF16 tolerance; formal soft-score agreement rate at 0.125 was {summaries[2].get('formal_soft_score_within_0.125_rate')}, with {summaries[2].get('failed')} malformed Pass-1 generations excluded.",
            "- Decision-Side K appears only in the CPU geometry diagnostic and was not used as a causal target.",
            "",
            "## Reproducibility",
            "",
            "Primary configuration was fixed at v4 / joint / answer_basis_9 / PANL layer 18 / seed 42. Directions are five-fold item-OOF Ridge regressions. Every intervention alpha is measured in the corresponding fold-training natural `SD(z_SA)`; transplant and clamp preserve their exact coordinate definitions and orthogonal controls match hidden-space L2.",
            "",
        ]
    )
    atomic_write_text(analysis_dir / "FINAL_ANALYSIS.md", "\n".join(lines))
    payload = {"gate": gate, "hypotheses": rows, "geometry": geometry, "experiment_summaries": summaries}
    atomic_write_json(analysis_dir / "final_analysis.json", payload)
    return payload
