# Delayed-SA Answer-force

This experiment asks whether the fixed answer shown in Phase 1 conditionally
updates delayed source attribution. It reuses the frozen clean delayed-SA
capture and its item-level OOF probe; it never treats the forced answer as a
naturally generated Phase-0 answer.

The package writes only to a new run directory under
`dp_SA/answer_force/outputs/`. Existing delayed-SA artifacts are read-only.

## Commands

Run the formal grid explicitly after the smoke gate:

```bash
python -m dp_SA.answer_force.run --output-root dp_SA/answer_force/outputs/<run_name>
python -m dp_SA.answer_force.analyze --output-root dp_SA/answer_force/outputs/<run_name>
```

Run the guarded CPU gate and 2+2 GPU smoke (this command never launches the
formal grid):

```bash
python -m dp_SA.answer_force.run_pipeline --run-name pipeline_seed42
```

`--resume` is accepted only when the complete run fingerprint and frozen
manifests match. The primary probe cell is always `P1_PANL`, layer 14.
Existing run and pipeline directories are never overwritten; choose a new
`--run-name`/`--output-root` or explicitly resume a completed run.

Each run also records `input_fingerprint.json` (with the plural compatibility
alias), `clean_parity_audit.json`, `probe_leakage_audit.json`,
`token_matched_aggregate_metrics.csv`, `correlations.csv`,
`clean_force_soft_sa_correlations.csv`,
`specificity_metrics.csv`, `hard_class_directional_proportions.csv`,
`delta_sa_absolute_overall.png`, and `failures.jsonl`. The overall absolute
delta chart intentionally pools text/image origins; the original faceted
charts remain available.
