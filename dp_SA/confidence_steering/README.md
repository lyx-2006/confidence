# LAT confidence measured-subspace orthogonal steering

This package constructs a confidence-related LAT direction using only the frozen
training split, removes measured difficulty and LAT-SA subspaces, and evaluates
the resulting intervention at a downstream PANL readout and the final SAC.

The permanent development split is `outer_fold != 0` for construction and
`outer_fold == 0` for direction audit. The frozen 100-record formal test manifest
is opened only by the locked formal runtime after CPU audit and a matching GPU
smoke have succeeded.

The shared PANL SA readout is `P1_PANL × L18`. A block-output intervention at
LAT L16 cannot affect the earlier PANL L14 representation; PANL L18 is after all
four steering layers L10/L12/L14/L16 and is therefore the common downstream
readout.

Run the audit-set smoke:

```bash
python -m dp_SA.confidence_steering.run_pipeline --smoke --num-gpus 1
```

After a successful matching smoke, the one-GPU formal command is:

```bash
python -m dp_SA.confidence_steering.run_pipeline \
  --output-root dp_SA/confidence_steering/output/orthogonal_results \
  --num-gpus 1
```

For two GPUs, change the final argument to `--num-gpus 2`. Use `--resume` only
with an existing output directory whose complete semantic fingerprint matches.

The natural-scale SA decomposition is a separate mechanism-diagnostic run spec:

```bash
python -m dp_SA.confidence_steering.run_pipeline --smoke --num-gpus 1 \
  --directions confidence_raw confidence_parallel_sa confidence_perp_sa_natural_scale \
  --layers 14 16 \
  --alphas -2 -1 0 1 2
```

After that exact smoke succeeds, its formal command uses the same arguments plus
`--output-root dp_SA/confidence_steering/output/natural_decomposition`. This
configuration has 3,000 main trials and 2,700 GPU forwards for 100 cases. It
does not request a shuffle direction, so it creates no null trials or null
analysis. Its paired statistics are S1 and S2 only.

Result figures are split by endpoint and direction. Final soft-SA plots are in
`figures/final/<direction>.png`; PANL L18 probe-SA plots are in
`figures/panl/<direction>.png`. Each figure contains one direction only, with
LAT layer on the x-axis and alpha encoded by line style and color.

The experiment only claims deletion of measured linear SA/difficulty subspaces.
Numerical orthogonality is an implementation check, not evidence that every
possible SA or difficulty signal was removed.

The matched random-SA-subspace null is an isolated follow-up. Its smoke reuses
the completed 24-case natural-decomposition smoke read-only:

```bash
python -m dp_SA.confidence_steering.run_pipeline --smoke --num-gpus 1 \
  --random-sa-null-repeats 3 --random-sa-null-layer 14 --random-sa-null-dose 2
```

After the matching 168-forward smoke is locked, the formal one-GPU command is:

```bash
python -m dp_SA.confidence_steering.run_pipeline --resume --num-gpus 1 \
  --random-sa-null-repeats 20 --random-sa-null-layer 14 --random-sa-null-dose 2
```

Use `--num-gpus 2` for the two-GPU form. The formal stage adds 100 fresh clean
validation forwards and 4,000 null forwards. It writes only the dedicated
random-null namespace inside `output/natural_decomposition`; it neither runs
the main pipeline nor reads or updates the older shuffle null.
