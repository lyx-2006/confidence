# Delayed-SA unimodal-difficulty LAT swap

This package runs the preregistered same-answer LAT residual-state swap for image and text difficulty. It reads existing delayed-SA and PANL-information artifacts but never modifies them.

Fresh preflight, CPU tests, and the bounded GPU smoke:

```bash
python -m dp_SA.lat_difficulty_swap.run_pipeline
```

The pipeline stops after smoke and writes the formal commands to `output/results/progress/formal_command.txt`. It does not launch the formal experiment.

The stages can also be run explicitly:

```bash
python -m dp_SA.lat_difficulty_swap.build_pairs
python -m dp_SA.lat_difficulty_swap.run --resume
python -m dp_SA.lat_difficulty_swap.analyze --resume
```

Any non-empty formal result directory requires `--resume`, and resume is accepted only when the frozen fingerprints and trial grid match.
