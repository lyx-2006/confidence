# Delayed-SA steering → selective probe

This isolated package runs Phase 0 answer generation, delayed-SA clean capture, activation
steering, FDR candidate selection, and Ridge probing in that order. The primary outcome is
the restricted-nine-class soft SA image score. SAC is steered as a direct-output reference
but excluded from the default probe family.

```bash
python -m dp_SA.capture --resume
python -m dp_SA.steering --resume
python -m dp_SA.analysis
python -m dp_SA.probe
python -m dp_SA.run_pipeline --resume
```

GPU smoke uses a separate output root. The formal experiment is launched with nohup:

```bash
nohup bash dp_SA/scripts/run_full_nohup.sh >/dev/null 2>&1 & echo $! > dp_SA/outputs/wrapper.pid
python dp_SA/scripts/status.py
```

Test selection is seeded random sampling from clean argmax classes 5–8 (image-side) and
0–3 (text-side), after excluding construction items. Class 4 is excluded. Construction
directions still use 25 highest and 25 lowest soft-SA records with unique items.
