"""Plot short-prompt attention-block paired metrics."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "output" / "attention_block"
METRICS = ["logit_change_diff", "token_change_rate"]
LABELS = {"logit_change_diff": "Logit change diff", "token_change_rate": "Token change rate"}

for exp in ("PANL2CLE", "CLE2SAC"):
    rows = list(csv.DictReader((ROOT / exp / "tables" / "condition_effects.csv").open()))
    for metric in METRICS:
        use = [r for r in rows if r["prompt"] == "short" and r["group"] == "answer_equal_macro" and r["metric"] == metric]
        windows = sorted({(int(r["window_start"]), int(r["window_end"])) for r in use})
        labels = [f"L{s}–{e}" for s, e in windows]
        fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=160)
        for condition, label, color in (("C1_main_block", "Main block", "C0"), ("C2_source_plus_1_control", "Source+1 control", "C1")):
            selected = {(int(r["window_start"]), int(r["window_end"])): r for r in use if r["condition"] == condition}
            vals = [float(selected[w]["mean"]) for w in windows]
            lows = [float(selected[w]["ci95_low"]) for w in windows]
            highs = [float(selected[w]["ci95_high"]) for w in windows]
            yerr = [[v - lo for v, lo in zip(vals, lows)], [hi - v for v, hi in zip(vals, highs)]]
            ax.errorbar(range(len(windows)), vals, yerr=yerr, fmt="o-", capsize=4, lw=1.5, color=color, label=label)
        ax.axhline(0, color="black", lw=0.8, alpha=0.7)
        ax.set_xticks(range(len(windows)), labels)
        ax.set_ylabel(LABELS[metric])
        ax.set_xlabel("Layer window")
        ax.set_title(f"Short prompt {exp}: {LABELS[metric]}")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out = ROOT / exp / "figures" / f"short_{metric}_main_vs_control_lines.png"
        fig.savefig(out)
        plt.close(fig)
        print(out)
