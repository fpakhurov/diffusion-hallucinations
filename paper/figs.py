"""Publication figures from the parameter sweep."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "results" / "sweep.csv")
FIG = HERE / "results"

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130})

SCHEMES = ["row", "snake", "hilbert", "col", "diag"]
DS = [3, 4, 5, 6]


def heatmap(task, tau, ax):
    sub = df[(df.task == task) & (df.family == "1d") & (df.tau == tau)]
    M = np.full((len(SCHEMES), len(DS)), np.nan)
    for i, s in enumerate(SCHEMES):
        for j, d in enumerate(DS):
            v = sub[(sub.scheme == s) & (sub.d == d)]["auroc"]
            if len(v):
                M[i, j] = v.iloc[0]
    im = ax.imshow(M, cmap="viridis", vmin=0.45, vmax=0.87, aspect="auto")
    ax.set_xticks(range(len(DS)), DS)
    ax.set_yticks(range(len(SCHEMES)), SCHEMES)
    ax.set_xlabel("embedding dim d")
    ax.set_title(f"{task}\n(tau={tau})", fontsize=10)
    for i in range(len(SCHEMES)):
        for j in range(len(DS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.7 else "black", fontsize=8)
    return im


# Figure 1: AUROC heatmaps, real-vs-synthetic, tau 1 vs 2
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
im = heatmap("real_vs_synthetic", 1, axes[0])
heatmap("real_vs_synthetic", 2, axes[1])
fig.colorbar(im, ax=axes, label="AUROC", fraction=0.046, pad=0.04)
fig.suptitle("Parameter sensitivity: real-vs-synthetic discrimination", y=1.02)
fig.savefig(FIG / "fig1_heatmap_real_synthetic.png", bbox_inches="tight")

# Figure 2: the two tasks side by side, tau=1, with chance line
fig2, ax = plt.subplots(figsize=(7.5, 4))
for task, color, marker in [("real_vs_synthetic", "C0", "o"),
                            ("bad_vs_good", "C3", "s")]:
    sub = df[(df.task == task)].sort_values("auroc").reset_index(drop=True)
    x = np.arange(len(sub))
    ax.errorbar(x, sub.auroc,
                yerr=[sub.auroc - sub.auroc_lo, sub.auroc_hi - sub.auroc],
                fmt=marker, color=color, ms=4, lw=0.8, capsize=2,
                label=task, alpha=0.8)
ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
ax.set_xlabel("parameterizations (sorted by AUROC)")
ax.set_ylabel("AUROC (95% bootstrap CI)")
ax.set_title("CE detects synthetic images, not hallucinations")
ax.legend()
fig2.savefig(FIG / "fig2_two_tasks.png", bbox_inches="tight")

# Figure 3: tau effect (paired) for real-vs-synthetic
fig3, ax = plt.subplots(figsize=(5, 4))
piv = df[(df.task == "real_vs_synthetic") & (df.family == "1d")].pivot_table(
    index=["scheme", "d"], columns="tau", values="auroc")
for (scheme, d), row in piv.iterrows():
    ax.plot([1, 2], [row[1], row[2]], "-o", color="C0", alpha=0.4, ms=4)
ax.set_xticks([1, 2])
ax.set_xlabel("delay tau")
ax.set_ylabel("AUROC")
ax.set_title("Delay tau=1 dominates tau=2\n(real-vs-synthetic)")
fig3.savefig(FIG / "fig3_tau_effect.png", bbox_inches="tight")

print("wrote 3 figures to", FIG)
print("\nrange summary:")
for task in df.task.unique():
    s = df[df.task == task].auroc
    print(f"  {task:<18} AUROC {s.min():.3f}-{s.max():.3f}  "
          f"(spread {s.max()-s.min():.3f})")
