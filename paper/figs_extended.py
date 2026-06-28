"""Plots: AUROC vs embedding dimension d and vs delay tau (wide ranges)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "results" / "extended_sweep.csv")
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

# --- d-axis ---
d = df[df.axis == "d"].sort_values("value")
ax = axes[0]
ax.errorbar(d.value, d.real_vs_synthetic_auroc,
            yerr=[d.real_vs_synthetic_auroc - d.real_vs_synthetic_lo,
                  d.real_vs_synthetic_hi - d.real_vs_synthetic_auroc],
            fmt="-o", color="C0", capsize=2, label="H1: real vs synthetic", lw=1.5, ms=4)
ax.errorbar(d.value, d.bad_vs_good_auroc,
            yerr=[d.bad_vs_good_auroc - d.bad_vs_good_lo,
                  d.bad_vs_good_hi - d.bad_vs_good_auroc],
            fmt="-s", color="C3", capsize=2, label="H2: bad vs good", lw=1, ms=3, alpha=0.7)
ax.axhline(0.5, color="k", ls="--", lw=1)
ax2 = ax.twinx()
ax2.plot(d.value, d.mean_H, ":", color="C2", lw=2, label="средняя норм. энтропия")
ax2.set_ylabel("средняя норм. энтропия H", color="C2")
ax2.tick_params(axis="y", labelcolor="C2")
ax.set_xscale("log")
ax.set_xlabel("размерность вложения d (log)")
ax.set_ylabel("AUROC (95% ДИ)")
ax.set_title("Качество vs d: коллапс при N/d!→0")
ax.legend(loc="lower left", fontsize=9)

# --- tau-axis ---
t = df[df.axis == "tau"].sort_values("value")
ax = axes[1]
ax.errorbar(t.value, t.real_vs_synthetic_auroc,
            yerr=[t.real_vs_synthetic_auroc - t.real_vs_synthetic_lo,
                  t.real_vs_synthetic_hi - t.real_vs_synthetic_auroc],
            fmt="-o", color="C0", capsize=2, label="H1: real vs synthetic", lw=1.5, ms=4)
ax.errorbar(t.value, t.bad_vs_good_auroc,
            yerr=[t.bad_vs_good_auroc - t.bad_vs_good_lo,
                  t.bad_vs_good_hi - t.bad_vs_good_auroc],
            fmt="-s", color="C3", capsize=2, label="H2: bad vs good", lw=1, ms=3, alpha=0.7)
ax.axhline(0.5, color="k", ls="--", lw=1)
ax.set_xscale("log")
ax.set_xlabel("задержка τ (log)")
ax.set_ylabel("AUROC (95% ДИ)")
ax.set_title("Качество vs τ (порядки величины)")
ax.legend(loc="lower left", fontsize=9)

fig.tight_layout()
fig.savefig(HERE / "results" / "fig_extended.png", bbox_inches="tight")
print("wrote results/fig_extended.png")
print("\nd-axis H1:", list(zip(d.value, d.real_vs_synthetic_auroc.round(3))))
print("tau-axis H1:", list(zip(t.value, t.real_vs_synthetic_auroc.round(3))))
