"""E5 figure: AUROC vs artifact severity, CE vs FFT, per artifact type."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "results" / "inject.csv")
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})

arts = ["blur", "junk", "swirl", "repeat"]
titles = {"blur": "Размытие (текстура)", "junk": "Шумовой патч (текстура)",
          "swirl": "Геом. искажение", "repeat": "Копия патча (структура)"}
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
for ax, art in zip(axes, arts):
    sub = df[df.artifact == art]
    for feat, color in [("CE", "C0"), ("FFT", "C3")]:
        s = sub[sub.feat == feat].sort_values("severity")
        ax.errorbar(s.severity, s.auroc,
                    yerr=[s.auroc - s.lo, s.hi - s.auroc],
                    fmt="-o", color=color, capsize=3, label=feat, lw=1.5, ms=5)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_title(titles[art], fontsize=10)
    ax.set_xlabel("сила артефакта")
    ax.set_xticks([0.1, 0.2, 0.4])
axes[0].set_ylabel("AUROC (95% ДИ)")
axes[0].legend(loc="upper left", fontsize=9)
fig.suptitle("E5: обнаружение контролируемых артефактов (CE слеп к структуросохраняющему «repeat»)", y=1.04)
fig.savefig(HERE / "results" / "fig_inject.png", bbox_inches="tight")
print("wrote results/fig_inject.png")
