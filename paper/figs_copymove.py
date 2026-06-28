"""Figures for the ordinal copy-move localizer: qualitative panel + F1 bars."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import copymove as cm
import evaluate as ev

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 10, "figure.dpi": 130})

# --- qualitative panel (gamma-edited paste) ---
ds = ev.load_dataset(size=256)
reals = [im for im, g in zip(ds.images, ds.group) if g == "real"]
rng = np.random.default_rng(3)
img, gt = cm.make_copymove(reals[3], 0.08, rng, gamma=2.2)
mo, _ = cm.ordinal_localize(img)
mi, _ = cm.intensity_localize(img)

fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
for a in ax:
    a.axis("off")
ax[0].imshow(img, cmap="gray"); ax[0].set_title("подделка (вставка + гамма 2.2)")
ax[1].imshow(gt, cmap="gray"); ax[1].set_title("истинная маска дубля")
ax[2].imshow(mo, cmap="gray"); ax[2].set_title(f"ординальный (F1={cm.f1(mo,gt):.2f})")
ax[3].imshow(mi, cmap="gray"); ax[3].set_title(f"интенсивностный (F1={cm.f1(mi,gt):.2f})")
fig.suptitle("Локализация copy-move при тоновой правке вставки", y=1.02)
fig.savefig(HERE / "results" / "fig_copymove_qual.png", bbox_inches="tight")
print("wrote fig_copymove_qual.png")

# --- F1 by condition x method ---
df = pd.read_csv(HERE / "results" / "copymove.csv")
conds = list(dict.fromkeys(df["cond"]))
x = np.arange(len(conds)); w = 0.36
fig2, ax2 = plt.subplots(figsize=(7.2, 4))
for i, (mth, color) in enumerate([("ordinal", "C0"), ("intensity", "C3")]):
    vals = [df[(df["cond"] == c) & (df.method == mth)].locF1.iloc[0] for c in conds]
    ax2.bar(x + (i - 0.5) * w, vals, w, label=mth, color=color)
ax2.set_xticks(x); ax2.set_xticklabels(["без правки", "гамма 2.2", "гамма 0.45"])
ax2.set_ylabel("локализация F1"); ax2.set_ylim(0, 1)
ax2.set_title("Ординальный детектор устойчив к тоновой правке вставки")
ax2.legend()
fig2.savefig(HERE / "results" / "fig_copymove_f1.png", bbox_inches="tight")
print("wrote fig_copymove_f1.png")
