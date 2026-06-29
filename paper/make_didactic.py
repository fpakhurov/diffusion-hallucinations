"""Didactic illustrations: method pipeline + ordinal-pattern / invariance."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "results"
BLUE, ACC, INK = "#102D69", "#1C7293", "#222222"
plt.rcParams.update({"font.size": 11, "figure.dpi": 140})


# ---------------------------------------------------------------- pipeline ---
def pipeline():
    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 2.5); ax.axis("off")
    steps = ["Изображение", "Оттенки серого\n+ развёртка 2D→1D",
             "Ординальные\nшаблоны (d, τ)", "Кривая Реньи\n(Hα, Cα)",
             "Логистическая\nрегрессия", "H1 / H2"]
    w, h, gap = 1.62, 1.2, 0.21
    x = 0.15
    for i, s in enumerate(steps):
        fc = BLUE if i in (0, len(steps) - 1) else "white"
        tc = "white" if i in (0, len(steps) - 1) else INK
        ax.add_patch(FancyBboxPatch((x, 0.65), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=BLUE, lw=1.6))
        ax.text(x + w / 2, 1.25, s, ha="center", va="center", color=tc, fontsize=10.5)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 1.25), (x + w + gap, 1.25),
                                         arrowstyle="-|>", mutation_scale=16, color=ACC, lw=2))
        x += w + gap
    ax.text(5.5, 0.25, "детерминированный, лёгкий ($O(N)$) дескриптор; обучается только финальный классификатор",
            ha="center", va="center", fontsize=9.5, style="italic", color="#555")
    fig.savefig(OUT / "fig_pipeline.png", bbox_inches="tight"); print("fig_pipeline.png")


# ------------------------------------------------- ordinal pattern + invar ---
def ordinal():
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for a in ax:
        a.axis("off")
    # panel 1: 2x2 patch with values
    vals = np.array([[40, 90], [60, 120]])
    ax[0].set_title("Окно 2×2 (яркости)", fontsize=11, color=BLUE)
    for (i, j), v in np.ndenumerate(vals):
        ax[0].add_patch(plt.Rectangle((j, 1 - i), 1, 1, fc="#EAF0FB", ec=BLUE, lw=1.6))
        ax[0].text(j + 0.5, 1 - i + 0.5, str(v), ha="center", va="center", fontsize=15, color=INK)
    ax[0].set_xlim(-0.2, 2.2); ax[0].set_ylim(-0.2, 2.2)
    ax[0].text(1, -0.15, "значения по возрастанию:\n40 < 60 < 90 < 120", ha="center", va="top", fontsize=9.5, color="#555")
    # panel 2: ranks -> pattern index
    ax[1].set_title("Порядковый шаблон", fontsize=11, color=BLUE)
    ax[1].text(0.5, 0.78, "ранги (порядок): ", ha="center", fontsize=11, color=INK)
    ax[1].text(0.5, 0.62, "(0, 2, 1, 3)", ha="center", fontsize=15, color=ACC)
    ax[1].text(0.5, 0.40, "→ шаблон #k из 24", ha="center", fontsize=12, color=INK)
    ax[1].text(0.5, 0.18, "признак = частоты\nвсех 24 шаблонов", ha="center", fontsize=9.5, color="#555")
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
    # panel 3: monotone invariance
    vals2 = np.clip((vals / 255.0) ** 2.2 * 255, 0, 255).astype(int)
    ax[2].set_title("Инвариантность к гамме", fontsize=11, color=BLUE)
    for (i, j), v in np.ndenumerate(vals2):
        ax[2].add_patch(plt.Rectangle((j, 1 - i), 1, 1, fc="#F3ECFB", ec=BLUE, lw=1.6))
        ax[2].text(j + 0.5, 1 - i + 0.5, str(v), ha="center", va="center", fontsize=14, color=INK)
    ax[2].set_xlim(-0.2, 2.2); ax[2].set_ylim(-0.4, 2.2)
    ax[2].text(1, -0.2, "гамма меняет значения,\nно порядок и шаблон — те же", ha="center", va="top", fontsize=9.5, color="#555")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ordinal.png", bbox_inches="tight"); print("fig_ordinal.png")


if __name__ == "__main__":
    pipeline()
    ordinal()


def taxonomy():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.3)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    rows = [("Синтетичность", "стат. отличие от реального", "видит", "#2C7A3F"),
            ("Текстурный артефакт", "размытие, шум", "видит", "#2C7A3F"),
            ("Дублирование", "повтор фрагмента", "только спец. метод", "#B8860B"),
            ("Семантическая ошибка", "лишние конечности, текст", "не видит", "#C0392B")]
    ax.text(2.0, 3.75, "Тип ошибки", fontsize=11, color="#102D69", ha="center", weight="bold")
    ax.text(8.05, 3.75, "Глобальный CE", fontsize=11, color="#102D69", ha="center", weight="bold")
    y = 3.0
    for name, desc, verdict, col in rows:
        ax.add_patch(plt.Rectangle((0.3, y - 0.35), 5.6, 0.72, fc="#EAF0FB", ec="#102D69", lw=1.3))
        ax.text(0.5, y + 0.08, name, fontsize=12, color="#102D69", va="center", weight="bold")
        ax.text(0.5, y - 0.18, desc, fontsize=9, color="#555", va="center")
        ax.add_patch(plt.Rectangle((6.3, y - 0.35), 3.4, 0.72, fc=col, ec="none", alpha=0.16))
        ax.text(8.0, y, verdict, fontsize=11.5, color=col, ha="center", va="center", weight="bold")
        y -= 0.85
    fig.savefig(OUT / "fig_taxonomy.png", bbox_inches="tight"); print("fig_taxonomy.png")


if __name__ == "__main__":
    taxonomy()
