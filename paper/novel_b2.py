"""Develop Б2: a stronger non-local ordinal duplication descriptor + severity curve.

b2_offset: offset-autocorrelation of the ordinal-pattern-index map. A copied
patch at displacement delta makes pattern indices match across that shift; the
maximum match fraction over non-trivial offsets (minus the chance collision
rate) is a deterministic copy-move signal robust to unknown displacement.
"""
from __future__ import annotations

import math

import numpy as np

import ce_features as cef
import evaluate as ev
import inject as inj
import novel


def b2_offset(img: np.ndarray, d: int = 2, step: int = 8, max_shift: int = 128) -> np.ndarray:
    idx = novel.pattern_map(img, d)
    K = math.factorial(d * d)
    p = np.bincount(idx.ravel(), minlength=K) / idx.size
    chance = (p ** 2).sum()                       # expected match by chance
    h, w = idx.shape
    fracs = []
    for dy in range(0, max_shift + 1, step):
        for dx in range(0, max_shift + 1, step):
            if dy == 0 and dx == 0:
                continue
            a = idx[:h - dy, :w - dx]
            b = idx[dy:, dx:]
            fracs.append((a == b).mean())
        if dy == 0:
            continue
    fracs = np.array(fracs) - chance
    return np.array([fracs.max(), np.percentile(fracs, 99), np.percentile(fracs, 95),
                     fracs.mean()])


def b2_combined(img):
    return np.concatenate([novel.b2_duplication(img), b2_offset(img)])


VARIANTS = {"Б2-block (orig)": novel.b2_duplication,
            "Б2-offset (new)": b2_offset,
            "Б2-combined": b2_combined,
            "CE-hist (baseline)": lambda im: cef.image_features(im, scheme="row", dx=5)}

SEVS = [0.1, 0.2, 0.4]


def repeat_set(reals, sev):
    imgs, y, groups = [], [], []
    for i, base in enumerate(reals):
        rng = np.random.default_rng(1000 * i + int(sev * 100))
        imgs.append(base); y.append(0); groups.append(i)
        imgs.append(inj.repeat_patch(base, sev, rng)); y.append(1); groups.append(i)
    return imgs, np.array(y), np.array(groups)


def main():
    ds = ev.load_dataset(size=256)
    reals = [im for im, g in zip(ds.images, ds.group) if g == "real"]
    print(f"{len(reals)} real base images")
    rows = []
    for name, fn in VARIANTS.items():
        line = []
        for sev in SEVS:
            imgs, y, groups = repeat_set(reals, sev)
            X = np.vstack([fn(im) for im in imgs])
            pt, lo, hi = inj.grouped_auroc(X, y, groups)
            rows.append({"variant": name, "sev": sev, "auroc": pt, "lo": lo, "hi": hi})
            line.append(f"sev{sev}:{pt:.3f}[{lo:.3f},{hi:.3f}]{'*' if lo > 0.5 else ' '}")
        print(f"  {name:<20} " + "  ".join(line))
    import pandas as pd
    pd.DataFrame(rows).to_csv(ev.Path(__file__).resolve().parent / "results" / "novel_b2.csv", index=False)
    print("wrote results/novel_b2.csv")


if __name__ == "__main__":
    main()
