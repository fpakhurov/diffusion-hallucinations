"""Local CE maps + pooling (extension А3).

Global CE averages over the whole image and may miss localized defects. We
compute CE on a grid of patches and pool the per-patch (H,C) by mean/std/min/
max/percentiles. Max-pooling of local complexity is the lever for localized
artifacts. We test whether pooled local CE recovers signal that global CE
misses, on H1, H2, and the localized E5 artifacts.
"""
from __future__ import annotations

import numpy as np
import ordpy

import ce_features as cef
import evaluate as ev
import inject as inj


def local_ce(img, patch=64, stride=32, dx=2, dy=2):
    Hs, Cs = [], []
    h, w = img.shape
    for i in range(0, h - patch + 1, stride):
        for j in range(0, w - patch + 1, stride):
            H, C = ordpy.complexity_entropy(img[i:i+patch, j:j+patch], dx=dx, dy=dy)
            Hs.append(H); Cs.append(C)
    Hs, Cs = np.array(Hs), np.array(Cs)

    def st(a):
        return [a.mean(), a.std(), a.min(), a.max(),
                np.percentile(a, 10), np.percentile(a, 90)]
    return np.array(st(Hs) + st(Cs))


def global_ce(img):
    return cef.image_features(img, scheme="row", dx=5, taux=1)


def main():
    ds = ev.load_dataset(size=256)
    tasks = ev.tasks_from_groups(ds.group)
    feats = {"local-CE+pool (12)": local_ce, "global CE (20)": global_ce}

    print("=== H1 / H2 ===")
    for name, fn in feats.items():
        X = np.vstack([fn(im) for im in ds.images])
        for tname, mask, y in tasks:
            r = ev.evaluate_task(X[mask], y, tname)
            print(f"  {name:<20} {tname:<18} AUROC={r['auroc']:.3f} "
                  f"[{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}]")

    print("\n=== localized E5 artifacts (sev 0.2), grouped CV ===")
    reals = [im for im, g in zip(ds.images, ds.group) if g == "real"]
    arts = {"blur": inj.blur_patch, "junk": inj.junk_patch, "repeat": inj.repeat_patch}
    for art_name, art_fn in arts.items():
        imgs, y, groups = [], [], []
        for i, base in enumerate(reals):
            rng = np.random.default_rng(50 * i + 7)
            imgs.append(base); y.append(0); groups.append(i)
            imgs.append(art_fn(base, 0.2, rng)); y.append(1); groups.append(i)
        y = np.array(y); groups = np.array(groups)
        for name, fn in feats.items():
            X = np.vstack([fn(im) for im in imgs])
            pt, lo, hi = inj.grouped_auroc(X, y, groups)
            sig = "*" if lo > 0.5 else " "
            print(f"  {art_name:<7} {name:<20} AUROC={pt:.3f} [{lo:.3f},{hi:.3f}] {sig}")


if __name__ == "__main__":
    main()
