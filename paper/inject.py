"""E5: controlled artifact injection (removes the source confound of H2).

Take REAL images and inject local artifacts of graded severity, producing
matched clean-vs-corrupted pairs that differ ONLY by the injected artifact.
Measure how well CE vs a spectral baseline detects each artifact as a function
of severity. CV is grouped by base image so a clean image and its corrupted
twin never span train/test.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import swirl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ce_features as cef
import baselines as bl
import evaluate as ev


# ----------------------------------------------------------------------------
# Artifact injectors: img in [0,1], deterministic via per-call rng
# ----------------------------------------------------------------------------

def _rect(rng, h, w, frac):
    ph, pw = max(2, int(h * frac)), max(2, int(w * frac))
    y0 = rng.integers(0, h - ph + 1)
    x0 = rng.integers(0, w - pw + 1)
    return y0, x0, ph, pw


def blur_patch(img, frac, rng):
    out = img.copy()
    y0, x0, ph, pw = _rect(rng, *img.shape, frac)
    sigma = 1 + 6 * frac
    out[y0:y0+ph, x0:x0+pw] = gaussian_filter(img[y0:y0+ph, x0:x0+pw], sigma)
    return out


def junk_patch(img, frac, rng):
    out = img.copy()
    y0, x0, ph, pw = _rect(rng, *img.shape, frac)
    out[y0:y0+ph, x0:x0+pw] = rng.random((ph, pw))
    return out


def swirl_patch(img, frac, rng):
    h, w = img.shape
    cy, cx = rng.integers(h // 4, 3 * h // 4), rng.integers(w // 4, 3 * w // 4)
    radius = int(min(h, w) * (0.15 + 0.5 * frac))
    return swirl(img, center=(cx, cy), strength=4 * frac * 5, radius=radius, mode="reflect")


def repeat_patch(img, frac, rng):
    out = img.copy()
    y0, x0, ph, pw = _rect(rng, *img.shape, frac)
    y1 = rng.integers(0, img.shape[0] - ph + 1)
    x1 = rng.integers(0, img.shape[1] - pw + 1)
    out[y1:y1+ph, x1:x1+pw] = img[y0:y0+ph, x0:x0+pw]
    return out


ARTIFACTS = {"blur": blur_patch, "junk": junk_patch,
             "swirl": swirl_patch, "repeat": repeat_patch}
SEVERITIES = [0.1, 0.2, 0.4]


# ----------------------------------------------------------------------------
# Grouped evaluation (group = base image id)
# ----------------------------------------------------------------------------

def grouped_auroc(X, y, groups, n_splits=5, n_repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    acc, cnt = np.zeros(n), np.zeros(n)
    for _ in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=int(rng.integers(1 << 30)))
        for tr, te in sgkf.split(X, y, groups):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(X[tr], y[tr])
            acc[te] += clf.predict_proba(X[te])[:, 1]
            cnt[te] += 1
    score = acc / np.maximum(cnt, 1)
    # bootstrap over base groups (resample whole pairs)
    uniq = np.unique(groups)
    boot = np.empty(2000)
    for b in range(2000):
        gsamp = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in gsamp])
        if len(np.unique(y[idx])) < 2:
            boot[b] = np.nan
            continue
        boot[b] = roc_auc_score(y[idx], score[idx])
    point = roc_auc_score(y, score)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return point, lo, hi


def ce_feat(img):
    return cef.image_features(img, scheme="row", dx=5, taux=1)


FEATS = {"CE": ce_feat, "FFT": bl.fft_radial}


def main():
    ds = ev.load_dataset(size=256)
    reals = [im for im, g in zip(ds.images, ds.group) if g == "real"]
    print(f"{len(reals)} real base images")

    rows = []
    for art_name, art_fn in ARTIFACTS.items():
        for sev in SEVERITIES:
            imgs, y, groups = [], [], []
            for i, base in enumerate(reals):
                rng = np.random.default_rng(1000 * i + int(sev * 100))
                imgs.append(base);      y.append(0); groups.append(i)
                imgs.append(art_fn(base, sev, rng)); y.append(1); groups.append(i)
            y = np.array(y); groups = np.array(groups)
            for fname, ffn in FEATS.items():
                X = np.vstack([ffn(im) for im in imgs])
                pt, lo, hi = grouped_auroc(X, y, groups)
                rows.append({"artifact": art_name, "severity": sev,
                             "feat": fname, "auroc": pt, "lo": lo, "hi": hi})
                sig = "*" if lo > 0.5 else " "
                print(f"{art_name:<7} sev={sev:<4} {fname:<4} "
                      f"AUROC={pt:.3f} [{lo:.3f},{hi:.3f}] {sig}")

    import pandas as pd
    pd.DataFrame(rows).to_csv(
        ev.Path(__file__).resolve().parent / "results" / "inject.csv", index=False)
    print("\nwrote results/inject.csv")


if __name__ == "__main__":
    main()
