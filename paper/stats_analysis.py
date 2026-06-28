"""Statistical-rigor analysis: DeLong tests and multiple-comparison control.

(a) H1: pairwise DeLong among feature sets (is CE significantly different from
    FFT / LBP / combo?).
(b) E5-repeat: DeLong of Б2 vs Б1 vs plain CE histogram (is Б2's gain real?).
(c) H2: DeLong-vs-chance p-value for each of the 44 sweep configs, then
    Benjamini-Hochberg FDR control (how many survive?).
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import baselines as bl
import ce_features as cef
import delong
import evaluate as ev
import inject as inj
import novel


def oof_scores(X, y, n_splits=5, n_repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    acc, cnt = np.zeros(len(y)), np.zeros(len(y))
    for _ in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=int(rng.integers(1 << 30)))
        for tr, te in skf.split(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[tr], y[tr])
            acc[te] += clf.predict_proba(X[te])[:, 1]; cnt[te] += 1
    return acc / np.maximum(cnt, 1)


def grouped_oof(X, y, groups, n_splits=5, n_repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    acc, cnt = np.zeros(len(y)), np.zeros(len(y))
    for _ in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=int(rng.integers(1 << 30)))
        for tr, te in sgkf.split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[tr], y[tr])
            acc[te] += clf.predict_proba(X[te])[:, 1]; cnt[te] += 1
    return acc / np.maximum(cnt, 1)


def part_a(ds):
    print("\n=== (a) H1 pairwise DeLong (real vs synthetic) ===")
    g = ds.group
    y = (g == "real").astype(int)
    fsets = {
        "CE": np.vstack([cef.image_features(im, scheme="row", dx=5, taux=1) for im in ds.images]),
        "FFT": bl.build(ds, bl.fft_radial),
        "LBP": bl.build(ds, bl.lbp_hist),
        "combo": bl.build(ds, bl.BASELINES["FFT+GLCM+LBP"]),
    }
    scores = {k: oof_scores(X, y) for k, X in fsets.items()}
    base = "CE"
    for other in ["FFT", "LBP", "combo"]:
        a1, a2, z, p = delong.delong_test(y, scores[base], scores[other])
        print(f"  {base} ({a1:.3f})  vs  {other} ({a2:.3f}):  z={z:+.2f}  p={p:.3f}"
              f"  {'(diff significant)' if p < 0.05 else '(not significant)'}")


def part_b(ds, sev=0.4):
    print("\n=== (b) E5-repeat DeLong (does Б2 beat Б1 / plain CE?) ===")
    reals = [im for im, gg in zip(ds.images, ds.group) if gg == "real"]
    imgs, y, groups = [], [], []
    for i, base in enumerate(reals):
        rng = np.random.default_rng(1000 * i + int(sev * 100))
        imgs.append(base); y.append(0); groups.append(i)
        imgs.append(inj.repeat_patch(base, sev, rng)); y.append(1); groups.append(i)
    y = np.array(y); groups = np.array(groups)
    fsets = {
        "Б2": np.vstack([novel.b2_duplication(im) for im in imgs]),
        "Б1": np.vstack([novel.b1_transition(im) for im in imgs]),
        "CE-hist": np.vstack([cef.image_features(im, scheme="row", dx=5) for im in imgs]),
    }
    scores = {k: grouped_oof(X, y, groups) for k, X in fsets.items()}
    for other in ["Б1", "CE-hist"]:
        a1, a2, z, p = delong.delong_test(y, scores["Б2"], scores[other])
        print(f"  Б2 ({a1:.3f})  vs  {other} ({a2:.3f}):  z={z:+.2f}  p={p:.3f}"
              f"  {'(Б2 significantly better)' if p < 0.05 and a1 > a2 else '(not significant)'}")
    auc, se, z, p = delong.delong_vs_chance(y, scores["Б2"])
    print(f"  Б2 vs chance: AUROC={auc:.3f}  SE={se:.3f}  one-sided p={p:.4f}")


def part_c(ds):
    print("\n=== (c) H2 multiple-comparison control over 44 configs ===")
    mask = np.isin(ds.group, ["good", "bad"])
    y = (ds.group[mask] == "bad").astype(int)
    configs = []
    for scheme, d, tau in itertools.product(["row", "col", "snake", "hilbert", "diag"],
                                            [3, 4, 5, 6], [1, 2]):
        configs.append(dict(scheme=scheme, dx=d, taux=tau))
    for dx, dy in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        configs.append(dict(scheme="2d", dx=dx, dy=dy))
    pvals, aucs = [], []
    for c in configs:
        X = np.vstack([cef.image_features(im, **c) for im in ds.images])[mask]
        s = oof_scores(X, y, n_repeats=3)
        auc, se, z, p = delong.delong_vs_chance(y, s)
        pvals.append(p); aucs.append(auc)
    pvals = np.array(pvals); aucs = np.array(aucs)
    raw_sig = (pvals < 0.05).sum()
    rej = delong.benjamini_hochberg(pvals, alpha=0.05)
    print(f"  configs: {len(configs)}   best AUROC={aucs.max():.3f} (p={pvals[aucs.argmax()]:.3f})")
    print(f"  raw p<0.05 (one-sided, uncorrected): {raw_sig}/{len(configs)}")
    print(f"  survive Benjamini-Hochberg FDR 0.05: {rej.sum()}/{len(configs)}")


if __name__ == "__main__":
    ds = ev.load_dataset(size=256)
    print(f"loaded {len(ds.images)} images")
    part_a(ds)
    part_b(ds)
    part_c(ds)
