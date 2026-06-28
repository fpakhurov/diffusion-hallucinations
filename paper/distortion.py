"""Distortion view: real vs synthetic as a continuous deviation in CE space.

(a) Fit a Gaussian to the CE features of REAL images; the Mahalanobis distance
    to that "real manifold" is a threshold-free distortion / anomaly score.
    Report AUROC(real vs synthetic) from the score alone (no class labels in
    the scorer) and its monotonic growth under graded injected distortion.
(b) Migration test: along the real-vs-synthetic LDA axis, where do
    blur/noise-distorted REAL images land relative to genuine synthetic ones?
    Tests whether "synthetic looks like an over-smoothed real".
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import ce_features as cef
import evaluate as ev

FKW = dict(scheme="row", dx=5, taux=1)


def feats(images):
    return np.vstack([cef.image_features(im, **FKW) for im in images])


def blur(img, sigma):
    return gaussian_filter(img, sigma)


def add_noise(img, std, seed):
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, std, img.shape), 0, 1)


# ----------------------------------------------------------------------------
# (a) Mahalanobis distortion score
# ----------------------------------------------------------------------------

def mahalanobis_scores(X_real_train, X_eval):
    mu = X_real_train.mean(0)
    cov = LedoitWolf().fit(X_real_train)
    prec = cov.precision_
    d = X_eval - mu
    return np.einsum("ij,jk,ik->i", d, prec, d)  # squared Mahalanobis


def run_mahalanobis(ds, seed=0):
    Xall = feats(ds.images)
    g = ds.group
    real = Xall[g == "real"]
    synth_mask = np.isin(g, ["good", "bad"])
    y = (~synth_mask).astype(int)  # 1 = real, 0 = synthetic; but we score distortion
    # CV over real for an honest "real manifold" (don't fit on eval reals)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    score = np.zeros(len(ds.images))
    # fit manifold on real-train folds, score everything OOF-style for reals;
    # synthetics scored by manifold fit on all real (they're never in the fit)
    real_idx = np.where(g == "real")[0]
    for tr, te in skf.split(real, np.zeros(len(real))):
        mu_prec_train = real[tr]
        s = mahalanobis_scores(mu_prec_train, real[te])
        score[real_idx[te]] = s
    score[synth_mask] = mahalanobis_scores(real, Xall[synth_mask])
    # AUROC: higher distortion => more synthetic
    auroc = roc_auc_score(synth_mask.astype(int), score)
    print(f"(a) Mahalanobis distortion -> AUROC(synthetic) = {auroc:.3f}")
    print(f"    median dist: real={np.median(score[g=='real']):.1f} "
          f"good={np.median(score[g=='good']):.1f} bad={np.median(score[g=='bad']):.1f}")
    return score, auroc


def run_severity(ds, seed=0):
    """Mahalanobis distortion vs injected blur/noise severity on REAL images."""
    reals = [im for im, gg in zip(ds.images, ds.group) if gg == "real"]
    Xreal = feats(reals)
    print("(a') distortion score vs injected severity (real images):")
    for kind in ("blur", "noise"):
        line = []
        for s in [0.0, 0.5, 1.0, 2.0, 4.0]:
            if s == 0:
                imgs = reals
            elif kind == "blur":
                imgs = [blur(im, s) for im in reals]
            else:
                imgs = [add_noise(im, s * 0.05, 7) for im in reals]
            X = feats(imgs)
            d = mahalanobis_scores(Xreal, X)
            line.append((s, float(np.median(d))))
        print(f"    {kind:<6} " + "  ".join(f"s={s}:{v:.1f}" for s, v in line))


# ----------------------------------------------------------------------------
# (b) Migration along the real-vs-synthetic LDA axis
# ----------------------------------------------------------------------------

def run_migration(ds, seed=0):
    Xall = feats(ds.images)
    g = ds.group
    synth = np.isin(g, ["good", "bad"]).astype(int)
    lda = LinearDiscriminantAnalysis().fit(Xall, synth)
    proj = lda.transform(Xall).ravel()  # higher => more synthetic side
    reals = [im for im, gg in zip(ds.images, ds.group) if gg == "real"]

    def proj_of(imgs):
        return lda.transform(feats(imgs)).ravel()

    real_proj = proj[g == "real"]
    synth_proj = proj[synth == 1]
    blur_proj = proj_of([blur(im, 2.0) for im in reals])
    noise_proj = proj_of([add_noise(im, 0.15, 7) for im in reals])
    print("(b) LDA axis (higher = more synthetic side); medians:")
    print(f"    real={np.median(real_proj):+.2f}  synthetic={np.median(synth_proj):+.2f}")
    print(f"    real+blur(σ=2)={np.median(blur_proj):+.2f}  "
          f"real+noise={np.median(noise_proj):+.2f}")
    frac_blur = (blur_proj > np.median(real_proj)).mean()
    print(f"    fraction of blurred reals past the real median: {frac_blur:.2f} "
          f"(toward synthetic if >0.5)")


if __name__ == "__main__":
    ds = ev.load_dataset(size=256)
    print(f"loaded {len(ds.images)} images")
    run_mahalanobis(ds)
    run_severity(ds)
    run_migration(ds)
