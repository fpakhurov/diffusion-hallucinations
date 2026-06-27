"""E4: baseline feature extractors vs CE on the same evaluation protocol.

Answers: is CE's real-vs-synthetic signal competitive, or trivially beaten by
a cheap spectral/texture baseline? Same LogReg + repeated CV + bootstrap CI.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sstats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

import ce_features as cef
import evaluate as ev


# ----------------------------------------------------------------------------
# Baseline feature extractors (grayscale image in [0,1], 256x256)
# ----------------------------------------------------------------------------

def fft_radial(img: np.ndarray, n_bins: int = 24) -> np.ndarray:
    """Log radial power spectrum — captures generator high-freq fingerprints."""
    f = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    power = np.abs(f) ** 2
    h, w = img.shape
    cy, cx = h / 2, w / 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = r.max()
    bins = np.linspace(0, r_max, n_bins + 1)
    idx = np.clip(np.digitize(r.ravel(), bins) - 1, 0, n_bins - 1)
    prof = np.zeros(n_bins)
    pr = power.ravel()
    for b in range(n_bins):
        m = idx == b
        prof[b] = pr[m].mean() if m.any() else 0.0
    prof = np.log1p(prof)
    hf = prof[n_bins // 2:].sum() / (prof.sum() + 1e-9)  # high-freq fraction
    return np.concatenate([prof, [hf]])


def lbp_hist(img: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    lbp = local_binary_pattern((img * 255).astype(np.uint8), P, R, method="uniform")
    n_bins = P + 2
    h, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return h


def glcm_feats(img: np.ndarray, levels: int = 32) -> np.ndarray:
    q = (img * (levels - 1)).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 2, 4], angles=[0, np.pi / 2],
                        levels=levels, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    return np.concatenate([graycoprops(glcm, p).ravel() for p in props])


def pixel_stats(img: np.ndarray) -> np.ndarray:
    gx = np.diff(img, axis=1)
    gy = np.diff(img, axis=0)
    grad = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)
    flat = img.ravel()
    return np.array([
        flat.mean(), flat.std(), sstats.skew(flat), sstats.kurtosis(flat),
        grad.mean(), grad.std(),
        np.percentile(flat, 10), np.percentile(flat, 90),
    ])


BASELINES = {
    "FFT-radial": fft_radial,
    "LBP": lbp_hist,
    "GLCM/Haralick": glcm_feats,
    "pixel-stats": pixel_stats,
    "FFT+GLCM+LBP": lambda im: np.concatenate([fft_radial(im), glcm_feats(im), lbp_hist(im)]),
}


def build(ds, fn):
    return np.vstack([fn(im) for im in ds.images])


def main():
    ds = ev.load_dataset(size=256)
    tasks = ev.tasks_from_groups(ds.group)

    rows = []
    # reference: CE best config
    Xce = np.vstack([cef.image_features(im, scheme="row", dx=5, taux=1) for im in ds.images])
    feats = {"CE (row d5)": Xce}
    for name, fn in BASELINES.items():
        feats[name] = build(ds, fn)

    for name, X in feats.items():
        for task, mask, y in tasks:
            r = ev.evaluate_task(X[mask], y, task)
            rows.append({"feat": name, "task": task, "dim": X.shape[1],
                         "auroc": r["auroc"], "lo": r["auroc_lo"], "hi": r["auroc_hi"],
                         "ap": r["ap"]})
            print(f"{name:<16} {task:<18} dim={X.shape[1]:<3} "
                  f"AUROC={r['auroc']:.3f} [{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}]")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(ev.Path(__file__).resolve().parent / "results" / "baselines.csv", index=False)
    print("\n=== H1 ranking (real_vs_synthetic) ===")
    h1 = df[df.task == "real_vs_synthetic"].sort_values("auroc", ascending=False)
    print(h1[["feat", "dim", "auroc", "lo", "hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
