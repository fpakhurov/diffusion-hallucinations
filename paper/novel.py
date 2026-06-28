"""Novel deterministic CE extensions (Б-variants), tested in direct order.

Б1 ordinal transition network (2D)  — spatial arrangement beyond the histogram
Б2 non-local ordinal autocorrelation — duplication / copy detector
Б3 Renyi-curve geometry              — compact geometric invariants
Б4 CE-anisotropy                     — directional CE differences

All are lightweight and deterministic. Each is evaluated on H1/H2 (and the
key structure-preserving E5 `repeat` case for Б1/Б2).
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import ce_features as cef
import evaluate as ev
import inject as inj


# ----------------------------------------------------------------------------
# shared: 2x2 ordinal-pattern index map (Lehmer code, 0..23)
# ----------------------------------------------------------------------------

def pattern_map(img: np.ndarray, d: int = 2) -> np.ndarray:
    w = sliding_window_view(img, (d, d))            # (h,w,d,d)
    h2, w2 = w.shape[:2]
    flat = w.reshape(h2, w2, d * d)
    ranks = np.argsort(np.argsort(flat, axis=-1), axis=-1)  # permutation of 0..n-1
    n = d * d
    idx = np.zeros((h2, w2), dtype=np.int64)
    for i in range(n):
        smaller = (ranks[..., i + 1:] < ranks[..., i:i + 1]).sum(-1)
        idx += smaller * math.factorial(n - 1 - i)
    return idx                                      # 0..n!-1


# ----------------------------------------------------------------------------
# Б1. Ordinal transition network
# ----------------------------------------------------------------------------

def b1_transition(img: np.ndarray, d: int = 2) -> np.ndarray:
    idx = pattern_map(img, d)
    K = math.factorial(d * d)
    T = np.zeros((K, K))
    # horizontal + vertical adjacent transitions
    for a, b in ((idx[:, :-1].ravel(), idx[:, 1:].ravel()),
                 (idx[:-1, :].ravel(), idx[1:, :].ravel())):
        np.add.at(T, (a, b), 1)
    T /= T.sum() + 1e-12
    p = T.ravel()
    nz = p[p > 0]
    trans_entropy = -(nz * np.log(nz)).sum() / math.log(K * K)
    self_frac = np.trace(T)                              # stay-in-pattern mass
    node = T.sum(1)
    node_imbalance = node.std()
    off = T - np.diag(np.diag(T))
    return np.array([trans_entropy, self_frac, node_imbalance, off.max()])


# ----------------------------------------------------------------------------
# Б2. Non-local ordinal autocorrelation (duplication detector)
# ----------------------------------------------------------------------------

def b2_duplication(img: np.ndarray, block: int = 16, d: int = 2) -> np.ndarray:
    idx = pattern_map(img, d)
    K = math.factorial(d * d)
    h, w = idx.shape
    nb_h, nb_w = h // block, w // block
    hists = []
    centers = []
    for i in range(nb_h):
        for j in range(nb_w):
            b = idx[i*block:(i+1)*block, j*block:(j+1)*block].ravel()
            hist = np.bincount(b, minlength=K).astype(float)
            hist /= hist.sum() + 1e-12
            hists.append(hist)
            centers.append((i, j))
    hists = np.array(hists)
    centers = np.array(centers)
    # pairwise L1 distance between NON-adjacent blocks (spatial sep > 1)
    best = []
    n = len(hists)
    for a, bb in combinations(range(n), 2):
        if np.abs(centers[a] - centers[bb]).max() <= 1:
            continue
        best.append(np.abs(hists[a] - hists[bb]).sum())
    best = np.array(best)
    # low min-distance => a near-duplicate block pair exists somewhere
    return np.array([best.min(), np.percentile(best, 1), np.percentile(best, 5),
                     (best < 0.15).mean()])


# ----------------------------------------------------------------------------
# Б3. Renyi-curve geometry
# ----------------------------------------------------------------------------

def b3_geometry(img: np.ndarray, scheme: str = "row", d: int = 5) -> np.ndarray:
    sig = cef.to_signal(img, scheme)
    out = np.asarray(__import__("ordpy").renyi_complexity_entropy(
        sig, alpha=cef.DEFAULT_ALPHAS, dx=d))
    H, C = out[:, 0], out[:, 1]
    dH, dC = np.diff(H), np.diff(C)
    seg = np.hypot(dH, dC)
    arc = seg.sum()
    init_slope = dC[0] / (dH[0] + 1e-9)
    curv = np.diff(np.arctan2(dC, dH))                  # turning per step
    return np.array([arc, init_slope, curv.mean(), curv.std(),
                     H.max() - H.min(), C.max()])


# ----------------------------------------------------------------------------
# Б4. CE-anisotropy
# ----------------------------------------------------------------------------

def b4_anisotropy(img: np.ndarray, d: int = 5) -> np.ndarray:
    feats = {s: cef.ce_curve_1d(cef.to_signal(img, s), dx=d) for s in ("row", "col", "diag")}
    row, col, diag = feats["row"], feats["col"], feats["diag"]
    return np.concatenate([np.abs(row - col), np.abs(row - diag), np.abs(col - diag)])


VARIANTS = {"Б1-transition": b1_transition, "Б2-duplication": b2_duplication,
            "Б3-geometry": b3_geometry, "Б4-anisotropy": b4_anisotropy}


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def eval_h1h2(ds, fn, name):
    X = np.vstack([fn(im) for im in ds.images])
    out = []
    for task, mask, y in ev.tasks_from_groups(ds.group):
        r = ev.evaluate_task(X[mask], y, task)
        out.append((task, r))
        print(f"  {name:<15} {task:<18} dim={X.shape[1]:<2} "
              f"AUROC={r['auroc']:.3f} [{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}]")
    return out


def eval_repeat(ds, fn, name, sev=0.4):
    """Key test: does the variant detect the structure-preserving repeat artifact?"""
    reals = [im for im, g in zip(ds.images, ds.group) if g == "real"]
    imgs, y, groups = [], [], []
    for i, base in enumerate(reals):
        rng = np.random.default_rng(1000 * i + int(sev * 100))
        imgs.append(base); y.append(0); groups.append(i)
        imgs.append(inj.repeat_patch(base, sev, rng)); y.append(1); groups.append(i)
    X = np.vstack([fn(im) for im in imgs])
    y = np.array(y); groups = np.array(groups)
    pt, lo, hi = inj.grouped_auroc(X, y, groups)
    sig = "*" if lo > 0.5 else " "
    print(f"  {name:<15} E5-repeat(sev={sev}) AUROC={pt:.3f} [{lo:.3f},{hi:.3f}] {sig}")
    return pt, lo, hi


if __name__ == "__main__":
    import sys
    ds = ev.load_dataset(size=256)
    print(f"loaded {len(ds.images)} images")
    rows = []
    order = ["Б1-transition", "Б2-duplication", "Б3-geometry", "Б4-anisotropy"]
    for name in order:
        fn = VARIANTS[name]
        print(f"=== {name} ===")
        h = eval_h1h2(ds, fn, name)
        rep = (None, None, None)
        if name in ("Б1-transition", "Б2-duplication"):
            rep = eval_repeat(ds, fn, name)
        for task, r in h:
            rows.append({"variant": name, "task": task, "dim": None,
                         "auroc": r["auroc"], "lo": r["auroc_lo"], "hi": r["auroc_hi"]})
        if rep[0] is not None:
            rows.append({"variant": name, "task": "E5-repeat",
                         "auroc": rep[0], "lo": rep[1], "hi": rep[2]})
    import pandas as pd
    pd.DataFrame(rows).to_csv(
        ev.Path(__file__).resolve().parent / "results" / "novel.csv", index=False)
    print("wrote results/novel.csv")
