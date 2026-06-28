"""A lightweight, deterministic copy-move (duplication) detector & localizer
built on ordinal patterns — the standalone development of Б2.

Practical edge (from Lemma 1, monotone invariance): ordinal patterns are
invariant to any monotone tone change of the pasted region, so the detector
survives brightness/gamma manipulation that breaks intensity-correlation
baselines. Deterministic, O(N * #offsets), fits in <1 GB RAM at 256x256.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import binary_opening, label

import ce_features as cef
import evaluate as ev

# block-hashing copy-move: identical source/paste blocks vote the exact offset
BLK, BSTEP, MINSH = 10, 1, 24        # block size, sampling step, min shift
MIN_DIV = 5                          # min distinct symbols per block (texture filter)
MIN_AREA = 150                       # min connected matched area (px)


# ----------------------------------------------------------------------------
# ground-truth copy-move sample (paste may get a monotone tone change)
# ----------------------------------------------------------------------------

def make_copymove(img, frac, rng, gamma=1.0, bright=0.0):
    h, w = img.shape
    ph = pw = max(8, int(round(min(h, w) * np.sqrt(frac))))
    sy, sx = rng.integers(0, h - ph + 1), rng.integers(0, w - pw + 1)
    dy, dx = rng.integers(0, h - ph + 1), rng.integers(0, w - pw + 1)
    out = img.copy()
    patch = img[sy:sy+ph, sx:sx+pw]
    patch = np.clip(patch ** gamma + bright, 0, 1)      # monotone tone edit
    out[dy:dy+ph, dx:dx+pw] = patch
    src = np.zeros((h, w), bool); src[sy:sy+ph, sx:sx+pw] = True
    pst = np.zeros((h, w), bool); pst[dy:dy+ph, dx:dx+pw] = True
    return out, (src | pst)


# ----------------------------------------------------------------------------
# ordinal pattern map (2x2 Lehmer index)
# ----------------------------------------------------------------------------

def pattern_map(img, d=2):
    w = sliding_window_view(img, (d, d))
    flat = w.reshape(w.shape[0], w.shape[1], d * d)
    ranks = np.argsort(np.argsort(flat, -1), -1)
    n = d * d
    idx = np.zeros(w.shape[:2], np.int64)
    for i in range(n):
        idx += (ranks[..., i+1:] < ranks[..., i:i+1]).sum(-1) * math.factorial(n-1-i)
    return idx


def _match_at(P, dy, dx):
    """boolean map where pattern(p) == pattern(p+(dy,dx))."""
    h, w = P.shape
    a = P[max(0, -dy):h-max(0, dy), max(0, -dx):w-max(0, dx)]
    b = P[max(0, dy):h-max(0, -dy), max(0, dx):w-max(0, -dx)]
    eq = (a == b)
    full = np.zeros((h, w), bool)
    full[max(0, -dy):h-max(0, dy), max(0, -dx):w-max(0, dx)] = eq
    return full


def _clean(m):
    m = binary_opening(m, iterations=2)
    lab, n = label(m)
    keep = np.zeros_like(m)
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() >= MIN_AREA:
            keep |= comp
    return keep


def _vote_offset(S):
    """S: integer signal/pattern map. Identical BLKxBLK blocks far apart vote
    their offset; the peak is the exact copy-move displacement."""
    from collections import Counter, defaultdict
    h, w = S.shape
    groups = defaultdict(list)
    Sb = S.astype(np.int16)
    for i in range(0, h - BLK, BSTEP):
        for j in range(0, w - BLK, BSTEP):
            blk = Sb[i:i+BLK, j:j+BLK]
            if np.unique(blk).size < MIN_DIV:           # skip low-texture
                continue
            groups[blk.tobytes()].append((i, j))
    votes = Counter()
    for pts in groups.values():
        if not (2 <= len(pts) <= 60):
            continue
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                dy, dx = pts[b][0]-pts[a][0], pts[b][1]-pts[a][1]
                if dy*dy + dx*dx >= MINSH*MINSH:
                    if dy < 0 or (dy == 0 and dx < 0):
                        dy, dx = -dy, -dx
                    votes[(dy, dx)] += 1
    return votes.most_common(1)[0] if votes else ((0, 0), 0)


def _localize(P):
    (dy, dx), nvotes = _vote_offset(P)
    if nvotes == 0:
        return np.zeros(P.shape, bool), 0.0
    m = _clean(_match_at(P, dy, dx) | _match_at(P, -dy, -dx))
    return m, m.sum() / P.size                          # score = duplicated-area fraction


def ordinal_localize(img):
    mask, score = _localize(pattern_map(img))
    full = np.zeros(img.shape, bool)
    full[:mask.shape[0], :mask.shape[1]] = mask
    return full, score


def intensity_localize(img):
    """Baseline: same pipeline on quantized intensities (no ordinal coding)."""
    Q = np.round(img * 23).astype(np.int64)[:-1, :-1]   # match #pattern symbols
    mask, score = _localize(Q)
    full = np.zeros(img.shape, bool)
    full[:mask.shape[0], :mask.shape[1]] = mask
    return full, score


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def f1(pred, gt):
    tp = (pred & gt).sum()
    if tp == 0:
        return 0.0
    prec = tp / pred.sum() if pred.sum() else 0.0
    rec = tp / gt.sum() if gt.sum() else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    ds = ev.load_dataset(size=256)
    reals = [im for im, g in zip(ds.images, ds.group) if g == "real"]
    print(f"{len(reals)} base images\n")
    print(f"{'cond':<22}{'method':<10}{'locF1':>8}{'detAUROC':>10}")
    rows = []
    for gamma, tag in [(1.0, "copy (no tone edit)"), (2.2, "copy + gamma 2.2"),
                       (0.45, "copy + gamma 0.45")]:
        for mname, fn in [("ordinal", ordinal_localize), ("intensity", intensity_localize)]:
            f1s, scores, ys = [], [], []
            for i, base in enumerate(reals):
                rng = np.random.default_rng(100 * i + int(gamma * 100))
                # positive: copy-move; negative: clean
                cm, gt = make_copymove(base, 0.08, rng, gamma=gamma)
                pm, sc = fn(cm); f1s.append(f1(pm, gt)); scores.append(sc); ys.append(1)
                _, sc0 = fn(base); scores.append(sc0); ys.append(0)
            auroc = roc_auc_score(ys, scores)
            print(f"{tag:<22}{mname:<10}{np.mean(f1s):>8.3f}{auroc:>10.3f}")
            rows.append({"cond": tag, "gamma": gamma, "method": mname,
                         "locF1": float(np.mean(f1s)), "detAUROC": float(auroc)})
    import pandas as pd
    pd.DataFrame(rows).to_csv(ev.Path(__file__).resolve().parent / "results" / "copymove.csv", index=False)
    print("\nwrote results/copymove.csv")
