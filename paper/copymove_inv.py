"""Rotation/flip/scale-tolerant ordinal copy-move localizer.

Per-pixel descriptor = canonical cyclic-minimum of the ordinal ranks of N
points sampled on concentric rings. Image rotation cyclically shifts ring
samples, so the canonical form is rotation-invariant; ordinal ranks keep the
monotone-tone invariance (Lemma 1); several radii give scale tolerance.
Pixels whose descriptor recurs at a far-away location, forming a contiguous
region, are flagged as duplicated — no constant-offset assumption, so it
handles rotated/reflected copies that pure translation voting misses.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_opening, label, map_coordinates, rotate, zoom

import copymove as cmv
import evaluate as ev

N = 8                      # ring samples
RADII = (2.0, 3.5)        # concentric rings (scale tolerance)
MINSH = 24                # min separation to count as a copy
MIN_AREA = 150


# ----------------------------------------------------------------------------
# geometric copy-move generator (transform applied to the pasted patch)
# ----------------------------------------------------------------------------

def make_copymove_geom(img, frac, rng, transform="translate", gamma=1.0):
    h, w = img.shape
    ph = pw = max(12, int(round(min(h, w) * np.sqrt(frac))))
    sy, sx = rng.integers(0, h - ph + 1), rng.integers(0, w - pw + 1)
    patch = img[sy:sy+ph, sx:sx+pw].copy()
    if transform == "flip":
        patch = patch[:, ::-1]
    elif transform == "rot90":
        patch = np.rot90(patch, 1)
    elif transform == "rot30":
        patch = rotate(patch, 30, reshape=False, order=1, mode="reflect")
    elif transform == "scale0.8":
        z = zoom(patch, 0.8, order=1)
        patch = np.zeros((ph, pw)); patch[:z.shape[0], :z.shape[1]] = z
    patch = np.clip(patch ** gamma, 0, 1)
    dy, dx = rng.integers(0, h - ph + 1), rng.integers(0, w - pw + 1)
    out = img.copy(); out[dy:dy+ph, dx:dx+pw] = patch
    gt = np.zeros((h, w), bool)
    gt[sy:sy+ph, sx:sx+pw] = True
    gt[dy:dy+ph, dx:dx+pw] = True
    return out, gt


# ----------------------------------------------------------------------------
# rotation-invariant ring-ordinal signature
# ----------------------------------------------------------------------------

def _ring_canon(img, r, n=N):
    h, w = img.shape
    Y, X = np.mgrid[0:h, 0:w]
    ang = 2 * np.pi * np.arange(n) / n
    S = np.stack([map_coordinates(img, [Y + r * np.sin(a), X + r * np.cos(a)],
                                  order=1, mode="reflect") for a in ang])  # (n,h,w)
    ranks = np.argsort(np.argsort(S, axis=0), axis=0).astype(np.int64)
    base = (n ** np.arange(n)).reshape(n, 1, 1)
    codes = None
    for seq in (ranks, ranks[::-1]):                 # dihedral: shifts + reflection
        for s in range(n):
            c = (np.roll(seq, -s, axis=0) * base).sum(0)
            codes = c if codes is None else np.minimum(codes, c)
    return codes


def signature(img):
    K = N ** N
    sig = np.zeros(img.shape, np.int64)
    for i, r in enumerate(RADII):
        sig = sig * K + _ring_canon(img, r)
    return sig


def localize_inv(img, bstep=1):
    """Flag pixels whose ring-signature recurs >= MINSH away in a contiguous region."""
    from collections import defaultdict
    sig = signature(img)
    h, w = sig.shape
    groups = defaultdict(list)
    for i in range(0, h, bstep):
        row = sig[i]
        for j in range(0, w, bstep):
            groups[int(row[j])].append((i, j))
    matched = np.zeros((h, w), bool)
    for pts in groups.values():
        if len(pts) < 2 or len(pts) > 60:          # skip unique & over-common
            continue
        P = np.array(pts)
        far = (np.abs(P[:, None, 0] - P[None, :, 0]) >= MINSH) | \
              (np.abs(P[:, None, 1] - P[None, :, 1]) >= MINSH)
        keep = far.any(1)
        for (yy, xx), k in zip(pts, keep):
            if k:
                matched[yy:yy+bstep, xx:xx+bstep] = True
    matched = binary_opening(matched, iterations=1)
    lab, n = label(matched)
    out = np.zeros((h, w), bool)
    for kk in range(1, n + 1):
        comp = lab == kk
        if comp.sum() >= MIN_AREA:
            out |= comp
    return out, out.sum() / sig.size


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    ds = ev.load_dataset(size=256)
    reals = [im for im, g in zip(ds.images, ds.group) if g == "real"]
    print(f"{len(reals)} base images\n{'transform':<12}{'method':<14}{'locF1':>8}{'detAUROC':>10}")
    rows = []
    methods = {"ordinal-ring(inv)": localize_inv,
               "block-hash(transl)": cmv.ordinal_localize}
    for tf in ["translate", "flip", "rot90", "rot30", "scale0.8"]:
        for mname, fn in methods.items():
            f1s, sc, ys = [], [], []
            for k, base in enumerate(reals):
                rng = np.random.default_rng(7 * k + 1)
                x, gt = make_copymove_geom(base, 0.09, rng, transform=tf, gamma=2.2)
                pm, s = fn(x); f1s.append(cmv.f1(pm, gt)); sc.append(s); ys.append(1)
                _, s0 = fn(base); sc.append(s0); ys.append(0)
            auroc = roc_auc_score(ys, sc)
            print(f"{tf:<12}{mname:<14}{np.mean(f1s):>8.3f}{auroc:>10.3f}")
            rows.append({"transform": tf, "method": mname,
                         "locF1": float(np.mean(f1s)), "detAUROC": float(auroc)})
    import pandas as pd
    pd.DataFrame(rows).to_csv(ev.Path(__file__).resolve().parent / "results" / "copymove_inv.csv", index=False)
    print("\nwrote results/copymove_inv.csv")
