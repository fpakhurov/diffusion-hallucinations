"""А-extensions: amplitude-aware dispersion CE (А1) and multiscale 2D-CE (А2).

Both are deterministic and O(N). А1 incorporates amplitude via the NCDF
quantization of dispersion entropy (Rostaghi & Azami 2016); А2 follows the
multiscale 2D complexity-entropy plane (Zunino & Ribeiro 2016).
"""
from __future__ import annotations

import numpy as np
import ordpy
from scipy.stats import norm

import ce_features as cef

ALPHAS = cef.DEFAULT_ALPHAS


# ----------------------------------------------------------------------------
# А1. Dispersion complexity-entropy (amplitude-aware)
# ----------------------------------------------------------------------------

def dispersion_probs(sig: np.ndarray, c: int = 6, m: int = 3, tau: int = 1) -> np.ndarray:
    s = np.asarray(sig, float)
    y = norm.cdf((s - s.mean()) / (s.std() + 1e-9))
    z = np.clip(np.round(c * y + 0.5), 1, c).astype(np.int64) - 1   # 0..c-1
    N = len(z)
    cols = [z[i * tau: N - (m - 1 - i) * tau] for i in range(m)]
    L = min(len(col) for col in cols)
    emb = np.stack([col[:L] for col in cols], axis=1)
    base = c ** np.arange(m)[::-1]
    pid = emb @ base
    p = np.bincount(pid, minlength=c ** m).astype(float)
    return p / p.sum()


def _shannon(p):
    p = p[p > 0]
    return -(p * np.log(p)).sum()


def _js_complexity(p):
    n = len(p)
    pu = np.ones(n) / n
    mm = 0.5 * (p + pu)
    djs = _shannon(mm) - 0.5 * _shannon(p) - 0.5 * _shannon(pu)
    dmax = -0.5 * (((n + 1) / n) * np.log(n + 1) - 2 * np.log(2 * n) + np.log(n))
    H = _shannon(p) / np.log(n)
    return (djs / dmax) * H


def _renyi_norm(p, alpha, n):
    q = p[p > 0]
    if abs(alpha - 1) < 1e-9:
        H = -(q * np.log(q)).sum()
    elif alpha == 0:
        H = np.log(len(q))
    else:
        H = 1.0 / (1.0 - alpha) * np.log((q ** alpha).sum())
    return H / np.log(n)


def dispersion_ce(img: np.ndarray, c: int = 6, m: int = 3, tau: int = 1,
                  scheme: str = "row", alphas=ALPHAS) -> np.ndarray:
    p = dispersion_probs(cef.to_signal(img, scheme), c=c, m=m, tau=tau)
    n = c ** m
    H = np.array([_renyi_norm(p, a, n) for a in alphas])
    C = _js_complexity(p)
    return np.concatenate([H, [C]])


# ----------------------------------------------------------------------------
# А2. Multiscale 2D complexity-entropy (Shannon, several spatial scales)
# ----------------------------------------------------------------------------

def multiscale2d_ce(img: np.ndarray, scales=(1, 2, 4, 8, 16),
                    dx: int = 2, dy: int = 2) -> np.ndarray:
    out = []
    for s in scales:
        H, C = ordpy.complexity_entropy(img, dx=dx, dy=dy, taux=s, tauy=s)
        out += [H, C]
    return np.array(out)


# weighted permutation entropy gap (amplitude sensitivity), compact aux feature
def wpe_gap(img: np.ndarray, scheme: str = "row", ds=(3, 4, 5, 6)) -> np.ndarray:
    sig = cef.to_signal(img, scheme)
    feats = []
    for d in ds:
        wpe = ordpy.weighted_permutation_entropy(sig, dx=d)
        pe = ordpy.permutation_entropy(sig, dx=d)
        feats += [wpe, pe, wpe - pe]
    return np.array(feats)


METHODS = {
    "CE (base)": lambda im: cef.image_features(im, scheme="row", dx=5, taux=1),
    "А1 dispersion": dispersion_ce,
    "А2 multiscale2d": multiscale2d_ce,
    "WPE-gap": wpe_gap,
}
