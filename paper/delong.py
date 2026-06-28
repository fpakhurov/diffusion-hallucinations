"""Fast DeLong test for correlated/uncorrelated ROC AUC comparisons.

Implements Sun & Xu (2014) fast DeLong. Used to test whether AUROC differences
between feature sets are significant, and whether a single AUROC exceeds 0.5.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(preds, m):
    """preds: (k, n) scores, positives (m) first then negatives (n)."""
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01).reshape(k, k)
    sy = np.cov(v10).reshape(k, k)
    cov = sx / m + sy / n
    return aucs, cov


def _order(y):
    order = np.argsort(-y, kind="stable")   # positives (label 1) first
    return order, int(y.sum())


def delong_test(y, s1, s2):
    """Two-sided DeLong test that AUC(s1) != AUC(s2) on the same samples."""
    y = np.asarray(y); order, m = _order(y)
    preds = np.vstack([np.asarray(s1)[order], np.asarray(s2)[order]])
    aucs, cov = _fast_delong(preds, m)
    l = np.array([1.0, -1.0])
    var = l @ cov @ l
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * norm.sf(abs(z)))


def delong_vs_chance(y, s, one_sided=True):
    """Test AUC(s) > 0.5."""
    y = np.asarray(y); order, m = _order(y)
    aucs, cov = _fast_delong(np.asarray(s)[order][None, :], m)
    auc, var = float(aucs[0]), float(cov[0, 0])
    z = (auc - 0.5) / np.sqrt(var) if var > 0 else 0.0
    p = norm.sf(z) if one_sided else 2 * norm.sf(abs(z))
    return auc, np.sqrt(var), float(z), float(p)


def benjamini_hochberg(pvals, alpha=0.05):
    """Return boolean array of rejections under BH-FDR."""
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    rej = np.zeros(n, dtype=bool)
    if passed.any():
        kmax = np.where(passed)[0].max()
        rej[order[:kmax + 1]] = True
    return rej
