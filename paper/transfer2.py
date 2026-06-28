"""Lightweight transferable variants: fix the resolution confound + test
shape-based and unsupervised descriptors for zero-shot transfer.

Resize BOTH datasets to a common canonical size, then transfer. Compare:
  CE (absolute H,C),  Б3 geometry (curve shape),  Mahalanobis (unsupervised).
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ce_features as cef
import cifake as cf
import evaluate as ev
import novel

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

SIZES = [32, 64, 128]


def ce_feat(im):
    return cef.image_features(im, scheme="row", dx=5, taux=1)


def b3_feat(im):
    return novel.b3_geometry(im, scheme="row", d=5)


SUP_METHODS = {"CE": ce_feat, "Б3-geometry": b3_feat}


def cv_auroc(X, y, seed=0):
    rng = np.random.default_rng(seed)
    acc, cnt = np.zeros(len(y)), np.zeros(len(y))
    for _ in range(3):
        skf = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1 << 30)))
        for tr, te in skf.split(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[tr], y[tr]); acc[te] += clf.predict_proba(X[te])[:, 1]; cnt[te] += 1
    return roc_auc_score(y, acc / np.maximum(cnt, 1))


def transfer(Xtr, ytr, Xte, yte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def maha_scores(real_train, X):
    mu = real_train.mean(0)
    prec = LedoitWolf().fit(real_train).precision_
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, prec, d)


def maha_transfer(Xa, ya, Xb, yb):
    """Fit real-manifold on A reals, score B (unsupervised on B); AUROC(synthetic)."""
    real_a = Xa[ya == 1]
    s_b = maha_scores(real_a, Xb)          # higher = more synthetic
    return roc_auc_score((yb == 0).astype(int), s_b)


def main(n_cifake=400):
    path = cf.get_parquet()
    rows = []
    for S in SIZES:
        ds = ev.load_dataset(size=S)
        y_loc = (ds.group == "real").astype(int)
        buckets = cf.collect_from_parquet(path, n_cifake, S)
        cif = buckets[0] + buckets[1]
        y_cif = np.array([0] * len(buckets[0]) + [1] * len(buckets[1]))
        print(f"\n--- canonical size {S}x{S} ---")
        for name, fn in SUP_METHODS.items():
            Xl = np.vstack([fn(im) for im in ds.images])
            Xc = np.vstack([fn(im) for im in cif])
            a_ll, a_cc = cv_auroc(Xl, y_loc), cv_auroc(Xc, y_cif)
            a_lc, a_cl = transfer(Xl, y_loc, Xc, y_cif), transfer(Xc, y_cif, Xl, y_loc)
            # unsupervised Mahalanobis transfer (CE features only)
            m_lc = maha_transfer(Xl, y_loc, Xc, y_cif)
            m_cl = maha_transfer(Xc, y_cif, Xl, y_loc)
            print(f"  {name:<12} in:L={a_ll:.3f} in:C={a_cc:.3f} | sup L→C={a_lc:.3f} C→L={a_cl:.3f}"
                  f" | maha L→C={m_lc:.3f} C→L={m_cl:.3f}")
            rows.append({"size": S, "method": name, "in_local": a_ll, "in_cifake": a_cc,
                         "sup_LtoC": a_lc, "sup_CtoL": a_cl, "maha_LtoC": m_lc, "maha_CtoL": m_cl})
    try:
        os.remove(path); print("\nremoved parquet")
    except OSError:
        pass
    import pandas as pd
    pd.DataFrame(rows).to_csv(ev.Path(__file__).resolve().parent / "results" / "transfer2.csv", index=False)
    print("wrote results/transfer2.csv")


if __name__ == "__main__":
    main()
