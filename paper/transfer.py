"""Cross-dataset transfer: does the real-vs-synthetic signal carry to a totally
different dataset WITHOUT retraining? Lightweight statistical descriptors vs a
heavier hand-crafted fingerprint, train-on-A / test-on-B (zero-shot).

Local web images (256px, Midjourney/SDXL/...) <-> CIFAKE (32px, CIFAR-10 vs SD).
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import amethods as am
import baselines as bl
import cifake as cf
import evaluate as ev

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

METHODS = {
    "CE (20)": am.METHODS["CE (base)"],
    "А1 dispersion (11)": am.dispersion_ce,
    "А2 multiscale2d (10)": am.multiscale2d_ce,
    "heavy combo FFT+GLCM+LBP (71)": bl.BASELINES["FFT+GLCM+LBP"],
}


def build(images, fn):
    return np.vstack([fn(im) for im in images])


def cv_auroc(X, y, seed=0):
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(seed)
    acc, cnt = np.zeros(len(y)), np.zeros(len(y))
    for _ in range(3):
        skf = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1 << 30)))
        for tr, te in skf.split(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[tr], y[tr])
            acc[te] += clf.predict_proba(X[te])[:, 1]; cnt[te] += 1
    return roc_auc_score(y, acc / np.maximum(cnt, 1))


def transfer_auroc(Xtr, ytr, Xte, yte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def main(n_cifake=400):
    # local
    ds = ev.load_dataset(size=256)
    y_loc = (ds.group == "real").astype(int)
    print(f"local: {len(ds.images)} imgs ({y_loc.sum()} real)")
    # cifake (stream parquet, delete after)
    path = cf.get_parquet()
    buckets = cf.collect_from_parquet(path, n_cifake, None)
    try:
        os.remove(path); print("removed parquet")
    except OSError:
        pass
    cif_imgs = buckets[0] + buckets[1]
    y_cif = np.array([0] * len(buckets[0]) + [1] * len(buckets[1]))  # 0 fake,1 real
    print(f"cifake: {len(cif_imgs)} imgs ({y_cif.sum()} real)")

    print(f"\n{'method':<32}{'in:local':>9}{'in:cifake':>11}{'L→C':>8}{'C→L':>8}{'retain':>8}")
    rows = []
    for name, fn in METHODS.items():
        Xl = build(ds.images, fn)
        Xc = build(cif_imgs, fn)
        a_ll = cv_auroc(Xl, y_loc)
        a_cc = cv_auroc(Xc, y_cif)
        a_lc = transfer_auroc(Xl, y_loc, Xc, y_cif)
        a_cl = transfer_auroc(Xc, y_cif, Xl, y_loc)
        retain = 0.5 * ((a_lc - 0.5) / (a_cc - 0.5 + 1e-9) + (a_cl - 0.5) / (a_ll - 0.5 + 1e-9))
        print(f"{name:<32}{a_ll:>9.3f}{a_cc:>11.3f}{a_lc:>8.3f}{a_cl:>8.3f}{retain:>8.2f}")
        rows.append({"method": name, "in_local": a_ll, "in_cifake": a_cc,
                     "transfer_LtoC": a_lc, "transfer_CtoL": a_cl, "retain": retain})
    import pandas as pd
    pd.DataFrame(rows).to_csv(ev.Path(__file__).resolve().parent / "results" / "transfer.csv", index=False)
    print("\nretain = mean fraction of (AUROC-0.5) margin kept under zero-shot transfer")
    print("wrote results/transfer.csv")


if __name__ == "__main__":
    main()
