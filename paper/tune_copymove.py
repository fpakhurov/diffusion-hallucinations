"""Threshold calibration for the ordinal copy-move detector.

The detector score is the duplicated-area fraction. We split BASE images into
train/val (no base leaks), pick the decision threshold on train (max F1 and
Youden's J), and report operating-point metrics on the held-out val split:
precision, recall, F1, balanced accuracy, MCC — plus threshold-free AUROC.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)

import copymove as cm
import evaluate as ev

GAMMAS = [0.6, 1.0, 1.6, 2.2]


def build_scores(bases, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for bi, base in enumerate(bases):
        # positive: translation copy-move with random size + tone edit
        r = np.random.default_rng(1000 + bi)
        frac = float(r.uniform(0.05, 0.12))
        gamma = float(GAMMAS[r.integers(len(GAMMAS))])
        x, _ = cm.make_copymove(base, frac, r, gamma=gamma)
        _, sp = cm.ordinal_localize(x)
        rows.append((bi, 1, sp))
        # negative: clean
        _, sn = cm.ordinal_localize(base)
        rows.append((bi, 0, sn))
    return np.array(rows, dtype=float)   # (base, label, score)


def metrics_at(y, s, t):
    pred = (s >= t).astype(int)
    return dict(prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                bacc=balanced_accuracy_score(y, pred),
                mcc=matthews_corrcoef(y, pred) if len(set(pred)) > 1 else 0.0)


def main():
    ds = ev.load_dataset(size=256)
    bases = [im for im, g in zip(ds.images, ds.group) if g == "real"]
    data = build_scores(bases)
    bidx = data[:, 0].astype(int)
    y = data[:, 1].astype(int)
    s = data[:, 2]

    # split by BASE image (no leakage)
    rng = np.random.default_rng(0)
    uniq = np.unique(bidx)
    rng.shuffle(uniq)
    n_tr = int(0.6 * len(uniq))
    tr_bases, va_bases = set(uniq[:n_tr]), set(uniq[n_tr:])
    tr = np.array([b in tr_bases for b in bidx])
    va = ~tr
    print(f"{len(uniq)} bases -> train {len(tr_bases)} / val {len(va_bases)}; "
          f"samples train {tr.sum()} / val {va.sum()}")

    print(f"\nval AUROC (threshold-free): {roc_auc_score(y[va], s[va]):.3f}")

    # tune threshold on TRAIN
    ts = np.unique(s[tr])
    grid = np.concatenate([[0.0], (ts[:-1] + ts[1:]) / 2, [ts.max() + 1e-6]]) if len(ts) > 1 else np.array([0.0, s.max()+1e-6])
    f1s = [metrics_at(y[tr], s[tr], t)["f1"] for t in grid]
    t_f1 = grid[int(np.argmax(f1s))]
    # Youden's J on train
    fpr, tpr, thr = roc_curve(y[tr], s[tr])
    t_j = thr[int(np.argmax(tpr - fpr))]

    for name, t in [("max-F1 (train)", t_f1), ("Youden-J (train)", t_j)]:
        m = metrics_at(y[va], s[va], t)
        print(f"\nthreshold {name} = {t:.4f}")
        print(f"  val: prec={m['prec']:.3f} rec={m['rec']:.3f} F1={m['f1']:.3f} "
              f"bal.acc={m['bacc']:.3f} MCC={m['mcc']:.3f}")

    import pandas as pd
    pd.DataFrame({"base": bidx, "label": y, "score": s, "split": np.where(tr, "train", "val")}
                 ).to_csv(ev.Path(__file__).resolve().parent / "results" / "tune_copymove.csv", index=False)
    print("\nwrote results/tune_copymove.csv")


if __name__ == "__main__":
    main()
