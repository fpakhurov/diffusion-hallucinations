"""Evaluation harness: does a CE parameterization carry signal for
real-vs-synthetic and good-vs-bad image discrimination?

Reports cross-validated AUROC / PR-AUC with bootstrap confidence intervals,
replacing the thesis's ad-hoc `score` metric.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ce_features as cef

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent / "bach_code" / "data"
FOLDERS = {"real": "real_imgs", "good": "good_imgs", "bad": "bad_imgs"}


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

@dataclass
class Dataset:
    images: list[np.ndarray]
    group: np.ndarray  # 'real' | 'good' | 'bad'
    paths: list[str]


def load_dataset(size: int = 256) -> Dataset:
    images, group, paths = [], [], []
    for g, folder in FOLDERS.items():
        for fp in sorted((DATA / folder).glob("*.*")):
            try:
                images.append(cef.load_gray(fp, size=size))
                group.append(g)
                paths.append(str(fp))
            except Exception as e:  # noqa: BLE001
                print(f"skip {fp.name}: {e}")
    return Dataset(images, np.array(group), paths)


def build_features(ds: Dataset, **fkw) -> np.ndarray:
    return np.vstack([cef.image_features(im, **fkw) for im in ds.images])


# ----------------------------------------------------------------------------
# Cross-validated evaluation with bootstrap CIs
# ----------------------------------------------------------------------------

def _cv_scores(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
               n_repeats: int = 5, seed: int = 0):
    """Return ONE out-of-fold score per true sample, averaged over repeats.

    Averaging each sample's OOF probability across the `n_repeats` repeats
    yields exactly `n` predictions (one per real sample). Bootstrapping then
    resamples true samples — avoiding the pseudo-replication that pooling all
    repeats would cause (which artificially narrows the CIs).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=int(rng.integers(1 << 30)))
        for tr, te in skf.split(X, y):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(X[tr], y[tr])
            acc[te] += clf.predict_proba(X[te])[:, 1]
            cnt[te] += 1
    return y.copy(), acc / np.maximum(cnt, 1)


def _bootstrap_ci(y_true, y_score, metric, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric(y_true, y_score)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        # guarantee both classes present
        if len(np.unique(y_true[idx])) < 2:
            stats[b] = np.nan
            continue
        stats[b] = metric(y_true[idx], y_score[idx])
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return point, lo, hi


def evaluate_task(X: np.ndarray, y: np.ndarray, name: str, seed: int = 0):
    yt, ys = _cv_scores(X, y, seed=seed)
    auroc, a_lo, a_hi = _bootstrap_ci(yt, ys, roc_auc_score, seed=seed)
    ap, p_lo, p_hi = _bootstrap_ci(yt, ys, average_precision_score, seed=seed)
    base = y.mean()  # positive rate
    return {
        "task": name, "n": len(y), "pos_rate": base,
        "auroc": auroc, "auroc_lo": a_lo, "auroc_hi": a_hi,
        "ap": ap, "ap_lo": p_lo, "ap_hi": p_hi,
    }


def tasks_from_groups(group: np.ndarray):
    """Return list of (name, mask, y) for the two discrimination tasks."""
    out = []
    # Task A: real (1) vs synthetic (good+bad) (0)
    mask_a = np.ones(len(group), dtype=bool)
    y_a = (group == "real").astype(int)
    out.append(("real_vs_synthetic", mask_a, y_a))
    # Task B: bad (1) vs good (0), synthetic only
    mask_b = np.isin(group, ["good", "bad"])
    y_b = (group[mask_b] == "bad").astype(int)
    out.append(("bad_vs_good", mask_b, y_b))
    return out


def run(fkw: dict, size: int = 256, seed: int = 0, ds: Dataset | None = None):
    if ds is None:
        ds = load_dataset(size=size)
    X = build_features(ds, **fkw)
    results = []
    for name, mask, y in tasks_from_groups(ds.group):
        results.append(evaluate_task(X[mask], y, name, seed=seed))
    return results, ds


def _fmt(r):
    return (f"{r['task']:<18} n={r['n']:>3}  "
            f"AUROC={r['auroc']:.3f} [{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}]  "
            f"AP={r['ap']:.3f} [{r['ap_lo']:.3f},{r['ap_hi']:.3f}]  (base={r['pos_rate']:.2f})")


if __name__ == "__main__":
    import sys
    scheme = sys.argv[1] if len(sys.argv) > 1 else "row"
    if scheme == "2d":
        fkw = dict(scheme="2d", dx=2, dy=2)
    else:
        fkw = dict(scheme=scheme, dx=5, taux=1)
    print(f"=== parameterization: {fkw} ===")
    res, ds = run(fkw)
    for r in res:
        print(_fmt(r))
