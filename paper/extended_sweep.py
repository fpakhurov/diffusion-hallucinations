"""Extended study of embedding dimension d and delay tau over wide ranges.

d: ordpy computes any d quickly (it only stores observed patterns), but the
statistics collapse once N/d! -> 0 (every window becomes a unique pattern,
normalized entropy saturates to 1, complexity -> 0). We push d up to 200 to
show the resulting collapse of discriminative power. tau is pushed by orders
of magnitude (1..1024).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ce_features as cef
import evaluate as ev

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

D_LIST = [3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 40, 50, 60]
# Note: d>=171 overflows ordpy's float(d!) ; and the descriptor is already
# statistically degenerate (N/d!->0) well before that, so we cap at 60.
TAU_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def mean_shannon_H(ds, d, tau):
    """Mean normalized Shannon entropy (alpha=1 point) across images — saturation probe."""
    hs = []
    for im in ds.images:
        sig = cef.to_signal(im, "row")
        out = np.asarray(__import__("ordpy").renyi_complexity_entropy(
            sig, alpha=[1.0], dx=d, taux=tau))
        hs.append(out[0, 0])
    return float(np.mean(hs))


def features_for(ds, d, tau, alphas):
    return np.vstack([cef.ce_curve_1d(cef.to_signal(im, "row"), dx=d, taux=tau, alphas=alphas)
                      for im in ds.images])


def run_axis(ds, axis: str):
    tasks = ev.tasks_from_groups(ds.group)
    rows = []
    alphas = cef.DEFAULT_ALPHAS
    N = ds.images[0].size
    vals = D_LIST if axis == "d" else TAU_LIST
    for v in vals:
        d, tau = (v, 1) if axis == "d" else (5, v)
        t0 = time.time()
        X = features_for(ds, d, tau, alphas)
        Hbar = mean_shannon_H(ds, d, tau)
        rec = {"axis": axis, "value": v, "d": d, "tau": tau,
               "N_over_dfact": N / math.factorial(min(d, 170)) if d <= 170 else 0.0,
               "mean_H": Hbar}
        for name, mask, y in tasks:
            r = ev.evaluate_task(X[mask], y, name)
            rec[f"{name}_auroc"] = r["auroc"]
            rec[f"{name}_lo"] = r["auroc_lo"]
            rec[f"{name}_hi"] = r["auroc_hi"]
        rows.append(rec)
        print(f"  {axis}={v:<5} d={d} tau={tau}  H1={rec['real_vs_synthetic_auroc']:.3f} "
              f"H2={rec['bad_vs_good_auroc']:.3f}  meanH={Hbar:.3f}  ({time.time()-t0:.1f}s)")
    return rows


def main():
    ds = ev.load_dataset(size=256)
    print(f"loaded {len(ds.images)} images, N={ds.images[0].size}")
    rows = []
    print("=== d-axis (tau=1) ===")
    rows += run_axis(ds, "d")
    print("=== tau-axis (d=5) ===")
    rows += run_axis(ds, "tau")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "extended_sweep.csv", index=False)
    print(f"wrote {OUT/'extended_sweep.csv'}")


if __name__ == "__main__":
    main()
