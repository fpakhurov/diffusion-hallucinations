"""Parameter-sensitivity sweep (paper core): how do CE parameters
(linearization scheme, embedding dim d, delay tau, Renyi alpha grid)
affect discrimination performance on both tasks?

Loads the dataset once, then evaluates every parameterization and writes a
tidy CSV of AUROC/AP with CIs.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as ev

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def sweep(size: int = 256, seed: int = 0):
    ds = ev.load_dataset(size=size)
    print(f"loaded {len(ds.images)} images ({(ds.group=='real').sum()} real / "
          f"{(ds.group=='good').sum()} good / {(ds.group=='bad').sum()} bad)")

    schemes_1d = ["row", "col", "snake", "hilbert", "diag"]
    d_values = [3, 4, 5, 6]
    tau_values = [1, 2]
    rows = []

    # --- 1D linearization sweep: scheme x d x tau --------------------------
    for scheme, d, tau in itertools.product(schemes_1d, d_values, tau_values):
        fkw = dict(scheme=scheme, dx=d, taux=tau)
        t0 = time.time()
        try:
            res, _ = ev.run(fkw, ds=ds, seed=seed)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {fkw}: {e}")
            continue
        for r in res:
            rows.append({**r, "family": "1d", "scheme": scheme,
                         "d": d, "tau": tau, "dy": None})
        print(f"  1d scheme={scheme} d={d} tau={tau}  "
              f"({time.time()-t0:.1f}s)  "
              + " | ".join(f"{r['task']}={r['auroc']:.3f}" for r in res))

    # --- native 2D ordinal sweep: (dx, dy) --------------------------------
    for dx, dy in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        fkw = dict(scheme="2d", dx=dx, dy=dy)
        t0 = time.time()
        try:
            res, _ = ev.run(fkw, ds=ds, seed=seed)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {fkw}: {e}")
            continue
        for r in res:
            rows.append({**r, "family": "2d", "scheme": "2d",
                         "d": dx, "tau": 1, "dy": dy})
        print(f"  2d dx={dx} dy={dy}  ({time.time()-t0:.1f}s)  "
              + " | ".join(f"{r['task']}={r['auroc']:.3f}" for r in res))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sweep.csv", index=False)
    print(f"\nwrote {OUT/'sweep.csv'}  ({len(df)} rows)")
    return df


def summarize(df: pd.DataFrame):
    for task in df["task"].unique():
        sub = df[df["task"] == task].sort_values("auroc", ascending=False)
        print(f"\n===== {task}: top 5 by AUROC =====")
        cols = ["family", "scheme", "d", "tau", "dy", "auroc", "auroc_lo", "auroc_hi"]
        print(sub[cols].head(5).to_string(index=False))
        print(f"  worst AUROC: {sub['auroc'].min():.3f}  best: {sub['auroc'].max():.3f}")


if __name__ == "__main__":
    df = sweep()
    summarize(df)
