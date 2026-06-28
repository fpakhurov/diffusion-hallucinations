"""Source-confound control (review item #1) + dataset manifest.

Builds a datasheet manifest (class, format, native resolution, file size,
sha256) for the collected corpus, and tests whether trivial source metadata
(format, resolution, file size) alone predicts the H1/H2 labels. If metadata
predicts the labels well, the CE signal is partly confounded by source.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import evaluate as ev

DATA = Path(__file__).resolve().parent.parent / "bach_code" / "data"
FOLDERS = {"real": "real_imgs", "good": "good_imgs", "bad": "bad_imgs"}


def manifest():
    rows = []
    for cls, folder in FOLDERS.items():
        for fp in sorted((DATA / folder).glob("*.*")):
            try:
                with Image.open(fp) as im:
                    w, h = im.size
                    fmt = im.format
                    mode = im.mode
                b = fp.read_bytes()
                rows.append({
                    "file": fp.name, "class": cls, "format": fmt, "mode": mode,
                    "width": w, "height": h, "megapixels": round(w * h / 1e6, 3),
                    "aspect": round(w / h, 3), "bytes": len(b),
                    "sha256": hashlib.sha256(b).hexdigest()[:16],
                })
            except Exception as e:  # noqa: BLE001
                print(f"skip {fp.name}: {e}")
    return pd.DataFrame(rows)


def meta_features(df):
    fmts = ["JPEG", "PNG", "WEBP"]
    X = np.column_stack([
        df.width.values, df.height.values, df.megapixels.values,
        df.aspect.values, np.log1p(df.bytes.values),
        *[(df.format == f).astype(float).values for f in fmts],
    ])
    return X


def main():
    df = manifest()
    out = Path(__file__).resolve().parent / "results" / "dataset_manifest.csv"
    df.to_csv(out, index=False)
    print(f"manifest: {len(df)} files -> {out}")
    print("\n=== format x class ===")
    print(pd.crosstab(df["class"], df["format"]))
    print("\n=== median native resolution (megapixels) by class ===")
    print(df.groupby("class").megapixels.median())
    print(f"\n=== median file size (KB) by class ===")
    print((df.groupby("class").bytes.median() / 1024).round(1))

    X = meta_features(df)
    g = df["class"].values
    print("\n=== source-metadata-only classifier (AUROC) ===")
    # H1: real vs synthetic
    y1 = (g == "real").astype(int)
    r1 = ev.evaluate_task(X, y1, "H1_meta")
    print(ev._fmt(r1))
    # H2: bad vs good
    m = np.isin(g, ["good", "bad"])
    y2 = (g[m] == "bad").astype(int)
    r2 = ev.evaluate_task(X[m], y2, "H2_meta")
    print(ev._fmt(r2))


if __name__ == "__main__":
    main()
