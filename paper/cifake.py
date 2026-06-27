"""Validate the real-vs-synthetic CE result on CIFAKE (streamed, memory-safe).

Streams a bounded balanced subset from HuggingFace (no full download / disk
cache), converts to small grayscale arrays, builds CE features and reports
AUROC/PR-AUC with bootstrap CIs using the same harness as the local data.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
from skimage.color import rgb2gray

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")  # flaky-network safe

import ce_features as cef
import evaluate as ev

DS_ID = "dragonintelligence/CIFAKE-image-dataset"


def get_parquet(split_file="data/test-00000-of-00001.parquet"):
    """Download one parquet shard (resume-safe) and return its local path."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DS_ID, split_file, repo_type="dataset",
                           resume_download=True)
    print(f"downloaded {split_file} -> {path}")
    return path


def to_gray(img, size: int | None) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    g = rgb2gray(arr)
    if size is not None and g.shape != (size, size):
        from skimage.transform import resize
        g = resize(g, (size, size), anti_aliasing=True, mode="reflect")
    return g.astype(np.float64)


def collect_from_parquet(path, n_per_class, size):
    """Read row groups incrementally; keep n_per_class small gray arrays each."""
    import pyarrow.parquet as pq
    from PIL import Image
    pf = pq.ParquetFile(path)
    buckets = {0: [], 1: []}
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["image", "label"]).to_pydict()
        for img_obj, y in zip(tbl["image"], tbl["label"]):
            y = int(y)
            if len(buckets[y]) >= n_per_class:
                continue
            b = img_obj["bytes"] if isinstance(img_obj, dict) else img_obj
            img = Image.open(io.BytesIO(b))
            buckets[y].append(to_gray(img, size))
        if all(len(v) >= n_per_class for v in buckets.values()):
            break
    return buckets


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 600
    cleanup = "--keep" not in sys.argv
    size = None  # keep native 32x32 (CIFAKE); avoid upsampling artifacts

    path = get_parquet()
    print(f"reading {n} per class (native 32x32)...")
    buckets = collect_from_parquet(path, n, size)
    if cleanup:
        try:
            os.remove(path)
            print("removed parquet to free disk")
        except OSError as e:
            print("could not remove parquet:", e)
    print("collected:", {k: len(v) for k, v in buckets.items()})

    # CIFAKE: label 0 = FAKE (synthetic), 1 = REAL
    images = buckets[0] + buckets[1]
    # task = real(1) vs synthetic(0); positive class = real to mirror local eval
    y = np.array([0] * len(buckets[0]) + [1] * len(buckets[1]))

    for fkw, tag in [(dict(scheme="row", dx=5, taux=1), "row d5"),
                     (dict(scheme="row", dx=4, taux=1), "row d4"),
                     (dict(scheme="2d", dx=2, dy=2), "2d 2x2")]:
        X = np.vstack([cef.image_features(im, **fkw) for im in images])
        r = ev.evaluate_task(X, y, f"CIFAKE real_vs_synth [{tag}]")
        print(ev._fmt(r))
