"""Complexity-entropy feature extraction for diffusion-error detection.

Clean, deterministic re-implementation of the thesis pipeline. Unlike the
notebook version, this module:
  * never applies a random untrained convolution,
  * exposes the image -> 1D linearization scheme as an explicit parameter,
  * supports native 2D ordinal patterns (no linearization at all),
  * returns full Renyi complexity-entropy *curves* as feature vectors.

The output of every ordpy CE call is ordered (entropy, complexity).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import ordpy
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize

# ----------------------------------------------------------------------------
# Image loading
# ----------------------------------------------------------------------------

def load_gray(path: str | Path, size: int = 256) -> np.ndarray:
    """Load an image as a float grayscale array in [0, 1], resized to size*size."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = rgb2gray(arr)
    gray = resize(gray, (size, size), anti_aliasing=True, mode="reflect")
    return gray.astype(np.float64)


# ----------------------------------------------------------------------------
# 2D -> 1D linearization schemes
# ----------------------------------------------------------------------------

def _hilbert_d2xy_order(n_bits: int) -> np.ndarray:
    """Return flat indices that traverse a 2^n_bits square along a Hilbert curve."""
    size = 1 << n_bits
    order = np.empty(size * size, dtype=np.int64)
    for d in range(size * size):
        rx = ry = 0
        t = d
        x = y = 0
        s = 1
        while s < size:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            # rotate
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        order[d] = y * size + x
    return order


_HILBERT_CACHE: dict[int, np.ndarray] = {}


def to_signal(img: np.ndarray, scheme: str = "row") -> np.ndarray:
    """Linearize a 2D image into a 1D signal.

    scheme: 'row'   row-major (C order)
            'col'   column-major (F order)
            'snake' boustrophedon (alternate row direction)
            'hilbert' Hilbert space-filling curve (img must be square, 2^k)
            'diag'  anti-diagonal scan
    """
    if scheme == "row":
        return img.ravel(order="C")
    if scheme == "col":
        return img.ravel(order="F")
    if scheme == "snake":
        rows = img.copy()
        rows[1::2] = rows[1::2, ::-1]
        return rows.ravel(order="C")
    if scheme == "diag":
        h, w = img.shape
        return np.concatenate([np.diagonal(img[:, ::-1], k) for k in range(-h + 1, w)])
    if scheme == "hilbert":
        h, w = img.shape
        assert h == w and (h & (h - 1)) == 0, "Hilbert needs square 2^k image"
        n_bits = int(np.log2(h))
        if n_bits not in _HILBERT_CACHE:
            _HILBERT_CACHE[n_bits] = _hilbert_d2xy_order(n_bits)
        return img.ravel(order="C")[_HILBERT_CACHE[n_bits]]
    raise ValueError(f"unknown scheme {scheme!r}")


# ----------------------------------------------------------------------------
# Complexity-entropy feature vectors
# ----------------------------------------------------------------------------

DEFAULT_ALPHAS = np.linspace(0.0, 1.0, 10)


def ce_curve_1d(signal: np.ndarray, dx: int = 5, taux: int = 1,
                alphas: np.ndarray = DEFAULT_ALPHAS) -> np.ndarray:
    """Renyi CE curve of a 1D signal -> flat vector [H_a0..H_an, C_a0..C_an]."""
    out = np.asarray(ordpy.renyi_complexity_entropy(
        signal, alpha=alphas, dx=dx, taux=taux))  # (n_alpha, 2): (H, C)
    return np.concatenate([out[:, 0], out[:, 1]])


def ce_curve_2d(img: np.ndarray, dx: int = 2, dy: int = 2,
                alphas: np.ndarray = DEFAULT_ALPHAS) -> np.ndarray:
    """Renyi CE curve using native 2D ordinal patterns -> flat vector."""
    out = np.asarray(ordpy.renyi_complexity_entropy(
        img, alpha=alphas, dx=dx, dy=dy))
    return np.concatenate([out[:, 0], out[:, 1]])


def image_features(img: np.ndarray, *, scheme: str = "row", dx: int = 5,
                   dy: int = 1, taux: int = 1,
                   alphas: np.ndarray = DEFAULT_ALPHAS) -> np.ndarray:
    """CE feature vector for one grayscale image under a given parameterization.

    scheme == '2d'  -> native 2D ordinal patterns with (dx, dy).
    otherwise       -> linearize with `scheme`, then 1D ordinal patterns (dx, taux).
    """
    if scheme == "2d":
        return ce_curve_2d(img, dx=dx, dy=dy, alphas=alphas)
    sig = to_signal(img, scheme)
    return ce_curve_1d(sig, dx=dx, taux=taux, alphas=alphas)


def feature_names(n_alpha: int) -> list[str]:
    return [f"H_a{i}" for i in range(n_alpha)] + [f"C_a{i}" for i in range(n_alpha)]
