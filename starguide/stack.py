"""Temporal stacking to pull faint stars out of the noise.

A single 4K night frame buries real stars in amplifier noise. Averaging the
clip raises every persistent star by ~sqrt(N) in SNR while transient noise,
aircraft and meteor streaks wash out. Over 42s the stars only rotate ~0.18deg
about the pole (a few pixels), so a plain mean already sharpens them; this is
what makes the brightest stars jump to SNR>200 and gives the solver a clean,
complete centroid list.
"""

from __future__ import annotations

import cv2
import numpy as np


def mean_stack(video_path: str, budget: int = 60) -> np.ndarray:
    """Grayscale mean of at most `budget` evenly-spaced frames (float32).

    Bounded sampling keeps the cost independent of clip length.
    """
    from .video import sampled_indices
    cap = cv2.VideoCapture(video_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(sampled_indices(n_total, budget))
    acc = None
    n = idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx in want:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            acc = g if acc is None else acc + g
            n += 1
        idx += 1
    cap.release()
    return acc / max(n, 1)


def prep_snr(stack: np.ndarray):
    """Background-subtract once; reuse across many SNR queries."""
    bg = cv2.GaussianBlur(cv2.medianBlur(stack.astype(np.uint8), 5),
                          (0, 0), 9).astype(np.float32)
    resid = stack - bg
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
    return resid, float(noise)


def snr_query(resid, noise, xs, ys, win: int = 10):
    """Peak SNR in a window around each (x, y) on a prepared residual."""
    out = []
    for x, y in zip(xs, ys):
        x, y = int(x), int(y)
        patch = resid[max(0, y - win):y + win, max(0, x - win):x + win]
        out.append(float(patch.max()) / noise if patch.size else 0.0)
    return np.array(out)


def snr_at(stack: np.ndarray, xs, ys, win: int = 10):
    """Background-subtracted peak SNR in a window around each (x, y)."""
    resid, noise = prep_snr(stack)
    return snr_query(resid, noise, xs, ys, win), noise
