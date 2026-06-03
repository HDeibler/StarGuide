"""Bounded, single-pass video sampling — the basis of near-constant runtime.

We never process every frame. `sampled_pass` reads at most `budget` frames,
evenly spaced across the clip, and in one decode pass yields each as both a
full-resolution gray frame (for stacking) and a detection-resolution gray frame
(downscaled to `work_width`, for tracking). So the work depends on (budget,
work_width) — not on how long the clip is or how many megapixels the sensor has.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class FrameMeta:
    width: int
    height: int
    fps: float
    n_total: int
    indices: list[int]      # frame indices actually sampled


def sampled_indices(n_total: int, budget: int) -> list[int]:
    if n_total <= budget:
        return list(range(n_total))
    return list(np.linspace(0, n_total - 1, budget).round().astype(int))


def sampled_pass(path: str, budget: int, work_width: int):
    """Yield (order_i, time_s, gray_full, gray_work, scale) for sampled frames.

    `scale` maps work-resolution pixels back to full resolution (multiply work
    centroids by `scale`). Returns the generator; iterate it, then read `.meta`
    off the returned object via `pass_meta` if needed.
    """
    cap = cv2.VideoCapture(path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(sampled_indices(n_total, budget))
    scale = max(1.0, W / float(work_width))
    ww, wh = int(round(W / scale)), int(round(H / scale))
    meta = FrameMeta(W, H, fps, n_total, sorted(want))

    def gen():
        idx = order = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if idx in want:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                gw = (cv2.resize(gray, (ww, wh), interpolation=cv2.INTER_AREA)
                      if scale > 1.01 else gray)
                yield order, idx / fps, gray, gw, scale
                order += 1
            idx += 1
        cap.release()

    return gen(), meta
