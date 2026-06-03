"""Point-source (star) detection for a single frame.

The whole pipeline rests on one cheap, robust primitive: given a frame, return
the sub-pixel centroids of star-like point sources, with each blob's flux and
shape so we can tell stars apart from streaks (planes/satellites) and noise.

Design notes
------------
- Background is estimated with a large median blur and subtracted. Wide-angle
  night frames have a smooth gradient (light pollution, moonglow) that a single
  global threshold can't handle; local background removal fixes that.
- A star is a *compact* bright blob a few pixels across. We threshold on a
  robust sigma above the local background, label connected components, and keep
  only ones whose size and roundness match a real point source.
- Aircraft / satellites are long, thin, fast streaks — high elongation. We flag
  those separately instead of feeding them to the solver.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class Blob:
    x: float          # sub-pixel centroid (image columns)
    y: float          # sub-pixel centroid (image rows)
    flux: float       # background-subtracted summed intensity
    peak: float       # peak background-subtracted intensity
    area: int         # pixel count
    elong: float      # major/minor axis ratio (1.0 = round, >>1 = streak)


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def estimate_background(gray: np.ndarray, ksize: int = 51) -> np.ndarray:
    """Smooth background via median blur (robust to bright point sources)."""
    k = ksize | 1  # force odd
    # medianBlur caps kernel at 5 for non-8U on some builds; downscale trick is
    # faster and smoother for large kernels on 4K frames.
    small = cv2.resize(gray, (gray.shape[1] // 8, gray.shape[0] // 8),
                       interpolation=cv2.INTER_AREA)
    bg_small = cv2.medianBlur(small, k if k <= 99 else 99)
    bg_small = cv2.GaussianBlur(bg_small, (0, 0), 6)
    return cv2.resize(bg_small, (gray.shape[1], gray.shape[0]),
                      interpolation=cv2.INTER_LINEAR)


def detect(frame: np.ndarray, mask: np.ndarray | None = None,
           sigma: float = 6.0, min_area: int = 3, max_area: int = 400,
           max_elong: float = 2.5, min_peak: float = 6.0,
           max_keep: int = 500, round_only: bool = False) -> tuple[list[Blob], list[Blob]]:
    """Return (stars, streaks) found in a frame.

    `mask` is a uint8 array, 0 where pixels should be ignored (timestamp text,
    tree line, etc.). `sigma` sets the detection threshold above local noise.

    Speed: component *filtering* is fully vectorized (bincount over a flood-fill
    labelling) so the cost is independent of the (huge) raw blob count; we only
    build Python Blob objects for the few hundred that survive the area/peak cuts.
    """
    gray = to_gray(frame).astype(np.float32)
    bg = estimate_background(gray.astype(np.uint8)).astype(np.float32)
    resid = gray - bg
    if mask is not None:
        resid[mask == 0] = 0

    # Matched filter: convolving with a PSF-sized Gaussian is the optimal linear
    # detector for point sources — it lifts real stars above the per-pixel noise
    # while averaging isolated speckle down.
    det = cv2.GaussianBlur(resid, (0, 0), 1.0)

    med = float(np.median(det))
    mad = float(np.median(np.abs(det - med))) + 1e-6
    noise = 1.4826 * mad
    binary = (det > med + sigma * noise).astype(np.uint8)
    # Opening removes 1-2px noise survivors; real stars are >=3px and persist.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _KERNEL3)
    if binary.sum() == 0:
        return [], []

    n, labels = cv2.connectedComponents(binary, connectivity=8)
    if n <= 1:
        return [], []

    # Restrict every reduction to *foreground* pixels (a few ten-thousand) so
    # cost scales with detections, not with the 8-megapixel frame.
    flat = labels.ravel()
    fg = np.flatnonzero(flat)
    lab_fg = flat[fg]
    pos = np.clip(resid, 0, None).ravel()[fg]
    raw = resid.ravel()[fg]
    yy, xx = _grids(resid.shape)
    x_fg, y_fg = xx.ravel()[fg], yy.ravel()[fg]

    area = np.bincount(lab_fg, minlength=n)
    fsum = np.bincount(lab_fg, weights=pos, minlength=n)
    sx = np.bincount(lab_fg, weights=pos * x_fg, minlength=n)
    sy = np.bincount(lab_fg, weights=pos * y_fg, minlength=n)
    peak = np.zeros(n)
    np.maximum.at(peak, lab_fg, raw)

    w = fsum + 1e-9
    cx, cy = sx / w, sy / w
    idx = np.arange(1, n)
    a = area[1:]
    pk = peak[1:]
    is_star = (a >= min_area) & (a <= max_area) & (pk >= min_peak)
    is_streak = (a > max_area) & (pk >= min_peak)

    elong = np.ones(n)
    if round_only:
        # Second moments (vectorized) give each blob's elongation. A star is a
        # round PSF (~1); text strokes, planes and trails are elongated. This is
        # a shape test — it makes no assumption about WHERE clutter sits.
        sxx = np.bincount(lab_fg, weights=pos * x_fg * x_fg, minlength=n)
        syy = np.bincount(lab_fg, weights=pos * y_fg * y_fg, minlength=n)
        sxy = np.bincount(lab_fg, weights=pos * x_fg * y_fg, minlength=n)
        cxx = sxx / w - cx * cx
        cyy = syy / w - cy * cy
        cxy = sxy / w - cx * cy
        tr = cxx + cyy
        disc = np.sqrt(np.maximum(tr * tr / 4 - (cxx * cyy - cxy * cxy), 0))
        l1 = tr / 2 + disc
        l2 = np.maximum(tr / 2 - disc, 1e-6)
        elong = np.sqrt(l1 / l2)
        is_star &= elong[1:] <= max_elong

    def pack(sel):
        return [Blob(float(cx[i]), float(cy[i]), float(fsum[i]),
                     float(peak[i]), int(area[i]), float(elong[i]))
                for i in idx[sel]]

    stars = pack(is_star)
    streaks = pack(is_streak)
    stars.sort(key=lambda b: -b.flux)
    return stars[:max_keep], streaks


_KERNEL3 = np.ones((3, 3), np.uint8)
_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _grids(shape):
    if shape not in _GRID_CACHE:
        h, w = shape
        _GRID_CACHE[shape] = np.mgrid[0:h, 0:w].astype(np.float32)
    return _GRID_CACHE[shape]


def _slice_elong(sl, labels: np.ndarray, pos: np.ndarray, lab: int) -> float:
    """Elongation of one component, evaluated on its bounding-box slice."""
    if sl is None:
        return 999.0
    sub_lab = labels[sl]
    sub_w = np.where(sub_lab == lab, pos[sl], 0.0)
    ys, xs = np.nonzero(sub_w)
    if xs.size < 2:
        return 1.0
    return _elongation(xs.astype(np.float32), ys.astype(np.float32),
                       sub_w[ys, xs])


def _elongation(xs: np.ndarray, ys: np.ndarray, w: np.ndarray) -> float:
    wsum = w.sum() + 1e-9
    mx, my = (xs * w).sum() / wsum, (ys * w).sum() / wsum
    dx, dy = xs - mx, ys - my
    cxx = (w * dx * dx).sum() / wsum
    cyy = (w * dy * dy).sum() / wsum
    cxy = (w * dx * dy).sum() / wsum
    tr = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = max(tr * tr / 4 - det, 0.0)
    l1 = tr / 2 + np.sqrt(disc)
    l2 = tr / 2 - np.sqrt(disc)
    if l2 <= 1e-6:
        return 999.0
    return float(np.sqrt(l1 / l2))


def detect_auto(frame: np.ndarray, target_width: int = 1600,
                mask: np.ndarray | None = None, **kw):
    """Resolution-adaptive detection: normalize the frame to a canonical width
    so stars are always ~2-4 px (the regime the fixed detector parameters suit),
    detect there, and scale the centroids back to the original pixels.

    This makes detection behave the same whether the camera is 800 px or 6000 px
    wide — the caller never tunes anything per image. Returns (stars, streaks,
    scale) with centroids in ORIGINAL-image coordinates.
    """
    H, W = frame.shape[:2]
    scale = W / float(target_width)
    if 0.8 < scale < 1.25:
        small, scale = frame, 1.0
    else:
        small = cv2.resize(frame, (target_width, int(round(H / scale))),
                           interpolation=(cv2.INTER_AREA if scale > 1
                                          else cv2.INTER_LINEAR))
    smask = None
    if mask is not None and scale != 1.0:
        smask = cv2.resize(mask, (small.shape[1], small.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    elif mask is not None:
        smask = mask
    stars, streaks = detect(small, smask, **kw)
    for b in (*stars, *streaks):
        b.x *= scale
        b.y *= scale
    return stars, streaks, scale


def build_mask(shape: tuple[int, int], tree_frac: float = 0.92) -> np.ndarray:
    """Default mask: drop the timestamp text (top-right) and tree line (bottom)."""
    h, w = shape
    mask = np.full((h, w), 255, np.uint8)
    mask[int(h * tree_frac):, :] = 0                 # tree line
    mask[: int(h * 0.06), int(w * 0.66):] = 0        # timestamp overlay
    return mask
