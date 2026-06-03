"""Motion-based star confirmation and pole recovery — pure kinematics.

The most discriminating fact in the video: real stars rigidly rotate about the
projected celestial pole at the sidereal rate, so each star's drift is a small
vector *perpendicular* to its line from the pole. This does two jobs with no
catalog at all:

  1. recover the pole pixel (linear least squares: drift . (pos - pole) = 0),
  2. certify genuine stars (tangential drift), rejecting fixed-pattern noise
     (no motion) and aircraft/satellites (wrong motion).

The functions here operate on tracks/arrays, so they are trivially unit-testable
and reusable; `analyze` is a convenience that reads a video itself.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .track import Tracker, Track
from .config import SolveConfig


@dataclass
class MotionResult:
    pole: np.ndarray            # projected celestial pole pixel (full-res)
    stars: list                 # confirmed (x, y, flux), full-res
    tangentiality: float        # median |cos(drift, radius)|; 0 = ideal
    mean_drift_px: float        # mean sidereal drift over the clip
    n_processed: int


def _endpoints(t: Track):
    xs, ys = np.array(t.xs), np.array(t.ys)
    k = max(2, t.n // 5)
    p0 = np.array([xs[:k].mean(), ys[:k].mean()])
    p1 = np.array([xs[-k:].mean(), ys[-k:].mean()])
    return (p0 + p1) / 2, p1 - p0


def fit_pole(mids: np.ndarray, drifts: np.ndarray) -> np.ndarray:
    """Least-squares pole where every drift is tangential: drift.(mid-P)=0."""
    P, *_ = np.linalg.lstsq(drifts, (drifts * mids).sum(1), rcond=None)
    return P


def confirm(tracker: Tracker, n_proc: int, scale: float = 1.0,
            min_frac: float = 0.6, drift_lo: float = 0.6,
            drift_hi: float = 12.0, tang_max: float = 0.35) -> MotionResult:
    """Fit the pole and confirm stars from a populated Tracker.

    `scale` maps the tracker's (possibly downscaled) pixels back to full-res.
    Drift thresholds are in tracker pixels (pre-scale).
    """
    mids, drifts, info = [], [], []
    for t in tracker.tracks:
        if t.n < min_frac * n_proc:
            continue
        mid, d = _endpoints(t)
        nd = float(np.hypot(*d))
        if drift_lo < nd < drift_hi and t.drift < drift_hi:
            mids.append(mid); drifts.append(d)
            info.append((t.centroid(), nd))
    mids = np.array(mids); drifts = np.array(drifts)
    if len(mids) < 8:
        raise RuntimeError("too few moving stars to fit a pole")

    pole = fit_pole(mids, drifts)
    rad = mids - pole
    cosang = np.abs((drifts * rad).sum(1)) / (
        np.hypot(*drifts.T) * np.hypot(*rad.T) + 1e-9)
    keep = cosang < tang_max
    stars = [(info[i][0][0] * scale, info[i][0][1] * scale, info[i][0][2])
             for i in range(len(info)) if keep[i]]
    return MotionResult(
        pole=pole * scale, stars=stars,
        tangentiality=float(np.median(cosang)),
        mean_drift_px=float(np.hypot(*drifts.T).mean()) * scale,
        n_processed=n_proc)


def analyze(path: str, cfg: SolveConfig | None = None) -> MotionResult:
    """Convenience: read the video (bounded sampling) and confirm stars."""
    from .detect import detect, build_mask
    from .video import sampled_pass

    cfg = cfg or SolveConfig()
    gen, meta = sampled_pass(path, cfg.track_budget, cfg.work_width)
    trk = Tracker(link_radius=6.0)
    n = 0
    mask = None
    for order, _t, _full, gw, scale in gen:
        if mask is None:
            mask = build_mask(gw.shape)
            mask[: int(gw.shape[0] * 0.10), int(gw.shape[1] * 0.62):] = 0
        s, _ = detect(gw, mask, sigma=cfg.detect_sigma, min_area=2,
                      min_peak=4.0, max_area=120)
        trk.update(order, s)
        n += 1
    return confirm(trk, n, scale=scale, min_frac=cfg.min_persist_frac)
