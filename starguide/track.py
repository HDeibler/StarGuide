"""Temporal consolidation of per-frame detections into confirmed stars.

A single frame at 4K night exposure is full of false positives: amplifier
noise, cosmic-ray hits, compression speckle, cloud edges. The video gives us
the one thing a still image can't — *time*. Real stars reappear at nearly the
same pixel in frame after frame (drifting only ~a few px over the whole clip
from Earth's rotation). Almost nothing else does that.

So we link detections across frames into tracks and keep only those seen in a
healthy fraction of frames. This simultaneously:
  - kills transient noise (low persistence),
  - kills moving streaks (a plane never revisits the same pixel), and
  - flags hot pixels (perfectly zero motion + often present in EVERY frame even
    through clouds) for optional removal.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from .detect import Blob


@dataclass
class Track:
    x: float
    y: float
    flux: float
    frames: list[int] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    fluxes: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def drift(self) -> float:
        """Total path length of the centroid across its life (pixels)."""
        if self.n < 2:
            return 0.0
        dx = np.diff(self.xs)
        dy = np.diff(self.ys)
        return float(np.sum(np.hypot(dx, dy)))

    def centroid(self) -> tuple[float, float, float]:
        w = np.array(self.fluxes) + 1e-6
        cx = float(np.sum(np.array(self.xs) * w) / w.sum())
        cy = float(np.sum(np.array(self.ys) * w) / w.sum())
        return cx, cy, float(np.median(self.fluxes))


class Tracker:
    """Greedy nearest-neighbour linker. O(detections) with a coarse grid."""

    def __init__(self, link_radius: float = 6.0):
        self.link_radius = link_radius
        self.tracks: list[Track] = []

    def update(self, frame_idx: int, blobs: list[Blob]) -> None:
        if not self.tracks:
            for b in blobs:
                self._spawn(frame_idx, b)
            return
        tx = np.array([t.x for t in self.tracks])
        ty = np.array([t.y for t in self.tracks])
        used = set()
        for b in blobs:
            d2 = (tx - b.x) ** 2 + (ty - b.y) ** 2
            j = int(np.argmin(d2))
            if d2[j] <= self.link_radius ** 2 and j not in used:
                t = self.tracks[j]
                t.frames.append(frame_idx)
                t.xs.append(b.x); t.ys.append(b.y); t.fluxes.append(b.flux)
                t.x, t.y, t.flux = b.x, b.y, b.flux
                used.add(j)
            else:
                self._spawn(frame_idx, b)

    def _spawn(self, frame_idx: int, b: Blob) -> None:
        self.tracks.append(Track(b.x, b.y, b.flux, [frame_idx],
                                 [b.x], [b.y], [b.flux]))

    def confirmed(self, n_frames: int, min_frac: float = 0.5,
                  max_drift: float = 12.0, drop_static_hotpixels: bool = True):
        """Return consolidated (x, y, flux) for tracks that look like stars."""
        out = []
        for t in self.tracks:
            if t.n < max(2, int(min_frac * n_frames)):
                continue
            if t.drift > max_drift:
                continue  # erratic / streak fragment
            # Hot pixels: present in essentially every frame with ~zero motion.
            if drop_static_hotpixels and t.n > 0.95 * n_frames and t.drift < 0.4:
                continue
            out.append(t.centroid())
        return out
