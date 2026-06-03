"""Overlay live star identification onto a video stream — the video stays intact.

`overlay_stream` reads a clip or an RTSP feed, solves the geometry from a short
burst of frames (the blind frame-stack path — no site or timestamp needed), and
then draws the constellation lines, star rings and names onto *every* frame and
writes them out. The frame itself is never altered; the labels ride on top.

A live sky drifts, so the solve refreshes every `resolve_every_s` seconds: the
overlay is recomputed in the background-cheap blind solve and the labels follow
the stars. Output goes to an `.mp4` and/or a per-frame callback (e.g. to pipe to
an ffmpeg RTSP restreamer), so the same code serves "annotate this clip" and
"annotate this live feed".
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from .config import BlindConfig
from .overlay import STAR
from .stack import snr_at


# A warm gold for constellation strokes/labels that reads on any night frame.
LINE = (150, 200, 120)
LABEL = (235, 235, 240)


@dataclass
class _Overlay:
    """Precomputed, pixel-space overlay for the current solve — drawn fast per
    frame so the per-frame cost is a few cv2 calls, not a re-solve."""
    segments: list          # [((x1,y1),(x2,y2)), ...] constellation strokes
    stars: list             # [(x, y, radius), ...] star rings
    labels: list            # [(text, x, y), ...] named-star labels
    n_inliers: int
    rms: float


def _build_overlay(model, matches, full, cra, cdec, names, hips, snr_ref,
                   shape, max_mag, snr_min=4.0):
    """Project the catalog through the solved model and keep only stars that are
    in frame and actually present (SNR), then assemble strokes/rings/labels."""
    from .astro import load_constellations
    H, W = shape
    px, py, _ = model.project(cra, cdec)
    on = (px > 1) & (px < W - 1) & (py > 1) & (py < H - 1) & ~np.isnan(px)
    snr = np.zeros(len(cra))
    idx = np.flatnonzero(on)
    if len(idx):
        snr[idx], _ = snr_at(snr_ref, px[idx], py[idx])
    vis = on & (snr >= snr_min)
    pos = {int(hips[i]): (int(px[i]), int(py[i]))
           for i in np.flatnonzero(vis)}

    cons = load_constellations()
    diag = (W * W + H * H) ** 0.5
    segments = []
    members = set()
    for pairs in cons.values():
        for h1, h2 in pairs:
            if h1 in pos and h2 in pos:
                (x1, y1), (x2, y2) = pos[h1], pos[h2]
                if (x2 - x1) ** 2 + (y2 - y1) ** 2 <= (0.45 * diag) ** 2:
                    segments.append((pos[h1], pos[h2]))
                    members.add(h1); members.add(h2)

    stars, labels = [], []
    for i in np.flatnonzero(vis):
        x, y = int(px[i]), int(py[i])
        mag = float(full[i])
        if int(hips[i]) not in members:
            stars.append((x, y, max(2, int(round(5 - mag)))))
        if mag <= 2.6 and not names[i].startswith(("HIP", "HR ")):
            labels.append((names[i], x, y))
    return _Overlay(segments, stars, labels, len(matches), 0.0)


def _draw(frame, ov: _Overlay, sc: float):
    """Draw the precomputed overlay onto an (untouched) frame copy at scale `sc`."""
    vis = frame.copy()
    for (x1, y1), (x2, y2) in ov.segments:
        cv2.line(vis, (int(x1 * sc), int(y1 * sc)), (int(x2 * sc), int(y2 * sc)),
                 LINE, 1, cv2.LINE_AA)
    for x, y, r in ov.stars:
        cv2.circle(vis, (int(x * sc), int(y * sc)), r, STAR, 1, cv2.LINE_AA)
    for text, x, y in ov.labels:
        cv2.putText(vis, text, (int(x * sc) + 6, int(y * sc) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, LABEL, 1, cv2.LINE_AA)
    return vis


def _solve_stack(frames, cfg, index):
    """Blind-solve a mean stack of recent frames; return (model, matches) or None."""
    from .blind import solve_blind, get_index
    from .detect import detect_auto
    if index is None:
        index = get_index(cfg.index_max_mag, cfg.index_k_near)
    stk = np.mean(frames, axis=0)
    bgr = cv2.cvtColor(np.clip(stk, 0, 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
    H, W = stk.shape
    for tw in (None, 2200, 2800):
        kw = {} if tw is None else {"target_width": tw}
        blobs, _, _ = detect_auto(bgr, sigma=cfg.detect_sigma, min_area=2,
                                  min_peak=6, max_keep=400, round_only=True, **kw)
        det = [(b.x, b.y, b.flux) for b in blobs]
        sol = solve_blind(det, (H, W), index, cfg)
        if sol is not None:
            return sol, stk
    return None, stk


def overlay_stream(source, out_path: str | None = None,
                   cfg: BlindConfig | None = None, resolve_every_s: float = 20.0,
                   max_seconds: float | None = None, max_mag: float = 4.5,
                   out_width: int = 1280, on_frame=None, fourcc: str = "mp4v",
                   verbose: bool = True) -> int:
    """Annotate a video/RTSP `source` with live star identification.

    Writes annotated frames to `out_path` (an .mp4) when given, and/or passes each
    annotated frame to `on_frame(bgr)` — use that to push frames to an ffmpeg
    process restreaming to RTSP. The geometry is re-solved every `resolve_every_s`
    seconds so the labels track the moving sky. Returns the number of frames
    written. Raises RuntimeError if the opening burst never yields a solve.
    """
    from .blind import get_index
    cfg = cfg or BlindConfig()
    index = get_index(cfg.index_max_mag, cfg.index_k_near)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open stream: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OW = min(out_width, W); OH = int(round(OW * H / W)); sc = OW / W

    full = cra = cdec = names = hips = None
    writer = None
    if out_path:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc),
                                 fps, (OW, OH))

    burst = max(8, int(round(fps * 1.5)))         # frames stacked per solve
    ring, ov, written, idx = [], None, 0, 0
    next_solve = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_seconds is not None and idx / fps > max_seconds:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ring.append(gray)
        if len(ring) > burst:
            ring.pop(0)

        # (re)solve on schedule once we have a full burst
        if len(ring) >= burst and idx >= next_solve:
            if full is None:
                cat = _catalog(max_mag)
                full, cra, cdec, names, hips = cat
            sol, stk = _solve_stack(ring, cfg, index)
            if sol is not None:
                ov = _build_overlay(sol.model, sol.matches, full, cra, cdec,
                                    names, hips, stk, (H, W), max_mag)
                ov.rms = sol.rms
                if verbose:
                    print(f"  t={idx/fps:5.1f}s  solved {ov.n_inliers} stars "
                          f"(RMS {ov.rms:.1f}px), {len(ov.segments)} strokes")
            elif verbose:
                print(f"  t={idx/fps:5.1f}s  no solve this window")
            next_solve = idx + int(round(fps * resolve_every_s))

        out_frame = cv2.resize(frame, (OW, OH))
        if ov is not None:
            out_frame = _draw(out_frame, ov, sc)
        if writer is not None:
            writer.write(out_frame)
        if on_frame is not None:
            on_frame(out_frame)
        written += 1
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if ov is None:
        raise RuntimeError("could not identify the sky in this stream "
                           "(too few stars, or not a recognizable field)")
    return written


def _catalog(max_mag):
    from .astro import load_catalog
    cat = load_catalog(max_mag=max(6.0, max_mag + 1.0))
    cra = np.array([s.ra for s in cat])
    cdec = np.array([s.dec for s in cat])
    names = [s.name for s in cat]
    hips = np.array([s.hip for s in cat])
    full = np.array([s.mag for s in cat])
    return full, cra, cdec, names, hips
