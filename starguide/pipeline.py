"""End-to-end: a night video in, an identified sky out — no manual calibration.

    solve_video(path, site) ->
        one bounded decode pass:  track stars (work-res)  +  stack (full-res)
        motion.confirm            pole pixel + confirmed stars (kinematics)
        detect on the stack       complete, high-SNR centroid list
        solve                     pole-anchored plate solve + distortion fit
    -> SkyModel: the camera model + everything needed to overlay any frame.

The expensive work happens once, over a *bounded* number of frames, so runtime
is a function of SolveConfig (track_budget, stack_budget, work_width) — not of clip length or
sensor resolution. Afterwards, overlaying a frame is a single sidereal rotation
of the catalog plus a polynomial eval (see project.rotate_uv) — thousands of fps.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from .config import SiteConfig, SolveConfig, BlindConfig, SKYSTREAM_SITE
from .detect import detect, build_mask
from . import motion, stack
from .astro import load_catalog, parse_timestamp, horizon_altaz
from .solve import solve, Solution


@dataclass
class SkyModel:
    solution: Solution
    pole: np.ndarray
    when_utc: object
    site: SiteConfig
    stack_img: np.ndarray
    tangentiality: float
    mean_drift_px: float
    n_confirmed: int

    @property
    def model(self):
        return self.solution.model


def solve_video(path: str, site: SiteConfig = SKYSTREAM_SITE,
                cfg: SolveConfig | None = None, verbose: bool = True) -> SkyModel:
    cfg = cfg or SolveConfig()
    when = parse_timestamp(path, site.utc_offset_h)

    # ---- pass 1 (bounded, costly): track stars -> pole + confirmed stars ----
    mr = motion.analyze(path, cfg)
    if verbose:
        print(f"[1/3] tracked {cfg.track_budget} frames; "
              f"pole=({mr.pole[0]:.0f},{mr.pole[1]:.0f}) "
              f"confirmed={len(mr.stars)} tangentiality={mr.tangentiality:.3f} "
              f"drift={mr.mean_drift_px:.2f}px")

    # ---- pass 2 (bounded, cheap decode-only): high-SNR stack ----------------
    stk = stack.mean_stack(path, cfg.stack_budget)
    H, W = stk.shape
    smask = build_mask((H, W))
    smask[: int(H * 0.10), int(W * 0.62):] = 0
    blobs, _ = detect(cv2.cvtColor(stk.astype(np.uint8), cv2.COLOR_GRAY2BGR),
                      smask, sigma=cfg.detect_sigma, min_peak=5, max_area=120,
                      max_keep=600)
    det = [(b.x, b.y, b.flux) for b in blobs]
    if verbose:
        print(f"[2/3] detected {len(det)} stars on the stack; plate solving …")

    cat = load_catalog(max_mag=cfg.max_mag)
    alt, az = horizon_altaz(cat, when, site, min_alt=0)
    sol = solve(det, cat, alt, az, (H, W), mr.pole, site.lat, cfg, stack=stk)
    if sol is None:
        raise RuntimeError("plate solve found no consensus")
    if verbose:
        named = [s.name for s, *_ in sol.matches
                 if not s.name.startswith(("HIP", "HR "))]
        print(f"[3/3] solved: RMS={sol.rms:.2f}px, {sol.n_inliers} fit stars; "
              f"e.g. {', '.join(named[:6])}")
    return SkyModel(sol, mr.pole, when, site, stk, mr.tangentiality,
                    mr.mean_drift_px, len(mr.stars))


@dataclass
class SkyImage:
    """Result of a one-off blind solve. `.model.project(ra, dec)` maps catalog
    directions to pixels; `.matches` are the identified stars; `.planets` are
    named solar-system bodies (only when planets are enabled AND a time given)."""
    solution: object
    image: np.ndarray
    source: str
    planets: list = None        # (name, px, py, mag, confirmed)
    when_utc: object = None

    @property
    def model(self):
        return self.solution.model

    @property
    def matches(self):
        return self.solution.matches


def solve_image(source, cfg: BlindConfig | None = None, index=None,
                when=None, site=None, verbose: bool = True) -> SkyImage:
    """Blind-identify a single still image (path or BGR array). No time,
    location or pole needed — recovered from the star pattern alone.

    A *video* path is accepted too: its frames are mean-stacked into one
    high-SNR image first (see `solve_video_blind`), which is the right path for a
    clip with no clean sidereal drift (too short, or handheld) that the motion
    solver can't use.

    `when` (a UTC datetime) is the one extra input that lets us name *planets*:
    their positions move with time, so without it a bright planet can only be
    flagged as a non-catalog object. `site` sharpens the Moon (parallax)."""
    from .config import BlindConfig
    from .blind import solve_blind, get_index
    from .detect import detect_auto
    from .loader import imread, is_video
    cfg = cfg or BlindConfig()
    if isinstance(source, str) and is_video(source):
        return solve_video_blind(source, cfg, when=when, site=site,
                                 verbose=verbose)
    img = imread(source) if isinstance(source, str) else source
    H, W = img.shape[:2]
    if index is None:
        index = get_index(cfg.index_max_mag, cfg.index_k_near)
    # Detection normalizes any frame to a canonical width so stars come out a few
    # px wide — the same settings work from an 800px phone clip to a 6K DSLR.
    # round_only rejects non-star shapes (text, planes, trails) by elongation, no
    # assumption about where they sit. A low-resolution wide field (e.g. an 886px
    # all-sky screen-grab) packs its stars near 1px, so if the default width finds
    # no confident match we re-detect finer; the bright-star guard keeps every
    # accepted solve honest, so trying a finer scale only ever recovers a true one.
    sol, scale, det = None, 1.0, []
    for tw in (None, 2200, 2800):
        kw = {} if tw is None else {"target_width": tw}
        blobs, _, scale = detect_auto(img, sigma=cfg.detect_sigma, min_area=2,
                                      min_peak=6, max_keep=400, round_only=True,
                                      **kw)
        det = [(b.x, b.y, b.flux) for b in blobs]
        if verbose:
            print(f"blind: {len(det)} stars detected (frame {W}x{H}, "
                  f"detect scale {scale:.2f}); matching quad hashes …")
        sol = solve_blind(det, (H, W), index, cfg)
        if sol is not None:
            break
    if sol is None:
        raise RuntimeError("no confident star match — not a recognizable sky, "
                           "or too few stars")
    if verbose:
        named = [s.name for s, *_ in sol.matches
                 if not s.name.startswith(("HIP", "HR "))]
        print(f"blind: identified {sol.n_inliers} stars (RMS {sol.rms:.2f}px); "
              f"e.g. {', '.join(sorted(set(named))[:6])}")

    planets = None
    if cfg.planets:
        if when is None:
            raise ValueError("planets=True needs a date — pass when=<UTC datetime>")
        # Planets bloom into large saturated blobs the round-star filter drops,
        # so confirm them against a pass that keeps bright blobs of any shape.
        pblobs, _, _ = detect_auto(img, sigma=cfg.detect_sigma, min_area=2,
                                   min_peak=10, max_area=4000, max_keep=150)
        pdet = [(b.x, b.y, b.flux) for b in pblobs]
        planets = _planets(sol.model, pdet, (H, W), when, site, sol.matches)
        if verbose:
            inframe = ", ".join(f"{p[0]}{'' if p[4] else '?'}" for p in planets)
            print(f"blind: planets in frame: {inframe or '(none up)'}")
    return SkyImage(sol, img, source if isinstance(source, str) else "<array>",
                    planets=planets, when_utc=when)


def solve_video_blind(path: str, cfg: BlindConfig | None = None,
                      budget: int = 60, when=None, site=None,
                      verbose: bool = True) -> SkyImage:
    """Identify a video by mean-stacking its frames into one high-SNR image and
    blind-solving that. This is the right path for a clip the *motion* solver
    can't use — too short for measurable sidereal drift, or handheld so the
    apparent motion is camera shake, not star rotation. Needs no site or
    timestamp; the stack is just a cleaner still the blind solver identifies."""
    from .config import BlindConfig
    cfg = cfg or BlindConfig()
    stk = stack.mean_stack(path, budget)
    bgr = cv2.cvtColor(np.clip(stk, 0, 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
    if verbose:
        print(f"blind-video: mean-stacked {budget} frames -> "
              f"{bgr.shape[1]}x{bgr.shape[0]} high-SNR image; identifying …")
    sky = solve_image(bgr, cfg, when=when, site=site, verbose=verbose)
    return SkyImage(sky.solution, sky.image, path, planets=sky.planets,
                    when_utc=sky.when_utc)


def _planets(model, det, shape, when, site, matches):
    """Identify planets conservatively. A body is shown ONLY if (a) it is above
    the horizon — when a site is known, which alone rejects the Moon/Mercury that
    fold into a gnomonic frame though they aren't in the sky — and (b) a genuine
    bright object sits at its predicted spot, to which the marker snaps. Anything
    we can't actually see in the frame is dropped, not guessed."""
    from scipy.spatial import cKDTree
    from .astro import planets_radec, planets_altaz
    H, W = shape
    blobs = np.array(det, float)                      # (x, y, flux)
    if not len(blobs):
        return []
    tree = cKDTree(blobs[:, :2])
    tol_local = 0.05 * max(H, W)                      # model must be constrained here
    tol_snap = 0.035 * max(H, W)                      # planet's blob within this

    ra0, dec0 = model.center1, model.center2
    seps = np.array([_ang_sep(s.ra, s.dec, ra0, dec0) for s, *_ in matches])
    field_rho = float(np.percentile(seps, 95)) + 5.0 if len(seps) else 90.0
    mp = np.array([(x, y) for _, x, y in matches], float)
    mtree = cKDTree(mp) if len(mp) else None

    up = None
    if site is not None:
        up = {n: al for n, al, az, mg in planets_altaz(when, site, min_alt=-90)}

    out = []
    for name, ra, dec, mag in planets_radec(when, site):
        if up is not None and up.get(name, -90) < 3.0:
            continue                                  # below the horizon
        if _ang_sep(ra, dec, ra0, dec0) > field_rho:
            continue                                  # outside the imaged field
        px, py, _ = model.project(np.array([ra]), np.array([dec]))
        x, y = float(px[0]), float(py[0])
        if np.isnan(x) or not (0 < x < W and 0 < y < H):
            continue
        # A bright planet (mag<0) outshines every star, so the brightest blob
        # near its prediction must be it — trust that even where the plate model
        # is loose. A faint planet is only trusted where the model is well
        # constrained (a matched star nearby).
        if mag < 0.0:
            tol = 0.12 * max(H, W)
        elif mtree is None or mtree.query([x, y])[0] <= tol_local:
            tol = tol_snap
        else:
            continue          # sparse region — a faint planet can't be trusted
        near = tree.query_ball_point([x, y], tol)
        if not near:
            continue                                  # no object where predicted
        b = near[int(np.argmax(blobs[near, 2]))]      # brightest blob in window
        out.append((name, float(blobs[b, 0]), float(blobs[b, 1]), mag, True))
    return out


def _ang_sep(ra1, dec1, ra2, dec2):
    a = np.radians([ra1, dec1, ra2, dec2])
    return np.degrees(np.arccos(np.clip(
        np.sin(a[1]) * np.sin(a[3]) +
        np.cos(a[1]) * np.cos(a[3]) * np.cos(a[0] - a[2]), -1, 1)))


def identify(source, site: SiteConfig | None = None, mode: str = "auto",
             cfg=None, when=None, save: str | None = None, style="auto",
             verbose: bool = True):
    """Unified entry — routes a live-stream/clip to the motion solver and a
    one-off image to the blind solver.

    mode: 'auto' (decide from the source), 'video' (fixed-camera motion solve),
    'video-blind' (stack a clip and blind-solve it), or 'image' (one-off photo).
    The motion solver ('video') recovers a full alt/az camera model from sidereal
    drift and needs a fixed camera with a `site` + timestamp; when that isn't
    available (handheld, too short, unknown site) it gives way to the blind
    frame-stack — reported, never silent. `when` (UTC datetime) enables planet
    identification when `BlindConfig(planets=True)`. If `save` is given, an
    annotated overlay is written using `style` ('auto'/'pro', 'classic', or an
    OverlayStyle).
    """
    from .loader import is_video
    from .config import SolveConfig, BlindConfig
    if mode == "auto":
        mode = "video" if (isinstance(source, str) and is_video(source)) else "image"
    if mode == "video-blind":
        bcfg = cfg if isinstance(cfg, BlindConfig) else None
        sky = solve_video_blind(source, bcfg, when=when, site=site, verbose=verbose)
    elif mode == "video":
        try:
            if site is None:
                raise RuntimeError("no camera SiteConfig given")
            scfg = cfg if isinstance(cfg, SolveConfig) else None
            sky = solve_video(source, site, scfg, verbose=verbose)
        except (RuntimeError, ValueError) as e:
            if verbose:
                print(f"[motion] cannot solve this clip ({e}); "
                      f"identifying via blind frame-stack instead")
            bcfg = cfg if isinstance(cfg, BlindConfig) else None
            sky = solve_video_blind(source, bcfg, when=when, site=site,
                                    verbose=verbose)
    else:
        sky = solve_image(source, cfg, when=when, site=site, verbose=verbose)
    if save:
        _save_overlay(sky, save, style)
    return sky


def _save_overlay(sky, path, style):
    """Render and write an annotated overlay for either result type."""
    import os
    from .overlay import annotate_image, render
    from .astro import load_catalog, horizon_altaz, load_constellations
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if isinstance(sky, SkyImage):
        annotate_image(sky, style=style, out_path=path)
    else:
        stars = load_catalog(max_mag=5.0)
        alt, az = horizon_altaz(stars, sky.when_utc, sky.site, min_alt=0)
        bg = cv2.cvtColor(sky.stack_img.astype("uint8"), cv2.COLOR_GRAY2BGR)
        vis = render(bg, sky.model, stars, alt, az,
                     constellations=load_constellations(),
                     snr_ref=sky.stack_img, style=style)
        cv2.imwrite(path, vis)


def when_from_planets(sky: SkyImage, years=(2015, 2026), step_days: int = 2,
                      max_planet_mag: float = 1.2):
    """Estimate WHEN a solved photo was taken from a bright planet's position.

    The inverse of planet identification: the brightest non-stellar dot is
    matched against where each bright planet would have been across a date range,
    and the best fit dates the photo. Precision is limited by how fast the planet
    moves — near opposition it's near-stationary, so the answer is good to weeks,
    not days. Returns (body, datetime, residual_px) or None."""
    from datetime import datetime, timezone, timedelta
    from scipy.spatial import cKDTree
    from .detect import detect_auto
    from .astro import planets_radec
    H, W = sky.image.shape[:2]
    blobs, _, _ = detect_auto(sky.image, min_peak=6, round_only=False)
    det = np.array([(b.x, b.y, b.flux) for b in blobs])
    det = det[det[:, 1] < 0.78 * H]                  # drop foreground/ground lights
    mp = np.array([(x, y) for _, x, y in sky.matches])
    stree = cKDTree(mp)
    obj = None
    for i in np.argsort(-det[:, 2]):
        if stree.query(det[i, :2])[0] > 10:          # bright + not a catalog star
            obj = det[i, :2]; break
    if obj is None:
        return None
    best = None
    d = datetime(years[0], 1, 1, 3, tzinfo=timezone.utc)
    end = datetime(years[1], 1, 1, tzinfo=timezone.utc)
    while d < end:
        for name, ra, dec, mag in planets_radec(d):
            if mag > max_planet_mag:
                continue
            px, py, _ = sky.model.project(np.array([ra]), np.array([dec]))
            if np.isnan(px[0]):
                continue
            r = float(np.hypot(px[0] - obj[0], py[0] - obj[1]))
            if best is None or r < best[2]:
                best = (name, d, r)
        d += timedelta(days=step_days)
    return best


def identification_yield(sky: SkyModel, max_mag: float = 4.5, snr_min: float = 4.0):
    """Honest accuracy: fraction of catalog stars whose predicted pixel lands on
    a real star in the stack (SNR > snr_min). Independent of the detector."""
    cat = load_catalog(max_mag=max_mag)
    alt, az = horizon_altaz(cat, sky.when_utc, sky.site, min_alt=0)
    px, py, _ = sky.model.project(alt, az)
    H, W = sky.stack_img.shape
    on = (px > 3) & (px < W - 3) & (py > 3) & (py < H - 3) & (~np.isnan(alt))
    snr, _ = stack.snr_at(sky.stack_img, px[on], py[on])
    return int((snr > snr_min).sum()), int(on.sum())
