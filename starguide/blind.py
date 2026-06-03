"""Blind plate solving for one-off photos — no time, location, or pole.

A still image from an unknown camera pointed who-knows-where can't use the
motion/pole machinery the live camera relies on. Instead we identify it from the
star pattern alone, the astrometry.net way:

  1. Index the catalog as geometric *quad hashes* — for groups of four nearby
     stars, a 4-number code invariant to translation, rotation and scale.
  2. Hash the image's star quads the same way and look them up: a code match
     proposes which four dots are which four catalog stars.
  3. RANSAC a gnomonic (TAN) pose from a match, verify by projecting the whole
     catalog and counting how many land on a detected star.
  4. Refine the distortion with the shared 2-D-polynomial `refine_poly`.

Everything downstream (PolyModel, refine_poly, overlay, catalog) is the same
infrastructure the video path uses — only the *plane* differs (gnomonic about a
tangent point here, azimuthal about the pole there). A non-sky image simply
fails to reach the inlier threshold and returns None.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations
from dataclasses import dataclass
from scipy.spatial import cKDTree

from .astro import load_catalog
from .project import (gnomonic, azimuthal_equidistant, radec_to_vec,
                      CameraModel, RaDecCamera, fit_model)
from .solve import Solution, refine_poly


@dataclass
class QuadIndex:
    tree: cKDTree           # KDTree over 4-D quad codes
    quads: np.ndarray       # (N, 4) catalog-star indices per quad, order [A,B,C,D]
    ra: np.ndarray          # catalog RA (deg), indexed by the quad columns
    dec: np.ndarray
    mag: np.ndarray
    names: list
    hip: np.ndarray


def _quad_code(xy):
    """4-number translation/rotation/scale-invariant code for 4 points, with a
    consistent [A,B,C,D] ordering so matches give correspondences directly."""
    z = xy[:, 0] + 1j * xy[:, 1]
    A = B = 0
    dmax = -1.0
    for i in range(4):
        for j in range(i + 1, 4):
            d = abs(z[i] - z[j])
            if d > dmax:
                dmax, A, B = d, i, j
    others = [k for k in range(4) if k not in (A, B)]

    def code(a, b):
        w = (z - z[a]) / (z[b] - z[a])
        c, d = w[others[0]], w[others[1]]
        if c.real > d.real:
            c, d = d, c
        return c, d

    c, d = code(A, B)
    order = [A, B, others[0], others[1]]
    if (c.real + d.real) > 1.0:                 # A/B swap symmetry: z -> 1-z
        c, d = code(B, A)
        order = [B, A, others[0], others[1]]
    # keep only "compact" quads (C,D inside the AB circle) — distinctive codes
    if abs(c - 0.5) > 0.58 or abs(d - 0.5) > 0.58:
        return None, None
    if c.real > d.real:
        order[2], order[3] = order[3], order[2]
        c, d = d, c
    return np.array([c.real, c.imag, d.real, d.imag]), order


def build_index(max_mag: float = 5.5, k_near: int = 6) -> QuadIndex:
    """Precompute catalog quad hashes from bright stars + nearby neighbours."""
    stars = load_catalog(max_mag=max_mag)
    ra = np.array([s.ra for s in stars])
    dec = np.array([s.dec for s in stars])
    vec = radec_to_vec(ra, dec)
    tree = cKDTree(vec)
    codes, quads = [], []
    for i in range(len(stars)):
        _, nbr = tree.query(vec[i], k=k_near + 1)
        nbr = [j for j in nbr if j != i][:k_near]
        # local tangent plane at star i so the quad is planar
        for trio in combinations(nbr, 3):
            idx = [i, *trio]
            uv = gnomonic(ra[idx], dec[idx], ra[i], dec[i])
            if np.isnan(uv).any():
                continue
            code, order = _quad_code(uv)
            if code is None:
                continue
            codes.append(code)
            quads.append([idx[o] for o in order])
    return QuadIndex(cKDTree(np.array(codes)), np.array(quads), ra, dec,
                     np.array([s.mag for s in stars]),
                     [s.name for s in stars],
                     np.array([s.hip for s in stars]))


def _image_quads(xy, k_near: int = 6):
    tree = cKDTree(xy)
    codes, quads = [], []
    for i in range(len(xy)):
        _, nbr = tree.query(xy[i], k=min(k_near + 1, len(xy)))
        nbr = [j for j in np.atleast_1d(nbr) if j != i][:k_near]
        for trio in combinations(nbr, 3):
            idx = [i, *trio]
            code, order = _quad_code(xy[idx])
            if code is None:
                continue
            codes.append(code)
            quads.append([idx[o] for o in order])
    return np.array(codes), np.array(quads)


def _tangent(ra, dec):
    v = radec_to_vec(ra, dec).mean(0)
    v /= np.linalg.norm(v) + 1e-12
    return np.degrees(np.arctan2(v[1], v[0])), np.degrees(np.arcsin(v[2]))


def _eqd(ra, dec, ra0, dec0):
    """True RA/Dec equidistant projection about (ra0, dec0) — the wide-field
    seed plane. Matches plane_project(..., 'equidistant') exactly."""
    return azimuthal_equidistant(dec, ra, dec0, ra0)


def _affine_fit(uv, px, py):
    M = np.column_stack([uv[:, 0], uv[:, 1], np.ones(len(uv))])
    a, *_ = np.linalg.lstsq(M, px, rcond=None)
    b, *_ = np.linalg.lstsq(M, py, rcond=None)
    return a, b


_INDEX_CACHE: dict = {}


def get_index(max_mag=5.5, k_near=7) -> QuadIndex:
    """Memoized catalog quad index (building it is the one upfront cost)."""
    key = (max_mag, k_near)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = build_index(max_mag, k_near)
    return _INDEX_CACHE[key]


def solve_blind(detected, img_shape, index: QuadIndex | None = None,
                cfg=None, inlier_px: float = 6.0):
    """Identify an image from its star pattern alone. Returns a Solution (with a
    gnomonic PolyModel) or None if no confident match is found."""
    from .config import BlindConfig
    cfg = cfg or BlindConfig()
    n_stars, code_tol = cfg.n_stars, cfg.code_tol
    min_inliers, max_candidates, n_refine = (cfg.min_inliers, cfg.max_candidates,
                                             cfg.n_refine)
    if index is None:
        index = get_index(cfg.index_max_mag, cfg.index_k_near)
    H, W = img_shape
    det = np.array(sorted(detected, key=lambda d: -d[2])[:n_stars], float)[:, :2]
    if len(det) < 8:
        return None

    icodes, iquads = _image_quads(det)
    if len(icodes) == 0:
        return None
    dist, j = index.tree.query(icodes, k=1)
    cand = [o for o in np.argsort(dist) if dist[o] < code_tol][:max_candidates]

    det70_tree = cKDTree(det)
    cands = []
    for o in cand:
        iq, cq = iquads[o], index.quads[j[o]]
        ra4, dec4 = index.ra[cq], index.dec[cq]
        ra0, dec0 = _tangent(ra4, dec4)                 # quad centre = initial axis
        uv4 = _eqd(ra4, dec4, ra0, dec0)
        a, b = _affine_fit(uv4, det[iq, 0], det[iq, 1])
        # Grow the local quad pose into a whole-field one. We project in the
        # *equidistant* plane (well-behaved to the horizon, where gnomonic
        # diverges past ~60deg and so could never lock a fisheye) and, each
        # round, recentre the axis on the matched stars: once the axis sits near
        # the lens axis the projection is a near-similarity the affine fits
        # cleanly, and the later polynomial only has to absorb radial distortion.
        for tol in (40.0, 18.0, 12.0):
            uv = _eqd(index.ra, index.dec, ra0, dec0)
            ok = np.hypot(uv[:, 0], uv[:, 1]) < 100.0   # one hemisphere; no wrap
            qx = a[0] * uv[:, 0] + a[1] * uv[:, 1] + a[2]
            qy = b[0] * uv[:, 0] + b[1] * uv[:, 1] + b[2]
            inb = ok & (qx > 0) & (qx < W) & (qy > 0) & (qy < H)
            if inb.sum() < 6:
                break
            dq, di = det70_tree.query(np.stack([qx[inb], qy[inb]], 1), k=1)
            sel = dq < tol
            if sel.sum() < 6:
                break
            ii = np.flatnonzero(inb)[sel]
            ra0, dec0 = _tangent(index.ra[ii], index.dec[ii])   # recentre on field
            uvc = _eqd(index.ra[ii], index.dec[ii], ra0, dec0)
            a, b = _affine_fit(uvc, det[di[sel], 0], det[di[sel], 1])
        uv = _eqd(index.ra, index.dec, ra0, dec0)
        ok = np.hypot(uv[:, 0], uv[:, 1]) < 100.0
        qx = a[0] * uv[:, 0] + a[1] * uv[:, 1] + a[2]
        qy = b[0] * uv[:, 0] + b[1] * uv[:, 1] + b[2]
        inb = ok & (qx > 0) & (qx < W) & (qy > 0) & (qy < H)
        if inb.sum() < 8:
            continue
        dq, _ = det70_tree.query(np.stack([qx[inb], qy[inb]], 1), k=1)
        nin = int((dq < inlier_px).sum())
        if nin >= 6:
            cands.append((nin, ra0, dec0, a, b))

    if not cands:
        return None

    # Each surviving candidate seeds TWO models and we keep whichever the
    # bright-star guard trusts with the most stars:
    #   • a 2-D polynomial on the equidistant plane — locks ordinary (narrow,
    #     rectilinear) photos tightly, and absorbs mild wide-angle curvature;
    #   • the physical CameraModel (`_wide_solve`) — a radial a1·ρ+a3·ρ³+a5·ρ⁵
    #     fisheye map a degree-3 polynomial cannot represent past ~70° (its ρ⁵
    #     edge term reaches thousands of px), so it reaches >100° all-sky frames
    #     the polynomial folds on.
    # Only the visible hemisphere is handed to the polynomial fit, so a
    # back-of-sky star can never wrap into the frame.
    full = load_catalog(max_mag=6.5)
    cra = np.array([s.ra for s in full]); cdec = np.array([s.dec for s in full])
    bmag = np.array([s.mag for s in full])
    bright = bmag < 3.2
    bra, bdec = cra[bright], cdec[bright]
    det_full = np.array([(x, y) for x, y, _ in detected], float)
    detF_tree = cKDTree(det_full)
    best = None
    for nin, ra0, dec0, a, b in sorted(cands, key=lambda c: -c[0])[:n_refine]:
        uv = _eqd(cra, cdec, ra0, dec0)
        loc = np.hypot(uv[:, 0], uv[:, 1]) < 95.0       # local hemisphere only
        idx_loc = np.flatnonzero(loc)
        px0 = a[0] * uv[idx_loc, 0] + a[1] * uv[idx_loc, 1] + a[2]
        py0 = b[0] * uv[idx_loc, 0] + b[1] * uv[idx_loc, 1] + b[2]
        poly = refine_poly(px0, py0, det, cra[idx_loc], cdec[idx_loc],
                           [full[k] for k in idx_loc], (H, W), ra0, dec0,
                           "equidistant", (ra0, dec0),
                           schedule=[(1, 40), (1, 25), (2, 16), (2, 11), (3, 8),
                                     (3, 6), (3, 4.5)])
        wide = _wide_solve(a, b, ra0, dec0, index, det, det70_tree, det_full,
                           detF_tree, full, cra, cdec, (H, W))
        for s in (poly, wide):
            if s is None or s.n_inliers < min_inliers:
                continue
            if not _trustworthy(s.model, detF_tree, bra, bdec, (H, W), s.rms):
                continue          # reject overfits whose bright stars don't align
            if best is None or s.n_inliers > best.n_inliers:
                best = s
    return best


def _valid_cam(m: CameraModel) -> bool:
    """A physically plausible pose: a sane pixels-per-degree scale and an axis on
    the sphere. Rejects the degenerate a1->0 collapse a free least-squares fit
    drifts toward when its correspondences are inconsistent."""
    return 2.0 < abs(m.a1) < 300.0 and abs(m.alt0) <= 92.0


def _grow_camera(m, detF, detF_tree, cra, cdec, W, H):
    """Grow a CameraModel from a local seed to the whole fisheye. Each pass widens
    the radius it trusts and tightens the match tolerance, refitting the physical
    model — which (unlike an affine) extrapolates the radial distortion correctly,
    so a correct seed reaches the frame edge. Two guards stop the collapse: the
    matched stars must stay spread across the frame, and the refit must stay
    physical. Distortion (a3,a5) is only freed once wide-radius stars constrain it."""
    diag = (W * W + H * H) ** 0.5
    for rcap, tol, fit_dist in [(30, 32, False), (50, 22, False), (70, 15, False),
                                (88, 10, True), (88, 6, True)]:
        px, py, rho = m.project(cdec, cra)             # CameraModel: project(alt,az)
        on = (px > 1) & (px < W - 1) & (py > 1) & (py < H - 1) & (rho < rcap)
        if on.sum() < 10:
            return None
        ci = np.flatnonzero(on)
        dq, di = detF_tree.query(np.stack([px[on], py[on]], 1), k=1)
        sel = dq < tol
        if sel.sum() < 10:
            return None
        ci, dj = ci[sel], di[sel]
        if np.hypot(detF[dj, 0].ptp(), detF[dj, 1].ptp()) < 0.30 * diag:
            return None                                # collapsed onto a cluster
        try:
            m, _ = fit_model(list(zip(cdec[ci], cra[ci])),
                             list(zip(detF[dj, 0], detF[dj, 1])), m, fit_dist)
        except Exception:
            return None
        if not _valid_cam(m):
            return None
    return m


def _wide_solve(a, b, ra0, dec0, index, det, det70_tree, det_full, detF_tree,
                full, cra, cdec, shape):
    """Wide-field / fisheye identification: seed a physical CameraModel from the
    candidate's affine inliers, grow it to the frame edge, and return a Solution
    whose model carries the full radial distortion. None if it doesn't lock."""
    H, W = shape
    uv = _eqd(index.ra, index.dec, ra0, dec0)
    ok = np.hypot(uv[:, 0], uv[:, 1]) < 100.0
    qx = a[0] * uv[:, 0] + a[1] * uv[:, 1] + a[2]
    qy = b[0] * uv[:, 0] + b[1] * uv[:, 1] + b[2]
    inb = ok & (qx > 0) & (qx < W) & (qy > 0) & (qy < H)
    if inb.sum() < 8:
        return None
    dq, di = det70_tree.query(np.stack([qx[inb], qy[inb]], 1), k=1)
    sel = dq < 12.0
    if sel.sum() < 6:
        return None
    ii, dj = np.flatnonzero(inb)[sel], di[sel]
    a1 = max((abs(a[0] * b[1] - a[1] * b[0])) ** 0.5, 1.0)   # affine scale (px/deg)
    init = CameraModel(cx=a[2], cy=b[2], roll=float(np.arctan2(a[1], a[0])),
                       alt0=dec0, az0=ra0, a1=a1)
    if not _valid_cam(init):
        return None
    try:                                               # lock pose before distortion
        m, _ = fit_model(list(zip(index.dec[ii], index.ra[ii])),
                         list(zip(det[dj, 0], det[dj, 1])), init, False)
    except Exception:
        return None
    if not _valid_cam(m):
        return None
    m = _grow_camera(m, det_full, detF_tree, cra, cdec, W, H)
    if m is None:
        return None
    px, py, rho = m.project(cdec, cra)
    onf = (px > 0) & (px < W) & (py > 0) & (py < H) & (rho < 89)
    ci = np.flatnonzero(onf)
    dq, di = detF_tree.query(np.stack([px[onf], py[onf]], 1), k=1)
    sel = dq < 7.0
    if sel.sum() < 8:
        return None
    ci, dj = ci[sel], di[sel]
    rms = float(np.sqrt(np.mean(dq[sel] ** 2)))
    matches = [(full[i], float(det_full[k, 0]), float(det_full[k, 1]))
               for i, k in zip(ci, dj)]
    rad0 = m.az0 % 360.0
    model = RaDecCamera(m, rad0, m.alt0)
    return Solution(model, rms, matches, len(matches),
                    np.array([rad0, m.alt0]), rad0)


def _trustworthy(model, det_tree, bra, bdec, shape, rms=1.0,
                 min_bright=4, frac=0.45, min_excess=0.38):
    """The decisive false-positive guard: a *correct* solve puts the brightest
    catalog stars onto actual detected stars. An overfit (e.g. a polynomial bent
    onto a tree line) matches faint clutter but leaves the bright, unambiguous
    stars stranded — so we require the in-frame bright stars to land on real
    detections. The tolerance scales with the fit's own RMS: a wide fisheye
    locks at ~3px RMS where a narrow photo locks at <1px, and a correct bright
    star sits within a few RMS either way (an overfit's bright stars are tens of
    px off, far beyond that), so the guard stays just as decisive at both scales.

    But a *low-resolution* frame packs its detections so densely that a random
    position lands near one ~half the time — enough to fake the bright-star rate.
    So we also measure that random baseline and require the bright stars to beat
    it decisively: a true solve hits far above chance (excess +0.44…+0.82 in
    practice), a dense-field coincidence hits only at chance (excess ~+0.32)."""
    H, W = shape
    tol = max(4.0, 3.0 * rms)
    bx, by, _ = model.project(bra, bdec)
    inf = (bx > 0) & (bx < W) & (by > 0) & (by < H) & ~np.isnan(bx)
    if inf.sum() < min_bright:
        return False
    dq, _ = det_tree.query(np.stack([bx[inf], by[inf]], 1), k=1)
    hit = dq < tol
    if hit.sum() < min_bright or hit.mean() < frac:
        return False
    rng = np.random.default_rng(0)
    rp = np.column_stack([rng.uniform(0, W, 800), rng.uniform(0, H, 800)])
    rq, _ = det_tree.query(rp, k=1)
    baseline = (rq < tol).mean()              # chance hit-rate at this density
    return hit.mean() - baseline >= min_excess
