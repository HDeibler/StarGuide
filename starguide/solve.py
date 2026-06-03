"""Fully automatic plate solve: detected pixels -> identified catalog stars.

No manual calibration. The priors are only what we genuinely know: the site
latitude (which fixes the pole's *sky* direction), the frame time, the projected
pole *pixel* (from star motion, see motion.py), and that the lens is azimuthal
about an axis near that pole.

Method — a constrained, not blind, solve:
  1. 2-point RANSAC anchored at the pole. Two correspondences are proposed only
     when they agree on two transform-free invariants: the difference in
     position-angle about the pole (independent of roll AND lens distortion) and
     the ratio of image radii. Each surviving pair fixes (roll, scale).
  2. The winner seeds a radial fit R(rho) (the fisheye curve), then a 2D
     polynomial (SIP-style) mops up residual decentering.

Every loop is bounded by SolveConfig, so the cost does not grow with the star
count of a richer sky — runtime is effectively constant.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree

from .config import SolveConfig
from .project import (altaz_to_vec, _axis_basis,
                      plane_project, PolyModel, fit_poly, _poly_terms)


@dataclass
class Solution:
    model: object               # RadialModel or PolyModel; both have .project()
    rms: float
    matches: list               # (CatalogStar, px, py)
    n_inliers: int
    pole: np.ndarray
    lat: float


@dataclass
class RadialModel:
    """Azimuthal model about a fixed pole with a free polynomial radius(rho).

    pixel = pole + R(rho) * (sin(theta+roll), -cos(theta+roll)); theta and rho
    (co-declination) come from the sky about the pole. R is a plain polynomial,
    so any monotonic fisheye curve fits. The pole pixel is pinned, which is what
    makes the fit well-conditioned.
    """
    pole: np.ndarray
    roll: float
    coef: np.ndarray
    parity: int
    lat: float

    def project(self, alt_deg, az_deg):
        theta, rho = catalog_polar(np.asarray(alt_deg, float),
                                   np.asarray(az_deg, float), self.lat)
        R = np.polyval(self.coef[::-1], rho)
        ang = self.parity * theta + self.roll
        return (self.pole[0] + R * np.sin(ang),
                self.pole[1] - R * np.cos(ang), rho)


def catalog_polar(alt_deg, az_deg, lat):
    """Position angle (theta, rad) and angular separation (rho, deg) from the
    celestial pole, whose sky direction is (alt=lat, az=0)."""
    vec = altaz_to_vec(alt_deg, az_deg)
    u, v, w = _axis_basis(lat, 0.0)
    rho = np.degrees(np.arccos(np.clip(vec @ w, -1, 1)))
    theta = np.arctan2(vec @ u, vec @ v)
    return theta, rho


def solve(detected, stars, alt, az, img_shape, pole, lat,
          cfg: SolveConfig | None = None, stack=None, rho_max=82.0,
          sep_tol_deg=1.2, ratio_tol=0.22, match_px=26.0) -> Solution | None:
    """Identify stars from detections given the pole pixel and site latitude.

    detected: list of (x, y, flux). stars/alt/az: aligned catalog arrays.
    `stack`: optional grayscale stack image — if given, candidate models are
    validated by image SNR at the predicted positions (the honest signal, robust
    to an incomplete detection list); otherwise by detection proximity.
    Returns a Solution, or None if no consensus is found.
    """
    cfg = cfg or SolveConfig()
    H, W = img_shape
    ranked = sorted(detected, key=lambda d: -d[2])
    # Anchors come from the brightest few; scoring/matching uses a bounded set of
    # the brightest detections so a flood of faint noise can't inflate scores.
    det_all = ranked[: cfg.n_detect]
    det_score = np.array([(x, y) for x, y, _ in ranked[: cfg.n_score]], float)
    det = np.array([(x, y) for x, y, _ in det_all], float)
    cx, cy = float(pole[0]), float(pole[1])
    phi = np.arctan2(det[:, 0] - cx, cy - det[:, 1])
    rr = np.hypot(det[:, 0] - cx, det[:, 1] - cy)

    vis = ~np.isnan(alt)
    vstars = [(s, alt[i], az[i]) for i, (s, v) in enumerate(zip(stars, vis))
              if v and s.mag <= cfg.max_mag]
    vstars.sort(key=lambda t: t[0].mag)
    theta0, rho0 = catalog_polar(np.array([t[1] for t in vstars]),
                                 np.array([t[2] for t in vstars]), lat)
    keep_field = rho0 < rho_max
    vstars = [vstars[i] for i in np.flatnonzero(keep_field)][: cfg.n_catalog]
    if len(vstars) < 8:
        return None
    calt = np.array([t[1] for t in vstars])
    caz = np.array([t[2] for t in vstars])
    theta_c, rho_c = catalog_polar(calt, caz, lat)
    # Score hypotheses against the BRIGHT anchor set only: real stars dominate
    # it, so the true (roll, scale) wins. Scoring against the full faint list
    # lets chance matches inflate wrong hypotheses.
    det_tree = cKDTree(det)
    sep_tol = np.radians(sep_tol_deg)

    # Precompute the catalog pair table ONCE per parity (the n^2 build used to
    # sit inside the detected-pair loop). dth[p][a,b] = wrapped position-angle
    # difference; we just index it in the hot loop.
    pair_dth = {p: _wrap(p * (theta_c[:, None] - theta_c[None, :]))
                for p in (1, -1)}

    # ---- propose (roll, scale, parity) hypotheses with the cheap invariants --
    # The two invariants (position-angle difference + radius ratio) prune pairs
    # to a handful of distinct (roll, scale); we DEDUP near-duplicates so the
    # expensive scoring (project + nearest-neighbour) runs once per real
    # hypothesis, not once per star pair.
    hyps = {}
    for p in (1, -1):
        dth = pair_dth[p]
        for i in range(len(det)):
            for k in range(i + 1, len(det)):
                dphi = _wrap(phi[i] - phi[k])
                rat = rr[i] / (rr[k] + 1e-9)
                if not (0.3 < rat < 3.0):
                    continue
                cand = np.argwhere(np.abs(_wrap(dth - dphi)) < sep_tol)
                for a, b in cand:
                    if a == b:
                        continue
                    if abs(rho_c[a] / (rho_c[b] + 1e-9) - rat) > ratio_tol * rat:
                        continue
                    roll = _wrap(phi[i] - p * theta_c[a])
                    s = rr[i] / (rho_c[a] + 1e-6)
                    if not (cfg.scale_lo < s < cfg.scale_hi):
                        continue
                    key = (p, round(float(roll), 2), round(float(s), 1))
                    hyps[key] = (float(roll), float(s), p)

    scored = []
    for roll, s, p in hyps.values():
        ang = p * theta_c + roll
        qx = cx + s * rho_c * np.sin(ang)
        qy = cy - s * rho_c * np.cos(ang)
        on = (qx > 0) & (qx < W) & (qy > 0) & (qy < H)
        if on.sum() < 8:
            continue
        dq, di = det_tree.query(np.stack([qx[on], qy[on]], 1), k=1)
        nin = int(np.unique(di[dq < match_px]).size)
        scored.append((nin, roll, s, p))
    scored.sort(key=lambda t: -t[0])
    if not scored or scored[0][0] < cfg.min_consensus:
        return None

    # Deeper catalog for the refine stages.
    deep = [(s2, alt[i], az[i]) for i, (s2, v) in enumerate(zip(stars, vis))
            if v and s2.mag <= cfg.max_mag]
    dcal = np.array([d[1] for d in deep]); dcaz = np.array([d[2] for d in deep])
    dcst = [d[0] for d in deep]
    score_tree = cKDTree(det_score)
    snr_prep = None
    if stack is not None:
        from .stack import prep_snr
        snr_prep = prep_snr(stack)            # background-subtract once

    # Refine each top hypothesis on the bright anchors (faint detections would
    # corrupt the polynomial), then SCORE it by how many catalog stars the
    # refined model actually identifies against the full detection list. A real
    # solution identifies hundreds; a wrong/noise hypothesis only a handful.
    best = None
    for _, roll, s, p in scored[: cfg.n_refine]:
        seed = RadialModel(np.array([cx, cy]), float(roll),
                           np.array([0.0, s]), p, lat)
        rad = _refine_radial(seed, det, dcal, dcaz, dcst, (H, W), lat,
                             match_px=90.0)
        if rad is not None:
            seed = rad.model
        px0, py0, _ = seed.project(dcal, dcaz)
        sol = refine_poly(px0, py0, det, dcal, dcaz, dcst, (H, W), lat, 0.0,
                          "azimuthal", pole)
        if sol is None:
            continue
        n_id, matches = _identify(sol.model, dcal, dcaz, dcst, (H, W),
                                  score_tree, det_score, snr_prep)
        if best is None or n_id > best[0]:
            best = (n_id, sol.model, sol.rms, matches)
    if best is None or best[0] < cfg.min_inliers:
        return None
    n_id, model, rms, matches = best
    return Solution(model, rms, matches, n_id, np.asarray(pole), lat)


def _identify(model, dcal, dcaz, dcst, img_shape, score_tree, det_score, snr_prep):
    """Count + list catalog stars the model places on a real star — by image
    SNR if a prepared stack is given (robust to a sparse detector), else by
    detection proximity. Separates a true solution (hundreds) from a wrong one."""
    H, W = img_shape
    qx, qy, _ = model.project(dcal, dcaz)
    keep = (qx > 0) & (qx < W) & (qy > 0) & (qy < H)
    ci = np.flatnonzero(keep)
    if snr_prep is not None:
        from .stack import snr_query
        snr = snr_query(snr_prep[0], snr_prep[1], qx[keep], qy[keep])
        m = snr > 4.0
        matches = [(dcst[i], float(qx[i]), float(qy[i])) for i in ci[m]]
    else:
        dq, dj = score_tree.query(np.stack([qx[keep], qy[keep]], 1), k=1)
        m = dq < 6.0
        idx = np.flatnonzero(m)
        matches = [(dcst[ci[t]], float(det_score[dj[t]][0]),
                    float(det_score[dj[t]][1])) for t in idx]
    return int(m.sum()), matches


def _refine_radial(model, det, calt, caz, cstars, img_shape, lat, match_px):
    """Reproject -> NN match -> refit roll + radius R(rho), tightening gate."""
    H, W = img_shape
    cx, cy = model.pole
    det_tree = cKDTree(det)
    matches, rms = [], 1e9
    for it in range(6):
        qx, qy, _ = model.project(calt, caz)
        keep = (qx > 0) & (qx < W) & (qy > 0) & (qy < H)
        if keep.sum() < 10:
            break
        dq, di = det_tree.query(np.stack([qx[keep], qy[keep]], 1), k=1)
        tol = max(match_px * (0.7 ** it), 3.0)
        sel = dq < tol
        ci = np.flatnonzero(keep)[sel]; dj = di[sel]
        if len(ci) < 10:
            break
        th, rh = catalog_polar(calt[ci], caz[ci], lat)
        ang_match = np.arctan2(det[dj, 0] - cx, cy - det[dj, 1])
        roll = float(np.angle(np.mean(np.exp(1j * (ang_match - model.parity * th)))))
        rr = np.hypot(det[dj, 0] - cx, det[dj, 1] - cy)
        coef = _fit_through_origin(rh, rr, 3 if len(ci) > 18 else 2)
        model = RadialModel(np.array([cx, cy]), roll, coef, model.parity, lat)
        qx2, qy2, _ = model.project(calt[ci], caz[ci])
        rms = float(np.sqrt(np.mean((qx2 - det[dj, 0]) ** 2 +
                                    (qy2 - det[dj, 1]) ** 2)))
        matches = [(cstars[i], float(det[k][0]), float(det[k][1]))
                   for i, k in zip(ci, dj)]
    if not matches:
        return None
    return Solution(model, rms, matches, len(matches), np.array([cx, cy]), lat)


def refine_poly(px0, py0, det, c1, c2, cstars, img_shape, center1, center2,
                plane, anchor, schedule=None):
    """Lock distortion with a 2D polynomial: reproject -> NN match -> robust
    polynomial fit, growing degree and tightening tolerance each round. Seeded
    from an initial projection (px0, py0). Plane-agnostic: drives both the
    azimuthal all-sky camera and the gnomonic blind solve.

    `c1, c2` are the catalog's native coords (alt/az or RA/Dec); `center1,
    center2, plane` define the undistorted plane; `anchor` is stashed on the
    Solution (the pole pixel, or the tangent RA/Dec)."""
    H, W = img_shape
    det = np.asarray(det, float)
    det_tree = cKDTree(det)
    uv = plane_project(c1, c2, center1, center2, plane)
    px, py = np.asarray(px0, float), np.asarray(py0, float)
    model, matches, rms = None, [], 1e9
    schedule = schedule or [(1, 45), (1, 30), (2, 22), (2, 15), (3, 11), (3, 8),
                            (3, 6), (3, 4.5)]
    for degree, tol in schedule:
        keep = (px > 0) & (px < W) & (py > 0) & (py < H) & ~np.isnan(px)
        if keep.sum() < 6:
            break
        ci_all = np.flatnonzero(keep)
        dq, di = det_tree.query(np.stack([px[keep], py[keep]], 1), k=1)
        sel = dq < tol
        ci, dj = ci_all[sel], di[sel]
        need = (degree + 1) * (degree + 2) // 2 + 4
        if len(ci) < need:
            continue
        suv, sx, sy = uv[ci], det[dj, 0], det[dj, 1]
        for _ in range(3):
            A, B = fit_poly(suv, sx, sy, degree)
            res = np.hypot(_poly_terms(suv[:, 0], suv[:, 1], degree) @ A - sx,
                           _poly_terms(suv[:, 0], suv[:, 1], degree) @ B - sy)
            good = res < max(3.0, 2.5 * np.median(res))
            if good.sum() < need or good.all():
                break
            suv, sx, sy, ci, dj = suv[good], sx[good], sy[good], ci[good], dj[good]
        model = PolyModel(A=A, B=B, degree=degree, center1=center1,
                          center2=center2, plane=plane)
        px, py, _ = model.project(c1, c2)
        rms = float(np.sqrt(np.mean(res[good] ** 2))) if good.any() else rms
        matches = [(cstars[i], float(det[k][0]), float(det[k][1]))
                   for i, k in zip(ci, dj)]
    if not matches or model is None:
        return None
    return Solution(model, rms, matches, len(matches), np.asarray(anchor),
                    center1)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _fit_through_origin(rho, r, deg=3):
    """Fit R(rho) = c1*rho + ... (no constant: the pole has R=0)."""
    M = np.stack([rho ** k for k in range(1, deg + 1)], axis=1)
    c, *_ = np.linalg.lstsq(M, r, rcond=None)
    return np.concatenate([[0.0], c])
