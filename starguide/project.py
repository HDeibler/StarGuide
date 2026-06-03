"""Camera model: sky direction <-> image pixel, with lens distortion.

The old SkyStream code used a *linear gnomonic* map (az/alt scaled straight to
pixels). That cannot work for this lens: gnomonic blows up past ~60deg from the
axis, and this camera is a wide-angle pointed near the celestial pole spanning
horizon-over-zenith. The honest model for such a lens is **azimuthal** about the
optical axis with a radial polynomial:

    direction (alt,az)
      -> unit vector in the horizontal frame
      -> rotate so the optical axis is +Z  (axis given by alt0, az0)
      -> (rho, theta): rho = angle from axis, theta = azimuth around axis
      -> r_px = a1*rho + a3*rho^3 + a5*rho^5      (odd radial distortion, deg->px)
      -> pixel = (cx + r_px*sin(theta+roll),  cy - r_px*cos(theta+roll))

This single family covers equidistant/fisheye lenses smoothly all the way to the
horizon. Eight parameters (cx, cy, roll, alt0, az0, a1, a3, a5) are fit to the
star matches by least squares — no hand calibration.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


def altaz_to_vec(alt_deg, az_deg):
    """Horizontal-frame unit vectors. x=East, y=North, z=Up."""
    alt = np.radians(alt_deg)
    az = np.radians(az_deg)
    ca = np.cos(alt)
    return np.stack([ca * np.sin(az), ca * np.cos(az), np.sin(alt)], axis=-1)


def _axis_basis(alt0, az0):
    """Orthonormal basis with w = optical axis, (u,v) spanning the image plane."""
    w = altaz_to_vec(alt0, az0)                      # axis direction
    up = np.array([0.0, 0.0, 1.0])
    u = np.cross(up, w)                               # points to the right (East-ish)
    nu = np.linalg.norm(u)
    if nu < 1e-6:                                     # axis at zenith: pick North
        u = np.array([0.0, 1.0, 0.0]); nu = 1.0
    u = u / nu
    v = np.cross(w, u)                               # completes right-handed frame
    return u, v, w


@dataclass
class CameraModel:
    cx: float
    cy: float
    roll: float          # radians
    alt0: float          # optical axis altitude (deg)
    az0: float           # optical axis azimuth (deg)
    a1: float            # px per degree (linear scale)
    a3: float = 0.0
    a5: float = 0.0

    def project(self, alt_deg, az_deg):
        """(alt,az) arrays -> (px, py) arrays. Vectorized."""
        vec = altaz_to_vec(np.asarray(alt_deg, float), np.asarray(az_deg, float))
        u, v, w = _axis_basis(self.alt0, self.az0)
        cw = np.clip(vec @ w, -1, 1)
        rho = np.degrees(np.arccos(cw))              # angle from axis (deg)
        theta = np.arctan2(vec @ u, vec @ v)         # azimuth around axis
        r = self.a1 * rho + self.a3 * rho ** 3 + self.a5 * rho ** 5
        ang = theta + self.roll
        px = self.cx + r * np.sin(ang)
        py = self.cy - r * np.cos(ang)
        return px, py, rho

    def params(self):
        return np.array([self.cx, self.cy, self.roll, self.alt0, self.az0,
                         self.a1, self.a3, self.a5])

    @staticmethod
    def from_params(p):
        return CameraModel(*p)


@dataclass
class RaDecCamera:
    """Adapter giving a `CameraModel` the PolyModel `.project(ra, dec)` interface,
    so overlay / planet code treats it uniformly. The physical fisheye model is
    fit in the RA/Dec-relabelled frame (dec -> the model's 'altitude' slot, ra ->
    'azimuth'), which is `radec_to_vec` up to an x<->y swap (a pure reflection),
    so angles are exact. Unlike a degree-3 polynomial it carries the a5 radial
    term a >100deg fisheye needs at its edge — where a polynomial folds."""
    cam: CameraModel
    center1: float          # ra0  (deg)
    center2: float          # dec0 (deg)

    def project(self, c1, c2):
        # c1 = RA, c2 = Dec  ->  CameraModel.project(alt=Dec, az=RA)
        return self.cam.project(np.asarray(c2, float), np.asarray(c1, float))


def _poly_terms(u, v, degree):
    """Design matrix of 2D polynomial terms up to `degree`. (N, T)."""
    cols = []
    for d in range(degree + 1):
        for i in range(d + 1):
            cols.append((u ** (d - i)) * (v ** i))
    return np.stack(cols, axis=1)


def plane_project(c1, c2, center1, center2, plane):
    """Map celestial coords onto the undistorted plane used by a PolyModel.

    'azimuthal'   — alt/az equidistant about the pole, for the all-sky video camera.
    'gnomonic'    — RA/Dec tangent plane, for ordinary narrow-field blind photos.
    'equidistant' — RA/Dec equidistant about (ra0, dec0), for *wide-field / fisheye*
                    blind photos where gnomonic diverges. Feeding (dec, ra) into the
                    equidistant kernel is the genuine RA/Dec projection (it differs
                    from radec_to_vec only by an x<->y swap, a pure reflection, so
                    `rho` is the true angular distance from centre).
    All three yield a plane the true image relates to by a 2-D polynomial.
    """
    if plane == "gnomonic":
        return gnomonic(c1, c2, center1, center2)
    if plane == "equidistant":
        return azimuthal_equidistant(c2, c1, center2, center1)
    return azimuthal_equidistant(c1, c2, center1, center2)


@dataclass
class PolyModel:
    """Sky -> pixel via an undistorted plane + a 2D polynomial (the SIP idea).

    The plane removes the bulk of the geometry; a low-order polynomial absorbs
    all remaining smooth lens distortion. `plane` selects the base projection so
    the SAME model, fit and overlay serve both the all-sky video camera
    ('azimuthal', centre = pole) and ordinary photos ('gnomonic', centre =
    tangent point). `center1/center2` are (alt0, az0) or (ra0, dec0) accordingly.
    """
    A: np.ndarray          # (T,) x-coefficients
    B: np.ndarray          # (T,) y-coefficients
    degree: int
    center1: float = 0.0    # plane centre coord 1 (alt0 / ra0)
    center2: float = 0.0    # plane centre coord 2 (az0 / dec0)
    plane: str = "azimuthal"

    def project(self, c1, c2):
        uv = plane_project(np.asarray(c1, float), np.asarray(c2, float),
                           self.center1, self.center2, self.plane)
        return self.project_uv(uv)

    def project_uv(self, uv):
        T = _poly_terms(uv[:, 0], uv[:, 1], self.degree)
        rho = np.hypot(uv[:, 0], uv[:, 1])
        return T @ self.A, T @ self.B, rho


# Sidereal rotation: the sky turns 360deg about the pole per sidereal day.
SIDEREAL_DEG_PER_SEC = 360.0 / 86164.0905


def rotate_uv(uv, angle_deg):
    """Rigid rotation of azimuthal coords about the pole — this is exactly what
    the passage of sidereal time does, so advancing the clock is one rotation,
    not a full re-projection. Enables realtime overlay after a one-time solve."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.stack([uv[:, 0] * c - uv[:, 1] * s,
                     uv[:, 0] * s + uv[:, 1] * c], axis=1)


def fit_poly(uv, px, py, degree, weights=None):
    """Linear least-squares 2D polynomial fit (u,v)->(x,y)."""
    T = _poly_terms(uv[:, 0], uv[:, 1], degree)
    if weights is not None:
        w = np.sqrt(weights)[:, None]
        A, *_ = np.linalg.lstsq(T * w, px * w[:, 0], rcond=None)
        B, *_ = np.linalg.lstsq(T * w, py * w[:, 0], rcond=None)
    else:
        A, *_ = np.linalg.lstsq(T, px, rcond=None)
        B, *_ = np.linalg.lstsq(T, py, rcond=None)
    return A, B


def gnomonic(ra_deg, dec_deg, ra0, dec0):
    """Gnomonic (TAN) tangent-plane coords (degrees) about (ra0, dec0).

    The standard rectilinear-lens projection: ordinary camera photos are
    gnomonic, so a low-order polynomial on top captures their mild distortion.
    Directions behind the tangent plane get NaN (off-image).
    """
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    ra0r, dec0r = np.radians(ra0), np.radians(dec0)
    cosc = (np.sin(dec0r) * np.sin(dec) +
            np.cos(dec0r) * np.cos(dec) * np.cos(ra - ra0r))
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.cos(dec) * np.sin(ra - ra0r) / cosc
        eta = (np.cos(dec0r) * np.sin(dec) -
               np.sin(dec0r) * np.cos(dec) * np.cos(ra - ra0r)) / cosc
    xi = np.where(cosc > 1e-6, xi, np.nan)
    eta = np.where(cosc > 1e-6, eta, np.nan)
    return np.stack([np.degrees(xi), np.degrees(eta)], axis=-1)


def radec_to_vec(ra_deg, dec_deg):
    """ICRS RA/Dec (deg) -> unit vectors. Rows are (x, y, z)."""
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=-1)


def azimuthal_equidistant(alt_deg, az_deg, alt0, az0):
    """Ideal (undistorted) planar coords in degrees about an axis.

    This is the seed space for matching: it is related to the true image by a
    similarity (scale+rotation+shift) plus the radial distortion, so *local*
    star patterns keep their shape and can be matched before any fit exists.
    """
    vec = altaz_to_vec(np.asarray(alt_deg, float), np.asarray(az_deg, float))
    u, v, w = _axis_basis(alt0, az0)
    cw = np.clip(vec @ w, -1, 1)
    rho = np.degrees(np.arccos(cw))
    theta = np.arctan2(vec @ u, vec @ v)
    # -cos matches the pixel convention used by CameraModel.project (y grows
    # downward), so the ideal plane relates to the image by a pure similarity
    # (no reflection) — which is exactly what the 2-point RANSAC can fit.
    return np.stack([rho * np.sin(theta), -rho * np.cos(theta)], axis=-1)


def fit_model(matches_altaz, matches_px, init: CameraModel,
              fit_distortion: bool = True):
    """Least-squares fit of the camera model to (sky, pixel) correspondences."""
    from scipy.optimize import least_squares

    alt = np.array([m[0] for m in matches_altaz])
    az = np.array([m[1] for m in matches_altaz])
    px = np.array([p[0] for p in matches_px])
    py = np.array([p[1] for p in matches_px])

    def resid(p):
        m = CameraModel.from_params(p if fit_distortion
                                    else np.concatenate([p, init.params()[6:]]))
        qx, qy, _ = m.project(alt, az)
        return np.concatenate([qx - px, qy - py])

    p0 = init.params() if fit_distortion else init.params()[:6]
    sol = least_squares(resid, p0, method="lm", max_nfev=200)
    full = sol.x if fit_distortion else np.concatenate([sol.x, init.params()[6:]])
    model = CameraModel.from_params(full)
    rms = float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    return model, rms
