"""Blind plate solve on a SYNTHETIC gnomonic scene — deterministic, no network.

Project the real catalog through a known tangent + affine to make a star photo,
then ask the blind solver to recover which stars they are, given nothing.
"""

import numpy as np
import pytest

from starguide.astro import load_catalog
from starguide.project import gnomonic, CameraModel
from starguide.blind import build_index, solve_blind
from starguide.config import BlindConfig

W, H = 1200, 900
RA0, DEC0 = 83.0, 5.0           # Orion/Taurus/Gemini: a star-rich field
SCALE = 42.0                    # px per degree -> ~28x21 deg, many stars
ROT = np.radians(12.0)


def _scene(rng, drop=0.1, n_false=25):
    cat = load_catalog(max_mag=6.0)
    ra = np.array([s.ra for s in cat]); dec = np.array([s.dec for s in cat])
    uv = gnomonic(ra, dec, RA0, DEC0)
    c, s = np.cos(ROT), np.sin(ROT)
    px = W / 2 + SCALE * (uv[:, 0] * c - uv[:, 1] * s)
    py = H / 2 + SCALE * (uv[:, 0] * s + uv[:, 1] * c)
    on = (px > 0) & (px < W) & (py > 0) & (py < H) & ~np.isnan(px)
    det, truth = [], {}
    for i in np.flatnonzero(on):
        if cat[i].mag > 5.8 or rng.random() < drop:
            continue
        det.append((px[i] + rng.normal(0, 0.5), py[i] + rng.normal(0, 0.5),
                    10 ** (-0.4 * cat[i].mag) * 1000 + 50))
        truth[cat[i].hip] = (px[i], py[i])
    for _ in range(n_false):
        det.append((rng.uniform(0, W), rng.uniform(0, H), rng.uniform(60, 300)))
    return det, truth


@pytest.fixture(scope="module")
def index():
    return build_index(max_mag=5.5, k_near=7)


def test_blind_identifies_known_field(index):
    rng = np.random.default_rng(0)
    det, truth = _scene(rng)
    sol = solve_blind(det, (H, W), index, BlindConfig())
    assert sol is not None, "blind solver found no match on a clear field"
    correct = sum(1 for cs, mx, my in sol.matches
                  if cs.hip in truth
                  and np.hypot(*np.subtract(truth[cs.hip], (mx, my))) < 6)
    assert correct >= 15
    assert correct / len(sol.matches) > 0.8


def test_blind_rejects_random_field(index):
    rng = np.random.default_rng(1)
    det = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(60, 300))
           for _ in range(120)]
    assert solve_blind(det, (H, W), index, BlindConfig()) is None


# --- wide-field / fisheye: a gnomonic-only solver cannot recover this scene ---
FW, FH = 1600, 1200
F_RA0, F_DEC0 = 270.0, -28.0     # Sagittarius/Scorpius: a dense wide field
F_CAM = CameraModel(cx=FW / 2, cy=FH / 2, roll=np.radians(8.0),
                    alt0=F_DEC0, az0=F_RA0, a1=24.0, a3=0.0016)   # ~73deg FOV


def _fisheye_scene(rng, drop=0.1, n_false=30):
    """Project the catalog through a known equidistant fisheye (CameraModel),
    treating (dec, ra) as (alt, az) exactly as the blind solver does."""
    cat = load_catalog(max_mag=6.0)
    ra = np.array([s.ra for s in cat]); dec = np.array([s.dec for s in cat])
    px, py, rho = F_CAM.project(dec, ra)
    on = (px > 0) & (px < FW) & (py > 0) & (py < FH) & (rho < 85)
    det, truth = [], {}
    for i in np.flatnonzero(on):
        if cat[i].mag > 5.8 or rng.random() < drop:
            continue
        det.append((px[i] + rng.normal(0, 0.5), py[i] + rng.normal(0, 0.5),
                    10 ** (-0.4 * cat[i].mag) * 1000 + 50))
        truth[cat[i].hip] = (px[i], py[i])
    for _ in range(n_false):
        det.append((rng.uniform(0, FW), rng.uniform(0, FH), rng.uniform(60, 300)))
    return det, truth


def test_blind_identifies_wide_fisheye(index):
    rng = np.random.default_rng(3)
    det, truth = _fisheye_scene(rng)
    sol = solve_blind(det, (FH, FW), index, BlindConfig())
    assert sol is not None, "blind solver failed on a wide fisheye field"
    correct = sum(1 for cs, mx, my in sol.matches
                  if cs.hip in truth
                  and np.hypot(*np.subtract(truth[cs.hip], (mx, my))) < 6)
    assert correct >= 30          # a degree-3 gnomonic refine folds here; the
    assert correct / len(sol.matches) > 0.8     # CameraModel path locks it
