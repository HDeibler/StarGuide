"""End-to-end plate solve on a SYNTHETIC sky — the core correctness test.

We build a known camera model, project the real catalog through it to make
synthetic detections (with centroid noise, dropped stars to mimic cloud, and
false detections to mimic sensor noise), then ask the solver to recover it given
only the pole pixel and site latitude. We assert it both reproduces the true
geometry and identifies the right stars. Fully deterministic; no video.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from starguide.config import SiteConfig, SolveConfig
from starguide.solve import solve, RadialModel, catalog_polar
from starguide.astro import load_catalog, horizon_altaz

SITE = SiteConfig(lat=40.19, lon=-76.33)
WHEN = datetime(2026, 6, 2, 2, 5, 41, tzinfo=timezone.utc)
W, H = 3840, 2160
TRUE_POLE = np.array([2252.0, 1587.0])


def _truth():
    # mild fisheye: R(rho) = 18*rho - 0.02*rho^2, pole lower-right, slight roll
    return RadialModel(pole=TRUE_POLE, roll=np.radians(-9.0),
                       coef=np.array([0.0, 18.0, -0.02]), parity=1, lat=SITE.lat)


def _synthetic_scene(cat, alt, az, truth, rng, lim_mag=4.2, drop=0.15,
                     n_false=40):
    """Realistic detections + a matching stack image. Only stars brighter than a
    limiting magnitude are 'seen'; some are dropped (cloud); false detections and
    pixel noise mimic the sensor. The stack lets the solver validate by SNR, the
    same path production uses."""
    px, py, _ = truth.project(alt, az)
    on = (px > 0) & (px < W) & (py > 0) & (py < H) & (~np.isnan(alt))
    stack = rng.normal(20, 2.0, (H, W)).astype(np.float32)
    ay, ax = np.mgrid[-5:6, -5:6]
    blob = np.exp(-(ax ** 2 + ay ** 2) / (2 * 1.3 ** 2))
    det, truth_xy = [], {}
    for i in np.flatnonzero(on):
        if cat[i].mag > lim_mag or rng.random() < drop:
            continue
        x = px[i] + rng.normal(0, 0.7)
        y = py[i] + rng.normal(0, 0.7)
        det.append((x, y, 10 ** (-0.4 * cat[i].mag) * 1000 + 50))
        truth_xy[cat[i].hip] = (px[i], py[i])
        ix, iy = int(round(x)), int(round(y))
        if 5 <= ix < W - 5 and 5 <= iy < H - 5:
            stack[iy - 5:iy + 6, ix - 5:ix + 6] += 80 * blob
    for _ in range(n_false):
        det.append((rng.uniform(0, W), rng.uniform(0, H), rng.uniform(60, 300)))
    return det, px, py, on, truth_xy, np.clip(stack, 0, 255)


@pytest.fixture(scope="module")
def catalog():
    cat = load_catalog(max_mag=5.5)
    alt, az = horizon_altaz(cat, WHEN, SITE, min_alt=0)
    return cat, alt, az


def test_synthetic_solve_recovers_geometry(catalog):
    cat, alt, az = catalog
    truth = _truth()
    rng = np.random.default_rng(0)
    det, px, py, on, _, stk = _synthetic_scene(cat, alt, az, truth, rng)

    sol = solve(det, cat, alt, az, (H, W), TRUE_POLE, SITE.lat, SolveConfig(), stack=stk)
    assert sol is not None, "solver found no consensus"

    # Evaluate in the well-sampled central field (rho < 80deg). Beyond that the
    # polynomial extrapolates into the extreme corners — a known all-sky limit.
    _, rho = catalog_polar(alt, az, SITE.lat)
    bright = [i for i in np.flatnonzero(on)
              if cat[i].mag < 3.0 and rho[i] < 80.0]
    rpx, rpy, _ = sol.model.project(alt, az)
    err = np.hypot(rpx[bright] - px[bright], rpy[bright] - py[bright])
    assert np.median(err) < 5.0, f"median bright-star error {np.median(err):.1f}px"


def test_synthetic_solve_identifies_correct_stars(catalog):
    cat, alt, az = catalog
    truth = _truth()
    rng = np.random.default_rng(0)
    det, px, py, on, truth_xy, stk = _synthetic_scene(cat, alt, az, truth, rng)

    sol = solve(det, cat, alt, az, (H, W), TRUE_POLE, SITE.lat, SolveConfig(), stack=stk)
    assert sol is not None

    # truth position of EVERY in-frame star (a correct match puts the right
    # catalog star where it truly is, regardless of whether it was detectable)
    truth_pos = {cat[i].hip: (px[i], py[i]) for i in np.flatnonzero(on)}
    rho_of = {cat[i].hip: r for i, r in
              zip(range(len(cat)), catalog_polar(alt, az, SITE.lat)[1])}
    # judged in the well-sampled central field (corners extrapolate, as above)
    central = [(cs, mx, my) for cs, mx, my in sol.matches
               if rho_of.get(cs.hip, 999) < 80.0]
    assert len(central) > 60, "too few central identifications"
    correct = sum(np.hypot(*(np.subtract(truth_pos.get(cs.hip, (1e9, 1e9)),
                                         (mx, my)))) < 8.0
                  for cs, mx, my in central)
    frac = correct / len(central)
    assert frac > 0.85, f"only {frac:.0%} of central identifications are correct"


def test_solver_returns_none_on_random_field(catalog):
    cat, alt, az = catalog
    rng = np.random.default_rng(3)
    det = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(60, 300))
           for _ in range(200)]
    sol = solve(det, cat, alt, az, (H, W), TRUE_POLE, SITE.lat, SolveConfig())
    # pure noise should not yield a confident solution
    assert sol is None or sol.n_inliers < 12
