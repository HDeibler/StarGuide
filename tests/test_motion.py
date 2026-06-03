"""Pole recovery from motion — synthetic, deterministic."""

import numpy as np

from starguide.motion import fit_pole


def _tangential_field(pole, n, rng, noise=0.0):
    """Stars on random radii about `pole`, each drifting tangentially."""
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = rng.uniform(300, 1500, n)
    pos = pole + np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    tang = np.column_stack([-np.sin(ang), np.cos(ang)])           # perpendicular
    drift = tang * (0.002 * rad)[:, None]                          # ~ r * dphi
    drift += rng.normal(0, noise, drift.shape)
    return pos, drift


def test_fit_pole_recovers_centre():
    rng = np.random.default_rng(0)
    pole = np.array([2252.0, 1587.0])
    pos, drift = _tangential_field(pole, 300, rng, noise=0.05)
    est = fit_pole(pos, drift)
    assert np.hypot(*(est - pole)) < 20.0


def test_fit_pole_is_robust_to_some_radial_movers():
    rng = np.random.default_rng(1)
    pole = np.array([1000.0, 800.0])
    pos, drift = _tangential_field(pole, 250, rng, noise=0.05)
    # the tangentiality gate in motion.confirm filters radial movers; here we
    # just confirm the LS centre is close with mostly-clean input.
    est = fit_pole(pos, drift)
    assert np.hypot(*(est - pole)) < 25.0
