"""Geometry invariants — fast, deterministic, no video or network."""

import numpy as np

from starguide.project import (altaz_to_vec, azimuthal_equidistant, rotate_uv,
                               PolyModel, fit_poly)
from starguide.solve import catalog_polar


def test_altaz_to_vec_unit_and_axes():
    v = altaz_to_vec(np.array([0.0, 90.0, 0.0]), np.array([0.0, 0.0, 90.0]))
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(v[1], [0, 0, 1])          # zenith -> +Up
    assert np.allclose(v[0], [0, 1, 0], atol=1e-9)   # N horizon -> +North
    assert np.allclose(v[2], [1, 0, 0], atol=1e-9)   # E horizon -> +East


def test_pole_direction_has_zero_separation():
    lat = 40.19
    # The pole's sky direction is (alt=lat, az=0); rho there must be ~0.
    _, rho = catalog_polar(np.array([lat]), np.array([0.0]), lat)
    assert rho[0] < 1e-6


def test_rotate_uv_is_a_pure_rotation():
    rng = np.random.default_rng(0)
    uv = rng.normal(size=(50, 2))
    r = rotate_uv(uv, 37.0)
    assert np.allclose(np.linalg.norm(r, axis=1), np.linalg.norm(uv, axis=1))
    back = rotate_uv(r, -37.0)
    assert np.allclose(back, uv, atol=1e-9)
    assert np.allclose(rotate_uv(uv, 360.0), uv, atol=1e-9)


def test_azimuthal_radius_equals_separation():
    # Azimuthal-equidistant radius (deg) must equal angular separation from axis.
    lat = 40.19
    alt = np.array([lat, lat + 10, lat - 25])
    az = np.array([0.0, 0.0, 0.0])
    uv = azimuthal_equidistant(alt, az, lat, 0.0)
    _, rho = catalog_polar(alt, az, lat)
    assert np.allclose(np.hypot(uv[:, 0], uv[:, 1]), rho, atol=1e-6)


def test_poly_model_fits_an_affine_exactly():
    rng = np.random.default_rng(1)
    uv = rng.normal(size=(60, 2)) * 20
    # a known affine map (degree-1 polynomial)
    px = 1900 + 18 * uv[:, 0] - 4 * uv[:, 1]
    py = 1000 + 4 * uv[:, 0] + 18 * uv[:, 1]
    A, B = fit_poly(uv, px, py, degree=1)
    m = PolyModel(A=A, B=B, degree=1)
    qx, qy, _ = m.project_uv(uv) if hasattr(m, "project_uv") else (None, None, None)
    qx = (np.column_stack([np.ones(len(uv)), uv[:, 0], uv[:, 1]]) @ A)
    qy = (np.column_stack([np.ones(len(uv)), uv[:, 0], uv[:, 1]]) @ B)
    assert np.allclose(qx, px, atol=1e-6) and np.allclose(qy, py, atol=1e-6)
