"""Catalog + astronomy: what is in the sky, and where (on the celestial sphere).

Pure astronomy, independent of the camera: load the bundled star catalog, parse
the frame timestamp, and convert catalog RA/Dec into local Alt/Az for a given
`SiteConfig` at a given instant. Astropy does the Alt/Az transform so precession,
nutation, aberration and sidereal time are SOFA-accurate — one vectorized call
per timestamp over the whole catalog.

Catalogs are bundled in `starguide/data/`, so the package is self-contained and
drops into any codebase.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .config import SiteConfig

_DATA = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class CatalogStar:
    hip: int
    ra: float    # degrees, J2000
    dec: float   # degrees, J2000
    mag: float
    name: str


def load_catalog(max_mag: float = 6.5) -> list[CatalogStar]:
    hip = json.load(open(os.path.join(_DATA, "hip_catalog.json")))
    names = json.load(open(os.path.join(_DATA, "hip_names.json")))
    import math
    out = [CatalogStar(int(h), d["ra"], d["dec"], d["mag"],
                       names.get(h, f"HIP {h}"))
           for h, d in hip.items()
           if d["mag"] <= max_mag
           and math.isfinite(d["ra"]) and math.isfinite(d["dec"])
           and math.isfinite(d["mag"])]
    out.sort(key=lambda s: s.mag)
    return out


def load_constellations() -> dict[str, list[tuple[int, int]]]:
    raw = json.load(open(os.path.join(_DATA, "constellation_data.json")))
    return {c: [(int(a), int(b)) for a, b in pairs] for c, pairs in raw.items()}


def parse_timestamp(name: str, utc_offset_h: float = 0.0) -> datetime:
    """Parse 'prefix_YYYYMMDD-HHMMSS-...' -> aware UTC datetime.

    `utc_offset_h` converts the camera's wall-clock to UTC (e.g. 4 for EDT).
    """
    m = re.search(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})", name)
    if not m:
        raise ValueError(f"no timestamp in {name!r}")
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    local = datetime(y, mo, d, h, mi, s)
    return (local + timedelta(hours=utc_offset_h)).replace(tzinfo=timezone.utc)


def horizon_altaz(stars: list[CatalogStar], when_utc: datetime,
                  site: SiteConfig, min_alt: float = 0.0):
    """Vectorized Alt/Az for all catalog stars. Stars below `min_alt` get NaN."""
    from astropy.coordinates import EarthLocation, AltAz, SkyCoord
    from astropy.time import Time
    import astropy.units as u

    ra = np.array([s.ra for s in stars])
    dec = np.array([s.dec for s in stars])
    loc = EarthLocation(lat=site.lat * u.deg, lon=site.lon * u.deg,
                        height=site.height_m * u.m)
    aa = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(
        AltAz(obstime=Time(when_utc), location=loc))
    alt = aa.alt.deg.astype(np.float64)
    az = aa.az.deg.astype(np.float64)
    alt[alt < min_alt] = np.nan
    return alt, az


def planets_radec(when_utc: datetime, site: SiteConfig | None = None):
    """Solar-system bodies as ICRS RA/Dec (deg) at `when_utc`, for projecting
    through a plate-solved model. Planets are essentially geocentric at our
    accuracy; the Moon's ~1deg parallax wants a site, so pass one if you have it.

    Returns [(name, ra_deg, dec_deg, approx_mag), ...].
    """
    from astropy.coordinates import EarthLocation, get_body
    from astropy.time import Time
    import astropy.units as u

    t = Time(when_utc)
    loc = None
    if site is not None:
        loc = EarthLocation(lat=site.lat * u.deg, lon=site.lon * u.deg,
                            height=site.height_m * u.m)
    out = []
    for name, mag in [("mercury", 0.0), ("venus", -4.0), ("mars", 0.7),
                      ("jupiter", -2.2), ("saturn", 0.6), ("moon", -11.0)]:
        c = get_body(name, t, loc).icrs
        out.append((name.capitalize(), float(c.ra.deg), float(c.dec.deg), mag))
    return out


def planets_altaz(when_utc: datetime, site: SiteConfig, min_alt: float = 0.0):
    """Bright solar-system bodies above the horizon: (name, alt, az, mag)."""
    from astropy.coordinates import EarthLocation, AltAz, get_body
    from astropy.time import Time
    import astropy.units as u

    loc = EarthLocation(lat=site.lat * u.deg, lon=site.lon * u.deg,
                        height=site.height_m * u.m)
    frame = AltAz(obstime=Time(when_utc), location=loc)
    out = []
    for name, mag in [("jupiter", -2.2), ("saturn", 0.7), ("mars", 1.0),
                      ("venus", -4.0), ("mercury", 0.0), ("moon", -11.0)]:
        c = get_body(name, Time(when_utc), loc).transform_to(frame)
        if c.alt.deg > min_alt:
            out.append((name.capitalize(), float(c.alt.deg), float(c.az.deg), mag))
    return out
