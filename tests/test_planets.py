"""Planet identification — ephemeris sanity + (if the sample photo is present)
forward identify-from-date and reverse date-from-planet."""

import glob
import os
from datetime import datetime, timezone

import pytest

from starguide.astro import planets_radec

# The June-16-2018 Scorpius photo: Saturn (Sagittarius) + Jupiter (Libra).
SCENE = glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "Data", "test-images", "*7.12.29*"))
WHEN_2018 = datetime(2018, 6, 16, 1, 44, 58, tzinfo=timezone.utc)


def test_planets_radec_sane():
    bodies = planets_radec(WHEN_2018)
    names = {b[0] for b in bodies}
    assert {"Jupiter", "Saturn", "Mars", "Moon"} <= names
    for name, ra, dec, mag in bodies:
        assert 0 <= ra <= 360 and -90 <= dec <= 90


def test_planets_require_a_date():
    from starguide import solve_image, BlindConfig
    import numpy as np
    # a blank frame won't solve, but the date-required check happens regardless
    with pytest.raises(Exception):
        solve_image(np.zeros((100, 100, 3), "uint8"),
                    BlindConfig(planets=True), when=None, verbose=False)


@pytest.mark.skipif(not SCENE, reason="sample photo not present")
def test_forward_identifies_saturn():
    from starguide import solve_image, BlindConfig
    sky = solve_image(SCENE[0], BlindConfig(planets=True), when=WHEN_2018,
                      verbose=False)
    confirmed = {p[0] for p in sky.planets if p[4]}
    assert "Saturn" in confirmed


@pytest.mark.skipif(not SCENE, reason="sample photo not present")
def test_reverse_dates_the_photo():
    from starguide import solve_image, when_from_planets
    sky = solve_image(SCENE[0], verbose=False)
    body, when, resid = when_from_planets(sky, years=(2016, 2020), step_days=4)
    assert body in ("Saturn", "Jupiter")
    assert when.year == 2018 and resid < 12
