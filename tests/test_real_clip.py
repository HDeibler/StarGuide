"""Integration regression on the bundled sample clip (skipped if absent)."""

import os

import pytest

from starguide import solve_video, identification_yield
from starguide.config import SKYSTREAM_SITE

VIDEO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "examples", "video", "input",
                     "skystream_20260601-220541.mp4")


@pytest.mark.skipif(not os.path.exists(VIDEO), reason="sample clip not present")
def test_real_clip_solves_and_identifies():
    sky = solve_video(VIDEO, SKYSTREAM_SITE, verbose=False)
    # motion is a clean rigid rotation
    assert sky.tangentiality < 0.2
    assert 2.0 < sky.mean_drift_px < 6.0
    # the solve identifies most bright stars correctly
    real, tot = identification_yield(sky, max_mag=4.0)
    assert tot > 80 and real / tot > 0.7, f"yield {real}/{tot}"
