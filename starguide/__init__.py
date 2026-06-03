"""StarGuide — realtime, calibration-free star identification from sky video.

Public API:

    from starguide import solve_video, SiteConfig, SolveConfig

    sky = solve_video("clip.mp4", SiteConfig(lat=40.19, lon=-76.33,
                                             utc_offset_h=4))
    # sky.model.project(alt_deg, az_deg) -> (px, py, rho) for any frame
    real, total = identification_yield(sky)        # honest accuracy

To target a different camera, supply your own SiteConfig (location + how its
timestamps map to UTC). Nothing else is location-specific.
"""

from .config import SiteConfig, SolveConfig, BlindConfig, SKYSTREAM_SITE
from .pipeline import (solve_video, solve_image, solve_video_blind, identify,
                       identification_yield, when_from_planets,
                       SkyModel, SkyImage)
from .stream import overlay_stream
from .solve import Solution, RadialModel, catalog_polar
from .project import (PolyModel, CameraModel, rotate_uv, azimuthal_equidistant,
                      gnomonic)
from .overlay import OverlayStyle, annotate_image, render

__all__ = [
    "identify", "solve_video", "solve_image", "solve_video_blind",
    "overlay_stream",
    "identification_yield", "when_from_planets", "SkyModel", "SkyImage",
    "SiteConfig", "SolveConfig", "BlindConfig", "SKYSTREAM_SITE",
    "Solution", "RadialModel", "PolyModel", "CameraModel",
    "OverlayStyle", "annotate_image", "render",
    "rotate_uv", "azimuthal_equidistant", "gnomonic", "catalog_polar",
]
