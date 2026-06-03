"""Configuration — everything that changes between cameras lives here.

To point StarGuide at a different camera/codebase you only touch a `SiteConfig`
(where the camera is) and optionally a `SolveConfig` (how hard to work). Nothing
else in the package hardcodes a location, a path, or a frame count.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteConfig:
    """Where and when the camera observes.

    lat/lon in degrees (North/East positive); height in metres. `utc_offset_h`
    converts the camera's wall-clock timestamp to UTC (e.g. US Eastern Daylight
    = 4). If your frames are already stamped in UTC, set it to 0.
    """
    lat: float
    lon: float
    height_m: float = 100.0
    utc_offset_h: float = 0.0
    name: str = "camera"


# The SkyStream reference camera (Pennsylvania, frames stamped in US Eastern,
# June -> EDT = UTC-4). Replace with your own SiteConfig elsewhere.
SKYSTREAM_SITE = SiteConfig(
    lat=40.192526, lon=-76.330737, height_m=100.0, utc_offset_h=4.0,
    name="skystream")


@dataclass(frozen=True)
class BlindConfig:
    """One-off image mode: identify an arbitrary photo with no time/location/pole
    via quad-hash blind plate solving."""
    index_max_mag: float = 5.5    # catalog depth for the quad-hash index
    index_k_near: int = 7         # neighbours per star when building quads
    detect_sigma: float = 4.0
    n_stars: int = 70             # brightest image stars used to form quads
    code_tol: float = 0.02        # quad-code match tolerance
    min_inliers: int = 20         # stars the refined solution must identify
    max_candidates: int = 800
    n_refine: int = 10            # top candidates refined + validated
    planets: bool = False         # identify planets — needs a date (`when`) too


@dataclass(frozen=True)
class SolveConfig:
    """Live-stream / video mode. Knobs that bound the work, so runtime is a
    function of these — not of clip length or sensor megapixels."""
    track_budget: int = 60        # frames sampled for motion tracking (costly pass)
    stack_budget: int = 250       # frames stacked for SNR (decode-only, cheap)
    work_width: int = 2560        # track at this width; centroids scaled back
    max_mag: float = 6.0          # faintest catalog star considered
    detect_sigma: float = 5.0
    min_persist_frac: float = 0.6  # fraction of sampled frames a star must appear in
    # plate-solve bounds (all loops are capped by these)
    n_detect: int = 70            # brightest detected stars used as anchor pairs
    n_score: int = 320            # brightest detected stars used for scoring/refine
    n_catalog: int = 140          # brightest in-field catalog stars used as anchors
    min_consensus: int = 9        # RANSAC inliers a hypothesis needs to be refined
    n_refine: int = 6             # top hypotheses refined; best kept
    min_inliers: int = 40         # catalog stars the final model must identify
    scale_lo: float = 6.0         # px/deg sanity gate
    scale_hi: float = 60.0
