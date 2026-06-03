"""Point-source detection on a synthetic frame."""

import numpy as np

from starguide.detect import detect


def _frame_with_stars(centres, H=400, W=600, peak=40.0, sigma=1.4):
    img = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for cx, cy in centres:
        img += peak * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    img += np.random.default_rng(0).normal(0, 1.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_detect_finds_stars_at_right_positions():
    centres = [(120, 90), (300, 200), (450, 320), (520, 80)]
    img = _frame_with_stars(centres)
    stars, _ = detect(img, sigma=4.0, min_area=2, min_peak=5.0)
    assert len(stars) >= len(centres)
    for cx, cy in centres:
        nearest = min(np.hypot(b.x - cx, b.y - cy) for b in stars)
        assert nearest < 2.0, f"star ({cx},{cy}) not recovered (off {nearest:.1f}px)"


def test_detect_centroids_are_subpixel():
    img = _frame_with_stars([(200.5, 150.5)], peak=60)
    stars, _ = detect(img, sigma=4.0, min_area=2, min_peak=5.0)
    b = min(stars, key=lambda s: np.hypot(s.x - 200.5, s.y - 150.5))
    assert abs(b.x - 200.5) < 0.7 and abs(b.y - 150.5) < 0.7
