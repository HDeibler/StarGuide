"""Video identification via the blind frame-stack and stream-overlay paths.

The motion solver needs a fixed camera with a site, a timestamp, and real
sidereal drift across the clip. A short or handheld clip has none of those, so
`identify` mean-stacks the frames and blind-solves the result instead, and
`overlay_stream` draws the labels on the (intact) frames.

The stream-overlay test always runs — it builds a tiny clip from a committed
sample photo. The handheld/low-res tests run only when those sample clips are
present (they aren't committed; see Data/README in git history).
"""

import glob
import os

import cv2
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, "examples", "image", "input")
# Optional large sample clips (not committed) — tests self-skip when absent.
HANDHELD = glob.glob(os.path.join(ROOT, "**", "RPReplay_Final1721192009.*"),
                     recursive=True)
LOWRES = glob.glob(os.path.join(ROOT, "**", "RPReplay_Final1723476666.*"),
                   recursive=True)


def test_overlay_stream_writes_annotated_video(tmp_path):
    """The RTSP/video-overlay path keeps the video intact and draws labels on
    top: solve once from a burst, then annotate every frame. We synthesize a
    short clip from a committed Orion photo so this always runs."""
    from starguide import overlay_stream
    photo = cv2.imread(os.path.join(IMAGES, "orion.jpg"))
    assert photo is not None, "committed sample photo missing"
    clip = str(tmp_path / "clip.mp4")
    vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 20,
                         (photo.shape[1], photo.shape[0]))
    for _ in range(30):
        vw.write(photo)
    vw.release()

    out = str(tmp_path / "annotated.mp4")
    n = overlay_stream(clip, out_path=out, resolve_every_s=30, verbose=False)
    assert n >= 30
    assert os.path.exists(out) and os.path.getsize(out) > 0


@pytest.mark.skipif(not HANDHELD, reason="handheld sample clip not present")
def test_handheld_clip_identifies_via_stack():
    """A handheld night-sky clip: the motion solver can't use its jittery
    'drift', but stacking + blind solve recovers the Gemini/Auriga/Taurus field."""
    from starguide import identify, SkyImage
    sky = identify(HANDHELD[0], mode="auto", verbose=False)
    assert isinstance(sky, SkyImage)
    hips = {s.hip for s, *_ in sky.matches}
    # Capella (24608) and Pollux (37826) / Castor (36850) anchor that region.
    assert sky.solution.n_inliers >= 15
    assert 24608 in hips or 37826 in hips or 36850 in hips


@pytest.mark.skipif(not LOWRES, reason="low-res sample clip not present")
def test_lowres_allsky_clip_declined_not_faked():
    """An 886px all-sky screen-grab is below the resolution where a >100deg
    fisheye can be blind-solved: its detections are so dense that a wrong pose
    matches bright stars at the random rate. The excess-over-chance guard must
    reject it rather than emit a confident but false solve — declining honestly
    beats a fabricated answer."""
    from starguide import identify
    with pytest.raises(RuntimeError):
        identify(LOWRES[0], mode="auto", verbose=False)
