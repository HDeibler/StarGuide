#!/usr/bin/env python3
"""Identify a fixed-camera night video and overlay the labels ON the video.

    python examples/video/identify_video.py                 # the bundled clip
    python examples/video/identify_video.py path/to/clip.mp4

The bundled `input/skystream.mp4` is a tripod-fixed all-sky camera. The motion
solver recovers the celestial pole from the stars' sidereal drift (no
calibration), fits the wide-angle lens once, and then every frame is overlaid by a
single rotation of the catalog — so the labelled video is produced in realtime
after the one-time solve. The result is written to `output/` (the committed
`output/skystream_overlay.mp4` was produced by exactly this command).

The video itself is preserved; the constellations and star names ride on top.

Live / RTSP feeds (no fixed site) — overlay without a motion solve:

    python examples/video/identify_video.py rtsp://CAM/stream --stream
    #   ... --stream --rtsp-out=rtsp://localhost:8554/identified   (restream it)
"""

import glob
import os
import sys
import time

from starguide import identify, overlay_stream, SkyModel
from starguide.config import SKYSTREAM_SITE
from starguide.overlay import render_video

SITE = SKYSTREAM_SITE          # <-- replace with your camera's SiteConfig
HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "input")
OUT = os.path.join(HERE, "output")


def _opt(flags, name, default=None):
    for f in flags:
        if f.startswith(name + "="):
            return f.split("=", 1)[1]
    return default


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bundled = sorted(glob.glob(os.path.join(IN, "skystream*.mp4")))
    source = args[0] if args else (bundled[0] if bundled else "")
    if not source:
        print("no input video found; pass a path: identify_video.py CLIP.mp4")
        return
    os.makedirs(OUT, exist_ok=True)
    name = os.path.splitext(os.path.basename(source.rstrip("/")))[0] or "stream"
    out = os.path.join(OUT, f"{name}_overlay.mp4")

    if "--stream" in flags:
        # General path: solve periodically and draw on the intact frames. Works
        # on any clip or rtsp:// feed; no site or timestamp needed.
        rtsp_out = _opt(flags, "--rtsp-out")
        on_frame = _ffmpeg_rtsp(rtsp_out) if rtsp_out else None
        n = overlay_stream(source, out_path=out,
                           resolve_every_s=float(_opt(flags, "--resolve", "20")),
                           max_seconds=(float(_opt(flags, "--seconds"))
                                        if _opt(flags, "--seconds") else None),
                           on_frame=on_frame, verbose=True)
        print(f"done: {n} frames -> {out}")
        return

    # Fixed-camera path: one motion solve, then a realtime overlay video.
    t0 = time.time()
    sky = identify(source, site=SITE, mode="video", verbose=True)
    print(f"solved in {time.time() - t0:.1f}s")
    if not isinstance(sky, SkyModel):
        print("not a fixed-camera clip (no usable star motion) — "
              "re-run with --stream to overlay it the general way")
        return
    us, n = render_video(source, out, sky, star_rings="--no-rings" not in flags)
    print(f"wrote {out}  ({n} frames, {1e6 / us:.0f} fps projection after solve)")


def _ffmpeg_rtsp(url, fps=25):
    """Pipe annotated frames to ffmpeg -> RTSP. Needs ffmpeg + a running server."""
    import subprocess
    import numpy as np
    state = {"proc": None}

    def on_frame(bgr):
        h, w = bgr.shape[:2]
        if state["proc"] is None:
            state["proc"] = subprocess.Popen(
                ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                 "-c:v", "libx264", "-preset", "ultrafast", "-tune",
                 "zerolatency", "-f", "rtsp", url], stdin=subprocess.PIPE)
        state["proc"].stdin.write(np.ascontiguousarray(bgr).tobytes())

    return on_frame


if __name__ == "__main__":
    main()
