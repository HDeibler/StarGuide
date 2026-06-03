#!/usr/bin/env python3
"""StarGuide — automatic, calibration-free star identification.

    python identify.py <source> [--video|--image] [--out PATH] [--style auto|classic]

`<source>` may be a live-stream/clip (motion solver, needs a camera SiteConfig)
or a one-off photo (blind solver, needs nothing). The mode is auto-detected;
force it with --video / --image. `--out` sets where the annotated image is
written; `--style` picks the look. Edit SITE below for your own camera.
"""

import os
import sys
import time
from datetime import datetime, timezone

from starguide import identify, identification_yield, SkyImage, BlindConfig
from starguide.config import SKYSTREAM_SITE
from starguide.overlay import render_video

SITE = SKYSTREAM_SITE          # replace with your camera's SiteConfig
DEFAULT = "examples/video/input/skystream_20260601-220541.mp4"


def _opt(flags, name, default=None):
    for f in flags:
        if f.startswith(name + "="):
            return f.split("=", 1)[1]
    return default


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    source = args[0] if args else DEFAULT
    mode = ("video" if "--video" in flags else
            "image" if "--image" in flags else "auto")
    style = _opt(flags, "--style", "auto")
    os.makedirs("out", exist_ok=True)
    base = os.path.splitext(os.path.basename(source))[0]
    out = _opt(flags, "--out", f"out/{base}_identified.png")

    when = None
    ws = _opt(flags, "--when")          # e.g. --when=2018-06-16T01:44:58
    if ws:
        when = datetime.fromisoformat(ws).replace(tzinfo=timezone.utc)
    cfg = BlindConfig(planets=("--planets" in flags or when is not None)) \
        if mode != "video" else None

    t0 = time.time()
    sky = identify(source, site=SITE, mode=mode, cfg=cfg, when=when,
                   save=out, style=style)
    print(f"      solved in {time.time() - t0:.1f}s; wrote {out}")

    if not isinstance(sky, SkyImage):
        for ml in (3.0, 4.0, 4.5):
            real, tot = identification_yield(sky, max_mag=ml)
            print(f"      yield(mag<{ml}): {real}/{tot} = "
                  f"{100*real/max(tot,1):.0f}% on a real star")
        us, n = render_video(source, "out/starguide_realtime.mp4", sky)
        print(f"      wrote out/starguide_realtime.mp4 ({1e6/us:.0f} fps projection)")


if __name__ == "__main__":
    main()
