#!/usr/bin/env python3
"""Identify the stars and constellations in a still photo — no calibration.

    python examples/image/identify_image.py [IMAGE ...]

With no argument it runs every photo in `input/`. Each is *blind-solved* from the
star pattern alone — no time, location, lens or pole given — and an annotated
overlay is written to `output/`. The committed `output/` images were produced by
exactly this command, so you can compare your run against them.

To name planets too, pass a UTC capture time:

    python examples/image/identify_image.py input/orion.jpg --when=2023-12-10T06:00:00
"""

import glob
import os
import sys
from datetime import datetime, timezone

from starguide import identify, BlindConfig

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "input")
OUT = os.path.join(HERE, "output")


def run(path, when):
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(OUT, f"{name}_identified.jpg")
    cfg = BlindConfig(planets=when is not None)
    sky = identify(path, mode="image", cfg=cfg, when=when, save=out, verbose=False)
    named = sorted({s.name for s, *_ in sky.matches
                    if not s.name.startswith(("HIP", "HR "))})
    print(f"  {name}: {sky.solution.n_inliers} stars identified "
          f"(RMS {sky.solution.rms:.1f}px)")
    print(f"      e.g. {', '.join(named[:6]) or '(no named bright stars)'}")
    if sky.planets:
        print(f"      planets: {', '.join(p[0] for p in sky.planets if p[4])}")
    print(f"      -> {out}")


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    when = None
    for f in flags:
        if f.startswith("--when="):
            when = datetime.fromisoformat(f.split("=", 1)[1]).replace(
                tzinfo=timezone.utc)
    os.makedirs(OUT, exist_ok=True)
    paths = args or sorted(glob.glob(os.path.join(IN, "*.jpg")))
    print(f"Identifying {len(paths)} image(s) from {IN}:")
    for p in paths:
        try:
            run(p, when)
        except Exception as e:
            print(f"  {os.path.basename(p)}: declined — {e}")


if __name__ == "__main__":
    main()
