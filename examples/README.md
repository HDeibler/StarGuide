# Examples

Two self-contained use cases. Each folder has an `input/` (what ships in), an
`output/` (the result, **pre-computed and committed** so you can compare), and one
runnable script. Run everything from the repo root.

```text
examples/
  image/
    input/     orion.jpg, bootes.jpg          # sample photos
    output/    *_identified.jpg               # the identified overlays
    identify_image.py
  video/
    input/     skystream_*.mp4                # a fixed all-sky camera clip
    output/    *_overlay.mp4, *_poster.jpg    # labels drawn on the video
    identify_video.py
```

## Image — identify a still photo (blind solve)

```bash
python examples/image/identify_image.py
```

Blind-solves every photo in `image/input/` from the star pattern alone — no time,
location, or lens data — and writes the annotated overlay to `image/output/`. The
committed outputs were produced by this exact command. Expected:

```text
  orion:  225 stars identified (RMS 0.9px)  e.g. Alnilam, Alnitak, Betelgeuse…
  bootes:  72 stars identified (RMS 0.9px)  e.g. Arcturus
```

Identify your own photo, and (optionally) name planets at a UTC capture time:

```bash
python examples/image/identify_image.py my_photo.jpg --when=2018-06-16T05:44:58
```

## Video — identify a clip and overlay labels ON the video

```bash
python examples/video/identify_video.py            # the bundled skystream clip
```

The bundled clip is a tripod-fixed all-sky camera. The **motion solver** recovers
the celestial pole from the stars' sidereal drift (no calibration), fits the lens
once, then overlays every frame by a single rotation of the catalog — realtime
after the one-time solve. The video is preserved; the constellations and names
ride on top. Output (committed): `video/output/skystream_*_overlay.mp4`.

For a feed with no known camera/site — a handheld clip or an `rtsp://` stream —
use the general overlay path, which re-solves periodically and draws on the intact
frames:

```bash
python examples/video/identify_video.py path/to/clip.mov --stream
python examples/video/identify_video.py rtsp://CAM/stream --stream \
    --rtsp-out=rtsp://localhost:8554/identified      # also restream it back out
```

`--rtsp-out` pipes annotated frames to `ffmpeg` → an RTSP server (e.g.
[MediaMTX](https://github.com/bluenviron/mediamtx)); watch with
`ffplay rtsp://localhost:8554/identified`.
