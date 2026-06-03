# How StarGuide works

A deep tour of the pipeline: how an image of dots becomes named stars,
constellations and planets — with **no calibration, no human in the loop, and no
confident wrong answers**. Read the [README](../README.md) first for the
what/why; this is the how.

---

## Contents

1. [The core problem](#1-the-core-problem)
2. [Detection — from pixels to a star list](#2-detection)
3. [Two ways to recover geometry](#3-two-ways-to-recover-the-geometry)
4. [The motion solver (fixed camera)](#4-the-motion-solver)
5. [The blind solver (any photo)](#5-the-blind-solver)
6. [Projection planes, and why the lens model adapts](#6-projection-planes)
7. [Wide-field fisheye: the physical camera model](#7-wide-field-fisheye)
8. [The trust gate — being right or silent](#8-the-trust-gate)
9. [Planets](#9-planets)
10. [Video, stacking, and the stream overlay](#10-video-paths)
11. [Realtime overlay](#11-realtime-overlay)
12. [Module map](#12-module-map)
13. [Configuration & tuning](#13-configuration--tuning)

---

## 1. The core problem

You have an image full of bright dots. You don't know the camera, the lens, where
it pointed, or when the shot was taken. You want to label each dot with the right
catalog star — i.e. recover the map

```text
sky direction (RA, Dec or alt/az)  ->  pixel (x, y)
```

that the camera applied, then invert it for every catalog star. That map is a
**pointing** (where the axis looks), a **rotation** (camera roll), a **scale**
(pixels per degree) and a **lens distortion**. Recovering it from dots alone,
robustly, is the whole game.

The old SkyStream approach assumed a *linear gnomonic* lens and asked a human to
click stars to fit it. That fails structurally: the camera is wide-angle (gnomonic
diverges past ~60°), and clicking isn't automation. StarGuide throws both out.

---

## 2. Detection

`detect.py` turns a frame into a centroid list, and it has to behave identically
whether the frame is an 800 px phone clip or a 6000 px DSLR.

- **Resolution-adaptive.** `detect_auto` rescales every frame to a canonical
  working width (~1600 px) so a star is a few pixels across regardless of sensor
  size, runs detection there, then scales the centroids back. One set of
  thresholds works everywhere. For very low-resolution wide frames the solver can
  ask for a finer width (a ladder, see §8/§10).
- **Background + matched filter.** A local background is subtracted and a small
  Gaussian matched filter sharpens point sources before thresholding at
  `sigma · noise`.
- **Vectorized components.** Connected components are labelled and *filtered*
  (area, peak) with `bincount` over the label image, so the cost is independent of
  the (huge) raw blob count — only the few hundred survivors become Python objects.
- **Shape, not position.** Overlay text, aircraft trails and satellite streaks are
  rejected by **elongation** (the ratio of second-moment eigenvalues), via the
  `round_only` flag. This makes *no assumption about where* clutter sits — an
  earlier version masked image edges and wrongly ate Orion's Belt; that is gone.

Output: a list of `(x, y, flux)` centroids, brightest first.

---

## 3. Two ways to recover the geometry

There are two fundamentally different situations, and StarGuide uses the cheapest
method that fits:

- **A fixed camera filming the sky** has something a photo doesn't: **time**. Over
  seconds the stars rigidly rotate about the celestial pole. That motion hands you
  the geometry almost for free — no catalog needed to find the pole. → the
  **motion solver** (§4).
- **A single photo** (or a handheld/short clip with no usable motion) has only the
  *pattern* of the stars. You must recognize it against the catalog. → the **blind
  solver** (§5).

`identify(source, mode="auto")` routes to the right one and, for video, falls back
from motion to blind-stack when there's no clean drift — always reported.

---

## 4. The motion solver

For a tripod-fixed camera (`solve_video`, `motion.py`, `solve.py`):

1. **Track stars across a bounded set of frames.** `video.sampled_pass` reads at
   most `track_budget` frames, evenly spaced, in one decode pass — so runtime
   depends on the budget, not the clip length. A simple tracker links detections
   into short tracks.
2. **Recover the pole from drift, with no catalog.** Every star moves tangentially
   around the projected celestial pole. So for each star the drift vector is
   perpendicular to (position − pole). Stacking that constraint over all stars is a
   linear least-squares problem whose solution is the **pole pixel**. This also
   *certifies* real stars: fixed-pattern noise doesn't drift, aircraft drift the
   wrong way — both fall out of the fit (the `tangentiality` score reports how
   cleanly the motion is rotational; a handheld clip scores near zero and is
   rejected).
3. **Stack for SNR.** A second bounded, decode-only pass mean-stacks up to
   `stack_budget` frames into a high-SNR image used both for detection and for
   *validating* the solve (§8).
4. **Pole-anchored plate solve.** With the pole pinned, a 2-point RANSAC uses two
   transform-free invariants — the **position-angle difference** about the pole
   (independent of roll and distortion) and the **image-radius ratio** — to match
   image stars to catalog stars. The winning hypothesis is refined into a full lens
   model (§6).

The result is a `SkyModel`: a complete alt/az → pixel map plus everything needed
to overlay any frame.

---

## 5. The blind solver

For a single image (`blind.py`), the astrometry.net idea:

1. **Index the catalog as quad hashes.** For groups of four nearby catalog stars,
   compute a 4-number **code** that is invariant to translation, rotation and
   scale (the two "inner" stars' coordinates in the frame where the two
   widest-apart stars are pinned to (0,0) and (1,1)). Only distinctive, compact
   quads are kept. The codes go in a KD-tree. This is built once and memoized.
2. **Hash the image's quads the same way** and look each up. A code match proposes
   "these four dots are those four catalog stars" — a correspondence, not just a
   location.
3. **Grow each candidate into a pose.** From the four-star correspondence, fit an
   affine map on the seed plane, project the whole catalog, match within a
   shrinking radius, and re-fit — one quad's corner becomes a whole-field estimate.
   Recentring the projection axis on the matched stars turns the lens into a near-
   similarity the affine can track.
4. **Refine the distortion** with the shared 2-D polynomial / physical model (§6,
   §7), then **validate** (§8). The richest *trustworthy* identification wins; a
   non-sky image simply never clears the bar and `solve_blind` returns `None`.

---

## 6. Projection planes

The undistorted "plane" you fit the polynomial on must match the lens, or the fit
fights the geometry. `project.py` offers three, chosen by field:

- **`gnomonic`** (RA/Dec tangent plane) — the rectilinear-lens projection. Ordinary
  camera photos *are* gnomonic, so a low-order polynomial on top captures their
  mild distortion. But gnomonic diverges past ~60° from the axis.
- **`azimuthal`** (alt/az equidistant about the pole) — the all-sky video camera's
  natural plane; this is what makes sidereal time a single rotation.
- **`equidistant`** (RA/Dec equidistant about a tangent point) — for *wide-field
  blind* photos where gnomonic blows up. It is the same equidistant kernel as the
  video camera, fed `(dec, ra)` so that, up to a harmless x↔y reflection, the plane
  radius **is** the true angular distance from the centre.

A `PolyModel` is just `plane + a 2-D polynomial (u,v) -> (x,y)`. The same fit and
overlay code (`refine_poly`) drives all three planes; only the kernel differs.

---

## 7. Wide-field fisheye

A degree-3 polynomial can absorb mild curvature, but a real fisheye has a strong
*odd-radial* distortion: the pixel radius grows like `a1·ρ + a3·ρ³ + a5·ρ⁵`. At
the edge of a 180° frame the `ρ⁵` term reaches **thousands of pixels** — far
beyond what a degree-3 polynomial can represent. Push a polynomial there and it
*folds*: it piles distant catalog stars onto a few detections to game the inlier
count, producing a high-score but geometrically nonsensical "solve".

So for wide fields StarGuide fits the **physical camera model** directly
(`CameraModel`: `cx, cy, roll, axis, a1, a3, a5`) — the same eight-parameter model
the all-sky video camera uses. It cannot fold, because it is a genuine monotonic
radial map.

Getting there robustly (`_wide_solve`, `_grow_camera` in `blind.py`):

- **Seed** the camera model from a candidate's affine inliers.
- **Grow** it outward: each pass widens the radius it trusts and tightens the match
  tolerance, refitting — the physical model *extrapolates* the distortion correctly,
  so a correct seed reaches the frame edge.
- **Two anti-collapse guards.** A free least-squares fit, re-matching each round,
  will happily drive `a1 → 0` (everything projects to one point, matching the
  nearest blob — a runaway). So a refit is rejected unless (a) the matched stars
  stay **spread across the frame** (>30% of the diagonal) and (b) the model stays
  **physical** (sane scale, axis on the sphere).

A thin `RaDecCamera` adapter gives the `CameraModel` the same `.project(ra, dec)`
interface as a `PolyModel`, so overlay and planet code treat both uniformly.

This is what lets one blind solver handle an Orion phone snapshot *and* a 150°
all-sky fisheye.

---

## 8. The trust gate

This is the most important part: **how StarGuide knows it's right**, so it can
decline instead of hallucinating.

A correct solve places the **brightest catalog stars onto real detected stars**.
An overfit (a polynomial bent onto a tree line, a folded fisheye) matches faint
clutter but leaves the bright, unambiguous stars stranded. So `_trustworthy`
requires the in-frame bright stars (mag < 3.2) to land on detections:

- **RMS-scaled tolerance.** A wide fisheye legitimately locks at ~2–3 px RMS where
  a narrow photo locks at <1 px; a correct bright star sits within a few RMS either
  way, an overfit's are tens of px off. So the tolerance is `max(4, 3·RMS)`.
- **Excess over chance.** A *low-resolution* frame packs its detections so densely
  that a random point lands near one ~half the time — enough to fake the bright-star
  rate. So the gate also samples 800 random positions to measure that **baseline
  hit-rate** and requires the bright stars to beat it decisively
  (`bright_hit − baseline ≥ 0.38`). In practice correct solves clear it by
  +0.44…+0.82; a dense-field coincidence only reaches ~+0.32. This single check
  killed a 373-inlier *false* "southern-sky" solve on an 886 px screen-grab that the
  absolute-rate test had passed.

For the *motion* solver, validation is analogous but uses image SNR on the stack:
the fraction of predicted catalog positions that land on a real star (SNR-bright)
in the stacked image — detector-independent, and the honest accuracy number the
examples print.

---

## 9. Planets

Planets move, so they're not in the star catalog. Given a **UTC time**, StarGuide
asks Astropy for each body's RA/Dec (`astro.planets_radec`) and projects it through
the solved model. A body is marked **only when it's genuinely there**
(`pipeline._planets`):

- **Above the horizon** — if a `site` is given, an altitude gate rejects a body
  that a gnomonic frame would otherwise fold in though it isn't in the sky (this is
  what correctly drops a below-horizon Moon).
- **Inside the imaged field** — within the field radius implied by the matched stars.
- **On a real bright object** — the marker snaps to a genuine bright blob at the
  prediction. A bright planet (mag < 0, Jupiter/Venus) is trusted by its brightness
  even in a sparse region; a faint one only where a matched star is nearby (the
  model is locally well-constrained there). Nothing is drawn as a guess.

And in **reverse** (`when_from_planets`): take the brightest non-stellar dot, sweep
candidate dates, and report when a bright planet would have sat there — dating a
photo to within weeks near opposition.

---

## 10. Video paths

Three ways video is handled, all reusing the pieces above:

- **Motion solve** (`solve_video`) — the §4 path, for a fixed camera with a known
  site and real drift. Produces a full `SkyModel`.
- **Blind frame-stack** (`solve_video_blind`) — mean-stack the frames into one
  high-SNR still and blind-solve it. The right path for a clip the motion solver
  can't use: too short for measurable sidereal drift, handheld (shake, not
  rotation), or an unknown camera. `identify(mode="auto")` falls back to this and
  says so; a detection-width ladder recovers tiny sub-pixel stars in low-res frames,
  and the trust gate keeps every accepted solve honest.
- **Stream overlay** (`stream.overlay_stream`) — for "label the live video and keep
  it intact". Solve the geometry from a short burst, then draw constellation lines,
  star rings and names **on top of the untouched frame**, re-solving every few
  seconds so labels follow the moving sky. Works on a file or an `rtsp://` feed and
  can pass each annotated frame to a callback (e.g. an ffmpeg RTSP restreamer).

---

## 11. Realtime overlay

After a *motion* solve, advancing the clock is a single rotation of the catalog in
the azimuthal plane (`project.rotate_uv`) followed by a polynomial eval — about
100 µs per frame, thousands of fps. The expensive solve happens once, over a
bounded number of frames; everything after is nearly free. That's the "realtime" in
the name: identification of a live fixed camera is a one-time cost, not a per-frame
one.

---

## 12. Module map

```text
starguide/
  config.py     SiteConfig / SolveConfig / BlindConfig — the only knobs
  loader.py     read images (incl. AVIF, non-ASCII paths) and detect video files
  detect.py     resolution-adaptive star detection; elongation (round_only) filter
  video.py      bounded single-pass frame sampling (near-constant runtime)
  stack.py      mean-stack frames; SNR-at-pixel queries
  track.py      link detections into short tracks (motion solver)
  motion.py     recover the pole pixel from star drift (no catalog)
  project.py    projection planes, PolyModel, CameraModel, RaDecCamera, fits
  solve.py      pole-anchored plate solve; shared refine_poly
  blind.py      quad-hash index, blind solve, wide-field growth, the trust gate
  astro.py      bundled catalog, constellations, horizon geometry, planet ephemerides
  overlay.py    annotated stills (PIL styling) + realtime overlay video
  stream.py     overlay_stream: label a live/streamed video, frame intact
  pipeline.py   the public verbs: identify / solve_image / solve_video / *_blind
  data/         Hipparcos catalog + constellation lines (bundled)
```

---

## 13. Configuration & tuning

Everything camera- or effort-specific is in `config.py`:

- **`SiteConfig`** — where a fixed camera is (`lat`, `lon`, `height_m`,
  `utc_offset_h`). The *only* thing the motion path needs about your camera.
- **`SolveConfig`** — how hard the motion solver works: `track_budget` /
  `stack_budget` (the bounded frame counts that make runtime constant),
  `work_width`, catalog depth, and the RANSAC/refine bounds. Every loop is capped
  by these, so cost is a function of the config, not the clip.
- **`BlindConfig`** — the blind/still path: quad-index depth (`index_max_mag`,
  `index_k_near`), detection sigma, `n_stars` per quad search, `min_inliers`,
  candidate/refine counts, and `planets` (with a `when`).

Good defaults ship for all of them. For one-off photos and stream overlay there's
nothing to set; for a new fixed camera, write a `SiteConfig` and you're done.
