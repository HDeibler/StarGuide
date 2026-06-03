"""Render the solved sky back onto a frame: stars, names, constellations.

Labels are gated by image SNR so they only appear where a real star actually
sits — cloud-covered and tree-blocked regions stay clean instead of being
littered with predictions for stars the camera can't see.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from .stack import snr_at

# Soft, professional palette (BGR): muted blue-white lines, gentle markers.
LINE = (236, 224, 198)        # soft blue-white
STAR = (228, 224, 214)        # near-white ring
NAME = (245, 238, 222)        # bright soft label
CONST = (210, 188, 150)       # dimmer blue for constellation names
PLANET = (90, 190, 255)
CYAN = (235, 200, 70)         # kept for the bold 'classic' preset


@dataclass
class OverlayStyle:
    """How the overlay is drawn — every line, dot and label is configurable.

    Presets: `pro(shape)` (default — Sky-Guide-like: thin translucent lines that
    stop short of each star so the star stays the clean vertex, tiny soft rings
    only on known-but-unlabelled stars, small offset labels), `auto` (alias of
    pro), and `classic` (bold high-contrast, for presentation)."""
    line_color: tuple = LINE
    line_thickness: int = 1
    line_alpha: float = 0.7           # <1 => translucent lines (soft, layered)
    line_gap: float = 7.0             # stop the line this far short of each star
    max_line_frac: float = 0.45       # skip segments longer than this x diagonal
    star_color: tuple = STAR
    star_radius: float = 2.6          # base ring radius (px)
    star_thickness: int = 1
    star_alpha: float = 0.65
    min_star_radius: float = 1.6
    brightness_scaled: bool = False   # brighter stars -> bigger ring
    ring_constellation_stars: bool = False  # rings on stars a line touches
    label_color: tuple = NAME
    label_scale: float = 0.46
    label_thickness: int = 1
    label_mag: float = 2.2            # label named stars up to this magnitude
    label_offset: int = 6
    constellation_labels: bool = True
    constellation_label_color: tuple = CONST
    constellation_label_scale: float = 0.5
    title_scale: float = 0.8
    planet_color: tuple = (90, 200, 255)     # warm gold (BGR)
    planet_radius: float = 5.0
    planet_alpha: float = 0.9

    @classmethod
    def pro(cls, shape):
        """Default professional styling, sized to the image."""
        s = max(0.6, max(shape) / 1500.0)
        return cls(
            line_thickness=max(1, round(1.1 * s)), line_gap=7.0 * s,
            star_radius=2.6 * s, min_star_radius=1.6 * s,
            label_scale=0.46 * s, label_thickness=max(1, round(0.95 * s)),
            label_offset=max(3, round(5 * s)),
            constellation_label_scale=0.52 * s, title_scale=0.75 * s)

    auto = pro

    @classmethod
    def classic(cls):
        return cls(line_color=CYAN, line_thickness=2, line_alpha=1.0,
                   line_gap=0.0, star_color=(120, 255, 140), star_radius=6.0,
                   star_thickness=2, star_alpha=1.0, min_star_radius=3.0,
                   brightness_scaled=True, ring_constellation_stars=True,
                   label_color=(140, 255, 210), label_scale=0.7,
                   label_thickness=2, label_mag=2.8,
                   constellation_label_color=CYAN,
                   constellation_label_scale=0.85, title_scale=1.1)


def _resolve_style(style, shape):
    if style is None or style in ("auto", "pro"):
        return OverlayStyle.pro(shape)
    if style == "classic":
        return OverlayStyle.classic()
    return style


def stretch(frame: np.ndarray, p: float = 99.7, gamma: float = 0.5) -> np.ndarray:
    g = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3
         else frame).astype(np.float32)
    hi = np.percentile(g, p)
    out = (np.clip(g / max(hi, 1), 0, 1) ** gamma * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def render(frame, model, stars, c1, c2, constellations=None, snr_ref=None,
           snr_min=4.0, style=None, base_stretch=True, title=None,
           planets=None):
    """Draw projected catalog stars + constellation lines onto `frame`.

    `style` is an OverlayStyle, or 'classic'/'auto'. `snr_ref` is a grayscale
    stack used to confirm each star is really present (suppresses labels over
    clouds/trees). `c1, c2` are the catalog's native coords for `model`
    (alt/az for the video camera, RA/Dec for a blind-solved photo). `planets`
    are (name, px, py, mag, confirmed) — solar-system bodies to mark.
    """
    H, W = frame.shape[:2]
    st = _resolve_style(style, (H, W))
    vis = stretch(frame) if base_stretch else frame.copy()
    px, py, _ = model.project(c1, c2)
    on = (px > 0) & (px < W) & (py > 0) & (py < H) & (~np.isnan(px))

    snr = None
    if snr_ref is not None:
        snr = np.zeros(len(stars))
        idx = np.flatnonzero(on)
        sv, _ = snr_at(snr_ref, px[idx], py[idx])
        snr[idx] = sv

    visible = on if snr is None else on & (snr >= snr_min)
    pos = {s.hip: (float(px[i]), float(py[i]))
           for i, s in enumerate(stars) if visible[i]}

    # Lines and rings are drawn on a separate layer, then alpha-blended in, so
    # they read as soft translucent ink over the photo rather than hard paint.
    layer = vis.copy()
    members = set()
    max_len = st.max_line_frac * (W * W + H * H) ** 0.5
    if constellations:
        for pairs in constellations.values():
            for h1, h2 in pairs:
                if h1 in pos and h2 in pos:
                    (x1, y1), (x2, y2) = pos[h1], pos[h2]
                    if (x2 - x1) ** 2 + (y2 - y1) ** 2 > max_len * max_len:
                        continue          # skip sprawling cross-frame segments
                    a, b = _gapped(pos[h1], pos[h2], st.line_gap)
                    if a is not None:
                        cv2.line(layer, a, b, st.line_color, st.line_thickness,
                                 cv2.LINE_AA)
                        members.add(h1); members.add(h2)
    vis = cv2.addWeighted(layer, st.line_alpha, vis, 1 - st.line_alpha, 0)

    ring = vis.copy()
    for i, s in enumerate(stars):
        if not visible[i] or (s.hip in members and not st.ring_constellation_stars):
            continue
        if st.brightness_scaled:
            r = int(max(st.min_star_radius, st.star_radius - s.mag * 1.4))
        else:
            r = int(round(max(st.min_star_radius, st.star_radius)))
        cv2.circle(ring, (int(px[i]), int(py[i])), r, st.star_color,
                   st.star_thickness, cv2.LINE_AA)
    vis = cv2.addWeighted(ring, st.star_alpha, vis, 1 - st.star_alpha, 0)

    # Planets: a small gold disk (a planet shows a disk, not a point), filled if
    # a detection confirms it, hollow if only predicted.
    if planets:
        glyph = vis.copy()
        pr = int(round(st.planet_radius * max(0.7, max(H, W) / 1500.0)))
        for name, x, y, mag, confirmed in planets:
            cv2.circle(glyph, (int(x), int(y)), pr, st.planet_color,
                       -1 if confirmed else 1, cv2.LINE_AA)
        vis = cv2.addWeighted(glyph, st.planet_alpha, vis, 1 - st.planet_alpha, 0)

    # Collect every label, then render once with a real font, faded, placed to
    # dodge the stars and each other.
    obstacles = np.array([[px[i], py[i]] for i in range(len(stars))
                          if visible[i]], float).reshape(-1, 2)
    items = []
    for name, x, y, mag, confirmed in (planets or []):
        items.append(dict(text=name, x=x, y=y, size=st.label_scale * 36,
                          color=st.planet_color, alpha=0.95,
                          gap=st.label_offset + 4, priority=-1))
    if constellations and st.constellation_labels:
        drawn = set()
        for cname, pairs in constellations.items():
            seg = [(h1, h2) for h1, h2 in pairs if h1 in pos and h2 in pos]
            if seg and cname not in drawn:
                drawn.add(cname)
                h1, h2 = seg[len(seg) // 2]
                items.append(dict(
                    text=cname.upper(),
                    x=(pos[h1][0] + pos[h2][0]) / 2,
                    y=(pos[h1][1] + pos[h2][1]) / 2,
                    size=st.constellation_label_scale * 30,
                    color=st.constellation_label_color, alpha=0.62,
                    gap=st.label_offset, priority=1))
    for i, s in enumerate(stars):
        if visible[i] and s.mag <= st.label_mag \
                and not s.name.startswith(("HIP", "HR ")):
            items.append(dict(text=s.name, x=px[i], y=py[i],
                              size=st.label_scale * 34, color=st.label_color,
                              alpha=0.92, gap=st.label_offset, priority=0))
    vis = _draw_labels(vis, items, obstacles)

    if title:
        vis = _draw_labels(vis, [dict(
            text=title, x=24, y=24 + st.title_scale * 30,
            size=st.title_scale * 34, color=(255, 255, 255), alpha=0.9,
            gap=0, priority=2)], np.empty((0, 2)))
    return vis


def _gapped(p1, p2, gap):
    """Shorten a segment by `gap` px at each end so it stops short of the stars."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = (dx * dx + dy * dy) ** 0.5
    if L <= 2.2 * gap:
        return None, None
    ux, uy = dx / L, dy / L
    a = (int(round(p1[0] + ux * gap)), int(round(p1[1] + uy * gap)))
    b = (int(round(p2[0] - ux * gap)), int(round(p2[1] - uy * gap)))
    return a, b


# Clean sans-serif text via PIL (OpenCV's Hershey font looks dated). We try a
# few good system faces and fall back to PIL's bundled DejaVu, then Hershey.
_FONT_PATHS = [
    "/System/Library/Fonts/SFNS.ttf",            # Apple San Francisco
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]
_FONT_CACHE: dict = {}


def _font(px):
    px = max(9, int(round(px)))
    if px in _FONT_CACHE:
        return _FONT_CACHE[px]
    from PIL import ImageFont
    f = None
    for path in _FONT_PATHS:
        try:
            f = ImageFont.truetype(path, px)
            break
        except OSError:
            continue
    if f is None:
        f = ImageFont.load_default()
    _FONT_CACHE[px] = f
    return f


def _place(x, y, tw, th, obstacles, placed, gap):
    """Pick a label top-left near (x, y) that best avoids stars and other labels.
    Tries eight offsets and keeps the one with the most clearance."""
    best, best_score = None, -1e18
    for dx, dy in [(gap, -th / 2), (-tw - gap, -th / 2), (-tw / 2, -th - gap),
                   (-tw / 2, gap), (gap, gap), (-tw - gap, gap),
                   (gap, -th - gap), (-tw - gap, -th - gap)]:
        tx, ty = x + dx, y + dy
        cx, cy = tx + tw / 2, ty + th / 2
        if obstacles.size:
            d = np.min((obstacles[:, 0] - cx) ** 2 + (obstacles[:, 1] - cy) ** 2)
            score = d ** 0.5
        else:
            score = 1e6
        for (a, b, c2, d2) in placed:           # penalise overlapping labels
            if tx < c2 and a < tx + tw and ty < d2 and b < ty + th:
                score -= 400
        if best is None or score > best_score:
            best, best_score = (tx, ty), score
    return best


def _draw_labels(vis, items, obstacles):
    """Render all labels at once with a real font, faded, avoiding stars."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        for it in items:                         # graceful Hershey fallback
            cv2.putText(vis, it["text"], (int(it["x"]), int(it["y"])),
                        cv2.FONT_HERSHEY_SIMPLEX, it["size"] / 30.0,
                        it["color"], 1, cv2.LINE_AA)
        return vis
    pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    placed = []
    for it in sorted(items, key=lambda t: t.get("priority", 1)):
        font = _font(it["size"])
        l, t, r, b = draw.textbbox((0, 0), it["text"], font=font)
        tw, th = r - l, b - t
        x, y = _place(it["x"], it["y"], tw, th, obstacles, placed, it["gap"])
        placed.append((x, y, x + tw, y + th))
        cr, cg, cb = it["color"][2], it["color"][1], it["color"][0]  # BGR->RGB
        draw.text((x, y - t), it["text"], font=font,
                  fill=(cr, cg, cb, int(255 * it["alpha"])))
    out = Image.alpha_composite(pil, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)


def annotate_image(sky_image, max_mag=5.0, style="auto", width=None,
                   out_path=None):
    """Render a labelled overlay for a one-off blind solve (gnomonic model).

    `style` is 'auto' (subtle, default), 'classic', or an OverlayStyle. If
    `out_path` is given the image is written there (parent dirs created)."""
    import os
    from .astro import load_catalog, load_constellations
    stars = load_catalog(max_mag=max_mag)
    ra = np.array([s.ra for s in stars])
    dec = np.array([s.dec for s in stars])
    # style at full image resolution so 'auto' sizing matches the photo
    vis = render(sky_image.image, sky_image.model, stars, ra, dec,
                 constellations=load_constellations(), style=style,
                 base_stretch=False, planets=sky_image.planets)
    if width:
        H, W = vis.shape[:2]
        vis = cv2.resize(vis, (width, int(width * H / W)))
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        cv2.imwrite(out_path, vis)
    return vis


def render_video(video_path, out_path, sky, max_mag=4.2, width=1366):
    """Realtime overlay video: every frame is one sidereal rotation + a poly
    eval, demonstrating that after the one-time solve, tracking is ~free."""
    import time
    from .astro import load_catalog, horizon_altaz, load_constellations
    from .project import azimuthal_equidistant, rotate_uv

    lat = sky.site.lat
    stars = load_catalog(max_mag=max_mag)
    alt, az = horizon_altaz(stars, sky.when_utc, sky.site, min_alt=0)
    vmask = ~np.isnan(alt)
    cons = load_constellations()
    uv0 = azimuthal_equidistant(alt, az, lat, 0.0)

    H, W = sky.stack_img.shape
    px0, py0, _ = sky.model.project(alt, az)
    keep = np.zeros(len(stars), bool)
    ii = np.flatnonzero(vmask & (px0 > 0) & (px0 < W) & (py0 > 0) & (py0 < H))
    snr, _ = snr_at(sky.stack_img, px0[ii], py0[ii])
    keep[ii[snr >= 4.0]] = True

    # sidereal angular rate (signed) in the azimuthal plane
    a2, z2 = horizon_altaz(stars, sky.when_utc.__class__.fromtimestamp(
        sky.when_utc.timestamp() + 200, sky.when_utc.tzinfo), sky.site, min_alt=0)
    uv2 = azimuthal_equidistant(a2, z2, lat, 0.0)
    ok = vmask & ~np.isnan(a2)
    d = (np.degrees(np.arctan2(uv0[ok, 0], uv0[ok, 1])) -
         np.degrees(np.arctan2(uv2[ok, 0], uv2[ok, 1])) + 180) % 360 - 180
    omega = float(np.median(d)) / 200.0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(5); N = int(cap.get(7))
    OW = width; OH = int(OW * H / W); sc = OW / W
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (OW, OH))
    idx = 0; proj_us = []
    while True:
        ok_, f = cap.read()
        if not ok_:
            break
        t0 = time.perf_counter()
        uv = rotate_uv(uv0, omega * idx / fps)
        px, py, _ = sky.model.project_uv(uv)
        proj_us.append((time.perf_counter() - t0) * 1e6)
        vis = cv2.resize(stretch(f), (OW, OH))
        on = (px > 0) & (px < W) & (py > 0) & (py < H) & vmask & keep
        pos = {s.hip: (int(px[i] * sc), int(py[i] * sc))
               for i, s in enumerate(stars) if on[i]}
        for cn, prs in cons.items():
            for a, b in prs:
                if a in pos and b in pos:
                    cv2.line(vis, pos[a], pos[b], CYAN, 1, cv2.LINE_AA)
        for i, s in enumerate(stars):
            if not on[i]:
                continue
            p = (int(px[i] * sc), int(py[i] * sc))
            cv2.circle(vis, p, max(2, int(6 - s.mag)), STAR, 1, cv2.LINE_AA)
            if s.mag < 2.6 and not s.name.startswith(("HIP", "HR ")):
                cv2.putText(vis, s.name, (p[0] + 5, p[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, NAME, 1, cv2.LINE_AA)
        cv2.putText(vis, f"StarGuide realtime  {idx+1}/{N}  proj {proj_us[-1]:.0f}us",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        out.write(vis); idx += 1
    cap.release(); out.release()
    return float(np.median(proj_us)), idx
